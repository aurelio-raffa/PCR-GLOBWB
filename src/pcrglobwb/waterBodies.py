"""Lakes and reservoirs of PCR-GLOBWB 2: the WaterBodies class.

Annotated copy of src/pcrglobwb/waterBodies.py on the RAWS branch of
UU-Hydro/PCR-GLOBWB_model (commit da91a875, 2026-10-15). Only comments and
docstrings were added; the code is unchanged. Apart from __init__, every method is
semantically identical to upstream master (model/waterBodies.py at 41e08fa).

Purpose
-------
WaterBodies holds, for every lake and reservoir of the model domain, the static
description (ids, outlet cell, surface area, type, capacity), three state variables
(storage, 30-day average inflow, 1825-day average outflow) and the rules that turn
the storage into a release for the current time step:

- lakes (waterBodyTyp == 1): a broad-crested weir formula, getLakeOutflow;
- reservoirs (waterBodyTyp == 2): a storage-fraction rule based on the long-term
  average outflow, the 30-day average inflow, an environmental-flow floor and a
  flood release, getReservoirOutflow.

Wetlands (input type 0) are dropped. A water body may span many cells; it releases
water through a single outlet cell.

Call sequence
-------------
1. routing.py Routing.__init__ creates the object: WaterBodies(iniItems, landmask).
2. routing.py Routing.update calls getParameterFiles at the first time step (with
   the routing state, so that getICs sets the initial states) and on every
   1 January afterwards (the water-body maps are yearly).
3. routing.py Routing.simple_update (routingMethod accuTravelTime or
   simplifiedKinematicWave, i.e. all RAWS configurations; once per day) or
   Routing.kinematic_wave_update (kinematicWave; once per sub-step) moves the
   channel storage of the water-body cells out of the channel and calls update:
   moveFromChannelToWaterBody (net inflow, avgInflow), then getWaterBodyOutflow
   (getLakeOutflow, getReservoirOutflow, 25 % cap, avgOutflow, storage).
4. routing.py takes waterBodyOutflow at the outlet cell, adds it to the channel
   storage (of the outlet cell; spread over the body's cells under
   accuTravelTime), routes the river network, and puts the remaining storage back
   into the channel storage of the water-body cells
   (Routing.return_water_body_storage_to_channel). The next step starts from there.

Units and spatial support
-------------------------
All maps are PCRaster maps on the model clone (scalars are 32-bit floats). Volumes
are in m3, flows in m3/s, areas in m2, heads and widths in m, time-step lengths in
s; "m3 per time step" is a volume over one call of update (one day in the RAWS
configurations). Spatial support:
- per cell: may differ from cell to cell;
- per water body: constant over all cells of a water body (result of
  pcr.areatotal, areaaverage, areamaximum or areamajority over waterBodyIds), MV
  (missing value) or 0 elsewhere. A per-body total such as the storage in m3 is
  repeated on every cell of the body: it must not be summed over cells;
- outlet only: defined only at the outlet cell (waterBodyOut).

Tags in the comments
--------------------
INTERFACE: a value enters from, or leaves to, another module.
CAUTION (<id>): behaviour that looks unintended or is easy to get wrong.
CAUTION: (no id) the comment just above it does not match what the code does.
EXTENSION POINT: where a new release scheme or a new input would be plugged in,
and the contract it must respect.

N1 to N5 are the issues of the re-check of 2026-10-29 ("reservoir findings
re-checked on the RAWS branch"):
- N1: the reservoir reduction factor is a step at minResvrFrac, not a ramp
  (operator precedence);
- N2: the environmental-flow floor is negligible (m3/s divided by m3);
- N3: the upper-limit rule compares m3 per time step with m3/s;
- N4: cold-start transient of the long-term average outflow (its fallback for new
  reservoirs practically never fires);
- N5: allocation-zone keys of the RAWS configurations are not read, so surface
  water is allocated locally (water_management/main_water_management.py); this
  shapes the abstraction already netted out of the inflow seen here.
W1 to W11 are numbered in this file:
- W1: debugWaterBalance = "False" makes update fail (preStorage undefined);
- W2: onlyNaturalWaterBodies=True as a constructor argument without the
  configuration key makes getParameterFiles fail;
- W3: minResvrFrac / maxResvrFrac: "None" fails, spatial maps are used per cell;
- W4: the per-body capacity and surface area depend on how the input stores them;
- W5: the water-body ids used by the model are re-derived from the outlets;
- W6: the consistency check in getParameterFiles cannot fail;
- W7: timing of the information seen by the release rules;
- W8: inflow spike on the first day of new or enlarged water bodies;
- W9: every release is capped at 25 % of the storage per time step;
- W10: per-cell updates of quantities assumed uniform over a water body;
- W11: hard-coded 86400 s in the flood release.
"""

import logging

import pcraster as pcr

# vos (common/virtualOS.py) helpers used here: readPCRmapClone and
# netcdf2PCRobjClone (input reading, resampling to the clone), getFullPath,
# getMinMaxMean, getValDivZero (division with a default where the denominator is
# <= a limit), smallNumber (1e-39), secondsPerDay (86400.0) and waterBalanceCheck
# (logs an error, does not stop the run).
from pcrglobwb.common import virtualOS as vos

logger = logging.getLogger(__name__)


class WaterBodies(object):
    """Lakes and reservoirs: static maps, states and release rules.

    One instance per run, created by routing.py Routing.__init__ and kept as
    routing.WaterBodies. See the module docstring for the call sequence, the units
    and the meaning of "per cell", "per water body" and "outlet only".

    Static attributes (set in __init__):
        tmpDir, inputDir, iniItems: configuration; landmask: boolean, per cell;
        lddMap: ldd read from routingOptions["lddMap"], not clipped to the landmask;
        debugWaterBalance, onlyNaturalWaterBodies: bool; dateForNaturalCondition:
        "1900-01-01" (natural runs only); useNetCDF: bool; ncFileInp: path of the
        netCDF input, or the five file-name stems fracWaterInp, waterBodyIdsInp,
        waterBodyTypInp, resMaxCapInp, resSfAreaInp; minWeirWidth: 10 m;
        minResvrFrac (0.10) and maxResvrFrac (0.75): fractions of the capacity.

    Yearly maps (set in getParameterFiles, reloaded every 1 January):
        cellArea (m2, per cell); fracWat (fraction, per cell); waterBodyIds
        (nominal, per cell, MV outside water bodies); waterBodyOut (boolean, True at
        the outlet cell only); waterBodyArea (m2, per body); waterBodyTyp (1 lake,
        2 reservoir, per body); resMaxCap (m3, per body, MV without capacity);
        waterBodyCap (m3, per body, 0 without capacity).

    States (restart variables, also kept by routing.py Routing.getState):
        waterBodyStorage (m3, per-body total repeated on every cell of the body,
        0 elsewhere); avgInflow (m3/s, 30-day running mean of the net inflow);
        avgOutflow (m3/s, 1825-day running mean of the release).

    Results of the last call of update:
        inflow (m3 per time step), inflowInM3PerSec (m3/s), waterBodyOutflow (m3
        per time step), waterBodyBalance (m), timestepsToAvgDischarge (days).
    """

    def __init__(self, iniItems, landmask, onlyNaturalWaterBodies=False, lddMap=None):
        """Read the static options of the lake and reservoir module.

        Called once, by routing.py Routing.__init__ as WaterBodies(iniItems,
        self.landmask); the last two arguments are never passed by the model. No
        water-body map is read here; that is done by getParameterFiles.

        Args:
            iniItems: parsed configuration (configuration.py); all values are
                strings. Read: iniItems.tmpDir, globalOptions["inputDir"] and the
                [routingOptions] keys lddMap, debugWaterBalance,
                onlyNaturalWaterBodies, waterBodyInputNC, minResvrFrac,
                maxResvrFrac and, only if waterBodyInputNC is "None",
                fracWaterInp, waterBodyIds, waterBodyTyp, resMaxCapInp and
                resSfAreaInp.
            landmask: boolean map of the active cells (routing.landmask).
            onlyNaturalWaterBodies: initial value of the natural-run flag (default
                False); see W2.
            lddMap: local drainage direction map; if None (as in the model) it is
                read again from routingOptions["lddMap"].
        """
        object.__init__(self)

        # temporary directory (str), used by vos when a map must be resampled
        self.tmpDir = iniItems.tmpDir
        # input directory (str); relative input paths are resolved against it
        self.inputDir = iniItems.globalOptions["inputDir"]
        # INTERFACE: boolean map of the active cells, routing.landmask from
        # routing.py Routing.__init__; every map of this class is cropped to it
        self.landmask = landmask

        # configuration object; stored, but not read again in this file
        self.iniItems = iniItems

        # Local drainage direction (ldd). The model never passes lddMap, so this
        # branch always runs: the ldd is read again from routingOptions["lddMap"]
        # and repaired but, unlike routing.py Routing.lddMap, it is not clipped to
        # the landmask (no pcr.lddmask). It is used only in getReservoirOutflow (the
        # pcr.downstream fallback of the average outflow); getParameterFiles
        # receives the clipped routing ldd as an argument instead.
        if lddMap is None:
            self.lddMap = vos.readPCRmapClone(
                iniItems.routingOptions["lddMap"],
                self.tmpDir,
                self.inputDir,
                True,
            )
            self.lddMap = pcr.lddrepair(pcr.ldd(self.lddMap))
            self.lddMap = pcr.lddrepair(self.lddMap)
        else:
            self.lddMap = lddMap

        # option to activate the water balance check
        # True unless routingOptions["debugWaterBalance"] is exactly "False" (all
        # RAWS configurations set "True"). When True, update logs an error if the
        # storage balance of a water body does not close (logging only).
        # CAUTION (W1): with "False" the run stops in update, where preStorage is
        # used without being defined (see update).
        self.debugWaterBalance = True
        if (
            "debugWaterBalance" in list(iniItems.routingOptions.keys())
            and iniItems.routingOptions["debugWaterBalance"] == "False"
        ):
            self.debugWaterBalance = False

        # option to run with only natural lakes (without reservoirs)
        # The constructor argument (default False; routing.py never sets it) is
        # overridden to True if routingOptions["onlyNaturalWaterBodies"] is "True"
        # (all RAWS configurations set "False"). Then getParameterFiles uses the
        # water-body maps of 1900 for the whole run and turns every reservoir into a
        # lake.
        # CAUTION (W2): if the constructor argument were True without the key,
        # self.dateForNaturalCondition would be undefined and getParameterFiles
        # would fail with AttributeError.
        self.onlyNaturalWaterBodies = onlyNaturalWaterBodies
        if (
            "onlyNaturalWaterBodies" in list(iniItems.routingOptions.keys())
            and iniItems.routingOptions["onlyNaturalWaterBodies"] == "True"
        ):
            logger.info(
                "Using only natural water bodies identified in the year 1900. All reservoirs in 1900 are assumed as lakes."
            )
            self.onlyNaturalWaterBodies = True
            # a natural run only uses this date
            self.dateForNaturalCondition = "1900-01-01"

        # files with water body parameters
        # Two input modes, read every year in getParameterFiles:
        # (a) waterBodyInputNC is a netCDF file (all RAWS configurations, e.g.
        #     5arcminutes/routing/lakes_and_reservoirs.nc at 5 arcmin and
        #     30arcminutes/routing/waterBodies.nc at 30 arcmin) holding the yearly
        #     variables fracWaterInp, waterBodyIds, waterBodyTyp, resMaxCapInp and
        #     resSfAreaInp;
        # (b) waterBodyInputNC is "None": the same five quantities come from yearly
        #     PCRaster maps named <stem><year>.map, with the stems given by the keys
        #     fracWaterInp, waterBodyIds, waterBodyTyp, resMaxCapInp, resSfAreaInp.
        # The source database of these inputs is not determined from the code.
        self.useNetCDF = True
        if iniItems.routingOptions["waterBodyInputNC"] == str(None):
            self.useNetCDF = False
            # file-name stems (str); a stem "None" keeps the default of that
            # quantity (fracWat 0, no ids, type 0, capacity 0, area 0); no ids or no
            # types mean no water bodies at all
            self.fracWaterInp = iniItems.routingOptions["fracWaterInp"]
            self.waterBodyIdsInp = iniItems.routingOptions["waterBodyIds"]
            self.waterBodyTypInp = iniItems.routingOptions["waterBodyTyp"]
            self.resMaxCapInp = iniItems.routingOptions["resMaxCapInp"]
            self.resSfAreaInp = iniItems.routingOptions["resSfAreaInp"]
        else:
            self.useNetCDF = True
            # full path of the netCDF file (relative to inputDir unless absolute)
            self.ncFileInp = vos.getFullPath(
                iniItems.routingOptions["waterBodyInputNC"], self.inputDir
            )

        # minimum width (m) in the weir formula; TODO: base minWeirWidth on GLWD, GRanD and/or the bankfull discharge formula
        # (m) lower bound of the weir width of lakes in getLakeOutflow; not used
        # for reservoirs
        self.minWeirWidth = 10.0

        # storage fractions below which reservoir release stops and above which it equals
        # the long-term average outflow (defaults)
        # Fractions of the capacity waterBodyCap (dimensionless); the defaults are
        # Python floats, the same for every reservoir.
        # CAUTION (N1): as coded in getReservoirOutflow, the reduction factor is a
        # step at minResvrFrac, not a ramp up to maxResvrFrac: the base release is 0
        # below minResvrFrac * capacity and equal to the long-term average outflow
        # (capped by the average inflow) just above it. maxResvrFrac acts only in
        # the flood-release terms of getReservoirOutflow.
        self.minResvrFrac = 0.10
        self.maxResvrFrac = 0.75
        # Optional keys [routingOptions] minResvrFrac and maxResvrFrac: a number
        # (read as a uniform map) or a map / netCDF file (read per cell) through
        # vos.readPCRmapClone. No configuration file in the repository sets them.
        # CAUTION (W3): a value "None" makes vos.readPCRmapClone return None and the
        # rule then fails with a TypeError. A spatial map is used cell by cell; it is
        # not aggregated per water body, so the cells of one reservoir can get
        # different fractions and therefore different releases (see W10).
        # EXTENSION POINT: options of a new release scheme (a scheme selector,
        # parameter files) can be read here in the same way. Values of iniItems are
        # strings; keep a default for an absent key, so that existing
        # configurations still run.
        if "minResvrFrac" in list(iniItems.routingOptions.keys()):
            minResvrFrac = iniItems.routingOptions["minResvrFrac"]
            self.minResvrFrac = vos.readPCRmapClone(
                minResvrFrac, self.tmpDir, self.inputDir
            )
        if "maxResvrFrac" in list(iniItems.routingOptions.keys()):
            maxResvrFrac = iniItems.routingOptions["maxResvrFrac"]
            self.maxResvrFrac = vos.readPCRmapClone(
                maxResvrFrac, self.tmpDir, self.inputDir
            )

    def getParameterFiles(
        self,
        currTimeStep,
        cellArea,
        ldd,
        initial_condition_dictionary=None,
        currTimeStepInDateTimeFormat=False,
    ):
        """Build the lake and reservoir maps for the current year.

        Called by routing.py Routing.update at the first time step, with
        initial_condition_dictionary = Routing.getState() (which triggers getICs),
        and on 1 January of every later year, without initial conditions. The maps
        are therefore yearly: water bodies can appear, disappear, or change shape,
        type or capacity from one year to the next.

        Args:
            currTimeStep: ModelTime (common/currTimeStep.py); its fulldate
                ("YYYY-MM-DD") and year select the yearly slice. A datetime if
                currTimeStepInDateTimeFormat is True (no caller does this).
            cellArea: cell area (m2, per cell), routing.cellArea.
            ldd: routing.lddMap (clipped to the landmask); used for the outlets and
                to re-derive the water-body ids.
            initial_condition_dictionary: routing state (Routing.getState()) at the
                first time step, None afterwards.
            currTimeStepInDateTimeFormat: see currTimeStep.

        Sets fracWat, waterBodyIds, waterBodyOut, waterBodyArea, waterBodyTyp,
        waterBodyCap (all cropped to the landmask at the end), resMaxCap and
        cellArea; at the first step also waterBodyStorage, avgInflow and avgOutflow
        (through getICs).

        Order: fracWat; input ids; outlet = the cell with the most upstream cells
        per input id; ids re-derived as the sub-catchment of each outlet within the
        input water-body cells; surface area = max(fracWat-based area, reservoir
        surface area); bodies with zero area dropped; type by majority (wetlands,
        type 0, dropped); capacity; reservoirs without capacity become lakes;
        natural-run override; consistency check; initial conditions; landmask crop.
        """

        # water body parameters: fracWat, waterBodyIds, waterBodyOut, waterBodyArea, waterBodyTyp, waterBodyCap

        # INTERFACE: cell area (m2, per cell), routing.cellArea (routing.py,
        # cellAreaMap); used below for the fracWat-based water-body area
        self.cellArea = cellArea
        # routing.lddMap, already clipped to the landmask by routing.py (lddmask)
        ldd = pcr.ifthen(self.landmask, ldd)

        # date used to extract the water body information
        # date_used: "YYYY-MM-DD" (or a datetime) for the netCDF reader, which reads
        # the time step dated 1 January of that year, or else the last one before it
        # (useDoy="yearly"; a year outside the file is replaced by its first or last
        # year, with a warning; vos.netcdf2PCRobjClone). year_used: appended to the
        # PCRaster file stems.
        if currTimeStepInDateTimeFormat:
            date_used = currTimeStep
            year_used = currTimeStep.year
        else:
            date_used = currTimeStep.fulldate
            year_used = currTimeStep.year
        # natural run: always the 1900 maps (year_used becomes the string "1900")
        if self.onlyNaturalWaterBodies:
            date_used = self.dateForNaturalCondition
            year_used = self.dateForNaturalCondition[0:4]

        # fraction of surface water bodies (-)
        # fracWat: fraction of the cell area covered by lakes and reservoirs
        # (dimensionless, per cell, 0 to 1), input variable fracWaterInp.
        # INTERFACE: read by routing.py (Routing.update, simplifiedKinematicWave,
        # kinematic_wave_update: lower bound of dynamicFracWat, the open-water
        # fraction used for evaporation) and by landSurface.py (fraction FRACWAT of
        # the capillary-rise computation). On the first day landSurface.py, which
        # runs before routing, reads fracWaterInp itself and assigns
        # routing.WaterBodies.fracWat (without the 0-1 clipping below, and with the
        # model year even in a natural run); from the second day it uses the value
        # set here.
        self.fracWat = pcr.spatial(pcr.scalar(0.0))

        if self.useNetCDF:
            self.fracWat = vos.netcdf2PCRobjClone(
                self.ncFileInp,
                "fracWaterInp",
                date_used,
                useDoy="yearly",
            )
        else:
            if self.fracWaterInp != "None":
                self.fracWat = vos.readPCRmapClone(
                    self.fracWaterInp + str(year_used) + ".map",
                    self.tmpDir,
                    self.inputDir,
                )

        # missing values -> 0, then clipped to [0, 1]
        self.fracWat = pcr.cover(self.fracWat, pcr.spatial(pcr.scalar(0.0)))
        self.fracWat = pcr.max(0.0, self.fracWat)
        self.fracWat = pcr.min(1.0, self.fracWat)

        # defaults: no water body anywhere
        self.waterBodyIds = pcr.spatial(pcr.nominal(0))
        self.waterBodyOut = pcr.spatial(pcr.boolean(0))
        self.waterBodyArea = pcr.spatial(pcr.scalar(0.0))

        # Input ids (variable waterBodyIds): a positive integer per lake or
        # reservoir, on all its cells. The netCDF reader returns a scalar map; the
        # PCRaster branch reads a nominal map (isNomMap=True, values <= 0 dropped).
        if self.useNetCDF:
            self.waterBodyIds = vos.netcdf2PCRobjClone(
                self.ncFileInp,
                "waterBodyIds",
                date_used,
                useDoy="yearly",
            )
        else:
            if self.waterBodyIdsInp != "None":
                self.waterBodyIds = vos.readPCRmapClone(
                    self.waterBodyIdsInp + str(year_used) + ".map",
                    self.tmpDir,
                    self.inputDir,
                    False,
                    None,
                    True,
                )
        # keep positive ids only, as nominal; other cells become MV
        self.waterBodyIds = pcr.ifthen(
            pcr.scalar(self.waterBodyIds) > 0.0, pcr.nominal(self.waterBodyIds)
        )

        # water body outlets (correcting the outlet positions)
        # wbCatchment: number of cells upstream of each cell, itself included
        # (pcr.catchmenttotal of 1 along the clipped ldd; a cell count, not an area).
        # Candidate outlet of a body: its cell(s) with the largest count
        # (pcr.areamaximum over the input ids), normally the most downstream cell.
        # The candidates carry the input id (nominal).
        wbCatchment = pcr.catchmenttotal(pcr.scalar(1), ldd)
        # this may give more than one outlet, e.g. if several cells have the largest upstream area
        self.waterBodyOut = pcr.ifthen(
            wbCatchment == pcr.areamaximum(wbCatchment, self.waterBodyIds),
            self.waterBodyIds,
        )
        # make sure there is only one outlet per water body
        # pcr.areaorder ranks the candidates of each id by their value, which is
        # the same for all of them (the id), so the tie is broken by cell position:
        # PCRaster ranks tied cells row by row from the top-left. The first is kept.
        self.waterBodyOut = pcr.ifthen(
            pcr.areaorder(pcr.scalar(self.waterBodyOut), self.waterBodyOut) == 1.0,
            self.waterBodyOut,
        )
        # redundant guard: outlets only where the input id is positive
        self.waterBodyOut = pcr.ifthen(
            pcr.scalar(self.waterBodyIds) > 0.0, self.waterBodyOut
        )

        # TODO: also consider endorheic lakes

        # Re-derive the ids from the outlets: pcr.subcatchment gives every cell the
        # id of the first outlet downstream of it (the outlet itself included), or 0
        # if it drains through no outlet; the result is kept only on cells with a
        # positive input id. A cell of input body A that drains into the outlet of
        # body B before reaching A's outlet therefore takes B's id; cells with 0 are
        # dropped later (they get no type). After this line the id of a body is the
        # input id found at its outlet cell.
        # CAUTION (W5): the ids used by the model can differ from the input ids on
        # some cells, can change from one year to the next with the yearly maps,
        # and are not reported (variable_list.py has no entry for them). A scheme
        # keyed on an external reservoir identifier (e.g. a dam database id) must be
        # linked through the outlet cell, and the link checked every year.
        self.waterBodyIds = pcr.ifthen(
            pcr.scalar(self.waterBodyIds) > 0.0,
            pcr.subcatchment(ldd, self.waterBodyOut),
        )

        # from here waterBodyOut is boolean: True at the single outlet cell of each
        # water body, MV elsewhere (support: outlet only)
        self.waterBodyOut = pcr.ifthen(
            pcr.scalar(self.waterBodyOut) > 0.0, pcr.spatial(pcr.boolean(1))
        )

        # reservoir surface area (m2)
        # input variable resSfAreaInp times 1e6: the input is in km2 and resSfArea
        # in m2. Local variable, used only for waterBodyArea below.
        if self.useNetCDF:
            resSfArea = (
                1000.0
                * 1000.0
                * vos.netcdf2PCRobjClone(
                    self.ncFileInp,
                    "resSfAreaInp",
                    date_used,
                    useDoy="yearly",
                )
            )
        else:
            if self.resSfAreaInp != "None":
                resSfArea = (
                    1000.0
                    * 1000.0
                    * vos.readPCRmapClone(
                        self.resSfAreaInp + str(year_used) + ".map",
                        self.tmpDir,
                        self.inputDir,
                    )
                )
            else:
                resSfArea = pcr.spatial(pcr.scalar(0.0))
        # one value per body (pcr.areaaverage over the re-derived ids), 0 elsewhere.
        # CAUTION (W4): areaaverage skips MV cells but counts cells holding 0: if
        # the input stores the area on one cell of the body and 0 on its other
        # cells, the result is the area divided by the number of cells. Whether the
        # input holds MV or 0 there is not determined from the code. The capacity
        # (below) is filtered to values > 0 before averaging; the area is not.
        resSfArea = pcr.areaaverage(resSfArea, self.waterBodyIds)
        resSfArea = pcr.cover(resSfArea, 0.0)

        # water body surface area of lakes and reservoirs (m2)
        # waterBodyArea (m2, per water body) = max(sum over the body's cells of
        # fracWat * cellArea, reservoir surface area). The second areaaverage is
        # redundant. Constant within a year (it does not vary with the storage).
        # Bodies with zero area get MV. It is not masked again by the type filters
        # below, so it can stay defined on cells of bodies dropped later (e.g.
        # wetlands); every use is masked by waterBodyIds, so this is harmless.
        # Not read outside this file: used for the lake head (getLakeOutflow) and to
        # express the balance check in m (update).
        self.waterBodyArea = pcr.max(
            pcr.areatotal(
                pcr.cover(self.fracWat * self.cellArea, 0.0), self.waterBodyIds
            ),
            pcr.areaaverage(pcr.cover(resSfArea, 0.0), self.waterBodyIds),
        )
        self.waterBodyArea = pcr.ifthen(self.waterBodyArea > 0.0, self.waterBodyArea)

        # exclude all water bodies with zero surface area
        self.waterBodyIds = pcr.ifthen(self.waterBodyArea > 0.0, self.waterBodyIds)
        self.waterBodyOut = pcr.ifthen(
            pcr.boolean(self.waterBodyIds), self.waterBodyOut
        )

        # water body types: 2 = reservoir (regulated discharge), 1 = lake (weir formula),
        # 0 = neither (e.g. wetland)
        # waterBodyTyp (nominal, per water body): 1 lake, 2 reservoir, MV elsewhere.
        # Input variable waterBodyTyp (the netCDF reader returns a scalar map).
        self.waterBodyTyp = pcr.nominal(0)

        if self.useNetCDF:
            self.waterBodyTyp = vos.netcdf2PCRobjClone(
                self.ncFileInp,
                "waterBodyTyp",
                date_used,
                useDoy="yearly",
            )
        else:
            if self.waterBodyTypInp != "None":
                self.waterBodyTyp = vos.readPCRmapClone(
                    self.waterBodyTypInp + str(year_used) + ".map",
                    self.tmpDir,
                    self.inputDir,
                    False,
                    None,
                    True,
                )

        # exclude wetlands (type 0) from all lake/reservoir functions
        self.waterBodyTyp = pcr.ifthen(
            pcr.scalar(self.waterBodyTyp) > 0, pcr.nominal(self.waterBodyTyp)
        )
        # types only on cells with a re-derived id
        self.waterBodyTyp = pcr.ifthen(
            pcr.scalar(self.waterBodyIds) > 0, pcr.nominal(self.waterBodyTyp)
        )
        # choose one type per water body: lake or reservoir
        # pcr.areamajority: the most frequent type over the body's cells, assigned
        # to all its cells; on a tie PCRaster takes the highest class, i.e.
        # reservoir (2) over lake (1)
        self.waterBodyTyp = pcr.areamajority(self.waterBodyTyp, self.waterBodyIds)
        self.waterBodyTyp = pcr.ifthen(
            pcr.scalar(self.waterBodyTyp) > 0, pcr.nominal(self.waterBodyTyp)
        )
        self.waterBodyTyp = pcr.ifthen(
            pcr.boolean(self.waterBodyIds), self.waterBodyTyp
        )

        # correct the lake and reservoir ids and outlets
        # keep only bodies of type 1 or 2 (wetlands and id-0 cells are dropped)
        self.waterBodyIds = pcr.ifthen(
            pcr.scalar(self.waterBodyTyp) > 0, self.waterBodyIds
        )
        self.waterBodyOut = pcr.ifthen(
            pcr.scalar(self.waterBodyIds) > 0, self.waterBodyOut
        )

        # reservoir maximum capacity (m3)
        # input variable resMaxCapInp times 1e6: the input is in million m3 and
        # resMaxCap in m3
        self.resMaxCap = pcr.scalar(0.0)
        self.waterBodyCap = pcr.scalar(0.0)

        if self.useNetCDF:
            self.resMaxCap = (
                1000.0
                * 1000.0
                * vos.netcdf2PCRobjClone(
                    self.ncFileInp,
                    "resMaxCapInp",
                    date_used,
                    useDoy="yearly",
                )
            )
        else:
            if self.resMaxCapInp != "None":
                self.resMaxCap = (
                    1000.0
                    * 1000.0
                    * vos.readPCRmapClone(
                        self.resMaxCapInp + str(year_used) + ".map",
                        self.tmpDir,
                        self.inputDir,
                    )
                )

        # keep positive values, then one value per body: the average over the
        # body's cells that have a positive value (resMaxCap: m3, per body, MV for
        # bodies without capacity; not cropped to the landmask, not read outside
        # this file).
        # CAUTION (W4): this is the body's capacity if the input repeats it on the
        # body's cells or stores it on one cell; it would be the mean cell value if
        # the input split the capacity over the cells. The convention of the input
        # file is not determined from the code. The capacity is re-read every year,
        # so it can change between years.
        self.resMaxCap = pcr.ifthen(self.resMaxCap > 0.0, self.resMaxCap)
        self.resMaxCap = pcr.areaaverage(self.resMaxCap, self.waterBodyIds)

        # water body capacity of lakes and reservoirs (m3); most lakes have a capacity > 0
        # waterBodyCap (m3, per water body; 0 for bodies without capacity). For
        # reservoirs it is the reference of minResvrFrac and maxResvrFrac; for lakes
        # it acts as the storage below the weir sill (getLakeOutflow uses the head
        # (storage - waterBodyCap) / waterBodyArea).
        # EXTENSION POINT: static per-reservoir inputs of a new scheme (rule-curve
        # parameters, storage targets, trained coefficients, a link to an external
        # reservoir id) can be loaded here in the same way: a netCDF variable read
        # with useDoy="yearly" (or yearly maps), aggregated per body over
        # self.waterBodyIds (areaaverage or areamaximum, after the ids have been
        # re-derived above), covered with a neutral value, and cropped to the
        # landmask with the other maps at the end of this method. Like everything
        # here, they are reloaded every 1 January.
        self.waterBodyCap = pcr.cover(self.resMaxCap, 0.0)
        self.waterBodyCap = pcr.ifthen(
            pcr.boolean(self.waterBodyIds), self.waterBodyCap
        )

        # reservoirs with zero capacity are assumed to be lakes
        self.waterBodyTyp = pcr.ifthen(
            pcr.scalar(self.waterBodyTyp) > 0.0, self.waterBodyTyp
        )
        # type 2 with waterBodyCap == 0 becomes type 1; other types are kept
        self.waterBodyTyp = pcr.ifthenelse(
            self.waterBodyCap > 0.0,
            self.waterBodyTyp,
            pcr.ifthenelse(
                pcr.scalar(self.waterBodyTyp) == 2, pcr.nominal(1), self.waterBodyTyp
            ),
        )

        # final corrections: all lakes and reservoirs must have surface areas
        self.waterBodyTyp = pcr.ifthen(self.waterBodyArea > 0.0, self.waterBodyTyp)
        # only types 1 and 2 are considered in the lake/reservoir functions
        self.waterBodyTyp = pcr.ifthen(
            pcr.scalar(self.waterBodyTyp) > 0.0, self.waterBodyTyp
        )
        # all lakes and reservoirs must have ids
        self.waterBodyIds = pcr.ifthen(
            pcr.scalar(self.waterBodyTyp) > 0.0, self.waterBodyIds
        )
        # all lakes and reservoirs must have outlets
        self.waterBodyOut = pcr.ifthen(
            pcr.scalar(self.waterBodyIds) > 0.0, self.waterBodyOut
        )

        # a natural run (onlyNaturalWaterBodies) only uses the year 1900: all reservoirs are lakes
        # date_used always equals dateForNaturalCondition in a natural run (set at
        # the top of this method), so the second test is redundant
        if self.onlyNaturalWaterBodies and date_used == self.dateForNaturalCondition:
            logger.info(
                "Using only natural water bodies identified in the year 1900. All reservoirs in 1900 are assumed as lakes."
            )
            self.waterBodyTyp = pcr.ifthen(
                pcr.scalar(self.waterBodyTyp) > 0.0, pcr.nominal(1)
            )

        # check that all lakes and reservoirs have types, ids, surface areas and outlets
        # test: True on cells of bodies with a type, an area, an id and an outlet.
        # CAUTION (W6): the check cannot fail. Type, area and id are defined on all
        # remaining body cells by construction; and for a body without an outlet
        # the areamaximum is MV, so test is MV there (PCRaster's & propagates MV)
        # and pcr.cover(..., 1.0) counts it as passing. By construction every body
        # does keep its outlet, so this hides no current error, but the warning is
        # no safeguard against changes to the code above.
        test = (
            pcr.defined(self.waterBodyTyp)
            & pcr.defined(self.waterBodyArea)
            & pcr.defined(self.waterBodyIds)
            & pcr.boolean(
                pcr.areamaximum(pcr.scalar(self.waterBodyOut), self.waterBodyIds)
            )
        )
        # min and max of (test - 1) over the map: both 0 if every cell passes
        a, b, c = vos.getMinMaxMean(pcr.cover(pcr.scalar(test), 1.0) - pcr.scalar(1.0))
        threshold = 1e-3
        if abs(a) > threshold or abs(b) > threshold:
            logger.warning("Missing information in some lakes and/or reservoirs.")

        # get the initial conditions at the first time step
        # INTERFACE: at the first time step routing.py Routing.update passes
        # Routing.getState() (keys avgLakeReservoirInflowShort,
        # avgLakeReservoirOutflowLong, waterBodyStorage, channelStorage, ...). getICs
        # runs here, after the ids are final, because its fallback uses them.
        if initial_condition_dictionary is not None and currTimeStep.timeStepPCR == 1:
            self.getICs(initial_condition_dictionary)

        # initialize storage and average inflow and outflow for new reservoirs (introduced at the
        # beginning of the year)
        # CAUTION (N4): pcr.cover only replaces missing values, and these three
        # maps are defined on the whole landmask (they are updated on every cell,
        # inside and outside water bodies), so nothing is initialised here for a
        # reservoir that appears in a later year: its cells keep their values,
        # i.e. storage 0 if they were not in a water body, and avgInflow and
        # avgOutflow decayed towards 0 but positive (or the values of the body the
        # cells belonged to, e.g. a lake turned into a reservoir). The next update
        # then counts the channel storage of the new cells as inflow (W8), and the
        # avgOutflow fallback of getReservoirOutflow does not fire.
        # The try/except silences the AttributeError raised when the states do not
        # exist yet (on master it was meant for offline MODFLOW runs); in the RAWS
        # model getICs always runs first, so it can only hide errors.
        try:
            self.waterBodyStorage = pcr.cover(self.waterBodyStorage, 0.0)
            self.avgInflow = pcr.cover(self.avgInflow, 0.0)
            self.avgOutflow = pcr.cover(self.avgOutflow, 0.0)
            self.waterBodyStorage = pcr.ifthen(self.landmask, self.waterBodyStorage)
            self.avgInflow = pcr.ifthen(self.landmask, self.avgInflow)
            self.avgOutflow = pcr.ifthen(self.landmask, self.avgOutflow)
        except Exception:
            pass
        # TODO: remove try/except

        # crop to the landmask
        # INTERFACE: the maps below are read by routing.py (waterBodyIds and
        # waterBodyOut in many places: storage transfer, release placement,
        # characteristic distances, discharge on water-body cells; fracWat for the
        # open-water fraction), by reporting.py (waterBodyIds, to mask the reported
        # storage) and by landSurface.py (fracWat). If includeWaterBodies is False,
        # routing.py Routing.update replaces waterBodyIds by nominal -1 on every
        # cell right after this method, which disables every water body (the
        # computations here test scalar(waterBodyIds) > 0).
        self.fracWat = pcr.ifthen(self.landmask, self.fracWat)
        self.waterBodyIds = pcr.ifthen(self.landmask, self.waterBodyIds)
        self.waterBodyOut = pcr.ifthen(self.landmask, self.waterBodyOut)
        self.waterBodyArea = pcr.ifthen(self.landmask, self.waterBodyArea)
        self.waterBodyTyp = pcr.ifthen(self.landmask, self.waterBodyTyp)
        self.waterBodyCap = pcr.ifthen(self.landmask, self.waterBodyCap)

    def getICs(self, initial_condition):
        """Set the three water-body states from the routing initial conditions.

        Called only by getParameterFiles at the first time step (timeStepPCR == 1),
        after the ids have been built.

        Args:
            initial_condition: dict produced by routing.py Routing.getState(). Keys
                read: avgLakeReservoirInflowShort (m3/s), avgLakeReservoirOutflowLong
                (m3/s), waterBodyStorage (m3, per water body, or None) and, only if
                waterBodyStorage is None, channelStorage (m3, per cell).

        Sets avgInflow, avgOutflow and waterBodyStorage (MV replaced by 0, cropped
        to the landmask). The values are used as they are: a warm state is uniform
        per water body only if it comes from a run with the same water-body maps;
        a configured constant gives the same value on every cell (all RAWS
        configurations use 0.001 for the three ...Ini keys, i.e. a storage of
        0.001 m3 per body and averages of 0.001 m3/s: a cold start, see N4).
        """

        # INTERFACE: restart states, written by routing.py Routing.getState and read
        # by Routing.getICs from the configuration keys avgLakeReservoirInflowShortIni,
        # avgLakeReservoirOutflowLongIni and waterBodyStorageIni (or from memory
        # after a spin-up).
        avgInflow = initial_condition["avgLakeReservoirInflowShort"]
        avgOutflow = initial_condition["avgLakeReservoirOutflowLong"]

        # The fallback (else) needs waterBodyStorage to be Python None in the dict.
        # routing.py sets it to None only if routingOptions["waterBodyStorageIni"] is
        # None, which a configuration string never is (a value "None" makes
        # routing.py fail earlier, in pcr.cover), so the fallback is not reached in
        # practice.
        if initial_condition["waterBodyStorage"] is not None:
            waterBodyStorage = initial_condition["waterBodyStorage"]
        else:
            # waterBodyStorage at the lake and reservoir cells
            # fallback: storage = sum over the body's cells of the channel storage
            # (m3 per cell), negative values set to 0, rounded down to whole m3
            storageAtLakeAndReservoirs = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.waterBodyIds) > 0.0,
                    initial_condition["channelStorage"],
                ),
                0.0,
            )
            # only non-negative, rounded-down values
            storageAtLakeAndReservoirs = pcr.max(
                0.00, pcr.rounddown(storageAtLakeAndReservoirs)
            )
            # lake and reservoir storage (m3), given for all lake/reservoir cells
            waterBodyStorage = pcr.ifthen(
                pcr.scalar(self.waterBodyIds) > 0.0,
                pcr.areatotal(storageAtLakeAndReservoirs, self.waterBodyIds),
            )

        # EXTENSION POINT: a state of a new scheme (e.g. a running mean of a gross
        # inflow, a hedging memory, the previous release) must be read here from a
        # new key of initial_condition, added to routing.py Routing.getState (so
        # that it is written to the restart files) and Routing.getICs (new ...Ini
        # configuration key), and covered with a neutral value for new reservoirs
        # in getParameterFiles, like avgInflow and avgOutflow.
        # (m3/s)
        # avgInflow: 30-day running mean of the net inflow (moveFromChannelToWaterBody)
        self.avgInflow = pcr.cover(avgInflow, 0.0)
        # (m3/s)
        # avgOutflow: 1825-day running mean of the release (getWaterBodyOutflow)
        self.avgOutflow = pcr.cover(avgOutflow, 0.0)
        # (m3)
        # waterBodyStorage: storage per water body (the total, repeated on its cells)
        self.waterBodyStorage = pcr.cover(waterBodyStorage, 0.0)

        # MV -> 0 above, then MV outside the landmask
        self.avgInflow = pcr.ifthen(self.landmask, self.avgInflow)
        self.avgOutflow = pcr.ifthen(self.landmask, self.avgOutflow)
        self.waterBodyStorage = pcr.ifthen(self.landmask, self.waterBodyStorage)

    def update(
        self,
        newStorageAtLakeAndReservoirs,
        timestepsToAvgDischarge,
        maxTimestepsToAvgDischargeShort,
        maxTimestepsToAvgDischargeLong,
        currTimeStep,
        avgChannelDischarge,
        length_of_time_step=vos.secondsPerDay(),
        downstreamDemand=None,
    ):
        """Advance the lakes and reservoirs by one time step.

        INTERFACE: called by routing.py Routing.simple_update once per day
        (routingMethod accuTravelTime or simplifiedKinematicWave, used by all RAWS
        configurations) with length_of_time_step = 86400 s, and by
        Routing.kinematic_wave_update once per sub-step (routingMethod
        kinematicWave, used by no RAWS configuration) with the sub-step length.

        Order: moveFromChannelToWaterBody (net inflow, avgInflow, storage before
        release), then getWaterBodyOutflow (release, 25 % cap, avgOutflow, storage
        after release), then the optional balance check.

        Args:
            newStorageAtLakeAndReservoirs: channel storage of the water-body cells
                (m3 per cell, 0 elsewhere; negative values set to 0 and rounded down
                to whole m3 by routing.py), which the caller removes from the
                channel. With the daily methods it holds: the body's storage after
                the previous release and the river water left on the body's cells
                after the previous routing (both put back into the channel by
                Routing.return_water_body_storage_to_channel), minus the previous
                step's riverbed seepage, plus this step's local terms on those
                cells (runoff and baseflow, minus surface-water abstraction, plus
                non-irrigation return flows, minus open-water evaporation) and, only
                for sub-runs with upstream_discharge_input_files (no RAWS
                configuration), inflow from other sub-runs.
            timestepsToAvgDischarge: day counter k of the running averages (spatial
                scalar, the same on every cell, restart state of routing.py; >= 1
                here, because routing.py adds 1 before this call).
            maxTimestepsToAvgDischargeShort: window N_s of avgInflow (30 days, set
                in routing.py Routing.__init__).
            maxTimestepsToAvgDischargeLong: window N_l of avgOutflow (1825 days).
            currTimeStep: ModelTime; only its date is used (balance-check log).
            avgChannelDischarge: long-term average channel discharge (m3/s, per
                cell), routing.avgDischarge; only a fallback for avgOutflow.
            length_of_time_step: dt (s).
            downstreamDemand: environmental-flow discharge (m3/s, per cell) from
                routing.py Routing.estimate_discharge_for_environmental_flow,
                max(0, avgDischarge - 1.2816 sd, 0.1 avgDischarge); not a
                withdrawal demand.

        Sets timestepsToAvgDischarge, inflow, inflowInM3PerSec, avgInflow,
        waterBodyOutflow, avgOutflow, waterBodyStorage and waterBodyBalance.
        Returns None.
        """

        if self.debugWaterBalance:
            # (m)
            # CAUTION (W1): the unit is m3, not m: the per-body storage before this
            # step. preStorage is defined only when debugWaterBalance is True, but
            # the waterBodyBalance expression at the end of this method uses it
            # unconditionally: with debugWaterBalance = "False" the run stops there
            # with UnboundLocalError.
            preStorage = self.waterBodyStorage

        # INTERFACE: day counter of the running averages (routing.py state
        # timestepsToAvgDischarge, incremented by 1 at the start of the routing
        # step). Stored because moveFromChannelToWaterBody and getWaterBodyOutflow
        # read self.timestepsToAvgDischarge rather than an argument.
        # CAUTION (W7): timing of what the release rules see (daily methods). The
        # storage already contains this step's local inflow on the body's cells, and
        # avgInflow includes this step's net inflow (closer to perfect foresight
        # than to a rule that knows only the previous day); river water from
        # upstream that reaches the body during this step's routing enters the
        # storage only at the next step. The release decided now is routed in this
        # step. A scheme trained on observed daily series must be fed with this
        # alignment in mind.
        # EXTENSION POINT: new inputs of a scheme (e.g. the per-body abstraction,
        # open-water evaporation and seepage needed for a gross inflow, or an
        # upstream inflow) are added as keyword arguments of update with a default
        # of None, computed per water body in routing.py (pcr.areatotal over
        # waterBodyIds) and passed by both callers, Routing.simple_update and
        # Routing.kinematic_wave_update, the way downstreamDemand is.
        self.timestepsToAvgDischarge = (
            # TODO: include this in currTimeStep
            timestepsToAvgDischarge
        )

        # step 1: net inflow, 30-day average inflow, storage before release
        self.moveFromChannelToWaterBody(
            newStorageAtLakeAndReservoirs,
            timestepsToAvgDischarge,
            maxTimestepsToAvgDischargeShort,
            length_of_time_step,
        )

        # step 2: release, 25 % cap, 1825-day average outflow, storage after release
        self.getWaterBodyOutflow(
            maxTimestepsToAvgDischargeLong,
            avgChannelDischarge,
            length_of_time_step,
            downstreamDemand,
        )

        # Balance check per cell, in m of water over the body's area: inflow -
        # outflow - (storage after - storage before), each divided by waterBodyArea
        # (MV outside bodies, covered with 0). vos.waterBalanceCheck logs an error
        # if the minimum or maximum exceeds 5e-3 m and does not stop the run. The
        # balance closes by construction (the inflow is defined as a storage
        # difference), up to 32-bit rounding.
        if self.debugWaterBalance:
            vos.waterBalanceCheck(
                [pcr.cover(self.inflow / self.waterBodyArea, 0.0)],
                [pcr.cover(self.waterBodyOutflow / self.waterBodyArea, 0.0)],
                [pcr.cover(preStorage / self.waterBodyArea, 0.0)],
                [pcr.cover(self.waterBodyStorage / self.waterBodyArea, 0.0)],
                "WaterBodyStorage (unit: m)",
                True,
                currTimeStep.fulldate,
                threshold=5e-3,
            )

        # waterBodyBalance (m, per cell): the same residual, kept for diagnostics.
        # INTERFACE: copied to routing.waterBodyBalance by routing.py
        # Routing.update; no other module reads it.
        # CAUTION (W1): preStorage is undefined here if debugWaterBalance is False.
        self.waterBodyBalance = (
            pcr.cover(self.inflow / self.waterBodyArea, 0.0)
            - pcr.cover(self.waterBodyOutflow / self.waterBodyArea, 0.0)
        ) - (
            pcr.cover(self.waterBodyStorage / self.waterBodyArea, 0.0)
            - pcr.cover(preStorage / self.waterBodyArea, 0.0)
        )

    def moveFromChannelToWaterBody(
        self,
        newStorageAtLakeAndReservoirs,
        timestepsToAvgDischarge,
        maxTimestepsToAvgDischargeShort,
        length_of_time_step=vos.secondsPerDay(),
    ):
        """Compute the net inflow of every water body and its 30-day average.

        Called by update (step 1), before getWaterBodyOutflow.

        Args:
            newStorageAtLakeAndReservoirs: m3 per cell, see update.
            timestepsToAvgDischarge: not used; the method reads
                self.timestepsToAvgDischarge (the same value, set in update).
            maxTimestepsToAvgDischargeShort: averaging window N_s (30 days).
            length_of_time_step: dt (s).

        With S_prev the storage after the previous release (per body) and S_new the
        sum of newStorageAtLakeAndReservoirs over the body's cells:

            inflow           = S_new - S_prev           (m3 per time step)
            inflowInM3PerSec = inflow / dt              (m3/s)
            n                = max(1, min(N_s, k - 1 + dt / 86400))
            avgInflow       <- max(0, avgInflow
                                      + (inflowInM3PerSec - avgInflow)
                                      * (dt / 86400) / n)
            waterBodyStorage <- S_new

        k is timestepsToAvgDischarge. With daily steps n = min(30, k): a cumulative
        mean over the first 30 days of the averaging period, then an exponential
        moving average with weight 1/30. The inflow is a net residual and can be
        negative; avgInflow is floored at 0.
        """

        # new lake and reservoir storage (m3)
        # per-body sum of the channel storage moved in by routing.py (m3), repeated
        # on every cell of the body; 0 outside water bodies
        newStorageAtLakeAndReservoirs = pcr.cover(
            pcr.areatotal(newStorageAtLakeAndReservoirs, self.waterBodyIds), 0.0
        )

        # incoming volume (m3)
        # Net inflow (m3 per time step, per water body; 0 outside bodies, except on
        # cells that left a body this year, where it is minus their old storage).
        # It is a residual, not a routed flux: net of surface-water abstraction on
        # the body's cells (a local allocation with the standard RAWS
        # configurations, re-check N5), of open-water evaporation and of the
        # previous step's riverbed seepage; it includes local runoff and baseflow,
        # non-irrigation return flows on those cells and the river water that
        # reached the body's cells in the previous routing step. It can be negative.
        # CAUTION (W8): on the first step of a body that appears in a later year,
        # or of cells added to a body, the old storage of the new cells is 0, so
        # their whole channel storage counts as inflow on that day (a one-day spike
        # in inflow and avgInflow). When cells are added to an existing body, the
        # inflow also differs between its cells on that day, and so does avgInflow
        # afterwards (W10).
        self.inflow = newStorageAtLakeAndReservoirs - self.waterBodyStorage

        # TODO: check whether this inflow includes evaporation losses
        # Answer to the TODO above, from routing.py: yes. Routing.calculate_evaporation
        # removes the open-water evaporation from the channel storage of every cell,
        # the body's cells included, before routing.py moves that storage here.

        # (m3/s)
        # INTERFACE: reported as lake_and_reservoir_inflow by reporting.py (per
        # body, repeated on the body's cells; under kinematicWave only the last
        # sub-step's value). The rules use it only through avgInflow.
        self.inflowInM3PerSec = self.inflow / length_of_time_step

        # update the (short-term) average inflow (m3/s), needed to constrain the lake outflow;
        # see the "weighted incremental algorithm" in http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        # temp = n, the number of days in the running mean:
        # max(1, min(N_s, k - 1 + dt/86400)) with N_s = 30 and k the day counter;
        # with daily steps n = min(30, k). avgInflow is used by both rules: the lake
        # release is at least avgInflow (getLakeOutflow); the base reservoir release
        # is at most avgInflow * dt, and the N3 floor uses it (getReservoirOutflow).
        temp = pcr.max(
            1.0,
            pcr.min(
                maxTimestepsToAvgDischargeShort,
                self.timestepsToAvgDischarge
                - 1.0
                + length_of_time_step / vos.secondsPerDay(),
            ),
        )
        # deviation of this step's inflow from the running mean (m3/s)
        deltaInflow = self.inflowInM3PerSec - self.avgInflow
        # increment: weighted by the step length in days (dt/86400), divided by n
        R = deltaInflow * (length_of_time_step / vos.secondsPerDay()) / temp
        self.avgInflow = self.avgInflow + R
        # floored at 0 (the inflow itself can be negative). Updated on every cell:
        # outside bodies the inflow is 0, so avgInflow decays towards 0 there.
        self.avgInflow = pcr.max(0.0, self.avgInflow)

        # storage before release (m3, per water body), including this step's inflow;
        # this is the S seen by getLakeOutflow and getReservoirOutflow.
        # EXTENSION POINT: a decision inflow other than this net residual (e.g. a
        # gross inflow that adds back abstraction, evaporation and seepage passed in
        # through update) and its running mean would be computed here with the same
        # incremental formula; a new running mean is a state (see getICs).
        self.waterBodyStorage = newStorageAtLakeAndReservoirs

    def getWaterBodyOutflow(
        self,
        maxTimestepsToAvgDischargeLong,
        avgChannelDischarge,
        length_of_time_step=vos.secondsPerDay(),
        downstreamDemand=None,
    ):
        """Compute the release of every lake and reservoir and update the storage.

        Called by update (step 2), after moveFromChannelToWaterBody.

        Args:
            maxTimestepsToAvgDischargeLong: averaging window N_l of avgOutflow
                (1825 days).
            avgChannelDischarge: routing.avgDischarge (m3/s, per cell), passed on to
                the two rules as a fallback.
            length_of_time_step: dt (s).
            downstreamDemand: environmental-flow discharge (m3/s, per cell), passed
                to getReservoirOutflow; 0 if None.

        With S the storage after this step's inflow (m3, per body):

            Q_lake = getLakeOutflow(...)        m3, defined where waterBodyTyp == 1
            Q_res  = getReservoirOutflow(...)   m3, defined where waterBodyTyp == 2
            Q = max(0, cover(cover(Q_res, Q_lake), 0))
            Q = rounddown(min(0.25 S, Q))       whole m3 per time step
            n = max(1, min(N_l, k - 1 + dt / 86400))
            avgOutflow <- max(0, avgOutflow + (Q / dt - avgOutflow)
                                               * (dt / 86400) / n)
            S <- max(0, S - Q)

        Sets waterBodyOutflow (m3 per time step), avgOutflow (m3/s) and
        waterBodyStorage (m3).
        """

        # lake outflow (m3)
        # m3 per time step, defined only on lake cells (waterBodyTyp == 1)
        lakeOutflow = self.getLakeOutflow(avgChannelDischarge, length_of_time_step)

        # reservoir outflow (m3)
        # EXTENSION POINT: this is where a new reservoir release scheme is plugged
        # in, e.g. by branching on a configuration option read in __init__ and
        # calling another method instead of getReservoirOutflow. Contract of the
        # returned map, so that everything downstream keeps working:
        # - unit: m3 released during this time step (not m3/s), >= 0;
        # - support: one value per water body, the same on all cells of the body
        #   (routing.py takes the value at the outlet cell as the release, while
        #   self.waterBodyStorage is reduced cell by cell below, see W10);
        # - masking: defined only where waterBodyTyp == 2 and waterBodyIds > 0, MV
        #   elsewhere, so that pcr.cover below falls back to the lake outflow;
        # - clamping: <= self.waterBodyStorage (the storage after this step's
        #   inflow); the result is then capped at 25 % of the storage and rounded
        #   down to whole m3 below (W9), whatever the scheme decides.
        # Inputs available here: self.waterBodyStorage (S, m3, after inflow),
        # self.waterBodyCap, self.waterBodyArea, self.avgInflow, self.avgOutflow,
        # self.inflow and self.inflowInM3PerSec (this step's net inflow),
        # self.minResvrFrac, self.maxResvrFrac, avgChannelDischarge,
        # downstreamDemand and length_of_time_step. Any other model variable must
        # be passed in through update from routing.py (simple_update and
        # kinematic_wave_update), as downstreamDemand is.
        # routing.py always passes downstreamDemand; 0 is a default for other callers
        if downstreamDemand is None:
            downstreamDemand = pcr.scalar(0.0)
        reservoirOutflow = self.getReservoirOutflow(
            avgChannelDischarge, length_of_time_step, downstreamDemand
        )

        # outflow from lakes and reservoirs
        # reservoir value where defined, otherwise the lake value; MV elsewhere
        self.waterBodyOutflow = pcr.cover(reservoirOutflow, lakeOutflow)

        # make sure all water bodies have an outflow
        # MV -> 0 (cells outside water bodies get 0), negative values -> 0
        self.waterBodyOutflow = pcr.max(0.0, pcr.cover(self.waterBodyOutflow, 0.0))

        # limit the outflow to the available storage, to avoid flip-flopping
        # CAUTION (W9): the limit is 25 % of the storage, not the storage: at most a
        # quarter of S (the storage after this step's inflow) leaves per time step,
        # for lakes and reservoirs alike, whatever the rule decided, flood release
        # included. Under kinematicWave it applies per sub-step. A scheme whose
        # release exceeds 0.25 S (e.g. emptying a small reservoir or passing a flood
        # peak through a full one) is clipped here without notice.
        # EXTENSION POINT: to let a new scheme's release through unchanged, the cap
        # can be restricted to the water bodies that use the native rules (e.g. with
        # pcr.ifthenelse on waterBodyTyp or on a scheme mask); the clamp to the
        # storage (release <= S) must be kept.
        factor = 0.25
        # (m3)
        self.waterBodyOutflow = pcr.min(
            self.waterBodyStorage * factor, self.waterBodyOutflow
        )
        # round values (m3)
        # rounded down to whole m3. PCRaster scalars are 32-bit floats, so for
        # releases above 2**24 m3 (about 1.7e7 m3) per step the spacing of
        # representable values exceeds 1 m3 and the rounding is below precision.
        self.waterBodyOutflow = pcr.rounddown(self.waterBodyOutflow / 1.0) * 1.0

        # (m3/s)
        # release as a mean rate over the time step (m3/s)
        waterBodyOutflowInM3PerSec = self.waterBodyOutflow / length_of_time_step

        # update the (long-term) average outflow (m3/s), needed to constrain the reservoir outflow;
        # see the "weighted incremental algorithm" in http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        # Same weighted incremental update as avgInflow, with N_l = 1825 days:
        # n = max(1, min(N_l, k - 1 + dt/86400)). Computed on every landmask cell;
        # outside water bodies the release is 0, so avgOutflow decays there towards
        # 0 but stays positive (N4). avgOutflow is used by getReservoirOutflow (base
        # release, flood release) and by getLakeOutflow (weir width).
        temp = pcr.max(
            1.0,
            pcr.min(
                maxTimestepsToAvgDischargeLong,
                self.timestepsToAvgDischarge
                - 1.0
                + length_of_time_step / vos.secondsPerDay(),
            ),
        )
        # deviation of this step's release rate from the running mean (m3/s)
        deltaOutflow = waterBodyOutflowInM3PerSec - self.avgOutflow
        R = deltaOutflow * (length_of_time_step / vos.secondsPerDay()) / temp
        self.avgOutflow = self.avgOutflow + R
        self.avgOutflow = pcr.max(0.0, self.avgOutflow)

        # update waterBodyStorage after outflow
        # S - Q (m3); the max(0, ...) cannot bind, since Q <= 0.25 S.
        # INTERFACE: routing.py copies this storage to routing.waterBodyStorage
        # (restart state waterBodyStorage), returns it to the channel storage of the
        # body's cells (Routing.return_water_body_storage_to_channel, which takes
        # the pcr.areaaverage of this map over the body), and reporting.py reports
        # it as waterBodyStorage (the per-body total repeated on every cell).
        # CAUTION (W10): the update is done cell by cell. The model assumes that the
        # storage, avgInflow, avgOutflow and the release are uniform over the cells
        # of a body, but nothing enforces it: downstreamDemand and
        # avgChannelDischarge are per cell, minResvrFrac and maxResvrFrac can be
        # per-cell maps (W3), and avgInflow becomes non-uniform when cells are added
        # to a body (W8) or when a warm state comes from other water-body maps.
        # Where the release differs between cells, routing.py releases the outlet's
        # value but puts back the cell average of this storage, so the water total
        # changes by (outlet release - mean release over the body's cells). With the
        # native rule the amounts are small (mainly through the negligible floor of
        # N2); a new scheme should compute one release per body.
        self.waterBodyStorage = self.waterBodyStorage - self.waterBodyOutflow
        self.waterBodyStorage = pcr.max(0.0, self.waterBodyStorage)

    # (m3/s)
    def weirFormula(self, waterHeight, weirWidth):
        """Broad-crested weir discharge (m3/s).

        Q = 1.7 * C_w * max(0, h - z_sill)**1.5 * B, with C_w = weirCoef = 1,
        z_sill = sillElev = 0 m, h = waterHeight (m) and B = weirWidth (m). The
        constant 1.7 is close to (2/3)**1.5 * sqrt(g) = 1.705 m**0.5/s, the textbook
        broad-crested weir coefficient. Used only by getLakeOutflow; reservoirs do
        not use it.
        """
        sillElev = pcr.scalar(0.0)
        weirCoef = pcr.scalar(1.0)
        weirFormula = (
            1.7 * weirCoef * pcr.max(0, waterHeight - sillElev) ** 1.5
        ) * weirWidth
        return weirFormula

    def getLakeOutflow(
        self, avgChannelDischarge, length_of_time_step=vos.secondsPerDay()
    ):
        """Release of the lakes (waterBodyTyp == 1) for this time step.

        Called by getWaterBodyOutflow. Returns m3 per time step, defined only on
        lake cells (MV elsewhere). With S the storage after this step's inflow (m3),
        C = waterBodyCap (m3, the storage below the sill; 0 for lakes without
        capacity data) and A = waterBodyArea (m2):

            h      = max(0.001, (S - C) / A)                              (m)
            Qbar   = avgOutflow if > 0, else max(avgChannelDischarge,
                     avgInflow, 0.001); then the maximum over the body   (m3/s)
            B      = max(4.8 * Qbar**0.5, minWeirWidth = 10)              (m)
            Q_lake = min(S, max(1.7 * h**1.5 * B, avgInflow) * dt)        (m3)

        The release is at least the 30-day average net inflow (if the storage
        allows), and the 0.001 m minimum head gives a small weir flow even below the
        sill. getWaterBodyOutflow then caps it at 0.25 S (W9).
        """

        # water height (m), a function of storage; Rens used 0.001 m as minimum to make sure there
        # is always lake outflow, but it is still limited by the available waterBodyStorage
        minWaterHeight = 0.001
        # waterHeight (m, per water body): head above the sill, (S - C) / A with
        # C = waterBodyCap, at least minWaterHeight; MV outside bodies -> 0
        waterHeight = pcr.cover(
            pcr.max(
                minWaterHeight,
                (self.waterBodyStorage - pcr.cover(self.waterBodyCap, 0.0))
                / self.waterBodyArea,
            ),
            0.0,
        )

        # weir width (m), estimated from avgOutflow (m3/s) using the bankfull discharge formula
        avgOutflow = self.avgOutflow
        # needed for new lakes/reservoirs (their avgOutflow is still zero)
        # CAUTION (N4): the fallback max(avgChannelDischarge, avgInflow, 0.001)
        # applies only where avgOutflow == 0 exactly, which practically never
        # happens once avgOutflow has a positive initial value (0.001 m3/s in the
        # RAWS configurations): it decays but stays positive, also outside water
        # bodies. A cold-started or new lake then gets a tiny width estimate, and
        # minWeirWidth (10 m) applies.
        avgOutflow = pcr.ifthenelse(
            avgOutflow > 0.0,
            avgOutflow,
            pcr.max(avgChannelDischarge, self.avgInflow, 0.001),
        )
        # one value per body: the maximum over its cells (m3/s)
        avgOutflow = pcr.areamaximum(avgOutflow, self.waterBodyIds)
        # bankfull width (m) = 4.8 * avgOutflow**0.5; MV -> 0
        bankfullWidth = pcr.cover(pcr.scalar(4.8) * ((avgOutflow) ** (0.5)), 0.0)
        weirWidthUsed = bankfullWidth
        # TODO: base minWeirWidth on the GRanD database
        weirWidthUsed = pcr.max(weirWidthUsed, self.minWeirWidth)
        # only on body cells, 0 elsewhere
        weirWidthUsed = pcr.cover(
            pcr.ifthen(pcr.scalar(self.waterBodyIds) > 0.0, weirWidthUsed), 0.0
        )

        # (m3/s)
        # weir flow, but at least the 30-day average net inflow (m3/s)
        lakeOutflowInM3PerSec = pcr.max(
            self.weirFormula(waterHeight, weirWidthUsed), self.avgInflow
        )

        # volume released by lakes (m3)
        # m3 per time step; then at most the storage, and kept only on lake cells
        lakeOutflow = lakeOutflowInM3PerSec * length_of_time_step
        lakeOutflow = pcr.min(self.waterBodyStorage, lakeOutflow)
        lakeOutflow = pcr.ifthen(pcr.scalar(self.waterBodyIds) > 0.0, lakeOutflow)
        lakeOutflow = pcr.ifthen(pcr.scalar(self.waterBodyTyp) == 1, lakeOutflow)

        # TODO: consider endorheic lakes/basins (no outflow)

        return lakeOutflow

    def getReservoirOutflow(
        self, avgChannelDischarge, length_of_time_step, downstreamDemand
    ):
        """Native PCR-GLOBWB release rule of the reservoirs (waterBodyTyp == 2).

        Called by getWaterBodyOutflow. Returns the release in m3 per time step,
        defined only on reservoir cells (MV elsewhere). This is the method a
        data-driven scheme would replace; the contract is given at the EXTENSION
        POINT in getWaterBodyOutflow.

        Notation: S storage after this step's inflow (m3, per body); C =
        waterBodyCap (m3); f_min = minResvrFrac (0.10); f_max = maxResvrFrac (0.75);
        I = avgInflow (m3/s, 30-day mean of the net inflow, per cell); O =
        avgOutflow after the fallbacks, maximum over the body (m3/s); D =
        downstreamDemand (m3/s, per cell); dt (s). As coded:

            r  = min(1, max(0, S - f_min C) / (f_max - f_min) * C)        (N1)
            R1 = max(0, min(r O dt, I dt))
            D' = min(D, D * D / (f_min C))    (D' = 0 if f_min C <= 1e-39) (N2)
            R2 = max(R1, D' dt)
            S' = max(0, S - R2)
            F  = max(0, S' - C)
                 + max(0, S' - f_max C) / ((1 - f_max) C)
                   * max(0, 2.3 O 86400 - R2)
            F  = max(0, min(F, S' - 0.75 f_max C))
            R3 = R2 + F
            R4 = min(R3, max(0, S - 0.75 f_max C))  if S > f_max C, else R3
            R5 = max(0, R4, I)                       if S > f_max C, else R4 (N3)
            R  = min(S, R5)

        In words: (1) a base release equal to the long-term average outflow,
        switched on as soon as the storage exceeds f_min C (N1: a step, not a ramp)
        and never more than the 30-day average net inflow; (2) a negligible
        environmental-flow floor (N2); (3) above f_max C, a flood release made of
        all the storage above capacity plus a share, growing with the filling above
        f_max C, of 2.3 times the daily long-term average outflow, but never
        drawing the storage below 0.75 f_max C (0.5625 C with the defaults); (4) a
        floor at the average inflow above f_max C that has no effect (N3, units);
        (5) at most the storage. getWaterBodyOutflow then applies the 25 % cap
        (W9).
        """

        # (m3/s)
        # O: long-term average outflow (m3/s), with two fallbacks for new
        # reservoirs, then one value per body
        avgOutflow = self.avgOutflow
        # needed for new lakes/reservoirs (their avgOutflow is still zero)
        # CAUTION (N4): the fallbacks apply only where avgOutflow == 0 exactly. With
        # a positive initial value (0.001 m3/s in the RAWS configurations)
        # avgOutflow stays positive everywhere (it decays outside water bodies but
        # does not reach 0), so they practically never fire. After a cold start a
        # reservoir begins nearly empty, its first releases are about 0, and O falls
        # to about 1e-6 m3/s after the first day (weight 1/1.001 with
        # timestepsToAvgDischargeIni = 0.001); the base release O * dt is then
        # negligible and the reservoir releases mainly by spilling above capacity
        # until O builds up. The length of this transient is not determined from
        # the code; it argues for a spin-up or warm states.
        # First fallback: max(avgChannelDischarge, avgInflow) on the same cell.
        avgOutflow = pcr.ifthenelse(
            avgOutflow > 0.0, avgOutflow, pcr.max(avgChannelDischarge, self.avgInflow)
        )
        # second fallback: the value on the next cell downstream (self.lddMap, not
        # clipped to the landmask)
        avgOutflow = pcr.ifthenelse(
            avgOutflow > 0.0, avgOutflow, pcr.downstream(self.lddMap, avgOutflow)
        )
        # one value per body: the maximum over its cells (m3/s)
        avgOutflow = pcr.areamaximum(avgOutflow, self.waterBodyIds)

        # reservoir outflow based on reservoir storage and avgDischarge, using a reduction factor:
        # release stops if relativeCapacity < minResvrFrac, and equals the long-term average if
        # relativeCapacity > maxResvrFrac
        # CAUTION (N1): operator precedence makes this
        #     min(1, (max(0, S - f_min C) / (f_max - f_min)) * C),
        # not the ramp max(0, S - f_min C) / ((f_max - f_min) C) that the comment
        # above describes (and it has units of m3 instead of being dimensionless).
        # The factor is 0 for S <= f_min C and 1 as soon as S exceeds f_min C by
        # (f_max - f_min) / C m3 (much less than 1 m3): a step at f_min. The base
        # release is therefore O (capped by I) above 10 % of capacity and 0 below,
        # and f_max plays no role here. A reproduction of the native rule must copy
        # the expression as written. cover(..., 0): 0 where C is MV.
        reductionFactor = pcr.cover(
            pcr.min(
                1.0,
                pcr.max(
                    0.0, self.waterBodyStorage - self.minResvrFrac * self.waterBodyCap
                )
                / (self.maxResvrFrac - self.minResvrFrac)
                * self.waterBodyCap,
            ),
            0.0,
        )
        # (m3)
        # base release before the inflow cap (m3 per time step)
        resvOutflow = reductionFactor * avgOutflow * length_of_time_step

        # maximum release <= average inflow (especially in dry conditions) (m3)
        # CAUTION: the cap applies at all times, not only in dry conditions: the
        # base release never exceeds the 30-day average net inflow, I * dt. I is per
        # cell, so R1 differs between the cells of a body if I does (W10).
        resvOutflow = pcr.max(
            0, pcr.min(resvOutflow, self.avgInflow * length_of_time_step)
        )

        # downstream demand (m3/s), reduced if storage < lower limit
        # CAUTION (N2): the reduction factor is D / (f_min C), a discharge (m3/s)
        # divided by a volume (m3), not a function of the storage. Since f_min C is
        # large, D' = min(D, D * D / (f_min C)) is tiny: D = 100 m3/s and
        # C = 1e8 m3 give D' = 1e-3 m3/s. The floor D' * dt therefore almost never
        # binds. D is the environmental-flow estimate of routing.py, not the demand
        # of downstream users: no downstream withdrawal signal reaches the
        # reservoir.
        # INTERFACE: downstreamDemand (m3/s, per cell) comes from routing.py
        # Routing.update (estimate_discharge_for_environmental_flow).
        reductionFactor = vos.getValDivZero(
            downstreamDemand, self.minResvrFrac * self.waterBodyCap, vos.smallNumber
        )
        reductionFactor = pcr.cover(reductionFactor, 0.0)
        downstreamDemand = pcr.min(downstreamDemand, downstreamDemand * reductionFactor)
        # resvOutflow > downstreamDemand (m3)
        # R2 = max(R1, D' * dt); D' is per cell, so R2 can differ between the cells
        # of a body (W10)
        resvOutflow = pcr.max(resvOutflow, downstreamDemand * length_of_time_step)

        # floodOutflow: additional release if storage > upper limit
        # ratioQBankfull (dimensionless): the flood release is scaled with 2.3
        # times the long-term average outflow (by its name, a bankfull-to-mean
        # discharge ratio)
        ratioQBankfull = 2.3
        # S' (estmStorage): storage left after R2 (m3)
        estmStorage = pcr.max(0.0, self.waterBodyStorage - resvOutflow)
        # F (m3) = all storage above capacity, plus a share of
        # max(0, 2.3 * O * 86400 - R2) equal to the relative filling above f_max C,
        # (S' - f_max C) / ((1 - f_max) C): 0 at f_max C, 1 at capacity, above 1
        # beyond capacity. cover(..., 0) where C is MV.
        # CAUTION (W11): the daily volume uses vos.secondsPerDay() instead of
        # length_of_time_step. This matches the daily calls of the RAWS
        # configurations, but under kinematicWave a daily volume is offered at every
        # sub-step.
        floodOutflow = pcr.max(0.0, estmStorage - self.waterBodyCap) + pcr.cover(
            pcr.max(0.0, estmStorage - self.maxResvrFrac * self.waterBodyCap)
            / ((1.0 - self.maxResvrFrac) * self.waterBodyCap),
            0.0,
        ) * pcr.max(
            0.0, ratioQBankfull * avgOutflow * vos.secondsPerDay() - resvOutflow
        )
        # limit floodOutflow: only bring the storage down to 3/4 of the upper limit capacity
        # F <= S' - 0.75 f_max C: the flood release never draws the storage below
        # 0.75 * f_max * C (0.5625 C with the defaults), i.e. below the upper limit
        # itself
        floodOutflow = pcr.max(
            0.0,
            pcr.min(
                floodOutflow, estmStorage - self.maxResvrFrac * self.waterBodyCap * 0.75
            ),
        )

        # update resvOutflow after floodOutflow
        # R3 = R2 + F (MV -> 0)
        resvOutflow = pcr.cover(resvOutflow, 0.0) + pcr.cover(floodOutflow, 0.0)

        # maximum release if storage > upper limit: only bring the storage down to 3/4 of the upper limit capacity
        # R4: if S > f_max C, the total release is limited to S - 0.75 f_max C
        resvOutflow = pcr.ifthenelse(
            self.waterBodyStorage > self.maxResvrFrac * self.waterBodyCap,
            pcr.min(
                resvOutflow,
                pcr.max(
                    0,
                    self.waterBodyStorage
                    - self.maxResvrFrac * self.waterBodyCap * 0.75,
                ),
            ),
            resvOutflow,
        )

        # if storage > upper limit: resvOutflow > avgInflow
        # CAUTION (N3): resvOutflow is in m3 per time step but avgInflow is in m3/s,
        # so with daily steps this floor is 1/86400 of the intended one and has no
        # practical effect. It is applied after the limit R4, so in principle it
        # could lift the release above that limit, by a negligible amount.
        resvOutflow = pcr.ifthenelse(
            self.waterBodyStorage > self.maxResvrFrac * self.waterBodyCap,
            pcr.max(0.0, resvOutflow, self.avgInflow),
            resvOutflow,
        )

        # resvOutflow < waterBodyStorage
        # R = min(S, R5): at most the storage
        resvOutflow = pcr.min(self.waterBodyStorage, resvOutflow)

        # reservoir cells only, MV elsewhere: getWaterBodyOutflow relies on this to
        # fall back to the lake outflow
        resvOutflow = pcr.ifthen(pcr.scalar(self.waterBodyIds) > 0.0, resvOutflow)
        resvOutflow = pcr.ifthen(pcr.scalar(self.waterBodyTyp) == 2, resvOutflow)
        # (m3)
        return resvOutflow
