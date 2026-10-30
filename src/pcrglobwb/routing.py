import datetime
import logging
import os
from copy import deepcopy

import numpy as np
import pcraster as pcr

from pcrglobwb import waterBodies
from pcrglobwb.common import virtualOS as vos
from pcrglobwb.ncConverter import PCR2netCDF

logger = logging.getLogger(__name__)


class Routing(object):
    """
    Surface-water routing of PCR-GLOBWB 2, including the coupling with lakes and
    reservoirs.

    Water-body overview (annotation for the RAWS reservoir work). Routing owns one
    WaterBodies object (waterBodies.py, class WaterBodies, held as self.WaterBodies) and
    does all the bookkeeping between the river channel storage and the lake/reservoir
    storage. Within one model day (Routing.update, called by pcrglobwb.py after meteo,
    landSurface and groundwater):

    (1) local fluxes are added to self.channelStorage (m3 per cell) on every cell,
    water-body cells included: runoff and baseflow, minus surface-water abstraction,
    plus non-irrigation return flows, minus open-water evaporation;

    (2) at water-body cells the channel storage is moved into the water body (rounded
    down to whole m3) and WaterBodies.update is called, once per day for accuTravelTime
    and simplifiedKinematicWave (simple_update), once per sub-step for kinematicWave
    (kinematic_wave_update). WaterBodies forms the inflow as a storage residual, decides
    the release (lakes: weir formula; reservoirs: storage-fraction rule,
    WaterBodies.getReservoirOutflow) and updates its storage;

    (3) the release (m3 per time step) is put back into channelStorage: at the outlet
    cell (simplifiedKinematicWave), spread over the body's cells (accuTravelTime), or
    into the cell downstream of the outlet (kinematicWave); then the channel storage is
    routed;

    (4) after routing, the water-body storage is spread back over the body's cells in
    proportion to cell area (return_water_body_storage_to_channel), so that between days
    channelStorage at a water-body cell holds that cell's share of the lake or reservoir
    storage. The next day's abstraction (readAvlChannelStorage), open-water evaporation
    and riverbed infiltration therefore act on lake/reservoir water directly, and
    WaterBodies sees them only through the storage residual.

    Support conventions used in the comments: "per cell" is a value for each grid cell;
    "per water body" is a value constant over all cells of one water body (built with
    pcr.areatotal, pcr.areaaverage or pcr.areamaximum over WaterBodies.waterBodyIds);
    "outlet cell only" is defined at the cell flagged by WaterBodies.waterBodyOut. Tags:
    INTERFACE marks values exchanged with other modules; CAUTION (N1) to (N5) refer to
    the 2026-10-29 re-check of the reservoir findings on the RAWS branch, CAUTION (R1),
    (R2), ... are specific to this file; EXTENSION POINT marks where a new release
    scheme or a new input to it would be plugged in.
    """

    def getPseudoState(self):
        """
        Return the pseudo-states of routing: none (empty dictionary). Kept for the
        interface of pcrglobwb.py (Model.getPseudoState, Model.getAllState).
        """
        result = {}
        return result

    def getState(self):
        """
        Return the routing states needed to restart the model (dictionary of PCRaster
        maps).

        INTERFACE: called by pcrglobwb.py (Model.getState, Model.getAllState) to dump
        restart maps, and by Routing.update at the first time step, which passes this
        dictionary to WaterBodies.getParameterFiles and so to WaterBodies.getICs
        (waterBodies.py): this is how the initial water-body states read in getICs reach
        WaterBodies. The keys must match those read back in getICs (memory branch) and
        in WaterBodies.getICs.

        Water-body related states (all PCRaster maps on the landmask): channelStorage
        (m3 per cell; at water-body cells the cell's area share of the lake/reservoir
        storage), waterBodyStorage (m3, per water body, repeated on its cells),
        avgLakeReservoirOutflowLong and avgLakeReservoirInflowShort (m3/s, per water
        body), timestepsToAvgDischarge (days, one value for the whole map),
        readAvlChannelStorage (m3 per cell), avgDischargeLong, m2tDischargeLong and
        avgDischargeShort (inputs of the environmental flow, i.e. of downstreamDemand),
        riverbedExchange (m3 per day per cell).

        EXTENSION POINT: a state of a new release scheme (e.g. another running mean kept
        in WaterBodies) must be added here, read back in getICs (ini-map and memory
        branches, with a new ini key for cold starts) and read by WaterBodies.getICs,
        which receives this dictionary at the first time step.
        """
        result = {}

        # hydrology states
        # (day)
        # number of days already in the running means (days), one value for the whole
        # map; also passed to WaterBodies.update, where it sets the weights of avgInflow
        # and avgOutflow
        result["timestepsToAvgDischarge"] = self.timestepsToAvgDischarge
        # channel storage, including lake and reservoir storage (m3)
        # at water-body cells: the cell's share (by cell area) of the lake/reservoir
        # storage after the day's release, plus channel water left in the body's cells
        # after routing, minus the riverbed infiltration (see
        # return_water_body_storage_to_channel)
        result["channelStorage"] = self.channelStorage
        # readily available channel storage that can be abstracted to satisfy water demand (m3)
        # INTERFACE: read on the next day by water_management/main_water_management.py
        # (update, as available_surface_water_volume); not read when using_qualloc is
        # True
        result["readAvlChannelStorage"] = self.readAvlChannelStorage
        # avgDischargeLong, m2tDischargeLong and avgDischargeShort feed
        # estimate_discharge_for_environmental_flow, i.e. downstreamDemand (passed to
        # WaterBodies.update) and the environmental-flow reduction of
        # readAvlChannelStorage
        # long-term average discharge (m3/s)
        result["avgDischargeLong"] = self.avgDischarge
        # ((m3/s)^2)
        result["m2tDischargeLong"] = self.m2tDischarge
        # long-term average baseflow (m3/s)
        # INTERFACE: avgBaseflow is read by water_management/main_water_management.py
        # (surface-water / groundwater partitioning)
        result["avgBaseflowLong"] = self.avgBaseflow
        # river bed infiltration from surface water bodies to groundwater (m3/day)
        # INTERFACE: read on the next day by groundwater.py (update_storage, as
        # surfaceWaterInf); at water-body cells it was taken from the lake/reservoir
        # storage share
        result["riverbedExchange"] = self.riverbedExchange
        # lake and reservoir storage (m3), per water body id (not per cell)
        # copy of WaterBodies.waterBodyStorage made in simple_update /
        # kinematic_wave_update, i.e. the total storage of each water body after
        # outflow, repeated on each of its cells (not a per-cell share); 0 outside water
        # bodies after the first update
        result["waterBodyStorage"] = self.waterBodyStorage
        # long-term average lake and reservoir outflow (m3/s), per water body id (not per cell)
        # self.avgOutflow and self.avgInflow are copies of WaterBodies.avgOutflow and
        # WaterBodies.avgInflow made in calculate_statistics; the running means used by
        # the release rule live in WaterBodies. Both are also kept (and updated) outside
        # water bodies
        result["avgLakeReservoirOutflowLong"] = self.avgOutflow
        # short-term average lake and reservoir inflow (m3/s), per water body id (not per cell)
        result["avgLakeReservoirInflowShort"] = self.avgInflow
        # short-term average discharge (m3/s)
        # (30-day window; used by estimate_available_volume_for_abstraction)
        result["avgDischargeShort"] = self.avgDischargeShort
        # sub-time step discharge (m3/s), needed for the kinematic wave methods
        # accuTravelTime: Q / 86400 (m3/s); kinematic methods: discharge of the last
        # sub-step (m3/s); read by estimate_length_of_sub_time_step at the start of the
        # next day
        result["subDischarge"] = self.subDischarge

        # QUAlloc: long-term water availability
        # QUAlloc (using_qualloc, see __init__) runs inside landSurface.update, before
        # routing, and reads routing.discharge (m3/s per cell, previous day; the cell
        # discharge, not disChanWaterBody), routing.runoff (m/day, previous day) and
        # routing.channelStorage / cellArea (m, surface-water availability). discharge
        # and runoff are therefore states, so that the first day after a restart sees
        # the previous day's values.
        #
        # INTERFACE: landSurface.py (update, QUAlloc branch) ->
        # src/qualloc/qualloc_main.py (update).
        if self.using_qualloc:
            # discharge (m3/s)
            result["discharge"] = self.discharge
            # total runoff (m/day)
            result["runoff"] = self.runoff

            # with quality also True, the DynQual concentrations are handed to QUAlloc
            # (landSurface.py), where they can restrict surface-water withdrawals
            # through QUAlloc's suitability per sector (src/qualloc/water_management.py,
            # update_surfacewater_potential_withdrawals), hence also withdrawals at
            # reservoir cells
            if self.quality:
                # TDS concentration (mg/L)
                result["salinity"] = self.salinity
                # BOD concentration (mg/L)
                result["organic"] = self.organic
                # FC concentration (cfu/100mL)
                result["pathogen"] = self.pathogen

        # the DynQual states below are water-quality variables only; none is read by the
        # water-body quantity path
        # DynQual: irrigation return flows
        if self.quality:
            # average irrigation gross demand (m/day)
            result["avg_irrGrossDemand"] = self.avg_irrGrossDemand
            # average net liquid water transferred to the soil (m/day)
            result["avg_netLqWaterToSoil"] = self.avg_netLqWaterToSoil
        else:
            logger.info("Irrigation return flows not estimated")

        # water quality states
        if self.quality:
            # water temperature (K)
            result["waterTemperature"] = self.waterTemp
            # ice thickness (m)
            result["iceThickness"] = self.iceThickness
            # routed TDS load (g), converted to salinity pollution (mg/L)
            result["routedTDS"] = self.routedTDS
            # routed BOD load (g), converted to organic pollution (mg/L)
            result["routedBOD"] = self.routedBOD
            # routed FC load (cfu), converted to pathogen pollution (cfu/100mL)
            result["routedFC"] = self.routedFC

            if not self.offlineRun and self.calculateLoads and self.loadsPerSector:
                # pollutants routed per sector (for the analysis of sectoral contributions): domestic
                # (g TDS)
                result["routedDomTDS"] = self.routedDomTDS
                # (g BOD)
                result["routedDomBOD"] = self.routedDomBOD
                # (10^6 cfu)
                result["routedDomFC"] = self.routedDomFC
                # manufacturing
                # (g TDS)
                result["routedManTDS"] = self.routedManTDS
                # (g BOD)
                result["routedManBOD"] = self.routedManBOD
                # (10^6 cfu)
                result["routedManFC"] = self.routedManFC
                # urban surface runoff
                # (g TDS)
                result["routedUSRTDS"] = self.routedUSRTDS
                # (g BOD)
                result["routedUSRBOD"] = self.routedUSRBOD
                # (10^6 cfu)
                result["routedUSRFC"] = self.routedUSRFC
                # intensive livestock
                # (g BOD)
                result["routedintLivBOD"] = self.routedintLivBOD
                # (10^6 cfu)
                result["routedintLivFC"] = self.routedintLivFC
                # extensive livestock
                # (g BOD)
                result["routedextLivBOD"] = self.routedextLivBOD
                # (10^6 cfu)
                result["routedextLivFC"] = self.routedextLivFC
                # irrigation
                # (g TDS)
                result["routedIrrTDS"] = self.routedIrrTDS
            else:
                logger.info("Water quality per sector not tracked")
        else:
            logger.info("Water quality elements not simulated")
        return result

    def __init__(self, iniItems, initialConditions, lddMap):
        """
        Read the routing options, create the WaterBodies object and read the initial
        conditions.

        Water-body relevant settings read here (routingOptions unless stated):
        routingMethod (selects the daily sequence, see update), includeWaterBodies, the
        averaging windows (1825 and 30 days, hard-coded), cropCoefficientWaterNC and
        minCropWaterKC (open-water evaporation), dynamicFloodPlain and the channel
        geometry (they set dynamicFracWat, hence open-water evaporation and riverbed
        infiltration at water-body cells), maxiumLengthOfSubTimeStep (sub-steps of the
        kinematic methods), upstream_discharge_input_files, quality (DynQual),
        WWtPlantsNC (return flows) and using_qualloc (waterManagementOptions).
        Lake/reservoir specific keys (waterBodyInputNC or the per-year map keys,
        onlyNaturalWaterBodies, minResvrFrac, maxResvrFrac) are read by
        WaterBodies.__init__ (waterBodies.py). Initial states are read in getICs.

        Arguments: iniItems (configuration.Configuration), initialConditions (dictionary
        of states from memory, e.g. after a spin-up, or None to read the ini maps),
        lddMap (not used: self.lddMap is read again from routingOptions lddMap below).
        """
        object.__init__(self)

        self.lddMap = lddMap

        self.tmpDir = iniItems.tmpDir
        self.inputDir = iniItems.globalOptions["inputDir"]

        # option to activate the water balance check
        # also switches the balance checks around the WaterBodies.update call in
        # simple_update and kinematic_wave_update; WaterBodies reads the same key
        # itself. True in all RAWS configurations (log messages only, no effect on the
        # results)
        self.debugWaterBalance = True
        if iniItems.routingOptions["debugWaterBalance"] == "False":
            self.debugWaterBalance = False

        # routingMethod: "accuTravelTime", "simplifiedKinematicWave" or "kinematicWave"
        # (default "accuTravelTime", set by configuration.py, repair_ini_key_names, if
        # the key is missing). It decides how often WaterBodies.update is called and
        # where the release enters the channel (see update). RAWS configurations:
        # pcrglobwb_5arcminutes.ini and online_pcrglobwb_qualloc_5arcminutes.ini use
        # simplifiedKinematicWave; pcrglobwb_30arcminutes.ini,
        # pcrglobwb_30arcseconds.ini and online_pcrglobwb_qualloc_30arcseconds.ini use
        # accuTravelTime; none uses kinematicWave. Any other value is not caught here;
        # update would then run neither branch and fail later (self.Q undefined).
        self.method = iniItems.routingOptions["routingMethod"]

        # option to include lakes and reservoirs
        # includeWaterBodies (default True; "False" or "None" switch it off; True in all
        # RAWS configurations). Used only in update, where waterBodyIds is replaced by
        # -1 everywhere.
        #
        # CAUTION (R6): only the ids are removed: WaterBodies.fracWat still enters
        # dynamicFracWat (open-water evaporation, riverbed infiltration), waterBodyOut
        # is kept, and WaterBodies.update still runs every day with zero storage.
        self.includeWaterBodies = True
        if "includeWaterBodies" in list(iniItems.routingOptions.keys()):
            if (
                iniItems.routingOptions["includeWaterBodies"] == "False"
                or iniItems.routingOptions["includeWaterBodies"] == "None"
            ):
                self.includeWaterBodies = False

        # sectors evaluated
        self.includeSectors = {}
        self.includeSectors["industry"] = False
        self.includeSectors["manufacture"] = False
        if iniItems.waterDemandOptions["includeIndustryWaterDemand"] == "True":
            self.includeSectors["industry"] = True
        elif iniItems.waterDemandOptions["includeManufactureWaterDemand"] == "True":
            self.includeSectors["manufacture"] = True

        self.lddMap = vos.readPCRmapClone(
            iniItems.routingOptions["lddMap"],
            self.tmpDir,
            self.inputDir,
            True,
        )
        self.lddMap = pcr.lddrepair(pcr.ldd(self.lddMap))
        self.lddMap = pcr.lddrepair(self.lddMap)

        # complete ldd before clipping to the landmask (see below); needed to route upstream discharge
        # (from other sub-runs) into this basin, as the cells supplying it lie just outside the landmask
        self.ldd_complete = self.lddMap

        if iniItems.globalOptions["landmask"] != "None":
            self.landmask = vos.readPCRmapClone(
                iniItems.globalOptions["landmask"],
                self.tmpDir,
                self.inputDir,
            )
        else:
            self.landmask = pcr.defined(self.lddMap)
        self.landmask = pcr.ifthen(pcr.defined(self.lddMap), self.landmask)
        self.landmask = pcr.cover(self.landmask, pcr.boolean(0))

        self.lddMap = pcr.lddmask(self.lddMap, self.landmask)

        # cell area (m2)
        # cellArea (m2 per cell): converts every m/day flux into m3, and is the weight
        # used to spread a water body's storage over its cells
        # (return_water_body_storage_to_channel)
        self.cellArea = vos.readPCRmapClone(
            iniItems.routingOptions["cellAreaMap"],
            self.tmpDir,
            self.inputDir,
        )

        # model resolution (arc-degree)
        self.cellSizeInArcDeg = vos.getCloneAttributes()["cellsize"]

        # maximum number of time steps for long-term average flows (default: 5 years = 1825 days)
        self.maxTimestepsToAvgDischargeLong = 1825.0

        # maximum number of time steps for short-term average values (default: 1 month = 30 days)
        self.maxTimestepsToAvgDischargeShort = 30.0

        # The two windows above are passed to WaterBodies.update:
        # maxTimestepsToAvgDischargeShort is the window (days) of WaterBodies.avgInflow
        # (and of avgDischargeShort here), maxTimestepsToAvgDischargeLong that of
        # WaterBodies.avgOutflow (and of avgDischarge, m2tDischarge and avgBaseflow
        # here). They are not configurable.
        #
        # EXTENSION POINT: a release scheme that needs other windows (e.g. a 7-day
        # inflow mean) must keep its own running mean in WaterBodies, with its own state
        # in getState / getICs.
        routingParameters = ["gradient", "manningsN"]
        for var in routingParameters:
            input = iniItems.routingOptions[str(var)]
            vars(self)[var] = vos.readPCRmapClone(input, self.tmpDir, self.inputDir)

        # parameters to estimate the channel dimensions (used in getRoutingParamAvgDischarge)
        self.eta = 0.25
        self.nu = 0.40
        self.tau = 8.00
        self.phi = 0.58

        # option to use a minimum channel width (m)
        self.minChannelWidth = pcr.scalar(0.0)
        if "minimumChannelWidth" in list(iniItems.routingOptions.keys()):
            if iniItems.routingOptions["minimumChannelWidth"] != "None":
                self.minChannelWidth = pcr.cover(
                    vos.readPCRmapClone(
                        iniItems.routingOptions["minimumChannelWidth"],
                        self.tmpDir,
                        self.inputDir,
                    ),
                    0.0,
                )

        # option to use a predefined channel width (m)
        self.predefinedChannelWidth = None
        if "constantChannelWidth" in list(iniItems.routingOptions.keys()):
            if iniItems.routingOptions["constantChannelWidth"] != "None":
                self.predefinedChannelWidth = pcr.cover(
                    vos.readPCRmapClone(
                        iniItems.routingOptions["constantChannelWidth"],
                        self.tmpDir,
                        self.inputDir,
                    ),
                    0.0,
                )

        # option to use a predefined channel depth (m)
        self.predefinedChannelDepth = None
        if "constantChannelDepth" in list(iniItems.routingOptions.keys()):
            if iniItems.routingOptions["constantChannelDepth"] != "None":
                self.predefinedChannelDepth = pcr.cover(
                    vos.readPCRmapClone(
                        iniItems.routingOptions["constantChannelDepth"],
                        self.tmpDir,
                        self.inputDir,
                    ),
                    0.0,
                )

        # assumption for broad sheet flow in the kinematic wave methods
        self.beta = 0.6

        # channel length (m), approximated by the cell diagonal
        cellSizeInArcMin = self.cellSizeInArcDeg * 60.0
        verticalSizeInMeter = cellSizeInArcMin * 1852.0
        self.cellLengthFD = (
            (self.cellArea / verticalSizeInMeter) ** (2) + (verticalSizeInMeter) ** (2)
        ) ** (0.5)
        self.channelLength = self.cellLengthFD
        # channel length (m)
        if "channelLength" in list(iniItems.routingOptions.keys()):
            if iniItems.routingOptions["channelLength"] != "None":
                self.channelLength = pcr.cover(
                    vos.readPCRmapClone(
                        iniItems.routingOptions["channelLength"],
                        self.tmpDir,
                        self.inputDir,
                    ),
                    self.channelLength,
                )

        # dist2celllength (m/arcDegree), needed in the accuTravelTime function
        nrCellsDownstream = pcr.ldddist(self.lddMap, pcr.nominal(self.lddMap) == 5, 1.0)
        distanceDownstream = pcr.ldddist(
            self.lddMap, pcr.nominal(self.lddMap) == 5, self.channelLength
        )
        # (m)
        channelLengthDownstream = (self.channelLength + distanceDownstream) / (
            nrCellsDownstream + 1
        )
        # (m/arcDegree)
        self.dist2celllength = channelLengthDownstream / self.cellSizeInArcDeg

        self.distance_to_pit = 0.5 * self.channelLength + distanceDownstream

        # the channel gradient must be >= minGradient
        minGradient = 0.00005
        self.gradient = pcr.max(minGradient, pcr.cover(self.gradient, minGradient))

        # INTERFACE: waterBodies.py, WaterBodies.__init__(iniItems, landmask). It reads
        # the lake/reservoir options from routingOptions (waterBodyInputNC or the
        # per-year map keys, onlyNaturalWaterBodies, minResvrFrac, maxResvrFrac,
        # debugWaterBalance) and its own lddMap. The water-body maps (fracWat,
        # waterBodyIds, waterBodyOut, waterBodyArea, waterBodyTyp, waterBodyCap) only
        # exist after WaterBodies.getParameterFiles, called from update at the first
        # time step and on 1 January.
        #
        # EXTENSION POINT: options of a new release scheme (e.g. a scheme selector, or
        # the path to trained parameters per reservoir) can be read in
        # WaterBodies.__init__ from the same iniItems.routingOptions; nothing needs to
        # change here.
        self.WaterBodies = waterBodies.WaterBodies(iniItems, self.landmask)

        # crop evaporation coefficient of surface water bodies
        # cropCoefficientWaterNC: monthly open-water crop coefficient kc (-), netCDF
        # variable "kc", read in calculate_potential_evaporation. If "None", kc = 0 and
        # no open-water evaporation is taken from channels, lakes or reservoirs. All
        # RAWS configurations set it (30arcminutes/routing/kc_open_water.nc, also for
        # the 5 arcmin and 30 arcsec runs).
        self.no_zero_crop_water_coefficient = True
        if iniItems.routingOptions["cropCoefficientWaterNC"] == "None":
            self.no_zero_crop_water_coefficient = False
        else:
            self.fileCropKC = vos.getFullPath(
                iniItems.routingOptions["cropCoefficientWaterNC"], self.inputDir
            )

        # Sub-time steps (kinematic methods only; accuTravelTime routes in one daily
        # step). limit_num_of_sub_time_steps (integer) is the minimum number of
        # sub-steps per day: at least 24, at least 86400 / (0.5 * minimum channelLength
        # / 5 m/s), and at least ceil(86400 / maxiumLengthOfSubTimeStep). The RAWS 5
        # arcmin configurations set maxiumLengthOfSubTimeStep = 720 (s), hence at least
        # 120 sub-steps. Under simplifiedKinematicWave the sub-steps route channel water
        # only and WaterBodies.update is still called once per day (simple_update);
        # under kinematicWave it is called at every sub-step (kinematic_wave_update).
        # Courant number criterion for numerical stability in the kinematic wave methods
        self.courantNumber = 0.50

        # empirical values for the minimum number of sub-time steps (m/s)
        design_flood_speed = 5.00
        design_length_of_sub_time_step = pcr.cellvalue(
            pcr.mapminimum(
                self.courantNumber * self.channelLength / design_flood_speed
            ),
            1,
        )[0]
        self.limit_num_of_sub_time_steps = np.ceil(
            vos.secondsPerDay() / design_length_of_sub_time_step
        )
        # at least 24 sub-time steps (hourly, as in Van Beek et al., 2011)
        self.limit_num_of_sub_time_steps = max(24.0, self.limit_num_of_sub_time_steps)

        # minimum number of sub-time steps from the ini file
        if "maxiumLengthOfSubTimeStep" in list(iniItems.routingOptions.keys()):
            maxiumLengthOfSubTimeStep = float(
                iniItems.routingOptions["maxiumLengthOfSubTimeStep"]
            )
            minimum_number_of_sub_time_step = np.ceil(
                vos.secondsPerDay() / maxiumLengthOfSubTimeStep
            )
            self.limit_num_of_sub_time_steps = max(
                minimum_number_of_sub_time_step, self.limit_num_of_sub_time_steps
            )
        self.limit_num_of_sub_time_steps = int(self.limit_num_of_sub_time_steps)

        # critical water height (m) to select a stable sub-time step length in the kinematic wave methods,
        self.critical_water_height = 0.25
        # as in Van Beek et al. (2011)

        # minimum fracWat used to calculate the water height (-)
        # min_fracwat_for_water_height (-): floor of the water fraction when a water
        # height (m) is computed as storage / (fraction * cellArea); max_water_height
        # (m): cap of that height in simplifiedKinematicWave
        self.min_fracwat_for_water_height = 0.001
        self.max_water_height = 50000000

        # minimum crop coefficient of surface water bodies
        # minCropWaterKC (default 0.0): lower bound of kc. The RAWS configurations set
        # 1.00, so kc >= 1 and the open-water potential evaporation is at least the
        # reference ET0.
        self.minCropWaterKC = 0.00
        if "minCropWaterKC" in list(iniItems.routingOptions.keys()):
            self.minCropWaterKC = float(iniItems.routingOptions["minCropWaterKC"])

        # floodplain options
        # dynamicFloodPlain (default "False", set by configuration.py if missing; True
        # in all RAWS configurations). For water bodies: (i) in update, the inundated
        # fraction derived from channelStorage enters dynamicFracWat; at water-body
        # cells channelStorage holds the cell's share of the lake/reservoir storage, so
        # the floodplain profile of the cell, not the geometry of the reservoir, sets
        # the evaporating fraction above fracWat (CAUTION R8, in update); (ii) under
        # simplifiedKinematicWave, alpha and the initial discharge come from kinAlpha
        # (floodplain Manning), also at the body's cells.
        self.floodPlain = iniItems.routingOptions["dynamicFloodPlain"] == "True"
        if self.floodPlain:

            logger.info("Flood plain extents can vary during the simulation.")

            # Manning's n of the floodplain areas
            input = iniItems.routingOptions["floodplainManningsN"]
            self.floodplainManN = vos.readPCRmapClone(input, self.tmpDir, self.inputDir)

            # reduction parameter of the smoothing interval and error threshold
            self.reductionKK = 0.5
            if "reductionKK" in list(iniItems.routingOptions.keys()):
                self.reductionKK = float(iniItems.routingOptions["reductionKK"])
            self.criterionKK = 40.0
            if "criterionKK" in list(iniItems.routingOptions.keys()):
                self.criterionKK = float(iniItems.routingOptions["criterionKK"])

            # profile of relative elevation above the floodplain per cell (including smoothing parameters)
            (
                self.nrZLevels,
                self.areaFractions,
                self.relZ,
                self.floodVolume,
                self.kSlope,
                self.mInterval,
            ) = self.getElevationProfile(iniItems)

            # bankfull capacity (m3)
            # River bankfull capacity (m3 per cell). With bankfullCapacity = None and
            # both constantChannelWidth and constantChannelDepth given (RAWS 5 arcmin
            # configurations) it is fixed from those maps (estimateBankfullCapacity).
            # Storage above it is floodplain water
            # (returnInundationFractionAndFloodDepth, kinAlpha), also at water-body
            # cells.
            self.predefinedBankfullCapacity = None
            self.usingFixedBankfullCapacity = False
            if iniItems.routingOptions["bankfullCapacity"] != "None":

                self.usingFixedBankfullCapacity = True
                self.predefinedBankfullCapacity = vos.readPCRmapClone(
                    iniItems.routingOptions["bankfullCapacity"],
                    self.tmpDir,
                    self.inputDir,
                )
                self.predefinedBankfullCapacity = pcr.cover(
                    self.predefinedBankfullCapacity, pcr.spatial(pcr.scalar(0.0))
                )

            else:
                msg = "The bankfull channel storage capacity is NOT defined in the configuration file. "

                if (
                    self.predefinedChannelWidth is None
                    or self.predefinedChannelDepth is None
                ):
                    msg += "The bankfull capacity is estimated from average discharge (5 year long term average)."
                else:
                    msg += "The bankfull capacity is estimated from the given channel depth and channel width."
                    self.usingFixedBankfullCapacity = True
                    self.predefinedBankfullCapacity = self.estimateBankfullCapacity(
                        self.predefinedChannelWidth, self.predefinedChannelDepth
                    )

                logger.info(msg)

        # option to limit the flood depth (to avoid unrealistic flood depths)
        self.maxFloodDepth = None
        if "maxFloodDepth" in list(iniItems.routingOptions.keys()):
            self.maxFloodDepth = vos.readPCRmapClone(
                iniItems.routingOptions["maxFloodDepth"],
                self.tmpDir,
                self.inputDir,
            )

        # input file for upstream discharge (from upstream basins)
        # upstream_discharge_input_files (comma-separated list; not set in any RAWS
        # configuration): discharge (m3/s) from other sub-runs, added in simple_update
        # to the channel storage of the first cell inside the landmask, before
        # WaterBodies.update. If that cell is a water-body cell, the volume becomes part
        # of the same day's inflow residual. Ignored by kinematic_wave_update.
        self.upstream_discharge_input_files = None
        if (
            "upstream_discharge_input_files" in list(iniItems.routingOptions.keys())
            and iniItems.routingOptions["upstream_discharge_input_files"] != "None"
        ):
            self.upstream_discharge_input_files = iniItems.routingOptions[
                "upstream_discharge_input_files"
            ].split(",")

        # DynQual
        # quality (DynQual: water temperature, salinity, organic and pathogen
        # pollution). True only in online_pcrglobwb_qualloc_5arcminutes.ini among the
        # RAWS configurations. DynQual reads water-body quantities
        # (WaterBodies.waterBodyOutflow, waterBodyIds, waterBodyOut,
        # self.waterBodyStorageTimeBefore, channelStorage copies) but writes back to
        # none of channelStorage, waterBodyStorage or the WaterBodies attributes; it
        # only overwrites the reported self.waterBodyEvaporation on freezing days
        # (qualityLocal, CAUTION R13). Through QUAlloc it can change withdrawals (see
        # getState).
        self.quality = False
        if (
            "quality" in iniItems.routingOptions.keys()
            and iniItems.routingOptions["quality"] == "True"
        ):
            self.quality = True
            logger.info("Water quality modelling initiated.")
        else:
            logger.info("Water quality modelling not initiated.")

        # WWtPlantsNC: wastewater-treatment-plant ids and service zones. The presence of
        # the key alone switches the option on (a value of "None" is not handled). When
        # on, simple_update moves domestic and manufacturing return flows to the plant
        # cells (return_flows_to_wastewater_treatment_plants), which moves local input
        # into or out of water-body cells. Set in both RAWS 5 arcmin configurations and
        # in online_pcrglobwb_qualloc_30arcseconds.ini (which points to the 5 arcmin
        # file). Not used by kinematic_wave_update.
        self.WWtPlants = False
        if "WWtPlantsNC" in list(iniItems.routingOptions.keys()):
            self.WWtPlants = True
            # wastewater pathways and removal efficiencies (treatment: tertiary, secondary, primary; collected but untreated; basic sanitation; open defecation; direct)
            self.WWtPlantsNC = vos.getFullPath(
                iniItems.routingOptions["WWtPlantsNC"], self.inputDir
            )

        if self.quality:

            # discharge threshold for estimating concentrations (default: 0.1 m3/s)
            self.WQ_discharge_threshold = 0.1
            if "WQ_discharge_threshold" in iniItems.routingOptions.keys():
                self.WQ_discharge_threshold = float(
                    iniItems.routingOptions["WQ_discharge_threshold"]
                )

            # water temperature parameters: temperature threshold for snowmelt (K)
            self.iceThresTemp = pcr.scalar(273.15)
            # density of water (kg/m3)
            self.densityWater = pcr.scalar(1000.0)
            # latent heat of vaporization (J/kg)
            self.latentHeatVapor = pcr.scalar(2.5e6)
            # latent heat of fusion (J/kg)
            self.latentHeatFusion = pcr.scalar(3.34e5)
            # specific heat of water (J/kg/degC)
            self.specificHeatWater = pcr.scalar(4190.0)
            # heat transfer coefficient of water (W/m2/degC)
            self.heatTransferWater = pcr.scalar(20.0)
            # heat transfer coefficient of ice (W/m2/degC)
            self.heatTransferIce = pcr.scalar(8.0)
            # albedo of water (-)
            self.albedoWater = pcr.scalar(0.15)
            # albedo of snow and ice (-)
            self.albedoSnow = pcr.scalar(0.50)
            # energy balance proxy for the groundwater temperature: mean annual temperature and the temperature reduction for falling rain
            self.deltaTPrec = pcr.scalar(1.5)

            self.radCon = 0.25
            self.radSlope = 0.50
            # Stefan-Boltzmann constant (W/m2/K4)
            self.stefanBoltzman = 5.67e-8
            # maximum river temperature: 322.15 K (50 degC)
            self.maxThresTemp = pcr.scalar(322.15)

            self.maxIceThickness = 3.0
            self.deltaIceThickness = 0.0

            if iniItems.meteoOptions["sunhoursTable"] != "Default":
                # convert cloud cover to sunshine hours (Doornkamp & Pruitt)
                self.sunFracTBL = vos.getFullPath(
                    iniItems.meteoOptions["sunhoursTable"], self.inputDir
                )
            else:
                self.sunFracTBL = vos.getFullPath(
                    "sunhoursfrac.tbl", os.path.abspath(os.path.dirname(__file__))
                )
                msg = "Using the default sunhoursfrac.tbl stored on " + self.sunFracTBL
                # sunshine fraction table
                logger.info(msg)

            # paths to additional meteorological variables
            self.cloudFileNC = vos.getFullPath(
                iniItems.meteoOptions["cloudcoverNC"], self.inputDir
            )
            self.radFileNC = vos.getFullPath(
                iniItems.meteoOptions["radiationNC"], self.inputDir
            )
            self.vapFileNC = vos.getFullPath(
                iniItems.meteoOptions["vaporNC"], self.inputDir
            )
            self.annualTFileNC = vos.getFullPath(
                iniItems.meteoOptions["annualAvgTNC"], self.inputDir
            )

            # paths to power plant data
            self.TlmaxNC = vos.getFullPath(
                iniItems.routingOptions["TlmaxNC"], self.inputDir
            )
            self.powerplants_fwNC = vos.getFullPath(
                iniItems.routingOptions["powerplants_fwNC"], self.inputDir
            )
            self.powerplants_fwfixedNC = vos.getFullPath(
                iniItems.routingOptions["powerplants_fwfixedNC"], self.inputDir
            )
            self.powerplants_swNC = vos.getFullPath(
                iniItems.routingOptions["powerplants_swNC"], self.inputDir
            )

            # salinity: background TDS concentration (mg/L)
            self.backgroundSalinityNC = vos.getFullPath(
                iniItems.routingOptions["backgroundSalinity"], self.inputDir
            )
            # (mg/L)
            self.backgroundSalinity = vos.netcdf2PCRobjCloneWithoutTime(
                self.backgroundSalinityNC, "bgTDS"
            )

            # organic pollution: first-order degradation coefficient at 20 degC (van Vliet et al., 2021)
            self.k_BOD = pcr.scalar(0.35)
            # temperature correction (van Vliet et al., 2021; Wen et al., 2017)
            self.watertempcorrection_BOD = pcr.scalar(1.047)

            # dissolved oxygen: elevation data
            self.elevation_path = vos.getFullPath(
                iniItems.landSurfaceOptions["topographyNC"], self.inputDir
            )
            self.elevation = vos.netcdf2PCRobjCloneWithoutTime(
                self.elevation_path,
                "dem_average",
                True,
                None,
                self.inputDir,
            )

            # fecal coliforms: temperature-dependent decay (day-1; Reder et al., 2015)

            self.darkinactivation_FC = pcr.scalar(0.82)
            # Reder et al. (2015)
            self.watertempcorrection_FC = pcr.scalar(1.07)

            # solar radiation dependent decay: total suspended solids (Beusen et al., 2005)
            self.tss = vos.readPCRmapClone(
                iniItems.routingOptions["TSSmap"],
                self.tmpDir,
                self.inputDir,
            )
            # (m2/W; Reder et al., 2015)
            self.sunlightinactivation_FC = pcr.scalar(0.0068)
            # (m-1; Reder et al., 2015)
            self.attenuation_FC = 0.0931 * self.tss + 0.881

            # sedimentation: the stream depth must exceed 0.5 m for sedimentation to occur (m)
            self.threshold_FC_settlingdepth = pcr.scalar(0.5)
            # (m/day; Reder et al., 2015)
            self.settlingvelocity_FC = pcr.scalar(1.656)

            # options for offline runs
            if iniItems.routingOptions["offlineRun"] == "True":
                self.offlineRun = True
                logger.info("DynQual running in offline configuration")

                # baseflow, interflow and direct runoff, required for offline DynQual runs
                self.baseflowNC = vos.getFullPath(
                    iniItems.routingOptions["baseflowNC"], self.inputDir
                )
                self.interflowNC = vos.getFullPath(
                    iniItems.routingOptions["interflowNC"], self.inputDir
                )
                self.directRunoffNC = vos.getFullPath(
                    iniItems.routingOptions["directRunoffNC"], self.inputDir
                )

                if iniItems.routingOptions["calculateLoads"] == "True":
                    logger.info(
                        "WARNING: Cannot calculate pollutant loadings in offline configuration."
                    )
                    logger.info(
                        "Switch to online configuration or prescribe loadings directly."
                    )

            else:
                self.offlineRun = False
                logger.info("DynQual running online")

            # calculate the loadings within the model run
            if (
                iniItems.routingOptions["calculateLoads"] == "True"
                and not self.offlineRun
            ):
                self.calculateLoads = True
                logger.info("Loadings calculated within model run.")

                if iniItems.routingOptions["loadsPerSector"] == "True":
                    self.loadsPerSector = True
                    logger.info("Option to report loads per sector enabled")
                else:
                    self.loadsPerSector = False

            else:
                self.calculateLoads = False
                self.loadsPerSector = False
                logger.info(
                    "Loadings are prescribed (i.e. akin to a forcing) to the model"
                )

            if self.calculateLoads:

                # pollutant loading input data: domestic, gridded annual population at 5 arcmin (Lange & Geiger, 2020)

                self.PopulationNC = vos.getFullPath(
                    iniItems.routingOptions["PopulationNC"], self.inputDir
                )
                # average (regional) excretion rates
                self.Dom_ExcrLoadNC = vos.getFullPath(
                    iniItems.routingOptions["Dom_ExcrLoadNC"], self.inputDir
                )
                # (g/capita/day)
                self.DomTDS_ExcrLoad = vos.netcdf2PCRobjCloneWithoutTime(
                    self.Dom_ExcrLoadNC, "Dom_Fixed_TDSload"
                )
                # (g/capita/day)
                self.DomBOD_ExcrLoad = vos.netcdf2PCRobjCloneWithoutTime(
                    self.Dom_ExcrLoadNC, "Dom_Fixed_BODload"
                )
                # (cfu/capita/day)
                self.DomFC_ExcrLoad = vos.netcdf2PCRobjCloneWithoutTime(
                    self.Dom_ExcrLoadNC, "Dom_Fixed_FCload"
                )

                # manufacturing: average (regional) effluent concentrations
                self.Man_EfflConcNC = vos.getFullPath(
                    iniItems.routingOptions["Man_EfflConcNC"], self.inputDir
                )
                # (mg/L, i.e. g/m3)
                self.ManTDS_EfflConc = vos.netcdf2PCRobjCloneWithoutTime(
                    self.Man_EfflConcNC, "Man_Fixed_TDSload"
                )
                # (mg/L, i.e. g/m3)
                self.ManBOD_EfflConc = vos.netcdf2PCRobjCloneWithoutTime(
                    self.Man_EfflConcNC, "Man_Fixed_BODload"
                )
                # (cfu/100mL)
                self.ManFC_EfflConc = vos.netcdf2PCRobjCloneWithoutTime(
                    self.Man_EfflConcNC, "Man_Fixed_FCload"
                )

                # urban surface runoff: urban fraction, from 0 (no urban) to 1 (all urban)
                self.UrbanFractionNC = vos.getFullPath(
                    iniItems.routingOptions["UrbanFractionNC"], self.inputDir
                )
                # average (regional) effluent concentrations
                self.USR_EfflConcNC = vos.getFullPath(
                    iniItems.routingOptions["USR_EfflConcNC"], self.inputDir
                )
                # (mg/L, i.e. g/m3)
                self.USRTDS_EfflConc = vos.netcdf2PCRobjCloneWithoutTime(
                    self.USR_EfflConcNC, "USR_Fixed_TDSload"
                )
                # (mg/L, i.e. g/m3)
                self.USRBOD_EfflConc = vos.netcdf2PCRobjCloneWithoutTime(
                    self.USR_EfflConcNC, "USR_Fixed_BODload"
                )
                # (cfu/100mL)
                self.USRFC_EfflConc = vos.netcdf2PCRobjCloneWithoutTime(
                    self.USR_EfflConcNC, "USR_Fixed_FCload"
                )

                # livestock: gridded populations, 2010, 5 arcmin (Gilbert et al., 2018)
                self.LivPopulationNC = vos.getFullPath(
                    iniItems.routingOptions["LivPopulationNC"], self.inputDir
                )
                # average (regional) excretion rates
                self.Liv_ExcrLoadNC = vos.getFullPath(
                    iniItems.routingOptions["Liv_ExcrLoadNC"], self.inputDir
                )
                # (g/stock/day)
                self.Bufallo_BODload = vos.netcdf2PCRobjCloneWithoutTime(
                    self.Liv_ExcrLoadNC, "bufallo_BODload"
                )
                # (g/stock/day)
                self.Chicken_BODload = vos.netcdf2PCRobjCloneWithoutTime(
                    self.Liv_ExcrLoadNC, "chicken_BODload"
                )
                # (g/stock/day)
                self.Cow_BODload = vos.netcdf2PCRobjCloneWithoutTime(
                    self.Liv_ExcrLoadNC, "cow_BODload"
                )
                # (g/stock/day)
                self.Duck_BODload = vos.netcdf2PCRobjCloneWithoutTime(
                    self.Liv_ExcrLoadNC, "duck_BODload"
                )
                # (g/stock/day)
                self.Goat_BODload = vos.netcdf2PCRobjCloneWithoutTime(
                    self.Liv_ExcrLoadNC, "goat_BODload"
                )
                # (g/stock/day)
                self.Horse_BODload = vos.netcdf2PCRobjCloneWithoutTime(
                    self.Liv_ExcrLoadNC, "horse_BODload"
                )
                # (g/stock/day)
                self.Pig_BODload = vos.netcdf2PCRobjCloneWithoutTime(
                    self.Liv_ExcrLoadNC, "pig_BODload"
                )
                # (g/stock/day)
                self.Sheep_BODload = vos.netcdf2PCRobjCloneWithoutTime(
                    self.Liv_ExcrLoadNC, "sheep_BODload"
                )
                # (cfu/stock/day)
                self.Bufallo_FCload = vos.netcdf2PCRobjCloneWithoutTime(
                    self.Liv_ExcrLoadNC, "bufallo_FCload"
                )
                # (cfu/stock/day)
                self.Chicken_FCload = vos.netcdf2PCRobjCloneWithoutTime(
                    self.Liv_ExcrLoadNC, "chicken_FCload"
                )
                # (cfu/stock/day)
                self.Cow_FCload = vos.netcdf2PCRobjCloneWithoutTime(
                    self.Liv_ExcrLoadNC, "cow_FCload"
                )
                # (cfu/stock/day)
                self.Duck_FCload = vos.netcdf2PCRobjCloneWithoutTime(
                    self.Liv_ExcrLoadNC, "duck_FCload"
                )
                # (cfu/stock/day)
                self.Goat_FCload = vos.netcdf2PCRobjCloneWithoutTime(
                    self.Liv_ExcrLoadNC, "goat_FCload"
                )
                # (cfu/stock/day)
                self.Horse_FCload = vos.netcdf2PCRobjCloneWithoutTime(
                    self.Liv_ExcrLoadNC, "horse_FCload"
                )
                # (cfu/stock/day)
                self.Pig_FCload = vos.netcdf2PCRobjCloneWithoutTime(
                    self.Liv_ExcrLoadNC, "pig_FCload"
                )
                # (cfu/stock/day)
                self.Sheep_FCload = vos.netcdf2PCRobjCloneWithoutTime(
                    self.Liv_ExcrLoadNC, "sheep_FCload"
                )

                # irrigation: soil concentration averaged over the topsoil and subsoil
                self.Irr_EfflConcNC = vos.getFullPath(
                    iniItems.routingOptions["Irr_EfflConcNC"], self.inputDir
                )
                # (mg/L)
                self.IrrTDS_EfflConc = vos.netcdf2PCRobjCloneWithoutTime(
                    self.Irr_EfflConcNC, "soil_TDS"
                )

            else:
                # paths to (non-natural) TDS, BOD and FC loading inputs
                self.TDSloadNC = vos.getFullPath(
                    iniItems.routingOptions["TDSloadNC"], self.inputDir
                )
                self.BODloadNC = vos.getFullPath(
                    iniItems.routingOptions["BODloadNC"], self.inputDir
                )
                self.FCloadNC = vos.getFullPath(
                    iniItems.routingOptions["FCloadNC"], self.inputDir
                )

        else:
            self.calculateLoads = False
            self.loadsPerSector = False

        # using_qualloc (waterManagementOptions): QUAlloc does the water allocation
        # instead of water_management/main_water_management.py (landSurface.py, update).
        # Effects on routing and water bodies: (i) extra states discharge and runoff
        # (getState, getICs), because QUAlloc reads them before routing runs; (ii)
        # QUAlloc takes the surface-water availability as routing.channelStorage /
        # cellArea, i.e. the whole channel storage, including the lake/reservoir share
        # at water-body cells, without the environmental-flow reduction, the 90 percent
        # cap and the 0.1 m3 rounding of estimate_available_volume_for_abstraction, and
        # withdraws min(potential, available) (src/qualloc/qualloc_main.py, update);
        # (iii) readAvlChannelStorage is still computed and saved but has no reader.
        # Routing subtracts landSurface.actSurfaceWaterAbstract in the same way in both
        # cases. True in online_pcrglobwb_qualloc_5arcminutes.ini and
        # online_pcrglobwb_qualloc_30arcseconds.ini.
        #
        # CAUTION (R12): with QUAlloc, one day's withdrawals at a water-body cell can
        # take up to the whole storage share of that cell (not 90 percent), and, as in
        # the standard path, the reservoir's lower limit minResvrFrac is not known to
        # the allocation (K2 of the gross-vs-net live log).
        self.using_qualloc = False
        if (
            "using_qualloc" in iniItems.waterManagementOptions
            and iniItems.waterManagementOptions["using_qualloc"] == "True"
        ):
            self.using_qualloc = True

        # initial conditions, including the staged water-body states avgInflow,
        # avgOutflow and waterBodyStorage (handed to WaterBodies at the first update)
        self.getICs(iniItems, initialConditions)

        # old-style reporting (useful for debugging)
        # inactive in all RAWS configurations (no out* keys in routingOptions)
        self.initiate_old_style_routing_reporting(iniItems)

    def getICs(self, iniItems, iniConditions=None):
        """
        Read the routing initial conditions, from the ini maps (iniConditions None) or
        from memory (iniConditions["routing"], from a spin-up or a previous getState).

        Water-body related states and their ini keys (routingOptions):
        avgLakeReservoirInflowShortIni -> self.avgInflow (m3/s),
        avgLakeReservoirOutflowLongIni -> self.avgOutflow (m3/s), waterBodyStorageIni ->
        self.waterBodyStorage (m3), channelStorageIni, readAvlChannelStorageIni,
        avgDischargeLongIni, avgDischargeShortIni, m2tDischargeLongIni,
        riverbedExchangeIni, timestepsToAvgDischargeIni. The three water-body states are
        only staged here: WaterBodies receives them at the first time step (update ->
        getState -> WaterBodies.getParameterFiles -> WaterBodies.getICs, waterBodies.py)
        and from then on keeps its own copies.

        All RAWS configurations cold-start every one of these with the scalar 0.001 and
        set maxSpinUpsInYears = 0, so lakes and reservoirs start nearly empty with
        avgOutflow = 0.001 m3/s on every cell.

        CAUTION (N4): the new-reservoir fallback of WaterBodies.getReservoirOutflow
        applies only where avgOutflow == 0 exactly, so it never fires after this
        initialisation; the base release starts near zero and builds up from spills.
        """

        if iniConditions is None:
            logger.info(
                "Reading initial conditions from pcraster maps listed in the .ini file"
            )
            # read the initial conditions from the PCRaster maps in the ini file (at the start of the model)
            self.timestepsToAvgDischarge = vos.readPCRmapClone(
                iniItems.routingOptions["timestepsToAvgDischargeIni"],
                self.tmpDir,
                self.inputDir,
            )

            self.channelStorage = vos.readPCRmapClone(
                iniItems.routingOptions["channelStorageIni"],
                self.tmpDir,
                self.inputDir,
            )
            self.readAvlChannelStorage = vos.readPCRmapClone(
                iniItems.routingOptions["readAvlChannelStorageIni"],
                self.tmpDir,
                self.inputDir,
            )
            self.avgDischarge = vos.readPCRmapClone(
                iniItems.routingOptions["avgDischargeLongIni"],
                self.tmpDir,
                self.inputDir,
            )
            self.m2tDischarge = vos.readPCRmapClone(
                iniItems.routingOptions["m2tDischargeLongIni"],
                self.tmpDir,
                self.inputDir,
            )
            self.avgBaseflow = vos.readPCRmapClone(
                iniItems.routingOptions["avgBaseflowLongIni"],
                self.tmpDir,
                self.inputDir,
            )
            self.riverbedExchange = vos.readPCRmapClone(
                iniItems.routingOptions["riverbedExchangeIni"],
                self.tmpDir,
                self.inputDir,
            )

            # initial condition introduced in version 2.0.2: avgDischargeShort
            self.avgDischargeShort = vos.readPCRmapClone(
                iniItems.routingOptions["avgDischargeShortIni"],
                self.tmpDir,
                self.inputDir,
            )

            # initial conditions needed for the kinematic wave methods
            self.subDischarge = vos.readPCRmapClone(
                iniItems.routingOptions["subDischargeIni"],
                self.tmpDir,
                self.inputDir,
            )

            # initial conditions needed for coupling with QUAlloc
            # dischargeIni (m3/s) and runoffIni (m/day) are required when using_qualloc
            # is True (KeyError otherwise); online_pcrglobwb_qualloc_5arcminutes.ini
            # sets both to 0.001. They feed QUAlloc on the first day only.
            if self.using_qualloc:
                self.discharge = vos.readPCRmapClone(
                    iniItems.routingOptions["dischargeIni"],
                    self.tmpDir,
                    self.inputDir,
                )
                self.runoff = vos.readPCRmapClone(
                    iniItems.routingOptions["runoffIni"],
                    self.tmpDir,
                    self.inputDir,
                )

                if self.quality:
                    # salinity pollution
                    self.salinity = vos.readPCRmapClone(
                        iniItems.routingOptions["salinityIni"],
                        self.tmpDir,
                        self.inputDir,
                    )
                    # organic pollution
                    self.organic = vos.readPCRmapClone(
                        iniItems.routingOptions["organicIni"],
                        self.tmpDir,
                        self.inputDir,
                    )
                    # pathogen pollution
                    self.pathogen = vos.readPCRmapClone(
                        iniItems.routingOptions["pathogenIni"],
                        self.tmpDir,
                        self.inputDir,
                    )

            # initial conditions needed for the water quality module
            if self.quality:
                self.waterTemp = vos.readPCRmapClone(
                    iniItems.routingOptions["waterTemperatureIni"],
                    self.tmpDir,
                    self.inputDir,
                )
                self.iceThickness = vos.readPCRmapClone(
                    iniItems.routingOptions["iceThicknessIni"],
                    self.tmpDir,
                    self.inputDir,
                )
                # salinity pollution
                self.routedTDS = vos.readPCRmapClone(
                    iniItems.routingOptions["routedTDSIni"],
                    self.tmpDir,
                    self.inputDir,
                )
                # organic pollution
                self.routedBOD = vos.readPCRmapClone(
                    iniItems.routingOptions["routedBODIni"],
                    self.tmpDir,
                    self.inputDir,
                )
                # pathogen pollution
                self.routedFC = vos.readPCRmapClone(
                    iniItems.routingOptions["routedFCIni"],
                    self.tmpDir,
                    self.inputDir,
                )

                # initial average irrigation demand and net liquid water to the soil, for the irrigation return flows
                if self.calculateLoads and not self.offlineRun:
                    self.avg_irrGrossDemand = vos.readPCRmapClone(
                        iniItems.routingOptions["avg_irrGrossDemandIni"],
                        self.tmpDir,
                        self.inputDir,
                    )
                    self.avg_netLqWaterToSoil = vos.readPCRmapClone(
                        iniItems.routingOptions["avg_netLqWaterToSoilIni"],
                        self.tmpDir,
                        self.inputDir,
                    )

                    # per water quality sector
                    if self.loadsPerSector:
                        self.routedDomTDS = vos.readPCRmapClone(
                            iniItems.routingOptions["routedDomTDSIni"],
                            self.tmpDir,
                            self.inputDir,
                        )
                        self.routedDomBOD = vos.readPCRmapClone(
                            iniItems.routingOptions["routedDomBODIni"],
                            self.tmpDir,
                            self.inputDir,
                        )
                        self.routedDomFC = vos.readPCRmapClone(
                            iniItems.routingOptions["routedDomFCIni"],
                            self.tmpDir,
                            self.inputDir,
                        )
                        self.routedManTDS = vos.readPCRmapClone(
                            iniItems.routingOptions["routedManTDSIni"],
                            self.tmpDir,
                            self.inputDir,
                        )
                        self.routedManBOD = vos.readPCRmapClone(
                            iniItems.routingOptions["routedManBODIni"],
                            self.tmpDir,
                            self.inputDir,
                        )
                        self.routedManFC = vos.readPCRmapClone(
                            iniItems.routingOptions["routedManFCIni"],
                            self.tmpDir,
                            self.inputDir,
                        )
                        self.routedUSRTDS = vos.readPCRmapClone(
                            iniItems.routingOptions["routedUSRTDSIni"],
                            self.tmpDir,
                            self.inputDir,
                        )
                        self.routedUSRBOD = vos.readPCRmapClone(
                            iniItems.routingOptions["routedUSRBODIni"],
                            self.tmpDir,
                            self.inputDir,
                        )
                        self.routedUSRFC = vos.readPCRmapClone(
                            iniItems.routingOptions["routedUSRFCIni"],
                            self.tmpDir,
                            self.inputDir,
                        )
                        self.routedintLivBOD = vos.readPCRmapClone(
                            iniItems.routingOptions["routedintLivBODIni"],
                            self.tmpDir,
                            self.inputDir,
                        )
                        self.routedintLivFC = vos.readPCRmapClone(
                            iniItems.routingOptions["routedintLivFCIni"],
                            self.tmpDir,
                            self.inputDir,
                        )
                        self.routedextLivBOD = vos.readPCRmapClone(
                            iniItems.routingOptions["routedextLivBODIni"],
                            self.tmpDir,
                            self.inputDir,
                        )
                        self.routedextLivFC = vos.readPCRmapClone(
                            iniItems.routingOptions["routedextLivFCIni"],
                            self.tmpDir,
                            self.inputDir,
                        )
                        self.routedIrrTDS = vos.readPCRmapClone(
                            iniItems.routingOptions["routedIrrTDSIni"],
                            self.tmpDir,
                            self.inputDir,
                        )

        else:
            logger.info("Reading initial conditions from memory.")
            # read the initial conditions from memory
            self.timestepsToAvgDischarge = iniConditions["routing"][
                "timestepsToAvgDischarge"
            ]

            self.channelStorage = iniConditions["routing"]["channelStorage"]
            self.readAvlChannelStorage = iniConditions["routing"][
                "readAvlChannelStorage"
            ]
            self.avgDischarge = iniConditions["routing"]["avgDischargeLong"]
            self.m2tDischarge = iniConditions["routing"]["m2tDischargeLong"]
            self.avgBaseflow = iniConditions["routing"]["avgBaseflowLong"]
            self.riverbedExchange = iniConditions["routing"]["riverbedExchange"]
            self.avgDischargeShort = iniConditions["routing"]["avgDischargeShort"]

            self.subDischarge = iniConditions["routing"]["subDischarge"]

            # initial conditions needed for coupling with QUAlloc
            # QUAlloc states from memory (same meaning as above)
            if self.using_qualloc:
                self.discharge = iniConditions["routing"]["discharge"]
                self.runoff = iniConditions["routing"]["runoff"]

                if self.quality:
                    self.salinity = iniConditions["routing"]["salinity"]
                    self.organic = iniConditions["routing"]["organic"]
                    self.pathogen = iniConditions["routing"]["pathogen"]

            # initial conditions needed for the water quality module (DynQual)
            if self.quality:
                self.waterTemp = iniConditions["routing"]["waterTemperature"]
                self.iceThickness = iniConditions["routing"]["iceThickness"]
                self.routedTDS = iniConditions["routing"]["routedTDS"]
                self.routedBOD = iniConditions["routing"]["routedBOD"]
                self.routedFC = iniConditions["routing"]["routedFC"]

                # initial average irrigation demand and net liquid water to the soil, for the irrigation return flows
                if self.calculateLoads and not self.offlineRun:
                    self.avg_irrGrossDemand = iniConditions["routing"][
                        "avg_irrGrossDemand"
                    ]
                    self.avg_netLqWaterToSoil = iniConditions["routing"][
                        "avg_netLqWaterToSoil"
                    ]

                    # per water quality sector
                    if self.loadsPerSector:
                        self.routedDomTDS = iniConditions["routing"]["routedDomTDS"]
                        self.routedDomBOD = iniConditions["routing"]["routedDomBOD"]
                        self.routedDomFC = iniConditions["routing"]["routedDomFC"]
                        self.routedManTDS = iniConditions["routing"]["routedManTDS"]
                        self.routedManBOD = iniConditions["routing"]["routedManBOD"]
                        self.routedManFC = iniConditions["routing"]["routedManFC"]
                        self.routedUSRTDS = iniConditions["routing"]["routedUSRTDS"]
                        self.routedUSRBOD = iniConditions["routing"]["routedUSRBOD"]
                        self.routedUSRFC = iniConditions["routing"]["routedUSRFC"]
                        self.routedintLivBOD = iniConditions["routing"][
                            "routedintLivBOD"
                        ]
                        self.routedintLivFC = iniConditions["routing"]["routedintLivFC"]
                        self.routedextLivBOD = iniConditions["routing"][
                            "routedextLivBOD"
                        ]
                        self.routedextLivFC = iniConditions["routing"]["routedextLivFC"]
                        self.routedIrrTDS = iniConditions["routing"]["routedIrrTDS"]

        # values within the landmask
        self.channelStorage = pcr.ifthen(
            self.landmask, pcr.cover(self.channelStorage, 0.0)
        )
        self.readAvlChannelStorage = pcr.ifthen(
            self.landmask, pcr.cover(self.readAvlChannelStorage, 0.0)
        )
        self.avgDischarge = pcr.ifthen(self.landmask, pcr.cover(self.avgDischarge, 0.0))
        self.m2tDischarge = pcr.ifthen(self.landmask, pcr.cover(self.m2tDischarge, 0.0))
        self.avgDischargeShort = pcr.ifthen(
            self.landmask, pcr.cover(self.avgDischargeShort, 0.0)
        )
        self.avgBaseflow = pcr.ifthen(self.landmask, pcr.cover(self.avgBaseflow, 0.0))
        self.riverbedExchange = pcr.ifthen(
            self.landmask, pcr.cover(self.riverbedExchange, 0.0)
        )
        self.subDischarge = pcr.ifthen(self.landmask, pcr.cover(self.subDischarge, 0.0))

        # readAvlChannelStorage (m3 per cell): non-negative and at most channelStorage
        self.readAvlChannelStorage = pcr.min(
            self.readAvlChannelStorage, self.channelStorage
        )
        self.readAvlChannelStorage = pcr.max(self.readAvlChannelStorage, 0.0)

        # QUAlloc
        # QUAlloc states restricted to the landmask, missing values set to 0
        if self.using_qualloc:
            self.discharge = pcr.ifthen(self.landmask, pcr.cover(self.discharge, 0.0))
            self.runoff = pcr.ifthen(self.landmask, pcr.cover(self.runoff, 0.0))

            if self.quality:
                self.salinity = pcr.ifthen(self.landmask, pcr.cover(self.salinity, 0.0))
                self.organic = pcr.ifthen(self.landmask, pcr.cover(self.organic, 0.0))
                self.pathogen = pcr.ifthen(self.landmask, pcr.cover(self.pathogen, 0.0))

        # DynQual
        if self.quality:
            self.waterTemp = pcr.ifthen(self.landmask, pcr.cover(self.waterTemp, 0.0))
            self.iceThickness = pcr.ifthen(
                self.landmask, pcr.cover(self.iceThickness, 0.0)
            )
            self.dissolved_oxygen = (1 - 0.0001148 * self.elevation) * pcr.exp(
                -139.34411
                + (157570.1) / (self.waterTemp)
                - (66423080.0) / (self.waterTemp**2)
                + (12438000000.0) / (self.waterTemp**3)
                - (862194900000.0) / (self.waterTemp**4)
            )
            self.channelStorageTimeBefore = self.channelStorage
            self.totEW = (
                self.channelStorage
                * self.waterTemp
                * self.specificHeatWater
                * self.densityWater
            )
            self.temp_water_height = self.eta * pow(self.avgDischarge, self.nu)
            self.routedTDS = pcr.ifthen(self.landmask, pcr.cover(self.routedTDS, 0.0))
            self.routedBOD = pcr.ifthen(self.landmask, pcr.cover(self.routedBOD, 0.0))
            self.routedFC = pcr.ifthen(self.landmask, pcr.cover(self.routedFC, 0.0))

            # per water quality sector
            if self.calculateLoads and not self.offlineRun:
                self.avg_irrGrossDemand = pcr.ifthen(
                    self.landmask, pcr.cover(self.avg_irrGrossDemand, 0.0)
                )
                self.avg_netLqWaterToSoil = pcr.ifthen(
                    self.landmask, pcr.cover(self.avg_netLqWaterToSoil, 0.0)
                )

                if self.loadsPerSector:
                    self.routedDomTDS = pcr.ifthen(
                        self.landmask, pcr.cover(self.routedDomTDS, 0.0)
                    )
                    self.routedDomBOD = pcr.ifthen(
                        self.landmask, pcr.cover(self.routedDomBOD, 0.0)
                    )
                    self.routedDomFC = pcr.ifthen(
                        self.landmask, pcr.cover(self.routedDomFC, 0.0)
                    )
                    self.routedManTDS = pcr.ifthen(
                        self.landmask, pcr.cover(self.routedManTDS, 0.0)
                    )
                    self.routedManBOD = pcr.ifthen(
                        self.landmask, pcr.cover(self.routedManBOD, 0.0)
                    )
                    self.routedManFC = pcr.ifthen(
                        self.landmask, pcr.cover(self.routedManFC, 0.0)
                    )
                    self.routedUSRTDS = pcr.ifthen(
                        self.landmask, pcr.cover(self.routedUSRTDS, 0.0)
                    )
                    self.routedUSRBOD = pcr.ifthen(
                        self.landmask, pcr.cover(self.routedUSRBOD, 0.0)
                    )
                    self.routedUSRFC = pcr.ifthen(
                        self.landmask, pcr.cover(self.routedUSRFC, 0.0)
                    )
                    self.routedintLivBOD = pcr.ifthen(
                        self.landmask, pcr.cover(self.routedintLivBOD, 0.0)
                    )
                    self.routedintLivFC = pcr.ifthen(
                        self.landmask, pcr.cover(self.routedintLivFC, 0.0)
                    )
                    self.routedextLivBOD = pcr.ifthen(
                        self.landmask, pcr.cover(self.routedextLivBOD, 0.0)
                    )
                    self.routedextLivFC = pcr.ifthen(
                        self.landmask, pcr.cover(self.routedextLivFC, 0.0)
                    )
                    self.routedIrrTDS = pcr.ifthen(
                        self.landmask, pcr.cover(self.routedIrrTDS, 0.0)
                    )

        # make sure timestepsToAvgDischarge is the same for the entire map
        # timestepsToAvgDischarge (days): 0.001 in the RAWS cold starts, so the first
        # update uses a weight of about 1 and the running means start at the first day's
        # values
        try:
            self.timestepsToAvgDischarge = pcr.mapmaximum(self.timestepsToAvgDischarge)
        except Exception:
            # try/except because pcr.mapmaximum cannot handle scalar values
            pass

        # for netCDF reporting, timestepsToAvgDischarge must be spatial and scalar (e.g. for pcr2numpy)
        self.timestepsToAvgDischarge = pcr.spatial(
            pcr.scalar(self.timestepsToAvgDischarge)
        )
        self.timestepsToAvgDischarge = pcr.ifthen(
            self.landmask, self.timestepsToAvgDischarge
        )

        # initial conditions of water bodies: short-term average inflow (m3/s) and long-term average outflow (m3/s)
        # (m3/s and m3; maps on every landmask cell, water-body values repeated on the
        # body's cells. INTERFACE: they reach WaterBodies.getICs, waterBodies.py, at the
        # first time step through update and getState.)
        if iniConditions is None:
            # read the initial conditions from the PCRaster maps in the ini file (at the start of the model)
            self.avgInflow = vos.readPCRmapClone(
                iniItems.routingOptions["avgLakeReservoirInflowShortIni"],
                self.tmpDir,
                self.inputDir,
            )
            self.avgOutflow = vos.readPCRmapClone(
                iniItems.routingOptions["avgLakeReservoirOutflowLongIni"],
                self.tmpDir,
                self.inputDir,
            )
            # CAUTION (R5): routingOptions values are strings, so this test is always
            # true when the key exists. With waterBodyStorageIni = None,
            # readPCRmapClone("None") returns None (common/virtualOS.py,
            # singleTryReadPCRmapClone) and the next statement passes None to pcr.cover,
            # which is expected to fail (not run). The memory branch never passes None
            # either (getState always returns a map), so the fallback of
            # WaterBodies.getICs that derives the storage from channelStorage is not
            # reachable in practice.
            if iniItems.routingOptions["waterBodyStorageIni"] is not None:
                self.waterBodyStorage = vos.readPCRmapClone(
                    iniItems.routingOptions["waterBodyStorageIni"],
                    self.tmpDir,
                    self.inputDir,
                )
                self.waterBodyStorage = pcr.ifthen(
                    self.landmask, pcr.cover(self.waterBodyStorage, 0.0)
                )
            else:
                self.waterBodyStorage = None
        else:
            # read the initial conditions from memory
            self.avgInflow = iniConditions["routing"]["avgLakeReservoirInflowShort"]
            self.avgOutflow = iniConditions["routing"]["avgLakeReservoirOutflowLong"]
            self.waterBodyStorage = iniConditions["routing"]["waterBodyStorage"]

        # CAUTION (N4): outside water bodies these maps keep their initial value (0.001
        # in the RAWS cold starts) and WaterBodies keeps updating them on every landmask
        # cell, so a reservoir that appears later (yearly maps) inherits a small
        # positive avgOutflow and the new-reservoir fallback does not fire
        self.avgInflow = pcr.ifthen(self.landmask, pcr.cover(self.avgInflow, 0.0))
        self.avgOutflow = pcr.ifthen(self.landmask, pcr.cover(self.avgOutflow, 0.0))
        if self.waterBodyStorage is not None:
            self.waterBodyStorage = pcr.ifthen(
                self.landmask, pcr.cover(self.waterBodyStorage, 0.0)
            )

    def estimateBankfullCapacity(self, width, depth, minWidth=5.0, minDepth=1.0):

        # bankfull capacity (m3)
        bankfullCapacity = (
            pcr.max(minWidth, width) * pcr.max(minDepth, depth) * self.channelLength
        )

        return bankfullCapacity

    def getElevationProfile(self, iniItems):

        # profile of relative elevation above the floodplain (per cell); output: dictionaries kSlope,

        # mInterval, relZ and floodVolume with keys iCnt (index):
        # nrZLevels: number of levels; areaFractions (-): fraction of flooded area; relZ (m): relative
        # elevation above the floodplain; floodVolume (m3): flood volume above the bankfull capacity;
        # kSlope (-): slope used in the interpolation; mInterval (m3): smoothing interval of the interpolation

        msg = "Get the profile of relative elevation (relZ, unit: m) !!!"
        logger.info(msg)

        relativeElevationFileNC = (
            # TODO: define the relative elevation files in a netCDF file
            None
        )
        if relativeElevationFileNC is not None:
            # TODO: use a netCDF file
            pass

        if relativeElevationFileNC is None:

            relZFileName = vos.getFullPath(
                iniItems.routingOptions["relativeElevationFiles"],
                iniItems.globalOptions["inputDir"],
            )

            # fractions of flooded areas (-)
            areaFractions = list(
                map(
                    float,
                    str(iniItems.routingOptions["relativeElevationLevels"]).split(","),
                )
            )
            # number of levels
            nrZLevels = len(areaFractions)
            # TODO: read areaFractions and nrZLevels automatically

        # patch the elevations of sills using the floodplain gradient, with local distances deltaX per
        # increment up to z[N] and the sum over sills; fill all lists (including smoothing interval and slopes)

        relZ = [0.0] * nrZLevels
        for iCnt in range(0, nrZLevels):

            if relativeElevationFileNC is None:
                inputName = relZFileName % (areaFractions[iCnt] * 100)
                relZ[iCnt] = vos.readPCRmapClone(inputName, self.tmpDir, self.inputDir)
            if relativeElevationFileNC is not None:
                # TODO: use a netCDF file
                pass

            # cover the elevation values
            relZ[iCnt] = pcr.ifthen(self.landmask, pcr.cover(relZ[iCnt], 0.0))

            # make sure that relZ[iCnt] >= relZ[iCnt-1] (added by Edwin)
            if iCnt > 0:
                relZ[iCnt] = pcr.max(relZ[iCnt], relZ[iCnt - 1])

        # minimum floodplain slope, defined by the longest sill (first used to retrieve the longest
        # cumulative distance)
        deltaX = [self.cellArea**0.5] * nrZLevels
        deltaX[0] = 0.0
        sumX = deltaX[:]
        minSlope = 0.0
        for iCnt in range(nrZLevels):
            if iCnt < nrZLevels - 1:
                deltaX[iCnt] = (
                    areaFractions[iCnt + 1] ** 0.5 - areaFractions[iCnt] ** 0.5
                ) * deltaX[iCnt]
            else:
                deltaX[iCnt] = (1.0 - areaFractions[iCnt - 1] ** 0.5) * deltaX[iCnt]
            if iCnt > 0:
                sumX[iCnt] = pcr.ifthenelse(
                    relZ[iCnt] == relZ[iCnt - 1], sumX[iCnt - 1] + deltaX[iCnt], 0.0
                )
                minSlope = pcr.ifthenelse(
                    relZ[iCnt] == relZ[iCnt - 1],
                    pcr.max(sumX[iCnt], minSlope),
                    minSlope,
                )
        # the floodplain slope is at most the channel gradient (flow is slower on the floodplain)
        minSlope = pcr.min(self.gradient, 0.5 * pcr.max(deltaX[1], minSlope) ** -1.0)

        # add a small increment to the elevation of each sill (except for lakes; TODO: verify this)
        for iCnt in range(nrZLevels):
            relZ[iCnt] = relZ[iCnt] + sumX[iCnt] * pcr.ifthenelse(
                relZ[nrZLevels - 1] > 0.0, minSlope, 0.0
            )
            # make sure that relZ[iCnt] >= relZ[iCnt-1] (added by Edwin)
            if iCnt > 0:
                relZ[iCnt] = pcr.max(relZ[iCnt], relZ[iCnt - 1])

        # slope and smoothing interval between dy = y(i+1)-y(i) and dx = x(i+1)-x(i), based on volume;
        # volume (m3)
        floodVolume = [0.0] * (nrZLevels)
        # smoothing interval (m3)
        mInterval = [0.0] * (nrZLevels)
        # slope (-)
        kSlope = [0.0] * (nrZLevels)
        for iCnt in range(1, nrZLevels):
            floodVolume[iCnt] = (
                floodVolume[iCnt - 1]
                + 0.5
                * (areaFractions[iCnt] + areaFractions[iCnt - 1])
                * (relZ[iCnt] - relZ[iCnt - 1])
                * self.cellArea
            )
            kSlope[iCnt - 1] = (
                areaFractions[iCnt] - areaFractions[iCnt - 1]
            ) / pcr.max(0.001, floodVolume[iCnt] - floodVolume[iCnt - 1])
        for iCnt in range(1, nrZLevels):
            if iCnt < (nrZLevels - 1):
                mInterval[iCnt] = (
                    0.5
                    * self.reductionKK
                    * pcr.min(
                        floodVolume[iCnt + 1] - floodVolume[iCnt],
                        floodVolume[iCnt] - floodVolume[iCnt - 1],
                    )
                )
            else:
                mInterval[iCnt] = (
                    0.5 * self.reductionKK * (floodVolume[iCnt] - floodVolume[iCnt - 1])
                )

        return nrZLevels, areaFractions, relZ, floodVolume, kSlope, mInterval

    def getRoutingParamAvgDischarge(self, avgDischarge, dist2celllength):
        """
        Channel dimensions from the long-term average discharge avgDischarge (m3/s per
        cell): yMean (m, average flow depth), wMean (m, average flow width, replaced by
        constantChannelWidth where given, at least minimumChannelWidth), and a
        characteristic distance (arc-degrees per day).

        Water bodies: avgDischarge at water-body cells is the running mean of the
        body-maximum discharge (calculate_statistics uses disChanWaterBody), so the
        channel width and depth there are those of the river at the body's largest
        discharge.

        CAUTION: the characteristicDistance returned here (with a 1.5-cell extension at
        water bodies) is stored by update as self.characteristicDistance and read
        nowhere; accuTravelTime uses getCharacteristicDistance (2.5 cells) instead.
        """
        # routing parameters from the long-term average discharge (m3/s); output: channel dimensions and
        # characteristicDistance (input of accuTravelTime)

        yMean = self.eta * pow(avgDischarge, self.nu)
        wMean = self.tau * pow(avgDischarge, self.phi)

        # option to use a predefined channel width (m)
        if self.predefinedChannelWidth is not None:
            wMean = pcr.cover(self.predefinedChannelWidth, wMean)
        wMean = pcr.max(self.minChannelWidth, wMean)

        # average flow width (m); may be used as channel width estimate (assuming rectangular channels)
        wMean = pcr.max(wMean, 0.01)
        wMean = pcr.cover(wMean, 0.01)
        # average flow depth (m); should not be used as channel depth estimate
        yMean = pcr.max(yMean, 0.01)
        yMean = pcr.cover(yMean, 0.01)

        # characteristicDistance (-), used for accutraveltimeflux (discharge: the total material flowing
        # through the cell, m3/s) and accutraveltimestate (storage: the material deposited in the cell, m3);
        # (m/day)
        characteristicDistance = (
            ((yMean * wMean) / (wMean + 2 * yMean)) ** (2.0 / 3.0)
            * ((self.gradient) ** (0.5))
            / self.manningsN
            * vos.secondsPerDay()
        )

        # (arcDeg/day)
        characteristicDistance = pcr.max(
            (self.cellSizeInArcDeg) * 0.000000001,
            characteristicDistance / dist2celllength,
        )

        # characteristicDistance of each lake/reservoir
        lakeReservoirCharacteristicDistance = pcr.ifthen(
            pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
            pcr.areaaverage(characteristicDistance, self.WaterBodies.waterBodyIds),
        )

        # make sure all outflow is released outside lakes and reservoirs
        # CAUTION: the result of this method is not used for routing (see the
        # docstring); the rule that is applied is the one in getCharacteristicDistance
        outlets = pcr.cover(
            pcr.ifthen(pcr.scalar(self.WaterBodies.waterBodyOut) > 0, pcr.boolean(1)),
            pcr.boolean(0),
        )
        distance_to_outlets = pcr.ifthen(
            pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
            pcr.ldddist(self.lddMap, outlets, pcr.scalar(1.0)),
        )
        lakeReservoirCharacteristicDistance = pcr.ifthen(
            pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
            pcr.max(
                distance_to_outlets + pcr.downstreamdist(self.lddMap) * 1.50,
                lakeReservoirCharacteristicDistance,
            ),
        )
        characteristicDistance = pcr.cover(
            lakeReservoirCharacteristicDistance, characteristicDistance
        )
        # (arcDeg/day)
        characteristicDistance = pcr.roundup(characteristicDistance * 100.0) / 100.0

        # minimum characteristicDistance
        characteristicDistance = pcr.cover(
            characteristicDistance, 0.1 * self.cellSizeInArcDeg
        )
        characteristicDistance = pcr.max(
            0.100 * self.cellSizeInArcDeg, characteristicDistance
        )

        return (yMean, wMean, characteristicDistance)

    def getCharacteristicDistance(self, yMean, wMean):
        """
        Characteristic distance (arc-degrees per day) of the accuTravelTime routing: the
        distance water travels in one day, Manning velocity (m/day) / dist2celllength.

        Water bodies: all cells of a water body get one value (pcr.areamaximum over
        waterBodyIds), at least the ldd distance (arc-degrees) from the cell to the
        body's outlet plus 2.5 downstream cell lengths, so that water placed in the
        body's cells (the release, spread over the body by simple_update) passes the
        outlet within the day. Used only by accuTravelTime (RAWS 30 arcmin and 30 arcsec
        configurations). See CAUTION (R7) in accuTravelTime for upstream water that
        reaches the body during routing.
        """

        # Manning's coefficient
        usedManningsN = self.manningsN

        # corrected Manning's coefficient
        if self.floodPlain:

            # wetted perimeter
            flood_only_wetted_perimeter = self.floodDepth * (2.0) + pcr.max(
                0.0,
                self.innundatedFraction * self.cellArea / self.channelLength
                - self.channelWidth,
            )
            channel_only_wetted_perimeter = (
                pcr.min(
                    self.channelDepth,
                    vos.getValDivZero(
                        self.channelStorage, self.channelLength * self.channelWidth, 0.0
                    ),
                )
                * 2.0
                + self.channelWidth
            )
            # total channel wetted perimeter (m)
            channel_wetted_perimeter = (
                channel_only_wetted_perimeter + flood_only_wetted_perimeter
            )
            # minimum channel wetted perimeter: 10 cm
            channel_wetted_perimeter = pcr.max(0.1, channel_wetted_perimeter)

            usedManningsN = (
                (channel_only_wetted_perimeter / channel_wetted_perimeter)
                * self.manningsN ** (1.5)
                + (flood_only_wetted_perimeter / channel_wetted_perimeter)
                * self.floodplainManN ** (1.5)
            ) ** (2.0 / 3.0)

        # characteristicDistance (-), used for accutraveltimeflux (discharge: the total material flowing
        # through the cell, m3/s) and accutraveltimestate (storage: the material deposited in the cell, m3);
        # (m/day)
        characteristicDistance = (
            ((yMean * wMean) / (wMean + 2 * yMean)) ** (2.0 / 3.0)
            * ((self.gradient) ** (0.5))
            / usedManningsN
            * vos.secondsPerDay()
        )

        # (arcDeg/day)
        characteristicDistance = pcr.max(
            (self.cellSizeInArcDeg) * 0.000000001,
            characteristicDistance / self.dist2celllength,
        )

        # characteristicDistance of each lake/reservoir
        # first estimate per water body: the mean of its cells' values (arc-degrees per
        # day)
        lakeReservoirCharacteristicDistance = pcr.ifthen(
            pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
            pcr.areaaverage(characteristicDistance, self.WaterBodies.waterBodyIds),
        )
        # make sure all outflow is released outside lakes and reservoirs
        # outlets: boolean map, True at WaterBodies.waterBodyOut; distance_to_outlets
        # (arc-degrees, along the ldd) from each water-body cell to its outlet
        outlets = pcr.cover(
            pcr.ifthen(
                pcr.scalar(self.WaterBodies.waterBodyOut) > 0,
                pcr.spatial(pcr.boolean(1)),
            ),
            pcr.spatial(pcr.boolean(0)),
        )
        distance_to_outlets = pcr.ifthen(
            pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
            pcr.ldddist(self.lddMap, outlets, pcr.scalar(1.0)),
        )

        # at least the distance to the outlet plus 2.5 downstream cell lengths
        # (pcr.downstreamdist, arc-degrees), then one value per body (its maximum)
        lakeReservoirCharacteristicDistance = pcr.ifthen(
            pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
            pcr.max(
                distance_to_outlets + pcr.downstreamdist(self.lddMap) * 2.50,
                lakeReservoirCharacteristicDistance,
            ),
        )
        lakeReservoirCharacteristicDistance = pcr.areamaximum(
            lakeReservoirCharacteristicDistance, self.WaterBodies.waterBodyIds
        )
        # TODO: calculate lakeReservoirCharacteristicDistance while obtaining the lake and reservoir parameters

        characteristicDistance = pcr.cover(
            lakeReservoirCharacteristicDistance, characteristicDistance
        )

        # note: in accutraveltime, a characteristicDistance (velocity) of 0 gives zero accutraveltimestate
        # and a very high accutraveltimeflux

        # TODO: consider using the downstreamdist function

        # current solution: roundup to ignore zero and very small values (arcDeg/day)
        characteristicDistance = pcr.roundup(characteristicDistance * 100.0) / 100.0

        # minimum characteristicDistance
        characteristicDistance = pcr.cover(
            characteristicDistance, 0.1 * self.cellSizeInArcDeg
        )
        # TODO: check the minimum distance for the accutraveltime function
        characteristicDistance = pcr.max(
            0.100 * self.cellSizeInArcDeg, characteristicDistance
        )

        return characteristicDistance

    def accuTravelTime(self):
        """
        Route channelStorage (m3 per cell) in one daily step with PCRaster
        accutraveltimeflux / accutraveltimestate, using the characteristic distance of
        getCharacteristicDistance (arc-degrees per day).

        Sets self.Q (m3/day per cell), self.channelStorage (m3, after routing) and
        self.subDischarge = Q / 86400 (m3/s). Negative storage and the fraction below 1
        m3 of each cell are not routed (channelStorageThatWillNotMove). Water bodies:
        called after WaterBodies.update and after the release has been spread over the
        body's cells (simple_update); the body's characteristic distance (at least the
        distance to the outlet plus 2.5 cells) carries the release past the outlet
        within the day.

        CAUTION (R7): upstream river water that reaches a body cell during this routing
        travels on with the same large characteristic distance, so it can pass the
        outlet within the day without ever entering WaterBodies.waterBodyStorage: it is
        neither inflow nor release of the water body (K1 of the gross-vs-net live log;
        by code reading, not verified with a run). Used by the RAWS 30 arcmin and 30
        arcsec configurations.
        """

        # accuTravelTime routing

        # only route non-negative channelStorage (otherwise it stays)
        channelStorageThatWillNotMove = pcr.ifthenelse(
            self.channelStorage < 0.0, self.channelStorage, 0.0
        )
        self.channelStorage = pcr.max(0.0, self.channelStorage)

        # at least 1 m3 of water stays, to minimize numerical errors of the float32 PCRaster implementation
        channelStorageThatWillNotMove += self.channelStorage - pcr.rounddown(
            self.channelStorage
        )
        self.channelStorage = pcr.rounddown(self.channelStorage)

        # channelStorage passed to the routing
        channelStorageForAccuTravelTime = pcr.max(0.0, self.channelStorage)
        # TODO: check why the cover operation is needed
        channelStorageForAccuTravelTime = pcr.cover(
            channelStorageForAccuTravelTime, 0.0
        )

        characteristicDistance = self.getCharacteristicDistance(self.yMean, self.wMean)

        # channel discharge (m3/day)
        self.Q = pcr.accutraveltimeflux(
            self.lddMap,
            channelStorageForAccuTravelTime,
            pcr.max(0.0, characteristicDistance),
        )
        self.Q = pcr.cover(self.Q, 0.0)
        # for very small velocities (characteristicDistanceForAccuTravelTime), discharge can be missing, see
        # http://sourceforge.net/p/pcraster/bugs-and-feature-requests/543/ and
        # http://karssenberg.geo.uu.nl/tt/TravelTimeSpecification.htm; also avoid negative discharge (m3/day)
        self.Q = pcr.max(0.0, self.Q)

        # update channelStorage after routing (m3)
        self.channelStorage = pcr.accutraveltimestate(
            self.lddMap,
            channelStorageForAccuTravelTime,
            pcr.max(0.0, characteristicDistance),
        )

        # return channelStorageThatWillNotMove to channelStorage (m3)
        self.channelStorage += channelStorageThatWillNotMove

        # for non-kinematic wave approaches, subDischarge is Q in m3/s
        self.subDischarge = self.Q / vos.secondsPerDay()
        self.subDischarge = pcr.ifthen(self.landmask, self.subDischarge)

    def estimate_length_of_sub_time_step(self):
        """
        Length (s) and number of the sub-steps of the kinematic methods for the whole
        day.

        From the previous sub-step discharge self.subDischarge (m3/s), self.water_height
        (m) and self.dynamicFracWat: the shortest residence time of the cells with a
        water height above critical_water_height (0.25 m, excluding pits) gives the
        number of sub-steps, times 1.25 plus 1, at least limit_num_of_sub_time_steps
        (see __init__). One value for the whole map. Water bodies: under
        simplifiedKinematicWave the body's storage has already been moved into
        WaterBodies when this is called; under kinematicWave it has not (the body's
        storage share is still in the channel storage of its cells).
        """

        # sub-time step length (s), the shorter the better, estimated from the initial or latest
        # sub-time step discharge (m3/s)
        length_of_sub_time_step = pcr.ifthenelse(
            self.subDischarge > 0.0,
            self.water_height * self.dynamicFracWat * self.cellArea / self.subDischarge,
            vos.secondsPerDay(),
        )

        # number of sub-time steps (Rens van Beek's method)
        critical_condition = (
            (length_of_sub_time_step < vos.secondsPerDay())
            & (self.water_height > self.critical_water_height)
            & (self.lddMap != pcr.ldd(5))
        )
        number_of_sub_time_steps = vos.secondsPerDay() / pcr.cover(
            pcr.areaminimum(
                pcr.ifthen(critical_condition, length_of_sub_time_step), self.landmask
            ),
            vos.secondsPerDay() / self.limit_num_of_sub_time_steps,
        )
        number_of_sub_time_steps = 1.25 * number_of_sub_time_steps + 1
        number_of_sub_time_steps = pcr.roundup(number_of_sub_time_steps)
        # at least one sub-time step
        number_of_loops = max(
            1.0, pcr.cellvalue(pcr.mapmaximum(number_of_sub_time_steps), 1)[1]
        )
        number_of_loops = int(max(self.limit_num_of_sub_time_steps, number_of_loops))

        # actual sub-time step length (s)
        length_of_sub_time_step = vos.secondsPerDay() / number_of_loops

        return (length_of_sub_time_step, number_of_loops)

    def simplifiedKinematicWave(self, meteo, landSurface, groundwater):
        """
        The 'simplifiedKinematicWave':
        1. First, assume that all local fluxes has been added to 'channelStorage'. This is done outside of this function/method.
        2. Then, the 'channelStorage' is routed by using 'pcr.kinematic function' with 'lateral_inflow' = 0.0.

        Annotation (water bodies). Called by simple_update after WaterBodies.update and
        after the day's release has been added to channelStorage at the outlet cell
        only. The body's own storage is not in channelStorage at this point (it was
        moved into WaterBodies), so the body's cells hold only remainders below 1 m3,
        the release (outlet cell) and whatever arrives from upstream during the day;
        they are routed like river cells. The sub-step loop routes channel water only
        and never calls WaterBodies.update (the release rule runs once per day). Sets
        self.Q (m3/day per cell, sum over the sub-steps), self.subDischarge (m3/s, last
        sub-step), self.channelStorage (m3), and self.dynamicFracWat and
        self.water_height (values of the last sub-step); with quality, runs
        qualityRouting at every sub-step (water quality only). The arguments meteo,
        landSurface and groundwater are not used. See CAUTION (R2) and (R3) in the loop.
        """

        logger.info("Using the simplifiedKinematicWave method!")

        # only route non-negative channelStorage (otherwise it stays)
        channelStorageThatWillNotMove = pcr.ifthenelse(
            self.channelStorage < 0.0, self.channelStorage, 0.0
        )

        # channelStorage passed to the routing (m3)
        channelStorageForRouting = pcr.max(0.0, self.channelStorage)

        # water height (m), needed to estimate the sub-time step length and the channel wetted area (for
        # alpha and dischargeInitial)
        # (start of the day; used here only to estimate the number of sub-steps)
        self.water_height = pcr.min(
            self.max_water_height,
            channelStorageForRouting
            / (
                pcr.max(self.min_fracwat_for_water_height, self.dynamicFracWat)
                * self.cellArea
            ),
        )

        # sub-time step length (s)
        length_of_sub_time_step, number_of_loops = (
            self.estimate_length_of_sub_time_step()
        )

        for i_loop in range(number_of_loops):

            # alpha parameter and initial discharge for the kinematic wave
            # floodplain on (all RAWS configurations): kinAlpha returns (flooded
            # fraction including the channel fraction (-), flood depth above the
            # floodplain (m), alpha, initial discharge (m3/s)); the first two are stored
            # as self.dynamicFracWat and self.water_height, so inside the loop
            # water_height is the flood depth, not storage / (fraction * cellArea).
            # kinAlpha also overwrites self.channelFraction. The fraction is floored at
            # WaterBodies.fracWat (water-body cells) and capped at 1.
            if self.floodPlain:
                self.dynamicFracWat, self.water_height, alpha, dischargeInitial = (
                    self.kinAlpha(channelStorageForRouting)
                )
                self.dynamicFracWat = pcr.min(
                    pcr.max(self.dynamicFracWat, self.WaterBodies.fracWat), 1.0
                )
            else:
                # alpha parameter and initial discharge for the kinematic wave
                alpha, dischargeInitial = (
                    self.calculate_alpha_and_initial_discharge_for_kinematic_wave(
                        channelStorageForRouting,
                        self.water_height,
                        self.innundatedFraction,
                        self.floodDepth,
                    )
                )

            # at the lake/reservoir outlets, use the water body outflow
            # CAUTION: the water-body outflow is used only as the initial (old)
            # discharge of each sub-step, see (R2) and (R3) below.
            #
            # WaterBodies.waterBodyOutflow is the release in m3 per day (time step of
            # 86400 s); at the outlet cell it is converted to m3/s; the other body cells
            # get 0, cells outside water bodies keep the kinematic-wave estimate.
            #
            # CAUTION (R2): pcr.kinematic computes the new discharge of each cell from
            # this old value and the new discharge arriving from its upstream cells.
            # Water arriving from upstream during the day therefore flows through the
            # body's cells and past the outlet without entering
            # WaterBodies.waterBodyStorage: it is counted neither as inflow nor as
            # release of the water body, and the rule cannot hold it back (bypass).
            #
            # CAUTION (R3): the release is not enforced either. With little upstream
            # inflow the new discharge at the outlet is below the old one
            # (kinematic-wave recession), so part of the release can remain in the
            # outlet cell at the end of the day; with upstream water passing through,
            # the outlet discharge can exceed the release.
            # return_water_body_storage_to_channel then merges what is left in the
            # body's cells into the water-body storage, and WaterBodies counts it as
            # inflow the next day. The magnitudes are not determined from the code (a
            # run is needed). Consequence for a replacement release scheme: the volume
            # leaving the body differs from the decided release, and the decision
            # signals (storage, avgInflow) include undeparted release and miss bypassed
            # inflow.
            waterBodyOutflowInM3PerSec = (
                pcr.cover(
                    pcr.ifthen(
                        self.WaterBodies.waterBodyOut, self.WaterBodies.waterBodyOutflow
                    ),
                    0.0,
                )
                / vos.secondsPerDay()
            )
            waterBodyOutflowInM3PerSec = pcr.ifthen(
                pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                waterBodyOutflowInM3PerSec,
            )
            dischargeInitial = pcr.cover(waterBodyOutflowInM3PerSec, dischargeInitial)

            # discharge (m3/s) from the kinematic wave approximation
            # new discharge of the sub-step (m3/s per cell), lateral inflow 0, one time
            # slice
            self.subDischarge = pcr.kinematic(
                self.lddMap,
                dischargeInitial,
                0.0,
                alpha,
                self.beta,
                1,
                length_of_sub_time_step,
                self.channelLength,
            )
            self.subDischarge = pcr.cover(self.subDischarge, 0.0)
            self.subDischarge = pcr.max(0.0, pcr.cover(self.subDischarge, 0.0))

            # avoid negative channel storage
            # outflow volume of a cell in one sub-step limited to its storage plus the
            # inflow from upstream in the same sub-step (m3), converted back to m3/s
            self.subDischarge = (
                pcr.min(
                    self.subDischarge * length_of_sub_time_step,
                    pcr.max(
                        0.0,
                        channelStorageForRouting
                        + pcr.upstream(
                            self.lddMap, self.subDischarge * length_of_sub_time_step
                        ),
                    ),
                )
                / length_of_sub_time_step
            )

            # update channelStorage (m3)
            storage_change_in_volume = (
                pcr.upstream(self.lddMap, self.subDischarge * length_of_sub_time_step)
                - self.subDischarge * length_of_sub_time_step
            )
            channelStorageForRouting += storage_change_in_volume
            # only route non-negative channelStorage (otherwise it stays)
            channelStorageThatWillNotMove += pcr.ifthenelse(
                channelStorageForRouting < 0.0, channelStorageForRouting, 0.0
            )
            channelStorageForRouting = pcr.max(0.000, channelStorageForRouting)

            # update the flood fraction and flood depth (without floodplain)
            # CAUTION (R9): returnInundationFractionAndFloodDepth is stored in
            # self.inundatedFraction (one n), while dynamicFracWat below reads
            # self.innundatedFraction (two n), the start-of-day value from update: the
            # inundated fraction is not updated within the loop. Same on master. Not
            # active in the RAWS configurations (dynamicFloodPlain = True).
            if not self.floodPlain:
                self.inundatedFraction, self.floodDepth = (
                    self.returnInundationFractionAndFloodDepth(channelStorageForRouting)
                )

                # dynamicFracWat: fraction of surface water bodies (-), including lakes and reservoirs; lake and
                # reservoir surface water fraction
                self.dynamicFracWat = pcr.cover(
                    pcr.min(1.0, self.WaterBodies.fracWat), 0.0
                )
                # fraction of channel (including its excess above bankfull capacity)
                self.dynamicFracWat += pcr.max(
                    0.0, 1.0 - self.dynamicFracWat
                ) * pcr.max(self.channelFraction, self.innundatedFraction)

                # dynamicFracWat is at most 1
                self.dynamicFracWat = pcr.ifthen(
                    self.landmask,
                    pcr.min(
                        pcr.max(self.dynamicFracWat, self.WaterBodies.fracWat), 1.0
                    ),
                )
                self.dynamicFracWat = pcr.ifthen(
                    self.landmask, pcr.min(1.0, self.dynamicFracWat)
                )
                self.dynamicFracWat = pcr.ifthen(
                    self.landmask, pcr.max(1e-6, self.dynamicFracWat)
                )

                # water height for the next loop, needed to estimate the channel wetted area (for alpha and dischargeInitial)
                self.water_height = channelStorageForRouting / (
                    pcr.max(self.min_fracwat_for_water_height, self.dynamicFracWat)
                    * self.cellArea
                )

            # total discharge volume (m3) until this loop
            # discharge_volume (m3 per cell): sum of the sub-step outflow volumes; at
            # the outlet cell this is the volume that actually left the water body that
            # day (release plus any pass-through, see R2 and R3)
            if i_loop == 0:
                discharge_volume = pcr.scalar(0.0)
            discharge_volume += self.subDischarge * length_of_sub_time_step

            # quality routing
            # DynQual: water-quality routing per sub-step on a copy of the storage
            # (channelStorageNow); channelStorageForRouting and subDischarge are not
            # changed
            if self.quality:
                self.channelStorageNow = pcr.max(0.0, channelStorageForRouting)
                self.qualityRouting(length_of_sub_time_step)
                self.channelStorageTimeBefore = pcr.max(0.0, self.channelStorageNow)

        # channel discharge (m3/day)
        # (INTERFACE: Q / 86400 is self.discharge in simple_update)
        self.Q = discharge_volume

        # update channelStorage after routing
        self.channelStorage = channelStorageForRouting

        # return channelStorageThatWillNotMove to channelStorage
        self.channelStorage += channelStorageThatWillNotMove

        # channel storage cannot be negative
        # (clipping creates water where storage was negative; not on master)
        self.channelStorage = pcr.max(0.0, self.channelStorage)

    def update(self, landSurface, groundwater, currTimeStep, meteo):
        """
        Daily routing step.

        INTERFACE: called by pcrglobwb.py (Model.update) after meteo.update,
        landSurface.update and groundwater.update of the same day.

        Water-body sequence: (1) water-body maps from WaterBodies.getParameterFiles at
        the first time step (with the initial states from getState) and on 1 January;
        (2) downstreamDemand from the previous day's discharge statistics; (3) channel
        geometry and the start-of-day water fraction dynamicFracWat (used for open-water
        evaporation); (4) the method-specific update, which contains the
        WaterBodies.update call(s): simple_update (accuTravelTime,
        simplifiedKinematicWave; once per day) or kinematic_wave_update (kinematicWave;
        once per sub-step); (5) riverbed infiltration, readAvlChannelStorage for the
        next day, and the discharge statistics (calculate_statistics).

        Inputs used: landSurface (landSurfaceRunoff, actSurfaceWaterAbstract, return
        flows, actualET), groundwater (baseflow, riverBedConductivity, abstractions),
        meteo (referencePotET), currTimeStep (timeStepPCR, doy, day, fulldate).
        """

        logger.info("routing in progress")

        # water bodies: get the parameters at the start of each year or simulation; this must be called first
        # to define the initial conditions at the start of the simulation
        # INTERFACE: waterBodies.py, WaterBodies.getParameterFiles(currTimeStep,
        # cellArea, ldd, initial_condition_dictionary). At the first time step it also
        # receives getState() (the staged avgInflow, avgOutflow, waterBodyStorage and
        # channelStorage) and calls WaterBodies.getICs. It (re)builds, from
        # waterBodyInputNC (yearly slices): fracWat (-, per cell), waterBodyIds
        # (nominal, per body), waterBodyOut (boolean, the cell with the largest upstream
        # area), waterBodyArea (m2), waterBodyTyp (1 lake, 2 reservoir) and waterBodyCap
        # (m3), the last three per body on its cells. landSurface.py also reads
        # WaterBodies.fracWat (capillary rise) and sets it itself on the first day.
        if currTimeStep.timeStepPCR == 1:
            initial_conditions_for_water_bodies = self.getState()
            # the last argument is for the initial conditions of lakes/reservoirs
            self.WaterBodies.getParameterFiles(
                currTimeStep,
                self.cellArea,
                self.lddMap,
                initial_conditions_for_water_bodies,
            )

        # yearly reload on 1 January (a new year may add reservoirs or change extents).
        # CAUTION (R18): WaterBodies keeps its storage and running means per cell across
        # the reload: a new reservoir starts with storage 0 and the small positive
        # avgOutflow its cells carried (N4); the river water already on newly added body
        # cells counts as inflow on the first day; cells that leave a body keep their
        # storage share in channelStorage, as river water. For a body whose extent
        # changed, WaterBodies' storage differs between its old and new cells at the
        # first update, so its inflow and avgInflow (30-day window) differ per cell, and
        # so can the release, which triggers R1. How often the extents change in the
        # input file is not determined from the code.
        if (currTimeStep.doy == 1) and (currTimeStep.timeStepPCR > 1):
            self.WaterBodies.getParameterFiles(currTimeStep, self.cellArea, self.lddMap)
        # CAUTION (R6): only waterBodyIds is replaced (see __init__)
        if not self.includeWaterBodies:
            # ignore all lakes and reservoirs
            self.WaterBodies.waterBodyIds = pcr.ifthen(self.landmask, pcr.nominal(-1))

        # downstream demand (m3/s) for reservoirs, estimated from the environmental flow discharge; must be
        # called before updating timestepsToAvgDischarge
        # INTERFACE: passed to WaterBodies.update (simple_update,
        # kinematic_wave_update). downstreamDemand (m3/s per cell) = max(0.1 *
        # avgDischarge, avgDischarge - 1.2816 * std), from the previous day's
        # statistics: an environmental-flow discharge of the cell itself, not a
        # downstream withdrawal demand. At water-body cells it is computed from the
        # statistics of the body-maximum discharge; WaterBodies uses it per cell and the
        # outlet value is the one that reaches the channel (R1).
        #
        # CAUTION (N2): its floor on the reservoir release is negligible as
        # WaterBodies.getReservoirOutflow is written.
        #
        # CAUTION (R4): the std estimate grows with run length, so the value tends to
        # 0.1 * avgDischarge.
        #
        # EXTENSION POINT: a demand-aware scheme can receive a real downstream demand
        # signal here instead (e.g. the surface-water demand of the cells downstream of
        # each outlet, summed per body), with the same units (m3/s) and support (per
        # body on its cells).
        self.downstreamDemand = self.estimate_discharge_for_environmental_flow(
            self.channelStorage
        )

        # routing/channel parameters (based on avgDischarge) and water body fraction, needed for the
        # evaporation from water bodies
        # (self.characteristicDistance is not used: accuTravelTime calls
        # getCharacteristicDistance)
        self.yMean, self.wMean, self.characteristicDistance = (
            self.getRoutingParamAvgDischarge(self.avgDischarge, self.dist2celllength)
        )

        # channel width (m), depth (m) and fraction
        # channelWidth (m), channelDepth (m, from avgDischarge; constantChannelDepth is
        # used only for the bankfull capacity, unlike master) and channelFraction (-),
        # also at water-body cells
        self.channelWidth = self.wMean
        self.channelDepth = pcr.max(0.0, self.yMean)
        self.channelFraction = pcr.max(
            0.0, pcr.min(1.0, self.channelWidth * self.channelLength / (self.cellArea))
        )

        # water height for the first time step
        # CAUTION (R17): this first-time-step block has no lasting effect:
        # dynamicFracWat is recomputed unconditionally below, and water_height is
        # recomputed before the kinematic methods use it
        if currTimeStep.timeStepPCR == 1:
            if self.floodPlain:
                self.dynamicFracWat, self.water_height = self.returnFloodedFraction(
                    self.channelStorage
                )
                self.dynamicFracWat = pcr.min(
                    pcr.max(self.dynamicFracWat, self.WaterBodies.fracWat), 1.0
                )
            else:
                self.dynamicFracWat = pcr.max(
                    self.channelFraction, self.WaterBodies.fracWat
                )
            self.dynamicFracWat = pcr.ifthen(self.landmask, self.dynamicFracWat)

        # channel bankfull capacity (m3)
        # (m3 per cell; fixed in the RAWS 5 arcmin configurations, from constant width
        # and depth)
        if self.floodPlain:
            if self.usingFixedBankfullCapacity:
                self.channelStorageCapacity = self.predefinedBankfullCapacity
            else:
                self.channelStorageCapacity = self.estimateBankfullCapacity(
                    self.channelWidth, self.channelDepth
                )

        # flood inundation fraction (-) and depth (m)
        # from the end-of-previous-day channelStorage; at water-body cells it holds the
        # cell's share of the lake/reservoir storage, so a full reservoir appears as a
        # large flood volume above the river bankfull capacity of the cell
        self.innundatedFraction, self.floodDepth = (
            self.returnInundationFractionAndFloodDepth(self.channelStorage)
        )
        # dynamicFracWat: fraction of surface water bodies (-), including lakes and reservoirs; lake and
        # reservoir surface water fraction
        # dynamicFracWat (-) = fracWat + (1 - fracWat) * max(channelFraction, inundated
        # fraction), at least fracWat and 1e-6, at most 1. This start-of-day value sets
        # the open-water evaporation of the day (calculate_evaporation).
        #
        # CAUTION (R8): at water-body cells with fracWat < 1, the inundated fraction
        # comes from the cell's floodplain elevation profile applied to the reservoir
        # storage share (with dynamicFloodPlain), not from the reservoir's own
        # area-storage relation, so the evaporating area grows with storage in a way the
        # reservoir data do not define. During simplifiedKinematicWave routing,
        # dynamicFracWat is overwritten per sub-step from the channel water only (the
        # body storage is then inside WaterBodies); that end-of-routing value is the one
        # used for riverbed infiltration (calculate_exchange_to_groundwater) and
        # reported (reporting.py, dynamicFracWat). Under accuTravelTime it is not
        # changed during routing.
        self.dynamicFracWat = pcr.cover(pcr.min(1.0, self.WaterBodies.fracWat), 0.0)
        # fraction of channel (including its excess above bankfull capacity)
        self.dynamicFracWat += pcr.max(0.0, 1.0 - self.dynamicFracWat) * pcr.max(
            self.channelFraction, self.innundatedFraction
        )
        # dynamicFracWat is at most 1
        self.dynamicFracWat = pcr.ifthen(
            self.landmask,
            pcr.min(pcr.max(self.dynamicFracWat, self.WaterBodies.fracWat), 1.0),
        )
        self.dynamicFracWat = pcr.ifthen(
            self.landmask, pcr.min(1.0, self.dynamicFracWat)
        )
        self.dynamicFracWat = pcr.ifthen(
            self.landmask, pcr.max(1e-6, self.dynamicFracWat)
        )

        # routing methods
        # accuTravelTime and simplifiedKinematicWave: WaterBodies.update once per day
        # (simple_update). kinematicWave: once per sub-step (kinematic_wave_update).
        if self.method == "accuTravelTime" or self.method == "simplifiedKinematicWave":
            self.simple_update(landSurface, groundwater, currTimeStep, meteo)
        if self.method == "kinematicWave":
            self.kinematic_wave_update(landSurface, groundwater, currTimeStep, meteo)
        # note: this method requires abstraction from fossil groundwater

        # infiltration from surface water bodies (channels, lakes and reservoirs) to groundwater, passed on in
        # the next time step
        # riverbedExchange (m3/day) is taken from channelStorage after the water-body
        # storage has been spread back over the body's cells, so at body cells it is
        # lake/reservoir water; it reaches WaterBodies only as a lower storage, i.e.
        # inside the next day's inflow residual
        self.calculate_exchange_to_groundwater(groundwater, currTimeStep)

        # volume released in pits (losses to the ocean/endorheic basins)
        # (m3/day; INTERFACE: pcrglobwb.py, water balance at pits)
        self.outgoing_volume_at_pits = pcr.ifthen(
            self.landmask, pcr.cover(pcr.ifthen(self.lddMap == pcr.ldd(5), self.Q), 0.0)
        )

        # volume that can be abstracted in the next time step
        # readAvlChannelStorage (m3 per cell) for the next day's surface-water
        # allocation. INTERFACE: water_management/main_water_management.py (update,
        # available_surface_water_volume); not read when using_qualloc is True. At
        # water-body cells it is computed from the cell's share of the lake/reservoir
        # storage, so lakes and reservoirs are an abstraction source, with no lower
        # limit at minResvrFrac (K2 of the gross-vs-net live log).
        #
        # CAUTION (N5): the RAWS configurations set allocation-zone keys that the RAWS
        # code does not read, so surface water is allocated locally and a reservoir's
        # cells supply only the demand located on those cells.
        #
        # CAUTION (R15): computed before calculate_statistics, i.e. with the statistics
        # of the previous day (on master: after).
        #
        # EXTENSION POINT: masking water-body cells here, or flooring the available
        # volume at a reservoir's lower limit (e.g. minResvrFrac * waterBodyCap spread
        # by cell area), changes what demand can take from reservoirs. Contract: m3 per
        # cell, >= 0, <= channelStorage, defined on the landmask. The QUAlloc path reads
        # channelStorage instead (landSurface.py).
        self.readAvlChannelStorage = pcr.max(
            0.0, self.estimate_available_volume_for_abstraction(self.channelStorage)
        )

        # water body balance check
        # (m over the body area; WaterBodies' own balance residual, diagnostic only)
        self.waterBodyBalance = self.WaterBodies.waterBodyBalance

        # total runoff from local land surface runoff and local changes in water bodies (m)
        # (m/day per cell: runoff, abstraction, return flow, evaporation and riverbed
        # infiltration; no reader outside this file, reporting.py builds its own
        # totalRunoff)
        self.totalRunoff = self.local_input_to_surface_water / self.cellArea

        # statistics of long- and short-term flow values
        # (also copies WaterBodies.avgInflow and WaterBodies.avgOutflow into routing's
        # states)
        self.calculate_statistics(groundwater, landSurface)

        # old-style reporting; TODO: remove
        self.old_style_routing_reporting(currTimeStep)

    def calculate_potential_evaporation(
        self, landSurface, currTimeStep, meteo, definedDynamicFracWat=None
    ):
        """
        Potential open-water evaporation over the whole cell area (m/day per cell);
        returned, not stored.

        E_pot = max(0, kc * ET0 - ETact) * dynamicFracWat, with kc the monthly
        open-water crop coefficient (cropCoefficientWaterNC, at least minCropWaterKC;
        read on the first day of each month and at the first time step), ET0 =
        meteo.referencePotET (m/day) and ETact = landSurface.actualET (m/day, over the
        whole cell, all land-cover tiles). Only the part of kc * ET0 not already
        evaporated by the land-surface scheme is taken from surface water. INTERFACE:
        meteo.py (referencePotET), landSurface.py (actualET). Water bodies:
        dynamicFracWat at body cells is at least WaterBodies.fracWat (CAUTION R8 in
        update).

        CAUTION (R16): if definedDynamicFracWat is passed (no caller does), the local
        name dynamicFracWat is undefined and the call fails.
        """

        if not self.no_zero_crop_water_coefficient:
            self.waterKC = 0.0

        # potential evaporation from water bodies; if landSurface.actualET < waterKC *
        # meteo.referencePotET * fracWat, more evaporation is added
        # CAUTION: the code compares actualET with waterKC * referencePotET (no fracWat
        # factor) and multiplies the excess by dynamicFracWat afterwards
        if (
            (currTimeStep.day == 1) or (currTimeStep.timeStepPCR == 1)
        ) and self.no_zero_crop_water_coefficient:
            waterKC = vos.netcdf2PCRobjClone(
                self.fileCropKC,
                "kc",
                currTimeStep.fulldate,
                useDoy="month",
            )
            self.waterKC = pcr.ifthen(self.landmask, pcr.cover(waterKC, 0.0))
            self.waterKC = pcr.max(self.minCropWaterKC, self.waterKC)

        # potential evaporation from water bodies (m/day), reduced by the evaporation calculated in the
        # landSurface module; not over the entire cell area
        # (m/day per unit of water surface)
        waterBodyPotEvapOvesSurfaceWaterArea = pcr.ifthen(
            self.landmask,
            pcr.max(0.0, self.waterKC * meteo.referencePotET - landSurface.actualET),
        )

        # potential evaporation from water bodies over the entire cell area (m/day)
        # (CAUTION R16: no else branch)
        if definedDynamicFracWat is None:
            dynamicFracWat = self.dynamicFracWat
        waterBodyPotEvap = pcr.max(
            0.0, waterBodyPotEvapOvesSurfaceWaterArea * dynamicFracWat
        )
        return waterBodyPotEvap

    def calculate_evaporation(self, landSurface, groundwater, currTimeStep, meteo):
        """
        Open-water evaporation taken from channelStorage (simple_update only;
        kinematic_wave_update computes its own per sub-step).

        Sets self.waterBodyPotEvap (m/day over the cell), self.waterBodyEvaporation
        (actual, m/day over the cell; volume = min(max(0, channelStorage),
        waterBodyPotEvap * cellArea)) and self.remainWaterBodyPotEvap (m/day, unmet
        part; no reader in src/). Water bodies: called before the water-body storage is
        moved into WaterBodies, so at body cells the evaporation is drawn from the
        cell's storage share plus the day's local inputs; it lowers the inflow residual
        seen by WaterBodies, which never receives evaporation as a separate term.

        INTERFACE: reporting.py (waterBodyActEvaporation, waterBodyPotEvaporation,
        totalEvaporation) and pcrglobwb.py (water-balance accumulators).

        EXTENSION POINT: the per-body evaporation volume
        pcr.areatotal(volLocEvapWaterBody, self.WaterBodies.waterBodyIds) (m3/day) is
        the open-water loss to pass to WaterBodies.update if a scheme needs a gross
        inflow.
        """

        # potential evaporation from water bodies over the entire cell area, not only the water bodies (m/day)
        self.waterBodyPotEvap = self.calculate_potential_evaporation(
            landSurface, currTimeStep, meteo
        )

        # evaporation volume from water bodies (m3), not limited to the available channelStorage
        # (m3/day per cell)
        volLocEvapWaterBody = self.waterBodyPotEvap * self.cellArea
        # limited to the available channelStorage
        volLocEvapWaterBody = pcr.min(
            pcr.max(0.0, self.channelStorage), volLocEvapWaterBody
        )

        # update channelStorage (m3) after evaporation from water bodies
        # at water-body cells this removes water from the lake/reservoir storage share
        # (moved into WaterBodies later in simple_update)
        self.channelStorage = self.channelStorage - volLocEvapWaterBody
        self.local_input_to_surface_water -= volLocEvapWaterBody

        # evaporation from water bodies (m)
        # (m/day over the whole cell area, not over the water surface)
        self.waterBodyEvaporation = volLocEvapWaterBody / self.cellArea
        self.waterBodyEvaporation = pcr.ifthen(self.landmask, self.waterBodyEvaporation)

        # remaining potential evaporation from water bodies (m)
        # (not read anywhere in src/; not set by kinematic_wave_update)
        self.remainWaterBodyPotEvap = pcr.max(
            0.0, self.waterBodyPotEvap - self.waterBodyEvaporation
        )

    def calculate_exchange_to_groundwater(self, groundwater, currTimeStep):
        """
        Riverbed infiltration from surface water to groundwater, self.riverbedExchange
        (m3/day per cell), subtracted from channelStorage after routing and after the
        water-body storage has been spread back over the body's cells.

        Rate: min(0.1, riverBedConductivity) (m/day) * dynamicFracWat * cellArea, only
        where 0 < baseflow < total groundwater abstraction, at most 75 percent of the
        (non-negative) channel storage.

        INTERFACE: reads groundwater.py (riverBedConductivity, nonFossilGroundwaterAbs,
        fossilGroundwaterAbstr, baseflow; m/day); groundwater.py (update_storage) adds
        riverbedExchange / cellArea to groundwater storage on the next day. Water
        bodies: at body cells the flux is drawn from the lake/reservoir storage share
        and reaches WaterBodies only as a negative term in the next day's inflow
        residual. dynamicFracWat here is the end-of-routing value (CAUTION R8 in
        update).

        EXTENSION POINT: the per-body seepage pcr.areatotal(self.riverbedExchange,
        self.WaterBodies.waterBodyIds) (m3/day) can be passed to WaterBodies.update on
        the next day.
        """

        if self.debugWaterBalance:
            # (m3)
            preStorage = self.channelStorage

        # riverbed infiltration (m3/day), following Inge's principle: only if 0 < baseflow < total groundwater abstraction
        # (fossil and non-fossil); the rate is based on the aquifer saturated conductivity, limited to fracWat
        # and the available channelStorage, and passed to groundwater in the next time step (de Graaf et al.,
        # 2014; Wada et al., 2012; Wada et al., 2010); TODO: improve this concept
        # (m/day)
        riverbedConductivity = groundwater.riverBedConductivity
        # maximum conductivity of 0.1 m/day (Marc Bierkens: resistance of 1 day for a 0.1 m river bed)
        riverbedConductivity = pcr.min(0.1, riverbedConductivity)
        # (m)
        total_groundwater_abstraction = pcr.max(
            0.0,
            groundwater.nonFossilGroundwaterAbs + groundwater.fossilGroundwaterAbstr,
        )
        self.riverbedExchange = pcr.max(
            0.0,
            pcr.min(
                pcr.max(0.0, self.channelStorage),
                pcr.ifthenelse(
                    groundwater.baseflow > 0.0,
                    pcr.ifthenelse(
                        total_groundwater_abstraction > groundwater.baseflow,
                        riverbedConductivity * self.dynamicFracWat * self.cellArea,
                        0.0,
                    ),
                    0.0,
                ),
            ),
        )
        self.riverbedExchange = pcr.cover(self.riverbedExchange, 0.0)
        # avoid flip-flopping
        factor = 0.25
        self.riverbedExchange = pcr.min(
            self.riverbedExchange,
            (1.0 - factor) * pcr.max(0.0, self.channelStorage),
        )
        self.riverbedExchange = pcr.ifthenelse(
            self.channelStorage < 0.0, 0.0, self.riverbedExchange
        )
        self.riverbedExchange = pcr.cover(self.riverbedExchange, 0.0)
        self.riverbedExchange = pcr.ifthen(self.landmask, self.riverbedExchange)

        # update channelStorage (m3) after riverbedExchange (m3)
        # (at water-body cells: lake/reservoir water)
        self.channelStorage -= self.riverbedExchange
        self.local_input_to_surface_water -= self.riverbedExchange

        if self.debugWaterBalance:
            vos.waterBalanceCheck(
                [pcr.scalar(0.0)],
                [self.riverbedExchange / self.cellArea],
                [preStorage / self.cellArea],
                [self.channelStorage / self.cellArea],
                "channelStorage after surface water infiltration",
                True,
                currTimeStep.fulldate,
                threshold=1e-4,
            )

    def return_flows_to_wastewater_treatment_plants(self, currTimeStep, landSurface):
        """
        Redirect domestic and manufacturing return flows to wastewater-treatment-plant
        cells (WWtPlantsNC) and set self.nonIrrReturnFlow (m3/day per cell) for
        simple_update.

        INTERFACE: landSurface.nonIrrReturnFlowVolumePerSector (m3/day per cell and
        sector; landSurface.py, standard or QUAlloc branch of update). Within a
        treatment zone (zone_id != 0) the domestic + manufacturing return flows of all
        cells are summed (pcr.areatotal over zone_id) and put at the cell where zone_id
        == plant_id; the other cells of the zone get 0; cells outside zones keep their
        own value. Livestock, thermoelectric and industry return flows stay at their own
        cells. The behaviour where plant_id is missing is not determined from the code.
        Water bodies: a lake/reservoir cell can lose its local return flow, or receive a
        whole zone's return flow if the plant cell lies in the body; either way it
        enters that day's inflow residual.
        """

        # wastewater treatment plants: ids
        # CAUTION (R14): the zone maps are read only when doy == 1, so a run that does
        # not start on 1 January fails at the first use below (attributes undefined; not
        # run)
        if currTimeStep.doy == 1:
            # wastewater treatment plant (point) locations
            self.WWt_plantID = vos.netcdf2PCRobjClone(
                self.WWtPlantsNC,
                "plant_id",
                str(currTimeStep.fulldate),
                useDoy="yearly",
                LatitudeLongitude=True,
                specificFillValue=None,
            )
            self.WWt_plantID = pcr.nominal(self.WWt_plantID)

            # wastewater treatment plant service zones
            self.WWt_zoneID = vos.netcdf2PCRobjClone(
                self.WWtPlantsNC,
                "zone_id",
                str(currTimeStep.fulldate),
                useDoy="yearly",
                LatitudeLongitude=True,
                specificFillValue=None,
            )
            self.WWt_zoneID = pcr.nominal(self.WWt_zoneID)

        # return flow: the sum of the domestic and manufacturing return flows only
        self.nonIrrReturnFlow = (
            landSurface.nonIrrReturnFlowVolumePerSector["domestic"]
            + landSurface.nonIrrReturnFlowVolumePerSector["manufacture"]
        )

        # redirect the return flow to the wastewater treatment plant locations, with removal at the plant (m3/day)
        # CAUTION: no removal is applied here; the zone's domestic and manufacturing
        # return flow is moved to the plant cell in full
        self.nonIrrReturnFlow = pcr.ifthenelse(
            # cells not in a wastewater treatment zone
            self.WWt_zoneID == 0,
            self.nonIrrReturnFlow,
            pcr.ifthenelse(
                self.WWt_zoneID
                # accumulate water over the treatment zone to the plant location
                == self.WWt_plantID,
                pcr.areatotal(
                    pcr.ifthenelse(self.WWt_zoneID != 0, self.nonIrrReturnFlow, 0),
                    self.WWt_zoneID,
                ),
                # after accumulation, set locations without a treatment plant to 0
                0.0,
            ),
        )

        # return flow of the non-irrigation water demand, calculated in landSurface.py (m3)
        # (livestock, thermoelectric and industry return flows stay at their own cells)
        self.nonIrrReturnFlow = (
            self.nonIrrReturnFlow
            + landSurface.nonIrrReturnFlowVolumePerSector["livestock"]
            + landSurface.nonIrrReturnFlowVolumePerSector["thermoelectric"]
            + landSurface.nonIrrReturnFlowVolumePerSector["industry"]
        )

    def simple_update(self, landSurface, groundwater, currTimeStep, meteo):
        """
        Daily update for accuTravelTime and simplifiedKinematicWave: local fluxes, the
        water-body step (WaterBodies.update, once per day) and the channel routing.

        Order (volumes in m3 per cell unless stated): (1) runoff + baseflow added to
        channelStorage; (2) optional upstream discharge from other sub-runs; (3)
        surface-water abstraction subtracted; (4) non-irrigation return flows added
        (moved to treatment plants if WWtPlantsNC is set); (5) open-water evaporation
        subtracted (calculate_evaporation); (6) at water-body cells the channel storage
        (rounded down to whole m3) is moved into the water body and WaterBodies.update
        is called once with a time step of 86400 s; (7) the release is put back into
        channelStorage (outlet cell for simplifiedKinematicWave, spread over the body's
        cells for accuTravelTime); (8) channel routing; (9) the water-body storage is
        spread back over the body's cells (return_water_body_storage_to_channel). Steps
        (1) to (5) act on every cell, water-body cells included, i.e. directly on the
        lake/reservoir storage share of those cells.

        What the release rule sees on day t: storage = end of day t-1 storage after
        release, plus the river water that reached the body's cells during the routing
        of day t-1 (merged in step 9 of day t-1, including any undeparted release, R3),
        minus the riverbed infiltration of day t-1, plus the local terms of day t on the
        body's cells (runoff, baseflow and return flows, minus abstraction and
        open-water evaporation). Upstream river water routed on day t reaches the
        storage only on day t+1 (one-day lag), except what passes through the body
        within the day (R2, R7). The release decided on day t is routed on day t.

        The same sequence applies to both RAWS 5 arcmin configurations. With
        online_pcrglobwb_qualloc_5arcminutes.ini (using_qualloc and quality True), the
        abstraction of step (3) comes from QUAlloc (CAUTION R12 in __init__), the return
        flows of step (4) from QUAlloc's per-sector return flows, and the DynQual steps
        marked below run in addition; they do not change any water quantity used here.
        """

        # update the time steps for the long- and short-term statistics (avgDischarge, avgInflow, avgOutflow, etc.)
        # (days; the same counter is passed to WaterBodies.update below; incremented
        # after downstreamDemand was computed in update)
        self.timestepsToAvgDischarge += 1.0

        if self.debugWaterBalance:
            # (m3)
            preStorage = self.channelStorage

        # total local change (input) to surface water storage (m3); only local processes, no routing;
        # initialized at zero
        # local_input_to_surface_water (m3 per cell): runoff, abstraction, return flow
        # and evaporation (and riverbed infiltration, in update); not the upstream input
        # nor the release. INTERFACE: reporting.py (local_water_body_flux, totalRunoff),
        # pcrglobwb.py (water balance).
        self.local_input_to_surface_water = pcr.scalar(0.0)

        # runoff from land surface cells (m/day)
        # INTERFACE: landSurface.landSurfaceRunoff (direct runoff + interflow, m/day;
        # landSurface.py) and groundwater.baseflow (m/day; groundwater.py), both over
        # the whole cell area. At water-body cells this is the only way precipitation
        # reaches the lake/reservoir: the land-cover fractions cover the whole cell
        # (landSurface.py), so rain on the water surface first passes through the soil
        # scheme. self.runoff is also read by QUAlloc on the next day and by
        # reporting.py.
        self.runoff = landSurface.landSurfaceRunoff + groundwater.baseflow

        # update channelStorage (m3) after runoff
        self.channelStorage += self.runoff * self.cellArea
        self.local_input_to_surface_water += self.runoff * self.cellArea

        # upstream discharge (m3/s)
        # optional (upstream_discharge_input_files): discharge (m3/s) of other sub-runs
        # at cells just outside the landmask, moved one cell downstream with
        # ldd_complete and added as a daily volume (m3/s * 86400). Not included in
        # local_input_to_surface_water, nor in the balance check below. If the receiving
        # cell is a water-body cell it becomes part of today's inflow residual.
        total_upstream_discharge = pcr.spatial(pcr.scalar(0.0))
        if self.upstream_discharge_input_files is not None:
            for i_ups_file in range(0, len(self.upstream_discharge_input_files)):
                upstream_discharge_input_file = self.upstream_discharge_input_files[
                    i_ups_file
                ]
                self.upstream_discharge = vos.readUpstreamDischarge(
                    upstream_discharge_input_file,
                    "automatic",
                    str(currTimeStep.fulldate),
                    useDoy=None,
                )
                total_upstream_discharge = total_upstream_discharge + pcr.cover(
                    self.upstream_discharge, 0.0
                )
            # add the upstream discharge to the current basin
            total_upstream_discharge = pcr.upstream(
                self.ldd_complete, total_upstream_discharge
            )
        # only values within the landmask
        self.total_upstream_discharge = pcr.ifthen(
            self.landmask, total_upstream_discharge
        )
        # add the upstream discharge to channelStorage (m3)
        self.channelStorage = (
            pcr.cover(self.total_upstream_discharge, 0.0) * 3600.0 * 24.0
            + self.channelStorage
        )

        # update channelStorage (m3) after actSurfaceWaterAbstraction
        # INTERFACE: landSurface.actSurfaceWaterAbstract (m/day per cell): the
        # surface-water withdrawal decided earlier on the same day, by
        # water_management/main_water_management.py (bounded by readAvlChannelStorage of
        # the previous day) or by QUAlloc (bounded by channelStorage / cellArea of the
        # previous day), copied by landSurface.py. At water-body cells it is taken from
        # the cell's share of the lake/reservoir storage before WaterBodies.update, so
        # the release rule sees post-withdrawal storage and the withdrawal also lowers
        # the inflow residual and avgInflow.
        #
        # EXTENSION POINT: the per-body withdrawal
        # pcr.areatotal(landSurface.actSurfaceWaterAbstract * self.cellArea,
        # self.WaterBodies.waterBodyIds) (m3/day) can be formed here and passed to
        # WaterBodies.update (for a gross inflow, or a demand-aware rule).
        self.channelStorage -= landSurface.actSurfaceWaterAbstract * self.cellArea
        self.local_input_to_surface_water -= (
            landSurface.actSurfaceWaterAbstract * self.cellArea
        )

        # channelStorage after surface water abstraction, for reporting (m3)
        # (diagnostic; not read elsewhere in src/)
        self.channelStorageAfterAbstraction = pcr.ifthen(
            self.landmask, self.channelStorage
        )

        # redirect the return flow to the wastewater treatment plant locations (m3)
        # non-irrigation return flows (m3/day per cell). INTERFACE: landSurface.py
        # (nonIrrReturnFlowVolume, nonIrrReturnFlowVolumePerSector). Irrigation return
        # flows reach the channel only through the soil (runoff and baseflow).
        if self.WWtPlants:
            self.return_flows_to_wastewater_treatment_plants(currTimeStep, landSurface)
        else:
            self.nonIrrReturnFlow = landSurface.nonIrrReturnFlowVolume

        # add the return flows to channelStorage (m3)
        # (at water-body cells: part of today's inflow residual)
        self.channelStorage += self.nonIrrReturnFlow
        self.local_input_to_surface_water += self.nonIrrReturnFlow

        # evaporation from water bodies (sets self.waterBodyEvaporation, in m)
        # subtracts open-water evaporation (m3) on every cell, water-body cells
        # included, before the storage is moved into the water bodies
        self.calculate_evaporation(landSurface, groundwater, currTimeStep, meteo)

        # per-cell check in m over the cell area (log message only)
        if self.debugWaterBalance:
            vos.waterBalanceCheck(
                [self.runoff, self.nonIrrReturnFlow / self.cellArea],
                [landSurface.actSurfaceWaterAbstract, self.waterBodyEvaporation],
                [preStorage / self.cellArea],
                [self.channelStorage / self.cellArea],
                "channelStorage (unit: m) before lake/reservoir outflow",
                True,
                currTimeStep.fulldate,
                threshold=5e-3,
            )

        # lake and reservoir operations
        # From here to the routing call, everything is done once per day.
        if self.debugWaterBalance:
            # (m3)
            preStorage = self.channelStorage

        # move channelStorage to waterBodyStorage at lake and reservoir cells
        # channel storage at water-body cells (m3 per cell): the cell's share of the end
        # of day t-1 lake/reservoir storage, the channel water that reached the cell
        # during the routing of day t-1, minus the riverbed infiltration of day t-1,
        # plus today's local terms added above
        storageAtLakeAndReservoirs = pcr.ifthen(
            pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0, self.channelStorage
        )
        storageAtLakeAndReservoirs = pcr.cover(storageAtLakeAndReservoirs, 0.0)

        # only non-negative, rounded-down values
        # rounded down to whole m3 and only if positive; the remainder (< 1 m3) and any
        # negative storage stay in channelStorage at the body's cells and are routed as
        # river water
        storageAtLakeAndReservoirs = pcr.max(
            0.00, pcr.rounddown(storageAtLakeAndReservoirs)
        )
        # (m3)
        self.channelStorage -= storageAtLakeAndReservoirs

        # update waterBodyStorage (inflow, storage and outflow)
        # INTERFACE: waterBodies.py, WaterBodies.update(newStorageAtLakeAndReservoirs,
        # timestepsToAvgDischarge, maxTimestepsToAvgDischargeShort,
        # maxTimestepsToAvgDischargeLong, currTimeStep, avgChannelDischarge,
        # length_of_time_step, downstreamDemand). Called once per day here
        # (accuTravelTime and simplifiedKinematicWave). Arguments:
        #
        # storageAtLakeAndReservoirs: m3 per cell at water-body cells, 0 elsewhere;
        # WaterBodies sums it per body (pcr.areatotal) and the difference with its
        # previous storage is the day's net inflow residual (WaterBodies.inflow, m3;
        # inflowInM3PerSec, m3/s);
        #
        # self.timestepsToAvgDischarge: days (already incremented today), weight of the
        # running means; self.maxTimestepsToAvgDischargeShort / Long: 30 / 1825 days,
        # windows of avgInflow / avgOutflow; currTimeStep: used only in the
        # balance-check message;
        #
        # self.avgDischarge: m3/s per cell, long-term mean channel discharge up to the
        # previous day; used by WaterBodies only as a fallback of avgOutflow and for the
        # lake weir width;
        #
        # vos.secondsPerDay(): 86400 s, the length of the step (release in m3 per day);
        #
        # self.downstreamDemand: m3/s per cell, environmental-flow discharge (see
        # update);
        #
        # CAUTION (N2): its floor on the release is negligible as the rule is written.
        #
        # The native reservoir rule (WaterBodies.getReservoirOutflow) also has the
        # quirks N1 (the reduction factor is a step at minResvrFrac, not a ramp) and N3
        # (m3 compared with m3/s above the upper limit).
        #
        # EXTENSION POINT: a new release scheme is selected inside WaterBodies
        # (getWaterBodyOutflow); new inputs are passed here as extra arguments, and in
        # the same way at the kinematic_wave_update call. Contract: (i) inputs per water
        # body, constant over its cells (pcr.areatotal / areaaverage over
        # self.WaterBodies.waterBodyIds), covered with 0 outside bodies, in m3 per step
        # or m3/s with the step length length_of_time_step; (ii) the release is written
        # to WaterBodies.waterBodyOutflow in m3 per step, >= 0, <= storage, the same on
        # every cell of the body (CAUTION R1), and subtracted from
        # WaterBodies.waterBodyStorage inside WaterBodies; (iii) WaterBodies applies the
        # 25 percent cap and the round-down to whole m3 after the rule
        # (getWaterBodyOutflow) unless that is changed too; (iv) running means a scheme
        # needs must be kept in WaterBodies and exposed as routing states (getState,
        # getICs, calculate_statistics).
        self.WaterBodies.update(
            storageAtLakeAndReservoirs,
            self.timestepsToAvgDischarge,
            self.maxTimestepsToAvgDischargeShort,
            self.maxTimestepsToAvgDischargeLong,
            currTimeStep,
            self.avgDischarge,
            vos.secondsPerDay(),
            self.downstreamDemand,
        )

        # waterBodyStorage (m3) after outflow, per water body id (not per cell)
        # (INTERFACE: kept for getState and DynQual; reporting.py reads
        # WaterBodies.waterBodyStorage directly)
        self.waterBodyStorage = pcr.ifthen(
            self.landmask, self.WaterBodies.waterBodyStorage
        )

        # DynQual only: storage before outflow (m3 per body), read by qualityWaterBody
        if self.quality:
            self.waterBodyStorageTimeBefore = (
                self.waterBodyStorage + self.WaterBodies.waterBodyOutflow
            )

        # transfer the outflow from lakes and reservoirs to channelStorage (m3/day)
        # release (m3/day) taken at the outlet cell only (WaterBodies.waterBodyOut).
        # CAUTION (R1): WaterBodies computes waterBodyOutflow on every cell of a body
        # and subtracts each cell's own value from that cell's storage, while only the
        # outlet value enters the channel here and return_water_body_storage_to_channel
        # averages the storage over the body's cells. If the release differed between
        # cells of one body, mass would not be conserved. In the native rule a
        # difference can come only from per-cell inputs (downstreamDemand, spatial
        # minResvrFrac / maxResvrFrac maps) or from per-cell values of
        # WaterBodies.avgInflow after a change of a body's extent (R18); with the RAWS
        # cold starts, scalar fractions and unchanged extents they are uniform per body,
        # so the issue is latent. A replacement scheme must return the same release on
        # every cell of a body.
        waterBodyOutflow = pcr.cover(
            pcr.ifthen(
                self.WaterBodies.waterBodyOut, self.WaterBodies.waterBodyOutflow
            ),
            0.0,
        )

        # accuTravelTime: the outlet value is spread equally over the body's cells
        # (pcr.areaaverage of a map that is the release at the outlet and 0 elsewhere:
        # release / number of cells, by count, not by area); the body's characteristic
        # distance (getCharacteristicDistance) then carries it past the outlet within
        # the day. simplifiedKinematicWave: the release stays at the outlet cell only.
        if self.method == "accuTravelTime":
            # distribute the outflow over the lake/reservoir cells, to avoid waterBodyOutflow skipping cells
            waterBodyOutflow = pcr.areaaverage(
                waterBodyOutflow, self.WaterBodies.waterBodyIds
            )
            waterBodyOutflow = pcr.ifthen(
                pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0, waterBodyOutflow
            )
        # (m3/day)
        # (release as placed in the channel; not set by kinematic_wave_update; no reader
        # outside this file)
        self.waterBodyOutflow = pcr.cover(waterBodyOutflow, 0.0)

        # update channelStorage (m3) after waterBodyOutflow (m3)
        # EXTENSION POINT: this is where the release enters the channel. To make it
        # leave the body within the day under simplifiedKinematicWave (R3), it could be
        # added to the cell downstream of the outlet instead, as kinematic_wave_update
        # does (pcr.upstream(self.lddMap, ...)).
        self.channelStorage += self.waterBodyOutflow
        # note: local_input_to_surface_water does not include waterBodyOutflow

        if self.debugWaterBalance:
            vos.waterBalanceCheck(
                [self.waterBodyOutflow / self.cellArea],
                [storageAtLakeAndReservoirs / self.cellArea],
                [preStorage / self.cellArea],
                [self.channelStorage / self.cellArea],
                "channelStorage (unit: m) after lake reservoir/outflow fluxes (errors here are most likely due to pcraster implementation in float_32)",
                True,
                currTimeStep.fulldate,
                threshold=1e-3,
            )

        # DynQual (quality = True): meteo and pollutant inputs, a copy of the storage
        # (channelStorageTimeBefore), qualityLocal and qualityWaterBody. They change
        # water-quality variables only (and the reported waterBodyEvaporation on
        # freezing days, CAUTION R13), not channelStorage or the water-body storage.
        if self.quality:
            # daily input data, unless adjusted in the function
            self.readExtensiveMeteo(currTimeStep)

            # pollutant loadings
            if self.calculateLoads:
                # daily input data for estimating the pollutant loadings, unless adjusted in the function
                self.readPollutantLoadingsInputData(currTimeStep)
                # pollutant loadings calculated (or read) daily, unless adjusted in the function
                self.calculatePollutantLoadings(currTimeStep, landSurface, groundwater)
            else:
                self.readPollutantLoadings(currTimeStep)

            # heat dumps from water temperature dependent power plants (J/day)
            self.PowTwload = pcr.cover(
                landSurface.nonIrrReturnFlowVolumePerSector["thermoelectric"]
                * self.specificHeatWater
                * self.densityWater
                * landSurface.water_demand.water_demand_thermoelectric.min_Tlmax_dTlmax,
                0.0,
            )

            self.channelStorageTimeBefore = pcr.max(0.0, self.channelStorage)
            self.qualityLocal(meteo, landSurface, groundwater, currTimeStep)
            self.qualityWaterBody()

        # routing: returns the new channelStorage (still without waterBodyStorage) and self.Q, the channel
        # discharge (m3/day)
        # accuTravelTime: one daily step. simplifiedKinematicWave: sub-steps of at most
        # maxiumLengthOfSubTimeStep seconds, with the body's cells routed as river cells
        # and the outlet's initial discharge set to the release rate at each sub-step
        # (CAUTION R2, R3). Neither calls WaterBodies.update.
        if self.method == "accuTravelTime":
            self.accuTravelTime()
        if self.method == "simplifiedKinematicWave":
            self.simplifiedKinematicWave(meteo, landSurface, groundwater)
        # channel discharge of the current time step (m3/s)
        # (daily mean, m3/s per cell. INTERFACE: QUAlloc on the next day,
        # landSurface.py)
        self.discharge = self.Q / vos.secondsPerDay()
        # reported channel discharge cannot be negative
        self.discharge = pcr.max(0.0, self.discharge)
        self.discharge = pcr.ifthen(self.landmask, self.discharge)
        # disChanWaterBody (m3/s): at water-body cells the maximum discharge over the
        # body (pcr.areamaximum; master took the outlet value), elsewhere the cell
        # discharge.
        #
        # INTERFACE: reporting.py ("discharge"). It also enters avgDischarge
        # (calculate_statistics), hence downstreamDemand and readAvlChannelStorage at
        # body cells. Under simplifiedKinematicWave the maximum may include upstream
        # water passing through the body (R2).
        self.disChanWaterBody = pcr.ifthen(
            pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
            pcr.areamaximum(self.discharge, self.WaterBodies.waterBodyIds),
        )
        self.disChanWaterBody = pcr.cover(self.disChanWaterBody, self.discharge)
        self.disChanWaterBody = pcr.ifthen(self.landmask, self.disChanWaterBody)
        # reported channel discharge cannot be negative
        self.disChanWaterBody = pcr.max(0.0, self.disChanWaterBody)

        # DynQual only (water temperature, ice, loads averaged over each body)
        if self.quality:
            self.qualityWaterBodyAverage(currTimeStep)

        # return waterBodyStorage to channelStorage
        # spread the lake/reservoir storage (after release) plus the channel water left
        # in the body's cells back over the body's cells by area: until the next
        # WaterBodies.update, channelStorage at a body cell is its share of the water
        # body. Under simplifiedKinematicWave this includes any part of the release that
        # did not leave the outlet cell and upstream water still in the body's cells;
        # both enter the next day's inflow residual (CAUTION R3).
        self.channelStorage = self.return_water_body_storage_to_channel(
            self.channelStorage
        )

        # reported channel storage cannot be negative
        # (creates water where storage was negative; not on master)
        self.channelStorage = pcr.max(0.0, self.channelStorage)

        # DynQual only: concentrations from routed loads and channelStorage (read only)
        if self.quality:
            self.estimate_concentrations()

    def calculate_alpha_and_initial_discharge_for_kinematic_wave(
        self, channelStorage, water_height, innundatedFraction, floodDepth
    ):
        """
        Kinematic-wave alpha and initial discharge (m3/s) of each cell, assuming a
        rectangular channel, from channelStorage (m3), water_height (m), the inundated
        fraction (-) and the flood depth (m). alpha links wetted area and discharge, A =
        alpha * Q ** beta (m2 per (m3/s) ** beta). Used by simplifiedKinematicWave when
        dynamicFloodPlain is False and by kinematic_wave_update. Water bodies: at body
        cells the initial discharge is then overwritten (release rate at the outlet; 0
        at the other body cells) but alpha is kept, so the body's cells route like river
        cells of the same geometry.
        """
        # alpha (-), the roughness coefficient of the kinematic wave (see
        # http://pcraster.geo.uu.nl/pcraster/4.0.0/doc/manual/op_kinematic.html), from the wetted area (m2),
        # wetted perimeter (m) and beta (-), assuming a rectangular channel

        # channel wetted area (m2): at least water height x channel width (Edwin), mainly based on
        # channelStorage and channelLength (Rens's approach)
        channel_wetted_area = water_height * self.channelWidth
        channel_wetted_area = pcr.max(
            channel_wetted_area, channelStorage / self.channelLength
        )

        # wetted perimeter
        flood_only_wetted_perimeter = floodDepth * (2.0) + pcr.max(
            0.0,
            innundatedFraction * self.cellArea / self.channelLength - self.channelWidth,
        )
        channel_only_wetted_perimeter = (
            pcr.min(
                self.channelDepth,
                vos.getValDivZero(
                    channelStorage, self.channelLength * self.channelWidth, 0.0
                ),
            )
            * 2.0
            + self.channelWidth
        )
        # total channel wetted perimeter (m)
        channel_wetted_perimeter = (
            channel_only_wetted_perimeter + flood_only_wetted_perimeter
        )
        # minimum channel wetted perimeter: 10 cm
        channel_wetted_perimeter = pcr.max(0.1, channel_wetted_perimeter)

        # corrected Manning's coefficient
        if self.floodPlain:
            usedManningsN = (
                (channel_only_wetted_perimeter / channel_wetted_perimeter)
                * self.manningsN ** (1.5)
                + (flood_only_wetted_perimeter / channel_wetted_perimeter)
                * self.floodplainManN ** (1.5)
            ) ** (2.0 / 3.0)
        else:
            usedManningsN = self.manningsN

        # alpha (-) and initial estimate of the channel discharge (m3/s)
        alpha = (
            usedManningsN
            * channel_wetted_perimeter ** (2.0 / 3.0)
            * self.gradient ** (-0.5)
        ) ** self.beta
        # (m3)
        # CAUTION: dischargeInitial is a discharge (m3/s), not a volume
        dischargeInitial = pcr.ifthenelse(
            alpha > 0.0, (channel_wetted_area / alpha) ** (1.0 / self.beta), 0.0
        )

        return (alpha, dischargeInitial)

    def returnInundationFractionAndFloodDepth(self, channelStorage):
        """
        Inundated fraction (-) and flood depth (m) of each cell for a given channel
        storage (m3 per cell). Without floodplain: channelFraction and 0. With
        floodplain: the volume above the bankfull capacity (channelStorageCapacity) is
        matched to the flood volumes of the relative-elevation profile (smoothed with a
        logistic function), giving a fraction between channelFraction and 1. Water
        bodies: update calls it with the end-of-day channelStorage, which at body cells
        holds the lake/reservoir storage share, so the fraction there reflects the
        reservoir volume spread on the cell's floodplain profile (CAUTION R8 in update);
        the kinematic loops call it with routed channel water only.
        """

        # flood inundation depth above the floodplain (m)
        floodDepth = 0.0

        # channel and flood inundated fraction (-), at least channelFraction
        inundatedFraction = self.channelFraction

        if self.floodPlain:

            # flooded fraction and associated water height for the given channel volume, using a logistic
            # smoother near intersections (K&K, 2007)

            # flood/excess volume above the bankfull capacity (m3)
            excessVolume = pcr.max(0.0, channelStorage - self.channelStorageCapacity)

            # find the match based on the shortest distance to the available intersections or steps
            deltaXMin = self.floodVolume[self.nrZLevels - 1]
            y_i = pcr.scalar(1.0)
            k = [pcr.scalar(0.0)] * 2
            mInt = pcr.scalar(0.0)
            for iCnt in range(self.nrZLevels - 1, 0, -1):
                # find x_i for the current volume and update the match, slope and intercept if applicable
                deltaX = excessVolume - self.floodVolume[iCnt]
                mask = pcr.abs(deltaX) < pcr.abs(deltaXMin)
                deltaXMin = pcr.ifthenelse(mask, deltaX, deltaXMin)
                y_i = pcr.ifthenelse(mask, self.areaFractions[iCnt], y_i)
                k[0] = pcr.ifthenelse(mask, self.kSlope[iCnt - 1], k[0])
                k[1] = pcr.ifthenelse(mask, self.kSlope[iCnt], k[1])
                mInt = pcr.ifthenelse(mask, self.mInterval[iCnt], mInt)

            # scaled deltaX and smoothed function, based on the integrated logistic functions PHI(x) and 1-PHI(x)
            deltaX = deltaXMin
            deltaXScaled = pcr.ifthenelse(
                deltaX < 0.0, pcr.scalar(-1.0), 1.0
            ) * pcr.min(self.criterionKK, pcr.abs(deltaX / pcr.max(1.0, mInt)))
            logInt = self.integralLogisticFunction(deltaXScaled)

            # fractional inundated area
            inundatedFraction = pcr.ifthenelse(
                excessVolume > 0.0,
                pcr.ifthenelse(
                    pcr.abs(deltaXScaled) < self.criterionKK,
                    y_i - k[0] * mInt * logInt[0] + k[1] * mInt * logInt[1],
                    y_i + pcr.ifthenelse(deltaX < 0.0, k[0], k[1]) * deltaX,
                ),
                0.0,
            )
            # at least channelFraction
            inundatedFraction = pcr.max(self.channelFraction, inundatedFraction)
            # at most 1 (-)
            inundatedFraction = pcr.max(0.0, pcr.min(1.0, inundatedFraction))

            # inundation depth above the floodplain (m), zero if excessVolume == 0
            floodDepth = pcr.ifthenelse(
                inundatedFraction > 0.0,
                excessVolume
                / (
                    pcr.max(self.min_fracwat_for_water_height, inundatedFraction)
                    * self.cellArea
                ),
                0.0,
            )

            # maximum flood depth
            if self.maxFloodDepth is not None:
                floodDepth = pcr.max(0.0, pcr.min(self.maxFloodDepth, floodDepth))

        return inundatedFraction, floodDepth

    def return_water_body_storage_to_channel(self, channelStorage):
        """
        Spread the water-body storage back over the body's cells and return the new
        channelStorage (m3 per cell).

        For each body: total = WaterBodies.waterBodyStorage (m3, the body value, taken
        as the mean over its cells with pcr.areaaverage) + the channel water in its
        cells (pcr.areatotal of channelStorage); each body cell receives total *
        cellArea / (sum of the body's cell areas). Other cells keep channelStorage.

        INTERFACE: reads WaterBodies.waterBodyStorage and waterBodyIds (waterBodies.py).
        Called once per day after routing (simple_update), at every sub-step
        (kinematic_wave_update), and by DynQual (qualityLocal, qualityWaterBodyAverage)
        on copies whose result is not assigned to channelStorage.

        CAUTION (R1): the areaaverage assumes that WaterBodies.waterBodyStorage is the
        same on every cell of a body, which holds only if the release
        (WaterBodies.waterBodyOutflow) is the same on every cell; mass is conserved only
        then, because simple_update injects the outlet value of the release.
        """

        # return waterBodyStorage to channelStorage
        # (m3 per body, on its cells)
        waterBodyStorageTotal = pcr.ifthen(
            pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
            pcr.areaaverage(
                pcr.ifthen(self.landmask, self.WaterBodies.waterBodyStorage),
                pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
            )
            + pcr.areatotal(
                pcr.cover(pcr.ifthen(self.landmask, channelStorage), 0.0),
                pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
            ),
        )
        # (m3 per cell: the body total times the cell's share of the body's cell area)
        waterBodyStoragePerCell = (
            waterBodyStorageTotal
            * self.cellArea
            / pcr.areatotal(
                pcr.cover(self.cellArea, 0.0),
                pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
            )
        )
        # (m3)
        waterBodyStoragePerCell = pcr.ifthen(
            pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0, waterBodyStoragePerCell
        )
        # (m3)
        channelStorage = pcr.ifthen(
            self.landmask, pcr.cover(waterBodyStoragePerCell, channelStorage)
        )
        return channelStorage

    def kinematic_wave_update(self, landSurface, groundwater, currTimeStep, meteo):
        """
        Daily update for routingMethod = kinematicWave (no RAWS configuration uses it).

        Abstraction once per day, then for each sub-step (length dt, s): runoff +
        baseflow and landSurface.nonIrrReturnFlow (m/day) scaled by dt / 86400;
        open-water evaporation; at body cells the channel storage (rounded down) is
        moved into WaterBodies and WaterBodies.update is called with length_of_time_step
        = dt (release in m3 per sub-step); the release is added to the cell downstream
        of the outlet; the body's cells get zero discharge (upstream water is stored, no
        pass-through), and the storage is spread back over the body's cells at the end
        of each sub-step. WaterBodies then updates its running means per sub-step with
        weight dt / 86400, while its flood term uses a daily volume (vos.secondsPerDay
        in WaterBodies.getReservoirOutflow) per sub-step.

        CAUTION (R11): compared with simple_update there is no WWtPlants redirection and
        no upstream_discharge_input_files; the return flow is
        landSurface.nonIrrReturnFlow (m/day) rather than nonIrrReturnFlowVolume;
        self.waterBodyOutflow and self.remainWaterBodyPotEvap are not set;
        WaterBodies.inflowInM3PerSec (reported as lake_and_reservoir_inflow) is that of
        the last sub-step only. See also (R9) and (R10) inline.
        """

        logger.info("Using the fully kinematic wave method! ")

        # update the time steps for the long- and short-term statistics (avgDischarge, avgInflow, avgOutflow, etc.)
        self.timestepsToAvgDischarge += 1.0

        # total local change (input) to surface water storage (m3); only local processes, no routing;
        # initialized at zero
        self.local_input_to_surface_water = pcr.scalar(0.0)

        # for simplicity, surface water abstraction is done outside the sub-daily time steps;
        # update channelStorage (m3) after actSurfaceWaterAbstraction
        # INTERFACE: landSurface.actSurfaceWaterAbstract (m/day), see simple_update; at
        # water-body cells taken from the storage share once, at the start of the day
        self.channelStorage -= landSurface.actSurfaceWaterAbstract * self.cellArea
        self.local_input_to_surface_water -= (
            landSurface.actSurfaceWaterAbstract * self.cellArea
        )
        # channelStorage after surface water abstraction, for reporting (m3)
        self.channelStorageAfterAbstraction = pcr.ifthen(
            self.landmask, self.channelStorage
        )

        # runoff from land surface cells, over the entire cell area (m/day)
        self.runoff = landSurface.landSurfaceRunoff + groundwater.baseflow

        # only route non-negative channelStorage (otherwise it stays)
        channelStorageThatWillNotMove = pcr.ifthenelse(
            self.channelStorage < 0.0, self.channelStorage, 0.0
        )

        # channelStorage passed to the routing (m3)
        channelStorageForRouting = pcr.max(0.0, self.channelStorage)

        # water height (m), needed to estimate the sub-time step length
        self.water_height = channelStorageForRouting / (
            pcr.max(self.min_fracwat_for_water_height, self.dynamicFracWat)
            * self.cellArea
        )

        # sub-time step length (s)
        length_of_sub_time_step, number_of_loops = (
            self.estimate_length_of_sub_time_step()
        )

        for i_loop in range(number_of_loops):

            msg = (
                "sub-daily time step "
                + str(i_loop + 1)
                + " from "
                + str(number_of_loops)
            )
            logger.info(msg)

            if self.debugWaterBalance:
                preStorage = pcr.ifthen(self.landmask, channelStorageForRouting)

            # initialize the accumulated values
            if i_loop == 0:
                # (m3)
                acc_local_input_to_surface_water = pcr.scalar(0.0)
                # (m3)
                acc_water_body_evaporation_volume = pcr.scalar(0.0)
                # (m3)
                acc_discharge_volume = pcr.scalar(0.0)

            # update channelStorageForRouting after runoff and non-irrigation return flow (m3)
            # (INTERFACE: landSurface.nonIrrReturnFlow is in m/day, landSurface.py)
            channelStorageForRouting += (
                (self.runoff + landSurface.nonIrrReturnFlow)
                * self.cellArea
                * length_of_sub_time_step
                / vos.secondsPerDay()
            )
            # (m3)
            acc_local_input_to_surface_water += (
                (self.runoff + landSurface.nonIrrReturnFlow)
                * self.cellArea
                * length_of_sub_time_step
                / vos.secondsPerDay()
            )

            # potential evaporation within the sub-time step, over the entire cell area (m)
            water_body_potential_evaporation = (
                self.calculate_potential_evaporation(landSurface, currTimeStep, meteo)
                * length_of_sub_time_step
                / vos.secondsPerDay()
            )

            # accumulate the potential evaporation
            if i_loop == 0:
                self.waterBodyPotEvap = pcr.scalar(0.0)
            self.waterBodyPotEvap += water_body_potential_evaporation

            # update channelStorageForRouting after evaporation
            # CAUTION (R10): water_body_potential_evaporation is already a depth per
            # sub-step (m/day * dt / 86400); multiplying again by dt / 86400 here makes
            # the evaporation volume about number_of_loops times too small, while
            # self.waterBodyPotEvap accumulates the daily potential correctly. Same on
            # master.
            water_body_evaporation_volume = pcr.min(
                pcr.max(channelStorageForRouting, 0.0),
                water_body_potential_evaporation
                * self.cellArea
                * length_of_sub_time_step
                / vos.secondsPerDay(),
            )
            channelStorageForRouting -= water_body_evaporation_volume
            acc_local_input_to_surface_water -= water_body_evaporation_volume
            acc_water_body_evaporation_volume += water_body_evaporation_volume

            if self.debugWaterBalance:
                vos.waterBalanceCheck(
                    [
                        self.runoff * length_of_sub_time_step / vos.secondsPerDay(),
                        landSurface.nonIrrReturnFlow
                        * length_of_sub_time_step
                        / vos.secondsPerDay(),
                    ],
                    [water_body_evaporation_volume / self.cellArea],
                    [preStorage / self.cellArea],
                    [channelStorageForRouting / self.cellArea],
                    "channelStorageForRouting (local fluxes)",
                    True,
                    currTimeStep.fulldate,
                    threshold=5e-5,
                )

            # move channelStorageForRouting (m3) to waterBodyStorage at lake and reservoir cells
            # (at every sub-step; the body's storage was spread back over its cells at
            # the end of the previous sub-step)
            storageAtLakeAndReservoirs = pcr.ifthen(
                pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                channelStorageForRouting,
            )
            storageAtLakeAndReservoirs = pcr.cover(storageAtLakeAndReservoirs, 0.0)

            # only non-negative, rounded-down values
            storageAtLakeAndReservoirs = pcr.max(
                0.00, pcr.rounddown(storageAtLakeAndReservoirs)
            )
            channelStorageForRouting = pcr.max(
                0.0, channelStorageForRouting - storageAtLakeAndReservoirs
            )

            # update waterBodyStorage (inflow, storage and outflow)
            # INTERFACE: waterBodies.py, WaterBodies.update, here once per sub-step with
            # length_of_time_step = length_of_sub_time_step (s); the same arguments as
            # in simple_update, with the release in m3 per sub-step.
            #
            # EXTENSION POINT: extra inputs of a new scheme must be passed here too,
            # scaled to the sub-step (contract as in simple_update).
            self.WaterBodies.update(
                storageAtLakeAndReservoirs,
                self.timestepsToAvgDischarge,
                self.maxTimestepsToAvgDischargeShort,
                self.maxTimestepsToAvgDischargeLong,
                currTimeStep,
                self.avgDischarge,
                length_of_sub_time_step,
                self.downstreamDemand,
            )

            # waterBodyOutflow (m3 per sub-time step) at lake/reservoir outlet cells
            # (release at the outlet cell, m3 per sub-step; 0 at the other cells)
            waterBodyOutflow = pcr.cover(
                pcr.ifthen(
                    self.WaterBodies.waterBodyOut, self.WaterBodies.waterBodyOutflow
                ),
                0.0,
            )
            waterBodyOutflow = pcr.ifthen(self.landmask, waterBodyOutflow)

            # (m3/s)
            waterBodyOutflowInM3PerSec = waterBodyOutflow / length_of_sub_time_step

            # waterBodyStorage (m3) after outflow, per water body id (not per cell)
            self.waterBodyStorage = pcr.ifthen(
                self.landmask, self.WaterBodies.waterBodyStorage
            )

            # update channelStorage (m3) after waterBodyOutflow (m3); local_input_to_surface_water does not
            # include waterBodyOutflow
            # the release is added to the cell downstream of the outlet (pcr.upstream
            # sums the values of each cell's upstream neighbours)
            storage_change_in_volume = pcr.upstream(self.lddMap, waterBodyOutflow)
            channelStorageForRouting += storage_change_in_volume

            # water height (m) in channels (excluding lakes and reservoirs), needed to estimate the channel
            # wetted area (for alpha and dischargeInitial)
            self.water_height = channelStorageForRouting / (
                pcr.max(self.min_fracwat_for_water_height, self.dynamicFracWat)
                * self.cellArea
            )
            # deactivating this line gives negative channelStorage (probably due to too high water heights in lakes and reservoirs)

            # alpha parameter and initial discharge estimate for the kinematic wave
            alpha, dischargeInitial = (
                self.calculate_alpha_and_initial_discharge_for_kinematic_wave(
                    channelStorageForRouting,
                    self.water_height,
                    self.innundatedFraction,
                    self.floodDepth,
                )
            )

            # at lake and reservoir outlet cells, the initial discharge is waterBodyOutflowInM3PerSec
            # (the discharge of all body cells is zeroed after pcr.kinematic below, so
            # this value does not move water)
            dischargeInitial = pcr.cover(
                pcr.ifthen(self.WaterBodies.waterBodyOut, waterBodyOutflowInM3PerSec),
                dischargeInitial,
            )

            # also for cells with zero channelStorageForRouting
            dischargeInitial = pcr.ifthenelse(
                channelStorageForRouting > 0.0, dischargeInitial, 0.0
            )

            # (m3/s)
            dischargeInitial = pcr.cover(dischargeInitial, 0.0)
            dischargeInitial = pcr.ifthen(self.landmask, dischargeInitial)

            # discharge (m3/s) from the kinematic wave approximation
            self.subDischarge = pcr.kinematic(
                self.lddMap,
                dischargeInitial,
                0.0,
                alpha,
                self.beta,
                1,
                length_of_sub_time_step,
                self.channelLength,
            )
            self.subDischarge = pcr.max(0.0, pcr.cover(self.subDischarge, 0.0))

            # zero discharge at lake and reservoir cells
            # (upstream water stays in the body's cells and is merged into the water
            # body by return_water_body_storage_to_channel below: no pass-through under
            # kinematicWave)
            self.subDischarge = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0, pcr.scalar(0.0)
                ),
                self.subDischarge,
            )

            # avoid negative channel storage
            self.subDischarge = (
                pcr.min(
                    self.subDischarge * length_of_sub_time_step,
                    pcr.max(
                        0.0,
                        channelStorageForRouting
                        + pcr.upstream(
                            self.lddMap, self.subDischarge * length_of_sub_time_step
                        ),
                    ),
                )
                / length_of_sub_time_step
            )

            # update channelStorage (m3) after lateral flows in channels
            storage_change_in_volume = (
                pcr.upstream(self.lddMap, self.subDischarge * length_of_sub_time_step)
                - self.subDischarge * length_of_sub_time_step
            )
            channelStorageForRouting += storage_change_in_volume

            # return waterBodyStorage to channelStorage
            # (every sub-step; extracted again at the start of the next sub-step)
            channelStorageForRouting = self.return_water_body_storage_to_channel(
                channelStorageForRouting
            )

            # add waterBodyOutflowInM3PerSec to subDischarge
            # (the outlet's reported discharge is the release rate)
            self.subDischarge += waterBodyOutflowInM3PerSec
            self.subDischarge = pcr.ifthen(self.landmask, self.subDischarge)

            # total discharge volume (m3) until this loop
            acc_discharge_volume += self.subDischarge * length_of_sub_time_step

            # update the flood fraction and flood depth
            # CAUTION (R9): stored as inundatedFraction (one n) while dynamicFracWat
            # below reads innundatedFraction (two n), the start-of-day value; same on
            # master
            self.inundatedFraction, self.floodDepth = (
                self.returnInundationFractionAndFloodDepth(channelStorageForRouting)
            )

            # dynamicFracWat: fraction of surface water bodies (-), including lakes and reservoirs; lake and
            # reservoir surface water fraction
            self.dynamicFracWat = pcr.cover(pcr.min(1.0, self.WaterBodies.fracWat), 0.0)

            # fraction of channel (including its excess above bankfull capacity)
            self.dynamicFracWat += pcr.max(0.0, 1.0 - self.dynamicFracWat) * pcr.max(
                self.channelFraction, self.innundatedFraction
            )

            # dynamicFracWat is at most 1
            self.dynamicFracWat = pcr.ifthen(
                self.landmask,
                pcr.min(pcr.max(self.dynamicFracWat, self.WaterBodies.fracWat), 1.0),
            )
            self.dynamicFracWat = pcr.ifthen(
                self.landmask, pcr.min(1.0, self.dynamicFracWat)
            )

            # for the next loop, only route non-negative channelStorage
            channelStorageThatWillNotMove += pcr.ifthenelse(
                channelStorageForRouting < 0.0, channelStorageForRouting, 0.0
            )
            channelStorageForRouting = pcr.max(0.000, channelStorageForRouting)

            # water height, including lakes and reservoirs
            self.water_height = pcr.max(0.0, channelStorageForRouting) / (
                pcr.max(self.min_fracwat_for_water_height, self.dynamicFracWat)
                * self.cellArea
            )

        # evaporation (m/day)
        # (m/day over the cell; see CAUTION R10)
        self.waterBodyEvaporation = acc_water_body_evaporation_volume / self.cellArea

        # local input to surface water (m3)
        self.local_input_to_surface_water += acc_local_input_to_surface_water

        # channel discharge (m3/day)
        self.Q = acc_discharge_volume

        # update channelStorage after routing
        self.channelStorage = channelStorageForRouting

        # return channelStorageThatWillNotMove to channelStorage
        self.channelStorage += channelStorageThatWillNotMove

        # reported channel storage cannot be negative
        self.channelStorage = pcr.max(0.0, self.channelStorage)

        # channel discharge of the current time step (m3/s)
        self.discharge = self.Q / vos.secondsPerDay()
        # reported channel discharge cannot be negative
        self.discharge = pcr.max(0.0, self.discharge)
        self.discharge = pcr.ifthen(self.landmask, self.discharge)

        # (as in simple_update: body-maximum discharge at water-body cells)
        self.disChanWaterBody = pcr.ifthen(
            pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
            pcr.areamaximum(self.discharge, self.WaterBodies.waterBodyIds),
        )
        self.disChanWaterBody = pcr.cover(self.disChanWaterBody, self.discharge)
        self.disChanWaterBody = pcr.ifthen(self.landmask, self.disChanWaterBody)

        # reported channel discharge cannot be negative
        self.disChanWaterBody = pcr.max(0.0, self.disChanWaterBody)

    def calculate_statistics(self, groundwater, landSurface):
        """
        Update the running means used by water bodies and water management, after
        routing.

        Copies WaterBodies.avgInflow (m3/s, 30-day window) and WaterBodies.avgOutflow
        (m3/s, 1825-day window) into self.avgInflow and self.avgOutflow for getState;
        WaterBodies keeps the values it actually uses. Updates avgDischarge (m3/s),
        m2tDischarge ((m3/s)^2), avgDischargeShort (m3/s) and avgBaseflow (m3/s) with
        the incremental mean x_new = x + (q - x) / min(window, timestepsToAvgDischarge).
        The discharge used is max(discharge, disChanWaterBody), i.e. the body-maximum
        discharge at water-body cells. These feed downstreamDemand and
        readAvlChannelStorage on the next day; avgBaseflow feeds the surface-water /
        groundwater partitioning (water_management/main_water_management.py).
        """

        # short-term average inflow (m3/s) and long-term average outflow (m3/s) of lakes and reservoirs
        # (INTERFACE: copies of the WaterBodies running means, for getState only; per
        # body, also defined outside water bodies)
        self.avgInflow = pcr.ifthen(
            self.landmask, pcr.cover(self.WaterBodies.avgInflow, 0.0)
        )
        self.avgOutflow = pcr.ifthen(
            self.landmask, pcr.cover(self.WaterBodies.avgOutflow, 0.0)
        )

        # short- and long-term average discharge (m3/s), using the online algorithm of
        # http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

        # long-term average discharge
        # (at water-body cells the body-maximum discharge, disChanWaterBody)
        dishargeUsed = pcr.max(0.0, self.discharge)
        dishargeUsed = pcr.max(dishargeUsed, self.disChanWaterBody)

        deltaAnoDischarge = dishargeUsed - self.avgDischarge
        self.avgDischarge = self.avgDischarge + deltaAnoDischarge / pcr.min(
            self.maxTimestepsToAvgDischargeLong, self.timestepsToAvgDischarge
        )
        self.avgDischarge = pcr.max(0.0, self.avgDischarge)
        # CAUTION (R4): m2tDischarge ((m3/s)^2) keeps accumulating squared anomalies
        # after the window is full, while its divisor in
        # estimate_discharge_for_environmental_flow stops at 1824, so the variance
        # estimate grows roughly linearly with the run length beyond 1825 days (by the
        # arithmetic of the code, not run). The environmental flow then tends to 0.1 *
        # avgDischarge, which changes downstreamDemand and readAvlChannelStorage over a
        # long run. Same on master.
        self.m2tDischarge = self.m2tDischarge + pcr.abs(
            deltaAnoDischarge * (dishargeUsed - self.avgDischarge)
        )

        # short-term average discharge
        deltaAnoDischargeShort = dishargeUsed - self.avgDischargeShort
        self.avgDischargeShort = (
            self.avgDischargeShort
            + deltaAnoDischargeShort
            / pcr.min(
                self.maxTimestepsToAvgDischargeShort, self.timestepsToAvgDischarge
            )
        )
        self.avgDischargeShort = pcr.max(0.0, self.avgDischargeShort)

        # long-term average baseflow (m3/s), used as proxy to partition groundwater and surface water abstractions
        baseflowM3PerSec = groundwater.baseflow * self.cellArea / vos.secondsPerDay()

        deltaAnoBaseflow = baseflowM3PerSec - self.avgBaseflow
        self.avgBaseflow = self.avgBaseflow + deltaAnoBaseflow / pcr.min(
            self.maxTimestepsToAvgDischargeLong, self.timestepsToAvgDischarge
        )
        self.avgBaseflow = pcr.max(0.0, self.avgBaseflow)

        # DynQual: average irrigation water allocated over the last 30 days (needed for online runs where
        # loadings are calculated in the loop)
        if self.quality:
            if self.calculateLoads and not self.offlineRun:
                # average irrigation gross demand over the last 30 days
                irrGrossDemand = deepcopy(landSurface.irrGrossDemand)
                deltaAno_irrGrossDemand = (
                    pcr.max(0.0, irrGrossDemand) - self.avg_irrGrossDemand
                )
                self.avg_irrGrossDemand = (
                    self.avg_irrGrossDemand
                    + deltaAno_irrGrossDemand
                    / pcr.min(30.0, self.timestepsToAvgDischarge)
                )
                self.avg_irrGrossDemand = pcr.max(0.0, self.avg_irrGrossDemand)

                # average netLqWaterToSoil over the last 30 days
                deltaAno_netLqWaterToSoil = (
                    pcr.max(0.0, landSurface.netLqWaterToSoil)
                    - self.avg_netLqWaterToSoil
                )
                self.avg_netLqWaterToSoil = (
                    self.avg_netLqWaterToSoil
                    + deltaAno_netLqWaterToSoil
                    / pcr.min(30.0, self.timestepsToAvgDischarge)
                )
                self.avg_netLqWaterToSoil = pcr.max(0.0, self.avg_netLqWaterToSoil)

    def estimate_discharge_for_environmental_flow(self, channelStorage):
        """
        Environmental-flow discharge (m3/s per cell): max(0.1 * avgDischarge,
        avgDischarge - 1.2816 * std), std = sqrt(m2tDischarge / max(1, min(1825,
        timestepsToAvgDischarge) - 1)). 1.2816 is the z-score of the 90th percentile, so
        avgDischarge - 1.28 std approximates the 10th percentile of a normal
        distribution. The argument channelStorage is not used. Used for downstreamDemand
        (update; INTERFACE: WaterBodies.update) and in
        estimate_available_volume_for_abstraction. At water-body cells the statistics
        are those of the body-maximum discharge.

        CAUTION (R4): see calculate_statistics.
        """

        # statistical assumption: z-score of the 90th percentile
        z_score = 1.2816

        # long-term variance and standard deviation of discharge, using the online algorithm of
        # http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        # ((m3/s)^2; CAUTION R4: the divisor stops growing at 1824, the numerator does
        # not)
        varDischarge = self.m2tDischarge / pcr.max(
            1.0,
            pcr.min(self.maxTimestepsToAvgDischargeLong, self.timestepsToAvgDischarge)
            - 1.0,
        )
        stdDischarge = pcr.max(varDischarge**0.5, 0.0)

        # minimum discharge for environmental flow (m3/s)
        minDischargeForEnvironmentalFlow = pcr.max(
            0.0, self.avgDischarge - z_score * stdDischarge
        )
        # avoid flip-flopping
        factor = 0.10
        # (m3/s)
        minDischargeForEnvironmentalFlow = pcr.max(
            factor * self.avgDischarge, minDischargeForEnvironmentalFlow
        )
        minDischargeForEnvironmentalFlow = pcr.max(
            0.0, minDischargeForEnvironmentalFlow
        )

        return minDischargeForEnvironmentalFlow

    def estimate_available_volume_for_abstraction(
        self, channelStorage, length_of_time_step=vos.secondsPerDay()
    ):
        """
        Volume of surface water that may be abstracted on the next day (m3 per cell),
        from channelStorage S (m3 per cell, end of today).

        A = S * min(1, min(avgDischargeShort, avgDischarge) / Q_env); where
        avgDischargeShort >= Q_env, A = max(A, (avgDischargeShort - Q_env) *
        length_of_time_step); then A <= 0.9 * S, A >= 0, rounded down to 0.1 m3. Since
        Q_env <= avgDischarge, this is 0.9 * S whenever avgDischargeShort >= Q_env, and
        S * avgDischargeShort / Q_env (at most 0.9 * S) otherwise. Water bodies: at body
        cells S is the cell's share of the lake/reservoir storage (plus channel water
        left after routing, minus riverbed infiltration), and the discharge statistics
        are those of the body-maximum discharge, so the same factor applies to all cells
        of a body. Nothing here knows waterBodyCap or minResvrFrac (K2).

        INTERFACE: stored as self.readAvlChannelStorage in update; read by
        water_management/main_water_management.py.
        """
        # input: channelStorage (m3)

        # minimum discharge for environmental flow (m3/s)
        minDischargeForEnvironmentalFlow = (
            self.estimate_discharge_for_environmental_flow(channelStorage)
        )

        # channelStorage available for surface water abstraction
        readAvlChannelStorage = pcr.max(0.0, channelStorage)

        # reduce readAvlChannelStorage if the average discharge < minDischargeForEnvironmentalFlow
        # (factor 1 when the short- and long-term discharge are at least Q_env; 0 if
        # Q_env is 0)
        readAvlChannelStorage *= pcr.min(
            1.0,
            vos.getValDivZero(
                pcr.max(0.0, pcr.min(self.avgDischargeShort, self.avgDischarge)),
                minDischargeForEnvironmentalFlow,
                vos.smallNumber,
            ),
        )

        # maintain the environmental flow if the average discharge > minDischargeForEnvironmentalFlow; TODO: check why this is needed
        # CAUTION: this branch can only increase the available volume (the surplus of
        # avgDischargeShort over the environmental flow, as a volume per time step); it
        # does not reserve the environmental flow. Where it applies, the factor above is
        # already 1.
        readAvlChannelStorage = pcr.ifthenelse(
            self.avgDischargeShort < minDischargeForEnvironmentalFlow,
            readAvlChannelStorage,
            pcr.max(
                readAvlChannelStorage,
                pcr.max(0.0, self.avgDischargeShort - minDischargeForEnvironmentalFlow)
                * length_of_time_step,
            ),
        )

        # maximum fraction of water that can be abstracted from the channel (to avoid flip-flopping)
        # (at water-body cells: at most 90 percent of the cell's lake/reservoir storage
        # share per day, whatever the reservoir's fill level)
        maximum_percentage = 0.90
        readAvlChannelStorage = pcr.min(
            readAvlChannelStorage, maximum_percentage * channelStorage
        )
        readAvlChannelStorage = pcr.max(0.0, readAvlChannelStorage)

        # ignore small volumes (less than 0.1 m3)
        readAvlChannelStorage = pcr.rounddown(readAvlChannelStorage * 10.0) / 10.0
        readAvlChannelStorage = pcr.ifthen(self.landmask, readAvlChannelStorage)

        # (m3)
        return readAvlChannelStorage

    def initiate_old_style_routing_reporting(self, iniItems):
        """
        Old-style netCDF reporting of routing attributes by name, configured in
        routingOptions (outDailyTotNC, outMonthTotNC, outMonthAvgNC, outMonthEndNC,
        outAnnuaTotNC, outAnnuaAvgNC, outAnnuaEndNC). All seven keys must be present,
        otherwise self.report is False and nothing is written; no RAWS configuration
        sets them in routingOptions (their outDailyTotNC is in reportingOptions and is
        handled by reporting.py).

        EXTENSION POINT: for debugging a release scheme, any Routing attribute that is a
        PCRaster map at the end of update can be written without a variable_list.py
        entry, e.g. waterBodyStorage (m3 per body), waterBodyOutflow (m3/day as placed
        in the channel; simple_update only), downstreamDemand (m3/s),
        readAvlChannelStorage (m3), disChanWaterBody (m3/s); the unit is written as
        "undefined". WaterBodies attributes are not reachable by name here.
        """

        self.report = True
        try:
            self.outDailyTotNC = iniItems.routingOptions["outDailyTotNC"].split(",")
            self.outMonthTotNC = iniItems.routingOptions["outMonthTotNC"].split(",")
            self.outMonthAvgNC = iniItems.routingOptions["outMonthAvgNC"].split(",")
            self.outMonthEndNC = iniItems.routingOptions["outMonthEndNC"].split(",")
            self.outAnnuaTotNC = iniItems.routingOptions["outAnnuaTotNC"].split(",")
            self.outAnnuaAvgNC = iniItems.routingOptions["outAnnuaAvgNC"].split(",")
            self.outAnnuaEndNC = iniItems.routingOptions["outAnnuaEndNC"].split(",")
        except Exception:
            self.report = False
        if self.report:
            # daily netCDF output
            self.outNCDir = iniItems.outNCDir
            self.netcdfObj = PCR2netCDF(iniItems)
            if self.outDailyTotNC[0] != "None":
                for var in self.outDailyTotNC:
                    self.netcdfObj.createNetCDF(
                        str(self.outNCDir) + "/" + str(var) + "_dailyTot.nc",
                        var,
                        "undefined",
                    )
            # monthly netCDF output: totals
            if self.outMonthTotNC[0] != "None":
                for var in self.outMonthTotNC:
                    # accumulator
                    vars(self)[var + "MonthTot"] = None
                    self.netcdfObj.createNetCDF(
                        str(self.outNCDir) + "/" + str(var) + "_monthTot.nc",
                        var,
                        "undefined",
                    )
            # averages
            if self.outMonthAvgNC[0] != "None":
                for var in self.outMonthAvgNC:
                    # accumulator
                    vars(self)[var + "MonthTot"] = None
                    vars(self)[var + "MonthAvg"] = None
                    self.netcdfObj.createNetCDF(
                        str(self.outNCDir) + "/" + str(var) + "_monthAvg.nc",
                        var,
                        "undefined",
                    )
            # end of month
            if self.outMonthEndNC[0] != "None":
                for var in self.outMonthEndNC:
                    self.netcdfObj.createNetCDF(
                        str(self.outNCDir) + "/" + str(var) + "_monthEnd.nc",
                        var,
                        "undefined",
                    )
            # yearly netCDF output: totals
            if self.outAnnuaTotNC[0] != "None":
                for var in self.outAnnuaTotNC:
                    # accumulator
                    vars(self)[var + "AnnuaTot"] = None
                    self.netcdfObj.createNetCDF(
                        str(self.outNCDir) + "/" + str(var) + "_annuaTot.nc",
                        var,
                        "undefined",
                    )
            # averages
            if self.outAnnuaAvgNC[0] != "None":
                for var in self.outAnnuaAvgNC:
                    vars(self)[var + "AnnuaAvg"] = None
                    # accumulator
                    vars(self)[var + "AnnuaTot"] = None
                    self.netcdfObj.createNetCDF(
                        str(self.outNCDir) + "/" + str(var) + "_annuaAvg.nc",
                        var,
                        "undefined",
                    )
            # end of year
            if self.outAnnuaEndNC[0] != "None":
                for var in self.outAnnuaEndNC:
                    self.netcdfObj.createNetCDF(
                        str(self.outNCDir) + "/" + str(var) + "_annuaEnd.nc",
                        var,
                        "undefined",
                    )

    def old_style_routing_reporting(self, currTimeStep):
        """
        Write the old-style routing outputs (see initiate_old_style_routing_reporting);
        called at the end of update. Daily values are those at the end of the day;
        monthly and yearly totals and averages sum the daily values (meaningful for
        fluxes, not for storages).
        """

        if self.report:
            timeStamp = datetime.datetime(
                currTimeStep.year, currTimeStep.month, currTimeStep.day, 0
            )
            # daily netCDF output
            timestepPCR = currTimeStep.timeStepPCR
            if self.outDailyTotNC[0] != "None":
                for var in self.outDailyTotNC:
                    self.netcdfObj.data2NetCDF(
                        str(self.outNCDir) + "/" + str(var) + "_dailyTot.nc",
                        var,
                        pcr.pcr2numpy(self.__getattribute__(var), vos.MV),
                        timeStamp,
                        timestepPCR - 1,
                    )

            # monthly netCDF output: totals
            if self.outMonthTotNC[0] != "None":
                for var in self.outMonthTotNC:

                    # initialize at the start of the simulation or reset at the start of the month
                    if currTimeStep.timeStepPCR == 1 or currTimeStep.day == 1:
                        vars(self)[var + "MonthTot"] = pcr.scalar(0.0)

                    vars(self)[var + "MonthTot"] += vars(self)[var]

                    if currTimeStep.endMonth:
                        self.netcdfObj.data2NetCDF(
                            str(self.outNCDir) + "/" + str(var) + "_monthTot.nc",
                            var,
                            pcr.pcr2numpy(
                                self.__getattribute__(var + "MonthTot"), vos.MV
                            ),
                            timeStamp,
                            currTimeStep.monthIdx - 1,
                        )
            # averages
            if self.outMonthAvgNC[0] != "None":
                for var in self.outMonthAvgNC:
                    # only if no accumulator is defined
                    if var not in self.outMonthTotNC:

                        # initialize at the start of the simulation or reset at the start of the month
                        if currTimeStep.timeStepPCR == 1 or currTimeStep.day == 1:
                            vars(self)[var + "MonthTot"] = pcr.scalar(0.0)
                        vars(self)[var + "MonthTot"] += vars(self)[var]

                    if currTimeStep.endMonth:
                        vars(self)[var + "MonthAvg"] = (
                            vars(self)[var + "MonthTot"] / currTimeStep.day
                        )
                        self.netcdfObj.data2NetCDF(
                            str(self.outNCDir) + "/" + str(var) + "_monthAvg.nc",
                            var,
                            pcr.pcr2numpy(
                                self.__getattribute__(var + "MonthAvg"), vos.MV
                            ),
                            timeStamp,
                            currTimeStep.monthIdx - 1,
                        )
            # end of month
            if self.outMonthEndNC[0] != "None":
                for var in self.outMonthEndNC:
                    if currTimeStep.endMonth:
                        self.netcdfObj.data2NetCDF(
                            str(self.outNCDir) + "/" + str(var) + "_monthEnd.nc",
                            var,
                            pcr.pcr2numpy(self.__getattribute__(var), vos.MV),
                            timeStamp,
                            currTimeStep.monthIdx - 1,
                        )

            # yearly netCDF output: totals
            if self.outAnnuaTotNC[0] != "None":
                for var in self.outAnnuaTotNC:

                    # initialize at the start of the simulation or reset at the start of the year
                    if currTimeStep.timeStepPCR == 1 or currTimeStep.doy == 1:
                        vars(self)[var + "AnnuaTot"] = pcr.scalar(0.0)

                    vars(self)[var + "AnnuaTot"] += vars(self)[var]

                    if currTimeStep.endYear:
                        self.netcdfObj.data2NetCDF(
                            str(self.outNCDir) + "/" + str(var) + "_annuaTot.nc",
                            var,
                            pcr.pcr2numpy(
                                self.__getattribute__(var + "AnnuaTot"), vos.MV
                            ),
                            timeStamp,
                            currTimeStep.annuaIdx - 1,
                        )
            # averages
            if self.outAnnuaAvgNC[0] != "None":
                for var in self.outAnnuaAvgNC:
                    # only if no accumulator is defined
                    if var not in self.outAnnuaTotNC:
                        # initialize at the start of the simulation or reset at the start of the year
                        if currTimeStep.timeStepPCR == 1 or currTimeStep.doy == 1:
                            vars(self)[var + "AnnuaTot"] = pcr.scalar(0.0)
                        vars(self)[var + "AnnuaTot"] += vars(self)[var]
                    if currTimeStep.endYear:
                        vars(self)[var + "AnnuaAvg"] = (
                            vars(self)[var + "AnnuaTot"] / currTimeStep.doy
                        )
                        self.netcdfObj.data2NetCDF(
                            str(self.outNCDir) + "/" + str(var) + "_annuaAvg.nc",
                            var,
                            pcr.pcr2numpy(
                                self.__getattribute__(var + "AnnuaAvg"), vos.MV
                            ),
                            timeStamp,
                            currTimeStep.annuaIdx - 1,
                        )
            # end of year
            if self.outAnnuaEndNC[0] != "None":
                for var in self.outAnnuaEndNC:
                    if currTimeStep.endYear:
                        self.netcdfObj.data2NetCDF(
                            str(self.outNCDir) + "/" + str(var) + "_annuaEnd.nc",
                            var,
                            pcr.pcr2numpy(self.__getattribute__(var), vos.MV),
                            timeStamp,
                            currTimeStep.annuaIdx - 1,
                        )

    def returnFloodedFraction(self, channelStorage):
        """
        Flooded fraction (-) and flood depth (m) for a given flood volume (m3 per cell),
        from the relative-elevation profile. Used by kinAlpha (flood volume above
        bankfull) and in update at the first time step (with the whole channelStorage,
        R17).
        """
        # flooded fraction and associated water height for the given flood volume, using a logistic smoother
        # near intersections (K&K, 2007); find the match based on the shortest distance to the available
        # intersections or steps
        deltaXMin = self.floodVolume[self.nrZLevels - 1]
        y_i = pcr.scalar(1.0)
        k = [pcr.scalar(0.0)] * 2
        mInt = pcr.scalar(0.0)
        for iCnt in range(self.nrZLevels - 1, 0, -1):
            # find x_i for the current volume and update the match, slope and intercept if applicable
            deltaX = channelStorage - self.floodVolume[iCnt]
            mask = pcr.abs(deltaX) < pcr.abs(deltaXMin)
            deltaXMin = pcr.ifthenelse(mask, deltaX, deltaXMin)
            y_i = pcr.ifthenelse(mask, self.areaFractions[iCnt], y_i)
            k[0] = pcr.ifthenelse(mask, self.kSlope[iCnt - 1], k[0])
            k[1] = pcr.ifthenelse(mask, self.kSlope[iCnt], k[1])
            mInt = pcr.ifthenelse(mask, self.mInterval[iCnt], mInt)

        # scaled deltaX and smoothed function, based on the integrated logistic functions PHI(x) and 1-PHI(x)
        deltaX = deltaXMin
        deltaXScaled = pcr.ifthenelse(deltaX < 0.0, pcr.scalar(-1.0), 1.0) * pcr.min(
            self.criterionKK, pcr.abs(deltaX / pcr.max(1.0, mInt))
        )
        logInt = self.integralLogisticFunction(deltaXScaled)

        # fractional flooded area and flood depth
        floodedFraction = pcr.ifthenelse(
            channelStorage > 0.0,
            pcr.ifthenelse(
                pcr.abs(deltaXScaled) < self.criterionKK,
                y_i - k[0] * mInt * logInt[0] + k[1] * mInt * logInt[1],
                y_i + pcr.ifthenelse(deltaX < 0.0, k[0], k[1]) * deltaX,
            ),
            0.0,
        )
        floodedFraction = pcr.max(0.0, pcr.min(1.0, floodedFraction))
        floodDepth = pcr.ifthenelse(
            floodedFraction > 0.0,
            channelStorage / (floodedFraction * self.cellArea),
            0.0,
        )
        floodDepth = pcr.min(self.max_water_height, floodDepth)

        return floodedFraction, floodDepth

    def integralLogisticFunction(self, x):
        # tuple with the integrals of the logistic functions of x and -x
        logInt = pcr.ln(pcr.exp(-x) + 1)
        return logInt, x + logInt

    def kinAlpha(self, channelStorage):
        """
        Floodplain-aware kinematic-wave parameters, used by simplifiedKinematicWave when
        dynamicFloodPlain is True (all RAWS configurations).

        Returns (flooded fraction including the channel fraction (-), flood depth above
        the floodplain (m), alpha, initial discharge (m3/s)); simplifiedKinematicWave
        stores the first two as self.dynamicFracWat and self.water_height. Side effect:
        overwrites self.channelFraction (computed with cellLengthFD). The static branch
        (ice-dependent Manning with quality = True) is computed but discarded, since
        mask is always True, so DynQual does not change alpha. Water bodies: at body
        cells channelStorage is the routed channel water only (the body's storage is
        inside WaterBodies), so they route like river cells; their initial discharge is
        then overwritten by simplifiedKinematicWave.
        """
        mask = pcr.boolean(1)

        # kinAlphaDynamic logic
        floodVol = pcr.max(0, channelStorage - self.channelStorageCapacity)
        floodedFraction, floodDepth = self.returnFloodedFraction(floodVol)

        self.channelFraction = pcr.max(
            0.0, pcr.min(1.0, self.wMean * self.cellLengthFD / self.cellArea)
        )
        floodFrac_dyn = floodedFraction + self.channelFraction

        wetA_dyn = channelStorage / self.channelLength
        wetPFld = (
            pcr.max(
                0.0, floodFrac_dyn * self.cellArea / self.channelLength - self.wMean
            )
            + 2.0 * floodDepth
        )
        wetPCh = self.wMean + 2.0 * pcr.min(
            self.channelDepth, channelStorage / (self.channelLength * self.wMean)
        )
        wetP_dyn = wetPFld + wetPCh
        manQ = (
            wetPCh / wetP_dyn * self.manningsN**1.5
            + wetPFld / wetP_dyn * self.floodplainManN**1.5
        ) ** (2.0 / 3.0)
        dynamicAlphaQ = (
            manQ * wetP_dyn ** (2.0 / 3.0) * self.gradient**-0.5
        ) ** self.beta
        dynamicDischargeInitial = pcr.ifthenelse(
            dynamicAlphaQ > 0.0, (wetA_dyn / dynamicAlphaQ) ** (1 / self.beta), 0.0
        )

        # kinAlphaStatic logic
        if self.quality:
            manIce = pcr.max(
                self.manningsN,
                0.0493
                * pcr.max(
                    0.01, self.channelStorage / (self.dynamicFracWat * self.cellArea)
                )
                ** (-0.23)
                * self.iceThickness**0.57,
            )
            manningsWithIce = (0.5 * (self.manningsN**1.5 + manIce**1.5)) ** (2.0 / 3.0)
            wetA_sta = self.channelStorage / self.channelLength
            wetP_sta = 2.0 * wetA_sta / self.wMean + self.wMean
            staticAlphaQ = (
                manningsWithIce * wetP_sta ** (2.0 / 3.0) * self.gradient**-0.5
            ) ** self.beta
        else:
            wetA_sta = channelStorage / self.channelLength
            wetP_sta = 2.0 * wetA_sta / self.wMean + self.wMean
            staticAlphaQ = (
                self.manningsN * wetP_sta ** (2.0 / 3.0) * self.gradient**-0.5
            ) ** self.beta

        staticDischargeInitial = pcr.ifthenelse(
            staticAlphaQ > 0.0, (wetA_sta / staticAlphaQ) ** (1 / self.beta), 0.0
        )

        # combine using the mask
        floodedFraction = pcr.ifthenelse(mask, floodFrac_dyn, 0.0)
        floodDepth = pcr.ifthenelse(mask, floodDepth, 0.0)
        alphaQ = pcr.ifthenelse(mask, dynamicAlphaQ, staticAlphaQ)
        dischargeInitial = pcr.ifthenelse(
            mask, dynamicDischargeInitial, staticDischargeInitial
        )

        return floodedFraction, floodDepth, alphaQ, dischargeInitial

    def readExtensiveMeteo(self, currTimeStep):
        # read the meteorological input directly from netCDF files
        if currTimeStep.day == 1:
            self.cloudCover = vos.netcdf2PCRobjClone(
                self.cloudFileNC,
                "cld",
                str(currTimeStep.fulldate),
                useDoy="monthly",
                LatitudeLongitude=True,
                specificFillValue=-999.0,
            ) / pcr.scalar(100)

            self.vaporPressure = vos.netcdf2PCRobjClone(
                self.vapFileNC,
                "vap",
                str(currTimeStep.fulldate),
                useDoy="monthly",
                LatitudeLongitude=True,
                specificFillValue=-999.0,
            )

            self.annualT = vos.netcdf2PCRobjClone(
                self.annualTFileNC,
                "tas",
                str(currTimeStep.fulldate),
                useDoy="yearly",
                LatitudeLongitude=True,
                specificFillValue=-999.0,
            ) + pcr.scalar(273.15)

        self.radiation = vos.netcdf2PCRobjClone(
            self.radFileNC,
            "rsds",
            str(currTimeStep.fulldate),
            useDoy="daily",
            LatitudeLongitude=True,
            specificFillValue=-999.0,
        )

        # atmospheric emissivity (-) from vapour pressure:
        # min(1, (0.53 + 0.0065 * vaporPressure**0.5) * (1 + 0.4 * cloudCover))
        self.atmosEmis = pcr.scalar(1.0)
        cld1 = pcr.roundoff(10 * self.cloudCover + 0.5)
        cld0 = cld1 - 1
        sun0 = pcr.lookupscalar(self.sunFracTBL, cld0)
        deltaSun = (pcr.lookupscalar(self.sunFracTBL, cld1) - sun0) / (cld1 - cld0)
        sunFrac = sun0 + (10 * self.cloudCover - cld0) * deltaSun
        radFrac = self.radCon + self.radSlope * sunFrac
        self.rsw = radFrac * self.radiation

    def readPollutantLoadingsInputData(self, currTimeStep):

        if currTimeStep.doy == 1:
            # domestic: gridded population (5 arcmin)
            self.Population = vos.netcdf2PCRobjClone(
                self.PopulationNC,
                "Population",
                str(currTimeStep.fulldate),
                useDoy=None,
                LatitudeLongitude=True,
                specificFillValue=None,
            )

            # urban surface runoff: urban fraction (0-1)
            self.urban_area_fraction = vos.netcdf2PCRobjClone(
                self.UrbanFractionNC,
                "urban_fraction",
                str(currTimeStep.fulldate),
                useDoy=None,
                LatitudeLongitude=True,
                specificFillValue=None,
            )
            # zero urban fraction if missing (e.g. for lakes)
            self.urban_area_fraction = pcr.cover(self.urban_area_fraction, 0)

            # livestock populations (5 arcmin): buffalo
            self.BufalloPopulation = vos.netcdf2PCRobjClone(
                self.LivPopulationNC,
                "BufalloPop",
                str(currTimeStep.fulldate),
                useDoy=None,
                LatitudeLongitude=True,
                specificFillValue=None,
            )
            self.BufalloPopulation = pcr.cover(self.BufalloPopulation, 0.0)

            # chicken
            self.ChickenPopulation = vos.netcdf2PCRobjClone(
                self.LivPopulationNC,
                "ChickenPop",
                str(currTimeStep.fulldate),
                useDoy=None,
                LatitudeLongitude=True,
                specificFillValue=None,
            )
            self.ChickenPopulation = pcr.cover(self.ChickenPopulation, 0.0)

            # cow
            self.CowPopulation = vos.netcdf2PCRobjClone(
                self.LivPopulationNC,
                "CowPop",
                str(currTimeStep.fulldate),
                useDoy=None,
                LatitudeLongitude=True,
                specificFillValue=None,
            )
            self.CowPopulation = pcr.cover(self.CowPopulation, 0.0)

            # duck
            self.DuckPopulation = vos.netcdf2PCRobjClone(
                self.LivPopulationNC,
                "DuckPop",
                str(currTimeStep.fulldate),
                useDoy=None,
                LatitudeLongitude=True,
                specificFillValue=None,
            )
            self.DuckPopulation = pcr.cover(self.DuckPopulation, 0.0)

            # goat
            self.GoatPopulation = vos.netcdf2PCRobjClone(
                self.LivPopulationNC,
                "GoatPop",
                str(currTimeStep.fulldate),
                useDoy=None,
                LatitudeLongitude=True,
                specificFillValue=None,
            )
            self.GoatPopulation = pcr.cover(self.GoatPopulation, 0.0)

            # horse
            self.HorsePopulation = vos.netcdf2PCRobjClone(
                self.LivPopulationNC,
                "HorsePop",
                str(currTimeStep.fulldate),
                useDoy=None,
                LatitudeLongitude=True,
                specificFillValue=None,
            )
            self.HorsePopulation = pcr.cover(self.HorsePopulation, 0.0)

            # pig
            self.PigPopulation = vos.netcdf2PCRobjClone(
                self.LivPopulationNC,
                "PigPop",
                str(currTimeStep.fulldate),
                useDoy=None,
                LatitudeLongitude=True,
                specificFillValue=None,
            )
            self.PigPopulation = pcr.cover(self.PigPopulation, 0.0)

            # sheep
            self.SheepPopulation = vos.netcdf2PCRobjClone(
                self.LivPopulationNC,
                "SheepPop",
                str(currTimeStep.fulldate),
                useDoy=None,
                LatitudeLongitude=True,
                specificFillValue=None,
            )
            self.SheepPopulation = pcr.cover(self.SheepPopulation, 0.0)

            # livestock densities accounting for livestock units (Wen et al., 2018); convert m2 to km2
            self.cellArea_km2 = self.cellArea / 1000000.0
            self.LivDensityThres = pcr.scalar(25.0)

            self.BufalloDensity = self.BufalloPopulation / self.cellArea_km2
            self.ChickenDensity = (self.ChickenPopulation * 0.01) / self.cellArea_km2
            self.CowDensity = self.CowPopulation / self.cellArea_km2
            self.DuckDensity = (self.DuckPopulation * 0.01) / self.cellArea_km2
            self.GoatDensity = (self.GoatPopulation * 0.1) / self.cellArea_km2
            self.HorseDensity = self.HorsePopulation / self.cellArea_km2
            self.PigDensity = (self.PigPopulation * 0.3) / self.cellArea_km2
            self.SheepDensity = (self.SheepPopulation * 0.1) / self.cellArea_km2

            # wastewater treatment: fraction of wastewater that is collected and treated
            self.WWt_ct = vos.netcdf2PCRobjClone(
                self.WWtPlantsNC,
                "WW_ct",
                str(currTimeStep.fulldate),
                useDoy="yearly",
                LatitudeLongitude=True,
                specificFillValue=None,
            )

            # country-level fraction of uncollected wastewater with basic sanitation
            self.WWt_bs = vos.netcdf2PCRobjClone(
                self.WWtPlantsNC,
                "WW_bs",
                str(currTimeStep.fulldate),
                useDoy="yearly",
                LatitudeLongitude=True,
                specificFillValue=None,
            )

            # country-level fraction of uncollected wastewater that is open defecation
            self.WWt_od = vos.netcdf2PCRobjClone(
                self.WWtPlantsNC,
                "WW_od",
                str(currTimeStep.fulldate),
                useDoy="yearly",
                LatitudeLongitude=True,
                specificFillValue=None,
            )

            # fraction of TDS removed at each wastewater treatment plant
            self.WWt_TDS_removal = vos.netcdf2PCRobjClone(
                self.WWtPlantsNC,
                "TDS_removal",
                str(currTimeStep.fulldate),
                useDoy="yearly",
                LatitudeLongitude=True,
                specificFillValue=None,
            )

            # fraction of BOD removed at each wastewater treatment plant
            self.WWt_BOD_removal = vos.netcdf2PCRobjClone(
                self.WWtPlantsNC,
                "BOD_removal",
                str(currTimeStep.fulldate),
                useDoy="yearly",
                LatitudeLongitude=True,
                specificFillValue=None,
            )

            # fraction of FC removed at each wastewater treatment plant
            self.WWt_FC_removal = vos.netcdf2PCRobjClone(
                self.WWtPlantsNC,
                "FC_removal",
                str(currTimeStep.fulldate),
                useDoy="yearly",
                LatitudeLongitude=True,
                specificFillValue=None,
            )

    def calculatePollutantLoadings(self, currTimeStep, landSurface, groundwater):
        # calculate the pollutant loadings
        logger.info("Calculating pollutant loadings")

        # fraction of direct runoff (direct runoff / total runoff), used to transport pollution from open
        # defecation and extensive livestock
        self.frac_surfaceRunoff = vos.getValDivZero(
            landSurface.directRunoff,
            (landSurface.landSurfaceRunoff + groundwater.baseflow),
        )

        # pollution from municipal wastewater (domestic, manufacturing and urban surface runoff)

        # gross domestic loadings: population (capita) * excretion rate per capita (g/capita/day; cfu/capita/day),
        # with removal at the wastewater treatment plant (g/day)
        self.Dom_TDSload = pcr.ifthenelse(
            # cells not in a wastewater treatment zone
            self.WWt_zoneID == 0,
            (self.Population * self.DomTDS_ExcrLoad * self.WWt_bs)
            + (
                self.Population
                * self.DomTDS_ExcrLoad
                * self.WWt_od
                * self.frac_surfaceRunoff
            ),
            pcr.ifthenelse(
                self.WWt_zoneID
                # accumulate the loadings over the treatment zone to the plant location
                == self.WWt_plantID,
                pcr.areatotal(
                    pcr.ifthenelse(
                        self.WWt_zoneID != 0, self.Population * self.DomTDS_ExcrLoad, 0
                    ),
                    self.WWt_zoneID,
                )
                * (1 - (self.WWt_TDS_removal * self.WWt_ct)),
                # after accumulation, set locations without a treatment plant to 0
                0.0,
            ),
        )

        # (g/day)
        self.Dom_BODload = pcr.ifthenelse(
            self.WWt_zoneID == 0,
            (self.Population * self.DomBOD_ExcrLoad * self.WWt_bs)
            + (
                self.Population
                * self.DomBOD_ExcrLoad
                * self.WWt_od
                * self.frac_surfaceRunoff
            ),
            pcr.ifthenelse(
                self.WWt_zoneID == self.WWt_plantID,
                pcr.areatotal(
                    pcr.ifthenelse(
                        self.WWt_zoneID != 0, self.Population * self.DomBOD_ExcrLoad, 0
                    ),
                    self.WWt_zoneID,
                )
                * (1 - (self.WWt_BOD_removal * self.WWt_ct)),
                0.0,
            ),
        )

        # (million cfu/day)
        self.Dom_FCload = pcr.ifthenelse(
            self.WWt_zoneID == 0,
            (self.Population * (self.DomFC_ExcrLoad / 1000000.0) * self.WWt_bs)
            + (
                self.Population
                * (self.DomFC_ExcrLoad / 1000000.0)
                * self.WWt_od
                * self.frac_surfaceRunoff
            ),
            pcr.ifthenelse(
                self.WWt_zoneID == self.WWt_plantID,
                pcr.areatotal(
                    pcr.ifthenelse(
                        self.WWt_zoneID != 0,
                        self.Population * (self.DomFC_ExcrLoad / 1000000.0),
                        0,
                    ),
                    self.WWt_zoneID,
                )
                * (1 - (self.WWt_FC_removal * self.WWt_ct)),
                0.0,
            ),
        )

        # gross manufacturing loadings: manufacturing wastewater (m3/day) * average effluent concentration (mg/L; cfu/100mL)
        if self.includeSectors["industry"]:
            self.ManWWp = landSurface.nonIrrReturnFlowVolumePerSector["industry"]
        elif self.includeSectors["manufacture"]:
            self.ManWWp = landSurface.nonIrrReturnFlowVolumePerSector["manufacture"]

        # (g/day)
        self.Man_TDSload = pcr.ifthenelse(
            self.WWt_zoneID == 0,
            self.ManWWp * self.ManTDS_EfflConc,
            pcr.ifthenelse(
                self.WWt_zoneID == self.WWt_plantID,
                pcr.areatotal(
                    pcr.ifthenelse(
                        self.WWt_zoneID != 0, self.ManWWp * self.ManTDS_EfflConc, 0
                    ),
                    self.WWt_zoneID,
                )
                * (1 - (self.WWt_TDS_removal * self.WWt_ct)),
                0.0,
            ),
        )

        # (g/day)
        self.Man_BODload = pcr.ifthenelse(
            self.WWt_zoneID == 0,
            self.ManWWp * self.ManBOD_EfflConc,
            pcr.ifthenelse(
                self.WWt_zoneID == self.WWt_plantID,
                pcr.areatotal(
                    pcr.ifthenelse(
                        self.WWt_zoneID != 0, self.ManWWp * self.ManBOD_EfflConc, 0
                    ),
                    self.WWt_zoneID,
                )
                * (1 - (self.WWt_BOD_removal * self.WWt_ct)),
                0.0,
            ),
        )

        # (million cfu/day)
        self.Man_FCload = pcr.ifthenelse(
            self.WWt_zoneID == 0,
            self.ManWWp * (self.ManFC_EfflConc / 100.0),
            pcr.ifthenelse(
                self.WWt_zoneID == self.WWt_plantID,
                pcr.areatotal(
                    pcr.ifthenelse(
                        self.WWt_zoneID != 0,
                        self.ManWWp * (self.ManFC_EfflConc / 100.0),
                        0,
                    ),
                    self.WWt_zoneID,
                )
                * (1 - (self.WWt_FC_removal * self.WWt_ct)),
                0.0,
            ),
        )

        # gross urban surface runoff loadings: USR return flow (m3/day) * average effluent concentration (mg/L; cfu/100mL)
        # (m3/day)
        self.USR_RF = (
            landSurface.directRunoff * self.urban_area_fraction * self.cellArea
        )

        # (g/day)
        self.USR_TDSload = pcr.ifthenelse(
            self.WWt_zoneID == 0,
            self.USR_RF * self.USRTDS_EfflConc,
            pcr.ifthenelse(
                self.WWt_zoneID == self.WWt_plantID,
                pcr.areatotal(
                    pcr.ifthenelse(
                        self.WWt_zoneID != 0, self.USR_RF * self.USRTDS_EfflConc, 0
                    ),
                    self.WWt_zoneID,
                )
                * (1 - (self.WWt_TDS_removal * self.WWt_ct)),
                0.0,
            ),
        )

        # (g/day)
        self.USR_BODload = pcr.ifthenelse(
            self.WWt_zoneID == 0,
            self.USR_RF * self.USRBOD_EfflConc,
            pcr.ifthenelse(
                self.WWt_zoneID == self.WWt_plantID,
                pcr.areatotal(
                    pcr.ifthenelse(
                        self.WWt_zoneID != 0, self.USR_RF * self.USRBOD_EfflConc, 0
                    ),
                    self.WWt_zoneID,
                )
                * (1 - (self.WWt_BOD_removal * self.WWt_ct)),
                0.0,
            ),
        )

        # (million cfu/day)
        self.USR_FCload = pcr.ifthenelse(
            self.WWt_zoneID == 0,
            self.USR_RF * (self.USRFC_EfflConc / 100.0),
            pcr.ifthenelse(
                self.WWt_zoneID == self.WWt_plantID,
                pcr.areatotal(
                    pcr.ifthenelse(
                        self.WWt_zoneID != 0,
                        self.USR_RF * (self.USRFC_EfflConc / 100.0),
                        0,
                    ),
                    self.WWt_zoneID,
                )
                * (1 - (self.WWt_FC_removal * self.WWt_ct)),
                0.0,
            ),
        )

        # pollution from livestock (intensive and extensive): BOD
        self.intLiv_Bufallo_BODload = pcr.ifthenelse(
            self.BufalloDensity > self.LivDensityThres,
            self.BufalloPopulation * self.Bufallo_BODload,
            0.0,
        )
        self.intLiv_Chicken_BODload = pcr.ifthenelse(
            self.ChickenDensity > self.LivDensityThres,
            self.ChickenPopulation * self.Chicken_BODload,
            0.0,
        )
        self.intLiv_Cow_BODload = pcr.ifthenelse(
            self.CowDensity > self.LivDensityThres,
            self.CowPopulation * self.Cow_BODload,
            0.0,
        )
        self.intLiv_Duck_BODload = pcr.ifthenelse(
            self.DuckDensity > self.LivDensityThres,
            self.DuckPopulation * self.Duck_BODload,
            0.0,
        )
        self.intLiv_Goat_BODload = pcr.ifthenelse(
            self.GoatDensity > self.LivDensityThres,
            self.GoatPopulation * self.Goat_BODload,
            0.0,
        )
        self.intLiv_Horse_BODload = pcr.ifthenelse(
            self.HorseDensity > self.LivDensityThres,
            self.HorsePopulation * self.Horse_BODload,
            0.0,
        )
        self.intLiv_Pig_BODload = pcr.ifthenelse(
            self.PigDensity > self.LivDensityThres,
            self.PigPopulation * self.Pig_BODload,
            0.0,
        )
        self.intLiv_Sheep_BODload = pcr.ifthenelse(
            self.SheepDensity > self.LivDensityThres,
            self.SheepPopulation * self.Sheep_BODload,
            0.0,
        )

        # (g/day)
        self.intLiv_BODload = (
            self.intLiv_Bufallo_BODload
            + self.intLiv_Chicken_BODload
            + self.intLiv_Cow_BODload
            + self.intLiv_Duck_BODload
            + self.intLiv_Goat_BODload
            + self.intLiv_Horse_BODload
            + self.intLiv_Pig_BODload
            + self.intLiv_Sheep_BODload
        )

        # (g/day)
        self.intLiv_BODload = pcr.ifthenelse(
            self.WWt_zoneID == 0,
            self.intLiv_BODload * self.frac_surfaceRunoff,
            pcr.ifthenelse(
                self.WWt_zoneID == self.WWt_plantID,
                pcr.areatotal(
                    pcr.ifthenelse(self.WWt_zoneID != 0, self.intLiv_BODload, 0),
                    self.WWt_zoneID,
                )
                * (1 - (self.WWt_BOD_removal * self.WWt_ct))
                * self.frac_surfaceRunoff,
                0.0,
            ),
        )

        self.extLiv_Bufallo_BODload = pcr.ifthenelse(
            self.BufalloDensity <= self.LivDensityThres,
            self.BufalloPopulation * self.Bufallo_BODload,
            0.0,
        )
        self.extLiv_Chicken_BODload = pcr.ifthenelse(
            self.ChickenDensity <= self.LivDensityThres,
            self.ChickenPopulation * self.Chicken_BODload,
            0.0,
        )
        self.extLiv_Cow_BODload = pcr.ifthenelse(
            self.CowDensity <= self.LivDensityThres,
            self.CowPopulation * self.Cow_BODload,
            0.0,
        )
        self.extLiv_Duck_BODload = pcr.ifthenelse(
            self.DuckDensity <= self.LivDensityThres,
            self.DuckPopulation * self.Duck_BODload,
            0.0,
        )
        self.extLiv_Goat_BODload = pcr.ifthenelse(
            self.GoatDensity <= self.LivDensityThres,
            self.GoatPopulation * self.Goat_BODload,
            0.0,
        )
        self.extLiv_Horse_BODload = pcr.ifthenelse(
            self.HorseDensity <= self.LivDensityThres,
            self.HorsePopulation * self.Horse_BODload,
            0.0,
        )
        self.extLiv_Pig_BODload = pcr.ifthenelse(
            self.PigDensity <= self.LivDensityThres,
            self.PigPopulation * self.Pig_BODload,
            0.0,
        )
        self.extLiv_Sheep_BODload = pcr.ifthenelse(
            self.SheepDensity <= self.LivDensityThres,
            self.SheepPopulation * self.Sheep_BODload,
            0.0,
        )

        # (g/day)
        self.extLiv_BODload = (
            self.extLiv_Bufallo_BODload
            + self.extLiv_Chicken_BODload
            + self.extLiv_Cow_BODload
            + self.extLiv_Duck_BODload
            + self.extLiv_Goat_BODload
            + self.extLiv_Horse_BODload
            + self.extLiv_Pig_BODload
            + self.extLiv_Sheep_BODload
        ) * self.frac_surfaceRunoff

        # FC
        self.intLiv_Bufallo_FCload = pcr.ifthenelse(
            self.BufalloDensity > self.LivDensityThres,
            self.BufalloPopulation * self.Bufallo_FCload,
            0.0,
        )
        self.intLiv_Chicken_FCload = pcr.ifthenelse(
            self.ChickenDensity > self.LivDensityThres,
            self.ChickenPopulation * self.Chicken_FCload,
            0.0,
        )
        self.intLiv_Cow_FCload = pcr.ifthenelse(
            self.CowDensity > self.LivDensityThres,
            self.CowPopulation * self.Cow_FCload,
            0.0,
        )
        self.intLiv_Duck_FCload = pcr.ifthenelse(
            self.DuckDensity > self.LivDensityThres,
            self.DuckPopulation * self.Duck_FCload,
            0.0,
        )
        self.intLiv_Goat_FCload = pcr.ifthenelse(
            self.GoatDensity > self.LivDensityThres,
            self.GoatPopulation * self.Goat_FCload,
            0.0,
        )
        self.intLiv_Horse_FCload = pcr.ifthenelse(
            self.HorseDensity > self.LivDensityThres,
            self.HorsePopulation * self.Horse_FCload,
            0.0,
        )
        self.intLiv_Pig_FCload = pcr.ifthenelse(
            self.PigDensity > self.LivDensityThres,
            self.PigPopulation * self.Pig_FCload,
            0.0,
        )
        self.intLiv_Sheep_FCload = pcr.ifthenelse(
            self.SheepDensity > self.LivDensityThres,
            self.SheepPopulation * self.Sheep_FCload,
            0.0,
        )

        self.intLiv_FCload = (
            self.intLiv_Bufallo_FCload
            + self.intLiv_Chicken_FCload
            + self.intLiv_Cow_FCload
            + self.intLiv_Duck_FCload
            + self.intLiv_Goat_FCload
            + self.intLiv_Horse_FCload
            + self.intLiv_Pig_FCload
            + self.intLiv_Sheep_FCload
        ) / 1000000.0

        # (million cfu/day)
        self.intLiv_FCload = pcr.ifthenelse(
            self.WWt_zoneID == 0,
            self.intLiv_FCload * self.frac_surfaceRunoff,
            pcr.ifthenelse(
                self.WWt_zoneID == self.WWt_plantID,
                pcr.areatotal(
                    pcr.ifthenelse(self.WWt_zoneID != 0, self.intLiv_FCload, 0),
                    self.WWt_zoneID,
                )
                * (1 - (self.WWt_FC_removal * self.WWt_ct))
                * self.frac_surfaceRunoff,
                0.0,
            ),
        )

        self.extLiv_Bufallo_FCload = pcr.ifthenelse(
            self.BufalloDensity <= self.LivDensityThres,
            self.BufalloPopulation * self.Bufallo_FCload,
            0.0,
        )
        self.extLiv_Chicken_FCload = pcr.ifthenelse(
            self.ChickenDensity <= self.LivDensityThres,
            self.ChickenPopulation * self.Chicken_FCload,
            0.0,
        )
        self.extLiv_Cow_FCload = pcr.ifthenelse(
            self.CowDensity <= self.LivDensityThres,
            self.CowPopulation * self.Cow_FCload,
            0.0,
        )
        self.extLiv_Duck_FCload = pcr.ifthenelse(
            self.DuckDensity <= self.LivDensityThres,
            self.DuckPopulation * self.Duck_FCload,
            0.0,
        )
        self.extLiv_Goat_FCload = pcr.ifthenelse(
            self.GoatDensity <= self.LivDensityThres,
            self.GoatPopulation * self.Goat_FCload,
            0.0,
        )
        self.extLiv_Horse_FCload = pcr.ifthenelse(
            self.HorseDensity <= self.LivDensityThres,
            self.HorsePopulation * self.Horse_FCload,
            0.0,
        )
        self.extLiv_Pig_FCload = pcr.ifthenelse(
            self.PigDensity <= self.LivDensityThres,
            self.PigPopulation * self.Pig_FCload,
            0.0,
        )
        self.extLiv_Sheep_FCload = pcr.ifthenelse(
            self.SheepDensity <= self.LivDensityThres,
            self.SheepPopulation * self.Sheep_FCload,
            0.0,
        )

        # (million cfu/day)
        self.extLiv_FCload = (
            (
                self.extLiv_Bufallo_FCload
                + self.extLiv_Chicken_FCload
                + self.extLiv_Cow_FCload
                + self.extLiv_Duck_FCload
                + self.extLiv_Goat_FCload
                + self.extLiv_Horse_FCload
                + self.extLiv_Pig_FCload
                + self.extLiv_Sheep_FCload
            )
            / 1000000.0
        ) * self.frac_surfaceRunoff

        # pollution from irrigation: irrigation return flow (m3/day) * soil TDS (mg/L); the return flows
        # follow from direct runoff, interflow and baseflow
        self.irr_rf_from_direct_runoff = landSurface.directRunoff * vos.getValDivZero(
            self.avg_irrGrossDemand,
            (self.avg_irrGrossDemand + self.avg_netLqWaterToSoil),
        )
        self.irr_rf_from_interflow = landSurface.interflowTotal * vos.getValDivZero(
            self.avg_irrGrossDemand,
            (self.avg_irrGrossDemand + self.avg_netLqWaterToSoil),
        )
        self.irr_rf_from_baseflow = groundwater.baseflow * vos.getValDivZero(
            self.avg_irrGrossDemand,
            (self.avg_irrGrossDemand + self.avg_netLqWaterToSoil),
        )

        # total irrigation return flow (m3/day)
        self.Irr_RF = (
            self.irr_rf_from_direct_runoff
            + self.irr_rf_from_interflow
            + self.irr_rf_from_baseflow
        ) * self.cellArea
        # (g/day)
        self.Irr_TDSload = self.Irr_RF * self.IrrTDS_EfflConc

        # pollution from all human activities combined (g/day)
        self.TDSload = (
            self.Dom_TDSload + self.Man_TDSload + self.USR_TDSload + self.Irr_TDSload
        )
        # (g/day)
        self.BODload = (
            self.Dom_BODload
            + self.Man_BODload
            + self.USR_BODload
            + self.intLiv_BODload
            + self.extLiv_BODload
        )
        # (million cfu/day)
        self.FCload = (
            self.Dom_FCload
            + self.Man_FCload
            + self.USR_FCload
            + self.intLiv_FCload
            + self.extLiv_FCload
        )

    def readPollutantLoadings(self, currTimeStep):
        # use precalculated pollutant loadings
        logger.info("Reading loadings directly")

        # TDS loadings (all sectors combined)
        self.TDSload = vos.netcdf2PCRobjClone(
            self.TDSloadNC,
            "TDSload",
            str(currTimeStep.fulldate),
            useDoy=None,
            LatitudeLongitude=True,
        )
        self.TDSload = pcr.ifthen(self.landmask, self.TDSload)

        # BOD loadings (all sectors combined)
        self.BODload = vos.netcdf2PCRobjClone(
            self.BODloadNC,
            "BODload",
            str(currTimeStep.fulldate),
            useDoy=None,
            LatitudeLongitude=True,
        )
        self.BODload = pcr.ifthen(self.landmask, self.BODload)

        # FC loadings (all sectors combined)
        self.FCload = vos.netcdf2PCRobjClone(
            self.FCloadNC,
            "FCload",
            str(currTimeStep.fulldate),
            useDoy=None,
            LatitudeLongitude=True,
        )
        self.FCload = pcr.ifthen(self.landmask, self.FCload)

    def qualityLocal(
        self, meteo, landSurface, groundwater, currTimeStep, timeSec=vos.secondsPerDay()
    ):
        """
        DynQual (quality = True): local water-temperature energy balance, ice formation
        and local pollutant inputs; called in simple_update after WaterBodies.update and
        the release injection, before routing.

        Water-body quantities read: self.dynamicFracWat, self.runoff,
        landSurface.actSurfaceWaterAbstract, self.waterBodyEvaporation,
        self.channelStorageTimeBefore (copy of channelStorage after the release) and
        return_water_body_storage_to_channel applied to that copy (result not assigned
        to channelStorage). Writes no water-quantity state: channelStorage,
        waterBodyStorage and the WaterBodies attributes are untouched.

        CAUTION (R13): it sets self.waterBodyEvaporation to 0 where the air temperature
        is below 273.15 K, after the evaporation volume was already removed from
        channelStorage (calculate_evaporation). On those days the reported open-water
        evaporation (reporting.py, waterBodyActEvaporation, totalEvaporation) and the
        water-balance accumulators of pcrglobwb.py miss a volume that was taken from
        storage, including lake/reservoir storage. Only in
        online_pcrglobwb_qualloc_5arcminutes.ini among the RAWS configurations.
        """

        # water temperature: surface water energy fluxes (W/m2) within the current time step; ice formation
        # is evaluated before routing to account for the loss in water height, the vertical energy change is
        # evaluated and warming is capped at the air temperature

        # constants and variables
        self.temperatureKelvin = meteo.temperature + pcr.scalar(273.15)
        landRunoff = self.runoff
        self.correctPrecip = pcr.scalar(0.0)
        self.dynamicFracWatBeforeRouting = self.dynamicFracWat

        # land temperature considering the runoff fractions
        landT = pcr.cover(
            landSurface.directRunoff
            / landRunoff
            * pcr.max(self.iceThresTemp + 0.1, self.temperatureKelvin - self.deltaTPrec)
            + landSurface.interflowTotal
            / landRunoff
            * pcr.max(self.iceThresTemp + 0.1, self.temperatureKelvin)
            + groundwater.baseflow
            / landRunoff
            * pcr.max(self.iceThresTemp + 5.0, self.annualT),
            self.temperatureKelvin,
        )

        # heat transfer (ice and water)
        iceHeatTransfer = self.heatTransferWater * (
            self.temperatureKelvin - self.iceThresTemp
        )
        waterHeatTransfer = self.heatTransferIce * (self.iceThresTemp - self.waterTemp)

        # check for ice
        noIce = pcr.ifthenelse(
            self.iceThickness > 0,
            pcr.boolean(0),
            pcr.ifthenelse(
                ((iceHeatTransfer - waterHeatTransfer) < 0)
                & (self.temperatureKelvin < self.iceThresTemp),
                pcr.boolean(0),
                pcr.boolean(1),
            ),
        )

        # update the water heat transfer based on the presence of ice
        waterHeatTransfer = pcr.ifthenelse(
            noIce,
            self.heatTransferWater * (self.temperatureKelvin - self.waterTemp),
            waterHeatTransfer,
        )

        # radiative heat transfer
        radiativHeatTransfer = (
            1 - pcr.ifthenelse(noIce, self.albedoWater, self.albedoSnow)
        ) * self.rsw
        radiativHeatTransfer -= self.stefanBoltzman * (
            pcr.ifthenelse(noIce, self.waterTemp, self.iceThresTemp) ** 4
            - self.atmosEmis * self.temperatureKelvin**4
        )

        # advected energy due to precipitation and inflow
        advectedEnergyPrecip = (
            pcr.max(0, self.correctPrecip)
            * pcr.max(self.iceThresTemp + 0.1, self.temperatureKelvin - self.deltaTPrec)
            * self.specificHeatWater
            * self.densityWater
            / timeSec
        )
        advectedEnergyInflow = (
            (1 - self.dynamicFracWat)
            / self.dynamicFracWat
            * landRunoff
            * landT
            * self.specificHeatWater
            * self.densityWater
            / timeSec
        )
        advectedEnergyInflow -= (
            landSurface.actSurfaceWaterAbstract
            / self.dynamicFracWat
            * self.waterTemp
            * self.specificHeatWater
            * self.densityWater
            / timeSec
        )

        # ice formation and thickness
        diceHeatTransfer = pcr.ifthenelse(
            noIce, 0, iceHeatTransfer - waterHeatTransfer + radiativHeatTransfer
        )
        self.deltaIceThickness = (
            -diceHeatTransfer * timeSec / (self.densityWater * self.latentHeatFusion)
        )
        self.deltaIceThickness = pcr.max(-self.iceThickness, self.deltaIceThickness)
        self.deltaIceThickness = pcr.min(
            self.deltaIceThickness, pcr.max(0, self.maxIceThickness - self.iceThickness)
        )

        # direct gain over the water surface
        watQ = pcr.ifthenelse(
            self.temperatureKelvin >= self.iceThresTemp,
            pcr.max(0, self.correctPrecip)
            - self.waterBodyEvaporation / self.dynamicFracWat,
            0,
        )
        # TODO: move
        self.waterBodyEvaporation = pcr.ifthenelse(
            self.temperatureKelvin >= self.iceThresTemp, self.waterBodyEvaporation, 0
        )

        # vertical gains/losses
        deltaIceThickness_melt = pcr.max(0, -self.deltaIceThickness)
        verticalGain = (
            watQ
            + (landRunoff - landSurface.actSurfaceWaterAbstract) / (self.dynamicFracWat)
            + deltaIceThickness_melt
        )

        # water storage change
        dtotStorLoc = verticalGain
        totStorLoc = (
            self.channelStorageTimeBefore / (self.dynamicFracWat * self.cellArea)
            - dtotStorLoc
        )
        totStorLoc = (
            self.return_water_body_storage_to_channel(self.channelStorageTimeBefore)
            / (self.dynamicFracWat * self.cellArea)
            - dtotStorLoc
        )

        # energy in the channel
        self.totEW = (
            totStorLoc * self.waterTemp * self.specificHeatWater * self.densityWater
        )
        dtotStorLoc = pcr.max(-totStorLoc, dtotStorLoc)

        # latent heat flux and advected energy of water evaporation
        latentHeat = (
            -self.waterBodyEvaporation
            / self.dynamicFracWat
            * self.densityWater
            * self.latentHeatVapor
            / timeSec
        )
        advectedEnergyPrecip += (
            deltaIceThickness_melt
            * self.iceThresTemp
            * self.specificHeatWater
            * self.densityWater
            / timeSec
        )

        # total energy change
        totEWC = totStorLoc * self.specificHeatWater * self.densityWater
        dtotEWC = dtotStorLoc * self.specificHeatWater * self.densityWater
        dtotEWLoc = (
            waterHeatTransfer + pcr.scalar(noIce) * (radiativHeatTransfer + latentHeat)
        ) * timeSec
        dtotEWAdv = (advectedEnergyInflow + advectedEnergyPrecip) * timeSec

        dtotEWLoc = pcr.min(
            dtotEWLoc,
            pcr.max(0, totEWC * self.temperatureKelvin - self.totEW)
            + pcr.ifthenelse(
                dtotStorLoc > 0,
                pcr.max(0, dtotEWC * self.temperatureKelvin - dtotEWAdv),
                0,
            ),
        )
        dtotEWLoc = pcr.ifthenelse(
            self.waterTemp > self.temperatureKelvin, pcr.min(0, dtotEWLoc), dtotEWLoc
        )
        dtotEWLoc = pcr.max(
            dtotEWLoc,
            pcr.min(
                0,
                (totEWC + dtotEWC)
                * pcr.max(self.temperatureKelvin, self.iceThresTemp + 0.1)
                - (self.totEW + dtotEWAdv),
            ),
        )

        # update the water height, energy and water temperature
        self.temp_water_height = pcr.max(1e-16, totStorLoc + dtotStorLoc)
        self.totEW = pcr.max(0, self.totEW + dtotEWLoc + dtotEWAdv)

        # temporary fix for an error in the water temperature in the first time step
        if currTimeStep.timeStepPCR != 1:
            self.waterTemp = pcr.ifthenelse(
                self.temp_water_height > self.critical_water_height,
                self.totEW
                / self.temp_water_height
                / (self.specificHeatWater * self.densityWater),
                self.temperatureKelvin,
            )
        self.waterTemp = pcr.min(
            pcr.ifthenelse(
                self.waterTemp < self.iceThresTemp + 0.1,
                self.iceThresTemp + 0.1,
                self.waterTemp,
            ),
            self.maxThresTemp,
        )
        # water temperature (degC)
        self.waterTemp_C = self.waterTemp - pcr.scalar(273.15)

        # BOD (non-conservative; depends on water temperature only): temperature-dependent decay
        self.BODdecay_temperature = pcr.cover(
            self.k_BOD * (self.watertempcorrection_BOD ** (self.waterTemp_C - 20)), 0.0
        )

        # FC (non-conservative; depends on temperature, solar radiation and sedimentation): temperature-dependent decay
        self.FCdecay_temperature = pcr.cover(
            self.darkinactivation_FC
            * (self.watertempcorrection_FC ** (self.waterTemp_C - 20)),
            0.0,
        )

        # solar radiation dependent decay, with a minimum water depth of 0.1 m for the decay coefficients
        self.water_height_pathogen = pcr.max(self.water_height, 0.1)
        self.FCdecay_solarradiation = pcr.cover(
            self.sunlightinactivation_FC
            * (self.rsw / (self.attenuation_FC * self.water_height_pathogen))
            * (1 - (pcr.exp(-(self.attenuation_FC * self.water_height_pathogen)))),
            0.0,
        )

        # sedimentation (day-1)
        self.FCdecay_sedimentation = pcr.ifthenelse(
            self.water_height_pathogen > self.threshold_FC_settlingdepth,
            pcr.cover(self.settlingvelocity_FC / self.water_height, 0.0),
            0,
        )

    def qualityRouting(self, timeSec):
        """
        DynQual: route heat and pollutant loads along the ldd over one sub-step of
        timeSec seconds, moving the fraction subDischarge * timeSec /
        channelStorageTimeBefore (at most 1) of each cell's load downstream. Called only
        by simplifiedKinematicWave, once per sub-step (with accuTravelTime and quality =
        True the loads are not routed; whether this is intended is not determined from
        the code). Water-body quantities read: self.subDischarge (m3/s) and
        self.channelStorageTimeBefore (m3); at the body's cells these are the routed
        channel values (the body's storage is inside WaterBodies during routing). Writes
        no water-quantity variable.
        """

        channelTransFrac = pcr.cover(
            pcr.max(
                pcr.min(
                    (self.subDischarge * timeSec) / self.channelStorageTimeBefore, 1.0
                ),
                0.0,
            ),
            0.0,
        )

        # energy (water temperature) routing, adding heat effluents from power plants (J per timeSec)
        self.volumeEW = self.volumeEW + (
            self.PowTwload * (timeSec / vos.secondsPerDay())
        )
        dtotEWLat = channelTransFrac * self.volumeEW
        self.volumeEW = self.volumeEW + pcr.upstream(self.lddMap, dtotEWLat) - dtotEWLat

        # salinity (TDS) routing
        self.routedTDS = self.routedTDS + (
            self.TDSload * (timeSec / vos.secondsPerDay())
        )
        dTDSLat = channelTransFrac * self.routedTDS
        self.routedTDS = self.routedTDS + pcr.upstream(self.lddMap, dTDSLat) - dTDSLat

        if self.loadsPerSector:
            self.routedDomTDS = self.routedDomTDS + (
                self.Dom_TDSload * (timeSec / vos.secondsPerDay())
            )
            dDomTDSLat = channelTransFrac * self.routedDomTDS
            self.routedDomTDS = (
                self.routedDomTDS + pcr.upstream(self.lddMap, dDomTDSLat) - dDomTDSLat
            )

            self.routedManTDS = self.routedManTDS + (
                self.Man_TDSload * (timeSec / vos.secondsPerDay())
            )
            dManTDSLat = channelTransFrac * self.routedManTDS
            self.routedManTDS = (
                self.routedManTDS + pcr.upstream(self.lddMap, dManTDSLat) - dManTDSLat
            )

            self.routedUSRTDS = self.routedUSRTDS + (
                self.USR_TDSload * (timeSec / vos.secondsPerDay())
            )
            dUSRTDSLat = channelTransFrac * self.routedUSRTDS
            self.routedUSRTDS = (
                self.routedUSRTDS + pcr.upstream(self.lddMap, dUSRTDSLat) - dUSRTDSLat
            )

            self.routedIrrTDS = self.routedIrrTDS + (
                self.Irr_TDSload * (timeSec / vos.secondsPerDay())
            )
            dIrrTDSLat = channelTransFrac * self.routedIrrTDS
            self.routedIrrTDS = (
                self.routedIrrTDS + pcr.upstream(self.lddMap, dIrrTDSLat) - dIrrTDSLat
            )

        # organic (BOD) routing
        self.routedBOD = self.routedBOD + (
            self.BODload * (timeSec / vos.secondsPerDay())
        )
        dBODLat = channelTransFrac * self.routedBOD
        self.BODdecay = pcr.exp(
            -(self.BODdecay_temperature) * (timeSec / vos.secondsPerDay())
        )
        self.routedBOD = (
            self.routedBOD + pcr.upstream(self.lddMap, dBODLat) - dBODLat
        ) * self.BODdecay

        if self.loadsPerSector:
            self.routedDomBOD = self.routedDomBOD + (
                self.Dom_BODload * (timeSec / vos.secondsPerDay())
            )
            dDomBODLat = channelTransFrac * self.routedDomBOD
            self.routedDomBOD = (
                self.routedDomBOD + pcr.upstream(self.lddMap, dDomBODLat) - dDomBODLat
            ) * self.BODdecay

            self.routedManBOD = self.routedManBOD + (
                self.Man_BODload * (timeSec / vos.secondsPerDay())
            )
            dManBODLat = channelTransFrac * self.routedManBOD
            self.routedManBOD = (
                self.routedManBOD + pcr.upstream(self.lddMap, dManBODLat) - dManBODLat
            ) * self.BODdecay

            self.routedUSRBOD = self.routedUSRBOD + (
                self.USR_BODload * (timeSec / vos.secondsPerDay())
            )
            dUSRBODLat = channelTransFrac * self.routedUSRBOD
            self.routedUSRBOD = (
                self.routedUSRBOD + pcr.upstream(self.lddMap, dUSRBODLat) - dUSRBODLat
            ) * self.BODdecay

            self.routedintLivBOD = self.routedintLivBOD + (
                self.intLiv_BODload * (timeSec / vos.secondsPerDay())
            )
            dintLivBODLat = channelTransFrac * self.routedintLivBOD
            self.routedintLivBOD = (
                self.routedintLivBOD
                + pcr.upstream(self.lddMap, dintLivBODLat)
                - dintLivBODLat
            ) * self.BODdecay

            self.routedextLivBOD = self.routedextLivBOD + (
                self.extLiv_BODload * (timeSec / vos.secondsPerDay())
            )
            dextLivBODLat = channelTransFrac * self.routedextLivBOD
            self.routedextLivBOD = (
                self.routedextLivBOD
                + pcr.upstream(self.lddMap, dextLivBODLat)
                - dextLivBODLat
            ) * self.BODdecay

        # pathogen (FC) routing
        self.routedFC = self.routedFC + (self.FCload * (timeSec / vos.secondsPerDay()))
        dFCLat = channelTransFrac * self.routedFC
        self.FCdecay = pcr.exp(
            -(
                self.FCdecay_temperature
                + self.FCdecay_solarradiation
                + self.FCdecay_sedimentation
            )
            * (timeSec / vos.secondsPerDay())
        )
        self.routedFC = (
            self.routedFC + pcr.upstream(self.lddMap, dFCLat) - dFCLat
        ) * self.FCdecay

        if self.loadsPerSector:
            self.routedDomFC = self.routedDomFC + (
                self.Dom_FCload * (timeSec / vos.secondsPerDay())
            )
            dDomFCLat = channelTransFrac * self.routedDomFC
            self.routedDomFC = (
                self.routedDomFC + pcr.upstream(self.lddMap, dDomFCLat) - dDomFCLat
            ) * self.FCdecay

            self.routedManFC = self.routedManFC + (
                self.Man_FCload * (timeSec / vos.secondsPerDay())
            )
            dManFCLat = channelTransFrac * self.routedManFC
            self.routedManFC = (
                self.routedManFC + pcr.upstream(self.lddMap, dManFCLat) - dManFCLat
            ) * self.FCdecay

            self.routedUSRFC = self.routedUSRFC + (
                self.USR_FCload * (timeSec / vos.secondsPerDay())
            )
            dUSRFCLat = channelTransFrac * self.routedUSRFC
            self.routedUSRFC = (
                self.routedUSRFC + pcr.upstream(self.lddMap, dUSRFCLat) - dUSRFCLat
            ) * self.FCdecay

            self.routedintLivFC = self.routedintLivFC + (
                self.intLiv_FCload * (timeSec / vos.secondsPerDay())
            )
            dintLivFCLat = channelTransFrac * self.routedintLivFC
            self.routedintLivFC = (
                self.routedintLivFC
                + pcr.upstream(self.lddMap, dintLivFCLat)
                - dintLivFCLat
            ) * self.FCdecay

            self.routedextLivFC = self.routedextLivFC + (
                self.extLiv_FCload * (timeSec / vos.secondsPerDay())
            )
            dextLivFCLat = channelTransFrac * self.routedextLivFC
            self.routedextLivFC = (
                self.routedextLivFC
                + pcr.upstream(self.lddMap, dextLivFCLat)
                - dextLivFCLat
            ) * self.FCdecay

    def qualityWaterBody(self):
        """
        DynQual: mix heat and pollutant loads within each lake/reservoir before routing.
        Reads WaterBodies.waterBodyOutflow (m3 per day), self.waterBodyStorageTimeBefore
        (storage before outflow, m3 per body), WaterBodies.waterBodyIds and
        waterBodyOut, self.dynamicFracWat and cellArea: the body's totals (pcr.areatotal
        over waterBodyIds) are split into the fraction outflow / storage-before-outflow,
        released at the outlet cell, and the remainder kept in the body. Writes no
        water-quantity variable.
        """

        lakeTransFrac = pcr.max(
            pcr.min(
                (self.WaterBodies.waterBodyOutflow) / (self.waterBodyStorageTimeBefore),
                1.0,
            ),
            0.0,
        )
        lakeTransFrac = pcr.cover(
            pcr.ifthen(self.WaterBodies.waterBodyOut, lakeTransFrac), 0.0
        )

        # water temperature (energy in the water body)
        energyTotal = pcr.cover(
            pcr.ifthen(
                pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                pcr.areatotal(
                    pcr.ifthen(
                        self.landmask, self.totEW * self.dynamicFracWat * self.cellArea
                    ),
                    pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                ),
            ),
            self.totEW * self.dynamicFracWat * self.cellArea,
        )
        self.volumeEW = pcr.cover(
            pcr.ifthen(
                pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                energyTotal * lakeTransFrac,
            ),
            energyTotal,
        )
        self.remainingVolumeEW = pcr.cover(
            pcr.ifthen(
                self.WaterBodies.waterBodyOut, (1 - lakeTransFrac) * energyTotal
            ),
            0.0,
        )

        # salinity (TDS in the water body)
        wbTDSTotal = pcr.cover(
            pcr.ifthen(
                pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                pcr.areatotal(
                    pcr.ifthen(self.landmask, self.routedTDS),
                    pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                ),
            ),
            self.routedTDS,
        )
        self.routedTDS = pcr.cover(
            pcr.ifthen(
                pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                wbTDSTotal * lakeTransFrac,
            ),
            wbTDSTotal,
        )
        self.wbRemainingTDS = pcr.cover(
            pcr.ifthen(self.WaterBodies.waterBodyOut, (1 - lakeTransFrac) * wbTDSTotal),
            0.0,
        )

        if self.loadsPerSector:
            wbDomTDSTotal = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    pcr.areatotal(
                        pcr.ifthen(self.landmask, self.routedDomTDS),
                        pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                    ),
                ),
                self.routedDomTDS,
            )
            self.routedDomTDS = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    wbDomTDSTotal * lakeTransFrac,
                ),
                wbDomTDSTotal,
            )
            self.wbRemainingDomTDS = pcr.cover(
                pcr.ifthen(
                    self.WaterBodies.waterBodyOut, (1 - lakeTransFrac) * wbDomTDSTotal
                ),
                0.0,
            )

            wbManTDSTotal = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    pcr.areatotal(
                        pcr.ifthen(self.landmask, self.routedManTDS),
                        pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                    ),
                ),
                self.routedManTDS,
            )
            self.routedManTDS = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    wbManTDSTotal * lakeTransFrac,
                ),
                wbManTDSTotal,
            )
            self.wbRemainingManTDS = pcr.cover(
                pcr.ifthen(
                    self.WaterBodies.waterBodyOut, (1 - lakeTransFrac) * wbManTDSTotal
                ),
                0.0,
            )

            wbUSRTDSTotal = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    pcr.areatotal(
                        pcr.ifthen(self.landmask, self.routedUSRTDS),
                        pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                    ),
                ),
                self.routedUSRTDS,
            )
            self.routedUSRTDS = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    wbUSRTDSTotal * lakeTransFrac,
                ),
                wbUSRTDSTotal,
            )
            self.wbRemainingUSRTDS = pcr.cover(
                pcr.ifthen(
                    self.WaterBodies.waterBodyOut, (1 - lakeTransFrac) * wbUSRTDSTotal
                ),
                0.0,
            )

            wbIrrTDSTotal = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    pcr.areatotal(
                        pcr.ifthen(self.landmask, self.routedIrrTDS),
                        pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                    ),
                ),
                self.routedIrrTDS,
            )
            self.routedIrrTDS = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    wbIrrTDSTotal * lakeTransFrac,
                ),
                wbIrrTDSTotal,
            )
            self.wbRemainingIrrTDS = pcr.cover(
                pcr.ifthen(
                    self.WaterBodies.waterBodyOut, (1 - lakeTransFrac) * wbIrrTDSTotal
                ),
                0.0,
            )

        # organic (BOD in the water body)
        wbBODTotal = pcr.cover(
            pcr.ifthen(
                pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                pcr.areatotal(
                    pcr.ifthen(self.landmask, self.routedBOD),
                    pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                ),
            ),
            self.routedBOD,
        )
        self.routedBOD = pcr.cover(
            pcr.ifthen(
                pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                wbBODTotal * lakeTransFrac,
            ),
            wbBODTotal,
        )
        self.wbRemainingBOD = pcr.cover(
            pcr.ifthen(self.WaterBodies.waterBodyOut, (1 - lakeTransFrac) * wbBODTotal),
            0.0,
        )

        if self.loadsPerSector:
            wbDomBODTotal = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    pcr.areatotal(
                        pcr.ifthen(self.landmask, self.routedDomBOD),
                        pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                    ),
                ),
                self.routedDomBOD,
            )
            self.routedDomBOD = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    wbDomBODTotal * lakeTransFrac,
                ),
                wbDomBODTotal,
            )
            self.wbRemainingDomBOD = pcr.cover(
                pcr.ifthen(
                    self.WaterBodies.waterBodyOut, (1 - lakeTransFrac) * wbDomBODTotal
                ),
                0.0,
            )

            wbManBODTotal = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    pcr.areatotal(
                        pcr.ifthen(self.landmask, self.routedManBOD),
                        pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                    ),
                ),
                self.routedManBOD,
            )
            self.routedManBOD = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    wbManBODTotal * lakeTransFrac,
                ),
                wbManBODTotal,
            )
            self.wbRemainingManBOD = pcr.cover(
                pcr.ifthen(
                    self.WaterBodies.waterBodyOut, (1 - lakeTransFrac) * wbManBODTotal
                ),
                0.0,
            )

            wbUSRBODTotal = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    pcr.areatotal(
                        pcr.ifthen(self.landmask, self.routedUSRBOD),
                        pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                    ),
                ),
                self.routedUSRBOD,
            )
            self.routedUSRBOD = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    wbUSRBODTotal * lakeTransFrac,
                ),
                wbUSRBODTotal,
            )
            self.wbRemainingUSRBOD = pcr.cover(
                pcr.ifthen(
                    self.WaterBodies.waterBodyOut, (1 - lakeTransFrac) * wbUSRBODTotal
                ),
                0.0,
            )

            wbintLivBODTotal = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    pcr.areatotal(
                        pcr.ifthen(self.landmask, self.routedintLivBOD),
                        pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                    ),
                ),
                self.routedintLivBOD,
            )
            self.routedintLivBOD = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    wbintLivBODTotal * lakeTransFrac,
                ),
                wbintLivBODTotal,
            )
            self.wbRemainingintLivBOD = pcr.cover(
                pcr.ifthen(
                    self.WaterBodies.waterBodyOut,
                    (1 - lakeTransFrac) * wbintLivBODTotal,
                ),
                0.0,
            )

            wbextLivBODTotal = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    pcr.areatotal(
                        pcr.ifthen(self.landmask, self.routedextLivBOD),
                        pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                    ),
                ),
                self.routedextLivBOD,
            )
            self.routedextLivBOD = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    wbextLivBODTotal * lakeTransFrac,
                ),
                wbextLivBODTotal,
            )
            self.wbRemainingextLivBOD = pcr.cover(
                pcr.ifthen(
                    self.WaterBodies.waterBodyOut,
                    (1 - lakeTransFrac) * wbextLivBODTotal,
                ),
                0.0,
            )

        # pathogen (FC in the water body)
        wbFCTotal = pcr.cover(
            pcr.ifthen(
                pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                pcr.areatotal(
                    pcr.ifthen(self.landmask, self.routedFC),
                    pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                ),
            ),
            self.routedFC,
        )
        self.routedFC = pcr.cover(
            pcr.ifthen(
                pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                wbFCTotal * lakeTransFrac,
            ),
            wbFCTotal,
        )
        self.wbRemainingFC = pcr.cover(
            pcr.ifthen(self.WaterBodies.waterBodyOut, (1 - lakeTransFrac) * wbFCTotal),
            0.0,
        )

        if self.loadsPerSector:
            wbDomFCTotal = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    pcr.areatotal(
                        pcr.ifthen(self.landmask, self.routedDomFC),
                        pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                    ),
                ),
                self.routedDomFC,
            )
            self.routedDomFC = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    wbDomFCTotal * lakeTransFrac,
                ),
                wbDomFCTotal,
            )
            self.wbRemainingDomFC = pcr.cover(
                pcr.ifthen(
                    self.WaterBodies.waterBodyOut, (1 - lakeTransFrac) * wbDomFCTotal
                ),
                0.0,
            )

            wbManFCTotal = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    pcr.areatotal(
                        pcr.ifthen(self.landmask, self.routedManFC),
                        pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                    ),
                ),
                self.routedManFC,
            )
            self.routedManFC = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    wbManFCTotal * lakeTransFrac,
                ),
                wbManFCTotal,
            )
            self.wbRemainingManFC = pcr.cover(
                pcr.ifthen(
                    self.WaterBodies.waterBodyOut, (1 - lakeTransFrac) * wbManFCTotal
                ),
                0.0,
            )

            wbUSRFCTotal = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    pcr.areatotal(
                        pcr.ifthen(self.landmask, self.routedUSRFC),
                        pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                    ),
                ),
                self.routedUSRFC,
            )
            self.routedUSRFC = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    wbUSRFCTotal * lakeTransFrac,
                ),
                wbUSRFCTotal,
            )
            self.wbRemainingUSRFC = pcr.cover(
                pcr.ifthen(
                    self.WaterBodies.waterBodyOut, (1 - lakeTransFrac) * wbUSRFCTotal
                ),
                0.0,
            )

            wbintLivFCTotal = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    pcr.areatotal(
                        pcr.ifthen(self.landmask, self.routedintLivFC),
                        pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                    ),
                ),
                self.routedintLivFC,
            )
            self.routedintLivFC = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    wbintLivFCTotal * lakeTransFrac,
                ),
                wbintLivFCTotal,
            )
            self.wbRemainingintLivFC = pcr.cover(
                pcr.ifthen(
                    self.WaterBodies.waterBodyOut, (1 - lakeTransFrac) * wbintLivFCTotal
                ),
                0.0,
            )

            wbextLivFCTotal = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    pcr.areatotal(
                        pcr.ifthen(self.landmask, self.routedextLivFC),
                        pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                    ),
                ),
                self.routedextLivFC,
            )
            self.routedextLivFC = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    wbextLivFCTotal * lakeTransFrac,
                ),
                wbextLivFCTotal,
            )
            self.wbRemainingextLivFC = pcr.cover(
                pcr.ifthen(
                    self.WaterBodies.waterBodyOut, (1 - lakeTransFrac) * wbextLivFCTotal
                ),
                0.0,
            )

    def qualityWaterBodyAverage(self, currTimeStep):
        """
        DynQual: after routing, average the water-body heat and loads over each body (by
        cell area) and update ice thickness and water temperature. Reads
        WaterBodies.waterBodyIds, dynamicFracWat (before and after routing),
        self.channelStorageNow (copy of the routed storage) and
        return_water_body_storage_to_channel of that copy. It subtracts ice growth from
        self.channelStorageNow, a DynQual copy; channelStorage itself is not changed, so
        ice does not remove water from channels, lakes or reservoirs. Writes no
        water-quantity variable.
        """

        # water temperature (energy averaged over the water body)
        self.totalVolumeEW = self.volumeEW + self.remainingVolumeEW

        energyTotal = pcr.cover(
            pcr.ifthen(
                pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                pcr.areatotal(
                    pcr.ifthen(self.landmask, self.totalVolumeEW),
                    pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                ),
            ),
            self.totalVolumeEW,
        )
        energyAverageLakeCell = pcr.cover(
            energyTotal
            * self.cellArea
            / pcr.areatotal(
                pcr.cover(self.cellArea, 0.0),
                pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
            ),
            energyTotal,
        )
        self.totEW = pcr.cover(
            energyAverageLakeCell / (self.dynamicFracWat * self.cellArea), 1e-16
        )

        self.temp_water_height = self.return_water_body_storage_to_channel(
            self.channelStorageNow
        ) / (self.dynamicFracWat * self.cellArea)

        iceReductionFactor = pcr.ifthen(
            self.landmask,
            pcr.cover(self.dynamicFracWatBeforeRouting / self.dynamicFracWat, 1.0),
        )

        self.deltaIceThickness = iceReductionFactor * self.deltaIceThickness
        self.deltaIceThickness = pcr.min(self.deltaIceThickness, self.temp_water_height)

        self.iceThickness = iceReductionFactor * self.iceThickness
        self.iceThickness = pcr.max(
            0,
            self.iceThickness
            + (
                self.deltaIceThickness
                + pcr.ifthenelse(
                    self.temperatureKelvin >= self.iceThresTemp, 0, self.correctPrecip
                )
            ),
        )
        self.iceThickness = pcr.ifthenelse(
            (self.iceThickness <= 0.001) & (self.deltaIceThickness < 0),
            0,
            self.iceThickness,
        )
        self.iceThickness = pcr.min(self.iceThickness, self.maxIceThickness)

        self.channelStorageNow = (
            self.channelStorageNow
            - self.deltaIceThickness * self.dynamicFracWat * self.cellArea
        )

        if currTimeStep.timeStepPCR == 1:
            logger.info(
                "Issue with estimating water temperature in first timestep: retrieving water temperature from initial condition"
            )
        else:
            self.waterTemp = pcr.ifthenelse(
                self.temp_water_height > self.critical_water_height,
                self.totEW
                / self.temp_water_height
                / (self.specificHeatWater * self.densityWater),
                self.temperatureKelvin,
            )

        self.waterTemp = pcr.min(
            pcr.ifthenelse(
                self.waterTemp < self.iceThresTemp + 0.1,
                self.iceThresTemp + 0.1,
                self.waterTemp,
            ),
            self.maxThresTemp,
        )
        # water temperature (degC)
        self.waterTemp_C = self.waterTemp - pcr.scalar(273.15)

        # salinity (TDS averaged over the water body)
        wbTDSTotal = pcr.cover(
            pcr.ifthen(
                pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                pcr.areatotal(
                    pcr.ifthen(self.landmask, self.routedTDS + self.wbRemainingTDS),
                    pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                ),
            ),
            self.routedTDS,
        )
        self.routedTDS = pcr.cover(
            wbTDSTotal
            * self.cellArea
            / pcr.areatotal(
                pcr.cover(self.cellArea, 0.0),
                pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
            ),
            wbTDSTotal,
        )

        if self.loadsPerSector:
            wbDomTDSTotal = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    pcr.areatotal(
                        pcr.ifthen(
                            self.landmask, self.routedDomTDS + self.wbRemainingDomTDS
                        ),
                        pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                    ),
                ),
                self.routedDomTDS,
            )
            self.routedDomTDS = pcr.cover(
                wbDomTDSTotal
                * self.cellArea
                / pcr.areatotal(
                    pcr.cover(self.cellArea, 0.0),
                    pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                ),
                wbDomTDSTotal,
            )

            wbManTDSTotal = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    pcr.areatotal(
                        pcr.ifthen(
                            self.landmask, self.routedManTDS + self.wbRemainingManTDS
                        ),
                        pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                    ),
                ),
                self.routedManTDS,
            )
            self.routedManTDS = pcr.cover(
                wbManTDSTotal
                * self.cellArea
                / pcr.areatotal(
                    pcr.cover(self.cellArea, 0.0),
                    pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                ),
                wbManTDSTotal,
            )

            wbUSRTDSTotal = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    pcr.areatotal(
                        pcr.ifthen(
                            self.landmask, self.routedUSRTDS + self.wbRemainingUSRTDS
                        ),
                        pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                    ),
                ),
                self.routedUSRTDS,
            )
            self.routedUSRTDS = pcr.cover(
                wbUSRTDSTotal
                * self.cellArea
                / pcr.areatotal(
                    pcr.cover(self.cellArea, 0.0),
                    pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                ),
                wbUSRTDSTotal,
            )

            wbIrrTDSTotal = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    pcr.areatotal(
                        pcr.ifthen(
                            self.landmask, self.routedIrrTDS + self.wbRemainingIrrTDS
                        ),
                        pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                    ),
                ),
                self.routedIrrTDS,
            )
            self.routedIrrTDS = pcr.cover(
                wbIrrTDSTotal
                * self.cellArea
                / pcr.areatotal(
                    pcr.cover(self.cellArea, 0.0),
                    pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                ),
                wbIrrTDSTotal,
            )

        # organic (BOD averaged over the water body)
        wbBODTotal = pcr.cover(
            pcr.ifthen(
                pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                pcr.areatotal(
                    pcr.ifthen(self.landmask, self.routedBOD + self.wbRemainingBOD),
                    pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                ),
            ),
            self.routedBOD,
        )
        self.routedBOD = pcr.cover(
            wbBODTotal
            * self.cellArea
            / pcr.areatotal(
                pcr.cover(self.cellArea, 0.0),
                pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
            ),
            wbBODTotal,
        )

        if self.loadsPerSector:
            wbDomBODTotal = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    pcr.areatotal(
                        pcr.ifthen(
                            self.landmask, self.routedDomBOD + self.wbRemainingDomBOD
                        ),
                        pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                    ),
                ),
                self.routedDomBOD,
            )
            self.routedDomBOD = pcr.cover(
                wbDomBODTotal
                * self.cellArea
                / pcr.areatotal(
                    pcr.cover(self.cellArea, 0.0),
                    pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                ),
                wbDomBODTotal,
            )

            wbManBODTotal = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    pcr.areatotal(
                        pcr.ifthen(
                            self.landmask, self.routedManBOD + self.wbRemainingManBOD
                        ),
                        pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                    ),
                ),
                self.routedManBOD,
            )
            self.routedManBOD = pcr.cover(
                wbManBODTotal
                * self.cellArea
                / pcr.areatotal(
                    pcr.cover(self.cellArea, 0.0),
                    pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                ),
                wbManBODTotal,
            )

            wbUSRBODTotal = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    pcr.areatotal(
                        pcr.ifthen(
                            self.landmask, self.routedUSRBOD + self.wbRemainingUSRBOD
                        ),
                        pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                    ),
                ),
                self.routedUSRBOD,
            )
            self.routedUSRBOD = pcr.cover(
                wbUSRBODTotal
                * self.cellArea
                / pcr.areatotal(
                    pcr.cover(self.cellArea, 0.0),
                    pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                ),
                wbUSRBODTotal,
            )

            wbintLivBODTotal = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    pcr.areatotal(
                        pcr.ifthen(
                            self.landmask,
                            self.routedintLivBOD + self.wbRemainingintLivBOD,
                        ),
                        pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                    ),
                ),
                self.routedintLivBOD,
            )
            self.routedintLivBOD = pcr.cover(
                wbintLivBODTotal
                * self.cellArea
                / pcr.areatotal(
                    pcr.cover(self.cellArea, 0.0),
                    pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                ),
                wbintLivBODTotal,
            )

            wbextLivBODTotal = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    pcr.areatotal(
                        pcr.ifthen(
                            self.landmask,
                            self.routedextLivBOD + self.wbRemainingextLivBOD,
                        ),
                        pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                    ),
                ),
                self.routedextLivBOD,
            )
            self.routedextLivBOD = pcr.cover(
                wbextLivBODTotal
                * self.cellArea
                / pcr.areatotal(
                    pcr.cover(self.cellArea, 0.0),
                    pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                ),
                wbextLivBODTotal,
            )

        # pathogen (FC averaged over the water body)
        wbFCTotal = pcr.cover(
            pcr.ifthen(
                pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                pcr.areatotal(
                    pcr.ifthen(self.landmask, self.routedFC + self.wbRemainingFC),
                    pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                ),
            ),
            self.routedFC,
        )
        self.routedFC = pcr.cover(
            wbFCTotal
            * self.cellArea
            / pcr.areatotal(
                pcr.cover(self.cellArea, 0.0),
                pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
            ),
            wbFCTotal,
        )

        if self.loadsPerSector:
            wbDomFCTotal = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    pcr.areatotal(
                        pcr.ifthen(
                            self.landmask, self.routedDomFC + self.wbRemainingDomFC
                        ),
                        pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                    ),
                ),
                self.routedDomFC,
            )
            self.routedDomFC = pcr.cover(
                wbDomFCTotal
                * self.cellArea
                / pcr.areatotal(
                    pcr.cover(self.cellArea, 0.0),
                    pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                ),
                wbDomFCTotal,
            )

            wbManFCTotal = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    pcr.areatotal(
                        pcr.ifthen(
                            self.landmask, self.routedManFC + self.wbRemainingManFC
                        ),
                        pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                    ),
                ),
                self.routedManFC,
            )
            self.routedManFC = pcr.cover(
                wbManFCTotal
                * self.cellArea
                / pcr.areatotal(
                    pcr.cover(self.cellArea, 0.0),
                    pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                ),
                wbManFCTotal,
            )

            wbUSRFCTotal = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    pcr.areatotal(
                        pcr.ifthen(
                            self.landmask, self.routedUSRFC + self.wbRemainingUSRFC
                        ),
                        pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                    ),
                ),
                self.routedUSRFC,
            )
            self.routedUSRFC = pcr.cover(
                wbUSRFCTotal
                * self.cellArea
                / pcr.areatotal(
                    pcr.cover(self.cellArea, 0.0),
                    pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                ),
                wbUSRFCTotal,
            )

            wbintLivFCTotal = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    pcr.areatotal(
                        pcr.ifthen(
                            self.landmask,
                            self.routedintLivFC + self.wbRemainingintLivFC,
                        ),
                        pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                    ),
                ),
                self.routedintLivFC,
            )
            self.routedintLivFC = pcr.cover(
                wbintLivFCTotal
                * self.cellArea
                / pcr.areatotal(
                    pcr.cover(self.cellArea, 0.0),
                    pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                ),
                wbintLivFCTotal,
            )

            wbextLivFCTotal = pcr.cover(
                pcr.ifthen(
                    pcr.scalar(self.WaterBodies.waterBodyIds) > 0.0,
                    pcr.areatotal(
                        pcr.ifthen(
                            self.landmask,
                            self.routedextLivFC + self.wbRemainingextLivFC,
                        ),
                        pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                    ),
                ),
                self.routedextLivFC,
            )
            self.routedextLivFC = pcr.cover(
                wbextLivFCTotal
                * self.cellArea
                / pcr.areatotal(
                    pcr.cover(self.cellArea, 0.0),
                    pcr.ifthen(self.landmask, self.WaterBodies.waterBodyIds),
                ),
                wbextLivFCTotal,
            )

    def estimate_concentrations(self):
        """
        DynQual: concentrations from the routed loads, channelStorage (m3) and
        disChanWaterBody (m3/s, threshold WQ_discharge_threshold); reads water
        quantities only. With QUAlloc, salinity, organic and pathogen are handed to
        QUAlloc on the next day (landSurface.py).
        """

        # channel storage with discharge threshold, to estimate in-stream concentrations
        self.channelStorage_Qthres = pcr.ifthenelse(
            pcr.cover(self.disChanWaterBody, vos.MV) > self.WQ_discharge_threshold,
            self.channelStorage,
            vos.MV,
        )

        # total dissolved solids concentration, salinity indicator (mg/L)
        self.salinity = pcr.ifthenelse(
            self.channelStorage_Qthres != vos.MV,
            (self.routedTDS / self.channelStorage_Qthres) + self.backgroundSalinity,
            vos.MV,
        )

        # biochemical oxygen demand concentration, organic indicator (mg/L)
        self.organic = pcr.ifthenelse(
            self.channelStorage_Qthres != vos.MV,
            self.routedBOD / self.channelStorage_Qthres,
            vos.MV,
        )

        # dissolved oxygen concentration (Streeter-Phelps equation)
        self.k1 = self.BODdecay_temperature * self.organic
        # oxygen saturation (mg/L)
        self.DOsat = (1 - 0.0001148 * self.elevation) * pcr.exp(
            -139.34411
            + (157570.1) / (self.waterTemp)
            - (66423080.0) / (self.waterTemp**2)
            + (12438000000.0) / (self.waterTemp**3)
            - (862194900000.0) / (self.waterTemp**4)
        )
        # velocity assuming a rectangular channel (m/s)
        self.velocity = self.avgDischarge / (self.yMean * self.wMean)
        # reaeration rate (day-1; O'Connor and Dobbins, 1958)
        self.k2 = 3.93 * (self.velocity**0.5) / (self.yMean**1.5)
        self.k2 = pcr.ifthenelse(
            self.k2 > 1.5, 1.5, pcr.ifthenelse(self.k2 < 0.4, 0.4, self.k2)
        )
        self.dissolved_oxygen = pcr.ifthenelse(
            self.dissolved_oxygen
            - self.k1
            + self.k2 * (self.DOsat - self.dissolved_oxygen)
            < 0.0,
            0.0,
            self.dissolved_oxygen
            - self.k1
            + self.k2 * (self.DOsat - self.dissolved_oxygen),
        )
        # dissolved oxygen concentration (mg/L)
        self.dissolved_oxygen = pcr.ifthenelse(
            self.channelStorage_Qthres != vos.MV, self.dissolved_oxygen, vos.MV
        )

        # fecal coliform concentration, pathogen indicator (cfu/100mL)
        self.pathogen = pcr.ifthenelse(
            self.channelStorage_Qthres != vos.MV,
            self.routedFC * 100.0 / self.channelStorage_Qthres,
            vos.MV,
        )
