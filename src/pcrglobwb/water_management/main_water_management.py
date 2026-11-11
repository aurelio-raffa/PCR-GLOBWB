import logging

import pcraster as pcr

from pcrglobwb.common import virtualOS as vos

logger = logging.getLogger(__name__)

# ANNOTATION OVERVIEW (comments and docstrings only; the code is unchanged).
#
# This module allocates the gross water demand of every cell to four sources, in this
# order: desalinated water, surface water, renewable (non-fossil) groundwater and
# non-renewable (fossil) groundwater. It is used only when QUAlloc is off
# ([waterManagementOptions] using_qualloc = False, as in
# configuration/run_with_args/pcrglobwb/pcrglobwb_5arcminutes.ini:142). With
# using_qualloc = True (online_pcrglobwb_qualloc_5arcminutes.ini:159) this class is
# never built; QUAlloc (src/qualloc/) allocates instead, see the QUAlloc branch of
# landSurface.LandSurface.update.
#
# Units and spatial support used throughout:
# - inside this module every demand, availability, abstraction and allocation is a
#   volume per daily time step (m3, i.e. m3/day), per cell (5 arcmin in the RAWS runs);
# - the results handed to other modules are divided by the cell area, so they are
#   water heights over the WHOLE cell area per day (m/day); multiply by cellArea (m2)
#   to obtain m3/day again;
# - "abstraction" (or "withdrawal") is located at the SOURCE cell, where the water is
#   physically taken; "allocation" is located at the DEMAND cell, where the water is
#   used. Without allocation zones both are the same cell and the same value.
#
# Link to lakes and reservoirs: the surface water available for abstraction is
# routing.readAvlChannelStorage (m3 per cell), computed at the end of the PREVIOUS
# day's routing from routing.channelStorage. At lake and reservoir cells the channel
# storage at the end of a day is the water-body storage spread over the body's cells
# in proportion to cell area (routing.Routing.return_water_body_storage_to_channel), so
# lake and reservoir storage is an abstraction source like any river cell, with no
# lower limit tied to the reservoir operation (no minResvrFrac). The surface water
# abstraction computed here (actSurfaceWaterAbstract, m/day at the source cell) is
# subtracted from routing.channelStorage on the SAME day, before the reservoir release
# rule runs (routing.Routing.simple_update), so it lowers both the storage the rule
# sees and the inflow residual of the water body
# (waterBodies.WaterBodies.moveFromChannelToWaterBody). Nothing in this module knows
# which cells belong to a lake or reservoir.


class WaterManagement(object):
    """Allocate gross sectoral water demand to desalinated water, surface water and
    groundwater (standard PCR-GLOBWB allocation, used when using_qualloc is False).

    Built once in landSurface.LandSurface.__init__ and updated once per daily time step
    from landSurface.LandSurface.update, after the water demand of the day is known and
    before the soil water balance, groundwater and routing of the same day
    (pcrglobwb.PCRGlobWB.update).

    Attributes read by other modules after update() (per cell, m/day over the whole
    cell area unless stated otherwise):

    - actSurfaceWaterAbstract: surface water taken at each source cell; copied to
      landSurface.actSurfaceWaterAbstract and subtracted from routing.channelStorage
      (lake and reservoir cells included) on the same day.
    - allocSurfaceWaterAbstract: surface water delivered to each demand cell.
    - desalinationAbstraction, desalinationAllocation: desalinated water.
    - nonFossilGroundwaterAbs, allocNonFossilGroundwater: renewable groundwater taken /
      delivered; the abstraction reaches groundwater.Groundwater.update_storage.
    - fossilGroundwaterAbstr, fossilGroundwaterAlloc: non-renewable groundwater.
    - reducedCapRise: renewable groundwater abstraction (m/day), used by landCover to
      reduce capillary rise.
    - satisfied_gross_sectoral_water_demands[sector]: water delivered to each sector
      (m3/day, per demand cell), used for return flows and reported withdrawals.
    """

    def __init__(self, iniItems, landmask):
        """Read the options of [waterManagementOptions] and the static input maps.

        Keys read here (an absent key gets the default shown). Keys with other names
        are ignored: the RAWS 5-arcmin ini files set allocationSegmentsForGroundSurface
        Water, treshold_to_maximize_irrigation_surface_water and treshold_to_minimize_
        fossil_groundwater_irrigation (old or misspelled names), which are not read
        here (re-check issue N5).

        - limitAbstraction (default "False"): no fossil groundwater use (see M7).
        - pumpingCapacityNC (default "None"): regional annual pumping limit (netCDF).
        - prioritizeLocalSourceToMeetWaterDemand (default True; see M1).
        - desalinationWater: monthly desalinated water supply (m/day); "None" or
          "False" switches it off. There is no default: the key must be present.
        - allocationSegmentsForDesalinatedWaterSource,
          allocationSegmentsForSurfaceWaterSource,
          allocationSegmentsForGroundwaterSource: nominal zone maps; absent, "None" or
          "False" means local supply only (each cell uses only its own source).
        - irrigationSurfaceWaterAbstractionFractionData and ...DataQuality: predefined
          surface-water share for irrigation (Siebert et al., 2010) and its quality.
        - threshold_to_maximize_irrigation_surface_water,
          threshold_to_minimize_fossil_groundwater_irrigation (default "1.0", which
          in effect disables both rules; see M9).
        - maximumDailyGroundwaterAbstraction (default 0.050 m/day),
          maximumDailyFossilGroundwaterAbstraction (default 0.020 m/day).
        - maximumNonIrrigationSurfaceWaterAbstractionFractionData (default 1.0),
          predefinedNonIrrigationSurfaceWaterAbstractionFractionData (default None).
        - noParameterExtrapolation (used only for the zone maps).

        Also reads [routingOptions] cellAreaMap (m2). Defaults are written back into
        iniItems.waterManagementOptions, which later modules (groundwater.py) read.

        Arguments: iniItems is the parsed configuration (configuration.Configuration);
        landmask is the boolean model mask.
        """
        object.__init__(self)

        self.iniItems = iniItems

        self.tmpDir = iniItems.tmpDir
        self.inputDir = iniItems.globalOptions["inputDir"]
        self.landmask = landmask

        # groundwater abstraction options: if limitAbstraction, only renewable groundwater is used
        # Neither RAWS 5-arcmin ini sets limitAbstraction, so it is False. The default
        # is written back into iniItems so that groundwater.Groundwater.__init__ (built
        # after this object in pcrglobwb.PCRGlobWB.createSubmodels) finds the key.
        if "limitAbstraction" not in list(iniItems.waterManagementOptions.keys()):
            iniItems.waterManagementOptions["limitAbstraction"] = False
        self.limitAbstraction = False
        if iniItems.waterManagementOptions["limitAbstraction"] == "True":
            self.limitAbstraction = True
        # CAUTION (M7): with limitAbstraction = True, abstraction_and_allocation_from_
        # groundwater uses volFossilGroundwaterAllocation and volFossilGroundwater
        # Abstraction without defining them (UnboundLocalError by code reading), so this
        # option is not usable in this revision. Not set in the RAWS inis.

        # option for the groundwater pumping capacity (limitRegionalAnnualGroundwaterAbstraction)
        # CAUTION (N5): both RAWS 5-arcmin inis set pumpingCapacityNC under
        # [groundwaterOptions] (pcrglobwb_5arcminutes.ini:376), not under
        # [waterManagementOptions], so the default "None" below applies and the regional
        # pumping limit is off. Only groundwater use changes directly; the surface-water
        # request of the same day is fixed before groundwater is allocated, and later
        # days are affected only through the groundwater allocation averages.
        if "pumpingCapacityNC" not in list(iniItems.waterManagementOptions.keys()):
            msg = 'The "pumpingCapacityNC" (annual groundwater pumping capacity limit netcdf file)'
            msg += "is not defined in the configuration file. "
            msg += "We assume no annual groundwater pumping limit used in this run. "
            msg += "It may result too high groundwater abstraction."
            logger.warning(msg)
            iniItems.waterManagementOptions["pumpingCapacityNC"] = "None"

        # option to limit regional groundwater abstraction
        if iniItems.waterManagementOptions["pumpingCapacityNC"] != "None":
            logger.info("Limit for annual regional groundwater abstraction is used.")
            self.limitRegionalAnnualGroundwaterAbstraction = True
            self.pumpingCapacityNC = vos.getFullPath(
                iniItems.waterManagementOptions["pumpingCapacityNC"],
                self.inputDir,
                False,
            )
        else:
            logger.warning(
                "NO LIMIT for regional groundwater (annual) pumping. It may result too high groundwater abstraction."
            )
            self.limitRegionalAnnualGroundwaterAbstraction = False

        # option to prioritize local sources before abstracting water from neighbouring cells
        # (used only by waterAbstractionAndAllocation, i.e. only where allocation zones
        # are active; with local-only supply it has no effect)
        self.prioritizeLocalSourceToMeetWaterDemand = True
        # CAUTION (M1): the condition below tests the value of pumpingCapacityNC, not of
        # prioritizeLocalSourceToMeetWaterDemand. Setting
        # prioritizeLocalSourceToMeetWaterDemand = False therefore has no effect unless
        # pumpingCapacityNC is literally "False"; in practice the option is always True.
        # Neither RAWS ini sets the key. Consequence where zones are active: a reservoir
        # cell first supplies the demand located on its own cells, and only the rest of
        # its availability is shared pro rata within the zone.
        if (
            "prioritizeLocalSourceToMeetWaterDemand"
            in list(iniItems.waterManagementOptions.keys())
            and iniItems.waterManagementOptions["pumpingCapacityNC"] == "False"
        ):
            self.prioritizeLocalSourceToMeetWaterDemand = False
            logger.info(
                "The option prioritizeLocalSourceToMeetWaterDemand is set to 'False'."
            )

        # option to prioritize surface water (currently always False)
        # (no configuration key; if True, the whole remaining demand would be requested
        # from surface water in abstraction_and_allocation_from_surface_water)
        self.surfaceWaterPiority = False

        # cell area (m2)
        # Same map as routing.cellArea ([routingOptions] cellAreaMap); used to convert
        # between m/day (height over the cell) and m3/day (volume per cell).
        cellArea = vos.readPCRmapClone(
            iniItems.routingOptions["cellAreaMap"],
            self.tmpDir,
            self.inputDir,
        )
        self.cellArea = pcr.ifthen(self.landmask, cellArea)

        # option to use desalinated water
        # The file holds a monthly supply capacity (m/day over the cell), read in
        # abstraction_and_allocation_from_desalination. pcrglobwb_5arcminutes.ini:145
        # sets "None" (off); desalinated water is allocated before surface water, so
        # where it is on it reduces the demand that can fall on rivers, lakes and
        # reservoirs.
        self.includeDesalination = False
        if iniItems.waterManagementOptions["desalinationWater"] not in [
            "None",
            "False",
        ]:
            logger.info("Monthly desalination water use is included.")
            self.includeDesalination = True
            self.desalinationWaterFile = vos.getFullPath(
                iniItems.waterManagementOptions["desalinationWater"], self.inputDir
            )
        else:
            logger.info("Monthly desalination water is NOT included.")

        # allocation zones for desalinated water, surface water and groundwater; if not defined,
        # only the local cell water availability is considered
        # For each source name below the loop sets three attributes:
        #   using_<source> (bool), <source> (nominal zone map, or None) and
        #   <source>Areas (zone area in m2, repeated on every cell of the zone, or
        #   None).
        # An allocation zone is a group of cells that share their available water: the
        # demand of any cell in the zone can be met from any other cell of the zone
        # (waterAbstractionAndAllocation). Spatial support of the result: per cell, but
        # pooled per zone.
        # CAUTION (N5): pcrglobwb_5arcminutes.ini:149 sets only the old key
        # allocationSegmentsForGroundSurfaceWater (60 arcmin zones), which is read
        # nowhere in src/pcrglobwb, and pcrglobwb_5arcminutes.ini:379 sets
        # allocationSegmentsForGroundwater under [groundwaterOptions] (also not read).
        # So with the standard RAWS configuration all three flags stay False and every
        # source is LOCAL ONLY: a reservoir cell can only supply the demand located on
        # that same cell (abstraction_and_allocation_from_surface_water, local branch).
        # Not confirmed by a run. The QUAlloc ini sets the new names for desalinated
        # water and groundwater (online_pcrglobwb_qualloc_5arcminutes.ini:174-175), but
        # there this class is not used.
        self.using_allocationSegmentsForDesalinatedWaterSource = False
        self.using_allocationSegmentsForSurfaceWaterSource = False
        self.using_allocationSegmentsForGroundwaterSource = False

        sources = [
            "allocationSegmentsForDesalinatedWaterSource",
            "allocationSegmentsForSurfaceWaterSource",
            "allocationSegmentsForGroundwaterSource",
        ]

        for source in sources:
            if (
                source in iniItems.waterManagementOptions.keys()
                and iniItems.waterManagementOptions[source] not in ["False", "None"]
            ):
                vars(self)["using_" + source] = True
                vars(self)[source], vars(self)[source + "Areas"] = (
                    self.get_allocation_zone(iniItems.waterManagementOptions[source])
                )
            else:
                vars(self)[source], vars(self)[source + "Areas"] = None, None

        # water sources and sectors to evaluate
        self.source_names = [
            "desalinated_water",
            "surface_water",
            "renewable_groundwater",
            "nonrenewable_groundwater",
        ]
        self.sector_names = [
            "domestic",
            "industry",
            "manufacture",
            "thermoelectric",
            "livestock",
            "irrigation",
        ]

        # allocated_withdrawal_per_sector[source_name][sector_name]: water taken from source_name for
        # sector_name, from the point of view of the source cells
        # (m3/day per source cell; diagnostic only: no other module reads it, see M13 in
        # allocate_withdrawal_to_each_sector for its accuracy when zones are active)
        self.allocated_withdrawal_per_sector = {}
        for source_name in self.source_names:
            self.allocated_withdrawal_per_sector[source_name] = {}
            for sector_name in self.sector_names:
                self.allocated_withdrawal_per_sector[source_name][sector_name] = (
                    pcr.scalar(0.0)
                )

        # allocated_demand_per_sector[source_name][sector_name]: water given to sector_name from
        # source_name, from the point of view of the demand cells
        # (m3/day per demand cell; reset by each source method every time step)
        # EXTENSION POINT: allocated_demand_per_sector["surface_water"][sector] is the
        # surface water delivered to each sector at its demand cell (m3/day). It is the
        # natural per-sector "served demand" signal; aggregate it over the command area
        # of a reservoir (e.g. pcr.areatotal with a zone map, or along the drainage
        # network) in routing before calling WaterBodies.update. It is available only on
        # the standard path (using_qualloc = False) as
        # landSurface.water_management.allocated_demand_per_sector.
        self.allocated_demand_per_sector = {}
        for source_name in self.source_names:
            self.allocated_demand_per_sector[source_name] = {}
            for sector_name in self.sector_names:
                self.allocated_demand_per_sector[source_name][sector_name] = pcr.scalar(
                    0.0
                )

        # met_demand_per_sector[sector_name]: demand met per sector, from the point of view of the
        # demand cells (the sum of allocated_demand_per_sector)
        self.met_demand_per_sector = {}
        for sector_name in self.sector_names:
            self.met_demand_per_sector[sector_name] = pcr.scalar(0.0)

        # predefined surface water fraction to satisfy the irrigation and livestock demand
        # (dimensionless, 0-1, per cell; set in both RAWS 5-arcmin inis,
        # pcrglobwb_5arcminutes.ini:154,156). Only cells with data quality <= 5 keep a
        # value. It is blended with the de Graaf et al. (2014) estimate in
        # partitioningGroundSurfaceAbstractionForIrrigation.
        self.swAbstractionFractionData = None
        self.swAbstractionFractionDataQuality = None
        if "irrigationSurfaceWaterAbstractionFractionData" in list(
            iniItems.waterManagementOptions.keys()
        ) and "irrigationSurfaceWaterAbstractionFractionDataQuality" in list(
            iniItems.waterManagementOptions.keys()
        ):

            if iniItems.waterManagementOptions[
                "irrigationSurfaceWaterAbstractionFractionData"
            ] not in ["None", "False"] or iniItems.waterManagementOptions[
                "irrigationSurfaceWaterAbstractionFractionDataQuality"
            ] not in [
                "None",
                "False",
            ]:

                logger.info(
                    "Using/incorporating the predefined surface water source of Siebert et al. (2010) for satisfying irrigation and livestock demand."
                )
                self.swAbstractionFractionData = pcr.cover(
                    vos.readPCRmapClone(
                        iniItems.waterManagementOptions[
                            "irrigationSurfaceWaterAbstractionFractionData"
                        ],
                        self.tmpDir,
                        self.inputDir,
                    ),
                    0.0,
                )
                self.swAbstractionFractionData = pcr.ifthen(
                    self.swAbstractionFractionData >= 0.0,
                    self.swAbstractionFractionData,
                )
                self.swAbstractionFractionDataQuality = pcr.cover(
                    vos.readPCRmapClone(
                        iniItems.waterManagementOptions[
                            "irrigationSurfaceWaterAbstractionFractionDataQuality"
                        ],
                        self.tmpDir,
                        self.inputDir,
                    ),
                    0.0,
                )
                # ignore values with a quality above 5 (very bad); only cells with quality <= 5 keep values
                self.swAbstractionFractionData = pcr.ifthen(
                    self.swAbstractionFractionDataQuality <= 5.0,
                    self.swAbstractionFractionData,
                )

        # thresholds (fractions) defining the preference for the irrigation water source:
        # threshold_to_maximize_irrigation_surface_water
        # CAUTION (N5): both RAWS 5-arcmin inis spell these keys "treshold_..."
        # (pcrglobwb_5arcminutes.ini:160,162, values 0.50 and 0.70), so the keys read
        # here are absent, a warning is logged and both thresholds default to 1.0.
        # Effect on surface water: the irrigation demand is NOT moved entirely to
        # surface water in cells whose surface-water fraction is between 0.50 and 1.0
        # (see abstraction_and_allocation_from_surface_water), so less irrigation demand
        # falls on rivers, lakes and reservoirs than the configuration intends. Values
        # are dimensionless fractions; a map or a constant is accepted
        # (readPCRmapClone).
        if "threshold_to_maximize_irrigation_surface_water" not in list(
            iniItems.waterManagementOptions.keys()
        ):
            msg = 'The option "threshold_to_maximize_irrigation_surface_water" is not defined in the "waterManagementOptions" of the configuration file. '
            msg += 'This run assumes "1.0" for this option.'
            logger.warning(msg)
            iniItems.waterManagementOptions[
                "threshold_to_maximize_irrigation_surface_water"
            ] = "1.0"
            # the default of 1.0 disables this threshold
        self.threshold_to_maximize_irrigation_surface_water = vos.readPCRmapClone(
            iniItems.waterManagementOptions[
                "threshold_to_maximize_irrigation_surface_water"
            ],
            self.tmpDir,
            self.inputDir,
        )

        # threshold_to_minimize_fossil_groundwater_irrigation
        # (groundwater only: above this surface-water fraction no fossil groundwater is
        # used for irrigation and livestock; it does not change surface-water
        # abstraction)
        if "threshold_to_minimize_fossil_groundwater_irrigation" not in list(
            iniItems.waterManagementOptions.keys()
        ):
            msg = 'The option "threshold_to_minimize_fossil_groundwater_irrigation" is not defined in the "waterManagementOptions" of configuration file. '
            msg += 'This run assumes "1.0" for this option.'
            logger.warning(msg)
            iniItems.waterManagementOptions[
                "threshold_to_minimize_fossil_groundwater_irrigation"
            ] = "1.0"
            # the default of 1.0 disables this threshold
        self.threshold_to_minimize_fossil_groundwater_irrigation = vos.readPCRmapClone(
            iniItems.waterManagementOptions[
                "threshold_to_minimize_fossil_groundwater_irrigation"
            ],
            self.tmpDir,
            self.inputDir,
        )

        # maximum daily rate of groundwater abstraction (m/day)
        # (over the whole cell area; caps renewable plus fossil groundwater abstraction;
        # not set in pcrglobwb_5arcminutes.ini, so 0.050 m/day applies. Demand that
        # groundwater cannot meet is NOT passed back to surface water: it stays unmet)
        if "maximumDailyGroundwaterAbstraction" not in list(
            iniItems.waterManagementOptions.keys()
        ):
            msg = 'The option "maximumDailyGroundwaterAbstraction" is not defined in the "waterManagementOptions" of the configuration file. '
            msg += 'This run assumes "0.050 m/day" for this option.'
            logger.warning(msg)
            iniItems.waterManagementOptions["maximumDailyGroundwaterAbstraction"] = (
                "0.050"
            )
        self.maximumDailyGroundwaterAbstraction = vos.readPCRmapClone(
            iniItems.waterManagementOptions["maximumDailyGroundwaterAbstraction"],
            self.tmpDir,
            self.inputDir,
        )

        # maximum daily rate of fossil groundwater abstraction (m/day)
        if "maximumDailyFossilGroundwaterAbstraction" not in list(
            iniItems.waterManagementOptions.keys()
        ):
            msg = 'The option "maximumDailyFossilGroundwaterAbstraction" is not defined in the "waterManagementOptions" of the configuration file. '
            msg += 'This run assumes "0.020 m/day" for this option.'
            logger.warning(msg)
            iniItems.waterManagementOptions[
                "maximumDailyFossilGroundwaterAbstraction"
            ] = "0.020"
        self.maximumDailyFossilGroundwaterAbstraction = vos.readPCRmapClone(
            iniItems.waterManagementOptions["maximumDailyFossilGroundwaterAbstraction"],
            self.tmpDir,
            self.inputDir,
        )

        # maximum predefined surface water fraction to satisfy the industrial and domestic demand
        # (default: the maximum)
        # (dimensionless, per cell; caps the surface-water share of the non-irrigation
        # sectors; it is used only when swAbstractionFractionData is set, see
        # partitioningGroundSurfaceAbstraction. pcrglobwb_5arcminutes.ini:165 sets a
        # map)
        self.maximumNonIrrigationSurfaceWaterAbstractionFractionData = pcr.scalar(1.0)
        if "maximumNonIrrigationSurfaceWaterAbstractionFractionData" in list(
            iniItems.waterManagementOptions.keys()
        ):
            # CAUTION (M2): the test below, (x != "None" or x != "False"), is always
            # True, so a value of "None" or "False" is passed to readPCRmapClone as a
            # file name. Harmless for the RAWS inis, which give a file.
            if (
                iniItems.waterManagementOptions[
                    "maximumNonIrrigationSurfaceWaterAbstractionFractionData"
                ]
                != "None"
                or iniItems.waterManagementOptions[
                    "maximumNonIrrigationSurfaceWaterAbstractionFractionData"
                ]
                != "False"
            ):

                logger.info(
                    "Set the maximum fraction for predefined surface water source for satisfying domestic and industrial demand."
                )
                self.maximumNonIrrigationSurfaceWaterAbstractionFractionData = pcr.min(
                    1.0,
                    pcr.cover(
                        vos.readPCRmapClone(
                            iniItems.waterManagementOptions[
                                "maximumNonIrrigationSurfaceWaterAbstractionFractionData"
                            ],
                            self.tmpDir,
                            self.inputDir,
                        ),
                        1.0,
                    ),
                )

        # predefined surface water fraction to satisfy the industrial and domestic demand
        # (dimensionless, per cell; not set in the RAWS inis, so it stays None. The same
        # always-True test as above applies, CAUTION (M2))
        self.predefinedNonIrrigationSurfaceWaterAbstractionFractionData = None
        if "predefinedNonIrrigationSurfaceWaterAbstractionFractionData" in list(
            iniItems.waterManagementOptions.keys()
        ) and (
            iniItems.waterManagementOptions[
                "predefinedNonIrrigationSurfaceWaterAbstractionFractionData"
            ]
            != "None"
            or iniItems.waterManagementOptions[
                "predefinedNonIrrigationSurfaceWaterAbstractionFractionData"
            ]
            != "False"
        ):

            logger.info(
                "Set the predefined fraction of surface water source for satisfying domestic and industrial demand."
            )
            self.predefinedNonIrrigationSurfaceWaterAbstractionFractionData = pcr.min(
                1.0,
                pcr.cover(
                    vos.readPCRmapClone(
                        iniItems.waterManagementOptions[
                            "predefinedNonIrrigationSurfaceWaterAbstractionFractionData"
                        ],
                        self.tmpDir,
                        self.inputDir,
                    ),
                    1.0,
                ),
            )
            self.predefinedNonIrrigationSurfaceWaterAbstractionFractionData = pcr.max(
                0.0,
                pcr.min(
                    self.maximumNonIrrigationSurfaceWaterAbstractionFractionData,
                    self.predefinedNonIrrigationSurfaceWaterAbstractionFractionData,
                ),
            )

    def get_allocation_zone(self, zonal_map_file_name):
        """Read an allocation-zone map and return (zones, zone areas).

        zonal_map_file_name: nominal map (PCRaster or netCDF) relative to inputDir.
        Returns allocSegments (nominal, per cell, within the landmask; every cell
        belongs to exactly one zone) and segmentAreas (m2, total cell area of the zone,
        repeated on every cell of the zone). Zones are made contiguous with pcr.clump,
        so one input zone split in two pieces becomes two zones. Cells without a zone
        become single-cell zones. Unless noParameterExtrapolation = "True", missing zone
        values are first filled with the majority of a 0.5 degree window.
        """
        allocSegments = vos.readPCRmapClone(
            zonal_map_file_name,
            self.tmpDir,
            self.inputDir,
            isLddMap=False,
            cover=None,
            isNomMap=True,
        )
        allocSegments = pcr.ifthen(self.landmask, allocSegments)
        allocSegments = pcr.clump(allocSegments)

        extrapolate = True
        if (
            "noParameterExtrapolation" in self.iniItems.waterManagementOptions.keys()
            and self.iniItems.waterManagementOptions["noParameterExtrapolation"]
            == "True"
        ):
            extrapolate = False

        if extrapolate:
            # extrapolate to half-degree resolution
            allocSegments = pcr.cover(
                allocSegments, pcr.windowmajority(allocSegments, 0.5)
            )

        allocSegments = pcr.ifthen(self.landmask, allocSegments)

        # clump and cover the rest with cell ids
        allocSegments = pcr.clump(allocSegments)
        cell_ids = (
            pcr.mapmaximum(pcr.scalar(allocSegments))
            + pcr.scalar(100.0)
            + pcr.uniqueid(pcr.boolean(1.0))
        )
        allocSegments = pcr.cover(allocSegments, pcr.nominal(cell_ids))
        allocSegments = pcr.clump(allocSegments)
        allocSegments = pcr.ifthen(self.landmask, allocSegments)

        # zone area (m2)
        segmentAreas = pcr.areatotal(pcr.cover(self.cellArea, 0.0), allocSegments)
        segmentAreas = pcr.ifthen(self.landmask, segmentAreas)

        return allocSegments, segmentAreas

    def waterAbstractionAndAllocation(
        self,
        water_demand_volume,
        available_water_volume,
        allocation_zones,
        zone_area=None,
        high_volume_threshold=None,
        debug_water_balance=True,
        extra_info_for_water_balance_reporting="",
        landmask=None,
        ignore_small_values=False,
        prioritizing_local_source=True,
    ):
        """Abstract water from the source cells of each zone and allocate it to the
        demand cells of the same zone (pro rata on both sides).

        Arguments (all maps per cell):
        - water_demand_volume (m3/day): demand to be met from this source, at the
          demand cell.
        - available_water_volume (m3/day): water that may be taken, at the source cell
          (for surface water: routing.readAvlChannelStorage, which at lake and reservoir
          cells is a share of the lake or reservoir storage).
        - allocation_zones (nominal) and zone_area (m2, only for the balance check).
        - prioritizing_local_source: meet each cell's demand from its own source first.
        - high_volume_threshold (m3) and ignore_small_values: numerical options; all
          calls in this module pass None and False.

        Equations, with D_c demand and A_c availability of cell c, and z the zone of c:
        - local step (if prioritizing_local_source):
          L_c = max(0, min(D_c, A_c)); D'_c = D_c - L_c; A'_c = A_c - L_c;
          otherwise L_c = 0, D'_c = D_c, A'_c = A_c.
        - zonal totals: D_z = sum_{c in z} D'_c; A'_c := min(A'_c, D_z);
          A_z = sum_{c in z} A'_c; X_z = min(A_z, D_z).
        - abstraction at source cells: X_c = min(X_z * A'_c / A_z, A'_c), i.e. every
          source cell of the zone gives the same fraction X_z / A_z of its availability;
          a lake or reservoir cell with large availability therefore supplies most of
          the zone's demand.
        - allocation at demand cells: Y_c = min(X_z * D'_c / D_z, D'_c).

        Returns (cellAbstraction, cellAllocation, zoneAbstraction), all in m3/day:
        cellAbstraction = X_c + L_c (per source cell), cellAllocation = Y_c + L_c (per
        demand cell), zoneAbstraction = X_z (zonal part only, WITHOUT the local part
        L_c, repeated on every cell of the zone). Divisions by zero return 0
        (vos.getValDivZero with vos.smallNumber).
        """

        logger.debug("Allocation of abstraction.")

        if landmask is not None:
            water_demand_volume = pcr.ifthen(
                landmask, pcr.cover(water_demand_volume, 0.0)
            )
            available_water_volume = pcr.ifthen(
                landmask, pcr.cover(available_water_volume, 0.0)
            )
            allocation_zones = pcr.ifthen(landmask, allocation_zones)

        # satisfy the demand with local sources
        localAllocation = pcr.scalar(0.0)
        localAbstraction = pcr.scalar(0.0)
        cellVolDemand = pcr.max(0.0, water_demand_volume)
        cellAvlWater = pcr.max(0.0, available_water_volume)
        if prioritizing_local_source:
            logger.debug(
                "Allocation of abstraction - first, satisfy demand with local source."
            )

            # demand volume per cell (m3)
            if landmask is not None:
                cellVolDemand = pcr.ifthen(landmask, pcr.cover(cellVolDemand, 0.0))

            # available water volume per cell
            if landmask is not None:
                cellAvlWater = pcr.ifthen(landmask, pcr.cover(cellAvlWater, 0.0))

            # first satisfy the demand with local sources
            # L_c = min(D_c, A_c) (m3): local abstraction equals local allocation
            localAllocation = pcr.max(0.0, pcr.min(cellVolDemand, cellAvlWater))
            localAbstraction = localAllocation * 1.0

        logger.debug(
            "Allocation of abstraction - satisfy demand with neighbour sources."
        )

        # remaining demand and available water
        # D'_c = D_c - L_c and A'_c = A_c - L_c (m3)
        cellVolDemand = pcr.max(0.0, cellVolDemand - localAllocation)
        cellAvlWater = pcr.max(0.0, cellAvlWater - localAbstraction)

        # ignore small values of water availability
        # CAUTION (M18): available_water_volume is not used after this point, so this
        # rounding has no effect on the result (cellAvlWater was already computed). All
        # calls pass ignore_small_values=False.
        if ignore_small_values:
            available_water_volume = pcr.max(0.0, pcr.rounddown(available_water_volume))

        # demand volume per cell (m3)
        cellVolDemand = pcr.max(0.0, cellVolDemand)
        if landmask is not None:
            cellVolDemand = pcr.ifthen(landmask, pcr.cover(cellVolDemand, 0.0))

        # total demand volume per zone (m3)
        # D_z, repeated on every cell of the zone
        zoneVolDemand = pcr.areatotal(cellVolDemand, allocation_zones)

        # avoid very high values of available water
        # A'_c := min(A'_c, D_z): no single source cell counts for more than the zone's
        # whole remaining demand (this changes the pro-rata weights below)
        cellAvlWater = pcr.min(cellAvlWater, zoneVolDemand)

        # available water volume per cell
        cellAvlWater = pcr.max(0.0, cellAvlWater)
        if landmask is not None:
            cellAvlWater = pcr.ifthen(landmask, pcr.cover(cellAvlWater, 0.0))

        # total available water volume per zone (m3)
        zoneAvlWater = pcr.areatotal(cellAvlWater, allocation_zones)

        # total actual abstraction volume per zone (m3), limited to the available water
        zoneAbstraction = pcr.min(zoneAvlWater, zoneVolDemand)

        # actual abstraction volume per cell (m3)
        # X_c = X_z * A'_c / A_z, at most A'_c: each source cell of the zone gives the
        # same fraction of its availability, so a lake or reservoir cell (large A'_c)
        # carries most of the zone's surface-water abstraction
        cellAbstraction = (
            vos.getValDivZero(cellAvlWater, zoneAvlWater, vos.smallNumber)
            * zoneAbstraction
        )
        cellAbstraction = pcr.min(cellAbstraction, cellAvlWater)

        # minimize numerical errors
        if high_volume_threshold is not None:
            # mask: 0 for small volumes, 1 for large volumes (e.g. lakes and reservoirs)
            mask = pcr.cover(
                pcr.ifthen(cellAbstraction > high_volume_threshold, pcr.boolean(1)),
                pcr.boolean(0),
            )
            zoneAbstraction = pcr.areatotal(
                pcr.ifthenelse(mask, 0.0, cellAbstraction), allocation_zones
            )
            zoneAbstraction += pcr.areatotal(
                pcr.ifthenelse(mask, cellAbstraction, 0.0), allocation_zones
            )

        # water allocated to meet the demand (m3)
        # Y_c = X_z * D'_c / D_z, at most D'_c (pro rata to the remaining demand)
        cellAllocation = (
            vos.getValDivZero(cellVolDemand, zoneVolDemand, vos.smallNumber)
            * zoneAbstraction
        )
        cellAllocation = pcr.min(cellAllocation, cellVolDemand)

        # add the local abstraction and allocation
        # (zoneAbstraction returned below keeps the zonal part X_z only)
        cellAbstraction = cellAbstraction + localAbstraction
        cellAllocation = cellAllocation + localAllocation

        if debug_water_balance and zone_area is not None:
            vos.waterBalanceCheck(
                [
                    pcr.cover(
                        pcr.areatotal(cellAbstraction, allocation_zones) / zone_area,
                        0.0,
                    )
                ],
                [
                    pcr.cover(
                        pcr.areatotal(cellAllocation, allocation_zones) / zone_area, 0.0
                    )
                ],
                [pcr.scalar(0.0)],
                [pcr.scalar(0.0)],
                "abstraction - allocation per zone/segment (PS: Error here may be caused by rounding error.)",
                True,
                extra_info_for_water_balance_reporting,
                threshold=1e-4,
            )

        return cellAbstraction, cellAllocation, zoneAbstraction

    def update(
        self, vol_gross_sectoral_water_demands, groundwater, routing, currTimeStep
    ):
        """Allocate the day's gross demand to the four sources, in a fixed order.

        Called once per day from landSurface.LandSurface.update (standard path only),
        after the water demand modules and before the soil water balance of the day.

        Arguments:
        - vol_gross_sectoral_water_demands: dict sector -> map (m3/day per demand cell)
          of POTENTIAL gross demand (not limited by availability), for "domestic",
          "industry", "manufacture", "thermoelectric", "livestock" and "irrigation";
          built in landSurface.LandSurface.update from water_demand (m/day) times
          routing.cellArea (and times fracVegCover for irrigation).
        - groundwater: groundwater.Groundwater, states at the end of the previous day
          (storGroundwater, storGroundwaterFossil in m; productive_aquifer; 7-day and
          365-day average allocations in m/day; pumpingCapacityNC;
          limitFossilGroundwaterAbstraction).
        - routing: routing.Routing, states at the end of the previous day
          (readAvlChannelStorage in m3; avgDischarge and avgBaseflow in m3/s; lddMap;
          timestepsToAvgDischarge in days).
        - currTimeStep: model time (ModelTime).

        Order: desalinated water, surface water, renewable groundwater, fossil
        groundwater. Each source sees only what the previous ones left. Demand left
        after fossil groundwater stays unmet; there is no second pass to surface water.

        Sets satisfied_gross_sectoral_water_demands and
        remaining_gross_sectoral_water_demands (m3/day per demand cell) and, through the
        source methods, the m/day attributes listed in the class docstring.
        """

        # total irrigation and livestock demand, not limited by available water; needed to allocate
        # groundwater (m3)
        self.volTotalIrrigationLivestockDemand = (
            vol_gross_sectoral_water_demands["irrigation"]
            + vol_gross_sectoral_water_demands["livestock"]
        )

        # remaining and satisfied gross sectoral water demands (m3)
        # (dict sector -> map, m3/day per demand cell; "remaining" starts at the
        # potential demand and is reduced source by source)
        # EXTENSION POINT: vol_gross_sectoral_water_demands (potential gross demand,
        # m3/day per demand cell, per sector) is the demand BEFORE any allocation. A
        # demand-driven release scheme that needs the full downstream demand (not only
        # what surface water could serve) should take it here or from
        # landSurface.LandSurface.update, where the same dict is built. Contract: store
        # it as an attribute (e.g. self.potential_gross_demand_volume), keep m3/day per
        # cell, and aggregate it to each water body in routing before
        # WaterBodies.update. The demand of day t is known before routing of day t.
        self.satisfied_gross_sectoral_water_demands = {}
        self.remaining_gross_sectoral_water_demands = {}

        for sector_name in self.sector_names:
            self.satisfied_gross_sectoral_water_demands[sector_name] = pcr.scalar(0.0)
            self.remaining_gross_sectoral_water_demands[sector_name] = (
                vol_gross_sectoral_water_demands[sector_name]
            )

        # remaining volumes of surface water and renewable and non-renewable groundwater (m3)
        # INTERFACE: routing.Routing.readAvlChannelStorage (routing.py, set at the end
        # of Routing.update of the PREVIOUS day by
        # estimate_available_volume_for_abstraction; restart state
        # "readAvlChannelStorage"). m3 per cell: channel storage S after the previous
        # day's routing, scaled by min(1, min(Q30, Qlong) / Qenv) (30-day and
        # 1825-day mean discharge over the environmental-flow discharge
        # Qenv = max(Qlong - 1.28 sd, 0.1 Qlong), m3/s), raised to (Q30 - Qenv) * 86400
        # where Q30 >= Qenv, then capped at 0.9 S and rounded down to 0.1 m3.
        # At lake and reservoir cells S is a share of the water-body storage (plus the
        # channel water left on the body's cells), spread over the body's cells by cell
        # area; no reservoir lower limit applies. One-day lag: the abstraction of day t
        # is bounded by the storage at the end of day t-1, while it is subtracted from
        # the storage of day t (routing.Routing.simple_update), after day t's runoff has
        # been added. On a cold start readAvlChannelStorageIni = 0.001
        # (pcrglobwb_5arcminutes.ini:457) gives almost no surface-water abstraction on
        # the first day (CAUTION (M19)).
        self.available_surface_water_volume = routing.readAvlChannelStorage
        # INTERFACE: groundwater.Groundwater storages (m, end of the previous day) times
        # cell area (m2) give m3 per cell; these two attributes are not used further in
        # this module (abstraction_and_allocation_from_groundwater recomputes them).
        self.available_renewable_groundwater = (
            groundwater.storGroundwater * self.cellArea
        )
        self.available_nonrenewable_groundwater = (
            groundwater.storGroundwaterFossil * self.cellArea
        )

        # abstract and allocate desalinated water
        # (first source; with desalinationWater = None, as in pcrglobwb_5arcminutes.ini,
        # it allocates zero and the whole demand passes to surface water)
        self.abstraction_and_allocation_from_desalination(
            remaining_gross_sectoral_water_demands=self.remaining_gross_sectoral_water_demands,
            currTimeStep=currTimeStep,
        )

        # update the remaining and satisfied gross sectoral water demands
        for sector_name in self.sector_names:
            self.satisfied_gross_sectoral_water_demands[
                sector_name
            ] += self.allocated_demand_per_sector["desalinated_water"][sector_name]
            self.remaining_gross_sectoral_water_demands[
                sector_name
            ] -= self.allocated_demand_per_sector["desalinated_water"][sector_name]
            self.remaining_gross_sectoral_water_demands[sector_name] = pcr.max(
                0.0, self.remaining_gross_sectoral_water_demands[sector_name]
            )

        # abstract and allocate surface water
        # (second source: rivers, lakes and reservoirs alike; sets
        # actSurfaceWaterAbstract and allocSurfaceWaterAbstract in m/day; does not
        # modify the dict it receives)
        self.abstraction_and_allocation_from_surface_water(
            remaining_gross_sectoral_water_demands=self.remaining_gross_sectoral_water_demands,
            available_surface_water_volume=self.available_surface_water_volume,
            routing=routing,
            groundwater=groundwater,
            currTimeStep=currTimeStep,
        )

        # update the remaining and satisfied gross sectoral water demands
        for sector_name in self.sector_names:
            self.satisfied_gross_sectoral_water_demands[
                sector_name
            ] += self.allocated_demand_per_sector["surface_water"][sector_name]
            self.remaining_gross_sectoral_water_demands[
                sector_name
            ] -= self.allocated_demand_per_sector["surface_water"][sector_name]
            self.remaining_gross_sectoral_water_demands[sector_name] = pcr.max(
                0.0, self.remaining_gross_sectoral_water_demands[sector_name]
            )

        # abstract and allocate groundwater
        # (renewable, then fossil; this method subtracts the groundwater allocations
        # from the dict it receives IN PLACE, i.e. from
        # self.remaining_gross_sectoral_water_ demands itself)
        self.abstraction_and_allocation_from_groundwater(
            remaining_gross_sectoral_water_demands=self.remaining_gross_sectoral_water_demands,
            routing=routing,
            groundwater=groundwater,
            currTimeStep=currTimeStep,
        )

        # update the remaining and satisfied demands after renewable groundwater allocation
        # CAUTION (M4): the groundwater method already subtracted both groundwater
        # allocations from self.remaining_gross_sectoral_water_demands (in place), so
        # the two loops below subtract them a second time (floored at 0). The satisfied
        # demands are correct; the remaining (unmet) demands left after update() are too
        # low. No other module reads remaining_gross_sectoral_water_demands today, so
        # the simulated fluxes are unaffected, but it must not be used as an "unmet
        # demand" signal for a release scheme. Use potential minus satisfied demand
        # instead.
        for sector_name in self.sector_names:
            self.satisfied_gross_sectoral_water_demands[
                sector_name
            ] += self.allocated_demand_per_sector["renewable_groundwater"][sector_name]
            self.remaining_gross_sectoral_water_demands[
                sector_name
            ] -= self.allocated_demand_per_sector["renewable_groundwater"][sector_name]
            self.remaining_gross_sectoral_water_demands[sector_name] = pcr.max(
                0.0, self.remaining_gross_sectoral_water_demands[sector_name]
            )

        # update the remaining and satisfied demands after non-renewable groundwater allocation
        for sector_name in self.sector_names:
            self.satisfied_gross_sectoral_water_demands[
                sector_name
            ] += self.allocated_demand_per_sector["nonrenewable_groundwater"][
                sector_name
            ]
            self.remaining_gross_sectoral_water_demands[
                sector_name
            ] -= self.allocated_demand_per_sector["nonrenewable_groundwater"][
                sector_name
            ]
            self.remaining_gross_sectoral_water_demands[sector_name] = pcr.max(
                0.0, self.remaining_gross_sectoral_water_demands[sector_name]
            )

    def allocate_satisfied_demand_to_each_sector(
        self,
        totalVolWaterAllocation,
        sectoral_remaining_demand_volume,
        total_remaining_demand_volume,
    ):
        """Split the water allocated to each demand cell over the sectors.

        allocated_demand_per_sector[s] = totalVolWaterAllocation * R_s / R, with R_s the
        remaining demand of sector s and R the total remaining demand of the cell
        (m3/day per demand cell; 0 where R = 0). Every sector of a cell is served in the
        same proportion; there is no sector priority at this stage.
        """

        allocated_demand_per_sector = {}

        # distribute the water allocated to the cell over the sectors, proportional to their remaining demands
        for sector_name in self.sector_names:
            allocated_demand_per_sector[sector_name] = pcr.ifthenelse(
                total_remaining_demand_volume > 0.0,
                totalVolWaterAllocation
                * vos.getValDivZero(
                    sectoral_remaining_demand_volume[sector_name],
                    total_remaining_demand_volume,
                ),
                0.0,
            )

        return allocated_demand_per_sector

    def allocate_withdrawal_to_each_sector(
        self,
        totalVolCellWaterAbstraction,
        totalVolZoneAbstraction,
        cellAllocatedDemandPerSector,
        allocation_zones=None,
    ):
        """Split the water abstracted at each source cell over the sectors.

        With zones: withdrawal_s(c) = cellAbstraction(c) * Y_s(z) / X_z, where Y_s(z) is
        the zonal total of the water allocated to sector s and X_z the zonal abstraction
        (both m3/day). Without zones the per-sector allocation at the cell is returned
        unchanged (abstraction and allocation are the same cell).

        CAUTION (M13): X_z (totalVolZoneAbstraction) is the ZONAL part only, while
        cellAbstraction and Y_s include the local part L_c (see
        waterAbstractionAndAllocation). With prioritizing_local_source (always True,
        M1) the per-sector withdrawals then do not add up to cellAbstraction (they
        exceed it, or are 0 where X_z = 0). No other module reads these per-sector
        withdrawals; do not use them to attribute a reservoir's withdrawal to sectors
        without fixing this.
        """

        # initialize the output dictionary
        allocated_withdrawal_per_sector = {}

        # with allocation zones
        if allocation_zones is not None:
            zonal_allocated_withdrawal_per_sector = {}

            for sector_name in self.sector_names:
                # total water allocated per sector over the allocation zone
                zonal_allocated_withdrawal_per_sector[sector_name] = pcr.areatotal(
                    cellAllocatedDemandPerSector[sector_name], allocation_zones
                )

                # distribute the water abstracted per sector, scaling the cell abstraction by the zonal
                # allocation per sector over the total allocation (total zonal allocation = total zonal abstraction)
                allocated_withdrawal_per_sector[sector_name] = (
                    totalVolCellWaterAbstraction
                    * vos.getValDivZero(
                        zonal_allocated_withdrawal_per_sector[sector_name],
                        totalVolZoneAbstraction,
                    )
                )

        # without allocation zones, allocation equals abstraction
        else:
            allocated_withdrawal_per_sector = cellAllocatedDemandPerSector

        return allocated_withdrawal_per_sector

    def abstraction_and_allocation_from_desalination(
        self, remaining_gross_sectoral_water_demands, currTimeStep
    ):
        """Allocate desalinated water (first source).

        Supply capacity: desalinationWater file, monthly, m/day over the cell, read on
        the first time step and on the first day of each month. Off in
        pcrglobwb_5arcminutes.ini (desalinationWater = None): everything below is then
        zero. Sets desalinationAbstraction and desalinationAllocation (m/day) and the
        per-sector dicts for "desalinated_water" (m3/day). Whatever it supplies is
        removed from the demand that surface water (and so reservoirs) must meet.
        """

        # total remaining demand (m3)
        volTotalRemainingDemand = pcr.scalar(0.0)
        for sector_name in remaining_gross_sectoral_water_demands.keys():
            volTotalRemainingDemand = (
                volTotalRemainingDemand
                + remaining_gross_sectoral_water_demands[sector_name]
            )

        # desalinated water use (m/day)
        if self.includeDesalination:
            logger.debug("Monthly desalination water use is included.")
            if currTimeStep.timeStepPCR == 1 or currTimeStep.day == 1:
                desalinationWaterUse = pcr.ifthen(
                    self.landmask,
                    pcr.cover(
                        vos.netcdf2PCRobjClone(
                            self.desalinationWaterFile,
                            "automatic",
                            currTimeStep.fulldate,
                            useDoy="monthly",
                        ),
                        0.0,
                    ),
                )
                self.desalinationWaterUse = pcr.max(0.0, desalinationWaterUse)
        else:
            logger.debug("Monthly desalination water use is NOT included.")
            self.desalinationWaterUse = pcr.scalar(0.0)

        # convert to volume (m3)
        volDesalinationWaterUse = pcr.max(
            0.0, self.desalinationWaterUse * self.cellArea
        )

        # abstraction and allocation of desalinated water
        if self.using_allocationSegmentsForDesalinatedWaterSource:
            logger.debug("Allocation of supply from desalination water.")
            (
                volDesalinationAbstraction,
                volDesalinationAllocation,
                volZoneDesalinationAbstraction,
            ) = self.waterAbstractionAndAllocation(
                water_demand_volume=volTotalRemainingDemand,
                available_water_volume=volDesalinationWaterUse,
                allocation_zones=self.allocationSegmentsForDesalinatedWaterSource,
                zone_area=self.allocationSegmentsForDesalinatedWaterSourceAreas,
                high_volume_threshold=None,
                debug_water_balance=True,
                extra_info_for_water_balance_reporting=str(currTimeStep.fulldate),
                landmask=self.landmask,
                ignore_small_values=False,
                prioritizing_local_source=self.prioritizeLocalSourceToMeetWaterDemand,
            )
        else:
            logger.debug(
                "Supply from desalination water is only for satisfying local demand (no network)."
            )
            volDesalinationAbstraction = pcr.min(
                volDesalinationWaterUse, volTotalRemainingDemand
            )
            volDesalinationAllocation = volDesalinationAbstraction
            volZoneDesalinationAbstraction = volDesalinationAbstraction

        # allocation of desalinated water per sector (m3)
        self.allocated_demand_per_sector["desalinated_water"] = (
            self.allocate_satisfied_demand_to_each_sector(
                totalVolWaterAllocation=volDesalinationAllocation,
                sectoral_remaining_demand_volume=remaining_gross_sectoral_water_demands,
                total_remaining_demand_volume=volTotalRemainingDemand,
            )
        )

        # abstraction of desalinated water per sector (m3)
        self.allocated_withdrawal_per_sector["desalinated_water"] = (
            self.allocate_withdrawal_to_each_sector(
                totalVolCellWaterAbstraction=volDesalinationAbstraction,
                totalVolZoneAbstraction=volZoneDesalinationAbstraction,
                cellAllocatedDemandPerSector=self.allocated_demand_per_sector[
                    "desalinated_water"
                ],
                allocation_zones=self.allocationSegmentsForDesalinatedWaterSource,
            )
        )

        # remaining desalinated water use (m3)
        self.volRemainingDesalinationWaterUse = pcr.max(
            0.0, volDesalinationWaterUse - volDesalinationAbstraction
        )

        # total desalinated water allocation and abstraction for other modules (m)
        # (m/day over the cell area; read by landSurface.LandSurface.update)
        self.desalinationAllocation = volDesalinationAllocation / self.cellArea
        self.desalinationAbstraction = volDesalinationAbstraction / self.cellArea

    def abstraction_and_allocation_from_surface_water(
        self,
        remaining_gross_sectoral_water_demands,
        available_surface_water_volume,
        routing,
        groundwater,
        currTimeStep,
    ):
        """Decide how much of the remaining demand is requested from surface water and
        take it from rivers, lakes and reservoirs (second source).

        Arguments: remaining_gross_sectoral_water_demands (dict sector -> m3/day per
        demand cell, after desalination); available_surface_water_volume (m3 per source
        cell, routing.readAvlChannelStorage of the previous day); routing and
        groundwater objects (previous-day states); currTimeStep.

        Steps, per demand cell c (all volumes m3/day):
        1. Surface-water fractions f (dimensionless) from
           partitioningGroundSurfaceAbstraction: f_irr for irrigation and livestock,
           f_nonirr for the other sectors.
        2. Estimate E = f_nonirr * R_nonirr + f_irr * R_irr, where R_irr is the
           remaining irrigation + livestock demand and R_nonirr the remaining demand of
           the other sectors; R_irr is used in full where f_irr >= the threshold
           threshold_to_maximize_irrigation_surface_water (1.0 under N5).
        3. Groundwater corrections: in non-productive aquifers E is replaced by
           max(0, R_irr - min(Gs, Gl) * cellArea); everywhere E is increased by
           max(0, max(Gs, Gl) * cellArea - (1 - f_irr) R_irr - (1 - f_nonirr) R_nonirr),
           with Gs, Gl the 7-day and 365-day mean groundwater allocation (m/day).
        4. Surface-water demand S = min(R_irr, E') + min(R_nonirr, max(0, E' - R_irr)),
           with E' = min(R_irr + R_nonirr, E): irrigation and livestock are served
           first from the surface-water request.
        5. Abstraction and allocation: with surface-water zones, by
           waterAbstractionAndAllocation; without (the standard RAWS case, N5),
           W_c = min(A_c, S_c) on the same cell.

        Sets (m/day over the cell area): actSurfaceWaterAbstract (at the source cell,
        subtracted from routing.channelStorage the same day) and
        allocSurfaceWaterAbstract (at the demand cell), plus the per-sector dicts for
        "surface_water" (m3/day).
        Surface-water demand that is not met passes to groundwater only partly: all
        remaining non-irrigation demand is requested from groundwater, but only
        (1 - f_irr) of the remaining irrigation and livestock demand
        (abstraction_and_allocation_from_groundwater).
        """

        # abstraction and allocation of surface water, with the surface water demand estimated from swAbstractionFractionDict

        # partitioning of the abstraction sources: groundwater and surface water
        # (dict of dimensionless maps per cell; see
        # partitioningGroundSurfaceAbstraction)
        self.swAbstractionFractionDict = self.partitioningGroundSurfaceAbstraction(
            routing
        )

        # surface water fraction for the industrial, domestic, manufacturing and thermoelectric sectors
        # (excluding irrigation and livestock)
        # f_nonirr = min(max_for_non_irrigation, estimate); replaced by the predefined
        # non-irrigation fraction when one is given (not in the RAWS inis)
        swAbstractionFraction_industrial_domestic = pcr.min(
            self.swAbstractionFractionDict["max_for_non_irrigation"],
            self.swAbstractionFractionDict["estimate"],
        )

        if self.swAbstractionFractionDict["non_irrigation"] is not None:
            swAbstractionFraction_industrial_domestic = self.swAbstractionFractionDict[
                "non_irrigation"
            ]

        # remaining demands of the combined sectors (m3)
        # R_nonirr: domestic + industry + manufacture + thermoelectric;
        # R_irr: irrigation + livestock (m3/day per demand cell, after desalination)
        remainingIndustrialDomestic = pcr.scalar(0.0)
        remainingIrrigationLivestock = pcr.scalar(0.0)

        for sector_name in remaining_gross_sectoral_water_demands.keys():
            if sector_name not in ["irrigation", "livestock"]:
                remainingIndustrialDomestic += remaining_gross_sectoral_water_demands[
                    sector_name
                ]
            else:
                remainingIrrigationLivestock += remaining_gross_sectoral_water_demands[
                    sector_name
                ]

        # total remaining demand of all sectors (m3)
        remainingTotalDemand = (
            remainingIndustrialDomestic + remainingIrrigationLivestock
        )

        # surface water demand estimate, first only for sectors other than irrigation and livestock (m3)
        # E = f_nonirr * R_nonirr
        surface_water_demand_estimate = (
            swAbstractionFraction_industrial_domestic * remainingIndustrialDomestic
        )

        # surface water demand estimate for irrigation and livestock (m3)
        surface_water_irrigation_demand_estimate = (
            self.swAbstractionFractionDict["irrigation"] * remainingIrrigationLivestock
        )

        # prioritize surface water if the groundwater irrigation fraction is relatively low (m3)
        # Where f_irr >= threshold_to_maximize_irrigation_surface_water, the whole R_irr
        # is requested from surface water. Under N5 the threshold is 1.0, so this
        # applies only where f_irr is already 1.0 and changes nothing; with the intended
        # 0.50 it would move all irrigation and livestock demand of cells with f_irr >=
        # 0.50 to surface water (and so to lakes and reservoirs where those are the
        # local source).
        surface_water_irrigation_demand_estimate = pcr.ifthenelse(
            self.swAbstractionFractionDict["irrigation"]
            >= self.swAbstractionFractionDict[
                "threshold_to_maximize_irrigation_surface_water"
            ],
            remainingIrrigationLivestock,
            surface_water_irrigation_demand_estimate,
        )

        # update the estimate of the surface water demand (m3)
        surface_water_demand_estimate += surface_water_irrigation_demand_estimate

        # prioritize surface water in non-productive aquifers with limited groundwater supply (m3)
        # In non-productive aquifers E is REPLACED by max(0, R_irr - min(Gs, Gl) *
        # cellArea), with Gs, Gl the 7-day and 365-day mean total groundwater allocation
        # (groundwater.avgAllocationShort / avgAllocation, m/day, previous day).
        # CAUTION (M10): this drops the non-irrigation part f_nonirr * R_nonirr: in
        # non-productive aquifers the domestic and industrial demand is requested from
        # surface water only through the term added below, and renewable groundwater is
        # not available there either (abstraction_and_allocation_from_groundwater), so
        # most of it stays unmet instead of falling on rivers, lakes and reservoirs.
        # Same expression on upstream master (landCover.py). Whether intended is not
        # determined from the code.
        surface_water_demand_estimate = pcr.ifthenelse(
            groundwater.productive_aquifer,
            surface_water_demand_estimate,
            pcr.max(
                0.0,
                remainingIrrigationLivestock
                - pcr.min(groundwater.avgAllocationShort, groundwater.avgAllocation)
                * self.cellArea,
            ),
        )

        # maximize surface water use in areas where groundwater supply is overestimated (m3)
        # E += max(0, max(Gs, Gl) * cellArea - (1 - f_irr) R_irr - (1 - f_nonirr)
        # R_nonirr): if recent groundwater allocation exceeded today's groundwater
        # share, the excess is requested from surface water instead
        surface_water_demand_estimate += pcr.max(
            0.0,
            pcr.max(
                groundwater.avgAllocationShort * self.cellArea,
                groundwater.avgAllocation * self.cellArea,
            )
            - (1.0 - self.swAbstractionFractionDict["irrigation"])
            * remainingIrrigationLivestock
            - (1.0 - swAbstractionFraction_industrial_domestic)
            * (remainingIndustrialDomestic),
        )

        # total demand to allocate from surface water, limited by swAbstractionFractionDict and the
        # remaining demand (m3)
        # E' = min(R_irr + R_nonirr, E); then irrigation and livestock are served first:
        # S_irr = min(E', R_irr), S_nonirr = min(R_nonirr, max(0, E' - R_irr)), S = sum
        surface_water_demand_estimate = pcr.min(
            remainingTotalDemand, surface_water_demand_estimate
        )
        correctedRemainingIrrigationLivestock = pcr.min(
            surface_water_demand_estimate, remainingIrrigationLivestock
        )
        correctedRemainingIndustrialDomestic = pcr.min(
            remainingIndustrialDomestic,
            pcr.max(0.0, surface_water_demand_estimate - remainingIrrigationLivestock),
        )
        correctedSurfaceWaterDemandEstimate = (
            correctedRemainingIrrigationLivestock + correctedRemainingIndustrialDomestic
        )
        surface_water_demand = correctedSurfaceWaterDemandEstimate
        # S (m3/day per demand cell): the demand requested from surface water before any
        # availability limit.
        # EXTENSION POINT: surface_water_demand is the best "demand on surface water"
        # signal for a demand-driven release scheme (local variable today). Contract:
        # store it as an attribute (e.g. self.surface_water_demand, m3/day per demand
        # cell) so that routing can read
        # landSurface.water_management.surface_water_demand; the shortage S -
        # volSurfaceWaterAllocation (m3/day) is the demand that surface water could not
        # meet. Both are known before routing of the same day, but a release decided on
        # day t can only be abstracted on day t+1 (availability lag, see the INTERFACE
        # note on readAvlChannelStorage in update()).

        # surface water as the first priority
        if self.surfaceWaterPiority:
            surface_water_demand = remainingTotalDemand

        # TODO: incorporate Rens's environmental flow concept; this may affect the surface water demand estimate

        # abstraction and allocation of surface water
        # Two branches. With surface-water zones (key allocationSegmentsForSurfaceWater
        # Source), demand is pooled per zone and taken from all cells of the zone in
        # proportion to their availability, so a reservoir supplies demand anywhere in
        # its zone. Without zones (local branch below), each cell can only use the water
        # of its own cell.
        # CAUTION (N5): with the standard RAWS 5-arcmin configuration
        # (pcrglobwb_5arcminutes.ini:149 sets only the unread old key) the LOCAL branch
        # runs: a lake or reservoir supplies only the surface-water demand located on
        # its own cells, and demand downstream of a reservoir is met only from the river
        # cells where it is located (which receive the releases). On upstream master,
        # configurations that set allocationSegmentsForGroundSurfaceWater (there under
        # [landSurfaceOptions]) pooled this demand over 60 arcmin zones. Not confirmed
        # by a run.
        if self.using_allocationSegmentsForSurfaceWaterSource:
            logger.debug("Allocation of supply from surface water.")
            (
                volSurfaceWaterAbstraction,
                volSurfaceWaterAllocation,
                volZoneSurfaceWaterAbstraction,
            ) = self.waterAbstractionAndAllocation(
                water_demand_volume=surface_water_demand,
                available_water_volume=available_surface_water_volume,
                allocation_zones=self.allocationSegmentsForSurfaceWaterSource,
                zone_area=self.allocationSegmentsForSurfaceWaterSourceAreas,
                high_volume_threshold=None,
                debug_water_balance=True,
                extra_info_for_water_balance_reporting=str(currTimeStep.fulldate),
                landmask=self.landmask,
                ignore_small_values=False,
                prioritizing_local_source=self.prioritizeLocalSourceToMeetWaterDemand,
            )
        else:
            logger.debug(
                "Supply from surface water is only for satisfying local demand (no network)."
            )
            # W_c = min(A_c, S_c) (m3/day): abstraction = allocation on the same cell.
            # At a reservoir cell A_c is (at most 90 % of) the cell's share of the
            # reservoir storage at the end of the previous day; S_c is the demand of
            # that cell only. The message above is logged at debug level only, so a run
            # log at the default level does not show that this branch was taken.
            volSurfaceWaterAbstraction = pcr.min(
                available_surface_water_volume, surface_water_demand
            )
            volSurfaceWaterAllocation = volSurfaceWaterAbstraction
            volZoneSurfaceWaterAbstraction = volSurfaceWaterAbstraction

        # allocation of surface water per sector (m3)
        self.allocated_demand_per_sector["surface_water"] = (
            self.allocate_satisfied_demand_to_each_sector(
                totalVolWaterAllocation=volSurfaceWaterAllocation,
                sectoral_remaining_demand_volume=remaining_gross_sectoral_water_demands,
                total_remaining_demand_volume=remainingTotalDemand,
            )
        )

        # abstraction of surface water per sector (m3)
        self.allocated_withdrawal_per_sector["surface_water"] = (
            self.allocate_withdrawal_to_each_sector(
                totalVolCellWaterAbstraction=volSurfaceWaterAbstraction,
                totalVolZoneAbstraction=volZoneSurfaceWaterAbstraction,
                cellAllocatedDemandPerSector=self.allocated_demand_per_sector[
                    "surface_water"
                ],
                allocation_zones=self.allocationSegmentsForSurfaceWaterSource,
            )
        )

        # total surface water allocation and abstraction for other modules (m)
        # INTERFACE: both are m/day over the whole cell area (volume / cellArea).
        # actSurfaceWaterAbstract is located at the SOURCE cell and is copied to
        # landSurface.actSurfaceWaterAbstract (landSurface.LandSurface.update), which
        # routing.Routing.simple_update subtracts from channelStorage (times cellArea,
        # m3) on the same day, before WaterBodies.update; at lake and reservoir cells
        # this lowers the water-body storage and the inflow residual
        # (lake_and_reservoir_inflow). It is also reported as surfaceWaterAbstraction.
        # allocSurfaceWaterAbstract is located at the DEMAND cell and enters the
        # satisfied demand (landSurface.totalPotentialGrossDemand). Without zones they
        # are equal.
        self.allocSurfaceWaterAbstract = volSurfaceWaterAllocation / self.cellArea
        self.actSurfaceWaterAbstract = volSurfaceWaterAbstraction / self.cellArea

    def abstraction_and_allocation_from_groundwater(
        self, remaining_gross_sectoral_water_demands, routing, groundwater, currTimeStep
    ):
        """Allocate renewable, then fossil groundwater (third and fourth sources).

        Groundwater-only, annotated lightly. It does not change surface-water
        abstraction on the same day: the surface-water share was fixed before. It acts
        on surface water only through later days: (1) groundwater.avgAllocation /
        avgAllocationShort (built from these allocations) enter the surface-water
        estimate of the next days; (2) groundwater abstraction can switch on riverbed
        infiltration from channels, lakes and reservoirs (routing.Routing.calculate_
        exchange_to_groundwater, where total abstraction exceeds baseflow).

        Demand for groundwater: all remaining non-irrigation demand, plus
        (1 - f_irr) of the remaining irrigation and livestock demand (m3/day).
        Renewable availability: min(storGroundwater, maximumDailyGroundwaterAbstraction)
        * cellArea * 0.75, zero in non-productive aquifers. Fossil availability:
        0.10 * storGroundwaterFossil, capped by the daily maxima, zero in non-productive
        aquifers. Mutates remaining_gross_sectoral_water_demands in place (see M4).

        Sets (m/day): nonFossilGroundwaterAbs, allocNonFossilGroundwater,
        fossilGroundwaterAbstr, fossilGroundwaterAlloc, reducedCapRise.
        """

        # remaining demands of the combined sectors (m3)
        remainingIndustrialDomestic = pcr.scalar(0.0)
        remainingIrrigationLivestock = pcr.scalar(0.0)

        for sector_name in remaining_gross_sectoral_water_demands.keys():
            if sector_name not in ["irrigation", "livestock"]:
                remainingIndustrialDomestic += remaining_gross_sectoral_water_demands[
                    sector_name
                ]
            else:
                remainingIrrigationLivestock += remaining_gross_sectoral_water_demands[
                    sector_name
                ]

        # total remaining demand of all sectors (m3)
        remainingTotalDemand = (
            remainingIndustrialDomestic + remainingIrrigationLivestock
        )

        # abstraction and allocation of groundwater (renewable and non-renewable);
        # groundwater demand of the industrial and domestic sectors (all remaining demand must be satisfied)
        groundwater_demand_estimate = remainingIndustrialDomestic
        # demand of the irrigation and livestock sectors (only partly satisfied, as it may be too high
        # due to the uncertainty in the irrigation scheme)
        irrigationLivestockGroundwaterDemand = pcr.min(
            remainingIrrigationLivestock,
            pcr.max(
                0.0,
                (1.0 - self.swAbstractionFractionDict["irrigation"])
                * remainingIrrigationLivestock,
            ),
        )
        groundwater_demand_estimate += irrigationLivestockGroundwaterDemand

        # demand to be satisfied by groundwater abstraction, not limited by available water (m3/day)
        self.potVolGroundwaterAbstract = pcr.min(
            remainingTotalDemand, groundwater_demand_estimate
        )

        # update the regional annual groundwater pumping capacity at the start of the year or simulation
        # (groundwater_pumping_region_ids, regionalAnnualGroundwaterAbstractionLimit)

        if self.limitRegionalAnnualGroundwaterAbstraction:

            logger.debug(
                "Total groundwater abstraction is limited by regional annual pumping capacity."
            )
            if currTimeStep.doy == 1 or currTimeStep.timeStepPCR == 1:

                self.groundwater_pumping_region_ids = vos.netcdf2PCRobjClone(
                    groundwater.pumpingCapacityNC,
                    "region_ids",
                    currTimeStep.fulldate,
                    useDoy="yearly",
                )

                other_ids = (
                    pcr.mapmaximum(self.groundwater_pumping_region_ids)
                    + pcr.scalar(1000.0)
                    + pcr.uniqueid(self.landmask)
                )
                self.groundwater_pumping_region_ids = pcr.cover(
                    self.groundwater_pumping_region_ids, other_ids
                )
                self.groundwater_pumping_region_ids = pcr.ifthen(
                    self.landmask, pcr.nominal(self.groundwater_pumping_region_ids)
                )

                self.regionalAnnualGroundwaterAbstractionLimit = pcr.ifthen(
                    self.landmask,
                    pcr.cover(
                        vos.netcdf2PCRobjClone(
                            groundwater.pumpingCapacityNC,
                            "regional_pumping_limit",
                            currTimeStep.fulldate,
                            useDoy="yearly",
                        ),
                        0.0,
                    ),
                )

                self.regionalAnnualGroundwaterAbstractionLimit = pcr.areamaximum(
                    self.regionalAnnualGroundwaterAbstractionLimit,
                    self.groundwater_pumping_region_ids,
                )

                # (m3/year)
                self.regionalAnnualGroundwaterAbstractionLimit *= (
                    1000.0 * 1000.0 * 1000.0
                )
                self.regionalAnnualGroundwaterAbstractionLimit = pcr.ifthen(
                    self.landmask, self.regionalAnnualGroundwaterAbstractionLimit
                )
                # minimum value (m3/year, regional)
                minimum_value = 1000.0
                self.regionalAnnualGroundwaterAbstractionLimit = pcr.max(
                    minimum_value, self.regionalAnnualGroundwaterAbstractionLimit
                )
        else:
            logger.debug(
                "Total groundwater abstraction is NOT limited by regional annual pumping capacity."
            )
            self.groundwater_pumping_region_ids = None
            self.regionalAnnualGroundwaterAbstractionLimit = None

        # constrain groundwater abstraction with the regional annual pumping capacity
        if self.limitRegionalAnnualGroundwaterAbstraction:

            logger.debug(
                "Total groundwater abstraction is limited by regional annual pumping capacity."
            )

            # total groundwater abstraction over the last 365 days (m3)
            tolerating_days = 0.0
            annualGroundwaterAbstraction = (
                groundwater.avgAbstraction
                * self.cellArea
                * pcr.min(
                    pcr.max(0.0, 365.0 - tolerating_days),
                    routing.timestepsToAvgDischarge,
                )
            )
            # note: groundwater.avgAbstraction must be in m (consistent with previous versions)

            # total regional groundwater abstraction over the last 365 days (m3)
            regionalAnnualGroundwaterAbstraction = pcr.areatotal(
                pcr.cover(annualGroundwaterAbstraction, 0.0),
                self.groundwater_pumping_region_ids,
            )

            # remaining regional pumping capacity (m3)
            remainingRegionalAnnualGroundwaterAbstractionLimit = pcr.max(
                0.0,
                self.regionalAnnualGroundwaterAbstractionLimit
                - regionalAnnualGroundwaterAbstraction,
            )
            # safety factor (residence time, day-1)
            remainingRegionalAnnualGroundwaterAbstractionLimit *= 0.33

            # remaining regional pumping capacity (m3), limited by potVolGroundwaterAbstract
            remainingRegionalAnnualGroundwaterAbstractionLimit = pcr.min(
                remainingRegionalAnnualGroundwaterAbstractionLimit,
                pcr.areatotal(
                    self.potVolGroundwaterAbstract, self.groundwater_pumping_region_ids
                ),
            )

            # remaining pumping capacity per cell (m3), downscaled using potVolGroundwaterAbstract
            remainingPixelAnnualGroundwaterAbstractionLimit = (
                remainingRegionalAnnualGroundwaterAbstractionLimit
                * vos.getValDivZero(
                    self.potVolGroundwaterAbstract,
                    pcr.areatotal(
                        self.potVolGroundwaterAbstract,
                        self.groundwater_pumping_region_ids,
                    ),
                )
            )

            # reduced potential groundwater abstraction (m3), considering the pumping capacity and average recharge (baseflow)
            # CAUTION (M6): routing.avgBaseflow is in m3/s and is added here to a volume
            # in m3 (same mismatch on upstream master, there divided by the cell area).
            # Active only with a regional pumping limit, which is off in both RAWS inis
            # (N5).
            self.potVolGroundwaterAbstract = pcr.min(
                self.potVolGroundwaterAbstract,
                remainingPixelAnnualGroundwaterAbstractionLimit
                + pcr.max(0.0, routing.avgBaseflow),
            )

        else:
            logger.debug(
                "NO LIMIT for regional groundwater (annual) pumping. It may result too high groundwater abstraction."
            )

        # abstraction and allocation of non-fossil groundwater;
        # accessible non-fossil groundwater storage (all variables in m3)
        readAvlStorGroundwater = pcr.cover(
            pcr.max(0.00, groundwater.storGroundwater * self.cellArea), 0.0
        )
        # maximum daily groundwater abstraction
        readAvlStorGroundwater = pcr.min(
            readAvlStorGroundwater,
            self.maximumDailyGroundwaterAbstraction * self.cellArea,
        )
        # ignore groundwater storage in non-productive aquifers
        readAvlStorGroundwater = pcr.ifthenelse(
            groundwater.productive_aquifer, readAvlStorGroundwater, 0.0
        )
        # in non-productive aquifers, limit readAvlStorGroundwater to the current recharge (baseflow)
        # CAUTION: the previous statement already set readAvlStorGroundwater to 0 in
        # non-productive aquifers, so min(0, recharge) below is still 0: no renewable
        # groundwater is available there (same on upstream master). See M10 for the
        # consequence on the surface-water request. routing.avgBaseflow (m3/s) times
        # 86400 s gives m3/day.
        readAvlStorGroundwater = pcr.ifthenelse(
            groundwater.productive_aquifer,
            readAvlStorGroundwater,
            pcr.min(
                readAvlStorGroundwater,
                pcr.max(routing.avgBaseflow * 24.0 * 3600.0, 0.0),
            ),
        )
        # avoid abstracting the entire groundwater volume at once
        readAvlStorGroundwater *= 0.75

        # abstraction and allocation of renewable groundwater
        if self.using_allocationSegmentsForGroundwaterSource:
            logger.debug("Allocation of supply from renewable groundwater.")
            (
                volRenewGroundwaterAbstraction,
                volRenewGroundwaterAllocation,
                volZoneRenewGroundwaterAbstraction,
            ) = self.waterAbstractionAndAllocation(
                water_demand_volume=self.potVolGroundwaterAbstract,
                available_water_volume=readAvlStorGroundwater,
                allocation_zones=self.allocationSegmentsForGroundwaterSource,
                zone_area=self.allocationSegmentsForGroundwaterSourceAreas,
                high_volume_threshold=None,
                debug_water_balance=True,
                extra_info_for_water_balance_reporting=str(currTimeStep.fulldate),
                landmask=self.landmask,
                ignore_small_values=False,
                prioritizing_local_source=self.prioritizeLocalSourceToMeetWaterDemand,
            )
        else:
            logger.debug(
                "Supply from renewable groundwater is only for satisfying local demand (no network)."
            )
            volRenewGroundwaterAbstraction = pcr.min(
                readAvlStorGroundwater, self.potVolGroundwaterAbstract
            )
            volRenewGroundwaterAllocation = volRenewGroundwaterAbstraction
            volZoneRenewGroundwaterAbstraction = volRenewGroundwaterAbstraction

        # allocation of renewable groundwater per sector (m3)
        self.allocated_demand_per_sector["renewable_groundwater"] = (
            self.allocate_satisfied_demand_to_each_sector(
                totalVolWaterAllocation=volRenewGroundwaterAllocation,
                sectoral_remaining_demand_volume=remaining_gross_sectoral_water_demands,
                total_remaining_demand_volume=remainingTotalDemand,
            )
        )

        # abstraction of renewable groundwater per sector (m3)
        self.allocated_withdrawal_per_sector["renewable_groundwater"] = (
            self.allocate_withdrawal_to_each_sector(
                totalVolCellWaterAbstraction=volRenewGroundwaterAbstraction,
                totalVolZoneAbstraction=volZoneRenewGroundwaterAbstraction,
                cellAllocatedDemandPerSector=self.allocated_demand_per_sector[
                    "renewable_groundwater"
                ],
                allocation_zones=self.allocationSegmentsForGroundwaterSource,
            )
        )

        # total renewable groundwater allocation and abstraction for other modules (m)
        # INTERFACE: m/day over the cell; nonFossilGroundwaterAbs is subtracted from
        # groundwater storage in groundwater.Groundwater.update_storage the same day,
        # and (with the fossil abstraction) decides whether routing lets channels, lakes
        # and reservoirs infiltrate to groundwater
        # (routing.Routing.calculate_exchange_to_ groundwater; that loss is taken after
        # the release rule and enters the next day's water-body inflow residual).
        self.allocNonFossilGroundwater = volRenewGroundwaterAllocation / self.cellArea
        self.nonFossilGroundwaterAbs = volRenewGroundwaterAbstraction / self.cellArea

        # needed to allocate fossil groundwater
        self.satisfiedIrrigationDemandFromNonFossilGroundwater = (
            self.allocated_demand_per_sector["renewable_groundwater"]["irrigation"]
        )

        # update the remaining demands after renewable groundwater allocation
        for sector_name in remaining_gross_sectoral_water_demands.keys():
            remaining_gross_sectoral_water_demands[sector_name] = pcr.max(
                0.0,
                remaining_gross_sectoral_water_demands[sector_name]
                - self.allocated_demand_per_sector["renewable_groundwater"][
                    sector_name
                ],
            )

        # remaining renewable groundwater that can still be abstracted (m3)
        self.volRemainingRenewGroundwater = pcr.max(
            0.0, readAvlStorGroundwater - volRenewGroundwaterAbstraction
        )

        # reduce capillary rise so there is always enough water for non-fossil groundwater abstraction (m)
        self.reducedCapRise = volRenewGroundwaterAbstraction / self.cellArea

        # demand to be satisfied by fossil groundwater abstraction, not limited by available water (m3/day)
        self.potVolFossilGroundwaterAbstract = pcr.max(
            0.0, self.potVolGroundwaterAbstract - volRenewGroundwaterAllocation
        )

        # with limitAbstraction, there is no fossil groundwater abstraction
        if self.limitAbstraction:
            logger.debug("Fossil groundwater abstractions are NOT allowed")

        # abstraction and allocation of fossil groundwater; TODO: skip this for runs without water use

        if not self.limitAbstraction:

            logger.debug("Fossil groundwater abstractions are allowed.")

            # remaining water demand per sector (m3/day), not limited by potFossilGroundwaterAbstract
            remainingDomestic = remaining_gross_sectoral_water_demands["domestic"]
            remainingIndustry = remaining_gross_sectoral_water_demands["industry"]
            remainingLivestock = remaining_gross_sectoral_water_demands["livestock"]
            # irrigation (excluding livestock)
            remainingIrrigation = remaining_gross_sectoral_water_demands["irrigation"]

            remainingIrrigationLivestock = remainingIrrigation + remainingLivestock
            # industrial and domestic (excluding livestock)
            # CAUTION (M8): manufacture and thermoelectric are left out of these totals,
            # while the per-sector split of fossil groundwater below uses all six
            # sectors, so the sector shares can sum to more than 1 when those two
            # sectors have demand. Both are off in pcrglobwb_5arcminutes.ini:131-132 (no
            # effect there).
            remainingIndustrialDomestic = remainingIndustry + remainingDomestic
            remainingTotalDemand = (
                remainingIrrigationLivestock + remainingIndustrialDomestic
            )

            # TODO: make these variables more flexible, especially for more and different sectors

        # constrain fossil groundwater abstraction with the regional pumping capacity
        if self.limitRegionalAnnualGroundwaterAbstraction and not self.limitAbstraction:

            logger.debug(
                "Fossil groundwater abstraction is allowed, BUT limited by the regional annual pumping capacity."
            )

            # total groundwater abstraction over the last 365 days (m3), including non-fossil groundwater
            annualGroundwaterAbstraction += volRenewGroundwaterAbstraction

            regionalAnnualGroundwaterAbstraction = pcr.areatotal(
                pcr.cover(annualGroundwaterAbstraction, 0.0),
                self.groundwater_pumping_region_ids,
            )

            # fossil groundwater demand reduced by the pumping capacity (m3); the safety factor avoids
            # abstracting the remaining limit at once (due to overestimated groundwater demand)
            safety_factor_for_fossil_abstraction = 1.00
            self.potVolFossilGroundwaterAbstract *= pcr.min(
                1.00,
                pcr.cover(
                    pcr.ifthenelse(
                        self.regionalAnnualGroundwaterAbstractionLimit > 0.0,
                        pcr.max(
                            0.000,
                            self.regionalAnnualGroundwaterAbstractionLimit
                            * safety_factor_for_fossil_abstraction
                            - regionalAnnualGroundwaterAbstraction,
                        )
                        / self.regionalAnnualGroundwaterAbstractionLimit,
                        0.0,
                    ),
                    0.0,
                ),
            )

        # TODO: skip this for runs without water use
        if not self.limitAbstraction:

            # remaining total demand limited by potVolFossilGroundwaterAbstract (m3)

            correctedRemainingTotalDemand = pcr.min(
                self.potVolFossilGroundwaterAbstract, remainingTotalDemand
            )

            # remaining industrial, domestic and livestock demands limited by potVolFossilGroundwaterAbstract;
            # not corrected, as these demands are always satisfied first (m3)
            correctedRemainingIndustrialDomesticLivestock = pcr.min(
                remainingIndustrialDomestic + remainingLivestock,
                correctedRemainingTotalDemand,
            )

            # remaining irrigation demand limited by potFossilGroundwaterAbstract (m3)
            correctedRemainingIrrigation = pcr.min(
                remainingIrrigation,
                pcr.max(
                    0.0,
                    correctedRemainingTotalDemand
                    - correctedRemainingIndustrialDomesticLivestock,
                ),
            )

            # ignore small irrigation demands (less than 1 mm) (m3)
            correctedRemainingIrrigation = (
                pcr.rounddown(correctedRemainingIrrigation / self.cellArea * 1000.0)
                / 1000.0
                * self.cellArea
            )

            # corrected remaining total demand, limited by potVolFossilGroundwaterAbstract (m3)
            correctedRemainingTotalDemand = (
                correctedRemainingIndustrialDomesticLivestock
                + correctedRemainingIrrigation
            )

            # corrected remaining industrial and domestic demand, excluding livestock (m3)
            correctedRemainingIndustrialDomestic = pcr.min(
                remainingIndustrialDomestic, correctedRemainingTotalDemand
            )

            # remaining irrigation and livestock demand limited by potFossilGroundwaterAbstract (m3)
            correctedRemainingIrrigationLivestock = pcr.min(
                remainingIrrigationLivestock,
                pcr.max(
                    0.0,
                    correctedRemainingTotalDemand
                    - correctedRemainingIndustrialDomestic,
                ),
            )

            # corrected remaining total demand limited by potFossilGroundwaterAbstract (m3)
            correctedRemainingTotalDemand = (
                correctedRemainingIrrigationLivestock
                + correctedRemainingIndustrialDomestic
            )

            # TODO: check the water balance: correctedRemainingIrrigationLivestock + correctedRemainingIndustrialDomestic <= potFossilGroundwaterAbstract
            # constrain the irrigation groundwater demand with the groundwater source fraction (m3)
            correctedRemainingIrrigationLivestock = pcr.min(
                (1.0 - self.swAbstractionFractionDict["irrigation"])
                * remainingIrrigationLivestock,
                correctedRemainingIrrigationLivestock,
            )
            correctedRemainingIrrigationLivestock = pcr.max(
                0.0,
                pcr.min(
                    correctedRemainingIrrigationLivestock,
                    pcr.max(0.0, self.volTotalIrrigationLivestockDemand)
                    * (1.0 - self.swAbstractionFractionDict["irrigation"])
                    - self.satisfiedIrrigationDemandFromNonFossilGroundwater,
                ),
            )

            # no fossil groundwater abstraction in irrigation areas dominated by swAbstractionFractionDict['irrigation'] (m3)
            correctedRemainingIrrigationLivestock = pcr.ifthenelse(
                self.swAbstractionFractionDict["irrigation"]
                >= self.swAbstractionFractionDict[
                    "threshold_to_minimize_fossil_groundwater_irrigation"
                ],
                0.0,
                correctedRemainingIrrigationLivestock,
            )

            # reduce the fossil irrigation and livestock demands where there is enough non-fossil groundwater
            # (to minimize unrealistic fossil groundwater abstraction): supply from the average recharge
            # (baseflow) and non-fossil groundwater allocation (m3)
            nonFossilGroundwaterSupply = pcr.max(
                pcr.max(0.0, routing.avgBaseflow * vos.secondsPerDay()),
                groundwater.avgNonFossilAllocationShort * self.cellArea,
                groundwater.avgNonFossilAllocation * self.cellArea,
            )

            # irrigation supply from non-fossil groundwater (m3)
            nonFossilIrrigationGroundwaterSupply = (
                nonFossilGroundwaterSupply
                * vos.getValDivZero(remainingIrrigationLivestock, remainingTotalDemand)
            )

            # corrected irrigation and livestock demand (m3)
            correctedRemainingIrrigationLivestock = pcr.max(
                0.0,
                correctedRemainingIrrigationLivestock
                - nonFossilIrrigationGroundwaterSupply,
            )

            # corrected remaining total demand (m3)
            correctedRemainingTotalDemand = (
                correctedRemainingIndustrialDomestic
                + correctedRemainingIrrigationLivestock
            )

            # demand to be satisfied by fossil groundwater abstraction (m3)
            self.potVolFossilGroundwaterAbstract = pcr.min(
                self.potVolFossilGroundwaterAbstract, correctedRemainingTotalDemand
            )

            if (
                not groundwater.limitFossilGroundwaterAbstraction
                and not self.limitAbstraction
            ):

                # note: if limitFossilGroundwaterAbstraction is False, fossil groundwater allocation is not needed
                msg = "Fossil groundwater abstractions are without limit for satisfying local demand. "
                msg = "Allocation for fossil groundwater abstraction is NOT needed/implemented. "
                msg += "However, the fossil groundwater abstraction rate still consider the maximumDailyGroundwaterAbstraction."
                logger.debug(msg)

                # fossil groundwater abstraction (m3)
                self.fossilGroundwaterAbstrVol = self.potVolFossilGroundwaterAbstract
                self.fossilGroundwaterAbstrVol = pcr.min(
                    self.fossilGroundwaterAbstrVol,
                    pcr.max(
                        0.0,
                        self.maximumDailyGroundwaterAbstraction * self.cellArea
                        - volRenewGroundwaterAbstraction,
                    ),
                )

                # fossil groundwater allocation (m3)
                self.fossilGroundwaterAllocVol = self.fossilGroundwaterAbstrVol

            if (
                groundwater.limitFossilGroundwaterAbstraction
                and not self.limitAbstraction
            ):
                logger.debug(
                    "Fossil groundwater abstractions are allowed, but with limit."
                )

                # accessible fossil groundwater (m)
                readAvlFossilGroundwater = pcr.ifthenelse(
                    groundwater.productive_aquifer,
                    groundwater.storGroundwaterFossil,
                    0.0,
                )

                # residence time (day-1) or safety factor (avoids 'unrealistic' zero fossil groundwater)
                readAvlFossilGroundwater *= 0.10

                # maximum daily groundwater abstraction (m)
                readAvlFossilGroundwater = pcr.min(
                    readAvlFossilGroundwater,
                    self.maximumDailyFossilGroundwaterAbstraction,
                    pcr.max(
                        0.0,
                        self.maximumDailyGroundwaterAbstraction
                        - volRenewGroundwaterAbstraction / self.cellArea,
                    ),
                )
                readAvlFossilGroundwater = pcr.max(
                    pcr.cover(readAvlFossilGroundwater, 0.0), 0.0
                )

                # accessible fossil groundwater volume (m3)
                readAvlFossilGroundwaterVol = readAvlFossilGroundwater * self.cellArea

                # fossil groundwater abstraction and allocation (m3);
                # TODO: consider aquifer productivity in the allocation
                if self.using_allocationSegmentsForGroundwaterSource:
                    logger.debug("Allocation of fossil groundwater abstraction.")
                    (
                        volFossilGroundwaterAbstraction,
                        volFossilGroundwaterAllocation,
                        volZoneFossilGroundwaterAbstraction,
                    ) = self.waterAbstractionAndAllocation(
                        water_demand_volume=self.potVolFossilGroundwaterAbstract,
                        available_water_volume=pcr.max(
                            0.00, readAvlFossilGroundwaterVol
                        ),
                        allocation_zones=self.allocationSegmentsForGroundwaterSource,
                        zone_area=self.allocationSegmentsForGroundwaterSourceAreas,
                        high_volume_threshold=None,
                        debug_water_balance=True,
                        extra_info_for_water_balance_reporting=str(
                            currTimeStep.fulldate
                        ),
                        landmask=self.landmask,
                        ignore_small_values=False,
                        prioritizing_local_source=self.prioritizeLocalSourceToMeetWaterDemand,
                    )
                else:
                    logger.debug(
                        "Fossil groundwater abstraction is only for satisfying local demand. NO Allocation for fossil groundwater abstraction."
                    )

                    volFossilGroundwaterAbstraction = pcr.min(
                        pcr.max(0.0, readAvlFossilGroundwaterVol),
                        self.potVolFossilGroundwaterAbstract,
                    )
                    volFossilGroundwaterAllocation = volFossilGroundwaterAbstraction
                    volZoneFossilGroundwaterAbstraction = (
                        volFossilGroundwaterAbstraction
                    )

        # allocation of non-renewable groundwater per sector (m3)
        # CAUTION (M7): volFossilGroundwaterAllocation and
        # volFossilGroundwaterAbstraction are defined only in the branch
        # "groundwater.limitFossilGroundwaterAbstraction and not self.limitAbstraction".
        # groundwater.py sets limitFossilGroundwaterAbstraction = True whenever QUAlloc
        # is off, so on the standard path only limitAbstraction = True leaves them
        # undefined (error here).
        self.allocated_demand_per_sector["nonrenewable_groundwater"] = (
            self.allocate_satisfied_demand_to_each_sector(
                totalVolWaterAllocation=volFossilGroundwaterAllocation,
                sectoral_remaining_demand_volume=remaining_gross_sectoral_water_demands,
                total_remaining_demand_volume=remainingTotalDemand,
            )
        )

        # abstraction of non-renewable groundwater per sector (m3)
        self.allocated_withdrawal_per_sector["nonrenewable_groundwater"] = (
            self.allocate_withdrawal_to_each_sector(
                totalVolCellWaterAbstraction=volFossilGroundwaterAbstraction,
                totalVolZoneAbstraction=volZoneFossilGroundwaterAbstraction,
                cellAllocatedDemandPerSector=self.allocated_demand_per_sector[
                    "nonrenewable_groundwater"
                ],
                allocation_zones=self.allocationSegmentsForGroundwaterSource,
            )
        )

        # update the remaining demands after non-renewable groundwater allocation (m3)
        for sector_name in remaining_gross_sectoral_water_demands.keys():
            remaining_gross_sectoral_water_demands[sector_name] = pcr.max(
                0.0,
                remaining_gross_sectoral_water_demands[sector_name]
                - self.allocated_demand_per_sector["nonrenewable_groundwater"][
                    sector_name
                ],
            )

        # total non-renewable groundwater allocation and abstraction for other modules (m)
        # INTERFACE: m/day over the cell; read by landSurface.LandSurface.update and
        # then groundwater.Groundwater.update_storage (fossilGroundwaterAbstr) on the
        # same day.
        self.fossilGroundwaterAlloc = volFossilGroundwaterAllocation / self.cellArea
        self.fossilGroundwaterAbstr = volFossilGroundwaterAbstraction / self.cellArea

    def partitioningGroundSurfaceAbstraction(self, routing_module):
        """Return the surface-water fractions of demand (dimensionless, per cell).

        De Graaf et al. (2014): f = Q_up / (Q_up + Q_bf), with
        Q_up = max(avgDischarge, sum of avgDischarge of the cells draining into the
        cell) and Q_bf = avgBaseflow, both long-term (1825-day) running means in m3/s
        from routing at the end of the previous day. f is rounded UP to 0.01, clipped
        to [0, 1] and set to 1.0 where undefined. With surface-water zones, Q_bf is
        averaged and Q_up and f maximised over the zone (not the case with the
        standard RAWS configuration, N5).

        Returned dict keys: "estimate" (f), "irrigation" (f, or a blend with the
        Siebert map when given, as in the RAWS inis), "max_for_non_irrigation" (f, or
        the maximumNonIrrigation... map when the Siebert map is given, as in the RAWS
        inis), "non_irrigation" (None unless a predefined map is given), and the two
        thresholds.

        Lakes and reservoirs: at a water-body cell Q_up is the long-term discharge of
        the cell and of its upstream neighbours (routing.avgDischarge, which on
        water-body cells is built from the maximum discharge over the body), so by the
        formula f tends to 1 on and below large water bodies, and most demand there is
        requested from surface water (not checked by a run).
        """

        # partition the abstraction sources (groundwater and surface water) following de Graaf et al. (2014),
        # based on the local average baseflow (m3/s) and upstream average discharge (m3/s)
        # INTERFACE: routing.Routing.avgBaseflow, avgDischarge (m3/s, long-term means
        # updated in routing.Routing.calculate_statistics at the end of the previous
        # day; restart states avgBaseflowLong and avgDischargeLong) and lddMap.
        averageBaseflowInput = routing_module.avgBaseflow
        averageUpstreamInput = pcr.max(
            routing_module.avgDischarge,
            pcr.cover(
                pcr.upstream(routing_module.lddMap, routing_module.avgDischarge), 0.0
            ),
        )

        if self.using_allocationSegmentsForSurfaceWaterSource:
            averageBaseflowInput = pcr.max(
                0.0, pcr.ifthen(self.landmask, averageBaseflowInput)
            )
            averageUpstreamInput = pcr.max(
                0.0, pcr.ifthen(self.landmask, averageUpstreamInput)
            )

            averageBaseflowInput = pcr.cover(
                pcr.areaaverage(
                    averageBaseflowInput, self.allocationSegmentsForSurfaceWaterSource
                ),
                0.0,
            )
            averageUpstreamInput = pcr.cover(
                pcr.areamaximum(
                    averageUpstreamInput, self.allocationSegmentsForSurfaceWaterSource
                ),
                0.0,
            )

        else:
            logger.debug("Water demand can only be satisfied by local source.")

        # f = Q_up / (Q_up + Q_bf), rounded up to the next 0.01 and clipped to [0, 1]
        swAbstractionFraction = vos.getValDivZero(
            averageUpstreamInput,
            averageUpstreamInput + averageBaseflowInput,
            vos.smallNumber,
        )
        swAbstractionFraction = pcr.roundup(swAbstractionFraction * 100.0) / 100.0
        swAbstractionFraction = pcr.max(0.0, swAbstractionFraction)
        swAbstractionFraction = pcr.min(1.0, swAbstractionFraction)

        if self.using_allocationSegmentsForSurfaceWaterSource:
            swAbstractionFraction = pcr.areamaximum(
                swAbstractionFraction, self.allocationSegmentsForSurfaceWaterSource
            )

        swAbstractionFraction = pcr.cover(swAbstractionFraction, 1.0)
        swAbstractionFraction = pcr.ifthen(self.landmask, swAbstractionFraction)

        # surface water fractions for various purposes
        swAbstractionFractionDict = {}

        # default estimate (de Graaf et al., 2014)
        swAbstractionFractionDict["estimate"] = swAbstractionFraction

        # irrigation and livestock
        swAbstractionFractionDict["irrigation"] = swAbstractionFraction

        # industrial and domestic
        swAbstractionFractionDict["max_for_non_irrigation"] = swAbstractionFraction

        # threshold to maximize surface water withdrawal for irrigation: areas with
        # swAbstractionFractionDict['irrigation'] above it prioritize surface water for irrigation
        # (zero disables this)
        # CAUTION: see M9. The test is f_irr >= threshold
        # (abstraction_and_allocation_from_surface_water), so a threshold of zero
        # applies the rule in every cell (all irrigation and livestock demand to surface
        # water); it is the value 1.0 (the default, and the value in force under N5)
        # that in effect disables it.
        swAbstractionFractionDict["threshold_to_maximize_irrigation_surface_water"] = (
            self.threshold_to_maximize_irrigation_surface_water
        )

        # threshold to minimize (unrealistic) fossil groundwater withdrawal: areas with
        # swAbstractionFractionDict['irrigation'] above it do not abstract fossil groundwater
        swAbstractionFractionDict[
            "threshold_to_minimize_fossil_groundwater_irrigation"
        ] = self.threshold_to_minimize_fossil_groundwater_irrigation

        # by default the non-irrigation fraction is None (then the 'estimate' limited by 'max_for_non_irrigation' is used)
        swAbstractionFractionDict["non_irrigation"] = None

        # include the predefined surface water fraction (e.g. Siebert et al., 2014; McDonald et al., 2014)
        if self.swAbstractionFractionData is not None:
            logger.debug(
                "Using/incorporating the predefined fractions of surface water source."
            )
            swAbstractionFractionDict["estimate"] = swAbstractionFraction
            swAbstractionFractionDict["irrigation"] = (
                self.partitioningGroundSurfaceAbstractionForIrrigation(
                    swAbstractionFraction,
                    self.swAbstractionFractionData,
                    self.swAbstractionFractionDataQuality,
                )
            )
            swAbstractionFractionDict["max_for_non_irrigation"] = (
                self.maximumNonIrrigationSurfaceWaterAbstractionFractionData
            )

            if (
                self.predefinedNonIrrigationSurfaceWaterAbstractionFractionData
                is not None
            ):
                swAbstractionFractionDict["non_irrigation"] = pcr.cover(
                    self.predefinedNonIrrigationSurfaceWaterAbstractionFractionData,
                    swAbstractionFractionDict["estimate"],
                )
                swAbstractionFractionDict["non_irrigation"] = pcr.min(
                    swAbstractionFractionDict["non_irrigation"],
                    swAbstractionFractionDict["max_for_non_irrigation"],
                )
        else:
            logger.debug(
                "NOT using/incorporating the predefined fractions of surface water source."
            )

        return swAbstractionFractionDict

    def partitioningGroundSurfaceAbstractionForIrrigation(
        self,
        swAbstractionFractionEstimate,
        swAbstractionFractionData,
        swAbstractionFractionDataQuality,
    ):
        """Blend the predefined irrigation surface-water fraction with the estimate.

        f_irr = w * f_data + (1 - w) * f_estimate, w = 1 - 0.5 * min(5, max(0, q)) / 10,
        with q the data quality class (0 best, 5 worst kept), so w is between 0.75 and
        1.0. Where the data are missing f_irr = f_estimate, and 1.0 where both are
        missing. Dimensionless, per cell.
        """

        # surface water fraction from Stefan Siebert's map; with this factor, the minimum data_weight_value
        # is 0.75 (for swAbstractionFractionDataQuality == 5)
        factor = 0.5
        data_weight_value = (
            pcr.scalar(1.0)
            - (pcr.min(5.0, pcr.max(0.0, swAbstractionFractionDataQuality)) / 10.0)
            * factor
        )

        swAbstractionFractionForIrrigation = (
            data_weight_value * swAbstractionFractionData
            + (1.0 - data_weight_value) * swAbstractionFractionEstimate
        )

        swAbstractionFractionForIrrigation = pcr.cover(
            swAbstractionFractionForIrrigation, swAbstractionFractionEstimate
        )
        swAbstractionFractionForIrrigation = pcr.cover(
            swAbstractionFractionForIrrigation, 1.0
        )
        swAbstractionFractionForIrrigation = pcr.ifthen(
            self.landmask, swAbstractionFractionForIrrigation
        )

        return swAbstractionFractionForIrrigation
