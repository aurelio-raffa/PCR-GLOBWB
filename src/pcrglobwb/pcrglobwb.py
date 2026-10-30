import logging

import pcraster as pcr

from pcrglobwb import groundwater, landSurface, meteo, routing
from pcrglobwb.common import virtualOS as vos

logger = logging.getLogger(__name__)

# ANNOTATION OVERVIEW (comments and docstrings only; the code is unchanged). Scope: the
# order of the module updates within one daily time step, the saving and restoring of
# states (water-body states in particular) and spin-up, as they affect lakes and
# reservoirs. See PCRGlobWB.update for the order and the one-day lags it creates.


class PCRGlobWB(object):
    """The PCR-GLOBWB model: four sub-models updated once per daily time step.

    Built by deterministic_runner_with_arguments.DeterministicRunner (the entry point
    pcrglobwb-run-with-arguments used by the RAWS configurations), once per spin-up
    cycle and once for the transient run. Owns meteo, landSurface, groundwater and
    routing; the lakes and reservoirs (waterBodies.WaterBodies) are owned by routing.
    """

    def __init__(self, configuration, currTimeStep, initialState=None, spinUpRun=None):
        """Build the sub-models.

        initialState: None (read the initial states from the *Ini keys of the
        configuration, e.g. [routingOptions] waterBodyStorageIni) or the dict returned
        by getState() at the end of the previous spin-up cycle (states kept in memory).
        spinUpRun: stored only; not used elsewhere in this class.
        """
        self._configuration = configuration
        self._modelTime = currTimeStep

        vos.set_clone(configuration.cloneMap)

        self.lddMap = vos.readPCRmapClone(
            configuration.routingOptions["lddMap"],
            configuration.tmpDir,
            configuration.globalOptions["inputDir"],
            True,
        )
        # make sure the ldd map is correct and of type ldd
        self.lddMap = pcr.lddrepair(pcr.ldd(self.lddMap))

        if configuration.globalOptions["landmask"] != "None":
            self.landmask = vos.readPCRmapClone(
                configuration.globalOptions["landmask"],
                configuration.tmpDir,
                configuration.globalOptions["inputDir"],
            )
        else:
            self.landmask = pcr.defined(self.lddMap)

        # catchment areas
        self.catchment_class = 1.0

        # number of upper soil layers
        self.numberOfSoilLayers = int(
            configuration.landSurfaceOptions["numberOfUpperSoilLayers"]
        )

        self.createSubmodels(initialState)

        # option to save monthly end states
        self.save_monthly_end_states = False
        if "save_monthly_end_states" in list(configuration.reportingOptions.keys()):
            self.save_monthly_end_states = (
                configuration.reportingOptions["save_monthly_end_states"] == "True"
            )

        # whether this run is a spin-up run
        # (no other code in this class reads it: a spin-up cycle runs the same update,
        # dumps end states and is reported like the transient run; only its time window
        # and initial states differ, see deterministic_runner_with_arguments.main)
        self.spinUpRun = spinUpRun

    @property
    def configuration(self):
        return self._configuration

    def createSubmodels(self, initialState):
        """Build meteo, landSurface, groundwater and routing, in this order.

        The order matters for water management: landSurface builds
        water_management.WaterManagement (when using_qualloc is False), which writes
        default values into configuration.waterManagementOptions (e.g. limitAbstraction,
        pumpingCapacityNC = "None") that groundwater.Groundwater then reads. routing
        builds waterBodies.WaterBodies, but the lake and reservoir parameters and states
        are loaded only at the first routing update (routing.Routing.update,
        WaterBodies.getParameterFiles).
        """

        self.meteo = meteo.Meteo(self._configuration, self.landmask, initialState)
        self.landSurface = landSurface.LandSurface(
            self._configuration, self.landmask, initialState
        )
        self.groundwater = groundwater.Groundwater(
            self._configuration, self.landmask, initialState
        )
        self.routing = routing.Routing(self._configuration, initialState, self.lddMap)

        # short name for every land cover type (used in file names)
        self.shortNames = ["f", "g", "p", "n"]

    def dumpState(self, outputDirectory, specific_date_string=None):
        # write all states to disk to allow restarting
        # One PCRaster map per state, named <variable>_<YYYY-MM-DD>.map (land-cover
        # states: <variable>_<coverType>_<YYYY-MM-DD>.map), in outputDirectory (the
        # end-state directory of the run). The water-body states are routing states (see
        # routing.Routing.getState): waterBodyStorage (m3, per lake or reservoir, the
        # body total repeated on each of its cells), avgLakeReservoirInflowShort (m3/s,
        # 30-day mean inflow), avgLakeReservoirOutflowLong (m3/s, 1825-day mean
        # outflow), plus channelStorage (m3 per cell, which at water-body cells includes
        # the water-body storage spread by cell area), readAvlChannelStorage (m3,
        # surface water available for abstraction on the next day), avgDischargeLong,
        # avgDischargeShort, m2tDischargeLong, avgBaseflowLong, timestepsToAvgDischarge,
        # riverbedExchange and subDischarge. To warm-start, point the matching *Ini keys
        # of [routingOptions] (e.g. waterBodyStorageIni, avgLakeReservoirInflowShortIni,
        # avgLakeReservoirOutflowLongIni, readAvlChannelStorageIni) at these files.
        # QUAlloc keeps its own states (written by qualloc_model.finalize_year).

        if specific_date_string is None:
            specific_date_string = str(self._modelTime.fulldate)

        state = self.getState()

        meteoState = state["meteo"]
        for variable, map in list(meteoState.items()):
            vos.writePCRmapToDir(
                map,
                str(variable) + "_" + specific_date_string + ".map",
                outputDirectory,
            )

        landSurfaceState = state["landSurface"]
        for coverType, coverTypeState in list(landSurfaceState.items()):
            for variable, map in list(coverTypeState.items()):
                vos.writePCRmapToDir(
                    map,
                    str(variable)
                    + "_"
                    + coverType
                    + "_"
                    + specific_date_string
                    + ".map",
                    outputDirectory,
                )

        groundWaterState = state["groundwater"]
        for variable, map in list(groundWaterState.items()):
            vos.writePCRmapToDir(
                map,
                str(variable) + "_" + specific_date_string + ".map",
                outputDirectory,
            )

        routingState = state["routing"]
        for variable, map in list(routingState.items()):
            vos.writePCRmapToDir(
                map,
                str(variable) + "_" + specific_date_string + ".map",
                outputDirectory,
            )

    # TODO: implement

    def report_summary(
        self,
        landWaterStoresAtBeginning,
        landWaterStoresAtEnd,
        surfaceWaterStoresAtBeginning,
        surfaceWaterStoresAtEnd,
    ):

        # reset the totals on the first day of the year
        if self._modelTime.doy == 1 or self._modelTime.isFirstTimestep():

            self.precipitationAcc = pcr.ifthen(
                self.landmask, pcr.spatial(pcr.scalar(0.0))
            )

            self.list_of_land_surface_variables = self.landSurface.fluxVars + [
                "desalinationAbstraction",
                "desalinationAllocation",
                "actSurfaceWaterAbstract",
                "allocSurfaceWaterAbstract",
                "nonFossilGroundwaterAbs",
                "allocNonFossilGroundwater",
                "fossilGroundwaterAbstr",
                "fossilGroundwaterAlloc",
                "totalGroundwaterAbstraction",
                "totalGroundwaterAllocation",
                "nonIrrReturnFlow",
            ]

            for var in self.list_of_land_surface_variables:
                vars(self)[var + "Acc"] = pcr.ifthen(
                    self.landmask, pcr.spatial(pcr.scalar(0.0))
                )

            self.baseflowAcc = pcr.ifthen(self.landmask, pcr.spatial(pcr.scalar(0.0)))

            self.surfaceWaterInfAcc = pcr.ifthen(
                self.landmask, pcr.spatial(pcr.scalar(0.0))
            )

            self.runoffAcc = pcr.ifthen(self.landmask, pcr.spatial(pcr.scalar(0.0)))
            self.unmetDemandAcc = pcr.ifthen(
                self.landmask, pcr.spatial(pcr.scalar(0.0))
            )

            self.waterBalanceAcc = pcr.ifthen(
                self.landmask, pcr.spatial(pcr.scalar(0.0))
            )
            self.absWaterBalanceAcc = pcr.ifthen(
                self.landmask, pcr.spatial(pcr.scalar(0.0))
            )

            # non-irrigation water use (m)
            self.nonIrrigationWaterUseAcc = pcr.ifthen(
                self.landmask, pcr.spatial(pcr.scalar(0.0))
            )

            # non-irrigation return flow to water bodies and water body evaporation (m)
            self.nonIrrReturnFlowAcc = pcr.ifthen(
                self.landmask, pcr.spatial(pcr.scalar(0.0))
            )
            self.waterBodyEvaporationAcc = pcr.ifthen(
                self.landmask, pcr.spatial(pcr.scalar(0.0))
            )

            # surface water input/loss volume (m3) and outgoing volume at pits (m3)
            self.surfaceWaterInputAcc = pcr.ifthen(
                self.landmask, pcr.spatial(pcr.scalar(0.0))
            )
            self.dischargeAtPitAcc = pcr.ifthen(
                self.landmask, pcr.spatial(pcr.scalar(0.0))
            )

            # storages at the first day of the year (or first time step): land surface (m)
            self.storageAtFirstDay = pcr.ifthen(
                self.landmask, landWaterStoresAtBeginning
            )
            # channel storage (m3)
            self.channelVolumeAtFirstDay = pcr.ifthen(
                self.landmask, surfaceWaterStoresAtBeginning
            )

        # accumulate until the last day of the year
        self.precipitationAcc += self.meteo.precipitation
        for var in self.list_of_land_surface_variables:
            vars(self)[var + "Acc"] += vars(self.landSurface)[var]

        self.baseflowAcc += self.groundwater.baseflow

        self.surfaceWaterInfAcc += self.groundwater.surfaceWaterInf

        self.runoffAcc += self.routing.runoff
        self.unmetDemandAcc += self.groundwater.unmetDemand

        self.waterBalance = (
            landWaterStoresAtBeginning
            - landWaterStoresAtEnd
            + self.meteo.precipitation
            + self.landSurface.irrigationWaterWithdrawal / self.routing.cellArea
            + self.groundwater.surfaceWaterInf
            - self.landSurface.actualET
            - self.routing.runoff
            - self.groundwater.nonFossilGroundwaterAbs
        )

        self.waterBalanceAcc += self.waterBalance
        self.absWaterBalanceAcc += pcr.abs(self.waterBalance)

        # consumptive water use for non-irrigation demand (m)
        self.nonIrrigationWaterUseAcc += self.landSurface.nonIrrWaterConsumption

        self.waterBodyEvaporationAcc += self.routing.waterBodyEvaporation

        # (m3)
        self.surfaceWaterInputAcc += self.routing.local_input_to_surface_water
        # (m3)
        self.dischargeAtPitAcc += self.routing.outgoing_volume_at_pits

        if self._modelTime.isLastDayOfYear() or self._modelTime.isLastTimeStep():

            logger.info("")
            msg = "The following summary values do not include storages in surface water bodies (lake, reservoir and channel storages)."
            # TODO: improve these water balance checks
            logger.info(msg)

            totalCellArea = vos.getMapTotal(
                pcr.ifthen(self.landmask, self.routing.cellArea)
            )
            msg = "Total area = %e km2" % (totalCellArea / 1e6)
            logger.info(msg)

            deltaStorageOneYear = vos.getMapVolume(
                pcr.ifthen(self.landmask, landWaterStoresAtBeginning)
                - pcr.ifthen(self.landmask, self.storageAtFirstDay),
                self.routing.cellArea,
            )
            msg = "Delta total storage days 1 to %i in %i = %e km3 = %e mm" % (
                int(self._modelTime.doy),
                int(self._modelTime.year),
                deltaStorageOneYear / 1e9,
                deltaStorageOneYear * 1000 / totalCellArea,
            )
            logger.info(msg)

            variableList = [
                "precipitation",
                "baseflow",
                "surfaceWaterInf",
                "runoff",
                "unmetDemand",
            ]
            variableList += self.list_of_land_surface_variables

            variableList += ["waterBalance", "absWaterBalance", "nonIrrigationWaterUse"]

            for var in variableList:
                volume = vos.getMapVolume(
                    self.__getattribute__(var + "Acc"), self.routing.cellArea
                )

                # TODO: the calculation does not always start on day 1
                msg = "Accumulated %s days 1 to %i in %i = %e km3 = %e mm" % (
                    var,
                    int(self._modelTime.doy),
                    int(self._modelTime.year),
                    volume / 1e9,
                    volume * 1000 / totalCellArea,
                )
                logger.info(msg)

            logger.info("")
            msg = "The following summary is for surface water bodies."
            logger.info(msg)

            deltaChannelStorageOneYear = vos.getMapTotal(
                pcr.ifthen(self.landmask, surfaceWaterStoresAtEnd)
                - pcr.ifthen(self.landmask, self.channelVolumeAtFirstDay)
            )
            msg = "Delta surface water storage days 1 to %i in %i = %e km3 = %e mm" % (
                int(self._modelTime.doy),
                int(self._modelTime.year),
                deltaChannelStorageOneYear / 1e9,
                deltaChannelStorageOneYear * 1000 / totalCellArea,
            )
            logger.info(msg)

            variableList = ["waterBodyEvaporation"]
            for var in variableList:
                volume = vos.getMapVolume(
                    self.__getattribute__(var + "Acc"), self.routing.cellArea
                )
                msg = "Accumulated %s days 1 to %i in %i = %e km3 = %e mm" % (
                    var,
                    int(self._modelTime.doy),
                    int(self._modelTime.year),
                    volume / 1e9,
                    volume * 1000 / totalCellArea,
                )
                logger.info(msg)

            surfaceWaterInputTotal = vos.getMapTotal(self.surfaceWaterInputAcc)
            msg = "Accumulated %s days 1 to %i in %i = %e km3 = %e mm" % (
                "surfaceWaterInput",
                int(self._modelTime.doy),
                int(self._modelTime.year),
                surfaceWaterInputTotal / 1e9,
                surfaceWaterInputTotal * 1000 / totalCellArea,
            )
            logger.info(msg)

            dischargeAtPitTotal = vos.getMapTotal(self.dischargeAtPitAcc)
            msg = "Accumulated %s days 1 to %i in %i = %e km3 = %e mm" % (
                "dischargeAtPitTotal",
                int(self._modelTime.doy),
                int(self._modelTime.year),
                dischargeAtPitTotal / 1e9,
                dischargeAtPitTotal * 1000 / totalCellArea,
            )
            logger.info(msg)

            surfaceWaterBalance = (
                deltaChannelStorageOneYear
                - surfaceWaterInputTotal
                + dischargeAtPitTotal
            )
            msg = "Accumulated %s days 1 to %i in %i = %e km3 = %e mm" % (
                "surfaceWaterBalance",
                int(self._modelTime.doy),
                int(self._modelTime.year),
                surfaceWaterBalance / 1e9,
                surfaceWaterBalance * 1000 / totalCellArea,
            )
            logger.info(msg)

    def getState(self):
        """Return the restart states as {"meteo": ..., "landSurface": ...,
        "groundwater": ..., "routing": ...}, each a dict of PCRaster maps.

        Used by dumpState (files) and by the runner to pass states in memory from one
        spin-up cycle to the next and to the transient run (as initialState). The
        lake and reservoir states are in result["routing"]: waterBodyStorage,
        avgLakeReservoirInflowShort and avgLakeReservoirOutflowLong (see dumpState).
        They are consumed by routing.Routing.getICs and, on the first routing update, by
        waterBodies.WaterBodies.getICs.
        """
        result = {}

        result["meteo"] = self.meteo.getState()

        result["landSurface"] = self.landSurface.getState()
        result["groundwater"] = self.groundwater.getState()
        result["routing"] = self.routing.getState()

        return result

    def getPseudoState(self):
        result = {}

        result["meteo"] = self.meteo.getPseudoState()

        result["landSurface"] = self.landSurface.getPseudoState()
        result["groundwater"] = self.groundwater.getPseudoState()
        result["routing"] = self.routing.getPseudoState()

        return result

    def getAllState(self):
        result = {}

        result["meteo"] = self.meteo.getState()
        result["meteo"].update(self.meteo.getPseudoState())

        result["landSurface"] = self.landSurface.getState()
        result["landSurface"].update(self.landSurface.getPseudoState())

        result["groundwater"] = self.groundwater.getState()
        result["groundwater"].update(self.groundwater.getPseudoState())

        result["routing"] = self.routing.getState()
        result["routing"].update(self.routing.getPseudoState())

        return result

    def totalLandWaterStores(self):
        # (m), excluding surface water bodies

        if self.numberOfSoilLayers == 2:
            total = (
                self.landSurface.interceptStor
                + self.landSurface.snowFreeWater
                + self.landSurface.snowCoverSWE
                + self.landSurface.topWaterLayer
                + self.landSurface.storUpp
                + self.landSurface.storLow
                + self.groundwater.storGroundwater
            )

        if self.numberOfSoilLayers == 3:
            total = (
                self.landSurface.interceptStor
                + self.landSurface.snowFreeWater
                + self.landSurface.snowCoverSWE
                + self.landSurface.topWaterLayer
                + self.landSurface.storUpp000005
                + self.landSurface.storUpp005030
                + self.landSurface.storLow030150
                + self.groundwater.storGroundwater
            )

        total = pcr.ifthen(self.landmask, total)

        return total

    def totalSurfaceWaterStores(self):
        # (m3), surface water bodies only
        # (channels, lakes and reservoirs: at the start and end of a step,
        # routing.channelStorage holds the water-body storage spread over the body's
        # cells)

        return pcr.ifthen(self.landmask, self.routing.channelStorage)

    def checkLandSurfaceWaterBalance(self, storesAtBeginning, storesAtEnd):

        # all stores (snow, interception, soil, groundwater), excluding routing: incoming fluxes (m)
        precipitation = pcr.ifthen(self.landmask, self.meteo.precipitation)
        satisfiedIrrGrossDemand = pcr.ifthen(
            self.landmask,
            self.landSurface.irrigationWaterWithdrawal / self.routing.cellArea,
        )
        surfaceWaterInf = pcr.ifthen(self.landmask, self.groundwater.surfaceWaterInf)
        # outgoing fluxes (m)
        actualET = pcr.ifthen(self.landmask, self.landSurface.actualET)
        runoff = pcr.ifthen(self.landmask, self.routing.runoff)
        nonFossilGroundwaterAbs = pcr.ifthen(
            self.landmask, self.groundwater.nonFossilGroundwaterAbs
        )
        # added by Joren; TODO: add an option to switch between the new and old module
        transportVolSnow = self.landSurface.transportVolSnow / self.routing.cellArea
        incomingVolSnow = self.landSurface.incomingVolSnow / self.routing.cellArea

        incomingFreeWater = self.landSurface.incomingFreeWater / self.routing.cellArea
        transportFreeWater = self.landSurface.transportFreeWater / self.routing.cellArea

        vos.waterBalanceCheck(
            [
                precipitation,
                surfaceWaterInf,
                satisfiedIrrGrossDemand,
                incomingVolSnow,
                incomingFreeWater,
            ],
            [
                actualET,
                runoff,
                nonFossilGroundwaterAbs,
                transportVolSnow,
                transportFreeWater,
            ],
            [storesAtBeginning],
            [storesAtEnd],
            "all stores (snow + interception + soil + groundwater), but except river/routing",
            True,
            self._modelTime.fulldate,
            threshold=1e-3,
        )

    def read_forcings(self):
        logger.info("Reading forcings for time %s", self._modelTime)
        self.meteo.read_forcings(self._modelTime)

    def update(self, report_water_balance=False):
        """Advance all sub-models by one day, in the fixed order meteo, landSurface,
        groundwater, routing; then save states at the end of a year, of the run, or
        (optionally) of a month.

        Called by DeterministicRunner.dynamic after read_forcings and before
        reporting.Reporting.report (deterministic_runner_with_arguments.py).

        What each step passes on that matters for lakes and reservoirs, on day t:
        1. meteo.update: forcing of day t (precipitation, temperature, referencePotET in
           m/day); it reads routing only for its averaging counter.
        2. landSurface.update: water demand of day t and its allocation. Surface water
           available for abstraction is routing.readAvlChannelStorage from the END OF
           DAY t-1 (one-day lag); the surface/groundwater split uses
           routing.avgDischarge and avgBaseflow up to day t-1. Produces
           actSurfaceWaterAbstract (m/day at the source cell), non-irrigation return
           flows (m3/day), landSurfaceRunoff and actualET (m/day) of day t. With
           using_qualloc = True the allocation is done by QUAlloc, which instead takes
           the whole routing.channelStorage of the end of day t-1 as available (no
           environmental-flow reserve, no 90 % cap); the outputs passed on are the same.
        3. groundwater.update: adds the riverbed infiltration of day t-1
           (routing.riverbedExchange), subtracts the groundwater abstraction of day t,
           computes baseflow of day t (m/day).
        4. routing.update: on day t adds runoff + baseflow, subtracts the abstraction,
           adds return flows, removes open-water evaporation, then moves the storage of
           lake and reservoir cells into the water bodies and calls the release rule
           (WaterBodies.update) once (accuTravelTime and simplifiedKinematicWave), then
           routes. Upstream river water reaches a water body during this routing and so
           enters its storage (and inflow) only on day t+1. Finally it computes the
           riverbed infiltration (used by groundwater on day t+1), readAvlChannelStorage
           (used by landSurface on day t+1) and the discharge statistics.
        Consequences: the release decided on day t is routed on day t, but can be
        abstracted downstream only from day t+1 on; the abstraction of day t at a
        reservoir cell is bounded by its storage at the end of day t-1 and is taken
        before the rule of day t.
        """
        logger.info("Updating model for time %s", self._modelTime)

        if report_water_balance:
            # excluding surface water bodies
            landWaterStoresAtBeginning = self.totalLandWaterStores()
            surfaceWaterStoresAtBeginning = self.totalSurfaceWaterStores()

        # INTERFACE: meteo uses routing.timestepsToAvgDischarge only (running means).
        self.meteo.update(self.routing, self._modelTime)
        # INTERFACE: land surface reads the routing and groundwater states of the END OF
        # THE PREVIOUS DAY (readAvlChannelStorage, avgDischarge, avgBaseflow,
        # storGroundwater, ...), and sets the abstractions and return flows of today.
        self.landSurface.update(
            self.meteo, self.groundwater, self.routing, self._modelTime
        )
        # INTERFACE: groundwater reads today's landSurface abstractions and recharge and
        # yesterday's routing.riverbedExchange.
        self.groundwater.update(self.landSurface, self.routing, self._modelTime)
        # INTERFACE: routing reads today's landSurface (actSurfaceWaterAbstract,
        # nonIrrReturnFlowVolume, landSurfaceRunoff, actualET), groundwater (baseflow,
        # abstractions) and meteo (referencePotET); lakes and reservoirs are updated
        # inside (routing.Routing.simple_update -> waterBodies.WaterBodies.update).
        # EXTENSION POINT: any new decision signal for a release scheme that is computed
        # by landSurface or groundwater (e.g. a demand map) is available to routing here
        # on the same day, because both run before routing.
        self.routing.update(
            self.landSurface, self.groundwater, self._modelTime, self.meteo
        )

        # save states at the end of the year or simulation, optionally also at the end of each month
        # (after routing, so the saved water-body states are the end-of-day states after
        # the release; [reportingOptions] save_monthly_end_states is not set in the RAWS
        # inis, so states are written on 31 December and on the last day of the run)
        save_monthly_end_states = self.save_monthly_end_states
        if (
            self._modelTime.isLastDayOfYear()
            or self._modelTime.isLastTimeStep()
            or (self._modelTime.isLastDayOfMonth() and save_monthly_end_states)
        ):
            logger.info(
                "Saving/dumping states to pcraster maps for time %s to the directory %s",
                self._modelTime,
                self._configuration.endStateDir,
            )
            self.dumpState(self._configuration.endStateDir)

        if report_water_balance:
            # excluding surface water bodies
            landWaterStoresAtEnd = self.totalLandWaterStores()
            surfaceWaterStoresAtEnd = self.totalSurfaceWaterStores()

            # land surface water balance check
            self.checkLandSurfaceWaterBalance(
                landWaterStoresAtBeginning, landWaterStoresAtEnd
            )

            # TODO: include water balance checks for the surface water part and for land surface and surface water combined
            # (the lake and reservoir balance is checked separately inside
            # waterBodies.WaterBodies.update; report_summary below only logs yearly
            # totals of the surface-water inputs and of waterBodyEvaporation)

            self.report_summary(
                landWaterStoresAtBeginning,
                landWaterStoresAtEnd,
                surfaceWaterStoresAtBeginning,
                surfaceWaterStoresAtEnd,
            )
