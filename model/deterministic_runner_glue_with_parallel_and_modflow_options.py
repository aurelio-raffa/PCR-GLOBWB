#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# PCR-GLOBWB (PCRaster Global Water Balance) Global Hydrological Model
#
# Copyright (C) 2016, Ludovicus P. H. (Rens) van Beek, Edwin H. Sutanudjaja, Yoshihide Wada,
# Joyce H. C. Bosmans, Niels Drost, Inge E. M. de Graaf, Kor de Jong, Patricia Lopez Lopez,
# Stefanie Pessenteiner, Oliver Schmitz, Menno W. Straatsma, Niko Wanders, Dominik Wisser,
# and Marc F. P. Bierkens,
# Faculty of Geosciences, Utrecht University, Utrecht, The Netherlands
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import sys
import datetime
import time

import pcraster as pcr
from pcraster.framework import DynamicModel
from pcraster.framework import DynamicFramework

from configuration import Configuration
from currTimeStep import ModelTime
from reporting import Reporting
from spinUp import SpinUp

from pcrglobwb import PCRGlobWB

import logging

logger = logging.getLogger(__name__)

import disclaimer


class DeterministicRunner(DynamicModel):

    def __init__(self, configuration, modelTime, initialState=None, system_argument=None, spinUpRun=False):
        DynamicModel.__init__(self)

        self.modelTime = modelTime
        # spinUpRun MUST be forwarded to PCRGlobWB: a transient (spinUpRun=False) run is the only
        # one that writes the per-month "pcrglobwb_files_for_<date>_are_ready.txt" sentinel that the
        # merging process waits on (pcrglobwb.py, at isLastDayOfMonth). If it is left as the
        # PCRGlobWB default of None, the guard `spinUpRun is not None and spinUpRun == False` is
        # never True, NO clone ever writes the sentinel, and the merging process hangs until its
        # wait_timeout backstop -- with every clone reported missing. (Matches the proven
        # deterministic_runner_with_arguments.py, which always passed spinUpRun here.)
        self.model = PCRGlobWB(configuration, modelTime, initialState, spinUpRun)
        self.reporting = Reporting(configuration, self.model, modelTime)

        # the model will set paramaters based on global pre-multipliers given in the argument:
        if system_argument != None: self.adusting_parameters(configuration, system_argument)

        # option to include merging processes for pcraster maps and netcdf files:
        self.with_merging = True
        if ('with_merging' in configuration.globalOptions.keys()) and (
                configuration.globalOptions['with_merging'] == "False"):
            self.with_merging = False

        # polling interval (seconds) for the month-end wait that is used ONLY when this
        # run is coupled online to MODFLOW. A real time.sleep() replaces the former
        # CPU-spinning poll loop so a waiting clone does not burn a core on the shared
        # node. Optionally overridable via globalOptions for future MODFLOW runs.
        self.barrier_poll_seconds = 10
        if 'barrier_poll_seconds' in configuration.globalOptions.keys():
            self.barrier_poll_seconds = int(configuration.globalOptions['barrier_poll_seconds'])

        # hard timeout (seconds) for the MODFLOW/merging waits below; a backstop for an
        # upstream process that hangs while still alive (no exit code for the launcher
        # supervisor to catch). Only reached in online-MODFLOW runs. Overridable via
        # globalOptions (wait_timeout_minutes).
        self.wait_timeout_seconds = 180 * 60
        if 'wait_timeout_minutes' in configuration.globalOptions.keys():
            self.wait_timeout_seconds = int(configuration.globalOptions['wait_timeout_minutes']) * 60

        # make the configuration available for the other method/function
        self.configuration = configuration

    def adusting_parameters(self, configuration, system_argument):

        # default neutral multipliers (overridden by system_argument or prefactorOptions if present)
        multiplier_for_minSoilDepthFrac = 1.0
        multiplier_for_kSat = 0.0  # log scale: 10^0 = 1
        multiplier_for_recessionCoeff = 0.0  # log scale: 10^0 = 1
        multiplier_for_storCap = 1.0
        multiplier_for_degreeDayFactor = 1.0
        self.multiplier_for_refPotET = 1.0

        # global pre-multipliers given in the argument:
        if len(system_argument) > 4:
            logger.info("Adjusting some model parameters based on given values in the system argument.")

            # pre-multipliers for minSoilDepthFrac, kSat, recessionCoeff, storCap and degreeDayFactor
            multiplier_for_minSoilDepthFrac = float(system_argument[4])  # linear scale
            # Note that this one does NOT work for the changing WMIN or Joyce land cover options.
            multiplier_for_kSat = float(system_argument[5])  # log scale
            multiplier_for_recessionCoeff = float(system_argument[6])  # log scale
            multiplier_for_storCap = float(system_argument[7])  # linear scale
            multiplier_for_degreeDayFactor = float(system_argument[8])  # linear scale

            # pre-multiplier for the reference potential ET
            self.multiplier_for_refPotET = float(system_argument[9])  # linear scale

        # it is also possible to define prefactors via the ini/configuration file: 
        # - this will be overwrite any previous given pre-multipliers
        if 'prefactorOptions' in configuration.allSections:
            logger.info("Adjusting some model parameters based on given values in the ini/configuration file.")

            self.multiplier_for_refPotET = float(configuration.prefactorOptions['linear_multiplier_for_refPotET'])
            # linear scale  # Note that this one does NOT work for the changing WMIN or Joyce land cover options.
            multiplier_for_degreeDayFactor = float(
                configuration.prefactorOptions['linear_multiplier_for_degreeDayFactor']
            )  # linear scale
            multiplier_for_minSoilDepthFrac = float(
                configuration.prefactorOptions['linear_multiplier_for_minSoilDepthFrac']
            )  # linear scale
            multiplier_for_kSat = float(configuration.prefactorOptions['log_10_multiplier_for_kSat'])  # log scale
            multiplier_for_storCap = float(
                configuration.prefactorOptions['linear_multiplier_for_storCap']
            )  # linear scale
            multiplier_for_recessionCoeff = float(
                configuration.prefactorOptions['log_10_multiplier_for_recessionCoeff']
            )  # log scale

        # saving global pre-multipliers to the log file:
        msg = "\n"
        msg += "\n"
        msg += "Multiplier values used: " + "\n"
        msg += "For minSoilDepthFrac           : " + str(multiplier_for_minSoilDepthFrac) + "\n"
        msg += "For kSat (log-scale)           : " + str(multiplier_for_kSat) + "\n"
        msg += "For recessionCoeff (log-scale) : " + str(multiplier_for_recessionCoeff) + "\n"
        msg += "For storCap                    : " + str(multiplier_for_storCap) + "\n"
        msg += "For degreeDayFactor            : " + str(multiplier_for_degreeDayFactor) + "\n"
        msg += "For refPotET                   : " + str(self.multiplier_for_refPotET) + "\n"
        logger.info(msg)
        # - also to a txt file 
        f = open(
            "multiplier.txt",
            "w"
        )
        # this will be stored in the "map" folder of the 'outputDir' (as we set the current working directory to
        # this "map" folder, see configuration.py)
        f.write(msg)
        f.close()

        # set parameter "recessionCoeff" based on the given pre-multiplier
        # - also saving the adjusted parameter maps to pcraster files
        # - these will be stored in the "map" folder of the 'outputDir' (as we set the current working directory to
        #   this "map" folder, see configuration.py)
        # "recessionCoeff"
        # minimum value is zero and using log-scale
        self.model.groundwater.recessionCoeff = pcr.max(0.0, (
                    10 ** (multiplier_for_recessionCoeff)) * self.model.groundwater.recessionCoeff)
        self.model.groundwater.recessionCoeff = pcr.min(1.0, self.model.groundwater.recessionCoeff)
        # report the map
        pcr.report(self.model.groundwater.recessionCoeff, "recessionCoeff.map")

        # set parameters "kSat", "storCap", "minSoilDepthFrac", and "degreeDayFactor" based on the given pre-multipliers
        for coverType in self.model.landSurface.coverTypes:

            # "degreeDayFactor"
            self.model.landSurface.landCoverObj[coverType].degreeDayFactor = pcr.max(
                0.0,
                multiplier_for_degreeDayFactor * self.model.landSurface.landCoverObj[coverType].degreeDayFactor
            )
            # report the map
            pcraster_filename = "degreeDayFactor" + "_" + coverType + ".map"
            pcr.report(self.model.landSurface.landCoverObj[coverType].degreeDayFactor, pcraster_filename)

            # "kSat" and "storCap" for 2 layer model
            if self.model.landSurface.numberOfSoilLayers == 2:
                # "kSat"
                # minimum value is zero and using-log-scale
                self.model.landSurface.landCoverObj[coverType].parameters.kSatUpp = \
                    pcr.max(0.0, (10 ** (multiplier_for_kSat)) * self.model.landSurface.landCoverObj[
                        coverType].parameters.kSatUpp)
                self.model.landSurface.landCoverObj[coverType].parameters.kSatLow = \
                    pcr.max(0.0, (10 ** (multiplier_for_kSat)) * self.model.landSurface.landCoverObj[
                        coverType].parameters.kSatLow)
                # report the maps
                pcraster_filename = "kSatUpp" + "_" + coverType + ".map"
                pcr.report(self.model.landSurface.landCoverObj[coverType].parameters.kSatUpp, pcraster_filename)
                pcraster_filename = "kSatLow" + "_" + coverType + ".map"
                pcr.report(self.model.landSurface.landCoverObj[coverType].parameters.kSatLow, pcraster_filename)

                # "storCap"
                # minimum value is zero
                self.model.landSurface.landCoverObj[coverType].parameters.storCapUpp = pcr.max(
                    0.0,
                    multiplier_for_storCap * \
                    self.model.landSurface.landCoverObj[coverType].parameters.storCapUpp
                )
                self.model.landSurface.landCoverObj[coverType].parameters.storCapLow = pcr.max(
                    0.0,
                    multiplier_for_storCap * \
                    self.model.landSurface.landCoverObj[coverType].parameters.storCapLow
                )
                # report the maps
                pcraster_filename = "storCapUpp" + "_" + coverType + ".map"
                pcr.report(self.model.landSurface.landCoverObj[coverType].parameters.storCapUpp, pcraster_filename)
                pcraster_filename = "storCapLow" + "_" + coverType + ".map"
                pcr.report(self.model.landSurface.landCoverObj[coverType].parameters.storCapLow, pcraster_filename)

            # "kSat" and "storCap" for 3 layer model
            if self.model.landSurface.numberOfSoilLayers == 3:
                # "kSat"
                # minimum value is zero and using-log-scale
                self.model.landSurface.landCoverObj[coverType].parameters.kSatUpp000005 = pcr.max(
                    0.0,
                    (
                            10 ** multiplier_for_kSat
                    ) * self.model.landSurface.landCoverObj[coverType].parameters.kSatUpp000005)
                self.model.landSurface.landCoverObj[coverType].parameters.kSatUpp005030 = pcr.max(
                    0.0,
                    (
                            10 ** multiplier_for_kSat
                    ) * self.model.landSurface.landCoverObj[coverType].parameters.kSatUpp005030)
                self.model.landSurface.landCoverObj[coverType].parameters.kSatLow030150 = pcr.max(
                    0.0,
                    (
                            10 ** multiplier_for_kSat
                    ) * self.model.landSurface.landCoverObj[coverType].parameters.kSatLow030150)
                # report the maps
                pcraster_filename = "kSatUpp000005" + "_" + coverType + ".map"
                pcr.report(self.model.landSurface.landCoverObj[coverType].parameters.kSatUpp000005, pcraster_filename)
                pcraster_filename = "kSatUpp005030" + "_" + coverType + ".map"
                pcr.report(self.model.landSurface.landCoverObj[coverType].parameters.kSatUpp005030, pcraster_filename)
                pcraster_filename = "kSatLow030150" + "_" + coverType + ".map"
                pcr.report(self.model.landSurface.landCoverObj[coverType].parameters.kSatLow030150, pcraster_filename)

                # "storCap"
                # minimum value is zero
                self.model.landSurface.landCoverObj[coverType].parameters.storCapUpp000005 = pcr.max(
                    0.0,
                    multiplier_for_storCap * self.model.landSurface.landCoverObj[coverType].parameters.storCapUpp000005)
                self.model.landSurface.landCoverObj[coverType].parameters.storCapUpp005030 = pcr.max(
                    0.0,
                    multiplier_for_storCap * self.model.landSurface.landCoverObj[coverType].parameters.storCapUpp005030)
                self.model.landSurface.landCoverObj[coverType].parameters.storCapLow030150 = pcr.max(
                    0.0,
                    multiplier_for_storCap * self.model.landSurface.landCoverObj[coverType].parameters.storCapLow030150)
                # report the maps
                pcraster_filename = "storCapUpp000005" + "_" + coverType + ".map"
                pcr.report(
                    self.model.landSurface.landCoverObj[coverType].parameters.storCapUpp000005,
                    pcraster_filename
                )
                pcraster_filename = "storCapUpp005030" + "_" + coverType + ".map"
                pcr.report(
                    self.model.landSurface.landCoverObj[coverType].parameters.storCapUpp005030,
                    pcraster_filename
                )
                pcraster_filename = "storCapLow030150" + "_" + coverType + ".map"
                pcr.report(
                    self.model.landSurface.landCoverObj[coverType].parameters.storCapLow030150,
                    pcraster_filename
                )

            # re-calculate rootZoneWaterStorageCap as the consequence of the modification of "storCap"
            # This is WMAX in the oldcalc script.
            if self.model.landSurface.numberOfSoilLayers == 2:
                self.model.landSurface.landCoverObj[coverType].parameters.rootZoneWaterStorageCap = \
                self.model.landSurface.landCoverObj[coverType].parameters.storCapUpp + \
                self.model.landSurface.landCoverObj[coverType].parameters.storCapLow
            if self.model.landSurface.numberOfSoilLayers == 3:
                self.model.landSurface.landCoverObj[coverType].parameters.rootZoneWaterStorageCap = \
                self.model.landSurface.landCoverObj[coverType].parameters.storCapUpp000005 + \
                self.model.landSurface.landCoverObj[coverType].parameters.storCapUpp005030 + \
                self.model.landSurface.landCoverObj[coverType].parameters.storCapLow030150
            # report the map
            pcraster_filename = "rootZoneWaterStorageCap" + "_" + coverType + ".map"
            pcr.report(
                self.model.landSurface.landCoverObj[coverType].parameters.rootZoneWaterStorageCap,
                pcraster_filename
            )

            # "minSoilDepthFrac"
            if multiplier_for_minSoilDepthFrac != 1.0:
                # minimum value is zero
                self.model.landSurface.landCoverObj[coverType].minSoilDepthFrac = pcr.max(
                    0.0,
                    multiplier_for_minSoilDepthFrac * self.model.landSurface.landCoverObj[coverType].minSoilDepthFrac)
                # for minSoilDepthFrac - values will be limited by maxSoilDepthFrac
                self.model.landSurface.landCoverObj[coverType].minSoilDepthFrac = pcr.min(
                    self.model.landSurface.landCoverObj[coverType].minSoilDepthFrac,
                    self.model.landSurface.landCoverObj[coverType].maxSoilDepthFrac
                )
                # maximum value is 1.0
                self.model.landSurface.landCoverObj[coverType].minSoilDepthFrac = pcr.min(
                    1.0,
                    self.model.landSurface.landCoverObj[coverType].minSoilDepthFrac)
                # report the map
                pcraster_filename = "minSoilDepthFrac" + "_" + coverType + ".map"
                pcr.report(self.model.landSurface.landCoverObj[coverType].minSoilDepthFrac, pcraster_filename)

                # re-calculate arnoBeta (as the consequence of the modification of minSoilDepthFrac)
                self.model.landSurface.landCoverObj[coverType].arnoBeta = pcr.max(
                    0.001,
                    (
                            self.model.landSurface.landCoverObj[coverType].maxSoilDepthFrac - 1.
                    ) / (
                            1. - self.model.landSurface.landCoverObj[coverType].minSoilDepthFrac
                    ) + self.model.landSurface.landCoverObj[coverType].parameters.orographyBeta - 0.01
                )
                self.model.landSurface.landCoverObj[coverType].arnoBeta = pcr.cover(
                    pcr.max(
                        0.001,
                        self.model.landSurface.landCoverObj[coverType].arnoBeta
                    ),
                    0.001
                )
                # report the map
                pcraster_filename = "arnoBeta" + "_" + coverType + ".map"
                pcr.report(self.model.landSurface.landCoverObj[coverType].arnoBeta, pcraster_filename)

                # re-calculate rootZoneWaterStorageMin (as the consequence of the modification of minSoilDepthFrac)
                # This is WMIN in the oldcalc script.
                # WMIN (unit: m): minimum local soil water capacity within the grid-cell
                self.model.landSurface.landCoverObj[coverType].rootZoneWaterStorageMin = \
                self.model.landSurface.landCoverObj[coverType].minSoilDepthFrac * \
                self.model.landSurface.landCoverObj[coverType].parameters.rootZoneWaterStorageCap
                # report the map
                pcraster_filename = "rootZoneWaterStorageMin" + "_" + coverType + ".map"
                pcr.report(self.model.landSurface.landCoverObj[coverType].rootZoneWaterStorageMin, pcraster_filename)

                # re-calculate rootZoneWaterStorageRange (as the consequence of the modification of
                # rootZoneWaterStorageRange and minSoilDepthFrac)
                # WMAX - WMIN (unit: m)
                self.model.landSurface.landCoverObj[coverType].rootZoneWaterStorageRange = \
                self.model.landSurface.landCoverObj[coverType].parameters.rootZoneWaterStorageCap - \
                self.model.landSurface.landCoverObj[coverType].rootZoneWaterStorageMin
                # report the map
                pcraster_filename = "rootZoneWaterStorageRange" + "_" + coverType + ".map"
                pcr.report(self.model.landSurface.landCoverObj[coverType].rootZoneWaterStorageRange, pcraster_filename)

    def initial(self):
        pass

    def dynamic(self):

        # re-calculate current model time using current pcraster timestep value
        self.modelTime.update(self.currentTimeStep())

        # read model forcing (will pick up current model time from model time object)
        self.model.read_forcings()

        # adjust the reference potential ET according to the given pre-multiplier
        self.model.meteo.referencePotET = self.model.meteo.referencePotET * self.multiplier_for_refPotET

        # update model (will pick up current model time from model time object)
        # - for a run coupled to MODFLOW, water balance checks are not valid due to lateral flow. 
        if self.configuration.online_coupling_between_pcrglobwb_and_modflow:
            self.model.update(report_water_balance=False)
        else:
            self.model.update(report_water_balance=True)

        # do any needed reporting for this time step        
        self.reporting.report()

        # At the last day of the month, a clone only has to PAUSE here when there is a
        # genuine cross-process data dependency on the global step. That is the case
        # ONLY for an online MODFLOW coupling: on day 1 of the next month the clone
        # reads the MODFLOW-computed groundwater fields (relativeGroundwaterHead,
        # storGroundwater, baseflow) back into its own state -- see
        # groundwater.update_with_MODFLOW() -- so it must block until MODFLOW and the
        # related merging for this month are on disk.
        #
        # For a pure merging run (with_merging = True, online_coupling = False) NOTHING
        # produced by the merger is ever read back into the clone: the merging is output
        # aggregation only. The clone has already written its own month-end sentinel
        # ("<clone>/maps/pcrglobwb_files_for_<date>_are_ready.txt", in PCRGlobWB.update);
        # it therefore does NOT block here and proceeds straight to the next month. The
        # merging process consumes that sentinel and runs ASYNCHRONOUSLY, lagging behind
        # the compute front instead of forcing a hard per-month barrier across all clones.
        # (The launcher's supervisor still waits for the merger to finish before the job
        # exits, so no output is lost.)
        if self.modelTime.isLastDayOfMonth() and \
                self.configuration.online_coupling_between_pcrglobwb_and_modflow:

            # Block until the MODFLOW output and then the merged files for this month are on
            # disk. Each wait polls with a real sleep (no CPU spin) and gives up after a hard
            # timeout, so an upstream process that wedges while still alive fails this clone
            # fast (the launcher supervisor then tears the run down) rather than hanging the
            # whole job to its wall-clock limit.
            self._wait_until_ready(self.check_modflow_status, "MODFLOW output")
            self._wait_until_ready(self.check_merging_status, "merged files")

    def _wait_until_ready(self, status_check, description):
        # Poll status_check() (returns True once the awaited sentinel files exist) with a
        # real sleep, aborting with a non-zero exit if it is still not ready after
        # wait_timeout_seconds. The timeout is a backstop for an upstream process that
        # hangs while still alive (no exit code for the launcher supervisor to catch).
        wait_start = datetime.datetime.now()
        self.count_check = 0
        ready = False
        while ready == False:
            ready = status_check()
            if ready == False:
                waited = (datetime.datetime.now() - wait_start).total_seconds()
                if waited > self.wait_timeout_seconds:
                    logger.error("Timed out after %.0f min waiting for %s for %s; aborting clone.",
                                 waited / 60.0, description, str(self.modelTime.fulldate))
                    sys.exit(1)
                time.sleep(self.barrier_poll_seconds)

    def check_modflow_status(self):

        status_file = str(
            self.configuration.main_output_directory
        ) + "/modflow/transient/maps/modflow_files_for_" + str(
            self.modelTime.fulldate
        ) + "_are_ready.txt"
        msg = 'Waiting for the file: ' + status_file
        if self.count_check == 1: logger.info(msg)
        if self.count_check < 7:
            # ~ logger.debug(msg)			# INACTIVATE THIS AS THIS MAKE A HUGE DEBUG (dbg) FILE
            self.count_check += 1
        status = os.path.exists(status_file)
        if status == False: return status
        if status: self.count_check = 0
        return status

    def check_merging_status(self):

        status_file = str(
            self.configuration.main_output_directory
        ) + "/global/maps/merged_files_for_" + str(
            self.modelTime.fulldate
        ) + "_are_ready.txt"
        msg = 'Waiting for the file: ' + status_file
        if self.count_check == 1: logger.info(msg)
        if self.count_check < 7:
            # ~ logger.debug(msg)			# INACTIVATE THIS AS THIS MAKE A HUGE DEBUG (dbg) FILE
            self.count_check += 1
        status = os.path.exists(status_file)
        if status == False: return status
        if status: self.count_check = 0
        return status


def main():
    # get the full path of configuration/ini file given in the system argument
    iniFileName = os.path.abspath(sys.argv[1])

    # debug option
    debug_mode = False
    if len(sys.argv) > 2:
        if sys.argv[2] == "debug" or sys.argv[2] == "debug_parallel" or sys.argv[
            2] == "debug-parallel": debug_mode = True

    # object to handle configuration/ini file
    configuration = Configuration(
        iniFileName=iniFileName,
        debug_mode=debug_mode,
        no_modification=False
    )

    # parallel option
    this_run_is_part_of_a_set_of_parallel_run = False
    if len(sys.argv) > 2:
        if sys.argv[2] == "parallel" or sys.argv[2] == "debug_parallel" or sys.argv[
            2] == "debug-parallel": this_run_is_part_of_a_set_of_parallel_run = True

    # for a non parallel run (usually 30min), a specific directory given in the system argument (sys.argv[3]) will be
    # assigned for a given parameter combination:
    if this_run_is_part_of_a_set_of_parallel_run == False:
        # modfiying 'outputDir' (based on the given system argument)
        configuration.globalOptions['outputDir'] += "/" + str(sys.argv[3]) + "/"

        # for a parallel run (usually 5min), we assign a specific directory based on the clone number/code:
    if this_run_is_part_of_a_set_of_parallel_run:
        # modfiying outputDir, clone-map and landmask (based on the given system arguments)
        clone_code = str(sys.argv[3])
        configuration.globalOptions['outputDir'] += "/" + clone_code + "/"
        configuration.globalOptions['cloneMap'] = configuration.globalOptions['cloneMap'] % (clone_code)
        configuration.globalOptions['landmask'] = configuration.globalOptions['landmask'] % (clone_code)

    # set configuration
    configuration.set_configuration(system_arguments=sys.argv)

    # timeStep info: year, month, day, doy, hour, etc
    currTimeStep = ModelTime()

    # object for spin_up
    spin_up = SpinUp(configuration)

    # spinning-up 
    noSpinUps = int(configuration.globalOptions['maxSpinUpsInYears'])
    initial_state = None
    if noSpinUps > 0:

        logger.info('Spin-Up #Total Years: ' + str(noSpinUps))

        spinUpRun = 0;
        has_converged = False
        while spinUpRun < noSpinUps and has_converged == False:
            spinUpRun += 1
            currTimeStep.getStartEndTimeStepsForSpinUp(
                configuration.globalOptions['startTime'],
                spinUpRun, noSpinUps)
            logger.info('Spin-Up Run No. ' + str(spinUpRun))
            deterministic_runner = DeterministicRunner(configuration, currTimeStep, initial_state, sys.argv, spinUpRun=True)

            all_state_begin = deterministic_runner.model.getAllState()

            dynamic_framework = DynamicFramework(deterministic_runner, currTimeStep.nrOfTimeSteps)
            dynamic_framework.setQuiet(True)
            dynamic_framework.run()

            all_state_end = deterministic_runner.model.getAllState()

            has_converged = spin_up.checkConvergence(all_state_begin, all_state_end, spinUpRun,
                                                     deterministic_runner.model.routing.cellArea)

            initial_state = deterministic_runner.model.getState()
    #
    # Running the deterministic_runner (excluding DA scheme)
    currTimeStep.getStartEndTimeSteps(
        configuration.globalOptions['startTime'],
        configuration.globalOptions['endTime']
    )

    logger.info('Transient simulation run started.')
    deterministic_runner = DeterministicRunner(configuration, currTimeStep, initial_state, sys.argv, spinUpRun=False)

    dynamic_framework = DynamicFramework(deterministic_runner, currTimeStep.nrOfTimeSteps)
    dynamic_framework.setQuiet(True)
    dynamic_framework.run()


if __name__ == '__main__':
    # print disclaimer
    disclaimer.print_disclaimer(with_logger=True)
    sys.exit(main())
