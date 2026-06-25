#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# PCR-GLOBWB (PCRaster Global Water Balance) Global Hydrological Model
#
# Copyright (C) 2016, Edwin H. Sutanudjaja, Rens van Beek, Niko Wanders, Yoshihide Wada, 
# Joyce H. C. Bosmans, Niels Drost, Ruud J. van der Ent, Inge E. M. de Graaf, Jannis M. Hoch, 
# Kor de Jong, Derek Karssenberg, Patricia López López, Stefanie Peßenteiner, Oliver Schmitz, 
# Menno W. Straatsma, Ekkamol Vannametee, Dominik Wisser, and Marc F. P. Bierkens
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
import glob

import pcraster as pcr
from pcraster.framework import DynamicModel
from pcraster.framework import DynamicFramework

from configuration import Configuration
from currTimeStep import ModelTime

try:
    from reporting_for_modflow import Reporting
except:
    pass

try:
    from modflow import ModflowCoupling
except:
    pass

import virtualOS as vos

import logging

logger = logging.getLogger(__name__)

import disclaimer


def resolve_global_clone_map(configuration):
    """Return a valid whole-domain clone map for the merging ("global") process.

    The parallel runner shares ONE ini across every spawned process. Per-tile workers substitute their
    clone code into ``cloneMap``'s ``%s`` (see deterministic_runner_glue_with_parallel_and_modflow_
    options.py), but the merging/global process is launched *without* a clone code, so its ``cloneMap``
    still holds the literal template (e.g. ``.../clone_%s.map``). That is not a real PCRaster map, so
    ``pcr.setclone`` raises ``TypeError: Cannot open '.../clone_%s.map'`` and the merging process dies --
    after which the per-tile processes deadlock at the monthly merge barrier and the job is eventually
    killed. (The bare ``wait`` in the parallel runner means none of this changes the job's exit code.)

    To avoid that, when ``cloneMap`` is still a per-tile template we resolve a whole-domain clone, in
    priority order:
      1. the ``GLOBAL_CLONE_MAP`` environment variable (explicit override; e.g. set via the job's
         ``--env``),
      2. a ``globalCloneMap`` key under ``[globalOptions]`` of the ini (explicit override),
      3. the routing ``lddMap`` -- a global map at the run resolution that always exists. This is exactly
         the choice the official MODFLOW merging config makes (``[globalMergingAndModflowOptions]``
         ``cloneMap = .../lddsound_05min.map``, ``landmask = None``).

    A ``cloneMap`` that is already concrete (no ``%s``) is returned unchanged.
    """
    clone = configuration.cloneMap
    if "%s" not in str(clone):
        return clone

    input_dir = configuration.globalOptions['inputDir']

    env_override = os.environ.get('GLOBAL_CLONE_MAP', '').strip()
    if env_override:
        logger.info("Merging process: using the GLOBAL_CLONE_MAP override as the clone map: %s", env_override)
        return vos.getFullPath(env_override, input_dir)

    ini_override = str(configuration.globalOptions.get('globalCloneMap', '')).strip()
    if ini_override and ini_override.lower() != 'none' and "%s" not in ini_override:
        logger.info("Merging process: using [globalOptions] globalCloneMap as the clone map: %s", ini_override)
        return vos.getFullPath(ini_override, input_dir)

    ldd_map = vos.getFullPath(configuration.routingOptions['lddMap'], input_dir)
    logger.warning("Merging process: cloneMap is a per-tile template (%s) with no clone code to "
                   "substitute; falling back to the routing lddMap as the whole-domain clone: %s",
                   clone, ldd_map)
    return ldd_map


class DeterministicRunner(DynamicModel):

    def __init__(self, configuration, modelTime):
        DynamicModel.__init__(self)

        # model time object
        self.modelTime = modelTime

        # make the configuration available for the other method/function
        self.configuration = configuration

        # polling interval (seconds) used while waiting for the per-clone month-end
        # sentinels. A real time.sleep() replaces the former CPU-spinning poll so the
        # merging process does not burn a core on the shared node while it waits.
        # Overridable via globalOptions.
        self.barrier_poll_seconds = 10
        if 'barrier_poll_seconds' in self.configuration.globalOptions.keys():
            self.barrier_poll_seconds = int(self.configuration.globalOptions['barrier_poll_seconds'])

        # hard timeout (seconds) for waiting on the per-clone month-end sentinels. This is
        # a BACKSTOP for a clone that hangs while still ALIVE (no exit code for the launcher
        # supervisor to catch); on timeout the merging process exits non-zero so the
        # supervisor tears the run down instead of the job hanging to its wall-clock limit.
        # A crashed clone (process exits) is caught by the supervisor in seconds and never
        # reaches this timeout. Overridable via globalOptions (wait_timeout_minutes).
        self.wait_timeout_seconds = 180 * 60
        if 'wait_timeout_minutes' in self.configuration.globalOptions.keys():
            self.wait_timeout_seconds = int(self.configuration.globalOptions['wait_timeout_minutes']) * 60

        # how often (seconds) to emit a progress heartbeat while blocked on the per-clone
        # month-end sentinels. The merger is the ONLY process that knows which clone's
        # sentinel is missing, so this names the laggard(s) in real time instead of only at
        # the final wait_timeout abort -- a wedge is then visible in the log within minutes.
        # Overridable via globalOptions (wait_progress_report_minutes). Default: every 15 min.
        self.progress_report_seconds = 15 * 60
        if 'wait_progress_report_minutes' in self.configuration.globalOptions.keys():
            self.progress_report_seconds = int(self.configuration.globalOptions['wait_progress_report_minutes']) * 60

        # indicating whether this run includes modflow or merging processes
        # - Only the "Global" and "part_one" runs include modflow or merging processes 
        self.include_merging_or_modflow = True
        if self.configuration.globalOptions['cloneAreas'] == "part_two": self.include_merging_or_modflow = False

        if self.include_merging_or_modflow:

            # netcdf merging options
            # Be tolerant of a missing/incomplete [mergingOutputOptions] section so a
            # menial config gap cannot crash the merging ("Global") process at startup
            # and silently cost us the consolidated output. Missing scalar keys fall
            # back to sensible defaults; missing report-type keys default to "None"
            # (that report type is simply not merged). Per-tile NetCDF outputs written
            # by the worker processes are independent of these options and never lost.
            nc_report_list = [
                "outDailyTotNC", "outMonthTotNC", "outMonthAvgNC", "outMonthEndNC", "outMonthMaxNC",
                "outAnnuaTotNC", "outAnnuaAvgNC", "outAnnuaEndNC", "outAnnuaMaxNC"
            ]

            merging_defaults = {"formatNetCDF": "NETCDF4",
                                "zlib": "True",
                                "delete_unmerged_pcraster_maps": "False"}
            for nc_report_type in nc_report_list:
                merging_defaults[nc_report_type] = "None"

            merging_options = dict(getattr(self.configuration, "mergingOutputOptions", {}) or {})
            missing_keys = [k for k in merging_defaults if k not in merging_options]
            if not hasattr(self.configuration, "mergingOutputOptions") or missing_keys:
                logger.warning(
                    "Configuration is missing the [mergingOutputOptions] section or "
                    "keys %s; falling back to defaults. No per-tile output is lost, but "
                    "report types left as 'None' will NOT be merged into global files. "
                    "Add a [mergingOutputOptions] section to your INI template to "
                    "control merging explicitly." % str(missing_keys))
            for k, v in merging_defaults.items():
                merging_options.setdefault(k, v)
            self.configuration.mergingOutputOptions = merging_options

            self.netcdf_format = self.configuration.mergingOutputOptions['formatNetCDF']
            self.zlib_option = self.configuration.mergingOutputOptions['zlib']

            # output files/variables that will be merged
            for nc_report_type in nc_report_list:
                vars(self)[nc_report_type] = self.configuration.mergingOutputOptions[nc_report_type]

        # model and reporting objects
        # - Note that both are still needed even 
        if self.configuration.online_coupling_between_pcrglobwb_and_modflow:
            self.model = ModflowCoupling(configuration, modelTime)
            self.reporting = Reporting(configuration, self.model, modelTime)
        else:
            # somehow you need to set the clone map (as the dynamic framework needs it and the "self.model" is not made)
            # NOTE: this is the merging/global process. Its cloneMap is the per-tile template (no clone code
            # was substituted), so resolve a valid whole-domain clone instead of the literal '...clone_%s.map'.
            pcr.setclone(resolve_global_clone_map(configuration))

    def initial(self):

        # get or prepare the initial condition for groundwater head 
        if self.configuration.online_coupling_between_pcrglobwb_and_modflow:
            self.model.get_initial_heads()

    def dynamic(self):

        # re-calculate current model time using current pcraster timestep value
        self.modelTime.update(self.currentTimeStep())

        # update/calculate model and daily merging, and report ONLY at the last day of the month
        if self.modelTime.isLastDayOfMonth():

            # wait until all pcrglobwb model runs are done
            pcrglobwb_is_ready = False
            self.count_check = 0
            wait_start = datetime.datetime.now()
            last_progress_report = wait_start
            while pcrglobwb_is_ready == False:
                # poll with a real sleep instead of CPU-spinning at fixed clock seconds
                pcrglobwb_is_ready = self.check_pcrglobwb_status()
                if pcrglobwb_is_ready == False:
                    now = datetime.datetime.now()
                    waited = (now - wait_start).total_seconds()
                    # periodic heartbeat: name the clone(s) we are still blocked on so a
                    # wedge is visible in the log in real time, not only at the final abort.
                    if (now - last_progress_report).total_seconds() >= self.progress_report_seconds:
                        self.report_clones_still_pending(waited, level=logging.WARNING)
                        last_progress_report = now
                    if waited > self.wait_timeout_seconds:
                        # No exit code to observe: a clone is wedged ALIVE (it never wrote
                        # its sentinel and never crashed, or the launcher supervisor would
                        # already have caught it). Name the culprit clone(s) -- the merger is
                        # the only process that knows which sentinel is missing -- then fail
                        # fast so the supervisor tears the run down instead of the job hanging
                        # to its scheduler wall-clock limit.
                        logger.error("Timed out after %.0f min waiting for per-clone month-end "
                                     "files for %s; a clone appears wedged. Aborting merging.",
                                     waited / 60.0, str(self.modelTime.fulldate))
                        self.report_clones_still_pending(waited, level=logging.ERROR, detailed=True)
                        sys.exit(1)
                    time.sleep(self.barrier_poll_seconds)

            # merging netcdf files at daily resolution
            start_date = '%04i-%02i-01' % (self.modelTime.year,
                                           self.modelTime.month)  # TODO: Make it flexible for a run starting not on the 1st January.
            end_date = self.modelTime.fulldate
            self.merging_netcdf_files("outDailyTotNC", start_date, end_date)

            # for runs with modflow
            if self.configuration.online_coupling_between_pcrglobwb_and_modflow:

                # merging pcraster maps that are needed for MODFLOW calculation
                msg = "Merging pcraster map files that are needed for the MODFLOW calculation."
                logger.info(msg)
                cmd = 'python ' + self.configuration.path_of_this_module + "/merge_pcraster_maps.py " + str(
                    self.modelTime.fulldate) + " " + \
                      str(self.configuration.main_output_directory) + "/ maps 8 " + \
                      str("Global")
                vos.cmd_line(cmd, using_subprocess=False)

                # cleaning up unmerged files (not tested yet)
                clean_up_pcraster_maps = False
                if self.configuration.mergingOutputOptions[
                    "delete_unmerged_pcraster_maps"] == "True": clean_up_pcraster_maps = True  # TODO: FIXME: This is NOT working yet.
                if clean_up_pcraster_maps:
                    files_to_be_removed = glob.glob(str(self.configuration.main_output_directory) + "/M*/maps/*" + str(
                        self.modelTime.fulldate) + "*")
                    for f in files_to_be_removed:
                        print(f)
                        os.remove(f)

                # update MODFLOW model (It will pick up current model time from the modelTime object)
                self.model.update()
                # reporting is only done at the end of the month
                self.reporting.report()

        # merging initial conditions (pcraster maps) of PCR-GLOBWB
        # ~ if self.modelTime.isLastDayOfYear():
        if self.modelTime.isLastDayOfMonth():

            msg = "Merging pcraster map files belonging to initial conditions."
            logger.info(msg)
            cmd = 'python ' + self.configuration.path_of_this_module + "/merge_pcraster_maps.py " + str(
                self.modelTime.fulldate) + " " + \
                  str(self.configuration.main_output_directory) + "/ states 8 " + \
                  str("Global")
            vos.cmd_line(cmd, using_subprocess=False)

            # cleaning up unmerged files (not tested yet)
            clean_up_pcraster_maps = False
            if self.configuration.mergingOutputOptions[
                "delete_unmerged_pcraster_maps"] == "True": clean_up_pcraster_maps = True  # TODO: FIXME: This is NOT working yet.
            if clean_up_pcraster_maps:
                files_to_be_removed = glob.glob(
                    str(self.configuration.main_output_directory) + "/M*/states/*" + str(self.modelTime.fulldate) + "*")
                for f in files_to_be_removed:
                    print(f)
                    os.remove(f)

        # monthly and annual merging
        if self.modelTime.isLastDayOfYear():
            # merging netcdf files at monthly resolutions
            start_date = '%04i-01-31' % (
                self.modelTime.year)  # TODO: Make it flexible for a run starting not on the 1st January.
            self.merging_netcdf_files("outMonthTotNC", start_date, end_date)
            self.merging_netcdf_files("outMonthAvgNC", start_date, end_date)
            self.merging_netcdf_files("outMonthEndNC", start_date, end_date)
            self.merging_netcdf_files("outMonthMaxNC", start_date, end_date)

            # merging netcdf files at annual resolutions
            start_date = '%04i-12-31' % (
                self.modelTime.year)  # TODO: Make it flexible for a run starting not on the 1st January.
            end_date = self.modelTime.fulldate
            self.merging_netcdf_files("outAnnuaTotNC", start_date, end_date)
            self.merging_netcdf_files("outAnnuaAvgNC", start_date, end_date)
            self.merging_netcdf_files("outAnnuaEndNC", start_date, end_date)
            self.merging_netcdf_files("outAnnuaMaxNC", start_date, end_date)

        # make an empty file indicating that merging process is done
        if self.modelTime.isLastDayOfMonth() or self.modelTime.isLastDayOfYear():

            outputDirectory = str(self.configuration.main_output_directory) + "/global/maps/"
            filename = outputDirectory + "/merged_files_for_" + str(self.modelTime.fulldate) + "_are_ready.txt"
            if os.path.exists(filename): os.remove(filename)
            open(filename, "w").close()

    def merging_netcdf_files(self, nc_report_type, start_date, end_date, max_number_of_cores=20):

        if str(vars(self)[nc_report_type]) != "None":
            netcdf_files_that_will_be_merged = vars(self)[nc_report_type]

            msg = "Merging netcdf files for the files/variables: " + netcdf_files_that_will_be_merged
            logger.info(msg)

            cmd = 'python ' + self.configuration.path_of_this_module + "/merge_netcdf.py " + str(
                self.configuration.main_output_directory) + " " + \
                  str(self.configuration.main_output_directory) + "/global/netcdf/ " + \
                  str(nc_report_type) + " " + \
                  str(start_date) + " " + \
                  str(end_date) + " " + \
                  str(netcdf_files_that_will_be_merged) + " " + \
                  str(self.netcdf_format) + " " + \
                  str(self.zlib_option) + " " + \
                  str(max_number_of_cores) + " " + \
                  str("Global") + " "

            msg = "Using the following command line: " + cmd
            logger.info(msg)

            vos.cmd_line(cmd, using_subprocess=False)

    def _clone_areas(self):
        """The clone codes whose month-end sentinels this merger waits for.

        Mirrors the clone selection in parallel_pcrglobwb_runner.py so the merger waits for
        exactly the clones that were launched.
        """
        if self.configuration.globalOptions['cloneAreas'] == "Global" or \
                self.configuration.globalOptions['cloneAreas'] == "part_one":
            return ['M%02d' % i for i in range(1, 53 + 1, 1)]
        return list(set(self.configuration.globalOptions['cloneAreas'].split(",")))

    def _clone_sentinel_path(self, clone_area, fulldate):
        return str(self.configuration.main_output_directory) + "/" + str(clone_area) + \
            "/maps/pcrglobwb_files_for_" + str(fulldate) + "_are_ready.txt"

    def check_pcrglobwb_status(self):

        clone_areas = self._clone_areas()
        status = len(clone_areas) == 0
        for clone_area in clone_areas:
            status_file = self._clone_sentinel_path(clone_area, self.modelTime.fulldate)
            msg = 'Waiting for the file: ' + status_file
            if self.count_check == 1: logger.info(msg)
            if self.count_check < 7:
                # ~ logger.debug(msg)		# INACTIVATE THIS AS THIS MAKE A HUGE DEBUG (dbg) FILE
                self.count_check += 1
            status = os.path.exists(status_file)
            if status == False: return status
            if status: self.count_check = 0

        return status

    def _missing_clone_sentinels(self, fulldate):
        """Full list of clone codes whose <fulldate> month-end sentinel is NOT yet on disk.

        check_pcrglobwb_status short-circuits on the first missing clone (a cheap poll); this
        enumerates ALL of them so the timeout diagnostic can name every laggard.
        """
        return [clone_area for clone_area in self._clone_areas()
                if not os.path.exists(self._clone_sentinel_path(clone_area, fulldate))]

    def _resolve_clone_paths(self, clone_area):
        """Best-effort (cloneMap, landmask) paths the given clone is responsible for.

        The merger holds the UNSUBSTITUTED per-tile template (e.g. '.../clone_%s.map'); fill
        in the clone code so the operator can inspect that tile's geometry/size to see whether
        a degenerate/oversized partition is behind the stall.
        """
        def _sub(template):
            try:
                if template is not None and "%" in str(template):
                    return str(template) % clone_area
            except (TypeError, ValueError):
                pass
            return str(template)
        return (_sub(self.configuration.globalOptions.get('cloneMap')),
                _sub(self.configuration.globalOptions.get('landmask')))

    def _latest_clone_activity(self, clone_area):
        """Where a (possibly wedged) clone got to: the latest month-end it completed and the
        most recent write under its output dir (a proxy for "last sign of life")."""
        clone_dir = str(self.configuration.main_output_directory) + "/" + str(clone_area)
        # latest completed month (sentinels are written only at month end, in pcrglobwb.py)
        sentinels = glob.glob(clone_dir + "/maps/pcrglobwb_files_for_*_are_ready.txt")
        if sentinels:
            latest = max(sentinels, key=os.path.getmtime)
            done = os.path.basename(latest).replace("pcrglobwb_files_for_", "").replace("_are_ready.txt", "")
            last_done = "last completed month-end " + done
        else:
            last_done = "NO month-end completed (stalled within the first simulated month)"
        # freshest file mtime anywhere under the clone dir = last write activity
        last_seen = "unknown"
        newest_mtime = None
        try:
            for root, _dirs, files in os.walk(clone_dir):
                for f in files:
                    try:
                        m = os.path.getmtime(os.path.join(root, f))
                    except OSError:
                        continue
                    if newest_mtime is None or m > newest_mtime:
                        newest_mtime = m
        except OSError:
            pass
        if newest_mtime is not None:
            last_seen = datetime.datetime.fromtimestamp(newest_mtime).isoformat()
        return "%s; last output write: %s; per-clone log dir: %s/log/" % (last_done, last_seen, clone_dir)

    def report_clones_still_pending(self, waited_seconds, level=logging.WARNING, detailed=False):
        """Log which clone(s) the merger is still blocked on (and, when ``detailed``, how far
        each got and which clone map it owns). This is the only place in the system that knows
        the IDENTITY of a wedged clone, since the merger consumes every clone's month-end
        sentinel. Called periodically (heartbeat) and once more at the final timeout."""
        fulldate = self.modelTime.fulldate
        missing = self._missing_clone_sentinels(fulldate)
        if not missing:
            return
        total = len(self._clone_areas())
        logger.log(level,
                   "Merging still blocked after %.0f min: waiting on %d of %d clone(s) for "
                   "month-end %s -> %s",
                   waited_seconds / 60.0, len(missing), total, str(fulldate), ",".join(missing))
        # ALL clones missing is qualitatively different from one wedged tile: it means the sentinel
        # mechanism itself is not running. Point at the usual cause so it is diagnosed in seconds.
        if total > 1 and len(missing) == total:
            logger.log(level,
                       "ALL %d clones are missing the month-end sentinel for %s -- this is a "
                       "SYSTEMIC sentinel problem, not a single wedged tile: no clone is writing "
                       "pcrglobwb_files_for_<date>_are_ready.txt. Check that the per-clone runner "
                       "builds PCRGlobWB with spinUpRun=False for the transient run (the sentinel "
                       "in pcrglobwb.py is written only when 'spinUpRun is not None and == False'), "
                       "and that clones use the layout the merger checks: %s/<clone>/maps/.",
                       total, str(fulldate), str(self.configuration.main_output_directory))
        if not detailed:
            return
        for clone_area in missing:
            clone_map, landmask = self._resolve_clone_paths(clone_area)
            logger.log(level,
                       "  wedged-candidate clone %s | cloneMap=%s | landmask=%s | %s",
                       clone_area, clone_map, landmask, self._latest_clone_activity(clone_area))


def main():
    # print disclaimer
    disclaimer.print_disclaimer()

    # get the full path of configuration/ini file given in the system argument
    iniFileName = os.path.abspath(sys.argv[1])

    # debug option
    debug_mode = False
    if len(sys.argv) > 2:
        if sys.argv[2] == "debug" or sys.argv[2] == "debug_parallel": debug_mode = True

    # options to perform steady state calculation
    steady_state_only = False
    if len(sys.argv) > 3:
        if sys.argv[3] == "steady-state-only": steady_state_only = True

    # object to handle configuration/ini file
    configuration = Configuration(iniFileName=iniFileName, debug_mode=debug_mode)

    # timeStep info: year, month, day, doy, hour, etc
    currTimeStep = ModelTime()

    # Running the deterministic_runner
    currTimeStep.getStartEndTimeSteps(configuration.globalOptions['startTime'],
                                      configuration.globalOptions['endTime'])
    logger.info('Model run starts.')
    deterministic_runner = DeterministicRunner(configuration, currTimeStep)

    dynamic_framework = DynamicFramework(deterministic_runner, currTimeStep.nrOfTimeSteps)
    dynamic_framework.setQuiet(True)
    dynamic_framework.run()


if __name__ == '__main__':
    # print disclaimer
    disclaimer.print_disclaimer(with_logger=True)
    sys.exit(main())
