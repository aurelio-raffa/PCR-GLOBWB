"""
LSF Job File Generator for Parallel PCR-GLOBWB Simulations

This module generates IBM Spectrum LSF job submission scripts (.lsf files) for
running parallel PCR-GLOBWB (Global Water Resources and Hydrological Model)
simulations on a compute cluster. It implements Option A: a single LSF job that
spawns multiple Python processes onto a single compute node with OS-level job
control (shell background processes and wait).

The generated script:
- Invokes one of the parallel runner scripts (basic or with-arguments variant)
- Launches one OS process per clone/tile specified in the INI configuration
- Enforces consistency checks (core count, spinup+merging compatibility)
- Emits LSF directives (#BSUB) for resource allocation and queue submission

Typical usage:
    python create_parallel_batch_job_file.py \\
        --nc 54 --mem 128G --jq normal --wd /work/dir \\
        --pc my_project --conda_env globwb \\
        --config /path/to/config.ini

The generated file is named job_parallel_YYMMDDHHMM.lsf and should be submitted
with 'bsub < job_parallel_YYMMDDHHMM.lsf' from the PCR-GLOBWB project root.
"""
import argparse
import configparser
import os
from datetime import datetime


# LSF Job Parameter Keys
# These keys map command-line argument names to the placeholders used in
# the job template. They define which parameters the user must provide.

N_CORES = 'nc'                      # Number of CPU cores to request from LSF
TILE = 'tile'                        # Cores per node for span[ptile=...] affinity
MEM_LIMIT = 'mem'                    # Per-node memory limit (e.g., "128G")
JOB_QUEUE = 'jq'                     # LSF queue name (e.g., "normal", "gpu")
WORKING_DIR = 'wd'                   # Directory for log files (stdout/stderr)
PROJECT_CODE = 'pc'                  # Project identifier for billing/tracking
CONDA_ENV = 'conda_env'              # Name of conda environment to activate
CONFIG_INI_PATH = 'config'           # Path to PCR-GLOBWB configuration INI file
EXCLUSIVE = 'excl'                   # Boolean flag: reserve entire node exclusively
RUNNER = 'runner'                    # Which parallel runner variant to use
DEBUG_OPTION = 'debug_option'        # Debug flag passed to runner ("parallel" or "debug")
RUNNER_EXTRA_ARGS = 'runner_extra_args'  # Additional arguments for runner_with_arguments
CLONE_AREAS_KEY = 'cloneAreas'      # INI key for clone/tile codes (read-only, not a CLI arg)
_RESOURCES = '__resources__'         # Internal: resolved LSF resource directive
_PYTHON_INVOCATION = '__python_invocation__'  # Internal: complete Python command line

# Clone/Tile Definitions
# PCR-GLOBWB divides the global hydrological model into 53 major clone regions
# (M01 through M53). These can be processed in parallel. The partitions below
# allow grouping clones for distributed execution across multiple compute jobs.
DEFAULT_CLONE_CODES = [f"M{i:02d}" for i in range(1, 54)]  # All 53 clones

# These partitions mirror the hardcoded allocations in model/parallel_pcrglobwb_runner.py.
# They allow splitting the workload for better load balancing or to fit within
# resource constraints.
PART_ONE_CLONE_CODES = [
    "M17", "M19", "M26", "M13", "M18", "M20", "M05", "M03", "M21", "M46",
    "M27", "M49", "M16", "M44", "M52", "M25", "M09", "M08", "M11", "M42",
    "M12", "M39",
]
PART_TWO_CLONE_CODES = [
    "M07", "M15", "M38", "M48", "M40", "M41", "M22", "M14", "M23", "M51",
    "M04", "M06", "M10", "M02", "M45", "M35", "M47", "M50", "M24", "M01",
    "M36", "M53", "M33", "M43", "M34", "M37", "M31", "M32", "M28", "M30",
    "M29",
]

# Maps keyword names to (clone_codes, merging_override) tuples.
# merging_override=False forces merging off (e.g., for part_two).
# merging_override=None reads the setting from the INI file.
KEYWORD_CLONE_AREAS = {
    "Global": (DEFAULT_CLONE_CODES, None),
    "part_one": (PART_ONE_CLONE_CODES, None),
    "part_two": (PART_TWO_CLONE_CODES, False),  # Forces merging off per parallel_pcrglobwb_runner.py
}

# Parallel Runner Variants
# The job submission supports two runner scripts with different capabilities:
# - basic: Simple runner that processes clones defined in the INI config
# - with_arguments: Extended runner that accepts command-line overrides for
#   start/end dates, output directories, and other parameters

RUNNER_BASIC = 'basic'
RUNNER_WITH_ARGS = 'with_arguments'

RUNNER_SCRIPT_PATHS = {
    RUNNER_BASIC: 'model/parallel_pcrglobwb_runner.py',
    RUNNER_WITH_ARGS: 'model/parallel_pcrglobwb_runner_with_arguments.py',
}

# INI Configuration File Keys
# These define the expected structure of PCR-GLOBWB configuration INI files.
# Used for validation checks (e.g., spinup + merging compatibility).

GLOBAL_OPTIONS_SECTION = 'globalOptions'  # INI section containing global settings
MAX_SPINUPS_KEY = 'maxSpinUpsInYears'     # Key for spin-up period duration
WITH_MERGING_KEY = 'with_merging'         # Key for post-run merging of clone outputs


# Help Message for CLI
# Displayed when user runs with --help. Explains the script's purpose,
# execution model, and required behavior.

help_msg: str = f"""
LSF Job File Generator for parallel PCR-GLOBWB runs (Option A: a single
LSF job that fans out many Python processes onto one fat node).

The generated .lsf invokes one of the bundled parallel runners
(parallel_pcrglobwb_runner.py or parallel_pcrglobwb_runner_with_arguments.py),
which in turn launches one OS process per clone listed under `cloneAreas`
in the INI file. All clone processes run on the same node, scheduled by
the OS via shell job control (`&` / `wait`). Submit the generated file
with `bsub`.

This script is intentionally agnostic about the INI content; the only
consistency check it performs is to reject INIs that combine
`{MAX_SPINUPS_KEY}` > 0 with `{WITH_MERGING_KEY}` = True, because
parallel_pcrglobwb_runner.py refuses that combination at runtime.

Sizing the allocation is left to the caller (--{N_CORES} must match the
expanded clone count, plus +1 if in-line merging will run). Use --{EXCLUSIVE}
to reserve the whole node, otherwise `-R span[ptile=...]` is emitted so all
requested cores still land on a single host (Option A semantics).

The .lsf assumes it will be submitted from the PCR-GLOBWB project root
(so that `model/parallel_pcrglobwb_runner*.py` is a valid relative path).

Output filename pattern: job_parallel_{{YYMMDDHHMM}}.lsf
"""


# LSF Job Template
# Shell script template with embedded LSF directives (#BSUB). The script:
# 1. Loads the anaconda module
# 2. Activates the specified conda environment
# 3. Invokes the Python runner script with INI config and optional arguments
#
# LSF Directive Reference:
#   -n: number of requested cores
#   -R: host resource requirements (affinity, placement, memory)
#   -x: exclusive node reservation (overrides -R)
#   -M: per-job memory limit in MB or with suffix (G, T)
#   -q: queue name for job submission
#   -o: stdout log file (job ID %J is substituted by LSF)
#   -e: stderr log file (job ID %J is substituted by LSF)
#   -P: project code for billing/tracking

job_template = f"""#!/bin/sh
#BSUB -n {{{N_CORES}}}
#BSUB {{{_RESOURCES}}}
#BSUB -M {{{MEM_LIMIT}}}
#BSUB -q {{{JOB_QUEUE}}}
#BSUB -o {{{WORKING_DIR}}}/logfile.%J.txt
#BSUB -e {{{WORKING_DIR}}}/errfile.%J.txt
#BSUB -P {{{PROJECT_CODE}}}
module load anaconda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate {{{CONDA_ENV}}}
{{{_PYTHON_INVOCATION}}}
"""


# Command-Line Argument Parser
# Defines all required and optional parameters for the job file generator.
# Arguments are grouped below by required (must provide) and optional (defaults).

parser = argparse.ArgumentParser(
    description=help_msg,
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument(f"--{N_CORES}", required=True, type=int, help='Number of cores requested from LSF')
parser.add_argument(f"--{MEM_LIMIT}", required=True, help='Memory limit per host (e.g. "128G")')
parser.add_argument(f"--{JOB_QUEUE}", required=True, help='LSF queue to submit to')
parser.add_argument(f"--{WORKING_DIR}", required=True, help='Working directory for stdout/stderr log files')
parser.add_argument(f"--{PROJECT_CODE}", required=True, help='Project identifier (#BSUB -P)')
parser.add_argument(f"--{CONDA_ENV}", required=True, help='Name of the conda environment to activate')
parser.add_argument(f"--{CONFIG_INI_PATH}", required=True, help='Path to the (filled-in) PCR-GLOBWB INI configuration file')
parser.add_argument(
    f"--{TILE}",
    nargs='?',
    type=int,
    default=None,
    help=f'Cores per node for `-R span[ptile=...]` (ignored when --{EXCLUSIVE} is set); defaults to --{N_CORES} so all clones land on one host.',
)
parser.add_argument(
    f"--{EXCLUSIVE}",
    action='store_true',
    help='Reserve the entire node (emits `#BSUB -x`). Recommended for Option A.',
)
parser.add_argument(
    f"--{RUNNER}",
    choices=[RUNNER_BASIC, RUNNER_WITH_ARGS],
    default=RUNNER_BASIC,
    help=(
        f'Which parallel runner to invoke. '
        f'"{RUNNER_BASIC}" -> {RUNNER_SCRIPT_PATHS[RUNNER_BASIC]}; '
        f'"{RUNNER_WITH_ARGS}" -> {RUNNER_SCRIPT_PATHS[RUNNER_WITH_ARGS]}. '
        f'Default: {RUNNER_BASIC}.'
    ),
)
parser.add_argument(
    f"--{DEBUG_OPTION}",
    default='parallel',
    help='Second positional argument forwarded to the parallel runner (e.g. "parallel", "debug"). Default: "parallel".',
)
parser.add_argument(
    f"--{RUNNER_EXTRA_ARGS}",
    nargs=argparse.REMAINDER,
    default=[],
    help=(
        f'Extra arguments forwarded verbatim to the parallel runner. '
        f'Only meaningful with --{RUNNER}={RUNNER_WITH_ARGS}. '
        f'MUST be the last flag on the command line; everything after it is captured. '
        f'Example: --{RUNNER_EXTRA_ARGS} -mod /scratch/out -sd 1981-01-01 -ed 2019-12-31'
    ),
)


# Helper Functions for Configuration and Validation
def _read_global_options(ini_path: str) -> dict:
    """
    Read the [globalOptions] section from a PCR-GLOBWB INI configuration file.

    Parses the INI file and extracts all key-value pairs from the globalOptions
    section. If the section does not exist, returns an empty dict. Preserves
    original case sensitivity of keys (via config.optionxform = str).

    Args:
        ini_path: Absolute or relative path to the PCR-GLOBWB configuration INI file.

    Returns:
        Dictionary of key-value pairs from [globalOptions], or empty dict if section
        is missing. Keys and values are strings as read from the file (no parsing).

    Raises:
        SystemExit: If the file does not exist or cannot be parsed as valid INI.
    """
    if not os.path.isfile(ini_path):
        raise SystemExit(f"INI file not found: {ini_path}")

    # Use RawConfigParser to avoid interpolation of %(var)s references
    config = configparser.RawConfigParser(strict=False)
    config.optionxform = str  # Preserve case sensitivity of option names

    try:
        config.read(ini_path)
    except configparser.Error as exc:
        raise SystemExit(f"Failed to parse INI file '{ini_path}': {exc}")

    # Return globalOptions section if it exists, otherwise empty dict
    if not config.has_section(GLOBAL_OPTIONS_SECTION):
        return {}
    return dict(config.items(GLOBAL_OPTIONS_SECTION))


def _parsed_max_spinups(global_options: dict) -> float:
    """
    Extract and parse the maxSpinUpsInYears value from global options.

    Reads the spin-up period duration from the global options dictionary. The
    value is expected to be a string representation of a number (int or float).
    If the key is missing or the value cannot be parsed as a float, returns 0.0.

    Args:
        global_options: Dictionary of global configuration options (from _read_global_options).

    Returns:
        Float value of maxSpinUpsInYears. Returns 0.0 if missing or unparseable.
    """
    raw_value = global_options.get(MAX_SPINUPS_KEY, '0').strip()
    try:
        return float(raw_value)
    except ValueError:
        return 0.0


def _check_spinup_merging_consistency(ini_path: str) -> None:
    """
    Validate that spin-up and merging settings are not both enabled in the INI.

    The parallel runner (parallel_pcrglobwb_runner.py) does not support running
    spin-up periods with simultaneous output merging. This consistency check
    enforces that the INI file configuration complies with this constraint.

    If maxSpinUpsInYears > 0 and with_merging = True, the configuration is
    invalid and execution halts. The user must either:
    - Disable merging (set with_merging = False)
    - Run spin-up serially first, then run the main simulation with merging

    Args:
        ini_path: Path to the PCR-GLOBWB configuration INI file.

    Raises:
        SystemExit: If both maxSpinUpsInYears > 0 and with_merging = True.
    """
    global_options = _read_global_options(ini_path)
    max_spinups = _parsed_max_spinups(global_options)
    with_merging_raw = global_options.get(WITH_MERGING_KEY, 'True').strip()

    # Check for invalid configuration: spin-up + merging together
    if max_spinups > 0 and with_merging_raw == 'True':
        raise SystemExit(
            f"Inconsistent INI '{ini_path}': "
            f"{MAX_SPINUPS_KEY}={max_spinups:g} cannot be combined with "
            f"{WITH_MERGING_KEY}=True. "
            f"parallel_pcrglobwb_runner.py rejects this combination at runtime. "
            f"Either set {WITH_MERGING_KEY}=False, or run the spin-up serially first."
        )


def _build_python_invocation(args: argparse.Namespace) -> str:
    """
    Construct the complete Python command line for the parallel runner script.

    Assembles the python3 invocation with the appropriate runner script,
    configuration path, debug option, and any extra arguments. The resulting
    command is embedded in the job template and executed on the compute node.

    Command format:
        python3 model/parallel_pcrglobwb_runner[_with_arguments].py <config.ini> <debug_flag> [extra_args]

    Args:
        args: Parsed command-line arguments (argparse.Namespace).

    Returns:
        Complete command string ready for shell execution.
    """
    # Determine which runner script to invoke based on user selection
    script_path = RUNNER_SCRIPT_PATHS[getattr(args, RUNNER)]
    config_path = getattr(args, CONFIG_INI_PATH)
    debug_flag = getattr(args, DEBUG_OPTION)
    extra_args = ' '.join(getattr(args, RUNNER_EXTRA_ARGS) or [])

    # Build base invocation: python3 <script> <config> <debug_flag>
    invocation = f"python3 {script_path} {config_path} {debug_flag}"

    # Append extra arguments if provided (for with_arguments runner variant)
    if extra_args:
        invocation = f"{invocation} {extra_args}"

    return invocation


def _resolve_tile(args: argparse.Namespace) -> None:
    """
    Set the ptile (cores per node) value to match total cores if not provided.

    The ptile parameter controls job placement affinity via LSF's span[ptile=...]
    constraint. If the user does not explicitly specify --tile, it defaults to
    --nc (total core count), ensuring all clone processes land on a single node.

    This function modifies args in-place.

    Args:
        args: Parsed command-line arguments (argparse.Namespace).

    Returns:
        None (modifies args in-place).
    """
    # If --tile not provided, set it equal to total cores for single-node execution
    if getattr(args, TILE) is None:
        setattr(args, TILE, getattr(args, N_CORES))


def _resolve_resources(args: argparse.Namespace) -> None:
    """
    Generate the LSF resource directive based on exclusive vs. shared node mode.

    Determines the appropriate LSF -x or -R flag for the job template:
    - If --exclusive is set: use -x (reserve entire node exclusively)
    - Otherwise: use -R span[ptile=X] (place X cores per node for affinity)

    The span[ptile=...] constraint ensures all processes land on a single node
    even if more cores are available, enforcing Option A execution semantics.

    This function modifies args in-place, setting the _RESOURCES pseudo-argument.

    Args:
        args: Parsed command-line arguments (argparse.Namespace).

    Returns:
        None (modifies args in-place).
    """
    if getattr(args, EXCLUSIVE):
        # Exclusive node mode: reserve the entire node with -x
        setattr(args, _RESOURCES, '-x')
    else:
        # Shared mode: place up to ptile cores per node for affinity
        setattr(args, _RESOURCES, f'-R span[ptile={getattr(args, TILE)}]')


def _validate_runner_extra_args(args: argparse.Namespace) -> None:
    """
    Ensure extra runner arguments are only provided with the with_arguments runner.

    The basic runner does not accept command-line overrides. If the user specifies
    --runner_extra_args with the basic runner, this is an error and execution halts.

    Args:
        args: Parsed command-line arguments (argparse.Namespace).

    Raises:
        SystemExit: If extra_args are provided but runner is not with_arguments.
    """
    extra_args = getattr(args, RUNNER_EXTRA_ARGS) or []
    if extra_args and getattr(args, RUNNER) != RUNNER_WITH_ARGS:
        raise SystemExit(
            f"--{RUNNER_EXTRA_ARGS} is only valid with --{RUNNER}={RUNNER_WITH_ARGS}, "
            f"but got --{RUNNER}={getattr(args, RUNNER)}."
        )


def _check_core_count(args: argparse.Namespace) -> None:
    """
    Validate that requested cores are sufficient for the clone workload plus merging.

    Calculates the minimum core count required:
    - num_clones: the number of clone/tile processes to run in parallel
    - +1: if with_merging=True (reserved for the merge process)

    cloneAreas is read directly from the [globalOptions] section of the INI file.
    It can be:
    - A keyword: "Global", "part_one", "part_two" (expanded to predefined lists)
    - A CSV list: "M01,M03,M17" (custom clone selection)

    If merging_override is set (e.g., part_two forces merging off), it takes
    precedence over the INI setting.

    Args:
        args: Parsed command-line arguments (argparse.Namespace).

    Raises:
        SystemExit: If --nc is below the required minimum, if cloneAreas is
                    missing from the INI, or if the expanded clone list is empty.
    """
    ini_path = getattr(args, CONFIG_INI_PATH)
    global_options = _read_global_options(ini_path)

    # cloneAreas must be present in the INI; no default or override is allowed
    if CLONE_AREAS_KEY not in global_options:
        raise SystemExit(
            f"INI file '{ini_path}' is missing required key "
            f"'{CLONE_AREAS_KEY}' under [{GLOBAL_OPTIONS_SECTION}]."
        )
    raw_clone_areas = global_options[CLONE_AREAS_KEY].strip()

    # Resolve clone area specification: either keyword or CSV list
    if raw_clone_areas in KEYWORD_CLONE_AREAS:
        codes, merging_override = KEYWORD_CLONE_AREAS[raw_clone_areas]
    else:
        # Parse comma-separated clone codes
        codes = [token.strip() for token in raw_clone_areas.split(',') if token.strip()]
        merging_override = None

    # Validate that at least one clone is specified
    if not codes:
        raise SystemExit(
            f"'{CLONE_AREAS_KEY}' = '{raw_clone_areas}' in INI '{ini_path}' "
            f"expands to an empty clone list."
        )

    # Determine if merging is enabled: use override if set, otherwise read INI
    if merging_override is not None:
        with_merging = merging_override
    else:
        with_merging = global_options.get(WITH_MERGING_KEY, 'True').strip() == 'True'

    # Calculate required cores: clones + 1 if merging
    required = len(codes) + (1 if with_merging else 0)
    nc = getattr(args, N_CORES)

    # Validate that user requested enough cores
    if nc < required:
        merging_note = " + 1 for the merging process" if with_merging else ""
        raise SystemExit(
            f"--{N_CORES}={nc} is below the minimum required for "
            f"'{CLONE_AREAS_KEY}'='{raw_clone_areas}' (from INI): "
            f"{len(codes)} clone process(es){merging_note} "
            f"= {required} core(s). Re-submit with --{N_CORES} >= {required}."
        )


def create_parallel_batch_job_file() -> None:
    """
    Main entry point: parse arguments, validate configuration, and generate LSF job file.

    Execution flow:
    1. Parse command-line arguments via argparse
    2. Resolve default values (tile, resources)
    3. Validate configuration consistency (runner mode, core count, spinup+merging)
    4. Build the complete Python invocation string
    5. Generate the .lsf job file using the template
    6. Print the filename to stdout

    The generated file is named job_parallel_YYMMDDHHMM.lsf (with current datetime)
    and is ready for submission via `bsub < job_parallel_YYMMDDHHMM.lsf`.

    Returns:
        None (side effect: writes .lsf file to disk and prints filename).

    Raises:
        SystemExit: On any configuration error (via validation helper functions).
    """
    # Parse command-line arguments
    args = parser.parse_args()

    # Stage 1: Resolve Configuration Defaults
    _resolve_tile(args)        # Set ptile = total cores if not provided
    _resolve_resources(args)   # Emit -x or -R span[ptile=...] based on exclusive flag

    # Stage 2: Validate Runner Configuration
    _validate_runner_extra_args(args)  # Ensure extra_args only with with_arguments runner

    # Stage 3: Validate INI Configuration
    _check_spinup_merging_consistency(getattr(args, CONFIG_INI_PATH))
    _check_core_count(args)

    # Stage 4: Build Python Command
    setattr(args, _PYTHON_INVOCATION, _build_python_invocation(args))

    # Stage 5: Generate and Write LSF Job File
    # Generate filename with current date/time: job_parallel_YYMMDDHHMM.lsf
    filename = f'job_parallel_{datetime.now().strftime("%y%m%d%H%M")}.lsf'

    # Populate template with resolved arguments and write to file
    with open(filename, 'w') as handle:
        handle.write(job_template.format(**vars(args)))

    # Output the filename so user can reference it for bsub submission
    print(filename)


# Script Entry Point
if __name__ == "__main__":
    """Execute the job file generator when run as a standalone script."""
    create_parallel_batch_job_file()
