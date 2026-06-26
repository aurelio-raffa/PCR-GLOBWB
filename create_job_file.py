#!/usr/bin/env python3
"""Unified LSF job-file generator for PCR-GLOBWB.

Merges create_batch_job_file.py (serial runner) and create_parallel_batch_job_file.py (parallel runner) into
one tool, and adds a ``pipeline`` mode that generates an LSF file for the MLflow-tracked pipeline
(``run_project.py``).

Privacy: every real path (run dir, input dir, repo dir, tracking URI, ...) is supplied as a CLI argument at
submission time on the HPC and only ever appears in the *generated* ``.lsf`` file, never in the repository.
The committed pipeline YAML refers to those paths only through ``{{$ENV_VAR}}`` placeholders, so no real
server detail is ever pushed to the public repo.

Modes
-----
    deterministic   python3 model/deterministic_runner.py <config>
    parallel        python3 model/parallel_pcrglobwb_runner[_with_arguments].py <config> <debug> [extra...]
    pipeline        export <paths> ; python3 run_project.py --config_file=... --experiment_name=... [...]

Usage
-----
    python create_job_file.py deterministic --nc 8 --mem 32G --jq normal --wd /work --pc PRJ \
        --conda_env pcrglobwb_python3 --config config.ini

    python create_job_file.py parallel --nc 54 --mem 128G --jq normal --wd /work --pc PRJ \
        --conda_env pcrglobwb_python3 --config config.ini --excl

    python create_job_file.py pipeline --nc 54 --mem 128G --jq normal --wd /work --pc PRJ \
        --conda_env pcrglobwb_python3 --experiment_name run01 --excl \
        --run_dir /scratch/$USER/run01 --input_dir /data/inputs --repo_dir "$PWD"

The generated file is named ``<name>_<YYMMDDHHMM>.lsf`` and is submitted with
``bsub < <name>_<YYMMDDHHMM>.lsf`` from the PCR-GLOBWB project root.
"""
import argparse
import configparser
import os
import sys
from datetime import datetime


# --------------------------------------------------------------------------------------------------------------
# Clone/tile definitions + INI validation (ported from create_parallel_batch_job_file.py)
# --------------------------------------------------------------------------------------------------------------
DEFAULT_CLONE_CODES = [f"M{i:02d}" for i in range(1, 54)]  # all 53 clones
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
# (clone_codes, merging_override): merging_override=False forces merging off; None reads it from the INI.
KEYWORD_CLONE_AREAS = {
    "Global": (DEFAULT_CLONE_CODES, None),
    "part_one": (PART_ONE_CLONE_CODES, None),
    "part_two": (PART_TWO_CLONE_CODES, False),
}

RUNNER_SCRIPT_PATHS = {
    'basic': 'model/parallel_pcrglobwb_runner.py',
    'with_arguments': 'model/parallel_pcrglobwb_runner_with_arguments.py',
}

GLOBAL_OPTIONS_SECTION = 'globalOptions'
MAX_SPINUPS_KEY = 'maxSpinUpsInYears'
WITH_MERGING_KEY = 'with_merging'
CLONE_AREAS_KEY = 'cloneAreas'


def _read_global_options(ini_path: str) -> dict:
    """Read the [globalOptions] section of a PCR-GLOBWB INI (no interpolation, case-preserving)."""
    if not os.path.isfile(ini_path):
        raise SystemExit(f"INI file not found: {ini_path}")
    config = configparser.RawConfigParser(strict=False)
    config.optionxform = str
    try:
        config.read(ini_path)
    except configparser.Error as exc:
        raise SystemExit(f"Failed to parse INI file '{ini_path}': {exc}")
    if not config.has_section(GLOBAL_OPTIONS_SECTION):
        return {}
    return dict(config.items(GLOBAL_OPTIONS_SECTION))


def _parsed_max_spinups(global_options: dict) -> float:
    raw = global_options.get(MAX_SPINUPS_KEY, '0').strip()
    try:
        return float(raw)
    except ValueError:
        return 0.0


def _check_spinup_merging_consistency(ini_path: str) -> None:
    """The parallel runner rejects maxSpinUpsInYears > 0 combined with with_merging=True; enforce it here."""
    global_options = _read_global_options(ini_path)
    if _parsed_max_spinups(global_options) > 0 and global_options.get(WITH_MERGING_KEY, 'True').strip() == 'True':
        raise SystemExit(
            f"Inconsistent INI '{ini_path}': {MAX_SPINUPS_KEY} > 0 cannot be combined with "
            f"{WITH_MERGING_KEY}=True (parallel_pcrglobwb_runner.py rejects this at runtime). "
            f"Set {WITH_MERGING_KEY}=False, or run the spin-up serially first."
        )


def _check_core_count(nc: int, ini_path: str) -> None:
    """Validate that --nc covers the expanded clone count (+1 for merging when enabled)."""
    global_options = _read_global_options(ini_path)
    if CLONE_AREAS_KEY not in global_options:
        raise SystemExit(
            f"INI file '{ini_path}' is missing required key '{CLONE_AREAS_KEY}' "
            f"under [{GLOBAL_OPTIONS_SECTION}]."
        )
    raw = global_options[CLONE_AREAS_KEY].strip()
    if raw in KEYWORD_CLONE_AREAS:
        codes, merging_override = KEYWORD_CLONE_AREAS[raw]
    else:
        codes = [token.strip() for token in raw.split(',') if token.strip()]
        merging_override = None
    if not codes:
        raise SystemExit(f"'{CLONE_AREAS_KEY}'='{raw}' in INI '{ini_path}' expands to an empty clone list.")
    if merging_override is not None:
        with_merging = merging_override
    else:
        with_merging = global_options.get(WITH_MERGING_KEY, 'True').strip() == 'True'
    required = len(codes) + (1 if with_merging else 0)
    if nc < required:
        note = " + 1 for the merging process" if with_merging else ""
        raise SystemExit(
            f"--nc={nc} is below the minimum for '{CLONE_AREAS_KEY}'='{raw}': "
            f"{len(codes)} clone process(es){note} = {required} core(s). Re-submit with --nc >= {required}."
        )


# --------------------------------------------------------------------------------------------------------------
# Shared LSF header / resources / output
# --------------------------------------------------------------------------------------------------------------
# Legend: -n cores, -R/-x placement, -M mem, -q queue, -o/-e logs, -P project code.
HEADER_TEMPLATE = """#!/bin/sh
#BSUB -n {nc}
#BSUB {resources}
#BSUB -M {mem}
#BSUB -q {jq}
#BSUB -o {wd}/logfile.%J.txt
#BSUB -e {wd}/errfile.%J.txt
#BSUB -P {pc}
module load anaconda
source $(conda info --base)/etc/profile.d/conda.sh
# Activate the env by NAME (no hard-coded paths, so the env can be relocated freely).
conda activate {conda_env}
# Guard against the original new-HPC failure: when the env's dir is not yet in conda's envs_dirs, `conda activate`
# fails and the shell silently continues on base python, which then crashes confusingly (e.g. "No module named
# mlflow"). Catch it here. Every PCR-GLOBWB job (all three modes) needs pcraster and the base/anaconda env does
# not have it, so a failing `import pcraster` is a reliable, path-free, all-modes signal that the requested env
# is NOT active. We deliberately do NOT gate on `conda activate`'s exit status or on $CONDA_DEFAULT_ENV: an
# activate.d hook can make a fully-successful activation return non-zero, and older conda may not export
# $CONDA_DEFAULT_ENV -- either would falsely abort a previously-working run.
python3 -c "import pcraster" || {{ echo "FATAL: conda env '{conda_env}' is not active (import pcraster failed) -- activation likely fell back to base python; check 'conda env list' and that the env's dir is registered in conda's envs_dirs" >&2; exit 1; }}
"""


def _resources(nc: int, tile, excl: bool) -> str:
    """`-x` for an exclusive node, else `-R span[ptile=...]` so all cores land on one host (Option A)."""
    if excl:
        return '-x'
    return f'-R span[ptile={tile if tile is not None else nc}]'


def _header(args: argparse.Namespace) -> str:
    return HEADER_TEMPLATE.format(
        nc=args.nc, resources=_resources(args.nc, args.tile, args.excl),
        mem=args.mem, jq=args.jq, wd=args.wd, pc=args.pc, conda_env=args.conda_env,
    )


def _with_invocation_comment(body: str) -> str:
    """Record how this job file was generated, as a comment just after the ``#!`` shebang.

    Captures the exact ``create_job_file.py`` invocation (``sys.argv``) and a timestamp, so a generated
    ``.lsf`` is self-documenting and regenerable. This comment lands ONLY in the generated ``.lsf`` (which
    is git-ignored), so the real cluster paths it contains never reach the public repository.
    """
    comment = (
        f'#\n'
        f'# Generated by: python {" ".join(sys.argv)}\n'
        f'# Generated at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n'
        f'# (regenerate with the command above; edits here are overwritten on the next generation)\n'
        f'#\n'
    )
    shebang, _, rest = body.partition('\n')
    if shebang.startswith('#!'):
        return f'{shebang}\n{comment}{rest}'
    return comment + body


def _write(name: str, body: str) -> str:
    filename = f'{name}_{datetime.now().strftime("%y%m%d%H%M")}.lsf'
    with open(filename, 'w') as handle:
        handle.write(_with_invocation_comment(body))
    print(filename)
    return filename


# --------------------------------------------------------------------------------------------------------------
# Mode bodies
# --------------------------------------------------------------------------------------------------------------
def build_deterministic(args: argparse.Namespace) -> str:
    """Serial run: `python3 model/deterministic_runner.py <config>` (replaces create_batch_job_file.py)."""
    body = _header(args) + f'python3 model/deterministic_runner.py {args.config}\n'
    return _write(args.name, body)


def build_parallel(args: argparse.Namespace) -> str:
    """Parallel run (replaces create_parallel_batch_job_file.py), with the same consistency checks."""
    if args.runner_extra_args and args.runner != 'with_arguments':
        raise SystemExit("--runner_extra_args is only valid with --runner=with_arguments.")
    _check_spinup_merging_consistency(args.config)
    _check_core_count(args.nc, args.config)

    invocation = f'python3 {RUNNER_SCRIPT_PATHS[args.runner]} {args.config} {args.debug_option}'
    extra = ' '.join(args.runner_extra_args or [])
    if extra:
        invocation = f'{invocation} {extra}'
    return _write(args.name, _header(args) + invocation + '\n')


def build_pipeline(args: argparse.Namespace) -> str:
    """MLflow pipeline run: export the (real) paths from CLI args, then launch run_project.py."""
    body = _header(args)
    # Only the pipeline needs the MLflow/Fire harness deps -- assert them here, not in the shared header
    # (which also serves the legacy deterministic/parallel model runners that import neither).
    body += ('python3 -c "import mlflow, fire" || '
             '{ echo "FATAL: env activated but missing mlflow/fire (the pipeline needs them)" >&2; exit 1; }\n')
    if args.repo_dir:
        body += f'cd {args.repo_dir}\n'
    if args.run_dir:
        body += f'export PCRG_RUN_DIR={args.run_dir}\n'
    if args.input_dir:
        body += f'export PCRG_INPUT_DIR={args.input_dir}\n'
    for pair in (args.env or []):
        if '=' not in pair:
            raise SystemExit(f"--env expects KEY=VALUE, got '{pair}'.")
        body += f'export {pair}\n'
    cmd = f'python3 run_project.py --config_file={args.config_file} --experiment_name={args.experiment_name}'
    if args.tracking_uri:
        cmd += f' --tracking_uri={args.tracking_uri}'
    return _write(args.name, body + cmd + '\n')


DISPATCH = {
    'deterministic': build_deterministic,
    'parallel': build_parallel,
    'pipeline': build_pipeline,
}


# --------------------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------------------
def _add_common(sub, default_name: str) -> None:
    sub.add_argument('--nc', required=True, type=int, help='Number of cores requested from LSF')
    sub.add_argument('--mem', required=True, help='Memory limit per host (e.g. "128G")')
    sub.add_argument('--jq', required=True, help='LSF queue to submit to')
    sub.add_argument('--wd', required=True, help='Working directory for stdout/stderr log files')
    sub.add_argument('--pc', required=True, help='Project identifier (#BSUB -P)')
    sub.add_argument('--conda_env', required=True,
                     help="Name of the conda environment to activate (must appear in 'conda env list'; "
                          "if it lives in a custom dir, register that dir in conda's envs_dirs)")
    sub.add_argument('--tile', type=int, default=None,
                     help='Cores per node for `-R span[ptile=...]` (ignored with --excl); defaults to --nc')
    sub.add_argument('--excl', action='store_true', help='Reserve the whole node (#BSUB -x)')
    sub.add_argument('--name', default=default_name,
                     help=f'Identifier used in the output filename. Defaults to "{default_name}"')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest='mode', required=True, metavar='{deterministic,parallel,pipeline}')

    det = sub.add_parser('deterministic', help='Serial run via model/deterministic_runner.py')
    _add_common(det, 'job')
    det.add_argument('--config', required=True, help='Path to the (filled-in) PCR-GLOBWB INI file')

    par = sub.add_parser('parallel', help='Parallel run via model/parallel_pcrglobwb_runner*.py')
    _add_common(par, 'job_parallel')
    par.add_argument('--config', required=True, help='Path to the (filled-in) PCR-GLOBWB INI file')
    par.add_argument('--runner', choices=list(RUNNER_SCRIPT_PATHS), default='basic',
                     help='Which parallel runner to invoke. Default: basic')
    par.add_argument('--debug_option', default='parallel',
                     help='Second positional arg forwarded to the runner (e.g. "parallel", "debug")')
    par.add_argument('--runner_extra_args', nargs=argparse.REMAINDER, default=[],
                     help='Extra args forwarded verbatim to the runner (only with --runner=with_arguments). '
                          'MUST be last on the command line.')

    pipe = sub.add_parser('pipeline', help='MLflow-tracked pipeline via run_project.py')
    _add_common(pipe, 'pipeline')
    pipe.add_argument('--config_file', default='config/pipeline/pcrglobwb_pipeline.yaml',
                      help='Pipeline YAML passed to run_project.py')
    pipe.add_argument('--experiment_name', required=True, help='MLflow experiment name')
    pipe.add_argument('--tracking_uri', default=None, help='Optional MLflow tracking store URI')
    pipe.add_argument('--run_dir', default=None, help='Exported as PCRG_RUN_DIR (run root)')
    pipe.add_argument('--input_dir', default=None, help='Exported as PCRG_INPUT_DIR (input tree root)')
    pipe.add_argument('--repo_dir', default=None, help='If set, `cd <repo_dir>` before running run_project.py')
    pipe.add_argument('--env', action='append', default=[], metavar='KEY=VALUE',
                      help='Additional environment export(s) for {{$VAR}} placeholders (repeatable)')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    DISPATCH[args.mode](args)


if __name__ == '__main__':
    main()
