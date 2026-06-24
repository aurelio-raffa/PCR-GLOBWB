"""Stage 4 -- launch the PCR-GLOBWB model.

Configurable launch path (chosen with ``mode``):

  * ``parallel`` (default) -- drives the multi-clone runner on a single node, which spawns one process per
    tile/clone (and, optionally, the merging process). Matches the LDD-basins/tiles decomposition.
        runner=basic            -> ``model/parallel_pcrglobwb_runner.py <ini> <debug_option>``
        runner=with_arguments   -> ``model/parallel_pcrglobwb_runner_with_arguments.py <ini> <debug_option> <extra_args...>``
  * ``serial``             -- a single global/clone run:
        ``model/deterministic_runner.py <ini> [debug] [extra_args...]``

``config`` is the concrete INI produced by the ``create_ini`` stage. ``extra_args`` is a single string
(shell-split) appended verbatim -- use it for the ``with_arguments`` runner's overrides (``-sd``, ``-ed``,
``-mod``, clone codes, ...) or for ``deterministic_runner``'s ``--output_dir`` override.
"""
import os
import shlex

from fire import Fire

from __init__ import root_path
from src.utils.shell import run_command, python_tool

_PARALLEL_RUNNERS = {
    'basic': 'model/parallel_pcrglobwb_runner.py',
    'with_arguments': 'model/parallel_pcrglobwb_runner_with_arguments.py',
}


def run_model(
        config: str,
        mode: str = 'parallel',
        runner: str = 'basic',
        debug_option: str = 'parallel',
        serial_debug: bool = False,
        extra_args: str = '',
        log_file: str = '',
) -> None:
    """Launch PCR-GLOBWB with the rendered ``config`` INI.

    Args:
        config: Path to the concrete INI (the ``create_ini`` stage's ``output_path``).
        mode: ``parallel`` (default) or ``serial``.
        runner: For ``parallel`` mode, ``basic`` or ``with_arguments``.
        debug_option: Value passed as the runner's second positional arg in ``parallel`` mode (default
            ``parallel``; the runners also accept ``debug_parallel``).
        serial_debug: In ``serial`` mode, append the ``debug`` positional flag.
        extra_args: Extra CLI arguments appended verbatim (shell-split).
        log_file: If set, tee the model's combined stdout+stderr to this convenience logfile (inside the
            run output dir) while still streaming to the LSF job log. The ``diagnostics`` stage reads it.
            In parallel mode the runner backgrounds every per-tile/merging process with inherited fds, so
            this single tee captures the full model output (including a merging-process crash that the
            job's exit code would otherwise hide).
    """
    tail = shlex.split(extra_args) if extra_args else []

    if mode == 'parallel':
        if runner not in _PARALLEL_RUNNERS:
            raise ValueError(f"run_model: unknown parallel runner '{runner}' (expected one of "
                             f"{sorted(_PARALLEL_RUNNERS)})")
        cmd = python_tool(_PARALLEL_RUNNERS[runner], config, debug_option, *tail)
    elif mode == 'serial':
        cmd = python_tool('model/deterministic_runner.py', config, *(['debug'] if serial_debug else []), *tail)
    else:
        raise ValueError(f"run_model: unknown mode '{mode}' (expected 'parallel' or 'serial')")

    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)

    run_command(cmd, cwd=root_path, log_file=log_file or None)


if __name__ == '__main__':
    Fire(run_model)
