"""Tiny subprocess helper shared by the pipeline stages.

Each stage is itself a subprocess spawned by the orchestrator; inside it we shell out to the existing
PCR-GLOBWB command-line tools (``compute_ldd_basins.py``, ``create_tile_clone_maps.py``,
``create_ini_config.py``, ``model/*_runner*.py``). This keeps the stages thin wrappers that *reuse* those
tools rather than re-implementing them.
"""
import sys
import logging
import subprocess

from src import console_handler

logger = logging.getLogger(__name__)
logger.addHandler(console_handler)


def run_command(cmd, cwd=None, capture: bool = False) -> subprocess.CompletedProcess:
    """Run ``cmd`` (a list of args), raising on a non-zero exit so the failure propagates up the pipeline.

    Args:
        cmd: Argument list (no shell). The first element is usually ``sys.executable`` for a Python tool.
        cwd: Working directory; defaults to the caller's.
        capture: If True, capture and return stdout/stderr (text); otherwise they stream to this process's
            stdout/stderr (so model logs land in the LSF job log).

    Returns:
        The completed process (``.stdout``/``.stderr`` populated only when ``capture`` is True).
    """
    printable = ' '.join(str(part) for part in cmd)
    logger.info('running: %s%s', printable, f'  (cwd={cwd})' if cwd else '')
    result = subprocess.run(
        [str(part) for part in cmd],
        cwd=cwd,
        check=True,
        text=True,
        capture_output=capture,
    )
    if capture and result.stdout:
        # echo the captured stdout so it is still visible in the job log
        print(result.stdout, file=sys.stderr)
    return result


def python_tool(script: str, *args) -> list:
    """Build an argv list ``[python, script, *args]`` (None args are dropped)."""
    return [sys.executable, script, *[a for a in args if a is not None]]
