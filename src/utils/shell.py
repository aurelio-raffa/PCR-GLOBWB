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


def run_command(cmd, cwd=None, capture: bool = False, log_file: str = None) -> subprocess.CompletedProcess:
    """Run ``cmd`` (a list of args), raising on a non-zero exit so the failure propagates up the pipeline.

    Args:
        cmd: Argument list (no shell). The first element is usually ``sys.executable`` for a Python tool.
        cwd: Working directory; defaults to the caller's.
        capture: If True, capture and return stdout/stderr (text); otherwise they stream to this process's
            stdout/stderr (so model logs land in the LSF job log).
        log_file: If set, *tee* the child's combined stdout+stderr to this file while still streaming it
            live to this process's stderr. Used by ``run_model`` to persist the model's output to a
            convenience logfile inside the run output dir (the diagnostics stage reads it). The file is
            written even if the command dies, so a crash/kill still leaves an analysable log. Mutually
            exclusive with ``capture`` (which buffers instead of streaming).

    Returns:
        The completed process (``.stdout``/``.stderr`` populated only when ``capture`` is True).
    """
    printable = ' '.join(str(part) for part in cmd)
    logger.info('running: %s%s', printable, f'  (cwd={cwd})' if cwd else '')
    argv = [str(part) for part in cmd]

    if log_file:
        if capture:
            raise ValueError('run_command: `capture` and `log_file` are mutually exclusive')
        return _run_command_teed(argv, cwd=cwd, log_file=log_file)

    result = subprocess.run(
        argv,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=capture,
    )
    if capture and result.stdout:
        # echo the captured stdout so it is still visible in the job log
        print(result.stdout, file=sys.stderr)
    return result


def _run_command_teed(argv, cwd, log_file) -> subprocess.CompletedProcess:
    """Stream a child's combined stdout+stderr to both this process's stderr and ``log_file``.

    Line-buffered and flushed so the convenience logfile is usable even mid-run; the file handle is
    closed in ``finally`` so it survives a crash/kill. Preserves the raise-on-nonzero contract.
    """
    handle = open(log_file, 'w', encoding='utf-8', buffering=1)
    try:
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,   # fold stderr into one stream (matches the LSF errfile ordering)
            text=True,
            bufsize=1,
        )
        for line in process.stdout:        # iterate as the child emits lines
            sys.stderr.write(line)         # keep streaming to the LSF job log
            handle.write(line)             # ...and persist to the convenience logfile
        process.stdout.close()
        returncode = process.wait()
    finally:
        handle.flush()
        handle.close()

    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, argv)
    return subprocess.CompletedProcess(argv, returncode)


def python_tool(script: str, *args) -> list:
    """Build an argv list ``[python, script, *args]`` (None args are dropped)."""
    return [sys.executable, script, *[a for a in args if a is not None]]
