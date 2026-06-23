"""Stage 3 -- instantiate a PCR-GLOBWB INI from a template.

Thin wrapper over the existing ``create_ini_config.py`` (which does the ``template.format(**args)`` substitution
and the path validation). That tool writes a *timestamped* file name (``config_<ts>_<name>_<template>.ini``),
which is awkward to hand to the next stage; so this wrapper runs it, locates the file it produced, and moves it
to the deterministic ``output_path`` you specify. The ``run_model`` stage then reads that same path as its
``config``.

YAML keys map to Fire params (hyphens -> underscores), e.g. ``clone-map`` -> ``clone_map``. Give this stage the
``output-path`` key so the orchestrator can log the generated INI as an artifact.
"""
import os
import glob
import time
import shutil

from fire import Fire

from __init__ import root_path
from src.utils.shell import run_command, python_tool


def _locate_generated_ini(stdout: str, name: str, search_dirs, since: float) -> str:
    """Find the INI that ``create_ini_config.py`` just wrote.

    Strategy: trust its printed filename first (it prints the result path on the last line), then fall back to
    the newest ``config_*.ini`` that mentions ``name`` and was written after ``since`` across the candidate
    directories. Robust to whether the tool prints a bare name or a full path, and to its exact write dir.
    """
    # 1) parse the printed path/filename (last stdout line ending in .ini)
    for line in reversed((stdout or '').splitlines()):
        token = line.strip().strip('"').strip("'")
        if token.endswith('.ini'):
            for candidate in (token, os.path.join(root_path, token)):
                if os.path.isfile(candidate):
                    return candidate
            break

    # 2) fall back to the newest matching freshly-written file across candidate dirs
    hits = []
    for directory in search_dirs:
        if directory and os.path.isdir(directory):
            for path in glob.glob(os.path.join(directory, 'config_*.ini')):
                try:
                    mtime = os.path.getmtime(path)
                except OSError:
                    continue
                if mtime >= since:
                    hits.append((name in os.path.basename(path), mtime, path))
    if not hits:
        raise FileNotFoundError(
            'create_ini: could not locate the INI produced by create_ini_config.py '
            f'(searched {list(search_dirs)}). Its stdout was:\n{stdout}'
        )
    # prefer files whose name contains `name`, then most recently written
    hits.sort(reverse=True)
    return hits[0][2]


def create_ini(
        base_ini: str,
        name: str,
        output_dir: str,
        clone_map: str,
        input_dir: str,
        output_path: str,
        landmask: str = 'None',
        clone_areas: str = 'Global',
        with_merging: str = 'True',
        low_res_data: str = 'global_30min',
        high_res_data: str = 'global_05min',
        institution: str = '',
        title: str = '',
        description: str = '',
        novalidation: bool = False,
) -> None:
    """Render ``base_ini`` into a concrete INI at ``output_path``.

    Args mirror ``create_ini_config.py`` (snake_case here, mapped to its exact flags internally):
        base_ini: Template INI, e.g. ``config/05min_parallel.ini`` (``--base_ini``).
        name: Experiment identifier baked into the generated name (``--name``).
        output_dir: The run's ``outputDir`` written into the INI (``--outputDir``).
        clone_map: ``cloneMap`` value (for parallel runs must contain ``%s``, e.g. ``.../clonemap_%s.map``).
        input_dir: ``inputDir`` base for the model inputs (``--inputDir``).
        output_path: Deterministic destination for the rendered INI (this is what ``run_model`` consumes).
        landmask, clone_areas, with_merging, low_res_data, high_res_data, institution, title, description:
            forwarded to the corresponding ``create_ini_config.py`` flags.
        novalidation: Pass ``--novalidation`` to skip on-disk path checks (useful for local dry-runs).
    """
    cmd = python_tool(
        'create_ini_config.py',
        '--name', name,
        '--base_ini', base_ini,
        '--outputDir', output_dir,
        '--cloneMap', clone_map,
        '--inputDir', input_dir,
        '--landmask', landmask,
        '--cloneAreas', clone_areas,
        '--with_merging', with_merging,
        '--lowResData', low_res_data,
        '--highResData', high_res_data,
        '--institution', institution,
        '--title', title,
        '--description', description,
    )
    if novalidation:
        cmd.append('--novalidation')

    since = time.time() - 2  # small grace window for filesystem mtime granularity
    result = run_command(cmd, cwd=root_path, capture=True)

    generated = _locate_generated_ini(
        result.stdout, name,
        search_dirs=[root_path, os.path.dirname(os.path.abspath(output_path)),
                     os.path.join(root_path, output_dir), os.path.dirname(os.path.join(root_path, base_ini))],
        since=since,
    )

    dest = output_path if os.path.isabs(output_path) else os.path.join(root_path, output_path)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.abspath(generated) != os.path.abspath(dest):
        shutil.move(generated, dest)
    print(f'create_ini: rendered INI -> {dest}')


if __name__ == '__main__':
    Fire(create_ini)
