"""Stage 1 -- set up the output directories for a PCR-GLOBWB run.

This is the cheap, idempotent prerequisite the other stages depend on: it creates the parent output
directory plus any working directories for the INI files and the LDD basin/tile maps. (PCR-GLOBWB itself
creates the per-run ``netcdf/``, ``log/``, ``states/``, ``maps/`` and ``tmp/`` subtrees under ``outputDir``
at model-launch time, in ``model/configuration.py:create_output_directories`` -- this stage only guarantees
the parents exist first.)

Generic by design (vendored from plumber): every keyword argument value is treated as a directory to create.
Absolute paths (e.g. cluster scratch) are created as-is; relative paths are created under the repo root.
Use YAML keys *without* hyphens (e.g. ``output_dir``, ``config_dir``) so Fire routes them into ``**dirs``.

    - setup:
        hard_clean: false
        output_dir:     /scratch/me/run01
        config_dir:     /scratch/me/run01/ini
        clone_maps_dir: /scratch/me/run01/clone_maps
        basins_dir:     /scratch/me/run01/basins
"""
import os
import shutil

from fire import Fire

from __init__ import root_path


def make_dirs(dir_path: str, hard_clean: bool = False) -> None:
    """Create a directory, optionally wiping it clean first.

    Args:
        dir_path: Absolute or repo-root-relative path of the directory to create.
        hard_clean: If True and the directory already exists, delete it and all its contents before
            recreating it. Default: False.
    """
    if hard_clean and os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)


def setup(hard_clean: bool = False, **dirs) -> None:
    """Create each directory passed as a keyword argument.

    Args:
        hard_clean: Remove any pre-existing contents of each directory before recreating it.
        **dirs: Any other keyword args are directory paths to create (the key names are ignored).
    """
    for dir_name in dirs.values():
        # os.path.join(root, '/abs/path') == '/abs/path', so absolute cluster paths pass through unchanged
        make_dirs(os.path.join(root_path, str(dir_name)), hard_clean=hard_clean)


if __name__ == '__main__':
    Fire(setup)
