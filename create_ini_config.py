"""
Configuration File Generator for PCR-GLOBWB

This module generates PCR-GLOBWB configuration (.ini) files from a template
by substituting command-line arguments into placeholder fields. It enables
dynamic configuration generation for different data sources, output locations,
and model parameters without manual file editing.

Key features:
- Reads a base INI template with {placeholder} syntax
- Substitutes placeholders with command-line arguments
- Validates that referenced file paths exist on disk
- Suggests alternative files when paths are missing (same stem, different ext)
- Generates timestamped output filenames

Typical usage:
    python create_ini_config.py \\
        --name=experiment_v1 \\
        --base_ini=config/template_30min.ini \\
        --outputDir=/scratch/output \\
        --cloneMap=./maps/clone_30min.map \\
        --inputDir=/data/pcr_inputs

The generated file is named config_YYMMDDHHMM_<name>_<template_basename>.ini
and is ready for use with PCR-GLOBWB parallel runners.
"""
import configparser
import os
import pathlib
import argparse
from datetime import datetime


# Argument name keys: map CLI flags to placeholder names in INI templates
name = 'name'                       # Identifier for the run (used in output filename)
ini_template = 'base_ini'           # Path to base INI template file to read
outputDir = 'outputDir'             # Output directory for model results
cloneMap = 'cloneMap'               # Path to clone map file (geographic regions)
inputDir = 'inputDir'               # Base input data directory
landmask = 'landmask'               # Landmask file or identifier
institution = 'institution'         # Institution name (metadata)
title = 'title'                     # Configuration title (metadata)
description = 'description'         # Configuration description (metadata)
lowResData = 'lowResData'           # Low-resolution subdirectory (30-minute resolution)
highResData = 'highResData'         # High-resolution subdirectory (5-minute resolution)
novalidation = 'novalidation'       # Flag to skip path validation checks
cloneAreas = 'cloneAreas'           # Clone/tile selection for parallel execution
with_merging = 'with_merging'       # Whether to merge outputs after parallel run

# Default values for optional parallel execution parameters
CLONE_AREAS_DEFAULT = 'Global'      # Run all clones (M01-M53)
WITH_MERGING_DEFAULT = 'True'       # Enable output merging by default
WITH_MERGING_CHOICES = ('True', 'False')  # Valid values for merging option

# Help message and CLI argument parser definition
help_msg: str = f"""Generate a configuration (.ini) file from a template with dynamic arguments.
This script reads a base INI template file, replaces placeholders using
command-line arguments, and writes a new configuration file with a timestamped
filename. The naming pattern for the output file is 
\"config_{{%y%m%d%H%M}}_{{args.{name}}}_{{args.{ini_template}.filename}}\""""

parser = argparse.ArgumentParser(description=help_msg)
parser.add_argument(
    f"--{name}",
    type=str,
    help='Identifier used in the output filename'
)
parser.add_argument(
    f"--{ini_template}",
    type=str,
    help='Path to the base INI template file'
)
parser.add_argument(
    f"--{outputDir}",
    type=str,
    help='Output directory to be injected into the template'
)
parser.add_argument(
    f"--{cloneMap}",
    type=str,
    help='Path or identifier for the clone map'
)
parser.add_argument(
    f"--{inputDir}",
    type=str,
    help='Input data directory'
)
parser.add_argument(
    f"--{landmask}",
    default='None',
    type=str,
    help='Landmask file or identifier. Defaults to "None"'
)
parser.add_argument(
    f"--{institution}",
    default='""',
    type=str,
    help='Name of the institution (metadata). Defaults to empty string'
)
parser.add_argument(
    f"--{title}",
    default='""',
    type=str,
    help='Title metadata for the configuration. Defaults to empty string'
)
parser.add_argument(
    f"--{description}",
    default='""',
    type=str,
    help='Description metadata for the configuration. Defaults to empty string'
)
parser.add_argument(
    f"--{lowResData}",
    default='global_30min',
    type=str,
    help=f'{inputDir} subdirectory for low-resolution data. Defaults to "global_30min"'
)
parser.add_argument(
    f"--{highResData}",
    default='global_05min',
    type=str,
    help=f'{inputDir} subdirectory for high-resolution data. Defaults to "global_05min"'
)
parser.add_argument(
    f"--{novalidation}",
    action='store_true',
    default=False,
    help=f'If provided, skips path validation in the generated config file and raise an error if not found.'
)
parser.add_argument(
    f"--{cloneAreas}",
    default=CLONE_AREAS_DEFAULT,
    type=str,
    help=(
        f'Parallel-run clone selection (only consumed by parallel templates). '
        f'Allowed: "Global", "part_one", "part_two", or a comma-separated list '
        f'like "M01,M03,M17". Defaults to "{CLONE_AREAS_DEFAULT}".'
    )
)
parser.add_argument(
    f"--{with_merging}",
    default=WITH_MERGING_DEFAULT,
    type=str,
    choices=WITH_MERGING_CHOICES,
    help=(
        f'Whether the parallel runner performs in-line merging (only consumed '
        f'by parallel templates). Defaults to "{WITH_MERGING_DEFAULT}".'
    )
)


def validate_paths_in_ini(
        ini_content: str,
        data_dir: str = None,
        raise_on_missing: bool = False
) -> None:
    """
    Validate that all file paths referenced in an INI configuration exist on disk.

    Scans the INI configuration and checks each path-like value to ensure the
    corresponding file or directory exists. Path-like values are identified by
    the presence of forward slashes (/) or backslashes (\\).

    For missing paths that have a file extension, the parent directory is
    scanned to suggest alternative files with the same stem but different
    extension (e.g., if "file.nc" is missing, suggests "file.tif" if it exists).

    Args:
        ini_content: String containing the INI configuration to validate.
        data_dir: Optional base directory to prepend to relative paths.
                 Defaults to empty string (paths treated as-is).
        raise_on_missing: If True, raises ValueError if any paths are missing.
                         If False (default), only prints warnings.

    Returns:
        None. Prints warnings to stdout for missing paths and suggestions.
              May raise ValueError if raise_on_missing=True and paths are missing.

    Raises:
        ValueError: Only if raise_on_missing=True and at least one path is missing.
    """
    # Default empty data_dir to empty string for path joining
    data_dir = '' if data_dir is None else data_dir

    # Parse INI content as a config object
    config = configparser.RawConfigParser(strict=False)
    config.optionxform = str  # Preserve original case of keys
    try:
        config.read_string(ini_content)
    except configparser.Error:
        # Silently skip if INI is malformed
        return

    # Helper function to identify path-like values
    # (contains slashes/backslashes but is not only slashes)
    def _is_path_like(value: str) -> bool:
        s = value.strip()
        return bool(s) and ('/' in s or '\\' in s) and not set(s).issubset(set('\\/'))

    # Scan all sections and keys for missing paths
    missing: list[tuple[str, str, pathlib.Path]] = []
    for section in config.sections():
        for key, raw_value in config.items(section):
            # Handle multi-line values (some INI fields span multiple lines)
            for token in raw_value.splitlines():
                token = token.strip()
                if not _is_path_like(token):
                    continue

                # Construct full path and check existence
                p = pathlib.Path(os.path.join(data_dir, token))
                if not p.exists():
                    missing.append((section, key, p))

    # Exit silently if all paths exist
    if not missing:
        return

    # Print warnings for missing paths
    print(f"\nWARNING: {len(missing)} path(s) referenced in the config do not exist on disk:")
    for section, key, path in missing:
        print(f"\t[{section}] {key} = {path}")

        # Attempt to suggest alternative files with same stem but different extension
        if path.suffix and path.parent.is_dir():
            suggestions = sorted(
                p for p in path.parent.iterdir()
                if p.stem == path.stem and p.suffix != path.suffix and p.is_file()
            )
            if suggestions:
                print(f"\tPossible replacements (same name, different extension):")
                for suggestion in suggestions:
                    print(f"\t\t{suggestion}")
            else:
                print(f"\tNo replacements found!")
        else:
            # Parent directory itself is missing
            print(f"\tParent directory not found!")

    # Raise error if validation failure is not acceptable
    if raise_on_missing:
        raise ValueError(f'{len(missing)} missing path(s) in generated ini config file.')


def create_ini_config():
    """
    Main entry point: generate a PCR-GLOBWB configuration file from a template.

    Execution flow:
    1. Parse command-line arguments specifying template, substitution values, and metadata
    2. Read the base INI template file (containing {placeholder} syntax)
    3. Substitute all placeholders using the provided argument values
    4. Append a generation timestamp comment for traceability
    5. Validate that all referenced file paths exist on disk (optional, default: enabled)
    6. Write the final configuration to a timestamped output file
    7. Print the output filename to stdout

    Output filename format:
        config_YYMMDDHHMM_<name>_<template_basename>.ini

    Path validation behavior:
    - By default, missing paths raise an error and halt execution.
    - Use --novalidation flag to skip validation and continue even if paths are missing.
    - The script suggests alternative files with matching stem but different extension.

    Returns:
        None (side effects: writes .ini file to disk, prints filename to stdout).

    Raises:
        SystemExit: If the template file cannot be opened or read.
        ValueError: If path validation fails and --novalidation is not set.

    Example:
        python create_ini_config.py \\
            --name=experiment_v1 \\
            --institution="My Institution" \\
            --title="Global Simulation 2020-2021" \\
            --base_ini=config/template_30min.ini \\
            --outputDir=/scratch/output \\
            --cloneMap=./maps/clone_30min.map \\
            --inputDir=/data/pcr_inputs
    """
    # Stage 1: Parse and prepare arguments
    args = parser.parse_args()
    now = datetime.now()

    # Stage 2: Read the base INI template file
    # The template contains {placeholder} syntax that matches CLI argument names
    with open(vars(args)['base_ini'], 'r') as handle:
        base_ini = handle.read()

    # Stage 3: Substitute all placeholders with argument values
    # Uses Python's string.format() with the args namespace dict
    full_ini = base_ini.format(**vars(args))

    # Stage 4: Append generation metadata as a comment for traceability
    full_ini += f'\n# Automatically generated by create_ini_config.py from "{vars(args)["base_ini"]}" on '
    full_ini += now.strftime("%d/%m/%y at %H:%M")

    # Stage 5: Validate that all file paths in the config exist (unless --novalidation)
    # raise_on_missing=True forces strict validation; False allows warnings only
    validate_paths_in_ini(
        full_ini,
        data_dir=getattr(args, inputDir),
        raise_on_missing=not args.novalidation
    )

    # Stage 6: Generate timestamped output filename and write to disk
    # Pattern: config_YYMMDDHHMM_<experiment_name>_<template_filename>
    filename = f'config_{now.strftime("%y%m%d%H%M")}_{vars(args)["name"]}_' + os.path.split(vars(args)['base_ini'])[-1]
    with open(filename, 'w') as handle:
        handle.write(full_ini)

    # Stage 7: Print filename so user can reference it or pipe it to other commands
    print(filename)


# Script entry point: execute config generation when run as a standalone script
if __name__ == "__main__":
    create_ini_config()
