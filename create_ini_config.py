import configparser
import os
import pathlib

import argparse
from datetime import datetime

name = 'name'
ini_template = 'base_ini'
outputDir = 'outputDir'
cloneMap = 'cloneMap'
inputDir = 'inputDir'
landmask = 'landmask'
institution = 'institution'
title = 'title'
description = 'description'
lowResData = 'lowResData'
highResData = 'highResData'
validate = 'validate'

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
    f"--{validate}",
    default=True,
    type=bool,
    help=f'Validate the paths contained in the generated config file and raise an error if not found. Defaults to True'
)


def validate_paths_in_ini(
        ini_content: str,
        data_dir: str,
        raise_on_missing: bool = False
) -> None:
    """Check that all path-like values in the generated ini exist on disk.

    A value is treated as a path if it contains '/' or '\\' (but not only those).
    For each missing path that carries a file extension, the parent directory
    is scanned for files sharing the same stem but a different extension, and
    those are printed as suggested replacements.
    """
    config = configparser.RawConfigParser(strict=False)
    config.optionxform = str  # preserve original key casing
    try:
        config.read_string(ini_content)
    except configparser.Error:
        return  # malformed ini section; skip silently

    def _is_path_like(value: str) -> bool:
        s = value.strip()
        return bool(s) and ('/' in s or '\\' in s) and not set(s).issubset(set('\\/'))

    missing: list[tuple[str, str, pathlib.Path]] = []
    for section in config.sections():
        for key, raw_value in config.items(section):
            for token in raw_value.splitlines():
                token = token.strip()
                if not _is_path_like(token):
                    continue
                p = pathlib.Path(os.path.join(data_dir, token))
                if not p.exists():
                    missing.append((section, key, p))

    if not missing:
        return

    print(f"\nWARNING: {len(missing)} path(s) referenced in the config do not exist on disk:")
    for section, key, path in missing:
        print(f"\t[{section}] {key} = {path}")
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
            print(f"\tParent directory not found!")

    if raise_on_missing:
        raise ValueError(f'{len(missing)} missing path(s) in generated ini config file.')


def create_ini_config():
    f"""{help_msg}
    
    Behavior:
        1. Reads the base INI template file.
        2. Writes the new configuration (with comment timestamped) to a new file with a timestamped name.
        3. Prints the name of the generated file.
        
    Example usage:
        python create_ini_config.py \\
            --{name}=NAME_FOR_THE_EXPERIMENT \\
            --{institution}=YOUR_INSITUTION \\
            --{title}=TITLE_FOR_THE_RUN \\
            --{ini_template}=config/30min.ini \\
            --{outputDir}=YOUR_OUTPUT_DIR \\
            --{cloneMap}=./clone_landmask_maps/clone_global_30min.map \\
            --{inputDir}=YOUR_DATA_DIR
    """
    # parsing arguments
    args = parser.parse_args()
    now = datetime.now()

    # reading the base ini template
    with open(vars(args)['base_ini'], 'r') as handle:
        base_ini = handle.read()

    # filling in placeholders and checking the generated result
    full_ini = base_ini.format(**vars(args))
    full_ini += f'\n# Automatically generated by create_ini_config.py from "{vars(args)["base_ini"]}" on '
    full_ini += now.strftime("%d/%m/%y at %H:%M")
    validate_paths_in_ini(full_ini, data_dir=args.get(inputDir, ''), raise_on_missing=args.validate)

    # saving to disk and displaying
    filename = f'config_{now.strftime("%y%m%d%H%M")}_{vars(args)["name"]}_' + os.path.split(vars(args)['base_ini'])[-1]
    with open(filename, 'w') as handle:
        handle.write(full_ini)
    print(filename)


if __name__ == "__main__":
    create_ini_config()
