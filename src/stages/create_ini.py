"""Stage 3 -- instantiate a PCR-GLOBWB INI from a template.

Calls ``src.utils.ini_config.create_ini_config`` directly (no subprocess). The rendered INI is written to the
deterministic ``output_path`` you specify, which is exactly what the ``run_model`` stage consumes as its
``config`` -- so there is no timestamped-filename guessing.

YAML keys map to Fire params (hyphens -> underscores), e.g. ``clone-map`` -> ``clone_map``. Give this stage
the ``output-path`` key so the orchestrator can log the generated INI as an artifact.
"""
from __init__ import root_path  # noqa: F401  -- runs src/stages/__init__.py so `import src...` resolves
from fire import Fire

from src.utils.ini_config import create_ini_config


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
    """Render ``base_ini`` into a concrete INI at ``output_path`` (see src/utils/ini_config.create_ini_config)."""
    path = create_ini_config(
        name=name,
        base_ini=base_ini,
        outputDir=output_dir,
        cloneMap=clone_map,
        inputDir=input_dir,
        landmask=landmask,
        cloneAreas=clone_areas,
        with_merging=str(with_merging),
        lowResData=low_res_data,
        highResData=high_res_data,
        institution=institution,
        title=title,
        description=description,
        novalidation=bool(novalidation),
        output_path=output_path,
    )
    print(f'create_ini: rendered INI -> {path}')


if __name__ == '__main__':
    Fire(create_ini)
