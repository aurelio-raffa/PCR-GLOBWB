"""Entrypoint for a PCR-GLOBWB pipeline run, tracked as an MLflow experiment.

This is what the LSF job (or your shell) invokes:

    python run_project.py --config_file=config/pipeline/pcrglobwb_pipeline.yaml \
                          --experiment_name=pcrglobwb_my_run

It opens (or reuses) the named MLflow experiment and launches the orchestrator
(``src/stages/run.py``) as an MLflow *project run* in the current environment
(``env_manager='local'`` -> no conda env is created; the already-active env is used).

MLflow tracking store:
  * Default (no ``--tracking_uri``): the local file store ``./mlruns`` in the repo dir. On the cluster this
    sits on the shared filesystem, so runs from every job land in one place. View it with
    ``mlflow ui`` (requires ``mlflow<3.13``; see conda_env/pcrglobwb_pipeline.yml).
  * ``--tracking_uri sqlite:///mlflow.db`` or a remote ``http://host:port`` server are also supported.
"""
import os
import logging

import mlflow

from fire import Fire

from src import console_handler

logger = logging.getLogger(__name__)
logger.addHandler(console_handler)


def run_project(
        config_file: str,
        experiment_name: str,
        tracking_uri: str = None,
):
    """Run the pipeline described by ``config_file`` as an MLflow experiment named ``experiment_name``.

    Args:
        config_file (str): Path (relative to the repo root) to the pipeline YAML, e.g.
            ``config/pipeline/pcrglobwb_pipeline.yaml``.
        experiment_name (str): MLflow experiment name to log the run under.
        tracking_uri (str, optional): Connection string to an MLflow tracking server / store. Defaults to the
            local ``./mlruns`` file store.
    """
    if tracking_uri is not None:
        redacted = ("*" * (len(tracking_uri) - 3) + tracking_uri[-3:]) if len(tracking_uri) > 3 else tracking_uri
        logger.info('Using MLflow tracking store at %s (redacted)', redacted)
        mlflow.set_tracking_uri(tracking_uri)
    else:
        logger.info('Using the local MLflow file store (./mlruns)')

    mlflow.projects.run(
        uri=os.path.dirname(os.path.abspath(__file__)),
        entry_point='src/stages/run.py',
        parameters={'config-file': config_file},
        experiment_name=experiment_name,
        env_manager='local',
    )


if __name__ == '__main__':
    Fire(run_project)
