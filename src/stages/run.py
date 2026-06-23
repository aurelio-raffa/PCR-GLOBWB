"""Sequential MLflow orchestrator for the PCR-GLOBWB pipeline.

Reads a pipeline YAML (see config/pipeline/pcrglobwb_pipeline.yaml), opens a single parent ("orchestrator")
MLflow run and executes each stage strictly in order. Every stage is its own ``mlflow.run`` child process --
``python <project_uri>/<stage_name>.py --key value ...`` -- so the stage's parameters are logged automatically
and a failing stage (non-zero exit) aborts the pipeline.

This is the lean, mlflow-only port of plumber's orchestrator: the optional lazy/determinism cache, the Prefect
backend, the deterministic seeding and the welcome banner have all been dropped. What remains is the dual idea
that makes the harness useful here -- a tracked parent run plus per-stage child runs -- with a few logging
conveniences (artifacts/metrics) and one control key, ``skip``, used to make the LDD-basins stage optional.

Per-stage YAML keys with special meaning:
  * ``skip: true``         -- orchestrator-only; skip the stage entirely (never reaches the stage CLI).
  * ``config-path: <p>``   -- after the stage runs, log file ``<p>`` as an artifact on the parent run.
  * ``output-path: <p>``   -- if the pipeline's ``log_artifacts`` is true, log ``<p>`` (file or dir) as an
                              artifact on the stage's child run.
  * ``metrics-path: <p>``  -- read this flat ``{name: number}`` JSON and log every entry as a metric on both
                              the stage run and the parent run.
All other keys are forwarded verbatim to the stage's Fire CLI as ``--<key> <value>``.
"""
import os
import sys
import json
import logging
from dataclasses import dataclass

import mlflow

from fire import Fire

from __init__ import root_path, console_handler
from src.utils.io.parse_config import parse_config

logger = logging.getLogger(__name__)
logger.addHandler(console_handler)


def _coerce_bool(value, default: bool = False) -> bool:
    """Interpret a YAML scalar (or None) as a boolean, falling back to ``default`` when unset."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ('1', 'true', 'yes', 'on')


def _log_metrics_to_run(client, run_id: str, metrics_path: str) -> dict:
    """Read a stage's flat metrics JSON and log each scalar to the given run; return the metrics dict."""
    with open(metrics_path) as handle:
        metrics = json.load(handle)
    for key, value in metrics.items():
        client.log_metric(run_id=run_id, key=key, value=value)
    return metrics


@dataclass
class PipelineContext:
    """Shared per-run state threaded into every :func:`execute_stage` call."""
    tracking_client: object              # mlflow.tracking.MlflowClient
    orchestrator_run_id: str             # the parent ("orchestrator") MLflow run
    project_uri: str
    log_artifacts: bool


def execute_stage(stage_name: str, parameters: dict, index: int, total: int, ctx: PipelineContext) -> None:
    """Run (or skip) a single pipeline stage as its own ``mlflow.run`` child process."""
    client = ctx.tracking_client
    parameters = dict(parameters) if parameters else {}

    # `skip` is an orchestrator-only control key (makes e.g. the LDD-basins stage optional); it never reaches
    # the stage CLI.
    if _coerce_bool(parameters.pop('skip', None), False):
        logger.info('skipping stage "%s" (skip: true)', stage_name)
        client.set_tag(run_id=ctx.orchestrator_run_id, key=f'skipped_{stage_name}', value='true')
        return

    print(file=sys.stderr)
    print(f'==============> Stage {index + 1}/{total}: "{stage_name}" '.ljust(103, '='), file=sys.stderr)

    # mlflow.run with no MLproject runs `python <entry_point> --key value ...` in the repo dir; values are
    # stringified so YAML ints/bools pass through cleanly. A non-zero exit raises and aborts the pipeline.
    cli_parameters = {key: str(value) for key, value in parameters.items()}
    current_run = mlflow.run(
        uri='',
        entry_point=f'{ctx.project_uri}/{stage_name}.py',
        parameters=cli_parameters,
        env_manager='local',
    )

    # --- logging conveniences (addressed by explicit run_id, so they do not depend on the fluent active run) ---
    if 'config-path' in parameters and os.path.exists(parameters['config-path']):
        client.log_artifact(ctx.orchestrator_run_id, parameters['config-path'])

    if ctx.log_artifacts and 'output-path' in parameters and os.path.exists(parameters['output-path']):
        client.log_artifact(run_id=current_run.run_id, local_path=parameters['output-path'])

    if 'metrics-path' in parameters and os.path.exists(parameters['metrics-path']):
        metrics = _log_metrics_to_run(client, current_run.run_id, parameters['metrics-path'])
        for key, value in metrics.items():
            client.log_metric(run_id=ctx.orchestrator_run_id, key=key, value=value)


def run(config_file: str):
    """Parse the pipeline YAML and execute its stages sequentially under one MLflow run."""
    config = parse_config(os.path.join(root_path, config_file))

    project_uri = config['project_uri']
    log_artifacts = _coerce_bool(config.get('log_artifacts'), False)
    tags = config.get('tags', {}) or {}

    tracking_client = mlflow.tracking.MlflowClient()
    with mlflow.start_run() as orchestrator:
        mlflow.log_artifact(config_file)            # snapshot the exact pipeline config used
        if tags:
            mlflow.set_tags(tags)

        ctx = PipelineContext(
            tracking_client=tracking_client,
            orchestrator_run_id=orchestrator.info.run_id,
            project_uri=project_uri,
            log_artifacts=log_artifacts,
        )

        stages = config['stages']
        total_steps = len(stages)
        for index, stage in enumerate(stages):
            for stage_name, parameters in stage.items():
                execute_stage(stage_name, parameters, index, total_steps, ctx)

    logger.info('Pipeline complete (%d stage(s)).', total_steps)


if __name__ == '__main__':
    Fire(run)
