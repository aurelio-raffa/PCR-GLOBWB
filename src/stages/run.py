"""Sequential MLflow orchestrator for the PCR-GLOBWB pipeline.

Reads a pipeline YAML (see config/pipeline/pcrglobwb_pipeline.yaml), opens a single parent ("orchestrator")
MLflow run and executes each stage strictly in order. Every stage is its own ``mlflow.run`` child process --
``python <project_uri>/<stage_name>.py --key value ...`` -- so the stage's parameters are logged automatically
and a failing stage (non-zero exit) aborts the pipeline.

This is the lean, mlflow-only port of plumber's orchestrator: the optional lazy/determinism cache, the Prefect
backend, the deterministic seeding and the welcome banner have all been dropped. What remains is the dual idea
that makes the harness useful here -- a tracked parent run plus per-stage child runs -- with a few logging
conveniences (artifacts/metrics) and one control key, ``skip``, used to make the LDD-basins stage optional.

Per-stage YAML keys and how the orchestrator treats them:
  * ``skip: true`` -- the one control key: skip the stage entirely (it never reaches the stage CLI).
  * Every other key is ALWAYS forwarded verbatim to the stage's Fire CLI as ``--<key> <value>``. The
    orchestrator never renames, drops, or hides a parameter from its stage -- so each stage in the YAML
    reads exactly as the CLI invocation it becomes.
  * Additive artifact logging: *after* a stage runs, any parameter whose NAME is in
    :data:`ARTIFACT_PARAM_NAMES` has the file at its path logged as an MLflow artifact on the parent run
    -- by default only when the file extension is in :data:`ARTIFACT_DEFAULT_EXTENSIONS`, or for any
    extension when the pipeline's ``log_artifacts`` is true. This is pure metadata logging; it does not
    affect what the stage received.
  * ``metrics-path: <p>`` is, in addition, read as a flat ``{name: number}`` JSON and logged as a metric
    on both the stage run and the parent run.
The fully-substituted pipeline YAML is always logged as an artifact on the parent run.
"""
import os
import sys
import json
import logging
import tempfile
from dataclasses import dataclass

import yaml
import mlflow

from fire import Fire

from __init__ import root_path, console_handler
from src.utils.io.parse_config import parse_config

logger = logging.getLogger(__name__)
logger.addHandler(console_handler)

# --- artifact-name allowlist (see module docstring) -------------------------------------------------
# A parameter is a candidate for artifact logging when its NAME (normalised: lower-case, '_'->'-') is in
# this set. The handling is purely additive -- the parameter is still forwarded to the stage unchanged.
ARTIFACT_PARAM_NAMES = {
    'config', 'config-path', 'config-file', 'config-ini',
    'summary', 'summary-path', 'output-summary',
    'metrics', 'metrics-path', 'metrics-config', 'output-metrics',
    'csv-path', 'output-csv',
    'output-path', 'output-image',
}
# When ``log_artifacts`` is false (the default), only candidate files with one of these extensions are
# logged -- the small, always-worth-keeping text artifacts. With ``log_artifacts`` true, any extension is
# logged (e.g. .png images, .nc model output, the .log convenience file).
ARTIFACT_DEFAULT_EXTENSIONS = {'.ini', '.yaml', '.yml', '.json', '.txt', '.csv'}


def _normalize_param_name(name: str) -> str:
    """Normalise a YAML/CLI parameter name for allowlist matching (lower-case, '_'->'-')."""
    return str(name).strip().lower().replace('_', '-')


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


def _log_resolved_config(config: dict) -> None:
    """Always archive the fully-substituted pipeline YAML (placeholders resolved) on the active run.

    Captures the exact, resolved pipeline that ran -- the single source of truth for *what was executed*.
    It holds real cluster paths, but MLflow artifacts live under git-ignored ``mlruns/`` so nothing
    reaches the (public) git history.
    """
    resolved_dir = tempfile.mkdtemp(prefix='pcrglobwb_pipeline_')
    resolved_path = os.path.join(resolved_dir, 'pipeline_resolved.yaml')
    with open(resolved_path, 'w') as handle:
        yaml.safe_dump(config, handle, sort_keys=False, default_flow_style=False)
    mlflow.log_artifact(resolved_path)


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
    # stringified so YAML ints/bools pass through cleanly. EVERY parameter is forwarded (only `skip` was
    # popped above, as a control key). A non-zero exit raises and aborts the pipeline.
    cli_parameters = {key: str(value) for key, value in parameters.items()}
    current_run = mlflow.run(
        uri='',
        entry_point=f'{ctx.project_uri}/{stage_name}.py',
        parameters=cli_parameters,
        env_manager='local',
    )

    # --- additive artifact logging: archive allowlisted-name params that point at existing files -----------
    # Purely metadata; the stage already received these params unchanged. Logged on the parent run so the
    # orchestrator collects the whole run-defining bundle (generated INI, summaries, metrics, ...).
    already_logged: set = set()
    for key, value in parameters.items():
        if _normalize_param_name(key) not in ARTIFACT_PARAM_NAMES:
            continue
        path = str(value)
        if not os.path.isfile(path) or path in already_logged:
            continue
        extension = os.path.splitext(path)[1].lower()
        if ctx.log_artifacts or extension in ARTIFACT_DEFAULT_EXTENSIONS:
            client.log_artifact(ctx.orchestrator_run_id, path)
            already_logged.add(path)

    # --- metrics: `metrics-path` JSON is additionally parsed into MLflow metrics (stage + parent) ----------
    if 'metrics-path' in parameters and os.path.isfile(parameters['metrics-path']):
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
        mlflow.log_artifact(config_file)            # the raw YAML (with {{$VAR}} placeholders, reproducible)
        _log_resolved_config(config)                # ALWAYS archive the fully-substituted YAML actually used
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
