"""Sequential MLflow orchestrator for the PCR-GLOBWB pipeline.

Reads a pipeline YAML (see config/pipeline/pcrglobwb_pipeline.yaml), opens a single parent ("orchestrator")
MLflow run and executes each stage strictly in order. Every stage is its own ``mlflow.run`` child process --
``python <project_uri>/<stage_name>.py --key value ...`` -- so the stage's parameters are logged automatically
and a failing stage (non-zero exit) aborts the pipeline.

This is the lean, mlflow-only port of plumber's orchestrator: the optional lazy/determinism cache, the Prefect
backend, the deterministic seeding and the welcome banner have all been dropped. What remains is the dual idea
that makes the harness useful here -- a tracked parent run plus per-stage child runs -- with a few logging
conveniences (artifacts/metrics) and two control keys (``skip`` and ``continue-on-error``).

Per-stage YAML keys and how the orchestrator treats them:
  * ``skip: true`` -- control key: skip the stage entirely (it never reaches the stage CLI).
  * ``continue-on-error: true`` -- control key: a non-zero exit from this stage does NOT abort the pipeline
    immediately. The failure is recorded and the pipeline is failed only AFTER every remaining stage has run.
    This lets a crashed ``run_model`` still flow into the ``diagnostics`` stage (which writes + logs its
    outputs) before the pipeline is failed. Like ``skip``, this key never reaches the stage CLI.
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


def _log_stage_artifacts(client, ctx: 'PipelineContext', parameters: dict, stage_run_id) -> None:
    """Archive a stage's allowlisted-name file params on the parent run, and log its ``metrics-path``
    JSON as MLflow metrics (on the stage run, when known, and the parent).

    Side-effect only and deliberately NON-RAISING: it is called both after a successful stage and,
    best-effort, after a failed ``continue-on-error`` stage, so whatever outputs the stage managed to
    write (e.g. the diagnostics summary/CSV/metrics) are still captured even when the stage exited
    non-zero. ``stage_run_id`` is None when the stage run id is unknown (a failed stage).
    """
    already_logged: set = set()
    for key, value in parameters.items():
        if _normalize_param_name(key) not in ARTIFACT_PARAM_NAMES:
            continue
        path = str(value)
        if not os.path.isfile(path) or path in already_logged:
            continue
        extension = os.path.splitext(path)[1].lower()
        if ctx.log_artifacts or extension in ARTIFACT_DEFAULT_EXTENSIONS:
            try:
                client.log_artifact(ctx.orchestrator_run_id, path)
                already_logged.add(path)
            except Exception as exc:  # logging is a convenience; never let it break the pipeline
                logger.warning('could not log artifact %s: %s', path, exc)

    metrics_path = parameters.get('metrics-path')
    if metrics_path and os.path.isfile(metrics_path):
        try:
            with open(metrics_path) as handle:
                metrics = json.load(handle)
            for key, value in metrics.items():
                if stage_run_id is not None:
                    client.log_metric(run_id=stage_run_id, key=key, value=value)
                client.log_metric(run_id=ctx.orchestrator_run_id, key=key, value=value)
        except Exception as exc:
            logger.warning('could not log metrics from %s: %s', metrics_path, exc)


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


def execute_stage(stage_name: str, parameters: dict, index: int, total: int, ctx: PipelineContext) -> dict | None:
    """Run (or skip) a single pipeline stage as its own ``mlflow.run`` child process.

    Returns ``None`` on success or skip. When a stage marked ``continue-on-error: true`` exits non-zero,
    the failure is NOT raised; instead this returns a ``{'stage', 'error'}`` record so the caller can run
    the remaining stages and fail the pipeline at the end. Any other stage failure raises as before.
    """
    client = ctx.tracking_client
    parameters = dict(parameters) if parameters else {}

    # `skip` is an orchestrator-only control key (makes e.g. the LDD-basins stage optional); it never reaches
    # the stage CLI.
    if _coerce_bool(parameters.pop('skip', None), False):
        logger.info('skipping stage "%s" (skip: true)', stage_name)
        client.set_tag(run_id=ctx.orchestrator_run_id, key=f'skipped_{stage_name}', value='true')
        return None

    # `continue-on-error` is the other orchestrator-only control key (see module docstring); also popped so
    # it never reaches the stage CLI.
    continue_on_error = _coerce_bool(parameters.pop('continue-on-error', None), False)

    print(file=sys.stderr)
    print(f'==============> Stage {index + 1}/{total}: "{stage_name}" '.ljust(103, '='), file=sys.stderr)

    # mlflow.run with no MLproject runs `python <entry_point> --key value ...` in the repo dir; values are
    # stringified so YAML ints/bools pass through cleanly. EVERY parameter is forwarded (only the control
    # keys were popped above). A non-zero exit raises ExecutionException.
    cli_parameters = {key: str(value) for key, value in parameters.items()}
    try:
        current_run = mlflow.run(
            uri='',
            entry_point=f'{ctx.project_uri}/{stage_name}.py',
            parameters=cli_parameters,
            env_manager='local',
        )
    except Exception as exc:
        if not continue_on_error:
            raise
        # Deferred failure: record it, tag the parent run, and still capture whatever outputs the stage
        # wrote (so e.g. a crashed run_model does not stop the pipeline before diagnostics runs).
        logger.error('stage "%s" FAILED (%s); continuing because continue-on-error is set, so the '
                     'remaining stages still run. The pipeline will be failed at the end.', stage_name, exc)
        client.set_tag(run_id=ctx.orchestrator_run_id, key=f'failed_{stage_name}', value='true')
        _log_stage_artifacts(client, ctx, parameters, stage_run_id=None)
        return {'stage': stage_name, 'error': str(exc)}

    # --- additive artifact + metrics logging (purely metadata; the stage already received these params) ----
    # Logged on the parent run so the orchestrator collects the whole run-defining bundle (generated INI,
    # summaries, metrics, ...).
    _log_stage_artifacts(client, ctx, parameters, stage_run_id=current_run.run_id)
    return None


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
        deferred_failures: list = []
        for index, stage in enumerate(stages):
            for stage_name, parameters in stage.items():
                failure = execute_stage(stage_name, parameters, index, total_steps, ctx)
                if failure is not None:
                    deferred_failures.append(failure)

        # Fail the pipeline now if any `continue-on-error` stage failed -- but only after every remaining
        # stage ran (e.g. diagnostics computed and logged its outputs). Raising inside the active run marks
        # the parent run FAILED and exits non-zero, so the failure still propagates to the caller/scheduler.
        if deferred_failures:
            names = ', '.join(f['stage'] for f in deferred_failures)
            logger.error('Pipeline FAILED: stage(s) [%s] reported failure. The remaining stages still ran '
                         'and their outputs were written/logged; failing the pipeline now.', names)
            raise SystemExit(f'pipeline failed: stage(s) [{names}] reported failure')

    logger.info('Pipeline complete (%d stage(s)).', total_steps)


if __name__ == '__main__':
    Fire(run)
