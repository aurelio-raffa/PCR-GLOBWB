"""Stage (after run_model) -- diagnose the model run from its convenience logfile.

Parses ``log_file`` (the ``output/model_run.log`` that ``run_model`` tees out of the model), writes a
human summary + CSV + a flat metrics JSON, and -- importantly -- decides whether the run progressed or
terminated early. This catches *silent* failures: the parallel runner backgrounds every process and
ends with a bare ``wait``, so a fatal crash inside a subprocess (e.g. the merging/global process) never
changes the job's exit code and the pipeline would otherwise report success.

Wraps ``src.utils.error_log.diagnose``. The output paths are deliberately named with the orchestrator's
artifact-name allowlist keys (``summary-path``, ``csv-path``, ``metrics-path``): they are genuine
arguments this stage writes to, and ``src/stages/run.py`` *additionally* archives ``summary-path``/
``csv-path`` as MLflow artifacts and logs ``metrics-path`` as MLflow metrics -- without altering this
invocation.

Also runnable standalone:
    python src/stages/diagnostics.py --log-file run.log --summary-path summary.txt
"""
from __init__ import root_path  # noqa: F401  -- runs src/stages/__init__.py so `import src...` resolves
from fire import Fire

from src.utils.error_log import diagnose


def diagnostics(
        log_file: str,
        summary_path: str,
        csv_path: str = '',
        metrics_path: str = '',
        fail_on_error: bool = False,
) -> None:
    """Analyse ``log_file`` and write the diagnostic outputs.

    Args:
        log_file: The model convenience logfile to analyse (``run_model``'s ``log-file``).
        summary_path: Where to write the human-readable summary (diagnosis verdict + deduplicated log).
        csv_path: Optional CSV of every parsed log record.
        metrics_path: Optional flat ``{name: number}`` JSON (errors/tracebacks/etc.) for MLflow metrics.
        fail_on_error: If True, exit non-zero when a fatal model error is detected -- opt-in so a silent
            model crash can be made to fail the pipeline. Default False keeps the stage non-fatal.
    """
    analysis = diagnose(
        log_file,
        summary_path=summary_path or None,
        csv_path=csv_path or None,
        metrics_path=metrics_path or None,
    )
    print(f'diagnostics: {analysis["verdict"]}')

    if fail_on_error and analysis['has_fatal_error']:
        raise SystemExit(f'diagnostics: fatal model error detected -- {analysis["verdict"]}')


if __name__ == '__main__':
    Fire(diagnostics)
