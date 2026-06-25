"""PCR-GLOBWB run-log parser and diagnostic core.

This module holds the implementation behind two front-ends:

  * the root-level CLI shim ``parse_errlog.py`` (manual use on an LSF ``errfile`` or a Markdown-wrapped
    log), and
  * the pipeline stage ``src/stages/diagnostics.py`` (run automatically after ``run_model`` on the
    convenience logfile ``output/model_run.log`` that ``run_model`` tees out of the model).

It parses PCR-GLOBWB's logging output (``YYYY-MM-DD HH:MM:SS,ms  <issuer>  <LEVEL>  <msg>`` lines plus
Python tracebacks), de-duplicates messages by severity, and -- crucially for catching *silent* failures
-- analyses whether the run actually progressed or terminated early.

Why this exists: the parallel runner backgrounds every per-tile/merging process and ends with a bare
``wait``, so a fatal crash inside one subprocess never changes the job's exit code. ``run_model`` returns
0 and the pipeline reports success even when the model died. Reading the model's own logs is the only
way to surface that, so :func:`analyze_run` looks for tracebacks (and, specifically, a crash in the
merging/global process) and for a model run that stopped well before its configured number of steps.
"""
import csv
import json
import logging
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from src import console_handler

logger = logging.getLogger(__name__)
logger.addHandler(console_handler)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOG_PATTERN = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+)"  # timestamp
    r"\s+(\S+)"                                        # issuer
    r"\s+(DEBUG|INFO|WARNING|ERROR|CRITICAL|NOTSET)"   # level
    r"(.*)"                                            # rest of message (may be empty)
)

TRACEBACK_START = "Traceback (most recent call last):"
EXCEPTION_PATTERN = re.compile(r"^[A-Za-z][\w.]*(?:Error|Exception|Warning|Interrupt|Exit|Fault|Signal)[\w]*")

LEVEL_ORDER = ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"]
LEVEL_RANK = {level: i for i, level in enumerate(LEVEL_ORDER)}

TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S,%f"

# --- termination / completion analysis ---
# The merging ("global") process runs this script; a traceback naming it means the merge step crashed
# (e.g. the unsubstituted ``clone_%s.map`` clone-map bug) while the per-tile processes kept going.
MERGING_SCRIPT = "deterministic_runner_for_monthly_modflow_and_merging.py"
# PCR-GLOBWB logs the total number of daily steps once at start: "... number of time steps: 4018".
TIMESTEP_RE = re.compile(r"number of time steps:\s*(\d+)")
# Per-step progress markers carry the simulated date, e.g. "reporting for time 2000-01-31" or
# "Updating model for time 2000-01-31"; the max date seen is how far the run actually got.
MODEL_DATE_RE = re.compile(r"for time (\d{4}-\d{2}-\d{2})")
# The last frame of a Python traceback is the exception, e.g. "TypeError: Cannot open '...'".
PY_FILE_FRAME_RE = re.compile(r'File "([^"]+)", line (\d+)')
# Explicit fatal markers logged by the parallel launcher / supervisor / merger when the run is torn
# down WITHOUT a Python traceback: a clone wedges -> the merger logs "appears wedged" and sys.exit(1)s,
# the supervisor SIGKILLs the rest, the launcher logs the failure and exits. These are clean exits, so
# the traceback/CRITICAL heuristic below would otherwise miss a hard, run-ending failure. Each substring
# below appears at ERROR level in the model logfile that diagnostics parses.
FATAL_MARKER_RE = re.compile(
    r"(A PCR-GLOBWB clone or the merging process failed"
    r"|a clone appears wedged"
    r"|Aborting merging"
    r"|SYSTEMIC sentinel problem"
    r"|Parallel supervisor: process .* exited with code"
    r"|Timed out after .* aborting clone)"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def strip_markdown_fence(lines: list[str]) -> list[str]:
    """Return only lines inside the first ```shell ... ``` fence, or all lines if none found."""
    inside = False
    result = []
    for line in lines:
        stripped = line.rstrip("\n")
        if not inside:
            if re.match(r"^```(?:shell|bash|sh)?$", stripped.strip()):
                inside = True
        else:
            if stripped.strip() == "```":
                inside = False
            else:
                result.append(line)
    return result if result else lines


def parse_timestamp(ts: str) -> datetime:
    return datetime.strptime(ts, TIMESTAMP_FMT)


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------

def parse_log(lines: list[str]) -> list[dict]:
    """
    Parse log lines into a list of records:
        { "timestamp": datetime, "level": str, "issuer": str, "content": str }

    Handles:
      - Standard log entries (single and multi-line)
      - Tracebacks: grouped into one record, attributed to the issuer of the
        preceding log line (or "traceback" if unknown), with level "ERROR"
    """
    records = []
    current: dict | None = None
    in_traceback = False
    traceback_lines: list[str] = []
    traceback_issuer = "traceback"
    traceback_ts: datetime | None = None

    def flush_traceback():
        nonlocal in_traceback, traceback_lines
        if not in_traceback:
            return
        content = "\n".join(traceback_lines).strip()
        records.append({
            "timestamp": traceback_ts,
            "level": "ERROR",
            "issuer": traceback_issuer,
            "content": content,
        })
        traceback_lines = []
        in_traceback = False

    def flush_current():
        nonlocal current
        if current is not None:
            current["content"] = current["content"].strip()
            records.append(current)
            current = None

    for raw_line in lines:
        line = raw_line.rstrip("\n")

        # --- traceback accumulation ---
        if in_traceback:
            # A line matching LOG_PATTERN is from a concurrent process — handle normally,
            # but do NOT flush the traceback yet (it continues on subsequent non-log lines).
            if LOG_PATTERN.match(line):
                # Process as a regular log entry inline without exiting traceback mode.
                m = LOG_PATTERN.match(line)
                ts_str, issuer, level, rest = m.group(1), m.group(2), m.group(3), m.group(4)
                # Temporarily flush current (shouldn't be open during traceback), then emit.
                flush_current()
                records.append({
                    "timestamp": parse_timestamp(ts_str),
                    "level": level,
                    "issuer": issuer,
                    "content": rest.lstrip().strip(),
                })
                continue
            traceback_lines.append(line)
            # End of traceback: a line that looks like an exception class name.
            if EXCEPTION_PATTERN.match(line.strip()):
                flush_traceback()
            continue

        if line.strip() == TRACEBACK_START or line.strip().startswith(TRACEBACK_START):
            flush_current()
            in_traceback = True
            traceback_ts = records[-1]["timestamp"] if records else datetime.min
            traceback_issuer = records[-1]["issuer"] if records else "traceback"
            traceback_lines = [line.strip()]
            continue

        # --- standard log line ---
        m = LOG_PATTERN.match(line)
        if m:
            flush_current()
            ts_str, issuer, level, rest = m.group(1), m.group(2), m.group(3), m.group(4)
            current = {
                "timestamp": parse_timestamp(ts_str),
                "level": level,
                "issuer": issuer,
                "content": rest.lstrip(),
            }
        else:
            # continuation line: append to current record (or discard if no record open)
            if current is not None:
                current["content"] += "\n" + line

    flush_current()
    flush_traceback()

    return records


# ---------------------------------------------------------------------------
# Termination / completion analysis
# ---------------------------------------------------------------------------

def _exception_summary(content: str) -> str:
    """Return the last non-empty line of a traceback (the exception), trimmed for readability."""
    for ln in reversed(content.splitlines()):
        if ln.strip():
            return ln.strip()
    return ""


def analyze_run(records: list[dict]) -> dict:
    """Summarise whether the model run progressed or terminated early.

    Returns a dict mixing scalars (safe to log as MLflow metrics via :func:`write_metrics`) and a few
    human-readable strings (surfaced in the summary). Heuristics, in plain terms:

      * ``merging_crashed`` -- a traceback whose frames name :data:`MERGING_SCRIPT`. In parallel mode
        the merging/global process is spawned without a clone code, so an unsubstituted ``clone_%s.map``
        makes ``pcr.setclone`` raise here. The per-tile processes then deadlock at the monthly merge
        barrier and the job is eventually killed -- a failure the exit code never reflects.
      * ``last_model_date`` vs ``expected_timesteps`` -- how far the simulation actually advanced.
      * ``verdict`` -- a one-line plain-English call derived from the above.
    """
    counts_by_level: dict[str, int] = defaultdict(int)
    timestamps: list[datetime] = []
    tracebacks: list[dict] = []
    fatal_markers: list[str] = []
    expected_timesteps = None
    last_model_date = None

    for r in records:
        counts_by_level[r["level"]] += 1
        ts = r.get("timestamp")
        if ts is not None and ts != datetime.min:
            timestamps.append(ts)

        content = r["content"]
        if TRACEBACK_START in content:
            tracebacks.append(r)

        marker = FATAL_MARKER_RE.search(content)
        if marker:
            # keep the marker's own log line (first non-empty line) for a readable verdict
            line = next((ln.strip() for ln in content.splitlines() if ln.strip()), marker.group(0))
            fatal_markers.append(line)

        m = TIMESTEP_RE.search(content)
        if m:
            expected_timesteps = int(m.group(1))

        for dm in MODEL_DATE_RE.finditer(content):
            date_str = dm.group(1)
            if last_model_date is None or date_str > last_model_date:
                last_model_date = date_str

    tracebacks.sort(key=lambda r: r.get("timestamp") or datetime.min)
    merging_tracebacks = [t for t in tracebacks if MERGING_SCRIPT in t["content"]]

    first_traceback = tracebacks[0] if tracebacks else None
    earliest_exception = _exception_summary(first_traceback["content"]) if first_traceback else ""

    has_traceback = bool(tracebacks)
    merging_crashed = bool(merging_tracebacks)
    n_critical = counts_by_level.get("CRITICAL", 0)
    n_error = counts_by_level.get("ERROR", 0)
    # A traceback is emitted as an ERROR record, so ``n_error`` already includes it. "Fatal" means any
    # traceback or CRITICAL line, OR an explicit launcher/supervisor/merger abort marker (a hard failure
    # that exits cleanly, hence carries no traceback -- e.g. a wedged clone aborting the merger).
    has_fatal = has_traceback or n_critical > 0 or bool(fatal_markers)

    if merging_crashed:
        verdict = ("merging/global process crashed -- the parallel run cannot complete (per-tile "
                   "processes deadlock at the monthly merge barrier). Likely the merging clone map.")
    elif first_traceback is not None:
        verdict = f"fatal error: {earliest_exception or 'a traceback was logged'}"
    elif fatal_markers:
        verdict = f"run aborted: {fatal_markers[0]}"
    elif n_critical > 0:
        verdict = "CRITICAL message(s) logged -- inspect the summary"
    elif not records:
        verdict = "empty / unreadable log -- nothing to analyse"
    else:
        verdict = "no fatal errors detected in the model log"

    return {
        "n_log_entries": len(records),
        "n_critical": n_critical,
        "n_error": n_error,
        "n_warning": counts_by_level.get("WARNING", 0),
        "n_info": counts_by_level.get("INFO", 0),
        "n_tracebacks": len(tracebacks),
        "n_fatal_markers": len(fatal_markers),
        "merging_crashed": merging_crashed,
        "has_traceback": has_traceback,
        "has_fatal_error": has_fatal,
        "expected_timesteps": expected_timesteps,   # None if the model never logged it
        "last_model_date": last_model_date,         # 'YYYY-MM-DD' or None
        "first_timestamp": min(timestamps) if timestamps else None,
        "last_timestamp": max(timestamps) if timestamps else None,
        "earliest_exception": earliest_exception,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Output: CSV
# ---------------------------------------------------------------------------

def write_csv(records: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "level", "issuer", "content"])
        writer.writeheader()
        for r in records:
            ts = r["timestamp"]
            writer.writerow({
                "timestamp": ts.strftime(TIMESTAMP_FMT) if ts is not None else "",
                "level": r["level"],
                "issuer": r["issuer"],
                "content": r["content"],
            })
    print(f"[CSV]     {path}  ({len(records)} rows)")


# ---------------------------------------------------------------------------
# Output: machine-readable metrics (flat {name: number}) for the MLflow `metrics-path` convenience
# ---------------------------------------------------------------------------

def write_metrics(analysis: dict, path: Path) -> None:
    """Write the numeric subset of an analysis as a flat ``{name: number}`` JSON.

    The orchestrator's ``metrics-path`` convenience reads exactly this shape and logs each entry as an
    MLflow metric, so only numbers/bools are emitted here (bools become 0/1; ``None`` becomes -1).
    """
    def num(value):
        if isinstance(value, bool):
            return int(value)
        if value is None:
            return -1
        return value

    metrics = {
        key: num(analysis[key])
        for key in (
            "n_log_entries", "n_critical", "n_error", "n_warning", "n_info", "n_tracebacks",
            "n_fatal_markers", "merging_crashed", "has_traceback", "has_fatal_error", "expected_timesteps",
        )
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Metrics] {path}  ({len(metrics)} metrics)")


# ---------------------------------------------------------------------------
# Output: textual summary
# ---------------------------------------------------------------------------

def write_summary(records: list[dict], path: Path, analysis: dict | None = None) -> None:
    # Deduplicate by (level, issuer, content)
    key_to_meta: dict[tuple, dict] = {}
    for r in records:
        key = (r["level"], r["issuer"], r["content"])
        if key not in key_to_meta:
            key_to_meta[key] = {
                "level": r["level"],
                "issuer": r["issuer"],
                "content": r["content"],
                "count": 0,
                "first": r["timestamp"],
                "last": r["timestamp"],
            }
        entry = key_to_meta[key]
        entry["count"] += 1
        if _before(r["timestamp"], entry["first"]):
            entry["first"] = r["timestamp"]
        if _before(entry["last"], r["timestamp"]):
            entry["last"] = r["timestamp"]

    # Sort: by severity (asc rank = higher severity first), then issuer, then first occurrence
    sorted_entries = sorted(
        key_to_meta.values(),
        key=lambda e: (LEVEL_RANK.get(e["level"], 99), e["issuer"], e["first"] or datetime.min),
    )

    lines_out: list[str] = []
    lines_out.append("PCR-GLOBWB Log Summary")
    lines_out.append("=" * 70)

    # --- diagnosis header (termination / completion verdict) ---
    if analysis is not None:
        lines_out.extend(_format_diagnosis(analysis))

    lines_out.append(f"Total log entries  : {len(records)}")
    lines_out.append(f"Unique log messages: {len(sorted_entries)}")
    lines_out.append("")

    counts_by_level = defaultdict(int)
    for r in records:
        counts_by_level[r["level"]] += 1
    lines_out.append("Counts by level:")
    for level in LEVEL_ORDER:
        if level in counts_by_level:
            lines_out.append(f"  {level:<10} {counts_by_level[level]:>8}")
    lines_out.append("")

    current_level = None
    for entry in sorted_entries:
        if entry["level"] != current_level:
            current_level = entry["level"]
            lines_out.append("")
            lines_out.append(f"{'[' + current_level + ']':=<70}")
            lines_out.append("")

        content_preview = entry["content"].strip()
        # For multi-line content abbreviate to first non-empty line
        first_content_line = next(
            (ln for ln in content_preview.splitlines() if ln.strip()), content_preview
        )

        lines_out.append(f"  Issuer : {entry['issuer']}")
        lines_out.append(f"  Message: {first_content_line}")
        if content_preview != first_content_line:
            extra_lines = [ln for ln in content_preview.splitlines()[1:] if ln.strip()]
            for ln in extra_lines[:3]:
                lines_out.append(f"           {ln}")
            if len(extra_lines) > 3:
                lines_out.append(f"           ... ({len(extra_lines) - 3} more lines)")
        lines_out.append(f"  Count  : {entry['count']}")
        lines_out.append(f"  First  : {_fmt_ts(entry['first'])}")
        lines_out.append(f"  Last   : {_fmt_ts(entry['last'])}")
        lines_out.append("")

    text = "\n".join(lines_out)
    path.write_text(text, encoding="utf-8")
    print(f"[Summary] {path}  ({len(sorted_entries)} unique messages)")


def _before(a: datetime | None, b: datetime | None) -> bool:
    """True if ``a`` strictly precedes ``b``, treating None as 'unknown' (never replaces a known value)."""
    if a is None:
        return False
    if b is None:
        return True
    return a < b


def _fmt_ts(ts: datetime | None) -> str:
    return ts.strftime(TIMESTAMP_FMT) if ts is not None else "(unknown)"


def _format_diagnosis(analysis: dict) -> list[str]:
    """Render the termination/completion verdict block placed at the top of the summary."""
    out = ["", "DIAGNOSIS", "-" * 70, f"  Verdict : {analysis['verdict']}"]
    if analysis.get("earliest_exception"):
        out.append(f"  Cause   : {analysis['earliest_exception']}")
    if analysis.get("merging_crashed"):
        out.append("  Note    : the merging/global process crashed; see the [ERROR] section below.")
    progressed = analysis.get("last_model_date")
    expected = analysis.get("expected_timesteps")
    if progressed or expected is not None:
        out.append(
            f"  Progress: last simulated date = {progressed or '(none)'}"
            + (f"; configured number of time steps = {expected}" if expected is not None else "")
        )
    out.append(
        f"  Window  : {_fmt_ts(analysis.get('first_timestamp'))}"
        f"  ->  {_fmt_ts(analysis.get('last_timestamp'))}"
    )
    out.append(
        f"  Counts  : tracebacks={analysis['n_tracebacks']}  errors={analysis['n_error']}  "
        f"warnings={analysis['n_warning']}"
    )
    out.append("-" * 70)
    out.append("")
    return out


# ---------------------------------------------------------------------------
# Top-level entry points
# ---------------------------------------------------------------------------

def _read_log_lines(logfile: Path) -> list[str]:
    """Read a log file into lines, transparently stripping a Markdown ```shell ... ``` fence."""
    if not logfile.exists():
        logger.warning("error_log: logfile not found: %s", logfile)
        return []
    raw_lines = logfile.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    if logfile.suffix == ".md" or any("```" in ln for ln in raw_lines[:20]):
        return strip_markdown_fence(raw_lines)
    return raw_lines


def diagnose(
        logfile,
        summary_path=None,
        csv_path=None,
        metrics_path=None,
) -> dict:
    """Parse ``logfile`` and (optionally) write summary / CSV / metrics; return the analysis dict.

    Resilient by design: a missing or empty log yields an empty analysis and still writes (empty)
    outputs rather than raising, so the diagnostics stage never breaks an otherwise fine pipeline run.
    """
    logfile = Path(logfile)
    lines = _read_log_lines(logfile)
    records = parse_log(lines)
    analysis = analyze_run(records)
    logger.info("error_log: %s -> %s (%d entries)", logfile.name, analysis["verdict"], len(records))

    if csv_path is not None:
        write_csv(records, Path(csv_path))
    if summary_path is not None:
        write_summary(records, Path(summary_path), analysis=analysis)
    if metrics_path is not None:
        write_metrics(analysis, Path(metrics_path))

    return analysis


def run_cli(logfile: str, csv: str = None, summary: str = None) -> None:
    """Replicate the historical ``parse_errlog.py`` behaviour (used by the root CLI shim).

    Defaults mirror the original: CSV next to the log (``<log>.csv``) and summary ``<log>_summary.txt``.
    """
    src = Path(logfile)
    if not src.exists():
        raise SystemExit(f"File not found: {src}")

    csv_path = Path(csv) if csv else src.with_suffix(".csv")
    summary_path = Path(summary) if summary else src.with_name(src.stem + "_summary.txt")

    analysis = diagnose(src, summary_path=summary_path, csv_path=csv_path)
    print(f"[Parsed]  {analysis['n_log_entries']} log entries from {src.name}")
    print(f"[Verdict] {analysis['verdict']}")


# argparse parser re-exposed by the root shim (``parse_errlog.py``) so the original CLI is preserved.
import argparse  # noqa: E402  (kept next to the parser it builds)

parser = argparse.ArgumentParser(description="Parse PCR-GLOBWB error / run log files.")
parser.add_argument("logfile", help="Path to the log file (.txt, .log, or .md with shell fence)")
parser.add_argument("--csv", default=None, help="Output CSV path (default: <logfile>.csv)")
parser.add_argument("--summary", default=None, help="Output summary path (default: <logfile>_summary.txt)")
