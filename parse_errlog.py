"""
PCR-GLOBWB error log parser.

Usage:
    python parse_errlog.py <logfile> [--csv <output.csv>] [--summary <output.txt>]

Accepts:
  - Raw log files (stderr / .txt / .log)
  - Markdown files wrapping the log in a ```shell ... ``` code fence

Outputs:
  - CSV  : one row per log entry with columns timestamp, level, issuer, content
  - TXT  : deduplicated summary ordered by severity (CRITICAL > ERROR > WARNING > INFO > DEBUG)
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

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
# Output: CSV
# ---------------------------------------------------------------------------

def write_csv(records: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "level", "issuer", "content"])
        writer.writeheader()
        for r in records:
            writer.writerow({
                "timestamp": r["timestamp"].strftime(TIMESTAMP_FMT),
                "level": r["level"],
                "issuer": r["issuer"],
                "content": r["content"],
            })
    print(f"[CSV]     {path}  ({len(records)} rows)")


# ---------------------------------------------------------------------------
# Output: textual summary
# ---------------------------------------------------------------------------

def write_summary(records: list[dict], path: Path) -> None:
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
        if r["timestamp"] < entry["first"]:
            entry["first"] = r["timestamp"]
        if r["timestamp"] > entry["last"]:
            entry["last"] = r["timestamp"]

    # Sort: by severity (asc rank = higher severity first), then issuer, then first occurrence
    sorted_entries = sorted(
        key_to_meta.values(),
        key=lambda e: (LEVEL_RANK.get(e["level"], 99), e["issuer"], e["first"]),
    )

    lines_out: list[str] = []
    lines_out.append("PCR-GLOBWB Log Summary")
    lines_out.append("=" * 70)
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
        lines_out.append(f"  First  : {entry['first'].strftime(TIMESTAMP_FMT)}")
        lines_out.append(f"  Last   : {entry['last'].strftime(TIMESTAMP_FMT)}")
        lines_out.append("")

    text = "\n".join(lines_out)
    path.write_text(text, encoding="utf-8")
    print(f"[Summary] {path}  ({len(sorted_entries)} unique messages)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Parse PCR-GLOBWB error log files.")
    parser.add_argument("logfile", help="Path to the log file (.txt, .log, or .md with shell fence)")
    parser.add_argument("--csv", default=None, help="Output CSV path (default: <logfile>.csv)")
    parser.add_argument("--summary", default=None, help="Output summary path (default: <logfile>_summary.txt)")
    args = parser.parse_args()

    src = Path(args.logfile)
    if not src.exists():
        sys.exit(f"File not found: {src}")

    csv_path = Path(args.csv) if args.csv else src.with_suffix(".csv")
    summary_path = Path(args.summary) if args.summary else src.with_name(src.stem + "_summary.txt")

    raw_lines = src.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)

    # Strip markdown code fence if present
    if src.suffix == ".md" or any("```" in ln for ln in raw_lines[:20]):
        log_lines = strip_markdown_fence(raw_lines)
    else:
        log_lines = raw_lines

    records = parse_log(log_lines)
    print(f"[Parsed]  {len(records)} log entries from {src.name}")

    write_csv(records, csv_path)
    write_summary(records, summary_path)


if __name__ == "__main__":
    main()
