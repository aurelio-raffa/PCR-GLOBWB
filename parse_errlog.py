"""Root CLI shim for parsing a PCR-GLOBWB error / run log.

The implementation moved to ``src/utils/error_log.py`` so the pipeline stage
(``src/stages/diagnostics.py``, run automatically on each run's convenience logfile) and this
command-line tool share a single implementation. This shim only re-exposes the original
``python parse_errlog.py <logfile> [--csv ...] [--summary ...]`` interface for manual use on an LSF
``errfile`` or a Markdown-wrapped log.

Usage:
    python parse_errlog.py <logfile> [--csv <output.csv>] [--summary <output.txt>]

Outputs:
  - CSV  : one row per log entry with columns timestamp, level, issuer, content
  - TXT  : a termination/completion diagnosis followed by a deduplicated summary ordered by severity
"""
from src.utils.error_log import parser, run_cli

if __name__ == "__main__":
    run_cli(**vars(parser.parse_args()))
