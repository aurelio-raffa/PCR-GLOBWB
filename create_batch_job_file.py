#!/usr/bin/env python3
"""DEPRECATED -- use ``python create_job_file.py deterministic ...`` instead.

Thin shim kept for backward compatibility: it forwards the original CLI (``--nc --mem --jq --wd --pc
--conda_env --config [--tile] [--excl] [--name]``) to the unified generator's ``deterministic`` mode, which
holds the single implementation now.
"""
import sys
import warnings

from create_job_file import build_parser, build_deterministic


def create_batch_job_file() -> None:
    warnings.warn("create_batch_job_file.py is deprecated; use `create_job_file.py deterministic ...`",
                  DeprecationWarning, stacklevel=2)
    print("NOTE: create_batch_job_file.py is deprecated; use 'python create_job_file.py deterministic ...'",
          file=sys.stderr)
    args = build_parser().parse_args(['deterministic', *sys.argv[1:]])
    build_deterministic(args)


if __name__ == "__main__":
    create_batch_job_file()
