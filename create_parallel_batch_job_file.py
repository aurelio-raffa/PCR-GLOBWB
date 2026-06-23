#!/usr/bin/env python3
"""DEPRECATED -- use ``python create_job_file.py parallel ...`` instead.

Thin shim kept for backward compatibility: it forwards the original CLI (``--nc --mem --jq --wd --pc
--conda_env --config [--tile] [--excl] [--runner] [--debug_option] [--runner_extra_args ...]``) to the unified
generator's ``parallel`` mode, which holds the single implementation (and the same consistency checks) now.
"""
import sys
import warnings

from create_job_file import build_parser, build_parallel


def create_parallel_batch_job_file() -> None:
    warnings.warn("create_parallel_batch_job_file.py is deprecated; use `create_job_file.py parallel ...`",
                  DeprecationWarning, stacklevel=2)
    print("NOTE: create_parallel_batch_job_file.py is deprecated; use 'python create_job_file.py parallel ...'",
          file=sys.stderr)
    args = build_parser().parse_args(['parallel', *sys.argv[1:]])
    build_parallel(args)


if __name__ == "__main__":
    create_parallel_batch_job_file()
