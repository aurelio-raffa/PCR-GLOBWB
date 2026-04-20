import argparse
from datetime import datetime


N_CORES = 'nc'
TILE = 'tile'
MEM_LIMIT = 'mem'
JOB_QUEUE = 'jq'
WORKING_DIR = 'wd'
PROJECT_CODE = 'pc'
CONDA_ENV = 'conda_env'
CONFIG_INI_PATH = 'config'
EXCLUSIVE = 'excl'
_RESOURCES = '__resources__'

help_msg: str = f"""
LSF Job File Generator.

This script generates an IBM Spectrum LSF job submission file (.lsf)
based on command-line arguments. The generated file can be submitted
to an LSF cluster using the `bsub` command.

The script fills a predefined job template with user-provided parameters
such as number of cores, memory limits, queue, and environment settings.

The naming pattern for the output file is \"job_{{%y%m%d%H%M}}.lsf\"
"""

# legend:
# -n: number of requested cores
# -R or -x: host resource requirements
# -M: per-job/host mem limit
# -q: queue
# -o: output logs
# -e: error logs
# -P: project code
job_template = f"""#!/bin/sh
#BSUB -n {{{N_CORES}}}
#BSUB {{{_RESOURCES}}}
#BSUB -M {{{MEM_LIMIT}}}
#BSUB -q {{{JOB_QUEUE}}}
#BSUB -o {{{WORKING_DIR}}}/logfile.%J.txt
#BSUB -e {{{WORKING_DIR}}}/errfile.%J.txt
#BSUB -P {{{PROJECT_CODE}}}    
module load anaconda
source activate {{{CONDA_ENV}}}
cd ./model
python3 deterministic_runner.py ../{{{CONFIG_INI_PATH}}}
"""

parser = argparse.ArgumentParser(description=help_msg)
parser.add_argument(f"--{N_CORES}", help='Number of cores')
parser.add_argument(f"--{MEM_LIMIT}", help='Memory limit (e.g.: "5G")')
parser.add_argument(f"--{JOB_QUEUE}", help='Queue to submit jobs to')
parser.add_argument(f"--{WORKING_DIR}", help='Working directory (for error and log files)')
parser.add_argument(f"--{PROJECT_CODE}", help='Project identifier')
parser.add_argument(f"--{CONDA_ENV}", help='Name of conda environment to run the code')
parser.add_argument(f"--{CONFIG_INI_PATH}", help='INI configuration file')
parser.add_argument(
    f"--{TILE}",
    nargs='?',
    help=f'No. cores per node (not used in exclusive mode); defaults to "{N_CORES}"',
    type=int,
    default=None
)
parser.add_argument(
    f"--{EXCLUSIVE}",
    help='Whether the job should take the entire node',
    action='store_true'
)


def create_batch_job_file():
    f"""{help_msg}
    
    Behavior:
        - If --{TILE} is not specified, it defaults to the value of --{N_CORES}.
        - If --{EXCLUSIVE} is set, the resource requirement becomes "-x".
        - Otherwise, the resource requirement is "-R span[ptile={TILE}]".
        - The script prints the name of the generated file.
        
    Example usage:
        python create_batch_job_file.py \\
            --{N_CORES}=72 \\
            --{MEM_LIMIT}=128G \\
            --{JOB_QUEUE}=YOUR_QUEUE \\
            --{WORKING_DIR}=PATH/TO/WORKING/DIR \\
            --{PROJECT_CODE}=PRJC0D3 \\
            --{CONDA_ENV}=pcrglobwb_python3 \\
            --{CONFIG_INI_PATH}=PATH/TO/CONFIG/INI \\
            —-{EXCLUSIVE}
    """
    args = parser.parse_args()

    # filling in optional arguments
    if not hasattr(args, TILE) or getattr(args, TILE) is None:
        setattr(args, TILE, getattr(args, N_CORES))
    if hasattr(args, EXCLUSIVE) and getattr(args, EXCLUSIVE):
        setattr(args, _RESOURCES, '-x')
    else:
        setattr(args, _RESOURCES, f'-R span[ptile={getattr(args, TILE)}]')

    filename = f'job_{datetime.now().strftime("%y%m%d%H%M")}.lsf'

    with open(filename, 'w') as handle:
        handle.write(job_template.format(**vars(args)))

    print(filename)


if __name__ == "__main__":
    create_batch_job_file()
