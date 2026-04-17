import argparse
from datetime import datetime


N_CORES = 'nc'
TILE = 'tile'
MEM_LIMIT = 'mem'
JOB_QUEUE = 'jq'
WORKING_DIR = 'wd'
PROJECT_CODE = 'pc'
CONDA_ENV = 'conda_env'
CONFIG_YAML_PATH = 'config'
EXCLUSIVE = 'excl'
_RESOURCES = '__resources__'

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
python3 model/deterministic_runner.py {{{CONFIG_YAML_PATH}}}
"""


parser = argparse.ArgumentParser()
parser.add_argument(f"--{N_CORES}", help='Number of cores')
parser.add_argument(f"--{MEM_LIMIT}", help='Memory limit (e.g.: "5G")')
parser.add_argument(f"--{JOB_QUEUE}", help='Queue to submit jobs to')
parser.add_argument(f"--{WORKING_DIR}", help='Working directory (for error and log files)')
parser.add_argument(f"--{PROJECT_CODE}", help='Project identifier')
parser.add_argument(f"--{CONDA_ENV}", help='Name of conda environment to run the code')
parser.add_argument(f"--{CONFIG_YAML_PATH}", help='Pipeline YAML configuration file')
parser.add_argument(
    f"--{TILE}",
    nargs='?',
    help=f'No. cores per node (not used in excl. mode); defaults to "{N_CORES}"',
    type=int,
    default=None
)
parser.add_argument(
    f"--{EXCLUSIVE}",
    help='Whether the job should take the entire node',
    action='store_true'
)


if __name__ == "__main__":
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
