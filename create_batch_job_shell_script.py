import argparse
from datetime import datetime


N_CORES = 'nc'
REQ_MEM = 'rmem'
MEM_LIMIT = 'mem'
JOB_QUEUE = 'jq'
WORKING_DIR = 'wd'
PROJECT_CODE = 'pc'
CONFIG_YAML_PATH = 'config'

job_template = f"""#!/bin/sh
#BSUB -n {{{N_CORES}}}                              #number of requested cores
#BSUB -R rusage[mem={{{REQ_MEM}}}]"                 #memory host resource requirements
#BSUB -M {{{MEM_LIMIT}}}                            #per-job/host mem limit
#BSUB -q {{{JOB_QUEUE}}}                            #queue
#BSUB -o /{{{WORKING_DIR}}}/logfile.%J.txt          #output logs
#BSUB -e /{{{WORKING_DIR}}}/errfile.%J.txt          #error logs
#BSUB -P {{{PROJECT_CODE}}}                         #project code
module load anaconda
conda activate pcrglobwb_python3
python3 deterministic_runner.py {{{CONFIG_YAML_PATH}}}
"""


parser = argparse.ArgumentParser()
parser.add_argument(f"--{N_CORES}")
parser.add_argument(f"--{REQ_MEM}")
parser.add_argument(f"--{MEM_LIMIT}")
parser.add_argument(f"--{JOB_QUEUE}")
parser.add_argument(f"--{WORKING_DIR}")
parser.add_argument(f"--{PROJECT_CODE}")
parser.add_argument(f"--{CONFIG_YAML_PATH}")


if __name__ == "__main__":
    args = parser.parse_args()

    with open(f'job_PCR_GLOBWB{datetime.now().strftime("%y%m%d%H%M")}.sh', 'w') as handle:
        handle.write(job_template.format(**vars(args)))
