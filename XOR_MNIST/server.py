# Server module
# Allows for the execution of multiple parameters

import os
import sys

import submitit
from experiments import *
from main import main, parse_args

conf_path = os.getcwd() + "."
sys.path.append(conf_path)

if __name__ == "__main__":
    # start_main()
    args = parse_args()  #
    #     args = prepare_args() #  parse_args() #
    executor = submitit.AutoExecutor(
        folder="./logs", slurm_max_num_timeout=150
    )
    executor.update_parameters(
        mem_gb=4,
        gpus_per_node=1,
        tasks_per_node=1,  # one task per GPU
        cpus_per_task=10,
        nodes=1,
        timeout_min=120,
        # Below are cluster dependent parameters
        slurm_partition="long-disi",
        slurm_signal_delay_s=120,
        slurm_array_parallelism=4,
    )

    experiments = launch_KAND(args)
    executor.update_parameters(name="OID-KAND")
    jobs = executor.map_array(main, experiments)
