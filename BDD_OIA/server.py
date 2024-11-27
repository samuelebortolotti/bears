import os
import sys

import submitit
from experiments import launch_bdd
from main_bdd import main, parse_args

conf_path = os.getcwd() + "."
sys.path.append(conf_path)

if __name__ == "__main__":
    # start_main()
    args = parse_args()  # prepare_args() #

    # the number of task class
    args.nclasses = 5
    args.theta_dim = args.nclasses

    executor = submitit.AutoExecutor(
        folder="./logs", slurm_max_num_timeout=30
    )
    executor.update_parameters(
        mem_gb=4,
        gpus_per_node=1,
        tasks_per_node=1,  # one task per GPU
        cpus_per_gpu=10,
        nodes=1,
        timeout_min=60,  # max is 60 * 72
        # Below are cluster dependent parameters
        slurm_partition="long",
        slurm_signal_delay_s=120,
        slurm_array_parallelism=4,
    )

    experiments = launch_bdd(args)
    executor.update_parameters(name="OID-BOIA")
    jobs = executor.map_array(main, experiments)
