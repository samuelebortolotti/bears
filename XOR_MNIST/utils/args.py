# Args module

from argparse import ArgumentParser

from datasets import NAMES as DATASET_NAMES
from models import get_all_models


def add_experiment_args(parser: ArgumentParser) -> None:
    """Adds the arguments used by all the models.

    Args:
        parser: the parser instance

    Returns:
        None: This function does not return a value.
    """
    # dataset
    parser.add_argument(
        "--dataset",
        default="addmnist",
        type=str,
        choices=DATASET_NAMES,
        help="Which dataset to perform experiments on.",
    )
    parser.add_argument(
        "--task",
        default="addition",
        type=str,
        choices=[
            "addition",
            "product",
            "multiop",
            "base",
            "red_triangle",
            "triangle_circle",
            "patterns",
            "mini_patterns",
            "mini_patterns_bombazza",
        ],
        help="Which operation to choose.",
    )
    # model settings
    parser.add_argument(
        "--model",
        type=str,
        default="mnistdpl",
        help="Model name.",
        choices=get_all_models(),
    )
    parser.add_argument(
        "--c_sup",
        type=float,
        default=0,
        help="Fraction of concept supervision on concepts",
    )
    parser.add_argument(
        "--which_c",
        type=int,
        nargs="+",
        default=[-1],
        help="Which concepts explicitly supervise (-1 means all)",
    )
    parser.add_argument(
        "--joint",
        action="store_true",
        default=False,
        help="Process the image as a whole.",
    )
    parser.add_argument(
        "--splitted",
        action="store_true",
        default=False,
        help="Create different encoders.",
    )
    parser.add_argument(
        "--entropy",
        action="store_true",
        default=False,
        help="Activate entropy on batch.",
    )
    # weights of logic
    parser.add_argument(
        "--w_sl",
        type=float,
        default=10,
        help="Weight of Semantic Loss",
    )
    # weight of mitigation
    parser.add_argument(
        "--gamma", type=float, default=1, help="Weight of mitigation"
    )
    # additional hyperparams
    parser.add_argument(
        "--w_rec",
        type=float,
        default=1,
        help="Weight of Reconstruction",
    )
    parser.add_argument(
        "--beta", type=float, default=2, help="Multiplier of KL"
    )
    parser.add_argument(
        "--w_h", type=float, default=1, help="Weight of entropy"
    )
    parser.add_argument(
        "--w_c", type=float, default=1, help="Weight of concept sup"
    )

    # optimization params
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate."
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=2, help="Warmup epochs."
    )
    parser.add_argument(
        "--exp_decay",
        type=float,
        default=0.99,
        help="Exp decay of learning rate.",
    )

    # learning hyperams
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=50,
        help="Number of epochs per task.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size."
    )

    # deep ensembles
    parser.add_argument(
        "--n_ensembles",
        type=int,
        default=5,
        help="Number of model in DeepEnsembles",
    )


def add_management_args(parser: ArgumentParser) -> None:
    """Adds the arguments used in management

    Args:
        parser: the parser instance

    Returns:
        None: This function does not return a value.
    """
    # random seed
    parser.add_argument(
        "--seed", type=int, default=None, help="The random seed."
    )
    # verbosity
    parser.add_argument(
        "--notes", type=str, default=None, help="Notes for this run."
    )
    parser.add_argument("--non_verbose", action="store_true")
    # logging
    parser.add_argument(
        "--wandb",
        type=str,
        default=None,
        help="Enable wandb logging -- set name of project",
    )
    # checkpoints
    parser.add_argument(
        "--checkin",
        type=str,
        default=None,
        help="location and path FROM where to load ckpt.",
    )
    parser.add_argument(
        "--checkout",
        action="store_true",
        default=False,
        help="save the model to data/ckpts.",
    )
    # post-hoc evaluation
    parser.add_argument(
        "--posthoc",
        action="store_true",
        default=False,
        help="Used to evaluate only the loaded model",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=False,
        help="Used to evaluate on the validation set for hyperparameters search",
    )
    parser.add_argument(
        "--active-learning",
        action="store_true",
        default=False,
        help="For concept supervision based learning",
    )
    # preprocessing option
    parser.add_argument(
        "--preprocess",
        action="store_true",
        default=False,
        help="Used to preprocess dataset",
    )
    parser.add_argument(
        "--finetuning",
        type=int,
        default=0,
        help="Phase of active learning",
    )


def add_test_args(parser: ArgumentParser) -> None:
    """Arguments for the Test part of the code

    Args:
        parser: the parser instance

    Returns:
        None: This function does not return a value.
    """
    # random seed
    parser.add_argument(
        "--use_ood",
        action="store_true",
        help="Use Out of Distribution test samples.",
    )
    # verbosity
    parser.add_argument(
        "--type",
        type=str,
        default="frequentist",
        choices=[
            "frequentist",
            "mcdropout",
            "ensemble",
            "laplace",
            "bears",
        ],
        help="Evaluation type.",
    )
    parser.add_argument(
        "--deep-ens-kl",
        action="store_true",
        default=False,
        help="Employ KL to separate concept distributions in Ensemble.",
    )
    parser.add_argument(
        "--knowledge-aware-kl",
        action="store_true",
        default=False,
        help="Employ a knowledge aware KL.",
    )
    parser.add_argument(
        "--real-kl",
        action="store_true",
        default=False,
        help="Real paper KL.",
    )
    parser.add_argument(
        "--evaluate-all",
        action="store_true",
        default=False,
        help="Evaluate all the techniques",
    )
    # weight of deep separation kl
    parser.add_argument(
        "--lambda_h",
        type=float,
        default=1,
        help="Lambda for the KL divergence",
    )
    parser.add_argument(
        "--skip_laplace",
        action="store_true",
        default=False,
        help="Skip Laplace as Bayesian method",
    )
