import os

import numpy as np
import wandb
from datasets.utils.base_dataset import BaseDataset
from models.mnistdpl import MnistDPL
from utils import fprint
from utils.bayes import (
    activate_dropout,
    deep_ensemble,
    ensemble_predict,
    laplace_approximation,
    laplace_prediction,
    laplace_single_prediction,
    montecarlo_dropout,
)
from utils.checkpoint import get_model_name, load_checkpoint
from utils.metrics import (
    concept_accuracy,
    evaluate_metrics,
    evaluate_mix,
    get_concept_probability,
    get_concept_probability_ensemble,
    get_concept_probability_factorized_ensemble,
    get_concept_probability_factorized_laplace,
    get_concept_probability_factorized_mcdropout,
    get_concept_probability_laplace,
    get_concept_probability_mcdropout,
    world_accuracy,
)
from utils.test_utils import *
from utils.wandb_logger import *


class IllegalArgumentError(ValueError):
    """Simple ValueError"""

    pass


"""TOTAL EVALUATION METHODS
"""
TOTAL_METHODS = [member.value for member in EVALUATION_TYPE]


def test(model: MnistDPL, dataset: BaseDataset, args, **kwargs):
    """Test function

    Args:
        model (nn.Module): network
        dataset (BaseDataset): dataset
        args: command line args
        kwargs: key-value args

    Returns:
        None: This function does not return a value.
    """
    # If I have to evaluate all
    if args.evaluate_all:
        if len(TOTAL_METHODS) == 0:
            print("Done total evaluation!...")

            if not os.path.exists("dumps"):  # If not, create it
                os.makedirs("dumps")

            # save the dumps
            save_dump(args, kwargs)

            # end wandb
            if args.wandb:
                wandb.finish()

        args.type = TOTAL_METHODS[0]
        TOTAL_METHODS.pop(0)

        print(
            "Doing total evaluation on...",
            args.type,
            "remaining: ",
            TOTAL_METHODS,
        )

    # Wandb
    if args.wandb is not None:
        fprint("\n---Wandb on\n")
        wandb.init(
            project=args.project,
            entity=args.wandb,
            name=str(args.model)
            + "_lasthope_"
            + "_n_ens_"
            + str(args.n_ensembles)
            + "_lambda_"
            + str(args.lambda_h),
            config=args,
        )

    # Default Setting for Training
    model.to(model.device)
    train_loader, val_loader, test_loader = dataset.get_data_loaders()

    # override the OOD if specified
    if args.use_ood:
        test_loader = dataset.ood_loader

    fprint("Loading network....")
    model = load_checkpoint(model, args, args.checkin)
    laplace_model = None
    ensemble = None

    # Add the original weights
    if "ORIGINAL_WEIGHTS" not in kwargs:
        kwargs["ORIGINAL_WEIGHTS"] = [
            param.data.clone() for param in model.parameters()
        ]

    # Check whether to apply softmax
    apply_softmax = False
    if args.model in ["mnistsl", "mnistpcbmsl"]:
        apply_softmax = True

    if (
        args.type == EVALUATION_TYPE.LAPLACE.value
        and args.skip_laplace
    ):
        if args.evaluate_all:
            test(model, dataset, args, **kwargs)
        else:
            return

    # Retrieve the metrics according to the type of evaluation specified
    if args.type == EVALUATION_TYPE.NORMAL.value:
        fprint("## Not Bayesian model ##")
        y_true, c_true, y_pred, c_pred, p_cs, p_ys, p_cs_all, _ = (
            evaluate_metrics(
                model,
                test_loader,
                args,
                last=True,
                apply_softmax=apply_softmax,
            )
        )
        _, c_true_cc, _, c_pred_cc, _, _, _, _ = evaluate_metrics(
            model,
            test_loader,
            args,
            last=True,
            concatenated_concepts=False,
            apply_softmax=apply_softmax,
        )
    elif args.type == EVALUATION_TYPE.MC_DROPOUT.value:
        fprint("## Montecarlo dropout ##")
        (
            y_true,
            c_true,
            y_pred,
            c_pred,
            c_true_cc,
            c_pred_cc,
            p_cs,
            p_ys,
            p_cs_all,
            _,
        ) = montecarlo_dropout(
            model, test_loader, model.n_facts, 30, apply_softmax
        )
    elif (
        args.type == EVALUATION_TYPE.BEARS.value
        or args.type == EVALUATION_TYPE.ENSEMBLE.value
    ):
        if args.type == EVALUATION_TYPE.BEARS.value:
            fprint("### BEARS ###")
            args.deep_ens_kl = True
        else:
            fprint("### Deep Ensemble ###")
            args.deep_ens_kl = False

        fprint("Preparing the ensembles...")

        ensemble = deep_ensemble(
            seeds=[
                i + args.seed + 1 for i in range(args.n_ensembles)
            ],
            dataset=dataset,
            num_epochs=args.n_epochs,
            args=args,
            val_loader=val_loader,
            epsilon=0.01,
            separate_from_others=args.deep_ens_kl,
            lambda_h=args.lambda_h,
            use_wandb=args.wandb,
            n_facts=model.n_facts,
            knowledge_aware_kl=args.knowledge_aware_kl,
            real_kl=args.real_kl,
        )

        # ensemble predict
        (
            y_true,
            c_true,
            y_pred,
            c_pred,
            c_true_cc,
            c_pred_cc,
            p_cs,
            p_ys,
            p_cs_all,
            _,
        ) = ensemble_predict(
            ensemble, test_loader, model.n_facts, apply_softmax
        )
    elif args.type == EVALUATION_TYPE.LAPLACE.value:
        fprint("### Laplace Approximation ###")
        fprint("Preparing laplace model, please wait...")
        laplace_model = laplace_approximation(
            model, model.device, train_loader, val_loader
        )
        (
            y_true,
            c_true,
            y_pred,
            c_pred,
            c_true_cc,
            c_pred_cc,
            p_cs,
            p_ys,
            p_cs_all,
            _,
        ) = laplace_prediction(
            laplace_model,
            model.device,
            test_loader,
            30,
            model.nr_classes,
            model.n_facts,
            apply_softmax,
        )
    else:
        raise IllegalArgumentError("Mode argument not valid")

    (
        _,
        concept_labels_single,
        _,
        _,
    ) = generate_concept_labels(dataset.get_concept_labels())

    fprint("Evaluating", args.type)

    # Print the distances
    if args.type == EVALUATION_TYPE.LAPLACE.value:
        # Get the ensembles for the inner model
        ensemble = laplace_model.model.model.get_ensembles(
            laplace_model, args.n_ensembles
        )

    # metrix, h(c|y) and concept confusion matrix
    mean_h_c, yac, cac, cf1, yf1 = print_metrics(
        y_true,
        y_pred,
        c_true,
        c_pred,
        p_cs_all,
        model.n_facts,
        args.type,
    )

    # Log in Wandb
    if args.wandb is not None:
        ood_string = "-ood" if args.use_ood else ""
        to_log = {
            f"{args.type}-Mean-H(C)-test{ood_string}": mean_h_c,
            f"{args.type}-Acc-Y-test{ood_string}": yac,
            f"{args.type}-Acc-C-test{ood_string}": cac,
            f"{args.type}-F1-C-test{ood_string}": cf1,
            f"{args.type}-F1-Y-test{ood_string}": yf1,
        }
        wandb.log(to_log)

    # Add stuff to the dictionary
    if "mean_hc" not in kwargs:
        kwargs["mean_hc"] = []
        kwargs["yac"] = []
        kwargs["cac"] = []
        kwargs["yac_hard"] = []
        kwargs["cac_hard"] = []
        kwargs["cf1"] = []
        kwargs["yf1"] = []

    kwargs["mean_hc"].append(mean_h_c)
    kwargs["yf1"].append(yf1)
    kwargs["yac"].append(yac)

    if args.type != EVALUATION_TYPE.NORMAL.value:
        kwargs["yac_hard"].append(cf1)
        kwargs["cac_hard"].append(cac)

    # label and concept ece
    ece_y = produce_ece_curve(
        p_ys, y_pred, y_true, args.type, "labels"
    )

    if args.type == EVALUATION_TYPE.NORMAL.value:
        (
            worlds_prob,
            c_factorized_1,
            c_factorized_2,
            worlds_groundtruth,
        ) = get_concept_probability(
            model, test_loader
        )  # 2 arrays of size 256, 10 (concept 1 and concept 2 for all items)
        c_pred_normal = worlds_prob.argmax(axis=1)
        p_c_normal = worlds_prob.max(axis=1)

        ece = produce_ece_curve(
            p_c_normal,
            c_pred_normal,
            worlds_groundtruth,
            args.type,
            "concepts",
        )

        mean_h_c, yac, cac, cf1, yf1 = print_metrics(
            y_true,
            y_pred,
            worlds_groundtruth,
            c_pred_normal,
            worlds_prob,
            model.n_facts,
            args.type,
        )

        kwargs["yac_hard"].append(cf1)
        kwargs["cac_hard"].append(cac)

    else:
        ece = produce_ece_curve(
            p_cs, c_pred, c_true, args.type, "concepts"
        )

    if "ece" not in kwargs:
        kwargs["ece"] = []
        kwargs["ece y"] = []

    kwargs["ece"].append(ece)
    kwargs["ece y"].append(ece_y)

    # Log in Wandb
    if args.wandb is not None:
        ood_string = "-ood" if args.use_ood else ""
        to_log = {
            f"{args.type}-ECE-C-test{ood_string}": ece,
            f"{args.type}-ECE-Y-test{ood_string}": ece_y,
        }
        wandb.log(to_log)

    # Evaluate all the models in the ensemble
    if (
        args.type == EVALUATION_TYPE.BEARS.value
        or args.type == EVALUATION_TYPE.LAPLACE.value
        or args.type == EVALUATION_TYPE.ENSEMBLE.value
    ):
        fprint(f"{args.type} evaluation...")

        if args.type == EVALUATION_TYPE.LAPLACE.value:
            # Get the ensembles for the inner model
            ensemble = laplace_model.model.model.get_ensembles(
                laplace_model, 30
            )

        for i, model in enumerate(ensemble):
            fprint(f"-- Model {i} --")
            (
                y_true_ens,
                c_true_ens,
                y_pred_ens,
                c_pred_ens,
                p_cs_ens,
                p_ys_ens,
                p_cs_all_ens,
                _,
            ) = evaluate_metrics(
                model,
                test_loader,
                args,
                last=True,
                apply_softmax=apply_softmax,
            )
            _, c_true_cc_ens, _, c_pred_cc_ens, _, _, _, _ = (
                evaluate_metrics(
                    model,
                    test_loader,
                    args,
                    last=True,
                    concatenated_concepts=False,
                    apply_softmax=apply_softmax,
                )
            )

            mean_sh_c, syac, scac, scf1, syf1 = print_metrics(
                y_true_ens,
                y_pred_ens,
                c_true_ens,
                c_pred_ens,
                p_cs_all_ens,
                model.n_facts,
                args.type,
            )

            sece_y = produce_ece_curve(
                p_ys_ens,
                y_pred_ens,
                y_true_ens,
                args.type,
                "labels",
                ECEMODE.WHOLE,
                None,
                f"_{args.type}_{i}",
            )
            sece = produce_ece_curve(
                p_cs_ens,
                c_pred_ens,
                c_true_ens,
                args.type,
                "concepts",
                ECEMODE.WHOLE,
                None,
                f"_{args.type}_{i}",
            )

            if args.wandb is not None:
                ood_string = "-ood" if args.use_ood else ""
                to_log = {
                    f"{args.type}_model_{i}-Mean-H(C)-test{ood_string}": mean_sh_c,
                    f"{args.type}_model_{i}-Acc-Y-test{ood_string}": syac,
                    f"{args.type}_model_{i}-Acc-C-test{ood_string}": scac,
                    f"{args.type}_model_{i}-F1-C-test{ood_string}": scf1,
                    f"{args.type}_model_{i}-F1-Y-test{ood_string}": syf1,
                    f"{args.type}_model_{i}-ECE-C-test{ood_string}": sece,
                    f"{args.type}_model_{i}-ECE-Y-test{ood_string}": sece_y,
                }
                wandb.log(to_log)

    fprint("--- Computing the probability of each world... ---")

    c_factorized_1, c_factorized_2 = None, None

    # TODO: should be pc
    if args.type == EVALUATION_TYPE.MC_DROPOUT.value:
        worlds_prob = get_concept_probability_mcdropout(
            model, test_loader, activate_dropout, args.n_ensembles
        )  # 2 arrays of size 256, 10 (concept 1 and concept 2 for all items)

        # Obtain the factorized probabilities
        c_factorized_1, c_factorized_2, gt_factorized = (
            get_concept_probability_factorized_mcdropout(
                model, test_loader, activate_dropout, args.n_ensembles
            )
        )

    elif (
        args.type == EVALUATION_TYPE.BEARS.value
        or args.type == EVALUATION_TYPE.ENSEMBLE.value
    ):
        worlds_prob = get_concept_probability_ensemble(
            ensemble, test_loader
        )  # 2 arrays of size 256, 10 (concept 1 and concept 2 for all items)

        # Obtain the factorized probabilities
        c_factorized_1, c_factorized_2, gt_factorized = (
            get_concept_probability_factorized_ensemble(
                ensemble, test_loader
            )
        )
    elif args.type == EVALUATION_TYPE.LAPLACE.value:
        worlds_prob = get_concept_probability_laplace(
            model.device, test_loader, laplace_model, args.n_ensembles
        )  # 2 arrays of size 256, 10 (concept 1 and concept 2 for all items)

        # Obtain the factorized probabilities
        c_factorized_1, c_factorized_2, gt_factorized = (
            get_concept_probability_factorized_laplace(
                model.device,
                test_loader,
                laplace_single_prediction,
                laplace_model,
                model.nr_classes,
                model.n_facts,
            )
        )
    else:
        # NORMAL MODE
        (
            worlds_prob,
            c_factorized_1,
            c_factorized_2,
            worlds_groundtruth,
        ) = get_concept_probability(
            model, test_loader
        )  # 2 arrays of size 256, 10 (concept 1 and concept 2 for all items)

    # Change it for the concept factorized entropy and variance
    if args.type == EVALUATION_TYPE.NORMAL.value:
        p_cs_all = worlds_prob
        gt_factorized = c_true

    # factorized probability concatenated
    c_factorized_full = np.concatenate(
        (c_factorized_1, c_factorized_2), axis=0
    )
    # maximum element probability for the ECE count
    c_factorized_max_p = np.max(c_factorized_full, axis=1)
    # factorized predictions with argmax
    c_pred_factorized_full = np.argmax(c_factorized_full, axis=1)

    single_concepts_ece = []
    # ECE per concept NOTE factorized, otherwise only world is possible
    for c in concept_labels_single:
        ece_single_concept = produce_ece_curve(
            c_factorized_max_p,
            c_pred_factorized_full,
            gt_factorized,
            args.type,
            f"concepts {c}",
            ECEMODE.FILTERED_BY_CONCEPT,
            int(c),
        )
        single_concepts_ece.append(ece_single_concept)

    # add single concepts ECE
    kwargs[f"{args.type} ece single concept"] = single_concepts_ece

    cfe = compute_concept_factorized_entropy(
        c_factorized_1,
        c_factorized_2,
        p_cs_all,  # equal to c_pred in ensembles [#dati, facts^2]
    )

    cfvar = compute_concept_factorized_variance(
        c_factorized_1,
        c_factorized_2,
        p_cs_all,  # equal to c_pred in ensembles [#dati, facts^2]
    )

    cac, cf1 = evaluate_mix(c_pred_factorized_full, gt_factorized)
    kwargs["cf1"].append(cf1)
    kwargs["cac"].append(cac)

    if not any(
        key in kwargs for key in ["e_c1", "e_c2", "e_c", "e_(c1, c2)"]
    ):
        kwargs["e_c1"] = list()
        kwargs["e_c2"] = list()
        kwargs["e_c"] = list()
        kwargs["e_(c1, c2)"] = list()

        kwargs["var_c1"] = list()
        kwargs["var_c2"] = list()
        kwargs["var_c"] = list()
        kwargs["var_(c1, c2)"] = list()

    kwargs["e_c1"].append(cfe["c1"])
    kwargs["e_c2"].append(cfe["c2"])
    kwargs["e_c"].append(cfe["c"])
    kwargs["e_(c1, c2)"].append(cfe["(c1, c2)"])

    kwargs["var_c1"].append(cfvar["c1"])
    kwargs["var_c2"].append(cfvar["c2"])
    kwargs["var_c"].append(cfvar["c"])
    kwargs["var_(c1, c2)"].append(cfvar["(c1, c2)"])

    concept_counter_list, concept_acc_list = concept_accuracy(
        c_factorized_1, c_factorized_2, gt_factorized
    )

    if args.type == EVALUATION_TYPE.NORMAL.value:
        p_cs_all = worlds_prob
        c_true = worlds_groundtruth

    world_counter_list, world_acc_list = world_accuracy(
        p_cs_all, c_true, model.n_facts
    )

    if not any(
        key in kwargs
        for key in ["c_acc_count", "c_acc", "w_acc_count", "w_acc"]
    ):
        kwargs["c_acc_count"] = list()
        kwargs["c_acc"] = list()
        kwargs["w_acc_count"] = list()
        kwargs["w_acc"] = list()
        kwargs["c_ova_filtered"] = list()
        kwargs["c_all_filtered"] = list()

    kwargs["c_acc_count"].append(concept_counter_list)
    kwargs["c_acc"].append(concept_acc_list)
    kwargs["w_acc_count"].append(world_counter_list)
    kwargs["w_acc"].append(world_acc_list)

    e_per_c = compute_entropy_per_concept(
        c_factorized_full, gt_factorized
    )

    kwargs["c_ova_filtered"].append(e_per_c["c_ova_filtered"])
    kwargs["c_all_filtered"].append(e_per_c["c_all_filtered"])

    file_path = f"dumps/{get_model_name(args)}-seed_{args.seed}-{args.type}-nens_{args.n_ensembles}-ood_{args.use_ood}-lambda_{args.lambda_h}.csv"

    # save csv
    save_csv(
        y_true,
        c_true,
        y_pred,
        c_pred,
        c_true_cc,
        c_pred_cc,
        p_cs,
        p_ys,
        p_cs_all,
        c_factorized_1,
        c_factorized_2,
        worlds_prob,
        gt_factorized,
        file_path,
    )

    if args.evaluate_all:
        test(model, dataset, args, **kwargs)
    else:
        save_dump(args, kwargs, incomplete=True, eltype=args.type)
