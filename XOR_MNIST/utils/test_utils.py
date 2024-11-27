# Debugging stuff for test.py
import json
import math
import time
from enum import Enum
from itertools import product
from typing import List

import numpy as np
import torch
from numpy import ndarray
from utils import fprint
from utils.checkpoint import get_model_name
from utils.metrics import (
    class_mean_entropy,
    class_mean_variance,
    ensemble_p_c_x_distance,
    evaluate_mix,
    expected_calibration_error,
    expected_calibration_error_by_concept,
    get_alpha,
    get_alpha_single,
    laplace_p_c_x_distance,
    mcdropout_p_c_x_distance,
    mean_entropy,
)
from utils.visualization import (
    produce_alpha_matrix,
    produce_bar_plot,
    produce_calibration_curve,
    produce_confusion_matrix,
)
from utils.wandb_logger import *


class ECEMODE(Enum):
    """Modality in which the ECE is computed"""

    WHOLE = 1
    FILTERED_BY_CONCEPT = 2


class EVALUATION_TYPE(Enum):
    """Evaluation type"""

    NORMAL = "frequentist"
    LAPLACE = "laplace"
    MC_DROPOUT = "mcdropout"
    BEARS = "bears"
    ENSEMBLE = "ensemble"


class REDUCED_EVALUATION_TYPE(Enum):
    """Evaluation type without Laplace"""

    NORMAL = "frequentist"
    MC_DROPOUT = "mcdropout"
    BEARS = "bears"
    ENSEMBLE = "ensemble"


def euclidean_distance(w1, w2):
    """Simple euclidean distance

    Args:
        w1: first weight
        w2: second weight

    Returns:
        distance: euclidean distance
    """
    return torch.sqrt(
        sum(torch.sum((p1 - p2) ** 2) for p1, p2 in zip(w1, w2))
    )


def fprint_weights_distance(
    original_weights, ensemble, method_1, method_2
):
    """Function which prints euclidean distance between model and elements within the ensemble

    Args:
        original_weights: Original weights
        ensemble: model ensemble
        method_1 (str): name of the first method
        method_2 (str): name of the second method

    Returns:
        None: This function does not return a value.
    """
    distance = 0
    for model in ensemble:
        model_weights = [
            param.data.clone() for param in model.parameters()
        ]
        distance += euclidean_distance(
            original_weights, model_weights
        )
    distance = distance / len(ensemble)
    fprint(
        f"Euclidean Distance between {method_1} and {method_2}: ",
        distance.item(),
    )


def fprint_ensemble_distance(ensemble):
    """Function which prints euclidean distance between elements within the ensemble

    Args:
        ensemble: model ensemble

    Returns:
        None: This function does not return a value.
    """
    distance = 0
    for i in range(len(ensemble) - 1):
        original_weights = [
            param.data.clone() for param in ensemble[i].parameters()
        ]
        for j in range(i + 1, len(ensemble)):
            model_weights = [
                param.data.clone()
                for param in ensemble[j].parameters()
            ]
            distance = euclidean_distance(
                original_weights, model_weights
            )
            fprint(
                f"Euclidean Distance between #{i} and #{j}: ",
                distance.item(),
            )


def print_p_c_given_x_distance(
    model,
    laplace_model,
    ensemble,
    test_loader,
    recover_predictions_from_laplace,
    activate_dropout,
    type: str,
    num_ensembles: int,
) -> None:
    """Function which prints the euclidean distance of p(c|x)

    Args:
        model (nn.Module): network
        laplace_model (Laplace): laplace model
        ensemble (List[nn.Module]): ensemble of network
        test_loader: test loader
        recover_predictions_from_laplace: function which recovers the predictions from the laplace library
        activate_dropout: function which activates the dropout of the network
        type (str): which evaluation to perform
        num_ensembles (int): number of ensembles

    Returns:
        None: This function does not return a value.
    """

    dist = None
    if type == EVALUATION_TYPE.LAPLACE.value:
        dist = laplace_p_c_x_distance(
            laplace_model,
            test_loader,
            num_ensembles,
            recover_predictions_from_laplace,
            model.nr_classes,
            model.n_facts,
        )
    elif type == EVALUATION_TYPE.MC_DROPOUT.value:
        dist = mcdropout_p_c_x_distance(
            model, test_loader, activate_dropout, num_ensembles
        )
    elif (
        type == EVALUATION_TYPE.BEARS.value
        or type == EVALUATION_TYPE.ENSEMBLE.value
    ):
        dist = ensemble_p_c_x_distance(ensemble, test_loader)
    fprint(f"Mean P(C|X) for {type} distance L2 is {dist}")


def print_metrics(
    y_true: ndarray,
    y_pred: ndarray,
    c_true: ndarray,
    c_pred: ndarray,
    p_cs_all: ndarray,
    n_facts: int,
    mode,
):
    """Function which prints the values of the metrics

    Args:
        y_true (ndarray): groundtruth labels
        y_pred (ndarray): predicted labels
        c_true (ndarray): groundtruth concepts
        c_pred (ndarray): predicted concepts
        p_cs_all (ndarray): all probabilities
        n_facts (int): number of concepts
        mode,

    Returns:
        h_c: entropy computed on concepts
        yac: accuracy on y
        cac: accuracy on c
        cf1: concept f1 score
        yf1: label f1 score
    """
    yac, yf1 = evaluate_mix(y_true, y_pred)
    cac, cf1 = evaluate_mix(c_true, c_pred)

    if mode != EVALUATION_TYPE.NORMAL.value:
        n_facts = n_facts**2

    h_c = mean_entropy(p_cs_all, n_facts)

    fprint(f"Performances:")
    fprint(f"Concepts:\n    ACC: {cac}, F1: {cf1}")
    fprint(f"Labels:\n      ACC: {yac}, F1: {yf1}")
    fprint(f"Entropy:\n     H(C): {h_c}")

    return h_c, yac, cac, cf1, yf1


def produce_h_c_given_y(
    p_cs_all: ndarray,
    y_true: ndarray,
    nr_classes: int,
    mode: str,
    suffix: str,
) -> None:
    """Function which produces a bar plot of H(C|Y)

    Args:
        p_cs_all (ndarray): all probabilities
        y_true (ndarray): groundtruth labels
        nr_classes (int): number of classes
        mode (str): testing modality
        suffix (str): suffix

    Returns:
        None: This function does not return a value.
    """
    h_c_given_y = class_mean_entropy(
        p_cs_all, np.concatenate((y_true, y_true)), nr_classes
    )
    produce_bar_plot(
        h_c_given_y,
        "Groundtruth class",
        "Entropy",
        "H(C|Y)",
        f"h_c_given_y_{mode}{suffix}",
        True,
    )


def produce_var_c_given_y(
    p_cs_all: ndarray,
    y_true: ndarray,
    nr_classes: int,
    mode: str,
    suffix: str,
) -> None:
    """Function which produces a bar plot of Var(C|Y)

    Args:
        p_cs_all (ndarray): all probabilities
        y_true (ndarray): groundtruth labels
        nr_classes (int): number of classes
        mode (str): testing modality
        suffix (str): suffix

    Returns:
        None: This function does not return a value.
    """
    var_c_given_y = class_mean_variance(
        p_cs_all, np.concatenate((y_true, y_true)), nr_classes
    )
    produce_bar_plot(
        var_c_given_y,
        "Groundtruth class",
        "Variance",
        "Var(C|Y)",
        f"var_c_given_y_{mode}{suffix}",
        True,
    )


def compute_concept_factorized_entropy(
    c_fact_1: ndarray,
    c_fact_2: ndarray,
    p_w_x: ndarray,
):
    """Function which computes the factorized entropy given the factorized probabilities as input

    Args:
        c_fact_1 (ndarray): factorized probability for concept 1
        c_fact_2 (ndarray): factorized probability for concept 2
        p_w_x (ndarray): probability of the world given x

    Returns:
        conditional_entropies: dictionary of lists containing entropies
    """

    def ova_entropy(p: ndarray, c: int):
        """Compute the OVA entropy per concept

        Args:
            p (ndarray): probability vector
            c (int): concept

        Returns:
            h: ova entropy for concept c
        """

        p += 1e-5
        p /= 1 + (p.shape[0] * 1e-5)

        positive = p[c] * math.log2(p[c])

        # mask to exclude index of the world
        mask = np.arange(len(p)) != c

        p_against_c = np.sum(p[mask])

        negative = p_against_c * math.log2(p_against_c)

        return -(positive + negative)

    conditional_entropies = {
        "c1": list(),
        "c2": list(),
        "(c1, c2)": list(),
        "c": list(),
    }

    c_fact_stacked = np.vstack([c_fact_1, c_fact_2])

    for c_fact, key in zip(
        [c_fact_1, c_fact_2, c_fact_stacked, p_w_x],
        ["c1", "c2", "c", "(c1, c2)"],
    ):
        for c in range(c_fact.shape[1]):
            result = np.apply_along_axis(
                ova_entropy, axis=1, arr=c_fact, c=c
            )
            conditional_entropies[key].append(np.mean(result))

    return conditional_entropies


def compute_entropy_per_concept(
    c_fact_stacked: ndarray,
    c_true: ndarray,
):
    """Function which computes the entropy per each concept

    Args:
        c_fact_stacked (ndarray): conept factorized probabilities
        c_true (ndarray): groundtruth concepts

    Returns:
        conditional_entropies: dictionary of lists containing entropies
    """

    def ova_entropy(p: ndarray, c: int):
        """Compute the OVA entropy per concept

        Args:
            p (ndarray): probability vector
            c (int): concept

        Returns:
            h: ova entropy for concept c
        """
        p += 1e-5
        p /= 1 + (p.shape[0] * 1e-5)

        positive = p[c] * math.log2(p[c])

        # mask to exclude index of the world
        mask = np.arange(len(p)) != c

        p_against_c = np.sum(p[mask])

        negative = p_against_c * math.log2(p_against_c)

        return -(positive + negative)

    def entropy(p: ndarray):
        """Shannon Entropy

        Args:
            p (ndarray): probability vector

        Returns:
            h: shannon entropy
        """
        entropy = -np.sum(p * np.log2(p))

        # Normalize entropy
        vector_size = len(p)
        normalized_entropy = entropy / np.log2(vector_size)

        return normalized_entropy

    conditional_entropies = {
        "c_ova_filtered": list(),
        "c_all_filtered": list(),
    }

    for c in range(c_fact_stacked.shape[1]):
        indices = np.where(c_true == c)[0]
        c_fact_filtered = c_fact_stacked[indices]

        result = np.apply_along_axis(
            ova_entropy, axis=1, arr=c_fact_filtered, c=c
        )
        conditional_entropies["c_ova_filtered"].append(
            np.mean(result)
        )

        result = np.apply_along_axis(
            entropy, axis=1, arr=c_fact_filtered
        )
        conditional_entropies["c_all_filtered"].append(
            np.mean(result)
        )

    return conditional_entropies


def measure_execution_time(func, *args, **kwargs):
    """Function which measures the execution time of another function

    Args:
        func: function to evaluate
        args: arguments for func
        kwargs: key-value arguments for func

    Returns:
        t: execution time
    """
    start_time = time.time()
    func(*args, **kwargs)
    end_time = time.time()

    execution_time = end_time - start_time
    return execution_time


# NOTE all the concepts are threated like Bernoulli variable
def compute_concept_factorized_variance(
    c_fact_1: ndarray,
    c_fact_2: ndarray,
    p_w_x: ndarray,
):
    """Function which computes the concept factorized variance

    Args:
        c_fact_1 (ndarray): factorized probability for concept 1
        c_fact_2 (ndarray): factorized probability for concept 2
        p_w_x: p(w|x)

    Returns:
        conditional_variances:  dictionary of lists containing variances
    """

    def bernoulli_std(p: ndarray, c: int):
        """Bernoulli variance

        Args:
            p (ndarray): probability vector
            c (int): concept

        Returns:
            v: bernoulli variance
        """

        return math.sqrt(p[c] * (1 - p[c]))

    conditional_variances = {
        "c1": list(),
        "c2": list(),
        "(c1, c2)": list(),
        "c": list(),
    }

    c_fact_stacked = np.vstack([c_fact_1, c_fact_2])

    for c_fact, key in zip(
        [c_fact_1, c_fact_2, c_fact_stacked, p_w_x],
        ["c1", "c2", "c", "(c1, c2)"],
    ):
        for c in range(c_fact.shape[1]):
            result = np.apply_along_axis(
                bernoulli_std, axis=1, arr=c_fact, c=c
            )
            conditional_variances[key].append(np.mean(result) ** 2)

    return conditional_variances


def produce_ece_curve(
    p: ndarray,
    pred: ndarray,
    true: ndarray,
    exp_mode: str,
    purpose: str = "labels",
    ece_mode: ECEMODE = ECEMODE.WHOLE,
    concept: int = None,
    suffix: str = "",
):
    """Function which produces the ECE curve

    Args:
        p (ndarray): probability vector p
        pred (ndarray): predictions
        true (ndarray): groundtruth values
        exp_mode (str): experiment mode
        purpose (str, default="labels"): either labels or concepts
        ece_mode (ECEMODE, default=ECEMODE.WHOLE): which ECE modality to apply,
        concept (int, default=None): with respect to which concept to produce the curve
        suffix (str, default=""): suffix

    Returns:
        ece: ECE value
    """
    ece = None

    if ece_mode == ECEMODE.FILTERED_BY_CONCEPT:
        ece_data = expected_calibration_error_by_concept(
            p, pred, true, concept
        )
    else:
        ece_data = expected_calibration_error(p, pred, true)

    if ece_data:
        ece, ece_bins = ece_data
        fprint(
            f"Expected Calibration Error (ECE) {exp_mode} on {purpose}",
            ece,
        )
        concept_flag = True if purpose != "labels" else False
        produce_calibration_curve(
            ece_bins,
            ece,
            f"{purpose}_calibration_curve_{exp_mode}{suffix}",
            concept_flag,
        )

    return ece


def generate_concept_labels(concept_labels: List[str]):
    """Produces different concept labels for visualization purpose only

    Args:
        concept_labels (List[str]): original concept labels list

    Returns:
        concept_labels_full: full concept label list
        concept_labels_single: concept label list for single concept
        sklearn_concept_labels: sklearn full concept label list
        sklearn_concept_labels_single: sklearn concept label list for single concept
    """

    # Generate all the product with repetition of size two of the concept labels  (which indeed are all the possible words)

    concept_labels_full = [
        "".join(comb) for comb in product(concept_labels, repeat=2)
    ]
    concept_labels_single = [
        "".join(comb) for comb in product(concept_labels)
    ]
    sklearn_concept_labels = [
        str(int(el)) for el in concept_labels_full
    ]
    sklearn_concept_labels_single = [
        str(int(el)) for el in concept_labels_single
    ]

    return (
        concept_labels_full,
        concept_labels_single,
        sklearn_concept_labels,
        sklearn_concept_labels_single,
    )


def convert_numpy_to_list(obj):
    """Converts numpy elements to list

    Args:
        obj: dictionary objectt

    Returns:
        obj: object ready to be dumped
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, list):
        return [convert_numpy_to_list(item) for item in obj]
    else:
        return obj


def save_dump(args, kwargs, incomplete=False, eltype=""):
    """Save the args into a json dump

    Args:
        args: command line arguments
        kwargs: key-value command line arguments
        incomplete (bool, default=False): whether the dump is incomplete or not
        eltype (str, default=""): evaluation type

    Returns:
        None: This function does not return a value.
    """
    file_path = f"dumps/{get_model_name(args)}-seed_{args.seed}-nens_{args.n_ensembles}-ood_{args.use_ood}-lambda_{args.lambda_h}.json"

    if incomplete:
        file_path = f"dumps/{get_model_name(args)}-seed_{args.seed}-nens_{args.n_ensembles}-ood_{args.use_ood}-lambda_{args.lambda_h}_incomplete_{eltype}_real-kl_{args.real_kl}.json"

    # Convert ndarrays to nested lists in the dictionary
    for key, value in kwargs.items():
        kwargs[key] = convert_numpy_to_list(value)

    del kwargs["ORIGINAL_WEIGHTS"]

    # Dump the dictionary into the file
    with open(file_path, "w") as json_file:
        json.dump(kwargs, json_file)


def produce_confusion_matrices(
    c_true_cc: ndarray,
    c_pred_cc: ndarray,
    c_true: ndarray,
    c_pred: ndarray,
    sklearn_concept_labels: List[str],
    sklearn_concept_labels_single: List[str],
    n_facts: int,
    mode: str,
    suffix: str,
):
    """Saves confusion matrices as figures

    Args:
        c_true_cc (ndarray): groundtruth concepts divided in two dims
        c_pred_cc (ndarray): predicted concepts divided in two dims
        c_true (ndarray): groundtruth concepts
        c_pred (ndarray): predicted concepts
        sklearn_concept_labels (List[str]): sklearn labels
        sklearn_concept_labels_single (List[str]): sklearn labels for single concepts
        n_facts (int): number of elements
        mode (str): modality
        suffix (str): suffix

    Returns:
        None: This function does not return a value.
    """

    from itertools import product

    concept_labels = [
        "".join(comb)
        for comb in product([str(el) for el in range(10)], repeat=2)
    ]
    concept_labels_single = [
        "".join(comb)
        for comb in product([str(el) for el in range(10)])
    ]
    sklearn_concept_labels = [str(int(el)) for el in concept_labels]
    sklearn_concept_labels_single = [
        str(int(el)) for el in concept_labels_single
    ]

    # extend them in order to have a single element: e.g. 03 means that the first element was associated to 0 while the second with 3
    c_extended_true = np.array(
        [int(str(first) + str(second)) for first, second in c_true_cc]
    )
    c_extended_pred = np.array(
        [int(str(first) + str(second)) for first, second in c_pred_cc]
    )

    # arrays of concepts one after the other eg. 0, 1, 2...
    c_extended_true_merged = np.array([int(str(el)) for el in c_true])
    c_extended_pred_merged = np.array([int(str(el)) for el in c_pred])

    fprint("--- Saving the RSs Confusion Matrix ---")

    produce_confusion_matrix(
        "RSs Confusion Matrix on Combined Concepts",
        c_extended_true,
        c_extended_pred,
        sklearn_concept_labels,
        f"confusion_matrix_combined_concept_{mode}_{suffix}",
        "true",
        1,  # n_facts,
    )

    produce_confusion_matrix(
        "RSs Confusion Matrix on Concepts",
        c_extended_true_merged,
        c_extended_pred_merged,
        sklearn_concept_labels,
        f"concept_confusion_matrix_{mode}_{suffix}",
        "true",
        1,
    )


def produce_alpha(
    mode: str,
    worlds_prob: ndarray,
    c_prb_1: ndarray,
    c_prb_2: ndarray,
    c_true: ndarray,
    c_true_cc: ndarray,
    n_facts: int,
    concept_labels: List[str],
    concept_labels_single: List[str],
    type: str,
):
    """Saves alpha matrices as figures

    Args:
        mode (str): test modality
        worlds_prob (ndarray): worlds probabilities
        c_prb_1 (ndarray): concept probability for the first concept
        c_prb_2 (ndarray): concept probability for the second concept
        c_true (ndarray): groundtruth concepts
        c_true_cc (ndarray): groundtruth concepts divided per concepts
        n_facts (int): number of concepts
        concept_labels (List[str]): concepts labels
        concept_labels_single (List[str]): concept labels single concept
        type (str): type

    Returns:
        None: This function does not return a value.
    """
    fprint("--- Computing the probability of each world... ---")

    alpha_M, _ = get_alpha(worlds_prob, c_true_cc, n_facts=n_facts)

    produce_alpha_matrix(
        alpha_M,
        "p((C1, C2)| (G1, G2))",
        concept_labels,
        f"alpha_plot_{mode}",
        n_facts,
    )

    # Only the single model produces the single ALPHA
    if type == EVALUATION_TYPE.NORMAL.value:
        words_prob_single_concept = np.concatenate(
            (c_prb_1, c_prb_2), axis=0
        )
        alpha_M_single, _ = get_alpha_single(
            words_prob_single_concept, c_true, n_facts=n_facts
        )

        produce_alpha_matrix(
            alpha_M_single,
            "p(C | G)",
            concept_labels_single,
            f"alpha_plot_single_{mode}",
            1,
        )


def save_csv(
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
):
    """Saves predictions in a csv file

    Args:
        y_true (ndarray): label groundtruth
        c_true (ndarray): concept groundtruth
        y_pred (ndarray): label prediction
        c_pred (ndarray): concept prediction
        c_true_cc (ndarray): concept groundtruth (cc)
        c_pred_cc (ndarray): concept prediction (cc)
        p_cs (ndarray): concept probability
        p_ys (ndarray): label probability
        p_cs_all (ndarray): concept probability all
        c_factorized_1 (ndarray): concept 1 factorized probability
        c_factorized_2 (ndarray): concept 2 factorized probability
        worlds_prob (ndarray): worlds probability
        gt_factorized (ndarray): groundtruth concepts
        file_path (str): where to save the file

    Returns:
        None: This function does not return a value.
    """
    import csv

    gt_factorized = np.reshape(
        gt_factorized, (int(gt_factorized.shape[0] / 2), 2)
    )

    with open(file_path, "a", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        # save to the file
        for j in range(len(y_true)):
            row = []
            row.append(y_true[j])
            row.append(c_true[j])
            row.append(y_pred[j])
            row.append(c_pred[j])
            for k in range(c_true_cc.shape[1]):
                row.append(c_true_cc[j][k])
            for k in range(c_pred_cc.shape[1]):
                row.append(c_pred_cc[j][k])
            row.append(p_cs[j])
            row.append(p_ys[j])
            for k in range(p_cs_all.shape[1]):
                row.append(p_cs_all[j][k])
            for k in range(c_factorized_1.shape[1]):
                row.append(c_factorized_1[j][k])
            for k in range(c_factorized_2.shape[1]):
                row.append(c_factorized_2[j][k])
            for k in range(gt_factorized.shape[1]):
                row.append(gt_factorized[j][k])
            for k in range(worlds_prob.shape[1]):
                row.append(worlds_prob[j][k])
            csv_writer.writerow(row)
