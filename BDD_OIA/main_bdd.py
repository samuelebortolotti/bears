# -*- coding: utf-8 -*-
# Standard Imports
import argparse
import math
import operator
import os
import pdb
import pickle
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Torch-related
import torch
import torch.utils.data.dataloader as dataloader
import torchvision
import wandb
from aggregators_BDD import CBM_aggregator, additive_scalar_aggregator
from BDD.config import (
    BASE_DIR,
    LR_DECAY_SIZE,
    MIN_LR,
    N_ATTRIBUTES,
    N_CLASSES,
    UPWEIGHT_RATIO,
)
from BDD.dataset import find_class_imbalance, load_data
from conceptizers_BDD import (
    PCBMConceptizer,
    image_cnn_conceptizer,
    image_fcc_conceptizer,
)
from DPL.dpl import DPL
from DPL.dpl_auc import DPL_AUC
from DPL.dpl_auc_pcbm import DPL_AUC_PCBM
from models import GSENN
from parametrizers import dfc_parametrizer, image_parametrizer
from scipy.special import softmax
from SENN.arglist import get_senn_parser
from SENN.eval_utils import estimate_dataset_lipschitz

# Local imports
from SENN.utils import (
    concept_grid,
    generate_dir_names,
    noise_stability_plots,
    plot_theta_stability,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    multilabel_confusion_matrix,
    precision_recall_fscore_support,
    precision_score,
)
from testers_BDD import ClassificationTesterFactory
from torch.autograd import Variable
from torch.utils.data import TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from trainers_BDD import GradPenaltyTrainer
from visualization import (
    create_output_folder,
    plot_grouped_entropies,
    produce_alpha_matrix,
    produce_bar_plot,
    produce_calibration_curve,
    produce_confusion_matrix,
    produce_scatter_multi_class,
)


def convert_to_json_serializable(obj):
    if isinstance(obj, (np.ndarray, np.generic)):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {
            key: convert_to_json_serializable(value)
            for key, value in obj.items()
        }
    elif isinstance(obj, (int, float, bool, str, type(None))):
        return obj
    else:
        raise TypeError(f"Unsupported type: {type(obj)}")


def entropy(probabilities, n_values: int):
    # Compute entropy along the columns
    probabilities += 1e-5
    probabilities /= 1 + (n_values * 1e-5)

    entropy_values = -np.sum(
        probabilities * np.log(probabilities), axis=1
    ) / np.log(n_values)
    return entropy_values


def mean_entropy(probabilities, n_values: int) -> float:
    # Accepts a ndarray of dim n_examples * n_classes (or equivalently n_concepts)
    entropy_values = entropy(probabilities, n_values).mean()
    return entropy_values.item()


def print_metrics(
    y_true,
    y_pred,
    c_true,
    pc_pred,
    w_true,
    w_pred,
    p_cs_all,
    n_facts: int,
    prefix,
):
    yac = accuracy_score(y_true, y_pred)
    yf1 = f1_score(y_true, y_pred, average="weighted")

    cac = accuracy_score(c_true, pc_pred)
    cf1 = f1_score(c_true, pc_pred, average="weighted")

    wac = accuracy_score(c_true, pc_pred)
    wf1 = f1_score(c_true, pc_pred, average="weighted")

    h_c = mean_entropy(p_cs_all, n_facts)

    print(f"Performances on {prefix}:")
    print(f"Concepts:\n    ACC: {cac}, F1: {cf1}")
    print(f"Worlds:\n    ACC: {wac}, F1: {wf1}")
    print(f"Labels:\n      ACC: {yac}, F1: {yf1}")
    print(f"Entropy:\n     H(C): {h_c}")

    return h_c


def print_multiclass_metric(
    y_true,
    y_pred,
    c_true,
    c_pred,
    p_cs_all,
    n_facts: int,
    prefix,
):
    yac = precision_score(y_true, y_pred, average="micro")
    yf1 = f1_score(y_true, y_pred, average="micro")

    cac = precision_score(c_true, c_pred, average="micro")
    cf1 = f1_score(c_true, c_pred, average="micro")

    h_c = mean_entropy(p_cs_all, n_facts)

    print(f"Performances on {prefix}:")
    print(f"Concepts:\n    ACC: {cac}, F1: {cf1}")
    print(f"Labels:\n      ACC: {yac}, F1: {yf1}")
    print(f"Entropy:\n     H(C): {h_c}")

    return h_c


def produce_h_c_given_y(
    p_cs_all, y_true, nr_classes: int, mode: str, suffix: str
) -> None:
    h_c_given_y = class_mean_entropy(
        p_cs_all, np.concatenate((y_true, y_true)), nr_classes
    )
    if plt.get_fignums():
        plt.close("all")

    produce_bar_plot(
        h_c_given_y,
        "Groundtruth class",
        "Entropy",
        "H(C|Y)",
        f"h_c_given_y_{mode}{suffix}",
        True,
    )


def class_mean_entropy(probabilities, true_classes, n_classes: int):
    # Compute the mean entropy per class
    num_samples, num_classes = probabilities.shape

    class_mean_entropy_values = np.zeros(
        n_classes
    )  # all possible results by summing 2 digits
    class_counts = np.zeros(n_classes)

    for i in range(num_samples):
        sample_entropy = entropy(
            np.expand_dims(probabilities[i], axis=0), num_classes
        )
        class_mean_entropy_values[true_classes[i]] += sample_entropy
        class_counts[true_classes[i]] += 1

    # Avoid division by zero
    class_counts[class_counts == 0] = 1

    class_mean_entropy_values /= class_counts

    return class_mean_entropy_values


def produce_confusion_matrices(
    p_true, p_pred, n_values: int, mode: str, suffix: str
):
    sklearn_concept_labels = [str(int(el)) for el in range(n_values)]

    print("--- Saving the RSs Confusion Matrix ---")

    cm = produce_confusion_matrix(
        "RSs Confusion Matrix on Combined Concepts",
        p_true,
        p_pred,
        sklearn_concept_labels,
        f"confusion_matrix_combined_concept_{mode}_{suffix}",
        "true",
        1,  # n_facts,
    )

    return cm


def _bin_initializer(num_bins: int):
    # Builds the bin
    return {
        i: {
            "COUNT": 0,
            "CONF": 0,
            "ACC": 0,
            "BIN_ACC": 0,
            "BIN_CONF": 0,
        }
        for i in range(num_bins)
    }


def _populate_bins(
    confs, preds, labels, num_bins: int, multilabel=False
):
    # initializes n bins (a bin contains probability from x to x + smth (where smth is greater than zero))
    bin_dict = _bin_initializer(num_bins)

    for confidence, prediction, label in zip(confs, preds, labels):
        if multilabel:
            for i in range(confidence.shape[0]):
                binn = int(math.ceil(num_bins * confidence[i] - 1))
                bin_dict[binn]["COUNT"] += 1
                bin_dict[binn]["CONF"] += confidence[i]
                bin_dict[binn]["ACC"] += (
                    1 if label[i] == prediction[i] else 0
                )
        else:
            binn = int(math.ceil(num_bins * confidence - 1))
            bin_dict[binn]["COUNT"] += 1
            bin_dict[binn]["CONF"] += confidence
            bin_dict[binn]["ACC"] += 1 if label == prediction else 0

    for bin_info in bin_dict.values():
        bin_count = bin_info["COUNT"]
        if bin_count == 0:
            bin_info["BIN_ACC"] = 0
            bin_info["BIN_CONF"] = 0
        else:
            bin_info["BIN_ACC"] = bin_info["ACC"] / bin_count
            bin_info["BIN_CONF"] = bin_info["CONF"] / bin_count

    return bin_dict


def expected_calibration_error(
    confs, preds, labels, multilabel=False, num_bins: int = 10
):
    # Perfect calibration is achieved when the ECE is zero
    # Formula: ECE = sum 1 upto M of number of elements in bin m|Bm| over number of samples across all bins (n), times |(Accuracy of Bin m Bm) - Confidence of Bin m Bm)|

    bin_dict = _populate_bins(
        confs, preds, labels, num_bins, multilabel
    )  # populate the bins
    num_samples = len(labels)  # number of samples (n)
    if multilabel:
        num_samples *= labels.shape[1]
    ece = sum(
        (bin_info["BIN_ACC"] - bin_info["BIN_CONF"]).__abs__()
        * bin_info["COUNT"]
        / num_samples
        for bin_info in bin_dict.values()  # number of bins basically
    )
    return ece, bin_dict


def produce_ece_curve(
    p,
    pred,
    true,
    exp_mode: str,
    purpose: str = "labels",
    suffix: str = "",
    multilabel: bool = False,
):
    ece = None

    if multilabel:
        ece_data = list()
        for i in range(p.shape[1]):
            ece_data.append(
                expected_calibration_error(
                    p[:, i], pred[:, i], true[:, i]
                )[0]
            )
        ece_data = np.mean(np.asarray(ece_data), axis=0)
    else:
        ece_data = expected_calibration_error(p, pred, true)

    if ece_data:
        if multilabel:
            ece = ece_data
            print(
                f"Expected Calibration Error (ECE) {exp_mode} on {purpose}",
                ece,
            )
        else:
            ece, ece_bins = ece_data
            print(
                f"Expected Calibration Error (ECE) {exp_mode} on {purpose}",
                ece,
            )
            concept_flag = True if purpose != "labels" else False
            produce_calibration_curve(
                ece_bins,
                ece,
                f"{purpose}_calibration_curve_{exp_mode}_{suffix}",
                concept_flag,
            )

    return ece


def print_distance(
    tester,
    test_loader,
):
    dist_fs, dist_left, dist_right = tester.p_c_x_distance(
        test_loader
    )
    print(f"Distance FS: {dist_fs}")
    print(f"Distance LEFT: {dist_left}")
    print(f"Distance RIGHT: {dist_right}")


def plot_multilabel_confusion_matrices(
    conf_matrices, labels, num_labels, plot_title, fig_title
):
    num_rows = num_labels // 5 + 1
    num_cols = min(num_labels, 5)

    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(20, 5 * num_rows), sharey=True
    )

    if num_rows > 1:
        axes = axes.flatten()

    plt.ylabel("Actual")
    plt.xlabel("Predicted")

    for i, (matrix, ax) in enumerate(zip(conf_matrices, axes)):
        matrix_to_disp = matrix.astype("float") / matrix.sum()
        sns.heatmap(
            matrix_to_disp,
            annot=True,
            fmt=".2%",
            cmap="viridis",
            cbar=False,
            ax=ax,
        )
        ax.set_title(f"Label {labels[i]}")

        if num_rows > 1:
            ax.set_ylabel("Actual")
        if i >= len(conf_matrices) - num_cols:
            ax.set_xlabel("Predicted")

    plt.suptitle(f"{plot_title}", y=1.02)
    plt.savefig(
        f"./plots/normalized_multilabel_confusion_{fig_title}.png"
    )


def plot_statistics_single_model(
    bayes_method,
    y_true,
    y_predictions,
    y_predictions_prob,
    y_prob,
    w_groundtruths,
    w_predictions,
    w_probs,
    w_predictions_prob_value,
    pc_prob,
    pc_pred,
    c_true,
    mean_hc_dict,
    ece_dict,
    cfs,
    suffix,
):

    for i, direction in zip(
        [0, 1, 2], ["stop_forward", "left", "right"]
    ):
        mean_h_c = print_metrics(
            y_true,
            y_predictions,
            c_true,
            pc_pred,
            w_groundtruths[i],
            w_predictions[i],
            w_probs[i],
            w_probs[i].shape[1],
            bayes_method + f"_{direction}{suffix}",
        )

        # create the keys for the mean hc and ece
        if direction not in mean_hc_dict:
            mean_hc_dict[direction] = []
            ece_dict[direction] = []

        # add it to the mean hc list
        mean_hc_dict[direction].append(mean_h_c)
        # produce_h_c_given_y(w_probs[i], y_true_not_one_hot, 5, bayes_method, f"_{direction}{suffix}")

        cm = produce_confusion_matrices(
            w_groundtruths[i],
            w_predictions[i],
            w_probs[i].shape[1],
            bayes_method,
            f"worlds_{direction}{suffix}",
        )

        cfs[f"worlds_{direction}{suffix}"] = cm

        ece = produce_ece_curve(
            w_predictions_prob_value[i],
            w_predictions[i],
            w_groundtruths[i],
            bayes_method,
            "worlds",
            f"_{direction}{suffix}",
        )

        ece_dict[direction].append(ece)

        if i > 0:
            continue

        conf_matrix = multilabel_confusion_matrix(
            y_true, y_predictions
        )
        labels = [f"{i + 1}" for i in range(len(conf_matrix))]
        plot_multilabel_confusion_matrices(
            conf_matrix,
            labels,
            len(labels),
            f"Confusion Matrix on Labels in {bayes_method}",
            f"labels_{bayes_method}{suffix}",
        )
        cfs[f"labels_{bayes_method}{suffix}"] = conf_matrix

        conf_matrix = multilabel_confusion_matrix(c_true, pc_pred)
        labels = [f"{i + 1}" for i in range(len(conf_matrix))]
        plot_multilabel_confusion_matrices(
            conf_matrix,
            labels,
            len(labels),
            f"Confusion Matrix on Concepts in {bayes_method}",
            f"concepts_{bayes_method}{suffix}",
        )
        cfs[f"concepts_{bayes_method}{suffix}"] = conf_matrix

        produce_ece_curve(
            y_predictions_prob,
            y_predictions,
            y_true,
            bayes_method,
            "labels",
            suffix,
            True,
        )

        produce_ece_curve(
            pc_prob,
            pc_pred,
            c_true,
            bayes_method,
            "concepts",
            suffix,
            True,
        )


def compute_concept_factorized_entropy(c_fact, c_true):
    def ova_entropy(p):
        import math

        p += 1e-5
        p /= 1 + (p.shape[0] * 1e-5)

        positive = p * math.log2(p)
        negative = (1 - p) * math.log2(1 - p)

        return -(positive + negative)

    conditional_entropies = {"c_ova_filtered": list()}

    for c in range(c_fact.shape[1]):
        c_fact_filtered = c_fact[:, c]
        c_fact_filtered = np.expand_dims(c_fact_filtered, axis=-1)

        result = np.apply_along_axis(
            ova_entropy, axis=1, arr=c_fact_filtered
        )
        conditional_entropies["c_ova_filtered"].append(
            np.mean(result)
        )

    return conditional_entropies


def get_accuracy_and_counter(c_pred, c_true, n_concepts):
    from collections import OrderedDict

    concept_counter_list = OrderedDict()
    concept_acc_list = OrderedDict()

    for i in range(n_concepts):
        concept_acc_list[i] = 0
        concept_counter_list[i] = 0

    for i, lc in enumerate(c_true):
        for c in range(lc.shape[0]):
            if lc[c] == 1:
                concept_counter_list[c] += 1
                if c_pred[i][c] == 1:
                    concept_acc_list[c] += 1

    for i in concept_counter_list.keys():
        if concept_counter_list[i] != 0:
            concept_acc_list[i] /= concept_counter_list[i]

    return concept_counter_list.values(), concept_acc_list.values()


def concept_accuracy(c_pred, c_true, n_concepts):
    return get_accuracy_and_counter(c_pred, c_true, n_concepts)


def world_accuracy(world_pred, world_true):
    pred_dict = {}
    for i, direction, world_size in zip(
        [0, 1, 2],
        ["stop_forward", "left", "right"],
        [
            int(math.pow(2, 9)),
            int(math.pow(2, 6)),
            int(math.pow(2, 6)),
        ],
    ):
        acc_list, counter_list = get_accuracy_and_counter(
            world_pred[i], world_true[i], world_size
        )
        pred_dict[direction] = [acc_list, counter_list]
    return pred_dict


def evaluate_mix(true, pred):
    ac = accuracy_score(true, pred)
    f1 = f1_score(true, pred, average="weighted")

    return ac, f1


def evaluate_mix_multiclass(true, pred):
    ac = precision_score(true, pred, average="micro")
    f1 = f1_score(true, pred, average="micro")

    return ac, f1


def single_concept_ece(bayes_method, labels, p, pred, true, suffix):
    single_concepts_ece = []

    for c in labels:
        ece_single_concept = produce_ece_curve(
            p[:, c],
            pred[:, c],
            true[:, c],
            bayes_method,
            f"concept {c}",
            f"_{suffix}",
        )
        single_concepts_ece.append(ece_single_concept)

    return single_concepts_ece


def compute_mean_acc_f1(y_true, y_predictions, dim):
    f1_M_list, acc_list = list(), list()
    for i in range(dim):
        f1_M_list.append(
            f1_score(
                y_true[:, i], y_predictions[:, i], average="macro"
            )
        )
        acc_list.append(
            accuracy_score(y_true[:, i], y_predictions[:, i])
        )
    f1_M = np.mean(np.asarray(f1_M_list), axis=0)
    acc = np.mean(np.asarray(acc_list), axis=0)
    return f1_M, acc


def compute_acc_f1(
    y_true,
    y_predictions,
    c_true,
    pc_pred,
    w_groundtruths,
    w_predictions,
):
    f1, accuracy = compute_mean_acc_f1(
        y_true, y_predictions, y_true.shape[1]
    )
    precision_per_class, recall_per_class, f1_score_per_class, _ = (
        precision_recall_fscore_support(
            y_true, y_predictions, average=None
        )
    )

    concept_f1, concept_accuracy = compute_mean_acc_f1(
        c_true, pc_pred, c_true.shape[1]
    )
    worlds_test_accuracies, worlds_test_f1 = worlds_f1_acc(
        w_groundtruths, w_predictions
    )
    return (
        accuracy,
        f1,
        precision_per_class,
        recall_per_class,
        f1_score_per_class,
        concept_accuracy,
        concept_f1,
        worlds_test_accuracies,
        worlds_test_f1,
    )


def worlds_f1_acc(w_groundtruths, w_predictions):
    worlds_test_accuracies = []
    worlds_test_f1 = []
    for i in range(len(w_groundtruths)):
        accuracy = accuracy_score(w_groundtruths[i], w_predictions[i])
        f1 = f1_score(
            w_groundtruths[i], w_predictions[i], average="micro"
        )
        worlds_test_accuracies.append(accuracy)
        worlds_test_f1.append(f1)
    return worlds_test_accuracies, worlds_test_f1


def dump_dictionary(
    args,
    mean_hc_dict,
    ece_dict,
    factorized_concept_dict,
    count_acc_dict,
    mean_hc_dict_train,
    ece_dict_train,
    factorized_concept_dict_train,
    count_acc_dict_train,
    yacc_train,
    cacc_train,
    yacc_test,
    cacc_test,
    yf1_train,
    cf1_train,
    yf1_test,
    cf1_test,
    yacc_per_class,
    yf1_per_class,
    yacc_per_class_train,
    yf1_per_class_train,
    cacc_hard_train,
    cf1_hard_train,
    cacc_hard_test,
    cf1_hard_test,
    cfs,
    single_concept_ece_list_test,
    single_concept_ece_list_train,
    incomplete=False,
    category="none",
):
    import json

    kwargs = {
        "mean_hc": mean_hc_dict,
        "ece": ece_dict,
        "fact_concept": factorized_concept_dict,
        "count_acc": count_acc_dict,
        "mean_hc_train": mean_hc_dict_train,
        "ece_train": ece_dict_train,
        "fact_concept_train": factorized_concept_dict_train,
        "count_acc_train": count_acc_dict_train,
        "yacc": yacc_test,
        "cacc": cacc_test,
        "yacc_train": yacc_train,
        "cacc_train": cacc_train,
        "yf1_train": yf1_train,
        "cf1_train": cf1_train,
        "yf1_test": yf1_test,
        "cf1_test": cf1_test,
        "yacc_per_class": yacc_per_class,
        "yf1_per_class": yf1_per_class,
        "yacc_per_class_train": yacc_per_class_train,
        "yf1_per_class_train": yf1_per_class_train,
        "cacc_hard_train": cacc_hard_train,
        "cf1_hard_train": cf1_hard_train,
        "cacc_hard_test": cacc_hard_test,
        "cf1_hard_test": cf1_hard_test,
        "single_concept_ece": single_concept_ece_list_test,
        "single_concept_ece_train": single_concept_ece_list_train,
        "cfs": cfs,
    }

    if not os.path.exists("dumps"):
        # If not, create it
        os.makedirs("dumps")

    file_path = f"dumps/dpl-seed_{args.seed}-nens_{args.n_models}-lambda_{args.lambda_h}.json"

    if incomplete:
        print("Sono incompleto")
        file_path = f"dumps/dpl-seed_{args.seed}-nens_{args.n_models}-lambda_{args.lambda_h}_incomplete_{category}.json"
        print(file_path)

    # Convert ndarrays to nested lists in the dictionary
    for key, value in kwargs.items():
        kwargs[key] = convert_to_json_serializable(value)

    # Dump the dictionary into the file
    with open(file_path, "w") as json_file:
        json.dump(kwargs, json_file)


def total_evaluation_stuff(
    args,
    mean_hc_dict,
    ece_dict,
    factorized_concept_dict,
    count_acc_dict,
    mean_hc_dict_train,
    ece_dict_train,
    factorized_concept_dict_train,
    count_acc_dict_train,
    yacc_train,
    cacc_train,
    yacc_test,
    cacc_test,
    yf1_train,
    cf1_train,
    yf1_test,
    cf1_test,
    yacc_per_class,
    yf1_per_class,
    yacc_per_class_train,
    yf1_per_class_train,
    cacc_hard_train,
    cf1_hard_train,
    cacc_hard_test,
    cf1_hard_test,
    cfs,
    single_concept_ece_list_test,
    single_concept_ece_list_train,
    worlds_size,
):
    evals = [
        "frequentist",
        "laplace",
        "mcdropout",
        "biretta",
        "deep ensembles",
    ]
    categories = [i for i in range(21)]

    for direction in ["stop_forward", "left", "right"]:
        produce_scatter_multi_class(
            mean_hc_dict[direction],
            ece_dict[direction],
            evals,
            "bddoia",
            f"_{direction}",
        )

    for c_prob in ["c_ova_filtered"]:
        plot_grouped_entropies(
            categories,
            "bddoia",
            factorized_concept_dict[c_prob],
            evals,
            f"entropy_on_{c_prob}",
            f"Entropy on Concept: {c_prob}",
        )

    for acc, count in zip(["c_acc"], ["c_count"]):
        fig, axs = plt.subplots(2, 1, figsize=(8, 8))

        plot_grouped_entropies(
            categories,
            "bddoia",
            count_acc_dict[count],
            evals,
            f"{count}",
            f"Concept Count",
            False,
            axs[0],
            fig,
        )
        plot_grouped_entropies(
            categories,
            "bddoia",
            count_acc_dict[acc],
            evals,
            f"{acc}",
            "Concept Accuracy",
            False,
            axs[1],
            fig,
            set_lim=True,
        )
        file_path = f"plots/bddoia_fact_concept_acc.png"
        fig.tight_layout()
        plt.savefig(file_path, dpi=150)
        plt.close()

    dump_dictionary(
        args,
        mean_hc_dict,
        ece_dict,
        factorized_concept_dict,
        count_acc_dict,
        mean_hc_dict_train,
        ece_dict_train,
        factorized_concept_dict_train,
        count_acc_dict_train,
        yacc_train,
        cacc_train,
        yacc_test,
        cacc_test,
        yf1_train,
        cf1_train,
        yf1_test,
        cf1_test,
        yacc_per_class,
        yf1_per_class,
        yacc_per_class_train,
        yf1_per_class_train,
        cacc_hard_train,
        cf1_hard_train,
        cacc_hard_test,
        cf1_hard_test,
        cfs,
        single_concept_ece_list_test,
        single_concept_ece_list_train,
    )


def load_checkpoint(model, savepath=None, seed=42):
    if savepath == None:
        raise ValueError("Select the save path")

    if not os.path.exists(savepath):
        raise ValueError("The save path does not exists")

    filename = os.path.join(savepath, f"model_best-{seed}.pth.tar")
    # model data
    model_dict = torch.load(filename)

    # Extract the state_dict from the dictionary
    state_dict = model_dict["state_dict"]

    model.load_state_dict(state_dict)


# This function does not modification
def parse_args():
    senn_parser = get_senn_parser()

    ### Local ones
    parser = argparse.ArgumentParser(
        parents=[senn_parser],
        add_help=False,
        description="Interpteratbility robustness evaluation",
    )

    # #setup
    parser.add_argument(
        "-d",
        "--datasets",
        nargs="+",
        default=[
            "heart",
            "ionosphere",
            "breast-cancer",
            "wine",
            "heart",
            "glass",
            "diabetes",
            "yeast",
            "leukemia",
            "abalone",
        ],
        help="<Required> Set flag",
    )
    parser.add_argument(
        "--lip_calls",
        type=int,
        default=10,
        help="ncalls for bayes opt gp method in Lipschitz estimation",
    )
    parser.add_argument(
        "--lip_eps",
        type=float,
        default=0.01,
        help="eps for Lipschitz estimation",
    )
    parser.add_argument(
        "--lip_points",
        type=int,
        default=100,
        help="sample size for dataset Lipschitz estimation",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="gp",
        help="black-box optimization method",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="dpl",
        help="Choose model to fit",
    )

    parser.add_argument(
        "--exp_decay_lr",
        type=float,
        default=0.1,
        help="Exponential decay for the LR scheduler",
    )

    parser.add_argument(
        "--which_c",
        type=int,
        nargs="+",
        default=[-1],
        help="Which concepts explicitly supervise (-1 means all)",
    )
    parser.add_argument(
        "--wandb", type=str, default=None, help="Activate wandb"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="BDD-OIA",
        help="Select wandb project",
    )
    parser.add_argument(
        "--do-test",
        default=False,
        action="store_true",
        help="Test the model",
    )

    parser.add_argument(
        "--deep_sep",
        default=False,
        action="store_true",
        help="Use KL to differentiate the models",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.01,
        help="Use KL to differentiate the models",
    )
    parser.add_argument(
        "--lambda_h",
        type=float,
        default=1.0,
        help="Lambda parameter used to weight the entropy loss",
    )
    parser.add_argument(
        "--n-models", type=int, default=30, help="Number of runs"
    )
    parser.add_argument(
        "--knowledge_aware_kl",
        default=True,
        action="store_true",
        help="Use KL with Knowledge",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="none",
        choices=[
            "none",
            "frequentist",
            "mcdropout",
            "laplace",
            "bears",
            "deepensembles",
        ],
        help="Select method to run",
    )
    parser.add_argument(
        "--lambda_kl",
        type=float,
        default=1.0,
        help="Lambda parameter used to weight the KL loss",
    )
    parser.add_argument(
        "--pcbm",
        action="store_true",
        default=False,
        help="KL for PCBM",
    )
    #####

    args = parser.parse_args()

    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    return args


"""
Main function:
load data, set models, train and test, and save results.
After ending this function, you can see <./out/bdd/*> directory to check outputs.

Inputs:
    None
Returns:
    None

Inputs loaded in this function:
    ./data/BDD: images of CUB_200_2011
    ./data/BDD/train_BDD_OIA.pkl, val_BDD_OIA.pkl, test_BDD_OIA.pkl: train, val, test samples
    ./models/bdd100k_24.pth: Faster RCNN pretrained by BDD100K (RCNN_global())

Outputs made in this function (same as CUB):
    *.pkl: model
    grad*/training_losses.pdf: loss figure
    grad*/concept_grid.pdf: images which maximize and minimize each unit in the concept layer
    grad*/test_results_of_BDD.csv: predicted and correct labels, prSedicted and correct concepts, coefficient of each concept
"""


def main(args):

    # get hyperparameters
    if args.wandb is not None:
        print("\n---wandb on\n")
        wandb.init(
            project=args.project,
            entity=args.wandb,
            name=str(args.model_name),
            config=args,
        )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # set which GPU uses
    # if args.cuda:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    # else:
    #     device = torch.device("cpu")

    # load dataset
    train_data_path = "./data/bdd2048/train_BDD_OIA.pkl"
    val_data_path = "./data/bdd2048/val_BDD_OIA.pkl"
    test_data_path = "./data/bdd2048/test_BDD_OIA.pkl"

    # load_data. Detail is BDD/dataset.py, lines 149-. This function is made by CBM's authors

    image_dir = "data/bdd2048/"
    train_loader = load_data(
        [train_data_path],
        True,
        False,
        args.batch_size,
        uncertain_label=False,
        n_class_attr=2,
        image_dir=image_dir + "train",
        resampling=False,
    )
    train_loader_no_shuffle = load_data(
        [train_data_path],
        True,
        False,
        args.batch_size,
        uncertain_label=False,
        n_class_attr=2,
        image_dir=image_dir + "train",
        resampling=False,
        no_shuffle=True,
    )
    valid_loader = load_data(
        [val_data_path],
        True,
        False,
        args.batch_size,
        uncertain_label=False,
        n_class_attr=2,
        image_dir=image_dir + "val",
        resampling=False,
    )
    test_loader = load_data(
        [test_data_path],
        True,
        False,
        args.batch_size,
        uncertain_label=False,
        n_class_attr=2,
        image_dir=image_dir + "test",
        resampling=False,
    )

    # get paths (see SENN/utils.py, lines 34-). This function is made by SENN's authors
    model_path, log_path, results_path = generate_dir_names(
        "bdd", args
    )

    # Convert the arguments to a string representation
    arg_string = "\n".join(
        [f"{arg}={getattr(args, arg)}" for arg in vars(args)]
    )
    file_path = "%s/args.txt" % (results_path)
    with open(file_path, "w") as f:
        f.write(arg_string)

    """
    Next, we set four networks, conceptizer, parametrizer, aggregator, and pretrained_model
    Pretrained_model (h(x)): encoder (h) Faster RCNN (see ./BDD/template_model.py)
    Conceptizer (e1(h(x))): concepts layer (see conceptizer.py)
    Parametrizer (e2(h(x))): network to compute parameters to get concepts (see parametrizer.py)
    Aggregator (f(e1(h(x)),e2(h(x)))): output layer (see aggregators.py)
    """

    # only "fcc" conceptizer use, otherwise cannot use (not modifile so as to fit this task...)
    if args.h_type == "fcc":
        conceptizer1 = image_fcc_conceptizer(
            2048,
            args.nconcepts,
            args.nconcepts_labeled,
            args.concept_dim,
            args.h_sparsity,
            args.senn,
        )
    elif args.h_type == "cnn":
        print("[ERROR] please use fcc network")
        sys.exit(1)
    else:
        print("[ERROR] please use fcc network")
        sys.exit(1)

    parametrizer1 = dfc_parametrizer(
        2048,
        1024,
        512,
        256,
        128,
        args.nconcepts,
        args.theta_dim,
        layers=4,
    )
    buf = 1

    """
    If you train CBM model, set cbm, <python main_cub.py --cbm>.
    In this case, our model does not use unknown concepts even if you set the number of unknown concepts.
    NOTE: # of unknown concepts = args.nconcepts - args.nconcepts_labeled
    """
    if args.cbm == True:
        aggregator = CBM_aggregator(
            args.concept_dim, args.nclasses, args.nconcepts_labeled
        )
    else:
        aggregator = additive_scalar_aggregator(
            args.concept_dim, args.nclasses
        )

    # you should set load_model as True. If you set, you can use inception v.3 as the encoder, otherwise end.

    """
    Function GSENN is in models.py
    model: model using outputs of inception v.3
    model_aux: mdoel using auxiliary output of inception v.3
    """
    # model = GSENN(conceptizer1, parametrizer1, aggregator, args.cbm, args.senn)
    if args.model_name == "dpl":
        model = DPL(
            conceptizer1,
            parametrizer1,
            aggregator,
            args.cbm,
            args.senn,
            device,
        )
    elif args.model_name == "dpl_auc":
        model = DPL_AUC(
            conceptizer1,
            parametrizer1,
            aggregator,
            args.cbm,
            args.senn,
            device,
        )
    elif args.model_name == "dpl_auc_pcbm":
        args.pcbm = True
        conceptizer1 = PCBMConceptizer(
            2048,
            args.nconcepts,
            args.nconcepts_labeled,
            args.concept_dim,
            args.h_sparsity,
            args.senn,
            device,
        )
        model = DPL_AUC_PCBM(
            conceptizer1,
            parametrizer1,
            aggregator,
            args.cbm,
            args.senn,
            device,
        )

    # send models to device you want to use
    model = model.to(device)
    print("Res path", results_path)
    load_checkpoint(
        model, f"models/bdd/{args.model_name}-{args.seed}", args.seed
    )
    print("Model", model)

    # Test or train
    if args.do_test:
        # create output folder
        create_output_folder()

        # dictionaries for storing partial results
        mean_hc_dict = {}
        ece_dict = {}
        factorized_concept_dict = {}
        count_and_acc_dict = {}
        cfs = {}

        mean_hc_dict_train = {}
        ece_dict_train = {}
        factorized_concept_dict_train = {}
        count_and_acc_dict_train = {}

        yacc_train = []
        cacc_train = []
        cacc_hard_train = []
        yf1_train = []
        cf1_train = []
        cf1_hard_train = []
        yacc_test = []
        cacc_test = []
        cacc_hard_test = []
        yf1_test = []
        cf1_test = []
        cf1_hard_test = []
        yacc_per_class = []
        yf1_per_class = []
        yacc_per_class_train = []
        yf1_per_class_train = []

        single_concept_ece_list_test = {}
        single_concept_ece_list_train = {}

        # loop over all bayesian methods
        methods_to_do = [
            "frequentist",
            "mcdropout",
            "laplace",
            "bears",
            "deepensembles",
        ]

        if args.type != "none":
            methods_to_do = [args.type]

        for bayes_method in methods_to_do:
            # Call the factory to get the model done
            tester = ClassificationTesterFactory.get_model(
                bayes_method, model, args, device
            )

            separate_from_others = False

            if bayes_method == "bears":
                separate_from_others = True

            # Trains the ensemble only when ensemble is the model or approximate laplace
            tester.setup(
                train_loader,
                train_loader_no_shuffle,
                [
                    args.seed + seed + 1
                    for seed in range(args.n_models)
                ],
                valid_loader,
                epochs=args.epochs,
                save_path=model_path,
                separate_from_others=separate_from_others,
                epsilon=args.epsilon,
                lambda_h=args.lambda_h,
                lambda_kl=args.lambda_kl,
            )

            # plot the losses for deep ensembles only
            if (
                bayes_method == "deepensembles"
                or bayes_method == "bears"
            ):
                tester.plot_losses(
                    bayes_method, save_path=results_path
                )

            save_file_name = f"{results_path}/test_results_of_BDD_{args.seed}_{bayes_method}_{args.lambda_h}_{args.lambda_kl}.csv"
            fp = open(save_file_name, "w")
            fp.close()

            #### EM
            save_file_name_train = f"{results_path}/train_results_of_BDD_{args.seed}_{bayes_method}_{args.lambda_h}_{args.lambda_kl}.csv"
            fp_train = open(save_file_name_train, "w")
            fp_train.close()

            # evaluation by test dataset
            tester.test_and_save_csv(
                test_loader,
                save_file_name,
                fold="test",
                pcbm=args.pcbm,
            )
            tester.test_and_save_csv(
                train_loader,
                save_file_name_train,
                fold="train",
                pcbm=args.pcbm,
            )

            # for the ensemble method write everything as frequentist
            if bayes_method != "frequentist":
                if bayes_method == "laplace":
                    for i, (inputs, targets, concepts) in enumerate(
                        test_loader
                    ):
                        ensemble = tester.get_ensemble_from_bayes(
                            args.n_models, inputs
                        )
                        break
                else:
                    ensemble = tester.get_ensemble_from_bayes(
                        args.n_models
                    )

                for j in range(len(ensemble)):
                    frequentist_m_tester = (
                        ClassificationTesterFactory.get_model(
                            "frequentist", ensemble[j], args, device
                        )
                    )

                    # initialize the csv file (cleaning before training)
                    save_file_name = f"{results_path}/test_results_of_BDD_n_mod_{j}_{args.seed}_{bayes_method}_{args.lambda_h}_{args.lambda_kl}_real_kl.csv"
                    fp = open(save_file_name, "w")
                    fp.close()

                    #### EM
                    save_file_name_train = f"{results_path}/train_results_of_BDD_n_mod_{j}_{args.seed}_{bayes_method}_{args.lambda_h}_{args.lambda_kl}_real_kl.csv"
                    fp_train = open(save_file_name_train, "w")
                    fp_train.close()

                    # evaluation by test dataset
                    apply_dropout = False

                    if bayes_method == "mcdropout":
                        apply_dropout = True

                    frequentist_m_tester.test_and_save_csv(
                        test_loader,
                        save_file_name,
                        fold="test",
                        dropout=apply_dropout,
                        pcbm=args.pcbm,
                    )
                    frequentist_m_tester.test_and_save_csv(
                        train_loader,
                        save_file_name_train,
                        fold="train",
                        dropout=apply_dropout,
                        pcbm=args.pcbm,
                    )

            if bayes_method in ["bears", "deepensembles"]:
                tester.save_model_params_all(
                    save_path=results_path,
                    separate_from_others=separate_from_others,
                    lambda_h=args.lambda_h,
                )

        if args.type == "none":
            total_evaluation_stuff(
                args,
                mean_hc_dict,
                ece_dict,
                factorized_concept_dict,
                count_and_acc_dict,
                mean_hc_dict_train,
                ece_dict_train,
                factorized_concept_dict_train,
                count_and_acc_dict_train,
                yacc_train,
                cacc_train,
                yacc_test,
                cacc_test,
                yf1_train,
                cf1_train,
                yf1_test,
                cf1_test,
                yacc_per_class,
                yf1_per_class,
                yacc_per_class_train,
                yf1_per_class_train,
                cacc_hard_train,
                cf1_hard_train,
                cacc_hard_test,
                cf1_hard_test,
                cfs,
                single_concept_ece_list_test,
                single_concept_ece_list_train,
                [math.pow(2, 9), math.pow(2, 6), math.pow(2, 6)],
            )
        else:
            dump_dictionary(
                args,
                mean_hc_dict,
                ece_dict,
                factorized_concept_dict,
                count_and_acc_dict,
                mean_hc_dict_train,
                ece_dict_train,
                factorized_concept_dict_train,
                count_and_acc_dict_train,
                yacc_train,
                cacc_train,
                yacc_test,
                cacc_test,
                yf1_train,
                cf1_train,
                yf1_test,
                cf1_test,
                yacc_per_class,
                yf1_per_class,
                yacc_per_class_train,
                yf1_per_class_train,
                cacc_hard_train,
                cf1_hard_train,
                cacc_hard_test,
                cf1_hard_test,
                cfs,
                single_concept_ece_list_test,
                single_concept_ece_list_train,
                incomplete=True,
                category=args.type,
            )

        from track_stuff import get_stat

        # if args.wandb:
        get_stat(
            args.seed,
            args.n_models,
            args.lambda_h,
            args.lambda_kl,
            args.type,
            "train",
            set_wandb=args.wandb,
        )
        get_stat(
            args.seed,
            args.n_models,
            args.lambda_h,
            args.lambda_kl,
            args.type,
            "test",
            set_wandb=args.wandb,
        )

    else:
        # initialize the csv file (cleaning before training)
        save_file_name = "%s/test_results_of_BDD.csv" % (results_path)
        fp = open(save_file_name, "w")
        fp.close()

        #### EM
        save_file_name_train = "%s/train_results_of_BDD.csv" % (
            results_path
        )
        fp_train = open(save_file_name_train, "w")
        fp_train.close()

        # training all models. This function is in trainers.py
        trainer = GradPenaltyTrainer(model, args, device)

        # train
        trainer.train(
            train_loader,
            valid_loader,
            epochs=args.epochs,
            save_path=model_path,
            seed=args.seed,
            pcbm=args.pcbm,
        )

        # make figures
        trainer.plot_losses(save_path=results_path)

        # evaluation by test dataset
        trainer.test_and_save_csv(
            test_loader, save_file_name, fold="test", pcbm=args.pcbm
        )
        trainer.test_and_save_csv(
            train_loader,
            save_file_name_train,
            fold="train",
            pcbm=args.pcbm,
        )

        # send model result to cpu
        model.eval().to("cpu")

    if args.wandb is not None:
        wandb.finish()

    """
    This function is in SENN/utils.py (lines 591-). 
    This function makes figures "grad*/concept_grid.pdf", which represents the maximize and minimize each unit in the concept layer
    """
    # concept_grid(model, pretrained_model, test_loader, top_k = 10, device="cpu", save_path = results_path + '/concept_grid.pdf')


if __name__ == "__main__":
    args = parse_args()
    # the number of task class
    args.nclasses = 5
    args.theta_dim = args.nclasses

    print(args)

    import sys

    sys.stdout.flush()

    main(args)

    print("### All done! ###")
