import itertools
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import confusion_matrix, f1_score


def get_dfs_merged(
    seeds, category, lambda_h, lambda_kl, train="train", pcbm=False
):
    dfs = []
    for seed in seeds:
        if pcbm:
            df = pd.read_csv(
                f"out/bdd/dpl_auc_pcbm-{seed}/{train}_results_of_BDD_{seed}_{category}_{lambda_h}_{lambda_kl}.csv",
                header=None,
            )
            print(
                f"Taking: out/bdd/dpl_auc_pcbm-{seed}/{train}_results_of_BDD_{seed}_{category}_{lambda_h}_{lambda_kl}.csv"
            )
        else:
            df = pd.read_csv(
                f"out/bdd/dpl_auc-{seed}/{train}_results_of_BDD_{seed}_{category}_{lambda_h}_{lambda_kl}.csv",
                header=None,
            )
            print(
                f"Taking: out/bdd/dpl_auc-{seed}/{train}_results_of_BDD_{seed}_{category}_{lambda_h}_{lambda_kl}.csv"
            )
        dfs.append(df)
    return dfs


def get_dfs_single(
    seed, n_model, category, lambda_h, lambda_kl, train="train"
):
    df = pd.read_csv(
        f"out/bdd/dpl_auc-{seed}/{train}_results_of_BDD_n_mod_{n_model}_{seed}_{category}_{lambda_h}_{lambda_kl}_real_kl.csv",
        header=None,
    )
    print(
        f"Taking: out/bdd/dpl_auc-{seed}/{train}_results_of_BDD_n_mod_{n_model}_{seed}_{category}_{lambda_h}_{lambda_kl}_real_kl.csv"
    )
    return df


def compute_f1(df):
    to_rtn = {}
    y_true = df.values[:, :5]
    y_pred = df.values[:, 5:13]

    # process labels
    preds = np.split(y_pred, 4, axis=1)
    y_prob = []
    labels = []
    for i in range(4):
        y = preds[i]
        y_prob.append(np.max(y, axis=1))
        labels.append(np.argmax(y, axis=1))
    y_pred = np.vstack(labels).T
    y_prob = np.vstack(y_prob).T

    all_y_true, all_y_pred = [], []
    preds, f1_y, y_cfs, ece_y = [], [], [], []
    for i in range(4):
        all_y_true.append(y_true[:, i])
        all_y_pred.append(y_pred[:, i])
        preds.append(y_pred[:, i].reshape(-1, 1))
        f1_value = f1_score(
            y_true[:, i], y_pred[:, i], average="macro"
        )
        f1_y.append(f1_value)
        y_cfs.append(
            confusion_matrix(
                y_true[:, i], y_pred[:, i], normalize="true"
            )
        )
        to_rtn[f"F1 of Label {i}"] = f1_value
        ece_value = produce_ece_curve(
            y_prob[:, i], y_pred[:, i], y_true[:, i]
        )
        ece_y.append(ece_value)
        to_rtn[f"ECE of label {i}"] = ece_value

    prob_C = df.values[:, 13:34]
    c_true = df.values[:, 44:-1].astype(int)

    all_c_true, all_c_pred = [], []
    c_pred, f1_c, cfs, ece_c = [], [], [], []
    for i in range(21):
        all_c_true.append(c_true[:, i])
        all_c_pred.append(np.round(prob_C[:, i]))
        f1_value = f1_score(
            c_true[:, i], np.round(prob_C[:, i]), average="macro"
        )
        f1_c.append(f1_value)
        c_pred.append(
            np.round(prob_C[:, i]).astype(int).reshape(-1, 1)
        )
        cfs.append(
            confusion_matrix(
                c_true[:, i], np.round(prob_C[:, i]), normalize="true"
            )
        )
        ece_value = produce_ece_curve(
            prob_C[:, i],
            np.round(prob_C[:, i]).astype(int),
            c_true[:, i].astype(float),
        )
        ece_c.append(ece_value)
        to_rtn[f"ECE of concept {i}"] = ece_value

    to_rtn["Mean f1 labels"] = np.mean(f1_y)
    to_rtn["Mean ECE of label"] = np.mean(ece_y)
    to_rtn["Mean f1 concepts"] = np.mean(f1_c)
    to_rtn["Mean ECE concepts"] = np.mean(ece_c)

    all_y_true = np.concatenate(all_y_true, axis=0)
    all_y_pred = np.concatenate(all_y_pred, axis=0)

    all_c_true = np.concatenate(all_c_true, axis=0)
    all_c_pred = np.concatenate(all_c_pred, axis=0)

    to_rtn["F1 all labels"] = f1_score(
        all_y_true, all_y_pred, average="macro"
    )
    to_rtn["F1 all concepts"] = f1_score(
        all_c_true, all_c_pred, average="macro"
    )

    return to_rtn


def get_f1_per_dict(dfs, is_list=True):
    def merge_dict(d1, d2):
        nd = {}
        for k, v in d1.items():
            if isinstance(v, list):
                nd[k] = v
                nd[k].append(d2[k])
            else:
                nd[k] = [v, d2[k]]
        return nd

    to_rtn = {}
    if is_list:
        for df in dfs:
            tmp_dict = compute_f1(df)
            if not bool(to_rtn):
                to_rtn = tmp_dict
            else:
                to_rtn = merge_dict(to_rtn, tmp_dict)
    else:
        to_rtn = compute_f1(dfs)

    df = pd.DataFrame(to_rtn.values())
    df = df.T
    df = df.set_axis(to_rtn.keys(), axis=1)

    return df


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


def _populate_bins(confs, preds, labels, num_bins: int):
    # initializes n bins (a bin contains probability from x to x + smth (where smth is greater than zero))
    bin_dict = _bin_initializer(num_bins)

    for i, (confidence, prediction, label) in enumerate(
        zip(confs, preds, labels)
    ):
        try:
            binn = int(math.ceil(num_bins * confidence - 1))
        except:
            binn = 0
        if binn == -1:
            binn = 0
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
    confs, preds, labels, num_bins: int = 10
):
    bin_dict = _populate_bins(
        confs, preds, labels, num_bins
    )  # populate the bins
    num_samples = len(labels)  # number of samples (n)
    ece = sum(
        (bin_info["BIN_ACC"] - bin_info["BIN_CONF"]).__abs__()
        * bin_info["COUNT"]
        / num_samples
        for bin_info in bin_dict.values()  # number of bins basically
    )
    return ece, bin_dict


def produce_ece_curve(p, pred, true, multilabel: bool = False):
    ece = None

    if multilabel:
        ece_data = list()
        for i in range(p.shape[1]):
            ece_data.append(
                expected_calibration_error(
                    p[:, i], pred[:, i], true[:, i]
                )[0]
            )
        return np.mean(np.asarray(ece_data), axis=0)
    else:
        return expected_calibration_error(p, pred, true)[0]


def half_width(upper_bound, lower_bound):
    return (upper_bound - lower_bound) / 2


def to_dict(df):
    to_rtn = {}
    for column_name in df.columns.values.tolist():
        values = df[column_name].to_numpy()
        to_rtn[column_name] = values
    return to_rtn


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


def from_list_to_value(dicts):
    to_rtn = []
    for key, value in dicts.items():
        dict_rtn = {}
        to_ins = value
        if isinstance(value, (list, tuple)):
            to_ins = value[0]
        dict_rtn[key] = to_ins
        to_rtn.append(dict_rtn)
    return to_rtn


def filter_dict(dictionary, keys):
    tmp_dict = {}
    # to_rtn = []
    for key, value in dictionary.items():
        if key in keys:
            tmp_dict[key] = value
    # for k in keys:
    #     to_rtn.append(tmp_dict[k])
    return tmp_dict


def get_stat(
    seed,
    n_model,
    lambda_h="1.0",
    lambda_kl="1.0",
    category="resense",
    train="test",
    set_wandb=False,
):
    import wandb

    print("Tracking statistics for", train, "...")

    keys = [
        "Mean f1 labels",
        "Mean f1 concepts",
        "Mean ECE of label",
        "Mean ECE concepts",
    ]
    for i in range(21):
        keys.append(f"ECE of concept {i}")

    df_bir_full = get_dfs_merged(
        [seed], category, lambda_h, lambda_kl, train=train
    )
    df_bir_list = [get_f1_per_dict(df_bir_full[0], is_list=False)]
    bir_dict = filter_dict(
        to_dict(pd.concat(df_bir_list, ignore_index=True)), keys
    )

    bir_full_to_log = {}
    for key, value in bir_dict.items():
        bir_full_to_log[f"Factorized_on_{train}_{key}"] = value
    bir_full_to_log = from_list_to_value(
        convert_to_json_serializable(bir_full_to_log)
    )
    print(bir_full_to_log)

    if set_wandb:
        for d in bir_full_to_log:
            wandb.log(d)

    try:
        for i in range(n_model):
            df_bir_f = get_dfs_single(
                seed, i, category, lambda_h, lambda_kl, train
            )
            df_bir_list = [get_f1_per_dict(df_bir_f, is_list=False)]
            bir_dict = filter_dict(
                to_dict(pd.concat(df_bir_list, ignore_index=True)),
                keys,
            )

            bir_full_to_log = {}
            for key, value in bir_dict.items():
                bir_full_to_log[f"Single_n_{i}_on_{train}_{key}"] = (
                    value
                )
            bir_full_to_log = from_list_to_value(
                convert_to_json_serializable(bir_full_to_log)
            )
            print(bir_full_to_log)

            if set_wandb:
                for d in bir_full_to_log:
                    wandb.log(d)
    except:
        pass
