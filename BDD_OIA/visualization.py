from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

OUTPUT_FOLDER = "./plots"


def create_output_folder() -> None:
    import os

    if not os.path.exists(OUTPUT_FOLDER):
        os.mkdir(OUTPUT_FOLDER)


def convert_concepts_to_indexes(values, reference_array):
    # Create a mapping dictionary from values to indices in the second array
    value_to_index = {
        value: index for index, value in enumerate(reference_array)
    }
    # Map the elements of the first array to indices from the second array
    mapped_indices = np.array(
        [value_to_index[value] for value in values]
    )
    return mapped_indices


def produce_confusion_matrix(
    title: str,
    y_true: ndarray,
    y_pred: ndarray,
    labels: List[str],
    plot_title: str,
    normalize: str = "all",
    ntimes: int = 10,
) -> None:
    create_output_folder()

    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()

    # getting the string values
    ground_truth = y_true.astype(str)
    predictions = y_pred.astype(str)

    all_labels = np.union1d(
        np.unique(ground_truth), np.unique(predictions)
    )
    label_encoder.fit(all_labels)

    ground_truth = label_encoder.transform(ground_truth)
    predictions = label_encoder.transform(predictions)

    # All classes
    label_indices = np.argsort(label_encoder.classes_.astype(int))
    labels = np.array(label_encoder.classes_)[label_indices]
    classes = np.unique(np.concatenate([ground_truth, predictions]))

    cm = confusion_matrix(
        ground_truth, predictions, labels=classes, normalize=normalize
    )
    cm = cm[label_indices, :][:, label_indices]

    # Create a ConfusionMatrixDisplay
    rs_confusion_matrix_disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=label_encoder.classes_
    )

    # Create a figure with specified size and DPI
    fig, ax = plt.subplots(figsize=(16, 16), dpi=150)

    # Customize the colormap, color intensity, and format
    cmap = plt.get_cmap("viridis")
    im = rs_confusion_matrix_disp.plot(
        cmap=cmap, values_format="", ax=ax, include_values=False
    )

    x_labels = ax.get_xticklabels()
    tick_positions = np.arange(len(x_labels))
    tick_labels = [
        x_labels[i] if i % ntimes == 0 else ""
        for i in range(len(x_labels))
    ]
    ax.set_xticks(
        tick_positions, tick_labels, rotation=90, fontsize=10
    )

    y_labels = ax.get_yticklabels()
    tick_positions = np.arange(len(y_labels))
    tick_labels = [
        y_labels[i] if i % ntimes == 0 else ""
        for i in range(len(y_labels))
    ]
    ax.set_yticks(tick_positions, tick_labels, fontsize=10)

    # Set title and color bar label
    ax.set_title(title)

    # Specify the file path and name where you want to save the image
    file_path = f"{OUTPUT_FOLDER}/{plot_title}.png"

    # Save the confusion matrix visualization as an image
    plt.savefig(file_path, dpi=150)

    plt.close()

    return cm


def produce_world_probability_table(
    data_dict: Dict[str, float],
    title="Table",
    key_string: str = "Key",
    key_value: str = "Value",
):
    keys = list(data_dict.keys())
    values = list(data_dict.values())

    _, ax = plt.subplots(figsize=(10, 4))

    # Hide axes
    ax.axis("off")

    # Create the table with blue color
    table_data = [
        [key, f"{value:.2f}"] for key, value in zip(keys, values)
    ]
    table = ax.table(
        cellText=table_data,
        colLabels=[key_string, key_value],
        loc="center",
        cellLoc="center",
        colColours=["#99c2ff"] * 2,
    )

    # Customize the appearance
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.5, 1.5)

    ax.set_title(
        title, fontweight="bold", fontsize=16, color="#3366cc"
    )

    # Specify the file path and name where you want to save the image
    file_path = f"{OUTPUT_FOLDER}/mean_probability_table.png"

    # Save the table visualization as an image
    plt.savefig(file_path, dpi=150)

    plt.close()


def produce_alpha_matrix(
    data_dict: Dict[str, np.ndarray],
    title="Matrix Plot",
    concept_labels=[],
    plot_title: str = "alpha_plot",
    ntimes: int = 10,
):
    keys = list(data_dict.keys())
    values = list(data_dict.values())

    sorted_data = sorted(zip(keys, values), key=lambda x: int(x[0]))
    keys, values = zip(*sorted_data)

    # Assuming all arrays in the dictionary have the same dimensions
    matrix_data = np.array(values)

    plt.figure(figsize=(16, 16))

    # Plot the matrix
    plt.imshow(matrix_data, cmap="viridis", interpolation="nearest")

    # Set title
    plt.title(title, fontweight="bold", fontsize=16, color="#3366cc")

    # Set x-axis ticks and labels
    # tick_positions = np.arange(len(concept_labels))
    # tick_labels = [concept_labels[i] if i % ntimes == 0 else '' for i in range(len(concept_labels))]
    # plt.xticks(tick_positions, tick_labels, rotation=90, fontsize=10)
    plt.xticks(
        np.arange(len(concept_labels)),
        concept_labels,
        rotation=90,
        fontsize=10,
    )

    # Set y-axis ticks and labels
    # tick_positions = np.arange(len(keys))
    # tick_labels = [keys[i] if i % ntimes == 0 else '' for i in range(len(keys))]
    # plt.yticks(tick_positions, tick_labels, fontsize=10)
    plt.yticks(np.arange(len(keys)), keys, fontsize=10)

    # Add colorbar
    cbar = plt.colorbar(label="Values", shrink=0.8)

    # Adjust colorbar tick label font size
    cbar.ax.tick_params(labelsize=10)

    # Specify the file path and name where you want to save the image
    file_path = f"{OUTPUT_FOLDER}/{plot_title}.png"

    # Save the matrix plot as an image
    plt.savefig(file_path, dpi=150)

    plt.close()


def produce_scatter_multi_class(
    x_values_list,
    y_values_list,
    labels,
    dataset,
    suffix,
    colors=None,
    markers=None,
):
    if colors is None:
        colors = [
            "blue",
            "red",
            "green",
            "orange",
            "purple",
            "brown",
            "pink",
            "gray",
        ]

    if markers is None:
        markers = ["o", "s", "^", "D", "v", ">", "<", "p"]

    # Create a scatter plot for each class
    for i, (x_values, y_values) in enumerate(
        zip(x_values_list, y_values_list)
    ):
        plt.scatter(
            [x_values],
            [y_values],
            color=colors[i],
            marker=markers[i],
            label=labels[i],
        )

    max_x = max(x_values_list)
    max_y = max(y_values_list)

    # Customize the plot
    plt.title(f"Scatter Plot Dataset: {dataset}")
    plt.xlabel("Mean H(C)")
    plt.ylabel("Concepts ECE")
    plt.xlim(0, max(1, max_x + 0.2))
    plt.ylim(0, max(1, max_y + 0.2))
    plt.tight_layout()
    plt.legend()

    # Save or display the plot
    file_path = (
        f"{OUTPUT_FOLDER}/{dataset}_hc_ece_scatter_plot{suffix}.png"
    )
    plt.savefig(file_path, dpi=150)

    # Close
    plt.close()


def plot_grouped_entropies(
    categories,
    dataset,
    values_list,
    group_labels,
    prefix,
    title,
    save=True,
    ax=None,
    fig=None,
    add_numbers=False,
    set_lim=False,
):
    bar_width = 0.35
    group_gap = 0.35
    num_categories = len(categories)

    index = np.arange(num_categories, dtype=float)

    # Adjust the index positions to create separation between groups
    total_group_width = (
        num_categories * bar_width + (num_categories - 1) * group_gap
    )
    linspace_values = np.linspace(
        -total_group_width / 2, total_group_width / 2, num_categories
    )
    index += linspace_values

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    if save:
        fig, ax = plt.subplots()

    for i, values in enumerate(values_list):
        bars = ax.bar(
            index + i * bar_width,
            values,
            bar_width,
            label=group_labels[i],
            color=colors[i],
        )

        if add_numbers:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{height:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                )

    ax.set_xlabel("Categories", fontsize=12)
    ax.set_ylabel("Values", fontsize=12)
    ax.set_title(f"{title}", fontsize=14)
    ax.set_xticks(index + bar_width)
    if set_lim:
        ax.set_ylim(0, 1)
    ax.set_xticklabels(categories, fontsize=10)
    ax.legend()

    if save:
        file_path = (
            f"{OUTPUT_FOLDER}/{dataset}_{prefix}_hc_bar_plot.png"
        )
        fig.tight_layout()
        plt.savefig(file_path, dpi=150)

        plt.close()


def produce_calibration_curve(
    bin_info: Dict[str, float],
    ece: float,
    plot_title: str = "calibration_curve",
    concept: bool = True,
):
    num_bins = len(bin_info)

    # Extract relevant information from bin_info
    bin_confidence = [
        bin_info[i]["BIN_CONF"] for i in range(num_bins)
    ]
    bin_accuracy = [bin_info[i]["BIN_ACC"] for i in range(num_bins)]
    bin_counts = [bin_info[i]["COUNT"] for i in range(num_bins)]

    # Calculate the center of each bin
    bin_centers = np.linspace(
        1 / (2 * num_bins), 1 - 1 / (2 * num_bins), num_bins
    )

    # Create a subplot with two plots (2 rows, 1 column)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 10), sharex=True
    )

    txt = "Concept" if concept else "Label"
    fig.suptitle(f"{txt} Calibration Curve ECE: {ece:.2f}")

    # Plot the percentage of samples per bin in the second plot
    bin_percentages = np.array(bin_counts) / np.sum(bin_counts)
    bin_percentages = bin_percentages * 100
    ax1.bar(
        bin_centers,
        bin_percentages,
        width=1 / num_bins,
        color="blue",
        alpha=0.7,
        label="Percentage of Samples",
    )

    # Plot grey dashed vertical lines for weighted average confidence and accuracy
    avg_confidence = np.sum(
        np.array(bin_confidence) * bin_counts
    ) / np.sum(bin_counts)
    avg_accuracy = np.sum(
        np.array(bin_accuracy) * bin_counts
    ) / np.sum(bin_counts)
    ax1.axvline(
        x=avg_confidence,
        color="red",
        linestyle="--",
        label="Weighted Avg. Confidence",
    )
    ax1.axvline(
        x=avg_accuracy,
        color="black",
        linestyle="--",
        label="Weighted Avg. Accuracy",
    )

    # Customize the second plot
    ax1.set_ylabel("Percentage of Samples: %")
    ax1.legend(loc="upper left")
    ax1.grid(True)

    # Plot the calibration curve with bars
    ax2.bar(
        bin_centers,
        bin_accuracy,
        width=1 / num_bins,
        color="blue",
        alpha=0.7,
        label="Output",
    )

    # Plot the red bars on top to fill the gap
    ideal_line = np.linspace(0, 1, num_bins)
    gap_values = np.maximum(
        ideal_line - np.array(bin_accuracy), 0
    )  # Adjust the gap width as needed
    ax2.bar(
        bin_centers,
        gap_values,
        width=1 / num_bins,
        color="red",
        alpha=0.7,
        label="Gap",
        bottom=bin_accuracy,
    )

    # Plot the ideal line (diagonal)
    ax2.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        label="Perfect Calibration",
    )

    # Customize the first plot
    ax2.set_xlabel("Mean Predicted Probability (Confidence)")
    ax2.set_ylabel("Mean Accuracy")
    ax2.legend(loc="upper left")
    ax2.grid(True)

    # Save the plot as an image
    file_path = f"{OUTPUT_FOLDER}/{plot_title}.png"
    plt.savefig(file_path, dpi=150)
    plt.close()


def produce_bar_plot(
    data: ndarray,
    xlabel: str = "Groundtruth class",
    ylabel: str = "Occurrences",
    title: str = "",
    plot_name: str = "bar_plot",
    ylim=False,
):
    indices = np.arange(len(data))

    # Create the bar plot with improved styling
    plt.bar(
        indices, data, color="blue", edgecolor="black", linewidth=1.2
    )

    # Adding labels and title
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)

    if ylim:
        plt.ylim(0, 1)

    # xticks
    plt.xticks(np.arange(len(data)), np.arange(len(data)))

    # Adding grid lines for better readability
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Save the plot as an image
    file_path = f"{OUTPUT_FOLDER}/{plot_name}.png"
    plt.savefig(file_path, dpi=150)
    plt.close()
