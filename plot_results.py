#!/usr/bin/env python3

import sys

import pandas as pd

import matplotlib.pyplot as plt


def load_csv(file):
    csv = pd.read_csv(file)

    results = {}
    for _, row in csv.iterrows():
        dataset, run_config, dim, loss = parse_row(row)

        results.setdefault(dataset, {}).setdefault(run_config, {})
        assert dim not in results[dataset][run_config]
        results[dataset][run_config][dim] = loss

    return results


def parse_row(row):
    dim = row["dim"]
    dataset = row["dataset"]
    backbone = row["backbone"]
    gamma = row["gamma"]
    loss = row["test-relative-sse"]

    return dataset, (backbone, gamma), dim, loss


def plot_dataset(results, dataset, titles, labels, settings):
    plt.figure(figsize=(8, 5))
    plt.rcParams.update(settings)

    lines = plot(results[dataset], labels)

    plt.xlabel("Problem dimension ($d$)")
    plt.ylabel("Average test loss")
    plt.title(titles[dataset])

    plt.legend(*sort_lines_with_labels(lines, labels))
    plt.tight_layout()


def plot(results, labels):
    lines = {}
    for run_config, values in results.items():
        x = list(values.keys())
        y = list(values.values())

        if run_config in labels:
            item = labels[run_config]
            (line,) = plt.plot(x, y, ".-", label=item["label"], color=item["color"])
            lines[run_config] = line

    return lines


def sort_lines_with_labels(lines, labels):
    sorted_lines = []
    sorted_labels = []
    for run_config in labels:
        sorted_lines.append(lines[run_config])
        sorted_labels.append(labels[run_config]["label"])
    return sorted_lines, sorted_labels


if __name__ == "__main__":
    results = load_csv(sys.argv[1])

    for dataset in results:
        plot_dataset(
            results,
            dataset,
            {
                "cov": "Covariance estimation",
                "linreg": "Linear regression",
                "expm": "Matrix exponentiation",
                "inv": "Matrix inversion",
            },
            {
                ("mlp", "none"): {
                    "label": "Unsymmetrised",
                    "color": "tab:blue",
                },
                ("emlp", "none"): {
                    "label": r"EMLP",
                    "color": "tab:orange",
                },
                ("mlp", "haar"): {
                    "label": r"Kim et al. ($\gamma = \text{Haar}$)",
                    "color": "tab:green",
                },
                ("mlp", "emlp"): {
                    "label": r"Kim et al. ($\gamma = \text{EMLP}$)",
                    "color": "tab:red",
                },
                ("mlp", "mlp-haar"): {
                    "label": r"Us",
                    "color": "tab:purple",
                },
            },
            {
                # "figure.figsize": (10, 5),  # 4:3 aspect ratio
                "font.size": 11,  # Set font size to 11pt
                "axes.labelsize": 11,  # -> axis labels
                "legend.fontsize": 11,  # -> legends
                "font.family": "Helvetica",
                # "font.family": "lmodern",
                "text.usetex": True,
                "text.latex.preamble": (r"\usepackage{amsmath}"),  # LaTeX preamble
            },
        )

    plt.show()
