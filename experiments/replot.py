"""
Regenerate plots from saved JSON data without re-running experiments.

Usage:
    uv run python replot.py                  # regenerate all plots
    uv run python replot.py vary_NS          # regenerate only vary_NS plots
    uv run python replot.py vary_NB vary_gamma  # regenerate specific plots

Available plot names:
    Synthetic: vary_NS, vary_NB, vary_dimension, vary_gamma, weak_separation
    Real-world: newsgroups, newsgroups_vary_NS, mnist, mnist_vary_NS, covertype
"""

import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PLOTS_DIR = Path(__file__).parent / "plots"
DATA_DIR = Path(__file__).parent / "data"
PLOTS_DIR.mkdir(exist_ok=True)


def save_plot(fig, name):
    fig.savefig(PLOTS_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(PLOTS_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {name}.pdf")


def load(name):
    with open(DATA_DIR / f"{name}.json") as f:
        return json.load(f)


# ── Synthetic plots ──────────────────────────────────────────────────────────

def plot_vary_NS():
    d = load("vary_NS")
    xs = d["N_S_values"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(xs, d["tv_mean"], yerr=d["tv_std"],
                fmt="o-", capsize=4, label="Estimated TV distance", color="C0")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$N_S$ (number of target samples)")
    ax.set_ylabel("TV distance")
    ax.set_title(rf"TV distance vs $N_S$ ($N_B={d['N_B']:,}$, $p={d['p']}$)")
    ax.legend(); ax.grid(True, alpha=0.3)
    save_plot(fig, "vary_NS_tv")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(xs, d["acc_s_mean"], yerr=d["acc_s_std"],
                fmt="s-", capsize=4, label=r"$S$ samples only", color="C3")
    ax.errorbar(xs, d["acc_f_mean"], yerr=d["acc_f_std"],
                fmt="o-", capsize=4, label=r"$S$ + filtered $B$", color="C0")
    ax.set_xscale("log")
    ax.set_xlabel(r"$N_S$ (number of target samples)")
    ax.set_ylabel("Downstream classification accuracy")
    ax.set_title(rf"Downstream accuracy vs $N_S$ ($N_B={d['N_B']:,}$, $p={d['p']}$)")
    ax.legend(); ax.grid(True, alpha=0.3)
    save_plot(fig, "vary_NS_downstream")


def plot_vary_NB():
    d = load("vary_NB")
    xs = d["N_B_values"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(xs, d["tv_mean"], yerr=d["tv_std"],
                fmt="o-", capsize=4, label="Estimated TV distance", color="C0")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$N_B$ (number of big distribution samples)")
    ax.set_ylabel("TV distance")
    ax.set_title(rf"TV distance vs $N_B$ ($N_S={d['N_S']}$, $p={d['p']}$)")
    ax.legend(); ax.grid(True, alpha=0.3)
    save_plot(fig, "vary_NB_tv")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(xs, d["acc_s_mean"], yerr=d["acc_s_std"],
                fmt="s-", capsize=4, label=r"$S$ samples only", color="C3")
    ax.errorbar(xs, d["acc_f_mean"], yerr=d["acc_f_std"],
                fmt="o-", capsize=4, label=r"$S$ + filtered $B$", color="C0")
    ax.set_xscale("log")
    ax.set_xlabel(r"$N_B$ (number of big distribution samples)")
    ax.set_ylabel("Downstream classification accuracy")
    ax.set_title(rf"Downstream accuracy vs $N_B$ ($N_S={d['N_S']}$, $p={d['p']}$)")
    ax.legend(); ax.grid(True, alpha=0.3)
    save_plot(fig, "vary_NB_downstream")


def plot_vary_dimension():
    d = load("vary_dimension")
    xs = d["d_values"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(xs, d["tv_mean"], yerr=d["tv_std"], fmt="o-", capsize=4, color="C0")
    ax.set_xscale("log")
    ax.set_xlabel("Dimension $d$")
    ax.set_ylabel(r"TV bound (FN + FP/$p$)")
    ax.set_title(rf"TV bound vs dimension $d$ ($N_S={d['N_S']}$, $N_B={d['N_B']:,}$, $p={d['p']}$)")
    ax.grid(True, alpha=0.3)
    save_plot(fig, "vary_dimension_tv")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(xs, d["fp_mean"], "s-", label="False positive rate", color="C2")
    ax.plot(xs, d["fn_mean"], "^-", label="False negative rate", color="C3")
    ax.set_xscale("log")
    ax.set_xlabel("Dimension $d$"); ax.set_ylabel("Error rate")
    ax.set_title(r"Classification errors vs dimension $d$")
    ax.legend(); ax.grid(True, alpha=0.3)
    save_plot(fig, "vary_dimension_errors")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(xs, d["acc_s_mean"], yerr=d["acc_s_std"],
                fmt="s-", capsize=4, label=r"$S$ samples only", color="C3")
    ax.errorbar(xs, d["acc_f_mean"], yerr=d["acc_f_std"],
                fmt="o-", capsize=4, label=r"$S$ + filtered $B$", color="C0")
    ax.set_xscale("log")
    ax.set_xlabel("Dimension $d$")
    ax.set_ylabel("Downstream classification accuracy")
    ax.set_title(rf"Downstream accuracy vs dimension $d$ ($N_S={d['N_S']}$, $p={d['p']}$)")
    ax.legend(); ax.grid(True, alpha=0.3)
    save_plot(fig, "vary_dimension_downstream")


def plot_vary_gamma():
    d = load("vary_gamma")
    xs = d["gamma_values"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(xs, d["tv_mean"], yerr=d["tv_std"],
                fmt="o-", capsize=4, label=r"TV bound (FN + FP/$p$)", color="C0")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"Margin $\gamma$"); ax.set_ylabel(r"TV bound")
    ax.set_title(rf"TV bound vs margin $\gamma$ ($N_S={d['N_S']}$, $p={d['p']}$, $R={d['R']}$)")
    ax.legend(); ax.grid(True, alpha=0.3)
    save_plot(fig, "vary_gamma_tv")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(xs, d["acc_s_mean"], yerr=d["acc_s_std"],
                fmt="s-", capsize=4, label=r"$S$ samples only", color="C3")
    ax.errorbar(xs, d["acc_f_mean"], yerr=d["acc_f_std"],
                fmt="o-", capsize=4, label=r"$S$ + filtered $B$", color="C0")
    ax.set_xscale("log")
    ax.set_xlabel(r"Margin $\gamma$")
    ax.set_ylabel("Downstream classification accuracy")
    ax.set_title(rf"Downstream accuracy vs $\gamma$ ($N_S={d['N_S']}$, $p={d['p']}$)")
    ax.legend(); ax.grid(True, alpha=0.3)
    save_plot(fig, "vary_gamma_downstream")


def plot_weak_separation():
    d = load("weak_separation")
    p = d["p"]
    res_O = d["eps_O_results"]
    res_S = d["eps_S_results"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(d["eps_O_values"], res_O["tv_mean"], yerr=res_O["tv_std"],
                fmt="o-", capsize=4, color="C0", label=r"TV bound (FN + FP/$p$)")
    baseline = res_O["tv_mean"][0]
    ref = [baseline + e / p for e in d["eps_O_values"]]
    ax.plot(d["eps_O_values"], ref, "--", color="C1", alpha=0.7,
            label=r"baseline $+ \varepsilon_O / p$ (theory)")
    ax.set_xlabel(r"$\varepsilon_O$ (fraction of $O$ violating margin)")
    ax.set_ylabel(r"TV bound")
    ax.set_title(rf"Weak separation: varying $\varepsilon_O$ ($p={p}$)")
    ax.legend(); ax.grid(True, alpha=0.3)
    save_plot(fig, "weak_sep_eps_O_tv")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(d["eps_S_values"], res_S["tv_mean"], yerr=res_S["tv_std"],
                fmt="o-", capsize=4, color="C0", label=r"TV bound (FN + FP/$p$)")
    baseline = res_S["tv_mean"][0]
    ref = [baseline + e for e in d["eps_S_values"]]
    ax.plot(d["eps_S_values"], ref, "--", color="C1", alpha=0.7,
            label=r"baseline $+ \varepsilon_S$ (theory)")
    ax.set_xlabel(r"$\varepsilon_S$ (fraction of $S$ violating margin)")
    ax.set_ylabel(r"TV bound")
    ax.set_title(r"Weak separation: varying $\varepsilon_S$")
    ax.legend(); ax.grid(True, alpha=0.3)
    save_plot(fig, "weak_sep_eps_S_tv")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(d["eps_O_values"], res_O["acc_s_mean"], yerr=res_O["acc_s_std"],
                fmt="s-", capsize=4, label=r"$S$ samples only", color="C3")
    ax.errorbar(d["eps_O_values"], res_O["acc_f_mean"], yerr=res_O["acc_f_std"],
                fmt="o-", capsize=4, label=r"$S$ + filtered $B$", color="C0")
    ax.set_xlabel(r"$\varepsilon_O$ (fraction of $O$ violating margin)")
    ax.set_ylabel("Downstream classification accuracy")
    ax.set_title(rf"Downstream accuracy vs $\varepsilon_O$ ($p={p}$)")
    ax.legend(); ax.grid(True, alpha=0.3)
    save_plot(fig, "weak_sep_eps_O_downstream")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(d["eps_S_values"], res_S["acc_s_mean"], yerr=res_S["acc_s_std"],
                fmt="s-", capsize=4, label=r"$S$ samples only", color="C3")
    ax.errorbar(d["eps_S_values"], res_S["acc_f_mean"], yerr=res_S["acc_f_std"],
                fmt="o-", capsize=4, label=r"$S$ + filtered $B$", color="C0")
    ax.set_xlabel(r"$\varepsilon_S$ (fraction of $S$ violating margin)")
    ax.set_ylabel("Downstream classification accuracy")
    ax.set_title(r"Downstream accuracy vs $\varepsilon_S$")
    ax.legend(); ax.grid(True, alpha=0.3)
    save_plot(fig, "weak_sep_eps_S_downstream")


# ── Real-world plots ─────────────────────────────────────────────────────────

def plot_newsgroups():
    d = load("newsgroups")
    all_results = d["results"]
    topics = d["topics"]
    short_names = [t.split(".")[-1] for t in topics]

    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(len(topics))
    width = 0.2
    for i, (key, label, color) in enumerate([
        ("acc_s_only", r"$S$ only ($N_S=200$)", "C3"),
        ("acc_filtered", r"$S$ + filtered $B$", "C0"),
        ("acc_random", r"$S$ + random $B$", "C1"),
        ("acc_oracle", "Oracle (all labels)", "C2"),
    ]):
        means = [np.mean(all_results[t][key]) for t in topics]
        stds = [np.std(all_results[t][key]) for t in topics]
        ax.bar(x_pos + (i - 1.5)*width, means, width, yerr=stds, capsize=3,
               label=label, color=color, alpha=0.8)
    ax.set_xticks(x_pos); ax.set_xticklabels(short_names)
    ax.set_ylabel("Downstream balanced accuracy")
    ax.set_title("20 Newsgroups: filtering improves downstream classification")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
    save_plot(fig, "newsgroups_downstream")

    fig, ax = plt.subplots(figsize=(9, 5))
    w2 = 0.25
    fn_means = [np.mean(all_results[t]["fn_rates"]) for t in topics]
    fn_stds = [np.std(all_results[t]["fn_rates"]) for t in topics]
    fp_means = [np.mean(all_results[t]["fp_rates"]) for t in topics]
    fp_stds = [np.std(all_results[t]["fp_rates"]) for t in topics]
    prec_means = [np.mean(all_results[t]["precisions"]) for t in topics]
    ax.bar(x_pos - w2/2, fn_means, w2, yerr=fn_stds, capsize=4,
           label="False negative rate", color="C3", alpha=0.8)
    ax.bar(x_pos + w2/2, fp_means, w2, yerr=fp_stds, capsize=4,
           label="False positive rate", color="C2", alpha=0.8)
    for i, p in enumerate(prec_means):
        ax.annotate(f"prec={p:.2f}", (x_pos[i], max(fn_means[i], fp_means[i]) + 0.02),
                    ha="center", fontsize=9)
    ax.set_xticks(x_pos); ax.set_xticklabels(short_names)
    ax.set_ylabel("Error rate")
    ax.set_title("20 Newsgroups: filter FP/FN rates across topics")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    save_plot(fig, "newsgroups_error_rates")


def plot_newsgroups_vary_NS():
    d = load("newsgroups_vary_NS")
    xs = d["N_S_values"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(xs, d["acc_s_mean"], yerr=d["acc_s_std"],
                fmt="s-", capsize=4, label=r"$S$ samples only", color="C3")
    ax.errorbar(xs, d["acc_f_mean"], yerr=d["acc_f_std"],
                fmt="o-", capsize=4, label=r"$S$ + filtered $B$", color="C0")
    if "acc_r_mean" in d:
        ax.errorbar(xs, d["acc_r_mean"], yerr=d["acc_r_std"],
                    fmt="^-", capsize=4, label=r"$S$ + random $B$", color="C1")
    ax.set_xscale("log")
    ax.set_xlabel(r"$N_S$ (number of target samples)")
    ax.set_ylabel("Downstream balanced accuracy")
    ax.set_title("20 Newsgroups (sci.space): accuracy vs $N_S$")
    ax.legend(); ax.grid(True, alpha=0.3)
    save_plot(fig, "newsgroups_vary_NS_downstream")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(xs, d["fn_mean"], yerr=d["fn_std"],
                fmt="^-", capsize=4, label="False negative rate", color="C3")
    ax.errorbar(xs, d["fp_mean"], yerr=d["fp_std"],
                fmt="s-", capsize=4, label="False positive rate", color="C2")
    ax.set_xscale("log")
    ax.set_xlabel(r"$N_S$"); ax.set_ylabel("Error rate")
    ax.set_title("20 Newsgroups (sci.space): filter errors vs $N_S$")
    ax.legend(); ax.grid(True, alpha=0.3)
    save_plot(fig, "newsgroups_vary_NS_errors")


def plot_mnist():
    d = load("mnist")
    all_results = d["results"]
    digits = [str(x) for x in d["digits"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(len(digits))
    width = 0.2
    for i, (key, label, color) in enumerate([
        ("acc_s_only", r"$S$ only ($N_S=200$)", "C3"),
        ("acc_filtered", r"$S$ + filtered $B$", "C0"),
        ("acc_random", r"$S$ + random $B$", "C1"),
        ("acc_oracle", "Oracle (all labels)", "C2"),
    ]):
        means = [np.mean(all_results[dig][key]) for dig in digits]
        stds = [np.std(all_results[dig][key]) for dig in digits]
        ax.bar(x_pos + (i - 1.5)*width, means, width, yerr=stds, capsize=3,
               label=label, color=color, alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"Digit {dig}" for dig in digits])
    ax.set_ylabel("Downstream balanced accuracy")
    ax.set_title("MNIST: filtering improves downstream classification ($N_S=200$)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
    save_plot(fig, "mnist_downstream")

    fig, ax = plt.subplots(figsize=(9, 5))
    w2 = 0.25
    fn_means = [np.mean(all_results[dig]["fn_rates"]) for dig in digits]
    fp_means = [np.mean(all_results[dig]["fp_rates"]) for dig in digits]
    prec_means = [np.mean(all_results[dig]["precisions"]) for dig in digits]
    ax.bar(x_pos - w2/2, fn_means, w2, label="False negative rate", color="C3", alpha=0.8)
    ax.bar(x_pos + w2/2, fp_means, w2, label="False positive rate", color="C2", alpha=0.8)
    for i, p in enumerate(prec_means):
        ax.annotate(f"prec={p:.2f}", (x_pos[i], max(fn_means[i], fp_means[i]) + 0.005),
                    ha="center", fontsize=9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"Digit {dig}" for dig in digits])
    ax.set_ylabel("Error rate")
    ax.set_title("MNIST: filter error rates ($N_S=200$)")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    save_plot(fig, "mnist_error_rates")


def plot_mnist_vary_NS():
    d = load("mnist_vary_NS")
    xs = d["N_S_values"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(xs, d["acc_s_mean"], yerr=d["acc_s_std"],
                fmt="s-", capsize=4, label=r"$S$ samples only", color="C3")
    ax.errorbar(xs, d["acc_f_mean"], yerr=d["acc_f_std"],
                fmt="o-", capsize=4, label=r"$S$ + filtered $B$", color="C0")
    if "acc_r_mean" in d:
        ax.errorbar(xs, d["acc_r_mean"], yerr=d["acc_r_std"],
                    fmt="^-", capsize=4, label=r"$S$ + random $B$", color="C1")
    ax.set_xscale("log")
    ax.set_xlabel(r"$N_S$ (number of target samples)")
    ax.set_ylabel("Downstream balanced accuracy")
    ax.set_title("MNIST (digit 3): accuracy vs $N_S$")
    ax.legend(); ax.grid(True, alpha=0.3)
    save_plot(fig, "mnist_vary_NS_downstream")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(xs, d["fn_mean"], yerr=d["fn_std"],
                fmt="^-", capsize=4, label="False negative rate", color="C3")
    ax.errorbar(xs, d["fp_mean"], yerr=d["fp_std"],
                fmt="s-", capsize=4, label="False positive rate", color="C2")
    ax.set_xscale("log")
    ax.set_xlabel(r"$N_S$"); ax.set_ylabel("Error rate")
    ax.set_title("MNIST (digit 3): filter errors vs $N_S$")
    ax.legend(); ax.grid(True, alpha=0.3)
    save_plot(fig, "mnist_vary_NS_errors")


def plot_covertype():
    d = load("covertype")
    all_results = d["results"]
    classes = [str(c) for c in d["classes"]]
    class_names = {4: "Cottonwood/Willow\n(p=0.005)", 5: "Aspen\n(p=0.016)"}

    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(classes))
    width = 0.2
    for i, (key, label, color) in enumerate([
        ("acc_s_only", r"$S$ only ($N_S=200$)", "C3"),
        ("acc_filtered", r"$S$ + filtered $B$", "C0"),
        ("acc_random", r"$S$ + random $B$", "C1"),
        ("acc_oracle", "Oracle (all labels)", "C2"),
    ]):
        means = [np.mean(all_results[c][key]) for c in classes]
        stds = [np.std(all_results[c][key]) for c in classes]
        ax.bar(x_pos + (i - 1.5)*width, means, width, yerr=stds, capsize=3,
               label=label, color=color, alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([class_names[int(c)] for c in classes])
    ax.set_ylabel("Downstream balanced accuracy")
    ax.set_title("Covertype: rare class detection ($N_S=200$)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
    save_plot(fig, "covertype_downstream")


# ── Main ─────────────────────────────────────────────────────────────────────

ALL_PLOTS = {
    "vary_NS": plot_vary_NS,
    "vary_NB": plot_vary_NB,
    "vary_dimension": plot_vary_dimension,
    "vary_gamma": plot_vary_gamma,
    "weak_separation": plot_weak_separation,
    "newsgroups": plot_newsgroups,
    "newsgroups_vary_NS": plot_newsgroups_vary_NS,
    "mnist": plot_mnist,
    "mnist_vary_NS": plot_mnist_vary_NS,
    "covertype": plot_covertype,
}

if __name__ == "__main__":
    names = sys.argv[1:] if len(sys.argv) > 1 else list(ALL_PLOTS.keys())
    for name in names:
        if name not in ALL_PLOTS:
            print(f"Unknown plot: {name}")
            print(f"Available: {', '.join(ALL_PLOTS.keys())}")
            sys.exit(1)
        print(f"Plotting {name}...")
        ALL_PLOTS[name]()
    print("Done.")
