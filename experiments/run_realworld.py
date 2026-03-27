"""
Real-world data experiments for "Dimension-Free Data Filtering".

We apply the filtering algorithm to real datasets where one class/topic
serves as the target distribution S, and the full dataset serves as B.
"""

import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import time

PLOTS_DIR = Path(__file__).parent / "plots"
DATA_DIR = Path(__file__).parent / "data"
SKLEARN_DATA = Path(__file__).parent / "sklearn_data"
PLOTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
SKLEARN_DATA.mkdir(exist_ok=True)

import os
os.environ["SCIKIT_LEARN_DATA"] = str(SKLEARN_DATA)


def save_data(name, data):
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        return obj

    converted = {}
    for k, v in data.items():
        if isinstance(v, dict):
            converted[k] = {kk: convert(vv) for kk, vv in v.items()}
        elif isinstance(v, list):
            converted[k] = [convert(x) for x in v]
        else:
            converted[k] = convert(v)

    with open(DATA_DIR / f"{name}.json", "w") as f:
        json.dump(converted, f, indent=2)


def save_plot(fig, name):
    fig.savefig(PLOTS_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(PLOTS_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {name}.pdf")


def run_filter(x_S, x_B, C=10.0, weight_scale=1.0):
    """Run the SVM-based filter. Returns boolean mask of B samples that pass,
    and the classifier.

    We use logistic regression with L2 regularization for real-world data,
    which generalizes better than hard-margin SVM in high dimensions with limited N_S.
    """
    N_S = len(x_S)
    N_B = len(x_B)

    X_train = np.vstack([x_S, x_B])
    y_train = np.array([1]*N_S + [-1]*N_B)

    clf = LogisticRegression(
        C=C,
        class_weight={1: weight_scale * N_B / N_S, -1: 1.0},
        max_iter=5000,
        solver="lbfgs",
    )
    clf.fit(X_train, y_train)

    scores_B = clf.decision_function(x_B)
    passed = scores_B >= 0
    return passed, clf


def run_downstream(x_S, x_B, passed, X_train_pool, is_target_train,
                   X_test, y_test, rng, N_S):
    """
    Run the downstream classification task (target vs non-target).

    Returns dict with accuracies for:
    - S-only: train on N_S target + random negatives
    - Filtered: S + filtered B as positive + random negatives
    - Random B: S + random B subset as positive (should be bad)
    - Oracle: trained on all labeled data
    """
    non_target_idx = np.where(~is_target_train)[0]

    def score(clf, X, y):
        return balanced_accuracy_score(y, clf.predict(X))

    # (a) S-only: balanced with 5x negatives
    n_neg_s = min(N_S * 5, len(non_target_idx))
    neg_sample_s = rng.choice(non_target_idx, n_neg_s, replace=False)
    X_s_train = np.vstack([x_S, X_train_pool[neg_sample_s]])
    y_s_train = np.array([1]*N_S + [0]*n_neg_s)
    clf_s = LogisticRegression(max_iter=1000, C=1.0)
    clf_s.fit(X_s_train, y_s_train)
    acc_s = score(clf_s, X_test, y_test)

    # (b) Filtered: cap filtered samples to avoid overwhelming S with noise
    x_filtered = x_B[passed]
    n_filt = len(x_filtered)
    max_filt = max(5 * N_S, 500)
    if n_filt > max_filt:
        filt_idx = rng.choice(n_filt, max_filt, replace=False)
        x_filtered = x_filtered[filt_idx]
        n_filt = max_filt
    n_pos_f = N_S + n_filt
    n_neg_f = min(n_pos_f * 3, len(non_target_idx))
    neg_sample_f = rng.choice(non_target_idx, n_neg_f, replace=False)
    X_f_train = np.vstack([x_S, x_filtered, X_train_pool[neg_sample_f]])
    y_f_train = np.array([1]*N_S + [1]*n_filt + [0]*n_neg_f)
    clf_f = LogisticRegression(max_iter=1000, C=1.0)
    clf_f.fit(X_f_train, y_f_train)
    acc_f = score(clf_f, X_test, y_test)

    # (c) Random B: take same number of samples from B randomly
    n_rand = max(n_filt, 1)
    rand_idx = rng.choice(len(x_B), min(n_rand, len(x_B)), replace=False)
    x_rand = x_B[rand_idx]
    n_neg_r = min((N_S + len(x_rand)) * 3, len(non_target_idx))
    neg_sample_r = rng.choice(non_target_idx, n_neg_r, replace=False)
    X_r_train = np.vstack([x_S, x_rand, X_train_pool[neg_sample_r]])
    y_r_train = np.array([1]*N_S + [1]*len(x_rand) + [0]*n_neg_r)
    clf_r = LogisticRegression(max_iter=1000, C=1.0)
    clf_r.fit(X_r_train, y_r_train)
    acc_r = score(clf_r, X_test, y_test)

    # (d) Oracle: use class_weight='balanced' to handle imbalanced training data
    clf_o = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced')
    clf_o.fit(X_train_pool, is_target_train.astype(int))
    acc_o = score(clf_o, X_test, y_test)

    return {"acc_s": acc_s, "acc_f": acc_f, "acc_r": acc_r, "acc_o": acc_o}


def compute_metrics(passed, labels_B):
    """labels_B: 1 if from S component, 0 if from O."""
    s_in_B = labels_B == 1
    o_in_B = labels_B == 0
    fn_rate = 1 - passed[s_in_B].mean() if s_in_B.sum() > 0 else 0.0
    fp_rate = passed[o_in_B].mean() if o_in_B.sum() > 0 else 0.0
    precision = labels_B[passed].mean() if passed.sum() > 0 else 0.0
    return {"fn_rate": fn_rate, "fp_rate": fp_rate, "precision": precision,
            "n_passed": int(passed.sum()), "pass_rate": float(passed.mean())}


# ── 20 Newsgroups ────────────────────────────────────────────────────────────

def experiment_20newsgroups(n_trials=10):
    """
    20 Newsgroups: one topic is S, rest is O. B is the full dataset.

    We try multiple target topics to show the method works across
    different difficulty levels.
    """
    print("=" * 60)
    print("Real-world: 20 Newsgroups")
    print("=" * 60)

    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer

    data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    vectorizer = TfidfVectorizer(max_features=10000)
    X_all = vectorizer.fit_transform(data.data).toarray()
    y_all = data.target
    target_names = data.target_names

    # Pick topics spanning easy to hard
    topic_indices = [
        (15, "sci.space"),        # distinctive topic
        (7, "rec.autos"),         # moderately distinctive
        (1, "comp.graphics"),     # among several comp.* groups
        (10, "rec.sport.hockey"), # sports topic
    ]

    all_results = {}

    for target_idx, topic_name in topic_indices:
        print(f"\n  Target: {topic_name} (class {target_idx})")

        is_target = (y_all == target_idx)
        X_target_all = X_all[is_target]
        n_target_total = is_target.sum()
        p_true = n_target_total / len(y_all)
        print(f"    Total target samples: {n_target_total}, p={p_true:.4f}")

        # For downstream task: classify within the target class based on
        # presence of a discriminative feature (median split of a high-variance dim)
        # Actually, let's use a more natural task: train a classifier to predict
        # target vs non-target, and measure how well it works on held-out data.
        #
        # Downstream task: binary classification (target vs non-target) on held-out test set.
        # This directly measures if the filtered data helps the downstream classifier.

        results = {"fn_rates": [], "fp_rates": [], "precisions": [],
                   "acc_s_only": [], "acc_filtered": [], "acc_random": [],
                   "acc_oracle": [], "n_passed": [], "n_S_used": []}

        for trial in range(n_trials):
            rng = np.random.default_rng(100 * trial + target_idx)

            # Split: 70% for building B, 30% for test
            train_idx, test_idx = train_test_split(
                np.arange(len(y_all)), test_size=0.3, random_state=trial,
                stratify=y_all)

            X_train_pool = X_all[train_idx]
            y_train_pool = y_all[train_idx]
            X_test = X_all[test_idx]
            y_test = (y_all[test_idx] == target_idx).astype(int)

            is_target_train = (y_train_pool == target_idx)

            # S: subsample target class (simulate having few high-quality samples)
            target_indices = np.where(is_target_train)[0]
            N_S = min(200, len(target_indices))
            s_idx = rng.choice(target_indices, N_S, replace=False)
            x_S = X_train_pool[s_idx]

            # B: the full training pool
            x_B = X_train_pool
            labels_B = is_target_train.astype(int)

            results["n_S_used"].append(N_S)

            # Run filter
            passed, clf_filter = run_filter(x_S, x_B)
            m = compute_metrics(passed, labels_B)
            results["fn_rates"].append(m["fn_rate"])
            results["fp_rates"].append(m["fp_rate"])
            results["precisions"].append(m["precision"])
            results["n_passed"].append(m["n_passed"])

            # Downstream
            ds = run_downstream(x_S, x_B, passed, X_train_pool, is_target_train,
                                X_test, y_test, rng, N_S)
            results["acc_s_only"].append(ds["acc_s"])
            results["acc_filtered"].append(ds["acc_f"])
            results["acc_random"].append(ds["acc_r"])
            results["acc_oracle"].append(ds["acc_o"])

        # Print summary
        print(f"    FN rate: {np.mean(results['fn_rates']):.4f} "
              f"± {np.std(results['fn_rates']):.4f}")
        print(f"    FP rate: {np.mean(results['fp_rates']):.4f} "
              f"± {np.std(results['fp_rates']):.4f}")
        print(f"    Precision: {np.mean(results['precisions']):.4f}")
        print(f"    Passed: {np.mean(results['n_passed']):.0f} "
              f"(of {len(x_B)})")
        print(f"    Downstream acc (S-only):   {np.mean(results['acc_s_only']):.4f}")
        print(f"    Downstream acc (filtered): {np.mean(results['acc_filtered']):.4f}")
        print(f"    Downstream acc (random B): {np.mean(results['acc_random']):.4f}")
        print(f"    Downstream acc (oracle):   {np.mean(results['acc_oracle']):.4f}")

        all_results[topic_name] = {
            k: [float(x) for x in v] for k, v in results.items()
        }

    save_data("newsgroups", {"results": all_results,
                              "topics": [t[1] for t in topic_indices],
                              "n_trials": n_trials})

    # Bar chart comparing methods across topics
    topics = [t[1] for t in topic_indices]
    short_names = [t.split(".")[-1] for t in topics]

    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(len(topics))
    width = 0.2

    s_means = [np.mean(all_results[t]["acc_s_only"]) for t in topics]
    s_stds = [np.std(all_results[t]["acc_s_only"]) for t in topics]
    f_means = [np.mean(all_results[t]["acc_filtered"]) for t in topics]
    f_stds = [np.std(all_results[t]["acc_filtered"]) for t in topics]
    r_means = [np.mean(all_results[t]["acc_random"]) for t in topics]
    r_stds = [np.std(all_results[t]["acc_random"]) for t in topics]
    o_means = [np.mean(all_results[t]["acc_oracle"]) for t in topics]
    o_stds = [np.std(all_results[t]["acc_oracle"]) for t in topics]

    ax.bar(x_pos - 1.5*width, s_means, width, yerr=s_stds, capsize=3,
           label=r"$S$ only ($N_S=200$)", color="C3", alpha=0.8)
    ax.bar(x_pos - 0.5*width, f_means, width, yerr=f_stds, capsize=3,
           label=r"$S$ + filtered $B$", color="C0", alpha=0.8)
    ax.bar(x_pos + 0.5*width, r_means, width, yerr=r_stds, capsize=3,
           label=r"$S$ + random $B$", color="C1", alpha=0.8)
    ax.bar(x_pos + 1.5*width, o_means, width, yerr=o_stds, capsize=3,
           label="Oracle (all labels)", color="C2", alpha=0.8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(short_names)
    ax.set_ylabel("Downstream balanced accuracy")
    ax.set_title("20 Newsgroups: filtering improves downstream classification")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    save_plot(fig, "newsgroups_downstream")

    # Precision/recall plot
    fig, ax = plt.subplots(figsize=(9, 5))
    fn_means = [np.mean(all_results[t]["fn_rates"]) for t in topics]
    fn_stds = [np.std(all_results[t]["fn_rates"]) for t in topics]
    fp_means = [np.mean(all_results[t]["fp_rates"]) for t in topics]
    fp_stds = [np.std(all_results[t]["fp_rates"]) for t in topics]
    prec_means = [np.mean(all_results[t]["precisions"]) for t in topics]

    ax.bar(x_pos - width/2, fn_means, width, yerr=fn_stds, capsize=4,
           label="False negative rate", color="C3", alpha=0.8)
    ax.bar(x_pos + width/2, fp_means, width, yerr=fp_stds, capsize=4,
           label="False positive rate", color="C2", alpha=0.8)
    for i, p in enumerate(prec_means):
        ax.annotate(f"prec={p:.2f}", (x_pos[i], max(fn_means[i], fp_means[i]) + 0.02),
                    ha="center", fontsize=9)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(short_names)
    ax.set_ylabel("Error rate")
    ax.set_title("20 Newsgroups: filter FP/FN rates across topics")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    save_plot(fig, "newsgroups_error_rates")


# ── 20 Newsgroups: vary N_S ─────────────────────────────────────────────────

def experiment_newsgroups_vary_NS(n_trials=10):
    """Vary N_S on 20 Newsgroups to see the sample complexity curve."""
    print("=" * 60)
    print("Real-world: 20 Newsgroups — varying N_S")
    print("=" * 60)

    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer

    data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    vectorizer = TfidfVectorizer(max_features=10000)
    X_all = vectorizer.fit_transform(data.data).toarray()
    y_all = data.target

    target_idx = 15  # sci.space

    N_S_values = [20, 50, 100, 200, 400]
    results = {k: [] for k in [
        "fn_mean", "fn_std", "fp_mean", "fp_std", "prec_mean",
        "acc_s_mean", "acc_s_std", "acc_f_mean", "acc_f_std",
        "acc_r_mean", "acc_r_std"
    ]}

    for N_S in N_S_values:
        fns, fps, precs, accs_s, accs_f, accs_r = [], [], [], [], [], []

        for trial in range(n_trials):
            rng = np.random.default_rng(200 * trial + N_S)

            train_idx, test_idx = train_test_split(
                np.arange(len(y_all)), test_size=0.3, random_state=trial,
                stratify=y_all)

            X_train_pool = X_all[train_idx]
            y_train_pool = y_all[train_idx]
            X_test = X_all[test_idx]
            y_test = (y_all[test_idx] == target_idx).astype(int)

            is_target_train = (y_train_pool == target_idx)
            target_indices = np.where(is_target_train)[0]

            if N_S > len(target_indices):
                continue

            s_idx = rng.choice(target_indices, N_S, replace=False)
            x_S = X_train_pool[s_idx]
            x_B = X_train_pool
            labels_B = is_target_train.astype(int)

            passed, _ = run_filter(x_S, x_B)
            m = compute_metrics(passed, labels_B)
            fns.append(m["fn_rate"])
            fps.append(m["fp_rate"])
            precs.append(m["precision"])

            # Downstream
            ds = run_downstream(x_S, x_B, passed, X_train_pool, is_target_train,
                                X_test, y_test, rng, N_S)
            accs_s.append(ds["acc_s"])
            accs_f.append(ds["acc_f"])
            accs_r.append(ds["acc_r"])

        results["fn_mean"].append(np.mean(fns))
        results["fn_std"].append(np.std(fns))
        results["fp_mean"].append(np.mean(fps))
        results["fp_std"].append(np.std(fps))
        results["prec_mean"].append(np.mean(precs))
        results["acc_s_mean"].append(np.mean(accs_s))
        results["acc_s_std"].append(np.std(accs_s))
        results["acc_f_mean"].append(np.mean(accs_f))
        results["acc_f_std"].append(np.std(accs_f))
        results["acc_r_mean"].append(np.mean(accs_r))
        results["acc_r_std"].append(np.std(accs_r))
        print(f"  N_S={N_S:4d}: FN={np.mean(fns):.4f}, FP={np.mean(fps):.4f}, "
              f"S-acc={np.mean(accs_s):.4f}, Filt-acc={np.mean(accs_f):.4f}")

    data_out = {"N_S_values": N_S_values, "target": "sci.space",
                "n_trials": n_trials, **results}
    save_data("newsgroups_vary_NS", data_out)

    # Downstream accuracy
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(N_S_values, results["acc_s_mean"], yerr=results["acc_s_std"],
                fmt="s-", capsize=4, label=r"$S$ samples only", color="C3")
    ax.errorbar(N_S_values, results["acc_f_mean"], yerr=results["acc_f_std"],
                fmt="o-", capsize=4, label=r"$S$ + filtered $B$", color="C0")
    ax.errorbar(N_S_values, results["acc_r_mean"], yerr=results["acc_r_std"],
                fmt="^-", capsize=4, label=r"$S$ + random $B$", color="C1")
    ax.set_xscale("log")
    ax.set_xlabel(r"$N_S$ (number of target samples)")
    ax.set_ylabel("Downstream balanced accuracy")
    ax.set_title("20 Newsgroups (sci.space): accuracy vs $N_S$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot(fig, "newsgroups_vary_NS_downstream")

    # Error rates
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(N_S_values, results["fn_mean"], yerr=results["fn_std"],
                fmt="^-", capsize=4, label="False negative rate", color="C3")
    ax.errorbar(N_S_values, results["fp_mean"], yerr=results["fp_std"],
                fmt="s-", capsize=4, label="False positive rate", color="C2")
    ax.set_xscale("log")
    ax.set_xlabel(r"$N_S$")
    ax.set_ylabel("Error rate")
    ax.set_title("20 Newsgroups (sci.space): filter errors vs $N_S$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot(fig, "newsgroups_vary_NS_errors")


# ── MNIST ────────────────────────────────────────────────────────────────────

def experiment_mnist(n_trials=10):
    """
    MNIST: one digit is S, the full dataset is B.
    We use PCA to reduce dimensionality for speed.
    """
    print("=" * 60)
    print("Real-world: MNIST")
    print("=" * 60)

    from sklearn.datasets import fetch_openml
    from sklearn.decomposition import PCA

    print("  Loading MNIST...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X_raw = mnist.data.astype(np.float32) / 255.0
    y_raw = mnist.target.astype(int)

    # PCA to 50 dimensions for speed
    pca = PCA(n_components=50, random_state=0)
    X_all = pca.fit_transform(X_raw)

    # Try several target digits
    target_digits = [3, 7, 1, 9]

    all_results = {}

    for target_digit in target_digits:
        print(f"\n  Target digit: {target_digit}")

        is_target = (y_raw == target_digit)
        p_true = is_target.mean()
        print(f"    p = {p_true:.4f}")

        results = {"fn_rates": [], "fp_rates": [], "precisions": [],
                   "acc_s_only": [], "acc_filtered": [], "acc_random": [],
                   "acc_oracle": [], "n_passed": []}

        for trial in range(n_trials):
            rng = np.random.default_rng(300 * trial + target_digit)

            train_idx, test_idx = train_test_split(
                np.arange(len(y_raw)), test_size=0.2, random_state=trial,
                stratify=y_raw)

            X_train = X_all[train_idx]
            y_train = y_raw[train_idx]
            X_test = X_all[test_idx]
            y_test = (y_raw[test_idx] == target_digit).astype(int)

            is_target_train = (y_train == target_digit)
            target_indices = np.where(is_target_train)[0]

            N_S = 200
            s_idx = rng.choice(target_indices, N_S, replace=False)
            x_S = X_train[s_idx]
            x_B = X_train
            labels_B = is_target_train.astype(int)

            passed, _ = run_filter(x_S, x_B)
            m = compute_metrics(passed, labels_B)
            results["fn_rates"].append(m["fn_rate"])
            results["fp_rates"].append(m["fp_rate"])
            results["precisions"].append(m["precision"])
            results["n_passed"].append(m["n_passed"])

            # Downstream
            ds = run_downstream(x_S, x_B, passed, X_train, is_target_train,
                                X_test, y_test, rng, N_S)
            results["acc_s_only"].append(ds["acc_s"])
            results["acc_filtered"].append(ds["acc_f"])
            results["acc_random"].append(ds["acc_r"])
            results["acc_oracle"].append(ds["acc_o"])

        print(f"    FN: {np.mean(results['fn_rates']):.4f}, "
              f"FP: {np.mean(results['fp_rates']):.4f}, "
              f"Prec: {np.mean(results['precisions']):.4f}")
        print(f"    S-only acc: {np.mean(results['acc_s_only']):.4f}, "
              f"Filtered acc: {np.mean(results['acc_filtered']):.4f}, "
              f"Random B: {np.mean(results['acc_random']):.4f}, "
              f"Oracle: {np.mean(results['acc_oracle']):.4f}")

        all_results[str(target_digit)] = {
            k: [float(x) for x in v] for k, v in results.items()
        }

    save_data("mnist", {"results": all_results,
                         "digits": target_digits, "n_trials": n_trials,
                         "N_S": 200, "pca_dims": 50})

    # Bar chart
    digits = [str(d) for d in target_digits]
    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(len(digits))
    width = 0.2

    s_means = [np.mean(all_results[d]["acc_s_only"]) for d in digits]
    s_stds = [np.std(all_results[d]["acc_s_only"]) for d in digits]
    f_means = [np.mean(all_results[d]["acc_filtered"]) for d in digits]
    f_stds = [np.std(all_results[d]["acc_filtered"]) for d in digits]
    r_means = [np.mean(all_results[d]["acc_random"]) for d in digits]
    r_stds = [np.std(all_results[d]["acc_random"]) for d in digits]
    o_means = [np.mean(all_results[d]["acc_oracle"]) for d in digits]
    o_stds = [np.std(all_results[d]["acc_oracle"]) for d in digits]

    ax.bar(x_pos - 1.5*width, s_means, width, yerr=s_stds, capsize=3,
           label=r"$S$ only ($N_S=200$)", color="C3", alpha=0.8)
    ax.bar(x_pos - 0.5*width, f_means, width, yerr=f_stds, capsize=3,
           label=r"$S$ + filtered $B$", color="C0", alpha=0.8)
    ax.bar(x_pos + 0.5*width, r_means, width, yerr=r_stds, capsize=3,
           label=r"$S$ + random $B$", color="C1", alpha=0.8)
    ax.bar(x_pos + 1.5*width, o_means, width, yerr=o_stds, capsize=3,
           label="Oracle (all labels)", color="C2", alpha=0.8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"Digit {d}" for d in target_digits])
    ax.set_ylabel("Downstream balanced accuracy")
    ax.set_title("MNIST: filtering improves downstream classification ($N_S=200$)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    save_plot(fig, "mnist_downstream")

    # Error rates
    fig, ax = plt.subplots(figsize=(9, 5))
    fn_means = [np.mean(all_results[d]["fn_rates"]) for d in digits]
    fp_means = [np.mean(all_results[d]["fp_rates"]) for d in digits]
    prec_means = [np.mean(all_results[d]["precisions"]) for d in digits]

    ax.bar(x_pos - width/2, fn_means, width, label="False negative rate", color="C3", alpha=0.8)
    ax.bar(x_pos + width/2, fp_means, width, label="False positive rate", color="C2", alpha=0.8)
    for i, p in enumerate(prec_means):
        ax.annotate(f"prec={p:.2f}", (x_pos[i], max(fn_means[i], fp_means[i]) + 0.005),
                    ha="center", fontsize=9)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"Digit {d}" for d in target_digits])
    ax.set_ylabel("Error rate")
    ax.set_title("MNIST: filter error rates ($N_S=200$)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    save_plot(fig, "mnist_error_rates")


# ── MNIST: vary N_S ─────────────────────────────────────────────────────────

def experiment_mnist_vary_NS(n_trials=10):
    """Vary N_S on MNIST digit 3."""
    print("=" * 60)
    print("Real-world: MNIST — varying N_S (digit 3)")
    print("=" * 60)

    from sklearn.datasets import fetch_openml
    from sklearn.decomposition import PCA

    print("  Loading MNIST...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X_raw = mnist.data.astype(np.float32) / 255.0
    y_raw = mnist.target.astype(int)

    pca = PCA(n_components=50, random_state=0)
    X_all = pca.fit_transform(X_raw)

    target_digit = 3
    N_S_values = [10, 20, 50, 100, 200, 500, 1000]
    results = {k: [] for k in [
        "fn_mean", "fn_std", "fp_mean", "fp_std",
        "acc_s_mean", "acc_s_std", "acc_f_mean", "acc_f_std",
        "acc_r_mean", "acc_r_std"
    ]}

    for N_S in N_S_values:
        fns, fps, accs_s, accs_f, accs_r = [], [], [], [], []

        for trial in range(n_trials):
            rng = np.random.default_rng(400 * trial + N_S)

            train_idx, test_idx = train_test_split(
                np.arange(len(y_raw)), test_size=0.2, random_state=trial,
                stratify=y_raw)

            X_train = X_all[train_idx]
            y_train = y_raw[train_idx]
            X_test = X_all[test_idx]
            y_test = (y_raw[test_idx] == target_digit).astype(int)

            is_target_train = (y_train == target_digit)
            target_indices = np.where(is_target_train)[0]
            if N_S > len(target_indices):
                continue

            s_idx = rng.choice(target_indices, N_S, replace=False)
            x_S = X_train[s_idx]
            x_B = X_train
            labels_B = is_target_train.astype(int)

            passed, _ = run_filter(x_S, x_B)
            m = compute_metrics(passed, labels_B)
            fns.append(m["fn_rate"])
            fps.append(m["fp_rate"])

            # Downstream
            ds = run_downstream(x_S, x_B, passed, X_train, is_target_train,
                                X_test, y_test, rng, N_S)
            accs_s.append(ds["acc_s"])
            accs_f.append(ds["acc_f"])
            accs_r.append(ds["acc_r"])

        results["fn_mean"].append(np.mean(fns))
        results["fn_std"].append(np.std(fns))
        results["fp_mean"].append(np.mean(fps))
        results["fp_std"].append(np.std(fps))
        results["acc_s_mean"].append(np.mean(accs_s))
        results["acc_s_std"].append(np.std(accs_s))
        results["acc_f_mean"].append(np.mean(accs_f))
        results["acc_f_std"].append(np.std(accs_f))
        results["acc_r_mean"].append(np.mean(accs_r))
        results["acc_r_std"].append(np.std(accs_r))
        print(f"  N_S={N_S:5d}: FN={np.mean(fns):.4f}, "
              f"S-acc={np.mean(accs_s):.4f}, Filt-acc={np.mean(accs_f):.4f}")

    data_out = {"N_S_values": N_S_values, "target_digit": target_digit,
                "n_trials": n_trials, **results}
    save_data("mnist_vary_NS", data_out)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(N_S_values, results["acc_s_mean"], yerr=results["acc_s_std"],
                fmt="s-", capsize=4, label=r"$S$ samples only", color="C3")
    ax.errorbar(N_S_values, results["acc_f_mean"], yerr=results["acc_f_std"],
                fmt="o-", capsize=4, label=r"$S$ + filtered $B$", color="C0")
    ax.errorbar(N_S_values, results["acc_r_mean"], yerr=results["acc_r_std"],
                fmt="^-", capsize=4, label=r"$S$ + random $B$", color="C1")
    ax.set_xscale("log")
    ax.set_xlabel(r"$N_S$ (number of target samples)")
    ax.set_ylabel("Downstream balanced accuracy")
    ax.set_title("MNIST (digit 3): accuracy vs $N_S$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot(fig, "mnist_vary_NS_downstream")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(N_S_values, results["fn_mean"], yerr=results["fn_std"],
                fmt="^-", capsize=4, label="False negative rate", color="C3")
    ax.errorbar(N_S_values, results["fp_mean"], yerr=results["fp_std"],
                fmt="s-", capsize=4, label="False positive rate", color="C2")
    ax.set_xscale("log")
    ax.set_xlabel(r"$N_S$")
    ax.set_ylabel("Error rate")
    ax.set_title("MNIST (digit 3): filter errors vs $N_S$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot(fig, "mnist_vary_NS_errors")


# ── Covertype ────────────────────────────────────────────────────────────────

def experiment_covertype(n_trials=10):
    """
    Covertype: rare forest cover type as S.
    Already tabular with 54 features -- no embedding needed.
    """
    print("=" * 60)
    print("Real-world: Covertype (rare class detection)")
    print("=" * 60)

    from sklearn.datasets import fetch_covtype
    from sklearn.preprocessing import StandardScaler

    print("  Loading Covertype...")
    X_raw, y_raw = fetch_covtype(return_X_y=True)

    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_raw)

    # Class 4 (Cottonwood/Willow) is rare: ~2,747 of 581,012 => p ≈ 0.0047
    # Class 5 (Aspen) is also smallish: ~9,493 => p ≈ 0.016
    target_classes = [4, 5]

    all_results = {}

    for target_class in target_classes:
        is_target = (y_raw == target_class)
        n_target = is_target.sum()
        p_true = n_target / len(y_raw)
        print(f"\n  Target class {target_class}: n={n_target}, p={p_true:.4f}")

        # Subsample B for speed (581k is a lot)
        N_B_use = 100_000

        results = {"fn_rates": [], "fp_rates": [], "precisions": [],
                   "acc_s_only": [], "acc_filtered": [], "acc_random": [],
                   "acc_oracle": [], "n_passed": []}

        for trial in range(n_trials):
            rng = np.random.default_rng(500 * trial + target_class)

            # Subsample
            idx = rng.choice(len(y_raw), N_B_use, replace=False)
            X_sub = X_all[idx]
            y_sub = y_raw[idx]

            # Train/test split
            train_idx, test_idx = train_test_split(
                np.arange(len(y_sub)), test_size=0.3, random_state=trial,
                stratify=y_sub)

            X_train = X_sub[train_idx]
            y_train = y_sub[train_idx]
            X_test = X_sub[test_idx]
            y_test = (y_sub[test_idx] == target_class).astype(int)

            is_target_train = (y_train == target_class)
            target_indices = np.where(is_target_train)[0]

            N_S = min(200, len(target_indices))
            if N_S < 20:
                continue

            s_idx = rng.choice(target_indices, N_S, replace=False)
            x_S = X_train[s_idx]
            x_B = X_train
            labels_B = is_target_train.astype(int)

            passed, _ = run_filter(x_S, x_B)
            m = compute_metrics(passed, labels_B)
            results["fn_rates"].append(m["fn_rate"])
            results["fp_rates"].append(m["fp_rate"])
            results["precisions"].append(m["precision"])
            results["n_passed"].append(m["n_passed"])

            # Downstream
            ds = run_downstream(x_S, x_B, passed, X_train, is_target_train,
                                X_test, y_test, rng, N_S)
            results["acc_s_only"].append(ds["acc_s"])
            results["acc_filtered"].append(ds["acc_f"])
            results["acc_random"].append(ds["acc_r"])
            results["acc_oracle"].append(ds["acc_o"])

        print(f"    FN: {np.mean(results['fn_rates']):.4f}, "
              f"FP: {np.mean(results['fp_rates']):.4f}, "
              f"Prec: {np.mean(results['precisions']):.4f}")
        print(f"    S-only acc: {np.mean(results['acc_s_only']):.4f}, "
              f"Filtered: {np.mean(results['acc_filtered']):.4f}, "
              f"Random B: {np.mean(results['acc_random']):.4f}, "
              f"Oracle: {np.mean(results['acc_oracle']):.4f}")

        all_results[str(target_class)] = {
            k: [float(x) for x in v] for k, v in results.items()
        }

    save_data("covertype", {"results": all_results,
                              "classes": target_classes, "n_trials": n_trials,
                              "N_S": 200, "N_B_subsample": N_B_use})

    # Plot
    classes = [str(c) for c in target_classes]
    class_names = {4: "Cottonwood/Willow\n(p=0.005)", 5: "Aspen\n(p=0.016)"}
    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(classes))
    width = 0.2

    s_means = [np.mean(all_results[c]["acc_s_only"]) for c in classes]
    s_stds = [np.std(all_results[c]["acc_s_only"]) for c in classes]
    f_means = [np.mean(all_results[c]["acc_filtered"]) for c in classes]
    f_stds = [np.std(all_results[c]["acc_filtered"]) for c in classes]
    r_means = [np.mean(all_results[c]["acc_random"]) for c in classes]
    r_stds = [np.std(all_results[c]["acc_random"]) for c in classes]
    o_means = [np.mean(all_results[c]["acc_oracle"]) for c in classes]
    o_stds = [np.std(all_results[c]["acc_oracle"]) for c in classes]

    ax.bar(x_pos - 1.5*width, s_means, width, yerr=s_stds, capsize=3,
           label=r"$S$ only ($N_S=200$)", color="C3", alpha=0.8)
    ax.bar(x_pos - 0.5*width, f_means, width, yerr=f_stds, capsize=3,
           label=r"$S$ + filtered $B$", color="C0", alpha=0.8)
    ax.bar(x_pos + 0.5*width, r_means, width, yerr=r_stds, capsize=3,
           label=r"$S$ + random $B$", color="C1", alpha=0.8)
    ax.bar(x_pos + 1.5*width, o_means, width, yerr=o_stds, capsize=3,
           label="Oracle (all labels)", color="C2", alpha=0.8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([class_names[int(c)] for c in classes])
    ax.set_ylabel("Downstream balanced accuracy")
    ax.set_title("Covertype: rare class detection ($N_S=200$)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    save_plot(fig, "covertype_downstream")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.time()

    experiment_20newsgroups()
    experiment_newsgroups_vary_NS()
    experiment_mnist()
    experiment_mnist_vary_NS()
    experiment_covertype()

    elapsed = time.time() - t0
    print(f"\nAll real-world experiments completed in {elapsed:.1f}s")
    print(f"Plots saved to {PLOTS_DIR.resolve()}")
    print(f"Data saved to {DATA_DIR.resolve()}")
