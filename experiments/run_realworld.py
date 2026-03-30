"""
Real-world data experiments for "Dimension-Free Data Filtering".

We apply the filtering algorithm to real datasets where two related classes
form the target distribution S, and the full dataset serves as B.
The downstream task is to distinguish the two classes within S.
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


def run_filter(x_S, x_B, C=10.0, weight_scale=1.0, threshold=0.0):
    """Run the filter. Returns boolean mask of B samples that pass.

    Args:
        threshold: decision function threshold. Higher = more selective
                   (fewer FP, more FN, higher precision).
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
    passed = scores_B >= threshold
    return passed, clf


def run_downstream(x_S, y_S_ds, x_B, y_B_ds, passed,
                   x_S_all_train, y_S_all_train, X_test, y_test, rng, N_S):
    """
    Downstream task: classify class A vs class B within S.

    Args:
        x_S: N_S labeled samples from S = A ∪ B
        y_S_ds: true A/B labels for S samples (0 or 1)
        x_B: all B samples (the big pool)
        y_B_ds: downstream labels for all B: true for A/B points, random for O
        passed: boolean mask of which B samples pass the filter
        x_S_all_train: all true S samples in training pool (for oracle)
        y_S_all_train: true A/B labels for all S in training pool (for oracle)
        X_test: test features (only A and B samples)
        y_test: test labels (0 or 1)
        rng: random state
        N_S: number of S samples

    Returns dict with accuracies for S-only, filtered, random B, oracle.
    """
    def score(clf, X, y):
        return balanced_accuracy_score(y, clf.predict(X))

    # (a) S-only
    clf_s = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced')
    clf_s.fit(x_S, y_S_ds)
    acc_s = score(clf_s, X_test, y_test)

    # (b) Filtered: S + filtered B samples with their downstream labels
    x_filtered = x_B[passed]
    y_filtered = y_B_ds[passed]
    n_filt = len(x_filtered)
    max_filt = max(5 * N_S, 500)
    if n_filt > max_filt:
        filt_idx = rng.choice(n_filt, max_filt, replace=False)
        x_filtered = x_filtered[filt_idx]
        y_filtered = y_filtered[filt_idx]
        n_filt = max_filt
    X_f_train = np.vstack([x_S, x_filtered])
    y_f_train = np.concatenate([y_S_ds, y_filtered])
    clf_f = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced')
    clf_f.fit(X_f_train, y_f_train)
    acc_f = score(clf_f, X_test, y_test)

    # (c) Random B: same count of random (unfiltered) B samples
    n_rand = max(n_filt, 1)
    rand_idx = rng.choice(len(x_B), min(n_rand, len(x_B)), replace=False)
    x_rand = x_B[rand_idx]
    y_rand = y_B_ds[rand_idx]
    X_r_train = np.vstack([x_S, x_rand])
    y_r_train = np.concatenate([y_S_ds, y_rand])
    clf_r = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced')
    clf_r.fit(X_r_train, y_r_train)
    acc_r = score(clf_r, X_test, y_test)

    # (d) Oracle: all true S samples with true labels
    clf_o = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced')
    clf_o.fit(x_S_all_train, y_S_all_train)
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


def make_downstream_labels(y_pool, X_pool, class_a, class_b,
                           confound_a, confound_b):
    """Create downstream labels for all samples in a pool.

    class A → 1, class B → 0.
    O (everything else) → labels from a confounding classifier trained on
    (confound_a vs confound_b). This models a labeling oracle trained for
    a different task: O labels carry real signal about the confounding
    classification, which actively misleads the A-vs-B classifier.
    Returns labels array and boolean mask of which samples are in S = A ∪ B.
    """
    is_a = (y_pool == class_a)
    is_b = (y_pool == class_b)
    is_S = is_a | is_b

    # Train confounding classifier on a different pair of classes
    is_c = (y_pool == confound_a)
    is_d = (y_pool == confound_b)
    X_cd = np.vstack([X_pool[is_c], X_pool[is_d]])
    y_cd = np.array([1]*is_c.sum() + [0]*is_d.sum())
    clf_confound = LogisticRegression(max_iter=1000, C=1.0)
    clf_confound.fit(X_cd, y_cd)

    # O samples get labels from the confounding classifier
    y_ds = clf_confound.predict(X_pool).astype(int)

    # Override with true labels for A and B
    y_ds[is_a] = 1
    y_ds[is_b] = 0
    return y_ds, is_S


def plot_bar_chart(all_results, keys, labels, title, plot_name):
    """Generic bar chart comparing methods across settings."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(len(keys))
    width = 0.2

    for i, (metric, label, color) in enumerate([
        ("acc_s_only", r"$S$ only", "C3"),
        ("acc_filtered", r"$S$ + filtered $B$", "C0"),
        ("acc_random", r"$S$ + random $B$", "C1"),
        ("acc_oracle", "Oracle", "C2"),
    ]):
        means = [np.mean(all_results[k][metric]) for k in keys]
        stds = [np.std(all_results[k][metric]) for k in keys]
        ax.bar(x_pos + (i - 1.5)*width, means, width, yerr=stds, capsize=3,
               label=label, color=color, alpha=0.8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Downstream balanced accuracy")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    save_plot(fig, plot_name)


def plot_error_rates(all_results, keys, labels, title, plot_name):
    """Bar chart of FP/FN rates with precision annotations."""
    fig, ax = plt.subplots(figsize=(9, 5))
    x_pos = np.arange(len(keys))
    width = 0.25

    fn_means = [np.mean(all_results[k]["fn_rates"]) for k in keys]
    fn_stds = [np.std(all_results[k]["fn_rates"]) for k in keys]
    fp_means = [np.mean(all_results[k]["fp_rates"]) for k in keys]
    fp_stds = [np.std(all_results[k]["fp_rates"]) for k in keys]
    prec_means = [np.mean(all_results[k]["precisions"]) for k in keys]

    ax.bar(x_pos - width/2, fn_means, width, yerr=fn_stds, capsize=4,
           label="False negative rate", color="C3", alpha=0.8)
    ax.bar(x_pos + width/2, fp_means, width, yerr=fp_stds, capsize=4,
           label="False positive rate", color="C2", alpha=0.8)
    for i, p in enumerate(prec_means):
        ax.annotate(f"prec={p:.2f}",
                    (x_pos[i], max(fn_means[i], fp_means[i]) + 0.02),
                    ha="center", fontsize=9)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Error rate")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    save_plot(fig, plot_name)


# ── 20 Newsgroups ────────────────────────────────────────────────────────────

def experiment_20newsgroups(n_trials=10):
    """
    20 Newsgroups: S = two related topics, O = rest.
    Downstream: distinguish the two topics within S.
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

    # Pairs of related topics: (class_a, class_b, confound_a, confound_b, short_name)
    # Confounding pairs are from different topic families so their signal
    # actively misleads the A-vs-B classifier.
    topic_pairs = [
        (15, 18, 9, 10, "christianity vs politics.misc"),  # confound: baseball vs hockey
        (11, 12, 7, 8, "sci.crypt vs sci.electronics"),    # confound: rec.autos vs motorcycles
        (7, 8, 3, 4, "rec.autos vs rec.motorcycles"),      # confound: ibm.pc vs mac
        (3, 4, 16, 17, "ibm.pc vs mac"),                   # confound: guns vs mideast
    ]

    all_results = {}

    for class_a, class_b, confound_a, confound_b, pair_name in topic_pairs:
        name_a = data.target_names[class_a]
        name_b = data.target_names[class_b]
        print(f"\n  Pair: {name_a} vs {name_b}")
        print(f"    Confounding: {data.target_names[confound_a]} vs {data.target_names[confound_b]}")

        is_S = (y_all == class_a) | (y_all == class_b)
        p_true = is_S.mean()
        print(f"    Total S samples: {is_S.sum()}, p={p_true:.4f}")

        results = {"fn_rates": [], "fp_rates": [], "precisions": [],
                   "acc_s_only": [], "acc_filtered": [], "acc_random": [],
                   "acc_oracle": [], "n_passed": []}

        for trial in range(n_trials):
            rng = np.random.default_rng(100 * trial + class_a)

            train_idx, test_idx = train_test_split(
                np.arange(len(y_all)), test_size=0.3, random_state=trial,
                stratify=y_all)

            X_train_pool = X_all[train_idx]
            y_train_pool = y_all[train_idx]

            # Downstream labels: A→1, B→0, O→confounding classifier
            y_ds_train, is_S_train = make_downstream_labels(
                y_train_pool, X_train_pool, class_a, class_b,
                confound_a, confound_b)

            # Test set: only A and B samples
            test_is_S = (y_all[test_idx] == class_a) | (y_all[test_idx] == class_b)
            X_test = X_all[test_idx][test_is_S]
            y_test = (y_all[test_idx][test_is_S] == class_a).astype(int)

            # S: subsample S from training pool (equal from each class)
            a_indices = np.where(y_train_pool == class_a)[0]
            b_indices = np.where(y_train_pool == class_b)[0]
            N_S_per_class = min(100, len(a_indices), len(b_indices))
            N_S = 2 * N_S_per_class
            s_idx = np.concatenate([
                rng.choice(a_indices, N_S_per_class, replace=False),
                rng.choice(b_indices, N_S_per_class, replace=False),
            ])
            x_S = X_train_pool[s_idx]
            y_S_ds = y_ds_train[s_idx]

            # B: the full training pool
            x_B = X_train_pool
            labels_B = is_S_train.astype(int)  # for filter metrics

            # All true S samples in training (for oracle)
            s_all_idx = np.where(is_S_train)[0]
            x_S_all = X_train_pool[s_all_idx]
            y_S_all_ds = y_ds_train[s_all_idx]

            # Run filter
            passed, _ = run_filter(x_S, x_B)
            m = compute_metrics(passed, labels_B)
            results["fn_rates"].append(m["fn_rate"])
            results["fp_rates"].append(m["fp_rate"])
            results["precisions"].append(m["precision"])
            results["n_passed"].append(m["n_passed"])

            # Downstream
            ds = run_downstream(x_S, y_S_ds, x_B, y_ds_train, passed,
                                x_S_all, y_S_all_ds, X_test, y_test, rng, N_S)
            results["acc_s_only"].append(ds["acc_s"])
            results["acc_filtered"].append(ds["acc_f"])
            results["acc_random"].append(ds["acc_r"])
            results["acc_oracle"].append(ds["acc_o"])

        print(f"    FN rate: {np.mean(results['fn_rates']):.4f} "
              f"± {np.std(results['fn_rates']):.4f}")
        print(f"    FP rate: {np.mean(results['fp_rates']):.4f} "
              f"± {np.std(results['fp_rates']):.4f}")
        print(f"    Precision: {np.mean(results['precisions']):.4f}")
        print(f"    Downstream acc (S-only):   {np.mean(results['acc_s_only']):.4f}")
        print(f"    Downstream acc (filtered): {np.mean(results['acc_filtered']):.4f}")
        print(f"    Downstream acc (random B): {np.mean(results['acc_random']):.4f}")
        print(f"    Downstream acc (oracle):   {np.mean(results['acc_oracle']):.4f}")

        all_results[pair_name] = {
            k: [float(x) for x in v] for k, v in results.items()
        }

    pair_names = [p[4] for p in topic_pairs]
    save_data("newsgroups", {"results": all_results,
                              "pairs": pair_names,
                              "n_trials": n_trials, "N_S": 200})

    plot_bar_chart(all_results, pair_names, pair_names,
                   "20 Newsgroups: filtering improves within-S classification",
                   "newsgroups_downstream")
    plot_error_rates(all_results, pair_names, pair_names,
                     "20 Newsgroups: filter FP/FN rates",
                     "newsgroups_error_rates")


# ── 20 Newsgroups: vary N_S ─────────────────────────────────────────────────

def experiment_newsgroups_vary_NS(n_trials=10):
    """Vary N_S on 20 Newsgroups (sci.crypt vs sci.electronics)."""
    print("=" * 60)
    print("Real-world: 20 Newsgroups — varying N_S (sci.crypt vs sci.electronics)")
    print("=" * 60)

    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer

    data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    vectorizer = TfidfVectorizer(max_features=10000)
    X_all = vectorizer.fit_transform(data.data).toarray()
    y_all = data.target

    class_a, class_b = 11, 12  # sci.crypt vs sci.electronics
    confound_a, confound_b = 7, 8  # rec.autos vs rec.motorcycles

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

            y_ds_train, is_S_train = make_downstream_labels(
                y_train_pool, X_train_pool, class_a, class_b,
                confound_a, confound_b)

            test_is_S = (y_all[test_idx] == class_a) | (y_all[test_idx] == class_b)
            X_test = X_all[test_idx][test_is_S]
            y_test = (y_all[test_idx][test_is_S] == class_a).astype(int)

            a_indices = np.where(y_train_pool == class_a)[0]
            b_indices = np.where(y_train_pool == class_b)[0]
            N_S_per_class = N_S // 2
            if N_S_per_class > min(len(a_indices), len(b_indices)):
                continue

            s_idx = np.concatenate([
                rng.choice(a_indices, N_S_per_class, replace=False),
                rng.choice(b_indices, N_S_per_class, replace=False),
            ])
            x_S = X_train_pool[s_idx]
            y_S_ds = y_ds_train[s_idx]

            x_B = X_train_pool
            labels_B = is_S_train.astype(int)

            s_all_idx = np.where(is_S_train)[0]
            x_S_all = X_train_pool[s_all_idx]
            y_S_all_ds = y_ds_train[s_all_idx]

            passed, _ = run_filter(x_S, x_B)
            m = compute_metrics(passed, labels_B)
            fns.append(m["fn_rate"])
            fps.append(m["fp_rate"])
            precs.append(m["precision"])

            ds = run_downstream(x_S, y_S_ds, x_B, y_ds_train, passed,
                                x_S_all, y_S_all_ds, X_test, y_test, rng, N_S)
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

    data_out = {"N_S_values": N_S_values,
                "pair": "sci.crypt vs sci.electronics",
                "n_trials": n_trials, **results}
    save_data("newsgroups_vary_NS", data_out)

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
    ax.set_title("20 Newsgroups (sci.crypt vs sci.electronics): accuracy vs $N_S$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot(fig, "newsgroups_vary_NS_downstream")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(N_S_values, results["fn_mean"], yerr=results["fn_std"],
                fmt="^-", capsize=4, label="False negative rate", color="C3")
    ax.errorbar(N_S_values, results["fp_mean"], yerr=results["fp_std"],
                fmt="s-", capsize=4, label="False positive rate", color="C2")
    ax.set_xscale("log")
    ax.set_xlabel(r"$N_S$")
    ax.set_ylabel("Error rate")
    ax.set_title("20 Newsgroups (sci.crypt vs sci.electronics): filter errors vs $N_S$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot(fig, "newsgroups_vary_NS_errors")


# ── MNIST ────────────────────────────────────────────────────────────────────

def experiment_mnist(n_trials=10):
    """
    MNIST: S = two similar digits, O = rest.
    Downstream: distinguish the two digits within S.
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

    pca = PCA(n_components=50, random_state=0)
    X_all = pca.fit_transform(X_raw)

    # Pairs of similar digits: (digit_a, digit_b, confound_a, confound_b, name)
    digit_pairs = [
        (7, 9, 3, 8, "7 vs 9"),   # confound with 3 vs 8
        (3, 5, 4, 9, "3 vs 5"),   # confound with 4 vs 9
        (4, 9, 3, 8, "4 vs 9"),   # confound with 3 vs 8
    ]

    all_results = {}

    for digit_a, digit_b, confound_a, confound_b, pair_name in digit_pairs:
        print(f"\n  Pair: {pair_name} (confound: {confound_a} vs {confound_b})")

        is_S = (y_raw == digit_a) | (y_raw == digit_b)
        p_true = is_S.mean()
        print(f"    p = {p_true:.4f}")

        results = {"fn_rates": [], "fp_rates": [], "precisions": [],
                   "acc_s_only": [], "acc_filtered": [], "acc_random": [],
                   "acc_oracle": [], "n_passed": []}

        for trial in range(n_trials):
            rng = np.random.default_rng(300 * trial + digit_a)

            train_idx, test_idx = train_test_split(
                np.arange(len(y_raw)), test_size=0.2, random_state=trial,
                stratify=y_raw)

            X_train = X_all[train_idx]
            y_train = y_raw[train_idx]

            y_ds_train, is_S_train = make_downstream_labels(
                y_train, X_train, digit_a, digit_b,
                confound_a, confound_b)

            test_is_S = (y_raw[test_idx] == digit_a) | (y_raw[test_idx] == digit_b)
            X_test = X_all[test_idx][test_is_S]
            y_test = (y_raw[test_idx][test_is_S] == digit_a).astype(int)

            a_indices = np.where(y_train == digit_a)[0]
            b_indices = np.where(y_train == digit_b)[0]
            N_S_per_class = min(100, len(a_indices), len(b_indices))
            N_S = 2 * N_S_per_class
            s_idx = np.concatenate([
                rng.choice(a_indices, N_S_per_class, replace=False),
                rng.choice(b_indices, N_S_per_class, replace=False),
            ])
            x_S = X_train[s_idx]
            y_S_ds = y_ds_train[s_idx]

            x_B = X_train
            labels_B = is_S_train.astype(int)

            s_all_idx = np.where(is_S_train)[0]
            x_S_all = X_train[s_all_idx]
            y_S_all_ds = y_ds_train[s_all_idx]

            passed, _ = run_filter(x_S, x_B)
            m = compute_metrics(passed, labels_B)
            results["fn_rates"].append(m["fn_rate"])
            results["fp_rates"].append(m["fp_rate"])
            results["precisions"].append(m["precision"])
            results["n_passed"].append(m["n_passed"])

            ds = run_downstream(x_S, y_S_ds, x_B, y_ds_train, passed,
                                x_S_all, y_S_all_ds, X_test, y_test, rng, N_S)
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

        all_results[pair_name] = {
            k: [float(x) for x in v] for k, v in results.items()
        }

    pair_names = [p[4] for p in digit_pairs]
    save_data("mnist", {"results": all_results,
                         "pairs": pair_names, "n_trials": n_trials,
                         "N_S": 200, "pca_dims": 50})

    plot_bar_chart(all_results, pair_names, pair_names,
                   "MNIST: filtering improves within-S digit classification ($N_S=200$)",
                   "mnist_downstream")
    plot_error_rates(all_results, pair_names, pair_names,
                     "MNIST: filter error rates ($N_S=200$)",
                     "mnist_error_rates")


# ── MNIST: vary N_S ─────────────────────────────────────────────────────────

def experiment_mnist_vary_NS(n_trials=10):
    """Vary N_S on MNIST (7 vs 9)."""
    print("=" * 60)
    print("Real-world: MNIST — varying N_S (7 vs 9)")
    print("=" * 60)

    from sklearn.datasets import fetch_openml
    from sklearn.decomposition import PCA

    print("  Loading MNIST...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X_raw = mnist.data.astype(np.float32) / 255.0
    y_raw = mnist.target.astype(int)

    pca = PCA(n_components=50, random_state=0)
    X_all = pca.fit_transform(X_raw)

    digit_a, digit_b = 7, 9
    confound_a, confound_b = 3, 8  # confounding pair
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

            y_ds_train, is_S_train = make_downstream_labels(
                y_train, X_train, digit_a, digit_b,
                confound_a, confound_b)

            test_is_S = (y_raw[test_idx] == digit_a) | (y_raw[test_idx] == digit_b)
            X_test = X_all[test_idx][test_is_S]
            y_test = (y_raw[test_idx][test_is_S] == digit_a).astype(int)

            a_indices = np.where(y_train == digit_a)[0]
            b_indices = np.where(y_train == digit_b)[0]
            N_S_per_class = N_S // 2
            if N_S_per_class > min(len(a_indices), len(b_indices)):
                continue

            s_idx = np.concatenate([
                rng.choice(a_indices, N_S_per_class, replace=False),
                rng.choice(b_indices, N_S_per_class, replace=False),
            ])
            x_S = X_train[s_idx]
            y_S_ds = y_ds_train[s_idx]

            x_B = X_train
            labels_B = is_S_train.astype(int)

            s_all_idx = np.where(is_S_train)[0]
            x_S_all = X_train[s_all_idx]
            y_S_all_ds = y_ds_train[s_all_idx]

            passed, _ = run_filter(x_S, x_B)
            m = compute_metrics(passed, labels_B)
            fns.append(m["fn_rate"])
            fps.append(m["fp_rate"])

            ds = run_downstream(x_S, y_S_ds, x_B, y_ds_train, passed,
                                x_S_all, y_S_all_ds, X_test, y_test, rng, N_S)
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

    data_out = {"N_S_values": N_S_values,
                "pair": "7 vs 9",
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
    ax.set_title("MNIST (7 vs 9): accuracy vs $N_S$")
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
    ax.set_title("MNIST (7 vs 9): filter errors vs $N_S$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot(fig, "mnist_vary_NS_errors")


# ── Covertype ────────────────────────────────────────────────────────────────

def experiment_covertype(n_trials=10):
    """
    Covertype: S = two forest types, O = rest.
    Downstream: distinguish the two types within S.
    """
    print("=" * 60)
    print("Real-world: Covertype")
    print("=" * 60)

    from sklearn.datasets import fetch_covtype
    from sklearn.preprocessing import StandardScaler

    print("  Loading Covertype...")
    X_raw, y_raw = fetch_covtype(return_X_y=True)

    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_raw)

    # Pairs of forest types: (class_a, class_b, confound_a, confound_b, name)
    class_pairs = [
        (1, 2, 3, 6, "Spruce/Fir vs Lodgepole Pine"),    # confound: Ponderosa vs Douglas-fir
        (3, 6, 1, 2, "Ponderosa vs Douglas-fir"),         # confound: Spruce/Fir vs Lodgepole Pine
    ]

    N_B_use = 100_000
    all_results = {}

    for class_a, class_b, confound_a, confound_b, pair_name in class_pairs:
        is_S = (y_raw == class_a) | (y_raw == class_b)
        p_true = is_S.mean()
        print(f"\n  Pair: {pair_name} (p={p_true:.4f})")

        results = {"fn_rates": [], "fp_rates": [], "precisions": [],
                   "acc_s_only": [], "acc_filtered": [], "acc_random": [],
                   "acc_oracle": [], "n_passed": []}

        for trial in range(n_trials):
            rng = np.random.default_rng(500 * trial + class_a)

            idx = rng.choice(len(y_raw), N_B_use, replace=False)
            X_sub = X_all[idx]
            y_sub = y_raw[idx]

            train_idx, test_idx = train_test_split(
                np.arange(len(y_sub)), test_size=0.3, random_state=trial,
                stratify=y_sub)

            X_train = X_sub[train_idx]
            y_train = y_sub[train_idx]

            y_ds_train, is_S_train = make_downstream_labels(
                y_train, X_train, class_a, class_b,
                confound_a, confound_b)

            test_is_S = (y_sub[test_idx] == class_a) | (y_sub[test_idx] == class_b)
            X_test = X_sub[test_idx][test_is_S]
            y_test = (y_sub[test_idx][test_is_S] == class_a).astype(int)

            if len(X_test) < 10:
                continue

            a_indices = np.where(y_train == class_a)[0]
            b_indices = np.where(y_train == class_b)[0]
            N_S_per_class = min(100, len(a_indices), len(b_indices))
            N_S = 2 * N_S_per_class
            if N_S < 20:
                continue

            s_idx = np.concatenate([
                rng.choice(a_indices, N_S_per_class, replace=False),
                rng.choice(b_indices, N_S_per_class, replace=False),
            ])
            x_S = X_train[s_idx]
            y_S_ds = y_ds_train[s_idx]

            x_B = X_train
            labels_B = is_S_train.astype(int)

            s_all_idx = np.where(is_S_train)[0]
            x_S_all = X_train[s_all_idx]
            y_S_all_ds = y_ds_train[s_all_idx]

            passed, _ = run_filter(x_S, x_B)
            m = compute_metrics(passed, labels_B)
            results["fn_rates"].append(m["fn_rate"])
            results["fp_rates"].append(m["fp_rate"])
            results["precisions"].append(m["precision"])
            results["n_passed"].append(m["n_passed"])

            ds = run_downstream(x_S, y_S_ds, x_B, y_ds_train, passed,
                                x_S_all, y_S_all_ds, X_test, y_test, rng, N_S)
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

        all_results[pair_name] = {
            k: [float(x) for x in v] for k, v in results.items()
        }

    pair_names = [p[4] for p in class_pairs]
    save_data("covertype", {"results": all_results,
                              "pairs": pair_names, "n_trials": n_trials,
                              "N_S": 200, "N_B_subsample": N_B_use})

    # Compute approximate p for labels
    plot_bar_chart(all_results, pair_names,
                   ["Spruce/Fir vs\nLodgepole Pine\n(p=0.85)",
                    "Ponderosa vs\nDouglas-fir\n(p=0.09)"],
                   "Covertype: filtering improves within-S classification ($N_S=200$)",
                   "covertype_downstream")


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
