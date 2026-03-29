"""
Synthetic experiments for "Dimension-Free Data Filtering".

We implement the paper's filtering algorithm using SVM as a practical proxy
for the constrained optimization problem, and validate the theoretical predictions:
  - TV distance scales as ~1/N_S (Theorem 1)
  - TV distance scales as ~1/(p*N_B) (Theorem 1)
  - TV distance depends on R^2/gamma^2 but NOT on dimension d
  - Weak separation: graceful degradation with imperfect margins
  - Downstream classification improves with filtered data augmentation
"""

import json
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import time

PLOTS_DIR = Path(__file__).parent / "plots"
DATA_DIR = Path(__file__).parent / "data"
PLOTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)


def save_data(name, data):
    """Save experiment data as JSON."""
    # Convert numpy types to native Python for JSON serialization
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
        if isinstance(v, list):
            converted[k] = [convert(x) for x in v]
        else:
            converted[k] = convert(v)

    with open(DATA_DIR / f"{name}.json", "w") as f:
        json.dump(converted, f, indent=2)


def save_plot(fig, name):
    """Save a figure as both PDF and PNG."""
    fig.savefig(PLOTS_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(PLOTS_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {name}.pdf")


# ── Data generation ──────────────────────────────────────────────────────────

def generate_data(d, N_S, N_B, p, gamma, R=1.0, rng=None,
                  eps_S=0.0, eps_O=0.0, label_shift=1.0):
    """
    Generate synthetic data for the filtering problem.

    S and O are distributions in R^d separated by margin gamma along e_1.
    We generate the "noise" coordinates with variance scaled by R/sqrt(d)
    so that total norm stays ~R regardless of dimension, preserving the
    effective margin gamma.

    With weak separation (eps_S, eps_O > 0), a fraction eps_S of S points
    and eps_O of O points violate the margin.

    label_shift: S points get x[1] += label_shift * noise_scale,
    making sign(x[1]) informative for S. O points get NO shift,
    so sign(x[1]) is pure noise (50/50) for O. This means
    unfiltered O data actively dilutes the downstream classifier's
    training signal.

    B = p * S + (1-p) * O.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    def _make_samples(n, side, eps_violate):
        """Generate n samples on given side (+1 for S, -1 for O)."""
        X = np.zeros((n, d))
        X[:, 0] = side * (np.abs(rng.standard_normal(n)) * 0.5 + gamma + 0.05)
        if d > 1:
            noise_scale = R / max(np.sqrt(d), 1.0) * 0.5
            X[:, 1:] = rng.standard_normal((n, d - 1)) * noise_scale
            # Shift x[1] for S only so sign(x[1]) is informative for S
            # but pure noise for O.  Scale with noise_scale for d-stability.
            if side == 1:  # S distribution
                X[:, 1] += label_shift * noise_scale
        n_violate = int(eps_violate * n)
        if n_violate > 0:
            idx = rng.choice(n, n_violate, replace=False)
            X[idx, 0] = -side * (np.abs(rng.standard_normal(n_violate)) * 0.5
                                  + gamma + 0.05)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        scale = np.minimum(1.0, R / norms)
        X *= scale
        return X

    x_S = _make_samples(N_S, +1, eps_S)
    # Downstream labels for S: informative (based on shifted x[1])
    y_S = (x_S[:, 1] > 0).astype(int) if d > 1 else rng.integers(0, 2, N_S)

    n_from_S = rng.binomial(N_B, p)
    n_from_O = N_B - n_from_S

    x_B_S = _make_samples(n_from_S, +1, eps_S)
    x_B_O = _make_samples(n_from_O, -1, eps_O)

    # Downstream labels for B: S-in-B get informative labels, O-in-B get random
    y_B_S = (x_B_S[:, 1] > 0).astype(int) if d > 1 else rng.integers(0, 2, n_from_S)
    y_B_O = rng.integers(0, 2, n_from_O)  # random labels for O

    x_B = np.vstack([x_B_S, x_B_O]) if n_from_O > 0 else x_B_S
    labels_B = np.array([1]*n_from_S + [0]*n_from_O)  # provenance: 1=from S
    y_B = np.concatenate([y_B_S, y_B_O])  # downstream labels

    perm = rng.permutation(N_B)
    x_B = x_B[perm]
    labels_B = labels_B[perm]
    y_B = y_B[perm]

    return x_S, x_B, labels_B, y_S, y_B


# ── Filtering algorithm ─────────────────────────────────────────────────────

def run_filter(x_S, x_B, gamma, method="svm"):
    """
    Implement the paper's filter using SVM.

    The paper's optimization:
      w = argmin_{||w||<=1, <w,x_i^S> >= gamma for all i} sum 1[<w,x_i^B> >= -gamma]

    We approximate this as a binary classification:
      - Positive class: S samples (label +1)
      - Negative class: B samples (label -1)
      - High penalty on misclassifying S samples (C_pos >> C_neg)

    Returns: boolean mask of B samples that pass the filter, and the classifier.
    """
    N_S = len(x_S)
    N_B = len(x_B)

    X_train = np.vstack([x_S, x_B])
    y_train = np.array([1]*N_S + [-1]*N_B)

    clf = LinearSVC(
        C=100.0,
        class_weight={1: N_B / N_S, -1: 1.0},
        max_iter=10000,
        dual="auto",
    )
    clf.fit(X_train, y_train)

    scores_B = clf.decision_function(x_B)
    passed = scores_B >= 0

    return passed, clf


# ── Evaluation ───────────────────────────────────────────────────────────────

def estimate_tv_histogram(x_S, x_B, passed, n_bins=50):
    """Histogram-based TV distance estimate on the first coordinate."""
    if passed.sum() == 0:
        return 1.0

    vals_S = x_S[:, 0]
    vals_F = x_B[passed, 0]

    lo = min(vals_S.min(), vals_F.min()) - 0.5
    hi = max(vals_S.max(), vals_F.max()) + 0.5
    bins = np.linspace(lo, hi, n_bins + 1)

    hist_S, _ = np.histogram(vals_S, bins=bins, density=True)
    hist_F, _ = np.histogram(vals_F, bins=bins, density=True)

    bin_width = bins[1] - bins[0]
    tv = 0.5 * np.sum(np.abs(hist_S - hist_F)) * bin_width
    return tv


def compute_metrics(x_S, x_B, passed, labels_B):
    """Compute false positive rate, false negative rate, and pass rate."""
    s_in_B = labels_B == 1
    o_in_B = labels_B == 0

    fn_rate = 1 - passed[s_in_B].mean() if s_in_B.sum() > 0 else 0.0
    fp_rate = passed[o_in_B].mean() if o_in_B.sum() > 0 else 0.0
    pass_rate = passed.mean()

    return {"fn_rate": fn_rate, "fp_rate": fp_rate, "pass_rate": pass_rate}


def estimate_tv_from_rates(fn_rate, fp_rate, p):
    """TV upper bound from the paper: TV(B^f, S) <= FN + FP/p."""
    return fn_rate + fp_rate / p


def run_downstream_task(x_S, y_S, x_B, y_B, passed, d, gamma, R, rng,
                        N_test=5000, label_shift=1.0):
    """
    Evaluate downstream classification accuracy.

    We define a binary classification task *within* the S distribution:
    classify based on whether the second coordinate x[1] > 0.

    Labels are pre-generated: S labels are informative (based on shifted
    x[1]), but O-in-B labels are random coin flips.  This means unfiltered
    B data injects genuine label noise, hurting the downstream classifier.

    Returns: (accuracy using S only, accuracy using S + filtered B,
              accuracy using S + all B unfiltered).
    """
    def _fit_safe(X, y):
        """Fit logistic regression, returning a constant predictor if single class."""
        if len(np.unique(y)) < 2:
            class DummyClf:
                def __init__(self, label):
                    self._label = label
                def score(self, X, y):
                    return (y == self._label).mean()
            return DummyClf(y[0])
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, y)
        return clf

    # S-only classifier
    clf_s = _fit_safe(x_S, y_S)

    # S + filtered B classifier (filtered B ≈ S, so labels are informative)
    x_filtered = x_B[passed]
    y_filtered = y_B[passed]
    if len(x_filtered) > 0:
        x_aug = np.vstack([x_S, x_filtered])
        y_aug = np.concatenate([y_S, y_filtered])
        clf_f = _fit_safe(x_aug, y_aug)
    else:
        clf_f = clf_s

    # S + unfiltered B classifier — same number of B samples as filtered
    n_filt = len(x_filtered) if len(x_filtered) > 0 else max(int(0.01 * len(x_B)), 100)
    n_add = min(len(x_B), n_filt)
    idx = rng.choice(len(x_B), size=n_add, replace=False)
    x_rand = x_B[idx]
    y_rand = y_B[idx]  # uses pre-generated labels (random for O-in-B)
    x_all = np.vstack([x_S, x_rand])
    y_all = np.concatenate([y_S, y_rand])
    clf_r = _fit_safe(x_all, y_all)

    # Test on fresh S samples (with the same label_shift as S training data)
    x_test = np.zeros((N_test, d))
    x_test[:, 0] = np.abs(rng.standard_normal(N_test)) * 0.5 + gamma + 0.05
    if d > 1:
        noise_scale = R / max(np.sqrt(d), 1.0) * 0.5
        x_test[:, 1:] = rng.standard_normal((N_test, d - 1)) * noise_scale
        x_test[:, 1] += label_shift * noise_scale  # S distribution has positive shift
    norms = np.linalg.norm(x_test, axis=1, keepdims=True)
    scale = np.minimum(1.0, R / norms)
    x_test *= scale
    y_test = (x_test[:, 1] > 0).astype(int)

    return (clf_s.score(x_test, y_test), clf_f.score(x_test, y_test),
            clf_r.score(x_test, y_test))


# ── Experiments ──────────────────────────────────────────────────────────────

def experiment_visualization():
    """2D visualization of the filtering process."""
    print("=" * 60)
    print("Experiment: 2D Visualization")
    print("=" * 60)

    rng = np.random.default_rng(99)
    d = 2
    N_S = 500
    N_B = 50_000
    p = 0.01
    gamma = 0.5
    R = 4.0

    x_S, x_B, labels_B, y_S, y_B = generate_data(d, N_S, N_B, p, gamma, R, rng)
    passed, clf = run_filter(x_S, x_B, gamma)
    m = compute_metrics(x_S, x_B, passed, labels_B)
    tv = estimate_tv_histogram(x_S, x_B, passed)

    # Panel 1: Raw data
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x_B[labels_B == 0, 0], x_B[labels_B == 0, 1],
               alpha=0.15, s=5, c="C3", label=r"$B$ from $O$", rasterized=True)
    ax.scatter(x_B[labels_B == 1, 0], x_B[labels_B == 1, 1],
               alpha=0.4, s=8, c="C0", label=r"$B$ from $S$")
    ax.scatter(x_S[:, 0], x_S[:, 1],
               alpha=0.6, s=15, c="C2", marker="x", label=r"$S$ samples")
    ax.set_title(f"Input data (ground truth), $p={p}$")
    ax.legend(fontsize=9)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.grid(True, alpha=0.3)
    save_plot(fig, "viz_input_data")

    # Panel 2: After filtering
    fig, ax = plt.subplots(figsize=(7, 5))
    w = clf.coef_[0]
    b = clf.intercept_[0]
    x_range = np.linspace(-5, 5, 100)
    if abs(w[1]) > 1e-10:
        y_boundary = -(w[0] * x_range + b) / w[1]
        ax.plot(x_range, y_boundary, "k-", linewidth=2, label="Filter boundary")
    ax.scatter(x_B[~passed, 0], x_B[~passed, 1],
               alpha=0.1, s=5, c="gray", label="Rejected", rasterized=True)
    ax.scatter(x_B[passed, 0], x_B[passed, 1],
               alpha=0.5, s=8, c="C0", label="Passed filter")
    ax.scatter(x_S[:, 0], x_S[:, 1],
               alpha=0.6, s=15, c="C2", marker="x", label=r"$S$ samples")
    ax.set_title(f"After filtering (FP={m['fp_rate']:.3f}, FN={m['fn_rate']:.3f})")
    ax.legend(fontsize=9)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.grid(True, alpha=0.3)
    save_plot(fig, "viz_after_filtering")

    # Panel 3: Distribution comparison
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(-5, 5, 40)
    ax.hist(x_S[:, 0], bins=bins, density=True, alpha=0.5,
            label=r"$S$ distribution", color="C2")
    if passed.sum() > 0:
        ax.hist(x_B[passed, 0], bins=bins, density=True, alpha=0.5,
                label="Filtered distribution", color="C0")
    ax.set_title(f"Distribution comparison (TV$\\approx${tv:.3f})")
    ax.legend(fontsize=9)
    ax.set_xlabel("$x_1$ (separating coordinate)")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)
    save_plot(fig, "viz_distribution_comparison")


def experiment_vary_N_S(n_trials=10):
    """TV distance and downstream accuracy vs N_S."""
    print("=" * 60)
    print("Experiment: Varying N_S")
    print("=" * 60)

    d = 20
    N_B = 500_000
    p = 0.01
    gamma = 0.5
    R = 3.0

    N_S_values = [50, 100, 200, 500, 1000, 2000, 5000]
    results = {k: [] for k in [
        "tv_mean", "tv_std", "fp_mean", "fn_mean",
        "acc_s_mean", "acc_s_std", "acc_f_mean", "acc_f_std",
        "acc_r_mean", "acc_r_std"
    ]}

    for N_S in N_S_values:
        tvs, fps, fns, accs_s, accs_f, accs_r = [], [], [], [], [], []
        for trial in range(n_trials):
            rng = np.random.default_rng(1000 * trial + N_S)
            x_S, x_B, labels_B, y_S, y_B = generate_data(d, N_S, N_B, p, gamma, R, rng)
            passed, clf = run_filter(x_S, x_B, gamma)
            tv = estimate_tv_histogram(x_S, x_B, passed)
            m = compute_metrics(x_S, x_B, passed, labels_B)
            a_s, a_f, a_r = run_downstream_task(x_S, y_S, x_B, y_B, passed, d, gamma, R, rng)
            tvs.append(tv)
            fps.append(m["fp_rate"])
            fns.append(m["fn_rate"])
            accs_s.append(a_s)
            accs_f.append(a_f)
            accs_r.append(a_r)

        results["tv_mean"].append(np.mean(tvs))
        results["tv_std"].append(np.std(tvs))
        results["fp_mean"].append(np.mean(fps))
        results["fn_mean"].append(np.mean(fns))
        results["acc_s_mean"].append(np.mean(accs_s))
        results["acc_s_std"].append(np.std(accs_s))
        results["acc_f_mean"].append(np.mean(accs_f))
        results["acc_f_std"].append(np.std(accs_f))
        results["acc_r_mean"].append(np.mean(accs_r))
        results["acc_r_std"].append(np.std(accs_r))
        print(f"  N_S={N_S:5d}: TV={np.mean(tvs):.4f}, "
              f"S-acc={np.mean(accs_s):.4f}, Filt-acc={np.mean(accs_f):.4f}, "
              f"Unfilt-acc={np.mean(accs_r):.4f}")

    data = {"N_S_values": N_S_values, "d": d, "N_B": N_B, "p": p,
            "gamma": gamma, "R": R, "n_trials": n_trials, **results}
    save_data("vary_NS", data)

    # TV distance plot
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(N_S_values, results["tv_mean"], yerr=results["tv_std"],
                fmt="o-", capsize=4, label="Estimated TV distance", color="C0")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$N_S$ (number of target samples)")
    ax.set_ylabel("TV distance")
    ax.set_title(rf"TV distance vs $N_S$ ($N_B={N_B:,}$, $p={p}$)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot(fig, "vary_NS_tv")

    # Downstream accuracy plot
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(N_S_values, results["acc_s_mean"], yerr=results["acc_s_std"],
                fmt="s-", capsize=4, label=r"$S$ samples only", color="C3")
    ax.errorbar(N_S_values, results["acc_f_mean"], yerr=results["acc_f_std"],
                fmt="o-", capsize=4, label=r"$S$ + filtered $B$", color="C0")
    ax.errorbar(N_S_values, results["acc_r_mean"], yerr=results["acc_r_std"],
                fmt="D--", capsize=4, label=r"$S$ + unfiltered $B$", color="C2")
    ax.set_xscale("log")
    ax.set_xlabel(r"$N_S$ (number of target samples)")
    ax.set_ylabel("Downstream classification accuracy")
    ax.set_title(rf"Downstream accuracy vs $N_S$ ($N_B={N_B:,}$, $p={p}$)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot(fig, "vary_NS_downstream")


def experiment_vary_N_B(n_trials=10):
    """TV distance and downstream accuracy vs N_B."""
    print("=" * 60)
    print("Experiment: Varying N_B")
    print("=" * 60)

    d = 50
    N_S = 200
    p = 0.01
    gamma = 0.5
    R = 3.0

    N_B_values = [10_000, 50_000, 100_000, 200_000, 500_000, 1_000_000, 2_000_000]
    results = {k: [] for k in [
        "tv_mean", "tv_std", "acc_s_mean", "acc_s_std",
        "acc_f_mean", "acc_f_std", "acc_r_mean", "acc_r_std"
    ]}

    for N_B in N_B_values:
        tvs, accs_s, accs_f, accs_r = [], [], [], []
        for trial in range(n_trials):
            rng = np.random.default_rng(2000 * trial + N_B)
            x_S, x_B, labels_B, y_S, y_B = generate_data(d, N_S, N_B, p, gamma, R, rng)
            passed, clf = run_filter(x_S, x_B, gamma)
            tv = estimate_tv_histogram(x_S, x_B, passed)
            a_s, a_f, a_r = run_downstream_task(x_S, y_S, x_B, y_B, passed, d, gamma, R, rng)
            tvs.append(tv)
            accs_s.append(a_s)
            accs_f.append(a_f)
            accs_r.append(a_r)

        results["tv_mean"].append(np.mean(tvs))
        results["tv_std"].append(np.std(tvs))
        results["acc_s_mean"].append(np.mean(accs_s))
        results["acc_s_std"].append(np.std(accs_s))
        results["acc_f_mean"].append(np.mean(accs_f))
        results["acc_f_std"].append(np.std(accs_f))
        results["acc_r_mean"].append(np.mean(accs_r))
        results["acc_r_std"].append(np.std(accs_r))
        print(f"  N_B={N_B:>9,}: TV={np.mean(tvs):.4f}, "
              f"S-acc={np.mean(accs_s):.4f}, Filt-acc={np.mean(accs_f):.4f}, "
              f"Unfilt-acc={np.mean(accs_r):.4f}")

    data = {"N_B_values": N_B_values, "d": d, "N_S": N_S, "p": p,
            "gamma": gamma, "R": R, "n_trials": n_trials, **results}
    save_data("vary_NB", data)

    # TV distance plot
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(N_B_values, results["tv_mean"], yerr=results["tv_std"],
                fmt="o-", capsize=4, label="Estimated TV distance", color="C0")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$N_B$ (number of big distribution samples)")
    ax.set_ylabel("TV distance")
    ax.set_title(rf"TV distance vs $N_B$ ($N_S={N_S}$, $p={p}$)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot(fig, "vary_NB_tv")

    # Downstream accuracy plot
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(N_B_values, results["acc_s_mean"], yerr=results["acc_s_std"],
                fmt="s-", capsize=4, label=r"$S$ samples only", color="C3")
    ax.errorbar(N_B_values, results["acc_f_mean"], yerr=results["acc_f_std"],
                fmt="o-", capsize=4, label=r"$S$ + filtered $B$", color="C0")
    ax.errorbar(N_B_values, results["acc_r_mean"], yerr=results["acc_r_std"],
                fmt="D--", capsize=4, label=r"$S$ + unfiltered $B$", color="C2")
    ax.set_xscale("log")
    ax.set_xlabel(r"$N_B$ (number of big distribution samples)")
    ax.set_ylabel("Downstream classification accuracy")
    ax.set_title(rf"Downstream accuracy vs $N_B$ ($N_S={N_S}$, $p={p}$)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot(fig, "vary_NB_downstream")


def experiment_vary_dimension(n_trials=10):
    """TV bound and downstream accuracy vs dimension d."""
    print("=" * 60)
    print("Experiment: Varying dimension d")
    print("=" * 60)

    N_S = 1000
    N_B = 500_000
    p = 0.01
    gamma = 0.5
    R = 3.0

    d_values = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
    results = {k: [] for k in [
        "tv_mean", "tv_std", "fp_mean", "fn_mean",
        "acc_s_mean", "acc_s_std", "acc_f_mean", "acc_f_std",
        "acc_r_mean", "acc_r_std"
    ]}

    for d in d_values:
        tvs, fps, fns, accs_s, accs_f, accs_r = [], [], [], [], [], []
        for trial in range(n_trials):
            rng = np.random.default_rng(3000 * trial + d)
            x_S, x_B, labels_B, y_S, y_B = generate_data(d, N_S, N_B, p, gamma, R, rng)
            passed, clf = run_filter(x_S, x_B, gamma)
            m = compute_metrics(x_S, x_B, passed, labels_B)
            tv = estimate_tv_from_rates(m["fn_rate"], m["fp_rate"], p)
            a_s, a_f, a_r = run_downstream_task(x_S, y_S, x_B, y_B, passed, d, gamma, R, rng)
            tvs.append(tv)
            fps.append(m["fp_rate"])
            fns.append(m["fn_rate"])
            accs_s.append(a_s)
            accs_f.append(a_f)
            accs_r.append(a_r)

        results["tv_mean"].append(np.mean(tvs))
        results["tv_std"].append(np.std(tvs))
        results["fp_mean"].append(np.mean(fps))
        results["fn_mean"].append(np.mean(fns))
        results["acc_s_mean"].append(np.mean(accs_s))
        results["acc_s_std"].append(np.std(accs_s))
        results["acc_f_mean"].append(np.mean(accs_f))
        results["acc_f_std"].append(np.std(accs_f))
        results["acc_r_mean"].append(np.mean(accs_r))
        results["acc_r_std"].append(np.std(accs_r))
        print(f"  d={d:5d}: TV_bound={np.mean(tvs):.4f}, "
              f"S-acc={np.mean(accs_s):.4f}, Filt-acc={np.mean(accs_f):.4f}, "
              f"Unfilt-acc={np.mean(accs_r):.4f}")

    data = {"d_values": d_values, "N_S": N_S, "N_B": N_B, "p": p,
            "gamma": gamma, "R": R, "n_trials": n_trials, **results}
    save_data("vary_dimension", data)

    # TV bound plot
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(d_values, results["tv_mean"], yerr=results["tv_std"],
                fmt="o-", capsize=4, color="C0")
    ax.set_xscale("log")
    ax.set_xlabel("Dimension $d$")
    ax.set_ylabel(r"TV bound (FN + FP/$p$)")
    ax.set_title(rf"TV bound vs dimension $d$ ($N_S={N_S}$, $N_B={N_B:,}$, $p={p}$)")
    ax.grid(True, alpha=0.3)
    save_plot(fig, "vary_dimension_tv")

    # FP/FN plot
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(d_values, results["fp_mean"], "s-", label="False positive rate", color="C2")
    ax.plot(d_values, results["fn_mean"], "^-", label="False negative rate", color="C3")
    ax.set_xscale("log")
    ax.set_xlabel("Dimension $d$")
    ax.set_ylabel("Error rate")
    ax.set_title(rf"Classification errors vs dimension $d$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot(fig, "vary_dimension_errors")

    # Downstream accuracy plot
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(d_values, results["acc_s_mean"], yerr=results["acc_s_std"],
                fmt="s-", capsize=4, label=r"$S$ samples only", color="C3")
    ax.errorbar(d_values, results["acc_f_mean"], yerr=results["acc_f_std"],
                fmt="o-", capsize=4, label=r"$S$ + filtered $B$", color="C0")
    ax.errorbar(d_values, results["acc_r_mean"], yerr=results["acc_r_std"],
                fmt="D--", capsize=4, label=r"$S$ + unfiltered $B$", color="C2")
    ax.set_xscale("log")
    ax.set_xlabel("Dimension $d$")
    ax.set_ylabel("Downstream classification accuracy")
    ax.set_title(rf"Downstream accuracy vs dimension $d$ ($N_S={N_S}$, $p={p}$)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot(fig, "vary_dimension_downstream")


def experiment_vary_gamma(n_trials=15):
    """TV bound and downstream accuracy vs margin gamma."""
    print("=" * 60)
    print("Experiment: Varying margin gamma")
    print("=" * 60)

    d = 10
    N_S = 30
    N_B = 500_000
    p = 0.01
    R = 3.0

    gamma_values = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.5]
    results = {k: [] for k in [
        "tv_mean", "tv_std", "fn_mean",
        "acc_s_mean", "acc_s_std", "acc_f_mean", "acc_f_std",
        "acc_r_mean", "acc_r_std"
    ]}

    for gamma in gamma_values:
        tvs, fns, accs_s, accs_f, accs_r = [], [], [], [], []
        for trial in range(n_trials):
            rng = np.random.default_rng(4000 * trial + int(gamma * 1000))
            x_S, x_B, labels_B, y_S, y_B = generate_data(d, N_S, N_B, p, gamma, R, rng)
            passed, clf = run_filter(x_S, x_B, gamma)
            m = compute_metrics(x_S, x_B, passed, labels_B)
            tv = estimate_tv_from_rates(m["fn_rate"], m["fp_rate"], p)
            a_s, a_f, a_r = run_downstream_task(x_S, y_S, x_B, y_B, passed, d, gamma, R, rng)
            tvs.append(tv)
            fns.append(m["fn_rate"])
            accs_s.append(a_s)
            accs_f.append(a_f)
            accs_r.append(a_r)

        results["tv_mean"].append(np.mean(tvs))
        results["tv_std"].append(np.std(tvs))
        results["fn_mean"].append(np.mean(fns))
        results["acc_s_mean"].append(np.mean(accs_s))
        results["acc_s_std"].append(np.std(accs_s))
        results["acc_f_mean"].append(np.mean(accs_f))
        results["acc_f_std"].append(np.std(accs_f))
        results["acc_r_mean"].append(np.mean(accs_r))
        results["acc_r_std"].append(np.std(accs_r))
        print(f"  gamma={gamma:.2f}: TV_bound={np.mean(tvs):.4f}, "
              f"S-acc={np.mean(accs_s):.4f}, Filt-acc={np.mean(accs_f):.4f}, "
              f"Unfilt-acc={np.mean(accs_r):.4f}")

    data = {"gamma_values": gamma_values, "d": d, "N_S": N_S, "N_B": N_B,
            "p": p, "R": R, "n_trials": n_trials, **results}
    save_data("vary_gamma", data)

    # TV bound plot
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(gamma_values, results["tv_mean"], yerr=results["tv_std"],
                fmt="o-", capsize=4, label=r"TV bound (FN + FP/$p$)", color="C0")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Margin $\gamma$")
    ax.set_ylabel(r"TV bound")
    ax.set_title(rf"TV bound vs margin $\gamma$ ($N_S={N_S}$, $p={p}$, $R={R}$)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot(fig, "vary_gamma_tv")

    # Downstream accuracy plot
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(gamma_values, results["acc_s_mean"], yerr=results["acc_s_std"],
                fmt="s-", capsize=4, label=r"$S$ samples only", color="C3")
    ax.errorbar(gamma_values, results["acc_f_mean"], yerr=results["acc_f_std"],
                fmt="o-", capsize=4, label=r"$S$ + filtered $B$", color="C0")
    ax.errorbar(gamma_values, results["acc_r_mean"], yerr=results["acc_r_std"],
                fmt="D--", capsize=4, label=r"$S$ + unfiltered $B$", color="C2")
    ax.set_xscale("log")
    ax.set_xlabel(r"Margin $\gamma$")
    ax.set_ylabel("Downstream classification accuracy")
    ax.set_title(rf"Downstream accuracy vs $\gamma$ ($N_S={N_S}$, $p={p}$)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot(fig, "vary_gamma_downstream")


def experiment_weak_separation(n_trials=15):
    """TV bound and downstream accuracy vs eps_O and eps_S."""
    print("=" * 60)
    print("Experiment: Weak separation")
    print("=" * 60)

    d = 10
    N_S = 1000
    N_B = 500_000
    p = 0.01
    gamma = 0.5
    R = 3.0

    # --- Vary eps_O ---
    eps_O_values = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
    res_O = {k: [] for k in [
        "tv_mean", "tv_std", "fp_mean",
        "acc_s_mean", "acc_s_std", "acc_f_mean", "acc_f_std",
        "acc_r_mean", "acc_r_std"
    ]}

    for eps_O in eps_O_values:
        tvs, fps, accs_s, accs_f, accs_r = [], [], [], [], []
        for trial in range(n_trials):
            rng = np.random.default_rng(5000 * trial + int(eps_O * 10000))
            x_S, x_B, labels_B, y_S, y_B = generate_data(
                d, N_S, N_B, p, gamma, R, rng, eps_S=0.0, eps_O=eps_O)
            passed, clf = run_filter(x_S, x_B, gamma)
            m = compute_metrics(x_S, x_B, passed, labels_B)
            tv = estimate_tv_from_rates(m["fn_rate"], m["fp_rate"], p)
            a_s, a_f, a_r = run_downstream_task(x_S, y_S, x_B, y_B, passed, d, gamma, R, rng)
            tvs.append(tv)
            fps.append(m["fp_rate"])
            accs_s.append(a_s)
            accs_f.append(a_f)
            accs_r.append(a_r)

        res_O["tv_mean"].append(np.mean(tvs))
        res_O["tv_std"].append(np.std(tvs))
        res_O["fp_mean"].append(np.mean(fps))
        res_O["acc_s_mean"].append(np.mean(accs_s))
        res_O["acc_s_std"].append(np.std(accs_s))
        res_O["acc_f_mean"].append(np.mean(accs_f))
        res_O["acc_f_std"].append(np.std(accs_f))
        res_O["acc_r_mean"].append(np.mean(accs_r))
        res_O["acc_r_std"].append(np.std(accs_r))
        print(f"  eps_O={eps_O:.3f}: TV_bound={np.mean(tvs):.4f}, "
              f"Filt-acc={np.mean(accs_f):.4f}, Unfilt-acc={np.mean(accs_r):.4f}")

    # --- Vary eps_S ---
    eps_S_values = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
    res_S = {k: [] for k in [
        "tv_mean", "tv_std", "fn_mean",
        "acc_s_mean", "acc_s_std", "acc_f_mean", "acc_f_std",
        "acc_r_mean", "acc_r_std"
    ]}

    for eps_S in eps_S_values:
        tvs, fns, accs_s, accs_f, accs_r = [], [], [], [], []
        for trial in range(n_trials):
            rng = np.random.default_rng(6000 * trial + int(eps_S * 1000))
            x_S, x_B, labels_B, y_S, y_B = generate_data(
                d, N_S, N_B, p, gamma, R, rng, eps_S=eps_S, eps_O=0.0)
            passed, clf = run_filter(x_S, x_B, gamma)
            m = compute_metrics(x_S, x_B, passed, labels_B)
            tv = estimate_tv_from_rates(m["fn_rate"], m["fp_rate"], p)
            a_s, a_f, a_r = run_downstream_task(x_S, y_S, x_B, y_B, passed, d, gamma, R, rng)
            tvs.append(tv)
            fns.append(m["fn_rate"])
            accs_s.append(a_s)
            accs_f.append(a_f)
            accs_r.append(a_r)

        res_S["tv_mean"].append(np.mean(tvs))
        res_S["tv_std"].append(np.std(tvs))
        res_S["fn_mean"].append(np.mean(fns))
        res_S["acc_s_mean"].append(np.mean(accs_s))
        res_S["acc_s_std"].append(np.std(accs_s))
        res_S["acc_f_mean"].append(np.mean(accs_f))
        res_S["acc_f_std"].append(np.std(accs_f))
        res_S["acc_r_mean"].append(np.mean(accs_r))
        res_S["acc_r_std"].append(np.std(accs_r))
        print(f"  eps_S={eps_S:.2f}: TV_bound={np.mean(tvs):.4f}, "
              f"Filt-acc={np.mean(accs_f):.4f}, Unfilt-acc={np.mean(accs_r):.4f}")

    data = {
        "eps_O_values": eps_O_values, "eps_S_values": eps_S_values,
        "d": d, "N_S": N_S, "N_B": N_B, "p": p, "gamma": gamma, "R": R,
        "n_trials": n_trials,
        "eps_O_results": res_O, "eps_S_results": res_S,
    }
    save_data("weak_separation", data)

    # TV bound vs eps_O
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(eps_O_values, res_O["tv_mean"], yerr=res_O["tv_std"],
                fmt="o-", capsize=4, color="C0", label=r"TV bound (FN + FP/$p$)")
    baseline = res_O["tv_mean"][0]
    ref = [baseline + e / p for e in eps_O_values]
    ax.plot(eps_O_values, ref, "--", color="C1", alpha=0.7,
            label=r"baseline $+ \varepsilon_O / p$ (theory)")
    ax.set_xlabel(r"$\varepsilon_O$ (fraction of $O$ violating margin)")
    ax.set_ylabel(r"TV bound")
    ax.set_title(rf"Weak separation: varying $\varepsilon_O$ ($p={p}$)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot(fig, "weak_sep_eps_O_tv")

    # TV bound vs eps_S
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(eps_S_values, res_S["tv_mean"], yerr=res_S["tv_std"],
                fmt="o-", capsize=4, color="C0", label=r"TV bound (FN + FP/$p$)")
    baseline = res_S["tv_mean"][0]
    ref = [baseline + e for e in eps_S_values]
    ax.plot(eps_S_values, ref, "--", color="C1", alpha=0.7,
            label=r"baseline $+ \varepsilon_S$ (theory)")
    ax.set_xlabel(r"$\varepsilon_S$ (fraction of $S$ violating margin)")
    ax.set_ylabel(r"TV bound")
    ax.set_title(r"Weak separation: varying $\varepsilon_S$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot(fig, "weak_sep_eps_S_tv")

    # Downstream accuracy vs eps_O
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(eps_O_values, res_O["acc_s_mean"], yerr=res_O["acc_s_std"],
                fmt="s-", capsize=4, label=r"$S$ samples only", color="C3")
    ax.errorbar(eps_O_values, res_O["acc_f_mean"], yerr=res_O["acc_f_std"],
                fmt="o-", capsize=4, label=r"$S$ + filtered $B$", color="C0")
    ax.errorbar(eps_O_values, res_O["acc_r_mean"], yerr=res_O["acc_r_std"],
                fmt="D--", capsize=4, label=r"$S$ + unfiltered $B$", color="C2")
    ax.set_xlabel(r"$\varepsilon_O$ (fraction of $O$ violating margin)")
    ax.set_ylabel("Downstream classification accuracy")
    ax.set_title(rf"Downstream accuracy vs $\varepsilon_O$ ($p={p}$)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot(fig, "weak_sep_eps_O_downstream")

    # Downstream accuracy vs eps_S
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(eps_S_values, res_S["acc_s_mean"], yerr=res_S["acc_s_std"],
                fmt="s-", capsize=4, label=r"$S$ samples only", color="C3")
    ax.errorbar(eps_S_values, res_S["acc_f_mean"], yerr=res_S["acc_f_std"],
                fmt="o-", capsize=4, label=r"$S$ + filtered $B$", color="C0")
    ax.errorbar(eps_S_values, res_S["acc_r_mean"], yerr=res_S["acc_r_std"],
                fmt="D--", capsize=4, label=r"$S$ + unfiltered $B$", color="C2")
    ax.set_xlabel(r"$\varepsilon_S$ (fraction of $S$ violating margin)")
    ax.set_ylabel("Downstream classification accuracy")
    ax.set_title(r"Downstream accuracy vs $\varepsilon_S$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot(fig, "weak_sep_eps_S_downstream")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.time()

    experiment_visualization()
    experiment_vary_N_S()
    experiment_vary_N_B()
    experiment_vary_dimension()
    experiment_vary_gamma()
    experiment_weak_separation()

    elapsed = time.time() - t0
    print(f"\nAll experiments completed in {elapsed:.1f}s")
    print(f"Plots saved to {PLOTS_DIR.resolve()}")
    print(f"Data saved to {DATA_DIR.resolve()}")
