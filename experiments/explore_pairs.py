"""Explore downstream task difficulty and filtering benefit for different class pairs.

Finds pairs where:
1. N_S=200 leaves meaningful headroom (S-only << oracle)
2. Perfect filtering (true S samples from B, correct labels) closes most of that gap
3. Confounding labels make random B clearly harmful

Usage:
    uv run python explore_pairs.py
"""

import numpy as np
import scipy.sparse as sp
from sklearn.datasets import fetch_openml, fetch_20newsgroups, fetch_covtype
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

N_S = 200
N_FILTERED_CAP = 1000
SEED = 42


def vstack(arrays):
    """vstack that handles both dense and sparse arrays."""
    if any(sp.issparse(a) for a in arrays):
        return sp.vstack(arrays)
    return np.row_stack(arrays)


def nrows(X):
    """Number of rows, works for both dense and sparse."""
    return X.shape[0]


def evaluate_pair(X_pool, y_pool, X_test, y_test, class_a, class_b,
                  confound_a, confound_b, rng, label=""):
    """Evaluate a single pair with all baselines including perfect filter."""
    is_a = y_pool == class_a
    is_b = y_pool == class_b
    a_idx = np.where(is_a)[0]
    b_idx = np.where(is_b)[0]

    # S samples
    n_per = N_S // 2
    sa = rng.choice(a_idx, min(n_per, len(a_idx)), replace=False)
    sb = rng.choice(b_idx, min(n_per, len(b_idx)), replace=False)
    x_S = X_pool[np.concatenate([sa, sb])]
    y_S = np.array([1]*len(sa) + [0]*len(sb))

    # Test set: only A and B
    test_ab = (y_test == class_a) | (y_test == class_b)
    Xt = X_test[test_ab]
    yt = (y_test[test_ab] == class_a).astype(int)

    if Xt.shape[0] == 0:
        return None

    def score(X_train, y_train):
        clf = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced')
        clf.fit(X_train, y_train)
        return balanced_accuracy_score(yt, clf.predict(Xt))

    # S-only
    acc_s = score(x_S, y_S)

    # Oracle (all true S)
    xO = X_pool[np.concatenate([a_idx, b_idx])]
    yO = np.array([1]*len(a_idx) + [0]*len(b_idx))
    acc_o = score(xO, yO)

    # Perfect filter: remaining true S samples with correct labels
    remaining_a = np.setdiff1d(a_idx, sa)
    remaining_b = np.setdiff1d(b_idx, sb)
    x_perf = X_pool[np.concatenate([remaining_a, remaining_b])]
    y_perf = np.array([1]*len(remaining_a) + [0]*len(remaining_b))
    if nrows(x_perf) > N_FILTERED_CAP:
        fi = rng.choice(nrows(x_perf), N_FILTERED_CAP, replace=False)
        x_perf = x_perf[fi]
        y_perf = y_perf[fi]
    Xpf = vstack([x_S, x_perf])
    ypf = np.concatenate([y_S, y_perf])
    acc_pf = score(Xpf, ypf)

    # Confounding labels for O
    is_c = y_pool == confound_a
    is_d = y_pool == confound_b
    if is_c.sum() > 0 and is_d.sum() > 0:
        X_cd = vstack([X_pool[is_c], X_pool[is_d]])
        y_cd = np.array([1]*is_c.sum() + [0]*is_d.sum())
        clf_conf = LogisticRegression(max_iter=1000, C=1.0)
        clf_conf.fit(X_cd, y_cd)
        y_B_ds = clf_conf.predict(X_pool).astype(int)
    else:
        y_B_ds = rng.integers(0, 2, size=len(y_pool))
    y_B_ds[is_a] = 1
    y_B_ds[is_b] = 0

    # Random B (unfiltered, same count as filtered cap)
    n_rand = min(N_FILTERED_CAP, nrows(X_pool))
    rand_idx = rng.choice(nrows(X_pool), n_rand, replace=False)
    Xr = vstack([x_S, X_pool[rand_idx]])
    yr = np.concatenate([y_S, y_B_ds[rand_idx]])
    acc_r = score(Xr, yr)

    # Run actual filter
    N_B = nrows(X_pool)
    X_filt = vstack([x_S, X_pool])
    y_filt = np.array([1]*N_S + [-1]*N_B)
    clf_filter = LogisticRegression(
        C=10.0, class_weight={1: N_B/N_S, -1: 1.0},
        max_iter=5000, solver='lbfgs')
    clf_filter.fit(X_filt, y_filt)
    scores_B = clf_filter.decision_function(X_pool)
    passed = scores_B >= 0

    labels_B = (is_a | is_b).astype(int)
    fn = 1 - passed[labels_B == 1].mean()
    fp = passed[labels_B == 0].mean()
    prec = labels_B[passed].mean() if passed.sum() > 0 else 0

    # Filtered B with confounding labels
    x_f = X_pool[passed]
    y_f = y_B_ds[passed]
    if nrows(x_f) > N_FILTERED_CAP:
        fi = rng.choice(nrows(x_f), N_FILTERED_CAP, replace=False)
        x_f = x_f[fi]
        y_f = y_f[fi]
    if nrows(x_f) > 0:
        Xff = vstack([x_S, x_f])
        yff = np.concatenate([y_S, y_f])
        acc_ff = score(Xff, yff)
    else:
        acc_ff = acc_s

    print(f"  {label:35s}  S={acc_s:.3f}  perf={acc_pf:.3f}  filt={acc_ff:.3f}  "
          f"rand={acc_r:.3f}  orac={acc_o:.3f}  "
          f"FN={fn:.3f} FP={fp:.4f} prec={prec:.2f}  "
          f"headroom={acc_o-acc_s:.3f}")
    return {
        "acc_s": acc_s, "acc_pf": acc_pf, "acc_ff": acc_ff,
        "acc_r": acc_r, "acc_o": acc_o,
        "fn": fn, "fp": fp, "prec": prec,
    }


def explore_mnist():
    print("\n" + "="*80)
    print("MNIST (PCA 50)")
    print("="*80)
    print(f"  {'pair':35s}  {'S':>5s}  {'perf':>5s}  {'filt':>5s}  "
          f"{'rand':>5s}  {'orac':>5s}  "
          f"{'FN':>5s} {'FP':>6s} {'prec':>4s}  {'headroom':>8s}")

    X, y = fetch_openml('mnist_784', version=1, return_X_y=True,
                        as_frame=False, parser='auto')
    y = y.astype(int)
    pca = PCA(n_components=50, random_state=SEED)
    X = pca.fit_transform(X)

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(X))
    n_test = len(X) // 5
    X_pool = X[idx[n_test:]]
    y_pool = y[idx[n_test:]]
    X_test = X[idx[:n_test]]
    y_test = y[idx[:n_test]]

    # Pairs with most headroom, plus confounding pairs
    pairs = [
        (4, 9, 3, 8, "4 vs 9 (confound 3v8)"),
        (3, 5, 4, 9, "3 vs 5 (confound 4v9)"),
        (3, 8, 4, 9, "3 vs 8 (confound 4v9)"),
        (0, 5, 3, 8, "0 vs 5 (confound 3v8)"),
        (2, 8, 3, 5, "2 vs 8 (confound 3v5)"),
        (1, 8, 4, 9, "1 vs 8 (confound 4v9)"),
        (5, 6, 3, 8, "5 vs 6 (confound 3v8)"),
        (7, 9, 3, 8, "7 vs 9 (confound 3v8)"),
    ]

    for a, b, ca, cb, label in pairs:
        evaluate_pair(X_pool, y_pool, X_test, y_test, a, b, ca, cb,
                      np.random.default_rng(SEED), label)


def explore_newsgroups():
    print("\n" + "="*80)
    print("20 Newsgroups (TF-IDF 10K)")
    print("="*80)
    print(f"  {'pair':35s}  {'S':>5s}  {'perf':>5s}  {'filt':>5s}  "
          f"{'rand':>5s}  {'orac':>5s}  "
          f"{'FN':>5s} {'FP':>6s} {'prec':>4s}  {'headroom':>8s}")

    data = fetch_20newsgroups(subset='all',
                              remove=('headers', 'footers', 'quotes'))
    vec = TfidfVectorizer(max_features=10000)
    X = vec.fit_transform(data.data)
    y = np.array(data.target)
    names = data.target_names

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(y))
    n_test = len(y) // 5
    X_pool = X[idx[n_test:]]
    y_pool = y[idx[n_test:]]
    X_test = X[idx[:n_test]]
    y_test = y[idx[:n_test]]

    # Pairs with most headroom
    pairs = [
        (3, 4, 16, 17, f"{names[3][:15]} vs {names[4][:15]} (confound pol)"),
        (7, 8, 3, 4, f"{names[7][:15]} vs {names[8][:15]} (confound comp)"),
        (1, 3, 9, 10, f"{names[1][:15]} vs {names[3][:15]} (confound sport)"),
        (9, 10, 1, 5, f"{names[9][:15]} vs {names[10][:15]} (confound comp)"),
        (11, 12, 7, 8, f"{names[11][:15]} vs {names[12][:15]} (confound rec)"),
        (12, 14, 16, 17, f"{names[12][:15]} vs {names[14][:15]} (confound pol)"),
        (13, 14, 9, 10, f"{names[13][:15]} vs {names[14][:15]} (confound sport)"),
        (16, 17, 9, 10, f"{names[16][:15]} vs {names[17][:15]} (confound sport)"),
        (15, 18, 9, 10, f"{names[15][:15]} vs {names[18][:15]} (confound sport)"),
        (17, 18, 9, 10, f"{names[17][:15]} vs {names[18][:15]} (confound sport)"),
    ]

    for a, b, ca, cb, label in pairs:
        evaluate_pair(X_pool, y_pool, X_test, y_test, a, b, ca, cb,
                      np.random.default_rng(SEED), label)


def explore_covertype():
    print("\n" + "="*80)
    print("Covertype (54 features)")
    print("="*80)
    print(f"  {'pair':35s}  {'S':>5s}  {'perf':>5s}  {'filt':>5s}  "
          f"{'rand':>5s}  {'orac':>5s}  "
          f"{'FN':>5s} {'FP':>6s} {'prec':>4s}  {'headroom':>8s}")

    X, y = fetch_covtype(return_X_y=True, as_frame=False)
    rng = np.random.default_rng(SEED)

    # Subsample to 100K
    idx = rng.choice(len(X), 100_000, replace=False)
    X = X[idx]
    y = y[idx]

    n_test = len(X) // 5
    perm = rng.permutation(len(X))
    X_pool = X[perm[n_test:]]
    y_pool = y[perm[n_test:]]
    X_test = X[perm[:n_test]]
    y_test = y[perm[:n_test]]

    # Class distribution
    for c in sorted(np.unique(y)):
        print(f"  Class {c}: {(y == c).sum()} ({(y == c).mean()*100:.1f}%)")

    pairs = [
        (3, 6, 1, 2, "Ponderosa vs Douglas-fir"),
        (1, 2, 3, 6, "Spruce/Fir vs Lodgepole"),
        (1, 3, 2, 6, "Spruce/Fir vs Ponderosa"),
        (2, 3, 1, 6, "Lodgepole vs Ponderosa"),
    ]

    for a, b, ca, cb, label in pairs:
        evaluate_pair(X_pool, y_pool, X_test, y_test, a, b, ca, cb,
                      np.random.default_rng(SEED), label)


if __name__ == "__main__":
    explore_newsgroups()
    explore_mnist()
    explore_covertype()
