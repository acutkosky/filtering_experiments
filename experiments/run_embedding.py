"""
Embedding-based filtering experiment: Wikipedia vs C4.

S = Wikipedia paragraphs (high-quality encyclopedic text)
O = C4 web crawl paragraphs (lower-quality web text)
B = p*S + (1-p)*O mixture

We embed text using sentence transformers and run the linear filter,
reporting FP/FN/precision rates at varying N_S. No downstream task —
just filter quality metrics.

Embeddings are cached to disk to avoid re-computing on repeated runs.
"""

import json
import hashlib
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

PLOTS_DIR = Path(__file__).parent / "plots"
DATA_DIR = Path(__file__).parent / "data"
CACHE_DIR = Path(__file__).parent / "embedding_cache"
PLOTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)


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


def cache_path(name, model_short):
    return CACHE_DIR / f"{name}_{model_short}.npz"


def load_or_embed(texts, name, model_short, model):
    """Embed texts, caching to disk."""
    path = cache_path(name, model_short)
    if path.exists():
        print(f"    Loading cached embeddings: {path.name}")
        return np.load(path)["embeddings"]

    print(f"    Embedding {len(texts)} texts with {model_short}...")
    # Encode in batches to show progress
    batch_size = 256
    all_emb = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        emb = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
        all_emb.append(emb)
        if (i // batch_size) % 10 == 0:
            print(f"      {i}/{len(texts)}")
    embeddings = np.vstack(all_emb).astype(np.float32)
    np.savez_compressed(path, embeddings=embeddings)
    print(f"    Saved to {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
    return embeddings


def load_wikipedia_texts(n_samples=5000):
    """Load Wikipedia paragraphs from HuggingFace.

    Uses paragraph-level chunks (~500 chars). Sentence transformers handle
    up to ~256-512 tokens, so paragraphs are a natural unit. For our filter
    task (Wikipedia vs web crawl), paragraph-level features are sufficient
    since Wikipedia has a distinctive style even at this granularity.
    """
    from datasets import load_dataset

    cache_file = CACHE_DIR / "wiki_texts.json"
    if cache_file.exists():
        print(f"  Loading cached Wikipedia texts from {cache_file.name}")
        with open(cache_file) as f:
            texts = json.load(f)
        return texts[:n_samples]

    print(f"  Downloading Wikipedia paragraphs (streaming)...")
    # Use the wikimedia foundation's parquet-based dataset
    ds = load_dataset("wikimedia/wikipedia", "20231101.en",
                      streaming=True, split="train")

    texts = []
    for i, example in enumerate(ds):
        text = example["text"].strip()
        # Split into paragraphs, keep substantial ones
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 100]
        for para in paragraphs:
            # Truncate long paragraphs to ~500 chars for uniform embedding
            texts.append(para[:500])
            if len(texts) >= n_samples:
                break
        if len(texts) >= n_samples:
            break

    texts = texts[:n_samples]
    with open(cache_file, "w") as f:
        json.dump(texts, f)
    print(f"  Cached {len(texts)} Wikipedia paragraphs")
    return texts


def load_c4_texts(n_samples=50000):
    """Load C4 web crawl paragraphs from HuggingFace (streaming)."""
    from datasets import load_dataset

    cache_file = CACHE_DIR / "c4_texts.json"
    if cache_file.exists():
        print(f"  Loading cached C4 texts from {cache_file.name}")
        with open(cache_file) as f:
            texts = json.load(f)
        return texts[:n_samples]

    print(f"  Downloading C4 paragraphs (streaming, {n_samples} samples)...")
    ds = load_dataset("allenai/c4", "en", streaming=True, split="train",
                      trust_remote_code=True)

    texts = []
    for i, example in enumerate(ds):
        text = example["text"].strip()
        # Take first ~500 chars of each document
        if len(text) > 100:
            texts.append(text[:500])
        if len(texts) >= n_samples:
            break
        if len(texts) % 10000 == 0 and len(texts) > 0:
            print(f"    {len(texts)}/{n_samples}")

    texts = texts[:n_samples]
    with open(cache_file, "w") as f:
        json.dump(texts, f)
    print(f"  Cached {len(texts)} C4 paragraphs")
    return texts


def run_filter(x_S, x_B, C=10.0, weight_scale=1.0):
    """Run the linear filter. Returns boolean mask of B samples that pass."""
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
    passed = scores_B >= 0.0
    return passed, clf


def compute_metrics(passed, labels_B):
    """labels_B: 1 if from S (Wikipedia), 0 if from O (C4)."""
    s_in_B = labels_B == 1
    o_in_B = labels_B == 0
    fn_rate = 1 - passed[s_in_B].mean() if s_in_B.sum() > 0 else 0.0
    fp_rate = passed[o_in_B].mean() if o_in_B.sum() > 0 else 0.0
    precision = labels_B[passed].mean() if passed.sum() > 0 else 0.0
    recall = passed[s_in_B].mean() if s_in_B.sum() > 0 else 0.0
    n_passed = int(passed.sum())
    return {"fn_rate": fn_rate, "fp_rate": fp_rate, "precision": precision,
            "recall": recall, "n_passed": n_passed, "pass_rate": float(passed.mean())}


def experiment_vary_NS(model, model_short, emb_dim,
                       X_wiki, X_c4,
                       n_wiki_pool=5000, n_c4_pool=50000,
                       p=0.01, n_trials=10):
    """
    Vary N_S and measure filter quality.

    B = p * Wikipedia + (1-p) * C4 mixture.
    S = N_S samples from Wikipedia.
    """
    print(f"\n  Experiment: vary N_S ({model_short}, {emb_dim}d)")

    N_S_values = [20, 50, 100, 200, 500, 1000, 2000, 5000]
    # Number of S-component samples in B
    n_wiki_in_B = int(p * n_c4_pool / (1 - p))  # so that p = n_wiki_in_B / N_B
    n_wiki_in_B = min(n_wiki_in_B, n_wiki_pool)
    N_B = n_wiki_in_B + n_c4_pool

    print(f"    N_B = {N_B}, wiki in B = {n_wiki_in_B}, p = {n_wiki_in_B/N_B:.4f}")

    results = {k: [] for k in [
        "fn_mean", "fn_std", "fp_mean", "fp_std",
        "prec_mean", "prec_std", "recall_mean", "recall_std",
        "n_passed_mean"
    ]}

    for N_S in N_S_values:
        fns, fps, precs, recalls, n_pass = [], [], [], [], []

        for trial in range(n_trials):
            rng = np.random.default_rng(700 * trial + N_S)

            # S samples: random subset of Wikipedia embeddings
            # Use embeddings beyond the pool used for B to avoid overlap
            wiki_s_start = n_wiki_pool  # S comes from held-out wiki
            if wiki_s_start + N_S > len(X_wiki):
                # Not enough held-out wiki, sample from pool but track overlap
                s_idx = rng.choice(n_wiki_pool, N_S, replace=False)
            else:
                s_idx = rng.choice(
                    np.arange(n_wiki_pool, len(X_wiki)), N_S, replace=False)
            x_S = X_wiki[s_idx]

            # B pool: mix of wiki and C4
            wiki_b_idx = rng.choice(n_wiki_pool, n_wiki_in_B, replace=False)
            c4_idx = rng.choice(len(X_c4), n_c4_pool, replace=False)
            x_B = np.vstack([X_wiki[wiki_b_idx], X_c4[c4_idx]])
            labels_B = np.array([1]*n_wiki_in_B + [0]*n_c4_pool)

            # Shuffle B
            perm = rng.permutation(len(x_B))
            x_B = x_B[perm]
            labels_B = labels_B[perm]

            passed, _ = run_filter(x_S, x_B)
            m = compute_metrics(passed, labels_B)
            fns.append(m["fn_rate"])
            fps.append(m["fp_rate"])
            precs.append(m["precision"])
            recalls.append(m["recall"])
            n_pass.append(m["n_passed"])

        results["fn_mean"].append(np.mean(fns))
        results["fn_std"].append(np.std(fns))
        results["fp_mean"].append(np.mean(fps))
        results["fp_std"].append(np.std(fps))
        results["prec_mean"].append(np.mean(precs))
        results["prec_std"].append(np.std(precs))
        results["recall_mean"].append(np.mean(recalls))
        results["recall_std"].append(np.std(recalls))
        results["n_passed_mean"].append(np.mean(n_pass))

        print(f"    N_S={N_S:5d}: FN={np.mean(fns):.4f}, FP={np.mean(fps):.4f}, "
              f"prec={np.mean(precs):.4f}, recall={np.mean(recalls):.4f}, "
              f"passed={np.mean(n_pass):.0f}")

    data_name = f"embedding_{model_short}"
    save_data(data_name, {
        "N_S_values": N_S_values,
        "model": model_short,
        "emb_dim": emb_dim,
        "n_wiki_pool": n_wiki_pool,
        "n_c4_pool": n_c4_pool,
        "p": p,
        "n_trials": n_trials,
        **results,
    })

    # Plot 1: FP and FN rates vs N_S
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(N_S_values, results["fn_mean"], yerr=results["fn_std"],
                fmt="^-", capsize=4, label="False negative rate", color="C3")
    ax.errorbar(N_S_values, results["fp_mean"], yerr=results["fp_std"],
                fmt="s-", capsize=4, label="False positive rate", color="C2")
    ax.set_xscale("log")
    ax.set_xlabel(r"$N_S$ (number of Wikipedia samples)")
    ax.set_ylabel("Error rate")
    ax.set_title(f"Wikipedia vs C4 ({model_short}, {emb_dim}d): filter errors vs $N_S$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot(fig, f"{data_name}_errors")

    # Plot 2: Precision and recall vs N_S
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(N_S_values, results["prec_mean"], yerr=results["prec_std"],
                fmt="o-", capsize=4, label="Precision", color="C0")
    ax.errorbar(N_S_values, results["recall_mean"], yerr=results["recall_std"],
                fmt="s-", capsize=4, label="Recall", color="C1")
    ax.set_xscale("log")
    ax.set_xlabel(r"$N_S$ (number of Wikipedia samples)")
    ax.set_ylabel("Rate")
    ax.set_title(f"Wikipedia vs C4 ({model_short}, {emb_dim}d): precision/recall vs $N_S$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    save_plot(fig, f"{data_name}_precision_recall")

    return results


def oracle_test(X_wiki, X_c4, model_short, emb_dim):
    """
    Oracle test: train a logistic regression classifier on all available
    labeled Wikipedia vs C4 data. This tells us the ceiling for linear
    separability in this embedding space.
    """
    print(f"\n  Oracle test ({model_short}, {emb_dim}d)")

    # Use half for train, half for test
    n_wiki = len(X_wiki)
    n_c4 = len(X_c4)
    n_wiki_train = n_wiki // 2
    n_c4_train = n_c4 // 2

    X_train = np.vstack([X_wiki[:n_wiki_train], X_c4[:n_c4_train]])
    y_train = np.array([1]*n_wiki_train + [0]*n_c4_train)
    X_test = np.vstack([X_wiki[n_wiki_train:], X_c4[n_c4_train:]])
    y_test = np.array([1]*(n_wiki - n_wiki_train) + [0]*(n_c4 - n_c4_train))

    clf = LogisticRegression(C=10.0, max_iter=5000, solver="lbfgs")
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    # Metrics on wiki (positive class)
    wiki_test = y_test == 1
    c4_test = y_test == 0
    recall = preds[wiki_test].mean()     # TPR: fraction of wiki correctly identified
    fp_rate = preds[c4_test].mean()      # FPR: fraction of C4 incorrectly called wiki
    precision = y_test[preds == 1].mean() if (preds == 1).sum() > 0 else 0.0
    accuracy = (preds == y_test).mean()

    print(f"    Accuracy:  {accuracy:.4f}")
    print(f"    Recall (wiki TPR): {recall:.4f}")
    print(f"    FP rate (C4 → wiki): {fp_rate:.4f}")
    print(f"    Precision: {precision:.4f}")
    print(f"    Train: {n_wiki_train} wiki + {n_c4_train} C4")
    print(f"    Test:  {n_wiki - n_wiki_train} wiki + {n_c4 - n_c4_train} C4")

    return {"accuracy": accuracy, "recall": recall, "fp_rate": fp_rate,
            "precision": precision}


def experiment_vary_NB(model_short, emb_dim, X_wiki, X_c4,
                       n_wiki_pool=5000, N_S=1000, p=0.01, n_trials=10):
    """
    Vary N_B (via n_c4_pool) with fixed N_S=1000 and measure filter quality.
    """
    print(f"\n  Experiment: vary N_B ({model_short}, {emb_dim}d, N_S={N_S})")

    n_c4_values = [50000, 100000, 150000, 200000]
    results = {k: [] for k in [
        "NB", "fn_mean", "fn_std", "fp_mean", "fp_std",
        "prec_mean", "prec_std", "recall_mean", "recall_std",
        "n_passed_mean"
    ]}

    for n_c4_pool in n_c4_values:
        n_wiki_in_B = int(p * n_c4_pool / (1 - p))
        n_wiki_in_B = min(n_wiki_in_B, n_wiki_pool)
        N_B = n_wiki_in_B + n_c4_pool

        fns, fps, precs, recalls, n_pass = [], [], [], [], []

        for trial in range(n_trials):
            rng = np.random.default_rng(800 * trial + n_c4_pool)

            # S from held-out wiki
            s_idx = rng.choice(
                np.arange(n_wiki_pool, len(X_wiki)), N_S, replace=False)
            x_S = X_wiki[s_idx]

            # B pool
            wiki_b_idx = rng.choice(n_wiki_pool, n_wiki_in_B, replace=False)
            c4_idx = rng.choice(len(X_c4), n_c4_pool, replace=False)
            x_B = np.vstack([X_wiki[wiki_b_idx], X_c4[c4_idx]])
            labels_B = np.array([1]*n_wiki_in_B + [0]*n_c4_pool)

            perm = rng.permutation(len(x_B))
            x_B = x_B[perm]
            labels_B = labels_B[perm]

            passed, _ = run_filter(x_S, x_B)
            m = compute_metrics(passed, labels_B)
            fns.append(m["fn_rate"])
            fps.append(m["fp_rate"])
            precs.append(m["precision"])
            recalls.append(m["recall"])
            n_pass.append(m["n_passed"])

        results["NB"].append(N_B)
        results["fn_mean"].append(np.mean(fns))
        results["fn_std"].append(np.std(fns))
        results["fp_mean"].append(np.mean(fps))
        results["fp_std"].append(np.std(fps))
        results["prec_mean"].append(np.mean(precs))
        results["prec_std"].append(np.std(precs))
        results["recall_mean"].append(np.mean(recalls))
        results["recall_std"].append(np.std(recalls))
        results["n_passed_mean"].append(np.mean(n_pass))

        print(f"    N_B={N_B:>7d} (C4={n_c4_pool:>6d}): "
              f"FN={np.mean(fns):.4f}, FP={np.mean(fps):.4f}, "
              f"prec={np.mean(precs):.4f}, recall={np.mean(recalls):.4f}, "
              f"passed={np.mean(n_pass):.0f}")

    data_name = f"embedding_{model_short}_vary_NB"
    save_data(data_name, {
        "n_c4_values": n_c4_values,
        "N_S": N_S,
        "model": model_short,
        "emb_dim": emb_dim,
        "p": p,
        "n_trials": n_trials,
        **results,
    })

    return results


def main():
    t0 = time.time()

    # Load text data (cached after first run)
    n_wiki = 15000   # 5K for B pool + up to 10K held out for S
    n_c4 = 200000

    wiki_texts = load_wikipedia_texts(n_wiki)
    c4_texts = load_c4_texts(n_c4)

    models = [
        ("all-MiniLM-L6-v2", "minilm", 384),
        ("BAAI/bge-base-en-v1.5", "bge", 768),
    ]

    all_results = {}
    all_nb_results = {}

    for model_name, model_short, emb_dim in models:
        print(f"\n{'='*60}")
        print(f"Model: {model_name} ({emb_dim}d)")
        print(f"{'='*60}")

        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)

        X_wiki = load_or_embed(wiki_texts, "wiki", model_short, model)
        X_c4 = load_or_embed(c4_texts, "c4", model_short, model)

        print(f"  Wiki embeddings: {X_wiki.shape}")
        print(f"  C4 embeddings:   {X_c4.shape}")

        oracle_test(X_wiki, X_c4, model_short, emb_dim)

        results = experiment_vary_NS(
            model, model_short, emb_dim,
            X_wiki, X_c4,
            n_wiki_pool=5000, n_c4_pool=200000,
            p=0.01, n_trials=10,
        )
        all_results[model_short] = results

        nb_results = experiment_vary_NB(
            model_short, emb_dim, X_wiki, X_c4,
            n_wiki_pool=5000, N_S=1000, p=0.01, n_trials=10,
        )
        all_nb_results[model_short] = nb_results

    # Combined comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    N_S_values = [20, 50, 100, 200, 500, 1000, 2000, 5000]

    for ax, metric, ylabel, title_suffix in [
        (axes[0], "fp", "False positive rate", "FP rate"),
        (axes[1], "prec", "Precision", "Precision"),
    ]:
        for model_name, model_short, emb_dim in models:
            r = all_results[model_short]
            mean_key = f"{metric}_mean"
            std_key = f"{metric}_std"
            ax.errorbar(N_S_values, r[mean_key], yerr=r[std_key],
                        fmt="o-", capsize=4,
                        label=f"{model_short} ({emb_dim}d)")
        ax.set_xscale("log")
        ax.set_xlabel(r"$N_S$")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Wikipedia vs C4: {title_suffix}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_plot(fig, "embedding_comparison")

    elapsed = time.time() - t0
    print(f"\nAll embedding experiments completed in {elapsed:.1f}s")
    print(f"Plots saved to {PLOTS_DIR.resolve()}")
    print(f"Data saved to {DATA_DIR.resolve()}")


if __name__ == "__main__":
    main()
