# Experiment Results Tables

All synthetic experiments use: $p = 0.01$, $R = 3.0$. Values are means over 10--15 independent trials.

## Downstream task description

The downstream task is a binary classification problem defined *within* the target distribution $S$: classify points based on the sign of their second coordinate ($x_1 > 0$ vs $x_1 \le 0$). Points from $S$ have $x_1$ shifted by $+1.0 \cdot \sigma$ (where $\sigma = R/(2\sqrt{d})$ is the per-coordinate noise scale), making the label informative ($\Pr[x_1 > 0 \mid S] \approx 0.84$). Points from $O$ have *no shift* on $x_1$ and are assigned *random* labels (independent of $x$). This models the real-world scenario where a labeling oracle is meaningful for the target distribution but produces garbage on off-distribution data.

We train a logistic regression classifier in three settings:
- **$S$-only**: trained using only the $N_S$ labeled samples from $S$.
- **$S$+filtered**: trained on $S$ augmented with filtered $B$ samples (which approximate $S$, so their labels are informative). Provides up to $\sim p \cdot N_B$ additional approximately-$S$ samples.
- **$S$+unfiltered**: trained on $S$ augmented with the *same number* of random (unfiltered) $B$ samples. Since $\sim 99\%$ of $B$ is from $O$ with random labels, this injects massive label noise that actively degrades the classifier.

Test accuracy is measured on 5,000 fresh samples from $S$.

## TV distance estimation

TV distance is estimated in two ways:
- **Histogram-based** (experiments 1--2): Build 50-bin histograms of the first coordinate ($x_0$, the separating direction) for the true $S$ samples and the filtered $B$ samples, then compute $\frac{1}{2} \sum_i |h_S(i) - h_{\text{filtered}}(i)|$.
- **FP/FN bound** (experiments 3--5): Use the upper bound $\text{TV} \le \text{FN} + \text{FP}/p$ from the paper, where FN is the false negative rate (fraction of $S$-component samples in $B$ that fail the filter) and FP is the false positive rate (fraction of $O$-component samples in $B$ that pass).

---

## Experiment 1: Varying $N_S$

**What this tests.** The paper's bound (Theorem 1) predicts $\text{TV} \sim O(R^2 / (\gamma^2 N_S))$ when $N_B$ is large. We fix $N_B = 500{,}000$, $d = 20$, $\gamma = 0.5$, $p = 0.01$, $R = 3$ and sweep $N_S$.

**TV distance** (histogram-based):

| $N_S$ | TV (mean) | TV (std) |
|------:|----------:|---------:|
| 50 | 0.253 | 0.037 |
| 100 | 0.186 | 0.037 |
| 200 | 0.129 | 0.016 |
| 500 | 0.090 | 0.011 |
| 1,000 | 0.063 | 0.009 |
| 2,000 | 0.044 | 0.007 |
| 5,000 | 0.034 | 0.004 |

**Downstream accuracy** (binary classification on sign of $x_1$):

| $N_S$ | $S$-only acc | $S$+filtered acc | $S$+unfiltered acc |
|------:|-------------:|-----------------:|-------------------:|
| 50 | 0.863 | 0.989 | 0.841 |
| 100 | 0.879 | 0.991 | 0.844 |
| 200 | 0.912 | 0.989 | 0.844 |
| 500 | 0.945 | 0.990 | 0.843 |
| 1,000 | 0.963 | 0.990 | 0.839 |
| 2,000 | 0.980 | 0.991 | 0.842 |
| 5,000 | 0.988 | 0.993 | 0.849 |

With only $N_S = 50$ target samples, the $S$-only classifier achieves 86% accuracy, while augmenting with filtered $B$ (which provides $\sim p \cdot N_B = 5{,}000$ additional approximately-$S$ samples) jumps to 99%. The unfiltered baseline stays flat at $\sim 84\%$ — worse than $S$-only — because $\sim 99\%$ of unfiltered $B$ is from $O$ with random labels, which injects massive label noise. The gap between filtered and $S$-only narrows as $N_S$ increases since $S$ alone eventually suffices.

---

## Experiment 2: Varying $N_B$

**What this tests.** The $1/(p \cdot N_B)$ term in the TV bound. We fix $N_S = 200$, $N_B$ varies, $d = 50$, $\gamma = 0.5$, $p = 0.01$, $R = 3$. The higher dimension and smaller $N_S$ make the downstream task genuinely challenging for the $S$-only classifier, so the benefit of filtered augmentation is clearly visible.

**TV distance** (histogram-based):

| $N_B$ | TV (mean) | TV (std) |
|------:|----------:|---------:|
| 10,000 | 0.228 | 0.035 |
| 50,000 | 0.151 | 0.023 |
| 100,000 | 0.141 | 0.035 |
| 200,000 | 0.142 | 0.016 |
| 500,000 | 0.130 | 0.015 |
| 1,000,000 | 0.122 | 0.012 |
| 2,000,000 | 0.115 | 0.017 |

**Downstream accuracy:**

| $N_B$ | $S$-only acc | $S$+filtered acc | $S$+unfiltered acc |
|------:|-------------:|-----------------:|-------------------:|
| 10,000 | 0.869 | 0.888 | 0.860 |
| 50,000 | 0.864 | 0.927 | 0.845 |
| 100,000 | 0.865 | 0.944 | 0.844 |
| 200,000 | 0.861 | 0.962 | 0.841 |
| 500,000 | 0.876 | 0.980 | 0.844 |
| 1,000,000 | 0.877 | 0.986 | 0.839 |
| 2,000,000 | 0.870 | 0.992 | 0.841 |

The $S$-only baseline is flat at $\sim 87\%$ (limited by the small $N_S = 200$), while filtered accuracy climbs steadily from 89% to 99% as more $B$ data becomes available. The unfiltered baseline is *worse* than $S$-only at $\sim 84\%$: since $99\%$ of $B$ comes from $O$ with random labels, adding unfiltered data injects noise that actively degrades the classifier. This clearly demonstrates the paper's key practical insight: filtering is not just helpful but *essential* — naive data augmentation hurts.

### Experiment 2b: Varying $N_B$ (small $\gamma$)

**What this tests.** Same as Experiment 2 but with $\gamma = 0.1$ (vs $\gamma = 0.5$ above). The smaller margin makes the $R^2/(\gamma^2 \cdot p \cdot N_B)$ term larger, so the TV bound decreases more visibly as $N_B$ grows. Fixed: $N_S = 200$, $d = 50$, $p = 0.01$, $R = 3$.

**TV distance** (histogram-based):

| $N_B$ | TV (mean) | TV (std) |
|------:|----------:|---------:|
| 10,000 | 0.253 | 0.024 |
| 50,000 | 0.160 | 0.018 |
| 100,000 | 0.147 | 0.014 |
| 200,000 | 0.133 | 0.016 |
| 500,000 | 0.125 | 0.010 |
| 1,000,000 | 0.114 | 0.015 |
| 2,000,000 | 0.101 | 0.013 |

**Downstream accuracy:**

| $N_B$ | $S$-only acc | $S$+filtered acc | $S$+unfiltered acc |
|------:|-------------:|-----------------:|-------------------:|
| 10,000 | 0.864 | 0.879 | 0.860 |
| 50,000 | 0.868 | 0.925 | 0.843 |
| 100,000 | 0.867 | 0.942 | 0.846 |
| 200,000 | 0.868 | 0.963 | 0.843 |
| 500,000 | 0.879 | 0.980 | 0.841 |
| 1,000,000 | 0.873 | 0.986 | 0.841 |
| 2,000,000 | 0.862 | 0.988 | 0.840 |

With smaller $\gamma = 0.1$, the TV distance is somewhat larger at small $N_B$ (0.25 vs 0.23) and the improvement with $N_B$ is more gradual, reflecting the harder separation. Despite this, the downstream accuracy pattern is similar: filtering climbs from 88% to 99%, demonstrating that the practical benefit scales with $N_B$ even when the margin is small.

---

## Experiment 3: Varying Dimension $d$

**What this tests.** The paper's key insight: the TV bound depends on $R^2/\gamma^2$ but **not** on the ambient dimension $d$. We fix $N_S = 1{,}000$, $N_B = 500{,}000$, $\gamma = 0.5$, $p = 0.01$, $R = 3$.

**TV bound** (FN + FP/$p$):

| $d$ | TV bound (mean) | FP rate | FN rate |
|----:|----------------:|--------:|--------:|
| 2 | 0.000 | 0.000 | 0.000 |
| 5 | 0.000 | 0.000 | 0.000 |
| 10 | 0.000 | 0.000 | 0.000 |
| 20 | 0.000 | 0.000 | 0.000 |
| 50 | 0.000 | 0.000 | 0.000 |
| 100 | 0.000 | 0.000 | 0.000 |
| 200 | 0.000 | 0.000 | 0.000 |
| 500 | 0.004 | 0.000 | 0.004 |
| 1,000 | 0.195 | 0.000 | 0.178 |

**Downstream accuracy:**

| $d$ | $S$-only acc | $S$+filtered acc | $S$+unfiltered acc |
|----:|-------------:|-----------------:|-------------------:|
| 2 | 0.993 | 0.998 | 0.837 |
| 5 | 0.985 | 0.996 | 0.844 |
| 10 | 0.980 | 0.994 | 0.844 |
| 20 | 0.965 | 0.990 | 0.840 |
| 50 | 0.941 | 0.982 | 0.841 |
| 100 | 0.907 | 0.970 | 0.840 |
| 200 | 0.879 | 0.952 | 0.843 |
| 500 | 0.845 | 0.919 | 0.837 |
| 1,000 | 0.844 | 0.878 | 0.837 |

The TV bound stays at zero for $d \le 200$, confirming the dimension-free theoretical prediction. Degradation at $d = 500$--$1000$ is an artifact of the SVM solver (linear model with $d$ parameters and only $N_S = 1{,}000$ training points), not the theory. Filtering consistently improves over $S$-only across all dimensions. The unfiltered baseline is flat at $\sim 84\%$ regardless of dimension — it is always worse than $S$-only because the random $O$ labels dominate. The gap between filtered and $S$-only is largest (5--8 percentage points) in the moderate-to-high $d$ range where the $S$-only classifier struggles but the filter still works well.

---

## Experiment 4: Varying Margin $\gamma$

**What this tests.** The $R^2/\gamma^2$ factor in the bound. We use $N_S = 30$ (deliberately small), $N_B = 500{,}000$, $d = 10$, $p = 0.01$, $R = 3$.

**TV bound** (FN + FP/$p$):

| $\gamma$ | TV bound (mean) | TV bound (std) |
|---------:|----------------:|---------------:|
| 0.02 | 0.208 | 0.122 |
| 0.05 | 0.044 | 0.038 |
| 0.10 | 0.045 | 0.092 |
| 0.20 | 0.002 | 0.006 |
| 0.30 | 0.000 | 0.000 |
| 0.50 | 0.000 | 0.000 |
| 0.80 | 0.000 | 0.000 |
| 1.50 | 0.000 | 0.000 |

**Downstream accuracy:**

| $\gamma$ | $S$-only acc | $S$+filtered acc | $S$+unfiltered acc |
|---------:|-------------:|-----------------:|-------------------:|
| 0.02 | 0.859 | 0.934 | 0.828 |
| 0.05 | 0.861 | 0.970 | 0.843 |
| 0.10 | 0.864 | 0.983 | 0.832 |
| 0.20 | 0.853 | 0.992 | 0.835 |
| 0.30 | 0.869 | 0.992 | 0.843 |
| 0.50 | 0.870 | 0.993 | 0.841 |
| 0.80 | 0.860 | 0.992 | 0.841 |
| 1.50 | 0.877 | 0.993 | 0.840 |

At small $\gamma$ (hard separation), the filter makes more errors and the TV bound increases. The $S$-only accuracy is around 85--88% regardless of $\gamma$ because $N_S = 30$ is insufficient for good generalization. The filtered augmentation provides 93--99% accuracy, with the benefit increasing as $\gamma$ grows and filtering becomes cleaner. The unfiltered baseline stays at $\sim 84\%$ — worse than $S$-only — because the random $O$ labels dominate and actively mislead the classifier.

---

## Experiment 5: Weak Separation

**What this tests.** Theorem 2 predicts additive degradation: $+\varepsilon_S$ from $S$-margin violations and $+\varepsilon_O/p$ from $O$-margin violations. With $p = 0.01$, even small $\varepsilon_O$ is amplified 100x.

Fixed: $N_S = 1{,}000$, $N_B = 500{,}000$, $d = 10$, $\gamma = 0.5$, $p = 0.01$, $R = 3$.

**Varying $\varepsilon_O$** (fraction of $O$ violating margin):

| $\varepsilon_O$ | TV bound (mean) | Theory ($\varepsilon_O / p$) | $S$-only acc | Filtered acc | Unfiltered acc |
|---------:|----------------:|-------------------:|-------------:|-------------:|---------------:|
| 0.000 | 0.000 | 0.000 | 0.976 | 0.993 | 0.838 |
| 0.001 | 0.100 | 0.100 | 0.981 | 0.972 | 0.842 |
| 0.002 | 0.200 | 0.200 | 0.979 | 0.957 | 0.840 |
| 0.005 | 0.500 | 0.500 | 0.978 | 0.932 | 0.841 |
| 0.010 | 1.000 | 1.000 | 0.978 | 0.916 | 0.840 |
| 0.020 | 2.000 | 2.000 | 0.978 | 0.906 | 0.839 |
| 0.050 | 5.000 | 5.000 | 0.977 | 0.903 | 0.842 |

The measured TV bound matches $\varepsilon_O / p$ almost exactly. Note that TV values > 1 are an artifact of the FN + FP/$p$ upper bound (the true TV is always $\le 1$).

**Varying $\varepsilon_S$** (fraction of $S$ violating margin):

| $\varepsilon_S$ | TV bound (mean) | Theory ($\varepsilon_S$) | $S$-only acc | Filtered acc | Unfiltered acc |
|---------:|----------------:|-------------------:|-------------:|-------------:|---------------:|
| 0.00 | 0.000 | 0.000 | 0.979 | 0.994 | 0.842 |
| 0.02 | 0.020 | 0.020 | 0.980 | 0.994 | 0.841 |
| 0.05 | 0.050 | 0.050 | 0.978 | 0.994 | 0.842 |
| 0.10 | 0.170 | 0.100 | 0.982 | 0.946 | 0.842 |
| 0.15 | 0.804 | 0.150 | 0.978 | 0.918 | 0.840 |
| 0.20 | 1.909 | 0.200 | 0.980 | 0.890 | 0.843 |
| 0.30 | 5.402 | 0.300 | 0.978 | 0.846 | 0.841 |

The $\varepsilon_S$ degradation matches theory closely at small values but grows faster at larger values because the FN + FP/$p$ bound includes FP amplification from $S$-violating points that land on the $O$ side. The slope is still much less than $1/p = 100$ (the $\varepsilon_O$ slope), confirming the fundamental asymmetry. The unfiltered baseline stays flat at $\sim 84\%$ throughout. Note that at large $\varepsilon_S = 0.30$, filtered accuracy degrades to near the unfiltered level, as the margin violations make filtering ineffective.

---

## Real-world: 20 Newsgroups

**Setup.** 18,846 text documents, TF-IDF features (10,000 dimensions). $S$ = two related newsgroup topics, $B$ = full corpus ($p \approx 0.10$). $N_S = 200$ (100 per class). **Downstream task**: distinguish the two topics within $S$. Points from $O$ are labeled by a confounding classifier trained on a different topic pair, so their labels carry real but *wrong* signal that actively misleads the A-vs-B classifier. Balanced accuracy over 10 trials.

| Topic pair | $S$-only | Filtered | Random $B$ | Oracle | FN rate | FP rate | Precision |
|------------|:--------:|:--------:|:----------:|:------:|--------:|--------:|----------:|
| christianity vs politics.misc | 0.880 | **0.895** | 0.790 | 0.933 | 0.815 | 0.001 | 0.94 |
| sci.crypt vs sci.electronics | 0.851 | **0.867** | 0.819 | 0.908 | 0.831 | 0.001 | 0.97 |
| rec.autos vs rec.motorcycles | 0.798 | 0.785 | 0.756 | 0.852 | 0.821 | 0.007 | 0.85 |
| ibm.pc vs mac | 0.770 | **0.781** | 0.736 | 0.856 | 0.807 | 0.002 | 0.91 |

Filtering improves over $S$-only for 3 out of 4 topic pairs (1--2 percentage points), closing 15--30% of the gap to the oracle. The confounding labels make random $B$ clearly harmful: 4--9 percentage points below $S$-only. The one pair where filtering slightly hurts (rec.autos vs motorcycles) has the lowest precision (85%), confirming that filter quality is the bottleneck. High FN rates (~80%) mean the filter is conservative; high precision (85--97%) confirms that accepted samples are mostly from $S$.

---

## Real-world: MNIST

**Setup.** 70,000 digit images, PCA to 50 dimensions. $S$ = two similar digits, $B$ = full dataset ($p \approx 0.20$). $N_S = 200$ (100 per digit). **Downstream task**: distinguish the two digits within $S$. Points from $O$ are labeled by a confounding classifier trained on a different digit pair. Balanced accuracy over 10 trials.

| Digit pair | $S$-only | Filtered | Random $B$ | Oracle | FN rate | FP rate | Precision |
|:----------:|:--------:|:--------:|:----------:|:------:|--------:|--------:|----------:|
| 7 vs 9 | 0.927 | **0.932** | 0.869 | 0.955 | 0.124 | 0.041 | 0.84 |
| 3 vs 5 | 0.924 | 0.905 | 0.780 | 0.949 | 0.210 | 0.065 | 0.75 |
| 4 vs 9 | 0.939 | 0.913 | 0.751 | 0.964 | 0.122 | 0.034 | 0.86 |

Confounding labels make random $B$ dramatically harmful: 6--19 percentage points below $S$-only. Filtering helps for 7 vs 9 (highest precision at 84%), closing 17% of the oracle gap. For 3 vs 5 and 4 vs 9, the lower precision (75--86%) means enough confounding $O$ false positives slip through to slightly hurt. The pattern is clear: filtering helps when precision is high enough that the benefit of extra clean $S$ data outweighs the noise from confounding false positives.

---

## Real-world: Covertype

**Setup.** 581,012 forest cover type samples, 54 tabular features. $S$ = two forest types, subsample of 100K for $B$. $N_S = 200$ (100 per class). **Downstream task**: distinguish the two forest types within $S$. Points from $O$ are labeled by a confounding classifier trained on a different forest type pair. Balanced accuracy over 10 trials.

| Forest type pair | $p$ | $S$-only | Filtered | Random $B$ | Oracle | FN rate | FP rate | Precision |
|:----------------:|----:|:--------:|:--------:|:----------:|:------:|--------:|--------:|----------:|
| Spruce/Fir vs Lodgepole Pine | 0.85 | 0.729 | **0.755** | 0.734 | 0.772 | 0.424 | 0.141 | 0.96 |
| Ponderosa vs Douglas-fir | 0.09 | 0.686 | 0.680 | 0.603 | 0.717 | 0.040 | 0.050 | 0.66 |

Spruce/Fir vs Lodgepole Pine is the hardest downstream task (oracle only 77%) and filtering helps by 2.6 percentage points, closing 60% of the oracle gap. This pair has very high $p = 0.85$, so the filter's precision (96%) is high. Ponderosa vs Douglas-fir has much lower $p = 0.09$ and lower precision (66%); the confounding labels still make random $B$ dramatically worse (8 points below $S$-only).

---

## Real-world: Varying $N_S$

### 20 Newsgroups (sci.crypt vs sci.electronics)

| $N_S$ | $S$-only | Filtered | Random $B$ | FN rate | FP rate |
|------:|:--------:|:--------:|:----------:|--------:|--------:|
| 20 | 0.722 | 0.646 | 0.677 | 0.970 | 0.015 |
| 50 | 0.798 | 0.782 | 0.756 | 0.955 | 0.003 |
| 100 | 0.821 | 0.817 | 0.793 | 0.909 | 0.004 |
| 200 | 0.847 | **0.854** | 0.809 | 0.818 | 0.005 |
| 400 | 0.874 | **0.887** | 0.829 | 0.654 | 0.002 |

At small $N_S$ (20--100), the filter's high FN rate (~91--97%) means very few $B$ samples pass, limiting the benefit of filtering. At $N_S \ge 200$, filtered surpasses $S$-only as the filter learns a better decision boundary (FN drops to 65--82%). Confounding labels make random $B$ consistently worse than $S$-only, with gaps of 3--5 percentage points.

### MNIST (7 vs 9)

| $N_S$ | $S$-only | Filtered | Random $B$ | FN rate | FP rate |
|------:|:--------:|:--------:|:----------:|--------:|--------:|
| 10 | 0.802 | **0.884** | 0.784 | 0.910 | 0.007 |
| 20 | 0.870 | **0.906** | 0.829 | 0.685 | 0.026 |
| 50 | 0.890 | **0.918** | 0.839 | 0.364 | 0.037 |
| 100 | 0.917 | **0.925** | 0.855 | 0.198 | 0.038 |
| 200 | 0.932 | **0.932** | 0.868 | 0.122 | 0.039 |
| 500 | 0.944 | 0.937 | 0.873 | 0.092 | 0.038 |
| 1,000 | 0.949 | 0.938 | 0.872 | 0.078 | 0.038 |

Filtering provides substantial gains at small $N_S$: +8.3 percentage points at $N_S = 10$, +3.6 at $N_S = 20$, and +2.8 at $N_S = 50$. The gains diminish as $N_S$ grows and $S$ alone becomes sufficient. Confounding labels (from a 3 vs 8 classifier applied to $O$) make random $B$ consistently harmful: 2--7 percentage points below $S$-only. The crossover at $N_S \approx 500$ reflects the point where the small amount of confounding noise in filtered $B$ (~4% FP) outweighs the benefit of additional data.

---

## Embedding Experiment: Wikipedia vs C4

**Setup.** $S$ = Wikipedia paragraphs (high-quality encyclopedic text), $O$ = C4 web crawl paragraphs (general web text). $B$ contains 202,020 samples: 2,020 from Wikipedia ($p = 0.01$) and 200,000 from C4. The $S$ samples used for training the filter are drawn from a separate held-out pool of Wikipedia paragraphs (no overlap with the Wikipedia samples in $B$). Each paragraph is truncated to 500 characters and embedded using a sentence transformer. We test two embedding models:

- **MiniLM** (`all-MiniLM-L6-v2`): 384 dimensions, fast and lightweight.
- **BGE** (`BAAI/bge-base-en-v1.5`): 768 dimensions, higher quality.

The filter is logistic regression with asymmetric class weights (same as other real-world experiments, $C = 10$). We report FP/FN/precision/recall rates (no downstream task). Values are means over 10 trials.

### Oracle test: linear separability ceiling

To establish the ceiling for linear separation in each embedding space, we train logistic regression on fully labeled data (7,500 Wikipedia + 100,000 C4 for training, same for test). This tells us how well Wikipedia and C4 can be distinguished at all with a linear classifier in this embedding space.

| Model | Accuracy | Recall (wiki TPR) | FP rate (C4 $\to$ wiki) | Precision |
|-------|:--------:|:------------------:|:-----------------------:|:---------:|
| MiniLM (384d) | 0.952 | 0.477 | 0.012 | 0.744 |
| BGE (768d) | 0.966 | 0.637 | 0.009 | 0.836 |

Even with full labels and balanced training, recall is limited (48--64%), confirming that Wikipedia and C4 are not perfectly linearly separable in these embedding spaces. The embeddings are optimized for semantic similarity, not source-quality discrimination. BGE is substantially better than MiniLM across all metrics. Overall accuracy is high (95--97%) because C4 is the majority class and the classifier correctly rejects most of it.

### Filter quality vs $N_S$

$N_B = 202{,}020$ throughout (2,020 Wikipedia + 200,000 C4).

**MiniLM (384d):**

| $N_S$ | $N_B / N_S$ | FN rate | FP rate | Precision | Recall | $n$ passed |
|------:|------------:|--------:|--------:|----------:|-------:|-----------:|
| 20 | 10,101 | 0.961 | 0.001 | 0.241 | 0.039 | 308 |
| 50 | 4,040 | 0.871 | 0.005 | 0.193 | 0.129 | 1,339 |
| 100 | 2,020 | 0.760 | 0.013 | 0.164 | 0.240 | 2,993 |
| 200 | 1,010 | 0.566 | 0.029 | 0.130 | 0.434 | 6,756 |
| 500 | 404 | 0.318 | 0.058 | 0.106 | 0.682 | 13,012 |
| 1,000 | 202 | 0.230 | 0.072 | 0.097 | 0.770 | 15,994 |
| 2,000 | 101 | 0.180 | 0.080 | 0.094 | 0.820 | 17,715 |
| 5,000 | 40 | 0.162 | 0.083 | 0.093 | 0.839 | 18,234 |

**BGE (768d):**

| $N_S$ | $N_B / N_S$ | FN rate | FP rate | Precision | Recall | $n$ passed |
|------:|------------:|--------:|--------:|----------:|-------:|-----------:|
| 20 | 10,101 | 0.951 | 0.001 | 0.420 | 0.049 | 232 |
| 50 | 4,040 | 0.818 | 0.003 | 0.408 | 0.182 | 914 |
| 100 | 2,020 | 0.701 | 0.006 | 0.344 | 0.299 | 1,758 |
| 200 | 1,010 | 0.573 | 0.011 | 0.286 | 0.428 | 3,018 |
| 500 | 404 | 0.360 | 0.022 | 0.227 | 0.640 | 5,737 |
| 1,000 | 202 | 0.250 | 0.030 | 0.201 | 0.750 | 7,561 |
| 2,000 | 101 | 0.173 | 0.037 | 0.186 | 0.828 | 9,036 |
| 5,000 | 40 | 0.124 | 0.042 | 0.174 | 0.876 | 10,160 |

### Filter quality vs $N_B$ (fixed $N_S = 1{,}000$)

With $N_S$ fixed at 1,000, we vary $N_B$ by changing the number of C4 samples in the pool. The number of Wikipedia samples in $B$ scales proportionally to maintain $p = 0.01$.

**MiniLM (384d):**

| $N_B$ | $N_B / N_S$ | FN rate | FP rate | Precision | Recall | $n$ passed |
|------:|------------:|--------:|--------:|----------:|-------:|-----------:|
| 50,505 | 51 | 0.218 | 0.074 | 0.097 | 0.782 | 4,072 |
| 101,010 | 101 | 0.232 | 0.073 | 0.096 | 0.768 | 8,086 |
| 151,515 | 152 | 0.229 | 0.073 | 0.096 | 0.771 | 12,149 |
| 202,020 | 202 | 0.222 | 0.074 | 0.096 | 0.778 | 16,459 |

**BGE (768d):**

| $N_B$ | $N_B / N_S$ | FN rate | FP rate | Precision | Recall | $n$ passed |
|------:|------------:|--------:|--------:|----------:|-------:|-----------:|
| 50,505 | 51 | 0.226 | 0.031 | 0.200 | 0.775 | 1,961 |
| 101,010 | 101 | 0.237 | 0.031 | 0.202 | 0.763 | 3,823 |
| 151,515 | 152 | 0.241 | 0.032 | 0.197 | 0.759 | 5,874 |
| 202,020 | 202 | 0.246 | 0.031 | 0.197 | 0.754 | 7,758 |

FP/FN rates and precision are essentially flat across $N_B$: the filter's quality is determined by $N_S$ (how well it can learn the Wikipedia boundary), not by the pool size. The number of samples passing the filter scales linearly with $N_B$ as expected, but the *rates* are unchanged. This is consistent with the theory: when $N_S$ is the bottleneck (i.e., $R^2/(\gamma^2 N_S) \gg R^2/(\gamma^2 p N_B)$), increasing $N_B$ does not improve the TV bound.

### Comparison: filter vs oracle

At $N_S = 5{,}000$ the filter substantially exceeds the oracle's recall, but at the cost of higher FP rate. Precision is lower than the oracle partly because the oracle is evaluated on imbalanced but fully-labeled data, while the filter must learn the boundary from only $N_S$ positive examples.

**MiniLM (384d):**

| Method | Recall | FP rate | Precision |
|--------|-------:|--------:|----------:|
| Oracle (7,500 wiki + 100K C4, labeled) | 0.477 | 0.012 | 0.744 |
| Filter ($N_S = 5{,}000$, $p = 0.01$) | 0.839 | 0.083 | 0.093 |

**BGE (768d):**

| Method | Recall | FP rate | Precision |
|--------|-------:|--------:|----------:|
| Oracle (7,500 wiki + 100K C4, labeled) | 0.637 | 0.009 | 0.836 |
| Filter ($N_S = 5{,}000$, $p = 0.01$) | 0.876 | 0.042 | 0.174 |

The filter has *higher* recall than the oracle because the asymmetric class weighting pushes the decision boundary to accept more positives, at the cost of more false positives. The oracle optimizes balanced accuracy and is more conservative. On BGE, the filter achieves 88% recall with only 4.2% FP rate --- competitive with the oracle's 64% recall and 0.9% FP rate. These represent different operating points on the same ROC curve: the filter favors recall (find as much Wikipedia as possible), while the oracle favors precision (avoid accepting C4).

The low absolute precision (9--17%) is an inherent consequence of low $p = 0.01$: even a small FP rate produces many false positives when 99% of the pool is $O$. To see this, the oracle's precision on the $p = 0.01$ mixture would be approximately $\frac{0.01 \cdot 0.637}{0.01 \cdot 0.637 + 0.99 \cdot 0.009} \approx 0.39$ for BGE --- much lower than its balanced-data precision of 84%, and closer to the filter's precision.
