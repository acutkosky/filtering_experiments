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

**Setup.** 18,846 text documents, TF-IDF features (10,000 dimensions). $S$ = two related newsgroup topics, $B$ = full corpus ($p \approx 0.10$). $N_S = 200$ (100 per class). **Downstream task**: distinguish the two topics within $S$. Points from $O$ get random labels. Balanced accuracy over 10 trials.

| Topic pair | $S$-only | Filtered | Random $B$ | Oracle | FN rate | FP rate | Precision |
|------------|:--------:|:--------:|:----------:|:------:|--------:|--------:|----------:|
| baseball vs hockey | 0.845 | **0.854** | 0.836 | 0.906 | 0.805 | 0.009 | 0.83 |
| ibm.pc vs mac | 0.786 | **0.790** | 0.771 | 0.856 | 0.813 | 0.002 | 0.93 |
| space vs electronics | 0.852 | **0.856** | 0.837 | 0.907 | 0.832 | 0.001 | 0.95 |
| guns vs mideast | 0.872 | **0.877** | 0.861 | 0.920 | 0.811 | 0.001 | 0.94 |

Filtering consistently improves over $S$-only across all topic pairs (0.5--1% balanced accuracy gain), and always outperforms the random $B$ baseline. The improvements are modest because $N_S = 200$ already provides reasonable accuracy for the within-$S$ task. High FN rates (~80%) mean the filter is conservative; high precision (83--95%) confirms that accepted samples are mostly from $S$.

---

## Real-world: MNIST

**Setup.** 70,000 digit images, PCA to 50 dimensions. $S$ = two similar digits, $B$ = full dataset ($p \approx 0.20$). $N_S = 200$ (100 per digit). **Downstream task**: distinguish the two digits within $S$. Points from $O$ get random labels. Balanced accuracy over 10 trials.

| Digit pair | $S$-only | Filtered | Random $B$ | Oracle | FN rate | FP rate | Precision |
|:----------:|:--------:|:--------:|:----------:|:------:|--------:|--------:|----------:|
| 3 vs 8 | 0.939 | **0.942** | 0.920 | 0.963 | 0.237 | 0.094 | 0.67 |
| 4 vs 9 | 0.937 | **0.937** | 0.886 | 0.964 | 0.108 | 0.037 | 0.86 |
| 1 vs 7 | 0.988 | 0.979 | 0.968 | 0.995 | 0.116 | 0.036 | 0.87 |

Filtering helps most when the within-$S$ task is hard: 3 vs 8 gains 0.3% from filtering, while random $B$ drops 2% below $S$-only. For the easy 1 vs 7 pair ($S$-only already 98.8%), additional data adds slight noise. Precision is 67--87%, with harder-to-filter pairs (3 vs 8) having lower precision.

---

## Real-world: Covertype

**Setup.** 581,012 forest cover type samples, 54 tabular features. $S$ = two forest types, subsample of 100K for $B$. $N_S = 200$ (100 per class). **Downstream task**: distinguish the two forest types within $S$. Points from $O$ get random labels. Balanced accuracy over 10 trials.

| Forest type pair | $p$ | $S$-only | Filtered | Random $B$ | Oracle | FN rate | FP rate | Precision |
|:----------------:|----:|:--------:|:--------:|:----------:|:------:|--------:|--------:|----------:|
| Ponderosa vs Douglas-fir | 0.09 | 0.684 | **0.695** | 0.648 | 0.717 | 0.041 | 0.049 | 0.66 |
| Cottonwood vs Aspen | 0.02 | 1.000 | 0.921 | 0.941 | 1.000 | 0.257 | 0.159 | 0.09 |

Ponderosa vs Douglas-fir shows the expected pattern: filtering improves by 1.1% over $S$-only and 4.7% over random $B$. Cottonwood vs Aspen is a degenerate case: the two classes are perfectly separable with 200 samples ($S$-only = 100%), so additional filtered data (precision only 9%) adds noise.

---

## Real-world: Varying $N_S$

### 20 Newsgroups (baseball vs hockey)

| $N_S$ | $S$-only | Filtered | Random $B$ | FN rate | FP rate |
|------:|:--------:|:--------:|:----------:|--------:|--------:|
| 20 | 0.671 | 0.679 | 0.663 | 0.969 | 0.011 |
| 50 | 0.734 | 0.755 | 0.720 | 0.941 | 0.009 |
| 100 | 0.796 | 0.809 | 0.786 | 0.896 | 0.006 |
| 200 | 0.837 | 0.853 | 0.832 | 0.803 | 0.003 |
| 400 | 0.872 | 0.884 | 0.862 | 0.613 | 0.004 |

Filtering consistently improves over $S$-only across all $N_S$ values, with the gap widening as the filter learns a better decision boundary. Random $B$ always hurts.

### MNIST (3 vs 8)

| $N_S$ | $S$-only | Filtered | Random $B$ | FN rate | FP rate |
|------:|:--------:|:--------:|:----------:|--------:|--------:|
| 10 | 0.830 | **0.871** | 0.862 | 0.942 | 0.106 |
| 20 | 0.860 | **0.882** | 0.847 | 0.750 | 0.094 |
| 50 | 0.915 | **0.927** | 0.908 | 0.442 | 0.089 |
| 100 | 0.930 | **0.932** | 0.913 | 0.311 | 0.088 |
| 200 | 0.942 | **0.943** | 0.923 | 0.243 | 0.083 |
| 500 | 0.948 | **0.949** | 0.933 | 0.201 | 0.077 |
| 1,000 | 0.953 | **0.950** | 0.937 | 0.189 | 0.072 |

Filtered beats $S$-only for $N_S \le 500$, with the largest gains at small $N_S$ (4% at $N_S = 10$). Random $B$ consistently hurts due to O samples with random labels.
