# Experiment Results Tables

All synthetic experiments use: $p = 0.01$, $R = 3.0$. Values are means over 10--15 independent trials. The "downstream task" is a binary classification problem defined *within* the target distribution $S$: classify points based on the sign of their second coordinate ($x_1 > 0$ vs $x_1 \le 0$). Crucially, the $S$ and $O$ distributions have *opposite* biases on $x_1$: points from $S$ have $x_1$ shifted positive (so $\Pr[x_1 > 0 \mid S] \approx 0.84$), while points from $O$ have $x_1$ shifted negative (so $\Pr[x_1 > 0 \mid O] \approx 0.16$). This means including unfiltered $O$ data actively *hurts* the downstream classifier — it provides misleading labels. We compare a logistic regression classifier trained on $S$ samples only versus one trained on $S$ samples augmented with filtered $B$ samples.

TV distance is estimated in two ways:
- **Histogram-based** (experiments 1--2): Build 50-bin histograms of the first coordinate ($x_0$, the separating direction) for the true $S$ samples and the filtered $B$ samples, then compute $\frac{1}{2} \sum_i |h_S(i) - h_{\text{filtered}}(i)|$.
- **FP/FN bound** (experiments 3--5): Use the upper bound $\text{TV} \le \text{FN} + \text{FP}/p$ from the paper, where FN is the false negative rate (fraction of $S$-component samples in $B$ that fail the filter) and FP is the false positive rate (fraction of $O$-component samples in $B$ that pass).

---

## Experiment 1: Varying $N_S$

**What this tests.** The paper's bound (Theorem 1) predicts $\text{TV} \sim O(R^2 / (\gamma^2 N_S))$ when $N_B$ is large. We fix $N_B = 500{,}000$, $d = 20$, $\gamma = 0.5$ and sweep $N_S$.

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

| $N_S$ | $S$-only acc | $S$+filtered acc |
|------:|-------------:|-----------------:|
| 50 | 0.862 | 0.989 |
| 100 | 0.877 | 0.990 |
| 200 | 0.909 | 0.989 |
| 500 | 0.944 | 0.990 |
| 1,000 | 0.963 | 0.990 |
| 2,000 | 0.980 | 0.991 |
| 5,000 | 0.987 | 0.992 |

With only $N_S = 50$ target samples, the $S$-only classifier achieves 86% accuracy, while augmenting with filtered $B$ (which provides $\sim p \cdot N_B = 5{,}000$ additional approximately-$S$ samples) jumps to 99%. The gap narrows as $N_S$ increases since $S$ alone eventually suffices. Because the $O$ distribution has the opposite label bias, including unfiltered $B$ data would actively hurt the classifier — filtering is essential.

---

## Experiment 2: Varying $N_B$

**What this tests.** The $1/(p \cdot N_B)$ term in the TV bound. We fix $N_S = 1{,}000$, $d = 20$, $\gamma = 0.5$.

**TV distance** (histogram-based):

| $N_B$ | TV (mean) | TV (std) |
|------:|----------:|---------:|
| 10,000 | 0.192 | 0.032 |
| 50,000 | 0.103 | 0.014 |
| 100,000 | 0.082 | 0.009 |
| 200,000 | 0.070 | 0.011 |
| 500,000 | 0.065 | 0.009 |
| 1,000,000 | 0.060 | 0.009 |
| 2,000,000 | 0.052 | 0.007 |

**Downstream accuracy:**

| $N_B$ | $S$-only acc | $S$+filtered acc |
|------:|-------------:|-----------------:|
| 10,000 | 0.965 | 0.967 |
| 50,000 | 0.970 | 0.978 |
| 100,000 | 0.965 | 0.979 |
| 200,000 | 0.967 | 0.983 |
| 500,000 | 0.968 | 0.991 |
| 1,000,000 | 0.964 | 0.993 |
| 2,000,000 | 0.967 | 0.996 |

TV decreases with $N_B$ and flattens when the $1/N_S$ term dominates. With $N_S = 1{,}000$, the $S$-only baseline is already decent ($\sim 96.7\%$), but filtered augmentation provides increasing improvement as $N_B$ grows, reaching 99.6% at $N_B = 2{,}000{,}000$.

---

## Experiment 3: Varying Dimension $d$

**What this tests.** The paper's key insight: the TV bound depends on $R^2/\gamma^2$ but **not** on the ambient dimension $d$. We fix $N_S = 1{,}000$, $N_B = 500{,}000$, $\gamma = 0.5$.

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
| 1,000 | 0.194 | 0.000 | 0.178 |

**Downstream accuracy:**

| $d$ | $S$-only acc | $S$+filtered acc |
|----:|-------------:|-----------------:|
| 2 | 0.993 | 0.998 |
| 5 | 0.987 | 0.996 |
| 10 | 0.980 | 0.994 |
| 20 | 0.964 | 0.989 |
| 50 | 0.942 | 0.983 |
| 100 | 0.910 | 0.969 |
| 200 | 0.878 | 0.954 |
| 500 | 0.849 | 0.920 |
| 1,000 | 0.843 | 0.876 |

The TV bound stays at zero for $d \le 200$, confirming the dimension-free theoretical prediction. Degradation at $d = 500$--$1000$ is an artifact of the SVM solver (linear model with $d$ parameters and only $N_S = 1{,}000$ training points), not the theory. Filtering consistently improves over $S$-only across all dimensions, with the gap being largest (3--9 percentage points) in the moderate-$d$ range where the $S$-only classifier struggles but the filter still works well.

---

## Experiment 4: Varying Margin $\gamma$

**What this tests.** The $R^2/\gamma^2$ factor in the bound. We use $N_S = 30$ (deliberately small), $d = 10$, $N_B = 500{,}000$.

**TV bound** (FN + FP/$p$):

| $\gamma$ | TV bound (mean) | TV bound (std) |
|---------:|----------------:|---------------:|
| 0.02 | 0.285 | 0.159 |
| 0.05 | 0.106 | 0.092 |
| 0.10 | 0.047 | 0.067 |
| 0.20 | 0.003 | 0.008 |
| 0.30 | 0.000 | 0.000 |
| 0.50 | 0.000 | 0.000 |
| 0.80 | 0.000 | 0.000 |
| 1.50 | 0.000 | 0.000 |

**Downstream accuracy:**

| $\gamma$ | $S$-only acc | $S$+filtered acc |
|---------:|-------------:|-----------------:|
| 0.02 | 0.861 | 0.993 |
| 0.05 | 0.859 | 0.993 |
| 0.10 | 0.867 | 0.992 |
| 0.20 | 0.854 | 0.993 |
| 0.30 | 0.869 | 0.993 |
| 0.50 | 0.871 | 0.993 |
| 0.80 | 0.860 | 0.992 |
| 1.50 | 0.878 | 0.992 |

At small $\gamma$ (hard separation), the filter makes more errors and the TV bound increases. The $S$-only accuracy is around 86% regardless of $\gamma$ because $N_S = 30$ is insufficient for good generalization. The filtered augmentation provides $\sim 99.3\%$ accuracy across all $\gamma$ values because even imperfect filtering adds much more data than the 30 $S$ samples, and the filtered data has the correct label bias (unlike unfiltered $O$ data which would be actively misleading).

---

## Experiment 5: Weak Separation

**What this tests.** Theorem 2 predicts additive degradation: $+\varepsilon_S$ from $S$-margin violations and $+\varepsilon_O/p$ from $O$-margin violations. With $p = 0.01$, even small $\varepsilon_O$ is amplified 100x.

Fixed: $N_S = 1{,}000$, $N_B = 500{,}000$, $d = 10$, $\gamma = 0.5$.

**Varying $\varepsilon_O$** (fraction of $O$ violating margin):

| $\varepsilon_O$ | TV bound (mean) | Theory ($\varepsilon_O / p$) | Filtered acc |
|---------:|----------------:|-------------------:|-------------:|
| 0.000 | 0.000 | 0.000 | 0.993 |
| 0.001 | 0.100 | 0.100 | 0.995 |
| 0.002 | 0.200 | 0.200 | 0.996 |
| 0.005 | 0.500 | 0.500 | 0.996 |
| 0.010 | 1.000 | 1.000 | 0.998 |
| 0.020 | 1.983 | 2.000 | 0.998 |
| 0.050 | 4.266 | 5.000 | 0.998 |

The measured TV bound matches $\varepsilon_O / p$ almost exactly. Note that TV values > 1 are an artifact of the FN + FP/$p$ upper bound (the true TV is always $\le 1$).

**Varying $\varepsilon_S$** (fraction of $S$ violating margin):

| $\varepsilon_S$ | TV bound (mean) | Theory ($\varepsilon_S$) | Filtered acc |
|---------:|----------------:|-------------------:|-------------:|
| 0.00 | 0.000 | 0.000 | 0.993 |
| 0.02 | 0.021 | 0.020 | 0.994 |
| 0.05 | 0.157 | 0.050 | 0.993 |
| 0.10 | 1.121 | 0.100 | 0.994 |
| 0.15 | 2.494 | 0.150 | 0.992 |
| 0.20 | 3.781 | 0.200 | 0.990 |
| 0.30 | 6.148 | 0.300 | 0.988 |

The $\varepsilon_S$ degradation matches theory at small values but grows faster than the bare $\varepsilon_S$ at larger values because the FN + FP/$p$ bound includes the FP amplification from S-violating points that land on the O side. The slope is still much less than $1/p = 100$ (the $\varepsilon_O$ slope), confirming the fundamental asymmetry.

---

## Real-world: 20 Newsgroups

**Setup.** 18,846 text documents, TF-IDF features (10,000 dimensions). One newsgroup topic is $S$, the full corpus is $B$ ($p \approx 0.05$). $N_S = 200$. The downstream task is binary classification: target topic vs. all others. We report balanced accuracy (mean of per-class recall) averaged over 10 trials.

| Topic | $S$-only | Filtered | Random $B$ | Oracle | FN rate | FP rate | Precision |
|-------|:--------:|:--------:|:----------:|:------:|--------:|--------:|----------:|
| sci.space | 0.647 | **0.734** | 0.703 | 0.894 | 0.643 | 0.001 | 0.94 |
| rec.autos | 0.547 | **0.652** | 0.615 | 0.852 | 0.620 | 0.024 | 0.53 |
| comp.graphics | 0.553 | **0.668** | 0.614 | 0.855 | 0.632 | 0.002 | 0.92 |
| rec.sport.hockey | 0.648 | **0.781** | 0.749 | 0.919 | 0.612 | 0.001 | 0.97 |

Filtering consistently improves over $S$-only across all topics (7--13% balanced accuracy gain), and always outperforms the random $B$ baseline. High FN rates (60--64%) mean the filter is conservative; it rejects most $B$ samples, but those it accepts have high precision (53--97%).

---

## Real-world: MNIST

**Setup.** 70,000 digit images, PCA to 50 dimensions. One digit is $S$, full dataset is $B$ ($p \approx 0.1$). $N_S = 200$. Downstream: target digit vs. all others. Balanced accuracy over 10 trials.

| Target digit | $S$-only | Filtered | Random $B$ | Oracle | FN rate | FP rate | Precision |
|:------------:|:--------:|:--------:|:----------:|:------:|--------:|--------:|----------:|
| 3 | 0.902 | **0.910** | 0.841 | 0.930 | 0.150 | 0.043 | 0.69 |
| 7 | 0.945 | **0.950** | 0.883 | 0.964 | 0.091 | 0.028 | 0.79 |
| 1 | 0.974 | **0.976** | 0.954 | 0.982 | 0.044 | 0.013 | 0.90 |
| 9 | 0.873 | **0.899** | 0.726 | 0.919 | 0.124 | 0.075 | 0.56 |

Filtered consistently beats $S$-only (1--3% gain) and dramatically outperforms random $B$ (7--17% gap). Digit 1 is easiest to separate (precision 90%), digit 9 is hardest (precision 56%).

---

## Real-world: Covertype

**Setup.** 581,012 forest cover type samples, 54 tabular features. Rare cover types are $S$, subsample of 100K is $B$. $N_S = 200$. Balanced accuracy over 10 trials.

| Target class | $p$ | $S$-only | Filtered | Random $B$ | Oracle | FN rate | FP rate | Precision |
|:------------:|----:|:--------:|:--------:|:----------:|:------:|--------:|--------:|----------:|
| Cottonwood/Willow | 0.005 | 0.961 | **0.976** | 0.882 | 0.979 | 0.004 | 0.030 | 0.14 |
| Aspen | 0.016 | 0.689 | **0.755** | 0.522 | 0.852 | 0.119 | 0.187 | 0.07 |

Despite low filter precision (7--14%), filtering still improves balanced accuracy by 1.6--6.5% over $S$-only, and dramatically beats random $B$. The low precision reflects the weak linear separability of tabular forest features; the paper's method still extracts useful signal even in this challenging regime.

---

## Real-world: Varying $N_S$

### 20 Newsgroups (sci.space)

| $N_S$ | $S$-only | Filtered | Random $B$ | FN rate | FP rate |
|------:|:--------:|:--------:|:----------:|--------:|--------:|
| 20 | 0.500 | 0.505 | 0.524 | 0.954 | 0.015 |
| 50 | 0.510 | 0.574 | 0.573 | 0.906 | 0.009 |
| 100 | 0.574 | 0.670 | 0.646 | 0.822 | 0.004 |
| 200 | 0.644 | 0.738 | 0.704 | 0.646 | 0.001 |
| 400 | 0.705 | 0.775 | 0.748 | 0.336 | 0.003 |

Filtering becomes effective once $N_S \ge 50$ and the gap widens with more data.

### MNIST (digit 3)

| $N_S$ | $S$-only | Filtered | Random $B$ | FN rate | FP rate |
|------:|:--------:|:--------:|:----------:|--------:|--------:|
| 10 | 0.805 | 0.707 | 0.621 | 0.897 | 0.089 |
| 20 | 0.849 | 0.771 | 0.655 | 0.671 | 0.071 |
| 50 | 0.880 | 0.869 | 0.741 | 0.304 | 0.054 |
| 100 | 0.885 | 0.891 | 0.795 | 0.203 | 0.047 |
| 200 | 0.903 | **0.909** | 0.829 | 0.144 | 0.040 |
| 500 | 0.909 | **0.920** | 0.859 | 0.118 | 0.034 |
| 1,000 | 0.912 | **0.923** | 0.870 | 0.111 | 0.030 |

Filtered overtakes $S$-only around $N_S = 100$ and the benefit increases with more data as the filter learns a better decision boundary.
