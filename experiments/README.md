# Synthetic Experiments for "Dimension-Free Data Filtering"

Synthetic experiments validating the theoretical results of the paper. We implement the filtering algorithm using a linear SVM as a practical proxy for the paper's constrained optimization, and measure both the filtering quality (TV distance, FP/FN rates) and the benefit for a downstream classification task.

## Quick start

```bash
cd experiments
uv run python run_experiments.py
```

Plots are saved to `plots/`, raw data to `data/` (as JSON for easy re-plotting). Total runtime is ~15–20 minutes on a MacBook.

## Problem setup

- **S** (target): A small high-quality distribution. We have N_S i.i.d. samples.
- **B** (big): A large mixed distribution B = p S + (1-p) O, with N_B samples. Here p = 0.01 (only 1% of B comes from S).
- **O** (other): An unknown noise/irrelevant distribution.

**Goal**: Design a filter f such that B conditioned on f(x) = 1 is close to S in total variation distance.

**Algorithm**: The paper solves a constrained optimization: find w (||w|| <= 1) that classifies all S samples as positive (with margin gamma) while minimizing the number of B samples classified as positive. We approximate this with sklearn's `LinearSVC` using asymmetric class weights (weight ratio N_B/N_S on the positive class).

**Downstream task**: To measure practical benefit, we define a binary classification problem *within* the S distribution (classify based on sign of the second coordinate x[1]). Points from S have x[1] shifted by +1.0 * noise_scale, making the label informative (P(x[1]>0|S) ≈ 0.84). Points from O have *no shift* on x[1] and are assigned *random* Bernoulli(0.5) labels independent of x. This models the real-world scenario where a labeling oracle is meaningful for the target distribution but produces garbage on off-distribution data. We train logistic regression in three settings: (a) S-only, (b) S + filtered B samples, and (c) S + unfiltered B samples (same count as filtered), then test on fresh S data. Setting (c) is a strawman showing that naive data augmentation actively hurts because ~99% of B has random labels.

## Data generation

Data lives in R^d. The target distribution S has its first coordinate x[0] > gamma (plus Gaussian noise), while the other distribution O has x[0] < -gamma. Remaining coordinates are isotropic Gaussian noise scaled as noise_scale = R/(2*sqrt(d)) so that total norms stay bounded by R regardless of dimension. The second coordinate x[1] is shifted by +label_shift * noise_scale for S only (default label_shift=1.0); O has no shift on x[1]. For the downstream task, S points get informative labels y = sign(x[1]), while O points get random Bernoulli(0.5) labels independent of x. For weak separation experiments, a fraction eps_S (resp. eps_O) of S (resp. O) points violate the margin.

## Experiments

All experiments use p = 0.01 unless otherwise noted. Each data point is averaged over 10–15 independent trials with error bars showing one standard deviation.

### Varying N_S (`vary_NS_tv`, `vary_NS_downstream`)

**Setup**: N_S ranges from 50 to 5000. Fixed N_B = 500,000, d = 20, gamma = 0.5, R = 3.

**What this tests**: The paper's main theorem (Theorem 1) predicts TV distance ~ O(R^2 / (gamma^2 * N_S) + R^2 / (p * gamma^2 * N_B)). With N_B large and fixed, the bound should decrease as ~1/N_S.

**TV distance plot** (`vary_NS_tv`): Shows estimated TV distance (histogram-based on the separating coordinate) vs N_S on a log-log scale. A 1/N_S reference line is overlaid.

**Downstream plot** (`vary_NS_downstream`): Shows classification accuracy using S-only vs S+filtered-B training data. With small N_S (e.g., 50–100), the S-only classifier has limited accuracy due to few training points, while the filtered augmentation provides access to ~p*N_B = 5000 additional high-quality samples, dramatically improving accuracy. The gap narrows as N_S grows (since S alone eventually suffices).

### Varying N_B (`vary_NB_tv`, `vary_NB_downstream`)

**Setup**: N_B ranges from 10,000 to 2,000,000. Fixed N_S = 200, d = 50, gamma = 0.5, R = 3.

**What this tests**: The 1/(p * N_B) term in the TV bound. Since p = 0.01, the effective sample count from B is p*N_B, so N_B = 2M gives 20K effective samples.

**TV distance plot** (`vary_NB_tv`): TV decreases with N_B and flattens when the 1/N_S term dominates — exactly as predicted.

**Downstream plot** (`vary_NB_downstream`): With N_S = 200 and d = 50, the S-only baseline is limited (~86%). As N_B increases, filtered augmentation adds more clean data, improving the downstream classifier from 89% to 99%, demonstrating substantial practical benefit.

### Varying dimension d (`vary_dimension_tv`, `vary_dimension_errors`, `vary_dimension_downstream`)

**Setup**: d ranges from 2 to 1000. Fixed N_S = 1000, N_B = 500,000, gamma = 0.5, R = 3.

**What this tests**: The paper's key insight — the TV bound depends on R^2/gamma^2 but NOT on the ambient dimension d. This is crucial for modern high-dimensional embeddings.

**TV bound plot** (`vary_dimension_tv`): Uses the FN + FP/p upper bound (cleaner than histogram estimation). Should be flat across dimensions.

**Error rates plot** (`vary_dimension_errors`): Shows FP and FN rates separately. Both should remain near zero until d approaches N_S, at which point the SVM (not the theoretical algorithm) starts to struggle.

**Downstream plot** (`vary_dimension_downstream`): Downstream accuracy should remain stable across dimensions, confirming that the practical benefit of filtering is also dimension-free.

**Note on high d**: When d >> N_S, the SVM approximation degrades because a linear model with d parameters cannot generalize well from N_S samples. This is a practical limitation of the SVM solver, not the theoretical result. The paper's exact optimization would remain dimension-free. In practice, one would use regularization or dimensionality reduction for very high d.

### Varying margin gamma (`vary_gamma_tv`, `vary_gamma_downstream`)

**Setup**: gamma ranges from 0.02 to 1.5. Fixed N_S = 30 (deliberately small), N_B = 500,000, d = 10, R = 3.

**What this tests**: The R^2/gamma^2 factor in the bound. N_S is kept small so the error term R^2/(gamma^2 * N_S) is large enough to observe.

**TV bound plot** (`vary_gamma_tv`): Log-log plot showing TV bound vs gamma with a 1/gamma^2 reference line. At small gamma (hard separation), the filter makes more errors; at large gamma (easy separation), the bound approaches zero.

**Downstream plot** (`vary_gamma_downstream`): With N_S = 30, the S-only classifier is weak regardless of gamma. The filtered augmentation should help more when gamma is larger (cleaner filtering), and may hurt or not help when gamma is very small (filter lets through too much noise from O).

### Weak separation (`weak_sep_eps_O_*`, `weak_sep_eps_S_*`)

**Setup**: Varies eps_O (fraction of O violating margin) from 0 to 0.05, and eps_S (fraction of S violating margin) from 0 to 0.3. Fixed N_S = 1000, N_B = 500,000, d = 10, gamma = 0.5, R = 3.

**What this tests**: Theorem 2 predicts additive degradation terms: +eps_S from S violations and +eps_O/p from O violations. With p = 0.01, even small eps_O is amplified 100x.

**TV bound plots** (`weak_sep_eps_O_tv`, `weak_sep_eps_S_tv`): Linear relationship between violation fraction and TV bound, with slopes matching the theoretical predictions (slope 1 for eps_S, slope 1/p = 100 for eps_O).

**Downstream plots** (`weak_sep_eps_O_downstream`, `weak_sep_eps_S_downstream`): Show how downstream accuracy degrades as margin violations increase. The eps_O effect is amplified by 1/p, so even 5% violation in O can noticeably affect the filtered distribution.

### 2D Visualization (`viz_input_data`, `viz_after_filtering`, `viz_distribution_comparison`)

**Setup**: d = 2, N_S = 500, N_B = 50,000, p = 0.01, gamma = 0.5.

Three separate plots showing: (1) the raw data with ground truth labels, (2) the learned decision boundary and which points pass the filter, (3) histograms comparing the S distribution and the filtered B distribution along the separating coordinate.

## Real-world experiments

Run separately:

```bash
uv run python run_realworld.py
```

Runtime: ~2–3 minutes. Uses three datasets available through sklearn (no extra downloads needed).

All downstream plots compare four baselines (measured by **balanced accuracy**, i.e., average of per-class recall):
- **S only**: train on the N_S labeled target samples + random negatives
- **S + filtered B**: augment S with filtered B samples (labeled as positive)
- **S + random B**: augment S with random (unfiltered) B samples — a strawman showing naively adding unlabeled data hurts
- **Oracle**: train on the full labeled training pool with balanced class weights

### 20 Newsgroups (`newsgroups_*`)

**Dataset**: 18,846 text documents across 20 newsgroup topics. TF-IDF features (10,000 dimensions).

**Setup**: One newsgroup topic is S. The full corpus is B (so p ~ 1/20 = 0.05). We use N_S = 200 labeled target documents and treat the remaining ~13K training documents as unlabeled B. The downstream task is binary classification: does a held-out document belong to the target topic?

**Topics tested**: sci.space (distinctive), rec.autos (moderate), comp.graphics (hard — among several comp.* topics), rec.sport.hockey (sports).

**Plots**:
- `newsgroups_downstream`: Bar chart comparing all four baselines across topics
- `newsgroups_error_rates`: FP/FN rates and filter precision across topics
- `newsgroups_vary_NS_downstream`: Balanced accuracy vs N_S for sci.space (S-only, filtered, random B)
- `newsgroups_vary_NS_errors`: Filter error rates vs N_S

**Why this matters**: This is a high-dimensional (d=10,000) real-world setting where linear separation is natural. The paper's theory predicts dimension-free performance, and TF-IDF features should provide good margin separation for distinctive topics. The varying difficulty across topics illustrates how the effective margin affects filtering quality in practice.

### MNIST (`mnist_*`)

**Dataset**: 70,000 handwritten digit images (28x28). Reduced to 50 PCA components for speed.

**Setup**: One digit is S, the full dataset is B (p ~ 0.1). N_S = 200 labeled target digit samples. Downstream task: binary classification (target digit vs. all others).

**Digits tested**: 3, 7, 1, 9 — varying in how visually distinctive they are.

**Plots**:
- `mnist_downstream`: Bar chart comparing all four baselines across digits
- `mnist_error_rates`: FP/FN rates per digit
- `mnist_vary_NS_downstream`: Balanced accuracy vs N_S for digit 3
- `mnist_vary_NS_errors`: Filter error rates vs N_S

### Covertype (`covertype_*`)

**Dataset**: 581,012 forest cover type samples with 54 tabular features. No embedding needed.

**Setup**: A rare forest cover type is S. We subsample 100K points for B. Class 4 (Cottonwood/Willow, p ~ 0.005) and Class 5 (Aspen, p ~ 0.016) are tested as targets. This models the "rare class detection" scenario with very low p.

**Plots**:
- `covertype_downstream`: Bar chart comparing all four baselines for the two rare classes

## Output files

### Plots (`plots/`)

Each experiment produces one PDF+PNG per plot (no multi-panel figures). Naming convention: `{experiment}_{metric}.{pdf,png}`.

### Data (`data/`)

Each experiment saves a JSON file containing all parameters and results, enabling easy re-plotting without re-running experiments. Files contain: parameter values, means, standard deviations, and all experimental settings.

## Implementation notes

- **Synthetic filter (SVM proxy)**: The paper's optimization (Eq. 1) is NP-hard in general. Our synthetic experiments use LinearSVC with asymmetric class weights (class_weight ratio N_B/N_S on the positive class, C=100) to approximate the paper's hard-constraint optimization. This works well in practice but has limitations: (1) it may not exactly enforce the hard constraint that all S samples are classified correctly, and (2) it degrades when d >> N_S.

- **Real-world filter (Logistic Regression)**: For real-world data, we use L2-regularized logistic regression (C=10) instead of SVM, as it generalizes better in high dimensions with limited N_S. The asymmetric weighting is the same (class_weight ratio N_B/N_S).

- **TV estimation**: We use two approaches: (1) histogram-based on the separating coordinate (synthetic experiments 1–2), and (2) the FN + FP/p upper bound from the paper (synthetic experiments 3–5). The histogram approach captures the full distributional distance but has estimation noise; the FP/FN approach is cleaner but is an upper bound.

- **Downstream task (synthetic)**: The binary classification task (sign of x[1]) is deliberately simple, so that the benefit of more data is clear and measurable. Points from S have x[1] shifted positive (label_shift=1.0), making labels informative. Points from O get *random* Bernoulli(0.5) labels independent of x, modeling a labeling oracle that works for S but produces garbage for O. Including unfiltered B data actively hurts because ~99% of B has random labels that inject noise. With only N_S training points from S, the classifier has limited accuracy; adding filtered B data provides many more training points from (approximately) the S distribution, directly improving generalization. We compare three settings: S-only, S+filtered B, and S+unfiltered B (strawman).

- **Downstream task (real-world)**: We use balanced accuracy (average of per-class recall) instead of raw accuracy, which gives a more meaningful metric when classes are highly imbalanced (e.g., Covertype with p=0.005). To prevent noisy filtered samples from overwhelming clean S data, we cap the number of filtered samples added to at most 5·N_S and balance the number of negative training samples accordingly.

- **Balanced accuracy**: For imbalanced binary classification (e.g., rare class with p=0.005), raw accuracy is misleading — predicting all negative gives 99.5% accuracy. Balanced accuracy = (TPR + TNR)/2 treats both classes equally and gives 50% for the trivial all-negative classifier.
