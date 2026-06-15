# Physics-Informed Graph Neural Networks for Plant-Disease Spread Prediction, with Cross-Physics Generalization and Validated Uncertainty

*Workshop-paper draft. Numbers marked `[RUN]` are filled from `experiments/run_study.py`
output — do not invent them. This draft is intentionally conservative in its claims.*

---

## Abstract

We study next-step prediction of plant-disease spread on spatial crop-field graphs.
Existing GNNs for plant disease use a fixed adjacency; we instead make the
message-passing weight on each field-to-field connection a learned, differentiable
function of meteorological edge features (wind speed, humidity, wind-direction
alignment, distance) — a scheme we call **DAGCA** (Dynamic Atmospheric Graph
Construction). We define a **leakage-safe** prediction task (evaluate only on
currently-susceptible fields, with features taken at the prediction time), benchmark
against strong non-GNN baselines, and — to address the central criticism of synthetic
studies — evaluate **cross-physics generalization**: training on one ground-truth
dispersal kernel and testing on a structurally different one. We extend DAGCA with
Beta-distributed edge weights and **validate** the resulting uncertainty via
temperature scaling and an uncertainty-vs-error analysis. We report a spectral
diagnostic (SENR0) read from the learned adjacency, explicitly framed as a diagnostic
rather than a calibrated reproduction number. On the susceptible frontier (base rate
~10%), `[RUN]`.

## 1. Introduction

Plant diseases cause large annual crop losses; predicting which healthy fields are
about to be infected would enable targeted intervention. Spread is driven by
atmospheric transport of spores, yet standard GNN approaches treat the field graph as
static. We ask whether making the graph's edge weights an explicit, differentiable
function of weather improves next-step spread prediction, and — critically — whether
such a model generalizes across different dispersal physics rather than memorizing the
one it was trained on.

**Contributions (scoped honestly).**
1. **DAGCA**, a meteorology-conditioned edge-weighting scheme for plant-epidemic GNNs.
   The mechanism (edge-conditioned message weighting) follows ECC [Simonovsky &
   Komodakis 2017], MPNN [Gilmer 2017], GATv2 [Brody 2021]; our novelty is the domain
   application and physics-informed feature design — not a new graph-learning algorithm.
2. A **leakage-safe task** for monotone-SEIR spread prediction (susceptible-frontier
   evaluation) that prior synthetic setups get wrong.
3. A **cross-physics generalization** protocol that directly tests the
   "synthetic ⇒ circular" objection.
4. **Validated uncertainty** (temperature scaling + uncertainty-vs-error), not asserted.

## 2. Related Work
Edge-conditioned / attention GNNs (ECC, MPNN, GAT/GATv2); learned graph structure
(NRI [Kipf 2018], LDS [Franceschi 2019], IDGL [Chen 2020]) — these learn *which* edges
exist; we learn weights on a fixed k-NN skeleton from physical features. Network SEIR /
Next-Generation-Matrix epidemiology [Diekmann 1990]. Calibration of neural classifiers
[Guo 2017].

## 3. Method

### 3.1 Task and leakage-safe evaluation
A snapshot at time *t* is a k-NN graph over *N* fields. Node features = SEIR one-hot
**at t** ⧺ [x, y, humidity, crop type]. Edge features = [wind speed, humidity,
wind-direction alignment, normalized distance]. Label = infected (E/I/R) at *t+1*.
Because SEIR is monotone, already-infected nodes are trivial positives; we therefore
compute the loss and **all metrics only over nodes Susceptible at t**. The positive
class (newly infected) is ~10%, so we use inverse-frequency class weighting and report
AUPRC and ECE alongside F1/AUROC.

### 3.2 DAGCA
For edge (i,j) with features e_ij, the message weight is
`w_ij = σ(MLP(e_ij · exp(s)))`, s a learned log-scale. `w_ij` is the **sole** gate on
messages — no second in-GNN edge projection — so the DAGCA-on/off ablation isolates it.

### 3.3 Bayesian DAGCA
`w_ij ~ Beta(α_ij, β_ij)`, `α,β = softplus(BetaHead(e_ij)) + 1`. Training uses a single
reparameterized Kumaraswamy sample per pass (variance reaches the loss); inference uses
the posterior mean. Loss `= L_task + λ·KL[Beta(α,β)‖Beta(1,1)]`.

### 3.4 SENR0 (diagnostic)
`R = ρ((β/γ)·A)` via differentiable power iteration on the learned adjacency A.
Because A is trained on a classification loss, we present R as a connectivity
**diagnostic**, not a validated epidemiological R₀.

### 3.5 Two ground-truth physics
We generate data from two dispersal kernels: **cosine** (distance-decay × wind cosine
alignment) and **plume** (anisotropic Gaussian plume). The model receives identical
edge features in both, enabling a clean out-of-distribution transfer test.

## 4. Experiments

All configurations in a seed share identical train/val/test graphs; results are
aggregated over seeds {42,43,44}.

**4.1 Ablation (Table 1).** `[RUN — experiments/run_study.py]`
Headline comparison: DAGCA vs No-DAGCA on AUPRC/F1.

**4.2 External baselines (Table 2).** Infected-neighbor heuristic, logistic regression,
random forest, tabular MLP on the same frontier. `[RUN — experiments/baselines.py]`

**4.3 Cross-physics generalization (Table 3).** Train-physics × test-physics AUPRC
matrix; off-diagonal = OOD. `[RUN — experiments/generalization.py]`

**4.4 Validated calibration (Fig).** ECE raw vs temperature-scaled; error rises with
predictive uncertainty. `[RUN — experiments/calibration.py]`

## 5. Limitations
Synthetic data only (mitigated, not solved, by cross-physics transfer); SENR0 is a
diagnostic, not a validated R₀; real-world spatial data and field deployment are future
work.

## 6. Conclusion
A physics-informed, meteorology-conditioned edge-weighting GNN for plant-disease spread,
evaluated on a leakage-safe task, benchmarked against non-GNN baselines, tested for
cross-physics transfer, and shipped with validated uncertainty.

## References
ECC (Simonovsky & Komodakis 2017); MPNN (Gilmer et al. 2017); GATv2 (Brody et al. 2021);
NRI (Kipf et al. 2018); LDS (Franceschi et al. 2019); IDGL (Chen et al. 2020);
Diekmann et al. (1990); Guo et al. (2017).
