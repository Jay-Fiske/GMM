# GMM-BA++: Enhanced Online Source-Free Universal Domain Adaptation

An implementation and experimental study of **GMM-BA++**, an enhanced
version of GMM-BA for **Online Source-Free Universal Domain Adaptation
(SF-UniDA)**.

The project starts from the GMM-based method proposed by Schlachter et
al. and reproduces its results before introducing targeted improvements
for feature geometry, covariance stability, pseudo-label reliability,
class imbalance, OOD threshold adaptation, and likelihood computation.

## Overview

In real-world deployment, a model trained on a source domain can
encounter a target domain whose data distribution differs from the
training distribution. The problem becomes more challenging when:

-   Source data is unavailable during adaptation.
-   Target samples arrive as an online stream and are seen only once.
-   The target label space may differ from the source label space.
-   Previous target samples cannot be stored because of memory
    constraints.
-   Unknown classes must be detected without target labels.

**GMM-BA** addresses these requirements by maintaining a compact
Gaussian representation of known-class feature distributions rather than
storing previous samples.

**GMM-BA++** builds on this approach and introduces six targeted
modifications.

## Problem Setting

The project focuses on **Online Source-Free Universal Domain Adaptation
(SF-UniDA)**.

A source model is pretrained on a labelled source dataset:

\[ f = h `\circ `{=tex}g \]

where:

-   `g` is the feature extractor.
-   `h` is the classifier.

During adaptation, only an unlabeled target stream is available. Target
samples arrive in small batches, with each batch accessed once.

The method handles three category-shift scenarios:

  -----------------------------------------------------------------------
  Scenario                Relationship            Description
  ----------------------- ----------------------- -----------------------
  PDA                     (Y_t                    Target contains only a
                          `\subset `{=tex}Y_s)    subset of source
                                                  classes

  ODA                     (Y_s                    Target contains new
                          `\subset `{=tex}Y_t)    classes not present in
                                                  the source

  OPDA                    Partial overlap         Both source-private and
                                                  target-private classes
                                                  exist
  -----------------------------------------------------------------------

The objective is to classify known target classes while rejecting
unknown classes, without source data and with minimal memory.

## Why GMM-BA?

Traditional online adaptation approaches can rely on:

-   Memory queues containing previous target samples.
-   Mean-teacher models containing a second copy of the network.
-   Source prototypes or source statistics.

These approaches introduce memory or source-data requirements that are
incompatible with strict source-free and memory-constrained settings.

GMM-BA instead models each known class using a Gaussian distribution in
a compact feature space. Its stored state consists primarily of class
means, covariance matrices, and accumulated class weights.

## GMM-BA Pipeline

The baseline method consists of four main components:

1.  **GMM-based pseudo-labelling**
2.  **Entropy-based OOD detection**
3.  **Contrastive feature adaptation**
4.  **KL-divergence classifier optimisation**

### 1. GMM-Based Knowledge Transfer

Reduced target features are represented using a Gaussian distribution
for each known class.

For each incoming batch, the class mean and covariance are updated using
an EM-inspired weighted update. Model softmax outputs provide the
class-wise weights.

The original method uses a decay factor:

\[ `\alpha `{=tex}= 0.999 \]

to control the balance between historical information and information
from the current batch.

The GMM therefore acts as a compact memory of previously observed target
information.

### 2. Pseudo-Labelling and OOD Detection

For every target sample, the method evaluates its likelihood under the
Gaussian corresponding to each known class.

A maximum-likelihood assignment provides a pseudo-label.

However, assigning every sample to a known class would incorrectly
classify unknown classes. Therefore, GMM-BA computes the normalized
Shannon entropy of the likelihood vector.

-   **Low entropy:** confident known-class prediction.
-   **High entropy:** likely unknown/OOD sample.
-   **Intermediate entropy:** uncertain sample, which is discarded
    during adaptation to reduce negative transfer.

Two thresholds are used during pseudo-labelling:

-   `τk` --- known-class threshold.
-   `τu` --- unknown-class threshold.

### 3. Contrastive Adaptation

The contrastive loss adapts the feature extractor so that the target
feature space becomes more class-structured.

It:

-   Pulls samples toward samples of the same pseudo-labelled class.
-   Pulls samples toward the corresponding GMM class mean.
-   Pushes samples from different classes apart.
-   Separates known samples from the unknown class.

The GMM means act as class anchors, avoiding the need for source
prototypes.

Data augmentation is also used so that each batch is supplemented with
augmented samples.

### 4. KL-Divergence Classifier Optimisation

The KL-divergence loss optimizes the classifier head.

For known samples, the classifier is encouraged to produce a confident,
non-uniform prediction.

For unknown samples, the classifier is encouraged toward a uniform
distribution, corresponding to high entropy.

The overall objective is:

\[ L = L_C + `\lambda `{=tex}L\_{KLD} \]

with:

\[ `\lambda `{=tex}= 1 \]

### 5. Inference

During inference, classifier prediction and GMM-based OOD detection are
combined.

A single threshold is used:

\[ `\tau `{=tex}= `\frac{\tau_k+\tau_u}{2}`{=tex} \]

Samples below the threshold are classified using the maximum classifier
output; samples above it are predicted as unknown.

------------------------------------------------------------------------

# GMM-BA++ Improvements

The analysis of the original GMM-BA paper and released implementation
identified several limitations. GMM-BA++ introduces six targeted
modifications.

## 1. L2 Feature Normalisation

### Problem

Reduced features passed to the GMM have unconstrained magnitudes. This
can cause high-magnitude dimensions to disproportionately influence
covariance estimation.

The contrastive loss, meanwhile, uses cosine similarity, which naturally
operates on normalized feature directions.

### Modification

Features are L2-normalized before GMM updates and contrastive learning:

``` python
feat = F.normalize(feat, p=2, dim=1)
```

with a small epsilon for numerical stability.

This places features on the unit hypersphere and makes the geometry used
by the GMM and contrastive loss more consistent.

## 2. Tikhonov Covariance Regularisation

### Problem

Under-represented classes may have very few softmax-weighted samples in
early batches. Their estimated covariance matrices can become nearly
singular, resulting in numerical failures during likelihood computation.

### Modification

A small isotropic regularisation term is added:

\[ `\Sigma`{=tex}\^{reg}\_c = `\Sigma`{=tex}*c + `\lambda`{=tex}*{reg}I
\]

with:

\[ `\lambda`{=tex}\_{reg}=10\^{-6} \]

This improves numerical stability and prevents NaN likelihood values.

## 3. Confidence-Weighted Contrastive Loss

### Problem

The original contrastive loss treats all pseudo-labelled samples
equally. Incorrect early pseudo-labels can therefore generate false
positive pairs and cause negative transfer.

### Modification

Each known sample is weighted according to its maximum GMM likelihood:

\[ w_i = `\max`{=tex}\_c p(x_i\|c) \]

High-confidence samples contribute more strongly to the contrastive
objective, while uncertain pseudo-labels are automatically
down-weighted.

Conceptually:

``` python
confidence = likelihood.max(dim=1).values
used = used * confidence[known_mask].unsqueeze(1)
```

## 4. Adaptive Per-Class Decay

### Problem

A single global decay factor does not account for differences in class
frequency.

A rare class and a frequently observed class may require different
amounts of historical information when updating their Gaussian
distributions.

### Modification

GMM-BA++ uses a class-specific decay:

\[ `\alpha`{=tex}\_c = 1-`\frac{1}{1+s^{-}_k(c)}`{=tex} \]

where (s\^{-}\_k(c)) represents accumulated evidence for class `c`.

Properties:

-   On the first encounter, (`\alpha`{=tex}\_c=0), so the class is
    initialized primarily from the current batch.
-   As accumulated evidence increases, (`\alpha`{=tex}\_c) approaches 1.
-   Frequently observed classes gradually rely more on historical
    estimates.

## 5. Continual EMA OOD Threshold Recalibration

### Problem

The original OOD thresholds are estimated during an initial warm-up
period and then frozen.

As adaptation improves the GMM, the entropy distribution can change,
making the original thresholds less well calibrated.

### Modification

The thresholds are continuously updated using an exponential moving
average:

\[ `\tau `{=tex}`\leftarrow`{=tex} `\alpha`{=tex}*{EMA}`\tau`{=tex}+
(1-`\alpha`{=tex}*{EMA})`\tau`{=tex}\_{batch} \]

The experiment uses:

\[ `\alpha`{=tex}\_{EMA}=0.95 \]

while preserving a minimum separation between the known and unknown
thresholds.

This modification is particularly beneficial in the PDA experiment, but
the reported experiments show that its effect is scenario-dependent for
ODA and OPDA.

## 6. GPU-Native Batched Likelihood Computation

### Problem

The original likelihood computation uses a Python loop over classes with
`scipy.stats.multivariate_normal`, causing repeated GPU-to-CPU
transfers.

This becomes increasingly expensive as the number of classes grows.

### Modification

The implementation replaces the CPU loop with GPU-native batched
computation using `torch.distributions.MultivariateNormal`.

Conceptually:

``` python
dist = MvNormal(mu, C_pd)
lp = dist.log_prob(feat[:, None, :])
```

This evaluates the likelihood for all classes in a vectorized operation.

### Measured Speed Improvement

On VisDA-C:

  Implementation                    Speed
  --------------------------- -----------
  Original scipy CPU loop       1.77 it/s
  GPU-native implementation     4.44 it/s

This corresponds to approximately **2.5× higher training throughput**,
with numerically identical accuracy in the reported experiment.

------------------------------------------------------------------------

# Experimental Results

## Reproduction of GMM-BA

The project first reproduced the original GMM-BA implementation using
the released code and pretrained source models.

### VisDA-C

  Method             PDA Accuracy   ODA H-score   OPDA H-score
  ---------------- -------------- ------------- --------------
  GMM-BA (paper)             41.2          59.9           60.3
  GMM-BA (ours)              40.9          60.1           60.9
  Deviation                  -0.3          +0.2           +0.6

The reproduction remained within 0.6 percentage points of the reported
paper values.

### DomainNet

  Method             PDA Avg.   ODA Avg.   OPDA Avg.
  ---------------- ---------- ---------- -----------
  GMM-BA (paper)        37.44      49.13       50.56
  GMM-BA (ours)         36.90      48.78       50.41
  Deviation             -0.54      -0.35       -0.15

The small differences were attributed in the report to single-run
variance, since the original paper reports six-seed means.

------------------------------------------------------------------------

# GMM-BA++ Results

For the main state-of-the-art comparison, the report evaluates GMM-BA++
using contributions C1--C4, excluding EMA threshold recalibration in
order to preserve ODA/OPDA performance.

### VisDA-C and DomainNet

  -----------------------------------------------------------------------------------
  Method          VisDA PDA  VisDA ODA VisDA OPDA   DomainNet   DomainNet   DomainNet
                                                          PDA         ODA        OPDA
  -------------- ---------- ---------- ---------- ----------- ----------- -----------
  Source-only          17.1       31.7       26.9        26.3        43.6        45.1

  OWTTT                28.1       56.7       46.8        24.5        44.3        45.4

  COMET-P              32.1       50.9       42.9        36.2        49.3        50.5

  GMM-BA               41.2       59.9       60.3        37.4        49.1        50.6

  **GMM-BA++**     **40.9**   **62.4**   **62.0**    **36.9**    **49.8**    **50.7**
  -----------------------------------------------------------------------------------

The report notes that GMM-BA++ improves the VisDA-C ODA H-score by 2.5
percentage points and OPDA H-score by 1.7 percentage points over GMM-BA
in this configuration. On DomainNet, the reported ODA improvement is 0.7
percentage points.

## Ablation Study

The reported ablation study evaluates the individual GMM-BA++
contributions.

  ------------------------------------------------------------------------------
  Configuration                        PDA                ODA               OPDA
  --------------------- ------------------ ------------------ ------------------
  GMM-BA baseline                     40.9               60.1               60.9

  \+ C1/C2:                           40.6               62.4               60.6
  Normalisation +                                             
  covariance                                                  
  regularisation                                              

  \+ C3:                              40.6               62.3               61.2
  Confidence-weighted                                         
  contrastive loss                                            

  \+ C4: Adaptive                     42.2               62.3               62.0
  per-class decay                                             

  \+ C5: EMA threshold                51.6               58.9               57.5
  recalibration                                               
  (`αEMA=0.95`)                                               

  \+ C6: GPU-native          Same accuracy      Same accuracy      Same accuracy
  likelihood                                                  
  ------------------------------------------------------------------------------

The report highlights that the EMA threshold modification is strongly
scenario-dependent: it provides a large PDA improvement but decreases
the ODA and OPDA scores with `αEMA=0.95`. A more conservative EMA factor
is proposed as a future direction.

------------------------------------------------------------------------

# Memory Efficiency

One of the central motivations of GMM-BA is to avoid storing large
numbers of previous target samples or an additional teacher model.

The GMM representation stores, for each known class:

-   A mean vector.
-   A symmetric covariance matrix.
-   A class weight.

For a PDA setting with 345 source classes on DomainNet, the reported GMM
representation requires approximately:

-   **2.2%** of the memory required by the memory queue.
-   **3.1%** of the memory required by a mean-teacher model.

This compact representation makes the approach suitable for
memory-constrained and embedded scenarios.

------------------------------------------------------------------------

# Datasets and Evaluation

The project reports experiments on:

### VisDA-C

A domain adaptation benchmark involving a synthetic-to-real domain shift
across 12 classes.

### DomainNet

A larger multi-domain benchmark containing 345 classes and multiple
domain shifts.

### Metrics

-   **PDA:** Classification accuracy.
-   **ODA:** H-score.
-   **OPDA:** H-score.

The H-score jointly evaluates known-class classification and
unknown-class recognition.

------------------------------------------------------------------------

# Key Implementation Details

Reported experimental settings include:

-   Batch size: `64`
-   Optimizer: SGD
-   Momentum: `0.9`
-   VisDA-C learning rate: `1e-2`
-   DomainNet learning rate: `1e-3`
-   Reduced feature dimension: `64`
-   Contrastive temperature: `0.1`
-   KL loss weight: `λ = 1`
-   GMM covariance regularisation: `1e-6`
-   Feature normalisation: L2
-   EMA threshold factor in the reported adaptive-threshold experiment:
    `0.95`

These values reflect the settings documented in the project report.

------------------------------------------------------------------------

# Project Contributions

The project can be viewed as a progression:

``` text
Pretrained Source Model
        |
        v
 Target Data Stream
        |
        v
 Feature Extraction
        |
        v
 Dimension Reduction
        |
        v
 L2 Feature Normalisation
        |
        v
 GMM Update
        |
        +----------------------+
        |                      |
        v                      v
 GMM Likelihoods          OOD Entropy
        |                      |
        +----------+-----------+
                   |
                   v
             Pseudo-Labels
                   |
          +--------+--------+
          |                 |
          v                 v
 Confidence-Weighted   KL-Divergence
 Contrastive Loss          Loss
          |                 |
          +--------+--------+
                   |
                   v
             Model Update
                   |
                   v
          Next Target Batch
```

The central idea is to retain **distributional knowledge rather than
individual samples**.

------------------------------------------------------------------------

# Limitations and Future Work

The report identifies several directions for further improvement.

## 1. Multi-Modal / Hierarchical GMM

The current approach represents each class with a single Gaussian. This
can be restrictive for visually diverse classes with multiple modes.

A hierarchical GMM could represent a class using multiple Gaussian
sub-components:

\[ `\text{Class }`{=tex}c `\rightarrow`{=tex}
{G\_{c,1},G\_{c,2},...,G\_{c,K_c}} \]

The report suggests selecting the number of components using BIC or an
entropy-based splitting criterion.

## 2. Low-Rank Covariance

Full covariance matrices require quadratic memory with respect to
feature dimension.

A structured representation such as:

\[ `\Sigma`{=tex}\_c = `\mathrm{diag}`{=tex}(d_c)+U_cU_c\^T \]

could reduce memory requirements while allowing higher-dimensional
features.

## 3. Scenario-Adaptive EMA

The current EMA threshold behaviour differs across PDA, ODA and OPDA.

A future system could monitor the unknown-rejection rate and dynamically
adjust the EMA factor instead of relying on scenario-specific tuning.

## 4. Correct the Missing Decay in the Released Implementation

The report notes a discrepancy between the paper's equations and
released implementation regarding the decay factor in the GMM update.
Future work could make the implementation fully consistent with the
intended formulation while retaining adaptive per-class decay.

## 5. Vision Transformer Backbone

The original work includes preliminary ViT-B/16 results. Extending the
GMM-BA++ modifications to ViT features is identified as another
direction for evaluation.

------------------------------------------------------------------------

# Reference

The baseline method is:

> P. Schlachter, S. Wagner, and B. Yang. *Memory-efficient
> pseudo-labeling for online source-free universal domain adaptation
> using a Gaussian mixture model.* arXiv:2407.14208, 2024.

The project report presents:

> *GMM-BA++: Towards More Robust Online Source-Free Universal Domain
> Adaptation via Enhanced Gaussian Mixture Modelling.*

Authors:

-   **Jay Fiske**
-   **Udaya Kumar**
-   **C. Krishna Mohan**
-   Indian Institute of Technology Hyderabad

------------------------------------------------------------------------

# Summary

**GMM-BA++** extends GMM-BA while preserving its central advantages:

-   **Source-free:** no source dataset is required during adaptation.
-   **Online:** target data is processed batch-by-batch.
-   **Universal:** supports PDA, ODA and OPDA scenarios.
-   **Memory-efficient:** stores compact class distributions instead of
    large sample queues.
-   **More robust:** improves feature geometry, covariance stability and
    pseudo-label weighting.
-   **Faster:** GPU-native likelihood computation provides approximately
    2.5× higher training throughput in the reported VisDA-C experiment.

The project demonstrates that relatively targeted changes to the GMM-BA
framework can address practical implementation and robustness issues
while retaining the source-free and memory-efficient nature of the
original method.
