# Response to the reviewers

Manuscript: *Classification of trajectories of dynamic systems using physically-informed neural networks* (SN Computer Science)

We thank the editor and the reviewers for the detailed and constructive reports. All the comments have been addressed in the revised manuscript. Below we answer every point and indicate where the corresponding change was made. Section and equation numbers refer to the revised version.

---

## Reviewer #2

### 2.1 The "Loss function" topic should be further strengthened.

Section 5 has been rewritten and substantially extended. It now contains:

* an explicit statement of *why* the naive squared-error loss on the accelerations (Eq. 7) is unsuitable — it requires inverting `H(L̂)` at every point of the trajectory;
* a separate discussion of the two sources of ill-posedness: the scale invariance `L ↦ cL` (which is now formally linked to Remark 2) and its numerical consequence, the unbounded `H(L̂)⁻¹` that lets the optimizer drive the Lagrangian towards zero and produce non-physical oscillations;
* the motivation for the residual form of the loss (Eq. 23), which substitutes the *measured* acceleration into the linear system instead of solving it, so that no matrix inversion appears in the computational graph and the gradients stay bounded near degenerate configurations;
* the argument that under the constraint (Eq. 22) the residual loss dominates the acceleration error, so its minimization also minimizes the quantity of interest;
* a term-by-term analysis of the smooth barrier (Eq. 25): its asymptotic behaviour for small and for large eigenvalues, the location of its minimum, the role of `α` and `β`, and the limit `β → ∞` in which the hard constraint is recovered.

### 2.2 There are inconsistencies in Font types.

Several genuine causes were found and removed:

* the preamble loaded `\usepackage[T2A]{fontenc}` (Cyrillic encoding) together with `mathtext` in an English-only manuscript, which caused font substitution at several sizes. Replaced by `[T1]`, `mathtext` removed;
* the preamble loaded `\usepackage{a4wide}`, marked `%REMOVE AFTER` by ourselves, which overrode the journal page geometry. Removed, together with the leftover commented `\documentclass{article}` lines;
* `sn-jnl.cls` defines its theorem styles (`thmstyleone`, `thmstyletwo`, `thmstylethree`) only if `amsthm` is already loaded when the class is read, which never happens in the supplied template. As a result all theorem, lemma and remark headings silently fell back to the default `amsthm` style. The styles are now defined in the preamble under an `\@ifundefined` guard, so the headings follow the journal style;
* `\textbf{w}` was used inside math mode in several equations while `\mathbf{w}` was used in others; and `\text{max}` was used instead of `\max`. Unified throughout.

The manuscript now compiles without any LaTeX font, reference or citation warnings.

### 2.3 Mathematical equations and concepts can be explained for better clarity.

* Eq. (1)–(2): the non-degeneracy condition is now written with a named Hessian `H(L)` and explained physically — for a system with kinetic energy `½ q̇ᵀM(q)q̇` the Hessian is the mass matrix, so the condition states that the masses are non-zero (Section 1).
* The Lagrangian space `𝕃` is now defined properly, as twice continuously differentiable square-integrable functions on a compact domain `Ω ⊂ ℝ^{2r}`, instead of the previous informal notation (Section 3).
* Eq. (14) is now written in the equivalent non-inverted form `H(L)q̈ = A(L)`, which is the form actually used everywhere later, and the operators `A` and `H` are introduced once at that point.
* Lemma 2 has been corrected (see also 3.2 below) and now also covers `H`, which is used later.
* The derivation of the accelerations from the chain rule, the definition of the action, and the statements and the proofs of Theorems 2 and 3 have been rewritten for grammatical and mathematical clarity. In particular, the proof of Theorem 2 previously ended with the inequality `γᵢ + ε < γⱼ + ε`, which does not give strong separability; it should be, and now is, `γᵢ + ε < γⱼ − ε`, which is what the choice `εᵢⱼ = |γᵢ − γⱼ|/3` was made for.
* The dimension symbol is now consistently `r` throughout (`n` and `r` were mixed).
* The unlabelled figure illustrating the construction of the state vector from several sensors now has a caption, a label and a reference in the text (Figure 1).

### 2.4 Dataset description can be stated in more detail.

Section 7 now opens with a dedicated subsection 7.1 "Dataset", which states: the number and demographics of the subjects; the placement of the three inertial units (wrist, chest, **ankle** — the previous text incorrectly said "elbow"); the full sensor complement of each unit and the sampling rate of 100 Hz; the distinction between the protocol and the optional part of PAMAP2 and which one is used; the complete list of the 12 protocol activities; *why* only the gyroscope channels are used (their readings are the generalized velocities, whereas the accelerometers mix proper acceleration with gravity and depend on the unknown sensor orientation); and the resulting number of generalized coordinates `r = 9`.

A new subsection 7.2 gives the preprocessing in full: interpolation of the dropouts, the numerical differentiation and integration formulas with their error bounds, the time step `h = 0.01` s, the segmentation into non-overlapping windows of 500 ticks (5 s) with the justification of this length, the resulting number of trajectories, and the per-channel rescaling.

A new subsection 7.3 gives the architecture of the Lagrangian network (input layer, two residual blocks of width 100, scalar output, Softplus activation and the reason for it), the optimizer and all its hyperparameters, the values `α = 5`, `β = 12`, and the wall-clock cost of fitting.

### 2.5 The conclusion section should be extended.

Section 8 has been rewritten and is now about four times longer. It separates: what the theoretical part establishes (the seminorm, the identification of its kernel with the classical gauge freedom, Theorem 1, and the role of Theorems 2–3); what the computational part shows (reconstruction quality, the vectorization, the accuracies and the comparison with the time series forest, and the properties of the representation); and three explicit limitations with the corresponding directions of further work — the per-trajectory inference cost (see 3.1 below), the sensitivity to the regularization hyperparameters, and the two missing groups of experiments (robustness ablations and modern baselines).

### 2.6 A few recent references should be included.

Nine references have been added, all verified against Crossref:

* Landau & Lifshitz, *Mechanics*, 3rd ed. (for gauge invariance, requested by Reviewer #3);
* Cuomo et al., *Scientific Machine Learning Through Physics-Informed Neural Networks*, J. Sci. Comput. 92:88, 2022;
* Kaltsas, *Constrained Hamiltonian systems and physics-informed neural networks: Hamilton-Dirac neural networks*, Phys. Rev. E 111:025301, **2025**;
* Ismail Fawaz et al., *InceptionTime*, DMKD 34(6):1936–1962, 2020;
* Dempster et al., *MiniRocket*, KDD 2021, 248–257;
* Tan et al., *MultiRocket*, DMKD 36(5):1623–1646, 2022;
* Middlehurst, Schäfer & Bagnall, *Bake off redux*, DMKD 38(4):1958–2031, **2024**;
* Lu et al., *DeepONet*, Nature Machine Intelligence 3:218–229, 2021;
* Kontolati et al., *Learning nonlinear operators in latent spaces for real-time predictions of complex dynamics in physical systems*, Nature Communications 15:5101, **2024**.

### 2.7 Comprehensive proofreading.

The whole manuscript has been proofread. All the typographical errors listed by Reviewer #3 ("one hes to solve", "closed closed singular sets", the hyphenation artefact) have been corrected, together with a number of others found in the process, among them: several broken or ungrammatical sentences in Sections 2, 4 and 7; a fragment of a running page header that had been pasted into the body of Section 3; a duplicated equation label `eq:model`; two duplicated labels in the proof of Theorem 1; a broken math environment in the norm-approximation block; a dangling `\ref{remark3}` and a reference to a non-existent "remark to Lemma 2"; the description of the potential energy as `V(q,q̇)` instead of `V(q)`; and the description of `q̇` as "generalized accelerations".

---

## Reviewer #3

### 3.1 Methodological flaws: the inference bottleneck.

The reviewer is correct, and we thank them for stating the point so precisely. The current pipeline does initialize and fit a new Lagrangian neural network for every trajectory, at training *and* at test time, so the inference cost on a new trajectory equals the cost of an ODE-constrained optimization problem — about a minute of GPU time per 5-second segment in our setting.

We have made this explicit rather than leaving it implicit. Section 7.3 now states the per-trajectory fitting cost and the total cost of processing the dataset. Section 8 discusses the bottleneck as the first and most important limitation of the method: it notes that the classifier itself is fast and that the cost lies entirely in the map `p(L|X)`, that this is acceptable offline but prohibitive for the real-time monitoring which is the natural application, and that the way to remove it is to learn this map once as an operator, so that a trajectory becomes a point in the input space rather than a separate learning task. We cite DeepONet and its latent-space modification designed for real-time prediction of the dynamics of physical systems as the appropriate tools.

We would ask the reviewer to accept this as a stated direction of further work rather than as part of the present paper. Operator learning is a substantial piece of work in its own right and would change the subject of the manuscript, whose contribution is the structure of the Lagrangian space — the seminorm, its kernel, and the conditions under which the compactness hypothesis survives the finite-dimensional projection — and the demonstration that classification in that space is feasible. The two are complementary: the operator would replace the map `p(L|X)`, while everything downstream of it, which is what this paper analyses, would remain unchanged.

### 3.2 Physics and mathematics.

**Theorem 1 and gauge invariance.** Accepted in full. The result is now presented as what it is. A new Remark 1 states the classical fact, with the reference to Landau & Lifshitz §2, that Lagrangians differing by a total time derivative `dF(q,t)/dt` generate the same equations of motion, and then shows by direct computation that the kernel of our seminorm is *exactly* the set of such gauge terms in the autonomous case: `H(dF/dt) = 0` because the term is affine in `q̇`, `A(dF/dt) = 0` by the symmetry of the second derivatives, and conversely `H(L) = 0` forces `L = a(q)·q̇ + b(q)`, after which `A(L) = 0` for all `q̇` forces `b` to be constant and the form `Σaᵢdqᵢ` to be closed. The remark closes by stating explicitly that the seminorm is not a new equivalence relation but a way to compute the classical one from the values of a neural network.

The theorem itself (now Theorem 1) has been restated. The claimed equivalence was in fact false as written: `L' = cL` produces identical accelerations while `‖L' − L‖ ≠ 0`. The theorem now states the two directions separately — the implication that holds unconditionally, and the converse under the additional hypothesis `H(δL) = 0` — and a new Remark 2 gives the counterexample, explains that it is the scaling ambiguity, and connects it to the normalization imposed in Section 5. The proof has also been corrected: it previously confused `H(L)` with `H(δL)` and `q̈` with `δq̈`, and contained two duplicated equation labels.

**Lemma 1 proof logic.** Accepted. The proof no longer argues by contradiction. It is now the direct argument the reviewer describes: the entries of `H` are continuous, the determinant is a polynomial in them, hence `|det H|` is continuous on the compact `Ω` and attains its minimum by the extreme value theorem; that minimum is non-zero by hypothesis. The same argument is then given for the eigenvalues, using the continuity of the eigenvalues of a symmetric matrix and the fact that a vanishing eigenvalue would make the determinant vanish. The statement of the lemma has also been made precise (it previously omitted the hypotheses and the strictness of the bounds).

We take the opportunity to note that Lemma 2 also contained errors: the operator `A(L)` was written with `∂/∂q̇` and `q` where the derivation requires `∂/∂q` and `q̇`, and the conclusion of the proof read `αA(L) + βA(L)` instead of `αA(L₁) + βA(L₂)`. Both are corrected, and the lemma now covers `H` as well, since its linearity is used in the proof of Theorem 1.

**Loss function mismatch.** Accepted. Section 5 now presents both forms and the relation between them. Eq. (24) gives the hard constraint as an infinite barrier and states plainly that in this form the penalty is neither continuous nor differentiable and therefore cannot be used with a gradient method. Eq. (25) gives the smooth penalty that is actually implemented, written out completely — the SiLU activation, the shift by `min SiLU ≈ −0.2785` that makes every summand non-negative, the averaging over the eigenvalues and over the trajectory, and the hyperparameters `α` and `β`. The text then shows that the smooth penalty recovers the hard constraint in the limit `β → ∞`, and Eq. (26) states the total loss that is minimized. The duplicated and inconsistent presentation of the penalty that previously appeared in Section 7 has been removed; Section 7 now only reports the values `α = 5`, `β = 12` and the resulting location of the barrier minimum.

We also corrected the hard-constraint formula itself, which read `+∞[(H) > 1]` — the eigenvalue operator `λ` was missing and the inequality pointed the wrong way, since the penalty must be active when the constraint is *violated*.

### 3.3 Experimental inconsistencies.

**Missing robustness tests.** Accepted. We have removed the unsupported claims. The abstract no longer states that the model is "stable to random non-physical changes" and no longer mentions training on augmented samples, and the corresponding sentence has been removed from the conclusion. Section 8 now states the position honestly: the stability of the representation with respect to additive noise, time warping and sensor dropout is a plausible consequence of building the features from the equations of motion rather than from the waveform, but it remains a hypothesis until it is measured, and the ablation study is left for further work.

**Dataset class discrepancies.** Accepted; the reviewer's reading of the dataset is correct and the manuscript was wrong. PAMAP2 has 12 protocol activities, not 24. Moreover, the experiment does not use all 12: it uses **four** — standing, walking, running and cycling — which is why Figures 4 and 5 show four classes. The revised manuscript states this consistently in three places: Section 1 gives the 12 protocol activities of the dataset; Section 7.1 states that `K = 4`, lists the four activities, gives the reason for the choice (pure whole-body locomotion regimes, present for all nine subjects, differing in the mechanical model rather than in the manipulated object) and states explicitly that *all* reported results, both the projections in Figures 4–5 and the accuracies in Table 1, refer to this four-class problem; and the caption of Table 1 repeats that it is the four-class problem. The evaluation protocol behind the `±` values is also now stated: an 80:20 train/test split and 5-fold cross-validation on the training part.

**Weak baselines.** Accepted in principle. We agree that a comparison against ROCKET, InceptionTime and the modern ensembles is what would position the method among the state of the art, and the revised manuscript no longer leaves this unsaid: the introduction now describes the ROCKET family and the recent large-scale comparison of Middlehurst et al. properly, Section 7 states that such a comparison is the subject of a separate study, and Section 8 lists it explicitly as required further work, naming ROCKET, MiniRocket, MultiRocket, InceptionTime and the ensembles of the bake-off redux, on the full 12-class protocol.

> **Note to the authors before submission:** this is the one point where the revision is currently textual only. Decide whether to (a) run the benchmark now and report it, which would answer the comment fully, or (b) submit with the statement of further work as it stands. The response above is written for case (b); if you run the experiments, this paragraph and the corresponding text in Sections 7 and 8 must be rewritten.

### 3.4 Minor corrections and formatting.

* **"Cortes formula".** Corrected. The text now reads "Simpson's rule, one of the Newton–Cotes formulas". The formula itself has also been fixed: the summation index collided with the upper limit `k`, and the error constant has been replaced by the standard composite-Simpson bound `(t_k − t₀)h⁴/180 · max|q⁗|`. The stray `dx` in the integral is now `dt`.
* **`Q_x = 0` in Eq. (1).** Removed, and the equation now reads `∂L/∂q − (d/dt)(∂L/∂q̇) = 0` with a sentence explaining that the system is closed, so the generalized non-conservative forces are absent and the right-hand side is the zero vector of `ℝ^r`. (The original also had a typographical error, `∂L/q` instead of `∂L/∂q`.)
* **`‖ ‖` for a matrix in Eq. (2).** Corrected. The matrix is now written with square brackets and the determinant with `det(·)`; the matrix is named `H(L)` and its shape `ℝ^{r×r}` is given.
* **Norm approximation, Eqs. (26)–(28) of the original).** Rewritten. The three-line multi-index Riemann sum has been replaced by two equations: the definition of the `L²(Ω)` seminorm as an integral against the Lebesgue measure, with the vector and Frobenius norms named explicitly, and a single quadrature formula with `μ(Ω)/N` in front of a sum over `N` probe points. The accompanying text states both interpretations — a Riemann sum on a uniform grid and a Monte Carlo estimate — gives the `O(N^{−1/2})` rate of the latter, and says which one is used in the experiment and why (the number of grid nodes grows exponentially in the dimension `2r = 18`).
* **Typographical errors.** All corrected, see 2.7 above.

---

## Other corrections made in the course of the revision

Not requested by the reviewers, but corrected while revising:

* the author e-mail addresses were swapped between the two authors, and the affiliations were the unfilled template placeholders;
* the sensor placement was stated as "the elbow of the predominant hand"; PAMAP2 places the third unit on the **ankle**;
* the description of the network as "a fully-connected network with three layers" did not match the implementation (a four-layer network with two residual connections);
* two different equations shared the label `eq:model`, so one of the cross-references pointed at the wrong equation.

## One item requiring the authors' confirmation

In Table 1, the standard deviations in the "Logistic regression" row read `0.13` and `0.14` for balanced accuracy and F1-macro, while every other standard deviation in the table is of the order `0.01`. This looks like a lost leading zero. A `%` comment marking this has been left in the source next to the row. Please check against the experimental logs and correct the values before submission.
