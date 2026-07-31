import KakeyaLeanGate.RiemannHypothesisRoot
import Mathlib.Analysis.Analytic.Order
import Mathlib.Analysis.Calculus.LogDeriv
import Mathlib.Analysis.Calculus.IteratedDeriv.Lemmas
import Mathlib.Analysis.Complex.LocallyUniformLimit
import Mathlib.Analysis.SpecialFunctions.Complex.Analytic
import Mathlib.NumberTheory.LSeries.ZetaZeros

/-!
# A typed Li-coefficient route to the Riemann Hypothesis

This file separates definitions and elementary consequences from the two
substantial bridges that are not yet in Mathlib:

* identification of the derivative coefficients with the height-symmetric
  sum over non-trivial zeta zeros, counted with multiplicity;
* Li's theorem identifying non-negativity of every positive-index
  coefficient with `RiemannHypothesis`.

Neither bridge is assumed or asserted as a theorem here.
-/

open Complex Filter Finset
open scoped Topology

noncomputable section

/-- Li's normalization of the Riemann xi function.

Mathlib's `completedRiemannZeta` has assigned values at its poles, so the
pointwise expression `s * (s - 1) * completedRiemannZeta s` has the wrong
values at `0` and `1`.  The expression below uses the entire pole-subtracted
completion and is equal to that product away from the two poles.  It is
normalized by `riemannXiLi 0 = riemannXiLi 1 = 1`.
-/
def riemannXiLi (s : ℂ) : ℂ :=
  1 + s * (s - 1) * completedRiemannZeta₀ s

@[simp] theorem riemannXiLi_zero : riemannXiLi 0 = 1 := by
  simp [riemannXiLi]

@[simp] theorem riemannXiLi_one : riemannXiLi 1 = 1 := by
  simp [riemannXiLi]

theorem differentiable_riemannXiLi : Differentiable ℂ riemannXiLi := by
  unfold riemannXiLi
  exact (differentiable_const 1).add
    ((differentiable_id.mul (differentiable_id.sub (differentiable_const 1))).mul
      differentiable_completedZeta₀)

theorem riemannXiLi_one_sub (s : ℂ) :
    riemannXiLi (1 - s) = riemannXiLi s := by
  rw [riemannXiLi, riemannXiLi, completedRiemannZeta₀_one_sub]
  ring

theorem riemannXiLi_eq_mul_completedRiemannZeta
    {s : ℂ} (hs0 : s ≠ 0) (hs1 : s ≠ 1) :
    riemannXiLi s = s * (s - 1) * completedRiemannZeta s := by
  rw [riemannXiLi, completedRiemannZeta_eq]
  field_simp
  ring

theorem analyticAt_log_riemannXiLi_one :
    AnalyticAt ℂ (fun s ↦ Complex.log (riemannXiLi s)) 1 :=
  (differentiable_riemannXiLi.analyticAt 1).clog (by simp)

/-- The source-standard derivative expression for the `n`th Li coefficient.

The standard sequence starts at `n = 1`.  We totalize it at `0` by setting
that value to zero.  For positive `n` this is
`1 / (n - 1)! * d^n/ds^n (s^(n-1) log(xi(s)))` at `s = 1`.
-/
def liDerivativeCoefficient (xi : ℂ → ℂ) (n : ℕ) : ℂ :=
  if n = 0 then 0
  else
    iteratedDeriv n (fun s ↦ s ^ (n - 1) * Complex.log (xi s)) 1 /
      ((n - 1).factorial : ℂ)

/-- The concrete real sequence extracted from the derivative formula.

Taking the real part makes the target type appropriate for positivity.
The obligation that the complex derivative is actually real is kept explicit
below rather than silently assumed.
-/
def riemannLiCoefficient (n : ℕ) : ℝ :=
  (liDerivativeCoefficient riemannXiLi n).re

@[simp] theorem liDerivativeCoefficient_zero (xi : ℂ → ℂ) :
    liDerivativeCoefficient xi 0 = 0 := by
  simp [liDerivativeCoefficient]

@[simp] theorem riemannLiCoefficient_zero : riemannLiCoefficient 0 = 0 := by
  simp [riemannLiCoefficient]

theorem liDerivativeCoefficient_one (xi : ℂ → ℂ) :
    liDerivativeCoefficient xi 1 =
      deriv (fun s ↦ Complex.log (xi s)) 1 := by
  simp [liDerivativeCoefficient]

/-- Mathlib's analytic order gives the intrinsic multiplicity of a zeta zero.

The value lies in `ℕ∞`: finiteness still requires ruling out local identically
zero behavior.  This definition does not enumerate the zeros and therefore
does not by itself provide the multiset needed by the symmetric Li sum. -/
def riemannZetaZeroOrder (rho : ℂ) : ℕ∞ :=
  analyticOrderAt riemannZeta rho

/-- Away from zeta's pole, nonzero analytic order is exactly membership in
Mathlib's set of zeta zeros. -/
theorem riemannZetaZeroOrder_ne_zero_iff {rho : ℂ} (hρ : rho ≠ 1) :
    riemannZetaZeroOrder rho ≠ 0 ↔ rho ∈ riemannZetaZeros := by
  rw [riemannZetaZeroOrder, analyticOrderAt_ne_zero]
  simp only [mem_riemannZetaZeros]
  exact and_iff_right (analyticOn_riemannZeta rho hρ)

/-- Exact reality obligation for the derivative formula. -/
def LiDerivativeIsReal (xi : ℂ → ℂ) : Prop :=
  ∀ n : ℕ, 0 < n → (liDerivativeCoefficient xi n).im = 0

theorem riemannLiCoefficient_cast_eq
    (hreal : LiDerivativeIsReal riemannXiLi) {n : ℕ} (hn : 0 < n) :
    (riemannLiCoefficient n : ℂ) =
      liDerivativeCoefficient riemannXiLi n := by
  apply Complex.ext
  · simp [riemannLiCoefficient]
  · simp [riemannLiCoefficient, hreal n hn]

/-- Non-negativity at every source-relevant (positive) index. -/
def LiPositive (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, 0 < n → 0 ≤ a n

/-- Finite-prefix evidence through index `N`; this is not an RH statement. -/
def LiPositiveThrough (a : ℕ → ℝ) (N : ℕ) : Prop :=
  ∀ n : ℕ, 0 < n → n ≤ N → 0 ≤ a n

theorem LiPositive.through {a : ℕ → ℝ} (h : LiPositive a) (N : ℕ) :
    LiPositiveThrough a N :=
  fun n hn _ ↦ h n hn

theorem liPositive_iff_all_prefixes (a : ℕ → ℝ) :
    LiPositive a ↔ ∀ N : ℕ, LiPositiveThrough a N := by
  constructor
  · exact fun h N ↦ h.through N
  · intro h n hn
    exact h n n hn le_rfl

theorem LiPositiveThrough.mono {a : ℕ → ℝ} {M N : ℕ}
    (h : LiPositiveThrough a N) (hMN : M ≤ N) :
    LiPositiveThrough a M :=
  fun n hn hnM ↦ h n hn (hnM.trans hMN)

@[simp] theorem liPositiveThrough_zero (a : ℕ → ℝ) :
    LiPositiveThrough a 0 := by
  intro n hn hn0
  omega

/-- A concrete counterexample to the invalid inference from any fixed finite
positive prefix to all-index positivity. -/
def finitePrefixSpoof (N n : ℕ) : ℝ :=
  if n ≤ N then 1 else -1

theorem finitePrefixSpoof_positiveThrough (N : ℕ) :
    LiPositiveThrough (finitePrefixSpoof N) N := by
  intro n _ hn
  simp [finitePrefixSpoof, hn]

theorem finitePrefixSpoof_not_positive (N : ℕ) :
    ¬ LiPositive (finitePrefixSpoof N) := by
  intro h
  have := h (N + 1) (by omega)
  have hbad : (0 : ℝ) ≤ -1 := by
    simpa [finitePrefixSpoof, show ¬N + 1 ≤ N by omega] using this
  norm_num at hbad

/-- The all-index Li criterion, deliberately a proposition rather than a
theorem.  Proving this proposition is the source theorem bridge. -/
def RiemannLiCriterionStatement : Prop :=
  KakeyaRiemannHypothesisRoot ↔ LiPositive riemannLiCoefficient

/-- One term in the zero-sum formulation. -/
def liZeroSummand (n : ℕ) (rho : ℂ) : ℂ :=
  1 - (1 - rho⁻¹) ^ n

@[simp] theorem liZeroSummand_zero (rho : ℂ) :
    liZeroSummand 0 rho = 0 := by
  simp [liZeroSummand]

@[simp] theorem liZeroSummand_one (rho : ℂ) :
    liZeroSummand 1 rho = rho⁻¹ := by
  simp [liZeroSummand]

theorem liZeroSummand_conj (n : ℕ) (rho : ℂ) :
    liZeroSummand n (star rho) = star (liZeroSummand n rho) := by
  simp [liZeroSummand]

/-- A finite truncation of an indexed multiset of zeros.  Indices, rather
than a `Finset ℂ`, preserve repeated roots when multiplicities are supplied. -/
def liZeroPartialSum {ι : Type*} (root : ι → ℂ)
    (cutoff : Finset ι) (n : ℕ) : ℂ :=
  ∑ i ∈ cutoff, liZeroSummand n (root i)

@[simp] theorem liZeroPartialSum_zero {ι : Type*} [DecidableEq ι]
    (root : ι → ℂ) (cutoff : Finset ι) :
    liZeroPartialSum root cutoff 0 = 0 := by
  simp [liZeroPartialSum]

/-- A finite, multiplicity-aware family of nonzero zeros.

Repeated values at distinct `Fin` indices are retained, so this is an
indexed multiset rather than a set.  It is concrete data, not an assumption
about the zeros of zeta. -/
structure FiniteZeroMultiset where
  card : ℕ
  root : Fin card → ℂ
  root_ne_zero : ∀ i, root i ≠ 0

/-- One zero repeated with a prescribed positive analytic multiplicity. -/
def FiniteZeroMultiset.replicate (multiplicity : ℕ) (rho : ℂ)
    (hρ : rho ≠ 0) : FiniteZeroMultiset where
  card := multiplicity
  root := fun _ ↦ rho
  root_ne_zero := fun _ ↦ hρ

/-- The exact Li sum of a finite indexed multiset. -/
def FiniteZeroMultiset.liSum (zeros : FiniteZeroMultiset) (n : ℕ) : ℂ :=
  ∑ i, liZeroSummand n (zeros.root i)

@[simp] theorem FiniteZeroMultiset.liSum_replicate
    (multiplicity n : ℕ) (rho : ℂ) (hρ : rho ≠ 0) :
    (FiniteZeroMultiset.replicate multiplicity rho hρ).liSum n =
      multiplicity * liZeroSummand n rho := by
  change (∑ _ : Fin multiplicity, liZeroSummand n rho) =
    multiplicity * liZeroSummand n rho
  rw [Finset.sum_const, Finset.card_univ]
  norm_num [Fintype.card_fin, nsmul_eq_mul]

@[simp] theorem FiniteZeroMultiset.liSum_zero (zeros : FiniteZeroMultiset) :
    zeros.liSum 0 = 0 := by
  simp [FiniteZeroMultiset.liSum]

theorem FiniteZeroMultiset.liSum_one (zeros : FiniteZeroMultiset) :
    zeros.liSum 1 = ∑ i, (zeros.root i)⁻¹ := by
  simp [FiniteZeroMultiset.liSum]

/-- One zero contributes a nonnegative real part whenever its Li transform
`1 - 1 / rho` lies in the closed unit disk. -/
theorem liZeroSummand_re_nonneg_of_norm_le_one
    (n : ℕ) (rho : ℂ) (hρ : ‖1 - rho⁻¹‖ ≤ 1) :
    0 ≤ (liZeroSummand n rho).re := by
  have hp : ((1 - rho⁻¹) ^ n).re ≤ 1 := calc
    ((1 - rho⁻¹) ^ n).re
        ≤ |((1 - rho⁻¹) ^ n).re| := le_abs_self _
    _ ≤ ‖(1 - rho⁻¹) ^ n‖ := abs_re_le_norm _
    _ = ‖1 - rho⁻¹‖ ^ n := norm_pow _ _
    _ ≤ 1 := pow_le_one₀ (norm_nonneg _) hρ
  simpa [liZeroSummand] using sub_nonneg.mpr hp

/-- Exact finite positivity: no convergence or zero-enumeration theorem is
needed for a finite multiplicity-aware family. -/
theorem FiniteZeroMultiset.liSum_re_nonneg
    (zeros : FiniteZeroMultiset)
    (hunit : ∀ i, ‖1 - (zeros.root i)⁻¹‖ ≤ 1)
    (n : ℕ) :
    0 ≤ (zeros.liSum n).re := by
  change 0 ≤ Complex.reCLM (∑ i, liZeroSummand n (zeros.root i))
  rw [map_sum]
  exact Finset.sum_nonneg fun i _ ↦
    liZeroSummand_re_nonneg_of_norm_le_one n (zeros.root i) (hunit i)

/-- On the critical line, the Li transform has norm exactly one. -/
theorem norm_one_sub_inv_eq_one_of_re_eq_half
    {rho : ℂ} (hρ0 : rho ≠ 0) (hline : rho.re = 1 / 2) :
    ‖1 - rho⁻¹‖ = 1 := by
  have hnormSq : ‖rho - 1‖ ^ 2 = ‖rho‖ ^ 2 := by
    rw [Complex.sq_norm, Complex.sq_norm, Complex.normSq_apply, Complex.normSq_apply]
    simp only [sub_re, one_re, sub_im, one_im, sub_zero]
    rw [hline]
    ring
  have hnorm : ‖rho - 1‖ = ‖rho‖ := by
    nlinarith [norm_nonneg (rho - 1), norm_nonneg rho]
  have hid : 1 - rho⁻¹ = (rho - 1) / rho := by
    field_simp
  rw [hid, norm_div, hnorm, div_self]
  exact norm_ne_zero_iff.mpr hρ0

theorem FiniteZeroMultiset.liSum_re_nonneg_of_on_criticalLine
    (zeros : FiniteZeroMultiset)
    (hline : ∀ i, (zeros.root i).re = 1 / 2)
    (n : ℕ) :
    0 ≤ (zeros.liSum n).re :=
  zeros.liSum_re_nonneg
    (fun i ↦ (norm_one_sub_inv_eq_one_of_re_eq_half
      (zeros.root_ne_zero i) (hline i)).le) n

/-- The analytic convergence part of the height-symmetric zero formula.
Completeness and multiplicity-correctness of `root` are separate obligations,
because pinned Mathlib exposes only a zero set, not analytic zero orders. -/
def HeightSymmetricLiLimit {ι : Type*} [DecidableEq ι]
    (root : ι → ℂ) (cutoff : ℕ → Finset ι) (height : ℕ → ℝ)
    (coefficient : ℕ → ℂ) : Prop :=
  Tendsto height atTop atTop ∧
    (∀ N i, i ∈ cutoff N ↔ |(root i).im| ≤ height N) ∧
    ∀ n : ℕ, Tendsto (fun N ↦ liZeroPartialSum root (cutoff N) n)
      atTop (𝓝 (coefficient n))

/-- Cauchy form of the analytic convergence obligation.  This does not assert
that zeta zeros satisfy it. -/
def HeightSymmetricLiCauchy {ι : Type*} [DecidableEq ι]
    (root : ι → ℂ) (cutoff : ℕ → Finset ι) : Prop :=
  ∀ n : ℕ, CauchySeq (fun N ↦ liZeroPartialSum root (cutoff N) n)

/-- Completeness of `ℂ` transfers explicit Cauchy estimates into a coefficient
sequence and the corresponding symmetric-limit interface. -/
theorem exists_heightSymmetricLiLimit_of_cauchy
    {ι : Type*} [DecidableEq ι]
    (root : ι → ℂ) (cutoff : ℕ → Finset ι) (height : ℕ → ℝ)
    (hheight : Tendsto height atTop atTop)
    (hcutoff : ∀ N i, i ∈ cutoff N ↔ |(root i).im| ≤ height N)
    (hcauchy : HeightSymmetricLiCauchy root cutoff) :
    ∃ coefficient : ℕ → ℂ,
      HeightSymmetricLiLimit root cutoff height coefficient := by
  choose coefficient hcoefficient using
    fun n ↦ cauchySeq_tendsto_of_complete (hcauchy n)
  exact ⟨coefficient, hheight, hcutoff, hcoefficient⟩

/-- Transfer a symmetric zero-sum limit through exact finite approximants.
This isolates the derivative/zero-sum bridge: for xi, one must construct
`approximation` and prove both hypotheses below. -/
theorem HeightSymmetricLiLimit.coefficient_eq_of_approximation
    {ι : Type*} [DecidableEq ι]
    {root : ι → ℂ} {cutoff : ℕ → Finset ι} {height : ℕ → ℝ}
    {coefficient : ℕ → ℂ}
    (hlimit : HeightSymmetricLiLimit root cutoff height coefficient)
    (approximation : ℕ → ℕ → ℂ) (target : ℕ → ℂ)
    (hexact : ∀ N n, approximation N n =
      liZeroPartialSum root (cutoff N) n)
    (htarget : ∀ n, Tendsto (fun N ↦ approximation N n)
      atTop (𝓝 (target n))) :
    coefficient = target := by
  funext n
  apply tendsto_nhds_unique (hlimit.2.2 n)
  simpa only [hexact] using htarget n

/-- Locally uniform convergence of holomorphic functions propagates through
every iterated derivative on an open complex domain.

This is the all-order form of Mathlib's
`TendstoLocallyUniformlyOn.deriv`; the holomorphy needed at later induction
steps follows from complex analyticity. -/
theorem tendstoLocallyUniformlyOn_iteratedDeriv
    {ι : Type*} {p : Filter ι} {f : ι → ℂ → ℂ} {g : ℂ → ℂ}
    {s : Set ℂ} (hs : IsOpen s)
    (hconv : TendstoLocallyUniformlyOn f g p s)
    (hhol : ∀ᶠ i in p, DifferentiableOn ℂ (f i) s)
    (k : ℕ) :
    TendstoLocallyUniformlyOn
      (fun i ↦ iteratedDeriv k (f i)) (iteratedDeriv k g) p s := by
  induction k with
  | zero =>
      simpa only [iteratedDeriv_zero] using hconv
  | succ k ih =>
      rw [iteratedDeriv_succ]
      convert ih.deriv (by
        filter_upwards [hhol] with i hi
        rw [iteratedDeriv_eq_iterate]
        exact ((hi.analyticOnNhd hs).iterated_deriv k).differentiableOn) hs using 1
      ext i z
      simp [Function.comp_apply, iteratedDeriv_succ]

/-- Explicit nonvanishing-neighborhood transfer for logarithmic derivatives.

The denominator is required to be nonzero throughout `s`, not merely at the
coefficient center.  This stronger condition is what permits a locally
uniform quotient, which can then be differentiated to arbitrary order. -/
theorem tendstoLocallyUniformlyOn_logDeriv
    {ι : Type*} {p : Filter ι} [p.NeBot]
    {f : ι → ℂ → ℂ} {g : ℂ → ℂ}
    {s : Set ℂ} (hs : IsOpen s)
    (hconv : TendstoLocallyUniformlyOn f g p s)
    (hhol : ∀ᶠ i in p, DifferentiableOn ℂ (f i) s)
    (hg0 : ∀ z ∈ s, g z ≠ 0) :
    TendstoLocallyUniformlyOn
      (fun i z ↦ logDeriv (f i) z) (fun z ↦ logDeriv g z) p s := by
  have hgDiff : DifferentiableOn ℂ g s :=
    hconv.differentiableOn hhol hs
  have hderiv := hconv.deriv hhol hs
  simp only [logDeriv, Pi.div_apply]
  convert hderiv.div₀ hconv (hgDiff.deriv hs).continuousOn
    hgDiff.continuousOn hg0 using 1
  · ext i z
    rfl
  · ext z
    rfl

/-- All Taylor coefficients of logarithmic derivatives converge under local
uniform convergence, holomorphy, and explicit nonvanishing on a common open
neighborhood. -/
theorem iteratedDeriv_logDeriv_tendsto
    {ι : Type*} {p : Filter ι} [p.NeBot]
    {f : ι → ℂ → ℂ} {g : ℂ → ℂ} {s : Set ℂ} {x : ℂ}
    (hs : IsOpen s) (hx : x ∈ s)
    (hconv : TendstoLocallyUniformlyOn f g p s)
    (hhol : ∀ᶠ i in p, DifferentiableOn ℂ (f i) s)
    (hf0 : ∀ᶠ i in p, ∀ z ∈ s, f i z ≠ 0)
    (hg0 : ∀ z ∈ s, g z ≠ 0) (k : ℕ) :
    Tendsto (fun i ↦ iteratedDeriv k (logDeriv (f i)) x) p
      (𝓝 (iteratedDeriv k (logDeriv g) x)) := by
  have hlog := tendstoLocallyUniformlyOn_logDeriv hs hconv hhol hg0
  have hlogHol : ∀ᶠ i in p, DifferentiableOn ℂ (logDeriv (f i)) s := by
    filter_upwards [hhol, hf0] with i hi hi0
    change DifferentiableOn ℂ (deriv (f i) / f i) s
    exact (hi.deriv hs).div hi hi0
  exact (tendstoLocallyUniformlyOn_iteratedDeriv
    hs hlog hlogHol k).tendsto_at hx

/-- The finite factor whose logarithmic derivative generates one zero's Li
summands. -/
def liFiniteProductFactor (rho z : ℂ) : ℂ :=
  (1 - (1 - rho⁻¹) * z) / (1 - z)

/-- A finite product retaining every indexed occurrence of a repeated zero. -/
def FiniteZeroMultiset.generatingProduct
    (zeros : FiniteZeroMultiset) (z : ℂ) : ℂ :=
  ∏ i, liFiniteProductFactor (zeros.root i) z

/-- The rational generating term for one zero.  Its formal power-series
coefficients are `liZeroSummand (n + 1) rho`. -/
def liZeroGeneratingTerm (rho z : ℂ) : ℂ :=
  1 / (1 - z) - (1 - rho⁻¹) / (1 - (1 - rho⁻¹) * z)

theorem logDeriv_liFiniteProductFactor
    (rho z : ℂ) (hz : 1 - z ≠ 0)
    (hρz : 1 - (1 - rho⁻¹) * z ≠ 0) :
    logDeriv (liFiniteProductFactor rho) z =
      liZeroGeneratingTerm rho z := by
  rw [logDeriv_apply]
  unfold liFiniteProductFactor liZeroGeneratingTerm
  have hnum : HasDerivAt
      (fun w : ℂ ↦ 1 - (1 - rho⁻¹) * w) (-(1 - rho⁻¹)) z := by
    simpa [sub_eq_add_neg] using
      ((hasDerivAt_id z).const_mul (1 - rho⁻¹)).neg.const_add 1
  have hden : HasDerivAt (fun w : ℂ ↦ 1 - w) (-1) z := by
    simpa [sub_eq_add_neg] using (hasDerivAt_id z).neg.const_add 1
  have hρz' : 1 + rho⁻¹ * z - z ≠ 0 := by
    convert hρz using 1
    ring
  rw [(hnum.fun_div hden hz).deriv]
  field_simp [hz, hρz, hρz']
  ring

/-- Exact finite-product logarithmic-derivative identity.  The hypotheses
simply say that the displayed rational factors are defined and nonzero at
`z`; near `z = 0` they hold automatically. -/
theorem FiniteZeroMultiset.logDeriv_generatingProduct
    (zeros : FiniteZeroMultiset) (z : ℂ)
    (hz : 1 - z ≠ 0)
    (hfactor : ∀ i, 1 - (1 - (zeros.root i)⁻¹) * z ≠ 0) :
    logDeriv zeros.generatingProduct z =
      ∑ i, liZeroGeneratingTerm (zeros.root i) z := by
  unfold FiniteZeroMultiset.generatingProduct
  rw [logDeriv_prod]
  · apply Finset.sum_congr rfl
    intro i _
    exact logDeriv_liFiniteProductFactor (zeros.root i) z hz (hfactor i)
  · intro i _
    exact div_ne_zero (hfactor i) hz
  · intro i _
    unfold liFiniteProductFactor
    fun_prop

@[simp] theorem liZeroGeneratingTerm_zero (rho : ℂ) :
    liZeroGeneratingTerm rho 0 = liZeroSummand 1 rho := by
  simp [liZeroGeneratingTerm]

theorem FiniteZeroMultiset.logDeriv_generatingProduct_zero
    (zeros : FiniteZeroMultiset) :
    logDeriv zeros.generatingProduct 0 = zeros.liSum 1 := by
  rw [zeros.logDeriv_generatingProduct 0 (by norm_num) (by simp)]
  simp [FiniteZeroMultiset.liSum]

/-- The normalized Taylor coefficient of the finite product's logarithmic
derivative at the origin. -/
def FiniteZeroMultiset.logDerivCoefficient
    (zeros : FiniteZeroMultiset) (k : ℕ) : ℂ :=
  iteratedDeriv k (logDeriv zeros.generatingProduct) 0 /
    (k.factorial : ℂ)

/-- The exact finite all-order coefficient identity needed by the limit
transfer.  Keeping it named makes clear that this is a finite algebraic
obligation, distinct from every convergence assumption. -/
def FiniteLiTaylorIdentity (zeros : FiniteZeroMultiset) : Prop :=
  ∀ k : ℕ, zeros.logDerivCoefficient k = zeros.liSum (k + 1)

/-- The first finite Taylor coefficient is already unconditional. -/
theorem FiniteZeroMultiset.logDerivCoefficient_zero
    (zeros : FiniteZeroMultiset) :
    zeros.logDerivCoefficient 0 = zeros.liSum 1 := by
  simp [FiniteZeroMultiset.logDerivCoefficient,
    zeros.logDeriv_generatingProduct_zero]

/-- Rigorous all-index transfer from finite multiplicity-aware zero sums to
the Taylor coefficients of a locally uniform nonvanishing product limit.

The hypotheses separate the four independent obligations:

* local uniform convergence of the finite products;
* holomorphy on one common open neighborhood of `0`;
* nonvanishing of every approximant and the limit there;
* the finite all-order coefficient identity.

No zeta-specific product or convergence claim is hidden in this theorem. -/
theorem finiteZeroLiSums_tendsto_of_generatingProducts
    (zeros : ℕ → FiniteZeroMultiset) (g : ℂ → ℂ) (s : Set ℂ)
    (hs : IsOpen s) (h0 : (0 : ℂ) ∈ s)
    (hconv : TendstoLocallyUniformlyOn
      (fun N ↦ (zeros N).generatingProduct) g atTop s)
    (hhol : ∀ᶠ N in atTop,
      DifferentiableOn ℂ (zeros N).generatingProduct s)
    (happrox0 : ∀ᶠ N in atTop, ∀ z ∈ s,
      (zeros N).generatingProduct z ≠ 0)
    (hlimit0 : ∀ z ∈ s, g z ≠ 0)
    (hexact : ∀ N, FiniteLiTaylorIdentity (zeros N))
    (k : ℕ) :
    Tendsto (fun N ↦ (zeros N).liSum (k + 1)) atTop
      (𝓝 (iteratedDeriv k (logDeriv g) 0 / (k.factorial : ℂ))) := by
  have hderiv := iteratedDeriv_logDeriv_tendsto
    hs h0 hconv hhol happrox0 hlimit0 k
  have hnormalized := hderiv.div_const (k.factorial : ℂ)
  simpa only [FiniteLiTaylorIdentity,
    FiniteZeroMultiset.logDerivCoefficient] using
    hnormalized.congr' (Filter.Eventually.of_forall fun N ↦ hexact N k)

/-- Truncated generating polynomial for any coefficient sequence. -/
def liGeneratingPolynomial (a : ℕ → ℝ) (N : ℕ) (x : ℝ) : ℝ :=
  ∑ n ∈ range N, a n * x ^ n

@[simp] theorem liGeneratingPolynomial_zero (a : ℕ → ℝ) (x : ℝ) :
    liGeneratingPolynomial a 0 x = 0 := by
  simp [liGeneratingPolynomial]

theorem liGeneratingPolynomial_succ (a : ℕ → ℝ) (N : ℕ) (x : ℝ) :
    liGeneratingPolynomial a (N + 1) x =
      liGeneratingPolynomial a N x + a N * x ^ N := by
  simp [liGeneratingPolynomial, sum_range_succ]

end
