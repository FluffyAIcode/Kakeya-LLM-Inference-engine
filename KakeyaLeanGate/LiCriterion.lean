import KakeyaLeanGate.RiemannHypothesisRoot
import Mathlib.Analysis.Calculus.IteratedDeriv.Lemmas
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
