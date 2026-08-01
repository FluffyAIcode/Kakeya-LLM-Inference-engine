import Mathlib.Analysis.Calculus.ContDiff.Convolution
import Mathlib.Analysis.Calculus.BumpFunction.Normed
import Mathlib.Analysis.Distribution.TestFunction
import Mathlib.Analysis.Distribution.SchwartzSpace.Fourier
import Mathlib.Analysis.Calculus.IteratedDeriv.Lemmas
import Mathlib.Analysis.Calculus.Deriv.Star
import Mathlib.Analysis.Complex.JensenFormula
import Mathlib.Analysis.Fourier.Convolution
import Mathlib.Analysis.InnerProductSpace.GramMatrix
import Mathlib.Analysis.MellinTransform
import Mathlib.Analysis.Meromorphic.Divisor
import Mathlib.Analysis.Normed.Group.Tannery
import Mathlib.Analysis.PSeries
import Mathlib.Analysis.Real.Pi.Bounds
import Mathlib.Analysis.SpecialFunctions.Gamma.Deligne
import Mathlib.Analysis.SpecialFunctions.Gamma.Digamma
import Mathlib.Analysis.Analytic.Uniqueness
import Mathlib.LinearAlgebra.Matrix.NonsingularInverse
import Mathlib.LinearAlgebra.Vandermonde
import Mathlib.Logic.Encodable.Basic
import Mathlib.NumberTheory.LSeries.Dirichlet
import Mathlib.NumberTheory.LSeries.RiemannZeta
import Mathlib.NumberTheory.LSeries.ZetaZeros
import Mathlib.Topology.Algebra.InfiniteSum.Real
import Mathlib.Topology.Order.MonotoneConvergence
import Mathlib.Topology.UniformSpace.UniformConvergence

/-!
# The Weil-positivity route: unconditional interface and finite lemmas

This file deliberately stops before the Riemann-zeta explicit formula.  Mathlib
`v4.32.0-rc1` has Mellin transforms and the completed zeta function, but no
enumeration of nontrivial zeros and no Guinand--Weil explicit formula.

We use the critical-line-centred logarithmic normalization.  A multiplicative
test `g` on `(0, ∞)` corresponds to a test `f` on `ℝ` by `x = exp t`; its
centred Mellin transform is the bilateral Laplace transform

`F(z) = ∫ t, f(t) * exp(z*t) dt`, where `z = s - 1/2`.

Thus RH is the assertion that the spectral parameters `z` are purely
imaginary.  The involution is `f⋆(t) = conj (f(-t))`.
-/

open Complex Filter MeasureTheory Set TopologicalSpace
open scoped ComplexConjugate ComplexOrder Convolution ContDiff Distributions FourierTransform
  Pointwise SchwartzMap Topology

noncomputable section

namespace WeilPositivity

/-- Smooth compactly-supported complex tests on the logarithmic line. -/
abbrev Test := 𝓓((⊤ : Opens ℝ), ℂ)

/-- The ambient Schwartz space containing every logarithmic test.  Mathlib's
Fourier theory is formulated on this space. -/
abbrev SchwartzTest := 𝓢(ℝ, ℂ)

/-- The canonical inclusion `C_c^∞(ℝ) ↪ 𝓢(ℝ)`, assembled from Mathlib's
`HasCompactSupport.toSchwartzMap`. -/
def toSchwartz (f : Test) : SchwartzTest :=
  f.hasCompactSupport.toSchwartzMap f.contDiff

@[simp]
theorem toSchwartz_apply (f : Test) (x : ℝ) : toSchwartz f x = f x :=
  rfl

/-! ## Pinned zeta and prime-side declarations -/

/-- A compact disk contains only finitely many zeta zeros.  This is a genuine
Mathlib theorem about the zero set, but it does not record zero multiplicity. -/
def zetaZeroWindow (R : ℝ) : Set ℂ :=
  Metric.closedBall 0 R ∩ riemannZetaZeros

theorem zetaZeroWindow_finite (R : ℝ) : (zetaZeroWindow R).Finite :=
  (isCompact_closedBall (0 : ℂ) R).inter_riemannZetaZeros_finite

/-- The corresponding finite set of distinct zeros.  It is useful for local
restricted statements, but cannot be substituted for a multiplicity-aware
Guinand--Weil zero sum. -/
def zetaZeroWindowFinset (R : ℝ) : Finset ℂ :=
  (zetaZeroWindow_finite R).toFinset

@[simp]
theorem mem_zetaZeroWindowFinset {R : ℝ} {s : ℂ} :
    s ∈ zetaZeroWindowFinset R ↔ s ∈ Metric.closedBall 0 R ∧ riemannZeta s = 0 := by
  simp [zetaZeroWindowFinset, zetaZeroWindow, mem_riemannZetaZeros]

/-- Positive-height nontrivial zeta zeros up to height `T`. -/
def positiveNontrivialZetaZeroWindow (T : ℝ) : Set ℂ :=
  {z | riemannZeta z = 0 ∧ 0 < z.im ∧ z.im ≤ T ∧ 0 < z.re ∧ z.re < 1}

/-- Every bounded positive-height nontrivial-zero window is finite.  This uses
Mathlib's actual discreteness theorem for `riemannZetaZeros`. -/
theorem positiveNontrivialZetaZeroWindow_finite (T : ℝ) :
    (positiveNontrivialZetaZeroWindow T).Finite := by
  apply (zetaZeroWindow_finite (|T| + 2)).subset
  intro z hz
  rcases hz with ⟨hzeta, himpos, himT, hrepos, hreone⟩
  constructor
  · rw [Metric.mem_closedBall, dist_zero_right]
    calc
      ‖z‖ ≤ |z.re| + |z.im| := Complex.norm_le_abs_re_add_abs_im z
      _ ≤ 1 + |T| := by
        rw [abs_of_pos hrepos, abs_of_pos himpos]
        gcongr
        exact himT.trans (le_abs_self T)
      _ ≤ |T| + 2 := by linarith
  · exact hzeta

def positiveNontrivialZetaZeroWindowFinset (T : ℝ) : Finset ℂ :=
  (positiveNontrivialZetaZeroWindow_finite T).toFinset

@[simp]
theorem mem_positiveNontrivialZetaZeroWindowFinset {T : ℝ} {z : ℂ} :
    z ∈ positiveNontrivialZetaZeroWindowFinset T ↔
      riemannZeta z = 0 ∧ 0 < z.im ∧ z.im ≤ T ∧ 0 < z.re ∧ z.re < 1 := by
  simp [positiveNontrivialZetaZeroWindowFinset, positiveNontrivialZetaZeroWindow]

/-- In the open critical strip, zeta and completed zeta have exactly the same
zero locations because the real Gamma factor is nonvanishing. -/
theorem riemannZeta_eq_zero_iff_completedRiemannZeta_eq_zero_of_strip
    {ρ : ℂ} (hρ0 : 0 < ρ.re) :
    riemannZeta ρ = 0 ↔ completedRiemannZeta ρ = 0 := by
  have hρne : ρ ≠ 0 := by
    intro h
    subst ρ
    norm_num at hρ0
  have hGamma : Complex.Gammaℝ ρ ≠ 0 :=
    Complex.Gammaℝ_ne_zero_of_re_pos hρ0
  rw [riemannZeta_def_of_ne_zero hρne]
  simp [hGamma]

/-- Functional-equation reflection preserves every nontrivial zeta zero
location. -/
theorem riemannZeta_one_sub_eq_zero_of_strip
    {ρ : ℂ} (hzero : riemannZeta ρ = 0)
    (hρ0 : 0 < ρ.re) (hρ1 : ρ.re < 1) :
    riemannZeta (1 - ρ) = 0 := by
  have hcomp :
      completedRiemannZeta ρ = 0 :=
    (riemannZeta_eq_zero_iff_completedRiemannZeta_eq_zero_of_strip hρ0).mp hzero
  have hreflect : completedRiemannZeta (1 - ρ) = 0 := by
    rw [completedRiemannZeta_one_sub]
    exact hcomp
  have hre : 0 < (1 - ρ).re := by
    norm_num
    linarith
  exact
    (riemannZeta_eq_zero_iff_completedRiemannZeta_eq_zero_of_strip hre).mpr
      hreflect

/-- In centered coordinates, the holomorphic functional equation is the
reflection `z ↦ -z`. -/
theorem centeredRiemannZeta_neg_eq_zero
    {z : ℂ} (hzero : riemannZeta ((1 / 2 : ℂ) + z) = 0)
    (hz0 : -1 / 2 < z.re) (hz1 : z.re < 1 / 2) :
    riemannZeta ((1 / 2 : ℂ) - z) = 0 := by
  have hρ0 : 0 < ((1 / 2 : ℂ) + z).re := by
    norm_num
    linarith
  have hρ1 : ((1 / 2 : ℂ) + z).re < 1 := by
    norm_num
    linarith
  have h := riemannZeta_one_sub_eq_zero_of_strip hzero hρ0 hρ1
  convert h using 1 <;> ring

/-- Complex conjugation commutes with zeta in its absolutely convergent
half-plane. -/
theorem riemannZeta_conj_of_one_lt_re
    {s : ℂ} (hs : 1 < s.re) :
    riemannZeta (star s) = star (riemannZeta s) := by
  have hstars : 1 < (star s).re := by simpa using hs
  rw [zeta_eq_tsum_one_div_nat_add_one_cpow hstars,
    zeta_eq_tsum_one_div_nat_add_one_cpow hs]
  change (∑' n : ℕ, 1 / (n + 1 : ℂ) ^ star s) =
    Complex.conjLIE.toContinuousLinearEquiv
      (∑' n : ℕ, 1 / (n + 1 : ℂ) ^ s)
  rw [Complex.conjLIE.toContinuousLinearEquiv.map_tsum]
  apply tsum_congr
  intro n
  change 1 / (n + 1 : ℂ) ^ conj s =
    conj (1 / (n + 1 : ℂ) ^ s)
  rw [map_div₀, map_one]
  rw [Complex.cpow_conj]
  · simp
  · norm_num [Complex.arg]
    exact ⟨by positivity, ne_of_lt Real.pi_pos⟩

/-- The holomorphic conjugate reflection of zeta. -/
def conjugateReflectedRiemannZeta (s : ℂ) : ℂ :=
  (starRingEnd ℂ) (riemannZeta ((starRingEnd ℂ) s))

/-- Complex conjugation commutes with the meromorphic continuation of zeta
away from its pole.  The proof analytically continues the absolutely
convergent Dirichlet-series identity. -/
theorem riemannZeta_conj {s : ℂ} (hs : s ≠ 1) :
    riemannZeta (star s) = star (riemannZeta s) := by
  let U : Set ℂ := ({1} : Set ℂ)ᶜ
  have hUopen : IsOpen U := isOpen_compl_singleton
  have hUpre : IsPreconnected U :=
    (isConnected_compl_singleton_of_one_lt_rank (by simp) (1 : ℂ)).isPreconnected
  have hgdiff : DifferentiableOn ℂ conjugateReflectedRiemannZeta U := by
    intro z hz
    have hzstar : star z ≠ 1 := by
      intro h
      apply hz
      simp only [U, Set.mem_compl_iff, Set.mem_singleton_iff]
      calc
        z = star (star z) := by simp
        _ = star (1 : ℂ) := congrArg star h
        _ = 1 := by simp
    have h0 :
        DifferentiableAt ℂ (conj ∘ riemannZeta ∘ conj) (conj (conj z)) :=
      (differentiableAt_riemannZeta hzstar).conj_conj
    have h1 :
        DifferentiableWithinAt ℂ (conj ∘ riemannZeta ∘ conj) U (conj (conj z)) :=
      h0.differentiableWithinAt
    change DifferentiableWithinAt ℂ
      (fun x ↦ (starRingEnd ℂ) (riemannZeta ((starRingEnd ℂ) x))) U z
    simpa [Function.comp_def] using h1
  have hg : AnalyticOnNhd ℂ conjugateReflectedRiemannZeta U :=
    hgdiff.analyticOnNhd hUopen
  have htwo : (2 : ℂ) ∈ U := by simp [U]
  have hevent :
      conjugateReflectedRiemannZeta =ᶠ[𝓝 (2 : ℂ)] riemannZeta := by
    have hhalf : {z : ℂ | 1 < z.re} ∈ 𝓝 (2 : ℂ) := by
      exact (isOpen_lt continuous_const continuous_re).mem_nhds (by norm_num)
    filter_upwards [hhalf] with z hz
    have h := congrArg star (riemannZeta_conj_of_one_lt_re hz)
    simpa [conjugateReflectedRiemannZeta] using h
  have heq :
      Set.EqOn conjugateReflectedRiemannZeta riemannZeta U :=
    hg.eqOn_of_preconnected_of_eventuallyEq analyticOn_riemannZeta hUpre htwo hevent
  have hpoint := heq (show s ∈ U by simpa [U])
  have hstar := congrArg star hpoint
  simpa [conjugateReflectedRiemannZeta] using hstar

/-- Iterated complex derivatives commute with holomorphic conjugate
reflection. -/
theorem iteratedDeriv_conj_conj (f : ℂ → ℂ) (n : ℕ) :
    iteratedDeriv n (conj ∘ f ∘ conj) =
      conj ∘ iteratedDeriv n f ∘ conj := by
  induction n with
  | zero => rfl
  | succ n ih =>
      rw [iteratedDeriv_succ, iteratedDeriv_succ, ih, deriv_conj_conj]

/-- Conjugation preserves every zeta zero away from the pole. -/
theorem riemannZeta_star_eq_zero_iff {s : ℂ} (hs : s ≠ 1) :
    riemannZeta (star s) = 0 ↔ riemannZeta s = 0 := by
  rw [riemannZeta_conj hs]
  simp

/-- Completed zeta is analytic away from its two poles. -/
theorem analyticAt_completedRiemannZeta {s : ℂ} (hs0 : s ≠ 0) (hs1 : s ≠ 1) :
    AnalyticAt ℂ completedRiemannZeta s := by
  let U : Set ℂ := ({0} : Set ℂ)ᶜ ∩ ({1} : Set ℂ)ᶜ
  apply DifferentiableOn.analyticAt (s := U)
  · intro z hz
    exact (differentiableAt_completedZeta
      (by simpa [U] using hz.1) (by simpa [U] using hz.2)).differentiableWithinAt
  · exact (isOpen_compl_singleton.inter isOpen_compl_singleton).mem_nhds
      ⟨by simpa [U], by simpa [U]⟩

/-- Actual nontrivial zeta zeros in the open critical strip. -/
def nontrivialRiemannZetaZeros : Set ℂ :=
  {ρ | riemannZeta ρ = 0 ∧ 0 < ρ.re ∧ ρ.re < 1}

theorem nontrivialRiemannZetaZeros_countable :
    nontrivialRiemannZetaZeros.Countable := by
  have hzeta : riemannZetaZeros.Countable :=
    isClosed_riemannZetaZeros.isLindelof.countable_of_isDiscrete
      isDiscrete_riemannZetaZeros
  exact hzeta.mono fun ρ hρ ↦ hρ.1

/-- The exact actual zero-location type, before choosing symmetry-orbit
representatives. -/
def ActualNontrivialZetaZero := ↥nontrivialRiemannZetaZeros

noncomputable instance : Countable ActualNontrivialZetaZero :=
  nontrivialRiemannZetaZeros_countable

/-- The same actual zero set in coordinates centered at `1 / 2`. -/
def actualCenteredNontrivialZetaZeros : Set ℂ :=
  {z | riemannZeta ((1 / 2 : ℂ) + z) = 0 ∧
    -1 / 2 < z.re ∧ z.re < 1 / 2}

/-- Centering at `1 / 2` is an exact equivalence of actual nontrivial zeta
zero locations; this does not choose orbit representatives or discard
multiplicity. -/
def actualZetaZeroCenteringEquiv :
    ActualNontrivialZetaZero ≃ ↥actualCenteredNontrivialZetaZeros where
  toFun ρ := ⟨ρ.val - (1 / 2 : ℂ), by
    rcases ρ.property with ⟨hzero, hρ0, hρ1⟩
    constructor
    · convert hzero using 1 <;> ring
    constructor <;> norm_num <;> linarith⟩
  invFun z := ⟨(1 / 2 : ℂ) + z.val, by
    rcases z.property with ⟨hzero, hz0, hz1⟩
    exact ⟨hzero, by norm_num; linarith, by norm_num; linarith⟩⟩
  left_inv ρ := by
    apply Subtype.ext
    change (1 / 2 : ℂ) + (ρ.val - 1 / 2) = ρ.val
    ring
  right_inv z := by
    apply Subtype.ext
    change (1 / 2 : ℂ) + z.val - 1 / 2 = z.val
    ring

noncomputable instance : Countable ↥actualCenteredNontrivialZetaZeros :=
  Countable.of_equiv ActualNontrivialZetaZero actualZetaZeroCenteringEquiv

/-- Actual finite centered zero window, forgetting multiplicity only for the
purpose of constructing one zero factor per distinct location. -/
def centeredZetaZeroWindowFinset (T : ℝ) : Finset ℂ :=
  (positiveNontrivialZetaZeroWindowFinset T).image
    (fun ρ ↦ ρ - (1 / 2 : ℂ))

/-- All actual centered locations in a finite window except the selected
target and its reflection partner. -/
def localCenteredZetaNeighbors (T : ℝ) (z : ℂ) : List ℂ :=
  (((centeredZetaZeroWindowFinset T).erase z).erase (-star z)).toList

theorem localCenteredZetaNeighbors_nodup
    (T : ℝ) {z : ℂ} (hoff : z ≠ -star z) :
    (z :: -star z :: localCenteredZetaNeighbors T z).Nodup := by
  simp [localCenteredZetaNeighbors]
  exact ⟨hoff, Finset.nodup_toList _⟩

theorem mem_localCenteredZetaNeighbors
    {T : ℝ} {z w : ℂ} (hw : w ∈ centeredZetaZeroWindowFinset T)
    (hwz : w ≠ z) (hwp : w ≠ -star z) :
    w ∈ localCenteredZetaNeighbors T z := by
  simp [localCenteredZetaNeighbors, hw, hwz]
  exact hwp

/-- Actual analytic multiplicity of a zeta zero, defined by Mathlib's local
analytic order. -/
def zetaZeroMultiplicity (z : ℂ) : ℕ :=
  analyticOrderNatAt riemannZeta z

/-- Actual analytic multiplicity is invariant under complex conjugation. -/
theorem zetaZeroMultiplicity_star {s : ℂ} (hs : s ≠ 1) :
    zetaZeroMultiplicity (star s) = zetaZeroMultiplicity s := by
  have hsstar : star s ≠ 1 := by
    intro h
    apply hs
    calc
      s = star (star s) := by simp
      _ = star (1 : ℂ) := congrArg star h
      _ = 1 := by simp
  have han_s : AnalyticAt ℂ riemannZeta s :=
    analyticOn_riemannZeta s (by simpa)
  have han_star : AnalyticAt ℂ riemannZeta (star s) :=
    analyticOn_riemannZeta (star s) (by simpa)
  let g : ℂ → ℂ := conj ∘ riemannZeta ∘ conj
  have hevent : g =ᶠ[𝓝 (star s)] riemannZeta := by
    have hU : ({1} : Set ℂ)ᶜ ∈ 𝓝 (star s) :=
      isOpen_compl_singleton.mem_nhds (by simpa)
    filter_upwards [hU] with z hz
    have hz1 : z ≠ 1 := by simpa using hz
    have h := congrArg star (riemannZeta_conj hz1)
    simpa [g, Function.comp_def] using h
  have hang : AnalyticAt ℂ g (star s) :=
    han_star.congr hevent.symm
  have horder_g :
      analyticOrderAt g (star s) = analyticOrderAt riemannZeta (star s) :=
    analyticOrderAt_congr hevent
  have horders :
      analyticOrderAt g (star s) = analyticOrderAt riemannZeta s := by
    apply ENat.eq_of_forall_natCast_le_iff
    intro n
    rw [natCast_le_analyticOrderAt_iff_iteratedDeriv_eq_zero hang,
      natCast_le_analyticOrderAt_iff_iteratedDeriv_eq_zero han_s]
    constructor
    · intro h k hk
      have hk0 := h k hk
      rw [iteratedDeriv_conj_conj] at hk0
      simpa [g, Function.comp_def] using congrArg star hk0
    · intro h k hk
      rw [iteratedDeriv_conj_conj]
      simp [g, Function.comp_def, h k hk]
  unfold zetaZeroMultiplicity analyticOrderNatAt
  rw [← horder_g, horders]

/-- Analytic multiplicity for the completed zeta function. -/
def completedZetaZeroMultiplicity (z : ℂ) : ℕ :=
  analyticOrderNatAt completedRiemannZeta z

/-- In the open right half-plane, the nonvanishing inverse Gamma factor
shows that zeta and completed zeta have identical analytic multiplicities. -/
theorem zetaZeroMultiplicity_eq_completed
    {s : ℂ} (hs0 : 0 < s.re) (hs1 : s.re < 1) :
    zetaZeroMultiplicity s = completedZetaZeroMultiplicity s := by
  have hsne0 : s ≠ 0 := by
    intro h
    subst s
    norm_num at hs0
  have hsne1 : s ≠ 1 := by
    intro h
    subst s
    norm_num at hs1
  have hcomp := analyticAt_completedRiemannZeta hsne0 hsne1
  have hinv : AnalyticAt ℂ (fun z ↦ (Complex.Gammaℝ z)⁻¹) s :=
    Complex.differentiable_Gammaℝ_inv.analyticAt s
  have hinvne : (Complex.Gammaℝ s)⁻¹ ≠ 0 :=
    inv_ne_zero (Complex.Gammaℝ_ne_zero_of_re_pos hs0)
  have horderInv :
      analyticOrderAt (fun z ↦ (Complex.Gammaℝ z)⁻¹) s = 0 :=
    hinv.analyticOrderAt_eq_zero.mpr hinvne
  have hevent :
      riemannZeta =ᶠ[𝓝 s]
        completedRiemannZeta * fun z ↦ (Complex.Gammaℝ z)⁻¹ := by
    filter_upwards [eventually_ne_nhds hsne0] with z hz
    rw [riemannZeta_def_of_ne_zero hz]
    rfl
  have horder :
      analyticOrderAt riemannZeta s =
        analyticOrderAt completedRiemannZeta s := by
    rw [analyticOrderAt_congr hevent]
    rw [analyticOrderAt_mul hcomp hinv, horderInv, add_zero]
  unfold zetaZeroMultiplicity completedZetaZeroMultiplicity analyticOrderNatAt
  rw [horder]

/-- Functional-equation reflection preserves completed-zeta multiplicity. -/
theorem completedZetaZeroMultiplicity_one_sub
    {s : ℂ} (hs0 : 0 < s.re) (hs1 : s.re < 1) :
    completedZetaZeroMultiplicity (1 - s) =
      completedZetaZeroMultiplicity s := by
  let r : ℂ → ℂ := fun z ↦ 1 - z
  have hran : 0 < (1 - s).re := by
    norm_num
    linarith
  have hr0 : AnalyticAt ℂ r s := by fun_prop
  have hrderiv : deriv r s ≠ 0 := by
    simp [r]
  have hcomp :
      analyticOrderAt (completedRiemannZeta ∘ r) s =
        analyticOrderAt completedRiemannZeta (1 - s) := by
    simpa [r] using
      (analyticOrderAt_comp_of_deriv_ne_zero
        (f := completedRiemannZeta) (g := r) (z₀ := s) hr0 hrderiv)
  have hfun :
      completedRiemannZeta ∘ r =ᶠ[𝓝 s] completedRiemannZeta :=
    Eventually.of_forall fun z ↦ by
      simpa [r, Function.comp_def] using completedRiemannZeta_one_sub z
  have horder :
      analyticOrderAt completedRiemannZeta (1 - s) =
        analyticOrderAt completedRiemannZeta s := by
    rw [← hcomp, analyticOrderAt_congr hfun]
  unfold completedZetaZeroMultiplicity analyticOrderNatAt
  rw [horder]

/-- Functional-equation reflection preserves actual zeta multiplicity in the
open critical strip. -/
theorem zetaZeroMultiplicity_one_sub
    {s : ℂ} (hs0 : 0 < s.re) (hs1 : s.re < 1) :
    zetaZeroMultiplicity (1 - s) = zetaZeroMultiplicity s := by
  have hran0 : 0 < (1 - s).re := by
    norm_num
    linarith
  have hran1 : (1 - s).re < 1 := by
    norm_num
    linarith
  rw [zetaZeroMultiplicity_eq_completed hran0 hran1,
    completedZetaZeroMultiplicity_one_sub hs0 hs1,
    zetaZeroMultiplicity_eq_completed hs0 hs1]

/-- Complex conjugation preserves completed-zeta zero locations in the open
critical strip. -/
theorem completedRiemannZeta_star_eq_zero_iff
    {s : ℂ} (hs0 : 0 < s.re) (hs1 : s.re < 1) :
    completedRiemannZeta (star s) = 0 ↔
      completedRiemannZeta s = 0 := by
  have hstar0 : 0 < (star s).re := by simpa using hs0
  have hsne1 : s ≠ 1 := by
    intro h
    subst s
    norm_num at hs1
  rw [← riemannZeta_eq_zero_iff_completedRiemannZeta_eq_zero_of_strip hstar0,
    ← riemannZeta_eq_zero_iff_completedRiemannZeta_eq_zero_of_strip hs0,
    riemannZeta_star_eq_zero_iff hsne1]

/-- Complex conjugation preserves completed-zeta analytic multiplicity in
the open critical strip. -/
theorem completedZetaZeroMultiplicity_star
    {s : ℂ} (hs0 : 0 < s.re) (hs1 : s.re < 1) :
    completedZetaZeroMultiplicity (star s) =
      completedZetaZeroMultiplicity s := by
  have hstar0 : 0 < (star s).re := by simpa using hs0
  have hstar1 : (star s).re < 1 := by simpa using hs1
  have hsne1 : s ≠ 1 := by
    intro h
    subst s
    norm_num at hs1
  rw [← zetaZeroMultiplicity_eq_completed hstar0 hstar1,
    zetaZeroMultiplicity_star hsne1,
    zetaZeroMultiplicity_eq_completed hs0 hs1]

/-- Riemann's entire xi function, without the inessential factor `1 / 2`.
This pole-cancelled definition uses Mathlib's entire `Λ₀`. -/
def riemannXi (s : ℂ) : ℂ :=
  s * (s - 1) * completedRiemannZeta₀ s + 1

theorem differentiable_riemannXi : Differentiable ℂ riemannXi := by
  unfold riemannXi
  exact ((differentiable_id.mul
    (differentiable_id.sub (differentiable_const _))).mul
      differentiable_completedZeta₀).add (differentiable_const _)

theorem analyticOnNhd_riemannXi :
    AnalyticOnNhd ℂ riemannXi Set.univ :=
  differentiable_riemannXi.differentiableOn.analyticOnNhd isOpen_univ

/-- Xi inherits the completed-zeta functional equation. -/
theorem riemannXi_one_sub (s : ℂ) :
    riemannXi (1 - s) = riemannXi s := by
  rw [riemannXi, riemannXi, completedRiemannZeta₀_one_sub]
  ring

/-- Away from the two removed poles, xi is the usual polynomial multiple of
the completed zeta function. -/
theorem riemannXi_eq_mul_completedRiemannZeta
    {s : ℂ} (hs0 : s ≠ 0) (hs1 : s ≠ 1) :
    riemannXi s = s * (s - 1) * completedRiemannZeta s := by
  rw [riemannXi, completedRiemannZeta_eq]
  field_simp
  ring

/-- In the open critical strip, xi and zeta have exactly the same zeros. -/
theorem riemannXi_eq_zero_iff_riemannZeta_eq_zero_of_strip
    {s : ℂ} (hs0 : 0 < s.re) (hs1 : s.re < 1) :
    riemannXi s = 0 ↔ riemannZeta s = 0 := by
  have hsne0 : s ≠ 0 := by
    intro h
    subst s
    norm_num at hs0
  have hsne1 : s ≠ 1 := by
    intro h
    subst s
    norm_num at hs1
  rw [riemannXi_eq_mul_completedRiemannZeta hsne0 hsne1,
    mul_eq_zero, mul_eq_zero,
    riemannZeta_eq_zero_iff_completedRiemannZeta_eq_zero_of_strip hs0]
  simp [hsne0, sub_ne_zero.mpr hsne1]

/-- Analytic multiplicity for the entire xi function. -/
def riemannXiZeroMultiplicity (s : ℂ) : ℕ :=
  analyticOrderNatAt riemannXi s

/-- In the open strip, pole cancellation does not change zero
multiplicities. -/
theorem riemannXiZeroMultiplicity_eq_zeta
    {s : ℂ} (hs0 : 0 < s.re) (hs1 : s.re < 1) :
    riemannXiZeroMultiplicity s = zetaZeroMultiplicity s := by
  have hsne0 : s ≠ 0 := by
    intro h
    subst s
    norm_num at hs0
  have hsne1 : s ≠ 1 := by
    intro h
    subst s
    norm_num at hs1
  let p : ℂ → ℂ := fun z ↦ z * (z - 1)
  have hp : AnalyticAt ℂ p s := by
    dsimp [p]
    fun_prop
  have hpne : p s ≠ 0 := by
    exact mul_ne_zero hsne0 (sub_ne_zero.mpr hsne1)
  have hporder : analyticOrderAt p s = 0 :=
    hp.analyticOrderAt_eq_zero.mpr hpne
  have hcomp : AnalyticAt ℂ completedRiemannZeta s :=
    analyticAt_completedRiemannZeta hsne0 hsne1
  have hevent :
      riemannXi =ᶠ[𝓝 s] p * completedRiemannZeta := by
    filter_upwards [eventually_ne_nhds hsne0,
      eventually_ne_nhds hsne1] with z hz0 hz1
    exact riemannXi_eq_mul_completedRiemannZeta hz0 hz1
  have horder :
      analyticOrderAt riemannXi s =
        analyticOrderAt completedRiemannZeta s := by
    rw [analyticOrderAt_congr hevent,
      analyticOrderAt_mul hp hcomp, hporder, zero_add]
  unfold riemannXiZeroMultiplicity analyticOrderNatAt
  rw [horder]
  exact (zetaZeroMultiplicity_eq_completed hs0 hs1).symm

/-- Local logarithmic-derivative form of the argument principle: an analytic
zero of finite order contributes exactly `n / (z - ρ)`, while the remaining
factor is analytic and nonvanishing.  This is the residue computation needed
at each xi zero; summing it over a rectangle still requires a global residue
theorem. -/
theorem analytic_logDeriv_local_normalForm
    {f : ℂ → ℂ} {ρ : ℂ} (hf : AnalyticAt ℂ f ρ)
    (hfinite : analyticOrderAt f ρ ≠ ⊤) :
    ∃ g : ℂ → ℂ, AnalyticAt ℂ g ρ ∧ g ρ ≠ 0 ∧
      ∀ᶠ z in 𝓝[≠] ρ,
        deriv f z / f z =
          (analyticOrderNatAt f ρ : ℂ) / (z - ρ) + deriv g z / g z := by
  obtain ⟨g, hg, hg0, hfg⟩ :=
    hf.analyticOrderAt_ne_top.mp hfinite
  simp only [smul_eq_mul] at hfg
  refine ⟨g, hg, hg0, ?_⟩
  have hg_ne : ∀ᶠ z in 𝓝 ρ, g z ≠ 0 :=
    hg.continuousAt.preimage_mem_nhds
      (isOpen_compl_singleton.mem_nhds hg0)
  have hderiv := hfg.deriv
  filter_upwards [hfg.filter_mono nhdsWithin_le_nhds,
    hderiv.filter_mono nhdsWithin_le_nhds,
    hg.eventually_analyticAt.filter_mono nhdsWithin_le_nhds,
    hg_ne.filter_mono nhdsWithin_le_nhds,
    self_mem_nhdsWithin] with z hfz hdz hgz hgz0 hz
  have hz0 : z - ρ ≠ 0 := sub_ne_zero.mpr hz
  generalize hn : analyticOrderNatAt f ρ = n
  rw [hn] at hfz hdz
  rw [hfz, hdz]
  rw [deriv_fun_mul (by fun_prop) hgz.differentiableAt,
    deriv_fun_pow (by fun_prop), deriv_sub_const]
  simp only [deriv_id'', one_mul]
  cases n with
  | zero => simp
  | succ n =>
      simp only [Nat.cast_succ, Nat.succ_sub_one]
      field_simp
      rw [pow_succ]
      ring

/-- Xi is not the zero entire function. -/
theorem riemannXi_two_ne_zero : riemannXi 2 ≠ 0 := by
  have hzeta : riemannZeta 2 ≠ 0 :=
    riemannZeta_ne_zero_of_one_le_re (by norm_num)
  have hcomp : completedRiemannZeta 2 ≠ 0 := by
    intro hc
    apply hzeta
    rw [riemannZeta_def_of_ne_zero (by norm_num)]
    simp [hc]
  rw [riemannXi_eq_mul_completedRiemannZeta
    (by norm_num) (by norm_num)]
  exact mul_ne_zero (mul_ne_zero (by norm_num) (by norm_num)) hcomp

/-- Every xi zero has finite analytic order, by analytic uniqueness and
nonvanishing at `s = 2`. -/
theorem riemannXi_analyticOrderAt_ne_top (ρ : ℂ) :
    analyticOrderAt riemannXi ρ ≠ ⊤ := by
  intro htop
  have hevent : riemannXi =ᶠ[𝓝 ρ] (fun _z : ℂ ↦ (0 : ℂ)) :=
    analyticOrderAt_eq_top.mp htop
  have hzero : AnalyticOnNhd ℂ (fun _z : ℂ ↦ (0 : ℂ)) Set.univ :=
    analyticOnNhd_const
  have heq : Set.EqOn riemannXi (fun _z : ℂ ↦ (0 : ℂ)) Set.univ :=
    analyticOnNhd_riemannXi.eqOn_of_preconnected_of_eventuallyEq
      hzero isPreconnected_univ (Set.mem_univ ρ) hevent
  exact riemannXi_two_ne_zero (by simpa using heq (Set.mem_univ (2 : ℂ)))

/-- Xi's local log-derivative has residue equal to its analytic
multiplicity at every finite-order point. -/
theorem riemannXi_logDeriv_local_normalForm
    {ρ : ℂ} (hfinite : analyticOrderAt riemannXi ρ ≠ ⊤) :
    ∃ g : ℂ → ℂ, AnalyticAt ℂ g ρ ∧ g ρ ≠ 0 ∧
      ∀ᶠ z in 𝓝[≠] ρ,
        deriv riemannXi z / riemannXi z =
          (riemannXiZeroMultiplicity ρ : ℂ) / (z - ρ) +
            deriv g z / g z := by
  exact analytic_logDeriv_local_normalForm
    (differentiable_riemannXi.analyticAt ρ) hfinite

/-- Unconditional local residue normal form for xi. -/
theorem riemannXi_logDeriv_local
    (ρ : ℂ) :
    ∃ g : ℂ → ℂ, AnalyticAt ℂ g ρ ∧ g ρ ≠ 0 ∧
      ∀ᶠ z in 𝓝[≠] ρ,
        deriv riemannXi z / riemannXi z =
          (riemannXiZeroMultiplicity ρ : ℂ) / (z - ρ) +
            deriv g z / g z :=
  riemannXi_logDeriv_local_normalForm
    (riemannXi_analyticOrderAt_ne_top ρ)

/-- Uniform data on a closed puncture disk around one xi zero. -/
structure XiPunctureData (ρ : ℂ) where
  radius : ℝ
  radius_pos : 0 < radius
  factor : ℂ → ℂ
  factor_analytic :
    ∀ z ∈ Metric.closedBall ρ radius, AnalyticAt ℂ factor z
  factor_ne_zero :
    ∀ z ∈ Metric.closedBall ρ radius, factor z ≠ 0
  xi_ne_zero :
    ∀ z ∈ Metric.closedBall ρ radius, z ≠ ρ → riemannXi z ≠ 0
  logDeriv_eq :
    ∀ z ∈ Metric.closedBall ρ radius, z ≠ ρ →
      deriv riemannXi z / riemannXi z =
        (riemannXiZeroMultiplicity ρ : ℂ) / (z - ρ) +
          deriv factor z / factor z

/-- Every xi point admits a positive-radius puncture disk on which its local
residue decomposition is uniform, including the boundary circle. -/
theorem exists_riemannXiPunctureData (ρ : ℂ) :
    Nonempty (XiPunctureData ρ) := by
  obtain ⟨g, hg, hg0, heq⟩ := riemannXi_logDeriv_local ρ
  have hg_ne : ∀ᶠ z in 𝓝 ρ, g z ≠ 0 :=
    hg.continuousAt.preimage_mem_nhds
      (isOpen_compl_singleton.mem_nhds hg0)
  have heq' : ∀ᶠ z in 𝓝 ρ, z ≠ ρ →
      deriv riemannXi z / riemannXi z =
        (riemannXiZeroMultiplicity ρ : ℂ) / (z - ρ) +
          deriv g z / g z :=
    eventually_nhdsWithin_iff.mp heq
  have hxi_ne : ∀ᶠ z in 𝓝[≠] ρ, riemannXi z ≠ 0 :=
    (differentiable_riemannXi.analyticAt ρ).eventually_eq_zero_or_eventually_ne_zero.resolve_left
      (fun hzero ↦ riemannXi_analyticOrderAt_ne_top ρ
        (analyticOrderAt_eq_top.mpr hzero))
  have hxi_ne' : ∀ᶠ z in 𝓝 ρ, z ≠ ρ → riemannXi z ≠ 0 :=
    eventually_nhdsWithin_iff.mp hxi_ne
  obtain ⟨r, hr, hall⟩ :=
    Metric.nhds_basis_closedBall.mem_iff.mp
      (hg.eventually_analyticAt.and (hg_ne.and (hxi_ne'.and heq')))
  exact ⟨{
    radius := r
    radius_pos := hr
    factor := g
    factor_analytic := fun _z hz ↦ (hall hz).1
    factor_ne_zero := fun _z hz ↦ (hall hz).2.1
    xi_ne_zero := fun _z hz hzρ ↦ (hall hz).2.2.1 hzρ
    logDeriv_eq := fun _z hz hzρ ↦ (hall hz).2.2.2 hzρ
  }⟩

/-- A fixed noncomputable choice of valid puncture data at every xi point. -/
noncomputable def riemannXiPunctureData (ρ : ℂ) : XiPunctureData ρ :=
  Classical.choice (exists_riemannXiPunctureData ρ)

/-- Every smaller positive boundary circle lies inside the uniform puncture
disk, contains no xi zeros, and carries the exact residue decomposition. -/
theorem XiPunctureData.on_sphere
    {ρ : ℂ} (D : XiPunctureData ρ) {r : ℝ}
    (hr : 0 < r) (hrD : r ≤ D.radius) {z : ℂ}
    (hz : z ∈ Metric.sphere ρ r) :
    riemannXi z ≠ 0 ∧
      deriv riemannXi z / riemannXi z =
        (riemannXiZeroMultiplicity ρ : ℂ) / (z - ρ) +
          deriv D.factor z / D.factor z := by
  have hnorm : ‖z - ρ‖ = r := by
    simpa [Metric.mem_sphere] using hz
  have hzball : z ∈ Metric.closedBall ρ D.radius := by
    rw [Metric.mem_closedBall, dist_eq_norm']
    simpa only [norm_sub_rev] using hnorm.le.trans hrD
  have hzρ : z ≠ ρ := by
    intro h
    subst z
    simp at hnorm
    linarith
  exact ⟨D.xi_ne_zero z hzball hzρ, D.logDeriv_eq z hzball hzρ⟩

/-- The unweighted logarithmic-derivative integral around every sufficiently
small xi puncture is exactly `2πi` times the analytic multiplicity. -/
theorem XiPunctureData.circleIntegral_logDeriv_eq
    {ρ : ℂ} (D : XiPunctureData ρ) {r : ℝ}
    (hr : 0 < r) (hrD : r ≤ D.radius) :
    (∮ z in C(ρ, r), deriv riemannXi z / riemannXi z) =
      2 * Real.pi * I * (riemannXiZeroMultiplicity ρ : ℂ) := by
  let p : ℂ → ℂ :=
    fun z ↦ (riemannXiZeroMultiplicity ρ : ℂ) / (z - ρ)
  let q : ℂ → ℂ := fun z ↦ deriv D.factor z / D.factor z
  have hpcont : ContinuousOn p (Metric.sphere ρ r) := by
    apply continuousOn_const.div
      (continuousOn_id.sub continuousOn_const)
    intro z hz
    have hzρ : z ≠ ρ := by
      intro h
      subst z
      simp [Metric.mem_sphere] at hz
      linarith
    exact sub_ne_zero.mpr hzρ
  have hqan : AnalyticOnNhd ℂ q (Metric.closedBall ρ r) := by
    intro z hz
    have hzD : z ∈ Metric.closedBall ρ D.radius :=
      Metric.closedBall_subset_closedBall hrD hz
    exact (D.factor_analytic z hzD).deriv.div
      (D.factor_analytic z hzD) (D.factor_ne_zero z hzD)
  have hqdiff : DiffContOnCl ℂ q (Metric.ball ρ r) := by
    exact ⟨hqan.differentiableOn.mono Metric.ball_subset_closedBall,
      by simpa only [Metric.closedBall, closure_ball ρ hr.ne'] using hqan.continuousOn⟩
  have hpint : CircleIntegrable p ρ r :=
    hpcont.circleIntegrable hr.le
  have hqint : CircleIntegrable q ρ r :=
    (hqan.mono Metric.sphere_subset_closedBall).continuousOn.circleIntegrable hr.le
  calc
    (∮ z in C(ρ, r), deriv riemannXi z / riemannXi z) =
        ∮ z in C(ρ, r), p z + q z := by
      apply circleIntegral.integral_congr hr.le
      intro z hz
      exact (D.on_sphere hr hrD hz).2
    _ = (∮ z in C(ρ, r), p z) + ∮ z in C(ρ, r), q z :=
      circleIntegral.integral_add hpint hqint
    _ = (∮ z in C(ρ, r), (z - ρ)⁻¹) *
          (riemannXiZeroMultiplicity ρ : ℂ) + 0 := by
      rw [show (∮ z in C(ρ, r), q z) = 0 from
        hqdiff.circleIntegral_eq_zero hr.le]
      congr 1
      simpa [p, div_eq_inv_mul] using
        (circleIntegral.integral_smul_const
          (fun z : ℂ ↦ (z - ρ)⁻¹)
          (riemannXiZeroMultiplicity ρ : ℂ) ρ r)
    _ = _ := by
      rw [circleIntegral.integral_sub_inv_of_mem_ball
        (show ρ ∈ Metric.ball ρ r by simpa using hr)]
      ring

/-- Summing the exact small-circle identities gives the full finite sum of
local xi residues, with multiplicity, for independently chosen puncture
radii. -/
theorem riemannXi_finset_sum_puncture_integrals
    (s : Finset ℂ) (r : ℂ → ℝ)
    (hr : ∀ ρ ∈ s, 0 < r ρ)
    (hrD : ∀ ρ ∈ s, r ρ ≤ (riemannXiPunctureData ρ).radius) :
    ∑ ρ ∈ s, (∮ z in C(ρ, r ρ), deriv riemannXi z / riemannXi z) =
      2 * Real.pi * I *
        ∑ ρ ∈ s, (riemannXiZeroMultiplicity ρ : ℂ) := by
  calc
    ∑ ρ ∈ s, (∮ z in C(ρ, r ρ), deriv riemannXi z / riemannXi z) =
        ∑ ρ ∈ s, (2 * Real.pi * I *
          (riemannXiZeroMultiplicity ρ : ℂ)) := by
      apply Finset.sum_congr rfl
      intro ρ hρ
      exact (riemannXiPunctureData ρ).circleIntegral_logDeriv_eq
        (hr ρ hρ) (hrD ρ hρ)
    _ = _ := by
      rw [Finset.mul_sum]

/-- Weighted local residue identity.  This is the puncture contribution
needed after inserting an analytic admissible transform. -/
theorem XiPunctureData.circleIntegral_mul_logDeriv_eq
    {ρ : ℂ} (D : XiPunctureData ρ) {r : ℝ}
    (hr : 0 < r) (hrD : r ≤ D.radius)
    {h : ℂ → ℂ} (hh : AnalyticOnNhd ℂ h (Metric.closedBall ρ r)) :
    (∮ z in C(ρ, r), h z * (deriv riemannXi z / riemannXi z)) =
      2 * Real.pi * I * (riemannXiZeroMultiplicity ρ : ℂ) * h ρ := by
  let p : ℂ → ℂ :=
    fun z ↦ (riemannXiZeroMultiplicity ρ : ℂ) / (z - ρ)
  let q : ℂ → ℂ := fun z ↦ deriv D.factor z / D.factor z
  have hqan : AnalyticOnNhd ℂ q (Metric.closedBall ρ r) := by
    intro z hz
    have hzD : z ∈ Metric.closedBall ρ D.radius :=
      Metric.closedBall_subset_closedBall hrD hz
    exact (D.factor_analytic z hzD).deriv.div
      (D.factor_analytic z hzD) (D.factor_ne_zero z hzD)
  have hrem : DiffContOnCl ℂ (fun z ↦ h z * q z) (Metric.ball ρ r) := by
    have han := hh.mul hqan
    exact ⟨han.differentiableOn.mono Metric.ball_subset_closedBall,
      by simpa only [Metric.closedBall, closure_ball ρ hr.ne'] using han.continuousOn⟩
  have hres : DiffContOnCl ℂ
      (fun z ↦ (riemannXiZeroMultiplicity ρ : ℂ) * h z)
      (Metric.ball ρ r) := by
    have han : AnalyticOnNhd ℂ
        (fun z ↦ (riemannXiZeroMultiplicity ρ : ℂ) * h z)
        (Metric.closedBall ρ r) := by
      intro z hz
      exact analyticAt_const.mul (hh z hz)
    exact ⟨han.differentiableOn.mono Metric.ball_subset_closedBall,
      by simpa only [Metric.closedBall, closure_ball ρ hr.ne'] using han.continuousOn⟩
  have hpcont : ContinuousOn (fun z ↦ h z * p z) (Metric.sphere ρ r) := by
    apply (hh.mono Metric.sphere_subset_closedBall).continuousOn.mul
    apply continuousOn_const.div
      (continuousOn_id.sub continuousOn_const)
    intro z hz
    have hzρ : z ≠ ρ := by
      intro hzr
      subst z
      simp at hz
      linarith
    exact sub_ne_zero.mpr hzρ
  have hpint : CircleIntegrable (fun z ↦ h z * p z) ρ r :=
    hpcont.circleIntegrable hr.le
  have hqint : CircleIntegrable (fun z ↦ h z * q z) ρ r :=
    (hh.mul hqan).continuousOn.mono
      Metric.sphere_subset_closedBall |>.circleIntegrable hr.le
  have hpformula :
      (∮ z in C(ρ, r), h z * p z) =
        (2 * Real.pi * I) *
          ((riemannXiZeroMultiplicity ρ : ℂ) * h ρ) := by
    calc
      (∮ z in C(ρ, r), h z * p z) =
          ∮ z in C(ρ, r), (z - ρ)⁻¹ *
            ((riemannXiZeroMultiplicity ρ : ℂ) * h z) := by
        apply circleIntegral.integral_congr hr.le
        intro z _hz
        simp [p, div_eq_inv_mul]
        ring
      _ = _ := hres.circleIntegral_sub_inv_smul
        (show ρ ∈ Metric.ball ρ r by simpa using hr)
  calc
    (∮ z in C(ρ, r), h z * (deriv riemannXi z / riemannXi z)) =
        ∮ z in C(ρ, r), h z * p z + h z * q z := by
      apply circleIntegral.integral_congr hr.le
      intro z hz
      change h z * (deriv riemannXi z / riemannXi z) =
        h z * p z + h z * q z
      rw [(D.on_sphere hr hrD hz).2]
      simp only [p, q]
      ring
    _ = (∮ z in C(ρ, r), h z * p z) +
          ∮ z in C(ρ, r), h z * q z :=
      circleIntegral.integral_add hpint hqint
    _ = (2 * Real.pi * I) *
          ((riemannXiZeroMultiplicity ρ : ℂ) * h ρ) + 0 := by
      rw [show (∮ z in C(ρ, r), h z * q z) = 0 from
        hrem.circleIntegral_eq_zero hr.le]
      rw [hpformula]
    _ = _ := by ring

/-- Finite weighted residue sum after inserting one analytic transform. -/
theorem riemannXi_finset_sum_puncture_integrals_mul
    (s : Finset ℂ) (r : ℂ → ℝ) (h : ℂ → ℂ)
    (hr : ∀ ρ ∈ s, 0 < r ρ)
    (hrD : ∀ ρ ∈ s, r ρ ≤ (riemannXiPunctureData ρ).radius)
    (hh : ∀ ρ ∈ s,
      AnalyticOnNhd ℂ h (Metric.closedBall ρ (r ρ))) :
    ∑ ρ ∈ s, (∮ z in C(ρ, r ρ),
        h z * (deriv riemannXi z / riemannXi z)) =
      2 * Real.pi * I *
        ∑ ρ ∈ s, (riemannXiZeroMultiplicity ρ : ℂ) * h ρ := by
  calc
    ∑ ρ ∈ s, (∮ z in C(ρ, r ρ),
        h z * (deriv riemannXi z / riemannXi z)) =
        ∑ ρ ∈ s, (2 * Real.pi * I *
          (riemannXiZeroMultiplicity ρ : ℂ) * h ρ) := by
      apply Finset.sum_congr rfl
      intro ρ hρ
      exact (riemannXiPunctureData ρ).circleIntegral_mul_logDeriv_eq
        (hr ρ hρ) (hrD ρ hρ) (hh ρ hρ)
    _ = _ := by
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro ρ _hρ
      ring

/-- Exact Jensen zero-count identity for xi on a circle.  This replaces the
local residue sum by Mathlib's divisor formalism; obtaining RvM still requires
a quantitative bound for the boundary average. -/
theorem riemannXi_circleAverage_log_norm_eq_divisor_sum
    (c : ℂ) {R : ℝ} (hR : R ≠ 0) :
    Real.circleAverage (fun z ↦ Real.log ‖riemannXi z‖) c R =
      ∑ᶠ u, MeromorphicOn.divisor riemannXi (Metric.closedBall c |R|) u *
        Real.log (R * ‖c - u‖⁻¹) +
      MeromorphicOn.divisor riemannXi (Metric.closedBall c |R|) c *
        Real.log R +
      Real.log ‖meromorphicTrailingCoeffAt riemannXi c‖ := by
  apply MeromorphicOn.circleAverage_log_norm hR
  exact (analyticOnNhd_riemannXi.mono (Set.subset_univ _)).meromorphicOn

/-- Quantitative Jensen bound for xi.  Any explicit bound `M` on the outer
circle immediately bounds the multiplicity-weighted number of zeros in the
inner circle. -/
theorem riemannXi_sum_divisor_le
    {c : ℂ} {r R M : ℝ} (hr : 0 < |r|) (hrR : |r| < |R|)
    (hM : 1 ≤ M) (hc : riemannXi c ≠ 0)
    (hbound : ∀ z ∈ Metric.sphere c |R|, ‖riemannXi z‖ ≤ M) :
    ∑ᶠ u, MeromorphicOn.divisor riemannXi
        (Metric.closedBall c |r|) u ≤
      Real.log (M / ‖riemannXi c‖) / Real.log (R / r) := by
  exact AnalyticOnNhd.sum_divisor_le hr hrR hM
    (analyticOnNhd_riemannXi.mono (Set.subset_univ _)) hc hbound

/-- Xi's divisor coefficient is exactly its natural analytic multiplicity;
there is no pole or infinite-order exceptional case. -/
theorem riemannXi_divisor_apply
    {U : Set ℂ} {z : ℂ} (hz : z ∈ U) :
    MeromorphicOn.divisor riemannXi U z =
      (riemannXiZeroMultiplicity z : ℤ) := by
  rw [MeromorphicOn.AnalyticOnNhd.divisor_apply
    (analyticOnNhd_riemannXi.mono (Set.subset_univ _)) hz]
  have hfinite := riemannXi_analyticOrderAt_ne_top z
  generalize ho : analyticOrderAt riemannXi z = o at hfinite ⊢
  cases o with
  | top => exact (hfinite rfl).elim
  | coe n => simp [riemannXiZeroMultiplicity, analyticOrderNatAt, ho]

/-- On any finite zero window, the xi divisor sum is literally the sum of
analytic multiplicities. -/
theorem riemannXi_finset_divisor_sum_eq_multiplicity
    {U : Set ℂ} (s : Finset ℂ) (hs : ∀ z ∈ s, z ∈ U) :
    ∑ z ∈ s, MeromorphicOn.divisor riemannXi U z =
      ∑ z ∈ s, (riemannXiZeroMultiplicity z : ℤ) := by
  apply Finset.sum_congr rfl
  intro z hz
  exact riemannXi_divisor_apply (hs z hz)

/-- Actual analytic multiplicity expressed in centered coordinates. -/
def actualCenteredZetaMultiplicity (z : ℂ) : ℕ :=
  zetaZeroMultiplicity ((1 / 2 : ℂ) + z)

/-- Centering preserves multiplicity definitionally once the spectral point
is transported by `actualZetaZeroCenteringEquiv`. -/
theorem actualCenteredZetaMultiplicity_centering
    (ρ : ActualNontrivialZetaZero) :
    actualCenteredZetaMultiplicity (actualZetaZeroCenteringEquiv ρ) =
      zetaZeroMultiplicity ρ.val := by
  simp [actualCenteredZetaMultiplicity, actualZetaZeroCenteringEquiv]

/-- The actual centered nontrivial zero set is closed under the full
functional-equation/conjugation partner `z ↦ -conj z`. -/
theorem actualCenteredNontrivialZetaZeros_neg_star
    {z : ℂ} (hz : z ∈ actualCenteredNontrivialZetaZeros) :
    -star z ∈ actualCenteredNontrivialZetaZeros := by
  rcases hz with ⟨hzero, hz0, hz1⟩
  have hρne : (1 / 2 : ℂ) + z ≠ 1 := by
    intro h
    have hre := congrArg Complex.re h
    norm_num at hre
    linarith
  have hconj0 :
      riemannZeta ((1 / 2 : ℂ) + star z) = 0 := by
    have hc :=
      (riemannZeta_star_eq_zero_iff hρne).mpr hzero
    convert hc using 1 <;> simp
  refine ⟨centeredRiemannZeta_neg_eq_zero hconj0 ?_ ?_, ?_, ?_⟩
  · simpa using hz0
  · simpa using hz1
  · simp
    linarith
  · simp
    linarith

/-- The full centered partner preserves actual analytic multiplicity. -/
theorem actualCenteredZetaMultiplicity_neg_star
    {z : ℂ} (hz : z ∈ actualCenteredNontrivialZetaZeros) :
    actualCenteredZetaMultiplicity (-star z) =
      actualCenteredZetaMultiplicity z := by
  rcases hz with ⟨_hzero, hz0, hz1⟩
  let ρ : ℂ := (1 / 2 : ℂ) + z
  have hρ0 : 0 < ρ.re := by
    dsimp [ρ]
    norm_num
    linarith
  have hρ1 : ρ.re < 1 := by
    dsimp [ρ]
    norm_num
    linarith
  have hρne : ρ ≠ 1 := by
    intro h
    have hre := congrArg Complex.re h
    dsimp [ρ] at hre
    norm_num at hre
    linarith
  have hstar0 : 0 < (star ρ).re := by simpa using hρ0
  have hstar1 : (star ρ).re < 1 := by simpa using hρ1
  calc
    actualCenteredZetaMultiplicity (-star z) =
        zetaZeroMultiplicity (1 - star ρ) := by
          unfold actualCenteredZetaMultiplicity
          congr 1
          dsimp [ρ]
          rw [map_add]
          rw [map_div₀, map_one, map_ofNat]
          ring
    _ = zetaZeroMultiplicity (star ρ) :=
      zetaZeroMultiplicity_one_sub hstar0 hstar1
    _ = zetaZeroMultiplicity ρ := zetaZeroMultiplicity_star hρne
    _ = actualCenteredZetaMultiplicity z := rfl

/-- Positive-height centered zeros.  Quotienting this set by
`z ↦ -conj z` gives one index for a critical-line pair or an off-line
quartet. -/
def actualPositiveCenteredZetaZeros : Set ℂ :=
  {z | z ∈ actualCenteredNontrivialZetaZeros ∧ 0 < z.im}

def ActualPositiveCenteredZetaZero := ↥actualPositiveCenteredZetaZeros

theorem actualPositiveCenteredZetaZeros_countable :
    actualPositiveCenteredZetaZeros.Countable :=
  nontrivialRiemannZetaZeros_countable.image
      (fun ρ ↦ ρ - (1 / 2 : ℂ)) |>.mono <| by
    intro z hz
    exact ⟨z + (1 / 2 : ℂ), by
      refine ⟨?_, ?_⟩
      · rcases hz.1 with ⟨hzero, hz0, hz1⟩
        exact ⟨by simpa [add_comm] using hzero,
          by norm_num; linarith, by norm_num; linarith⟩
      · ring⟩

noncomputable instance : Countable ActualPositiveCenteredZetaZero :=
  actualPositiveCenteredZetaZeros_countable

/-- The actual positive-height partner involution. -/
def actualPositiveCenteredPartner :
    ActualPositiveCenteredZetaZero ≃ ActualPositiveCenteredZetaZero where
  toFun z := ⟨-star z.val, by
    exact ⟨actualCenteredNontrivialZetaZeros_neg_star z.property.1,
      by simpa using z.property.2⟩⟩
  invFun z := ⟨-star z.val, by
    exact ⟨actualCenteredNontrivialZetaZeros_neg_star z.property.1,
      by simpa using z.property.2⟩⟩
  left_inv z := by
    apply Subtype.ext
    simp
  right_inv z := by
    apply Subtype.ext
    simp

@[simp]
theorem actualPositiveCenteredPartner_apply
    (z : ActualPositiveCenteredZetaZero) :
    (actualPositiveCenteredPartner z).val = -star z.val := rfl

/-- Critical-line positive zeros are exactly the fixed points of the partner
involution. -/
theorem actualPositiveCenteredPartner_fixed_iff
    (z : ActualPositiveCenteredZetaZero) :
    actualPositiveCenteredPartner z = z ↔ z.val.re = 0 := by
  constructor
  · intro h
    have hre := congrArg (fun w : ActualPositiveCenteredZetaZero ↦ w.val.re) h
    simp at hre
    linarith
  · intro hre
    apply Subtype.ext
    apply Complex.ext
    · simp [hre]
    · simp

/-- Off-critical positive zeros form two-element partner pairs, corresponding
to four-element zero quartets after adjoining complex conjugates. -/
theorem actualPositiveCenteredPartner_ne_iff
    (z : ActualPositiveCenteredZetaZero) :
    actualPositiveCenteredPartner z ≠ z ↔ z.val.re ≠ 0 :=
  (actualPositiveCenteredPartner_fixed_iff z).not

/-- Two positive-height zeros represent the same quartet exactly when they
are equal or are functional-equation partners. -/
instance actualPositiveCenteredOrbitSetoid :
    Setoid ActualPositiveCenteredZetaZero where
  r a b := b = a ∨ b = actualPositiveCenteredPartner a
  iseqv := ⟨
    fun a ↦ Or.inl rfl,
    fun {a b} h ↦ by
      rcases h with rfl | h
      · exact Or.inl rfl
      · right
        rw [h]
        apply Subtype.ext
        simp,
    fun {a b c} hab hbc ↦ by
      rcases hab with rfl | hab
      · exact hbc
      · rcases hbc with hbc | hbc
        · exact Or.inr (hbc.trans hab)
        · left
          rw [hbc, hab]
          apply Subtype.ext
          simp⟩

/-- The exact countable type of positive-height zeta quartet orbits. -/
def ActualZetaZeroOrbit :=
  Quotient actualPositiveCenteredOrbitSetoid

noncomputable instance : Countable ActualZetaZeroOrbit :=
  (Quotient.mk'_surjective :
    Function.Surjective
      (Quotient.mk' : ActualPositiveCenteredZetaZero → ActualZetaZeroOrbit)).countable

theorem actualZetaZeroOrbit_mk_eq_iff
    (a b : ActualPositiveCenteredZetaZero) :
    (Quotient.mk' a : ActualZetaZeroOrbit) = Quotient.mk' b ↔
      b = a ∨ b = actualPositiveCenteredPartner a := by
  rw [Quotient.eq']
  rfl

/-- Canonical partial natural-number enumeration obtained from countability.
`none` marks unused codes, avoiding duplicate dummy orbits when the orbit
type is finite. -/
noncomputable def actualZetaZeroOrbitDecode :
    ℕ → Option ActualZetaZeroOrbit :=
  letI := Encodable.ofCountable ActualZetaZeroOrbit
  fun n ↦ Encodable.decode₂ ActualZetaZeroOrbit n

noncomputable def actualZetaZeroOrbitCode
    (q : ActualZetaZeroOrbit) : ℕ :=
  letI := Encodable.ofCountable ActualZetaZeroOrbit
  Encodable.encode q

/-- Every exact quartet orbit occurs in the partial natural enumeration. -/
theorem actualZetaZeroOrbitDecode_complete (q : ActualZetaZeroOrbit) :
    actualZetaZeroOrbitDecode (actualZetaZeroOrbitCode q) = some q := by
  simp [actualZetaZeroOrbitDecode, actualZetaZeroOrbitCode,
    Encodable.encodek₂]

/-- The partial enumeration never assigns two natural codes to the same
orbit. -/
theorem actualZetaZeroOrbitDecode_injective
    {m n : ℕ} {q : ActualZetaZeroOrbit}
    (hm : actualZetaZeroOrbitDecode m = some q)
    (hn : actualZetaZeroOrbitDecode n = some q) :
    m = n := by
  letI := Encodable.ofCountable ActualZetaZeroOrbit
  rw [show actualZetaZeroOrbitDecode m =
      Encodable.decode₂ ActualZetaZeroOrbit m by
        rfl, Encodable.decode₂_eq_some] at hm
  rw [show actualZetaZeroOrbitDecode n =
      Encodable.decode₂ ActualZetaZeroOrbit n by
        rfl, Encodable.decode₂_eq_some] at hn
  exact hm.symm.trans hn

/-- Multiplicity descends to quartet orbits because partner multiplicities
are equal. -/
def actualZetaZeroOrbitMultiplicity : ActualZetaZeroOrbit → ℕ :=
  Quotient.lift
    (fun z : ActualPositiveCenteredZetaZero ↦
      actualCenteredZetaMultiplicity z.val)
    (by
      intro a b hab
      rcases hab with rfl | h
      · rfl
      · rw [h]
        simpa using
          (actualCenteredZetaMultiplicity_neg_star a.property.1).symm)

/-- A bounded positive-height portion of the exact zero type. -/
def actualPositiveCenteredZeroWindow (T : ℝ) :
    Set ActualPositiveCenteredZetaZero :=
  {z | z.val.im ≤ T}

theorem actualPositiveCenteredZeroWindow_finite (T : ℝ) :
    (actualPositiveCenteredZeroWindow T).Finite := by
  let u : ActualPositiveCenteredZetaZero → ℂ :=
    fun z ↦ (1 / 2 : ℂ) + z.val
  apply Set.Finite.of_finite_image (f := u)
  · apply (positiveNontrivialZetaZeroWindow_finite T).subset
    rintro ρ ⟨z, hz, rfl⟩
    rcases z.property with ⟨⟨hzero, hz0, hz1⟩, hzim⟩
    have hzT : z.val.im ≤ T := hz
    exact ⟨hzero, by simpa [u] using hzim, by simpa [u] using hzT,
      by norm_num [u]; linarith, by norm_num [u]; linarith⟩
  · exact (fun _ _ _ _ h ↦ Subtype.ext <| by
      have := congrArg (fun w : ℂ ↦ w - 1 / 2) h
      simpa [u] using this)

/-- The bounded orbit window is finite, giving local finiteness of the exact
countable quartet enumeration. -/
def actualZetaZeroOrbitWindow (T : ℝ) : Set ActualZetaZeroOrbit :=
  Quotient.mk' '' actualPositiveCenteredZeroWindow T

theorem actualZetaZeroOrbitWindow_finite (T : ℝ) :
    (actualZetaZeroOrbitWindow T).Finite :=
  (actualPositiveCenteredZeroWindow_finite T).image Quotient.mk'

/-- Every actual positive-height zero belongs to its exact quartet class. -/
theorem actualZetaZeroOrbit_complete
    (z : ActualPositiveCenteredZetaZero) :
    Quotient.mk' z ∈ actualZetaZeroOrbitWindow z.val.im :=
  ⟨z, by simp [actualPositiveCenteredZeroWindow], rfl⟩

/-- Actual multiplicity-aware bounded zero count. -/
def positiveNontrivialZetaZeroCount (T : ℝ) : ℕ :=
  ∑ z ∈ positiveNontrivialZetaZeroWindowFinset T, zetaZeroMultiplicity z

/-- Monotonicity of the actual bounded multiplicity count. -/
theorem positiveNontrivialZetaZeroCount_mono {T U : ℝ} (hTU : T ≤ U) :
    positiveNontrivialZetaZeroCount T ≤ positiveNontrivialZetaZeroCount U := by
  apply Finset.sum_le_sum_of_subset_of_nonneg
  · intro z hz
    rw [mem_positiveNontrivialZetaZeroWindowFinset] at hz ⊢
    exact ⟨hz.1, hz.2.1, hz.2.2.1.trans hTU, hz.2.2.2⟩
  · intro z hzU hzT
    exact Nat.zero_le _

/-- The multiplicity mass in a unit height shell is bounded by the actual
cumulative multiplicity count at its upper endpoint. -/
theorem positiveNontrivialZetaZeroShellCount_le (k : ℕ) :
    (∑ z ∈ positiveNontrivialZetaZeroWindowFinset (k + 1),
        if (k : ℝ) < z.im then zetaZeroMultiplicity z else 0) ≤
      positiveNontrivialZetaZeroCount (k + 1) := by
  apply Finset.sum_le_sum
  intro z hz
  split_ifs
  · exact le_rfl
  · exact Nat.zero_le _

/-- Main term in the sourced Riemann--von Mangoldt estimate. -/
def riemannVonMangoldtMain (T : ℝ) : ℝ :=
  T / (2 * Real.pi) * Real.log (T / (2 * Real.pi * Real.exp 1))

/-- Explicit error envelope from Hasanalizade--Shen--Wong (2021).  Decimal
constants are represented as exact rationals. -/
def riemannVonMangoldtError (T : ℝ) : ℝ :=
  (1038 / 10000 : ℝ) * Real.log T +
    (2573 / 10000 : ℝ) * Real.log (Real.log T) +
    (93675 / 10000 : ℝ)

/-- Elementary quadratic envelope for the published main-plus-error terms.
The source gives the much sharper `O(T log T)` expression; this coarse bound
is convenient for summable transform tails. -/
theorem riemannVonMangoldtMain_add_error_le_quadratic
    {T : ℝ} (hT : 3 ≤ T) :
    riemannVonMangoldtMain T + riemannVonMangoldtError T ≤ 3 * T ^ 2 := by
  have hT0 : 0 ≤ T := by linarith
  have hT1 : 1 ≤ T := by linarith
  have hd : 1 ≤ 2 * Real.pi := by
    nlinarith [Real.pi_gt_three]
  have he : 1 ≤ Real.exp 1 := Real.one_le_exp (by norm_num)
  have hde : 1 ≤ 2 * Real.pi * Real.exp 1 :=
    one_le_mul_of_one_le_of_one_le hd he
  have hx0 : 0 ≤ T / (2 * Real.pi * Real.exp 1) := by positivity
  have hxT : T / (2 * Real.pi * Real.exp 1) ≤ T :=
    div_le_self hT0 hde
  have hlogx :
      Real.log (T / (2 * Real.pi * Real.exp 1)) ≤ T :=
    (Real.log_le_self hx0).trans hxT
  have hTd0 : 0 ≤ T / (2 * Real.pi) := by positivity
  have hTdT : T / (2 * Real.pi) ≤ T := div_le_self hT0 hd
  have hmain : riemannVonMangoldtMain T ≤ T ^ 2 := by
    rw [riemannVonMangoldtMain]
    calc
      T / (2 * Real.pi) *
          Real.log (T / (2 * Real.pi * Real.exp 1)) ≤
          (T / (2 * Real.pi)) * T :=
        mul_le_mul_of_nonneg_left hlogx hTd0
      _ ≤ T * T := mul_le_mul_of_nonneg_right hTdT hT0
      _ = T ^ 2 := by ring
  have hlogT0 : 0 ≤ Real.log T := Real.log_nonneg hT1
  have hlogT : Real.log T ≤ T := Real.log_le_self hT0
  have hloglog : Real.log (Real.log T) ≤ T :=
    (Real.log_le_self hlogT0).trans hlogT
  have herror : riemannVonMangoldtError T ≤ 2 * T ^ 2 := by
    rw [riemannVonMangoldtError]
    calc
      (1038 / 10000 : ℝ) * Real.log T +
            (2573 / 10000 : ℝ) * Real.log (Real.log T) +
            (93675 / 10000 : ℝ) ≤
          (1038 / 10000 : ℝ) * T +
            (2573 / 10000 : ℝ) * T +
            (93675 / 10000 : ℝ) := by
        gcongr
      _ ≤ 2 * T ^ 2 := by nlinarith
  linarith

/-- The sourced Riemann--von Mangoldt estimate specialized to Mathlib's
actual analytic-order multiplicity count.  This is the sole remaining
number-theoretic obligation for bounded counts. -/
structure ActualRiemannVonMangoldtBound where
  explicit_bound : ∀ N : ℕ, 3 ≤ N →
    |(positiveNontrivialZetaZeroCount N : ℝ) -
      riemannVonMangoldtMain N| ≤ riemannVonMangoldtError N

/-- The actual multiplicity mass in `(k+2, k+3]` satisfies the exact sourced
main-plus-error shell bound. -/
theorem ActualRiemannVonMangoldtBound.shell_count_le
    (H : ActualRiemannVonMangoldtBound) (k : ℕ) :
    (∑ z ∈ positiveNontrivialZetaZeroWindowFinset (k + 3),
        if (k + 2 : ℝ) < z.im then zetaZeroMultiplicity z else 0 : ℕ) ≤
      riemannVonMangoldtMain (k + 3) +
        riemannVonMangoldtError (k + 3) := by
  have hs :
      (∑ z ∈ positiveNontrivialZetaZeroWindowFinset (k + 3),
          if (k + 2 : ℝ) < z.im then zetaZeroMultiplicity z else 0 : ℕ) ≤
        positiveNontrivialZetaZeroCount (k + 3) := by
    apply Finset.sum_le_sum
    intro z hz
    split_ifs
    · exact le_rfl
    · exact Nat.zero_le _
  have hb := H.explicit_bound (k + 3) (by omega)
  have hb' :
      |(positiveNontrivialZetaZeroCount (k + 3) : ℝ) -
        riemannVonMangoldtMain ((k : ℝ) + 3)| ≤
        riemannVonMangoldtError ((k : ℝ) + 3) := by
    simpa [Nat.cast_add] using hb
  calc
    (∑ z ∈ positiveNontrivialZetaZeroWindowFinset (k + 3),
        if (k + 2 : ℝ) < z.im then zetaZeroMultiplicity z else 0 : ℕ) ≤
        (positiveNontrivialZetaZeroCount (k + 3) : ℝ) := by exact_mod_cast hs
    _ = ((positiveNontrivialZetaZeroCount (k + 3) : ℝ) -
        riemannVonMangoldtMain (k + 3)) +
        riemannVonMangoldtMain (k + 3) := by ring
    _ ≤ riemannVonMangoldtError (k + 3) +
        riemannVonMangoldtMain (k + 3) := by
      gcongr
      exact (le_abs_self _).trans hb'
    _ = riemannVonMangoldtMain (k + 3) +
        riemannVonMangoldtError (k + 3) := add_comm _ _

/-- Coarse actual shell-count consequence used by polynomial transform
decay. -/
theorem ActualRiemannVonMangoldtBound.shell_count_le_quadratic
    (H : ActualRiemannVonMangoldtBound) (k : ℕ) :
    (∑ z ∈ positiveNontrivialZetaZeroWindowFinset (k + 3),
        if (k + 2 : ℝ) < z.im then zetaZeroMultiplicity z else 0 : ℕ) ≤
      3 * ((k : ℝ) + 3) ^ 2 :=
  (H.shell_count_le k).trans <| by
    simpa [Nat.cast_add] using
      (riemannVonMangoldtMain_add_error_le_quadratic
        (T := (k : ℝ) + 3) (by
          have hk : (0 : ℝ) ≤ k := Nat.cast_nonneg k
          linarith))

/-- Mathlib's source-backed prime-side identity in the absolutely convergent
half-plane.  This is a useful restricted explicit-formula ingredient, not a
Guinand--Weil formula and not an RH statement. -/
theorem vonMangoldt_lseries_eq_neg_logDeriv {s : ℂ} (hs : 1 < s.re) :
    LSeries (fun n ↦ (ArithmeticFunction.vonMangoldt n : ℂ)) s =
      -deriv riemannZeta s / riemannZeta s := by
  simpa using ArithmeticFunction.LSeries_vonMangoldt_eq_deriv_riemannZeta_div hs

theorem vonMangoldt_lseries_summable {s : ℂ} (hs : 1 < s.re) :
    LSeriesSummable (fun n ↦ (ArithmeticFunction.vonMangoldt n : ℂ)) s := by
  simpa using ArithmeticFunction.LSeriesSummable_vonMangoldt hs

/-- The exact `n`th prime-power term used by Mathlib's von Mangoldt
L-series.  In particular the term at `n = 0` is definitionally zero, and the
terms for `n > 0` have normalization `Λ(n) / n^s`. -/
def vonMangoldtTerm (s : ℂ) (n : ℕ) : ℂ :=
  LSeries.term (fun m ↦ (ArithmeticFunction.vonMangoldt m : ℂ)) s n

theorem vonMangoldtTerm_zero (s : ℂ) : vonMangoldtTerm s 0 = 0 := by
  simp [vonMangoldtTerm, LSeries.term]

theorem vonMangoldtTerm_of_ne_zero (s : ℂ) {n : ℕ} (hn : n ≠ 0) :
    vonMangoldtTerm s n =
      (ArithmeticFunction.vonMangoldt n : ℂ) / (n : ℂ) ^ s := by
  simp [vonMangoldtTerm, LSeries.term, hn]

/-- Finite prime-power truncation in the exact Mathlib L-series
normalization.  This is a bounded right-half-plane explicit-formula piece,
not the prime side of a Guinand--Weil formula. -/
def vonMangoldtPartialSum (s : ℂ) (N : ℕ) : ℂ :=
  ∑ n ∈ Finset.range N, vonMangoldtTerm s n

/-- The actual von Mangoldt truncations converge to the zeta logarithmic
derivative throughout the half-plane `re s > 1`. -/
theorem vonMangoldtPartialSum_tendsto_logDeriv {s : ℂ} (hs : 1 < s.re) :
    Tendsto (vonMangoldtPartialSum s) atTop
      (𝓝 (-deriv riemannZeta s / riemannZeta s)) := by
  have hsum :
      Summable (fun n ↦ vonMangoldtTerm s n) := by
    exact vonMangoldt_lseries_summable hs
  have hlim :
      Tendsto (vonMangoldtPartialSum s) atTop
        (𝓝 (LSeries (fun n ↦ (ArithmeticFunction.vonMangoldt n : ℂ)) s)) := by
    change Tendsto
      (fun N ↦ ∑ n ∈ Finset.range N,
        LSeries.term (fun m ↦ (ArithmeticFunction.vonMangoldt m : ℂ)) s n)
      atTop
      (𝓝 (∑' n, LSeries.term
        (fun m ↦ (ArithmeticFunction.vonMangoldt m : ℂ)) s n))
    exact hsum.hasSum.tendsto_sum_nat
  simpa [vonMangoldt_lseries_eq_neg_logDeriv hs] using hlim

/-- Exact remainder after a finite von Mangoldt truncation. -/
def vonMangoldtTail (s : ℂ) (N : ℕ) : ℂ :=
  ∑' n, vonMangoldtTerm s (n + N)

/-- Genuine finite-plus-tail identity in the absolutely convergent
half-plane. -/
theorem vonMangoldtPartialSum_add_tail {s : ℂ} (hs : 1 < s.re) (N : ℕ) :
    vonMangoldtPartialSum s N + vonMangoldtTail s N =
      -deriv riemannZeta s / riemannZeta s := by
  have hsum : Summable (vonMangoldtTerm s) :=
    vonMangoldt_lseries_summable hs
  rw [← vonMangoldt_lseries_eq_neg_logDeriv hs]
  exact hsum.sum_add_tsum_nat_add N

/-- A bounded arithmetic smoothing weight.  The unit norm bound is chosen so
that Mathlib's absolute convergence theorem is itself a dominating series. -/
structure PrimeSmoothingWeight where
  coeff : ℕ → ℂ
  norm_le_one : ∀ n, ‖coeff n‖ ≤ 1

def smoothedVonMangoldtTerm (s : ℂ) (w : PrimeSmoothingWeight) (n : ℕ) : ℂ :=
  w.coeff n * vonMangoldtTerm s n

def smoothedVonMangoldtSeries (s : ℂ) (w : PrimeSmoothingWeight) : ℂ :=
  ∑' n, smoothedVonMangoldtTerm s w n

theorem smoothedVonMangoldt_summable
    {s : ℂ} (hs : 1 < s.re) (w : PrimeSmoothingWeight) :
    Summable (smoothedVonMangoldtTerm s w) := by
  rw [← summable_norm_iff]
  exact (vonMangoldt_lseries_summable hs).norm.of_nonneg_of_le
    (fun n ↦ norm_nonneg _) fun n ↦ by
      rw [smoothedVonMangoldtTerm, norm_mul]
      exact mul_le_of_le_one_left (norm_nonneg _) (w.norm_le_one n)

/-- Dominated convergence for bounded arithmetic smoothing weights.  The
limit is the exact negative logarithmic derivative, with no continuation
across `re s = 1` claimed. -/
theorem smoothedVonMangoldtSeries_tendsto_logDeriv
    {s : ℂ} (hs : 1 < s.re) (w : ℕ → PrimeSmoothingWeight)
    (hw : ∀ n, Tendsto (fun k ↦ (w k).coeff n) atTop (𝓝 1)) :
    Tendsto (fun k ↦ smoothedVonMangoldtSeries s (w k)) atTop
      (𝓝 (-deriv riemannZeta s / riemannZeta s)) := by
  have hdom : Summable (fun n ↦ ‖vonMangoldtTerm s n‖) :=
    (vonMangoldt_lseries_summable hs).norm
  have hterm (n : ℕ) :
      Tendsto (fun k ↦ smoothedVonMangoldtTerm s (w k) n) atTop
        (𝓝 (vonMangoldtTerm s n)) := by
    simpa [smoothedVonMangoldtTerm] using (hw n).mul_const (vonMangoldtTerm s n)
  have hbound :
      ∀ᶠ k in atTop, ∀ n, ‖smoothedVonMangoldtTerm s (w k) n‖ ≤
        ‖vonMangoldtTerm s n‖ := by
    exact Eventually.of_forall fun k n ↦ by
      rw [smoothedVonMangoldtTerm, norm_mul]
      exact mul_le_of_le_one_left (norm_nonneg _) ((w k).norm_le_one n)
  have hlim := tendsto_tsum_of_dominated_convergence hdom hterm hbound
  rw [show (∑' n, vonMangoldtTerm s n) =
      LSeries (fun n ↦ (ArithmeticFunction.vonMangoldt n : ℂ)) s by
        rfl] at hlim
  simpa [smoothedVonMangoldtSeries, vonMangoldt_lseries_eq_neg_logDeriv hs] using hlim

/-! ## The actual archimedean Gamma factor -/

/-- The archimedean logarithmic derivative in the completed-zeta
normalization used by Mathlib:
`Γℝ(s) = π^(-s/2) Γ(s/2)`. -/
def archimedeanLogDeriv (s : ℂ) : ℂ :=
  logDeriv Complex.Gammaℝ s

/-- Exact Gamma/digamma normalization on the right half-plane. -/
theorem archimedeanLogDeriv_eq_digamma {s : ℂ} (hs : 0 < s.re) :
    archimedeanLogDeriv s =
      -(Complex.log Real.pi) / 2 + Complex.digamma (s / 2) / 2 := by
  have hhalf : 0 < (s / 2).re := by
    simpa using div_pos hs (by norm_num : (0 : ℝ) < 2)
  have hnotpole : ∀ n : ℕ, s / 2 ≠ -(n : ℂ) := by
    intro n h
    have hre := congrArg Complex.re h
    simp at hre
    linarith
  have hGamma : Complex.Gamma (s / 2) ≠ 0 :=
    Complex.Gamma_ne_zero_of_re_pos hhalf
  have hpow : (Real.pi : ℂ) ^ (-s / 2) ≠ 0 :=
    by simp [Real.pi_ne_zero]
  have hpowDiff : DifferentiableAt ℂ (fun z : ℂ ↦ (Real.pi : ℂ) ^ (-z / 2)) s :=
    (by fun_prop : DifferentiableAt ℂ (fun z : ℂ ↦ -z / 2) s).const_cpow <|
      Or.inl (ofReal_ne_zero.mpr Real.pi_ne_zero)
  have hGammaDiff : DifferentiableAt ℂ (fun z : ℂ ↦ Complex.Gamma (z / 2)) s :=
    (Complex.differentiableAt_Gamma (s / 2) hnotpole).comp s (by fun_prop)
  have hexponent : HasDerivAt (fun z : ℂ ↦ -z / 2) (-1 / 2) s := by
    simpa only [Pi.neg_apply, id_eq] using (hasDerivAt_id s).neg.div_const (2 : ℂ)
  have hcomp :
      logDeriv (fun z : ℂ ↦ Complex.Gamma (z / 2)) s =
        Complex.digamma (s / 2) / 2 := by
    change logDeriv (Complex.Gamma ∘ fun z : ℂ ↦ z / 2) s = _
    rw [logDeriv_comp (f := Complex.Gamma) (g := fun z : ℂ ↦ z / 2)
      (Complex.differentiableAt_Gamma (s / 2) hnotpole)
      (show DifferentiableAt ℂ (fun z : ℂ ↦ z / 2) s by fun_prop), Complex.digamma_def]
    simp only [deriv_div_const, deriv_id'']
    ring
  rw [archimedeanLogDeriv]
  change logDeriv
    (fun z : ℂ ↦ (Real.pi : ℂ) ^ (-z / 2) * Complex.Gamma (z / 2)) s = _
  rw [logDeriv_mul s hpow hGamma hpowDiff hGammaDiff, hcomp]
  rw [logDeriv_apply, Complex.deriv_const_cpow (by fun_prop), hexponent.deriv]
  field_simp [hpow]

/-- At the central normalization point `s = 1`, Mathlib's derivative theorem
gives the exact Euler--Mascheroni and `log(4π)` constant. -/
theorem archimedeanLogDeriv_one :
    archimedeanLogDeriv 1 =
      -(Real.eulerMascheroniConstant + Complex.log (4 * Real.pi)) / 2 := by
  rw [archimedeanLogDeriv, logDeriv_apply,
    Complex.hasDerivAt_Gammaℝ_one.deriv, Complex.Gammaℝ_one, div_one]

/-- Mathlib's exact complex digamma value at positive integers. -/
theorem digamma_nat_add_one (n : ℕ) :
    Complex.digamma (n + 1) =
      -Real.eulerMascheroniConstant + (harmonic n : ℂ) := by
  rw [Complex.digamma_def, logDeriv_apply, Complex.deriv_Gamma_nat,
    Complex.Gamma_nat_eq_factorial]
  field_simp

/-- The positive-integer digamma asymptotic error from Mathlib's harmonic
number construction of the Euler--Mascheroni constant. -/
def digammaNatError (n : ℕ) : ℝ :=
  (harmonic n : ℝ) - Real.log n - Real.eulerMascheroniConstant

theorem digammaNatError_nonneg {n : ℕ} (hn : n ≠ 0) :
    0 ≤ digammaNatError n := by
  have h := Real.eulerMascheroniConstant_lt_eulerMascheroniSeq' n
  rw [Real.eulerMascheroniSeq', if_neg hn] at h
  exact sub_nonneg.mpr h.le

/-- Explicit source-backed error bound:
`0 ≤ Hₙ - log n - γ ≤ log(n+1) - log n`. -/
theorem digammaNatError_le_log_succ_sub_log (n : ℕ) :
    digammaNatError n ≤ Real.log (n + 1) - Real.log n := by
  have h := Real.eulerMascheroniSeq_lt_eulerMascheroniConstant n
  rw [Real.eulerMascheroniSeq] at h
  dsimp only [digammaNatError]
  linarith

theorem digammaNatError_tendsto_zero :
    Tendsto digammaNatError atTop (𝓝 0) := by
  change Tendsto
    (fun n : ℕ ↦ (harmonic n : ℝ) - Real.log n -
      Real.eulerMascheroniConstant) atTop (𝓝 0)
  simpa only [sub_self] using Real.tendsto_harmonic_sub_log.sub
    (show Tendsto (fun _ : ℕ ↦ Real.eulerMascheroniConstant) atTop
      (𝓝 Real.eulerMascheroniConstant) from tendsto_const_nhds)

/-- Norm error for the positive-integer digamma asymptotic. -/
theorem norm_digamma_nat_add_one_sub_log_le {n : ℕ} (hn : n ≠ 0) :
    ‖Complex.digamma (n + 1) - (Real.log n : ℂ)‖ ≤
      Real.log (n + 1) - Real.log n := by
  rw [digamma_nat_add_one]
  have hnonneg := digammaNatError_nonneg hn
  have hbound := digammaNatError_le_log_succ_sub_log n
  have heq :
      (-Real.eulerMascheroniConstant : ℂ) + (harmonic n : ℂ) -
        (Real.log n : ℂ) = (digammaNatError n : ℂ) := by
    rw [digammaNatError]
    push_cast
    ring
  rw [heq, norm_real, Real.norm_eq_abs, abs_of_nonneg hnonneg]
  exact hbound

/-- A genuine finite harmonic/digamma expression for the archimedean
logarithmic derivative at `s = 2`. -/
def archimedeanAtTwoStage (n : ℕ) : ℂ :=
  -(Complex.log Real.pi) / 2 +
    (Complex.digamma (n + 1) - (harmonic n : ℂ)) / 2

theorem archimedeanAtTwoStage_eq (n : ℕ) :
    archimedeanAtTwoStage n = archimedeanLogDeriv 2 := by
  rw [archimedeanLogDeriv_eq_digamma (s := 2) (by norm_num),
    archimedeanAtTwoStage, digamma_nat_add_one]
  norm_num
  exact Complex.digamma_one.symm

/-- A genuine finite restricted completed-log-derivative identity at `s = 2`.
It combines the actual finite von Mangoldt sum, its convergent tail, and the
actual finite harmonic/Gamma stage.  It is not a zero-sum explicit formula. -/
theorem finiteCompletedLogDerivAtTwo (N : ℕ) :
    vonMangoldtPartialSum 2 N + vonMangoldtTail 2 N +
      archimedeanAtTwoStage N =
    (-deriv riemannZeta 2 / riemannZeta 2) +
      archimedeanLogDeriv 2 := by
  rw [vonMangoldtPartialSum_add_tail (s := 2) (by norm_num),
    archimedeanAtTwoStage_eq]

/-- A quantitative approximation package for the actual Gamma-factor
logarithmic derivative.  Constructing such a package from a truncated
digamma integral/series is the remaining analytic task. -/
structure ArchimedeanGammaApproximation (s : ℂ) where
  stage : ℕ → ℂ
  error : ℕ → ℝ
  error_tendsto : Tendsto error atTop (𝓝 0)
  norm_sub_le : ∀ n, ‖stage n - archimedeanLogDeriv s‖ ≤ error n

theorem ArchimedeanGammaApproximation.tendsto
    {s : ℂ} (A : ArchimedeanGammaApproximation s) :
    Tendsto A.stage atTop (𝓝 (archimedeanLogDeriv s)) := by
  rw [tendsto_iff_norm_sub_tendsto_zero]
  exact squeeze_zero (fun n ↦ norm_nonneg (A.stage n - archimedeanLogDeriv s))
    A.norm_sub_le A.error_tendsto

/-- The positive-integer Gamma recurrence gives an exact finite
archimedean approximation at `s = 2`. -/
def archimedeanGammaApproximationAtTwo : ArchimedeanGammaApproximation 2 where
  stage := archimedeanAtTwoStage
  error := 0
  error_tendsto := tendsto_const_nhds
  norm_sub_le n := by simp [archimedeanAtTwoStage_eq n]

/-- The critical-line-centred Mellin transform in logarithmic coordinates. -/
def transform (f : Test) (z : ℂ) : ℂ :=
  ∫ t : ℝ, f t * exp (z * t)

/-- Transform evaluation as a continuous complex-linear functional on the LF
test space. -/
noncomputable def transformCLM (z : ℂ) : Test →L[ℂ] ℂ :=
  TestFunction.integralAgainstBilinCLM
    (ContinuousLinearMap.mul ℂ ℂ) volume (fun t : ℝ ↦ exp (z * t))

@[simp]
theorem transformCLM_apply (z : ℂ) (f : Test) :
    transformCLM z f = transform f z := by
  rw [transformCLM,
    TestFunction.integralAgainstBilinCLM_eq_integral]
  · rfl
  · have hc : Continuous (fun t : ℝ ↦ exp (z * t)) :=
      Complex.continuous_exp.comp
        (continuous_const.mul (Complex.continuous_ofReal.comp continuous_id))
    exact (hc.locallyIntegrable (μ := (volume : Measure ℝ))).locallyIntegrableOn univ

/-- Complex exponential modulation preserves compact support and shifts the
spectral transform. -/
def spectralShiftTest (f : Test) (z : ℂ) : Test where
  toFun t := f t * exp (-z * t)
  contDiff' := f.contDiff.mul <| Complex.contDiff_exp.comp <|
    contDiff_const.mul (Complex.ofRealCLM.contDiff.comp contDiff_id)
  hasCompactSupport' := f.hasCompactSupport.mul_right
  tsupport_subset' := tsupport_mul_subset_left.trans f.tsupport_subset

@[simp]
theorem spectralShiftTest_apply (f : Test) (z : ℂ) (t : ℝ) :
    spectralShiftTest f z t = f t * exp (-z * t) :=
  rfl

/-- Exact transform translation under complex exponential modulation. -/
theorem transform_spectralShiftTest (f : Test) (z w : ℂ) :
    transform (spectralShiftTest f z) w = transform f (w - z) := by
  rw [transform, transform]
  apply integral_congr_ae
  filter_upwards with t
  simp only [spectralShiftTest_apply]
  calc
    f t * exp (-z * t) * exp (w * t) =
        f t * (exp (-z * t) * exp (w * t)) := by ring
    _ = f t * exp ((-z * t) + (w * t)) := by rw [exp_add]
    _ = f t * exp ((w - z) * t) := by
      congr 2
      ring

theorem transform_spectralShiftTest_self (f : Test) (z : ℂ) :
    transform (spectralShiftTest f z) z = transform f 0 := by
  simpa using transform_spectralShiftTest f z z

/-- A fixed source-backed real smooth compactly supported test of integral
one. -/
def normalizedRealBumpTest : 𝓓((⊤ : Opens ℝ), ℝ) :=
  let b : ContDiffBump (0 : ℝ) := default
  { toFun := b.normed volume
    contDiff' := b.contDiff_normed
    hasCompactSupport' := b.hasCompactSupport_normed
    tsupport_subset' := Set.subset_univ _ }

/-- The corresponding complex test, obtained through Mathlib's continuous
postcomposition map. -/
def normalizedBumpTest : Test :=
  TestFunction.postcompCLM Complex.ofRealCLM normalizedRealBumpTest

theorem transform_normalizedBumpTest_zero :
    transform normalizedBumpTest 0 = 1 := by
  let b : ContDiffBump (0 : ℝ) := default
  rw [transform]
  simp only [zero_mul, exp_zero, mul_one, normalizedBumpTest,
    TestFunction.postcompCLM_apply, Function.comp_apply, normalizedRealBumpTest]
  change (∫ t : ℝ, (b.normed volume t : ℂ)) = 1
  rw [integral_complex_ofReal, b.integral_normed]
  norm_num

/-- Every complex spectral point admits an explicit compactly supported smooth
test whose transform is exactly one there. -/
def pointNormalizedTest (z : ℂ) : Test :=
  spectralShiftTest normalizedBumpTest z

theorem transform_pointNormalizedTest_self (z : ℂ) :
    transform (pointNormalizedTest z) z = 1 := by
  rw [pointNormalizedTest, transform_spectralShiftTest_self,
    transform_normalizedBumpTest_zero]

/-- Translation on the logarithmic line preserves the test class. -/
def translateTest (f : Test) (a : ℝ) : Test where
  toFun t := f (t - a)
  contDiff' := f.contDiff.comp (contDiff_id.sub contDiff_const)
  hasCompactSupport' :=
    f.hasCompactSupport.comp_homeomorph (Homeomorph.addRight (-a))
  tsupport_subset' := Set.subset_univ _

@[simp]
theorem translateTest_apply (f : Test) (a t : ℝ) :
    translateTest f a t = f (t - a) :=
  rfl

/-- Translation becomes multiplication by an exponential under the centered
transform. -/
theorem transform_translateTest (f : Test) (a : ℝ) (z : ℂ) :
    transform (translateTest f a) z = exp (z * a) * transform f z := by
  rw [transform, transform]
  simp only [translateTest_apply]
  rw [← integral_add_right_eq_self
    (fun t : ℝ ↦ f (t - a) * exp (z * t)) a]
  have hf : Integrable fun t : ℝ ↦ f t * exp (z * t) := by
    apply Continuous.integrable_of_hasCompactSupport
    · exact f.continuous.mul <| Complex.continuous_exp.comp
        (continuous_const.mul (Complex.continuous_ofReal.comp continuous_id))
    · exact f.hasCompactSupport.mul_right
  rw [← integral_const_mul]
  apply integral_congr_ae
  filter_upwards with t
  rw [show t + a - a = t by ring]
  rw [show (z * (↑(t + a) : ℂ)) = z * t + z * a by push_cast; ring, exp_add]
  ring

/-- A finite-difference test whose transform has a prescribed zero.  This is
the elementary compact-support Paley--Wiener separator factor
`1 - exp (a (z-q))`; it uses no interpolation matrix. -/
def spectralZeroFactor (f : Test) (a : ℝ) (q : ℂ) : Test :=
  f - exp (-q * a) • translateTest f a

theorem transform_spectralZeroFactor
    (f : Test) (a : ℝ) (q z : ℂ) :
    transform (spectralZeroFactor f a q) z =
      (1 - exp ((z - q) * a)) * transform f z := by
  rw [← transformCLM_apply]
  simp only [spectralZeroFactor, map_sub, map_smul, transformCLM_apply,
    transform_translateTest]
  change transform f z - exp (-q * a) *
      (exp (z * a) * transform f z) =
    (1 - exp ((z - q) * a)) * transform f z
  have hexp :
      exp (-q * a) * exp (z * a) = exp ((z - q) * a) := by
    rw [← exp_add]
    congr 1
    ring
  rw [← mul_assoc, hexp]
  ring

/-- Iterating finite differences inserts any finite list of exact spectral
zeros while preserving compact support and smoothness. -/
def spectralZeroProduct (f : Test) (a : ℝ) : List ℂ → Test
  | [] => f
  | q :: roots => spectralZeroProduct (spectralZeroFactor f a q) a roots

theorem transform_spectralZeroProduct
    (f : Test) (a : ℝ) (roots : List ℂ) (z : ℂ) :
    transform (spectralZeroProduct f a roots) z =
      (roots.map fun q ↦ (1 - exp ((z - q) * a))).prod * transform f z := by
  induction roots generalizing f with
  | nil => simp [spectralZeroProduct]
  | cons q roots ih =>
      rw [spectralZeroProduct, ih, transform_spectralZeroFactor]
      simp only [List.map_cons, List.prod_cons]
      ring

/-- A local separator normalized to one at `target` and forced to vanish on
the finite list `roots`.  Its only denominator is the product of local gap
factors, making near-collision dependence explicit. -/
noncomputable def localSpectralSeparator
    (a : ℝ) (target : ℂ) (roots : List ℂ) : Test :=
  ((roots.map fun q ↦ (1 - exp ((target - q) * a))).prod)⁻¹ •
    spectralZeroProduct (pointNormalizedTest target) a roots

theorem transform_localSpectralSeparator
    (a : ℝ) (target : ℂ) (roots : List ℂ) (z : ℂ) :
    transform (localSpectralSeparator a target roots) z =
      ((roots.map fun q ↦ (1 - exp ((target - q) * a))).prod)⁻¹ *
        ((roots.map fun q ↦ (1 - exp ((z - q) * a))).prod *
          transform (pointNormalizedTest target) z) := by
  rw [← transformCLM_apply]
  simp [localSpectralSeparator, transform_spectralZeroProduct]

theorem transform_localSpectralSeparator_self
    (a : ℝ) (target : ℂ) (roots : List ℂ)
    (hgap : (roots.map fun q ↦
      (1 - exp ((target - q) * a))).prod ≠ 0) :
    transform (localSpectralSeparator a target roots) target = 1 := by
  rw [transform_localSpectralSeparator,
    transform_pointNormalizedTest_self]
  simpa using inv_mul_cancel₀ hgap

theorem transform_localSpectralSeparator_eq_zero
    (a : ℝ) (target : ℂ) (roots : List ℂ) {q : ℂ} (hq : q ∈ roots) :
    transform (localSpectralSeparator a target roots) q = 0 := by
  rw [transform_localSpectralSeparator]
  have hprod :
      (roots.map fun w ↦ (1 - exp ((q - w) * a))).prod = 0 := by
    apply List.prod_eq_zero
    exact List.mem_map.mpr ⟨q, hq, by simp⟩
  rw [hprod]
  ring

/-- Exact local noncollision condition.  Unlike a global zero-spacing bound,
this concerns only one target and one finite neighboring list. -/
def LocalSeparatorGap (a : ℝ) (target : ℂ) (roots : List ℂ) : Prop :=
  ∀ q ∈ roots, 1 - exp ((target - q) * a) ≠ 0

theorem LocalSeparatorGap.prod_ne_zero
    {a : ℝ} {target : ℂ} {roots : List ℂ}
    (h : LocalSeparatorGap a target roots) :
    (roots.map fun q ↦ (1 - exp ((target - q) * a))).prod ≠ 0 := by
  apply List.prod_ne_zero
  intro hzero
  rcases List.mem_map.mp hzero with ⟨q, hq, hqzero⟩
  exact h q hq hqzero

/-- Two local finite-difference kernels, each annihilating the reflected
target and the prescribed nearby points. -/
noncomputable def localPairSeparator
    (a : ℝ) (z : ℂ) (nearby : List ℂ) : Test :=
  localSpectralSeparator a z (-star z :: nearby) -
    localSpectralSeparator a (-star z) (z :: nearby)

theorem transform_localPairSeparator_target
    (a : ℝ) (z : ℂ) (nearby : List ℂ)
    (hz : LocalSeparatorGap a z (-star z :: nearby)) :
    transform (localPairSeparator a z nearby) z = 1 := by
  rw [← transformCLM_apply]
  simp only [localPairSeparator, map_sub, transformCLM_apply]
  rw [transform_localSpectralSeparator_self _ _ _ hz.prod_ne_zero,
    transform_localSpectralSeparator_eq_zero _ _ _ (by simp)]
  ring

theorem transform_localPairSeparator_partner
    (a : ℝ) (z : ℂ) (nearby : List ℂ)
    (hpartner : LocalSeparatorGap a (-star z) (z :: nearby)) :
    transform (localPairSeparator a z nearby) (-star z) = -1 := by
  rw [← transformCLM_apply]
  simp only [localPairSeparator, map_sub, transformCLM_apply]
  rw [transform_localSpectralSeparator_eq_zero _ _ _ (by simp),
    transform_localSpectralSeparator_self _ _ _ hpartner.prod_ne_zero]
  ring

theorem transform_localPairSeparator_nearby_eq_zero
    (a : ℝ) (z : ℂ) (nearby : List ℂ) {q : ℂ} (hq : q ∈ nearby) :
    transform (localPairSeparator a z nearby) q = 0 := by
  rw [← transformCLM_apply]
  simp only [localPairSeparator, map_sub, transformCLM_apply]
  rw [transform_localSpectralSeparator_eq_zero _ _ _
      (List.mem_cons_of_mem _ hq),
    transform_localSpectralSeparator_eq_zero _ _ _
      (List.mem_cons_of_mem _ hq)]
  ring

/-- A finite family of transform evaluations has one common test at which
none of them vanish.  The proof avoids a finite set of bad coefficients at
each induction step; no determinant or analytic zero-set theorem is needed. -/
theorem exists_test_transform_ne_zero_on_finset (points : Finset ℂ) :
    ∃ base : Test, ∀ z ∈ points, transform base z ≠ 0 := by
  classical
  induction points using Finset.induction_on with
  | empty =>
      exact ⟨0, by simp⟩
  | @insert z s hz ih =>
      rcases ih with ⟨f, hf⟩
      let g : Test := pointNormalizedTest z
      let bad : Finset ℂ :=
        insert (-transform f z)
          (s.image fun w ↦ -transform f w / transform g w)
      obtain ⟨c, hc⟩ := bad.finite_toSet.exists_notMem
      refine ⟨f + c • g, ?_⟩
      intro w hw
      have heval :
          transform (f + c • g) w =
            transform f w + c * transform g w := by
        rw [← transformCLM_apply w (f + c • g),
          ← transformCLM_apply w f, ← transformCLM_apply w g]
        rw [map_add, map_smul]
        rfl
      rw [heval]
      rw [Finset.mem_insert] at hw
      rcases hw with rfl | hw
      · have hg : transform g w = 1 := by
          exact transform_pointNormalizedTest_self w
        rw [hg, mul_one]
        have hc0 : c ≠ -transform f w := by
          intro h
          apply hc
          change c ∈ bad
          exact Finset.mem_insert.mpr (Or.inl h)
        intro h
        apply hc0
        linear_combination h
      · have hfw : transform f w ≠ 0 := hf w hw
        by_cases hgw : transform g w = 0
        · simpa [hgw] using hfw
        · have hcbad :
              c ≠ -transform f w / transform g w := by
            intro h
            apply hc
            change c ∈ bad
            exact Finset.mem_insert.mpr <| Or.inr <|
              Finset.mem_image.mpr ⟨w, hw, h.symm⟩
          intro hsum
          apply hcbad
          apply (eq_div_iff hgw).2
          linear_combination hsum

/-- Complex exponential is injective between two points whose imaginary
coordinates differ by less than one full period. -/
theorem exp_eq_exp_of_abs_im_sub_lt_two_pi
    {x y : ℂ} (hxy : |x.im - y.im| < 2 * Real.pi)
    (hexp : exp x = exp y) : x = y := by
  rcases Complex.exp_eq_exp_iff_exists_int.mp hexp with ⟨m, hm⟩
  have him := congrArg Complex.im hm
  have him' : x.im - y.im = (m : ℝ) * (2 * Real.pi) := by
    norm_num [mul_re, mul_im] at him
    linarith
  have hm0 : m = 0 := by
    by_contra hm0
    have habsm : (1 : ℝ) ≤ |(m : ℝ)| := by
      exact_mod_cast Int.one_le_abs hm0
    have hperiod : 2 * Real.pi ≤ |(m : ℝ) * (2 * Real.pi)| := by
      rw [abs_mul, abs_of_pos Real.two_pi_pos]
      nlinarith [Real.two_pi_pos]
    rw [him'] at hxy
    linarith
  subst m
  simpa using hm

/-- A concrete positive translation step whose exponential nodes are
collision-free for every injective finite complex point family. -/
def collisionFreeTranslationStep {n : ℕ} (points : Fin n → ℂ) : ℝ :=
  1 / (2 * (∑ i, |(points i).im|) + 1)

theorem collisionFreeTranslationStep_pos
    {n : ℕ} (points : Fin n → ℂ) :
    0 < collisionFreeTranslationStep points := by
  apply one_div_pos.mpr
  have hsum : 0 ≤ ∑ i, |(points i).im| :=
    Finset.sum_nonneg fun _ _ ↦ abs_nonneg _
  linarith

theorem collisionFreeTranslationStep_nodes_injective
    {n : ℕ} (points : Fin n → ℂ) (hpoints : Function.Injective points) :
    Function.Injective
      (fun i ↦ exp (points i * collisionFreeTranslationStep points)) := by
  intro i j hij
  let R : ℝ := ∑ k, |(points k).im|
  let a : ℝ := collisionFreeTranslationStep points
  have hR : 0 ≤ R := Finset.sum_nonneg fun _ _ ↦ abs_nonneg _
  have hiR : |(points i).im| ≤ R := by
    exact Finset.single_le_sum (s := Finset.univ)
      (f := fun k ↦ |(points k).im|)
      (fun _ _ ↦ abs_nonneg _) (Finset.mem_univ i)
  have hjR : |(points j).im| ≤ R := by
    exact Finset.single_le_sum (s := Finset.univ)
      (f := fun k ↦ |(points k).im|)
      (fun _ _ ↦ abs_nonneg _) (Finset.mem_univ j)
  have ha : 0 < a := collisionFreeTranslationStep_pos points
  have haR : a * (2 * R + 1) = 1 := by
    dsimp [a, collisionFreeTranslationStep, R]
    field_simp
  have him :
      |(points i * a).im - (points j * a).im| < 2 * Real.pi := by
    have hdiff :
        |(points i).im - (points j).im| ≤ 2 * R := by
      calc
        |(points i).im - (points j).im| ≤
            |(points i).im| + |(points j).im| := abs_sub _ _
        _ ≤ 2 * R := by linarith
    have hone : a * (2 * R) < 1 := by nlinarith
    have hpi : (1 : ℝ) < 2 * Real.pi := by
      nlinarith [Real.pi_gt_three]
    simp only [mul_im, ofReal_re, ofReal_im, mul_zero, zero_add]
    rw [← sub_mul, abs_mul, abs_of_pos ha]
    calc
      |(points i).im - (points j).im| * a ≤ (2 * R) * a :=
        mul_le_mul_of_nonneg_right hdiff ha.le
      _ < 1 := by nlinarith
      _ < 2 * Real.pi := hpi
  apply hpoints
  have hscaled := exp_eq_exp_of_abs_im_sub_lt_two_pi
    (x := points i * (a : ℂ)) (y := points j * (a : ℂ)) him hij
  exact mul_right_cancel₀ (Complex.ofReal_ne_zero.mpr ha.ne') hscaled

/-- A finite locally discrete target, partner, and neighbor family admits one
common positive translation scale satisfying both local gap conditions. -/
theorem exists_pairLocalSeparatorGap
    (z : ℂ) (nearby : List ℂ)
    (hnodup : (z :: -star z :: nearby).Nodup) :
    ∃ a > 0,
      LocalSeparatorGap a z (-star z :: nearby) ∧
        LocalSeparatorGap a (-star z) (z :: nearby) := by
  let all := z :: -star z :: nearby
  let points : Fin all.length → ℂ := all.get
  let a := collisionFreeTranslationStep points
  have hpoints : Function.Injective points := by
    exact hnodup.injective_get
  have hnodes : Function.Injective fun i ↦ exp (points i * a) :=
    collisionFreeTranslationStep_nodes_injective points hpoints
  have hnode_ne {x y : ℂ} (hx : x ∈ all) (hy : y ∈ all) (hxy : x ≠ y) :
      exp (x * a) ≠ exp (y * a) := by
    obtain ⟨ix, hix⟩ := List.get_of_mem hx
    obtain ⟨iy, hiy⟩ := List.get_of_mem hy
    intro heq
    apply hxy
    have hnodeIndices :
        exp (points ix * a) = exp (points iy * a) := by
      change exp (all.get ix * a) = exp (all.get iy * a)
      rw [hix, hiy]
      exact heq
    have hindices : ix = iy := hnodes hnodeIndices
    rw [← hix, ← hiy, hindices]
  have hgap {target q : ℂ} (htarget : target ∈ all)
      (hq : q ∈ all) (hne : target ≠ q) :
      1 - exp ((target - q) * a) ≠ 0 := by
    intro hzero
    have hexp : exp ((target - q) * a) = 1 := by
      exact (sub_eq_zero.mp hzero).symm
    apply hnode_ne htarget hq hne
    calc
      exp (target * a) =
          exp (((target - q) * a) + q * a) := by
            congr 1
            ring
      _ = exp ((target - q) * a) * exp (q * a) := exp_add _ _
      _ = exp (q * a) := by rw [hexp, one_mul]
  have hzall : z ∈ all := by simp [all]
  have hpall : -star z ∈ all := by simp [all]
  have hznot : z ∉ -star z :: nearby := (List.nodup_cons.mp hnodup).1
  have hpnotnear : -star z ∉ nearby :=
    (List.nodup_cons.mp (List.nodup_cons.mp hnodup).2).1
  have hpnez : -star z ≠ z := by
    intro h
    apply hznot
    exact List.mem_cons.mpr (Or.inl h.symm)
  have hpnot : -star z ∉ z :: nearby := by
    simpa only [List.mem_cons, not_or] using And.intro hpnez hpnotnear
  refine ⟨a, collisionFreeTranslationStep_pos points, ?_, ?_⟩
  · intro q hq
    exact hgap hzall (by
      dsimp [all]
      exact List.mem_cons.mpr (Or.inr hq))
      (fun h ↦ hznot (by simpa [h] using hq))
  · intro q hq
    have hne : -star z ≠ q := by
      intro h
      apply hpnot
      rw [h]
      exact hq
    have hqall : q ∈ all := by
      rcases List.mem_cons.mp hq with rfl | hq
      · exact hzall
      · dsimp [all]
        exact List.mem_cons.mpr <| Or.inr <|
          List.mem_cons.mpr <| Or.inr hq
    exact hgap hpall hqall hne

/-- Evaluation matrix of translates of one base test.  This basis exposes the
only two finite interpolation degeneracies: a zero base transform, or
colliding exponential nodes. -/
def translatedBasisMatrix (n : ℕ) (base : Test) (a : ℝ)
    (points : Fin n → ℂ) : Matrix (Fin n) (Fin n) ℂ :=
  fun i j ↦ transform (translateTest base (a * (j : ℝ))) (points i)

theorem translatedBasisMatrix_eq_diagonal_mul_vandermonde
    (n : ℕ) (base : Test) (a : ℝ) (points : Fin n → ℂ) :
    translatedBasisMatrix n base a points =
      Matrix.diagonal (fun i ↦ transform base (points i)) *
        Matrix.vandermonde (fun i ↦ exp (points i * a)) := by
  ext i j
  rw [translatedBasisMatrix, transform_translateTest,
    Matrix.diagonal_mul, Matrix.vandermonde_apply]
  have hexp :
      exp (points i * ((a * (j : ℝ) : ℝ) : ℂ)) =
        exp (points i * a) ^ (j : ℕ) := by
    rw [← Complex.exp_nat_mul]
    congr 1
    push_cast
    ring
  rw [hexp]
  ring

theorem translatedBasisMatrix_det_ne_zero
    (n : ℕ) (base : Test) (a : ℝ) (points : Fin n → ℂ)
    (hbase : ∀ i, transform base (points i) ≠ 0)
    (hnodes : Function.Injective fun i ↦ exp (points i * a)) :
    (translatedBasisMatrix n base a points).det ≠ 0 := by
  rw [translatedBasisMatrix_eq_diagonal_mul_vandermonde,
    Matrix.det_mul, Matrix.det_diagonal]
  exact mul_ne_zero (Finset.prod_ne_zero_iff.mpr fun i _ ↦ hbase i)
    (Matrix.det_vandermonde_ne_zero_iff.mpr hnodes)

theorem translatedBasisMatrix_det_ne_zero_iff
    (n : ℕ) (base : Test) (a : ℝ) (points : Fin n → ℂ) :
    (translatedBasisMatrix n base a points).det ≠ 0 ↔
      (∀ i, transform base (points i) ≠ 0) ∧
        Function.Injective (fun i ↦ exp (points i * a)) := by
  rw [translatedBasisMatrix_eq_diagonal_mul_vandermonde,
    Matrix.det_mul, Matrix.det_diagonal, mul_ne_zero_iff,
    Finset.prod_ne_zero_iff, Matrix.det_vandermonde_ne_zero_iff]
  simp

/-- Exact two-point determinant, exposing the node-separation factor. -/
theorem translatedBasisMatrix_finTwo_det
    (base : Test) (a : ℝ) (z₀ z₁ : ℂ) :
    (translatedBasisMatrix 2 base a ![z₀, z₁]).det =
      transform base z₀ * transform base z₁ *
        (exp (z₁ * a) - exp (z₀ * a)) := by
  rw [Matrix.det_fin_two]
  simp [translatedBasisMatrix, transform_translateTest]
  ring

theorem translatedBasisMatrix_det_isUnit
    (n : ℕ) (base : Test) (a : ℝ) (points : Fin n → ℂ)
    (hbase : ∀ i, transform base (points i) ≠ 0)
    (hnodes : Function.Injective fun i ↦ exp (points i * a)) :
    IsUnit (translatedBasisMatrix n base a points).det :=
  (isUnit_iff_ne_zero.mpr
    (translatedBasisMatrix_det_ne_zero n base a points hbase hnodes))

/-- Arbitrary finite transform data, interpolated by translates of one test.
The explicit hypotheses are exactly the degeneracies in the determinant
factorization above. -/
def translatedInterpolatingTest (n : ℕ) (base : Test) (a : ℝ)
    (points values : Fin n → ℂ) : Test :=
  ∑ j, ((translatedBasisMatrix n base a points)⁻¹).mulVec values j •
    translateTest base (a * (j : ℝ))

/-- Coefficients of the translated Vandermonde interpolator. -/
def translatedInterpolationCoefficients (n : ℕ) (base : Test) (a : ℝ)
    (points values : Fin n → ℂ) : Fin n → ℂ :=
  ((translatedBasisMatrix n base a points)⁻¹).mulVec values

/-- Explicit coefficient/translation amplification on the centered strip. -/
def translatedInterpolationWeight (n : ℕ) (base : Test) (a : ℝ)
    (points values : Fin n → ℂ) : ℝ :=
  ∑ j, ‖translatedInterpolationCoefficients n base a points values j‖ *
    Real.exp (|a * (j : ℝ)| / 2)

/-- Exact transform expansion of the translated interpolator at every
complex point. -/
theorem transform_translatedInterpolatingTest_eq_sum
    (n : ℕ) (base : Test) (a : ℝ) (points values : Fin n → ℂ) (z : ℂ) :
    transform (translatedInterpolatingTest n base a points values) z =
      ∑ j, translatedInterpolationCoefficients n base a points values j *
        (exp (z * (a * (j : ℝ))) * transform base z) := by
  rw [← transformCLM_apply]
  simp [translatedInterpolatingTest, translatedInterpolationCoefficients,
    transformCLM_apply, transform_translateTest]

theorem transform_translatedInterpolatingTest
    (n : ℕ) (base : Test) (a : ℝ) (points values : Fin n → ℂ)
    (hbase : ∀ i, transform base (points i) ≠ 0)
    (hnodes : Function.Injective fun i ↦ exp (points i * a))
    (i : Fin n) :
    transform (translatedInterpolatingTest n base a points values) (points i) =
      values i := by
  let A : Matrix (Fin n) (Fin n) ℂ := translatedBasisMatrix n base a points
  let c : Fin n → ℂ := (A⁻¹).mulVec values
  have hA : IsUnit A.det :=
    translatedBasisMatrix_det_isUnit n base a points hbase hnodes
  calc
    transform (translatedInterpolatingTest n base a points values) (points i) =
        ∑ j, c j * A i j := by
      rw [← transformCLM_apply]
      simp [translatedInterpolatingTest, c, A, translatedBasisMatrix]
    _ = A.mulVec c i := by
      rw [Matrix.mulVec, dotProduct]
      apply Finset.sum_congr rfl
      intro j hj
      exact mul_comm _ _
    _ = values i := by
      dsimp [c]
      rw [Matrix.mulVec_mulVec, Matrix.mul_nonsing_inv A hA,
        Matrix.one_mulVec]

/-- Distinct finite spectral points always have a common nonvanishing base,
a collision-free translated Vandermonde basis, and arbitrary interpolation. -/
theorem exists_translatedInterpolatingTest
    (n : ℕ) (points values : Fin n → ℂ)
    (hpoints : Function.Injective points) :
    ∃ base : Test,
      ∀ i : Fin n,
        transform
          (translatedInterpolatingTest n base
            (collisionFreeTranslationStep points) points values)
          (points i) = values i := by
  classical
  let spectralSet : Finset ℂ := Finset.univ.image points
  rcases exists_test_transform_ne_zero_on_finset spectralSet with
    ⟨base, hbaseSet⟩
  have hbase : ∀ i, transform base (points i) ≠ 0 := by
    intro i
    apply hbaseSet (points i)
    exact Finset.mem_image.mpr ⟨i, Finset.mem_univ i, rfl⟩
  refine ⟨base, fun i ↦ ?_⟩
  exact transform_translatedInterpolatingTest n base
    (collisionFreeTranslationStep points) points values hbase
    (collisionFreeTranslationStep_nodes_injective points hpoints) i

/-- Finite closure under the functional-equation reflection
`z ↦ -conj z`. -/
def spectralSymmetryClosure (points : Finset ℂ) : Finset ℂ :=
  points ∪ points.image (fun z ↦ -star z)

theorem mem_spectralSymmetryClosure (points : Finset ℂ) {z : ℂ}
    (hz : z ∈ points) :
    z ∈ spectralSymmetryClosure points ∧
      -star z ∈ spectralSymmetryClosure points := by
  constructor
  · exact Finset.mem_union_left _ hz
  · exact Finset.mem_union_right _ <|
      Finset.mem_image.mpr ⟨z, hz, rfl⟩

theorem exists_test_transform_ne_zero_on_symmetryClosure
    (points : Finset ℂ) :
    ∃ base : Test, ∀ z ∈ spectralSymmetryClosure points,
      transform base z ≠ 0 :=
  exists_test_transform_ne_zero_on_finset (spectralSymmetryClosure points)

/-- Evaluation matrix of the explicit point-normalized tests at a finite
family of spectral points. -/
def pointInterpolationMatrix
    {ι : Type*} [Fintype ι] (points : ι → ℂ) : Matrix ι ι ℂ :=
  fun i j ↦ transform (pointNormalizedTest (points j)) (points i)

/-- Explicit finite simultaneous interpolant obtained by inverting the
point-normalized evaluation matrix. -/
def simultaneousInterpolatingTest
    {ι : Type*} [Fintype ι] [DecidableEq ι]
    (points values : ι → ℂ) : Test :=
  ∑ j, ((pointInterpolationMatrix points)⁻¹).mulVec values j •
    pointNormalizedTest (points j)

/-- Whenever the explicit evaluation matrix is nonsingular, the constructed
compactly supported smooth test takes all prescribed transform values
simultaneously. -/
theorem transform_simultaneousInterpolatingTest
    {ι : Type*} [Fintype ι] [DecidableEq ι]
    (points values : ι → ℂ)
    (hA : IsUnit (pointInterpolationMatrix points).det) (i : ι) :
    transform (simultaneousInterpolatingTest points values) (points i) =
      values i := by
  let A : Matrix ι ι ℂ := pointInterpolationMatrix points
  let c : ι → ℂ := (A⁻¹).mulVec values
  calc
    transform (simultaneousInterpolatingTest points values) (points i) =
        ∑ j, c j * transform (pointNormalizedTest (points j)) (points i) := by
      rw [← transformCLM_apply]
      simp [simultaneousInterpolatingTest, c, A]
    _ = A.mulVec c i := by
      rw [Matrix.mulVec, dotProduct]
      change ∑ j, c j * A i j = ∑ j, A i j * c j
      apply Finset.sum_congr rfl
      intro j hj
      exact mul_comm _ _
    _ = values i := by
      dsimp [c]
      rw [Matrix.mulVec_mulVec, Matrix.mul_nonsing_inv A hA,
        Matrix.one_mulVec]

theorem pointInterpolationMatrix_finOne_isUnit (z : ℂ) :
    IsUnit (pointInterpolationMatrix (fun _ : Fin 1 ↦ z)).det := by
  have hmatrix :
      pointInterpolationMatrix (fun _ : Fin 1 ↦ z) = 1 := by
    ext i j
    have hi : i = 0 := Fin.eq_zero i
    have hj : j = 0 := Fin.eq_zero j
    subst i
    subst j
    simp [pointInterpolationMatrix, transform_pointNormalizedTest_self]
  rw [hmatrix]
  simp

/-- On the critical line the centred transform is exactly Mathlib's Fourier
transform, after accounting for Mathlib's `2π` convention. -/
theorem transform_imaginary_eq_fourier (f : Test) (ξ : ℝ) :
    transform f (-2 * Real.pi * ξ * I) = 𝓕 (toSchwartz f) ξ := by
  rw [transform, SchwartzMap.fourier_coe, Real.fourier_eq']
  apply integral_congr_ae
  filter_upwards with t
  simp only [toSchwartz_apply, smul_eq_mul]
  rw [mul_comm]
  congr 1
  congr 1
  push_cast
  simp
  ring

/-- Exponentially weight a compactly supported smooth logarithmic test.
This remains a test function with the same support bound. -/
def exponentiallyWeightedTest (f : Test) (a : ℝ) : Test where
  toFun t := f t * (Real.exp (a * t) : ℂ)
  contDiff' := f.contDiff.mul <|
    Complex.ofRealCLM.contDiff.comp <|
      Real.contDiff_exp.comp (contDiff_const.mul contDiff_id)
  hasCompactSupport' := f.hasCompactSupport.mul_right
  tsupport_subset' := tsupport_mul_subset_left.trans f.tsupport_subset

/-- Explicit pointwise majorant for derivatives of `e^{a x} f(x)`, uniform
for `|a| ≤ 1/2`. -/
def stripDerivativeMajorant (f : Test) (n : ℕ) (x : ℝ) : ℝ :=
  ∑ i ∈ Finset.range (n + 1),
    n.choose i * (1 / 2 : ℝ) ^ i * Real.exp (|x| / 2) *
      ‖iteratedDeriv (n - i) f x‖

theorem hasCompactSupport_iteratedDeriv (f : Test) (n : ℕ) :
    HasCompactSupport (iteratedDeriv n f) := by
  induction n with
  | zero => simpa [iteratedDeriv_zero] using f.hasCompactSupport
  | succ n ih =>
      rw [iteratedDeriv_succ]
      exact ih.deriv

theorem integrable_stripDerivativeMajorant (f : Test) (n : ℕ) :
    Integrable (stripDerivativeMajorant f n) := by
  change Integrable fun x ↦ ∑ i ∈ Finset.range (n + 1),
    n.choose i * (1 / 2 : ℝ) ^ i * Real.exp (|x| / 2) *
      ‖iteratedDeriv (n - i) f x‖
  apply integrable_finsetSum
  intro i hi
  apply Continuous.integrable_of_hasCompactSupport
  · have hle : ((n - i : ℕ) : ℕ∞ω) ≤ (∞ : ℕ∞ω) :=
      WithTop.coe_le_coe.mpr (ENat.coe_lt_top (n - i)).le
    have hd : Continuous (iteratedDeriv (n - i) (f : ℝ → ℂ)) :=
      (f.contDiff.of_le hle).continuous_iteratedDeriv' _
    fun_prop
  · exact (hasCompactSupport_iteratedDeriv f (n - i)).norm.mul_left

theorem norm_iteratedDeriv_exponentiallyWeightedTest_le
    (f : Test) (n : ℕ) {a : ℝ}
    (ha : a ∈ Set.Icc (-1 / 2 : ℝ) (1 / 2)) (x : ℝ) :
    ‖iteratedDeriv n (exponentiallyWeightedTest f a) x‖ ≤
      stripDerivativeMajorant f n x := by
  have hfun :
      (exponentiallyWeightedTest f a : ℝ → ℂ) =
        (Real.exp ∘ fun y ↦ a * id y) • (f : ℝ → ℂ) := by
    ext y
    simp [exponentiallyWeightedTest, Algebra.smul_def, mul_comm]
  rw [hfun]
  rw [show iteratedDeriv n
      ((Real.exp ∘ fun y ↦ a * id y) • (f : ℝ → ℂ)) x =
        ∑ i ∈ Finset.range (n + 1),
          n.choose i • iteratedDeriv i (Real.exp ∘ fun y ↦ a * id y) x •
            iteratedDeriv (n - i) f x by
    simpa only [iteratedDerivWithin_univ] using
      (iteratedDerivWithin_smul (x := x) (s := Set.univ) (n := n)
        (Set.mem_univ x) uniqueDiffOn_univ
        ((Real.contDiff_exp.comp
          (contDiff_const.mul contDiff_id)).contDiffAt.of_le
            (mod_cast le_top)).contDiffWithinAt
        (f.contDiff.contDiffAt.of_le (mod_cast le_top)).contDiffWithinAt)]
  have hderiv (i : ℕ) :
      iteratedDeriv i (Real.exp ∘ fun y ↦ a * id y) x =
        a ^ i * Real.exp (a * x) := by
    simpa [Function.comp_def] using
      congrFun (iteratedDeriv_exp_const_mul i a) x
  simp_rw [hderiv]
  apply (norm_sum_le _ _).trans
  rw [stripDerivativeMajorant]
  apply Finset.sum_le_sum
  intro i hi
  simp [Real.norm_eq_abs, Complex.norm_exp]
  have haabs : |a| ≤ 1 / 2 := by
    rw [abs_le]
    constructor <;> linarith [ha.1, ha.2]
  have haexp : Real.exp (a * x) ≤ Real.exp (|x| / 2) := by
    apply Real.exp_le_exp.mpr
    calc
      a * x ≤ |a * x| := le_abs_self _
      _ = |a| * |x| := abs_mul a x
      _ ≤ (1 / 2) * |x| := mul_le_mul_of_nonneg_right haabs (abs_nonneg x)
      _ = |x| / 2 := by ring
  have hpow : |a| ^ i ≤ (1 / 2 : ℝ) ^ i :=
    pow_le_pow_left₀ (abs_nonneg a) haabs i
  have hprod :
      |a| ^ i * Real.exp (a * x) ≤
        (1 / 2 : ℝ) ^ i * Real.exp (|x| / 2) :=
    mul_le_mul hpow haexp (Real.exp_nonneg _) (pow_nonneg (by norm_num) _)
  have hchoose := mul_le_mul_of_nonneg_left hprod
    (show 0 ≤ (n.choose i : ℝ) from Nat.cast_nonneg _)
  have hnorm := mul_le_mul_of_nonneg_right hchoose
    (norm_nonneg (iteratedDeriv (n - i) f x))
  simpa [mul_assoc] using hnorm

theorem integral_norm_iteratedDeriv_exponentiallyWeightedTest_le
    (f : Test) (n : ℕ) {a : ℝ}
    (ha : a ∈ Set.Icc (-1 / 2 : ℝ) (1 / 2)) :
    (∫ x : ℝ, ‖iteratedDeriv n (exponentiallyWeightedTest f a) x‖) ≤
      ∫ x : ℝ, stripDerivativeMajorant f n x := by
  apply integral_mono
  · have hcont : Continuous
        (iteratedDeriv n (exponentiallyWeightedTest f a : ℝ → ℂ)) :=
      (exponentiallyWeightedTest f a).contDiff.continuous_iteratedDeriv n
        (WithTop.coe_le_coe.mpr (ENat.coe_lt_top n).le)
    exact hcont.norm.integrable_of_hasCompactSupport
      (hasCompactSupport_iteratedDeriv (exponentiallyWeightedTest f a) n).norm
  · exact integrable_stripDerivativeMajorant f n
  · exact norm_iteratedDeriv_exponentiallyWeightedTest_le f n ha

@[simp]
theorem exponentiallyWeightedTest_apply (f : Test) (a t : ℝ) :
    exponentiallyWeightedTest f a t = f t * (Real.exp (a * t) : ℂ) :=
  rfl

/-- Every vertical line is a Fourier transform after exponential weighting.
This is the exact bridge from compactly supported tests to Mathlib's
Schwartz-space rapid decay API. -/
theorem transform_vertical_eq_fourier (f : Test) (a ξ : ℝ) :
    transform f ((a : ℂ) - 2 * Real.pi * ξ * I) =
      𝓕 (toSchwartz (exponentiallyWeightedTest f a)) ξ := by
  rw [transform, SchwartzMap.fourier_coe, Real.fourier_eq']
  apply integral_congr_ae
  filter_upwards with t
  simp only [toSchwartz_apply, exponentiallyWeightedTest_apply, smul_eq_mul]
  simp only [RCLike.inner_apply, conj_trivial]
  rw [Complex.ofReal_exp]
  push_cast
  have himag :
      exp ((-2 * Real.pi * ξ * I) * (t : ℂ)) =
        exp ((-2 * Real.pi * (t * ξ) : ℝ) * I) := by
    congr 1
    push_cast
    ring_nf
  have himag' :
      exp ((-2 * Real.pi * (t * ξ) : ℝ) * I) =
        exp (-2 * (Real.pi : ℂ) * (ξ : ℂ) * (t : ℂ) * I) := by
    congr 1
    push_cast
    ring_nf
  have hsplit :
      exp (((a : ℂ) - 2 * Real.pi * ξ * I) * t) =
        exp ((a : ℂ) * t) * exp ((-2 * Real.pi * ξ * I) * t) := by
    rw [← exp_add]
    congr 2
    ring_nf
  rw [hsplit, himag, himag']
  ring_nf

/-- Quantitative rapid decay on every fixed vertical line, with the bound
given by an actual Schwartz seminorm of the Fourier transform. -/
theorem norm_transform_vertical_pow_le (f : Test) (a ξ : ℝ) (k : ℕ) :
    |ξ| ^ k * ‖transform f ((a : ℂ) - 2 * Real.pi * ξ * I)‖ ≤
      (SchwartzMap.seminorm ℂ k 0)
        (𝓕 (toSchwartz (exponentiallyWeightedTest f a))) := by
  rw [transform_vertical_eq_fourier]
  simpa [Real.norm_eq_abs] using
    SchwartzMap.norm_pow_mul_le_seminorm ℂ
      (𝓕 (toSchwartz (exponentiallyWeightedTest f a))) k ξ

/-- Explicit strip-decay constant obtained from the uniform derivative
majorant. -/
def stripTransformDecayConstant (f : Test) (n : ℕ) : ℝ :=
  ∫ x : ℝ, stripDerivativeMajorant f n x

/-- Non-vacuous strip-uniform rapid decay for every compactly supported smooth
test.  The constant is an explicit integral of finitely many derivatives of
`f`; no parameter-to-Schwartz boundedness assumption is used. -/
theorem norm_transform_strip_derivative_le
    (f : Test) (n : ℕ) {a : ℝ}
    (ha : a ∈ Set.Icc (-1 / 2 : ℝ) (1 / 2)) (ξ : ℝ) :
    (2 * Real.pi * |ξ|) ^ n *
        ‖transform f ((a : ℂ) - 2 * Real.pi * ξ * I)‖ ≤
      stripTransformDecayConstant f n := by
  let g := exponentiallyWeightedTest f a
  have hdegree : ((n : ℕ) : ℕ∞ω) ≤ (∞ : ℕ∞ω) :=
    WithTop.coe_le_coe.mpr (ENat.coe_lt_top n).le
  have hgcont : ContDiff ℝ (n : ℕ∞) (g : ℝ → ℂ) :=
    g.contDiff.of_le hdegree
  have hgint : ∀ j : ℕ, j ≤ (n : ℕ∞) →
      Integrable (iteratedDeriv j (g : ℝ → ℂ)) := by
    intro j hj
    have hjdegree : ((j : ℕ) : ℕ∞ω) ≤ (∞ : ℕ∞ω) :=
      WithTop.coe_le_coe.mpr (ENat.coe_lt_top j).le
    exact (g.contDiff.continuous_iteratedDeriv j hjdegree).integrable_of_hasCompactSupport
      (hasCompactSupport_iteratedDeriv g j)
  have hfour := congrFun
    (Real.fourier_iteratedDeriv hgcont hgint (show n ≤ (n : ℕ∞) by simp)) ξ
  have hto :
      (toSchwartz (exponentiallyWeightedTest f a) : ℝ → ℂ) =
        (g : ℝ → ℂ) := by rfl
  calc
    (2 * Real.pi * |ξ|) ^ n *
          ‖transform f ((a : ℂ) - 2 * Real.pi * ξ * I)‖ =
        ‖𝓕 (iteratedDeriv n (g : ℝ → ℂ)) ξ‖ := by
      rw [transform_vertical_eq_fourier, SchwartzMap.fourier_coe, hto, hfour]
      simp only [norm_smul, norm_pow, Complex.norm_mul,
        norm_real, Real.norm_eq_abs, abs_of_pos (Real.pi_pos), norm_I, mul_one,
        norm_ofNat]
    _ ≤ ∫ x : ℝ, ‖iteratedDeriv n (g : ℝ → ℂ) x‖ :=
      VectorFourier.norm_fourierIntegral_le_integral_norm
        𝐞 volume (innerₗ ℝ) _ ξ
    _ ≤ stripTransformDecayConstant f n := by
      exact integral_norm_iteratedDeriv_exponentiallyWeightedTest_le f n ha

theorem norm_transform_strip_pow_le_explicit
    (f : Test) (n : ℕ) {a : ℝ}
    (ha : a ∈ Set.Icc (-1 / 2 : ℝ) (1 / 2)) (ξ : ℝ) :
    |ξ| ^ n * ‖transform f ((a : ℂ) - 2 * Real.pi * ξ * I)‖ ≤
      stripTransformDecayConstant f n := by
  calc
    |ξ| ^ n * ‖transform f ((a : ℂ) - 2 * Real.pi * ξ * I)‖ ≤
        (2 * Real.pi * |ξ|) ^ n *
          ‖transform f ((a : ℂ) - 2 * Real.pi * ξ * I)‖ := by
      gcongr
      nlinarith [Real.pi_gt_three, abs_nonneg ξ]
    _ ≤ stripTransformDecayConstant f n :=
      norm_transform_strip_derivative_le f n ha ξ

theorem norm_transform_centered_strip_im_pow_le_explicit
    (f : Test) (n : ℕ) {z : ℂ}
    (hz : z.re ∈ Set.Icc (-1 / 2 : ℝ) (1 / 2)) :
    |z.im| ^ n * ‖transform f z‖ ≤ stripTransformDecayConstant f n := by
  let ξ : ℝ := -z.im / (2 * Real.pi)
  have h := norm_transform_strip_derivative_le f n hz ξ
  have hscale :
      2 * Real.pi * |ξ| = |z.im| := by
    dsimp [ξ]
    rw [abs_div, abs_neg, abs_of_pos (by positivity : 0 < 2 * Real.pi)]
    field_simp [Real.pi_ne_zero]
  have hξ : -(2 * Real.pi * ξ) = z.im := by
    dsimp [ξ]
    field_simp [Real.pi_ne_zero]
  have harg :
      (z.re : ℂ) -
          2 * Real.pi * (ξ : ℂ) * I = z := by
    apply Complex.ext
    · simp
    · simpa using hξ
  rwa [hscale, harg] at h

/-- Explicit condition-number-sensitive strip decay for the translated
Vandermonde interpolator.  All growth is isolated in
`translatedInterpolationWeight`. -/
theorem norm_transform_translatedInterpolatingTest_strip_pow_le
    (m n : ℕ) (base : Test) (a : ℝ)
    (points values : Fin m → ℂ) {z : ℂ}
    (hz : z.re ∈ Set.Icc (-1 / 2 : ℝ) (1 / 2)) :
    |z.im| ^ n *
        ‖transform (translatedInterpolatingTest m base a points values) z‖ ≤
      stripTransformDecayConstant base n *
        translatedInterpolationWeight m base a points values := by
  let c := translatedInterpolationCoefficients m base a points values
  let W := translatedInterpolationWeight m base a points values
  have hW : 0 ≤ W := by
    dsimp [W, translatedInterpolationWeight]
    exact Finset.sum_nonneg fun _ _ ↦
      mul_nonneg (norm_nonneg _) (Real.exp_pos _).le
  have hterm (j : Fin m) :
      ‖c j * (exp (z * (a * (j : ℝ))) * transform base z)‖ ≤
        (‖c j‖ * Real.exp (|a * (j : ℝ)| / 2)) *
          ‖transform base z‖ := by
    let b : ℝ := a * (j : ℝ)
    have harg :
        z * (a * (j : ℝ)) = z * (b : ℂ) := by
      congr 1
      dsimp [b]
      push_cast
      rfl
    rw [harg]
    have hre :
        (z * (b : ℂ)).re ≤ |a * (j : ℝ)| / 2 := by
      rw [mul_re]
      simp only [ofReal_re, ofReal_im, mul_zero, sub_zero]
      dsimp [b]
      calc
        z.re * (a * (j : ℝ)) ≤ |z.re * (a * (j : ℝ))| :=
          le_abs_self _
        _ = |z.re| * |a * (j : ℝ)| := abs_mul _ _
        _ ≤ (1 / 2 : ℝ) * |a * (j : ℝ)| := by
          gcongr
          rw [abs_le]
          constructor <;> linarith [hz.1, hz.2]
        _ = |a * (j : ℝ)| / 2 := by ring
    rw [norm_mul, norm_mul, Complex.norm_exp]
    have hexp := Real.exp_le_exp.mpr hre
    calc
      ‖c j‖ * (Real.exp (z * (b : ℂ)).re *
          ‖transform base z‖) ≤
          ‖c j‖ * (Real.exp (|a * (j : ℝ)| / 2) *
            ‖transform base z‖) := by
        gcongr
      _ = (‖c j‖ * Real.exp (|a * (j : ℝ)| / 2)) *
          ‖transform base z‖ := by ring
  have hsum :
      ‖transform (translatedInterpolatingTest m base a points values) z‖ ≤
        W * ‖transform base z‖ := by
    rw [transform_translatedInterpolatingTest_eq_sum]
    calc
      ‖∑ j, c j * (exp (z * (a * (j : ℝ))) * transform base z)‖ ≤
          ∑ j, ‖c j *
            (exp (z * (a * (j : ℝ))) * transform base z)‖ :=
        norm_sum_le _ _
      _ ≤ ∑ j, (‖c j‖ * Real.exp (|a * (j : ℝ)| / 2)) *
          ‖transform base z‖ :=
        Finset.sum_le_sum fun j _ ↦ hterm j
      _ = W * ‖transform base z‖ := by
        rw [← Finset.sum_mul]
        rfl
  have hbase :=
    norm_transform_centered_strip_im_pow_le_explicit base n hz
  calc
    |z.im| ^ n *
          ‖transform (translatedInterpolatingTest m base a points values) z‖ ≤
        |z.im| ^ n * (W * ‖transform base z‖) :=
      mul_le_mul_of_nonneg_left hsum (pow_nonneg (abs_nonneg _) _)
    _ = W * (|z.im| ^ n * ‖transform base z‖) := by ring
    _ ≤ W * stripTransformDecayConstant base n :=
      mul_le_mul_of_nonneg_left hbase hW
    _ = stripTransformDecayConstant base n * W := mul_comm _ _

/-- Two distinct points approaching each other on the critical line. -/
def nearCollisionSpectralPair (N : ℕ) : Fin 2 → ℂ :=
  ![0, I / (N + 1 : ℝ)]

theorem nearCollisionSpectralPair_injective (N : ℕ) :
    Function.Injective (nearCollisionSpectralPair N) := by
  have hden : (((N + 1 : ℝ) : ℂ)) ≠ 0 :=
    Complex.ofReal_ne_zero.mpr (by positivity)
  have hne : I / (((N + 1 : ℝ) : ℂ)) ≠ 0 :=
    div_ne_zero I_ne_zero hden
  intro i j hij
  fin_cases i <;> fin_cases j
  · rfl
  · exact False.elim <| hne <| by
      simpa [nearCollisionSpectralPair] using hij.symm
  · exact False.elim <| hne <| by
      simpa [nearCollisionSpectralPair] using hij
  · rfl

/-- Even for two distinct critical-line points, the translated Vandermonde
determinant can be arbitrarily ill-conditioned as the points coalesce.  This
quantitative upper bound shows why RvM counts and distinctness alone cannot
provide a uniform interpolation bound. -/
theorem norm_translatedBasisMatrix_nearCollision_det_le
    (base : Test) (a : ℝ) (N : ℕ) (ha : |a| ≤ (N + 1 : ℝ)) :
    ‖(translatedBasisMatrix 2 base a
      (nearCollisionSpectralPair N)).det‖ ≤
      (stripTransformDecayConstant base 0) ^ 2 *
        (2 * |a| / (N + 1 : ℝ)) := by
  let C := stripTransformDecayConstant base 0
  let z : ℂ := I / (N + 1 : ℝ)
  have hzre : z.re ∈ Set.Icc (-1 / 2 : ℝ) (1 / 2) := by
    dsimp [z]
    norm_num [div_re]
  have hzero :
      ‖transform base 0‖ ≤ C := by
    have h := norm_transform_centered_strip_im_pow_le_explicit
      base 0 (z := (0 : ℂ)) (by norm_num)
    simpa [C] using h
  have hz :
      ‖transform base z‖ ≤ C := by
    have h := norm_transform_centered_strip_im_pow_le_explicit
      base 0 hzre
    simpa [C] using h
  have hC : 0 ≤ C := (norm_nonneg (transform base 0)).trans hzero
  have hza_norm : ‖z * (a : ℂ)‖ = |a| / (N + 1 : ℝ) := by
    dsimp [z]
    rw [norm_mul, norm_div, norm_I]
    have hden :
        ‖(((N + 1 : ℝ) : ℂ))‖ = (N + 1 : ℝ) := by
      rw [Complex.norm_real, Real.norm_of_nonneg]
      positivity
    rw [hden]
    simp only [norm_real, Real.norm_eq_abs]
    rw [div_eq_mul_inv]
    ring
  have hza_le : ‖z * (a : ℂ)‖ ≤ 1 := by
    rw [hza_norm]
    exact (div_le_one (by positivity : (0 : ℝ) < N + 1)).mpr ha
  have hexp :
      ‖exp (z * (a : ℂ)) - 1‖ ≤ 2 * |a| / (N + 1 : ℝ) := by
    calc
      ‖exp (z * (a : ℂ)) - 1‖ ≤ 2 * ‖z * (a : ℂ)‖ :=
        Complex.norm_exp_sub_one_le hza_le
      _ = 2 * |a| / (N + 1 : ℝ) := by rw [hza_norm]; ring
  rw [show nearCollisionSpectralPair N = ![(0 : ℂ), z] by
    ext i; fin_cases i <;> rfl]
  rw [translatedBasisMatrix_finTwo_det]
  simp only [zero_mul, exp_zero]
  rw [norm_mul, norm_mul]
  have hprod :
      ‖transform base 0‖ * ‖transform base z‖ ≤ C ^ 2 := by
    nlinarith [hzero, hz, norm_nonneg (transform base 0),
      norm_nonneg (transform base z)]
  calc
    ‖transform base 0‖ * ‖transform base z‖ *
          ‖exp (z * (a : ℂ)) - 1‖ ≤
        C ^ 2 * ‖exp (z * (a : ℂ)) - 1‖ :=
      mul_le_mul_of_nonneg_right hprod (norm_nonneg _)
    _ ≤ C ^ 2 * (2 * |a| / (N + 1 : ℝ)) :=
      mul_le_mul_of_nonneg_left hexp (sq_nonneg C)

/-- Concrete admissibility condition for strip-uniform decay: one explicit
Schwartz seminorm bound for all exponential weights in the centered critical
strip. -/
def StripUniformSchwartzSeminormBound
    (f : Test) (k : ℕ) (C : ℝ) : Prop :=
  0 ≤ C ∧ ∀ a ∈ Set.Icc (-1 / 2 : ℝ) (1 / 2),
    (SchwartzMap.seminorm ℂ k 0)
      (𝓕 (toSchwartz (exponentiallyWeightedTest f a))) ≤ C

/-- The admissible seminorm condition gives a strip-uniform polynomial decay
estimate with the same explicit constant. -/
theorem norm_transform_strip_pow_le
    {f : Test} {k : ℕ} {C : ℝ}
    (hC : StripUniformSchwartzSeminormBound f k C)
    {a : ℝ} (ha : a ∈ Set.Icc (-1 / 2 : ℝ) (1 / 2)) (ξ : ℝ) :
    |ξ| ^ k * ‖transform f ((a : ℂ) - 2 * Real.pi * ξ * I)‖ ≤ C :=
  (norm_transform_vertical_pow_le f a ξ k).trans (hC.2 a ha)

theorem zero_stripUniformSchwartzSeminormBound (k : ℕ) :
    StripUniformSchwartzSeminormBound (0 : Test) k 0 := by
  constructor
  · exact le_rfl
  · intro a ha
    have hz : exponentiallyWeightedTest (0 : Test) a = 0 := by
      ext x
      simp [exponentiallyWeightedTest]
    have hsz : toSchwartz (exponentiallyWeightedTest (0 : Test) a) = 0 := by
      ext x
      simp [hz]
    rw [hsz]
    simp

/-- The multiplicative representative corresponding to a logarithmic test:
`g(x) = x⁻¹/² f(log x)` for `x > 0`, extended by zero.  This is the input to
Mathlib's `mellin`; proving the change-of-variables identity is kept as an
explicit bridge obligation below. -/
def multiplicativeRepresentative (f : Test) (x : ℝ) : ℂ :=
  if 0 < x then (Real.sqrt x : ℂ)⁻¹ * f (Real.log x) else 0

/-- The actual Mathlib Mellin expression in the source normalization. -/
def mathlibMellin (f : Test) (s : ℂ) : ℂ :=
  mellin (multiplicativeRepresentative f) s

/-- Exact normalization needed to identify Mathlib's one-sided Mellin
integral with the centred bilateral transform. -/
def MellinNormalizationObligation : Prop :=
  ∀ (f : Test) (z : ℂ),
    MellinConvergent (multiplicativeRepresentative f) (1 / 2 + z) ∧
      mathlibMellin f (1 / 2 + z) = transform f z

/-- Hermitian reflection on logarithmic test functions: `f⋆(t) = conj (f(-t))`. -/
def involution (f : Test) : Test where
  toFun t := star (f (-t))
  contDiff' :=
    Complex.conjCLE.contDiff.comp (f.contDiff.comp contDiff_neg)
  hasCompactSupport' :=
    (f.hasCompactSupport.comp_homeomorph (Homeomorph.neg ℝ)).comp_left
      (map_zero Complex.conjCLE)
  tsupport_subset' := by simp

@[simp]
theorem involution_apply (f : Test) (t : ℝ) :
    involution f t = star (f (-t)) :=
  rfl

@[simp]
theorem involution_involution (f : Test) : involution (involution f) = f := by
  ext t
  simp [involution]

@[simp]
theorem involution_zero : involution (0 : Test) = 0 := by
  ext t
  simp

theorem involution_add (f g : Test) :
    involution (f + g) = involution f + involution g := by
  ext t
  simp

theorem involution_smul (c : ℂ) (f : Test) :
    involution (c • f) = star c • involution f := by
  ext t
  simp

/-- Hermitian reflection at an arbitrary complex spectral parameter. -/
theorem transform_involution (f : Test) (z : ℂ) :
    transform (involution f) z =
      star (transform f (-star z)) := by
  rw [transform, transform]
  change _ = conj (∫ t : ℝ, f t * exp ((-conj z) * t))
  rw [← integral_conj, ← integral_neg_eq_self]
  apply integral_congr_ae
  filter_upwards with t
  simp only [involution_apply, map_mul, ← exp_conj, map_neg,
    conj_ofReal, neg_neg]
  congr 2
  simp [mul_comm]

/-- On the critical line, Hermitian reflection conjugates the centered
transform. -/
theorem transform_involution_imaginary (f : Test) (ξ : ℝ) :
    transform (involution f) (-2 * Real.pi * ξ * I) =
      star (transform f (-2 * Real.pi * ξ * I)) := by
  rw [transform, transform]
  change _ = conj (∫ t : ℝ, f t * exp ((-2 * Real.pi * ξ * I) * t))
  rw [← integral_conj, ← integral_neg_eq_self]
  apply integral_congr_ae
  filter_upwards with t
  simp only [involution_apply, map_mul, ← exp_conj, map_neg, map_ofNat,
    conj_ofReal, conj_I, neg_mul, neg_neg]
  congr 2
  push_cast
  ring_nf

/-- Additive convolution on the logarithmic line.  It corresponds to
multiplicative convolution with Haar measure `dx/x` on `(0, ∞)`. -/
def convolution (f g : Test) : Test where
  toFun := MeasureTheory.convolution f g (ContinuousLinearMap.mul ℝ ℂ) volume
  contDiff' :=
    g.hasCompactSupport.contDiff_convolution_right
      (ContinuousLinearMap.mul ℝ ℂ) f.continuous.locallyIntegrable g.contDiff
  hasCompactSupport' :=
    f.hasCompactSupport.convolution (ContinuousLinearMap.mul ℝ ℂ) g.hasCompactSupport
  tsupport_subset' := by simp

@[simp]
theorem convolution_apply (f g : Test) (x : ℝ) :
    convolution f g x = ∫ t : ℝ, f t * g (x - t) := by
  rfl

/-- The bilateral Laplace transform converts additive convolution into
multiplication at every complex spectral parameter. -/
theorem transform_convolution (f g : Test) (z : ℂ) :
    transform (convolution f g) z = transform f z * transform g z := by
  let F : ℝ → ℂ := fun t ↦ f t * exp (z * t)
  let G : ℝ → ℂ := fun t ↦ g t * exp (z * t)
  have hF : Integrable F := by
    apply Continuous.integrable_of_hasCompactSupport
    · exact f.continuous.mul <| Complex.continuous_exp.comp
        (continuous_const.mul (Complex.continuous_ofReal.comp continuous_id))
    · exact f.hasCompactSupport.mul_right
  have hG : Integrable G := by
    apply Continuous.integrable_of_hasCompactSupport
    · exact g.continuous.mul <| Complex.continuous_exp.comp
        (continuous_const.mul (Complex.continuous_ofReal.comp continuous_id))
    · exact g.hasCompactSupport.mul_right
  rw [transform, transform, transform]
  change (∫ x : ℝ, convolution f g x * exp (z * x)) =
    (∫ t : ℝ, F t) * ∫ t : ℝ, G t
  have hprod :
      (∫ t : ℝ, F t) * (∫ t : ℝ, G t) =
        ∫ x : ℝ, MeasureTheory.convolution F G
          (ContinuousLinearMap.mul ℝ ℂ) volume x := by
    symm
    simpa using MeasureTheory.integral_convolution
      (L := ContinuousLinearMap.mul ℝ ℂ) hF hG
  rw [hprod]
  apply integral_congr_ae
  filter_upwards with x
  change (∫ t : ℝ, f t * g (x - t)) * exp (z * x) =
    ∫ t : ℝ, F t * G (x - t)
  rw [← integral_mul_const]
  apply integral_congr_ae
  filter_upwards with t
  dsimp [F, G]
  have hexp :
      z * (t : ℂ) + z * ((x - t : ℝ) : ℂ) = z * (x : ℂ) := by
    push_cast
    ring
  have hsplit :
      exp (z * (t : ℂ)) * exp (z * ((x - t : ℝ) : ℂ)) =
        exp (z * (x : ℂ)) := by
    rw [← exp_add, hexp]
  rw [← hsplit]
  ring

/-- Exact off-critical convolution-square factorization. -/
theorem transform_convolution_involution (f : Test) (z : ℂ) :
    transform (convolution f (involution f)) z =
      transform f z * star (transform f (-star z)) := by
  rw [transform_convolution, transform_involution]

/-- The matrix-free local separator has the exact negative convolution-square
value at the selected off-line target. -/
theorem transform_localPairSeparator_square_target
    (a : ℝ) (z : ℂ) (nearby : List ℂ)
    (hz : LocalSeparatorGap a z (-star z :: nearby))
    (hpartner : LocalSeparatorGap a (-star z) (z :: nearby)) :
    transform
        (convolution (localPairSeparator a z nearby)
          (involution (localPairSeparator a z nearby))) z = -1 := by
  rw [transform_convolution_involution,
    transform_localPairSeparator_target _ _ _ hz,
    transform_localPairSeparator_partner _ _ _ hpartner]
  norm_num

theorem transform_localPairSeparator_square_partner
    (a : ℝ) (z : ℂ) (nearby : List ℂ)
    (hz : LocalSeparatorGap a z (-star z :: nearby))
    (hpartner : LocalSeparatorGap a (-star z) (z :: nearby)) :
    transform
        (convolution (localPairSeparator a z nearby)
          (involution (localPairSeparator a z nearby))) (-star z) = -1 := by
  rw [transform_convolution_involution,
    transform_localPairSeparator_partner _ _ _ hpartner]
  rw [show -star (-star z) = z by simp]
  rw [transform_localPairSeparator_target _ _ _ hz]
  norm_num

theorem transform_localPairSeparator_square_nearby_eq_zero
    (a : ℝ) (z : ℂ) (nearby : List ℂ) {q : ℂ} (hq : q ∈ nearby) :
    transform
        (convolution (localPairSeparator a z nearby)
          (involution (localPairSeparator a z nearby))) q = 0 := by
  rw [transform_convolution_involution,
    transform_localPairSeparator_nearby_eq_zero _ _ _ hq]
  ring

/-- Strip-uniform rapid decay is preserved quantitatively by convolution
squares, with doubled power and squared decay constant. -/
theorem norm_transform_convolution_involution_strip_pow_le
    (f : Test) (n : ℕ) {z : ℂ}
    (hz : z.re ∈ Set.Icc (-1 / 2 : ℝ) (1 / 2)) :
    |z.im| ^ (2 * n) *
        ‖transform (convolution f (involution f)) z‖ ≤
      (stripTransformDecayConstant f n) ^ 2 := by
  have hzpartner :
      (-star z).re ∈ Set.Icc (-1 / 2 : ℝ) (1 / 2) := by
    have hre : (-star z).re = -z.re := by simp
    rw [hre]
    constructor <;> linarith [hz.1, hz.2]
  have h₁ := norm_transform_centered_strip_im_pow_le_explicit f n hz
  have h₂ :=
    norm_transform_centered_strip_im_pow_le_explicit f n hzpartner
  have him : (-star z).im = z.im := by simp
  rw [him] at h₂
  have hC : 0 ≤ stripTransformDecayConstant f n :=
    (mul_nonneg (pow_nonneg (abs_nonneg z.im) n)
      (norm_nonneg (transform f z))).trans h₁
  have hmul :
      (|z.im| ^ n * ‖transform f z‖) *
          (|z.im| ^ n * ‖transform f (-star z)‖) ≤
        stripTransformDecayConstant f n *
          stripTransformDecayConstant f n :=
    mul_le_mul h₁ h₂
      (mul_nonneg (pow_nonneg (abs_nonneg z.im) n)
        (norm_nonneg (transform f (-star z)))) hC
  rw [transform_convolution_involution, norm_mul, norm_star]
  calc
    |z.im| ^ (2 * n) *
          (‖transform f z‖ * ‖transform f (-star z)‖) =
        (|z.im| ^ n * ‖transform f z‖) *
          (|z.im| ^ n * ‖transform f (-star z)‖) := by
      rw [show 2 * n = n + n by omega, pow_add]
      ring
    _ ≤ _ := hmul
    _ = (stripTransformDecayConstant f n) ^ 2 := by ring

/-- Distant spectral contributions of the local separator retain arbitrary
strip-uniform polynomial decay. -/
theorem norm_transform_localPairSeparator_square_strip_pow_le
    (a : ℝ) (target : ℂ) (nearby : List ℂ) (n : ℕ) {z : ℂ}
    (hz : z.re ∈ Set.Icc (-1 / 2 : ℝ) (1 / 2)) :
    |z.im| ^ (2 * n) *
        ‖transform
          (convolution (localPairSeparator a target nearby)
            (involution (localPairSeparator a target nearby))) z‖ ≤
      (stripTransformDecayConstant
        (localPairSeparator a target nearby) n) ^ 2 :=
  norm_transform_convolution_involution_strip_pow_le
    (localPairSeparator a target nearby) n hz

/-- Fully explicit condition-number-sensitive decay bound for a translated
interpolator's convolution square. -/
theorem norm_transform_translatedInterpolatorSquare_strip_pow_le
    (m n : ℕ) (base : Test) (a : ℝ)
    (points values : Fin m → ℂ) {z : ℂ}
    (hz : z.re ∈ Set.Icc (-1 / 2 : ℝ) (1 / 2)) :
    |z.im| ^ (2 * n) *
        ‖transform
          (convolution
            (translatedInterpolatingTest m base a points values)
            (involution
              (translatedInterpolatingTest m base a points values))) z‖ ≤
      (stripTransformDecayConstant base n *
        translatedInterpolationWeight m base a points values) ^ 2 := by
  let f := translatedInterpolatingTest m base a points values
  let C := stripTransformDecayConstant base n *
    translatedInterpolationWeight m base a points values
  have hzpartner :
      (-star z).re ∈ Set.Icc (-1 / 2 : ℝ) (1 / 2) := by
    have hre : (-star z).re = -z.re := by simp
    rw [hre]
    constructor <;> linarith [hz.1, hz.2]
  have h₁ :
      |z.im| ^ n * ‖transform f z‖ ≤ C :=
    norm_transform_translatedInterpolatingTest_strip_pow_le
      m n base a points values hz
  have h₂ :
      |z.im| ^ n * ‖transform f (-star z)‖ ≤ C := by
    have h := norm_transform_translatedInterpolatingTest_strip_pow_le
      m n base a points values hzpartner
    simpa using h
  have hC : 0 ≤ C :=
    (mul_nonneg (pow_nonneg (abs_nonneg z.im) n)
      (norm_nonneg (transform f z))).trans h₁
  have hmul :
      (|z.im| ^ n * ‖transform f z‖) *
          (|z.im| ^ n * ‖transform f (-star z)‖) ≤ C * C :=
    mul_le_mul h₁ h₂
      (mul_nonneg (pow_nonneg (abs_nonneg z.im) n)
        (norm_nonneg (transform f (-star z)))) hC
  change |z.im| ^ (2 * n) *
      ‖transform (convolution f (involution f)) z‖ ≤ C ^ 2
  rw [transform_convolution_involution, norm_mul, norm_star]
  calc
    |z.im| ^ (2 * n) *
          (‖transform f z‖ * ‖transform f (-star z)‖) =
        (|z.im| ^ n * ‖transform f z‖) *
          (|z.im| ^ n * ‖transform f (-star z)‖) := by
      rw [show 2 * n = n + n by omega, pow_add]
      ring
    _ ≤ C * C := hmul
    _ = C ^ 2 := by ring

/-- Finite interpolation closed under functional-equation reflection passes
exactly to convolution-square transform values. -/
theorem exists_convolutionSquare_transform_eq
    (n : ℕ) (points values : Fin n → ℂ) (partner : Fin n → Fin n)
    (hpoints : Function.Injective points)
    (hpartner : ∀ i, points (partner i) = -star (points i)) :
    ∃ f : Test, ∀ i,
      transform (convolution f (involution f)) (points i) =
        values i * star (values (partner i)) := by
  rcases exists_translatedInterpolatingTest n points values hpoints with
    ⟨base, hbase⟩
  let f : Test :=
    translatedInterpolatingTest n base
      (collisionFreeTranslationStep points) points values
  refine ⟨f, fun i ↦ ?_⟩
  rw [transform_convolution_involution, hbase i, ← hpartner i,
    hbase (partner i)]

/-- Every non-fixed symmetry pair admits a genuine convolution square whose
transform is `-1` at both members of the pair. -/
theorem exists_convolutionSquare_negative_on_offCritical_pair
    (z : ℂ) (hz : z ≠ -star z) :
    ∃ f : Test,
      transform (convolution f (involution f)) z = -1 ∧
        transform (convolution f (involution f)) (-star z) = -1 := by
  let points : Fin 2 → ℂ := ![z, -star z]
  let values : Fin 2 → ℂ := ![1, -1]
  let partner : Fin 2 → Fin 2 := ![1, 0]
  have hpoints : Function.Injective points := by
    intro i j hij
    fin_cases i <;> fin_cases j <;> simp_all [points]
  have hpartner : ∀ i, points (partner i) = -star (points i) := by
    intro i
    fin_cases i <;> simp [points, partner]
  rcases exists_convolutionSquare_transform_eq
      2 points values partner hpoints hpartner with ⟨f, hf⟩
  refine ⟨f, ?_, ?_⟩
  · simpa [points, values, partner] using hf 0
  · simpa [points, values, partner] using hf 1

theorem convolution_comm (f g : Test) :
    convolution f g = convolution g f := by
  ext x
  change MeasureTheory.convolution f g (ContinuousLinearMap.mul ℝ ℂ) volume x =
    MeasureTheory.convolution g f (ContinuousLinearMap.mul ℝ ℂ) volume x
  rw [MeasureTheory.convolution_eq_swap]
  apply integral_congr_ae
  filter_upwards with t
  exact mul_comm _ _

theorem iteratedDeriv_convolution_right (f g : Test) (n : ℕ) :
    iteratedDeriv n (convolution f g) =
      MeasureTheory.convolution f (iteratedDeriv n g)
        (ContinuousLinearMap.mul ℝ ℂ) volume := by
  induction n with
  | zero =>
      ext x
      rfl
  | succ n ih =>
      rw [iteratedDeriv_succ, ih]
      ext x
      have hg : ContDiff ℝ (n + 1 : ℕ) (g : ℝ → ℂ) :=
        g.contDiff.of_le
          (WithTop.coe_le_coe.mpr (ENat.coe_lt_top (n + 1)).le)
      have hd : ContDiff ℝ 1 (iteratedDeriv n (g : ℝ → ℂ)) :=
        (contDiff_nat_succ_iff_contDiff_one_iteratedDeriv.mp hg).2
      simpa only [iteratedDeriv_succ] using
        ((hasCompactSupport_iteratedDeriv g n).hasDerivAt_convolution_right
          (μ := volume)
          (ContinuousLinearMap.mul ℝ ℂ) f.continuous.locallyIntegrable
          hd x).deriv

theorem integral_norm_supported_le_measure_mul_seminorm
    (K : Compacts ℝ) (f : 𝓓^{⊤}_{K}(ℝ, ℂ)) :
    (∫ x : ℝ, ‖f x‖) ≤ volume.real (K : Set ℝ) * N[ℝ]_{K, 0} f := by
  have heq :
      (∫ x : ℝ, ‖f x‖) = ∫ x : ℝ in (K : Set ℝ), ‖f x‖ := by
    symm
    apply setIntegral_eq_integral_of_forall_compl_eq_zero
    intro x hx
    simp [f.zero_on_compl hx]
  rw [heq, ← Real.norm_of_nonneg (setIntegral_nonneg
    K.isCompact.measurableSet fun _ _ ↦ norm_nonneg _)]
  calc
    |∫ x : ℝ in (K : Set ℝ), ‖f x‖| ≤
        N[ℝ]_{K, 0} f * volume.real (K : Set ℝ) :=
      norm_setIntegral_le_of_norm_le_const
        (μ := volume) (f := fun x : ℝ ↦ ‖f x‖)
        K.isCompact.measure_lt_top fun x _ ↦ by
          simpa [norm_iteratedFDeriv_eq_norm_iteratedDeriv] using
            (ContDiffMapSupportedIn.norm_iteratedFDeriv_apply_le_seminorm_top
              (𝕜 := ℝ) (f := f) (x := x) (i := 0))
    _ = volume.real (K : Set ℝ) * N[ℝ]_{K, 0} f := mul_comm _ _

theorem norm_iteratedDeriv_supported_convolution_le
    (K L : Compacts ℝ) (f : 𝓓^{⊤}_{K}(ℝ, ℂ)) (g : 𝓓^{⊤}_{L}(ℝ, ℂ))
    (n : ℕ) (x : ℝ) :
    ‖iteratedDeriv n
        (convolution
          (TestFunction.ofSupportedIn (Set.subset_univ _) f)
          (TestFunction.ofSupportedIn (Set.subset_univ _) g)) x‖ ≤
      volume.real (K : Set ℝ) * N[ℝ]_{K, 0} f * N[ℝ]_{L, n} g := by
  let F : Test := TestFunction.ofSupportedIn (Set.subset_univ _) f
  let G : Test := TestFunction.ofSupportedIn (Set.subset_univ _) g
  have hgcont : Continuous (iteratedDeriv n (g : ℝ → ℂ)) :=
    (g.contDiff.continuous_iteratedDeriv n
      (WithTop.coe_le_coe.mpr (ENat.coe_lt_top n).le))
  have hint :
      Integrable fun t : ℝ ↦ ‖f t * iteratedDeriv n (g : ℝ → ℂ) (x - t)‖ := by
    apply Continuous.integrable_of_hasCompactSupport
    · exact (f.continuous.mul
        (hgcont.comp (continuous_const.sub continuous_id))).norm
    · have hs : HasCompactSupport
          (fun t : ℝ ↦ f t * iteratedDeriv n (g : ℝ → ℂ) (x - t)) :=
        f.hasCompactSupport.mul_right
      exact hs.norm
  have hmajor :
      Integrable fun t : ℝ ↦ ‖f t‖ * N[ℝ]_{L, n} g :=
    f.continuous.norm.integrable_of_hasCompactSupport f.hasCompactSupport.norm
      |>.mul_const _
  have hderiv (y : ℝ) :
      ‖iteratedDeriv n (g : ℝ → ℂ) y‖ ≤ N[ℝ]_{L, n} g := by
    rw [← norm_iteratedFDeriv_eq_norm_iteratedDeriv]
    exact ContDiffMapSupportedIn.norm_iteratedFDeriv_apply_le_seminorm_top ℝ
  rw [show
    (convolution F G : ℝ → ℂ) =
      convolution
        (TestFunction.ofSupportedIn (Set.subset_univ _) f)
        (TestFunction.ofSupportedIn (Set.subset_univ _) g) by rfl]
  rw [iteratedDeriv_convolution_right F G n]
  change ‖∫ t : ℝ, f t * iteratedDeriv n (g : ℝ → ℂ) (x - t)‖ ≤ _
  calc
    ‖∫ t : ℝ, f t * iteratedDeriv n (g : ℝ → ℂ) (x - t)‖ ≤
        ∫ t : ℝ, ‖f t * iteratedDeriv n (g : ℝ → ℂ) (x - t)‖ :=
      norm_integral_le_integral_norm _
    _ ≤ ∫ t : ℝ, ‖f t‖ * N[ℝ]_{L, n} g := by
      apply integral_mono hint hmajor
      intro t
      dsimp only
      rw [norm_mul]
      exact mul_le_mul_of_nonneg_left (hderiv (x - t)) (norm_nonneg _)
    _ = N[ℝ]_{L, n} g * ∫ t : ℝ, ‖f t‖ := by
      rw [← integral_const_mul]
      apply integral_congr_ae
      filter_upwards with t
      ring
    _ ≤ N[ℝ]_{L, n} g *
        (volume.real (K : Set ℝ) * N[ℝ]_{K, 0} f) := by
      gcongr
      exact integral_norm_supported_le_measure_mul_seminorm K f
    _ = volume.real (K : Set ℝ) * N[ℝ]_{K, 0} f * N[ℝ]_{L, n} g := by
      ring

/-- Minkowski sum of two compact supports. -/
def convolutionCompact (K L : Compacts ℝ) : Compacts ℝ :=
  ⟨(K : Set ℝ) + (L : Set ℝ), K.isCompact.add L.isCompact⟩

/-- Fixed-stage convolution with its sharp support target bundled in
`𝓓_{K+L}`. -/
noncomputable def supportedConvolution
    (K L : Compacts ℝ) :
    𝓓^{⊤}_{K}(ℝ, ℂ) × 𝓓^{⊤}_{L}(ℝ, ℂ) →
      𝓓^{⊤}_{convolutionCompact K L}(ℝ, ℂ) :=
  fun p ↦ ContDiffMapSupportedIn.of_support_subset
    ((p.1.hasCompactSupport.contDiff_convolution_left
      (ContinuousLinearMap.mul ℝ ℂ) p.1.contDiff
      p.2.continuous.locallyIntegrable))
    ((MeasureTheory.support_convolution_subset
      (μ := volume) (ContinuousLinearMap.mul ℝ ℂ)).trans
        (Set.add_subset_add p.1.support_subset p.2.support_subset))

@[simp]
theorem supportedConvolution_apply
    (K L : Compacts ℝ)
    (p : 𝓓^{⊤}_{K}(ℝ, ℂ) × 𝓓^{⊤}_{L}(ℝ, ℂ)) (x : ℝ) :
    supportedConvolution K L p x =
      MeasureTheory.convolution p.1 p.2
        (ContinuousLinearMap.mul ℝ ℂ) volume x := rfl

/-- The pointwise estimate upgrades to every defining seminorm of the exact
fixed support target. -/
theorem supportedConvolution_seminorm_le
    (K L : Compacts ℝ)
    (p : 𝓓^{⊤}_{K}(ℝ, ℂ) × 𝓓^{⊤}_{L}(ℝ, ℂ)) (n : ℕ) :
    N[ℝ]_{convolutionCompact K L, n} (supportedConvolution K L p) ≤
      volume.real (K : Set ℝ) * N[ℝ]_{K, 0} p.1 * N[ℝ]_{L, n} p.2 := by
  apply (ContDiffMapSupportedIn.seminorm_top_le_iff
    (𝕜 := ℝ) (K := convolutionCompact K L) (by positivity)
    n (supportedConvolution K L p)).2
  intro x hx
  rw [norm_iteratedFDeriv_eq_norm_iteratedDeriv]
  have hfun :
      (supportedConvolution K L p : ℝ → ℂ) =
        (convolution
          (TestFunction.ofSupportedIn (Set.subset_univ _) p.1)
          (TestFunction.ofSupportedIn (Set.subset_univ _) p.2) : ℝ → ℂ) := by
    ext y
    rfl
  rw [hfun]
  exact norm_iteratedDeriv_supported_convolution_le K L p.1 p.2 n x

/-- On a fixed compact-support stage, von Neumann boundedness is exactly
uniform boundedness for every defining derivative seminorm. -/
theorem supportedIn_isVonNBounded_iff_seminorm_bounded
    (K : Compacts ℝ) (B : Set 𝓓^{⊤}_{K}(ℝ, ℂ)) :
    Bornology.IsVonNBounded ℝ B ↔
      ∀ n : ℕ, ∃ r > 0, ∀ f ∈ B, N[ℝ]_{K, n} f < r :=
  WithSeminorms.isVonNBounded_iff_seminorm_bounded
    (ContDiffMapSupportedIn.withSeminorms ℝ ℝ ℂ ⊤ K)

/-- Uniform left-variable estimate on a bounded subset of a fixed support
stage. This is the concrete hypocontinuity estimate used below. -/
theorem supportedConvolution_uniform_left_of_isVonNBounded
    (K L : Compacts ℝ) (B : Set 𝓓^{⊤}_{K}(ℝ, ℂ))
    (hB : Bornology.IsVonNBounded ℝ B) (n : ℕ) :
    ∃ C ≥ 0, ∀ f ∈ B, ∀ g : 𝓓^{⊤}_{L}(ℝ, ℂ),
      N[ℝ]_{convolutionCompact K L, n} (supportedConvolution K L (f, g)) ≤
        C * N[ℝ]_{L, n} g := by
  rcases (supportedIn_isVonNBounded_iff_seminorm_bounded K B).mp hB 0 with
    ⟨r, hr, hBr⟩
  refine ⟨volume.real (K : Set ℝ) * r, mul_nonneg measureReal_nonneg hr.le,
    fun f hf g ↦ ?_⟩
  calc
    N[ℝ]_{convolutionCompact K L, n} (supportedConvolution K L (f, g)) ≤
        volume.real (K : Set ℝ) * N[ℝ]_{K, 0} f * N[ℝ]_{L, n} g :=
      supportedConvolution_seminorm_le K L (f, g) n
    _ ≤ (volume.real (K : Set ℝ) * r) * N[ℝ]_{L, n} g := by
      gcongr
      exact (hBr f hf).le

/-- Uniform right-variable estimate on a bounded subset of a fixed support
stage. Together with the preceding theorem, this is fixed-stage
hypocontinuity of convolution. -/
theorem supportedConvolution_uniform_right_of_isVonNBounded
    (K L : Compacts ℝ) (B : Set 𝓓^{⊤}_{L}(ℝ, ℂ))
    (hB : Bornology.IsVonNBounded ℝ B) (n : ℕ) :
    ∃ C ≥ 0, ∀ g ∈ B, ∀ f : 𝓓^{⊤}_{K}(ℝ, ℂ),
      N[ℝ]_{convolutionCompact K L, n} (supportedConvolution K L (f, g)) ≤
        C * N[ℝ]_{K, 0} f := by
  rcases (supportedIn_isVonNBounded_iff_seminorm_bounded L B).mp hB n with
    ⟨r, hr, hBr⟩
  refine ⟨volume.real (K : Set ℝ) * r, mul_nonneg measureReal_nonneg hr.le,
    fun g hg f ↦ ?_⟩
  calc
    N[ℝ]_{convolutionCompact K L, n} (supportedConvolution K L (f, g)) ≤
        volume.real (K : Set ℝ) * N[ℝ]_{K, 0} f * N[ℝ]_{L, n} g :=
      supportedConvolution_seminorm_le K L (f, g) n
    _ ≤ volume.real (K : Set ℝ) * N[ℝ]_{K, 0} f * r := by
      gcongr
      exact (hBr g hg).le
    _ = (volume.real (K : Set ℝ) * r) * N[ℝ]_{K, 0} f := by
      ring

/-- A subset of the LF test-function space is stage-bounded if it is carried
by a von Neumann bounded subset of one fixed compact-support stage. -/
def TestFunction.IsStageVonNBounded (B : Set Test) : Prop :=
  ∃ K : Compacts ℝ, ∃ S : Set 𝓓^{⊤}_{K}(ℝ, ℂ),
    Bornology.IsVonNBounded ℝ S ∧
      B ⊆ TestFunction.ofSupportedIn (Set.subset_univ _) '' S

/-- Every stage-bounded family is von Neumann bounded in the LF topology.
The converse is the regularity theorem for this strict LF limit; that theorem
is not presently supplied by Mathlib's test-function API. -/
theorem TestFunction.IsStageVonNBounded.isVonNBounded
    {B : Set Test} (hB : TestFunction.IsStageVonNBounded B) :
    Bornology.IsVonNBounded ℝ B := by
  rcases hB with ⟨K, S, hS, hsub⟩
  exact Bornology.IsVonNBounded.subset hsub
    (hS.image (TestFunction.ofSupportedInCLM ℝ (Set.subset_univ _)))

/-- Every individual test is bounded in its own compact-support stage. -/
theorem TestFunction.isStageVonNBounded_singleton (f : Test) :
    TestFunction.IsStageVonNBounded ({f} : Set Test) := by
  let K : Compacts ℝ := ⟨tsupport f, f.hasCompactSupport⟩
  let fK : 𝓓^{⊤}_{K}(ℝ, ℂ) :=
    ContDiffMapSupportedIn.of_support_subset f.contDiff subset_closure
  refine ⟨K, {fK}, Bornology.isVonNBounded_singleton fK, ?_⟩
  intro g hg
  rw [Set.mem_singleton_iff] at hg
  subst g
  refine ⟨fK, Set.mem_singleton fK, ?_⟩
  ext x
  rfl

/-- Pointwise convolution image of two test-function families. -/
def convolutionSet (B C : Set Test) : Set Test :=
  (fun p : Test × Test ↦ convolution p.1 p.2) '' (B ×ˢ C)

/-- The unconditional bornological substitute available from fixed-stage
seminorm estimates: convolution maps two stage-bounded families into one
stage-bounded family. -/
theorem TestFunction.IsStageVonNBounded.convolutionSet
    {B C : Set Test} (hB : TestFunction.IsStageVonNBounded B)
    (hC : TestFunction.IsStageVonNBounded C) :
    TestFunction.IsStageVonNBounded (convolutionSet B C) := by
  classical
  rcases hB with ⟨K, S, hS, hBS⟩
  rcases hC with ⟨L, T, hT, hCT⟩
  let U : Set 𝓓^{⊤}_{convolutionCompact K L}(ℝ, ℂ) :=
    supportedConvolution K L '' (S ×ˢ T)
  have hU : Bornology.IsVonNBounded ℝ U := by
    rw [supportedIn_isVonNBounded_iff_seminorm_bounded]
    intro n
    rcases (supportedIn_isVonNBounded_iff_seminorm_bounded K S).mp hS 0 with
      ⟨r, hr, hSr⟩
    rcases (supportedIn_isVonNBounded_iff_seminorm_bounded L T).mp hT n with
      ⟨q, hq, hTq⟩
    refine ⟨volume.real (K : Set ℝ) * r * q + 1, by positivity, ?_⟩
    intro u hu
    rcases hu with ⟨p, hp, rfl⟩
    calc
      N[ℝ]_{convolutionCompact K L, n} (supportedConvolution K L p) ≤
          volume.real (K : Set ℝ) * N[ℝ]_{K, 0} p.1 *
            N[ℝ]_{L, n} p.2 :=
        supportedConvolution_seminorm_le K L p n
      _ ≤ volume.real (K : Set ℝ) * r * q := by
        gcongr
        · exact (hSr p.1 hp.1).le
        · exact (hTq p.2 hp.2).le
      _ < volume.real (K : Set ℝ) * r * q + 1 := by linarith
  refine ⟨convolutionCompact K L, U, hU, ?_⟩
  intro h hh
  rcases hh with ⟨p, hp, rfl⟩
  rcases hBS hp.1 with ⟨f, hf, hpf⟩
  rcases hCT hp.2 with ⟨g, hg, hpg⟩
  change convolution p.1 p.2 ∈
    TestFunction.ofSupportedIn (Set.subset_univ _) '' U
  rw [← hpf, ← hpg]
  refine ⟨supportedConvolution K L (f, g), ⟨(f, g), ⟨hf, hg⟩, rfl⟩, ?_⟩
  ext x
  rfl

/-- In particular, every convolution square used by Weil positivity is an
element of a concrete bounded compact-support stage; strict-LF regularity is
not needed for individual-test admissibility. -/
theorem convolutionSquare_isStageVonNBounded (f : Test) :
    TestFunction.IsStageVonNBounded
      ({convolution f (involution f)} : Set Test) :=
  TestFunction.isStageVonNBounded_singleton _

/-- The finite-difference separator square is unconditionally admissible in
one concrete compact-support stage; global strict-LF regularity is irrelevant
for this individual Weil test. -/
theorem localPairSeparatorSquare_isStageVonNBounded
    (a : ℝ) (z : ℂ) (nearby : List ℂ) :
    TestFunction.IsStageVonNBounded
      ({convolution (localPairSeparator a z nearby)
        (involution (localPairSeparator a z nearby))} : Set Test) :=
  convolutionSquare_isStageVonNBounded _

/-- Exact regularity statement missing from the pinned strict-LF API: every
LF-bounded family is already bounded in one compact-support stage. -/
def TestFunction.StrictLFBoundedRegularity : Prop :=
  ∀ B : Set Test, Bornology.IsVonNBounded ℝ B →
    TestFunction.IsStageVonNBounded B

theorem TestFunction.strictLFBoundedRegularity_iff :
    TestFunction.StrictLFBoundedRegularity ↔
      ∀ B : Set Test,
        Bornology.IsVonNBounded ℝ B ↔ TestFunction.IsStageVonNBounded B := by
  constructor
  · intro h B
    exact ⟨h B, TestFunction.IsStageVonNBounded.isVonNBounded⟩
  · intro h B hB
    exact (h B).mp hB

/-- Under the missing strict-LF regularity theorem, the preceding unconditional
stage statement upgrades to all von Neumann bounded families. -/
theorem convolutionSet_isVonNBounded_of_strictLF
    (hregular : TestFunction.StrictLFBoundedRegularity)
    {B C : Set Test} (hB : Bornology.IsVonNBounded ℝ B)
    (hC : Bornology.IsVonNBounded ℝ C) :
    Bornology.IsVonNBounded ℝ (convolutionSet B C) :=
  ((hregular B hB).convolutionSet (hregular C hC)).isVonNBounded

theorem convolutionCompact_mono
    {K K' L L' : Compacts ℝ} (hK : K ≤ K') (hL : L ≤ L') :
    convolutionCompact K L ≤ convolutionCompact K' L' :=
  Set.add_subset_add hK hL

/-- Fixed-stage convolution commutes with all compact-support inclusions.
This is the exact inductive-system compatibility required before any LF
bilinear lifting theorem can be applied. -/
theorem supportedConvolution_mono
    {K K' L L' : Compacts ℝ} (hK : K ≤ K') (hL : L ≤ L')
    (f : 𝓓^{⊤}_{K}(ℝ, ℂ)) (g : 𝓓^{⊤}_{L}(ℝ, ℂ)) :
    ContDiffMapSupportedIn.monoCLM ℝ
        (supportedConvolution K L (f, g)) =
      supportedConvolution K' L'
        (ContDiffMapSupportedIn.monoCLM ℝ f,
          ContDiffMapSupportedIn.monoCLM ℝ g) := by
  ext x
  have hKL := convolutionCompact_mono hK hL
  simp only [ContDiffMapSupportedIn.monoCLM_apply, le_refl, and_self,
    hKL, if_true]
  rw [supportedConvolution_apply, supportedConvolution_apply]
  have hf :
      ((ContDiffMapSupportedIn.monoCLM ℝ f :
        𝓓^{⊤}_{K'}(ℝ, ℂ)) : ℝ → ℂ) = f := by
    simpa [hK] using
      (ContDiffMapSupportedIn.monoCLM_apply
        (𝕜 := ℝ) (n₁ := ⊤) (n₂ := ⊤) (K₁ := K) (K₂ := K') f)
  have hg :
      ((ContDiffMapSupportedIn.monoCLM ℝ g :
        𝓓^{⊤}_{L'}(ℝ, ℂ)) : ℝ → ℂ) = g := by
    simpa [hL] using
      (ContDiffMapSupportedIn.monoCLM_apply
        (𝕜 := ℝ) (n₁ := ⊤) (n₂ := ⊤) (K₁ := L) (K₂ := L') g)
  change MeasureTheory.convolution f g
      (ContinuousLinearMap.mul ℝ ℂ) volume x =
    MeasureTheory.convolution
      (ContDiffMapSupportedIn.monoCLM ℝ f)
      (ContDiffMapSupportedIn.monoCLM ℝ g)
      (ContinuousLinearMap.mul ℝ ℂ) volume x
  rw [hf, hg]

/-- A candidate explicit-formula distribution.  The actual zeta distribution
is not constructed here; this structure records only its typed domain. -/
structure Distribution where
  /-- Value of the distribution on a test function. -/
  eval : Test → ℂ
  /-- The zeta distribution is complex-linear on its test space. -/
  map_add : ∀ f g, eval (f + g) = eval f + eval g
  /-- The zeta distribution is complex-linear on its test space. -/
  map_smul : ∀ (c : ℂ) f, eval (c • f) = c * eval f

/-- The algebraic linear map underlying a candidate distribution. -/
def Distribution.toLinearMap (W : Distribution) : Test →ₗ[ℂ] ℂ where
  toFun := W.eval
  map_add' := W.map_add
  map_smul' c f := by simpa [smul_eq_mul] using W.map_smul c f

/-- A candidate distribution whose LF continuity is part of its type. -/
structure ContinuousDistribution where
  evalCLM : Test →L[ℂ] ℂ

def ContinuousDistribution.toDistribution (W : ContinuousDistribution) : Distribution where
  eval := W.evalCLM
  map_add := W.evalCLM.map_add
  map_smul c f := by simp

theorem ContinuousDistribution.evalContinuous (W : ContinuousDistribution) :
    Continuous W.toDistribution.toLinearMap :=
  W.evalCLM.continuous

/-- Weil's Hermitian quadratic functional on convolution squares. -/
def quadratic (W : Distribution) (f : Test) : ℝ :=
  (W.eval (convolution f (involution f))).re

/-- Positive type of a distribution on every admissible convolution square. -/
def IsPositive (W : Distribution) : Prop :=
  ∀ f : Test, 0 ≤ quadratic W f

/-- The precise final theorem obligation for the zeta distribution.

No inhabitant is supplied: proving this requires constructing the
Guinand--Weil distribution, proving the explicit formula with the chosen
normalization, and proving its positivity criterion. -/
def BridgeObligation (W : Distribution) : Prop :=
  IsPositive W ↔ RiemannHypothesis

/-- The two logical directions of the universal Weil criterion are kept
separate because neither follows from the explicit formula alone. -/
def PositivityImpliesRH (W : Distribution) : Prop :=
  IsPositive W → RiemannHypothesis

def RHImpliesPositivity (W : Distribution) : Prop :=
  RiemannHypothesis → IsPositive W

theorem bridgeObligation_iff_directions (W : Distribution) :
    BridgeObligation W ↔ PositivityImpliesRH W ∧ RHImpliesPositivity W := by
  constructor
  · intro h
    exact ⟨h.mp, h.mpr⟩
  · rintro ⟨hforward, hbackward⟩
    exact ⟨hforward, hbackward⟩

/-- Pullback of a distribution along a linear map of test spaces. -/
def Distribution.pullback (W : Distribution) (T : Test →ₗ[ℂ] Test) : Distribution where
  eval f := W.eval (T f)
  map_add f g := by simp [W.map_add]
  map_smul c f := by simp [W.map_smul]

/-! ## Finite explicit-formula approximants -/

/-- A finite spectral sum.  The index type carries multiplicity by repetition. -/
def finiteZeroSum {ι : Type*} [Fintype ι] (zeros : ι → ℂ) (f : Test) : ℂ :=
  ∑ i, transform f (zeros i)

/-- A finite prime-power sample in logarithmic coordinates.  The caller
supplies the source-normalized complex weights and signed log locations. -/
def finitePrimePowerSum {κ : Type*} [Fintype κ]
    (weight : κ → ℂ) (logLocation : κ → ℝ) (f : Test) : ℂ :=
  ∑ k, weight k * f (logLocation k)

/-- A multiplicity-aware enumeration of symmetry orbits in centred
coordinates.  If `z = ρ - 1/2`, its functional-equation partner is
`-conj z`.  Fixed orbits are marked explicitly so zeros on the critical line
are not counted twice.  This structure does not assert that its entries
enumerate the zeta zeros. -/
structure MultiplicityAwareSymmetricZeros where
  representative : ℕ → ℂ
  multiplicity : ℕ → ℕ
  fixedBySymmetry : ℕ → Bool
  fixed_spec : ∀ n, fixedBySymmetry n = true →
    representative n = -star (representative n)

/-- Exact specification saying that a modeled list is the quotient of the
actual centered nontrivial zeta zeros by `z ↦ -conj z`, with analytic
multiplicity.  `orbit_complete_unique` excludes duplicate modeled orbits.

The structure is intentionally not inhabited below: Mathlib currently proves
the holomorphic `z ↦ -z` symmetry, but has no conjugation or analytic-order
transport theorem for `riemannZeta`, so those facts cannot be manufactured
without an unsupported assumption. -/
structure ExactActualZetaOrbitModel (Z : MultiplicityAwareSymmetricZeros) : Prop where
  representative_mem : ∀ i,
    Z.representative i ∈ actualCenteredNontrivialZetaZeros
  multiplicity_eq : ∀ i,
    Z.multiplicity i =
      actualCenteredZetaMultiplicity (Z.representative i)
  fixed_iff : ∀ i,
    Z.fixedBySymmetry i = true ↔
      Z.representative i = -star (Z.representative i)
  orbit_complete_unique : ∀ z ∈ actualCenteredNontrivialZetaZeros,
    ∃! i, z = Z.representative i ∨ z = -star (Z.representative i)

/-- An exact actual-orbit model supplies both actual zero identities required
by the earlier shell-count bridge. -/
theorem ExactActualZetaOrbitModel.isZetaZero
    {Z : MultiplicityAwareSymmetricZeros}
    (H : ExactActualZetaOrbitModel Z) (i : ℕ) :
    riemannZeta ((1 / 2 : ℂ) + Z.representative i) = 0 :=
  (H.representative_mem i).1

/-- Partner membership is now a theorem, not an orbit-model assumption. -/
theorem ExactActualZetaOrbitModel.partner_mem
    {Z : MultiplicityAwareSymmetricZeros}
    (H : ExactActualZetaOrbitModel Z) (i : ℕ) :
    -star (Z.representative i) ∈ actualCenteredNontrivialZetaZeros :=
  actualCenteredNontrivialZetaZeros_neg_star (H.representative_mem i)

/-- Partner multiplicity equality is likewise forced by actual zeta
symmetry. -/
theorem ExactActualZetaOrbitModel.partner_multiplicity_eq
    {Z : MultiplicityAwareSymmetricZeros}
    (H : ExactActualZetaOrbitModel Z) (i : ℕ) :
    actualCenteredZetaMultiplicity (-star (Z.representative i)) =
      Z.multiplicity i :=
  (actualCenteredZetaMultiplicity_neg_star (H.representative_mem i)).trans
    (H.multiplicity_eq i).symm

theorem ExactActualZetaOrbitModel.partnerIsZetaZero
    {Z : MultiplicityAwareSymmetricZeros}
    (H : ExactActualZetaOrbitModel Z) (i : ℕ) :
    riemannZeta ((1 / 2 : ℂ) - star (Z.representative i)) = 0 := by
  simpa only [sub_eq_add_neg] using (H.partner_mem i).1

/-- Standard zero-location input needed to turn a critical-strip orbit model
into Mathlib's global `RiemannHypothesis`: every nontrivial zero is in the
open critical strip.  This follows classically from the functional equation
and the classification of Gamma/cosine zeros, but that classification is not
currently exposed by the pinned zeta API. -/
def NontrivialZetaZerosInCriticalStrip : Prop :=
  ∀ s : ℂ, riemannZeta s = 0 →
    (¬∃ n : ℕ, s = -2 * (n + 1)) → s ≠ 1 →
    0 < s.re ∧ s.re < 1

/-- The standard global strip classification, proved from Mathlib's
closed-right-half-plane nonvanishing, the completed functional equation, and
the exact zero set of `GammaR`. -/
theorem nontrivialZetaZerosInCriticalStrip :
    NontrivialZetaZerosInCriticalStrip := by
  intro s hzero hnontrivial hs1
  have hs0 : s ≠ 0 := by
    intro h
    subst s
    norm_num [riemannZeta_zero] at hzero
  constructor
  · by_contra hnot
    have hsre : s.re ≤ 0 := le_of_not_gt hnot
    let t : ℂ := 1 - s
    have htre : 1 ≤ t.re := by
      dsimp [t]
      norm_num
      linarith
    have ht0 : t ≠ 0 := by
      intro h
      have hre := congrArg Complex.re h
      dsimp [t] at hre
      norm_num at hre
      linarith
    have htzeta : riemannZeta t ≠ 0 :=
      riemannZeta_ne_zero_of_one_le_re htre
    have htcomp : completedRiemannZeta t ≠ 0 := by
      intro hcomp
      apply htzeta
      rw [riemannZeta_def_of_ne_zero ht0, hcomp, zero_div]
    have hscomp : completedRiemannZeta s ≠ 0 := by
      intro hcomp
      apply htcomp
      dsimp [t]
      rw [completedRiemannZeta_one_sub]
      exact hcomp
    have hsGamma : Complex.Gammaℝ s = 0 := by
      by_contra hGamma
      have hzeta_ne : riemannZeta s ≠ 0 := by
        rw [riemannZeta_def_of_ne_zero hs0]
        exact div_ne_zero hscomp hGamma
      exact hzeta_ne hzero
    obtain ⟨k, hk⟩ := Complex.Gammaℝ_eq_zero_iff.mp hsGamma
    cases k with
    | zero =>
        apply hs0
        simpa using hk
    | succ k =>
        apply hnontrivial
        refine ⟨k, ?_⟩
        convert hk using 1 <;> push_cast <;> ring
  · by_contra hnot
    have hsre : 1 ≤ s.re := le_of_not_gt hnot
    exact (riemannZeta_ne_zero_of_one_le_re hsre) hzero

/-- Exact orbit completeness plus critical-line representatives gives the
actual RH, once the standard global zero-location lemma is supplied. -/
theorem ExactActualZetaOrbitModel.riemannHypothesis_of_representatives
    {Z : MultiplicityAwareSymmetricZeros}
    (H : ExactActualZetaOrbitModel Z)
    (hloc : NontrivialZetaZerosInCriticalStrip)
    (hre : ∀ i, (Z.representative i).re = 0) :
    RiemannHypothesis := by
  intro s hzero hnontrivial hs1
  have hstrip := hloc s hzero hnontrivial hs1
  let z : ℂ := s - 1 / 2
  have hzmem : z ∈ actualCenteredNontrivialZetaZeros := by
    refine ⟨?_, ?_, ?_⟩
    · change riemannZeta ((1 / 2 : ℂ) + (s - 1 / 2)) = 0
      convert hzero using 1 <;> ring
    · dsimp [z]
      norm_num
      linarith [hstrip.1]
    · dsimp [z]
      norm_num
      linarith [hstrip.2]
  obtain ⟨i, hi, -⟩ := H.orbit_complete_unique z hzmem
  have hzre : z.re = 0 := by
    rcases hi with hi | hi
    · rw [hi]
      exact hre i
    · rw [hi]
      simp [hre i]
  dsimp [z] at hzre
  norm_num at hzre ⊢
  linarith

/-- With the global strip theorem now proved, exact orbit completeness and
critical-line representatives imply RH without a separate location input. -/
theorem ExactActualZetaOrbitModel.riemannHypothesis
    {Z : MultiplicityAwareSymmetricZeros}
    (H : ExactActualZetaOrbitModel Z)
    (hre : ∀ i, (Z.representative i).re = 0) :
    RiemannHypothesis :=
  H.riemannHypothesis_of_representatives
    nontrivialZetaZerosInCriticalStrip hre

/-- Contribution of one multiplicity-weighted symmetry orbit.  Fixed points
contribute once; non-fixed representatives contribute together with their
functional-equation partner. -/
def symmetricZeroPairTerm
    (Z : MultiplicityAwareSymmetricZeros) (f : Test) (n : ℕ) : ℂ :=
  if Z.fixedBySymmetry n then
    (Z.multiplicity n : ℂ) * transform f (Z.representative n)
  else
    (Z.multiplicity n : ℂ) *
      (transform f (Z.representative n) +
        transform f (-star (Z.representative n)))

/-- A positive-multiplicity non-fixed off-critical orbit has a genuine
convolution square with strictly negative selected pair contribution. -/
theorem exists_convolutionSquare_selectedPairTerm_neg
    (Z : MultiplicityAwareSymmetricZeros) (i : ℕ)
    (hfixed : Z.fixedBySymmetry i = false)
    (hoff : Z.representative i ≠ -star (Z.representative i))
    (hmult : 0 < Z.multiplicity i) :
    ∃ f : Test,
      (symmetricZeroPairTerm Z (convolution f (involution f)) i).re < 0 := by
  rcases exists_convolutionSquare_negative_on_offCritical_pair
      (Z.representative i) hoff with ⟨f, hzi, hpartner⟩
  refine ⟨f, ?_⟩
  rw [symmetricZeroPairTerm, if_neg (by simpa using hfixed), hzi, hpartner]
  norm_num
  exact_mod_cast hmult

/-- The local finite-difference construction gives the selected pair exactly,
with no global interpolation system.  Near collisions enter only through the
two explicit finite `LocalSeparatorGap` hypotheses. -/
theorem localPairSeparator_selectedPairTerm
    (Z : MultiplicityAwareSymmetricZeros) (i : ℕ)
    (a : ℝ) (nearby : List ℂ)
    (hfixed : Z.fixedBySymmetry i = false)
    (hz : LocalSeparatorGap a (Z.representative i)
      (-star (Z.representative i) :: nearby))
    (hpartner : LocalSeparatorGap a (-star (Z.representative i))
      (Z.representative i :: nearby)) :
    symmetricZeroPairTerm Z
        (convolution (localPairSeparator a (Z.representative i) nearby)
          (involution (localPairSeparator a (Z.representative i) nearby))) i =
      -(2 * Z.multiplicity i : ℕ) := by
  rw [symmetricZeroPairTerm, if_neg (by simpa using hfixed),
    transform_localPairSeparator_square_target _ _ _ hz hpartner,
    transform_localPairSeparator_square_partner _ _ _ hz hpartner]
  norm_num
  ring

/-- Every listed neighboring orbit is annihilated exactly, independently of
its multiplicity.  Both members are listed for a non-fixed symmetry orbit. -/
theorem localPairSeparator_nearbyPairTerm_eq_zero
    (Z : MultiplicityAwareSymmetricZeros) (j : ℕ)
    (a : ℝ) (z : ℂ) (nearby : List ℂ)
    (hrep : Z.representative j ∈ nearby)
    (hpartner : -star (Z.representative j) ∈ nearby) :
    symmetricZeroPairTerm Z
        (convolution (localPairSeparator a z nearby)
          (involution (localPairSeparator a z nearby))) j = 0 := by
  rw [symmetricZeroPairTerm]
  split_ifs
  · rw [transform_localPairSeparator_square_nearby_eq_zero _ _ _ hrep]
    ring
  · rw [transform_localPairSeparator_square_nearby_eq_zero _ _ _ hrep,
      transform_localPairSeparator_square_nearby_eq_zero _ _ _ hpartner]
    ring

/-- Symmetric finite truncation for a fixed multiplicity-aware enumeration. -/
def symmetricZeroPartialSum
    (Z : MultiplicityAwareSymmetricZeros) (f : Test) (N : ℕ) : ℂ :=
  ∑ n ∈ Finset.range N, symmetricZeroPairTerm Z f n

/-- A finite, not necessarily initial, collection of symmetry-orbit
contributions. -/
def symmetricZeroFinsetSum
    (Z : MultiplicityAwareSymmetricZeros) (f : Test) (S : Finset ℕ) : ℂ :=
  ∑ n ∈ S, symmetricZeroPairTerm Z f n

/-- Quantitative robust finite separation criterion. -/
theorem symmetricZeroFinsetSum_re_lt_of_target
    (Z : MultiplicityAwareSymmetricZeros) (f : Test)
    (S : Finset ℕ) {i : ℕ} (hi : i ∈ S) {R q : ℝ}
    (hrest : ‖∑ j ∈ S.erase i, symmetricZeroPairTerm Z f j‖ ≤ R)
    (htarget : (symmetricZeroPairTerm Z f i).re < q - R) :
    (symmetricZeroFinsetSum Z f S).re < q := by
  let rest : ℂ := ∑ j ∈ S.erase i, symmetricZeroPairTerm Z f j
  have hrestRe : rest.re ≤ R :=
    (Complex.re_le_norm rest).trans hrest
  have hsplit :
      rest + symmetricZeroPairTerm Z f i =
        symmetricZeroFinsetSum Z f S := by
    exact S.sum_erase_add (fun j ↦ symmetricZeroPairTerm Z f j) hi
  rw [← hsplit]
  rw [Complex.add_re]
  linarith

/-- If one selected orbit is more negative than a norm bound for all remaining
selected orbits, then the whole finite spectral sum is negative. -/
theorem symmetricZeroFinsetSum_re_neg_of_target
    (Z : MultiplicityAwareSymmetricZeros) (f : Test)
    (S : Finset ℕ) {i : ℕ} (hi : i ∈ S) {R : ℝ}
    (hrest : ‖∑ j ∈ S.erase i, symmetricZeroPairTerm Z f j‖ ≤ R)
    (htarget : (symmetricZeroPairTerm Z f i).re < -R) :
    (symmetricZeroFinsetSum Z f S).re < 0 := by
  apply symmetricZeroFinsetSum_re_lt_of_target Z f S hi hrest
  simpa using htarget

/-- The unconditional regularized value selected by absolute summability. -/
def symmetricZeroRegularized
    (Z : MultiplicityAwareSymmetricZeros) (f : Test) : ℂ :=
  ∑' n, symmetricZeroPairTerm Z f n

/-- Under the explicit summability hypothesis, symmetric finite truncations
converge to the selected multiplicity-aware regularization. -/
theorem symmetricZeroPartialSum_tendsto
    (Z : MultiplicityAwareSymmetricZeros) (f : Test)
    (hsum : Summable (symmetricZeroPairTerm Z f)) :
    Tendsto (symmetricZeroPartialSum Z f) atTop
      (𝓝 (symmetricZeroRegularized Z f)) := by
  exact hsum.hasSum.tendsto_sum_nat

/-- Limiting separation from a uniform residual-spectrum bound.  A fixed
negative margin survives passage from finite symmetric truncations to the
normally summed spectral value. -/
theorem symmetricZeroRegularized_re_neg_of_tail_domination
    (Z : MultiplicityAwareSymmetricZeros) (f : Test) (i : ℕ)
    {R ε : ℝ} (hε : 0 < ε)
    (hsum : Summable (symmetricZeroPairTerm Z f))
    (htarget : (symmetricZeroPairTerm Z f i).re < -ε - R)
    (hrest : ∀ᶠ N in atTop,
      i ∈ Finset.range N ∧
        ‖∑ j ∈ (Finset.range N).erase i,
          symmetricZeroPairTerm Z f j‖ ≤ R) :
    (symmetricZeroRegularized Z f).re < 0 := by
  have hfinite : ∀ᶠ N in atTop,
      (symmetricZeroPartialSum Z f N).re < -ε := by
    filter_upwards [hrest] with N hN
    change (symmetricZeroFinsetSum Z f (Finset.range N)).re < -ε
    apply symmetricZeroFinsetSum_re_lt_of_target Z f (Finset.range N)
      hN.1 hN.2
    linarith
  have hlim :
      Tendsto (fun N ↦ (symmetricZeroPartialSum Z f N).re) atTop
        (𝓝 (symmetricZeroRegularized Z f).re) :=
    Complex.continuous_re.tendsto
      (symmetricZeroRegularized Z f) |>.comp
        (symmetricZeroPartialSum_tendsto Z f hsum)
  have hle : (symmetricZeroRegularized Z f).re ≤ -ε :=
    le_of_tendsto hlim (hfinite.mono fun _ h ↦ h.le)
  linarith

/-- Absolute summability makes the regularized zero value invariant under
every bijective re-enumeration.  This does not manufacture an enumeration of
zeta zeros or prove the required summability. -/
theorem symmetricZeroRegularized_reindex
    (Z : MultiplicityAwareSymmetricZeros) (f : Test)
    (hsum : Summable (symmetricZeroPairTerm Z f)) (e : ℕ ≃ ℕ) :
    ∑' n, symmetricZeroPairTerm Z f (e n) =
      symmetricZeroRegularized Z f := by
  have hreindexed : Summable (fun n ↦ symmetricZeroPairTerm Z f (e n)) :=
    e.summable_iff.mpr hsum
  exact hreindexed.hasSum.tsum_eq.trans <|
    (e.hasSum_iff.mpr hsum.hasSum).tsum_eq.trans <| by
      rfl

/-- Normal convergence of the multiplicity-aware orbit series.  This is
strictly stronger than the classical conditionally regularized zero sum. -/
def ZeroOrbitNormallyConvergent
    (Z : MultiplicityAwareSymmetricZeros) (f : Test) : Prop :=
  Summable (fun n ↦ ‖symmetricZeroPairTerm Z f n‖)

/-- Total norm mass of all spectral orbits except one selected index. -/
def residualOrbitNormMass
    (Z : MultiplicityAwareSymmetricZeros) (f : Test) (i : ℕ) : ℝ :=
  ∑' j, if j = i then 0 else ‖symmetricZeroPairTerm Z f j‖

theorem ZeroOrbitNormallyConvergent.summable
    {Z : MultiplicityAwareSymmetricZeros} {f : Test}
    (h : ZeroOrbitNormallyConvergent Z f) :
    Summable (symmetricZeroPairTerm Z f) :=
  summable_norm_iff.mp h

theorem ZeroOrbitNormallyConvergent.residual_summable
    {Z : MultiplicityAwareSymmetricZeros} {f : Test}
    (h : ZeroOrbitNormallyConvergent Z f) (i : ℕ) :
    Summable (fun j ↦ if j = i then 0 else
      ‖symmetricZeroPairTerm Z f j‖) := by
  apply h.of_nonneg_of_le
  · intro j
    split_ifs <;> positivity
  · intro j
    split_ifs <;> simp

/-- Normal convergence gives a truncation-independent residual bound after
removing one selected orbit. -/
theorem ZeroOrbitNormallyConvergent.norm_sum_erase_le_residual
    {Z : MultiplicityAwareSymmetricZeros} {f : Test}
    (h : ZeroOrbitNormallyConvergent Z f) (i N : ℕ) :
    ‖∑ j ∈ (Finset.range N).erase i, symmetricZeroPairTerm Z f j‖ ≤
      residualOrbitNormMass Z f i := by
  let r : ℕ → ℝ := fun j ↦ if j = i then 0 else
    ‖symmetricZeroPairTerm Z f j‖
  have hrsum : Summable r := h.residual_summable i
  calc
    ‖∑ j ∈ (Finset.range N).erase i, symmetricZeroPairTerm Z f j‖ ≤
        ∑ j ∈ (Finset.range N).erase i,
          ‖symmetricZeroPairTerm Z f j‖ := norm_sum_le _ _
    _ = ∑ j ∈ (Finset.range N).erase i, r j := by
      apply Finset.sum_congr rfl
      intro j hj
      dsimp [r]
      rw [if_neg (Finset.ne_of_mem_erase hj)]
    _ ≤ ∑' j, r j :=
      hrsum.sum_le_tsum _ fun j hj ↦ by positivity
    _ = residualOrbitNormMass Z f i := rfl

/-- A summable count-times-decay majorant bounds the complete residual norm
mass away from a selected orbit. -/
theorem ZeroOrbitNormallyConvergent.residualOrbitNormMass_le
    {Z : MultiplicityAwareSymmetricZeros} {f : Test}
    (h : ZeroOrbitNormallyConvergent Z f) (i : ℕ)
    (majorant : ℕ → ℝ) (hmajorant : Summable majorant)
    (hnonneg : ∀ j, 0 ≤ majorant j)
    (hbound : ∀ j, j ≠ i →
      ‖symmetricZeroPairTerm Z f j‖ ≤ majorant j) :
    residualOrbitNormMass Z f i ≤ ∑' j, majorant j := by
  apply (h.residual_summable i).tsum_le_tsum
  · intro j
    split_ifs with hj
    · exact hnonneg j
    · exact hbound j hj
  · exact hmajorant

/-- A selected orbit whose negative real contribution strictly dominates the
total norm mass of every other orbit separates the full normally convergent
spectrum. -/
theorem symmetricZeroRegularized_re_neg_of_selected_dominates
    (Z : MultiplicityAwareSymmetricZeros) (f : Test) (i : ℕ)
    (hnormal : ZeroOrbitNormallyConvergent Z f)
    (htarget : (symmetricZeroPairTerm Z f i).re <
      -residualOrbitNormMass Z f i) :
    (symmetricZeroRegularized Z f).re < 0 := by
  let R : ℝ := residualOrbitNormMass Z f i
  let ε : ℝ := (-R - (symmetricZeroPairTerm Z f i).re) / 2
  have hε : 0 < ε := by
    dsimp [ε, R]
    linarith
  apply symmetricZeroRegularized_re_neg_of_tail_domination
    Z f i (R := R) (ε := ε) hε hnormal.summable
  · dsimp [ε, R]
    linarith
  · filter_upwards [eventually_gt_atTop i] with N hN
    exact ⟨Finset.mem_range.mpr hN,
      hnormal.norm_sum_erase_le_residual i N⟩

theorem symmetricZeroRegularized_re_neg_of_selected_dominates_majorant
    (Z : MultiplicityAwareSymmetricZeros) (f : Test) (i : ℕ)
    (hnormal : ZeroOrbitNormallyConvergent Z f)
    (majorant : ℕ → ℝ) (hmajorant : Summable majorant)
    (hnonneg : ∀ j, 0 ≤ majorant j)
    (hbound : ∀ j, j ≠ i →
      ‖symmetricZeroPairTerm Z f j‖ ≤ majorant j)
    (htarget : (symmetricZeroPairTerm Z f i).re < -(∑' j, majorant j)) :
    (symmetricZeroRegularized Z f).re < 0 := by
  apply symmetricZeroRegularized_re_neg_of_selected_dominates Z f i hnormal
  exact htarget.trans_le (neg_le_neg
    (hnormal.residualOrbitNormMass_le i majorant hmajorant hnonneg hbound))

/-- A summable radial/counting majorant is sufficient for normal zero-orbit
convergence.  Instantiating this requires zero-count and transform-decay
bounds absent from the pinned zeta API. -/
theorem ZeroOrbitNormallyConvergent.of_bound
    {Z : MultiplicityAwareSymmetricZeros} {f : Test}
    (majorant : ℕ → ℝ) (hmajorant : Summable majorant)
    (hbound : ∀ n, ‖symmetricZeroPairTerm Z f n‖ ≤ majorant n) :
    ZeroOrbitNormallyConvergent Z f :=
  hmajorant.of_nonneg_of_le (fun _ ↦ norm_nonneg _) hbound

/-- A partition of multiplicity-aware zero orbits into finite height shells. -/
structure ZeroOrbitShells where
  shell : ℕ → Finset ℕ
  unique_shell : ∀ i, ∃! k, i ∈ shell k

/-- The weakest shell-count/transform-decay summability theorem used by this
route.  A summable product of shell cardinality and a per-zero decay bound
implies normal convergence of the full orbit series. -/
theorem ZeroOrbitNormallyConvergent.of_shell_count_decay
    {Z : MultiplicityAwareSymmetricZeros} {f : Test}
    (S : ZeroOrbitShells) (decay : ℕ → ℝ)
    (hcountDecay : Summable fun k ↦ (S.shell k).card * decay k)
    (hterm : ∀ k i, i ∈ S.shell k →
      ‖symmetricZeroPairTerm Z f i‖ ≤ decay k) :
    ZeroOrbitNormallyConvergent Z f := by
  classical
  rw [ZeroOrbitNormallyConvergent]
  rw [summable_partition (fun i ↦ norm_nonneg (symmetricZeroPairTerm Z f i))
    S.unique_shell]
  constructor
  · intro k
    exact Summable.of_finite
  · apply hcountDecay.of_nonneg_of_le
    · intro k
      exact tsum_nonneg fun _ ↦ norm_nonneg _
    · intro k
      rw [tsum_fintype]
      calc
        ∑ i : {i // i ∈ S.shell k}, ‖symmetricZeroPairTerm Z f i‖
            ≤ ∑ _i : {i // i ∈ S.shell k}, decay k :=
          Finset.sum_le_sum fun i _ ↦ hterm k i i.property
        _ = (S.shell k).card * decay k := by simp

/-- Multiplicity-weighted version of shell-count decay.  This is the form
compatible with Riemann--von Mangoldt, which counts zeros with multiplicity. -/
theorem ZeroOrbitNormallyConvergent.of_shell_mass_decay
    {Z : MultiplicityAwareSymmetricZeros} {f : Test}
    (S : ZeroOrbitShells) (mass : ℕ → ℕ) (decay : ℕ → ℝ)
    (hmassDecay : Summable fun k ↦
      (∑ i ∈ S.shell k, mass i : ℕ) * decay k)
    (hterm : ∀ k i, i ∈ S.shell k →
      ‖symmetricZeroPairTerm Z f i‖ ≤ mass i * decay k) :
    ZeroOrbitNormallyConvergent Z f := by
  classical
  rw [ZeroOrbitNormallyConvergent]
  rw [summable_partition (fun i ↦ norm_nonneg (symmetricZeroPairTerm Z f i))
    S.unique_shell]
  constructor
  · intro k
    exact Summable.of_finite
  · apply hmassDecay.of_nonneg_of_le
    · intro k
      exact tsum_nonneg fun _ ↦ norm_nonneg _
    · intro k
      rw [tsum_fintype]
      calc
        ∑ i : {i // i ∈ S.shell k}, ‖symmetricZeroPairTerm Z f i‖
            ≤ ∑ i : {i // i ∈ S.shell k}, mass i * decay k :=
          Finset.sum_le_sum fun i _ ↦ hterm k i i.property
        _ = (∑ i ∈ S.shell k, mass i : ℕ) * decay k := by
          push_cast
          rw [Finset.sum_mul]
          simpa using
            (Finset.sum_attach (S.shell k)
              (fun i ↦ (mass i : ℝ) * decay k))

/-- Exact quantitative zeta-zero input requested from Riemann--von Mangoldt.
The statement is deliberately packaged as an obligation: pinned Mathlib does
not prove it.  Spectral points are centered by `ρ = 1/2 + z`, shells are unit
height intervals, and the shell count includes multiplicity.  The quadratic
bound is a convenient weakening of the sourced `O(T log T)` estimate. -/
structure ZetaZeroShellCountObligation
    (Z : MultiplicityAwareSymmetricZeros) (S : ZeroOrbitShells) where
  isZetaZero : ∀ i, riemannZeta ((1 / 2 : ℂ) + Z.representative i) = 0
  partnerIsZetaZero : ∀ i,
    riemannZeta ((1 / 2 : ℂ) - star (Z.representative i)) = 0
  coversNontrivialZeros : ∀ ρ,
    riemannZeta ρ = 0 → 0 < ρ.re → ρ.re < 1 →
      ∃ i, ρ = (1 / 2 : ℂ) + Z.representative i ∨
        ρ = (1 / 2 : ℂ) - star (Z.representative i)
  shell_by_height : ∀ k i,
    i ∈ S.shell k ↔
      (k : ℝ) ≤ |(Z.representative i).im| ∧
        |(Z.representative i).im| < (k : ℝ) + 1
  constant : ℝ
  constant_nonneg : 0 ≤ constant
  multiplicity_count_le : ∀ k,
    (∑ i ∈ S.shell k, Z.multiplicity i : ℕ) ≤
      constant * (k + 2 : ℝ) ^ 2

/-- Under RH, the two zeta-zero equations in the shell obligation force every
centered representative onto the imaginary axis.  The partner equation rules
out trivial negative-even zeros without adding a separate nontriviality
field. -/
theorem ZetaZeroShellCountObligation.representative_re_zero_of_rh
    {Z : MultiplicityAwareSymmetricZeros} {S : ZeroOrbitShells}
    (H : ZetaZeroShellCountObligation Z S) (hRH : RiemannHypothesis) (i : ℕ) :
    (Z.representative i).re = 0 := by
  let ρ : ℂ := (1 / 2 : ℂ) + Z.representative i
  have hρzero : riemannZeta ρ = 0 := H.isZetaZero i
  have hρtrivial : ¬∃ n : ℕ, ρ = -2 * (n + 1) := by
    rintro ⟨n, hn⟩
    have hρre : ρ.re = -2 * ((n : ℝ) + 1) := by
      simpa [ρ] using congrArg Complex.re hn
    have hpartnerRe :
        1 ≤ ((1 / 2 : ℂ) - star (Z.representative i)).re := by
      dsimp [ρ] at hρre
      norm_num at hρre ⊢
      nlinarith [show 0 ≤ (n : ℝ) from Nat.cast_nonneg n]
    exact
      (riemannZeta_ne_zero_of_one_le_re hpartnerRe)
        (H.partnerIsZetaZero i)
  have hρone : ρ ≠ 1 := by
    intro h
    rw [h] at hρzero
    exact riemannZeta_one_ne_zero hρzero
  have hline := hRH ρ hρzero hρtrivial hρone
  dsimp [ρ] at hline
  norm_num at hline
  linarith

/-- Exact multiplicity-aware Riemann--von Mangoldt theorem statement needed by
the route.  This is an obligation, not an asserted theorem: `zeroCount N`
counts the modeled positive-height nontrivial zeros with multiplicity, and
the shell relation connects that count to the formal orbit enumeration. -/
structure RiemannVonMangoldtExplicitObligation
    (Z : MultiplicityAwareSymmetricZeros) (S : ZeroOrbitShells) where
  zeroCount : ℕ → ℕ
  shell_le_zeroCount : ∀ k N, k < N →
    (∑ i ∈ S.shell k, Z.multiplicity i : ℕ) ≤ zeroCount N
  explicit_bound : ∀ N : ℕ, 3 ≤ N →
    |(zeroCount N : ℝ) - riemannVonMangoldtMain N| ≤
      riemannVonMangoldtError N

/-- The exact Riemann--von Mangoldt obligation gives a unit-shell upper bound
with no asymptotic simplification. -/
theorem RiemannVonMangoldtExplicitObligation.shell_count_le
    {Z : MultiplicityAwareSymmetricZeros} {S : ZeroOrbitShells}
    (H : RiemannVonMangoldtExplicitObligation Z S) (k : ℕ) :
    (∑ i ∈ S.shell k, Z.multiplicity i : ℕ) ≤
      riemannVonMangoldtMain (k + 3) +
        riemannVonMangoldtError (k + 3) := by
  have hs : (∑ i ∈ S.shell k, Z.multiplicity i : ℕ) ≤ H.zeroCount (k + 3) :=
    H.shell_le_zeroCount k (k + 3) (by omega)
  have hb := H.explicit_bound (k + 3) (by omega)
  have hb' :
      |(H.zeroCount (k + 3) : ℝ) -
        riemannVonMangoldtMain ((k : ℝ) + 3)| ≤
        riemannVonMangoldtError ((k : ℝ) + 3) := by
    simpa [Nat.cast_add] using hb
  calc
    (∑ i ∈ S.shell k, Z.multiplicity i : ℕ) ≤ (H.zeroCount (k + 3) : ℝ) := by
      exact_mod_cast hs
    _ = ((H.zeroCount (k + 3) : ℝ) -
        riemannVonMangoldtMain (k + 3)) +
        riemannVonMangoldtMain (k + 3) := by ring
    _ ≤ riemannVonMangoldtError (k + 3) +
        riemannVonMangoldtMain (k + 3) := by
      gcongr
      exact (le_abs_self _).trans hb'
    _ = riemannVonMangoldtMain (k + 3) +
        riemannVonMangoldtError (k + 3) := add_comm _ _

/-- Exact-source version of the normal-convergence reduction.  The only
remaining analytic input is summability of the explicit count envelope times
the chosen strip-uniform transform decay. -/
theorem RiemannVonMangoldtExplicitObligation.normalConvergence
    {Z : MultiplicityAwareSymmetricZeros} {S : ZeroOrbitShells}
    (H : RiemannVonMangoldtExplicitObligation Z S) {f : Test}
    (decay : ℕ → ℝ)
    (hseries : Summable fun k : ℕ ↦
      (riemannVonMangoldtMain (k + 3) +
        riemannVonMangoldtError (k + 3)) * decay k)
    (hdecay : ∀ k, 0 ≤ decay k)
    (hterm : ∀ k i, i ∈ S.shell k →
      ‖symmetricZeroPairTerm Z f i‖ ≤ Z.multiplicity i * decay k) :
    ZeroOrbitNormallyConvergent Z f := by
  apply ZeroOrbitNormallyConvergent.of_shell_mass_decay
    S Z.multiplicity decay
  · apply hseries.of_nonneg_of_le
    · intro k
      exact mul_nonneg (Nat.cast_nonneg _) (hdecay k)
    · intro k
      exact mul_le_mul_of_nonneg_right (H.shell_count_le k) (hdecay k)
  · exact hterm

/-- A sourced multiplicity shell count plus any summable shell-uniform
transform envelope yields convergence of the actual multiplicity-weighted
orbit terms. -/
theorem ZetaZeroShellCountObligation.normalConvergence
    {Z : MultiplicityAwareSymmetricZeros} {S : ZeroOrbitShells}
    (H : ZetaZeroShellCountObligation Z S) {f : Test} (decay : ℕ → ℝ)
    (hseries : Summable fun k : ℕ ↦
      H.constant * ((k : ℝ) + 2) ^ 2 * decay k)
    (hdecay : ∀ k, 0 ≤ decay k)
    (hterm : ∀ k i, i ∈ S.shell k →
      ‖symmetricZeroPairTerm Z f i‖ ≤ Z.multiplicity i * decay k) :
    ZeroOrbitNormallyConvergent Z f := by
  apply ZeroOrbitNormallyConvergent.of_shell_mass_decay
    S Z.multiplicity decay
  · apply hseries.of_nonneg_of_le
    · intro k
      exact mul_nonneg (Nat.cast_nonneg _) (hdecay k)
    · intro k
      exact mul_le_mul_of_nonneg_right (H.multiplicity_count_le k) (hdecay k)
  · exact hterm

/-- Every convergent real series has an arbitrarily small shifted tail.  This
is the quantitative finite-to-limit step used after annihilating finitely many
zero orbits. -/
theorem exists_tsum_nat_add_lt_of_summable
    {u : ℕ → ℝ} (hu : Summable u) {ε : ℝ} (hε : 0 < ε) :
    ∃ N, ∑' k, u (k + N) < ε := by
  have hnhds :
      Set.Ioo ((∑' k, u k) - ε) ((∑' k, u k) + ε) ∈
        𝓝 (∑' k, u k) :=
    Ioo_mem_nhds (by linarith) (by linarith)
  rcases (hu.hasSum.tendsto_sum_nat.eventually hnhds).exists with ⟨N, hN⟩
  refine ⟨N, ?_⟩
  have hsplit := hu.sum_add_tsum_nat_add N
  linarith

/-- The published explicit Riemann--von Mangoldt envelope, once supplied for
the actual Mathlib zero count, has an arbitrarily small count-times-decay
tail. -/
theorem ActualRiemannVonMangoldtBound.exists_envelope_tail_lt
    (H : ActualRiemannVonMangoldtBound) (decay : ℕ → ℝ)
    (hseries : Summable fun k : ℕ ↦
      (riemannVonMangoldtMain (k + 3) +
        riemannVonMangoldtError (k + 3)) * decay k)
    {ε : ℝ} (hε : 0 < ε) :
    ∃ N, ∑' k : ℕ,
      (riemannVonMangoldtMain ((k + N + 3 : ℕ) : ℝ) +
        riemannVonMangoldtError ((k + N + 3 : ℕ) : ℝ)) *
          decay (k + N) < ε := by
  simpa [Nat.cast_add] using
    exists_tsum_nat_add_lt_of_summable hseries hε

/-- Actual multiplicity-weighted unit-shell decay is bounded by the published
main-plus-error envelope. -/
theorem ActualRiemannVonMangoldtBound.shell_weighted_decay_le
    (H : ActualRiemannVonMangoldtBound) (decay : ℕ → ℝ)
    (hdecay : ∀ k, 0 ≤ decay k) (k : ℕ) :
    (∑ z ∈ positiveNontrivialZetaZeroWindowFinset (k + 3),
        if (k + 2 : ℝ) < z.im
        then (zetaZeroMultiplicity z : ℝ) * decay k else 0) ≤
      (riemannVonMangoldtMain (k + 3) +
        riemannVonMangoldtError (k + 3)) * decay k := by
  have heq :
      (∑ z ∈ positiveNontrivialZetaZeroWindowFinset (k + 3),
          if (k + 2 : ℝ) < z.im
          then (zetaZeroMultiplicity z : ℝ) * decay k else 0) =
        (∑ z ∈ positiveNontrivialZetaZeroWindowFinset (k + 3),
          if (k + 2 : ℝ) < z.im
          then (zetaZeroMultiplicity z : ℝ) else 0) * decay k := by
    rw [Finset.sum_mul]
    apply Finset.sum_congr rfl
    intro z hz
    split_ifs <;> simp
  rw [heq]
  have hcast :
      (∑ z ∈ positiveNontrivialZetaZeroWindowFinset (k + 3),
          if (k + 2 : ℝ) < z.im
          then (zetaZeroMultiplicity z : ℝ) else 0) ≤
        riemannVonMangoldtMain (k + 3) +
          riemannVonMangoldtError (k + 3) := by
    exact_mod_cast H.shell_count_le k
  exact mul_le_mul_of_nonneg_right hcast (hdecay k)

/-- The exact sourced RvM envelope is summable against fourth-power vertical
decay, with no extra summability hypothesis. -/
theorem ActualRiemannVonMangoldtBound.summable_fourthPower_envelope
    (H : ActualRiemannVonMangoldtBound) (C : ℝ) (hC : 0 ≤ C) :
    Summable fun k : ℕ ↦
      (riemannVonMangoldtMain (k + 3) +
        riemannVonMangoldtError (k + 3)) *
          (C / ((k : ℝ) + 2) ^ 4) := by
  have hp : Summable fun k : ℕ ↦ 1 / (k : ℝ) ^ 2 :=
    (Real.summable_one_div_nat_pow (p := 2)).2 (by norm_num)
  have hp' : Summable fun k : ℕ ↦ 1 / ((k + 2 : ℕ) : ℝ) ^ 2 :=
    (summable_nat_add_iff 2).2 hp
  have hmajor : Summable fun k : ℕ ↦
      12 * C * (1 / ((k : ℝ) + 2) ^ 2) := by
    simpa [Nat.cast_add] using hp'.mul_left (12 * C)
  apply hmajor.of_nonneg_of_le
  · intro k
    have henv0 : 0 ≤ riemannVonMangoldtMain (k + 3) +
        riemannVonMangoldtError (k + 3) := by
      exact (Nat.cast_nonneg
        (∑ z ∈ positiveNontrivialZetaZeroWindowFinset (k + 3),
          if (k + 2 : ℝ) < z.im then zetaZeroMultiplicity z else 0)).trans
        (H.shell_count_le k)
    exact mul_nonneg henv0 (div_nonneg hC (by positivity))
  · intro k
    have henv := riemannVonMangoldtMain_add_error_le_quadratic
      (T := (k : ℝ) + 3) (by
        have hk : (0 : ℝ) ≤ k := Nat.cast_nonneg k
        linarith)
    have hk : (0 : ℝ) < (k : ℝ) + 2 := by positivity
    have hratio :
        3 * ((k : ℝ) + 3) ^ 2 *
            (C / ((k : ℝ) + 2) ^ 4) ≤
          12 * C * (1 / ((k : ℝ) + 2) ^ 2) := by
      have hsq : ((k : ℝ) + 3) ^ 2 ≤
          4 * ((k : ℝ) + 2) ^ 2 := by nlinarith
      rw [div_eq_mul_inv]
      field_simp
      nlinarith
    exact (mul_le_mul_of_nonneg_right henv
      (div_nonneg hC (by positivity))).trans hratio

/-- Published RvM plus fourth-power transform decay gives a tail below any
positive fraction, in particular below a selected multiplicity. -/
theorem ActualRiemannVonMangoldtBound.exists_fourthPower_tail_lt
    (H : ActualRiemannVonMangoldtBound) (C : ℝ) (hC : 0 ≤ C)
    {ε : ℝ} (hε : 0 < ε) :
    ∃ N, ∑' k : ℕ,
      (riemannVonMangoldtMain ((k + N + 3 : ℕ) : ℝ) +
        riemannVonMangoldtError ((k + N + 3 : ℕ) : ℝ)) *
          (C / (((k + N : ℕ) : ℝ) + 2) ^ 4) < ε :=
  H.exists_envelope_tail_lt
    (fun k ↦ C / ((k : ℝ) + 2) ^ 4)
    (H.summable_fourthPower_envelope C hC) hε

/-- Actual multiplicity-weighted mass in the positive-height unit shell
`(k+2,k+3]`, for a prescribed per-zero decay envelope. -/
def actualZetaZeroShellWeightedMass (decay : ℕ → ℝ) (k : ℕ) : ℝ :=
  ∑ z ∈ positiveNontrivialZetaZeroWindowFinset (k + 3),
    if (k + 2 : ℝ) < z.im
    then (zetaZeroMultiplicity z : ℝ) * decay k else 0

/-- The sourced RvM estimate reduces the complete actual-zero tail to the
main-plus-error envelope.  This is the quantitative count-to-tail bridge; it
does not supply the separate interpolation-conditioning estimate. -/
theorem ActualRiemannVonMangoldtBound.exists_actual_shell_tail_lt
    (H : ActualRiemannVonMangoldtBound) (decay : ℕ → ℝ)
    (hdecay : ∀ k, 0 ≤ decay k)
    (hseries : Summable fun k : ℕ ↦
      (riemannVonMangoldtMain (k + 3) +
        riemannVonMangoldtError (k + 3)) * decay k)
    {ε : ℝ} (hε : 0 < ε) :
    ∃ N, Summable (fun k : ℕ ↦
        actualZetaZeroShellWeightedMass decay (k + N)) ∧
      ∑' k : ℕ, actualZetaZeroShellWeightedMass decay (k + N) < ε := by
  rcases H.exists_envelope_tail_lt decay hseries hε with ⟨N, hN⟩
  let A : ℕ → ℝ := actualZetaZeroShellWeightedMass decay
  let E : ℕ → ℝ := fun k ↦
    (riemannVonMangoldtMain (k + 3) +
      riemannVonMangoldtError (k + 3)) * decay k
  have hA_nonneg : ∀ k, 0 ≤ A k := by
    intro k
    dsimp [A, actualZetaZeroShellWeightedMass]
    exact Finset.sum_nonneg fun _ _ ↦ by
      split_ifs
      · exact mul_nonneg (Nat.cast_nonneg _) (hdecay k)
      · exact le_rfl
  have hAE : ∀ k, A k ≤ E k := by
    intro k
    exact H.shell_weighted_decay_le decay hdecay k
  have hEshift : Summable fun k : ℕ ↦ E (k + N) :=
    (summable_nat_add_iff N).mpr hseries
  have hAshift : Summable fun k : ℕ ↦ A (k + N) :=
    hEshift.of_nonneg_of_le (fun k ↦ hA_nonneg (k + N))
      (fun k ↦ hAE (k + N))
  refine ⟨N, hAshift, ?_⟩
  calc
    ∑' k : ℕ, A (k + N) ≤ ∑' k : ℕ, E (k + N) :=
      hAshift.tsum_le_tsum (fun k ↦ hAE (k + N)) hEshift
    _ < ε := by simpa [E, Nat.cast_add] using hN

/-- Actual analytic-multiplicity mass of a test transform in one
positive-height zeta shell, centered at `ρ - 1/2`. -/
def actualZetaZeroTransformShellMass (f : Test) (k : ℕ) : ℝ :=
  ∑ ρ ∈ positiveNontrivialZetaZeroWindowFinset (k + 3),
    if (k + 2 : ℝ) < ρ.im then
      (zetaZeroMultiplicity ρ : ℝ) *
        ‖transform f (ρ - (1 / 2 : ℂ))‖
    else 0

/-- Fourth-power strip decay bounds every actual zeta-zero transform shell by
the corresponding multiplicity-weighted decay mass. -/
theorem actualZetaZeroTransformShellMass_le
    (f : Test) (k : ℕ) :
    actualZetaZeroTransformShellMass f k ≤
      actualZetaZeroShellWeightedMass
        (fun j ↦ stripTransformDecayConstant f 4 /
          ((j : ℝ) + 2) ^ 4) k := by
  dsimp [actualZetaZeroTransformShellMass,
    actualZetaZeroShellWeightedMass]
  apply Finset.sum_le_sum
  intro ρ hρ
  split_ifs with hk
  · have hρdata :=
      (mem_positiveNontrivialZetaZeroWindowFinset.mp hρ)
    have hstrip :
        (ρ - (1 / 2 : ℂ)).re ∈ Set.Icc (-1 / 2 : ℝ) (1 / 2) := by
      norm_num
      constructor <;> linarith [hρdata.2.2.2.1, hρdata.2.2.2.2]
    have hdecay :=
      norm_transform_centered_strip_im_pow_le_explicit f 4 hstrip
    have him : (ρ - (1 / 2 : ℂ)).im = ρ.im := by norm_num
    rw [him, abs_of_pos hρdata.2.1] at hdecay
    have hk0 : (0 : ℝ) ≤ (k : ℝ) + 2 := by positivity
    have hpow :
        ((k : ℝ) + 2) ^ 4 ≤ ρ.im ^ 4 :=
      pow_le_pow_left₀ hk0 hk.le 4
    have hscaled :
        ((k : ℝ) + 2) ^ 4 *
            ‖transform f (ρ - (1 / 2 : ℂ))‖ ≤
          stripTransformDecayConstant f 4 :=
      (mul_le_mul_of_nonneg_right hpow (norm_nonneg _)).trans hdecay
    have htransform :
        ‖transform f (ρ - (1 / 2 : ℂ))‖ ≤
          stripTransformDecayConstant f 4 / ((k : ℝ) + 2) ^ 4 :=
      (le_div_iff₀ (pow_pos (by positivity) 4)).2 <| by
        simpa [mul_comm] using hscaled
    exact mul_le_mul_of_nonneg_left htransform (Nat.cast_nonneg _)
  · exact le_rfl

/-- For every fixed compactly supported smooth test, actual RvM makes its
complete positive-height transform tail smaller than any prescribed positive
margin.  Multiplicity is included in the finite shell sums. -/
theorem ActualRiemannVonMangoldtBound.exists_actual_transform_tail_lt
    (H : ActualRiemannVonMangoldtBound) (f : Test)
    {ε : ℝ} (hε : 0 < ε) :
    ∃ N, Summable (fun k : ℕ ↦
        actualZetaZeroTransformShellMass f (k + N)) ∧
      ∑' k : ℕ, actualZetaZeroTransformShellMass f (k + N) < ε := by
  let C := stripTransformDecayConstant f 4
  let decay : ℕ → ℝ := fun k ↦ C / ((k : ℝ) + 2) ^ 4
  have hC : 0 ≤ C := by
    have h := norm_transform_centered_strip_im_pow_le_explicit
      f 4 (z := (0 : ℂ)) (by norm_num)
    simpa [C] using h
  have hseries : Summable fun k : ℕ ↦
      (riemannVonMangoldtMain (k + 3) +
        riemannVonMangoldtError (k + 3)) * decay k :=
    H.summable_fourthPower_envelope C hC
  rcases H.exists_actual_shell_tail_lt decay
      (fun k ↦ div_nonneg hC (by positivity)) hseries hε with
    ⟨N, hweighted, hweightedTail⟩
  have hmass_nonneg : ∀ k, 0 ≤ actualZetaZeroTransformShellMass f k := by
    intro k
    dsimp [actualZetaZeroTransformShellMass]
    exact Finset.sum_nonneg fun _ _ ↦ by
      split_ifs <;> positivity
  have hmass_le : ∀ k,
      actualZetaZeroTransformShellMass f k ≤
        actualZetaZeroShellWeightedMass decay k := by
    intro k
    exact actualZetaZeroTransformShellMass_le f k
  have hmass : Summable fun k : ℕ ↦
      actualZetaZeroTransformShellMass f (k + N) :=
    hweighted.of_nonneg_of_le (fun k ↦ hmass_nonneg (k + N))
      (fun k ↦ hmass_le (k + N))
  refine ⟨N, hmass, ?_⟩
  exact (hmass.tsum_le_tsum (fun k ↦ hmass_le (k + N)) hweighted).trans_lt
    hweightedTail

/-- In particular, the actual multiplicity-aware distant tail can be made
strictly smaller than any positive selected multiplicity. -/
theorem ActualRiemannVonMangoldtBound.exists_actual_transform_tail_lt_multiplicity
    (H : ActualRiemannVonMangoldtBound) (f : Test)
    (m : ℕ) (hm : 0 < m) :
    ∃ N, Summable (fun k : ℕ ↦
        actualZetaZeroTransformShellMass f (k + N)) ∧
      ∑' k : ℕ, actualZetaZeroTransformShellMass f (k + N) < m :=
  H.exists_actual_transform_tail_lt f (by exact_mod_cast hm)

/-- Riemann--von Mangoldt shell counts turn a pointwise transform envelope
into a quantitative bound for the norm mass of one complete shell. -/
theorem ZetaZeroShellCountObligation.shell_norm_sum_le
    {Z : MultiplicityAwareSymmetricZeros} {S : ZeroOrbitShells}
    (H : ZetaZeroShellCountObligation Z S) {f : Test}
    (decay : ℕ → ℝ) (hdecay : ∀ k, 0 ≤ decay k)
    (hterm : ∀ k i, i ∈ S.shell k →
      ‖symmetricZeroPairTerm Z f i‖ ≤ Z.multiplicity i * decay k)
    (k : ℕ) :
    ∑ i ∈ S.shell k, ‖symmetricZeroPairTerm Z f i‖ ≤
      H.constant * ((k : ℝ) + 2) ^ 2 * decay k := by
  calc
    ∑ i ∈ S.shell k, ‖symmetricZeroPairTerm Z f i‖ ≤
        ∑ i ∈ S.shell k, Z.multiplicity i * decay k :=
      Finset.sum_le_sum fun i hi ↦ hterm k i hi
    _ = (∑ i ∈ S.shell k, Z.multiplicity i : ℕ) * decay k := by
      push_cast
      rw [Finset.sum_mul]
    _ ≤ H.constant * ((k : ℝ) + 2) ^ 2 * decay k :=
      mul_le_mul_of_nonneg_right (H.multiplicity_count_le k) (hdecay k)

/-- The sourced quadratic shell count and a summable transform envelope give
an explicit height cutoff beyond which the total envelope is below any
prescribed positive margin. -/
theorem ZetaZeroShellCountObligation.exists_envelope_tail_lt
    {Z : MultiplicityAwareSymmetricZeros} {S : ZeroOrbitShells}
    (H : ZetaZeroShellCountObligation Z S) (decay : ℕ → ℝ)
    (hseries : Summable fun k : ℕ ↦
      H.constant * ((k : ℝ) + 2) ^ 2 * decay k)
    {ε : ℝ} (hε : 0 < ε) :
    ∃ N, ∑' k : ℕ,
      H.constant * (((k + N : ℕ) : ℝ) + 2) ^ 2 * decay (k + N) < ε := by
  simpa [Nat.cast_add] using
    exists_tsum_nat_add_lt_of_summable hseries hε

set_option maxHeartbeats 800000 in
/-- The complete residual orbit mass is bounded directly by the sourced
count-times-decay shell envelope. -/
theorem ZetaZeroShellCountObligation.residualOrbitNormMass_le
    {Z : MultiplicityAwareSymmetricZeros} {S : ZeroOrbitShells}
    (H : ZetaZeroShellCountObligation Z S) {f : Test} (i : ℕ)
    (decay : ℕ → ℝ)
    (hseries : Summable fun k : ℕ ↦
      H.constant * ((k : ℝ) + 2) ^ 2 * decay k)
    (hdecay : ∀ k, 0 ≤ decay k)
    (hterm : ∀ k j, j ∈ S.shell k →
      ‖symmetricZeroPairTerm Z f j‖ ≤ Z.multiplicity j * decay k) :
    residualOrbitNormMass Z f i ≤
      ∑' k : ℕ, H.constant * ((k : ℝ) + 2) ^ 2 * decay k := by
  let g : ℕ → ℝ := fun j ↦ if j = i then 0 else
    ‖symmetricZeroPairTerm Z f j‖
  let e := Set.sigmaEquiv (fun k ↦ (S.shell k : Set ℕ)) S.unique_shell
  have hnormal : ZeroOrbitNormallyConvergent Z f :=
    H.normalConvergence decay hseries hdecay hterm
  have hg : Summable g := hnormal.residual_summable i
  have hge : Summable (fun p ↦ g (e p)) :=
    e.summable_iff.mpr hg
  calc
    residualOrbitNormMass Z f i = ∑' j, g j := rfl
    _ = ∑' p, g (e p) := (e.tsum_eq g).symm
    _ = ∑' k, ∑' j : {j // j ∈ S.shell k}, g j :=
      hge.tsum_sigma
    _ ≤ ∑' k : ℕ, H.constant * ((k : ℝ) + 2) ^ 2 * decay k := by
      apply hge.sigma.tsum_le_tsum
      · intro k
        rw [tsum_fintype]
        calc
          ∑ j : {j // j ∈ S.shell k}, g j ≤
              ∑ j : {j // j ∈ S.shell k},
                ‖symmetricZeroPairTerm Z f j‖ := by
            apply Finset.sum_le_sum
            intro j hj
            dsimp [g]
            split_ifs <;> simp
          _ = ∑ j ∈ S.shell k, ‖symmetricZeroPairTerm Z f j‖ := by
            simpa using (Finset.sum_attach (S.shell k)
              (fun j ↦ ‖symmetricZeroPairTerm Z f j‖))
          _ ≤ H.constant * ((k : ℝ) + 2) ^ 2 * decay k :=
            H.shell_norm_sum_le decay hdecay hterm k
      · exact hseries

set_option maxHeartbeats 800000 in
/-- Residual-only shell estimate.  The selected orbit is excluded from the
pointwise envelope, avoiding the impossible requirement that a negative term
dominate a majorant which already contains its own norm. -/
theorem ZetaZeroShellCountObligation.residualOrbitNormMass_le_excluding
    {Z : MultiplicityAwareSymmetricZeros} {S : ZeroOrbitShells}
    (H : ZetaZeroShellCountObligation Z S) {f : Test} (i : ℕ)
    (hnormal : ZeroOrbitNormallyConvergent Z f)
    (decay : ℕ → ℝ)
    (hseries : Summable fun k : ℕ ↦
      H.constant * ((k : ℝ) + 2) ^ 2 * decay k)
    (hdecay : ∀ k, 0 ≤ decay k)
    (hterm : ∀ k j, j ∈ S.shell k → j ≠ i →
      ‖symmetricZeroPairTerm Z f j‖ ≤ Z.multiplicity j * decay k) :
    residualOrbitNormMass Z f i ≤
      ∑' k : ℕ, H.constant * ((k : ℝ) + 2) ^ 2 * decay k := by
  let g : ℕ → ℝ := fun j ↦ if j = i then 0 else
    ‖symmetricZeroPairTerm Z f j‖
  let e := Set.sigmaEquiv (fun k ↦ (S.shell k : Set ℕ)) S.unique_shell
  have hg : Summable g := hnormal.residual_summable i
  have hge : Summable (fun p ↦ g (e p)) :=
    e.summable_iff.mpr hg
  calc
    residualOrbitNormMass Z f i = ∑' j, g j := rfl
    _ = ∑' p, g (e p) := (e.tsum_eq g).symm
    _ = ∑' k, ∑' j : {j // j ∈ S.shell k}, g j :=
      hge.tsum_sigma
    _ ≤ ∑' k : ℕ, H.constant * ((k : ℝ) + 2) ^ 2 * decay k := by
      apply hge.sigma.tsum_le_tsum
      · intro k
        rw [tsum_fintype]
        calc
          ∑ j : {j // j ∈ S.shell k}, g j ≤
              ∑ j : {j // j ∈ S.shell k},
                Z.multiplicity j * decay k := by
            apply Finset.sum_le_sum
            intro j hj
            dsimp [g]
            split_ifs with hji
            · exact mul_nonneg (Nat.cast_nonneg _) (hdecay k)
            · exact hterm k j j.property hji
          _ = (∑ j : {j // j ∈ S.shell k},
                (Z.multiplicity j : ℝ)) * decay k := by
            rw [Finset.sum_mul]
          _ = (∑ j ∈ S.shell k, Z.multiplicity j : ℕ) * decay k := by
            push_cast
            congr 1
            simpa using (Finset.sum_attach (S.shell k)
              (fun j ↦ (Z.multiplicity j : ℝ)))
          _ ≤ H.constant * ((k : ℝ) + 2) ^ 2 * decay k :=
            mul_le_mul_of_nonneg_right (H.multiplicity_count_le k) (hdecay k)
      · exact hseries

/-- Corrected infinite separation criterion with a residual-only envelope. -/
theorem ZetaZeroShellCountObligation.symmetricZeroRegularized_re_neg_excluding
    {Z : MultiplicityAwareSymmetricZeros} {S : ZeroOrbitShells}
    (H : ZetaZeroShellCountObligation Z S) (f : Test) (i : ℕ)
    (hnormal : ZeroOrbitNormallyConvergent Z f)
    (decay : ℕ → ℝ)
    (hseries : Summable fun k : ℕ ↦
      H.constant * ((k : ℝ) + 2) ^ 2 * decay k)
    (hdecay : ∀ k, 0 ≤ decay k)
    (hterm : ∀ k j, j ∈ S.shell k → j ≠ i →
      ‖symmetricZeroPairTerm Z f j‖ ≤ Z.multiplicity j * decay k)
    (htarget : (symmetricZeroPairTerm Z f i).re <
      -(∑' k : ℕ, H.constant * ((k : ℝ) + 2) ^ 2 * decay k)) :
    (symmetricZeroRegularized Z f).re < 0 := by
  apply symmetricZeroRegularized_re_neg_of_selected_dominates Z f i hnormal
  exact htarget.trans_le (neg_le_neg
    (H.residualOrbitNormMass_le_excluding i hnormal decay
      hseries hdecay hterm))

/-- Infinite matrix-free separator theorem.  Local neighbors are removed by
finite-difference zero factors; only the residual shell envelope must be
smaller than the exact selected mass `2 * multiplicity`. -/
theorem ZetaZeroShellCountObligation.localPairSeparator_regularized_re_neg
    {Z : MultiplicityAwareSymmetricZeros} {S : ZeroOrbitShells}
    (H : ZetaZeroShellCountObligation Z S) (i : ℕ)
    (a : ℝ) (nearby : List ℂ)
    (hfixed : Z.fixedBySymmetry i = false)
    (hz : LocalSeparatorGap a (Z.representative i)
      (-star (Z.representative i) :: nearby))
    (hpartner : LocalSeparatorGap a (-star (Z.representative i))
      (Z.representative i :: nearby))
    (hnormal : ZeroOrbitNormallyConvergent Z
      (convolution (localPairSeparator a (Z.representative i) nearby)
        (involution (localPairSeparator a (Z.representative i) nearby))))
    (decay : ℕ → ℝ)
    (hseries : Summable fun k : ℕ ↦
      H.constant * ((k : ℝ) + 2) ^ 2 * decay k)
    (hdecay : ∀ k, 0 ≤ decay k)
    (hterm : ∀ k j, j ∈ S.shell k → j ≠ i →
      ‖symmetricZeroPairTerm Z
        (convolution (localPairSeparator a (Z.representative i) nearby)
          (involution (localPairSeparator a (Z.representative i) nearby))) j‖ ≤
        Z.multiplicity j * decay k)
    (htail : (∑' k : ℕ,
      H.constant * ((k : ℝ) + 2) ^ 2 * decay k) <
        2 * (Z.multiplicity i : ℝ)) :
    (symmetricZeroRegularized Z
      (convolution (localPairSeparator a (Z.representative i) nearby)
        (involution (localPairSeparator a (Z.representative i) nearby)))).re < 0 := by
  apply H.symmetricZeroRegularized_re_neg_excluding
    _ i hnormal decay hseries hdecay hterm
  rw [localPairSeparator_selectedPairTerm Z i a nearby hfixed hz hpartner]
  norm_num
  exact htail

/-- Conditional contradiction with a genuine Guinand--Weil identity and
nonnegative arithmetic side, specialized to the constructed local square. -/
theorem ZetaZeroShellCountObligation.localPairSeparator_contradicts_formula
    {Z : MultiplicityAwareSymmetricZeros} {S : ZeroOrbitShells}
    (H : ZetaZeroShellCountObligation Z S) (i : ℕ)
    (a : ℝ) (nearby : List ℂ)
    (hfixed : Z.fixedBySymmetry i = false)
    (hz : LocalSeparatorGap a (Z.representative i)
      (-star (Z.representative i) :: nearby))
    (hpartner : LocalSeparatorGap a (-star (Z.representative i))
      (Z.representative i :: nearby))
    (hnormal : ZeroOrbitNormallyConvergent Z
      (convolution (localPairSeparator a (Z.representative i) nearby)
        (involution (localPairSeparator a (Z.representative i) nearby))))
    (decay : ℕ → ℝ)
    (hseries : Summable fun k : ℕ ↦
      H.constant * ((k : ℝ) + 2) ^ 2 * decay k)
    (hdecay : ∀ k, 0 ≤ decay k)
    (hterm : ∀ k j, j ∈ S.shell k → j ≠ i →
      ‖symmetricZeroPairTerm Z
        (convolution (localPairSeparator a (Z.representative i) nearby)
          (involution (localPairSeparator a (Z.representative i) nearby))) j‖ ≤
        Z.multiplicity j * decay k)
    (htail : (∑' k : ℕ,
      H.constant * ((k : ℝ) + 2) ^ 2 * decay k) <
        2 * (Z.multiplicity i : ℝ))
    (arithmeticSide : ℂ)
    (hformula : symmetricZeroRegularized Z
      (convolution (localPairSeparator a (Z.representative i) nearby)
        (involution (localPairSeparator a (Z.representative i) nearby))) =
      arithmeticSide)
    (harithmetic : 0 ≤ arithmeticSide.re) :
    False := by
  have hneg := H.localPairSeparator_regularized_re_neg i a nearby
    hfixed hz hpartner hnormal decay hseries hdecay hterm htail
  rw [hformula] at hneg
  linarith

/-- Actual finite-to-limit off-critical separation criterion: a sourced shell
count, a summable rapid-decay envelope, and one selected negative orbit imply
negative total regularized spectral value.  Finite interpolation is used to
make the envelope small on the finitely many unselected low shells. -/
theorem ZetaZeroShellCountObligation.symmetricZeroRegularized_re_neg
    {Z : MultiplicityAwareSymmetricZeros} {S : ZeroOrbitShells}
    (H : ZetaZeroShellCountObligation Z S) (f : Test) (i : ℕ)
    (decay : ℕ → ℝ)
    (hseries : Summable fun k : ℕ ↦
      H.constant * ((k : ℝ) + 2) ^ 2 * decay k)
    (hdecay : ∀ k, 0 ≤ decay k)
    (hterm : ∀ k j, j ∈ S.shell k →
      ‖symmetricZeroPairTerm Z f j‖ ≤ Z.multiplicity j * decay k)
    (htarget : (symmetricZeroPairTerm Z f i).re <
      -(∑' k : ℕ, H.constant * ((k : ℝ) + 2) ^ 2 * decay k)) :
    (symmetricZeroRegularized Z f).re < 0 := by
  have hnormal : ZeroOrbitNormallyConvergent Z f :=
    H.normalConvergence decay hseries hdecay hterm
  apply symmetricZeroRegularized_re_neg_of_selected_dominates Z f i hnormal
  exact htarget.trans_le (neg_le_neg
    (H.residualOrbitNormMass_le i decay hseries hdecay hterm))

/-- A fourth-power transform envelope is summable against the quadratic shell
count weakening of Riemann--von Mangoldt. -/
theorem summable_quadratic_shell_fourth_power_decay (C : ℝ) :
    Summable fun k : ℕ ↦
      C * ((k : ℝ) + 2) ^ 2 * (1 / ((k : ℝ) + 2) ^ 4) := by
  have hp : Summable fun k : ℕ ↦ 1 / (k : ℝ) ^ 2 :=
    (Real.summable_one_div_nat_pow (p := 2)).2 (by norm_num)
  have hp' : Summable fun k : ℕ ↦ 1 / ((k + 2 : ℕ) : ℝ) ^ 2 :=
    (summable_nat_add_iff 2).2 hp
  apply (hp'.mul_left C).congr
  intro k
  norm_num [Nat.cast_add]
  field_simp

/-- The sourced quadratic shell bound reduces zeta-zero normal convergence to
a uniform fourth-power vertical transform estimate on each shell. -/
theorem ZetaZeroShellCountObligation.normalConvergence_of_fourthPower
    {Z : MultiplicityAwareSymmetricZeros} {S : ZeroOrbitShells}
    (H : ZetaZeroShellCountObligation Z S) {f : Test}
    (hterm : ∀ k i, i ∈ S.shell k →
      ‖symmetricZeroPairTerm Z f i‖ ≤
        Z.multiplicity i * (1 / ((k : ℝ) + 2) ^ 4)) :
    ZeroOrbitNormallyConvergent Z f := by
  apply H.normalConvergence (fun k ↦ 1 / ((k : ℝ) + 2) ^ 4)
  · exact summable_quadratic_shell_fourth_power_decay H.constant
  · intro k
    positivity
  · exact hterm

/-- An arbitrary sequence of finite orbit windows exhausting all indices.
Because indices already represent complete symmetry orbits, no extra
partner-closure condition is required. -/
structure SymmetricZeroWindows where
  window : ℕ → Finset ℕ
  exhausts : Tendsto window atTop atTop

def symmetricZeroWindowSum
    (Z : MultiplicityAwareSymmetricZeros) (W : SymmetricZeroWindows)
    (f : Test) (N : ℕ) : ℂ :=
  ∑ n ∈ W.window N, symmetricZeroPairTerm Z f n

/-- Every exhausting finite-window scheme has the same limit under normal
convergence. -/
theorem symmetricZeroWindowSum_tendsto
    (Z : MultiplicityAwareSymmetricZeros) (W : SymmetricZeroWindows)
    (f : Test) (hnormal : ZeroOrbitNormallyConvergent Z f) :
    Tendsto (symmetricZeroWindowSum Z W f) atTop
      (𝓝 (symmetricZeroRegularized Z f)) := by
  exact hnormal.summable.hasSum.comp W.exhausts

/-- Concrete zeta-zero regularization consequence under the sourced
multiplicity shell-count obligation and an explicit shell-uniform transform
envelope.  The zero-count field is not asserted by this theorem. -/
theorem symmetricZeroWindowSum_tendsto_of_zetaShellCount
    (Z : MultiplicityAwareSymmetricZeros) (S : ZeroOrbitShells)
    (H : ZetaZeroShellCountObligation Z S) (W : SymmetricZeroWindows)
    (f : Test) (decay : ℕ → ℝ)
    (hseries : Summable fun k : ℕ ↦
      H.constant * ((k : ℝ) + 2) ^ 2 * decay k)
    (hdecay : ∀ k, 0 ≤ decay k)
    (hterm : ∀ k i, i ∈ S.shell k →
      ‖symmetricZeroPairTerm Z f i‖ ≤ Z.multiplicity i * decay k) :
    Tendsto (symmetricZeroWindowSum Z W f) atTop
      (𝓝 (symmetricZeroRegularized Z f)) := by
  exact symmetricZeroWindowSum_tendsto Z W f <|
    H.normalConvergence decay hseries hdecay hterm

/-- Two arbitrary exhausting symmetric-window schemes become asymptotically
independent under normal convergence. -/
theorem symmetricZeroWindows_difference_tendsto_zero
    (Z : MultiplicityAwareSymmetricZeros) (W V : SymmetricZeroWindows)
    (f : Test) (hnormal : ZeroOrbitNormallyConvergent Z f) :
    Tendsto
      (fun N ↦ symmetricZeroWindowSum Z W f N -
        symmetricZeroWindowSum Z V f N)
      atTop (𝓝 0) := by
  simpa using
    (symmetricZeroWindowSum_tendsto Z W f hnormal).sub
      (symmetricZeroWindowSum_tendsto Z V f hnormal)

/-- Concrete data for a finite-zero/finite-prime explicit-formula stage.
`IsExactAt` below records the analytic equality; no convergence is built in. -/
structure FiniteFormulaData (ι κ : Type*) [Fintype ι] [Fintype κ] where
  zeros : ι → ℂ
  primeWeight : κ → ℂ
  primeLogLocation : κ → ℝ
  poleTerm : Test → ℂ
  archimedeanTerm : Test → ℂ

def FiniteFormulaData.arithmeticSide
    {ι κ : Type*} [Fintype ι] [Fintype κ] (D : FiniteFormulaData ι κ) (f : Test) : ℂ :=
  D.poleTerm f - finitePrimePowerSum D.primeWeight D.primeLogLocation f -
    D.archimedeanTerm f

/-- The exact, deliberately local finite explicit-formula hypothesis. -/
def FiniteFormulaData.IsExactAt
    {ι κ : Type*} [Fintype ι] [Fintype κ] (D : FiniteFormulaData ι κ) (f : Test) : Prop :=
  finiteZeroSum D.zeros f = D.arithmeticSide f

/-- An exact finite Guinand--Weil stage transfers a negative separating
spectral value to the complete finite arithmetic side, including its
prime-power, pole, and archimedean terms. -/
theorem FiniteFormulaData.arithmeticSide_re_neg_of_exact
    {ι κ : Type*} [Fintype ι] [Fintype κ]
    (D : FiniteFormulaData ι κ) (f : Test)
    (hexact : D.IsExactAt f) (hneg : (finiteZeroSum D.zeros f).re < 0) :
    (D.arithmeticSide f).re < 0 := by
  rw [← hexact]
  exact hneg

/-- A restricted admissible datum in the honest half-plane `re s > 1`.
The prime side is the actual Mathlib-normalized smoothed von Mangoldt series.
The zero, pole, and archimedean series carry explicit summability hypotheses,
and `stageFormula` records the finite identity to be established by analysis
for a concrete subclass. -/
structure RestrictedExplicitFormulaData
    (Z : MultiplicityAwareSymmetricZeros) (s : ℂ) where
  test : Test
  primeWeight : PrimeSmoothingWeight
  poleTerm : ℕ → ℂ
  archimedeanTerm : ℕ → ℂ
  zeroSummable : Summable (symmetricZeroPairTerm Z test)
  poleSummable : Summable poleTerm
  archimedeanSummable : Summable archimedeanTerm
  stageFormula : ∀ N,
    symmetricZeroPartialSum Z test N =
      (∑ n ∈ Finset.range N, poleTerm n) -
      (∑ n ∈ Finset.range N, smoothedVonMangoldtTerm s primeWeight n) -
      (∑ n ∈ Finset.range N, archimedeanTerm n)

/-- On the restricted admissible subclass, the finite formula and the stated
summability obligations imply the limiting explicit formula.  No universal
test-space criterion or continuation to the critical strip is inferred. -/
theorem RestrictedExplicitFormulaData.explicitFormula
    {Z : MultiplicityAwareSymmetricZeros} {s : ℂ}
    (D : RestrictedExplicitFormulaData Z s) (hs : 1 < s.re) :
    symmetricZeroRegularized Z D.test =
      (∑' n, D.poleTerm n) -
      smoothedVonMangoldtSeries s D.primeWeight -
      (∑' n, D.archimedeanTerm n) := by
  apply tendsto_nhds_unique (symmetricZeroPartialSum_tendsto Z D.test D.zeroSummable)
  have hpole := D.poleSummable.hasSum.tendsto_sum_nat
  have hprime := (smoothedVonMangoldt_summable hs D.primeWeight).hasSum.tendsto_sum_nat
  have harch := D.archimedeanSummable.hasSum.tendsto_sum_nat
  exact ((hpole.sub hprime).sub harch).congr' <|
    Eventually.of_forall fun N ↦ (D.stageFormula N).symm

/-- Sharper restricted Guinand--Weil limit package.  Its prime target is the
actual negative zeta logarithmic derivative, its archimedean target is the
actual `Γℝ` logarithmic derivative, and its zero windows are
multiplicity-aware.  Only the finite stage formula and analytic convergence
inputs remain fields. -/
structure RestrictedGuinandWeilLimitData
    (Z : MultiplicityAwareSymmetricZeros) (s : ℂ) where
  test : Test
  zeroWindows : SymmetricZeroWindows
  zeroNormal : ZeroOrbitNormallyConvergent Z test
  primeSmoothing : ℕ → PrimeSmoothingWeight
  primeSmoothing_tendsto :
    ∀ n, Tendsto (fun k ↦ (primeSmoothing k).coeff n) atTop (𝓝 1)
  poleStage : ℕ → ℂ
  poleLimit : ℂ
  pole_tendsto : Tendsto poleStage atTop (𝓝 poleLimit)
  archimedean : ArchimedeanGammaApproximation s
  stageFormula : ∀ N,
    symmetricZeroWindowSum Z zeroWindows test N =
      poleStage N -
      smoothedVonMangoldtSeries s (primeSmoothing N) -
      archimedean.stage N

/-- All algebraic and limit-combination parts of the restricted
Guinand--Weil formula.  This theorem does not provide the finite-stage
identity, normal convergence of zeta zeros, or a Gamma approximation. -/
theorem RestrictedGuinandWeilLimitData.explicitFormula
    {Z : MultiplicityAwareSymmetricZeros} {s : ℂ}
    (D : RestrictedGuinandWeilLimitData Z s) (hs : 1 < s.re) :
    symmetricZeroRegularized Z D.test =
      D.poleLimit -
      (-deriv riemannZeta s / riemannZeta s) -
      archimedeanLogDeriv s := by
  apply tendsto_nhds_unique
    (symmetricZeroWindowSum_tendsto Z D.zeroWindows D.test D.zeroNormal)
  exact
    ((D.pole_tendsto.sub
      (smoothedVonMangoldtSeries_tendsto_logDeriv hs D.primeSmoothing
        D.primeSmoothing_tendsto)).sub D.archimedean.tendsto).congr' <|
      Eventually.of_forall fun N ↦ (D.stageFormula N).symm

/-- A sharper restricted package replacing abstract zero normal convergence by
finite shell counts and a summable count-times-decay majorant. -/
structure RestrictedGuinandWeilShellData
    (Z : MultiplicityAwareSymmetricZeros) (s : ℂ) where
  test : Test
  zeroWindows : SymmetricZeroWindows
  zeroShells : ZeroOrbitShells
  zeroDecay : ℕ → ℝ
  zeroCountDecaySummable :
    Summable fun k ↦ (zeroShells.shell k).card * zeroDecay k
  zeroTermBound : ∀ k i, i ∈ zeroShells.shell k →
    ‖symmetricZeroPairTerm Z test i‖ ≤ zeroDecay k
  primeSmoothing : ℕ → PrimeSmoothingWeight
  primeSmoothing_tendsto :
    ∀ n, Tendsto (fun k ↦ (primeSmoothing k).coeff n) atTop (𝓝 1)
  poleStage : ℕ → ℂ
  poleLimit : ℂ
  pole_tendsto : Tendsto poleStage atTop (𝓝 poleLimit)
  archimedean : ArchimedeanGammaApproximation s
  stageFormula : ∀ N,
    symmetricZeroWindowSum Z zeroWindows test N =
      poleStage N -
      smoothedVonMangoldtSeries s (primeSmoothing N) -
      archimedean.stage N

/-- Strongest restricted explicit formula currently obtained: shell-count and
transform-decay hypotheses discharge zero regularization, while the existing
prime, pole, and archimedean limits discharge every limit-combination step. -/
theorem RestrictedGuinandWeilShellData.explicitFormula
    {Z : MultiplicityAwareSymmetricZeros} {s : ℂ}
    (D : RestrictedGuinandWeilShellData Z s) (hs : 1 < s.re) :
    symmetricZeroRegularized Z D.test =
      D.poleLimit -
      (-deriv riemannZeta s / riemannZeta s) -
      archimedeanLogDeriv s := by
  let L : RestrictedGuinandWeilLimitData Z s :=
    { test := D.test
      zeroWindows := D.zeroWindows
      zeroNormal := ZeroOrbitNormallyConvergent.of_shell_count_decay
        D.zeroShells D.zeroDecay D.zeroCountDecaySummable D.zeroTermBound
      primeSmoothing := D.primeSmoothing
      primeSmoothing_tendsto := D.primeSmoothing_tendsto
      poleStage := D.poleStage
      poleLimit := D.poleLimit
      pole_tendsto := D.pole_tendsto
      archimedean := D.archimedean
      stageFormula := D.stageFormula }
  exact L.explicitFormula hs

/-- Spectral energy of finitely many ordinates on the critical line. -/
def finiteCriticalEnergy {ι : Type*} [Fintype ι] (ordinate : ι → ℝ) (f : Test) : ℝ :=
  ∑ i, normSq (transform f (ordinate i * I))

theorem finiteCriticalEnergy_nonneg
    {ι : Type*} [Fintype ι] (ordinate : ι → ℝ) (f : Test) :
    0 ≤ finiteCriticalEnergy ordinate f := by
  exact Finset.sum_nonneg fun i _ ↦ normSq_nonneg _

/-- The critical-line convolution-square transform factorization is fully
proved from Mathlib's Fourier convolution theorem. -/
theorem transform_convolution_involution_imaginary
    (f : Test) (ξ : ℝ) :
    transform (convolution f (involution f)) (-2 * Real.pi * ξ * I) =
      (normSq (transform f (-2 * Real.pi * ξ * I)) : ℂ) := by
  rw [transform_imaginary_eq_fourier]
  change 𝓕 (fun x : ℝ ↦ convolution f (involution f) x) ξ = _
  have hf : Integrable (f : ℝ → ℂ) :=
    f.continuous.integrable_of_hasCompactSupport f.hasCompactSupport
  have hi : Integrable (involution f : ℝ → ℂ) :=
    (involution f).continuous.integrable_of_hasCompactSupport
      (involution f).hasCompactSupport
  have hconv :
      (fun x : ℝ ↦ convolution f (involution f) x) =
        MeasureTheory.convolution f (involution f)
          (ContinuousLinearMap.mul ℂ ℂ) volume := by
    funext x
    apply integral_congr_ae
    filter_upwards with t
    rfl
  rw [hconv, Real.fourier_mul_convolution_eq hf hi f.continuous
    (involution f).continuous]
  have htf :
      𝓕 (f : ℝ → ℂ) ξ = transform f (-2 * Real.pi * ξ * I) := by
    calc
      _ = 𝓕 (toSchwartz f) ξ := by congr 2
      _ = _ := (transform_imaginary_eq_fourier f ξ).symm
  have hti :
      𝓕 (involution f : ℝ → ℂ) ξ =
        transform (involution f) (-2 * Real.pi * ξ * I) := by
    calc
      _ = 𝓕 (toSchwartz (involution f)) ξ := by congr 2
      _ = _ := (transform_imaginary_eq_fourier (involution f) ξ).symm
  rw [htf, hti, transform_involution_imaginary]
  simpa only [starRingEnd_apply] using
    Complex.mul_conj (transform f (-2 * Real.pi * ξ * I))

/-- Equivalent pure-imaginary parametrization of the proved
convolution-square factorization. -/
theorem transform_convolution_involution_pureImaginary
    (f : Test) (y : ℝ) :
    transform (convolution f (involution f)) (y * I) =
      (normSq (transform f (y * I)) : ℂ) := by
  have h := transform_convolution_involution_imaginary
    f (-y / (2 * Real.pi))
  have hz :
      -2 * (Real.pi : ℂ) * ((-y / (2 * Real.pi) : ℝ) : ℂ) * I =
        (y : ℂ) * I := by
    push_cast
    field_simp [Real.pi_ne_zero]
  rwa [hz] at h

/-- Exact finite positivity identity under the local convolution-transform
factorization hypothesis.  This isolates the analytic theorem still needed
to derive the hypothesis from Fubini and the chosen normalization. -/
theorem finiteZeroSum_convolution_eq_energy
    {ι : Type*} [Fintype ι] (ordinate : ι → ℝ) (f : Test)
    (hfactor : ∀ i,
      transform (convolution f (involution f)) (ordinate i * I) =
        (normSq (transform f (ordinate i * I)) : ℂ)) :
    (finiteZeroSum (fun i ↦ ordinate i * I) (convolution f (involution f))).re =
      finiteCriticalEnergy ordinate f := by
  simp only [finiteZeroSum, finiteCriticalEnergy]
  rw [Complex.re_sum]
  apply Finset.sum_congr rfl
  intro i _
  rw [hfactor i]
  simp

theorem finiteZeroSum_convolution_nonneg
    {ι : Type*} [Fintype ι] (ordinate : ι → ℝ) (f : Test)
    (hfactor : ∀ i,
      transform (convolution f (involution f)) (ordinate i * I) =
        (normSq (transform f (ordinate i * I)) : ℂ)) :
    0 ≤ (finiteZeroSum (fun i ↦ ordinate i * I)
      (convolution f (involution f))).re := by
  rw [finiteZeroSum_convolution_eq_energy ordinate f hfactor]
  exact finiteCriticalEnergy_nonneg ordinate f

/-- The finite critical-line energy identity no longer needs a factorization
hypothesis: it follows from the proved Fourier theorem. -/
theorem finiteZeroSum_convolution_eq_energy_exact
    {ι : Type*} [Fintype ι] (ordinate : ι → ℝ) (f : Test) :
    (finiteZeroSum (fun i ↦ ordinate i * I)
      (convolution f (involution f))).re =
      finiteCriticalEnergy ordinate f :=
  finiteZeroSum_convolution_eq_energy ordinate f fun i ↦
    transform_convolution_involution_pureImaginary f (ordinate i)

theorem finiteZeroSum_convolution_nonneg_exact
    {ι : Type*} [Fintype ι] (ordinate : ι → ℝ) (f : Test) :
    0 ≤ (finiteZeroSum (fun i ↦ ordinate i * I)
      (convolution f (involution f))).re := by
  rw [finiteZeroSum_convolution_eq_energy_exact]
  exact finiteCriticalEnergy_nonneg ordinate f

/-- Restricted infinite positivity criterion: if every modeled symmetry orbit
is fixed (hence lies on the centered critical line) and the convolution-square
zero series is normally convergent, then its regularized real part is
nonnegative.  This is one direction only and does not assert RH. -/
theorem symmetricZeroRegularized_convolution_nonneg
    (Z : MultiplicityAwareSymmetricZeros) (f : Test)
    (hfixed : ∀ n, Z.fixedBySymmetry n = true)
    (hnormal : ZeroOrbitNormallyConvergent Z (convolution f (involution f))) :
    0 ≤ (symmetricZeroRegularized Z (convolution f (involution f))).re := by
  rw [symmetricZeroRegularized, Complex.re_tsum hnormal.summable]
  apply tsum_nonneg
  intro n
  have hzfixed := Z.fixed_spec n (hfixed n)
  have hzre : (Z.representative n).re = 0 := by
    have : (Z.representative n).re = -(Z.representative n).re := by
      simpa using congrArg Complex.re hzfixed
    linarith
  have hz : Z.representative n = (Z.representative n).im * I := by
    apply Complex.ext
    · simp [hzre]
    · simp
  rw [symmetricZeroPairTerm, if_pos (hfixed n), hz,
    transform_convolution_involution_pureImaginary]
  simpa using
    (mul_nonneg (show 0 ≤ (Z.multiplicity n : ℝ) from Nat.cast_nonneg _)
      (normSq_nonneg (transform f ((Z.representative n).im * I))))

/-- The orbit metadata need not mark critical-line points as fixed:
if every representative is purely imaginary, both the fixed and paired
branches are nonnegative (the paired branch is twice the same energy). -/
theorem symmetricZeroRegularized_convolution_nonneg_of_re_zero
    (Z : MultiplicityAwareSymmetricZeros) (f : Test)
    (hre : ∀ n, (Z.representative n).re = 0)
    (hnormal : ZeroOrbitNormallyConvergent Z (convolution f (involution f))) :
    0 ≤ (symmetricZeroRegularized Z (convolution f (involution f))).re := by
  rw [symmetricZeroRegularized, Complex.re_tsum hnormal.summable]
  apply tsum_nonneg
  intro n
  have hz : Z.representative n = (Z.representative n).im * I := by
    apply Complex.ext
    · simp [hre n]
    · simp
  have hpartner : -star (Z.representative n) = Z.representative n := by
    rw [hz]
    apply Complex.ext <;> simp
  rw [symmetricZeroPairTerm]
  split_ifs
  · rw [hz, transform_convolution_involution_pureImaginary]
    simpa using
      (mul_nonneg (show 0 ≤ (Z.multiplicity n : ℝ) from Nat.cast_nonneg _)
        (normSq_nonneg (transform f ((Z.representative n).im * I))))
  · rw [hpartner, hz, transform_convolution_involution_pureImaginary]
    have hm : 0 ≤ (Z.multiplicity n : ℝ) := Nat.cast_nonneg _
    have he : 0 ≤ normSq (transform f ((Z.representative n).im * I)) :=
      normSq_nonneg _
    simpa using mul_nonneg hm (add_nonneg he he)

/-- Positivity of the normally convergent zero-orbit functional on every
convolution square.  This isolates the spectral part of the converse Weil
criterion from the prime/pole/Gamma identity. -/
def ZeroOrbitConvolutionSquarePositive
    (Z : MultiplicityAwareSymmetricZeros) : Prop :=
  ∀ f : Test, ZeroOrbitNormallyConvergent Z (convolution f (involution f)) →
    0 ≤ (symmetricZeroRegularized Z (convolution f (involution f))).re

/-- Exact all-test separation input needed for the converse: every modeled
off-critical orbit must be detectable by a normally convergent convolution
square with negative total spectral value.  This is deliberately stronger
than point-separation of transforms, because all other zero orbits remain in
the regularized sum. -/
def ConvolutionSquaresSeparateOffCriticalOrbits
    (Z : MultiplicityAwareSymmetricZeros) : Prop :=
  ∀ i : ℕ, (Z.representative i).re ≠ 0 →
    ∃ f : Test,
      ZeroOrbitNormallyConvergent Z (convolution f (involution f)) ∧
        (symmetricZeroRegularized Z (convolution f (involution f))).re < 0

/-- The nondegenerate separation condition only asks to detect orbits carrying
positive multiplicity. -/
def ConvolutionSquaresSeparatePositiveOffCriticalOrbits
    (Z : MultiplicityAwareSymmetricZeros) : Prop :=
  ∀ i : ℕ, 0 < Z.multiplicity i → (Z.representative i).re ≠ 0 →
    ∃ f : Test,
      ZeroOrbitNormallyConvergent Z (convolution f (involution f)) ∧
        (symmetricZeroRegularized Z (convolution f (involution f))).re < 0

/-- Exact quantitative diagonal-selection condition remaining after finite
interpolation: the selected negative square must dominate one sourced
summable shell envelope. -/
def RVMEnvelopeSeparatesPositiveOffCriticalOrbits
    (Z : MultiplicityAwareSymmetricZeros) (S : ZeroOrbitShells)
    (H : ZetaZeroShellCountObligation Z S) : Prop :=
  ∀ i : ℕ, 0 < Z.multiplicity i → (Z.representative i).re ≠ 0 →
    ∃ f : Test, ∃ decay : ℕ → ℝ,
      Summable (fun k : ℕ ↦
        H.constant * ((k : ℝ) + 2) ^ 2 * decay k) ∧
      (∀ k, 0 ≤ decay k) ∧
      (∀ k j, j ∈ S.shell k →
        ‖symmetricZeroPairTerm Z (convolution f (involution f)) j‖ ≤
          Z.multiplicity j * decay k) ∧
      (symmetricZeroPairTerm Z (convolution f (involution f)) i).re <
        -(∑' k : ℕ, H.constant * ((k : ℝ) + 2) ^ 2 * decay k)

/-- The quantitative RvM envelope condition produces actual infinite
off-critical convolution-square separators. -/
theorem ZetaZeroShellCountObligation.convolutionSquaresSeparate
    {Z : MultiplicityAwareSymmetricZeros} {S : ZeroOrbitShells}
    (H : ZetaZeroShellCountObligation Z S)
    (hsep : RVMEnvelopeSeparatesPositiveOffCriticalOrbits Z S H) :
    ConvolutionSquaresSeparatePositiveOffCriticalOrbits Z := by
  intro i hmult hoff
  rcases hsep i hmult hoff with
    ⟨f, decay, hseries, hdecay, hterm, htarget⟩
  refine ⟨f, H.normalConvergence decay hseries hdecay hterm, ?_⟩
  exact H.symmetricZeroRegularized_re_neg
    (convolution f (involution f)) i decay hseries hdecay hterm htarget

/-- The converse spectral implication, with its separation hypothesis made
explicit: positivity forces every modeled representative onto the centered
critical line. -/
theorem representative_re_zero_of_convolutionSquarePositive
    (Z : MultiplicityAwareSymmetricZeros)
    (hpositive : ZeroOrbitConvolutionSquarePositive Z)
    (hseparate : ConvolutionSquaresSeparateOffCriticalOrbits Z) :
    ∀ i, (Z.representative i).re = 0 := by
  intro i
  by_contra hi
  rcases hseparate i hi with ⟨f, hnormal, hneg⟩
  exact (not_lt_of_ge (hpositive f hnormal)) hneg

theorem representative_re_zero_of_positiveMultiplicity_separation
    (Z : MultiplicityAwareSymmetricZeros)
    (hmult : ∀ i, 0 < Z.multiplicity i)
    (hpositive : ZeroOrbitConvolutionSquarePositive Z)
    (hseparate : ConvolutionSquaresSeparatePositiveOffCriticalOrbits Z) :
    ∀ i, (Z.representative i).re = 0 := by
  intro i
  by_contra hi
  rcases hseparate i (hmult i) hi with ⟨f, hnormal, hneg⟩
  exact (not_lt_of_ge (hpositive f hnormal)) hneg

/-- Exact converse assembly from a genuine Guinand--Weil identity, arithmetic
positivity, and the quantitative RvM envelope separator.  The theorem adds no
formula or zero-count assumption silently. -/
theorem representative_re_zero_of_explicitFormula_and_rvmEnvelope
    (Z : MultiplicityAwareSymmetricZeros) (S : ZeroOrbitShells)
    (H : ZetaZeroShellCountObligation Z S)
    (hmult : ∀ i, 0 < Z.multiplicity i)
    (arithmeticSide : Test → ℂ)
    (hformula : ∀ f : Test,
      ZeroOrbitNormallyConvergent Z (convolution f (involution f)) →
        symmetricZeroRegularized Z (convolution f (involution f)) =
          arithmeticSide f)
    (harithmetic : ∀ f : Test, 0 ≤ (arithmeticSide f).re)
    (hsep : RVMEnvelopeSeparatesPositiveOffCriticalOrbits Z S H) :
    ∀ i, (Z.representative i).re = 0 := by
  apply representative_re_zero_of_positiveMultiplicity_separation Z hmult
  · intro f hnormal
    rw [hformula f hnormal]
    exact harithmetic f
  · exact H.convolutionSquaresSeparate hsep

/-- Exact conditional converse theorem for the actual zeta function.  The
formula/positivity and quantitative separator hypotheses first force every
modeled representative onto the centered critical line; exact orbit
completeness then turns this into Mathlib's global `RiemannHypothesis`. -/
theorem riemannHypothesis_of_exactActualOrbits_explicitFormula_and_rvmEnvelope
    (Z : MultiplicityAwareSymmetricZeros) (S : ZeroOrbitShells)
    (Hexact : ExactActualZetaOrbitModel Z)
    (hloc : NontrivialZetaZerosInCriticalStrip)
    (H : ZetaZeroShellCountObligation Z S)
    (hmult : ∀ i, 0 < Z.multiplicity i)
    (arithmeticSide : Test → ℂ)
    (hformula : ∀ f : Test,
      ZeroOrbitNormallyConvergent Z (convolution f (involution f)) →
        symmetricZeroRegularized Z (convolution f (involution f)) =
          arithmeticSide f)
    (harithmetic : ∀ f : Test, 0 ≤ (arithmeticSide f).re)
    (hsep : RVMEnvelopeSeparatesPositiveOffCriticalOrbits Z S H) :
    RiemannHypothesis :=
  Hexact.riemannHypothesis_of_representatives hloc
    (representative_re_zero_of_explicitFormula_and_rvmEnvelope
      Z S H hmult arithmeticSide hformula harithmetic hsep)

/-- The same exact converse with the former strip-classification hypothesis
discharged by `nontrivialZetaZerosInCriticalStrip`. -/
theorem riemannHypothesis_of_exactActualOrbits_formula_rvm
    (Z : MultiplicityAwareSymmetricZeros) (S : ZeroOrbitShells)
    (Hexact : ExactActualZetaOrbitModel Z)
    (H : ZetaZeroShellCountObligation Z S)
    (hmult : ∀ i, 0 < Z.multiplicity i)
    (arithmeticSide : Test → ℂ)
    (hformula : ∀ f : Test,
      ZeroOrbitNormallyConvergent Z (convolution f (involution f)) →
        symmetricZeroRegularized Z (convolution f (involution f)) =
          arithmeticSide f)
    (harithmetic : ∀ f : Test, 0 ≤ (arithmeticSide f).re)
    (hsep : RVMEnvelopeSeparatesPositiveOffCriticalOrbits Z S H) :
    RiemannHypothesis :=
  riemannHypothesis_of_exactActualOrbits_explicitFormula_and_rvmEnvelope
    Z S Hexact nontrivialZetaZerosInCriticalStrip H hmult arithmeticSide
      hformula harithmetic hsep

/-- A deliberately degenerate orbit model showing that pointwise transform
normalization is not by itself a separation theorem.  The existing orbit
interface permits zero multiplicities. -/
def zeroMultiplicityOffCriticalModel : MultiplicityAwareSymmetricZeros where
  representative _ := 1
  multiplicity _ := 0
  fixedBySymmetry _ := false
  fixed_spec n h := by simp at h

theorem pointNormalization_without_positiveMultiplicity_counterexample :
    (zeroMultiplicityOffCriticalModel.representative 0).re ≠ 0 ∧
      transform
        (pointNormalizedTest (zeroMultiplicityOffCriticalModel.representative 0))
        (zeroMultiplicityOffCriticalModel.representative 0) = 1 ∧
      ¬ ConvolutionSquaresSeparateOffCriticalOrbits
        zeroMultiplicityOffCriticalModel := by
  constructor
  · norm_num [zeroMultiplicityOffCriticalModel]
  constructor
  · exact transform_pointNormalizedTest_self _
  · intro h
    rcases h 0 (by norm_num [zeroMultiplicityOffCriticalModel]) with
      ⟨f, _hnormal, hneg⟩
    have hzero :
        symmetricZeroRegularized zeroMultiplicityOffCriticalModel
          (convolution f (involution f)) = 0 := by
      rw [symmetricZeroRegularized]
      have hterm :
          (fun n ↦ symmetricZeroPairTerm zeroMultiplicityOffCriticalModel
            (convolution f (involution f)) n) = 0 := by
        funext n
        simp [symmetricZeroPairTerm, zeroMultiplicityOffCriticalModel]
      rw [hterm]
      exact hasSum_zero.tsum_eq
    rw [hzero] at hneg
    norm_num at hneg

/-- Restricted infinite-formula positivity: shell count-times-decay
summability and the finite stage formula are combined by
`RestrictedGuinandWeilShellData`; if every modeled orbit is on the critical
line and the selected test is a convolution square, the resulting arithmetic
side has nonnegative real part. -/
theorem RestrictedGuinandWeilShellData.explicitFormula_re_nonneg
    {Z : MultiplicityAwareSymmetricZeros} {s : ℂ}
    (D : RestrictedGuinandWeilShellData Z s) (hs : 1 < s.re)
    (f : Test) (htest : D.test = convolution f (involution f))
    (hfixed : ∀ n, Z.fixedBySymmetry n = true) :
    0 ≤ (D.poleLimit -
      (-deriv riemannZeta s / riemannZeta s) -
      archimedeanLogDeriv s).re := by
  rw [← D.explicitFormula hs]
  rw [htest]
  apply symmetricZeroRegularized_convolution_nonneg Z f hfixed
  have hn : ZeroOrbitNormallyConvergent Z D.test :=
    ZeroOrbitNormallyConvergent.of_shell_count_decay
      D.zeroShells D.zeroDecay D.zeroCountDecaySummable D.zeroTermBound
  simpa [htest] using hn

/-- Stronger restricted positivity theorem needing only the actual
critical-line equation for representatives, not consistency of the auxiliary
fixed-orbit Boolean metadata. -/
theorem RestrictedGuinandWeilShellData.explicitFormula_re_nonneg_of_re_zero
    {Z : MultiplicityAwareSymmetricZeros} {s : ℂ}
    (D : RestrictedGuinandWeilShellData Z s) (hs : 1 < s.re)
    (f : Test) (htest : D.test = convolution f (involution f))
    (hre : ∀ n, (Z.representative n).re = 0) :
    0 ≤ (D.poleLimit -
      (-deriv riemannZeta s / riemannZeta s) -
      archimedeanLogDeriv s).re := by
  rw [← D.explicitFormula hs]
  rw [htest]
  apply symmetricZeroRegularized_convolution_nonneg_of_re_zero Z f hre
  have hn : ZeroOrbitNormallyConvergent Z D.test :=
    ZeroOrbitNormallyConvergent.of_shell_count_decay
      D.zeroShells D.zeroDecay D.zeroCountDecaySummable D.zeroTermBound
  simpa [htest] using hn

/-- RH supplies the centered critical-line hypothesis in the strengthened
restricted formula theorem when the shell model carries the actual pair of
zeta-zero equations. -/
theorem RestrictedGuinandWeilShellData.explicitFormula_re_nonneg_of_rh
    {Z : MultiplicityAwareSymmetricZeros} {s : ℂ}
    (D : RestrictedGuinandWeilShellData Z s)
    (H : ZetaZeroShellCountObligation Z D.zeroShells)
    (hs : 1 < s.re) (hRH : RiemannHypothesis)
    (f : Test) (htest : D.test = convolution f (involution f)) :
    0 ≤ (D.poleLimit -
      (-deriv riemannZeta s / riemannZeta s) -
      archimedeanLogDeriv s).re :=
  D.explicitFormula_re_nonneg_of_re_zero hs f htest
    (H.representative_re_zero_of_rh hRH)

/-- A stage-wise explicit formula packaged entirely in the typed
distribution interface. -/
structure FiniteExplicitStage where
  spectral : Distribution
  pole : Distribution
  primePower : Distribution
  archimedean : Distribution
  formula : ∀ f, spectral.eval f =
    pole.eval f - primePower.eval f - archimedean.eval f

/-- Pointwise convergence is the topology needed for passing positivity and
the linear explicit-formula identity to a limiting distribution. -/
def Distribution.TendsTo (Wn : ℕ → Distribution) (W : Distribution) : Prop :=
  ∀ f : Test, Tendsto (fun n ↦ (Wn n).eval f) atTop (𝓝 (W.eval f))

/-- Pointwise convergence of distributions gives pointwise convergence of
their real quadratic functionals, by continuity of `Complex.re`. -/
theorem Distribution.TendsTo.quadratic
    {Wn : ℕ → Distribution} {W : Distribution} (h : Distribution.TendsTo Wn W)
    (f : Test) :
    Tendsto (fun n ↦ quadratic (Wn n) f) atTop (𝓝 (quadratic W f)) := by
  exact Complex.continuous_re.continuousAt.tendsto.comp
    (h (convolution f (involution f)))

theorem explicitFormula_passes_to_limit
    (stage : ℕ → FiniteExplicitStage)
    (spectral pole primePower archimedean : Distribution)
    (hspectral : Distribution.TendsTo (fun n ↦ (stage n).spectral) spectral)
    (hpole : Distribution.TendsTo (fun n ↦ (stage n).pole) pole)
    (hprime : Distribution.TendsTo (fun n ↦ (stage n).primePower) primePower)
    (harch : Distribution.TendsTo (fun n ↦ (stage n).archimedean) archimedean) :
    ∀ f, spectral.eval f =
      pole.eval f - primePower.eval f - archimedean.eval f := by
  intro f
  apply tendsto_nhds_unique (hspectral f)
  exact ((hpole f).sub (hprime f)).sub (harch f) |>.congr' <|
    Eventually.of_forall fun n ↦ ((stage n).formula f).symm

/-! ## Positivity and monotone/pointwise limits -/

/-- Nonnegativity is closed under ordinary real limits. -/
theorem nonneg_of_tendsto
    (q : ℕ → ℝ) (qLimit : ℝ)
    (hlimit : Tendsto q atTop (𝓝 qLimit)) (hpos : ∀ n, 0 ≤ q n) :
    0 ≤ qLimit := by
  exact isClosed_Ici.mem_of_tendsto hlimit (Eventually.of_forall hpos)

/-- Pointwise convergence of quadratic forms is enough to pass universal
positivity to the limit; uniform convergence is not required. -/
theorem positivity_passes_to_pointwise_limit
    (Q : ℕ → Test → ℝ) (Qlimit : Test → ℝ)
    (hlimit : ∀ f, Tendsto (fun n ↦ Q n f) atTop (𝓝 (Qlimit f)))
    (hpos : ∀ n f, 0 ≤ Q n f) :
    ∀ f, 0 ≤ Qlimit f := by
  intro f
  exact nonneg_of_tendsto (fun n ↦ Q n f) (Qlimit f) (hlimit f) fun n ↦ hpos n f

/-- Eventual stage positivity for each fixed test is sufficient.  The cutoff
may depend on the test; neither one common cutoff nor uniform convergence on
the full test space is needed. -/
theorem positivity_passes_to_eventual_pointwise_limit
    (Q : ℕ → Test → ℝ) (Qlimit : Test → ℝ)
    (hlimit : ∀ f, Tendsto (fun n ↦ Q n f) atTop (𝓝 (Qlimit f)))
    (hpos : ∀ f, ∀ᶠ n in atTop, 0 ≤ Q n f) :
    ∀ f, 0 ≤ Qlimit f := by
  intro f
  exact isClosed_Ici.mem_of_tendsto (hlimit f) (hpos f)

/-- Uniform convergence on a selected admissible subclass implies pointwise
convergence there and hence preserves positivity on that subclass. -/
theorem positivity_passes_to_uniform_limit_on
    (Q : ℕ → Test → ℝ) (Qlimit : Test → ℝ) (S : Set Test)
    (hlimit : TendstoUniformlyOn Q Qlimit atTop S)
    (hpos : ∀ n f, f ∈ S → 0 ≤ Q n f) :
    ∀ f ∈ S, 0 ≤ Qlimit f := by
  intro f hf
  exact nonneg_of_tendsto (fun n ↦ Q n f) (Qlimit f)
    (hlimit.tendsto_at hf) fun n ↦ hpos n f hf

/-- Nonnegativity of a continuous functional extends from a set to its
closure.  This is the exact topological lemma needed for any dense-subclass
Weil argument. -/
theorem nonnegative_on_closure
    {α : Type*} [TopologicalSpace α] (Q : α → ℝ) (S : Set α)
    (hQ : ContinuousOn Q (closure S))
    (hpos : ∀ x ∈ S, 0 ≤ Q x) :
    ∀ x ∈ closure S, 0 ≤ Q x := by
  intro x hx
  exact le_on_closure (f := fun _ ↦ (0 : ℝ)) (g := Q)
    hpos continuousOn_const hQ hx

/-- Positivity on a dense subclass extends universally only when the target
functional is continuous in the chosen ambient topology. -/
theorem nonnegative_of_dense
    {α : Type*} [TopologicalSpace α] (Q : α → ℝ) (S : Set α)
    (hS : Dense S) (hQ : Continuous Q)
    (hpos : ∀ x ∈ S, 0 ≤ Q x) :
    ∀ x, 0 ≤ Q x := by
  intro x
  apply nonnegative_on_closure Q S hQ.continuousOn hpos
  rw [hS.closure_eq]
  exact mem_univ x

/-- Density alone is insufficient: a discontinuous functional may be
nonnegative on a dense subclass and negative at the omitted point. -/
theorem dense_nonnegative_counterexample_without_continuity :
    ∃ (Q : ℝ → ℝ) (S : Set ℝ),
      Dense S ∧ (∀ x ∈ S, 0 ≤ Q x) ∧ ¬ 0 ≤ Q 0 := by
  refine ⟨fun x ↦ if x = 0 then -1 else 0, ({0}ᶜ : Set ℝ), ?_, ?_, ?_⟩
  · exact dense_compl_singleton 0
  · intro x hx
    simp only [Set.mem_compl_iff, Set.mem_singleton_iff] at hx
    simp [hx]
  · norm_num

/-- Sequential form of the same closure argument, useful when density is
supplied by an explicit approximating sequence. -/
theorem nonnegative_of_tendsto_from
    {α : Type*} [TopologicalSpace α] (Q : α → ℝ) (S : Set α)
    {u : ℕ → α} {x : α}
    (hu : ∀ n, u n ∈ S) (hux : Tendsto u atTop (𝓝 x))
    (hQ : ContinuousAt Q x) (hpos : ∀ y ∈ S, 0 ≤ Q y) :
    0 ≤ Q x := by
  exact nonneg_of_tendsto (fun n ↦ Q (u n)) (Q x)
    (hQ.tendsto.comp hux) fun n ↦ hpos (u n) (hu n)

/-- Continuity obligation for a Weil quadratic functional in Mathlib's LF
topology on test functions.  Linearity of `Distribution.eval` alone does not
imply this property. -/
def Distribution.QuadraticContinuous (W : Distribution) : Prop :=
  Continuous (quadratic W)

/-- Continuity of the linear evaluation map in Mathlib's canonical LF
topology. -/
def Distribution.EvalContinuous (W : Distribution) : Prop :=
  Continuous W.toLinearMap

theorem ContinuousDistribution.toDistribution_evalContinuous
    (W : ContinuousDistribution) :
    W.toDistribution.EvalContinuous :=
  W.evalContinuous

/-- Mathlib's source-backed LF universal property, specialized to candidate
Weil distributions: evaluation is continuous exactly when every restriction
to a fixed compact support is continuous. -/
theorem Distribution.evalContinuous_iff_supported
    (W : Distribution) :
    W.EvalContinuous ↔
      ∀ (K : Compacts ℝ) (hK : (K : Set ℝ) ⊆ (⊤ : Opens ℝ)),
        Continuous (W.toLinearMap ∘ TestFunction.ofSupportedIn hK) := by
  exact TestFunction.continuous_iff_continuous_comp W.toLinearMap

/-- The other exact LF-continuity obligation: continuity of the nonlinear
convolution-square map.  Mathlib currently proves smoothness and compact
support of convolution, but exposes no theorem proving this LF continuity. -/
def ConvolutionSquareContinuous : Prop :=
  Continuous fun f : Test ↦ convolution f (involution f)

/-- Separate LF continuity obligation for Hermitian reflection. -/
def InvolutionContinuous : Prop :=
  Continuous involution

/-- Separate joint LF continuity obligation for test-function convolution. -/
def ConvolutionContinuous : Prop :=
  Continuous fun p : Test × Test ↦ convolution p.1 p.2

/-- Reflection of a compact support set through the origin. -/
def reflectedCompact (K : Compacts ℝ) : Compacts ℝ :=
  K.map (-·) continuous_neg

/-- Precomposition by reflection on one fixed-support Fréchet stage. -/
noncomputable def supportedReflectionPrecompLM (K : Compacts ℝ) :
    𝓓^{⊤}_{K}(ℝ, ℂ) →ₗ[ℝ] 𝓓^{⊤}_{reflectedCompact K}(ℝ, ℂ) where
  toFun f :=
    { toFun := fun x ↦ f (-x)
      contDiff' := f.contDiff.comp contDiff_neg
      zero_on_compl' := by
        intro x hx
        apply f.zero_on_compl
        intro hneg
        apply hx
        exact ⟨-x, hneg, by simp⟩ }
  map_add' f g := by ext x; rfl
  map_smul' r f := by ext x; rfl

/-- The only missing fixed-stage estimate for reflection, stated directly in
the seminorms defining `𝓓_K`.  The derivative identity under `x ↦ -x` is
available in Mathlib; packaging this estimate as a continuous map is the next
local topology step. -/
def SupportedReflectionSeminormBound : Prop :=
  ∀ (K : Compacts ℝ) (i : ℕ) (f : 𝓓^{⊤}_{K}(ℝ, ℂ)),
    N[ℝ]_{reflectedCompact K, i} (supportedReflectionPrecompLM K f) ≤
      N[ℝ]_{K, i} f

theorem supportedReflectionSeminormBound :
    SupportedReflectionSeminormBound := by
  intro K i f
  rw [ContDiffMapSupportedIn.seminorm_top_le_iff
    (𝕜 := ℝ) (E := ℝ) (F := ℂ) (K := reflectedCompact K)
    (apply_nonneg (ContDiffMapSupportedIn.seminorm ℝ ℝ ℂ ⊤ K i) f)]
  intro x hx
  rcases hx with ⟨y, hy, rfl⟩
  change ‖iteratedFDeriv ℝ i (fun x : ℝ ↦ f (-x)) (-y)‖ ≤
    N[ℝ]_{K, i} f
  rw [show (fun x : ℝ ↦ f (-x)) = fun x ↦ f ((-1 : ℝ) • x) by
    ext x
    simp]
  rw [iteratedFDeriv_comp_const_smul (-1 : ℝ) (f.contDiff.of_le (mod_cast le_top))]
  simp only [norm_smul, norm_pow, norm_neg, norm_one, one_pow, one_mul,
    neg_smul, one_smul, neg_neg]
  exact (ContDiffMapSupportedIn.seminorm_top_le_iff
    (𝕜 := ℝ) (E := ℝ) (F := ℂ) (K := K)
    (apply_nonneg (ContDiffMapSupportedIn.seminorm ℝ ℝ ℂ ⊤ K i) f) i f).1 le_rfl y hy

/-- Reflection precomposition is continuous on every fixed-support Fréchet
stage, proved from the defining seminorms. -/
noncomputable def supportedReflectionPrecompCLM (K : Compacts ℝ) :
    𝓓^{⊤}_{K}(ℝ, ℂ) →L[ℝ] 𝓓^{⊤}_{reflectedCompact K}(ℝ, ℂ) where
  toLinearMap := supportedReflectionPrecompLM K
  cont := by
    refine WithSeminorms.continuous_of_isBounded
      (ContDiffMapSupportedIn.withSeminorms ℝ ℝ ℂ ⊤ K)
      (ContDiffMapSupportedIn.withSeminorms ℝ ℝ ℂ ⊤ (reflectedCompact K))
      _ (fun i ↦ ⟨{i}, 1, fun f ↦ ?_⟩)
    simpa using supportedReflectionSeminormBound K i f

/-- The algebraic reflection map on one fixed-support Fréchet stage.  Mathlib's
test-function LF universal property reduces global reflection continuity to
continuity of these maps. -/
noncomputable def supportedInvolutionLM (K : Compacts ℝ) :
    𝓓^{⊤}_{K}(ℝ, ℂ) →ₗ[ℝ] Test where
  toFun f := involution (TestFunction.ofSupportedIn (Set.subset_univ _) f)
  map_add' f g := by
    ext x
    simp [involution]
  map_smul' r f := by
    ext x
    simp [involution]

/-- Fixed-support Hermitian reflection as an actual continuous real-linear
map: reflection precomposition, followed by complex conjugation and LF
inclusion. -/
noncomputable def supportedInvolutionCLM (K : Compacts ℝ) :
    𝓓^{⊤}_{K}(ℝ, ℂ) →L[ℝ] Test :=
  TestFunction.ofSupportedInCLM ℝ (Set.subset_univ _) ∘L
    (ContDiffMapSupportedIn.postcompCLM Complex.conjCLE ∘L
      supportedReflectionPrecompCLM K)

theorem supportedInvolutionCLM_toLinearMap (K : Compacts ℝ) :
    (supportedInvolutionCLM K).toLinearMap = supportedInvolutionLM K := by
  ext f x
  rfl

/-- Exact fixed-support prerequisite absent from the pinned Mathlib API:
reflection must be continuous on each `𝓓_K` stage. -/
def SupportedInvolutionContinuous : Prop :=
  ∀ K : Compacts ℝ, Continuous (supportedInvolutionLM K)

theorem supportedInvolutionContinuous : SupportedInvolutionContinuous := by
  intro K
  rw [← supportedInvolutionCLM_toLinearMap K]
  exact (supportedInvolutionCLM K).continuous

/-- The LF universal property proves global Hermitian-reflection continuity
once fixed-support reflection continuity is supplied. -/
theorem involutionContinuous_of_supported
    (h : SupportedInvolutionContinuous) : InvolutionContinuous := by
  let T : Test →L[ℝ] Test :=
    TestFunction.limitCLM ℝ involution
      (fun K _ ↦ ⟨supportedInvolutionLM K, h K⟩)
      (fun _ _ _ ↦ rfl)
  exact T.continuous

/-- Hermitian reflection is continuous in Mathlib's canonical LF test-space
topology. -/
theorem involutionContinuous : InvolutionContinuous :=
  involutionContinuous_of_supported supportedInvolutionContinuous

/-- Fixed-support joint continuity is the analytic prerequisite for
convolution.  The available LF universal property applies only to linear maps,
so this condition alone does not currently yield joint continuity on
`Test × Test` in Mathlib. -/
def SupportedConvolutionContinuous : Prop :=
  ∀ K L : Compacts ℝ, Continuous fun p : 𝓓^{⊤}_{K}(ℝ, ℂ) × 𝓓^{⊤}_{L}(ℝ, ℂ) ↦
    convolution
      (TestFunction.ofSupportedIn (Set.subset_univ _) p.1)
      (TestFunction.ofSupportedIn (Set.subset_univ _) p.2)

/-- Convolution by a fixed left test function is real-linear. -/
noncomputable def convolutionLeftLM (f : Test) : Test →ₗ[ℝ] Test where
  toFun g := convolution f g
  map_add' g h := by
    ext x
    change MeasureTheory.convolution f (g + h) (ContinuousLinearMap.mul ℝ ℂ) volume x =
      MeasureTheory.convolution f g (ContinuousLinearMap.mul ℝ ℂ) volume x +
        MeasureTheory.convolution f h (ContinuousLinearMap.mul ℝ ℂ) volume x
    exact congrArg (fun q : ℝ → ℂ ↦ q x) <|
      (f.hasCompactSupport.convolutionExists_left
        (ContinuousLinearMap.mul ℝ ℂ) f.continuous g.continuous.locallyIntegrable).distrib_add
        (f.hasCompactSupport.convolutionExists_left
          (ContinuousLinearMap.mul ℝ ℂ) f.continuous h.continuous.locallyIntegrable)
  map_smul' r g := by
    ext x
    change MeasureTheory.convolution f (r • g) (ContinuousLinearMap.mul ℝ ℂ) volume x =
      r • MeasureTheory.convolution f g (ContinuousLinearMap.mul ℝ ℂ) volume x
    exact congrFun MeasureTheory.convolution_smul x

/-- Fixed-left convolution is globally continuous once fixed-support joint
estimates are available.  This uses only Mathlib's one-variable LF universal
property. -/
noncomputable def convolutionLeftCLM
    (hlocal : SupportedConvolutionContinuous) (f : Test) : Test →L[ℝ] Test := by
  let K : Compacts ℝ := ⟨tsupport f, f.hasCompactSupport⟩
  let fK : 𝓓^{⊤}_{K}(ℝ, ℂ) :=
    .of_support_subset f.contDiff subset_closure
  refine TestFunction.limitCLM ℝ (convolution f)
    (fun L _ ↦ ⟨convolutionLeftLM f ∘ₗ
      (TestFunction.ofSupportedInCLM ℝ (Set.subset_univ _)).toLinearMap, ?_⟩) ?_
  · have hpair : Continuous fun g : 𝓓^{⊤}_{L}(ℝ, ℂ) ↦ (fK, g) :=
      continuous_const.prodMk continuous_id
    have h := (hlocal K L).comp hpair
    apply h.congr
    intro g
    rfl
  · intro L hL g
    rfl

theorem convolution_separatelyContinuous_left
    (hlocal : SupportedConvolutionContinuous) (f : Test) :
    Continuous fun g : Test ↦ convolution f g :=
  (convolutionLeftCLM hlocal f).continuous.congr fun _ ↦ rfl

theorem convolution_separatelyContinuous_right
    (hlocal : SupportedConvolutionContinuous) (g : Test) :
    Continuous fun f : Test ↦ convolution f g :=
  (convolution_separatelyContinuous_left hlocal g).congr fun f ↦
    convolution_comm g f

/-- Convolution bundled as an algebraic real-bilinear map. -/
noncomputable def convolutionBilinearLM :
    Test →ₗ[ℝ] Test →ₗ[ℝ] Test where
  toFun := convolutionLeftLM
  map_add' f g := by
    apply LinearMap.ext
    intro h
    calc
      convolution (f + g) h = convolution h (f + g) := convolution_comm _ _
      _ = convolution h f + convolution h g :=
        (convolutionLeftLM h).map_add f g
      _ = convolution f h + convolution g h := by
        rw [convolution_comm h f, convolution_comm h g]
  map_smul' r f := by
    apply LinearMap.ext
    intro g
    calc
      convolution (r • f) g = convolution g (r • f) := convolution_comm _ _
      _ = r • convolution g f := (convolutionLeftLM g).map_smul r f
      _ = r • convolution f g := by rw [convolution_comm g f]

theorem convolution_sub_sub (a b c d : Test) :
    convolution a b - convolution c d =
      convolution a (b - d) + convolution (a - c) d := by
  change convolutionBilinearLM a b - convolutionBilinearLM c d =
    convolutionBilinearLM a (b - d) + convolutionBilinearLM (a - c) d
  simp only [map_sub, LinearMap.sub_apply]
  abel

theorem supportedConvolution_sub_sub
    (K L : Compacts ℝ)
    (a c : 𝓓^{⊤}_{K}(ℝ, ℂ)) (b d : 𝓓^{⊤}_{L}(ℝ, ℂ)) :
    supportedConvolution K L (a, b) - supportedConvolution K L (c, d) =
      supportedConvolution K L (a, b - d) +
        supportedConvolution K L (a - c, d) := by
  let A : Test := TestFunction.ofSupportedIn (Set.subset_univ _) a
  let B : Test := TestFunction.ofSupportedIn (Set.subset_univ _) b
  let C : Test := TestFunction.ofSupportedIn (Set.subset_univ _) c
  let D : Test := TestFunction.ofSupportedIn (Set.subset_univ _) d
  ext x
  change (convolution A B - convolution C D) x =
    (convolution A (B - D) + convolution (A - C) D) x
  exact congrArg (fun f : Test ↦ f x) (convolution_sub_sub A B C D)

/-- The explicit seminorm estimate proves genuine joint continuity on every
fixed compact-support stage. -/
theorem supportedConvolution_continuous (K L : Compacts ℝ) :
    Continuous (supportedConvolution K L) := by
  rw [ContDiffMapSupportedIn.continuous_iff_comp]
  intro n
  rw [continuous_iff_continuousAt]
  intro x
  change Tendsto
    (fun e ↦ (ContDiffMapSupportedIn.structureMapCLM ℝ ⊤ n)
      (supportedConvolution K L e)) (𝓝 x)
    (𝓝 ((ContDiffMapSupportedIn.structureMapCLM ℝ ⊤ n)
      (supportedConvolution K L x)))
  rw [tendsto_iff_dist_tendsto_zero]
  let C : ℝ := volume.real (K : Set ℝ)
  let pK : Seminorm ℝ 𝓓^{⊤}_{K}(ℝ, ℂ) :=
    ContDiffMapSupportedIn.seminorm ℝ ℝ ℂ ⊤ K 0
  let pL : Seminorm ℝ 𝓓^{⊤}_{L}(ℝ, ℂ) :=
    ContDiffMapSupportedIn.seminorm ℝ ℝ ℂ ⊤ L n
  have hpK : Continuous pK := by
    exact (ContDiffMapSupportedIn.withSeminorms ℝ ℝ ℂ ⊤ K).continuous_seminorm 0
  have hpL : Continuous pL := by
    exact (ContDiffMapSupportedIn.withSeminorms ℝ ℝ ℂ ⊤ L).continuous_seminorm n
  have hle (e : 𝓓^{⊤}_{K}(ℝ, ℂ) × 𝓓^{⊤}_{L}(ℝ, ℂ)) :
      ‖(ContDiffMapSupportedIn.structureMapCLM ℝ ⊤ n)
          (supportedConvolution K L e) -
        (ContDiffMapSupportedIn.structureMapCLM ℝ ⊤ n)
          (supportedConvolution K L x)‖ ≤
        C * pK e.1 * pL (e.2 - x.2) +
          C * pK (e.1 - x.1) * pL x.2 := by
    calc
      ‖(ContDiffMapSupportedIn.structureMapCLM ℝ ⊤ n)
            (supportedConvolution K L e) -
          (ContDiffMapSupportedIn.structureMapCLM ℝ ⊤ n)
            (supportedConvolution K L x)‖ =
          N[ℝ]_{convolutionCompact K L, n}
            (supportedConvolution K L e - supportedConvolution K L x) := by
              change
                ‖(ContDiffMapSupportedIn.structureMapCLM ℝ ⊤ n)
                    (supportedConvolution K L e) -
                  (ContDiffMapSupportedIn.structureMapCLM ℝ ⊤ n)
                    (supportedConvolution K L x)‖ =
                ‖(ContDiffMapSupportedIn.structureMapCLM ℝ ⊤ n)
                  (supportedConvolution K L e - supportedConvolution K L x)‖
              exact congrArg norm <|
                (map_sub (ContDiffMapSupportedIn.structureMapCLM ℝ ⊤ n)
                  (supportedConvolution K L e)
                  (supportedConvolution K L x)).symm
      _ = N[ℝ]_{convolutionCompact K L, n}
          (supportedConvolution K L (e.1, e.2 - x.2) +
            supportedConvolution K L (e.1 - x.1, x.2)) := by
              rw [supportedConvolution_sub_sub]
      _ ≤ N[ℝ]_{convolutionCompact K L, n}
            (supportedConvolution K L (e.1, e.2 - x.2)) +
          N[ℝ]_{convolutionCompact K L, n}
            (supportedConvolution K L (e.1 - x.1, x.2)) :=
              map_add_le_add _ _ _
      _ ≤ C * pK e.1 * pL (e.2 - x.2) +
          C * pK (e.1 - x.1) * pL x.2 := by
            exact add_le_add
              (supportedConvolution_seminorm_le K L (e.1, e.2 - x.2) n)
              (supportedConvolution_seminorm_le K L (e.1 - x.1, x.2) n)
  refine squeeze_zero
    (g := fun e ↦
      C * pK e.1 * pL (e.2 - x.2) +
        C * pK (e.1 - x.1) * pL x.2)
    (fun e ↦ dist_nonneg) (fun e ↦ ?_) ?_
  · have hupper :
        0 ≤ C * pK e.1 * pL (e.2 - x.2) +
          C * pK (e.1 - x.1) * pL x.2 :=
      (norm_nonneg
        ((ContDiffMapSupportedIn.structureMapCLM ℝ ⊤ n)
            (supportedConvolution K L e) -
          (ContDiffMapSupportedIn.structureMapCLM ℝ ⊤ n)
            (supportedConvolution K L x))).trans (hle e)
    apply (BoundedContinuousFunction.dist_le hupper).2
    intro y
    calc
      dist
          ((ContDiffMapSupportedIn.structureMapCLM ℝ ⊤ n)
            (supportedConvolution K L e) y)
          ((ContDiffMapSupportedIn.structureMapCLM ℝ ⊤ n)
            (supportedConvolution K L x) y) =
          ‖((ContDiffMapSupportedIn.structureMapCLM ℝ ⊤ n)
              (supportedConvolution K L e) -
            (ContDiffMapSupportedIn.structureMapCLM ℝ ⊤ n)
              (supportedConvolution K L x)) y‖ := by
            rw [dist_eq_norm]
            rfl
      _ ≤ ‖(ContDiffMapSupportedIn.structureMapCLM ℝ ⊤ n)
              (supportedConvolution K L e) -
            (ContDiffMapSupportedIn.structureMapCLM ℝ ⊤ n)
              (supportedConvolution K L x)‖ :=
        BoundedContinuousFunction.norm_coe_le_norm _ _
      _ ≤ C * pK e.1 * pL (e.2 - x.2) +
          C * pK (e.1 - x.1) * pL x.2 := hle e
  · have hu : Continuous fun e :
        𝓓^{⊤}_{K}(ℝ, ℂ) × 𝓓^{⊤}_{L}(ℝ, ℂ) ↦
        C * pK e.1 * pL (e.2 - x.2) +
          C * pK (e.1 - x.1) * pL x.2 := by
      fun_prop
    simpa [C, pK, pL] using hu.tendsto x

/-- Fixed-stage joint convolution continuity is unconditional. -/
theorem supportedConvolutionContinuous : SupportedConvolutionContinuous := by
  intro K L
  have hinc : Continuous
      (TestFunction.ofSupportedIn (Set.subset_univ _) :
        𝓓^{⊤}_{convolutionCompact K L}(ℝ, ℂ) → Test) :=
    TestFunction.continuous_ofSupportedIn (Set.subset_univ _)
  exact (hinc.comp (supportedConvolution_continuous K L)).congr fun p ↦ by
    ext x
    rfl

theorem convolution_separatelyContinuous_left_unconditional (f : Test) :
    Continuous fun g : Test ↦ convolution f g :=
  convolution_separatelyContinuous_left supportedConvolutionContinuous f

theorem convolution_separatelyContinuous_right_unconditional (g : Test) :
    Continuous fun f : Test ↦ convolution f g :=
  convolution_separatelyContinuous_right supportedConvolutionContinuous g

/-- Pair of canonical fixed-support inclusions into the LF test space. -/
def TestFunction.stagePairInclusion (K L : Compacts ℝ) :
    𝓓^{⊤}_{K}(ℝ, ℂ) × 𝓓^{⊤}_{L}(ℝ, ℂ) → Test × Test :=
  fun p ↦
    (TestFunction.ofSupportedIn (Set.subset_univ _) p.1,
      TestFunction.ofSupportedIn (Set.subset_univ _) p.2)

theorem TestFunction.continuous_stagePairInclusion (K L : Compacts ℝ) :
    Continuous (TestFunction.stagePairInclusion K L) :=
  (TestFunction.continuous_ofSupportedIn (Set.subset_univ _)).comp continuous_fst
    |>.prodMk
      ((TestFunction.continuous_ofSupportedIn (Set.subset_univ _)).comp continuous_snd)

/-- Final topology generated by all pairs of fixed-support stages. -/
@[reducible] noncomputable def TestFunction.LFPairFinalTopology :
    TopologicalSpace (Test × Test) :=
  ⨆ K : Compacts ℝ, ⨆ L : Compacts ℝ,
    TopologicalSpace.coinduced
      (TestFunction.stagePairInclusion K L)
      (inferInstance : TopologicalSpace
        (𝓓^{⊤}_{K}(ℝ, ℂ) × 𝓓^{⊤}_{L}(ℝ, ℂ)))

/-- The pair-stage final topology is always no finer than the ordinary
product of the two LF topologies. -/
theorem TestFunction.lfPairFinalTopology_le_product :
    TestFunction.LFPairFinalTopology ≤
      (inferInstance : TopologicalSpace (Test × Test)) := by
  apply iSup_le
  intro K
  apply iSup_le
  intro L
  exact continuous_iff_coinduced_le.mp
    (TestFunction.continuous_stagePairInclusion K L)

/-- The exact remaining product-topology obligation.  Only the reverse
inequality is unknown; the forward inequality is proved above. -/
def TestFunction.LFPairProductTopologyObligation : Prop :=
  (inferInstance : TopologicalSpace (Test × Test)) ≤
    TestFunction.LFPairFinalTopology

/-- The reverse inequality is exactly continuity of the identity from the
ordinary LF product into the pair-stage final topology. -/
theorem TestFunction.lfPairProductTopologyObligation_iff :
    TestFunction.LFPairProductTopologyObligation ↔
      @Continuous (Test × Test) (Test × Test)
        (inferInstance : TopologicalSpace (Test × Test))
        TestFunction.LFPairFinalTopology id := by
  exact continuous_id_iff_le.symm

/-- First-principles neighborhood form of the product comparison: every
pair-stage-final neighborhood must contain a rectangle of ordinary LF
neighborhoods. -/
def TestFunction.LFPairProductNeighborhoodCriterion : Prop :=
  ∀ p : Test × Test, ∀ U : Set (Test × Test),
    U ∈ @nhds (Test × Test) TestFunction.LFPairFinalTopology p →
      ∃ V ∈ 𝓝 p.1, ∃ W ∈ 𝓝 p.2, V ×ˢ W ⊆ U

theorem TestFunction.lfPairProductTopologyObligation_iff_neighborhoods :
    TestFunction.LFPairProductTopologyObligation ↔
      TestFunction.LFPairProductNeighborhoodCriterion := by
  constructor
  · intro h p U hU
    have hmem : U ∈ 𝓝 p :=
      (le_iff_nhds
        (inferInstance : TopologicalSpace (Test × Test))
        TestFunction.LFPairFinalTopology).mp h p hU
    exact mem_nhds_prod_iff.mp hmem
  · intro h
    apply (le_iff_nhds
      (inferInstance : TopologicalSpace (Test × Test))
      TestFunction.LFPairFinalTopology).2
    intro p U hU
    rcases h p U hU with ⟨V, hV, W, hW, hVW⟩
    exact mem_of_superset (prod_mem_nhds hV hW) hVW

/-- Universal property of the explicitly defined pair-stage final topology. -/
theorem TestFunction.continuous_lfPairFinal
    {Y : Type*} [TopologicalSpace Y] (F : Test × Test → Y)
    (hstage : ∀ K L : Compacts ℝ,
      Continuous (F ∘ TestFunction.stagePairInclusion K L)) :
    @Continuous (Test × Test) Y TestFunction.LFPairFinalTopology
      inferInstance F := by
  rw [TestFunction.LFPairFinalTopology, continuous_iSup_dom]
  intro K
  rw [continuous_iSup_dom]
  intro L
  rw [continuous_coinduced_dom]
  exact hstage K L

/-- The reverse product-topology inequality upgrades all compatible
fixed-stage continuity statements to ordinary product continuity. -/
theorem TestFunction.continuous_of_lfPairProductTopology
    (hproduct : TestFunction.LFPairProductTopologyObligation)
    {Y : Type*} [TopologicalSpace Y] (F : Test × Test → Y)
    (hstage : ∀ K L : Compacts ℝ,
      Continuous (F ∘ TestFunction.stagePairInclusion K L)) :
    Continuous F := by
  exact continuous_le_dom hproduct
    (TestFunction.continuous_lfPairFinal F hstage)

/-- Convolution is unconditionally continuous when its domain carries the
actual pair-stage final topology. -/
theorem convolutionContinuous_lfPairFinal :
    @Continuous (Test × Test) Test
      TestFunction.LFPairFinalTopology inferInstance
      (fun p ↦ convolution p.1 p.2) := by
  apply TestFunction.continuous_lfPairFinal
  intro K L
  exact supportedConvolutionContinuous K L

/-- Global joint convolution continuity now depends only on the explicit
reverse topology inequality, not on any analytic estimate. -/
theorem convolutionContinuous_of_lfPairProductTopology
    (hproduct : TestFunction.LFPairProductTopologyObligation) :
    ConvolutionContinuous := by
  apply TestFunction.continuous_of_lfPairProductTopology hproduct
  intro K L
  exact supportedConvolutionContinuous K L

/-- Exact bilinear product-lifting statement needed for convolution.  Unlike
`LFProductUniversalProperty`, this asks only for bilinear maps and is the
appropriate locally-convex categorical obligation. -/
def TestFunction.LFBilinearProductUniversalProperty : Prop :=
  ∀ B : Test →ₗ[ℝ] Test →ₗ[ℝ] Test,
    (∀ K L : Compacts ℝ, Continuous fun
      p : 𝓓^{⊤}_{K}(ℝ, ℂ) × 𝓓^{⊤}_{L}(ℝ, ℂ) ↦
        B
          (TestFunction.ofSupportedIn (Set.subset_univ _) p.1)
          (TestFunction.ofSupportedIn (Set.subset_univ _) p.2)) →
    Continuous fun p : Test × Test ↦ B p.1 p.2

theorem TestFunction.lfBilinearProductUniversalProperty_of_topology
    (hproduct : TestFunction.LFPairProductTopologyObligation) :
    TestFunction.LFBilinearProductUniversalProperty := by
  intro B hstage
  apply TestFunction.continuous_of_lfPairProductTopology hproduct
  intro K L
  exact hstage K L

theorem convolutionContinuous_of_lfBilinearProduct
    (hproduct : TestFunction.LFBilinearProductUniversalProperty)
    (hlocal : SupportedConvolutionContinuous) :
    ConvolutionContinuous := by
  exact (hproduct convolutionBilinearLM hlocal).congr fun _ ↦ rfl

/-- Exact product/inductive-limit universal property missing from the pinned
API.  The existing `TestFunction.continuous_iff_continuous_comp` handles
linear maps from one LF factor; joint convolution needs this two-factor
lifting theorem (or an equivalent bounded-bilinear theorem). -/
def TestFunction.LFProductUniversalProperty : Prop :=
  ∀ F : Test × Test → Test,
    (∀ K L : Compacts ℝ, Continuous fun
      p : 𝓓^{⊤}_{K}(ℝ, ℂ) × 𝓓^{⊤}_{L}(ℝ, ℂ) ↦
        F
          (TestFunction.ofSupportedIn (Set.subset_univ _) p.1,
            TestFunction.ofSupportedIn (Set.subset_univ _) p.2)) →
    Continuous F

/-- Fixed-stage joint convolution plus the missing LF-product universal
property yields global joint continuity. -/
theorem convolutionContinuous_of_lfProduct
    (hproduct : TestFunction.LFProductUniversalProperty)
    (hlocal : SupportedConvolutionContinuous) :
    ConvolutionContinuous :=
  hproduct (fun p ↦ convolution p.1 p.2) hlocal

/-- Joint continuity of convolution and continuity of reflection imply
continuity of convolution squares. -/
theorem convolutionSquareContinuous_of_components
    (hconv : ConvolutionContinuous) (hinv : InvolutionContinuous) :
    ConvolutionSquareContinuous := by
  exact hconv.comp (continuous_id.prodMk hinv)

/-- Since reflection continuity is now proved, joint LF continuity of
convolution is the only remaining topological input for convolution-square
continuity. -/
theorem convolutionSquareContinuous_of_convolution
    (hconv : ConvolutionContinuous) : ConvolutionSquareContinuous :=
  convolutionSquareContinuous_of_components hconv involutionContinuous

/-- The explicit LF product-topology obligation is sufficient for global
convolution-square continuity. -/
theorem convolutionSquareContinuous_of_lfPairProductTopology
    (hproduct : TestFunction.LFPairProductTopologyObligation) :
    ConvolutionSquareContinuous :=
  convolutionSquareContinuous_of_convolution
    (convolutionContinuous_of_lfPairProductTopology hproduct)

/-- Continuous distribution evaluation plus continuity of convolution
squares implies continuity of the Weil quadratic functional. -/
theorem Distribution.quadraticContinuous_of_eval_of_convolutionSquare
    (W : Distribution) (hW : W.EvalContinuous)
    (hsquare : ConvolutionSquareContinuous) :
    W.QuadraticContinuous := by
  exact Complex.continuous_re.comp (hW.comp hsquare)

/-- Exact dense-subclass closure theorem for Weil positivity.  Applications
must separately prove both LF-density of the subclass and continuity of the
quadratic functional. -/
theorem Distribution.IsPositive.of_dense
    (W : Distribution) (S : Set Test) (hS : Dense S)
    (hcontinuous : W.QuadraticContinuous)
    (hpos : ∀ f ∈ S, 0 ≤ quadratic W f) :
    IsPositive W :=
  nonnegative_of_dense (quadratic W) S hS hcontinuous hpos

/-- A convergent error envelope is a convenient quantitative sufficient
condition for the pointwise convergence needed by positivity transfer. -/
theorem positivity_of_error_bound
    (q : ℕ → ℝ) (qLimit : ℝ) (ε : ℕ → ℝ)
    (hε : Tendsto ε atTop (𝓝 0))
    (herror : ∀ n, |q n - qLimit| ≤ ε n)
    (hpos : ∀ n, 0 ≤ q n) :
    0 ≤ qLimit := by
  apply nonneg_of_tendsto q qLimit
  · rw [tendsto_iff_norm_sub_tendsto_zero]
    simpa only [Real.norm_eq_abs] using
      squeeze_zero (fun n ↦ abs_nonneg (q n - qLimit)) herror hε
  · exact hpos

/-- The distribution-level form used by the Weil route. -/
theorem Distribution.IsPositive.of_tendsTo
    {Wn : ℕ → Distribution} {W : Distribution}
    (hlimit : Distribution.TendsTo Wn W)
    (hpos : ∀ n, IsPositive (Wn n)) :
    IsPositive W := by
  intro f
  exact nonneg_of_tendsto _ _ (hlimit.quadratic f) fun n ↦ hpos n f

/-- A bounded monotone family has the canonical pointwise supremum limit,
using Mathlib's conditionally-complete monotone convergence theorem. -/
theorem monotoneQuadratic_tendsto_ciSup
    (Q : ℕ → Test → ℝ) (f : Test)
    (hmono : Monotone fun n ↦ Q n f)
    (hbdd : BddAbove (range fun n ↦ Q n f)) :
    Tendsto (fun n ↦ Q n f) atTop (𝓝 (⨆ n, Q n f)) :=
  tendsto_atTop_ciSup hmono hbdd

theorem monotoneQuadratic_limit_nonneg
    (Q : ℕ → Test → ℝ)
    (hmono : ∀ f, Monotone fun n ↦ Q n f)
    (hbdd : ∀ f, BddAbove (range fun n ↦ Q n f))
    (hpos : ∀ n f, 0 ≤ Q n f) :
    ∀ f, 0 ≤ ⨆ n, Q n f := by
  intro f
  exact nonneg_of_tendsto (fun n ↦ Q n f) (⨆ n, Q n f)
    (monotoneQuadratic_tendsto_ciSup Q f (hmono f) (hbdd f)) fun n ↦ hpos n f

/-- Finite-stage positivity does not constrain a separately declared value
unless a convergence/regularization theorem connects them. -/
theorem finitePositivity_without_convergence_counterexample :
    (∀ _ : ℕ, 0 ≤ (0 : ℝ)) ∧
      ¬ Tendsto (fun _ : ℕ ↦ (0 : ℝ)) atTop (𝓝 (-1 : ℝ)) := by
  constructor
  · intro _
    positivity
  · intro h
    have hzero : Tendsto (fun _ : ℕ ↦ (0 : ℝ)) atTop (𝓝 (0 : ℝ)) :=
      tendsto_const_nhds
    have : (0 : ℝ) = -1 := tendsto_nhds_unique hzero h
    norm_num at this

/-! ## Named terminal bridge obligations -/

/-- The archimedean truncations must converge on every admissible test. -/
def ArchimedeanLimitObligation
    (stage : ℕ → Distribution) (limit : Distribution) : Prop :=
  Distribution.TendsTo stage limit

/-- Compact support should make a correctly enumerated prime-power
truncation eventually exact on each test. -/
def PrimePowerStabilizationObligation
    (stage : ℕ → Distribution) (limit : Distribution) : Prop :=
  ∀ f : Test, ∃ N, ∀ n ≥ N, (stage n).eval f = limit.eval f

theorem primePowerStabilization_tendsTo
    (stage : ℕ → Distribution) (limit : Distribution)
    (h : PrimePowerStabilizationObligation stage limit) :
    Distribution.TendsTo stage limit := by
  intro f
  rcases h f with ⟨N, hN⟩
  exact tendsto_const_nhds.congr' <|
    eventually_atTop.2 ⟨N, fun n hn ↦ (hN n hn).symm⟩

/-- Symmetric zero truncations must converge to the selected regularized
zero distribution, with multiplicities and ordering fixed by the caller. -/
def ZeroRegularizationObligation
    (stage : ℕ → Distribution) (regularized : Distribution) : Prop :=
  Distribution.TendsTo stage regularized

/-! ## Separately typed analytic approximants -/

/-- Symmetric zero truncations.  `symmetric` records the selected
regularization symmetry at distribution level; convergence and multiplicity
remain separate obligations. -/
structure SymmetricZeroApproximants where
  stage : ℕ → Distribution
  symmetric : ∀ n f,
    (stage n).eval (involution f) = star ((stage n).eval f)

structure ArchimedeanApproximants where
  stage : ℕ → Distribution

structure PrimePowerApproximants where
  stage : ℕ → Distribution

structure PoleApproximants where
  stage : ℕ → Distribution

def SymmetricZeroApproximants.ConvergesTo
    (A : SymmetricZeroApproximants) (limit : Distribution) : Prop :=
  ZeroRegularizationObligation A.stage limit

def ArchimedeanApproximants.ConvergesTo
    (A : ArchimedeanApproximants) (limit : Distribution) : Prop :=
  ArchimedeanLimitObligation A.stage limit

def PrimePowerApproximants.StabilizesTo
    (A : PrimePowerApproximants) (limit : Distribution) : Prop :=
  PrimePowerStabilizationObligation A.stage limit

def PoleApproximants.ConvergesTo
    (A : PoleApproximants) (limit : Distribution) : Prop :=
  Distribution.TendsTo A.stage limit

/-- A typed four-component approximation package.  The stage formula is an
obligation rather than an assumed zeta theorem. -/
structure TypedExplicitApproximants where
  zeros : SymmetricZeroApproximants
  pole : PoleApproximants
  primePower : PrimePowerApproximants
  archimedean : ArchimedeanApproximants
  formula : ∀ n f, (zeros.stage n).eval f =
    (pole.stage n).eval f - (primePower.stage n).eval f -
      (archimedean.stage n).eval f

def TypedExplicitApproximants.toFiniteStage
    (A : TypedExplicitApproximants) (n : ℕ) : FiniteExplicitStage where
  spectral := A.zeros.stage n
  pole := A.pole.stage n
  primePower := A.primePower.stage n
  archimedean := A.archimedean.stage n
  formula := A.formula n

/-- Algebraic combination theorem for the separately typed limits. -/
theorem TypedExplicitApproximants.explicitFormula
    (A : TypedExplicitApproximants)
    (zeros pole primePower archimedean : Distribution)
    (hzero : A.zeros.ConvergesTo zeros)
    (hpole : A.pole.ConvergesTo pole)
    (hprime : A.primePower.StabilizesTo primePower)
    (harch : A.archimedean.ConvergesTo archimedean) :
    ∀ f, zeros.eval f =
      pole.eval f - primePower.eval f - archimedean.eval f := by
  apply explicitFormula_passes_to_limit A.toFiniteStage
  · exact hzero
  · exact hpole
  · exact primePowerStabilization_tendsTo _ _ hprime
  · exact harch

/-- The finite identities and all component limits form the analytic
explicit-formula package.  It still does not assert either RH direction. -/
structure AnalyticBridgeObligations
    (stage : ℕ → FiniteExplicitStage)
    (spectral pole primePower archimedean : Distribution) : Prop where
  spectralRegularization :
    Distribution.TendsTo (fun n ↦ (stage n).spectral) spectral
  poleLimit : Distribution.TendsTo (fun n ↦ (stage n).pole) pole
  primePowerStabilization :
    PrimePowerStabilizationObligation (fun n ↦ (stage n).primePower) primePower
  archimedeanLimit :
    ArchimedeanLimitObligation (fun n ↦ (stage n).archimedean) archimedean

theorem AnalyticBridgeObligations.explicitFormula
    {stage : ℕ → FiniteExplicitStage}
    {spectral pole primePower archimedean : Distribution}
    (h : AnalyticBridgeObligations stage spectral pole primePower archimedean) :
    ∀ f, spectral.eval f =
      pole.eval f - primePower.eval f - archimedean.eval f :=
  explicitFormula_passes_to_limit stage spectral pole primePower archimedean
    h.spectralRegularization h.poleLimit
    (primePowerStabilization_tendsTo _ _ h.primePowerStabilization)
    h.archimedeanLimit

/-- The remaining all-test criterion is exactly the conjunction of the two
logical directions; finite-stage formulae and limits do not supply it. -/
def AllTestPositivityObligation (W : Distribution) : Prop :=
  PositivityImpliesRH W ∧ RHImpliesPositivity W

/-- A finite Gram kernel is the basic exact certificate behind every
finite-dimensional positive-semidefinite restriction. -/
def gramKernel {ι E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℂ E] (v : ι → E) :
    Matrix ι ι ℂ :=
  Matrix.gram ℂ v

/-- Every finite Gram kernel is positive semidefinite. -/
theorem gramKernel_posSemidef
    {ι E : Type*} [Finite ι] [NormedAddCommGroup E] [InnerProductSpace ℂ E] (v : ι → E) :
    (gramKernel v).PosSemidef :=
  Matrix.posSemidef_gram ℂ v

/-- Positive semidefiniteness survives every finite reindexing/restriction. -/
theorem gramKernel_restrict_posSemidef
    {ι κ E : Type*} [Finite ι] [NormedAddCommGroup E] [InnerProductSpace ℂ E]
    (v : ι → E) (e : κ → ι) :
    ((gramKernel v).submatrix e e).PosSemidef :=
  (gramKernel_posSemidef v).submatrix e

/-- Exact rank-one quadratic certificate over the reals. -/
theorem rankOne_quadratic_nonneg
    {ι : Type*} [Fintype ι] (u x : ι → ℝ) :
    0 ≤ (∑ i, u i * x i) ^ 2 :=
  sq_nonneg _

/-- A concrete Hermitian kernel can still have a negative direction.
This rules out replacing the exact Weil functional by a generic
"find a Hermitian kernel" plan. -/
def badKernel : Matrix (Fin 2) (Fin 2) ℝ :=
  !![1, 2; 2, 1]

def badDirection : Fin 2 → ℝ :=
  ![1, -1]

theorem badKernel_negative_direction :
    (∑ i, ∑ j, badDirection i * badKernel i j * badDirection j) = -2 := by
  norm_num [Fin.sum_univ_two, badDirection, badKernel]

theorem badKernel_not_posSemidef : ¬ badKernel.PosSemidef := by
  intro h
  have hnonneg := h.dotProduct_mulVec_nonneg badDirection
  change 0 ≤ ∑ i, badDirection i * (∑ j, badKernel i j * badDirection j) at hnonneg
  have heq :
      (∑ i, badDirection i * (∑ j, badKernel i j * badDirection j)) = -2 := by
    simpa only [Finset.mul_sum, mul_assoc] using badKernel_negative_direction
  rw [heq] at hnonneg
  norm_num at hnonneg

end WeilPositivity
