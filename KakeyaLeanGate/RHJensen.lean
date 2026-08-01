import KakeyaLeanGate.LiCriterion
import Mathlib.Analysis.Complex.TaylorSeries
import Mathlib.Analysis.Complex.Polynomial.GaussLucas
import Mathlib.Analysis.Analytic.IsolatedZeros
import Mathlib.Analysis.Analytic.OfScalars
import Mathlib.Analysis.Calculus.Deriv.Star
import Mathlib.Analysis.Calculus.Deriv.Shift
import Mathlib.Analysis.Calculus.IteratedDeriv.Lemmas
import Mathlib.Analysis.Complex.LocallyUniformLimit
import Mathlib.Analysis.Complex.JensenFormula
import Mathlib.Analysis.Complex.BranchLogRoot
import Mathlib.Analysis.Complex.BorelCaratheodory
import Mathlib.Analysis.Complex.AbsMax
import Mathlib.Analysis.Normed.Module.MultipliableUniformlyOn
import Mathlib.Analysis.Real.Pi.Bounds
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Log.Summable
import Mathlib.Analysis.SpecialFunctions.Complex.Analytic
import Mathlib.Analysis.SpecialFunctions.ImproperIntegrals
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Probability.Moments.IntegrableExpMul
import Mathlib.Probability.Moments.ComplexMGF
import Mathlib.Probability.Distributions.Poisson.PoissonLimitThm
import Mathlib.Analysis.Normed.Group.Tannery
import Mathlib.Analysis.Normed.Group.FunctionSeries
import Mathlib.LinearAlgebra.Matrix.Determinant.Basic
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.Topology.Algebra.Polynomial

/-!
# The Jensen-polynomial / Laguerre--Pólya route

This module formalizes only the unconditional beginning of the route.
In particular, `AllJensenHyperbolic xiGamma` is **not** proved here and no
equivalence with `RiemannHypothesis` is asserted.

The normalization follows Griffin--Ono--Rolen--Zagier, PNAS 116 (2019),
11103--11110, equations (1) and (2): their generating entire function is
`8 * ξ (1/2 + z)`, and
`J_a^{d,n}(X) = ∑_{j=0}^d choose(d,j) a(n+j) X^j`.

Mathlib does not currently expose a polynomial real-rootedness predicate, so
`Hyperbolic` below is a small standard project-local interface: every complex
root of the scalar-extended real polynomial is real.
-/

noncomputable section

open Complex Polynomial MeasureTheory Filter
open scoped ComplexConjugate Topology

namespace Kakeya.RHJensen

attribute [local fun_prop] differentiable_completedZeta₀

/-- The entire continuation of the classical
`ξ(s) = s(s-1) Λ(s) / 2`.

Mathlib's `completedRiemannZeta₀` is the entire function satisfying
`Λ(s) = Λ₀(s) - 1/s - 1/(1-s)`.  Multiplication by `s(s-1)` gives the
pole-free expression below. -/
def riemannXi (s : ℂ) : ℂ :=
  (1 + s * (s - 1) * completedRiemannZeta₀ s) / 2

theorem differentiable_riemannXi : Differentiable ℂ riemannXi := by
  unfold riemannXi
  fun_prop

attribute [local fun_prop] differentiable_riemannXi

/-- The actual modified Jacobi-theta kernel used by pinned Mathlib to define
the pole-removed completed Riemann zeta through a Mellin transform. -/
def completedZetaMellinKernel : ℝ → ℂ :=
  (HurwitzZeta.hurwitzEvenFEPair 0).f_modif

/-- Pinned Mathlib's exact integral normalization:
`Λ₀(s) = Mellin(completedZetaMellinKernel)(s/2) / 2`.

This representation is unconditional.  Its kernel is the *modified* weak
functional-equation kernel; coefficient positivity requires a further
integration-by-parts identity with Riemann's positive `Φ` kernel, which is not
present in the pinned library. -/
theorem completedRiemannZeta₀_eq_mellin (s : ℂ) :
    completedRiemannZeta₀ s =
      mellin completedZetaMellinKernel (s / 2) / 2 := by
  rfl

/-- The `n`-th summand of Riemann's classical even Fourier kernel `Φ`.
We use `n + 1` to index the source sum from `1`, and `|u|` to make the
authoritative positive-half-line formula even by definition.  The factored
form is algebraically

`(4 π² m⁴ exp(9|u|/2) - 6 π m² exp(5|u|/2))
  exp(-π m² exp(2|u|))`, where `m = n+1`. -/
def riemannPhiTerm (n : ℕ) (u : ℝ) : ℝ :=
  let m : ℝ := n + 1
  let v := |u|
  2 * Real.pi * m ^ 2 * Real.exp (5 * v / 2) *
    (2 * Real.pi * m ^ 2 * Real.exp (2 * v) - 3) *
      Real.exp (-Real.pi * m ^ 2 * Real.exp (2 * v))

theorem riemannPhiTerm_eq_source_formula (n : ℕ) (u : ℝ) :
    riemannPhiTerm n u =
      (4 * Real.pi ^ 2 * (n + 1 : ℝ) ^ 4 *
          Real.exp (9 * |u| / 2) -
        6 * Real.pi * (n + 1 : ℝ) ^ 2 *
          Real.exp (5 * |u| / 2)) *
        Real.exp (-Real.pi * (n + 1 : ℝ) ^ 2 *
          Real.exp (2 * |u|)) := by
  rw [riemannPhiTerm]
  rw [show Real.exp (9 * |u| / 2) =
      Real.exp (5 * |u| / 2) * Real.exp (2 * |u|) by
    rw [← Real.exp_add]
    congr 1
    ring]
  ring

/-- Positive-half-line theta summand whose shifted second derivative is the
corresponding Riemann `Φ` summand. -/
def riemannThetaTerm (n : ℕ) (u : ℝ) : ℝ :=
  Real.exp (u / 2) *
    Real.exp (-Real.pi * (n + 1 : ℝ) ^ 2 * Real.exp (2 * u))

theorem hasDerivAt_riemannThetaTerm_raw (n : ℕ) (u : ℝ) :
    HasDerivAt (riemannThetaTerm n)
      (Real.exp (u / 2) / 2 *
          Real.exp (-Real.pi * (n + 1 : ℝ) ^ 2 * Real.exp (2 * u)) +
        Real.exp (u / 2) *
          (Real.exp (-Real.pi * (n + 1 : ℝ) ^ 2 * Real.exp (2 * u)) *
            (-2 * Real.pi * (n + 1 : ℝ) ^ 2 * Real.exp (2 * u)))) u := by
  have hleft :
      HasDerivAt (fun x : ℝ ↦ Real.exp (x / 2))
        (Real.exp (u / 2) / 2) u :=
    by simpa only [id_eq, div_eq_mul_inv, one_mul] using
      ((hasDerivAt_id' u).div_const 2).exp
  have hinner := (((hasDerivAt_id' u).const_mul 2).exp.const_mul
    (-Real.pi * (n + 1 : ℝ) ^ 2))
  have hraw := hleft.mul hinner.exp
  rw [show
      Real.exp (u / 2) / 2 *
            Real.exp (-Real.pi * (n + 1 : ℝ) ^ 2 * Real.exp (2 * u)) +
          Real.exp (u / 2) *
            (Real.exp (-Real.pi * (n + 1 : ℝ) ^ 2 * Real.exp (2 * u)) *
              (-Real.pi * (n + 1 : ℝ) ^ 2 *
                (Real.exp (2 * u) * (2 * 1)))) =
        Real.exp (u / 2) / 2 *
            Real.exp (-Real.pi * (n + 1 : ℝ) ^ 2 * Real.exp (2 * u)) +
          Real.exp (u / 2) *
            (Real.exp (-Real.pi * (n + 1 : ℝ) ^ 2 * Real.exp (2 * u)) *
              (-2 * Real.pi * (n + 1 : ℝ) ^ 2 * Real.exp (2 * u))) by ring] at hraw
  change HasDerivAt
    ((fun x : ℝ ↦ Real.exp (x / 2)) *
      fun x : ℝ ↦
        Real.exp (-Real.pi * (n + 1 : ℝ) ^ 2 * Real.exp (2 * x))) _ u
  exact hraw

theorem hasDerivAt_riemannThetaTerm (n : ℕ) (u : ℝ) :
    HasDerivAt (riemannThetaTerm n)
      ((1 / 2 - 2 * Real.pi * (n + 1 : ℝ) ^ 2 * Real.exp (2 * u)) *
        riemannThetaTerm n u) u := by
  have h := hasDerivAt_riemannThetaTerm_raw n u
  rw [show
      Real.exp (u / 2) / 2 *
            Real.exp (-Real.pi * (n + 1 : ℝ) ^ 2 * Real.exp (2 * u)) +
          Real.exp (u / 2) *
            (Real.exp (-Real.pi * (n + 1 : ℝ) ^ 2 * Real.exp (2 * u)) *
              (-2 * Real.pi * (n + 1 : ℝ) ^ 2 * Real.exp (2 * u))) =
        (1 / 2 - 2 * Real.pi * (n + 1 : ℝ) ^ 2 * Real.exp (2 * u)) *
          riemannThetaTerm n u by
    unfold riemannThetaTerm
    ring] at h
  exact h

theorem deriv_riemannThetaTerm (n : ℕ) (u : ℝ) :
    deriv (riemannThetaTerm n) u =
      (1 / 2 - 2 * Real.pi * (n + 1 : ℝ) ^ 2 * Real.exp (2 * u)) *
        riemannThetaTerm n u := by
  exact (hasDerivAt_riemannThetaTerm n u).deriv

theorem hasDerivAt_deriv_riemannThetaTerm_raw (n : ℕ) (u : ℝ) :
    HasDerivAt (deriv (riemannThetaTerm n))
      (-4 * Real.pi * (n + 1 : ℝ) ^ 2 * Real.exp (2 * u) *
          riemannThetaTerm n u +
        (1 / 2 - 2 * Real.pi * (n + 1 : ℝ) ^ 2 * Real.exp (2 * u)) *
          ((1 / 2 - 2 * Real.pi * (n + 1 : ℝ) ^ 2 * Real.exp (2 * u)) *
            riemannThetaTerm n u)) u := by
  have hfun :
      deriv (riemannThetaTerm n) = fun x ↦
        (1 / 2 - 2 * Real.pi * (n + 1 : ℝ) ^ 2 * Real.exp (2 * x)) *
          riemannThetaTerm n x := by
    funext x
    exact deriv_riemannThetaTerm n x
  rw [hfun]
  have hcoeff := (hasDerivAt_const u (1 / 2 : ℝ)).sub
    (((hasDerivAt_id' u).const_mul 2).exp.const_mul
      (2 * Real.pi * (n + 1 : ℝ) ^ 2))
  have hraw := hcoeff.mul (hasDerivAt_riemannThetaTerm n u)
  rw [show
      (0 - 2 * Real.pi * (n + 1 : ℝ) ^ 2 *
          (Real.exp (2 * u) * (2 * 1))) * riemannThetaTerm n u =
        -4 * Real.pi * (n + 1 : ℝ) ^ 2 * Real.exp (2 * u) *
          riemannThetaTerm n u by ring] at hraw
  change HasDerivAt
    ((fun x : ℝ ↦
      1 / 2 - 2 * Real.pi * (n + 1 : ℝ) ^ 2 * Real.exp (2 * x)) *
        riemannThetaTerm n) _ u
  exact hraw

theorem deriv_deriv_riemannThetaTerm (n : ℕ) (u : ℝ) :
    deriv (deriv (riemannThetaTerm n)) u =
      (1 / 4 -
          6 * Real.pi * (n + 1 : ℝ) ^ 2 * Real.exp (2 * u) +
          4 * Real.pi ^ 2 * (n + 1 : ℝ) ^ 4 * Real.exp (4 * u)) *
        riemannThetaTerm n u := by
  rw [(hasDerivAt_deriv_riemannThetaTerm_raw n u).deriv]
  rw [show Real.exp (4 * u) = Real.exp (2 * u) ^ 2 by
    rw [pow_two, ← Real.exp_add]
    congr 1
    ring]
  ring

/-- Summable coefficient in a uniform exponential majorant for `Φ`. -/
def riemannPhiMajorantCoefficient (n : ℕ) : ℝ :=
  4 * Real.pi ^ 2 * (n + 1 : ℝ) ^ 4 *
    Real.exp (-Real.pi * (n + 1 : ℝ) ^ 2)

theorem riemannPhiMajorantCoefficient_pos (n : ℕ) :
    0 < riemannPhiMajorantCoefficient n := by
  unfold riemannPhiMajorantCoefficient
  positivity

set_option maxHeartbeats 400000 in
theorem summable_riemannPhiMajorantCoefficient :
    Summable riemannPhiMajorantCoefficient := by
  have hbase :
      Summable (fun n : ℕ ↦
        (n : ℝ) ^ 4 * Real.exp (-Real.pi * n)) :=
    Real.summable_pow_mul_exp_neg_nat_mul 4 Real.pi_pos
  have hshift :
      Summable (fun n : ℕ ↦
        4 * Real.pi ^ 2 * (n + 1 : ℝ) ^ 4 *
          Real.exp (-Real.pi * (n + 1 : ℝ))) := by
    have hs := ((summable_nat_add_iff 1).2 hbase).mul_left (4 * Real.pi ^ 2)
    refine hs.congr ?_
    intro n
    push_cast
    ring
  apply hshift.of_nonneg_of_le
  · exact fun n ↦ (riemannPhiMajorantCoefficient_pos n).le
  · intro n
    unfold riemannPhiMajorantCoefficient
    apply mul_le_mul_of_nonneg_left _ (by positivity)
    apply Real.exp_le_exp.mpr
    have hm : (n + 1 : ℝ) ≤ (n + 1 : ℝ) ^ 2 := by
      have hn : 0 ≤ (n : ℝ) := Nat.cast_nonneg n
      nlinarith
    nlinarith [Real.pi_pos]

/-- One summable coefficient controls the theta summands and their first two
derivatives on every bounded positive interval. -/
def riemannThetaC2Majorant (R : ℝ) (n : ℕ) : ℝ :=
  Real.exp (R / 2) *
    (1 + Real.exp (2 * R) + Real.exp (4 * R)) *
      riemannPhiMajorantCoefficient n

theorem summable_riemannThetaC2Majorant (R : ℝ) :
    Summable (riemannThetaC2Majorant R) := by
  exact summable_riemannPhiMajorantCoefficient.mul_left
    (Real.exp (R / 2) * (1 + Real.exp (2 * R) + Real.exp (4 * R)))

/-- A summable derivative majorant on the two-sided neighborhood `(-1,1)`.
This is only needed to identify the exact derivative at the modular boundary
`u = 0`; positive-half estimates alone do not justify that derivative. -/
def riemannThetaLocalDerivMajorant (n : ℕ) : ℝ :=
  Real.exp (1 / 2 : ℝ) *
    (1 + 2 * Real.pi * Real.exp 2) *
      (n + 1 : ℝ) ^ 4 *
        Real.exp (-(Real.pi * Real.exp (-2)) * (n + 1 : ℝ) ^ 2)

theorem summable_riemannThetaLocalDerivMajorant :
    Summable riemannThetaLocalDerivMajorant := by
  let c := Real.pi * Real.exp (-2)
  let K := Real.exp (1 / 2 : ℝ) * (1 + 2 * Real.pi * Real.exp 2)
  have hc : 0 < c := by
    dsimp only [c]
    positivity
  have hbase :
      Summable (fun n : ℕ ↦
        (n : ℝ) ^ 4 * Real.exp (-c * n)) :=
    Real.summable_pow_mul_exp_neg_nat_mul 4 hc
  have hshift :
      Summable (fun n : ℕ ↦
        K * (n + 1 : ℝ) ^ 4 * Real.exp (-c * (n + 1 : ℝ))) := by
    have hs := ((summable_nat_add_iff 1).2 hbase).mul_left K
    refine hs.congr ?_
    intro n
    push_cast
    ring
  apply hshift.of_nonneg_of_le
  · intro n
    unfold riemannThetaLocalDerivMajorant
    positivity
  · intro n
    have hm : (n + 1 : ℝ) ≤ (n + 1 : ℝ) ^ 2 := by
      have hn : 0 ≤ (n : ℝ) := Nat.cast_nonneg n
      nlinarith
    have hexp :
        Real.exp (-c * (n + 1 : ℝ) ^ 2) ≤
          Real.exp (-c * (n + 1 : ℝ)) := by
      apply Real.exp_le_exp.mpr
      nlinarith
    dsimp only [riemannThetaLocalDerivMajorant, K, c]
    gcongr

theorem norm_deriv_riemannThetaTerm_le_localMajorant
    (n : ℕ) {u : ℝ} (hu : u ∈ Set.Ioo (-1 : ℝ) 1) :
    ‖deriv (riemannThetaTerm n) u‖ ≤
      riemannThetaLocalDerivMajorant n := by
  let m : ℝ := n + 1
  let c := Real.pi * Real.exp (-2)
  have hm : 1 ≤ m := by
    dsimp only [m]
    exact_mod_cast Nat.succ_pos n
  have hm2 : 1 ≤ m ^ 2 := by nlinarith [sq_nonneg m]
  have hm24 : m ^ 2 ≤ m ^ 4 := by nlinarith [sq_nonneg (m ^ 2 - m)]
  have huL := hu.1
  have huU := hu.2
  have hhalf : Real.exp (u / 2) ≤ Real.exp (1 / 2 : ℝ) := by
    apply Real.exp_le_exp.mpr
    linarith
  have hEupper : Real.exp (2 * u) ≤ Real.exp 2 := by
    apply Real.exp_le_exp.mpr
    linarith
  have hElower : Real.exp (-2) ≤ Real.exp (2 * u) := by
    apply Real.exp_le_exp.mpr
    linarith
  have hdamp :
      Real.exp (-Real.pi * m ^ 2 * Real.exp (2 * u)) ≤
        Real.exp (-c * m ^ 2) := by
    apply Real.exp_le_exp.mpr
    dsimp only [c]
    have hp := mul_le_mul_of_nonneg_left hElower
      (mul_nonneg Real.pi_pos.le (sq_nonneg m))
    nlinarith
  have hcoef :
      |1 / 2 - 2 * Real.pi * m ^ 2 * Real.exp (2 * u)| ≤
        (1 + 2 * Real.pi * Real.exp 2) * m ^ 4 := by
    calc
      |1 / 2 - 2 * Real.pi * m ^ 2 * Real.exp (2 * u)| ≤
          1 / 2 + 2 * Real.pi * m ^ 2 * Real.exp (2 * u) := by
        calc
          _ ≤ |(1 / 2 : ℝ)| +
              |2 * Real.pi * m ^ 2 * Real.exp (2 * u)| := abs_sub _ _
          _ = _ := by
            rw [abs_of_nonneg (by norm_num),
              abs_of_nonneg (by positivity)]
      _ ≤ 1 + 2 * Real.pi * m ^ 4 * Real.exp 2 := by
        gcongr
        norm_num
      _ ≤ (1 + 2 * Real.pi * Real.exp 2) * m ^ 4 := by
        nlinarith [Real.pi_pos, Real.exp_pos 2]
  rw [deriv_riemannThetaTerm, Real.norm_eq_abs, abs_mul]
  have hthetaPos : 0 < riemannThetaTerm n u := by
    unfold riemannThetaTerm
    positivity
  rw [show |riemannThetaTerm n u| = riemannThetaTerm n u from
    abs_of_pos hthetaPos]
  calc
    |1 / 2 - 2 * Real.pi * (n + 1 : ℝ) ^ 2 * Real.exp (2 * u)| *
        riemannThetaTerm n u ≤
      ((1 + 2 * Real.pi * Real.exp 2) * m ^ 4) *
        (Real.exp (1 / 2 : ℝ) * Real.exp (-c * m ^ 2)) := by
      dsimp only [riemannThetaTerm, m]
      gcongr
    _ = riemannThetaLocalDerivMajorant n := by
      dsimp only [riemannThetaLocalDerivMajorant, m, c]
      ring

theorem norm_riemannThetaTerm_le_basicMajorant
    (n : ℕ) {u R : ℝ} (hu : 0 ≤ u) (huR : u ≤ R) :
    ‖riemannThetaTerm n u‖ ≤
      Real.exp (R / 2) *
        Real.exp (-Real.pi * (n + 1 : ℝ) ^ 2) := by
  let m : ℝ := n + 1
  have hE : 1 ≤ Real.exp (2 * u) :=
    Real.one_le_exp (mul_nonneg (by norm_num) hu)
  have hneg :
      Real.exp (-Real.pi * m ^ 2 * Real.exp (2 * u)) ≤
        Real.exp (-Real.pi * m ^ 2) := by
    apply Real.exp_le_exp.mpr
    have hp : Real.pi * m ^ 2 ≤
        Real.pi * m ^ 2 * Real.exp (2 * u) := by
      exact le_mul_of_one_le_right (mul_nonneg Real.pi_pos.le (sq_nonneg m)) hE
    linarith
  have hhalf : Real.exp (u / 2) ≤ Real.exp (R / 2) := by
    apply Real.exp_le_exp.mpr
    linarith
  rw [Real.norm_eq_abs, abs_of_pos (by unfold riemannThetaTerm; positivity)]
  unfold riemannThetaTerm
  exact mul_le_mul hhalf hneg (by positivity) (by positivity)

theorem norm_riemannThetaTerm_le_C2Majorant
    (n : ℕ) {u R : ℝ} (hu : 0 ≤ u) (huR : u ≤ R) :
    ‖riemannThetaTerm n u‖ ≤ riemannThetaC2Majorant R n := by
  have hbasic := norm_riemannThetaTerm_le_basicMajorant n hu huR
  let m : ℝ := n + 1
  have hm : 1 ≤ m := by
    dsimp only [m]
    exact_mod_cast Nat.succ_pos n
  unfold riemannThetaC2Majorant riemannPhiMajorantCoefficient
  have hm4 : 1 ≤ m ^ 4 := by
    exact one_le_pow₀ hm
  have hpi : 1 ≤ 4 * Real.pi ^ 2 := by
    nlinarith [Real.pi_gt_three, sq_nonneg Real.pi]
  have hA : 1 ≤ 4 * Real.pi ^ 2 * m ^ 4 := by
    calc
      (1 : ℝ) = 1 * 1 := by ring
      _ ≤ (4 * Real.pi ^ 2) * m ^ 4 :=
        mul_le_mul hpi hm4 zero_le_one (by positivity)
  have hfac :
      1 ≤ (1 + Real.exp (2 * R) + Real.exp (4 * R)) *
        (4 * Real.pi ^ 2 * m ^ 4) := by
    calc
      (1 : ℝ) ≤ 1 + Real.exp (2 * R) + Real.exp (4 * R) := by
        nlinarith [Real.exp_pos (2 * R), Real.exp_pos (4 * R)]
      _ ≤ (1 + Real.exp (2 * R) + Real.exp (4 * R)) *
          (4 * Real.pi ^ 2 * m ^ 4) := by
        exact le_mul_of_one_le_right
          (by nlinarith [Real.exp_pos (2 * R), Real.exp_pos (4 * R)]) hA
  calc
    ‖riemannThetaTerm n u‖ ≤
        Real.exp (R / 2) *
          Real.exp (-Real.pi * (n + 1 : ℝ) ^ 2) := hbasic
    _ ≤ Real.exp (R / 2) *
          ((1 + Real.exp (2 * R) + Real.exp (4 * R)) *
            (4 * Real.pi ^ 2 * (n + 1 : ℝ) ^ 4)) *
          Real.exp (-Real.pi * (n + 1 : ℝ) ^ 2) := by
      apply mul_le_mul_of_nonneg_right _ (Real.exp_pos _).le
      simpa [m] using
        (mul_le_mul_of_nonneg_left hfac (Real.exp_pos (R / 2)).le)
    _ = _ := by ring

theorem norm_deriv_riemannThetaTerm_le_C2Majorant
    (n : ℕ) {u R : ℝ} (hu : 0 ≤ u) (huR : u ≤ R) :
    ‖deriv (riemannThetaTerm n) u‖ ≤ riemannThetaC2Majorant R n := by
  rw [deriv_riemannThetaTerm, Real.norm_eq_abs, abs_mul]
  have htheta := norm_riemannThetaTerm_le_basicMajorant n hu huR
  rw [Real.norm_eq_abs] at htheta
  have hcoef :
      |1 / 2 - 2 * Real.pi * (n + 1 : ℝ) ^ 2 * Real.exp (2 * u)| ≤
        1 / 2 + 2 * Real.pi * (n + 1 : ℝ) ^ 2 * Real.exp (2 * R) := by
    calc
      |1 / 2 - 2 * Real.pi * (n + 1 : ℝ) ^ 2 * Real.exp (2 * u)| ≤
          |(1 / 2 : ℝ)| +
            |2 * Real.pi * (n + 1 : ℝ) ^ 2 * Real.exp (2 * u)| := abs_sub _ _
      _ = 1 / 2 + 2 * Real.pi * (n + 1 : ℝ) ^ 2 * Real.exp (2 * u) := by
        rw [abs_of_nonneg (by norm_num), abs_of_nonneg (by positivity)]
      _ ≤ 1 / 2 + 2 * Real.pi * (n + 1 : ℝ) ^ 2 * Real.exp (2 * R) := by
        gcongr
  let m : ℝ := n + 1
  let A := 4 * Real.pi ^ 2 * m ^ 4
  have hm : 1 ≤ m := by
    dsimp only [m]
    exact_mod_cast Nat.succ_pos n
  have hA0 : 0 ≤ A := by dsimp only [A]; positivity
  have hAhalf : (1 / 2 : ℝ) ≤ A := by
    dsimp only [A]
    have hm4 : 1 ≤ m ^ 4 := one_le_pow₀ hm
    have hpi : (1 / 2 : ℝ) ≤ 4 * Real.pi ^ 2 := by
      nlinarith [Real.pi_gt_three, sq_nonneg Real.pi]
    calc
      (1 / 2 : ℝ) = (1 / 2) * 1 := by ring
      _ ≤ (4 * Real.pi ^ 2) * m ^ 4 :=
        mul_le_mul hpi hm4 (by norm_num) (by positivity)
  have hAlin : 2 * Real.pi * m ^ 2 ≤ A := by
    dsimp only [A]
    have hone : 1 ≤ 2 * Real.pi * m ^ 2 := by
      have hm2 : 1 ≤ m ^ 2 := one_le_pow₀ hm
      nlinarith [Real.pi_gt_three]
    calc
      2 * Real.pi * m ^ 2 =
          (2 * Real.pi * m ^ 2) * 1 := by ring
      _ ≤ (2 * Real.pi * m ^ 2) * (2 * Real.pi * m ^ 2) := by gcongr
      _ = 4 * Real.pi ^ 2 * m ^ 4 := by ring
  have hfactor :
      1 / 2 + 2 * Real.pi * m ^ 2 * Real.exp (2 * R) ≤
        (1 + Real.exp (2 * R) + Real.exp (4 * R)) * A := by
    nlinarith [mul_le_mul_of_nonneg_right hAlin (Real.exp_pos (2 * R)).le,
      mul_nonneg hA0 (Real.exp_pos (4 * R)).le]
  unfold riemannThetaC2Majorant riemannPhiMajorantCoefficient
  dsimp only [m, A] at hcoef hfactor ⊢
  calc
    |1 / 2 - 2 * Real.pi * (n + 1 : ℝ) ^ 2 * Real.exp (2 * u)| *
        |riemannThetaTerm n u| ≤
      (1 / 2 + 2 * Real.pi * (n + 1 : ℝ) ^ 2 * Real.exp (2 * R)) *
        |riemannThetaTerm n u| := by gcongr
    _ ≤ (1 / 2 + 2 * Real.pi * (n + 1 : ℝ) ^ 2 * Real.exp (2 * R)) *
        (Real.exp (R / 2) *
          Real.exp (-Real.pi * (n + 1 : ℝ) ^ 2)) := by
      gcongr
    _ ≤ ((1 + Real.exp (2 * R) + Real.exp (4 * R)) *
          (4 * Real.pi ^ 2 * (n + 1 : ℝ) ^ 4)) *
        (Real.exp (R / 2) *
          Real.exp (-Real.pi * (n + 1 : ℝ) ^ 2)) := by
      gcongr
    _ = _ := by ring

theorem norm_deriv_deriv_riemannThetaTerm_le_C2Majorant
    (n : ℕ) {u R : ℝ} (hu : 0 ≤ u) (huR : u ≤ R) :
    ‖deriv (deriv (riemannThetaTerm n)) u‖ ≤
      riemannThetaC2Majorant R n := by
  rw [deriv_deriv_riemannThetaTerm, Real.norm_eq_abs, abs_mul]
  have htheta := norm_riemannThetaTerm_le_basicMajorant n hu huR
  rw [Real.norm_eq_abs] at htheta
  let m : ℝ := n + 1
  let A := 4 * Real.pi ^ 2 * m ^ 4
  have hm : 1 ≤ m := by
    dsimp only [m]
    exact_mod_cast Nat.succ_pos n
  have hcoef :
      |1 / 4 - 6 * Real.pi * m ^ 2 * Real.exp (2 * u) +
          4 * Real.pi ^ 2 * m ^ 4 * Real.exp (4 * u)| ≤
        1 / 4 + 6 * Real.pi * m ^ 2 * Real.exp (2 * R) +
          4 * Real.pi ^ 2 * m ^ 4 * Real.exp (4 * R) := by
    have he2 : Real.exp (2 * u) ≤ Real.exp (2 * R) :=
      Real.exp_le_exp.mpr (by linarith)
    have he4 : Real.exp (4 * u) ≤ Real.exp (4 * R) :=
      Real.exp_le_exp.mpr (by linarith)
    have hy : 6 * Real.pi * m ^ 2 * Real.exp (2 * u) ≤
        6 * Real.pi * m ^ 2 * Real.exp (2 * R) := by gcongr
    have hz : 4 * Real.pi ^ 2 * m ^ 4 * Real.exp (4 * u) ≤
        4 * Real.pi ^ 2 * m ^ 4 * Real.exp (4 * R) := by gcongr
    have hyu0 : 0 ≤ 6 * Real.pi * m ^ 2 * Real.exp (2 * u) := by positivity
    have hyR0 : 0 ≤ 6 * Real.pi * m ^ 2 * Real.exp (2 * R) := by positivity
    have hzu0 : 0 ≤ 4 * Real.pi ^ 2 * m ^ 4 * Real.exp (4 * u) := by positivity
    have hzR0 : 0 ≤ 4 * Real.pi ^ 2 * m ^ 4 * Real.exp (4 * R) := by positivity
    rw [abs_le]
    constructor <;> nlinarith
  have hA0 : 0 ≤ A := by dsimp only [A]; positivity
  have hAquarter : (1 / 4 : ℝ) ≤ A := by
    dsimp only [A]
    have hm4 : 1 ≤ m ^ 4 := one_le_pow₀ hm
    have hpi : (1 / 4 : ℝ) ≤ 4 * Real.pi ^ 2 := by
      nlinarith [Real.pi_gt_three, sq_nonneg Real.pi]
    calc
      (1 / 4 : ℝ) = (1 / 4) * 1 := by ring
      _ ≤ (4 * Real.pi ^ 2) * m ^ 4 :=
        mul_le_mul hpi hm4 (by norm_num) (by positivity)
  have hAsix : 6 * Real.pi * m ^ 2 ≤ A := by
    dsimp only [A]
    have hm2 : 1 ≤ m ^ 2 := one_le_pow₀ hm
    have hratio : 6 ≤ 4 * Real.pi * m ^ 2 := by
      nlinarith [Real.pi_gt_three]
    calc
      6 * Real.pi * m ^ 2 ≤
          (4 * Real.pi * m ^ 2) * (Real.pi * m ^ 2) := by
        simpa only [mul_assoc] using
          (mul_le_mul_of_nonneg_right hratio
            (mul_nonneg Real.pi_pos.le (sq_nonneg m)))
      _ = 4 * Real.pi ^ 2 * m ^ 4 := by ring
  have hfactor :
      1 / 4 + 6 * Real.pi * m ^ 2 * Real.exp (2 * R) +
          4 * Real.pi ^ 2 * m ^ 4 * Real.exp (4 * R) ≤
        (1 + Real.exp (2 * R) + Real.exp (4 * R)) * A := by
    nlinarith [mul_le_mul_of_nonneg_right hAsix (Real.exp_pos (2 * R)).le]
  unfold riemannThetaC2Majorant riemannPhiMajorantCoefficient
  dsimp only [m, A] at hcoef hfactor ⊢
  calc
    |1 / 4 - 6 * Real.pi * (n + 1 : ℝ) ^ 2 * Real.exp (2 * u) +
          4 * Real.pi ^ 2 * (n + 1 : ℝ) ^ 4 * Real.exp (4 * u)| *
        |riemannThetaTerm n u| ≤
      (1 / 4 + 6 * Real.pi * (n + 1 : ℝ) ^ 2 * Real.exp (2 * R) +
          4 * Real.pi ^ 2 * (n + 1 : ℝ) ^ 4 * Real.exp (4 * R)) *
        |riemannThetaTerm n u| := by gcongr
    _ ≤ (1 / 4 + 6 * Real.pi * (n + 1 : ℝ) ^ 2 * Real.exp (2 * R) +
          4 * Real.pi ^ 2 * (n + 1 : ℝ) ^ 4 * Real.exp (4 * R)) *
        (Real.exp (R / 2) *
          Real.exp (-Real.pi * (n + 1 : ℝ) ^ 2)) := by
      gcongr
    _ ≤ ((1 + Real.exp (2 * R) + Real.exp (4 * R)) *
          (4 * Real.pi ^ 2 * (n + 1 : ℝ) ^ 4)) *
        (Real.exp (R / 2) *
          Real.exp (-Real.pi * (n + 1 : ℝ) ^ 2)) := by
      gcongr
    _ = _ := by ring

theorem tendstoUniformlyOn_riemannThetaTerm_Icc (R : ℝ) :
    TendstoUniformlyOn
      (fun N u ↦ ∑ n ∈ Finset.range N, riemannThetaTerm n u)
      (fun u ↦ ∑' n : ℕ, riemannThetaTerm n u)
      Filter.atTop (Set.Icc 0 R) := by
  apply tendstoUniformlyOn_tsum_nat (summable_riemannThetaC2Majorant R)
  intro n u hu
  exact norm_riemannThetaTerm_le_C2Majorant n hu.1 hu.2

theorem tendstoUniformlyOn_deriv_riemannThetaTerm_Icc (R : ℝ) :
    TendstoUniformlyOn
      (fun N u ↦ ∑ n ∈ Finset.range N, deriv (riemannThetaTerm n) u)
      (fun u ↦ ∑' n : ℕ, deriv (riemannThetaTerm n) u)
      Filter.atTop (Set.Icc 0 R) := by
  apply tendstoUniformlyOn_tsum_nat (summable_riemannThetaC2Majorant R)
  intro n u hu
  exact norm_deriv_riemannThetaTerm_le_C2Majorant n hu.1 hu.2

theorem tendstoUniformlyOn_deriv_deriv_riemannThetaTerm_Icc (R : ℝ) :
    TendstoUniformlyOn
      (fun N u ↦ ∑ n ∈ Finset.range N,
        deriv (deriv (riemannThetaTerm n)) u)
      (fun u ↦ ∑' n : ℕ, deriv (deriv (riemannThetaTerm n)) u)
      Filter.atTop (Set.Icc 0 R) := by
  apply tendstoUniformlyOn_tsum_nat (summable_riemannThetaC2Majorant R)
  intro n u hu
  exact norm_deriv_deriv_riemannThetaTerm_le_C2Majorant n hu.1 hu.2

/-- The source kernel is termwise the shifted second derivative of the theta
tail on the positive half-line. -/
theorem riemannPhiTerm_eq_theta_shifted_second_deriv
    (n : ℕ) {u : ℝ} (hu : 0 ≤ u) :
    riemannPhiTerm n u =
      deriv (deriv (riemannThetaTerm n)) u -
        (1 / 4 : ℝ) * riemannThetaTerm n u := by
  rw [deriv_deriv_riemannThetaTerm]
  simp only [riemannPhiTerm, riemannThetaTerm, abs_of_nonneg hu]
  rw [show Real.exp (5 * u / 2) =
      Real.exp (2 * u) * Real.exp (u / 2) by
        rw [← Real.exp_add]; congr 1; ring]
  rw [show Real.exp (4 * u) =
      Real.exp (2 * u) * Real.exp (2 * u) by
        rw [← Real.exp_add]; congr 1; ring]
  ring

/-- Positive-half-line theta tail underlying the modified Mellin kernel. -/
def riemannThetaTail (u : ℝ) : ℝ :=
  ∑' n : ℕ, riemannThetaTerm n u

theorem continuousOn_riemannThetaTail_Icc (R : ℝ) :
    ContinuousOn riemannThetaTail (Set.Icc 0 R) := by
  apply (tendstoUniformlyOn_riemannThetaTerm_Icc R).continuousOn
  exact Filter.Frequently.of_forall fun N ↦ by
    apply Continuous.continuousOn
    apply continuous_finsetSum
    intro n _
    exact continuous_iff_continuousAt.mpr fun u ↦
      (hasDerivAt_riemannThetaTerm n u).continuousAt

theorem evenKernel_zero_sub_one_eq_two_mul_thetaSeries
    {x : ℝ} (hx : 0 < x) :
    HurwitzZeta.evenKernel 0 x - 1 =
      2 * ∑' n : ℕ,
        Real.exp (-Real.pi * (n + 1 : ℝ) ^ 2 * x) := by
  let f : ℤ → ℝ := fun n ↦
    if (n : ℝ) = 0 then 0 else Real.exp (-Real.pi * (n : ℝ) ^ 2 * x)
  have hs : HasSum f (HurwitzZeta.evenKernel 0 x - 1) := by
    simpa [f] using
      (HurwitzZeta.hasSum_int_evenKernel₀ (a := 0) hx)
  have heven : Function.Even f := by
    intro n
    simp only [f, Int.cast_neg, neg_eq_zero, neg_sq]
  rw [← hs.tsum_eq, tsum_int_eq_zero_add_two_mul_tsum_pnat heven hs.summable]
  simp only [f, Int.cast_zero, if_pos, zero_add, nsmul_eq_mul, Nat.cast_ofNat]
  change 2 * (∑' n : ℕ+, f (n : ℤ)) =
    2 * ∑' n : ℕ, Real.exp (-Real.pi * (n + 1 : ℝ) ^ 2 * x)
  rw [← Equiv.pnatEquivNat.symm.tsum_eq]
  congr 2
  funext n
  have hne : (n : ℤ) + 1 ≠ 0 := by omega
  simp [f, Equiv.pnatEquivNat, Nat.succPNat]
  exact fun h ↦ (hne h).elim

theorem two_mul_riemannThetaTail_eq_evenKernel (u : ℝ) :
    2 * riemannThetaTail u =
      Real.exp (u / 2) *
        (HurwitzZeta.evenKernel 0 (Real.exp (2 * u)) - 1) := by
  rw [evenKernel_zero_sub_one_eq_two_mul_thetaSeries (Real.exp_pos (2 * u))]
  unfold riemannThetaTail riemannThetaTerm
  rw [tsum_mul_left]
  ring

theorem cosKernel_zero_eq_evenKernel_zero (x : ℝ) :
    HurwitzZeta.cosKernel 0 x = HurwitzZeta.evenKernel 0 x := by
  rw [← Complex.ofReal_inj]
  change
    (HurwitzZeta.cosKernel ((0 : ℝ) : UnitAddCircle) x : ℂ) =
      (HurwitzZeta.evenKernel ((0 : ℝ) : UnitAddCircle) x : ℂ)
  rw [HurwitzZeta.cosKernel_def (0 : ℝ) x,
    HurwitzZeta.evenKernel_def (0 : ℝ) x]
  simp

theorem evenKernel_zero_exp_neg_two (u : ℝ) :
    HurwitzZeta.evenKernel 0 (Real.exp (-2 * u)) =
      Real.exp u * HurwitzZeta.evenKernel 0 (Real.exp (2 * u)) := by
  have hfe := HurwitzZeta.evenKernel_functional_equation
    (0 : UnitAddCircle) (Real.exp (-2 * u))
  rw [cosKernel_zero_eq_evenKernel_zero] at hfe
  have hrpow :
      Real.exp (-2 * u) ^ (1 / 2 : ℝ) = Real.exp (-u) := by
    rw [← Real.exp_mul]
    congr 1
    ring
  have hinv :
      1 / Real.exp (-2 * u) = Real.exp (2 * u) := by
    rw [one_div, ← Real.exp_neg]
    congr 1
    ring
  rw [hrpow, hinv] at hfe
  calc
    HurwitzZeta.evenKernel 0 (Real.exp (-2 * u)) =
        1 / Real.exp (-u) *
          HurwitzZeta.evenKernel 0 (Real.exp (2 * u)) := hfe
    _ = Real.exp u *
          HurwitzZeta.evenKernel 0 (Real.exp (2 * u)) := by
      rw [one_div, ← Real.exp_neg]
      congr 2
      ring

theorem riemannThetaTail_neg (u : ℝ) :
    riemannThetaTail (-u) =
      riemannThetaTail u +
        (Real.exp (u / 2) - Real.exp (-u / 2)) / 2 := by
  have hneg := two_mul_riemannThetaTail_eq_evenKernel (-u)
  have hpos := two_mul_riemannThetaTail_eq_evenKernel u
  have hfe := evenKernel_zero_exp_neg_two u
  have hexp :
      Real.exp (-u / 2) * Real.exp u = Real.exp (u / 2) := by
    rw [← Real.exp_add]
    congr 1
    ring
  have htwo :
      2 * riemannThetaTail (-u) =
        2 * riemannThetaTail u +
          Real.exp (u / 2) - Real.exp (-u / 2) := by
    calc
      2 * riemannThetaTail (-u) =
          Real.exp (-u / 2) *
            (HurwitzZeta.evenKernel 0 (Real.exp (-2 * u)) - 1) := by
        convert hneg using 1 <;> ring
      _ = Real.exp (-u / 2) *
            (Real.exp u *
              HurwitzZeta.evenKernel 0 (Real.exp (2 * u)) - 1) := by
        rw [hfe]
      _ = Real.exp (u / 2) *
            (HurwitzZeta.evenKernel 0 (Real.exp (2 * u)) - 1) +
            Real.exp (u / 2) - Real.exp (-u / 2) := by
        rw [← hexp]
        ring
      _ = 2 * riemannThetaTail u +
            Real.exp (u / 2) - Real.exp (-u / 2) := by
        rw [← hpos]
  linarith

theorem hurwitzEvenFEPair_zero_f_modif_of_one_lt
    {x : ℝ} (hx : 1 < x) :
    (HurwitzZeta.hurwitzEvenFEPair 0).f_modif x =
      (HurwitzZeta.evenKernel 0 x - 1 : ℝ) := by
  simp [WeakFEPair.f_modif, HurwitzZeta.hurwitzEvenFEPair,
    Set.mem_Ioi.mpr hx, Set.notMem_Ioo_of_ge hx.le]

theorem hurwitzEvenFEPair_zero_f_modif_of_mem_Ioo
    {x : ℝ} (hx : x ∈ Set.Ioo (0 : ℝ) 1) :
    (HurwitzZeta.hurwitzEvenFEPair 0).f_modif x =
      (HurwitzZeta.evenKernel 0 x - x ^ (-(1 / 2 : ℝ)) : ℝ) := by
  simp [WeakFEPair.f_modif, HurwitzZeta.hurwitzEvenFEPair,
    Set.notMem_Ioi.mpr hx.2.le, hx]

theorem hurwitzEvenFEPair_zero_f_modif_one :
    (HurwitzZeta.hurwitzEvenFEPair 0).f_modif 1 = 0 := by
  simp [WeakFEPair.f_modif]

theorem riemannThetaTail_eq_f_modif {u : ℝ} (hu : 0 < u) :
    (riemannThetaTail u : ℂ) =
      (Real.exp (u / 2) / 2 : ℝ) *
        (HurwitzZeta.hurwitzEvenFEPair 0).f_modif (Real.exp (2 * u)) := by
  have hx : 1 < Real.exp (2 * u) :=
    Real.one_lt_exp_iff.mpr (by linarith)
  rw [hurwitzEvenFEPair_zero_f_modif_of_one_lt hx]
  push_cast
  have h := congrArg ((↑) : ℝ → ℂ)
    (two_mul_riemannThetaTail_eq_evenKernel u)
  push_cast at h
  calc
    (riemannThetaTail u : ℂ) =
        (1 / 2 : ℂ) * (2 * (riemannThetaTail u : ℂ)) := by ring
    _ = (1 / 2 : ℂ) *
        (Complex.exp (u / 2) *
          ((HurwitzZeta.evenKernel 0 (Real.exp (2 * u)) : ℂ) - 1)) := by
      rw [h]
    _ = _ := by ring

theorem hurwitzEvenFEPair_zero_f_modif_exp_two_of_neg
    {u : ℝ} (hu : u < 0) :
    (HurwitzZeta.hurwitzEvenFEPair 0).f_modif (Real.exp (2 * u)) =
      (Real.exp (-u) : ℝ) *
        (HurwitzZeta.hurwitzEvenFEPair 0).f_modif (Real.exp (-2 * u)) := by
  have hxlow : Real.exp (2 * u) ∈ Set.Ioo (0 : ℝ) 1 := by
    exact ⟨Real.exp_pos _, Real.exp_lt_one_iff.mpr (by linarith)⟩
  have hxhigh : 1 < Real.exp (-2 * u) :=
    Real.one_lt_exp_iff.mpr (by linarith)
  rw [hurwitzEvenFEPair_zero_f_modif_of_mem_Ioo hxlow,
    hurwitzEvenFEPair_zero_f_modif_of_one_lt hxhigh]
  have hkernel := evenKernel_zero_exp_neg_two (-u)
  have hrpow :
      Real.exp (2 * u) ^ (-(1 / 2 : ℝ)) = Real.exp (-u) := by
    rw [Real.rpow_def_of_pos (Real.exp_pos _), Real.log_exp]
    congr 1
    ring
  rw [hrpow]
  push_cast
  rw [show -2 * -u = 2 * u by ring,
    show 2 * -u = -2 * u by ring] at hkernel
  rw [hkernel]
  push_cast
  ring

theorem completedRiemannZeta₀_eq_mellin_f_modif (s : ℂ) :
    completedRiemannZeta₀ s =
      mellin (HurwitzZeta.hurwitzEvenFEPair 0).f_modif (s / 2) / 2 := by
  rfl

theorem mellinConvergent_hurwitzEvenFEPair_zero_f_modif (q : ℂ) :
    MellinConvergent (HurwitzZeta.hurwitzEvenFEPair 0).f_modif q :=
  ((HurwitzZeta.hurwitzEvenFEPair 0).toStrongFEPair.hasMellin q).1

/-- Finite-interval form of the Mellin substitution `x = exp (2u)`.
This theorem deliberately uses Mathlib's monotone substitution API, which
requires no continuity of the piecewise `f_modif` integrand at `x = 1`. -/
theorem intervalIntegral_mellin_exp_two_substitution
    (q : ℂ) (a b : ℝ) :
    (∫ u in a..b, (2 * Real.exp (2 * u)) •
      (((Real.exp (2 * u) : ℂ) ^ q) *
        (HurwitzZeta.hurwitzEvenFEPair 0).f_modif (Real.exp (2 * u)))) =
      ∫ x in Real.exp (2 * a)..Real.exp (2 * b),
        (x : ℂ) ^ q *
          (HurwitzZeta.hurwitzEvenFEPair 0).f_modif x := by
  simpa only [Function.comp_apply] using
    (intervalIntegral.integral_deriv_smul_comp_of_deriv_nonneg
      (f := fun u : ℝ ↦ Real.exp (2 * u))
      (f' := fun u : ℝ ↦ 2 * Real.exp (2 * u))
      (g := fun x : ℝ ↦ (x : ℂ) ^ q *
        (HurwitzZeta.hurwitzEvenFEPair 0).f_modif x)
      (a := a) (b := b)
      (by fun_prop)
      (fun u _ ↦ by
        simpa only [id_eq, one_mul, mul_comm] using
          ((hasDerivAt_id u).const_mul 2).exp)
      (fun u _ ↦ by positivity))

theorem mellin_exp_two_integrand_eq_theta
    (s : ℂ) {u : ℝ} (hu : 0 < u) :
    (2 * Real.exp (2 * u)) •
      (((Real.exp (2 * u) : ℂ) ^ (s / 2 - 1)) *
        (HurwitzZeta.hurwitzEvenFEPair 0).f_modif (Real.exp (2 * u))) =
      4 * Complex.exp ((s - 1 / 2) * u) * riemannThetaTail u := by
  have hpow :
      (Real.exp (2 * u) : ℂ) ^ (s / 2 - 1) =
        Complex.exp ((s - 2) * u) := by
    rw [Complex.cpow_def_of_ne_zero
      (Complex.ofReal_ne_zero.mpr (Real.exp_ne_zero (2 * u)))]
    rw [← Complex.ofReal_log (Real.exp_pos (2 * u)).le, Real.log_exp]
    congr 1
    push_cast
    ring
  rw [hpow, Complex.real_smul]
  have htheta := riemannThetaTail_eq_f_modif hu
  have hexp₁ :
      Complex.exp (2 * u) * Complex.exp ((s - 2) * u) =
        Complex.exp (s * u) := by
    rw [← Complex.exp_add]
    congr 1
    ring
  have hexp₂ :
      Complex.exp ((s - 1 / 2) * u) * Complex.exp (u / 2) =
        Complex.exp (s * u) := by
    rw [← Complex.exp_add]
    congr 1
    ring
  calc
    (2 * Real.exp (2 * u) : ℝ) *
        (Complex.exp ((s - 2) * u) *
          (HurwitzZeta.hurwitzEvenFEPair 0).f_modif (Real.exp (2 * u))) =
      2 * (Complex.exp (2 * u) * Complex.exp ((s - 2) * u)) *
          (HurwitzZeta.hurwitzEvenFEPair 0).f_modif (Real.exp (2 * u)) := by
        push_cast
        ring
    _ = 2 * Complex.exp (s * u) *
          (HurwitzZeta.hurwitzEvenFEPair 0).f_modif (Real.exp (2 * u)) := by
      rw [hexp₁]
    _ = 4 * Complex.exp ((s - 1 / 2) * u) *
        ((Real.exp (u / 2) / 2 : ℝ) *
          (HurwitzZeta.hurwitzEvenFEPair 0).f_modif (Real.exp (2 * u))) := by
      push_cast
      rw [← hexp₂]
      ring
    _ = 4 * Complex.exp ((s - 1 / 2) * u) * riemannThetaTail u := by
      rw [← htheta]

theorem mellin_exp_two_integrand_eq_theta_neg
    (s : ℂ) {u : ℝ} (hu : u < 0) :
    (2 * Real.exp (2 * u)) •
      (((Real.exp (2 * u) : ℂ) ^ (s / 2 - 1)) *
        (HurwitzZeta.hurwitzEvenFEPair 0).f_modif (Real.exp (2 * u))) =
      4 * Complex.exp ((s - 1 / 2) * u) * riemannThetaTail (-u) := by
  have hpow :
      (Real.exp (2 * u) : ℂ) ^ (s / 2 - 1) =
        Complex.exp ((s - 2) * u) := by
    rw [Complex.cpow_def_of_ne_zero
      (Complex.ofReal_ne_zero.mpr (Real.exp_ne_zero (2 * u)))]
    rw [← Complex.ofReal_log (Real.exp_pos (2 * u)).le, Real.log_exp]
    congr 1
    push_cast
    ring
  rw [hpow, Complex.real_smul,
    hurwitzEvenFEPair_zero_f_modif_exp_two_of_neg hu]
  have htheta := riemannThetaTail_eq_f_modif (u := -u) (by linarith)
  have htheta' :
      (riemannThetaTail (-u) : ℂ) =
        (Real.exp (-u / 2) / 2 : ℝ) *
          (HurwitzZeta.hurwitzEvenFEPair 0).f_modif (Real.exp (-2 * u)) := by
    convert htheta using 1 <;> ring
  have hexp :
      Complex.exp (2 * u) * Complex.exp ((s - 2) * u) *
          Complex.exp (-u) =
        Complex.exp ((s - 1) * u) := by
    rw [← Complex.exp_add, ← Complex.exp_add]
    congr 1
    ring
  have hexpTheta :
      Complex.exp ((s - 1 / 2) * u) *
          Complex.exp (-u / 2) =
        Complex.exp ((s - 1) * u) := by
    rw [← Complex.exp_add]
    congr 1
    ring
  calc
    (2 * Real.exp (2 * u) : ℝ) *
        (Complex.exp ((s - 2) * u) *
          ((Real.exp (-u) : ℝ) *
            (HurwitzZeta.hurwitzEvenFEPair 0).f_modif (Real.exp (-2 * u)))) =
      2 * (Complex.exp (2 * u) * Complex.exp ((s - 2) * u) *
        Complex.exp (-u)) *
          (HurwitzZeta.hurwitzEvenFEPair 0).f_modif (Real.exp (-2 * u)) := by
        push_cast
        ring
    _ = 2 * Complex.exp ((s - 1) * u) *
          (HurwitzZeta.hurwitzEvenFEPair 0).f_modif (Real.exp (-2 * u)) := by
      rw [hexp]
    _ = 4 * Complex.exp ((s - 1 / 2) * u) *
        ((Real.exp (-u / 2) / 2 : ℝ) *
          (HurwitzZeta.hurwitzEvenFEPair 0).f_modif (Real.exp (-2 * u))) := by
      push_cast
      rw [← hexpTheta]
      ring
    _ = 4 * Complex.exp ((s - 1 / 2) * u) * riemannThetaTail (-u) := by
      rw [← htheta']

theorem intervalIntegral_mellin_f_modif_eq_theta
    (s : ℂ) {R : ℝ} (hR : 0 ≤ R) :
    (∫ x in (1 : ℝ)..Real.exp (2 * R),
        (x : ℂ) ^ (s / 2 - 1) *
          (HurwitzZeta.hurwitzEvenFEPair 0).f_modif x) =
      ∫ u in (0 : ℝ)..R,
        4 * Complex.exp ((s - 1 / 2) * u) * riemannThetaTail u := by
  have hsub := intervalIntegral_mellin_exp_two_substitution
    (q := s / 2 - 1) 0 R
  simp only [mul_zero, Real.exp_zero] at hsub
  rw [← hsub]
  apply intervalIntegral.integral_congr_uIoo
  intro u hu
  change min 0 R < u ∧ u < max 0 R at hu
  rw [min_eq_left hR, max_eq_right hR] at hu
  simpa only using mellin_exp_two_integrand_eq_theta s hu.1

theorem intervalIntegral_mellin_f_modif_eq_cosh_theta
    (s : ℂ) {R : ℝ} (hR : 0 ≤ R) :
    (∫ x in Real.exp (-2 * R)..Real.exp (2 * R),
        (x : ℂ) ^ (s / 2 - 1) *
          (HurwitzZeta.hurwitzEvenFEPair 0).f_modif x) =
      ∫ u in (0 : ℝ)..R,
        8 * Complex.cosh ((s - 1 / 2) * u) * riemannThetaTail u := by
  let M : ℝ → ℂ := fun u ↦
    (2 * Real.exp (2 * u)) •
      (((Real.exp (2 * u) : ℂ) ^ (s / 2 - 1)) *
        (HurwitzZeta.hurwitzEvenFEPair 0).f_modif (Real.exp (2 * u)))
  let L : ℝ → ℂ := fun u ↦
    4 * Complex.exp (-(s - 1 / 2) * u) * riemannThetaTail u
  let U : ℝ → ℂ := fun u ↦
    4 * Complex.exp ((s - 1 / 2) * u) * riemannThetaTail u
  have hsub := intervalIntegral_mellin_exp_two_substitution
    (q := s / 2 - 1) (-R) R
  have hsub' :
      (∫ u in -R..R, M u) =
        ∫ x in Real.exp (-2 * R)..Real.exp (2 * R),
          (x : ℂ) ^ (s / 2 - 1) *
            (HurwitzZeta.hurwitzEvenFEPair 0).f_modif x := by
    dsimp only [M]
    convert hsub using 1 <;> ring
  have hlower_raw :
      (∫ u in -R..0, M u) =
        ∫ u in -R..0,
          4 * Complex.exp ((s - 1 / 2) * u) *
            riemannThetaTail (-u) := by
    apply intervalIntegral.integral_congr_uIoo
    intro u hu
    change min (-R) 0 < u ∧ u < max (-R) 0 at hu
    rw [min_eq_left (by linarith), max_eq_right (by linarith)] at hu
    dsimp only [M]
    exact mellin_exp_two_integrand_eq_theta_neg s hu.2
  have hlower :
      (∫ u in -R..0, M u) = ∫ u in 0..R, L u := by
    rw [hlower_raw]
    have hneg := intervalIntegral.integral_comp_neg
      (f := L) (a := -R) (b := 0)
    dsimp only [L] at hneg ⊢
    rw [neg_zero, neg_neg] at hneg
    convert hneg using 1
    · apply intervalIntegral.integral_congr
      intro u _
      simp only
      have he :
          -(s - 1 / 2) * (-u : ℂ) = (s - 1 / 2) * (u : ℂ) := by
        push_cast
        ring
      push_cast
      rw [he]
  have hupper :
      (∫ u in 0..R, M u) = ∫ u in 0..R, U u := by
    apply intervalIntegral.integral_congr_uIoo
    intro u hu
    change min 0 R < u ∧ u < max 0 R at hu
    rw [min_eq_left hR, max_eq_right hR] at hu
    dsimp only [M, U]
    exact mellin_exp_two_integrand_eq_theta s hu.1
  have hLint : IntervalIntegrable L MeasureTheory.volume 0 R := by
    apply ContinuousOn.intervalIntegrable
    have hthetaCont : ContinuousOn riemannThetaTail (Set.uIcc 0 R) := by
      simpa [Set.uIcc_of_le hR] using continuousOn_riemannThetaTail_Icc R
    exact ((by fun_prop : Continuous fun u : ℝ ↦
      4 * Complex.exp (-(s - 1 / 2) * u)).continuousOn).mul
        (Complex.continuous_ofReal.comp_continuousOn
          hthetaCont)
  have hUint : IntervalIntegrable U MeasureTheory.volume 0 R := by
    apply ContinuousOn.intervalIntegrable
    have hthetaCont : ContinuousOn riemannThetaTail (Set.uIcc 0 R) := by
      simpa [Set.uIcc_of_le hR] using continuousOn_riemannThetaTail_Icc R
    exact ((by fun_prop : Continuous fun u : ℝ ↦
      4 * Complex.exp ((s - 1 / 2) * u)).continuousOn).mul
        (Complex.continuous_ofReal.comp_continuousOn
          hthetaCont)
  have hMlower : IntervalIntegrable M MeasureTheory.volume (-R) 0 := by
    have hN : IntervalIntegrable
        (fun u : ℝ ↦ 4 * Complex.exp ((s - 1 / 2) * u) *
          riemannThetaTail (-u)) MeasureTheory.volume (-R) 0 := by
      rw [IntervalIntegrable.iff_comp_neg]
      simp only [neg_neg, neg_zero]
      apply hLint.symm.congr_uIoo
      intro u hu
      dsimp only [L, Function.comp_apply]
      push_cast
      congr 2 <;> ring
    apply hN.congr_uIoo
    intro u hu
    change min (-R) 0 < u ∧ u < max (-R) 0 at hu
    rw [min_eq_left (by linarith), max_eq_right (by linarith)] at hu
    exact (mellin_exp_two_integrand_eq_theta_neg s hu.2).symm
  have hMupper : IntervalIntegrable M MeasureTheory.volume 0 R := by
    apply hUint.congr_uIoo
    intro u hu
    change min 0 R < u ∧ u < max 0 R at hu
    rw [min_eq_left hR, max_eq_right hR] at hu
    exact (mellin_exp_two_integrand_eq_theta s hu.1).symm
  rw [← hsub', ← intervalIntegral.integral_add_adjacent_intervals
    (a := -R) (b := 0) (c := R) hMlower hMupper, hlower, hupper,
    ← intervalIntegral.integral_add hLint hUint]
  apply intervalIntegral.integral_congr
  intro u _
  dsimp only [L, U]
  rw [Complex.cosh]
  ring

theorem tendsto_intervalIntegral_mellin_truncation
    {f : ℝ → ℂ} {q : ℂ} (hconv : MellinConvergent f q) :
    Filter.Tendsto
      (fun R : ℝ ↦
        ∫ x in Real.exp (-2 * R)..Real.exp (2 * R),
          (x : ℂ) ^ (q - 1) * f x)
      Filter.atTop (𝓝 (mellin f q)) := by
  let I : ℝ → ℂ := fun x ↦ (x : ℂ) ^ (q - 1) * f x
  let φ : ℝ → Set ℝ := fun R ↦
    Set.Ioc (Real.exp (-2 * R)) (Real.exp (2 * R))
  have hcover :
      MeasureTheory.AECover
        (MeasureTheory.volume.restrict (Set.Ioi 0)) Filter.atTop φ := by
    refine ⟨?_, fun R ↦ measurableSet_Ioc⟩
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with x hx
    filter_upwards [Filter.eventually_gt_atTop (|Real.log x| + 1)] with R hR
    dsimp only [φ]
    rw [← Real.exp_log hx]
    constructor
    · apply Real.exp_lt_exp.mpr
      nlinarith [neg_le_abs (Real.log x), abs_nonneg (Real.log x)]
    · exact (Real.exp_lt_exp.mpr (by
        nlinarith [le_abs_self (Real.log x), abs_nonneg (Real.log x)])).le
  have hlim :=
    hcover.integral_tendsto_of_countably_generated (show
      MeasureTheory.Integrable I
        (MeasureTheory.volume.restrict (Set.Ioi 0)) from hconv)
  have hevent :
      (fun R ↦ ∫ x in φ R, I x ∂
          (MeasureTheory.volume.restrict (Set.Ioi 0))) =ᶠ[Filter.atTop]
        fun R ↦ ∫ x in Real.exp (-2 * R)..Real.exp (2 * R), I x := by
    filter_upwards [Filter.eventually_ge_atTop (0 : ℝ)] with R hR
    have hle : Real.exp (-2 * R) ≤ Real.exp (2 * R) :=
      Real.exp_le_exp.mpr (by linarith)
    have hsub : Set.Ioc (Real.exp (-2 * R)) (Real.exp (2 * R)) ⊆
        Set.Ioi 0 := by
      intro x hx
      exact (Real.exp_pos _).trans hx.1
    dsimp only [φ, I]
    rw [Measure.restrict_restrict measurableSet_Ioc,
      Set.inter_eq_left.mpr hsub]
    exact (intervalIntegral.integral_of_le hle).symm
  simpa only [mellin, I, smul_eq_mul] using hlim.congr' hevent

/-- Explicit positive-endpoint decay for every theta summand.  Unlike the
compact `C²` majorant, this estimate decays uniformly as `u → +∞`. -/
theorem norm_riemannThetaTerm_le_decay
    (n : ℕ) {u : ℝ} (hu : 0 ≤ u) :
    ‖riemannThetaTerm n u‖ ≤
      riemannPhiMajorantCoefficient n *
        Real.exp (-(3 / 2 : ℝ) * u) := by
  let m : ℝ := n + 1
  have hm : 1 ≤ m := by
    dsimp only [m]
    exact_mod_cast Nat.succ_pos n
  have hm2 : 1 ≤ m ^ 2 := by nlinarith [sq_nonneg m]
  have hpm : 3 ≤ Real.pi * m ^ 2 := by
    nlinarith [Real.pi_gt_three, Real.pi_pos]
  have hexpLower : 1 + 2 * u ≤ Real.exp (2 * u) := by
    simpa [add_comm] using Real.add_one_le_exp (2 * u)
  have hmul :
      Real.pi * m ^ 2 * (1 + 2 * u) ≤
        Real.pi * m ^ 2 * Real.exp (2 * u) :=
    mul_le_mul_of_nonneg_left hexpLower (by positivity)
  have harg :
      u / 2 - Real.pi * m ^ 2 * Real.exp (2 * u) ≤
        -Real.pi * m ^ 2 - (3 / 2 : ℝ) * u := by
    nlinarith
  have hcoeff : 1 ≤ 4 * Real.pi ^ 2 * m ^ 4 := by
    have hpi : 3 < Real.pi := Real.pi_gt_three
    nlinarith [sq_nonneg (Real.pi * m ^ 2)]
  calc
    ‖riemannThetaTerm n u‖ =
        Real.exp (u / 2 - Real.pi * m ^ 2 * Real.exp (2 * u)) := by
      rw [show u / 2 - Real.pi * m ^ 2 * Real.exp (2 * u) =
          u / 2 + (-Real.pi * m ^ 2 * Real.exp (2 * u)) by ring,
        Real.exp_add]
      dsimp only [riemannThetaTerm, m]
      rw [Real.norm_eq_abs, abs_of_pos (by positivity)]
    _ ≤ Real.exp (-Real.pi * m ^ 2 - (3 / 2 : ℝ) * u) :=
      Real.exp_le_exp.mpr harg
    _ = Real.exp (-Real.pi * m ^ 2) *
        Real.exp (-(3 / 2 : ℝ) * u) := by
      rw [show -Real.pi * m ^ 2 - (3 / 2 : ℝ) * u =
          -Real.pi * m ^ 2 + (-(3 / 2 : ℝ) * u) by ring,
        Real.exp_add]
    _ ≤ (4 * Real.pi ^ 2 * m ^ 4 * Real.exp (-Real.pi * m ^ 2)) *
        Real.exp (-(3 / 2 : ℝ) * u) := by
      apply mul_le_mul_of_nonneg_right
      · have hc := mul_le_mul_of_nonneg_right hcoeff
          (Real.exp_nonneg (-Real.pi * m ^ 2))
        convert hc using 1
        all_goals ring
      · positivity
    _ = riemannPhiMajorantCoefficient n *
        Real.exp (-(3 / 2 : ℝ) * u) := by
      rfl

theorem norm_deriv_riemannThetaTerm_le_decay
    (n : ℕ) {u : ℝ} (hu : 0 ≤ u) :
    ‖deriv (riemannThetaTerm n) u‖ ≤
      riemannPhiMajorantCoefficient n *
        Real.exp (-(3 / 2 : ℝ) * u) := by
  let m : ℝ := n + 1
  let q := Real.pi * m ^ 2
  have hm : 1 ≤ m := by
    dsimp only [m]
    exact_mod_cast Nat.succ_pos n
  have hm2 : 1 ≤ m ^ 2 := by nlinarith [sq_nonneg m]
  have hq : 3 ≤ q := by
    dsimp only [q]
    nlinarith [Real.pi_gt_three, Real.pi_pos]
  have hE : 1 ≤ Real.exp (2 * u) :=
    Real.one_le_exp (by positivity)
  have hexpLower : 1 + 2 * u ≤ Real.exp (2 * u) := by
    simpa [add_comm] using Real.add_one_le_exp (2 * u)
  have hmul :
      q * (1 + 2 * u) ≤ q * Real.exp (2 * u) :=
    mul_le_mul_of_nonneg_left hexpLower (by positivity)
  have harg :
      5 * u / 2 - q * Real.exp (2 * u) ≤
        -q - (3 / 2 : ℝ) * u := by
    nlinarith
  have hcoef :
      |1 / 2 - 2 * q * Real.exp (2 * u)| ≤
        (1 / 2 + 2 * q) * Real.exp (2 * u) := by
    rw [abs_le]
    constructor <;> nlinarith [Real.exp_pos (2 * u)]
  have hcoeffMajor : 1 / 2 + 2 * q ≤ 4 * Real.pi ^ 2 * m ^ 4 := by
    have hq_sq : q ^ 2 = Real.pi ^ 2 * m ^ 4 := by
      dsimp only [q]
      ring
    calc
      1 / 2 + 2 * q ≤ 4 * q ^ 2 := by
        nlinarith [sq_nonneg q]
      _ = 4 * Real.pi ^ 2 * m ^ 4 := by rw [hq_sq]; ring
  rw [deriv_riemannThetaTerm, Real.norm_eq_abs, abs_mul]
  have hthetaPos : 0 < riemannThetaTerm n u := by
    unfold riemannThetaTerm
    positivity
  rw [show |riemannThetaTerm n u| = riemannThetaTerm n u from
    abs_of_pos hthetaPos]
  calc
    |1 / 2 - 2 * Real.pi * (n + 1 : ℝ) ^ 2 * Real.exp (2 * u)| *
        riemannThetaTerm n u =
      |1 / 2 - 2 * q * Real.exp (2 * u)| *
        (Real.exp (u / 2) *
          Real.exp (-q * Real.exp (2 * u))) := by
      dsimp only [riemannThetaTerm, m, q]
      congr 3 <;> ring
    _ ≤ ((1 / 2 + 2 * q) * Real.exp (2 * u)) *
        (Real.exp (u / 2) * Real.exp (-q * Real.exp (2 * u))) := by
      gcongr
    _ = (1 / 2 + 2 * q) *
        Real.exp (5 * u / 2 - q * Real.exp (2 * u)) := by
      rw [show 5 * u / 2 - q * Real.exp (2 * u) =
          2 * u + u / 2 + (-q * Real.exp (2 * u)) by ring,
        Real.exp_add, Real.exp_add]
      ring
    _ ≤ (1 / 2 + 2 * q) *
        Real.exp (-q - (3 / 2 : ℝ) * u) := by
      gcongr
    _ ≤ (4 * Real.pi ^ 2 * m ^ 4) *
        Real.exp (-q - (3 / 2 : ℝ) * u) := by
      gcongr
    _ = riemannPhiMajorantCoefficient n *
        Real.exp (-(3 / 2 : ℝ) * u) := by
      rw [show -q - (3 / 2 : ℝ) * u =
          -q + (-(3 / 2 : ℝ) * u) by ring, Real.exp_add]
      dsimp only [riemannPhiMajorantCoefficient, m, q]
      ring

theorem summable_riemannThetaTerm_of_nonneg {u : ℝ} (hu : 0 ≤ u) :
    Summable (fun n ↦ riemannThetaTerm n u) := by
  apply (summable_riemannThetaC2Majorant u).of_norm_bounded
  intro n
  exact norm_riemannThetaTerm_le_C2Majorant n hu le_rfl

theorem norm_riemannThetaTail_le_decay {u : ℝ} (hu : 0 ≤ u) :
    ‖riemannThetaTail u‖ ≤
      (∑' n : ℕ, riemannPhiMajorantCoefficient n) *
        Real.exp (-(3 / 2 : ℝ) * u) := by
  have hnorm :
      Summable (fun n : ℕ ↦ ‖riemannThetaTerm n u‖) :=
    (summable_riemannThetaTerm_of_nonneg hu).norm
  have hmajor :
      Summable (fun n : ℕ ↦
        riemannPhiMajorantCoefficient n *
          Real.exp (-(3 / 2 : ℝ) * u)) :=
    summable_riemannPhiMajorantCoefficient.mul_right _
  calc
    ‖riemannThetaTail u‖ ≤
        ∑' n : ℕ, ‖riemannThetaTerm n u‖ :=
      norm_tsum_le_tsum_norm hnorm
    _ ≤ ∑' n : ℕ, riemannPhiMajorantCoefficient n *
          Real.exp (-(3 / 2 : ℝ) * u) :=
      hnorm.tsum_le_tsum
        (fun n ↦ norm_riemannThetaTerm_le_decay n hu) hmajor
    _ = (∑' n : ℕ, riemannPhiMajorantCoefficient n) *
          Real.exp (-(3 / 2 : ℝ) * u) :=
      tsum_mul_right

theorem tendsto_riemannThetaTail_atTop_zero :
    Filter.Tendsto riemannThetaTail Filter.atTop (𝓝 0) := by
  let C := ∑' n : ℕ, riemannPhiMajorantCoefficient n
  have hlin :
      Filter.Tendsto (fun u : ℝ ↦ (3 / 2 : ℝ) * u)
        Filter.atTop Filter.atTop :=
    Filter.tendsto_id.const_mul_atTop (by norm_num)
  have hexp :
      Filter.Tendsto (fun u : ℝ ↦ Real.exp (-((3 / 2 : ℝ) * u)))
        Filter.atTop (𝓝 0) :=
    Real.tendsto_exp_neg_atTop_nhds_zero.comp hlin
  have hupper :
      Filter.Tendsto (fun u : ℝ ↦ C * Real.exp (-(3 / 2 : ℝ) * u))
        Filter.atTop (𝓝 0) := by
    have hraw := hexp.const_mul C
    convert hraw using 1
    · funext u
      congr 2
      ring
    · simp
  have hC : 0 ≤ C := by
    dsimp only [C]
    exact tsum_nonneg fun n ↦ by
      unfold riemannPhiMajorantCoefficient
      positivity
  rw [Metric.tendsto_nhds] at hupper ⊢
  intro ε hε
  filter_upwards [hupper ε hε, Filter.eventually_ge_atTop (0 : ℝ)] with u huε hu
  rw [dist_zero_right, Real.norm_eq_abs,
    abs_of_nonneg (mul_nonneg hC (Real.exp_nonneg _))] at huε
  rw [dist_zero_right]
  exact (norm_riemannThetaTail_le_decay hu).trans_lt (by simpa [C] using huε)

theorem summable_deriv_riemannThetaTerm_of_nonneg {u : ℝ} (hu : 0 ≤ u) :
    Summable (fun n ↦ deriv (riemannThetaTerm n) u) := by
  apply (summable_riemannThetaC2Majorant u).of_norm_bounded
  intro n
  exact norm_deriv_riemannThetaTerm_le_C2Majorant n hu le_rfl

theorem summable_deriv_deriv_riemannThetaTerm_of_nonneg {u : ℝ} (hu : 0 ≤ u) :
    Summable (fun n ↦ deriv (deriv (riemannThetaTerm n)) u) := by
  apply (summable_riemannThetaC2Majorant u).of_norm_bounded
  intro n
  exact norm_deriv_deriv_riemannThetaTerm_le_C2Majorant n hu le_rfl

theorem hasDerivAt_riemannThetaTail_zero :
    HasDerivAt riemannThetaTail
      (∑' n : ℕ, deriv (riemannThetaTerm n) 0) 0 := by
  have h :=
    hasDerivAt_tsum_of_isPreconnected
      (g := fun n y ↦ riemannThetaTerm n y)
      (g' := fun n y ↦ deriv (riemannThetaTerm n) y)
      (y₀ := 0) (y := 0)
      summable_riemannThetaLocalDerivMajorant
      isOpen_Ioo isPreconnected_Ioo
      (fun n y _ ↦ by
        simpa only [deriv_riemannThetaTerm] using
          (hasDerivAt_riemannThetaTerm n y))
      (fun n y hy ↦ norm_deriv_riemannThetaTerm_le_localMajorant n hy)
      (by norm_num)
      (summable_riemannThetaTerm_of_nonneg (le_refl 0))
      (by norm_num)
  change HasDerivAt (fun z ↦ ∑' n : ℕ, riemannThetaTerm n z)
    (∑' n : ℕ, deriv (riemannThetaTerm n) 0) 0
  exact h

theorem riemannThetaTailDerivSum_zero :
    (∑' n : ℕ, deriv (riemannThetaTerm n) 0) = -(1 / 4 : ℝ) := by
  let d := ∑' n : ℕ, deriv (riemannThetaTerm n) 0
  have htheta : HasDerivAt riemannThetaTail d 0 :=
    hasDerivAt_riemannThetaTail_zero
  have hthetaNeg : HasDerivAt riemannThetaTail d (-0) := by
    simpa using htheta
  have hleft := hthetaNeg.comp 0 (hasDerivAt_neg 0)
  have hg :
      HasDerivAt
        (fun u : ℝ ↦
          (Real.exp (u / 2) - Real.exp (-u / 2)) / 2)
        (1 / 2 : ℝ) 0 := by
    have hp :
        HasDerivAt (fun u : ℝ ↦ Real.exp (u / 2)) (1 / 2) 0 := by
      convert ((hasDerivAt_id (0 : ℝ)).div_const 2).exp using 1 <;> norm_num
    have hn :
        HasDerivAt (fun u : ℝ ↦ Real.exp (-u / 2)) (-(1 / 2)) 0 := by
      convert ((hasDerivAt_id (0 : ℝ)).neg.div_const 2).exp using 1 <;> norm_num
    have hmul := (hp.sub hn).const_mul (1 / 2 : ℝ)
    norm_num at hmul
    simpa [div_eq_mul_inv, mul_comm] using hmul
  have hright :
      HasDerivAt
        (fun u : ℝ ↦ riemannThetaTail u +
          (Real.exp (u / 2) - Real.exp (-u / 2)) / 2)
        (d + 1 / 2) 0 :=
    htheta.add hg
  have heq :
      riemannThetaTail ∘ Neg.neg =
        fun u : ℝ ↦ riemannThetaTail u +
          (Real.exp (u / 2) - Real.exp (-u / 2)) / 2 := by
    funext u
    exact riemannThetaTail_neg u
  have hd := congrArg (fun F : ℝ → ℝ ↦ deriv F 0) heq
  rw [hleft.deriv, hright.deriv] at hd
  change d = -(1 / 4 : ℝ)
  linarith

theorem deriv_riemannThetaTail_zero :
    deriv riemannThetaTail 0 = -(1 / 4 : ℝ) := by
  rw [hasDerivAt_riemannThetaTail_zero.deriv]
  exact riemannThetaTailDerivSum_zero

theorem hasDerivAt_riemannThetaTail {u : ℝ} (hu : 0 < u) :
    HasDerivAt riemannThetaTail
      (∑' n : ℕ, deriv (riemannThetaTerm n) u) u := by
  let R := u + 1
  have huR : u < R := by dsimp only [R]; linarith
  have h :=
    hasDerivAt_tsum_of_isPreconnected
      (u := riemannThetaC2Majorant R)
      (t := Set.Ioo (0 : ℝ) R)
      (g := fun n ↦ riemannThetaTerm n)
      (g' := fun n y ↦ deriv (riemannThetaTerm n) y)
      (y₀ := u) (y := u)
      (summable_riemannThetaC2Majorant R)
      isOpen_Ioo isPreconnected_Ioo
      (fun n y _ ↦ by
        simpa only [deriv_riemannThetaTerm] using
          (hasDerivAt_riemannThetaTerm n y))
      (fun n y hy ↦ norm_deriv_riemannThetaTerm_le_C2Majorant n hy.1.le hy.2.le)
      ⟨hu, huR⟩
      (summable_riemannThetaTerm_of_nonneg hu.le)
      ⟨hu, huR⟩
  change HasDerivAt (fun z ↦ ∑' n : ℕ, riemannThetaTerm n z)
    (∑' n : ℕ, deriv (riemannThetaTerm n) u) u
  exact h

theorem deriv_riemannThetaTail {u : ℝ} (hu : 0 < u) :
    deriv riemannThetaTail u =
      ∑' n : ℕ, deriv (riemannThetaTerm n) u :=
  (hasDerivAt_riemannThetaTail hu).deriv

def riemannThetaTailDeriv (u : ℝ) : ℝ :=
  ∑' n : ℕ, deriv (riemannThetaTerm n) u

theorem riemannThetaTailDeriv_zero :
    riemannThetaTailDeriv 0 = -(1 / 4 : ℝ) :=
  riemannThetaTailDerivSum_zero

theorem deriv_riemannThetaTail_eq_tailDeriv_of_nonneg
    {u : ℝ} (hu : 0 ≤ u) :
    deriv riemannThetaTail u = riemannThetaTailDeriv u := by
  rcases hu.eq_or_lt with rfl | hu
  · rw [deriv_riemannThetaTail_zero, riemannThetaTailDeriv_zero]
  · exact deriv_riemannThetaTail hu

theorem continuousOn_riemannThetaTailDeriv_Icc (R : ℝ) :
    ContinuousOn riemannThetaTailDeriv (Set.Icc 0 R) := by
  apply (tendstoUniformlyOn_deriv_riemannThetaTerm_Icc R).continuousOn
  exact Filter.Frequently.of_forall fun N ↦ by
    apply Continuous.continuousOn
    apply continuous_finsetSum
    intro n _
    rw [show deriv (riemannThetaTerm n) = fun u ↦
        (1 / 2 - 2 * Real.pi * (n + 1 : ℝ) ^ 2 * Real.exp (2 * u)) *
          riemannThetaTerm n u by
      funext u
      exact deriv_riemannThetaTerm n u]
    apply Continuous.mul (by fun_prop)
    exact continuous_iff_continuousAt.mpr fun u ↦
      (hasDerivAt_riemannThetaTerm n u).continuousAt

theorem continuousOn_deriv_riemannThetaTail_Icc (R : ℝ) :
    ContinuousOn (deriv riemannThetaTail) (Set.Icc 0 R) := by
  apply (continuousOn_riemannThetaTailDeriv_Icc R).congr
  intro u hu
  exact deriv_riemannThetaTail_eq_tailDeriv_of_nonneg hu.1

theorem hasDerivAt_riemannThetaTailDeriv {u : ℝ} (hu : 0 < u) :
    HasDerivAt riemannThetaTailDeriv
      (∑' n : ℕ, deriv (deriv (riemannThetaTerm n)) u) u := by
  let R := u + 1
  have huR : u < R := by dsimp only [R]; linarith
  have h :=
    hasDerivAt_tsum_of_isPreconnected
      (u := riemannThetaC2Majorant R)
      (t := Set.Ioo (0 : ℝ) R)
      (g := fun n ↦ deriv (riemannThetaTerm n))
      (g' := fun n y ↦ deriv (deriv (riemannThetaTerm n)) y)
      (y₀ := u) (y := u)
      (summable_riemannThetaC2Majorant R)
      isOpen_Ioo isPreconnected_Ioo
      (fun n y _ ↦
        (hasDerivAt_deriv_riemannThetaTerm_raw n y).differentiableAt.hasDerivAt)
      (fun n y hy ↦
        norm_deriv_deriv_riemannThetaTerm_le_C2Majorant n hy.1.le hy.2.le)
      ⟨hu, huR⟩
      (summable_deriv_riemannThetaTerm_of_nonneg hu.le)
      ⟨hu, huR⟩
  change HasDerivAt
    (fun z ↦ ∑' n : ℕ, deriv (riemannThetaTerm n) z)
    (∑' n : ℕ, deriv (deriv (riemannThetaTerm n)) u) u
  exact h

theorem deriv_deriv_riemannThetaTail {u : ℝ} (hu : 0 < u) :
    deriv (deriv riemannThetaTail) u =
      ∑' n : ℕ, deriv (deriv (riemannThetaTerm n)) u := by
  have heq :
      deriv riemannThetaTail =ᶠ[nhds u] riemannThetaTailDeriv := by
    filter_upwards [Ioi_mem_nhds hu] with y hy
    exact deriv_riemannThetaTail hy
  rw [heq.deriv_eq]
  exact (hasDerivAt_riemannThetaTailDeriv hu).deriv

theorem hasDerivAt_deriv_riemannThetaTail {u : ℝ} (hu : 0 < u) :
    HasDerivAt (deriv riemannThetaTail)
      (deriv (deriv riemannThetaTail) u) u := by
  have heq :
      deriv riemannThetaTail =ᶠ[nhds u] riemannThetaTailDeriv := by
    filter_upwards [Ioi_mem_nhds hu] with y hy
    exact deriv_riemannThetaTail hy
  rw [heq.deriv_eq]
  exact (hasDerivAt_riemannThetaTailDeriv hu).differentiableAt.hasDerivAt
    |>.congr_of_eventuallyEq heq

theorem riemannPhiTerm_pos (n : ℕ) (u : ℝ) :
    0 < riemannPhiTerm n u := by
  let m : ℝ := n + 1
  let v := |u|
  have hm : 1 ≤ m := by
    dsimp only [m]
    exact_mod_cast Nat.succ_pos n
  have hv : 0 ≤ v := abs_nonneg u
  have hE : 1 ≤ Real.exp (2 * v) :=
    Real.one_le_exp (mul_nonneg (by norm_num) hv)
  have hfactor : 0 < 2 * Real.pi * m ^ 2 * Real.exp (2 * v) - 3 := by
    have hpi : 3 < 2 * Real.pi := by nlinarith [Real.pi_gt_three]
    have hm2 : 1 ≤ m ^ 2 := by nlinarith [sq_nonneg m]
    have hprod :
        2 * Real.pi ≤ 2 * Real.pi * m ^ 2 * Real.exp (2 * v) := by
      calc
        2 * Real.pi = 2 * Real.pi * 1 * 1 := by ring
        _ ≤ 2 * Real.pi * m ^ 2 * Real.exp (2 * v) := by gcongr
    linarith
  dsimp only [riemannPhiTerm, m, v]
  positivity

/-- Quantitative source-kernel decay, uniform in the summation index. -/
theorem riemannPhiTerm_le_majorant (n : ℕ) (u : ℝ) :
    riemannPhiTerm n u ≤
      riemannPhiMajorantCoefficient n * Real.exp (-(3 / 2 : ℝ) * |u|) := by
  let m : ℝ := n + 1
  let v := |u|
  have hm : 1 ≤ m := by
    dsimp only [m]
    exact_mod_cast Nat.succ_pos n
  have hv : 0 ≤ v := abs_nonneg u
  have hm2 : 1 ≤ m ^ 2 := by nlinarith [sq_nonneg m]
  have hpm : 3 ≤ Real.pi * m ^ 2 := by
    nlinarith [Real.pi_gt_three, Real.pi_pos]
  have hexpLower : 1 + 2 * v ≤ Real.exp (2 * v) := by
    simpa [add_comm] using Real.add_one_le_exp (2 * v)
  have harg :
      9 * v / 2 - Real.pi * m ^ 2 * Real.exp (2 * v) ≤
        -Real.pi * m ^ 2 - (3 / 2 : ℝ) * v := by
    have hsix : 6 * v ≤ 2 * Real.pi * m ^ 2 * v := by
      nlinarith
    nlinarith
  have hfactor :
      riemannPhiTerm n u ≤
        4 * Real.pi ^ 2 * m ^ 4 * Real.exp (9 * v / 2) *
          Real.exp (-Real.pi * m ^ 2 * Real.exp (2 * v)) := by
    have hbracket :
        2 * Real.pi * m ^ 2 * Real.exp (2 * v) - 3 ≤
          2 * Real.pi * m ^ 2 * Real.exp (2 * v) := by
      linarith
    calc
      riemannPhiTerm n u =
          2 * Real.pi * m ^ 2 * Real.exp (5 * v / 2) *
            (2 * Real.pi * m ^ 2 * Real.exp (2 * v) - 3) *
              Real.exp (-Real.pi * m ^ 2 * Real.exp (2 * v)) := by
        rfl
      _ ≤ 2 * Real.pi * m ^ 2 * Real.exp (5 * v / 2) *
            (2 * Real.pi * m ^ 2 * Real.exp (2 * v)) *
              Real.exp (-Real.pi * m ^ 2 * Real.exp (2 * v)) := by
        gcongr
      _ = 4 * Real.pi ^ 2 * m ^ 4 * Real.exp (9 * v / 2) *
            Real.exp (-Real.pi * m ^ 2 * Real.exp (2 * v)) := by
        rw [show Real.exp (9 * v / 2) =
            Real.exp (5 * v / 2) * Real.exp (2 * v) by
          rw [← Real.exp_add]
          congr 1
          ring]
        ring
  calc
    riemannPhiTerm n u ≤
        4 * Real.pi ^ 2 * m ^ 4 * Real.exp (9 * v / 2) *
          Real.exp (-Real.pi * m ^ 2 * Real.exp (2 * v)) := hfactor
    _ = 4 * Real.pi ^ 2 * m ^ 4 *
          Real.exp (9 * v / 2 -
            Real.pi * m ^ 2 * Real.exp (2 * v)) := by
      rw [show 9 * v / 2 - Real.pi * m ^ 2 * Real.exp (2 * v) =
          9 * v / 2 + (-Real.pi * m ^ 2 * Real.exp (2 * v)) by ring,
        Real.exp_add]
      ring
    _ ≤ 4 * Real.pi ^ 2 * m ^ 4 *
          Real.exp (-Real.pi * m ^ 2 - (3 / 2 : ℝ) * v) := by
      gcongr
    _ = riemannPhiMajorantCoefficient n *
          Real.exp (-(3 / 2 : ℝ) * |u|) := by
      rw [show -Real.pi * m ^ 2 - (3 / 2 : ℝ) * v =
          -Real.pi * m ^ 2 + (-(3 / 2 : ℝ) * v) by ring,
        Real.exp_add]
      dsimp only [riemannPhiMajorantCoefficient, m, v]
      ring

theorem norm_deriv_deriv_riemannThetaTerm_le_decay
    (n : ℕ) {u : ℝ} (hu : 0 ≤ u) :
    ‖deriv (deriv (riemannThetaTerm n)) u‖ ≤
      2 * riemannPhiMajorantCoefficient n *
        Real.exp (-(3 / 2 : ℝ) * u) := by
  have hidentity :=
    riemannPhiTerm_eq_theta_shifted_second_deriv n hu
  have hdd :
      deriv (deriv (riemannThetaTerm n)) u =
        riemannPhiTerm n u + (1 / 4 : ℝ) * riemannThetaTerm n u := by
    linarith
  rw [hdd]
  calc
    ‖riemannPhiTerm n u + (1 / 4 : ℝ) * riemannThetaTerm n u‖ ≤
        ‖riemannPhiTerm n u‖ +
          ‖(1 / 4 : ℝ) * riemannThetaTerm n u‖ :=
      norm_add_le _ _
    _ = riemannPhiTerm n u +
        (1 / 4 : ℝ) * ‖riemannThetaTerm n u‖ := by
      rw [Real.norm_eq_abs, abs_of_pos (riemannPhiTerm_pos n u),
        norm_mul, Real.norm_eq_abs, abs_of_nonneg (by norm_num)]
    _ ≤ riemannPhiMajorantCoefficient n *
          Real.exp (-(3 / 2 : ℝ) * u) +
        (1 / 4 : ℝ) * (riemannPhiMajorantCoefficient n *
          Real.exp (-(3 / 2 : ℝ) * u)) := by
      gcongr
      · simpa [abs_of_nonneg hu] using riemannPhiTerm_le_majorant n u
      · exact norm_riemannThetaTerm_le_decay n hu
    _ ≤ 2 * riemannPhiMajorantCoefficient n *
        Real.exp (-(3 / 2 : ℝ) * u) := by
      have hc : 0 ≤ riemannPhiMajorantCoefficient n := by
        unfold riemannPhiMajorantCoefficient
        positivity
      have he : 0 ≤ Real.exp (-(3 / 2 : ℝ) * u) := Real.exp_nonneg _
      nlinarith

def riemannThetaTailSecondDerivSum (u : ℝ) : ℝ :=
  ∑' n : ℕ, deriv (deriv (riemannThetaTerm n)) u

theorem norm_riemannThetaTailDeriv_le_decay {u : ℝ} (hu : 0 ≤ u) :
    ‖riemannThetaTailDeriv u‖ ≤
      (∑' n : ℕ, riemannPhiMajorantCoefficient n) *
        Real.exp (-(3 / 2 : ℝ) * u) := by
  have hnorm :
      Summable (fun n : ℕ ↦ ‖deriv (riemannThetaTerm n) u‖) :=
    (summable_deriv_riemannThetaTerm_of_nonneg hu).norm
  have hmajor :
      Summable (fun n : ℕ ↦
        riemannPhiMajorantCoefficient n *
          Real.exp (-(3 / 2 : ℝ) * u)) :=
    summable_riemannPhiMajorantCoefficient.mul_right _
  calc
    ‖riemannThetaTailDeriv u‖ ≤
        ∑' n : ℕ, ‖deriv (riemannThetaTerm n) u‖ :=
      norm_tsum_le_tsum_norm hnorm
    _ ≤ ∑' n : ℕ, riemannPhiMajorantCoefficient n *
          Real.exp (-(3 / 2 : ℝ) * u) :=
      hnorm.tsum_le_tsum
        (fun n ↦ norm_deriv_riemannThetaTerm_le_decay n hu) hmajor
    _ = (∑' n : ℕ, riemannPhiMajorantCoefficient n) *
          Real.exp (-(3 / 2 : ℝ) * u) :=
      tsum_mul_right

theorem norm_riemannThetaTailSecondDerivSum_le_decay
    {u : ℝ} (hu : 0 ≤ u) :
    ‖riemannThetaTailSecondDerivSum u‖ ≤
      2 * (∑' n : ℕ, riemannPhiMajorantCoefficient n) *
        Real.exp (-(3 / 2 : ℝ) * u) := by
  have hnorm :
      Summable (fun n : ℕ ↦
        ‖deriv (deriv (riemannThetaTerm n)) u‖) :=
    (summable_deriv_deriv_riemannThetaTerm_of_nonneg hu).norm
  have hmajor :
      Summable (fun n : ℕ ↦
        (2 * riemannPhiMajorantCoefficient n) *
          Real.exp (-(3 / 2 : ℝ) * u)) :=
    (summable_riemannPhiMajorantCoefficient.mul_left 2).mul_right _
  calc
    ‖riemannThetaTailSecondDerivSum u‖ ≤
        ∑' n : ℕ, ‖deriv (deriv (riemannThetaTerm n)) u‖ :=
      norm_tsum_le_tsum_norm hnorm
    _ ≤ ∑' n : ℕ, (2 * riemannPhiMajorantCoefficient n) *
          Real.exp (-(3 / 2 : ℝ) * u) :=
      hnorm.tsum_le_tsum
        (fun n ↦ norm_deriv_deriv_riemannThetaTerm_le_decay n hu) hmajor
    _ = 2 * (∑' n : ℕ, riemannPhiMajorantCoefficient n) *
          Real.exp (-(3 / 2 : ℝ) * u) := by
      rw [tsum_mul_right, ← tsum_mul_left]

theorem tendsto_zero_of_norm_le_exp_decay
    {f : ℝ → ℝ} {C : ℝ} (hC : 0 ≤ C)
    (hbound : ∀ u, 0 ≤ u →
      ‖f u‖ ≤ C * Real.exp (-(3 / 2 : ℝ) * u)) :
    Filter.Tendsto f Filter.atTop (𝓝 0) := by
  have hlin :
      Filter.Tendsto (fun u : ℝ ↦ (3 / 2 : ℝ) * u)
        Filter.atTop Filter.atTop :=
    Filter.tendsto_id.const_mul_atTop (by norm_num)
  have hexp :
      Filter.Tendsto (fun u : ℝ ↦ Real.exp (-((3 / 2 : ℝ) * u)))
        Filter.atTop (𝓝 0) :=
    Real.tendsto_exp_neg_atTop_nhds_zero.comp hlin
  have hupper :
      Filter.Tendsto (fun u : ℝ ↦ C * Real.exp (-(3 / 2 : ℝ) * u))
        Filter.atTop (𝓝 0) := by
    have hraw := hexp.const_mul C
    convert hraw using 1
    · funext u
      congr 2
      ring
    · simp
  rw [Metric.tendsto_nhds] at hupper ⊢
  intro ε hε
  filter_upwards [hupper ε hε, Filter.eventually_ge_atTop (0 : ℝ)] with u huε hu
  rw [dist_zero_right, Real.norm_eq_abs,
    abs_of_nonneg (mul_nonneg hC (Real.exp_nonneg _))] at huε
  rw [dist_zero_right]
  exact (hbound u hu).trans_lt huε

theorem tendsto_exp_mul_zero_of_norm_le_exp_decay
    {f : ℝ → ℝ} {C r : ℝ} (hC : 0 ≤ C) (hr : r < 3 / 2)
    (hbound : ∀ u, 0 ≤ u →
      ‖f u‖ ≤ C * Real.exp (-(3 / 2 : ℝ) * u)) :
    Filter.Tendsto (fun u ↦ Real.exp (r * u) * f u)
      Filter.atTop (𝓝 0) := by
  let δ := 3 / 2 - r
  have hδ : 0 < δ := by
    dsimp only [δ]
    linarith
  have hlin :
      Filter.Tendsto (fun u : ℝ ↦ δ * u)
        Filter.atTop Filter.atTop :=
    Filter.tendsto_id.const_mul_atTop hδ
  have hexp :
      Filter.Tendsto (fun u : ℝ ↦ Real.exp (-(δ * u)))
        Filter.atTop (𝓝 0) :=
    Real.tendsto_exp_neg_atTop_nhds_zero.comp hlin
  have hupper :
      Filter.Tendsto (fun u : ℝ ↦ C * Real.exp (-(δ * u)))
        Filter.atTop (𝓝 0) := by
    simpa using hexp.const_mul C
  rw [Metric.tendsto_nhds] at hupper ⊢
  intro ε hε
  filter_upwards [hupper ε hε, Filter.eventually_ge_atTop (0 : ℝ)] with u huε hu
  rw [dist_zero_right, Real.norm_eq_abs,
    abs_of_nonneg (mul_nonneg hC (Real.exp_nonneg _))] at huε
  rw [dist_zero_right, norm_mul, Real.norm_eq_abs,
    abs_of_pos (Real.exp_pos _)]
  calc
    Real.exp (r * u) * ‖f u‖ ≤
        Real.exp (r * u) *
          (C * Real.exp (-(3 / 2 : ℝ) * u)) := by
      gcongr
      exact hbound u hu
    _ = C * (Real.exp (r * u) *
        Real.exp (-(3 / 2 : ℝ) * u)) := by ring
    _ = C * Real.exp (-(δ * u)) := by
      rw [← Real.exp_add]
      dsimp only [δ]
      congr 2
      ring
    _ < ε := huε

theorem integrableOn_cosh_mul_of_norm_le_exp_decay
    {f : ℝ → ℝ} {C r : ℝ} (hr : |r| < 3 / 2)
    (hcont : ContinuousOn f (Set.Ioi 0))
    (hbound : ∀ u, 0 < u →
      ‖f u‖ ≤ C * Real.exp (-(3 / 2 : ℝ) * u)) :
    MeasureTheory.IntegrableOn
      (fun u ↦ Real.cosh (r * u) * f u) (Set.Ioi 0) := by
  let g : ℝ → ℝ := fun u ↦
    (C / 2) * (Real.exp ((r - 3 / 2) * u) +
      Real.exp ((-r - 3 / 2) * u))
  have hrp : r - 3 / 2 < 0 := by linarith [(abs_lt.mp hr).2]
  have hrn : -r - 3 / 2 < 0 := by linarith [(abs_lt.mp hr).1]
  have hg : MeasureTheory.IntegrableOn g (Set.Ioi 0) := by
    exact ((integrableOn_exp_mul_Ioi hrp 0).add
      (integrableOn_exp_mul_Ioi hrn 0)).const_mul (C / 2)
  apply hg.mono'
  · exact
      (((by fun_prop : Continuous fun u : ℝ ↦ Real.cosh (r * u)).continuousOn.mul
        hcont).aestronglyMeasurable measurableSet_Ioi)
  · filter_upwards [ae_restrict_mem measurableSet_Ioi] with u hu
    rw [norm_mul, Real.norm_eq_abs, abs_of_pos (Real.cosh_pos _)]
    calc
      Real.cosh (r * u) * ‖f u‖ ≤
          Real.cosh (r * u) *
            (C * Real.exp (-(3 / 2 : ℝ) * u)) := by
        gcongr
        exact hbound u hu
      _ = g u := by
        rw [Real.cosh_eq]
        dsimp only [g]
        rw [show Real.exp ((r - 3 / 2) * u) =
            Real.exp (r * u) * Real.exp (-(3 / 2 : ℝ) * u) by
          rw [← Real.exp_add]
          congr 1
          ring]
        rw [show Real.exp ((-r - 3 / 2) * u) =
            Real.exp (-(r * u)) * Real.exp (-(3 / 2 : ℝ) * u) by
          rw [← Real.exp_add]
          congr 1
          ring]
        ring

theorem tendsto_riemannThetaTailDeriv_atTop_zero :
    Filter.Tendsto riemannThetaTailDeriv Filter.atTop (𝓝 0) := by
  apply tendsto_zero_of_norm_le_exp_decay
    (f := riemannThetaTailDeriv)
    (C := ∑' n : ℕ, riemannPhiMajorantCoefficient n)
  · exact tsum_nonneg fun n ↦ by
      unfold riemannPhiMajorantCoefficient
      positivity
  · exact fun u hu ↦ norm_riemannThetaTailDeriv_le_decay hu

theorem tendsto_riemannThetaTailSecondDerivSum_atTop_zero :
    Filter.Tendsto riemannThetaTailSecondDerivSum Filter.atTop (𝓝 0) := by
  apply tendsto_zero_of_norm_le_exp_decay
    (f := riemannThetaTailSecondDerivSum)
    (C := 2 * (∑' n : ℕ, riemannPhiMajorantCoefficient n))
  · exact mul_nonneg (by norm_num) (tsum_nonneg fun n ↦ by
      unfold riemannPhiMajorantCoefficient
      positivity)
  · exact fun u hu ↦ norm_riemannThetaTailSecondDerivSum_le_decay hu

theorem tendsto_deriv_riemannThetaTail_atTop_zero :
    Filter.Tendsto (deriv riemannThetaTail) Filter.atTop (𝓝 0) := by
  apply tendsto_riemannThetaTailDeriv_atTop_zero.congr'
  filter_upwards [Filter.eventually_gt_atTop (0 : ℝ)] with u hu
  exact (deriv_riemannThetaTail hu).symm

theorem tendsto_deriv_deriv_riemannThetaTail_atTop_zero :
    Filter.Tendsto (deriv (deriv riemannThetaTail)) Filter.atTop (𝓝 0) := by
  apply tendsto_riemannThetaTailSecondDerivSum_atTop_zero.congr'
  filter_upwards [Filter.eventually_gt_atTop (0 : ℝ)] with u hu
  exact (deriv_deriv_riemannThetaTail hu).symm

theorem summable_riemannPhiTerm (u : ℝ) :
    Summable (fun n ↦ riemannPhiTerm n u) := by
  let C : ℝ := 4 * Real.pi ^ 2 * Real.exp (9 * |u| / 2)
  have hbase :
      Summable (fun n : ℕ ↦
        (n : ℝ) ^ 4 * Real.exp (-Real.pi * n)) :=
    Real.summable_pow_mul_exp_neg_nat_mul 4 Real.pi_pos
  have hmajor :
      Summable (fun n : ℕ ↦
        C * (n + 1 : ℝ) ^ 4 *
          Real.exp (-Real.pi * (n + 1 : ℝ))) := by
    simpa [Nat.cast_add, Nat.cast_one, mul_assoc] using
      ((summable_nat_add_iff 1).2 hbase).mul_left C
  apply hmajor.of_norm_bounded
  intro n
  let m : ℝ := n + 1
  let v := |u|
  have hm : 1 ≤ m := by
    dsimp only [m]
    exact_mod_cast Nat.succ_pos n
  have hv : 0 ≤ v := abs_nonneg u
  have hE : 1 ≤ Real.exp (2 * v) :=
    Real.one_le_exp (mul_nonneg (by norm_num) hv)
  have hm2E : m ≤ m ^ 2 * Real.exp (2 * v) := by
    nlinarith [sq_nonneg (m - 1), Real.exp_pos (2 * v)]
  have hexp :
      Real.exp (-Real.pi * m ^ 2 * Real.exp (2 * v)) ≤
        Real.exp (-Real.pi * m) := by
    apply Real.exp_le_exp.mpr
    nlinarith [Real.pi_pos]
  have hterm := (riemannPhiTerm_pos n u).le
  have hbracket :
      2 * Real.pi * m ^ 2 * Real.exp (2 * v) - 3 ≤
        2 * Real.pi * m ^ 2 * Real.exp (2 * v) := by
    linarith
  rw [Real.norm_eq_abs, abs_of_nonneg hterm]
  calc
    riemannPhiTerm n u ≤
        2 * Real.pi * m ^ 2 * Real.exp (5 * v / 2) *
          (2 * Real.pi * m ^ 2 * Real.exp (2 * v)) *
            Real.exp (-Real.pi * m ^ 2 * Real.exp (2 * v)) := by
      dsimp only [riemannPhiTerm, m, v]
      gcongr
    _ = C * m ^ 4 *
          Real.exp (-Real.pi * m ^ 2 * Real.exp (2 * v)) := by
      dsimp only [C]
      rw [show Real.exp (9 * |u| / 2) =
          Real.exp (5 * v / 2) * Real.exp (2 * v) by
        rw [← Real.exp_add]
        congr 1
        dsimp only [v]
        ring]
      ring
    _ ≤ C * m ^ 4 * Real.exp (-Real.pi * m) := by
      gcongr

/-- Riemann's explicit Fourier kernel. -/
def riemannPhi (u : ℝ) : ℝ :=
  ∑' n : ℕ, riemannPhiTerm n u

theorem riemannPhi_eq_thetaSeries_shifted_second_deriv
    {u : ℝ} (hu : 0 ≤ u) :
    riemannPhi u =
      (∑' n : ℕ, deriv (deriv (riemannThetaTerm n)) u) -
        (1 / 4 : ℝ) * riemannThetaTail u := by
  rw [riemannPhi]
  unfold riemannThetaTail
  calc
    (∑' n : ℕ, riemannPhiTerm n u) =
        ∑' n : ℕ, (deriv (deriv (riemannThetaTerm n)) u -
          (1 / 4 : ℝ) * riemannThetaTerm n u) := by
      apply tsum_congr
      intro n
      exact riemannPhiTerm_eq_theta_shifted_second_deriv n hu
    _ = (∑' n : ℕ, deriv (deriv (riemannThetaTerm n)) u) -
        ∑' n : ℕ, (1 / 4 : ℝ) * riemannThetaTerm n u := by
      exact Summable.tsum_sub
        (summable_deriv_deriv_riemannThetaTerm_of_nonneg hu)
        ((summable_riemannThetaTerm_of_nonneg hu).mul_left (1 / 4 : ℝ))
    _ = _ := by rw [tsum_mul_left]

theorem riemannPhi_eq_thetaTail_shifted_second_deriv
    {u : ℝ} (hu : 0 < u) :
    riemannPhi u =
      deriv (deriv riemannThetaTail) u - (1 / 4 : ℝ) * riemannThetaTail u := by
  rw [riemannPhi_eq_thetaSeries_shifted_second_deriv hu.le,
    deriv_deriv_riemannThetaTail hu]

theorem riemannPhi_even (u : ℝ) :
    riemannPhi (-u) = riemannPhi u := by
  apply tsum_congr
  intro n
  simp only [riemannPhiTerm, abs_neg]

theorem riemannPhi_pos (u : ℝ) :
    0 < riemannPhi u := by
  exact (summable_riemannPhiTerm u).tsum_pos
    (fun n ↦ (riemannPhiTerm_pos n u).le) 0
    (riemannPhiTerm_pos 0 u)

theorem continuous_riemannPhiTerm (n : ℕ) :
    Continuous (riemannPhiTerm n) := by
  unfold riemannPhiTerm
  fun_prop

/-- The explicit kernel is continuous, by a global Weierstrass majorant. -/
theorem continuous_riemannPhi : Continuous riemannPhi := by
  unfold riemannPhi
  apply continuous_tsum continuous_riemannPhiTerm
      summable_riemannPhiMajorantCoefficient
  intro n u
  rw [Real.norm_eq_abs, abs_of_pos (riemannPhiTerm_pos n u)]
  calc
    riemannPhiTerm n u ≤
        riemannPhiMajorantCoefficient n *
          Real.exp (-(3 / 2 : ℝ) * |u|) :=
      riemannPhiTerm_le_majorant n u
    _ ≤ riemannPhiMajorantCoefficient n * 1 := by
      apply mul_le_mul_of_nonneg_left _
        (riemannPhiMajorantCoefficient_pos n).le
      apply Real.exp_le_one_iff.mpr
      exact mul_nonpos_of_nonpos_of_nonneg (by norm_num) (abs_nonneg u)
    _ = riemannPhiMajorantCoefficient n := mul_one _

def thetaCoshBoundary (r u : ℝ) : ℝ :=
  Real.cosh (r * u) * deriv riemannThetaTail u -
    r * Real.sinh (r * u) * riemannThetaTail u

theorem thetaCoshBoundary_zero (r : ℝ) :
    thetaCoshBoundary r 0 = -(1 / 4 : ℝ) := by
  rw [thetaCoshBoundary, deriv_riemannThetaTail_zero]
  simp

theorem hasDerivAt_thetaCoshBoundary
    (r : ℝ) {u : ℝ} (hu : 0 < u) :
    HasDerivAt (thetaCoshBoundary r)
      (Real.cosh (r * u) *
        (riemannPhi u + (1 / 4 - r ^ 2) * riemannThetaTail u)) u := by
  have ht := hasDerivAt_riemannThetaTail hu
  have hdt := hasDerivAt_deriv_riemannThetaTail hu
  have hc :
      HasDerivAt (fun x : ℝ ↦ Real.cosh (r * x))
        (r * Real.sinh (r * u)) u := by
    simpa [id_eq, mul_comm] using ((hasDerivAt_id u).const_mul r).cosh
  have hs :
      HasDerivAt (fun x : ℝ ↦ Real.sinh (r * x))
        (r * Real.cosh (r * u)) u := by
    simpa [id_eq, mul_comm] using ((hasDerivAt_id u).const_mul r).sinh
  have hraw := (hc.mul hdt).sub ((hs.const_mul r).mul ht)
  rw [show deriv (deriv riemannThetaTail) u =
      riemannPhi u + (1 / 4 : ℝ) * riemannThetaTail u by
    rw [riemannPhi_eq_thetaTail_shifted_second_deriv hu]
    ring] at hraw
  rw [← deriv_riemannThetaTail hu] at hraw
  unfold thetaCoshBoundary
  change HasDerivAt
    (fun x : ℝ ↦ Real.cosh (r * x) * deriv riemannThetaTail x -
      r * Real.sinh (r * x) * riemannThetaTail x) _ u at hraw
  convert hraw using 1
  ring

theorem continuousOn_thetaCoshBoundary_Icc (r R : ℝ) :
    ContinuousOn (thetaCoshBoundary r) (Set.Icc 0 R) := by
  unfold thetaCoshBoundary
  exact
    ((by fun_prop : Continuous fun u : ℝ ↦ Real.cosh (r * u)).continuousOn.mul
      (continuousOn_deriv_riemannThetaTail_Icc R)).sub
    (((by fun_prop : Continuous fun u : ℝ ↦ r * Real.sinh (r * u)).continuousOn).mul
      (continuousOn_riemannThetaTail_Icc R))

theorem intervalIntegral_cosh_mul_riemannPhi_double_ibp
    (r : ℝ) {R : ℝ} (hR : 0 ≤ R) :
    (∫ u in (0 : ℝ)..R, Real.cosh (r * u) * riemannPhi u) =
      thetaCoshBoundary r R + 1 / 4 +
        (r ^ 2 - 1 / 4) *
          ∫ u in (0 : ℝ)..R,
            Real.cosh (r * u) * riemannThetaTail u := by
  let A : ℝ → ℝ := fun u ↦ Real.cosh (r * u) * riemannPhi u
  let T : ℝ → ℝ := fun u ↦ Real.cosh (r * u) * riemannThetaTail u
  let F : ℝ → ℝ := fun u ↦
    Real.cosh (r * u) *
      (riemannPhi u + (1 / 4 - r ^ 2) * riemannThetaTail u)
  have hAcont : ContinuousOn A (Set.Icc 0 R) :=
    ((by fun_prop : Continuous fun u : ℝ ↦ Real.cosh (r * u)).continuousOn).mul
      continuous_riemannPhi.continuousOn
  have hTcont : ContinuousOn T (Set.Icc 0 R) :=
    ((by fun_prop : Continuous fun u : ℝ ↦ Real.cosh (r * u)).continuousOn).mul
      (continuousOn_riemannThetaTail_Icc R)
  have hFcont : ContinuousOn F (Set.Icc 0 R) := by
    have hsum := hAcont.add (hTcont.const_mul (1 / 4 - r ^ 2))
    apply hsum.congr
    intro u hu
    simp only [Pi.add_apply]
    dsimp only [F, A, T]
    ring
  have hAcont' : ContinuousOn A (Set.uIcc 0 R) := by
    simpa [Set.uIcc_of_le hR] using hAcont
  have hTcont' : ContinuousOn T (Set.uIcc 0 R) := by
    simpa [Set.uIcc_of_le hR] using hTcont
  have hFcont' : ContinuousOn F (Set.uIcc 0 R) := by
    simpa [Set.uIcc_of_le hR] using hFcont
  have hAint : IntervalIntegrable A MeasureTheory.volume 0 R :=
    hAcont'.intervalIntegrable
  have hTint : IntervalIntegrable T MeasureTheory.volume 0 R :=
    hTcont'.intervalIntegrable
  have hFint : IntervalIntegrable F MeasureTheory.volume 0 R :=
    hFcont'.intervalIntegrable
  have hfund :
      (∫ u in (0 : ℝ)..R, F u) =
        thetaCoshBoundary r R - thetaCoshBoundary r 0 :=
    intervalIntegral.integral_eq_sub_of_hasDerivAt_of_le hR
      (continuousOn_thetaCoshBoundary_Icc r R)
      (fun u hu ↦ hasDerivAt_thetaCoshBoundary r hu.1)
      hFint
  have hsplit :
      (∫ u in (0 : ℝ)..R, F u) =
        (∫ u in (0 : ℝ)..R, A u) +
          (1 / 4 - r ^ 2) * ∫ u in (0 : ℝ)..R, T u := by
    have heq : F = fun u ↦ A u + (1 / 4 - r ^ 2) * T u := by
      funext u
      dsimp only [F, A, T]
      ring
    rw [heq, intervalIntegral.integral_add hAint
      (hTint.const_mul (1 / 4 - r ^ 2)),
      intervalIntegral.integral_const_mul]
  rw [hsplit, thetaCoshBoundary_zero] at hfund
  dsimp only [A, T] at hfund ⊢
  linarith

theorem tendsto_thetaCoshBoundary_atTop_zero
    {r : ℝ} (hr : |r| < 3 / 2) :
    Filter.Tendsto (thetaCoshBoundary r) Filter.atTop (𝓝 0) := by
  let C := ∑' n : ℕ, riemannPhiMajorantCoefficient n
  have hC : 0 ≤ C := by
    dsimp only [C]
    exact tsum_nonneg fun n ↦ (riemannPhiMajorantCoefficient_pos n).le
  have hrp : r < 3 / 2 := (abs_lt.mp hr).2
  have hrn : -r < 3 / 2 := by linarith [(abs_lt.mp hr).1]
  have hdpos :
      Filter.Tendsto
        (fun u ↦ Real.exp (r * u) * deriv riemannThetaTail u)
        Filter.atTop (𝓝 0) :=
    tendsto_exp_mul_zero_of_norm_le_exp_decay hC hrp
      (fun u hu ↦ by
        rw [deriv_riemannThetaTail_eq_tailDeriv_of_nonneg hu]
        exact norm_riemannThetaTailDeriv_le_decay hu)
  have hdneg :
      Filter.Tendsto
        (fun u ↦ Real.exp (-r * u) * deriv riemannThetaTail u)
        Filter.atTop (𝓝 0) :=
    tendsto_exp_mul_zero_of_norm_le_exp_decay hC hrn
      (fun u hu ↦ by
        rw [deriv_riemannThetaTail_eq_tailDeriv_of_nonneg hu]
        exact norm_riemannThetaTailDeriv_le_decay hu)
  have htpos :
      Filter.Tendsto
        (fun u ↦ Real.exp (r * u) * riemannThetaTail u)
        Filter.atTop (𝓝 0) :=
    tendsto_exp_mul_zero_of_norm_le_exp_decay hC hrp
      (fun u hu ↦ norm_riemannThetaTail_le_decay hu)
  have htneg :
      Filter.Tendsto
        (fun u ↦ Real.exp (-r * u) * riemannThetaTail u)
        Filter.atTop (𝓝 0) :=
    tendsto_exp_mul_zero_of_norm_le_exp_decay hC hrn
      (fun u hu ↦ norm_riemannThetaTail_le_decay hu)
  have hcosh :
      Filter.Tendsto
        (fun u ↦ Real.cosh (r * u) * deriv riemannThetaTail u)
        Filter.atTop (𝓝 0) := by
    have h := (hdpos.add hdneg).const_mul (1 / 2 : ℝ)
    convert h using 1
    · funext u
      rw [Real.cosh_eq]
      ring
    · ring
  have hsinh :
      Filter.Tendsto
        (fun u ↦ r * Real.sinh (r * u) * riemannThetaTail u)
        Filter.atTop (𝓝 0) := by
    have h := (htpos.sub htneg).const_mul (r / 2)
    convert h using 1
    · funext u
      rw [Real.sinh_eq]
      ring
    · ring
  have h := hcosh.sub hsinh
  change Filter.Tendsto
    (fun u ↦ Real.cosh (r * u) * deriv riemannThetaTail u -
      r * Real.sinh (r * u) * riemannThetaTail u)
    Filter.atTop (𝓝 0)
  simpa using h

theorem integral_Ioi_cosh_mul_riemannPhi_double_ibp_of_integrable
    {r : ℝ} (hr : |r| < 3 / 2)
    (hPhi : MeasureTheory.IntegrableOn
      (fun u : ℝ ↦ Real.cosh (r * u) * riemannPhi u) (Set.Ioi 0))
    (hTheta : MeasureTheory.IntegrableOn
      (fun u : ℝ ↦ Real.cosh (r * u) * riemannThetaTail u) (Set.Ioi 0)) :
    (∫ u : ℝ in Set.Ioi 0, Real.cosh (r * u) * riemannPhi u) =
      1 / 4 + (r ^ 2 - 1 / 4) *
        ∫ u : ℝ in Set.Ioi 0,
          Real.cosh (r * u) * riemannThetaTail u := by
  let A : ℝ → ℝ := fun u ↦ Real.cosh (r * u) * riemannPhi u
  let T : ℝ → ℝ := fun u ↦ Real.cosh (r * u) * riemannThetaTail u
  have hAlim :
      Filter.Tendsto (fun R ↦ ∫ u in (0 : ℝ)..R, A u)
        Filter.atTop (𝓝 (∫ u : ℝ in Set.Ioi 0, A u)) :=
    intervalIntegral_tendsto_integral_Ioi 0 hPhi Filter.tendsto_id
  have hTlim :
      Filter.Tendsto (fun R ↦ ∫ u in (0 : ℝ)..R, T u)
        Filter.atTop (𝓝 (∫ u : ℝ in Set.Ioi 0, T u)) :=
    intervalIntegral_tendsto_integral_Ioi 0 hTheta Filter.tendsto_id
  have hBlim := tendsto_thetaCoshBoundary_atTop_zero hr
  have hright :
      Filter.Tendsto
        (fun R ↦ thetaCoshBoundary r R + 1 / 4 +
          (r ^ 2 - 1 / 4) * ∫ u in (0 : ℝ)..R, T u)
        Filter.atTop
        (𝓝 (1 / 4 + (r ^ 2 - 1 / 4) *
          ∫ u : ℝ in Set.Ioi 0, T u)) := by
    convert (hBlim.add_const (1 / 4)).add
      (hTlim.const_mul (r ^ 2 - 1 / 4)) using 1 <;> ring
  have heq :
      (fun R ↦ ∫ u in (0 : ℝ)..R, A u) =ᶠ[Filter.atTop]
        fun R ↦ thetaCoshBoundary r R + 1 / 4 +
          (r ^ 2 - 1 / 4) * ∫ u in (0 : ℝ)..R, T u := by
    filter_upwards [Filter.eventually_ge_atTop (0 : ℝ)] with R hR
    exact intervalIntegral_cosh_mul_riemannPhi_double_ibp r hR
  have hright' :
      Filter.Tendsto (fun R ↦ ∫ u in (0 : ℝ)..R, A u)
        Filter.atTop
        (𝓝 (1 / 4 + (r ^ 2 - 1 / 4) *
          ∫ u : ℝ in Set.Ioi 0, T u)) :=
    hright.congr' heq.symm
  have hunique := tendsto_nhds_unique hAlim hright'
  simpa only [A, T] using hunique

theorem continuousOn_riemannThetaTail_Ioi :
    ContinuousOn riemannThetaTail (Set.Ioi 0) := by
  intro u hu
  exact (hasDerivAt_riemannThetaTail hu).continuousAt.continuousWithinAt

/-- Global exponential decay of Riemann's `Φ`, with an explicit finite constant. -/
theorem riemannPhi_le_exponential_majorant (u : ℝ) :
    riemannPhi u ≤
      (∑' n : ℕ, riemannPhiMajorantCoefficient n) *
        Real.exp (-(3 / 2 : ℝ) * |u|) := by
  unfold riemannPhi
  calc
    (∑' n : ℕ, riemannPhiTerm n u) ≤
        ∑' n : ℕ, riemannPhiMajorantCoefficient n *
          Real.exp (-(3 / 2 : ℝ) * |u|) :=
      (summable_riemannPhiTerm u).tsum_le_tsum
        (fun n ↦ riemannPhiTerm_le_majorant n u)
        (summable_riemannPhiMajorantCoefficient.mul_right _)
    _ = (∑' n : ℕ, riemannPhiMajorantCoefficient n) *
          Real.exp (-(3 / 2 : ℝ) * |u|) := tsum_mul_right

theorem integral_Ioi_cosh_mul_riemannPhi_double_ibp
    {r : ℝ} (hr : |r| < 3 / 2) :
    (∫ u : ℝ in Set.Ioi 0, Real.cosh (r * u) * riemannPhi u) =
      1 / 4 + (r ^ 2 - 1 / 4) *
        ∫ u : ℝ in Set.Ioi 0,
          Real.cosh (r * u) * riemannThetaTail u := by
  let C := ∑' n : ℕ, riemannPhiMajorantCoefficient n
  apply integral_Ioi_cosh_mul_riemannPhi_double_ibp_of_integrable hr
  · apply integrableOn_cosh_mul_of_norm_le_exp_decay hr
      continuous_riemannPhi.continuousOn
    intro u hu
    rw [Real.norm_eq_abs, abs_of_pos (riemannPhi_pos u)]
    simpa [C, abs_of_pos hu] using riemannPhi_le_exponential_majorant u
  · apply integrableOn_cosh_mul_of_norm_le_exp_decay hr
      continuousOn_riemannThetaTail_Ioi
    intro u hu
    exact norm_riemannThetaTail_le_decay hu.le

theorem completedRiemannZeta₀_real_eq_cosh_theta
    {r : ℝ} (hr : |r| < 3 / 2) :
    completedRiemannZeta₀ (1 / 2 + r : ℝ) =
      (4 * ∫ u : ℝ in Set.Ioi 0,
        Real.cosh (r * u) * riemannThetaTail u : ℝ) := by
  let G : ℝ → ℝ := fun u ↦ Real.cosh (r * u) * riemannThetaTail u
  let GC : ℝ → ℂ := fun u ↦ (8 : ℂ) * (G u : ℂ)
  have hGint : MeasureTheory.IntegrableOn G (Set.Ioi 0) := by
    apply integrableOn_cosh_mul_of_norm_le_exp_decay hr
      continuousOn_riemannThetaTail_Ioi
    intro u hu
    exact norm_riemannThetaTail_le_decay hu.le
  have hGCint : MeasureTheory.IntegrableOn GC (Set.Ioi 0) := by
    apply MeasureTheory.Integrable.const_mul
    exact hGint.norm.mono'
      (Complex.continuous_ofReal.comp_aestronglyMeasurable hGint.1)
      (Filter.Eventually.of_forall fun u ↦ by simp)
  have hMellin :=
    tendsto_intervalIntegral_mellin_truncation
      (mellinConvergent_hurwitzEvenFEPair_zero_f_modif
        (((1 / 2 + r : ℝ) : ℂ) / 2))
  have hGlim :
      Filter.Tendsto (fun R ↦ ∫ u in (0 : ℝ)..R, GC u)
        Filter.atTop (𝓝 (∫ u : ℝ in Set.Ioi 0, GC u)) :=
    intervalIntegral_tendsto_integral_Ioi 0 hGCint Filter.tendsto_id
  have hevent :
      (fun R ↦ ∫ x in Real.exp (-2 * R)..Real.exp (2 * R),
        (x : ℂ) ^ ((((1 / 2 + r : ℝ) : ℂ) / 2) - 1) *
          (HurwitzZeta.hurwitzEvenFEPair 0).f_modif x) =ᶠ[Filter.atTop]
        fun R ↦ ∫ u in (0 : ℝ)..R, GC u := by
    filter_upwards [Filter.eventually_ge_atTop (0 : ℝ)] with R hR
    rw [intervalIntegral_mellin_f_modif_eq_cosh_theta
      ((1 / 2 + r : ℝ) : ℂ) hR]
    apply intervalIntegral.integral_congr
    intro u _
    dsimp only [GC, G]
    push_cast
    have harg :
        (1 / 2 + (r : ℂ) - 1 / 2) * (u : ℂ) =
          (r : ℂ) * (u : ℂ) := by ring
    rw [harg]
    ring
  have hMellin' :
      Filter.Tendsto
        (fun R ↦ ∫ x in Real.exp (-2 * R)..Real.exp (2 * R),
          (x : ℂ) ^ ((((1 / 2 + r : ℝ) : ℂ) / 2) - 1) *
            (HurwitzZeta.hurwitzEvenFEPair 0).f_modif x)
        Filter.atTop (𝓝 (∫ u : ℝ in Set.Ioi 0, GC u)) :=
    hGlim.congr' hevent.symm
  have hunique := tendsto_nhds_unique hMellin hMellin'
  rw [completedRiemannZeta₀_eq_mellin_f_modif, hunique]
  dsimp only [GC, G]
  rw [MeasureTheory.integral_const_mul]
  rw [show
    (∫ a : ℝ in Set.Ioi 0,
      ((Real.cosh (r * a) * riemannThetaTail a : ℝ) : ℂ)) =
        ((∫ a : ℝ in Set.Ioi 0,
          Real.cosh (r * a) * riemannThetaTail a : ℝ) : ℂ) from
    integral_ofReal]
  push_cast
  ring

theorem integrable_exp_neg_mul_abs {c : ℝ} (hc : 0 < c) :
    Integrable (fun u : ℝ ↦ Real.exp (-c * |u|)) := by
  rw [← integrableOn_univ, ← Set.Iic_union_Ioi (a := 0)]
  apply IntegrableOn.union
  · refine (integrableOn_exp_mul_Iic hc 0).congr_fun ?_ measurableSet_Iic
    intro u hu
    change u ≤ 0 at hu
    change Real.exp (c * u) = Real.exp (-c * |u|)
    rw [abs_of_nonpos hu]
    congr 1
    ring
  · refine (integrableOn_exp_mul_Ioi (neg_neg_of_pos hc) 0).congr_fun ?_
        measurableSet_Ioi
    intro u hu
    change 0 < u at hu
    change Real.exp (-c * u) = Real.exp (-c * |u|)
    rw [abs_of_pos hu]

/-- Every polynomial moment of the common exponential majorant is integrable. -/
theorem integrable_abs_pow_mul_exp_majorant (k : ℕ) :
    Integrable (fun u : ℝ ↦ |u| ^ k * Real.exp (-(3 / 2 : ℝ) * |u|)) := by
  have hslow : Integrable (fun u : ℝ ↦ Real.exp (-(1 / 2 : ℝ) * |u|)) :=
    integrable_exp_neg_mul_abs (by norm_num)
  have hfast : Integrable (fun u : ℝ ↦ Real.exp (-(5 / 2 : ℝ) * |u|)) :=
    integrable_exp_neg_mul_abs (by norm_num)
  have hpos :
      Integrable (fun u : ℝ ↦
        Real.exp ((-(3 / 2 : ℝ) + 1) * |u|)) := by
    convert hslow using 1
    funext u
    congr 1
    ring
  have hneg :
      Integrable (fun u : ℝ ↦
        Real.exp ((-(3 / 2 : ℝ) - 1) * |u|)) := by
    convert hfast using 1
    funext u
    congr 1
    ring
  have h := ProbabilityTheory.integrable_pow_abs_mul_exp_of_integrable_exp_mul
    (μ := volume) (X := fun u : ℝ ↦ |u|) (v := -(3 / 2 : ℝ))
    (t := 1) one_ne_zero
    hpos hneg k
  simpa [abs_abs] using h

/-- All moments of the concrete Riemann `Φ` kernel are Lebesgue integrable. -/
theorem integrable_riemannPhi_mul_pow (k : ℕ) :
    Integrable (fun u : ℝ ↦ riemannPhi u * u ^ k) := by
  let C := ∑' n : ℕ, riemannPhiMajorantCoefficient n
  have hC : 0 ≤ C :=
    tsum_nonneg fun n ↦ (riemannPhiMajorantCoefficient_pos n).le
  have hmajor :
      Integrable (fun u : ℝ ↦ C *
        (|u| ^ k * Real.exp (-(3 / 2 : ℝ) * |u|))) :=
    (integrable_abs_pow_mul_exp_majorant k).const_mul C
  apply hmajor.mono'
  · exact (continuous_riemannPhi.mul (continuous_id.pow k)).aestronglyMeasurable
  · filter_upwards with u
    rw [Real.norm_eq_abs, abs_mul, abs_of_pos (riemannPhi_pos u), abs_pow]
    have hdecay :
        riemannPhi u ≤ C * Real.exp (-(3 / 2 : ℝ) * |u|) := by
      simpa only [C] using riemannPhi_le_exponential_majorant u
    calc
      riemannPhi u * |u| ^ k ≤
          (C * Real.exp (-(3 / 2 : ℝ) * |u|)) * |u| ^ k := by
        gcongr
      _ = C * (|u| ^ k * Real.exp (-(3 / 2 : ℝ) * |u|)) := by ring

theorem integrable_riemannPhi_mul_exp_of_abs_lt_one
    {r : ℝ} (hr : |r| < 1) :
    Integrable (fun u : ℝ ↦ riemannPhi u * Real.exp (r * u)) := by
  let C := ∑' n : ℕ, riemannPhiMajorantCoefficient n
  have hC : 0 ≤ C :=
    tsum_nonneg fun n ↦ (riemannPhiMajorantCoefficient_pos n).le
  have hmajor :
      Integrable (fun u : ℝ ↦ C * Real.exp (-(1 / 2 : ℝ) * |u|)) :=
    (integrable_exp_neg_mul_abs (by norm_num)).const_mul C
  apply hmajor.mono'
  · exact (continuous_riemannPhi.mul (by fun_prop)).aestronglyMeasurable
  · filter_upwards with u
    rw [Real.norm_eq_abs, abs_mul,
      abs_of_pos (riemannPhi_pos u), abs_of_pos (Real.exp_pos _)]
    have hru : r * u ≤ |u| := by
      calc
        r * u ≤ |r| * |u| := (le_abs_self (r * u)).trans_eq (abs_mul r u)
        _ ≤ 1 * |u| := by
          gcongr
        _ = |u| := one_mul _
    have hdecay :
        riemannPhi u ≤ C * Real.exp (-(3 / 2 : ℝ) * |u|) := by
      simpa only [C] using riemannPhi_le_exponential_majorant u
    calc
      riemannPhi u * Real.exp (r * u) ≤
          (C * Real.exp (-(3 / 2 : ℝ) * |u|)) *
            Real.exp (r * u) := by gcongr
      _ ≤ (C * Real.exp (-(3 / 2 : ℝ) * |u|)) *
            Real.exp |u| := by gcongr
      _ = C * Real.exp (-(1 / 2 : ℝ) * |u|) := by
        rw [mul_assoc, ← Real.exp_add]
        congr 2
        ring

theorem riemannXi_one_sub (s : ℂ) : riemannXi (1 - s) = riemannXi s := by
  rw [riemannXi, riemannXi, completedRiemannZeta₀_one_sub]
  ring_nf

/-- The entire function in equation (1) of Griffin--Ono--Rolen--Zagier:
the entire continuation of
`(-1 + 4 z^2) Λ(1/2 + z) = 8 ξ(1/2 + z)`. -/
def xiJensenEntire (z : ℂ) : ℂ :=
  8 * riemannXi (1 / 2 + z)

theorem xiJensenEntire_eq_four_mul_riemannXiLi (z : ℂ) :
    xiJensenEntire z = 4 * riemannXiLi (1 / 2 + z) := by
  simp only [xiJensenEntire, riemannXi, riemannXiLi]
  ring

theorem differentiable_xiJensenEntire : Differentiable ℂ xiJensenEntire := by
  unfold xiJensenEntire
  fun_prop

/-- The centered xi function inherits the proved global order-one estimate
from the Li normalization. -/
theorem exists_xiJensenEntire_orderOneGrowthBound :
    ∃ C R : ℝ, 0 ≤ C ∧ 2 ≤ R ∧ ∀ z : ℂ, R ≤ ‖z‖ →
      ‖xiJensenEntire z‖ ≤
        Real.exp (C * ‖z‖ * Real.log (‖z‖ + 2)) := by
  obtain ⟨C, R, hC, hR, hbound⟩ := riemannXi_orderOneGrowthBound
  let D := 4 * C + 2
  let R' : ℝ := max 5 (R + 1)
  refine ⟨D, R', by dsimp [D]; positivity,
    (by dsimp [R']; linarith [le_max_left (5 : ℝ) (R + 1)]), ?_⟩
  intro z hz
  let r := ‖z‖
  let w : ℂ := 1 / 2 + z
  have hr5 : 5 ≤ r := (le_max_left _ _).trans hz
  have hRw : R ≤ ‖w‖ := by
    have hlower : r - 1 ≤ ‖w‖ := by
      have h := norm_sub_le w (1 / 2)
      have hw : w - 1 / 2 = z := by dsimp [w]; ring
      rw [hw] at h
      have hhalf : ‖(1 / 2 : ℂ)‖ = 1 / 2 := by norm_num
      rw [hhalf] at h
      linarith
    have hRr : R + 1 ≤ r := (le_max_right 5 (R + 1)).trans hz
    linarith
  have hwUpper : ‖w‖ ≤ 2 * r := by
    have h := norm_add_le (1 / 2 : ℂ) z
    have hhalf : ‖(1 / 2 : ℂ)‖ = 1 / 2 := by norm_num
    rw [hhalf] at h
    linarith
  have hlog0 : 0 ≤ Real.log (r + 2) :=
    Real.log_nonneg (by linarith)
  have hlogw : Real.log (‖w‖ + 2) ≤ 2 * Real.log (r + 2) := by
    have harg : ‖w‖ + 2 ≤ (r + 2) ^ 2 := by
      nlinarith [sq_nonneg (r + 2)]
    calc
      Real.log (‖w‖ + 2) ≤ Real.log ((r + 2) ^ 2) :=
        Real.log_le_log (by positivity) harg
      _ = 2 * Real.log (r + 2) := by
        rw [Real.log_pow]
        norm_num
  have hlogw0 : 0 ≤ Real.log (‖w‖ + 2) :=
    Real.log_nonneg (by linarith [norm_nonneg w])
  have hbase := hbound w hRw
  rw [xiJensenEntire_eq_four_mul_riemannXiLi, norm_mul] 
  norm_num
  calc
    4 * ‖riemannXiLi w‖ ≤
        4 * Real.exp (C * ‖w‖ * Real.log (‖w‖ + 2)) := by gcongr
    _ ≤ 4 * Real.exp (4 * C * r * Real.log (r + 2)) := by
      apply mul_le_mul_of_nonneg_left (Real.exp_le_exp.mpr ?_) (by norm_num)
      calc
        C * ‖w‖ * Real.log (‖w‖ + 2)
            ≤ C * (2 * r) * Real.log (‖w‖ + 2) := by
              exact mul_le_mul_of_nonneg_right
                (mul_le_mul_of_nonneg_left hwUpper hC) hlogw0
        _ ≤ C * (2 * r) * (2 * Real.log (r + 2)) := by
              exact mul_le_mul_of_nonneg_left hlogw
                (mul_nonneg hC (by positivity))
        _ = 4 * C * r * Real.log (r + 2) := by ring
    _ ≤ Real.exp (D * r * Real.log (r + 2)) := by
      rw [show D * r * Real.log (r + 2) =
        4 * C * r * Real.log (r + 2) +
          2 * r * Real.log (r + 2) by dsimp [D]; ring,
        Real.exp_add]
      have hlog2 : Real.log 2 ≤ Real.log (r + 2) :=
        Real.log_le_log (by norm_num) (by linarith)
      have hexp : 4 ≤ Real.exp (2 * r * Real.log (r + 2)) := by
        rw [← Real.exp_log (by norm_num : (0 : ℝ) < 4)]
        apply Real.exp_le_exp.mpr
        rw [show Real.log 4 = 2 * Real.log 2 by
          rw [show (4 : ℝ) = 2 ^ 2 by norm_num, Real.log_pow]; norm_num]
        nlinarith
      rw [show 4 * Real.exp (4 * C * r * Real.log (r + 2)) =
        Real.exp (4 * C * r * Real.log (r + 2)) * 4 by ring]
      exact mul_le_mul_of_nonneg_left hexp (Real.exp_pos _).le

theorem xiJensenEntire_real_eq_riemannPhi_cosh
    {r : ℝ} (hr : |r| < 3 / 2) :
    xiJensenEntire r =
      (16 * ∫ u : ℝ in Set.Ioi 0,
        riemannPhi u * Real.cosh (r * u) : ℝ) := by
  have hcompleted :
      completedRiemannZeta₀ (1 / 2 + (r : ℂ)) =
        (4 * ∫ u : ℝ in Set.Ioi 0,
          Real.cosh (r * u) * riemannThetaTail u : ℝ) := by
    convert completedRiemannZeta₀_real_eq_cosh_theta hr using 1 <;>
      push_cast <;> ring
  rw [xiJensenEntire, riemannXi, hcompleted]
  have hibp := integral_Ioi_cosh_mul_riemannPhi_double_ibp hr
  have hcomm :
      (∫ u : ℝ in Set.Ioi 0,
        riemannPhi u * Real.cosh (r * u)) =
      ∫ u : ℝ in Set.Ioi 0,
        Real.cosh (r * u) * riemannPhi u := by
    apply MeasureTheory.integral_congr_ae
    filter_upwards with u
    ring
  rw [hcomm, hibp]
  push_cast
  ring

noncomputable def riemannPhiDensity (u : ℝ) : ENNReal :=
  ENNReal.ofReal (riemannPhi u)

noncomputable def riemannPhiMeasure : Measure ℝ :=
  MeasureTheory.volume.withDensity riemannPhiDensity

theorem measurable_riemannPhiDensity : Measurable riemannPhiDensity := by
  unfold riemannPhiDensity
  exact ENNReal.measurable_ofReal.comp continuous_riemannPhi.measurable

theorem riemannPhiDensity_lt_top :
    ∀ᵐ u ∂(MeasureTheory.volume : Measure ℝ),
      riemannPhiDensity u < (⊤ : ENNReal) := by
  filter_upwards with u
  simp [riemannPhiDensity]

theorem Ioo_subset_interior_integrableExpSet_riemannPhiMeasure :
    Set.Ioo (-1 : ℝ) 1 ⊆ interior
      (ProbabilityTheory.integrableExpSet id riemannPhiMeasure) := by
  apply interior_maximal ?_ isOpen_Ioo
  intro r hr
  change Integrable (fun u : ℝ ↦ Real.exp (r * id u)) riemannPhiMeasure
  rw [riemannPhiMeasure,
    MeasureTheory.integrable_withDensity_iff_integrable_smul'
      measurable_riemannPhiDensity riemannPhiDensity_lt_top]
  have htoReal :
      ∀ u : ℝ, (riemannPhiDensity u).toReal = riemannPhi u := by
    intro u
    simp [riemannPhiDensity, (riemannPhi_pos u).le]
  simp_rw [id_eq, htoReal, smul_eq_mul, mul_comm]
  simpa only [mul_comm] using
    integrable_riemannPhi_mul_exp_of_abs_lt_one (abs_lt.mpr hr)

theorem zero_mem_interior_integrableExpSet_riemannPhiMeasure :
    (0 : ℝ) ∈ interior
      (ProbabilityTheory.integrableExpSet id riemannPhiMeasure) :=
  Ioo_subset_interior_integrableExpSet_riemannPhiMeasure (by norm_num)

theorem complexMGF_riemannPhiMeasure (z : ℂ) :
    ProbabilityTheory.complexMGF id riemannPhiMeasure z =
      ∫ u : ℝ, riemannPhi u * Complex.exp (z * u) := by
  rw [ProbabilityTheory.complexMGF, riemannPhiMeasure,
    integral_withDensity_eq_integral_toReal_smul
      measurable_riemannPhiDensity riemannPhiDensity_lt_top]
  apply MeasureTheory.integral_congr_ae
  filter_upwards with u
  simp [riemannPhiDensity, (riemannPhi_pos u).le]

theorem complexMGF_riemannPhiMeasure_real
    {r : ℝ} (hr : |r| < 1) :
    ProbabilityTheory.complexMGF id riemannPhiMeasure r =
      (2 * ∫ u : ℝ in Set.Ioi 0,
        riemannPhi u * Real.cosh (r * u) : ℝ) := by
  rw [complexMGF_riemannPhiMeasure]
  let fp : ℝ → ℂ := fun u ↦
    riemannPhi u * Complex.exp ((r : ℂ) * u)
  let fm : ℝ → ℂ := fun u ↦
    riemannPhi u * Complex.exp (-(r : ℂ) * u)
  have hfpReal := integrable_riemannPhi_mul_exp_of_abs_lt_one hr
  have hfmReal := integrable_riemannPhi_mul_exp_of_abs_lt_one
    (by simpa using hr : |-r| < 1)
  have hfp : Integrable fp := by
    apply hfpReal.mono'
    · dsimp only [fp]
      exact ((Complex.continuous_ofReal.comp continuous_riemannPhi).mul
        (Complex.continuous_exp.comp (by fun_prop))).aestronglyMeasurable
    · filter_upwards with u
      dsimp only [fp]
      rw [norm_mul, Complex.norm_real, Complex.norm_exp]
      simp only [mul_re, ofReal_re, ofReal_im, mul_zero, sub_zero]
      rw [Real.norm_eq_abs, abs_of_pos (riemannPhi_pos u)]
  have hfm : Integrable fm := by
    apply hfmReal.mono'
    · dsimp only [fm]
      exact ((Complex.continuous_ofReal.comp continuous_riemannPhi).mul
        (Complex.continuous_exp.comp (by fun_prop))).aestronglyMeasurable
    · filter_upwards with u
      dsimp only [fm]
      rw [norm_mul, Complex.norm_real, Complex.norm_exp]
      simp only [neg_re, mul_re, ofReal_re, ofReal_im, mul_zero, sub_zero]
      rw [Real.norm_eq_abs, abs_of_pos (riemannPhi_pos u)]
  have hlower :
      (∫ u : ℝ in Set.Iic 0, fp u) =
        ∫ u : ℝ in Set.Ioi 0, fm u := by
    calc
      (∫ u : ℝ in Set.Iic 0, fp u) =
          ∫ u : ℝ in Set.Ioi 0, fp (-u) := by
        simpa using (integral_comp_neg_Ioi 0 fp).symm
      _ = ∫ u : ℝ in Set.Ioi 0, fm u := by
        apply MeasureTheory.integral_congr_ae
        filter_upwards with u
        dsimp only [fp, fm]
        rw [riemannPhi_even]
        push_cast
        congr 2
        ring
  have hsplit := intervalIntegral.integral_Iic_add_Ioi
    (b := 0) hfp.integrableOn hfp.integrableOn
  rw [← hsplit, hlower, ← MeasureTheory.integral_add
    hfm.integrableOn hfp.integrableOn]
  rw [show (∫ u : ℝ in Set.Ioi 0, fm u + fp u) =
      ∫ u : ℝ in Set.Ioi 0,
        ((2 * riemannPhi u * Real.cosh (r * u) : ℝ) : ℂ) by
    apply MeasureTheory.integral_congr_ae
    filter_upwards with u
    dsimp only [fp, fm]
    push_cast
    rw [Complex.cosh]
    ring]
  rw [show
    (∫ u : ℝ in Set.Ioi 0,
      ((2 * riemannPhi u * Real.cosh (r * u) : ℝ) : ℂ)) =
      ((∫ u : ℝ in Set.Ioi 0,
        2 * riemannPhi u * Real.cosh (r * u) : ℝ) : ℂ) from integral_ofReal]
  rw [show
      (∫ u : ℝ in Set.Ioi 0,
        2 * riemannPhi u * Real.cosh (r * u)) =
        2 * ∫ u : ℝ in Set.Ioi 0,
          riemannPhi u * Real.cosh (r * u) by
    rw [← MeasureTheory.integral_const_mul]
    apply MeasureTheory.integral_congr_ae
    filter_upwards with u
    ring]

theorem xiJensenEntire_eq_complexMGF_on_strip :
    Set.EqOn xiJensenEntire
      (fun z ↦ 8 * ProbabilityTheory.complexMGF id riemannPhiMeasure z)
      {z : ℂ | z.re ∈ Set.Ioo (-1 : ℝ) 1} := by
  let U : Set ℂ := {z : ℂ | z.re ∈ Set.Ioo (-1 : ℝ) 1}
  let H : ℂ → ℂ := fun z ↦
    8 * ProbabilityTheory.complexMGF id riemannPhiMeasure z
  have hUopen : IsOpen U := by
    exact isOpen_Ioo.preimage continuous_re
  have hUconvex : Convex ℝ U := by
    intro x hx y hy a b ha hb hab
    change -1 < (a • x + b • y).re ∧
      (a • x + b • y).re < 1
    simp only [Complex.add_re, Complex.smul_re, smul_eq_mul]
    change (-1 < x.re ∧ x.re < 1) at hx
    change (-1 < y.re ∧ y.re < 1) at hy
    by_cases ha0 : a = 0
    · subst a
      have hb1 : b = 1 := by linarith
      subst b
      simpa using hy
    · have haPos : 0 < a := lt_of_le_of_ne ha (Ne.symm ha0)
      have haxL : a * (-1) < a * x.re :=
        mul_lt_mul_of_pos_left hx.1 haPos
      have haxU : a * x.re < a * 1 :=
        mul_lt_mul_of_pos_left hx.2 haPos
      have hbyL : b * (-1) ≤ b * y.re :=
        mul_le_mul_of_nonneg_left hy.1.le hb
      have hbyU : b * y.re ≤ b * 1 :=
        mul_le_mul_of_nonneg_left hy.2.le hb
      constructor <;> nlinarith
  have hXi : AnalyticOnNhd ℂ xiJensenEntire U :=
    ((analyticOnNhd_univ_iff_differentiable.mpr
      differentiable_xiJensenEntire).mono (Set.subset_univ U))
  have hMdiff :
      DifferentiableOn ℂ
        (ProbabilityTheory.complexMGF id riemannPhiMeasure) U := by
    apply ProbabilityTheory.differentiableOn_complexMGF.mono
    intro z hz
    exact Ioo_subset_interior_integrableExpSet_riemannPhiMeasure hz
  have hH : AnalyticOnNhd ℂ H U := by
    rw [analyticOnNhd_iff_differentiableOn hUopen]
    dsimp only [H]
    fun_prop
  have hclosure :
      (0 : ℂ) ∈ closure
        ({z : ℂ | xiJensenEntire z = H z} \ {(0 : ℂ)}) := by
    apply mem_closure_of_tendsto
      (tendsto_one_div_add_atTop_nhds_zero_nat (𝕜 := ℂ))
    filter_upwards [Filter.eventually_ge_atTop 1] with n hn
    have hrpos : 0 < (1 : ℝ) / (n + 1) := by positivity
    have hrlt : (1 : ℝ) / (n + 1) < 1 := by
      rw [div_lt_one (by positivity)]
      exact_mod_cast Nat.lt_add_one_iff.mpr hn
    constructor
    · change xiJensenEntire ((1 : ℂ) / (n + 1)) =
        8 * ProbabilityTheory.complexMGF id riemannPhiMeasure
          ((1 : ℂ) / (n + 1))
      have hcast :
          ((1 : ℂ) / (n + 1)) =
            (((1 : ℝ) / (n + 1) : ℝ) : ℂ) := by
        push_cast
        rfl
      rw [hcast,
        xiJensenEntire_real_eq_riemannPhi_cosh
          (by rw [abs_of_pos hrpos]; linarith),
        complexMGF_riemannPhiMeasure_real
          (by rw [abs_of_pos hrpos]; exact hrlt)]
      push_cast
      ring
    · simp only [Set.mem_singleton_iff]
      exact div_ne_zero one_ne_zero (by
        exact_mod_cast Nat.add_one_ne_zero n)
  have heq := hXi.eqOn_of_preconnected_of_mem_closure hH
    hUconvex.isPreconnected (show (0 : ℂ) ∈ U by
      dsimp only [U]
      norm_num) hclosure
  simpa only [U, H] using heq

theorem xiJensenEntire_neg (z : ℂ) :
    xiJensenEntire (-z) = xiJensenEntire z := by
  have harg : (1 / 2 : ℂ) + -z = 1 - (1 / 2 + z) := by ring
  rw [xiJensenEntire, xiJensenEntire, harg, riemannXi_one_sub]

/-- The exact source-level conjugation proposition for pinned Mathlib's
pole-removed completed zeta. -/
def CompletedZetaConjugation : Prop :=
  ∀ s : ℂ, completedRiemannZeta₀ (conj s) = conj (completedRiemannZeta₀ s)

/-- On the half-plane of absolute convergence, the Dirichlet series for
`riemannZeta` commutes with complex conjugation. -/
theorem riemannZeta_conj_of_one_lt_re (s : ℂ) (hs : 1 < s.re) :
    riemannZeta (conj s) = conj (riemannZeta s) := by
  rw [zeta_eq_tsum_one_div_nat_add_one_cpow (by simpa using hs),
    zeta_eq_tsum_one_div_nat_add_one_cpow hs, Complex.conj_tsum]
  congr 1
  funext n
  simp only [map_div₀, map_one]
  rw [Complex.cpow_conj]
  · simp
  · rw [show (n : ℂ) + 1 = ((n + 1 : ℕ) : ℂ) by norm_num,
      Complex.natCast_arg]
    exact Real.pi_ne_zero.symm

/-- The completed pole-removed zeta commutes with conjugation on `Re(s) > 1`.
The proof uses Mathlib's exact normalization
`Λ₀(s) = π^(-s/2) Γ(s/2) ζ(s) + 1/s + 1/(1-s)`. -/
theorem completedRiemannZeta₀_conj_of_one_lt_re (s : ℂ) (hs : 1 < s.re) :
    completedRiemannZeta₀ (conj s) =
      conj (completedRiemannZeta₀ s) := by
  have hs0 : s ≠ 0 := by
    intro h
    subst s
    norm_num at hs
  have hcs0 : conj s ≠ 0 := by
    intro h
    apply hs0
    have := congrArg conj h
    simpa using this
  have hzeta := riemannZeta_conj_of_one_lt_re s hs
  let D : ℂ → ℂ := fun z ↦
    (Real.pi : ℂ) ^ (-z / 2) * Gamma (z / 2)
  have hD (z : ℂ) (hz : 0 < z.re) : D z ≠ 0 := by
    apply mul_ne_zero
    · rw [Complex.cpow_ne_zero_iff]
      exact Or.inl (Complex.ofReal_ne_zero.mpr Real.pi_ne_zero)
    · exact Gamma_ne_zero_of_re_pos
        (by simpa [D] using div_pos hz zero_lt_two)
  have hDconj : D (conj s) = conj (D s) := by
    dsimp only [D]
    rw [map_mul]
    congr 1
    · rw [show -(conj s) / 2 = conj (-s / 2) by
          simp only [map_div₀, map_neg, Complex.conj_ofNat],
        Complex.cpow_conj]
      · simp
      · rw [Complex.arg_ofReal_of_nonneg Real.pi_nonneg]
        exact Real.pi_ne_zero.symm
    · rw [← Complex.Gamma_conj]
      congr 1
      simp only [map_div₀, Complex.conj_ofNat]
  have hformula :=
    riemannZeta_eq_completedRiemannZeta₀ (s := s) hs0
  have hcformula :=
    riemannZeta_eq_completedRiemannZeta₀ (s := conj s) hcs0
  have hsolve :
      completedRiemannZeta₀ s =
        riemannZeta s * D s + 1 / s + 1 / (1 - s) := by
    have hmul := (eq_div_iff (hD s (lt_trans zero_lt_one hs))).mp hformula
    dsimp only [D] at hmul
    linear_combination -hmul
  have hcsolve :
      completedRiemannZeta₀ (conj s) =
        riemannZeta (conj s) * D (conj s) +
          1 / conj s + 1 / (1 - conj s) := by
    have hmul :=
      (eq_div_iff
        (hD (conj s) (by simpa using lt_trans zero_lt_one hs))).mp hcformula
    dsimp only [D] at hmul
    linear_combination -hmul
  rw [hcsolve, hsolve, hzeta, hDconj]
  simp

/-- Global conjugation symmetry for pinned Mathlib's
`completedRiemannZeta₀`.  Both sides are entire; equality on the open
half-plane `Re(s) > 1` therefore extends by the analytic identity theorem. -/
theorem completedRiemannZeta₀_conj (s : ℂ) :
    completedRiemannZeta₀ (conj s) =
      conj (completedRiemannZeta₀ s) := by
  let f : ℂ → ℂ := completedRiemannZeta₀
  let g : ℂ → ℂ := conj ∘ f ∘ conj
  have hf : Differentiable ℂ f := differentiable_completedZeta₀
  have hg : Differentiable ℂ g := by
    intro z
    simpa [g] using (hf (conj z)).conj_conj
  have hlocal : ∀ z : ℂ, 1 < z.re → f z = g z := by
    intro z hz
    dsimp only [f, g, Function.comp_apply]
    rw [completedRiemannZeta₀_conj_of_one_lt_re z hz, conj_conj]
  have hfg : f = g := by
    apply AnalyticOnNhd.eq_of_eventuallyEq
        ((analyticOnNhd_univ_iff_differentiable).2 hf)
        ((analyticOnNhd_univ_iff_differentiable).2 hg)
    filter_upwards [
      (isOpen_lt continuous_const continuous_re).mem_nhds
        (show (2 : ℂ) ∈ {z : ℂ | 1 < z.re} by norm_num)] with z hz
    exact hlocal z hz
  simpa only [f, g, Function.comp_apply, conj_conj] using
    congrFun hfg (conj s)

/-- The previously explicit completed-zeta conjugation obligation is
unconditional in the pinned Mathlib environment. -/
theorem completedZetaConjugation : CompletedZetaConjugation :=
  completedRiemannZeta₀_conj

theorem riemannXi_conj_of_completedZetaConjugation
    (hconj : CompletedZetaConjugation) (s : ℂ) :
    riemannXi (conj s) = conj (riemannXi s) := by
  rw [riemannXi, riemannXi, hconj]
  simp only [map_div₀, map_add, map_one, map_mul, map_sub, Complex.conj_ofNat]

theorem xiJensenEntire_conj_of_completedZetaConjugation
    (hconj : CompletedZetaConjugation) (z : ℂ) :
    xiJensenEntire (conj z) = conj (xiJensenEntire z) := by
  rw [xiJensenEntire, xiJensenEntire]
  simp only [map_mul, Complex.conj_ofNat]
  rw [show (1 / 2 : ℂ) + conj z = conj (1 / 2 + z) by simp [Complex.conj_ofNat],
    riemannXi_conj_of_completedZetaConjugation hconj]

/-- Conjugation symmetry of a complex function forces every derivative at
zero to be real.  The proof uses Mathlib's exact derivative rule for
`conj ∘ f ∘ conj` and therefore does not assume a power-series uniqueness
principle. -/
theorem iteratedDeriv_im_zero_of_conj_symmetry
    (f : ℂ → ℂ) (hf : ∀ z, f (conj z) = conj (f z)) (n : ℕ) :
    (iteratedDeriv n f 0).im = 0 := by
  have hsym : conj ∘ f ∘ conj = f := by
    funext z
    simp only [Function.comp_apply]
    rw [hf]
    simp
  have hiter :
      conj ∘ iteratedDeriv n f ∘ conj = iteratedDeriv n f := by
    induction n with
    | zero => simpa using hsym
    | succ n ih =>
        have hderiv := congrArg deriv ih
        simpa [Nat.succ_eq_add_one, iteratedDeriv_succ] using hderiv
  apply Complex.conj_eq_iff_im.mp
  simpa [Function.comp_apply] using congrFun hiter 0

/-- The complex derivative normalization underlying the classical real
coefficient `γ(n)`: `n! ξ_J^(2n)(0) / (2n)!`. -/
def xiGammaComplex (n : ℕ) : ℂ :=
  (n.factorial : ℂ) / ((2 * n).factorial : ℂ) *
    iteratedDeriv (2 * n) xiJensenEntire 0

theorem iteratedDeriv_xiJensenEntire_eq_riemannPhi_moment (n : ℕ) :
    iteratedDeriv n xiJensenEntire 0 =
      (8 * ∫ u : ℝ, riemannPhi u * u ^ n : ℝ) := by
  let U : Set ℂ := {z : ℂ | z.re ∈ Set.Ioo (-1 : ℝ) 1}
  let M : ℂ → ℂ := fun z ↦
    ProbabilityTheory.complexMGF id riemannPhiMeasure z
  have heqOn :
      Set.EqOn xiJensenEntire (fun z ↦ 8 * M z) U := by
    simpa only [U, M] using xiJensenEntire_eq_complexMGF_on_strip
  have hUopen : IsOpen U := isOpen_Ioo.preimage continuous_re
  have hzero : (0 : ℂ) ∈ U := by
    dsimp only [U]
    norm_num
  have hevent : xiJensenEntire =ᶠ[𝓝 (0 : ℂ)] fun z ↦ 8 * M z := by
    filter_upwards [hUopen.mem_nhds hzero] with z hz
    exact heqOn hz
  have hiter := hevent.iteratedDeriv_eq n
  rw [iteratedDeriv_const_mul_field] at hiter
  have hmgf := ProbabilityTheory.iteratedDeriv_complexMGF
    (X := id) (μ := riemannPhiMeasure) (z := (0 : ℂ))
    zero_mem_interior_integrableExpSet_riemannPhiMeasure n
  dsimp only [M] at hiter
  rw [hmgf] at hiter
  simp only [id_eq, zero_mul, Complex.exp_zero, mul_one] at hiter
  rw [riemannPhiMeasure,
    integral_withDensity_eq_integral_toReal_smul
      measurable_riemannPhiDensity riemannPhiDensity_lt_top] at hiter
  have hdensity :
      ∀ u : ℝ, (riemannPhiDensity u).toReal = riemannPhi u := by
    intro u
    simp [riemannPhiDensity, (riemannPhi_pos u).le]
  change iteratedDeriv n xiJensenEntire 0 =
    8 * ∫ u : ℝ, ((riemannPhiDensity u).toReal : ℂ) * (u : ℂ) ^ n at hiter
  simp_rw [hdensity] at hiter
  rw [show
      (∫ u : ℝ, (riemannPhi u : ℂ) * (u : ℂ) ^ n) =
        ((∫ u : ℝ, riemannPhi u * u ^ n : ℝ) : ℂ) by
    calc
      (∫ u : ℝ, (riemannPhi u : ℂ) * (u : ℂ) ^ n) =
          ∫ u : ℝ, ((riemannPhi u * u ^ n : ℝ) : ℂ) := by
            apply MeasureTheory.integral_congr_ae
            filter_upwards with u
            push_cast
            rfl
      _ = ((∫ u : ℝ, riemannPhi u * u ^ n : ℝ) : ℂ) :=
        integral_ofReal] at hiter
  push_cast
  exact hiter

/-- Conjugation symmetry implies reality of every normalized xi coefficient. -/
theorem xiGammaComplex_im_zero_of_completedZetaConjugation
    (hconj : CompletedZetaConjugation) (n : ℕ) :
    (xiGammaComplex n).im = 0 := by
  have hderiv :=
    iteratedDeriv_im_zero_of_conj_symmetry xiJensenEntire
      (xiJensenEntire_conj_of_completedZetaConjugation hconj) (2 * n)
  simp [xiGammaComplex, Complex.mul_im, hderiv]

/-- Every normalized centered xi derivative is real, unconditionally. -/
theorem xiGammaComplex_im_zero (n : ℕ) :
    (xiGammaComplex n).im = 0 :=
  xiGammaComplex_im_zero_of_completedZetaConjugation
    completedZetaConjugation n

/-- Real xi coefficient sequence, initially defined by taking the real part;
`xiGammaComplex_eq_ofReal` below proves this loses no information. -/
def xiGamma (n : ℕ) : ℝ :=
  (xiGammaComplex n).re

/-- The complex normalized derivative is exactly the scalar extension of the
real xi coefficient sequence. -/
theorem xiGammaComplex_eq_ofReal (n : ℕ) :
    xiGammaComplex n = (xiGamma n : ℂ) := by
  apply Complex.ext
  · rfl
  · simpa using xiGammaComplex_im_zero n

/-- Evenness of the centered xi function kills every odd derivative at the
center. -/
theorem iteratedDeriv_xiJensenEntire_odd_zero (n : ℕ) :
    iteratedDeriv (2 * n + 1) xiJensenEntire 0 = 0 := by
  have hfun : (fun z : ℂ ↦ xiJensenEntire (-z)) = xiJensenEntire := by
    funext z
    exact xiJensenEntire_neg z
  have hder :=
    congrArg (fun f : ℂ → ℂ ↦ iteratedDeriv (2 * n + 1) f 0) hfun
  rw [iteratedDeriv_comp_neg] at hder
  norm_num [pow_succ, pow_mul] at hder
  simpa only [CharZero.neg_eq_self_iff] using hder

/-- The even Taylor term at index `2n` is precisely the sourced
`xiGamma n / n!` coefficient. -/
theorem xiJensenEntire_even_taylor_term (n : ℕ) (z : ℂ) :
    ((2 * n).factorial : ℂ)⁻¹ *
        iteratedDeriv (2 * n) xiJensenEntire 0 * z ^ (2 * n) =
      ((xiGamma n / n.factorial : ℝ) : ℂ) * z ^ (2 * n) := by
  push_cast
  rw [← xiGammaComplex_eq_ofReal]
  unfold xiGammaComplex
  have hn : (n.factorial : ℂ) ≠ 0 := by
    exact_mod_cast Nat.factorial_ne_zero n
  have h2n : ((2 * n).factorial : ℂ) ≠ 0 := by
    exact_mod_cast Nat.factorial_ne_zero (2 * n)
  field_simp [hn, h2n]

theorem xiJensenEntire_taylor (z : ℂ) :
    ∑' n : ℕ, (n.factorial : ℂ)⁻¹ *
        iteratedDeriv n xiJensenEntire 0 * z ^ n =
      xiJensenEntire z := by
  simpa using
    Complex.taylorSeries_eq_of_entire'
      differentiable_xiJensenEntire (c := 0) (z := z)

/-- The centered xi function has the real, even Taylor expansion used in the
Jensen-polynomial literature. -/
theorem xiJensenEntire_hasSum_xiGamma (z : ℂ) :
    HasSum (fun n : ℕ ↦
        ((xiGamma n / n.factorial : ℝ) : ℂ) * z ^ (2 * n))
      (xiJensenEntire z) := by
  let F : ℕ → ℂ := fun n ↦
    (n.factorial : ℂ)⁻¹ *
      iteratedDeriv n xiJensenEntire 0 * z ^ n
  have hfull₀ :=
    Complex.hasSum_taylorSeries_of_entire
      differentiable_xiJensenEntire 0 z
  have hfull : HasSum F (xiJensenEntire z) := by
    apply hfull₀.congr_fun
    intro n
    simp only [F, sub_zero, smul_eq_mul]
    ring
  have heven : Summable (fun n ↦ F (2 * n)) :=
    hfull.summable.comp_injective
      (mul_right_injective₀ (two_ne_zero' ℕ))
  have hodd : HasSum (fun n ↦ F (2 * n + 1)) 0 := by
    convert (hasSum_zero : HasSum (fun _ : ℕ ↦ (0 : ℂ)) 0) using 1
    simp [F, iteratedDeriv_xiJensenEntire_odd_zero]
  have hall :=
    heven.hasSum.even_add_odd hodd
  have hvalue : ∑' n, F (2 * n) = xiJensenEntire z := by
    simpa using (hfull.unique hall).symm
  have hgamma :
      HasSum (fun n : ℕ ↦
          ((xiGamma n / n.factorial : ℝ) : ℂ) * z ^ (2 * n))
        (∑' n, F (2 * n)) := by
    apply heven.hasSum.congr_fun
    intro n
    simpa [F] using (xiJensenEntire_even_taylor_term n z).symm
  rwa [hvalue] at hgamma

/-- Strict coefficient positivity. -/
def StrictlyPositive (a : ℕ → ℝ) : Prop :=
  ∀ n, 0 < a n

/-- The classical positive-kernel route to coefficient positivity, with the
normalization forced by
`xiJensenEntire z = ∑ gamma(n) z^(2n) / n!`.

For the standard even extension of Riemann's Fourier kernel `Φ`, the integral
over `ℝ` is twice the usual integral over `[0, ∞)`.  Pinned Mathlib defines
`Λ₀` by a modified-theta Mellin kernel (`completedRiemannZeta₀_eq_mellin`) but
does not supply this differentiated-kernel identity, so it remains explicit. -/
def XiGammaMomentRepresentation : Prop :=
  ∀ n : ℕ,
    xiGamma n =
      (8 * n.factorial : ℝ) / (2 * n).factorial *
        ∫ t : ℝ, riemannPhi t * t ^ (2 * n)

theorem xiGamma_momentRepresentation :
    XiGammaMomentRepresentation := by
  intro n
  have hcomplex :
      xiGammaComplex n =
        (((8 * n.factorial : ℝ) / (2 * n).factorial *
          ∫ t : ℝ, riemannPhi t * t ^ (2 * n) : ℝ) : ℂ) := by
    rw [xiGammaComplex,
      iteratedDeriv_xiJensenEntire_eq_riemannPhi_moment (2 * n)]
    push_cast
    ring
  rw [xiGamma]
  have hre := congrArg Complex.re hcomplex
  simpa using hre

/-- The actual positive-moment representation would prove strict positivity,
not merely nonvanishing. -/
theorem xiGamma_strictlyPositive_of_momentRepresentation
    (h : XiGammaMomentRepresentation) :
    StrictlyPositive xiGamma := by
  intro n
  have hint := integrable_riemannPhi_mul_pow (2 * n)
  have hformula := h n
  have hnonneg : 0 ≤ fun t : ℝ ↦ riemannPhi t * t ^ (2 * n) := by
    intro t
    exact mul_nonneg (riemannPhi_pos t).le (Even.pow_nonneg (by simp) t)
  have hsupp :
      Set.Ioo (1 : ℝ) 2 ⊆
        Function.support (fun t : ℝ ↦ riemannPhi t * t ^ (2 * n)) := by
    intro t ht
    exact mul_ne_zero (ne_of_gt (riemannPhi_pos t))
      (pow_ne_zero _ (ne_of_gt (lt_trans zero_lt_one ht.1)))
  have hintegral :
      0 < ∫ t : ℝ, riemannPhi t * t ^ (2 * n) := by
    apply (MeasureTheory.integral_pos_iff_support_of_nonneg hnonneg hint).2
    exact ((Measure.measure_Ioo_pos volume).mpr one_lt_two).trans_le
      (measure_mono hsupp)
  rw [hformula]
  exact mul_pos (div_pos (by positivity) (by positivity)) hintegral

/-- The degree-`d`, shift-`n` Jensen polynomial of a real sequence. -/
def jensenPolynomial (a : ℕ → ℝ) (d n : ℕ) : ℝ[X] :=
  ∑ j ∈ Finset.range (d + 1),
    Polynomial.monomial j ((d.choose j : ℝ) * a (n + j))

theorem coeff_jensenPolynomial (a : ℕ → ℝ) (d n j : ℕ) :
    (jensenPolynomial a d n).coeff j =
      if j ≤ d then (d.choose j : ℝ) * a (n + j) else 0 := by
  classical
  rw [jensenPolynomial, Polynomial.finsetSum_coeff]
  simp [Polynomial.coeff_monomial, Finset.sum_ite_eq, eq_comm]

theorem jensenPolynomial_degree_zero (a : ℕ → ℝ) (n : ℕ) :
    jensenPolynomial a 0 n = Polynomial.C (a n) := by
  classical
  norm_num [jensenPolynomial]

theorem jensenPolynomial_degree_one (a : ℕ → ℝ) (n : ℕ) :
    jensenPolynomial a 1 n =
      Polynomial.C (a n) + Polynomial.C (a (n + 1)) * Polynomial.X := by
  classical
  norm_num [jensenPolynomial, Finset.sum_range_succ]
  rw [← Polynomial.C_mul_X_eq_monomial]

theorem jensenPolynomial_degree_two (a : ℕ → ℝ) (n : ℕ) :
    jensenPolynomial a 2 n =
      Polynomial.C (a n) +
        Polynomial.C (2 * a (n + 1)) * Polynomial.X +
        Polynomial.C (a (n + 2)) * Polynomial.X ^ 2 := by
  classical
  norm_num [jensenPolynomial, Finset.sum_range_succ, pow_two]
  simp_rw [← Polynomial.C_mul_X_pow_eq_monomial]
  rw [Polynomial.C_mul]
  ring

/-- The explicit cubic, useful for coefficient-side checks beyond Turán's
quadratic inequality. -/
theorem jensenPolynomial_degree_three (a : ℕ → ℝ) (n : ℕ) :
    jensenPolynomial a 3 n =
      Polynomial.C (a n) +
        Polynomial.C (3 * a (n + 1)) * Polynomial.X +
        Polynomial.C (3 * a (n + 2)) * Polynomial.X ^ 2 +
        Polynomial.C (a (n + 3)) * Polynomial.X ^ 3 := by
  classical
  norm_num [jensenPolynomial, Finset.sum_range_succ, pow_two]
  simp_rw [← Polynomial.C_mul_X_pow_eq_monomial]
  simp [pow_two, Polynomial.C_mul, mul_assoc]

/-- O'Sullivan (2021), equation (3.1), in the shifted Jensen convention:
formal differentiation lowers the degree and raises the shift. -/
theorem derivative_jensenPolynomial (a : ℕ → ℝ) (d n : ℕ) :
    (jensenPolynomial a (d + 1) n).derivative =
      Polynomial.C (d + 1 : ℝ) * jensenPolynomial a d (n + 1) := by
  ext j
  rw [Polynomial.coeff_derivative, coeff_jensenPolynomial]
  simp only [Polynomial.coeff_C_mul, coeff_jensenPolynomial]
  by_cases hj : j ≤ d
  · rw [if_pos hj, if_pos (Nat.succ_le_succ hj)]
    rw [show n + (j + 1) = n + 1 + j by omega]
    have hchoose :
        ((d + 1).choose (j + 1) : ℝ) * (j + 1) =
          (d + 1 : ℝ) * d.choose j := by
      exact_mod_cast (Nat.add_one_mul_choose_eq d j).symm
    calc
      ((d + 1).choose (j + 1) : ℝ) * a (n + 1 + j) * (j + 1) =
          (((d + 1).choose (j + 1) : ℝ) * (j + 1)) * a (n + 1 + j) := by ring
      _ = ((d + 1 : ℝ) * d.choose j) * a (n + 1 + j) := by rw [hchoose]
      _ = (d + 1 : ℝ) * ((d.choose j : ℝ) * a (n + 1 + j)) := by ring
  · rw [if_neg hj, if_neg (by omega)]
    simp

/-- A real polynomial is hyperbolic when all roots of its complex scalar
extension are real. -/
def Hyperbolic (p : ℝ[X]) : Prop :=
  ∀ z : ℂ, (p.map (algebraMap ℝ ℂ)).IsRoot z →
    ∃ x : ℝ, z = (x : ℂ)

theorem hyperbolic_mul {p q : ℝ[X]}
    (hp : Hyperbolic p) (hq : Hyperbolic q) :
    Hyperbolic (p * q) := by
  intro z hz
  simp only [Polynomial.IsRoot.def, Polynomial.map_mul,
    Polynomial.eval_mul] at hz
  exact (mul_eq_zero.mp hz).elim (hp z) (hq z)

theorem hyperbolic_C_add_C_mul_X (a b : ℝ) (hb : b ≠ 0) :
    Hyperbolic (Polynomial.C a + Polynomial.C b * Polynomial.X) := by
  intro z hz
  simp only [Polynomial.IsRoot.def] at hz
  have hz' : (a : ℂ) + (b : ℂ) * z = 0 := by
    simpa using hz
  have hbc : (b : ℂ) ≠ 0 := by exact_mod_cast hb
  have hz_eq : z = -(a : ℂ) / (b : ℂ) := by
    apply (eq_div_iff hbc).2
    linear_combination hz'
  refine ⟨-a / b, hz_eq.trans ?_⟩
  push_cast
  rfl

/-- The standard discriminant of `A X³ + B X² + C X + D`. -/
def cubicDiscriminant (A B C D : ℝ) : ℝ :=
  B ^ 2 * C ^ 2 - 4 * A * C ^ 3 - 4 * B ^ 3 * D -
    27 * A ^ 2 * D ^ 2 + 18 * A * B * C * D

/-- A real cubic with nonzero leading coefficient and nonnegative standard
discriminant is hyperbolic.  If `x + iy` were a nonreal root, direct
elimination gives

`Δ = -4 y² ((B + 3Ax)² + A²y²)² < 0`,

contradicting the discriminant hypothesis. -/
theorem cubic_hyperbolic_of_discriminant_nonnegative
    (A B C D : ℝ) (hA : A ≠ 0)
    (hdisc : 0 ≤ cubicDiscriminant A B C D) :
    Hyperbolic
      (Polynomial.C D + Polynomial.C C * Polynomial.X +
        Polynomial.C B * Polynomial.X ^ 2 +
        Polynomial.C A * Polynomial.X ^ 3) := by
  intro z hz
  simp only [Polynomial.IsRoot, Polynomial.map_add, Polynomial.map_mul,
    Polynomial.map_C, Polynomial.map_X, Polynomial.map_pow,
    Polynomial.eval_add, Polynomial.eval_mul, Polynomial.eval_C,
    Polynomial.eval_X, Polynomial.eval_pow] at hz
  have hre := congrArg Complex.re hz
  have him := congrArg Complex.im hz
  norm_num [Complex.mul_re, Complex.mul_im, pow_succ, pow_two] at hre him
  have hy : z.im = 0 := by
    by_contra hy
    have himfac :
        A * (3 * z.re ^ 2 - z.im ^ 2) + 2 * B * z.re + C = 0 := by
      have hfactor :
          z.im *
            (A * (3 * z.re ^ 2 - z.im ^ 2) + 2 * B * z.re + C) = 0 := by
        nlinarith
      exact (mul_eq_zero.mp hfactor).resolve_left hy
    have hC :
        C = -2 * B * z.re - 3 * A * z.re ^ 2 + A * z.im ^ 2 := by
      nlinarith
    have hD :
        D = B * z.re ^ 2 + B * z.im ^ 2 +
          2 * A * z.re ^ 3 + 2 * A * z.re * z.im ^ 2 := by
      linear_combination hre - z.re * himfac
    let Q : ℝ :=
      (B + 3 * A * z.re) ^ 2 + A ^ 2 * z.im ^ 2
    have hidentity :
        cubicDiscriminant A B C D = -4 * z.im ^ 2 * Q ^ 2 := by
      rw [hC, hD]
      simp only [cubicDiscriminant, Q]
      ring
    have hy2 : 0 < z.im ^ 2 := sq_pos_of_ne_zero hy
    have hAy2 : 0 < A ^ 2 * z.im ^ 2 :=
      mul_pos (sq_pos_of_ne_zero hA) hy2
    have hQ : 0 < Q := by
      dsimp only [Q]
      nlinarith [sq_nonneg (B + 3 * A * z.re)]
    have hneg : cubicDiscriminant A B C D < 0 := by
      rw [hidentity]
      nlinarith [sq_pos_of_pos hQ]
    exact (not_lt_of_ge hdisc hneg).elim
  exact ⟨z.re, Complex.ext (by simp) (by simpa using hy)⟩

/-- The standard cubic discriminant specialized to the degree-three Jensen
polynomial coefficients. -/
def jensenCubicDiscriminant (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  cubicDiscriminant
    (a (n + 3)) (3 * a (n + 2)) (3 * a (n + 1)) (a n)

/-- Nonnegative cubic discriminant proves hyperbolicity of the degree-three
Jensen polynomial in the nondegenerate case. -/
theorem jensenPolynomial_degree_three_hyperbolic
    (a : ℕ → ℝ) (n : ℕ)
    (hc : a (n + 3) ≠ 0)
    (hdisc : 0 ≤ jensenCubicDiscriminant a n) :
    Hyperbolic (jensenPolynomial a 3 n) := by
  rw [jensenPolynomial_degree_three]
  exact cubic_hyperbolic_of_discriminant_nonnegative
    (a (n + 3)) (3 * a (n + 2)) (3 * a (n + 1)) (a n)
    hc hdisc

/-- The all-degree/all-shift statement occurring in the sourced
Pólya--Jensen criterion.  It is an explicit proof obligation, not a theorem. -/
def AllJensenHyperbolic (a : ℕ → ℝ) : Prop :=
  ∀ d n : ℕ, 1 ≤ d → Hyperbolic (jensenPolynomial a d n)

/-- A finite square of Jensen obligations.  The family is nested in `k`, and
each member asks only about positive degrees and shifts at most `k`. -/
def JensenSquare (a : ℕ → ℝ) (k : ℕ) : Prop :=
  ∀ d, 1 ≤ d → d ≤ k → ∀ n ≤ k, Hyperbolic (jensenPolynomial a d n)

/-- All-Jensen hyperbolicity is exactly a nested family of finite square
obligations.  This is an unconditional reduction of the infinite quantifier
shape; it does not claim that any square is decidable computationally. -/
theorem allJensenHyperbolic_iff_jensenSquares (a : ℕ → ℝ) :
    AllJensenHyperbolic a ↔ ∀ k, JensenSquare a k := by
  constructor
  · intro h k d hd _ n _
    exact h d n hd
  · intro h d n hd
    exact h (max d n) d hd (le_max_left _ _) n (le_max_right _ _)

theorem jensenSquare_mono (a : ℕ → ℝ) {k l : ℕ}
    (hkl : k ≤ l) (h : JensenSquare a l) :
    JensenSquare a k := by
  intro d hdpos hd n hn
  exact h d hdpos (hd.trans hkl) n (hn.trans hkl)

/-- Removing a constant polynomial factor cannot introduce a nonreal root. -/
theorem hyperbolic_of_C_mul {c : ℝ} {p : ℝ[X]}
    (h : Hyperbolic (Polynomial.C c * p)) :
    Hyperbolic p := by
  intro z hz
  apply h z
  simp only [Polynomial.map_mul, Polynomial.map_C, IsRoot.def,
    Polynomial.eval_mul, Polynomial.eval_C]
  rw [show (p.map (algebraMap ℝ ℂ)).eval z = 0 from hz]
  simp

/-- The zero polynomial is not hyperbolic under the project's root predicate. -/
theorem zero_not_hyperbolic : ¬ Hyperbolic (0 : ℝ[X]) := by
  intro h
  obtain ⟨x, hx⟩ := h Complex.I (by simp [Polynomial.IsRoot])
  have him := congrArg Complex.im hx
  norm_num at him

/-- Consequently, derivative closure without a nonzero-derivative hypothesis
is formally false: the constant polynomial `1` is a counterexample. -/
theorem not_hyperbolicity_preserved_by_every_derivative :
    ¬ (∀ p : ℝ[X], Hyperbolic p → Hyperbolic p.derivative) := by
  intro h
  have hone : Hyperbolic (1 : ℝ[X]) := by
    intro z hz
    simp [Polynomial.IsRoot] at hz
  exact zero_not_hyperbolic (by simpa using h 1 hone)

/-- The current `Hyperbolic` predicate deliberately excludes the zero
polynomial: every complex number is a root of zero.  Thus derivative closure
requires the derivative to be nonzero (the constant polynomial `1` is the
basic counterexample without this hypothesis).

Pinned Mathlib's Gauss--Lucas theorem gives exactly the required closure for a
nonzero derivative, including repeated roots and arbitrary degree drop. -/
theorem hyperbolic_derivative {p : ℝ[X]} (hp : Hyperbolic p)
    (hp' : p.derivative ≠ 0) :
    Hyperbolic p.derivative := by
  intro z hz
  let P : ℂ[X] := p.map (algebraMap ℝ ℂ)
  have hP' : P.derivative ≠ 0 := by
    change (p.map (algebraMap ℝ ℂ)).derivative ≠ 0
    rw [Polynomial.derivative_map,
      Polynomial.map_ne_zero_iff (FaithfulSMul.algebraMap_injective ℝ ℂ)]
    exact hp'
  have hPdeg : 0 < P.degree := by
    rw [← not_le]
    intro hdeg
    apply hP'
    rw [Polynomial.eq_C_of_degree_le_zero hdeg, Polynomial.derivative_C]
  have hzroot : P.derivative.IsRoot z := by
    simpa only [P, Polynomial.derivative_map] using hz
  have hzset : z ∈ P.derivative.rootSet ℂ := by
    rw [Polynomial.mem_rootSet]
    exact ⟨hP', by simpa [Polynomial.IsRoot, Polynomial.coe_aeval_eq_eval] using hzroot⟩
  have hroots : P.rootSet ℂ ⊆ {w : ℂ | w.im = 0} := by
    intro w hw
    rw [Polynomial.mem_rootSet] at hw
    have hwroot : (p.map (algebraMap ℝ ℂ)).IsRoot w := by
      simpa [P, Polynomial.IsRoot, Polynomial.coe_aeval_eq_eval] using hw.2
    obtain ⟨x, rfl⟩ := hp w hwroot
    simp
  have hreal : Convex ℝ {w : ℂ | w.im = 0} := by
    have h :=
      (convex_halfSpace_im_le 0).inter (convex_halfSpace_im_ge 0)
    convert h using 1
    ext w
    simp [le_antisymm_iff]
  have hzreal : z ∈ {w : ℂ | w.im = 0} :=
    convexHull_min hroots hreal
      (P.rootSet_derivative_subset_convexHull_rootSet hPdeg hzset)
  exact ⟨z.re, Complex.ext (by simp) (by simpa using hzreal)⟩

/-- A simple coefficient hypothesis ensuring that no positive-degree Jensen
polynomial degenerates to the zero polynomial. -/
def PointwiseNonzero (a : ℕ → ℝ) : Prop :=
  ∀ n, a n ≠ 0

/-- The exact nondegeneracy needed by shift propagation. -/
def NonzeroJensenFamily (a : ℕ → ℝ) : Prop :=
  ∀ d n : ℕ, 1 ≤ d → jensenPolynomial a d n ≠ 0

/-- Exact coefficient-side form of Jensen-family nondegeneracy: every
positive-length coefficient window contains a nonzero entry. -/
def JensenWindowNonzero (a : ℕ → ℝ) : Prop :=
  ∀ d n : ℕ, 1 ≤ d → ∃ j ≤ d, a (n + j) ≠ 0

/-- A Jensen polynomial is nonzero exactly when its coefficient window is not
identically zero. -/
theorem nonzeroJensenFamily_iff_windowNonzero (a : ℕ → ℝ) :
    NonzeroJensenFamily a ↔ JensenWindowNonzero a := by
  constructor
  · intro h d n hd
    by_contra hwindow
    push Not at hwindow
    apply h d n hd
    ext j
    rw [coeff_jensenPolynomial]
    by_cases hj : j ≤ d
    · rw [if_pos hj, hwindow j hj, mul_zero, Polynomial.coeff_zero]
    · rw [if_neg hj, Polynomial.coeff_zero]
  · intro h d n hd hzero
    obtain ⟨j, hj, haj⟩ := h d n hd
    have hcoeff := congrArg (fun p : ℝ[X] ↦ p.coeff j) hzero
    rw [coeff_jensenPolynomial, if_pos hj, Polynomial.coeff_zero] at hcoeff
    exact haj (by
      apply (mul_eq_zero.mp hcoeff).resolve_left
      exact_mod_cast (Nat.choose_pos hj).ne')

/-- The exact window condition has a much simpler equivalent form: two
consecutive coefficients may not both vanish.  Thus isolated zero
coefficients do not obstruct Jensen shift propagation. -/
def NoAdjacentZeros (a : ℕ → ℝ) : Prop :=
  ∀ n, a n ≠ 0 ∨ a (n + 1) ≠ 0

theorem jensenWindowNonzero_iff_noAdjacentZeros (a : ℕ → ℝ) :
    JensenWindowNonzero a ↔ NoAdjacentZeros a := by
  constructor
  · intro h n
    obtain ⟨j, hj, haj⟩ := h 1 n (by omega)
    interval_cases j
    · exact Or.inl (by simpa using haj)
    · exact Or.inr (by simpa using haj)
  · intro h d n hd
    rcases h n with hn | hn
    · exact ⟨0, by omega, by simpa using hn⟩
    · exact ⟨1, hd, by simpa using hn⟩

/-- Jensen-family nondegeneracy permits isolated zeros and is equivalent to
the absence of adjacent zeros. -/
theorem nonzeroJensenFamily_iff_noAdjacentZeros (a : ℕ → ℝ) :
    NonzeroJensenFamily a ↔ NoAdjacentZeros a :=
  (nonzeroJensenFamily_iff_windowNonzero a).trans
    (jensenWindowNonzero_iff_noAdjacentZeros a)

theorem jensenPolynomial_ne_zero_of_pointwiseNonzero
    {a : ℕ → ℝ} (ha : PointwiseNonzero a) (d n : ℕ) :
    jensenPolynomial a d n ≠ 0 := by
  intro hzero
  have hcoeff := congrArg (fun p : ℝ[X] ↦ p.coeff d) hzero
  rw [coeff_jensenPolynomial, if_pos le_rfl, Polynomial.coeff_zero] at hcoeff
  simpa using (mul_ne_zero (by simp) (ha (n + d))) hcoeff

/-- Under the exact Jensen nondegeneracy hypothesis, every shift follows from
the unshifted Jensen family. -/
theorem allJensenHyperbolic_iff_unshifted_of_nonzeroJensen
    (a : ℕ → ℝ) (ha : NonzeroJensenFamily a) :
    AllJensenHyperbolic a ↔
      ∀ d : ℕ, 1 ≤ d → Hyperbolic (jensenPolynomial a d 0) := by
  constructor
  · intro h d hd
    exact h d 0 hd
  · intro hzero d n hd
    induction n generalizing d with
    | zero => exact hzero d hd
    | succ n ih =>
        apply hyperbolic_of_C_mul
        rw [← derivative_jensenPolynomial]
        apply hyperbolic_derivative (ih (d + 1) (by omega))
        rw [derivative_jensenPolynomial]
        have hc : Polynomial.C (d + 1 : ℝ) ≠ 0 :=
          Polynomial.C_ne_zero.mpr (by positivity)
        exact mul_ne_zero
          hc
          (ha d (n + 1) hd)

/-- Pointwise coefficient nonvanishing is a convenient sufficient condition
for the exact Jensen nondegeneracy hypothesis. -/
theorem allJensenHyperbolic_iff_unshifted
    (a : ℕ → ℝ) (ha : PointwiseNonzero a) :
    AllJensenHyperbolic a ↔
      ∀ d : ℕ, 1 ≤ d → Hyperbolic (jensenPolynomial a d 0) :=
  allJensenHyperbolic_iff_unshifted_of_nonzeroJensen a fun d n _ ↦
    jensenPolynomial_ne_zero_of_pointwiseNonzero ha d n

/-- The unshifted reduction needs only exact window nondegeneracy, not
pointwise nonvanishing of every coefficient. -/
theorem allJensenHyperbolic_iff_unshifted_of_windowNonzero
    (a : ℕ → ℝ) (ha : JensenWindowNonzero a) :
    AllJensenHyperbolic a ↔
      ∀ d : ℕ, 1 ≤ d → Hyperbolic (jensenPolynomial a d 0) :=
  allJensenHyperbolic_iff_unshifted_of_nonzeroJensen a
    ((nonzeroJensenFamily_iff_windowNonzero a).2 ha)

theorem jensenPolynomial_degree_one_hyperbolic
    (a : ℕ → ℝ) (n : ℕ) (h : a (n + 1) ≠ 0) :
    Hyperbolic (jensenPolynomial a 1 n) := by
  intro z hz
  rw [jensenPolynomial_degree_one] at hz
  simp only [IsRoot.def] at hz
  simp at hz
  refine ⟨-a n / a (n + 1), ?_⟩
  have hc : (a (n + 1) : ℂ) ≠ 0 := Complex.ofReal_ne_zero.mpr h
  have hz' : z = -(a n : ℂ) / (a (n + 1) : ℂ) := by
    apply (eq_div_iff hc).2
    simpa [mul_comm] using eq_neg_of_add_eq_zero_right hz
  rw [hz']
  norm_num

/-- Evaluation form of the degree-two Jensen polynomial. -/
def jensenQuadratic (a : ℕ → ℝ) (n : ℕ) (x : ℝ) : ℝ :=
  a n + 2 * a (n + 1) * x + a (n + 2) * x ^ 2

theorem eval_jensenPolynomial_degree_two (a : ℕ → ℝ) (n : ℕ) (x : ℝ) :
    (jensenPolynomial a 2 n).eval x = jensenQuadratic a n x := by
  rw [jensenPolynomial_degree_two]
  simp [jensenQuadratic]

/-- The Jensen quadratic has the two quadratic-formula roots whenever its
Turán discriminant is nonnegative.  This is supporting finite algebra only. -/
theorem jensenQuadratic_has_two_real_roots
    (a : ℕ → ℝ) (n : ℕ)
    (hc : a (n + 2) ≠ 0)
    (hdisc : 0 ≤ a (n + 1) ^ 2 - a n * a (n + 2)) :
    jensenQuadratic a n
        ((-a (n + 1) + Real.sqrt
          (a (n + 1) ^ 2 - a n * a (n + 2))) / a (n + 2)) = 0 ∧
      jensenQuadratic a n
        ((-a (n + 1) - Real.sqrt
          (a (n + 1) ^ 2 - a n * a (n + 2))) / a (n + 2)) = 0 := by
  have hsqrt :
      (Real.sqrt (a (n + 1) ^ 2 - a n * a (n + 2))) ^ 2 =
        a (n + 1) ^ 2 - a n * a (n + 2) :=
    Real.sq_sqrt hdisc
  constructor <;>
    simp only [jensenQuadratic] <;>
    field_simp <;>
    nlinarith

/-- The precise degree-two condition in the nondegenerate case: existence of
two (not necessarily distinct) real roots is equivalent to the Turán
inequality. -/
theorem jensenQuadratic_has_real_roots_iff
    (a : ℕ → ℝ) (n : ℕ) (hc : a (n + 2) ≠ 0) :
    (∃ r₁ r₂ : ℝ,
        jensenQuadratic a n r₁ = 0 ∧ jensenQuadratic a n r₂ = 0) ↔
      0 ≤ a (n + 1) ^ 2 - a n * a (n + 2) := by
  constructor
  · rintro ⟨r, -, hr, -⟩
    by_contra hneg
    have hlt : a (n + 1) ^ 2 - a n * a (n + 2) < 0 := lt_of_not_ge hneg
    have hsquare : 0 ≤ (a (n + 2) * r + a (n + 1)) ^ 2 := sq_nonneg _
    have hid :
        (a (n + 2) * r + a (n + 1)) ^ 2 =
          a (n + 1) ^ 2 - a n * a (n + 2) := by
      calc
        (a (n + 2) * r + a (n + 1)) ^ 2 =
            a (n + 2) * jensenQuadratic a n r +
              (a (n + 1) ^ 2 - a n * a (n + 2)) := by
                simp only [jensenQuadratic]
                ring
        _ = a (n + 1) ^ 2 - a n * a (n + 2) := by rw [hr]; ring
    rw [hid] at hsquare
    exact (not_lt_of_ge hsquare hlt).elim
  · intro hdisc
    obtain ⟨h₁, h₂⟩ := jensenQuadratic_has_two_real_roots a n hc hdisc
    exact ⟨_, _, h₁, h₂⟩

/-- The nonnegative Turán discriminant makes the degree-two Jensen polynomial
hyperbolic (all of its complex roots are real). -/
theorem jensenPolynomial_degree_two_hyperbolic
    (a : ℕ → ℝ) (n : ℕ)
    (hc : a (n + 2) ≠ 0)
    (hdisc : 0 ≤ a (n + 1) ^ 2 - a n * a (n + 2)) :
    Hyperbolic (jensenPolynomial a 2 n) := by
  intro z hz
  rw [jensenPolynomial_degree_two] at hz
  simp only [IsRoot.def] at hz
  simp at hz
  have hre := congrArg Complex.re hz
  have him := congrArg Complex.im hz
  norm_num [Complex.mul_re, Complex.mul_im, pow_two] at hre him
  have hy : z.im = 0 := by
    by_contra hy
    have hy2 : 0 < z.im ^ 2 := sq_pos_of_ne_zero hy
    have hx : a (n + 2) * z.re + a (n + 1) = 0 := by
      have hfactor :
          2 * z.im * (a (n + 2) * z.re + a (n + 1)) = 0 := by
        nlinarith
      exact (mul_eq_zero.mp hfactor).resolve_left (mul_ne_zero (by norm_num) hy)
    have hxsq : (a (n + 2) * z.re + a (n + 1)) ^ 2 = 0 := by rw [hx]; norm_num
    have hscaled :
        a (n + 2) *
          (a n + 2 * a (n + 1) * z.re +
            a (n + 2) * (z.re * z.re - z.im * z.im)) = 0 := by
      rw [hre, mul_zero]
    nlinarith [hxsq, sq_pos_of_ne_zero hc]
  exact ⟨z.re, Complex.ext (by simp) (by simpa using hy)⟩

/-- The coefficient-side order-two condition: strict positivity supplies the
nondegenerate leading coefficient, while log-concavity is exactly Turán's
inequality. -/
def LogConcave (a : ℕ → ℝ) : Prop :=
  ∀ n, a n * a (n + 2) ≤ a (n + 1) ^ 2

/-- The lower-triangular Toeplitz matrix attached to a one-sided sequence.
This is the convention used for Pólya-frequency sequences. -/
def toeplitzMatrix (a : ℕ → ℝ) : Matrix ℕ ℕ ℝ :=
  fun i j ↦ if j ≤ i then a (i - j) else 0

/-- All finite minors of the one-sided Toeplitz matrix are nonnegative.
Injectivity makes `f` and `g` honest choices of rows and columns. -/
def ToeplitzTotallyNonnegative (a : ℕ → ℝ) : Prop :=
  ∀ k : ℕ, ∀ f g : Fin k → ℕ, Function.Injective f →
    Function.Injective g →
      0 ≤ ((toeplitzMatrix a).submatrix f g).det

/-- Pólya-frequency condition in its all-minor form. -/
abbrev PolyaFrequency := ToeplitzTotallyNonnegative

/-- Order-one Toeplitz minors show that every Pólya-frequency sequence is
coefficientwise nonnegative. -/
theorem toeplitzTotallyNonnegative_nonnegative
    {a : ℕ → ℝ} (h : ToeplitzTotallyNonnegative a) :
    ∀ n, 0 ≤ a n := by
  intro n
  have hn := h 1 (fun _ ↦ n) (fun _ ↦ 0)
    (fun _ _ _ ↦ Subsingleton.elim _ _)
    (fun _ _ _ ↦ Subsingleton.elim _ _)
  simpa [toeplitzMatrix, Matrix.det_fin_one] using hn

/-- Contiguous order-two Toeplitz minors give ordinary log-concavity.
This is only the first nontrivial layer of the all-minor PF condition. -/
theorem toeplitzTotallyNonnegative_logConcave
    {a : ℕ → ℝ} (h : ToeplitzTotallyNonnegative a) :
    LogConcave a := by
  intro n
  let f : Fin 2 → ℕ := fun i ↦ n + 1 + i
  let g : Fin 2 → ℕ := fun i ↦ i
  have hf : Function.Injective f := by
    intro i j hij
    apply Fin.ext
    exact Nat.add_left_cancel hij
  have hg : Function.Injective g := fun _ _ hfg ↦ Fin.ext hfg
  have hminor := h 2 f g hf hg
  simpa [toeplitzMatrix, Matrix.det_fin_two, f, g, pow_two] using hminor

/-- Strict positivity alone does not imply even the first nontrivial
Toeplitz/PF condition. -/
def positiveNonPFSequence (n : ℕ) : ℝ :=
  if n = 2 then 2 else 1

theorem positiveNonPFSequence_strictlyPositive :
    StrictlyPositive positiveNonPFSequence := by
  intro n
  simp only [positiveNonPFSequence]
  split_ifs <;> norm_num

theorem positiveNonPFSequence_not_logConcave :
    ¬ LogConcave positiveNonPFSequence := by
  intro h
  have h0 := h 0
  norm_num [positiveNonPFSequence] at h0

theorem positiveNonPFSequence_not_polyaFrequency :
    ¬ PolyaFrequency positiveNonPFSequence := by
  intro h
  exact positiveNonPFSequence_not_logConcave
    (toeplitzTotallyNonnegative_logConcave h)

/-- Extend the coefficient list of a Jensen polynomial by zeros. -/
def jensenCoefficientSequence (a : ℕ → ℝ) (d n j : ℕ) : ℝ :=
  if j ≤ d then (d.choose j : ℝ) * a (n + j) else 0

/-- A finite coefficient list viewed as a polynomial. -/
def finiteCoefficientPolynomial (b : ℕ → ℝ) (d : ℕ) : ℝ[X] :=
  ∑ j ∈ Finset.range (d + 1), Polynomial.monomial j (b j)

theorem finiteCoefficientPolynomial_jensenCoefficientSequence
    (a : ℕ → ℝ) (d n : ℕ) :
    finiteCoefficientPolynomial (jensenCoefficientSequence a d n) d =
      jensenPolynomial a d n := by
  classical
  rw [finiteCoefficientPolynomial, jensenPolynomial]
  apply Finset.sum_congr rfl
  intro j hj
  rw [jensenCoefficientSequence, if_pos]
  exact Nat.le_of_lt_succ (Finset.mem_range.mp hj)

/-- PF/all-minor condition for one Jensen coefficient row. -/
def JensenPolyaFrequency (a : ℕ → ℝ) (d n : ℕ) : Prop :=
  PolyaFrequency (jensenCoefficientSequence a d n)

def AllJensenPolyaFrequency (a : ℕ → ℝ) : Prop :=
  ∀ d n : ℕ, 1 ≤ d → JensenPolyaFrequency a d n

/-- Exact finite Aissen--Schoenberg--Whitney bridge needed by the PF route.
It is intentionally an explicit proposition: pinned Mathlib has determinants
and polynomial roots, but no theorem identifying finite PF sequences with
polynomials whose roots are all real and nonpositive. -/
def FinitePolyaFrequencyHyperbolicity : Prop :=
  ∀ (b : ℕ → ℝ) (d : ℕ),
    (∀ j, d < j → b j = 0) →
    PolyaFrequency b →
    finiteCoefficientPolynomial b d ≠ 0 →
    Hyperbolic (finiteCoefficientPolynomial b d)

/-- Stronger finite-ASW conclusion: every complex root is represented by a
nonpositive real number. -/
def NonpositiveRooted (p : ℝ[X]) : Prop :=
  ∀ z : ℂ, (p.map (algebraMap ℝ ℂ)).IsRoot z →
    ∃ x : ℝ, x ≤ 0 ∧ z = (x : ℂ)

theorem nonpositiveRooted_hyperbolic {p : ℝ[X]}
    (h : NonpositiveRooted p) :
    Hyperbolic p := by
  intro z hz
  obtain ⟨x, -, hx⟩ := h z hz
  exact ⟨x, hx⟩

theorem nonpositiveRooted_C {c : ℝ} (hc : c ≠ 0) :
    NonpositiveRooted (Polynomial.C c) := by
  intro z hz
  simp [Polynomial.IsRoot, hc] at hz

theorem nonpositiveRooted_X :
    NonpositiveRooted (Polynomial.X : ℝ[X]) := by
  intro z hz
  simp [Polynomial.IsRoot] at hz
  exact ⟨0, le_rfl, hz⟩

theorem nonpositiveRooted_mul {p q : ℝ[X]}
    (hp : NonpositiveRooted p) (hq : NonpositiveRooted q) :
    NonpositiveRooted (p * q) := by
  intro z hz
  simp only [Polynomial.IsRoot, Polynomial.map_mul,
    Polynomial.eval_mul] at hz
  rcases mul_eq_zero.mp hz with hz | hz
  · exact hp z (by simpa [Polynomial.IsRoot] using hz)
  · exact hq z (by simpa [Polynomial.IsRoot] using hz)

theorem nonpositiveRooted_one_add_inv_mul_X
    {r : ℝ} (hr : 0 < r) :
    NonpositiveRooted
      (Polynomial.C 1 + Polynomial.C r⁻¹ * Polynomial.X) := by
  intro z hz
  simp only [Polynomial.IsRoot, Polynomial.map_add, Polynomial.map_mul,
    Polynomial.map_C, Polynomial.map_X, Polynomial.eval_add,
    Polynomial.eval_mul, Polynomial.eval_C, Polynomial.eval_X] at hz
  refine ⟨-r, by linarith, ?_⟩
  norm_num at hz
  have hmul :
      (r : ℂ)⁻¹ * z = -(1 : ℂ) := by
    exact eq_neg_of_add_eq_zero_right hz
  calc
    z = (r : ℂ) * ((r : ℂ)⁻¹ * z) := by
      rw [← mul_assoc]
      norm_num [hr.ne']
    _ = (r : ℂ) * (-(1 : ℂ)) := by rw [hmul]
    _ = ((-r : ℝ) : ℂ) := by push_cast; ring

/-- Exact finite Aissen--Schoenberg--Whitney statement, including internal and
endpoint zero cases through finite support and an explicit nonzero-polynomial
hypothesis.  The reverse direction is total nonnegativity, not merely
coefficient log-concavity. -/
def FiniteAissenSchoenbergWhitney : Prop :=
  ∀ (b : ℕ → ℝ) (d : ℕ),
    (∀ j, d < j → b j = 0) →
    finiteCoefficientPolynomial b d ≠ 0 →
    (PolyaFrequency b ↔
      NonpositiveRooted (finiteCoefficientPolynomial b d))

theorem FiniteAissenSchoenbergWhitney.toHyperbolicity
    (hASW : FiniteAissenSchoenbergWhitney) :
    FinitePolyaFrequencyHyperbolicity := by
  intro b d hsupport hpf hne
  exact nonpositiveRooted_hyperbolic
    ((hASW b d hsupport hne).mp hpf)

theorem finiteCoefficientPolynomial_degree_zero (b : ℕ → ℝ) :
    finiteCoefficientPolynomial b 0 = Polynomial.C (b 0) := by
  classical
  norm_num [finiteCoefficientPolynomial]

theorem finiteCoefficientPolynomial_degree_one (b : ℕ → ℝ) :
    finiteCoefficientPolynomial b 1 =
      Polynomial.C (b 0) + Polynomial.C (b 1) * Polynomial.X := by
  classical
  norm_num [finiteCoefficientPolynomial, Finset.sum_range_succ]
  rw [← Polynomial.C_mul_X_eq_monomial]

theorem finiteCoefficientPolynomial_degree_two (b : ℕ → ℝ) :
    finiteCoefficientPolynomial b 2 =
      Polynomial.C (b 0) + Polynomial.C (b 1) * Polynomial.X +
        Polynomial.C (b 2) * Polynomial.X ^ 2 := by
  classical
  norm_num [finiteCoefficientPolynomial, Finset.sum_range_succ]
  rw [← Polynomial.C_mul_X_eq_monomial]
  rw [Polynomial.C_mul_X_pow_eq_monomial]

theorem finiteCoefficientPolynomial_degree_three (b : ℕ → ℝ) :
    finiteCoefficientPolynomial b 3 =
      Polynomial.C (b 0) + Polynomial.C (b 1) * Polynomial.X +
        Polynomial.C (b 2) * Polynomial.X ^ 2 +
          Polynomial.C (b 3) * Polynomial.X ^ 3 := by
  classical
  norm_num [finiteCoefficientPolynomial, Finset.sum_range_succ]
  rw [← Polynomial.C_mul_X_eq_monomial]
  rw [Polynomial.C_mul_X_pow_eq_monomial]
  rw [Polynomial.C_mul_X_pow_eq_monomial]

/-- The sharp quadratic coefficient condition; the factor `4`, rather than
ordinary log-concavity's factor `1`, is exactly the discriminant threshold. -/
def QuadraticASWCertificate (b : ℕ → ℝ) : Prop :=
  0 < b 0 ∧ 0 ≤ b 1 ∧ 0 < b 2 ∧ 4 * b 0 * b 2 ≤ b 1 ^ 2

/-- The coefficient signs, contiguous order-two minor, and leading
order-three Toeplitz minor commonly checked before the full ASW condition. -/
def SelectedQuadraticToeplitzMinorConditions (b : ℕ → ℝ) : Prop :=
  0 ≤ b 0 ∧ 0 ≤ b 1 ∧ 0 ≤ b 2 ∧
    b 0 * b 2 ≤ b 1 ^ 2 ∧
    0 ≤ b 1 ^ 3 - 2 * b 0 * b 1 * b 2

/-- A finite-support sequence satisfying the selected minors but missing the
sharp factor-four discriminant bound. -/
def selectedMinorCounterexample (n : ℕ) : ℝ :=
  if n = 0 then 1 else if n = 1 then 3 / 2 else if n = 2 then 1 else 0

theorem selectedMinorCounterexample_conditions :
    SelectedQuadraticToeplitzMinorConditions selectedMinorCounterexample := by
  norm_num [SelectedQuadraticToeplitzMinorConditions,
    selectedMinorCounterexample]

theorem selectedMinorCounterexample_not_sharp :
    ¬ 4 * selectedMinorCounterexample 0 * selectedMinorCounterexample 2 ≤
      selectedMinorCounterexample 1 ^ 2 := by
  norm_num [selectedMinorCounterexample]

theorem selected_quadratic_minors_do_not_imply_factor_four :
    ¬ (∀ b : ℕ → ℝ, SelectedQuadraticToeplitzMinorConditions b →
      4 * b 0 * b 2 ≤ b 1 ^ 2) := by
  intro h
  exact selectedMinorCounterexample_not_sharp
    (h selectedMinorCounterexample selectedMinorCounterexample_conditions)

/-- Constant-diagonal tridiagonal matrix whose determinant is the normalized
continuant attached to a quadratic Toeplitz sequence. -/
def toeplitzContinuantMatrix (a p q : ℝ) (k : ℕ) :
    Matrix (Fin k) (Fin k) ℝ :=
  fun i j ↦
    if i = j then a
    else if (i : ℕ) + 1 = j then p
    else if (j : ℕ) + 1 = i then q
    else 0

theorem toeplitzContinuantMatrix_drop_first
    (a p q : ℝ) (k : ℕ) :
    (toeplitzContinuantMatrix a p q (k + 1)).submatrix
        Fin.succ Fin.succ =
      toeplitzContinuantMatrix a p q k := by
  ext i j
  simp [toeplitzContinuantMatrix, Fin.ext_iff]

theorem toeplitzContinuantMatrix_skip_second_drop_first
    (a p q : ℝ) (k : ℕ) :
    (toeplitzContinuantMatrix a p q (k + 2)).submatrix
        (Fin.succ ∘ Fin.succ) (Fin.succAbove 1 ∘ Fin.succ) =
      toeplitzContinuantMatrix a p q k := by
  ext i j
  simp [toeplitzContinuantMatrix, Fin.ext_iff, Function.comp_apply]

theorem toeplitzContinuantMatrix_skip_second_det
    (a p q : ℝ) (k : ℕ) :
    ((toeplitzContinuantMatrix a p q (k + 2)).submatrix
        Fin.succ (Fin.succAbove 1)).det =
      q * (toeplitzContinuantMatrix a p q k).det := by
  rw [Matrix.det_succ_column_zero, Fin.sum_univ_succ]
  simp [toeplitzContinuantMatrix, Fin.ext_iff,
    toeplitzContinuantMatrix_skip_second_drop_first]

theorem toeplitzContinuantMatrix_det_recurrence
    (a p q : ℝ) (k : ℕ) :
    (toeplitzContinuantMatrix a p q (k + 2)).det =
      a * (toeplitzContinuantMatrix a p q (k + 1)).det -
        p * q * (toeplitzContinuantMatrix a p q k).det := by
  rw [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.sum_univ_succ]
  simp [toeplitzContinuantMatrix, toeplitzContinuantMatrix_drop_first,
    toeplitzContinuantMatrix_skip_second_det, Fin.ext_iff]
  ring

/-- The contiguous Toeplitz minor whose determinant is the quadratic
continuant: rows `1, ..., k` and columns `0, ..., k - 1`. -/
def quadraticToeplitzShiftMinor (b : ℕ → ℝ) (k : ℕ) : ℝ :=
  ((toeplitzMatrix b).submatrix
    (fun i : Fin k ↦ (i : ℕ) + 1)
    (fun j : Fin k ↦ (j : ℕ))).det

theorem quadraticToeplitzShiftMinor_eq_continuantMatrix_det
    (b : ℕ → ℝ) (hsupport : ∀ n, 3 ≤ n → b n = 0) (k : ℕ) :
    quadraticToeplitzShiftMinor b k =
      (toeplitzContinuantMatrix (b 1) (b 0) (b 2) k).det := by
  apply congrArg Matrix.det
  ext i j
  simp only [Matrix.submatrix_apply, toeplitzMatrix,
    toeplitzContinuantMatrix]
  by_cases hle : (j : ℕ) ≤ (i : ℕ) + 1
  · rw [if_pos hle]
    by_cases heq : i = j
    · rw [if_pos heq]
      congr 1
      omega
    · rw [if_neg heq]
      by_cases hsup : (i : ℕ) + 1 = (j : ℕ)
      · rw [if_pos hsup]
        congr 1
        omega
      · rw [if_neg hsup]
        by_cases hsub : (j : ℕ) + 1 = (i : ℕ)
        · rw [if_pos hsub]
          congr 1
          omega
        · rw [if_neg hsub]
          exact hsupport _ (by omega)
  · rw [if_neg hle]
    rw [if_neg (by intro h; subst j; omega)]
    rw [if_neg (by omega), if_neg (by omega)]

theorem quadraticToeplitzShiftMinor_nonnegative_of_all_minors
    {b : ℕ → ℝ} (hpf : PolyaFrequency b) (k : ℕ) :
    0 ≤ quadraticToeplitzShiftMinor b k := by
  apply hpf k
  · intro i j hij
    apply Fin.ext
    exact Nat.add_right_cancel hij
  · intro i j hij
    apply Fin.ext
    exact hij

theorem quadraticToeplitzShiftMinor_zero (b : ℕ → ℝ) :
    quadraticToeplitzShiftMinor b 0 = 1 := by
  simp [quadraticToeplitzShiftMinor]

theorem quadraticToeplitzShiftMinor_one (b : ℕ → ℝ) :
    quadraticToeplitzShiftMinor b 1 = b 1 := by
  simp [quadraticToeplitzShiftMinor, toeplitzMatrix]

theorem quadraticToeplitzShiftMinor_two (b : ℕ → ℝ) :
    quadraticToeplitzShiftMinor b 2 = b 1 ^ 2 - b 0 * b 2 := by
  simp [quadraticToeplitzShiftMinor, toeplitzMatrix, Matrix.det_fin_two]
  ring

theorem quadraticToeplitzShiftMinor_three
    (b : ℕ → ℝ) (hsupport : ∀ n, 3 ≤ n → b n = 0) :
    quadraticToeplitzShiftMinor b 3 =
      b 1 ^ 3 - 2 * b 0 * b 1 * b 2 := by
  rw [quadraticToeplitzShiftMinor, Matrix.det_fin_three]
  simp [toeplitzMatrix, hsupport 3 (le_refl 3)]
  ring

/-- Exact determinant principle still needed for degree-two ASW.  Under
quadratic support, nonnegativity of the whole unbounded family of shifted
Toeplitz continuants must force the discriminant's factor `4`.  The previous
counterexample shows that no replacement by the displayed minors of orders
at most three is valid. -/
def QuadraticToeplitzDeterminantPrinciple : Prop :=
  ∀ b : ℕ → ℝ,
    (∀ n, 3 ≤ n → b n = 0) →
    (∀ k, 0 ≤ quadraticToeplitzShiftMinor b k) →
      4 * b 0 * b 2 ≤ b 1 ^ 2

/-- Exact algebraic determinant recurrence needed to turn the unbounded
Toeplitz-minor family into a second-order continuant.  Mathlib currently has
row/column Laplace expansion, but no packaged tridiagonal determinant theorem. -/
def QuadraticToeplitzContinuantRecurrence : Prop :=
  ∀ (b : ℕ → ℝ), (∀ n, 3 ≤ n → b n = 0) → ∀ k,
    quadraticToeplitzShiftMinor b (k + 2) =
      b 1 * quadraticToeplitzShiftMinor b (k + 1) -
        b 0 * b 2 * quadraticToeplitzShiftMinor b k

theorem quadraticToeplitzContinuantRecurrence :
    QuadraticToeplitzContinuantRecurrence := by
  intro b hsupport k
  rw [quadraticToeplitzShiftMinor_eq_continuantMatrix_det b hsupport,
    quadraticToeplitzShiftMinor_eq_continuantMatrix_det b hsupport,
    quadraticToeplitzShiftMinor_eq_continuantMatrix_det b hsupport]
  exact toeplitzContinuantMatrix_det_recurrence (b 1) (b 0) (b 2) k

/-- Exact oscillation lemma needed after the determinant recurrence: a
nonnegative solution of the continuant recurrence cannot remain nonnegative
when its characteristic roots are nonreal. -/
def QuadraticContinuantOscillationPrinciple : Prop :=
  ∀ (a c : ℝ) (D : ℕ → ℝ),
    0 < c →
    D 0 = 1 →
    D 1 = a →
    (∀ k, D (k + 2) = a * D (k + 1) - c * D k) →
    (∀ k, 0 ≤ D k) →
    4 * c ≤ a ^ 2

theorem quadraticContinuantOscillationPrinciple :
    QuadraticContinuantOscillationPrinciple := by
  intro a c D hc hD0 hD1 hrec hnonneg
  have hD2 : D 2 = a ^ 2 - c := by
    rw [show 2 = 0 + 2 by omega, hrec 0, hD0, hD1]
    ring
  have ha0 : 0 ≤ a := by simpa [hD1] using hnonneg 1
  have ha : 0 < a := by
    have : c ≤ a ^ 2 := by
      have := hnonneg 2
      rw [hD2] at this
      linarith
    nlinarith
  have hDpos : ∀ n, 0 < D n := by
    intro n
    induction n using Nat.twoStepInduction with
    | zero => simpa [hD0]
    | one => simpa [hD1]
    | more n hn hn1 =>
        have hn2 := hnonneg (n + 2)
        rcases hn2.eq_or_lt with hz | hp
        · have hn3 := hnonneg (n + 3)
          have hr := hrec (n + 1)
          rw [show n + 1 + 2 = n + 3 by omega,
            show n + 1 + 1 = n + 2 by omega, ← hz] at hr
          nlinarith
        · exact hp
  let r : ℕ → ℝ := fun n ↦ D (n + 1) / D n
  have hrpos (n : ℕ) : 0 < r n :=
    div_pos (hDpos (n + 1)) (hDpos n)
  have hrzero : r 0 = a := by
    dsimp only [r]
    rw [hD0, hD1, div_one]
  have hrrec (n : ℕ) : r (n + 1) = a - c / r n := by
    dsimp only [r]
    rw [hrec n]
    field_simp [ne_of_gt (hDpos n), ne_of_gt (hDpos (n + 1))]
  have hrle (n : ℕ) : r n ≤ a := by
    cases n with
    | zero => rw [hrzero]
    | succ n =>
        rw [hrrec n]
        exact sub_le_self _ (div_nonneg hc.le (hrpos n).le)
  by_contra hsharp
  have hbad : a ^ 2 < 4 * c := lt_of_not_ge hsharp
  let δ : ℝ := c - a ^ 2 / 4
  have hδ : 0 < δ := by
    dsimp only [δ]
    nlinarith
  have hdecrease (n : ℕ) : r (n + 1) ≤ r n - δ / a := by
    have hquad : δ ≤ r n ^ 2 - a * r n + c := by
      dsimp only [δ]
      nlinarith [sq_nonneg (r n - a / 2)]
    have hratio : δ / a ≤
        (r n ^ 2 - a * r n + c) / r n := by
      apply (div_le_div_iff₀ ha (hrpos n)).mpr
      nlinarith [hrle n]
    rw [hrrec n]
    have hid :
        a - c / r n =
          r n - (r n ^ 2 - a * r n + c) / r n := by
      field_simp [ne_of_gt (hrpos n)]
      ring
    rw [hid]
    linarith
  have hlinear (n : ℕ) : r n ≤ a - n * (δ / a) := by
    induction n with
    | zero =>
        simp only [Nat.cast_zero, zero_mul, sub_zero]
        exact hrle 0
    | succ n ih =>
        rw [Nat.cast_add, Nat.cast_one, add_mul]
        have := hdecrease n
        linarith
  obtain ⟨n, hn⟩ := exists_nat_gt (a / (δ / a))
  have hδa : 0 < δ / a := div_pos hδ ha
  have hn' : a < n * (δ / a) := by
    apply (div_lt_iff₀ hδa).mp
    simpa [div_div] using hn
  have := hlinear n
  nlinarith [hrpos n]

theorem quadraticToeplitzDeterminantPrinciple_of_recurrence_oscillation
    (hrec : QuadraticToeplitzContinuantRecurrence)
    (hosc : QuadraticContinuantOscillationPrinciple) :
    QuadraticToeplitzDeterminantPrinciple := by
  intro b hsupport hnonneg
  by_cases hc : 0 < b 0 * b 2
  · have hbound := hosc (b 1) (b 0 * b 2)
      (quadraticToeplitzShiftMinor b) hc
      (quadraticToeplitzShiftMinor_zero b)
      (quadraticToeplitzShiftMinor_one b)
      (hrec b hsupport) hnonneg
    nlinarith
  · have hc' : b 0 * b 2 ≤ 0 := le_of_not_gt hc
    nlinarith [sq_nonneg (b 1)]

theorem quadraticToeplitzDeterminantPrinciple :
    QuadraticToeplitzDeterminantPrinciple :=
  quadraticToeplitzDeterminantPrinciple_of_recurrence_oscillation
    quadraticToeplitzContinuantRecurrence
    quadraticContinuantOscillationPrinciple

/-- Degree-two ASW's root conclusion from its sharp coefficient certificate.
Deriving the factor-four inequality from all Toeplitz minors is the remaining
infinite-minor step; no fixed minor order implies it. -/
theorem finiteASW_degree_two_of_certificate
    (b : ℕ → ℝ) (hcert : QuadraticASWCertificate b) :
    NonpositiveRooted (finiteCoefficientPolynomial b 2) := by
  let a : ℕ → ℝ := fun n ↦
    if n = 0 then b 0 else if n = 1 then b 1 / 2 else b 2
  have ha0 : a 0 = b 0 := by simp [a]
  have ha1 : a 1 = b 1 / 2 := by simp [a]
  have ha2 : a 2 = b 2 := by simp [a]
  have hpoly :
      finiteCoefficientPolynomial b 2 = jensenPolynomial a 2 0 := by
    rw [finiteCoefficientPolynomial_degree_two, jensenPolynomial_degree_two]
    simp only [ha0, ha1, ha2]
    ring_nf
  have hdisc : 0 ≤ a 1 ^ 2 - a 0 * a 2 := by
    rw [ha0, ha1, ha2]
    nlinarith [hcert.2.2.2]
  have hhyp : Hyperbolic (finiteCoefficientPolynomial b 2) := by
    rw [hpoly]
    exact jensenPolynomial_degree_two_hyperbolic a 0
      (by simpa [ha2] using hcert.2.2.1.ne') hdisc
  intro z hz
  obtain ⟨r, hr⟩ := hhyp z hz
  refine ⟨r, ?_, hr⟩
  have heval :
      (b 0 : ℂ) + b 1 * (r : ℂ) + b 2 * (r : ℂ) ^ 2 = 0 := by
    rw [finiteCoefficientPolynomial_degree_two] at hz
    simpa [Polynomial.IsRoot, hr] using hz
  have hre : b 0 + b 1 * r + b 2 * r ^ 2 = 0 := by
    exact_mod_cast heval
  by_contra hrpos
  have hr0 : 0 < r := lt_of_not_ge hrpos
  have ht1 : 0 ≤ b 1 * r := mul_nonneg hcert.2.1 hr0.le
  have ht2 : 0 ≤ b 2 * r ^ 2 :=
    mul_nonneg hcert.2.2.1.le (sq_nonneg r)
  nlinarith [hcert.1]

/-- PF supplies the sign conditions in the degree-two certificate.  The only
additional obligation is the sharp discriminant bound. -/
theorem finiteASW_degree_two_of_sharp_bound
    (b : ℕ → ℝ) (hpf : PolyaFrequency b)
    (hb0 : 0 < b 0) (hb2 : 0 < b 2)
    (hsharp : 4 * b 0 * b 2 ≤ b 1 ^ 2) :
    NonpositiveRooted (finiteCoefficientPolynomial b 2) := by
  apply finiteASW_degree_two_of_certificate b
  exact ⟨hb0, toeplitzTotallyNonnegative_nonnegative hpf 1, hb2, hsharp⟩

/-- Full all-minor degree-two ASW, reduced to the exact unbounded continuant
determinant principle rather than to any insufficient finite list of minors. -/
theorem finiteASW_degree_two_of_all_minors
    (hdet : QuadraticToeplitzDeterminantPrinciple)
    (b : ℕ → ℝ) (hsupport : ∀ n, 3 ≤ n → b n = 0)
    (hpf : PolyaFrequency b) (hb0 : 0 < b 0) (hb2 : 0 < b 2) :
    NonpositiveRooted (finiteCoefficientPolynomial b 2) := by
  apply finiteASW_degree_two_of_sharp_bound b hpf hb0 hb2
  exact hdet b hsupport
    (quadraticToeplitzShiftMinor_nonnegative_of_all_minors hpf)

theorem finiteASW_degree_two
    (b : ℕ → ℝ) (hsupport : ∀ n, 3 ≤ n → b n = 0)
    (hpf : PolyaFrequency b) (hb0 : 0 < b 0) (hb2 : 0 < b 2) :
    NonpositiveRooted (finiteCoefficientPolynomial b 2) :=
  finiteASW_degree_two_of_all_minors
    quadraticToeplitzDeterminantPrinciple b hsupport hpf hb0 hb2

/-- Exact cubic all-minor consequence still missing from pinned Mathlib.
Unlike degree two, one shifted continuant family is not enough: complete
Toeplitz total nonnegativity must force the full cubic discriminant. -/
def CubicToeplitzDiscriminantPrinciple : Prop :=
  ∀ b : ℕ → ℝ,
    (∀ n, 4 ≤ n → b n = 0) →
    PolyaFrequency b →
    0 ≤ cubicDiscriminant (b 3) (b 2) (b 1) (b 0)

theorem finiteASW_degree_three_of_discriminant
    (b : ℕ → ℝ) (hpf : PolyaFrequency b)
    (hb0 : 0 < b 0) (hb3 : 0 < b 3)
    (hdisc : 0 ≤ cubicDiscriminant (b 3) (b 2) (b 1) (b 0)) :
    NonpositiveRooted (finiteCoefficientPolynomial b 3) := by
  have hhyp : Hyperbolic (finiteCoefficientPolynomial b 3) := by
    rw [finiteCoefficientPolynomial_degree_three]
    exact cubic_hyperbolic_of_discriminant_nonnegative
      (b 3) (b 2) (b 1) (b 0) hb3.ne' hdisc
  have hb := toeplitzTotallyNonnegative_nonnegative hpf
  intro z hz
  obtain ⟨r, hr⟩ := hhyp z hz
  refine ⟨r, ?_, hr⟩
  have heval :
      (b 0 : ℂ) + b 1 * (r : ℂ) + b 2 * (r : ℂ) ^ 2 +
        b 3 * (r : ℂ) ^ 3 = 0 := by
    rw [finiteCoefficientPolynomial_degree_three] at hz
    simpa [Polynomial.IsRoot, hr] using hz
  have hre :
      b 0 + b 1 * r + b 2 * r ^ 2 + b 3 * r ^ 3 = 0 := by
    exact_mod_cast heval
  by_contra hrnonpos
  have hrpos : 0 < r := lt_of_not_ge hrnonpos
  have h1 : 0 ≤ b 1 * r := mul_nonneg (hb 1) hrpos.le
  have h2 : 0 ≤ b 2 * r ^ 2 :=
    mul_nonneg (hb 2) (sq_nonneg r)
  have h3 : 0 ≤ b 3 * r ^ 3 :=
    mul_nonneg (hb 3) (by positivity)
  nlinarith

theorem finiteASW_degree_three
    (hcubic : CubicToeplitzDiscriminantPrinciple)
    (b : ℕ → ℝ) (hsupport : ∀ n, 4 ≤ n → b n = 0)
    (hpf : PolyaFrequency b) (hb0 : 0 < b 0) (hb3 : 0 < b 3) :
    NonpositiveRooted (finiteCoefficientPolynomial b 3) :=
  finiteASW_degree_three_of_discriminant b hpf hb0 hb3
    (hcubic b hsupport hpf)

/-- A fully formal finite Aissen--Schoenberg--Whitney theorem in degree one:
all Toeplitz minors force nonnegative coefficients, hence the unique root is
real and nonpositive.  The constant degeneration is handled using the exact
nonzero-polynomial assumption. -/
theorem finiteASW_degree_one
    (b : ℕ → ℝ)
    (hpf : PolyaFrequency b)
    (hne : finiteCoefficientPolynomial b 1 ≠ 0) :
    NonpositiveRooted (finiteCoefficientPolynomial b 1) := by
  have hb := toeplitzTotallyNonnegative_nonnegative hpf
  intro z hz
  by_cases hb1 : b 1 = 0
  · have hb0 : b 0 ≠ 0 := by
      intro hb0
      apply hne
      rw [finiteCoefficientPolynomial_degree_one, hb0, hb1]
      simp
    rw [finiteCoefficientPolynomial_degree_one] at hz
    simp [Polynomial.IsRoot, hb0, hb1] at hz
  · rw [finiteCoefficientPolynomial_degree_one] at hz
    simp only [Polynomial.IsRoot, Polynomial.map_add, Polynomial.map_mul,
      Polynomial.map_C, Polynomial.map_X, Polynomial.eval_add,
      Polynomial.eval_mul, Polynomial.eval_C, Polynomial.eval_X] at hz
    refine ⟨-b 0 / b 1,
      div_nonpos_of_nonpos_of_nonneg (neg_nonpos.mpr (hb 0)) (hb 1), ?_⟩
    have hb1c : (b 1 : ℂ) ≠ 0 := Complex.ofReal_ne_zero.mpr hb1
    have hz' : z = -(b 0 : ℂ) / (b 1 : ℂ) := by
      apply (eq_div_iff hb1c).2
      simpa [mul_comm] using eq_neg_of_add_eq_zero_right hz
    simpa using hz'

/-- The finite ASW hyperbolicity implication is now unconditional through
degree one. -/
theorem finitePolyaFrequencyHyperbolicity_through_one :
    ∀ (b : ℕ → ℝ) (d : ℕ), d ≤ 1 →
      (∀ j, d < j → b j = 0) →
      PolyaFrequency b →
      finiteCoefficientPolynomial b d ≠ 0 →
      Hyperbolic (finiteCoefficientPolynomial b d) := by
  intro b d hd _hsupport hpf hne
  interval_cases d
  · rw [finiteCoefficientPolynomial_degree_zero]
    intro z hz
    simp only [Polynomial.IsRoot, Polynomial.map_C, Polynomial.eval_C] at hz
    have hb0 : b 0 ≠ 0 := by
      simpa [finiteCoefficientPolynomial_degree_zero] using hne
    exact (hb0 (Complex.ofReal_eq_zero.mp hz)).elim
  · exact nonpositiveRooted_hyperbolic (finiteASW_degree_one b hpf hne)

/-- Bounded-degree finite ASW statement.  This makes explicit that increasing
the Jensen degree requires a corresponding increase in the proved PF/root
theorem; no fixed minor order is silently extrapolated. -/
def FinitePolyaFrequencyHyperbolicityThrough (k : ℕ) : Prop :=
  ∀ (b : ℕ → ℝ) (d : ℕ), d ≤ k →
    (∀ j, d < j → b j = 0) →
    PolyaFrequency b →
    finiteCoefficientPolynomial b d ≠ 0 →
    Hyperbolic (finiteCoefficientPolynomial b d)

theorem finitePolyaFrequencyHyperbolicityThrough_one :
    FinitePolyaFrequencyHyperbolicityThrough 1 :=
  finitePolyaFrequencyHyperbolicity_through_one

def JensenHyperbolicThrough (a : ℕ → ℝ) (k : ℕ) : Prop :=
  ∀ d n : ℕ, 1 ≤ d → d ≤ k →
    Hyperbolic (jensenPolynomial a d n)

/-- A bounded ASW theorem plus PF for the corresponding Jensen rows proves
all shifts through that degree bound. -/
theorem jensenHyperbolicThrough_of_polyaFrequency
    {a : ℕ → ℝ} {k : ℕ}
    (hASW : FinitePolyaFrequencyHyperbolicityThrough k)
    (hpf : AllJensenPolyaFrequency a)
    (hne : NonzeroJensenFamily a) :
    JensenHyperbolicThrough a k := by
  intro d n hd hdk
  rw [← finiteCoefficientPolynomial_jensenCoefficientSequence]
  apply hASW (jensenCoefficientSequence a d n) d hdk
  · intro j hj
    simp [jensenCoefficientSequence, Nat.not_le.mpr hj]
  · exact hpf d n hd
  · rw [finiteCoefficientPolynomial_jensenCoefficientSequence]
    exact hne d n hd

/-- Bounded ASW theorems for every degree are exactly enough for the PF route
to recover the full all-Jensen target. -/
theorem allJensenHyperbolic_of_polyaFrequencyThrough
    {a : ℕ → ℝ}
    (hASW : ∀ k, FinitePolyaFrequencyHyperbolicityThrough k)
    (hpf : AllJensenPolyaFrequency a)
    (hne : NonzeroJensenFamily a) :
    AllJensenHyperbolic a := by
  intro d n hd
  exact jensenHyperbolicThrough_of_polyaFrequency
    (hASW d) hpf hne d n hd le_rfl

/-- Assuming precisely the finite ASW theorem, all-minor positivity of every
Jensen coefficient row implies all Jensen hyperbolicity. -/
theorem allJensenHyperbolic_of_polyaFrequency
    {a : ℕ → ℝ}
    (hASW : FinitePolyaFrequencyHyperbolicity)
    (hpf : AllJensenPolyaFrequency a)
    (hne : NonzeroJensenFamily a) :
    AllJensenHyperbolic a := by
  intro d n hd
  rw [← finiteCoefficientPolynomial_jensenCoefficientSequence]
  apply hASW (jensenCoefficientSequence a d n) d
  · intro j hj
    simp [jensenCoefficientSequence, Nat.not_le.mpr hj]
  · exact hpf d n hd
  · rw [finiteCoefficientPolynomial_jensenCoefficientSequence]
    exact hne d n hd

theorem degree_two_hyperbolic_of_positive_logConcave
    (a : ℕ → ℝ) (hpos : StrictlyPositive a) (hlc : LogConcave a) :
    ∀ n, Hyperbolic (jensenPolynomial a 2 n) := by
  intro n
  apply jensenPolynomial_degree_two_hyperbolic
  · exact ne_of_gt (hpos (n + 2))
  · exact sub_nonneg.mpr (hlc n)

/-- A positive log-concave sequence witnessing that all quadratic Jensen
polynomials can be hyperbolic while a cubic Jensen polynomial is not. -/
def turanOnlySequence (n : ℕ) : ℝ :=
  if n = 0 then 7 else 8

theorem turanOnlySequence_strictlyPositive :
    StrictlyPositive turanOnlySequence := by
  intro n
  simp only [turanOnlySequence]
  split_ifs <;> norm_num

theorem turanOnlySequence_logConcave :
    LogConcave turanOnlySequence := by
  intro n
  rcases n with _ | n
  · norm_num [turanOnlySequence]
  · simp [turanOnlySequence]
    norm_num

theorem turanOnlySequence_degree_two_hyperbolic :
    ∀ n, Hyperbolic (jensenPolynomial turanOnlySequence 2 n) :=
  degree_two_hyperbolic_of_positive_logConcave turanOnlySequence
    turanOnlySequence_strictlyPositive turanOnlySequence_logConcave

/-- Ordinary Turán/log-concavity conditions do not propagate to degree
three.  Here `J^{3,0}(X) = 8(X+1)^3 - 1`; a nonreal cube root of unity gives
an explicit nonreal root. -/
theorem turanOnlySequence_degree_three_not_hyperbolic :
    ¬ Hyperbolic (jensenPolynomial turanOnlySequence 3 0) := by
  intro h
  let ω : ℂ := -(1 : ℂ) / 2 + (Real.sqrt 3 : ℂ) / 2 * Complex.I
  let z : ℂ := ω / 2 - 1
  have hsqrt : (Real.sqrt 3) ^ 2 = 3 :=
    Real.sq_sqrt (by norm_num)
  have hsqrt3 : (Real.sqrt 3) ^ 3 = 3 * Real.sqrt 3 := by
    calc
      (Real.sqrt 3) ^ 3 = Real.sqrt 3 * (Real.sqrt 3) ^ 2 := by ring
      _ = Real.sqrt 3 * 3 := by rw [hsqrt]
      _ = 3 * Real.sqrt 3 := by ring
  have hω : ω ^ 3 = 1 := by
    apply Complex.ext
    · norm_num [ω, Complex.mul_re, Complex.mul_im, pow_succ, pow_two]
      nlinarith
    · norm_num [ω, Complex.mul_re, Complex.mul_im, pow_succ, pow_two]
      ring_nf at *
      rw [hsqrt3]
      nlinarith
  have hzroot :
      ((jensenPolynomial turanOnlySequence 3 0).map
        (algebraMap ℝ ℂ)).IsRoot z := by
    rw [jensenPolynomial_degree_three]
    simp only [Polynomial.IsRoot, Polynomial.map_add, Polynomial.map_mul,
      Polynomial.map_C, Polynomial.map_X, Polynomial.map_pow,
      Polynomial.eval_add, Polynomial.eval_mul, Polynomial.eval_C,
      Polynomial.eval_X, Polynomial.eval_pow]
    norm_num [turanOnlySequence]
    calc
      (7 : ℂ) + 24 * z + 24 * z ^ 2 + 8 * z ^ 3 =
          8 * (z + 1) ^ 3 - 1 := by ring
      _ = 8 * (ω / 2) ^ 3 - 1 := by simp [z]
      _ = 0 := by rw [div_pow, hω]; norm_num
  obtain ⟨x, hx⟩ := h z hzroot
  have him := congrArg Complex.im hx
  have hsqrtpos : 0 < Real.sqrt 3 :=
    Real.sqrt_pos.2 (by norm_num)
  norm_num [z, ω, Complex.mul_im] at him

/-- The ordinary exponential generating function to which the classical
Pólya--Jensen theorem applies.  The already proved Taylor identity says
`xiJensenEntire z` is this function evaluated at `z²`; proving that relation
globally in this variable and its analyticity is part of the analytic bridge,
not folded into RH. -/
def xiJensenGeneratingFunction (z : ℂ) : ℂ :=
  ∑' n : ℕ, ((xiGamma n / n.factorial : ℝ) : ℂ) * z ^ n

theorem exists_complex_square_root (w : ℂ) :
    ∃ z : ℂ, z ^ 2 = w := by
  let p : ℂ[X] := Polynomial.X ^ 2 - Polynomial.C w
  have hp : p.degree ≠ 0 := by
    change (Polynomial.X ^ 2 - Polynomial.C w).degree ≠ 0
    rw [Polynomial.degree_sub_eq_left_of_degree_lt]
    · simp
    · exact lt_of_le_of_lt Polynomial.degree_C_le (by norm_num)
  obtain ⟨z, hz⟩ := IsAlgClosed.exists_root p hp
  refine ⟨z, ?_⟩
  have hz' : z ^ 2 - w = 0 := by
    simpa [p, Polynomial.IsRoot] using hz
  exact sub_eq_zero.mp hz'

/-- The ordinary xi generating series converges at every complex argument.
This follows from the already proved entire even Taylor series after choosing
a complex square root; it does not use RH or coefficient positivity. -/
theorem xiJensenGeneratingFunction_hasSum (w : ℂ) :
    HasSum (fun n : ℕ ↦
      ((xiGamma n / n.factorial : ℝ) : ℂ) * w ^ n)
      (xiJensenGeneratingFunction w) := by
  obtain ⟨z, rfl⟩ := exists_complex_square_root w
  have hs := xiJensenEntire_hasSum_xiGamma z
  have hsummable :
      Summable (fun n : ℕ ↦
        ((xiGamma n / n.factorial : ℝ) : ℂ) * (z ^ 2) ^ n) := by
    apply hs.summable.congr
    intro n
    rw [← pow_mul]
  simpa [xiJensenGeneratingFunction] using hsummable.hasSum

/-- Exact relation between the even centered xi function and the ordinary
exponential generating function used by Pólya--Jensen. -/
theorem xiJensenGeneratingFunction_sq (z : ℂ) :
    xiJensenGeneratingFunction (z ^ 2) = xiJensenEntire z := by
  exact (xiJensenGeneratingFunction_hasSum (z ^ 2)).unique
    (by
      convert xiJensenEntire_hasSum_xiGamma z using 1
      ext n
      rw [← pow_mul])

/-- Exact change of variables from the ordinary Jensen generating function
back to the classical completed xi function.  This is the root-identification
map needed in the final LP-to-RH step. -/
theorem xiJensenGeneratingFunction_centered_square (s : ℂ) :
    xiJensenGeneratingFunction ((s - 1 / 2) ^ 2) = 8 * riemannXi s := by
  rw [xiJensenGeneratingFunction_sq, xiJensenEntire]
  congr 2
  ring

/-- Scalar formal power series underlying the ordinary xi generating
function. -/
def xiJensenGeneratingPowerSeries : FormalMultilinearSeries ℂ ℂ ℂ :=
  FormalMultilinearSeries.ofScalars ℂ
    (fun n ↦ ((xiGamma n / n.factorial : ℝ) : ℂ))

theorem xiJensenGeneratingPowerSeries_sum :
    xiJensenGeneratingPowerSeries.sum =
      xiJensenGeneratingFunction := by
  funext z
  change FormalMultilinearSeries.ofScalarsSum
    (fun n ↦ ((xiGamma n / n.factorial : ℝ) : ℂ)) z =
      xiJensenGeneratingFunction z
  rw [FormalMultilinearSeries.ofScalars_sum_eq]
  rfl

/-- The ordinary xi generating power series has infinite convergence radius.
The proof transfers convergence from the even entire xi Taylor series via
complex square roots. -/
theorem xiJensenGeneratingPowerSeries_radius :
    xiJensenGeneratingPowerSeries.radius = ⊤ := by
  apply FormalMultilinearSeries.radius_eq_top_of_summable_norm
  intro r
  have hs := (xiJensenGeneratingFunction_hasSum (r : ℂ)).summable
  rw [← summable_norm_iff] at hs
  simpa [xiJensenGeneratingPowerSeries,
    FormalMultilinearSeries.ofScalars_norm, norm_mul, norm_pow,
    abs_of_nonneg r.2] using hs

/-- The ordinary xi generating function is entire, proved without RH. -/
theorem differentiable_xiJensenGeneratingFunction :
    Differentiable ℂ xiJensenGeneratingFunction := by
  rw [← xiJensenGeneratingPowerSeries_sum]
  have hseries :=
    xiJensenGeneratingPowerSeries.hasFPowerSeriesOnBall
      (by rw [xiJensenGeneratingPowerSeries_radius]; simp)
  have hd := hseries.differentiableOn
  apply differentiableOn_univ.mp
  simpa [xiJensenGeneratingPowerSeries_radius] using hd

/-- The ordinary xi generating function is locally uniformly approximated by
its genuine power-series polynomials.  This closes the analytic convergence
part of the Laguerre--Pólya interface; proving these or alternative
approximants hyperbolic remains the Pólya--Jensen content. -/
theorem xiJensenGeneratingFunction_tendstoLocallyUniformly :
    TendstoLocallyUniformlyOn
      (fun n z ↦ xiJensenGeneratingPowerSeries.partialSum n z)
      xiJensenGeneratingFunction Filter.atTop Set.univ := by
  have hseries :=
    xiJensenGeneratingPowerSeries.hasFPowerSeriesOnBall
      (by rw [xiJensenGeneratingPowerSeries_radius]; simp)
  have hlim := hseries.tendstoLocallyUniformlyOn'
  rw [xiJensenGeneratingPowerSeries_radius] at hlim
  simpa [xiJensenGeneratingPowerSeries_sum] using hlim

/-- Evaluate a real polynomial after scalar extension to `ℂ`. -/
def complexPolynomialEval (p : ℝ[X]) (z : ℂ) : ℂ :=
  (p.map (algebraMap ℝ ℂ)).eval z

/-- A finite power sum converges locally uniformly when each coefficient
converges. This is the coefficient-to-function closure needed by the reverse
Jensen argument. -/
theorem tendstoLocallyUniformlyOn_finset_powerSum
    {b : ℕ → ℕ → ℂ} {a : ℕ → ℂ} (s : Finset ℕ)
    (hcoeff : ∀ j ∈ s, Tendsto (fun N ↦ b N j) atTop (𝓝 (a j))) :
    TendstoLocallyUniformlyOn
      (fun N z ↦ ∑ j ∈ s, b N j * z ^ j)
      (fun z ↦ ∑ j ∈ s, a j * z ^ j)
      atTop Set.univ := by
  classical
  induction s using Finset.induction_on with
  | empty =>
      simp only [Finset.sum_empty]
      intro u hu x hx
      exact ⟨Set.univ, Filter.univ_mem,
        Filter.Eventually.of_forall fun N z hz ↦ refl_mem_uniformity hu⟩
  | @insert j s hj ih =>
      have hjcoeff := hcoeff j (Finset.mem_insert_self j s)
      have hjconst : TendstoLocallyUniformlyOn
          (fun N (_z : ℂ) ↦ b N j) (fun _z : ℂ ↦ a j)
          atTop Set.univ := by
        intro u hu x hx
        exact ⟨Set.univ, Filter.univ_mem,
          (hjcoeff.tendstoUniformlyOn_const Set.univ) u hu⟩
      have hjpow : TendstoLocallyUniformlyOn
          (fun (_N : ℕ) (_z : ℂ) ↦ (_z : ℂ) ^ j)
          (fun z : ℂ ↦ z ^ j)
          atTop Set.univ := by
        intro u hu x hx
        exact ⟨Set.univ, Filter.univ_mem,
          Filter.Eventually.of_forall fun N z hz ↦ refl_mem_uniformity hu⟩
      have hjterm : TendstoLocallyUniformlyOn
          (fun N z ↦ b N j * z ^ j) (fun z ↦ a j * z ^ j)
          atTop Set.univ := by
        convert hjconst.mul₀ hjpow (by fun_prop) (by fun_prop) using 1 <;>
          rfl
      have hrest := ih fun k hk ↦ hcoeff k (Finset.mem_insert_of_mem hk)
      simp only [Finset.sum_insert hj]
      convert hjterm.add hrest using 1 <;> rfl

theorem hyperbolic_comp_scale {p : ℝ[X]} (hp : Hyperbolic p)
    {c : ℝ} (hc : c ≠ 0) :
    Hyperbolic (p.comp (Polynomial.C c * Polynomial.X)) := by
  intro z hz
  have hz' :
      (p.map (algebraMap ℝ ℂ)).IsRoot ((c : ℂ) * z) := by
    simpa [Polynomial.IsRoot, Polynomial.map_comp,
      Polynomial.eval_comp] using hz
  obtain ⟨x, hx⟩ := hp _ hz'
  refine ⟨x / c, ?_⟩
  apply (mul_left_cancel₀ (Complex.ofReal_ne_zero.mpr hc))
  push_cast
  rw [hx]
  rw [div_eq_mul_inv]
  calc
    (x : ℂ) = ((c : ℂ) * (c : ℂ)⁻¹) * (x : ℂ) := by
      rw [mul_inv_cancel₀ (Complex.ofReal_ne_zero.mpr hc), one_mul]
    _ = (c : ℂ) * ((x : ℂ) * (c : ℂ)⁻¹) := by ring

theorem tendsto_choose_mul_inv_pow (j : ℕ) :
    Filter.Tendsto
      (fun d : ℕ ↦
        (((d + 1).choose j : ℝ) * (((d + 1 : ℕ) : ℝ)⁻¹) ^ j))
      Filter.atTop (𝓝 ((j.factorial : ℝ)⁻¹)) := by
  have hbase :
      Filter.Tendsto (fun n : ℕ ↦ (n : ℝ) * (n : ℝ)⁻¹)
        Filter.atTop (𝓝 1) := by
    apply tendsto_const_nhds.congr'
    filter_upwards [Filter.eventually_ge_atTop 1] with n hn
    rw [mul_inv_cancel₀ (by exact_mod_cast (Nat.ne_zero_of_lt hn))]
  have h := ProbabilityTheory.tendsto_choose_mul_pow_atTop j hbase
  have hshift := h.comp (Filter.tendsto_add_atTop_nat 1)
  change Filter.Tendsto
    (fun d : ℕ ↦
      (((d + 1).choose j : ℝ) * (((d + 1 : ℕ) : ℝ)⁻¹) ^ j))
    Filter.atTop (𝓝 (1 ^ j / (j.factorial : ℝ))) at hshift
  simpa [div_eq_mul_inv] using hshift

theorem choose_mul_inv_pow_le_factorial_inv (d j : ℕ) :
    (((d + 1).choose j : ℝ) * (((d + 1 : ℕ) : ℝ)⁻¹) ^ j) ≤
      (j.factorial : ℝ)⁻¹ := by
  have hchoose := Nat.choose_le_pow_div (α := ℝ) j (d + 1)
  have hd : 0 < ((d + 1 : ℕ) : ℝ) := by positivity
  have hinv : 0 ≤ (((d + 1 : ℕ) : ℝ)⁻¹) ^ j := by positivity
  have hmul := mul_le_mul_of_nonneg_right hchoose hinv
  rw [div_eq_mul_inv] at hmul
  calc
    (((d + 1).choose j : ℝ) * (((d + 1 : ℕ) : ℝ)⁻¹) ^ j) ≤
        (((d + 1 : ℕ) : ℝ) ^ j * (j.factorial : ℝ)⁻¹) *
          (((d + 1 : ℕ) : ℝ)⁻¹) ^ j := hmul
    _ = (j.factorial : ℝ)⁻¹ := by
      calc
        ((d + 1 : ℕ) : ℝ) ^ j * (j.factorial : ℝ)⁻¹ *
            (((d + 1 : ℕ) : ℝ)⁻¹) ^ j =
            (j.factorial : ℝ)⁻¹ * (((d + 1 : ℕ) : ℝ) ^ j *
              (((d + 1 : ℕ) : ℝ)⁻¹) ^ j) := by ring
        _ = (j.factorial : ℝ)⁻¹ := by
          rw [← mul_pow, mul_inv_cancel₀ hd.ne', one_pow, mul_one]

/-- Rescaled Jensen polynomials used in the classical Pólya--Jensen
approximation. -/
def xiRescaledJensenPolynomial (d : ℕ) : ℝ[X] :=
  (jensenPolynomial xiGamma (d + 1) 0).comp
    (Polynomial.C (((d + 1 : ℕ) : ℝ)⁻¹) * Polynomial.X)

theorem complexPolynomialEval_xiRescaledJensenPolynomial
    (d : ℕ) (z : ℂ) :
    complexPolynomialEval (xiRescaledJensenPolynomial d) z =
      ∑ j ∈ Finset.range (d + 2),
        (((((d + 1).choose j : ℝ) *
          (((d + 1 : ℕ) : ℝ)⁻¹) ^ j * xiGamma j : ℝ) : ℂ) * z ^ j) := by
  classical
  simp only [xiRescaledJensenPolynomial, complexPolynomialEval,
    Polynomial.map_comp, Polynomial.eval_comp, Polynomial.map_mul,
    Polynomial.map_C, Polynomial.map_X, Polynomial.eval_mul,
    Polynomial.eval_C, Polynomial.eval_X, jensenPolynomial,
    Polynomial.map_sum, Polynomial.map_monomial, map_mul,
    Polynomial.eval_finset_sum, Polynomial.eval_monomial, zero_add]
  apply Finset.sum_congr rfl
  intro j hj
  push_cast
  rw [show algebraMap ℝ ℂ (xiGamma j) = (xiGamma j : ℂ) by rfl]
  ring

theorem xiRescaledJensenPolynomial_tendsto_pointwise (z : ℂ) :
    Filter.Tendsto
      (fun d ↦ complexPolynomialEval (xiRescaledJensenPolynomial d) z)
      Filter.atTop (𝓝 (xiJensenGeneratingFunction z)) := by
  let f : ℕ → ℕ → ℂ := fun d j ↦
    ((((d + 1).choose j : ℝ) * (((d + 1 : ℕ) : ℝ)⁻¹) ^ j : ℝ) : ℂ) *
      (xiGamma j : ℂ) * z ^ j
  let g : ℕ → ℂ := fun j ↦
    ((((j.factorial : ℝ)⁻¹ : ℝ) : ℂ) * (xiGamma j : ℂ) * z ^ j)
  let bound : ℕ → ℝ := fun j ↦
    (xiGamma j / j.factorial) * ‖z‖ ^ j
  have hpos : StrictlyPositive xiGamma :=
    xiGamma_strictlyPositive_of_momentRepresentation xiGamma_momentRepresentation
  have hsum : Summable bound := by
    have hs := (xiJensenGeneratingFunction_hasSum z).summable.norm
    convert hs using 1
    funext j
    simp [bound, norm_mul, norm_pow, Real.norm_eq_abs,
      abs_of_pos (hpos j), abs_of_pos (by positivity : 0 < (j.factorial : ℝ))]
  have hterm (j : ℕ) :
      Filter.Tendsto (fun d ↦ f d j) Filter.atTop (𝓝 (g j)) := by
    have hr := tendsto_choose_mul_inv_pow j
    have hc :
        Filter.Tendsto
          (fun d : ℕ ↦
            ((((d + 1).choose j : ℝ) *
              (((d + 1 : ℕ) : ℝ)⁻¹) ^ j : ℝ) : ℂ))
          Filter.atTop (𝓝 (((j.factorial : ℝ)⁻¹ : ℝ) : ℂ)) :=
      Complex.continuous_ofReal.continuousAt.tendsto.comp hr
    convert (hc.mul_const ((xiGamma j : ℝ) : ℂ)).mul_const (z ^ j) using 1 <;>
      simp only [f, g]
  have hbound : ∀ᶠ d in Filter.atTop, ∀ j, ‖f d j‖ ≤ bound j := by
    filter_upwards with d
    intro j
    have hscale := choose_mul_inv_pow_le_factorial_inv d j
    have hgamma : 0 ≤ xiGamma j := (hpos j).le
    have hz : 0 ≤ ‖z‖ ^ j := by positivity
    have hscale0 : 0 ≤
        ((d + 1).choose j : ℝ) * (((d + 1 : ℕ) : ℝ)⁻¹) ^ j := by
      positivity
    have h :=
      mul_le_mul_of_nonneg_right
        (mul_le_mul_of_nonneg_right hscale hgamma) hz
    have hnorm : ‖f d j‖ =
        (((d + 1).choose j : ℝ) * (((d + 1 : ℕ) : ℝ)⁻¹) ^ j) *
          xiGamma j * ‖z‖ ^ j := by
      dsimp only [f]
      rw [norm_mul, norm_mul, norm_pow]
      rw [Complex.norm_real, Real.norm_of_nonneg hscale0]
      rw [Complex.norm_real, Real.norm_of_nonneg hgamma]
    rw [hnorm]
    dsimp only [bound]
    calc
      _ ≤ (j.factorial : ℝ)⁻¹ * xiGamma j * ‖z‖ ^ j := h
      _ = xiGamma j / (j.factorial : ℝ) * ‖z‖ ^ j := by
        rw [div_eq_mul_inv]
        ring
  have ht := tendsto_tsum_of_dominated_convergence hsum hterm hbound
  have hf (d : ℕ) : ∑' j, f d j =
      complexPolynomialEval (xiRescaledJensenPolynomial d) z := by
    calc
      ∑' j, f d j = ∑ j ∈ Finset.range (d + 2), f d j := by
        apply tsum_eq_sum
        intro j hj
        have hj' : d + 1 < j := by
          simp only [Finset.mem_range, not_lt] at hj
          omega
        dsimp only [f]
        rw [Nat.choose_eq_zero_of_lt hj']
        norm_num
      _ = complexPolynomialEval (xiRescaledJensenPolynomial d) z := by
        rw [complexPolynomialEval_xiRescaledJensenPolynomial]
        apply Finset.sum_congr rfl
        intro j hj
        simp only [f]
        push_cast
        ring
  have hg : ∑' j, g j = xiJensenGeneratingFunction z := by
    rw [← (xiJensenGeneratingFunction_hasSum z).tsum_eq]
    apply tsum_congr
    intro j
    simp only [g, div_eq_mul_inv]
    push_cast
    ring
  simpa only [hf, hg] using ht

/-- Locally uniform convergence of the explicit rescaled Jensen sequence. -/
def XiJensenScalingConvergence : Prop :=
  TendstoLocallyUniformlyOn
    (fun d z ↦ complexPolynomialEval (xiRescaledJensenPolynomial d) z)
    xiJensenGeneratingFunction Filter.atTop Set.univ

theorem xiJensenScalingConvergence :
    XiJensenScalingConvergence := by
  rw [XiJensenScalingConvergence,
    tendstoLocallyUniformlyOn_iff_forall_isCompact isOpen_univ]
  intro K hKuniv hK
  letI : CompactSpace K := isCompact_iff_compactSpace.mp hK
  let Z : C(K, ℂ) :=
    ⟨fun z ↦ z.1, continuous_subtype_val⟩
  let F : ℕ → ℕ → C(K, ℂ) := fun d j ↦
    (ContinuousMap.const K
      ((((d + 1).choose j : ℝ) * (((d + 1 : ℕ) : ℝ)⁻¹) ^ j : ℝ) : ℂ)) *
      (ContinuousMap.const K (xiGamma j : ℂ)) * Z ^ j
  let G : ℕ → C(K, ℂ) := fun j ↦
    (ContinuousMap.const K ((((j.factorial : ℝ)⁻¹ : ℝ) : ℂ))) *
      (ContinuousMap.const K (xiGamma j : ℂ)) * Z ^ j
  let bound : ℕ → ℝ := fun j ↦
    (xiGamma j / j.factorial) * ‖Z‖ ^ j
  have hpos : StrictlyPositive xiGamma :=
    xiGamma_strictlyPositive_of_momentRepresentation xiGamma_momentRepresentation
  have hsum : Summable bound := by
    have hs :=
      (xiJensenGeneratingFunction_hasSum ((‖Z‖ : ℝ) : ℂ)).summable.norm
    convert hs using 1
    funext j
    simp [bound, norm_mul, norm_pow, Real.norm_eq_abs,
      abs_of_pos (hpos j), abs_of_nonneg (norm_nonneg Z)]
  have hconst {c : ℕ → ℂ} {a : ℂ}
      (hc : Filter.Tendsto c Filter.atTop (𝓝 a)) :
      Filter.Tendsto (fun d ↦ ContinuousMap.const K (c d))
        Filter.atTop (𝓝 (ContinuousMap.const K a)) := by
    rw [Metric.tendsto_nhds] at hc ⊢
    intro ε hε
    filter_upwards [hc ε hε] with d hd
    apply lt_of_le_of_lt
      ((ContinuousMap.dist_le dist_nonneg).2 fun x ↦ ?_) hd
    simp
  have hterm (j : ℕ) :
      Filter.Tendsto (fun d ↦ F d j) Filter.atTop (𝓝 (G j)) := by
    have hr := tendsto_choose_mul_inv_pow j
    have hc :
        Filter.Tendsto
          (fun d : ℕ ↦
            ((((d + 1).choose j : ℝ) *
              (((d + 1 : ℕ) : ℝ)⁻¹) ^ j : ℝ) : ℂ))
          Filter.atTop (𝓝 (((j.factorial : ℝ)⁻¹ : ℝ) : ℂ)) :=
      Complex.continuous_ofReal.continuousAt.tendsto.comp hr
    simpa only [F, G] using
      ((hconst hc).mul_const (ContinuousMap.const K (xiGamma j : ℂ))).mul_const
        (Z ^ j)
  have hbound (d j : ℕ) : ‖F d j‖ ≤ bound j := by
    have hgamma : 0 ≤ xiGamma j := (hpos j).le
    apply (ContinuousMap.norm_le _
      (mul_nonneg (div_nonneg hgamma (by positivity)) (by positivity))).2
    intro z
    have hscale := choose_mul_inv_pow_le_factorial_inv d j
    have hzpow :
        ‖(Z z)‖ ^ j ≤ ‖Z‖ ^ j :=
      pow_le_pow_left₀ (norm_nonneg _) (Z.norm_coe_le_norm z) j
    have hscale0 : 0 ≤
        ((d + 1).choose j : ℝ) * (((d + 1 : ℕ) : ℝ)⁻¹) ^ j := by
      positivity
    have hcoeff :
        (((d + 1).choose j : ℝ) * (((d + 1 : ℕ) : ℝ)⁻¹) ^ j) *
            xiGamma j ≤
          (j.factorial : ℝ)⁻¹ * xiGamma j :=
      mul_le_mul_of_nonneg_right hscale hgamma
    have h :=
      (mul_le_mul_of_nonneg_right hcoeff (by positivity : 0 ≤ ‖Z z‖ ^ j)).trans
        (mul_le_mul_of_nonneg_left hzpow
          (mul_nonneg (by positivity) hgamma))
    dsimp only [F, bound]
    simp only [ContinuousMap.mul_apply, ContinuousMap.const_apply,
      ContinuousMap.pow_apply]
    rw [norm_mul, norm_mul, norm_pow]
    rw [Complex.norm_real, Real.norm_of_nonneg hscale0]
    rw [Complex.norm_real, Real.norm_of_nonneg hgamma]
    calc
      _ ≤ (j.factorial : ℝ)⁻¹ * xiGamma j * ‖Z‖ ^ j := h
      _ = xiGamma j / (j.factorial : ℝ) * ‖Z‖ ^ j := by
        rw [div_eq_mul_inv]
        ring
  have ht := tendsto_tsum_of_dominated_convergence hsum hterm
    (Filter.Eventually.of_forall hbound)
  have hFsummable (d : ℕ) : Summable (F d) :=
    hsum.of_norm_bounded (hbound d)
  have hGsummable : Summable G :=
    hsum.of_norm_bounded fun j ↦ by
      have hgamma : 0 ≤ xiGamma j := (hpos j).le
      apply (ContinuousMap.norm_le _
        (mul_nonneg (div_nonneg hgamma (by positivity)) (by positivity))).2
      intro z
      have hzpow : ‖Z z‖ ^ j ≤ ‖Z‖ ^ j :=
        pow_le_pow_left₀ (norm_nonneg _) (Z.norm_coe_le_norm z) j
      have hnonneg : 0 ≤ (j.factorial : ℝ)⁻¹ * xiGamma j := by positivity
      dsimp only [G, bound]
      simp only [ContinuousMap.mul_apply, ContinuousMap.const_apply,
        ContinuousMap.pow_apply]
      rw [norm_mul, norm_mul, norm_pow]
      rw [Complex.norm_real, Real.norm_of_nonneg (by positivity :
        0 ≤ (j.factorial : ℝ)⁻¹)]
      rw [Complex.norm_real, Real.norm_of_nonneg hgamma]
      calc
        _ ≤ (j.factorial : ℝ)⁻¹ * xiGamma j * ‖Z‖ ^ j :=
          mul_le_mul_of_nonneg_left hzpow hnonneg
        _ = xiGamma j / (j.factorial : ℝ) * ‖Z‖ ^ j := by
          rw [div_eq_mul_inv]
          ring
  have hF (d : ℕ) :
      ∑' j, F d j =
        ⟨fun z : K ↦
          complexPolynomialEval (xiRescaledJensenPolynomial d) z.1,
          by
            unfold complexPolynomialEval
            fun_prop⟩ := by
    ext z
    rw [← ContinuousMap.tsum_apply (hFsummable d) z]
    change ∑' i, (F d i) z =
      complexPolynomialEval (xiRescaledJensenPolynomial d) z.1
    rw [complexPolynomialEval_xiRescaledJensenPolynomial]
    apply (tsum_eq_sum (s := Finset.range (d + 2)) ?_).trans
      (Finset.sum_congr rfl fun j hj ↦ ?_)
    · intro j hj
      have hj' : d + 1 < j := by
        simp only [Finset.mem_range, not_lt] at hj
        omega
      simp [F, Nat.choose_eq_zero_of_lt hj']
    · simp only [F, ContinuousMap.mul_apply, ContinuousMap.const_apply,
        ContinuousMap.pow_apply]
      rw [show Z z = z.1 by rfl]
      push_cast
      ring
  have hG :
      ∑' j, G j =
        ⟨fun z : K ↦ xiJensenGeneratingFunction z.1,
          differentiable_xiJensenGeneratingFunction.continuous.comp
            continuous_subtype_val⟩ := by
    ext z
    rw [← ContinuousMap.tsum_apply hGsummable z]
    change ∑' i, (G i) z = xiJensenGeneratingFunction z.1
    rw [← (xiJensenGeneratingFunction_hasSum z.1).tsum_eq]
    apply tsum_congr
    intro j
    simp only [G, ContinuousMap.mul_apply, ContinuousMap.const_apply,
      ContinuousMap.pow_apply, div_eq_mul_inv]
    rw [show Z z = z.1 by rfl]
    push_cast
    ring
  rw [Metric.tendstoUniformlyOn_iff]
  intro ε hε
  have ht' := (Metric.tendsto_nhds.1 ht) ε hε
  filter_upwards [ht'] with d hd
  intro z hz
  rw [hF, hG] at hd
  simpa only [ContinuousMap.coe_mk, dist_comm] using
    (ContinuousMap.dist_apply_le_dist (⟨z, hz⟩ : ↥K)).trans_lt hd

theorem xiRescaledJensenPolynomial_hyperbolic
    (hJ : AllJensenHyperbolic xiGamma) (d : ℕ) :
    Hyperbolic (xiRescaledJensenPolynomial d) := by
  apply hyperbolic_comp_scale (hJ (d + 1) 0 (by omega))
  positivity

theorem complexPolynomialEval_conj (p : ℝ[X]) (z : ℂ) :
    complexPolynomialEval p (conj z) = conj (complexPolynomialEval p z) := by
  induction p using Polynomial.induction_on' with
  | add p q hp hq =>
      calc
        complexPolynomialEval (p + q) (conj z) =
            complexPolynomialEval p (conj z) +
              complexPolynomialEval q (conj z) := by
          simp [complexPolynomialEval]
        _ = conj (complexPolynomialEval p z) +
              conj (complexPolynomialEval q z) := by rw [hp, hq]
        _ = conj (complexPolynomialEval p z +
              complexPolynomialEval q z) := (map_add conj _ _).symm
        _ = conj (complexPolynomialEval (p + q) z) := by
          simp [complexPolynomialEval]
  | monomial n a =>
      simp [complexPolynomialEval, map_monomial]

/-- A concrete Laguerre--Pólya-style interface: local uniform convergence on
the entire complex plane of real hyperbolic polynomials. -/
def LocallyUniformHyperbolicLimit (f : ℂ → ℂ) : Prop :=
  ∃ p : ℕ → ℝ[X],
    (∀ n, Hyperbolic (p n)) ∧
    TendstoLocallyUniformlyOn
      (fun n z ↦ complexPolynomialEval (p n) z) f Filter.atTop Set.univ

/-- The locally-uniform-limit interface really does force an entire limit;
this is Mathlib's Weierstrass theorem, not an extra bridge assumption. -/
theorem differentiable_of_locallyUniformHyperbolicLimit
    {f : ℂ → ℂ} (h : LocallyUniformHyperbolicLimit f) :
    Differentiable ℂ f := by
  obtain ⟨p, -, hlim⟩ := h
  rw [← differentiableOn_univ]
  apply hlim.differentiableOn
  · exact Filter.Eventually.of_forall fun n ↦
      ((p n).map (algebraMap ℝ ℂ)).differentiable.differentiableOn
  · exact isOpen_univ

/-- A locally uniform limit of real polynomials retains conjugation symmetry.
This is a genuine closure consequence independent of root preservation. -/
theorem conj_symmetry_of_locallyUniformHyperbolicLimit
    {f : ℂ → ℂ} (h : LocallyUniformHyperbolicLimit f) (z : ℂ) :
    f (conj z) = conj (f z) := by
  obtain ⟨p, -, hlim⟩ := h
  have hz := hlim.tendsto_at (Set.mem_univ z)
  have hcz := hlim.tendsto_at (Set.mem_univ (conj z))
  have hconj :
      Filter.Tendsto (fun n ↦ conj (complexPolynomialEval (p n) z))
        Filter.atTop (nhds (conj (f z))) :=
    (continuous_conj.tendsto (f z)).comp hz
  apply tendsto_nhds_unique hcz
  convert hconj using 1
  funext n
  exact complexPolynomialEval_conj (p n) z

/-- A quantitative project-local Rouché/Hurwitz lemma proved from Mathlib's
Jensen circle-average theorem.  If `f` has a zero at the center and is bounded
away from zero on the boundary circle, then an analytic function uniformly
closer than half that boundary gap must have a zero in the closed disk. -/
theorem exists_zero_closedBall_of_uniform_close
    {f g : ℂ → ℂ} {c : ℂ} {r m : ℝ}
    (hr : 0 < r) (hm : 0 < m)
    (hfc : f c = 0)
    (hboundary : ∀ z ∈ Metric.sphere c r, m ≤ ‖f z‖)
    (hg : AnalyticOnNhd ℂ g (Metric.closedBall c r))
    (hclose : ∀ z ∈ Metric.closedBall c r, ‖g z - f z‖ < m / 2) :
    ∃ z ∈ Metric.closedBall c r, g z = 0 := by
  by_contra! hzero
  have hcircle :
      CircleIntegrable (fun z ↦ Real.log ‖g z‖) c r :=
    by
      have hg' : MeromorphicOn g (Metric.sphere c |r|) := by
        simpa [abs_of_pos hr] using
          (hg.mono Metric.sphere_subset_closedBall).meromorphicOn
      exact hg'.circleIntegrable_log_norm
  have hgap : ∀ z ∈ Metric.sphere c r, m / 2 < ‖g z‖ := by
    intro z hz
    have htri := norm_le_norm_add_norm_sub' (f z) (g z)
    have hc := hclose z (Metric.sphere_subset_closedBall hz)
    rw [norm_sub_rev] at hc
    nlinarith [hboundary z hz]
  have havg :
      Real.log (m / 2) ≤
        Real.circleAverage (fun z ↦ Real.log ‖g z‖) c r := by
    rw [← Real.circleAverage_const (Real.log (m / 2)) c r]
    apply Real.circleAverage_mono (circleIntegrable_const _ _ _) hcircle
    intro z hz
    exact Real.log_le_log (half_pos hm)
      (hgap z (by simpa [abs_of_pos hr] using hz)).le
  have hgabs : AnalyticOnNhd ℂ g (Metric.closedBall c |r|) := by
    simpa [abs_of_pos hr] using hg
  have hzeroabs : ∀ z ∈ Metric.closedBall c |r|, g z ≠ 0 := by
    intro z hz
    exact hzero z (by simpa [abs_of_pos hr] using hz)
  rw [hgabs.circleAverage_log_norm_of_ne_zero hzeroabs] at havg
  have hgc : g c ≠ 0 := hzero c (Metric.mem_closedBall_self hr.le)
  have hnorm_lower : m / 2 ≤ ‖g c‖ :=
    (Real.log_le_log_iff (half_pos hm) (norm_pos_iff.mpr hgc)).mp havg
  have hcenter := hclose c (Metric.mem_closedBall_self hr.le)
  rw [hfc, sub_zero] at hcenter
  exact (not_lt_of_ge hnorm_lower) hcenter

/-- Project-local Hurwitz theorem on an arbitrary open preconnected domain.
It is proved from the quantitative disk lemma above, compact-uniform
convergence, and Mathlib's isolated-zero theorem. -/
theorem hurwitz_nonvanishing_or_eq_zero_on
    {F : ℕ → ℂ → ℂ} {f : ℂ → ℂ} {U : Set ℂ}
    (hUopen : IsOpen U) (hUpre : IsPreconnected U)
    (hF : ∀ n, Differentiable ℂ (F n))
    (hlim : TendstoLocallyUniformlyOn F f Filter.atTop U)
    (hnonzero : ∀ n z, z ∈ U → F n z ≠ 0) :
    (∀ z, z ∈ U → f z ≠ 0) ∨ Set.EqOn f 0 U := by
  by_cases hn : ∀ z, z ∈ U → f z ≠ 0
  · exact Or.inl hn
  right
  push Not at hn
  obtain ⟨c, hcU, hfc⟩ := hn
  have hfdiff : DifferentiableOn ℂ f U :=
    hlim.differentiableOn
      (Filter.Eventually.of_forall fun n ↦ (hF n).differentiableOn) hUopen
  have hfan : AnalyticOnNhd ℂ f U := hfdiff.analyticOnNhd hUopen
  rcases (hfan c hcU).eventually_eq_zero_or_eventually_ne_zero with hlocal | hiso
  · exact hfan.eqOn_zero_of_preconnected_of_eventuallyEq_zero
      hUpre hcU hlocal
  exfalso
  rcases Metric.mem_nhdsWithin_iff.mp hiso with ⟨ε, hε, hεsub⟩
  rcases (Metric.isOpen_iff.mp hUopen c hcU) with ⟨δ, hδ, hδsub⟩
  let r := min ε δ / 2
  have hr : 0 < r := by
    dsimp only [r]
    positivity
  have hrε : r < ε := by
    dsimp only [r]
    nlinarith [min_le_left ε δ]
  have hrδ : r < δ := by
    dsimp only [r]
    nlinarith [min_le_right ε δ]
  have hclosedU : Metric.closedBall c r ⊆ U := by
    intro z hz
    apply hδsub
    rw [Metric.mem_ball]
    exact hz.trans_lt hrδ
  have hsphere_ne : ∀ z ∈ Metric.sphere c r, f z ≠ 0 := by
    intro z hz
    apply hεsub
    constructor
    · rw [Metric.mem_ball, Metric.mem_sphere.mp hz]
      exact hrε
    · intro hzc
      subst z
      have hzero_r : (0 : ℝ) = r := by
        simpa [Metric.mem_sphere] using hz
      exact hr.ne' hzero_r.symm
  have hsphere : (Metric.sphere c r).Nonempty :=
    NormedSpace.sphere_nonempty.mpr hr.le
  have hcont :
      ContinuousOn (fun z ↦ ‖f z‖) (Metric.sphere c r) :=
    (hfdiff.continuousOn.mono
      (Metric.sphere_subset_closedBall.trans hclosedU)).norm
  obtain ⟨w, hw, hwmin⟩ :=
    (isCompact_sphere c r).exists_isMinOn hsphere hcont
  let m := ‖f w‖
  have hm : 0 < m := by
    dsimp only [m]
    exact norm_pos_iff.mpr (hsphere_ne w hw)
  have hboundary : ∀ z ∈ Metric.sphere c r, m ≤ ‖f z‖ :=
    fun _ hz ↦ hwmin hz
  have hcompact : IsCompact (Metric.closedBall c r) :=
    isCompact_closedBall c r
  have hunif : TendstoUniformlyOn F f Filter.atTop (Metric.closedBall c r) :=
    (tendstoLocallyUniformlyOn_iff_forall_isCompact hUopen).mp hlim
      (Metric.closedBall c r) hclosedU hcompact
  rw [Metric.tendstoUniformlyOn_iff] at hunif
  have hevent := hunif (m / 2) (half_pos hm)
  obtain ⟨n, hnclose⟩ := Filter.Eventually.exists hevent
  have hFn_an : AnalyticOnNhd ℂ (F n) (Metric.closedBall c r) :=
    ((hF n).differentiableOn.analyticOnNhd isOpen_univ).mono
      (Set.subset_univ _)
  have hclose :
      ∀ z ∈ Metric.closedBall c r, ‖F n z - f z‖ < m / 2 := by
    intro z hz
    simpa only [dist_eq_norm, norm_sub_rev] using hnclose z hz
  obtain ⟨z, hzball, hzero⟩ :=
    exists_zero_closedBall_of_uniform_close hr hm hfc hboundary hFn_an hclose
  exact hnonzero n z (hclosedU hzball) hzero

/-- The exact Hurwitz theorem needed for root-location closure, specialized to
the two connected half-planes. Pinned Mathlib does not package this
zero-free-limit dichotomy; `hurwitzHalfPlaneNonvanishingClosure` below
discharges the interface project-locally. -/
def HurwitzHalfPlaneNonvanishingClosure : Prop :=
  ∀ (F : ℕ → ℂ → ℂ) (f : ℂ → ℂ),
    (∀ n, Differentiable ℂ (F n)) →
    TendstoLocallyUniformlyOn F f Filter.atTop Set.univ →
    (∀ n z, z.im ≠ 0 → F n z ≠ 0) →
    ((∀ z, 0 < z.im → f z ≠ 0) ∨ (∀ z, 0 < z.im → f z = 0)) ∧
      ((∀ z, z.im < 0 → f z ≠ 0) ∨ (∀ z, z.im < 0 → f z = 0))

/-- The required half-plane Hurwitz principle, now proved project-locally from
Jensen's formula rather than retained as an analytic bridge assumption. -/
theorem hurwitzHalfPlaneNonvanishingClosure :
    HurwitzHalfPlaneNonvanishingClosure := by
  intro F f hF hlim hnonzero
  have hu := hurwitz_nonvanishing_or_eq_zero_on
    (F := F) (f := f) (U := {z : ℂ | 0 < z.im})
    (isOpen_lt continuous_const continuous_im)
    (convex_halfSpace_im_gt 0).isPreconnected hF
    (hlim.mono (Set.subset_univ _))
    (fun n z hz ↦ hnonzero n z (ne_of_gt hz))
  have hl := hurwitz_nonvanishing_or_eq_zero_on
    (F := F) (f := f) (U := {z : ℂ | z.im < 0})
    (isOpen_lt continuous_im continuous_const)
    (convex_halfSpace_im_lt 0).isPreconnected hF
    (hlim.mono (Set.subset_univ _))
    (fun n z hz ↦ hnonzero n z (ne_of_lt hz))
  constructor
  · rcases hu with hu | hu
    · exact Or.inl hu
    · exact Or.inr fun z hz ↦ hu hz
  · rcases hl with hl | hl
    · exact Or.inl hl
    · exact Or.inr fun z hz ↦ hl hz

/-- Under the precise Hurwitz principle, one nonreal zero forces the limit to
vanish on its entire open half-plane. -/
theorem nonreal_zero_forces_halfPlane_zero
    (hHurwitz : HurwitzHalfPlaneNonvanishingClosure)
    {f : ℂ → ℂ} (h : LocallyUniformHyperbolicLimit f) :
    (∀ {z : ℂ}, 0 < z.im → f z = 0 → ∀ w, 0 < w.im → f w = 0) ∧
      (∀ {z : ℂ}, z.im < 0 → f z = 0 → ∀ w, w.im < 0 → f w = 0) := by
  obtain ⟨p, hp, hlim⟩ := h
  have hdiff :
      ∀ n, Differentiable ℂ (fun z ↦ complexPolynomialEval (p n) z) :=
    fun n ↦ ((p n).map (algebraMap ℝ ℂ)).differentiable
  have hnonreal :
      ∀ n z, z.im ≠ 0 → complexPolynomialEval (p n) z ≠ 0 := by
    intro n z hz hzero
    have hroot :
        ((p n).map (algebraMap ℝ ℂ)).IsRoot z := by
      simpa [complexPolynomialEval, Polynomial.IsRoot] using hzero
    obtain ⟨x, hx⟩ := hp n z hroot
    have him := congrArg Complex.im hx
    simp at him
    exact hz him
  have hcases := hHurwitz
    (fun n z ↦ complexPolynomialEval (p n) z) f hdiff hlim hnonreal
  constructor
  · intro z hz hzero
    rcases hcases.1 with hnonzero | hallzero
    · exact (hnonzero z hz hzero).elim
    · exact hallzero
  · intro z hz hzero
    rcases hcases.2 with hnonzero | hallzero
    · exact (hnonzero z hz hzero).elim
    · exact hallzero

/-- If an entire locally uniform polynomial limit vanishes on either open
half-plane, Mathlib's analytic identity theorem propagates that vanishing to
the whole plane.  Thus the identically-zero branch in Hurwitz is global, not
an independent branch for each half-plane. -/
theorem eq_zero_of_locallyUniformHyperbolicLimit_of_halfPlane_zero
    {f : ℂ → ℂ} (h : LocallyUniformHyperbolicLimit f)
    (hall : (∀ z, 0 < z.im → f z = 0) ∨
      (∀ z, z.im < 0 → f z = 0)) :
    ∀ z, f z = 0 := by
  have han : AnalyticOnNhd ℂ f Set.univ :=
    (differentiable_of_locallyUniformHyperbolicLimit h).differentiableOn
      |>.analyticOnNhd isOpen_univ
  rcases hall with hu | hl
  · have hopen : IsOpen {z : ℂ | 0 < z.im} :=
      isOpen_lt continuous_const continuous_im
    have hI : Complex.I ∈ {z : ℂ | 0 < z.im} := by simp
    have hev : f =ᶠ[nhds Complex.I] 0 := by
      filter_upwards [hopen.mem_nhds hI] with z hz
      exact hu z hz
    intro z
    exact han.eqOn_zero_of_preconnected_of_eventuallyEq_zero
      isPreconnected_univ (Set.mem_univ Complex.I) hev (Set.mem_univ z)
  · have hopen : IsOpen {z : ℂ | z.im < 0} :=
      isOpen_lt continuous_im continuous_const
    have hI : -Complex.I ∈ {z : ℂ | z.im < 0} := by simp
    have hev : f =ᶠ[nhds (-Complex.I)] 0 := by
      filter_upwards [hopen.mem_nhds hI] with z hz
      exact hl z hz
    intro z
    exact han.eqOn_zero_of_preconnected_of_eventuallyEq_zero
      isPreconnected_univ (Set.mem_univ (-Complex.I)) hev (Set.mem_univ z)

/-- Hurwitz plus one nonzero witness in each half-plane forces every zero of
the locally uniform hyperbolic limit to be real. -/
theorem zero_reality_of_locallyUniformHyperbolicLimit
    (hHurwitz : HurwitzHalfPlaneNonvanishingClosure)
    {f : ℂ → ℂ} (h : LocallyUniformHyperbolicLimit f)
    (hupper : ∃ w : ℂ, 0 < w.im ∧ f w ≠ 0)
    (hlower : ∃ w : ℂ, w.im < 0 ∧ f w ≠ 0)
    {z : ℂ} (hz : f z = 0) :
    z.im = 0 := by
  obtain ⟨hup, hlow⟩ := nonreal_zero_forces_halfPlane_zero hHurwitz h
  rcases lt_trichotomy z.im 0 with hneg | hzero | hpos
  · obtain ⟨w, hw, hnz⟩ := hlower
    exact (hnz (hlow hneg hz w hw)).elim
  · exact hzero
  · obtain ⟨w, hw, hnz⟩ := hupper
    exact (hnz (hup hpos hz w hw)).elim

/-- After the analytic identity-theorem step, only one nonzero witness
anywhere in the plane is needed to exclude Hurwitz's identically-zero branch. -/
theorem zero_reality_of_locallyUniformHyperbolicLimit_of_nonzero
    (hHurwitz : HurwitzHalfPlaneNonvanishingClosure)
    {f : ℂ → ℂ} (h : LocallyUniformHyperbolicLimit f)
    (hnonzero : ∃ w : ℂ, f w ≠ 0)
    {z : ℂ} (hz : f z = 0) :
    z.im = 0 := by
  obtain ⟨hup, hlow⟩ := nonreal_zero_forces_halfPlane_zero hHurwitz h
  rcases lt_trichotomy z.im 0 with hneg | hzero | hpos
  · obtain ⟨w, hw⟩ := hnonzero
    exact (hw (eq_zero_of_locallyUniformHyperbolicLimit_of_halfPlane_zero h
      (Or.inr (hlow hneg hz)) w)).elim
  · exact hzero
  · obtain ⟨w, hw⟩ := hnonzero
    exact (hw (eq_zero_of_locallyUniformHyperbolicLimit_of_halfPlane_zero h
      (Or.inl (hup hpos hz)) w)).elim

/-- Unconditional zero-reality closure for a nonzero locally uniform limit of
real hyperbolic polynomials.  This discharges the Hurwitz side of the
project-local Laguerre--Pólya limit interface. -/
theorem zero_reality_of_locallyUniformHyperbolicLimit_unconditional
    {f : ℂ → ℂ} (h : LocallyUniformHyperbolicLimit f)
    (hnonzero : ∃ w : ℂ, f w ≠ 0)
    {z : ℂ} (hz : f z = 0) :
    z.im = 0 :=
  zero_reality_of_locallyUniformHyperbolicLimit_of_nonzero
    hurwitzHalfPlaneNonvanishingClosure h hnonzero hz

/-- Repeated complex derivatives preserve locally uniform convergence. This
is the analytic coefficient-extraction mechanism for reverse Jensen limits. -/
theorem tendstoLocallyUniformlyOn_iteratedDeriv
    {F : ℕ → ℂ → ℂ} {f : ℂ → ℂ}
    (hlim : TendstoLocallyUniformlyOn F f atTop Set.univ)
    (hF : ∀ N n, Differentiable ℂ ((deriv^[n]) (F N)))
    (n : ℕ) :
    TendstoLocallyUniformlyOn
      (fun N ↦ (deriv^[n]) (F N)) ((deriv^[n]) f)
      atTop Set.univ := by
  induction n with
  | zero => simpa using hlim
  | succ n ih =>
      have hd := ih.deriv
        (Filter.Eventually.of_forall fun N ↦ (hF N n).differentiableOn)
        isOpen_univ
      convert hd using 1
      · funext N
        rw [Function.iterate_succ_apply']
        rfl
      · rw [Function.iterate_succ_apply']

theorem tendsto_iteratedDeriv_apply_of_locallyUniform
    {F : ℕ → ℂ → ℂ} {f : ℂ → ℂ}
    (hlim : TendstoLocallyUniformlyOn F f atTop Set.univ)
    (hF : ∀ N n, Differentiable ℂ ((deriv^[n]) (F N)))
    (n : ℕ) (z : ℂ) :
    Tendsto (fun N ↦ (deriv^[n]) (F N) z) atTop
      (𝓝 ((deriv^[n]) f z)) :=
  (tendstoLocallyUniformlyOn_iteratedDeriv hlim hF n).tendsto_at
    (Set.mem_univ z)

/-- Hyperbolicity is closed under locally uniform limits of real polynomial
evaluations as soon as the limiting polynomial is visibly nonzero. -/
theorem hyperbolic_of_tendstoLocallyUniformly
    {p : ℝ[X]} {P : ℕ → ℝ[X]}
    (hP : ∀ N, Hyperbolic (P N))
    (hlim : TendstoLocallyUniformlyOn
      (fun N ↦ complexPolynomialEval (P N))
      (complexPolynomialEval p) atTop Set.univ)
    (hp0 : p.coeff 0 ≠ 0) :
    Hyperbolic p := by
  have hLP : LocallyUniformHyperbolicLimit (complexPolynomialEval p) :=
    ⟨P, hP, hlim⟩
  have hnonzero : ∃ w : ℂ, complexPolynomialEval p w ≠ 0 := by
    refine ⟨0, ?_⟩
    change (p.map (algebraMap ℝ ℂ)).eval 0 ≠ 0
    rw [Polynomial.eval_zero_map, ← Polynomial.coeff_zero_eq_eval_zero]
    exact
      Complex.ofReal_ne_zero.mpr hp0
  intro z hz
  have hzfun : complexPolynomialEval p z = 0 := by
    simpa [complexPolynomialEval, Polynomial.IsRoot] using hz
  have him :=
    zero_reality_of_locallyUniformHyperbolicLimit_unconditional
      hLP hnonzero hzfun
  exact ⟨z.re, Complex.ext (by simp) (by simpa using him)⟩

/-- Exact approximation datum sufficient for reverse Pólya--Jensen. The
finite rows already have the current project's binomial/shift-zero
normalization; only coefficientwise convergence is required. -/
def JensenCoefficientwiseHyperbolicApproximation (a : ℕ → ℝ) : Prop :=
  ∃ b : ℕ → ℕ → ℝ,
    (∀ N d : ℕ, 1 ≤ d → Hyperbolic (jensenPolynomial (b N) d 0)) ∧
    ∀ j : ℕ, Tendsto (fun N ↦ b N j) atTop (𝓝 (a j))

/-- Coefficientwise limits of hyperbolic Jensen rows remain hyperbolic. -/
theorem unshiftedJensenHyperbolic_of_coefficientwiseApproximation
    {a : ℕ → ℝ} (ha0 : a 0 ≠ 0)
    (happrox : JensenCoefficientwiseHyperbolicApproximation a) :
    ∀ d : ℕ, 1 ≤ d → Hyperbolic (jensenPolynomial a d 0) := by
  rcases happrox with ⟨b, hbhyper, hbconv⟩
  intro d hd
  let P : ℕ → ℝ[X] := fun N ↦ jensenPolynomial (b N) d 0
  have hcoeff : ∀ j ∈ Finset.range (d + 1),
      Tendsto (fun N ↦
        (((P N).coeff j : ℝ) : ℂ)) atTop
        (𝓝 ((((jensenPolynomial a d 0).coeff j : ℝ) : ℂ))) := by
    intro j hj
    have hjd : j ≤ d := Nat.le_of_lt_succ (Finset.mem_range.mp hj)
    have hr := (hbconv j).const_mul (d.choose j : ℝ)
    have hc := Complex.continuous_ofReal.continuousAt.tendsto.comp hr
    convert hc using 1
    · funext N
      simp [P, coeff_jensenPolynomial, hjd]
    · simp [coeff_jensenPolynomial, hjd]
  have hsum := tendstoLocallyUniformlyOn_finset_powerSum
    (Finset.range (d + 1)) hcoeff
  have hlim : TendstoLocallyUniformlyOn
      (fun N ↦ complexPolynomialEval (P N))
      (complexPolynomialEval (jensenPolynomial a d 0))
      atTop Set.univ := by
    convert hsum using 1
    · funext N z
      have hdeg_le : (P N).natDegree ≤ d := by
        rw [Polynomial.natDegree_le_iff_coeff_eq_zero]
        intro m hm
        rw [coeff_jensenPolynomial, if_neg (not_le_of_gt hm)]
      have hdeg :
          (P N).natDegree < d + 1 :=
        hdeg_le.trans_lt (Nat.lt_succ_self d)
      rw [complexPolynomialEval, ← Polynomial.eval₂_eq_eval_map,
        Polynomial.eval₂_eq_sum_range' _ hdeg]
      norm_num
    · funext z
      have hdeg_le : (jensenPolynomial a d 0).natDegree ≤ d := by
        rw [Polynomial.natDegree_le_iff_coeff_eq_zero]
        intro m hm
        rw [coeff_jensenPolynomial, if_neg (not_le_of_gt hm)]
      have hdeg :
          (jensenPolynomial a d 0).natDegree < d + 1 :=
        hdeg_le.trans_lt (Nat.lt_succ_self d)
      rw [complexPolynomialEval, ← Polynomial.eval₂_eq_eval_map,
        Polynomial.eval₂_eq_sum_range' _ hdeg]
      norm_num
  apply hyperbolic_of_tendstoLocallyUniformly (fun N ↦ hbhyper N d hd) hlim
  simpa [coeff_jensenPolynomial] using ha0

theorem allJensenHyperbolic_of_coefficientwiseApproximation
    {a : ℕ → ℝ} (ha : NonzeroJensenFamily a) (ha0 : a 0 ≠ 0)
    (happrox : JensenCoefficientwiseHyperbolicApproximation a) :
    AllJensenHyperbolic a :=
  (allJensenHyperbolic_iff_unshifted_of_nonzeroJensen a ha).2
    (unshiftedJensenHyperbolic_of_coefficientwiseApproximation ha0 happrox)

def XiLaguerrePolyaMembership : Prop :=
  LocallyUniformHyperbolicLimit xiJensenGeneratingFunction

theorem xiLaguerrePolyaMembership_of_allJensenHyperbolic
    (hconv : XiJensenScalingConvergence)
    (hJ : AllJensenHyperbolic xiGamma) :
    XiLaguerrePolyaMembership :=
  ⟨xiRescaledJensenPolynomial,
    xiRescaledJensenPolynomial_hyperbolic hJ, hconv⟩

theorem xiLaguerrePolyaMembership_of_allJensenHyperbolic_unconditional
    (hJ : AllJensenHyperbolic xiGamma) :
    XiLaguerrePolyaMembership :=
  xiLaguerrePolyaMembership_of_allJensenHyperbolic
    xiJensenScalingConvergence hJ

/-- Finite genus-zero product with negative real zeros `-root i`.  This is the
product obtained by pairing the centered xi zeros `± i sqrt(root i)`. -/
def positiveGenusZeroPolynomial
    (c : ℝ) (root : ℕ → ℝ) (N : ℕ) : ℝ[X] :=
  Polynomial.C c *
    ∏ i ∈ Finset.range N,
      (Polynomial.C 1 + Polynomial.C (root i)⁻¹ * Polynomial.X)

/-- Genus-one Weierstrass primary factor, recreated locally from the Li route. -/
def centeredGenusOnePrimaryFactor (w : ℂ) : ℂ :=
  (1 - w) * Complex.exp w

/-- Pairing the centered conjugate roots `±iγ` cancels both genus-one
exponentials and leaves the genus-zero factor in the squared variable. -/
theorem centeredGenusOnePrimaryFactor_pair
    (z : ℂ) {γ : ℝ} (hγ : γ ≠ 0) :
    centeredGenusOnePrimaryFactor (z / (Complex.I * γ)) *
        centeredGenusOnePrimaryFactor (z / (-Complex.I * γ)) =
      1 + z ^ 2 / (γ : ℂ) ^ 2 := by
  have hIγ : Complex.I * (γ : ℂ) ≠ 0 :=
    mul_ne_zero Complex.I_ne_zero (by exact_mod_cast hγ)
  have hnIγ : -Complex.I * (γ : ℂ) ≠ 0 :=
    mul_ne_zero (neg_ne_zero.mpr Complex.I_ne_zero) (by exact_mod_cast hγ)
  unfold centeredGenusOnePrimaryFactor
  rw [show
      (1 - z / (Complex.I * γ)) * Complex.exp (z / (Complex.I * γ)) *
          ((1 - z / (-Complex.I * γ)) *
            Complex.exp (z / (-Complex.I * γ))) =
        ((1 - z / (Complex.I * γ)) *
          (1 - z / (-Complex.I * γ))) *
            (Complex.exp (z / (Complex.I * γ)) *
              Complex.exp (z / (-Complex.I * γ))) by ring]
  rw [← Complex.exp_add]
  have hexp :
      z / (Complex.I * γ) + z / (-Complex.I * γ) = 0 := by
    rw [show -Complex.I * (γ : ℂ) = -(Complex.I * γ) by ring,
      div_neg, add_neg_cancel]
  rw [hexp, Complex.exp_zero, mul_one]
  have hneg :
      z / (-Complex.I * γ) = -(z / (Complex.I * γ)) := by
    rw [show -Complex.I * (γ : ℂ) = -(Complex.I * γ) by ring, div_neg]
  rw [hneg]
  have hsquare :
      (z / (Complex.I * γ)) ^ 2 = -z ^ 2 / (γ : ℂ) ^ 2 := by
    rw [div_pow, mul_pow, Complex.I_sq]
    field_simp [hγ]
  rw [show
      (1 - z / (Complex.I * γ)) *
          (1 - -(z / (Complex.I * γ))) =
        1 - (z / (Complex.I * γ)) ^ 2 by ring, hsquare]
  ring

theorem deriv_zero_of_even
    {f : ℂ → ℂ} (_hf : Differentiable ℂ f)
    (heven : ∀ z, f (-z) = f z) :
    deriv f 0 = 0 := by
  have hfun : (fun z ↦ f (-z)) = f := funext heven
  have hderiv := congrArg (fun g : ℂ → ℂ ↦ deriv g 0) hfun
  rw [deriv_comp_neg, neg_zero] at hderiv
  have htwo : (2 : ℂ) * deriv f 0 = 0 := by
    linear_combination -hderiv
  exact (mul_eq_zero.mp htwo).resolve_left (by norm_num)

/-- Once Hadamard quotient rigidity gives an exponential-affine prefactor,
centered evenness removes its linear term.  This is the final normalization
step needed after divisor equality and order-one growth. -/
theorem exp_affine_prefactor_eq_const_of_even
    {f P : ℂ → ℂ} (a b : ℂ)
    (hf : Differentiable ℂ f) (hP : Differentiable ℂ P)
    (hfeven : ∀ z, f (-z) = f z)
    (hPeven : ∀ z, P (-z) = P z)
    (hP0 : P 0 ≠ 0)
    (hrep : ∀ z, f z = Complex.exp (a + b * z) * P z) :
    ∀ z, f z = Complex.exp a * P z := by
  have hdf : deriv f 0 = 0 := deriv_zero_of_even hf hfeven
  have hdP : deriv P 0 = 0 := deriv_zero_of_even hP hPeven
  have hfun :
      f = (fun z ↦ Complex.exp (a + b * z)) * P := by
    funext z
    exact hrep z
  have hderiv := congrArg (fun g : ℂ → ℂ ↦ deriv g 0) hfun
  have haff : HasDerivAt (fun z : ℂ ↦ a + b * z) b 0 := by
    simpa only [Function.id_def, one_mul, mul_one, mul_comm] using
      ((hasDerivAt_id (𝕜 := ℂ) (0 : ℂ)).mul_const b).const_add a
  have hexp : HasDerivAt (fun z : ℂ ↦ Complex.exp (a + b * z))
      (Complex.exp a * b) 0 := by
    simpa only [mul_zero, add_zero] using haff.cexp
  rw [hdf] at hderiv
  have hmulderiv :
      deriv ((fun z : ℂ ↦ Complex.exp (a + b * z)) * P) 0 =
        (Complex.exp a * b) * P 0 + Complex.exp a * deriv P 0 := by
    simpa only [mul_zero, add_zero] using
      (hexp.mul (hP.differentiableAt.hasDerivAt)).deriv
  rw [hmulderiv, hdP, mul_zero, add_zero] at hderiv
  have hb : b = 0 := by
    have hprod : Complex.exp a * (b * P 0) = 0 := by
      linear_combination -hderiv
    have hbp : b * P 0 = 0 :=
      (mul_eq_zero.mp hprod).resolve_left (Complex.exp_ne_zero a)
    exact (mul_eq_zero.mp hbp).resolve_right hP0
  intro z
  simpa [hb] using hrep z

/-- Normal convergence of positive genus-zero factors over any countable
index type. -/
theorem hasProdLocallyUniformlyOn_positiveGenusZeroFactors_indexed
    {ι : Type*} [Countable ι] (root : ι → ℝ)
    (hsum : Summable (fun i ↦ |(root i)⁻¹|)) :
    HasProdLocallyUniformlyOn
      (fun (i : ι) (z : ℂ) ↦ 1 + ((root i)⁻¹ : ℝ) * z)
      (fun z : ℂ ↦ ∏' i : ι, (1 + ((root i)⁻¹ : ℝ) * z))
      Set.univ := by
  apply hasProdLocallyUniformlyOn_of_forall_compact isOpen_univ
  intro K hKu hK
  obtain ⟨R, hR⟩ := hK.isBounded.exists_norm_le
  have hu : Summable (fun i ↦ R * |(root i)⁻¹|) :=
    hsum.mul_left R
  apply hu.hasProdUniformlyOn_one_add hK
  · filter_upwards [] with i z hz
    rw [norm_mul, Complex.norm_real, Real.norm_eq_abs]
    simpa [mul_comm] using
      mul_le_mul_of_nonneg_right (hR z hz) (abs_nonneg (root i)⁻¹)
  · intro i
    fun_prop

/-- Inverse-square summability of paired centered zeros becomes the exact
inverse-first-power summability needed for the genus-zero product in `z²`. -/
theorem hasProdLocallyUniformlyOn_positiveGenusZeroFactors
    (root : ℕ → ℝ)
    (hsum : Summable (fun i ↦ |(root i)⁻¹|)) :
    HasProdLocallyUniformlyOn
      (fun (i : ℕ) (z : ℂ) ↦ 1 + ((root i)⁻¹ : ℝ) * z)
      (fun z : ℂ ↦ ∏' i : ℕ, (1 + ((root i)⁻¹ : ℝ) * z))
      Set.univ :=
  hasProdLocallyUniformlyOn_positiveGenusZeroFactors_indexed root hsum

theorem positiveGenusZeroPolynomial_tendstoLocallyUniformly
    (c : ℝ) (root : ℕ → ℝ)
    (hsum : Summable (fun i ↦ |(root i)⁻¹|)) :
    TendstoLocallyUniformlyOn
      (fun N z ↦ complexPolynomialEval
        (positiveGenusZeroPolynomial c root N) z)
      (fun z : ℂ ↦ (c : ℂ) *
        ∏' i : ℕ, (1 + ((root i)⁻¹ : ℝ) * z))
      Filter.atTop Set.univ := by
  have hprod : TendstoLocallyUniformlyOn
      (fun (N : ℕ) (z : ℂ) ↦ ∏ i ∈ Finset.range N,
        (1 + ((root i)⁻¹ : ℝ) * z))
      (fun z : ℂ ↦ ∏' i : ℕ,
        (1 + ((root i)⁻¹ : ℝ) * z))
      Filter.atTop Set.univ := by
    intro u hu x hx
    obtain ⟨t, ht, hevent⟩ :=
      hasProdLocallyUniformlyOn_positiveGenusZeroFactors root hsum
        u hu x hx
    exact ⟨t, ht, Filter.tendsto_finset_range.eventually hevent⟩
  have hconst : TendstoLocallyUniformlyOn
      (fun _ : ℕ ↦ fun _ : ℂ ↦ (c : ℂ))
      (fun _ : ℂ ↦ (c : ℂ)) Filter.atTop Set.univ := by
    intro u hu x hx
    exact ⟨Set.univ, Filter.univ_mem,
      Filter.Eventually.of_forall (fun _ y hy ↦ refl_mem_uniformity hu)⟩
  have hprodcont : ContinuousOn
      (fun z : ℂ ↦ ∏' i : ℕ,
        (1 + ((root i)⁻¹ : ℝ) * z)) Set.univ := by
    apply hprod.continuousOn
    exact Filter.Frequently.of_forall fun N ↦ by
      apply Continuous.continuousOn
      apply continuous_finsetProd
      intro i hi
      fun_prop
  have hmul := hconst.mul₀ hprod (by fun_prop) hprodcont
  convert hmul using 1
  · funext N z
    simp [positiveGenusZeroPolynomial, complexPolynomialEval]
  · funext z
    rfl

theorem hyperbolic_C {c : ℝ} (hc : c ≠ 0) :
    Hyperbolic (Polynomial.C c) := by
  intro z hz
  simp [Polynomial.IsRoot.def, hc] at hz

theorem positiveGenusZeroPolynomial_hyperbolic
    {c : ℝ} (hc : c ≠ 0) {root : ℕ → ℝ}
    (hroot : ∀ i, root i ≠ 0) (N : ℕ) :
    Hyperbolic (positiveGenusZeroPolynomial c root N) := by
  induction N with
  | zero =>
      simpa [positiveGenusZeroPolynomial] using hyperbolic_C hc
  | succ N ih =>
      rw [positiveGenusZeroPolynomial, Finset.prod_range_succ, ← mul_assoc]
      exact hyperbolic_mul ih
        (hyperbolic_C_add_C_mul_X 1 (root N)⁻¹
          (inv_ne_zero (hroot N)))

theorem positiveGenusZeroPolynomial_nonpositiveRooted
    {c : ℝ} (hc : c ≠ 0) {root : ℕ → ℝ}
    (hroot : ∀ i, 0 < root i) (N : ℕ) :
    NonpositiveRooted (positiveGenusZeroPolynomial c root N) := by
  unfold positiveGenusZeroPolynomial
  apply nonpositiveRooted_mul (nonpositiveRooted_C hc)
  induction N with
  | zero =>
      simpa using (nonpositiveRooted_C (c := 1) (by norm_num))
  | succ N ih =>
      rw [Finset.prod_range_succ]
      exact nonpositiveRooted_mul ih
        (nonpositiveRooted_one_add_inv_mul_X (hroot N))

/-- Exponential Taylor coefficients of a finite positive genus-zero product.
This is the coefficient normalization used by `jensenPolynomial`. -/
noncomputable def positiveGenusZeroExponentialCoefficient
    (c : ℝ) (root : ℕ → ℝ) (N n : ℕ) : ℝ :=
  n.factorial * (positiveGenusZeroPolynomial c root N).coeff n

/-- Degree-`n` Schur--Szegő composition in binomial coordinates:
if `p = ∑ binom(n,j) a_j X^j` and
`q = ∑ binom(n,j) b_j X^j`, this is
`∑ binom(n,j) a_j b_j X^j`. -/
noncomputable def schurSzegoComposition
    (n : ℕ) (p q : ℝ[X]) : ℝ[X] :=
  ∑ j ∈ Finset.range (n + 1),
    Polynomial.monomial j
      (p.coeff j * q.coeff j / (n.choose j : ℝ))

theorem coeff_schurSzegoComposition
    (n : ℕ) (p q : ℝ[X]) (j : ℕ) :
    (schurSzegoComposition n p q).coeff j =
      if j ≤ n then p.coeff j * q.coeff j / (n.choose j : ℝ)
      else 0 := by
  classical
  unfold schurSzegoComposition
  rw [Polynomial.finsetSum_coeff]
  by_cases hj : j ≤ n
  · rw [if_pos hj]
    rw [Finset.sum_eq_single_of_mem j (by simpa)]
    · simp
    · intro b hb hbj
      rw [Polynomial.coeff_monomial, if_neg hbj]
  · rw [if_neg hj]
    apply Finset.sum_eq_zero
    intro b hb
    rw [Polynomial.coeff_monomial, if_neg]
    intro hbj
    subst b
    exact hj (by simpa using hb)

/-- The second Schur--Szegő factor required by degree-`d` Jensen weighting. -/
noncomputable def schurSzegoJensenKernel (N d : ℕ) : ℝ[X] :=
  ∑ j ∈ Finset.range (N + 1),
    Polynomial.monomial j
      ((N.choose j : ℝ) * (d.descFactorial j : ℝ))

theorem coeff_schurSzegoJensenKernel (N d j : ℕ) :
    (schurSzegoJensenKernel N d).coeff j =
      if j ≤ N then
        (N.choose j : ℝ) * (d.descFactorial j : ℝ)
      else 0 := by
  classical
  unfold schurSzegoJensenKernel
  rw [Polynomial.finsetSum_coeff]
  by_cases hj : j ≤ N
  · rw [if_pos hj]
    rw [Finset.sum_eq_single_of_mem j (by simpa)]
    · simp
    · intro b hb hbj
      rw [Polynomial.coeff_monomial, if_neg hbj]
  · rw [if_neg hj]
    apply Finset.sum_eq_zero
    intro b hb
    rw [Polynomial.coeff_monomial, if_neg]
    intro hbj
    subst b
    exact hj (by simpa using hb)

theorem schurSzegoJensenKernel_zero_left (d : ℕ) :
    schurSzegoJensenKernel 0 d = 1 := by
  ext j
  rw [coeff_schurSzegoJensenKernel]
  cases j with
  | zero => norm_num
  | succ j =>
      rw [if_neg (by omega)]
      rw [Polynomial.coeff_one]
      simp

theorem schurSzegoJensenKernel_zero_right (N : ℕ) :
    schurSzegoJensenKernel N 0 = 1 := by
  ext j
  rw [coeff_schurSzegoJensenKernel]
  cases j with
  | zero => norm_num
  | succ j =>
      by_cases hj : j + 1 ≤ N
      · rw [if_pos hj,
          Nat.descFactorial_eq_zero_iff_lt.mpr (by omega)]
        rw [Polynomial.coeff_one]
        simp
      · rw [if_neg hj]
        rw [Polynomial.coeff_one]
        simp

theorem schurSzegoJensenKernel_nonpositiveRooted_zero_left (d : ℕ) :
    NonpositiveRooted (schurSzegoJensenKernel 0 d) := by
  rw [schurSzegoJensenKernel_zero_left]
  exact nonpositiveRooted_C (by norm_num)

theorem schurSzegoJensenKernel_nonpositiveRooted_zero_right (N : ℕ) :
    NonpositiveRooted (schurSzegoJensenKernel N 0) := by
  rw [schurSzegoJensenKernel_zero_right]
  exact nonpositiveRooted_C (by norm_num)

theorem schurSzegoJensenKernel_one_right (N : ℕ) :
    schurSzegoJensenKernel N 1 =
      1 + Polynomial.C (N : ℝ) * Polynomial.X := by
  ext j
  rw [coeff_schurSzegoJensenKernel]
  cases j with
  | zero => norm_num
  | succ j =>
      cases j with
      | zero =>
          rw [Polynomial.coeff_add, Polynomial.coeff_one,
            Polynomial.coeff_C_mul_X]
          by_cases hN : N = 0
          · subst N
            norm_num
          · rw [if_pos (Nat.one_le_iff_ne_zero.mpr hN)]
            norm_num
      | succ j =>
          rw [Nat.descFactorial_eq_zero_iff_lt.mpr (by omega)]
          rw [Polynomial.coeff_add, Polynomial.coeff_one,
            Polynomial.coeff_C_mul_X]
          simp

theorem schurSzegoJensenKernel_nonpositiveRooted_one_right (N : ℕ) :
    NonpositiveRooted (schurSzegoJensenKernel N 1) := by
  rw [schurSzegoJensenKernel_one_right]
  by_cases hN : N = 0
  · subst N
    simpa using (nonpositiveRooted_C (c := 1) (by norm_num))
  · simpa using
      (nonpositiveRooted_one_add_inv_mul_X
        (r := ((N : ℝ)⁻¹))
        (inv_pos.mpr (by exact_mod_cast (Nat.pos_of_ne_zero hN))))

theorem positiveGenusZeroPolynomial_natDegree_le
    (c : ℝ) (root : ℕ → ℝ) (N : ℕ) :
    (positiveGenusZeroPolynomial c root N).natDegree ≤ N := by
  unfold positiveGenusZeroPolynomial
  calc
    (Polynomial.C c *
      ∏ i ∈ Finset.range N,
        (Polynomial.C 1 + Polynomial.C (root i)⁻¹ *
          Polynomial.X)).natDegree
        ≤ (Polynomial.C c).natDegree +
            (∏ i ∈ Finset.range N,
              (Polynomial.C 1 + Polynomial.C (root i)⁻¹ *
                Polynomial.X)).natDegree :=
      Polynomial.natDegree_mul_le
    _ ≤ 0 + ∑ _i ∈ Finset.range N, 1 := by
      gcongr
      · simp
      · exact (Polynomial.natDegree_prod_le (Finset.range N)
          (fun i ↦ Polynomial.C 1 + Polynomial.C (root i)⁻¹ *
            Polynomial.X)).trans (Finset.sum_le_sum fun i hi ↦
              (Polynomial.natDegree_add_le _ _).trans
                (max_le (by simp)
                  (Polynomial.natDegree_mul_le.trans (by simp))))
    _ = N := by simp

theorem positiveGenusZeroPolynomial_ne_zero
    {c : ℝ} (hc : c ≠ 0) (root : ℕ → ℝ) (N : ℕ) :
    positiveGenusZeroPolynomial c root N ≠ 0 := by
  unfold positiveGenusZeroPolynomial
  apply mul_ne_zero (Polynomial.C_ne_zero.mpr hc)
  rw [Finset.prod_ne_zero_iff]
  intro i hi hfactor
  have hcoeff := congrArg (fun p : ℝ[X] ↦ p.coeff 0) hfactor
  norm_num at hcoeff

/-- The normalization check behind the reverse route: composing a finite
genus-zero product with the Jensen kernel gives exactly the current
`binom(d,j) * gamma(j)` polynomial, including `N < d`, `d < N`, and zero
coefficient boundary cases. -/
theorem schurSzegoComposition_positiveGenusZero_eq_jensenPolynomial
    (c : ℝ) (root : ℕ → ℝ) (N d : ℕ) :
    schurSzegoComposition N (positiveGenusZeroPolynomial c root N)
        (schurSzegoJensenKernel N d) =
      jensenPolynomial
        (positiveGenusZeroExponentialCoefficient c root N) d 0 := by
  ext j
  rw [coeff_schurSzegoComposition, coeff_jensenPolynomial]
  by_cases hjN : j ≤ N
  · rw [if_pos hjN, coeff_schurSzegoJensenKernel, if_pos hjN]
    by_cases hjd : j ≤ d
    · rw [if_pos hjd]
      have hchoose : (N.choose j : ℝ) ≠ 0 := by
        exact_mod_cast (Nat.choose_pos hjN).ne'
      field_simp [hchoose]
      unfold positiveGenusZeroExponentialCoefficient
      rw [Nat.descFactorial_eq_factorial_mul_choose]
      push_cast
      ring
    · rw [if_neg hjd,
        Nat.descFactorial_eq_zero_iff_lt.mpr (lt_of_not_ge hjd)]
      norm_num
  · rw [if_neg hjN]
    have hcoeff : (positiveGenusZeroPolynomial c root N).coeff j = 0 := by
      apply Polynomial.coeff_eq_zero_of_natDegree_lt
      exact (positiveGenusZeroPolynomial_natDegree_le c root N).trans_lt
        (lt_of_not_ge hjN)
    by_cases hjd : j ≤ d
    · rw [if_pos hjd]
      simp [positiveGenusZeroExponentialCoefficient, hcoeff]
    · rw [if_neg hjd]

/-- The superficially natural strict Schur--Szegő statement is false at
boundary supports: two nonzero rooted inputs can have zero composition. -/
def StrictFiniteSchurSzegoNonpositiveRootedness : Prop :=
  ∀ (n : ℕ) (p q : ℝ[X]),
    p ≠ 0 → p.natDegree ≤ n → q ≠ 0 → q.natDegree ≤ n →
    NonpositiveRooted p → NonpositiveRooted q →
      NonpositiveRooted (schurSzegoComposition n p q)

theorem not_strictFiniteSchurSzegoNonpositiveRootedness :
    ¬ StrictFiniteSchurSzegoNonpositiveRootedness := by
  intro h
  have hroot := h 1 (Polynomial.X : ℝ[X]) 1
    (by simp) (by simp) (by norm_num) (by simp)
    nonpositiveRooted_X (nonpositiveRooted_C (by norm_num))
  have hzero :
      schurSzegoComposition 1 (Polynomial.X : ℝ[X]) 1 = 0 := by
    ext j
    rw [coeff_schurSzegoComposition]
    by_cases hj : j ≤ 1
    · interval_cases j
      · norm_num
      · norm_num [Polynomial.coeff_one]
    · rw [if_neg hj]
      rfl
  rw [hzero] at hroot
  exact zero_not_hyperbolic (nonpositiveRooted_hyperbolic hroot)

/-- Precise finite Schur--Szegő theorem with the necessary zero-output
alternative. This handles missing endpoint coefficients and disjoint supports;
the Jensen specialization below separately proves that its output is nonzero. -/
def FiniteSchurSzegoNonpositiveRootedness : Prop :=
  ∀ (n : ℕ) (p q : ℝ[X]),
    p ≠ 0 → p.natDegree ≤ n → q ≠ 0 → q.natDegree ≤ n →
    NonpositiveRooted p → NonpositiveRooted q →
      schurSzegoComposition n p q = 0 ∨
        NonpositiveRooted (schurSzegoComposition n p q)

/-- The remaining special-function input: these reversed generalized
Laguerre kernels have only real nonpositive zeros, with multiplicities and
the `N < d`/`d < N` degree drops included. -/
def SchurSzegoJensenKernelNonpositiveRootedness : Prop :=
  ∀ N d : ℕ, NonpositiveRooted (schurSzegoJensenKernel N d)

/-- Exact finite real-rootedness theorem still needed in the reverse direction.
In classical language these are weighted matching/Jensen polynomials; proving
this is equivalent to the relevant Schur--Szegő multiplier-sequence step. -/
def FinitePositiveGenusZeroJensenHyperbolicity : Prop :=
  ∀ (c : ℝ) (root : ℕ → ℝ) (N d : ℕ),
    c ≠ 0 → (∀ i, 0 < root i) → 1 ≤ d →
      Hyperbolic (jensenPolynomial
        (positiveGenusZeroExponentialCoefficient c root N) d 0)

theorem finitePositiveGenusZeroJensenHyperbolicity_of_schurSzego
    (hschur : FiniteSchurSzegoNonpositiveRootedness)
    (hkernel : SchurSzegoJensenKernelNonpositiveRootedness) :
    FinitePositiveGenusZeroJensenHyperbolicity := by
  intro c root N d hc hroot hd
  rw [← schurSzegoComposition_positiveGenusZero_eq_jensenPolynomial]
  apply nonpositiveRooted_hyperbolic
  rcases hschur N
      (positiveGenusZeroPolynomial c root N)
      (schurSzegoJensenKernel N d)
      (positiveGenusZeroPolynomial_ne_zero hc root N)
      (positiveGenusZeroPolynomial_natDegree_le c root N)
      (by
        intro hq
        have hcoeff := congrArg (fun p : ℝ[X] ↦ p.coeff 0) hq
        rw [coeff_schurSzegoJensenKernel] at hcoeff
        norm_num at hcoeff)
      (by
        rw [Polynomial.natDegree_le_iff_coeff_eq_zero]
        intro j hj
        rw [coeff_schurSzegoJensenKernel, if_neg (not_le_of_gt hj)])
      (positiveGenusZeroPolynomial_nonpositiveRooted hc hroot N)
      (hkernel N d) with hzero | hrooted
  · have hcoeff := congrArg (fun p : ℝ[X] ↦ p.coeff 0) hzero
    rw [coeff_schurSzegoComposition, if_pos (Nat.zero_le N),
      coeff_schurSzegoJensenKernel, if_pos (Nat.zero_le N)] at hcoeff
    have hprod :
        (∏ i ∈ Finset.range N,
          (Polynomial.C 1 + Polynomial.C (root i)⁻¹ *
            Polynomial.X)).coeff 0 = 1 := by
      rw [Polynomial.coeff_zero_eq_eval_zero]
      change Polynomial.evalRingHom 0
        (∏ i ∈ Finset.range N,
          (Polynomial.C 1 + Polynomial.C (root i)⁻¹ *
            Polynomial.X)) = 1
      rw [map_prod]
      simp
    have hsource :
        (positiveGenusZeroPolynomial c root N).coeff 0 = c := by
      unfold positiveGenusZeroPolynomial
      rw [Polynomial.coeff_zero_eq_eval_zero]
      have hprodEval :
          Polynomial.eval 0
            (∏ i ∈ Finset.range N,
              (Polynomial.C 1 + Polynomial.C (root i)⁻¹ *
                Polynomial.X)) = 1 := by
        rwa [← Polynomial.coeff_zero_eq_eval_zero]
      rw [Polynomial.eval_mul, Polynomial.eval_C, hprodEval, mul_one]
    rw [hsource] at hcoeff
    norm_num at hcoeff
    exact (hc hcoeff).elim
  · exact hrooted

/-- Coefficient-extraction half of the positive-product reverse theorem.
Local uniform convergence should imply this by repeated derivative
convergence; it is stated separately from finite real-rootedness. -/
def XiPositiveGenusZeroCoefficientConvergence : Prop :=
  ∀ (c : ℝ) (root : ℕ → ℝ),
    c ≠ 0 → (∀ i, 0 < root i) →
    TendstoLocallyUniformlyOn
      (fun N z ↦ complexPolynomialEval
        (positiveGenusZeroPolynomial c root N) z)
      xiJensenGeneratingFunction atTop Set.univ →
    ∀ j : ℕ, Tendsto
      (fun N ↦ positiveGenusZeroExponentialCoefficient c root N j)
      atTop (𝓝 (xiGamma j))

/-- Source-shaped genus-zero Hadamard representation in the squared centered
variable.  Li's multiplicity-indexed genus-one product supplies the raw zero
product; even pairing and order `< 1` must identify this normalized form. -/
def PositiveGenusZeroProductRepresentation (f : ℂ → ℂ) : Prop :=
  ∃ c : ℝ, ∃ root : ℕ → ℝ,
    c ≠ 0 ∧ (∀ i, 0 < root i) ∧
      TendstoLocallyUniformlyOn
        (fun N z ↦ complexPolynomialEval
          (positiveGenusZeroPolynomial c root N) z)
        f Filter.atTop Set.univ

/-- The exact normalized product identity left after pairing the Li-route
genus-one divisor.  The constant is fixed at the already proved positive
center value `xiGamma 0`. -/
def XiNormalizedGenusZeroProductIdentification : Prop :=
  xiGamma 0 ≠ 0 ∧ ∃ root : ℕ → ℝ,
    (∀ i, 0 < root i) ∧
    Summable (fun i ↦ |(root i)⁻¹|) ∧
    ∀ z : ℂ, xiJensenGeneratingFunction z =
      (xiGamma 0 : ℂ) *
        ∏' i : ℕ, (1 + ((root i)⁻¹ : ℝ) * z)

theorem positiveGenusZeroProductRepresentation_of_identification
    (h : XiNormalizedGenusZeroProductIdentification) :
    PositiveGenusZeroProductRepresentation xiJensenGeneratingFunction := by
  rcases h with ⟨hgamma0, root, hroot, hsum, hid⟩
  refine ⟨xiGamma 0, root, hgamma0, hroot, ?_⟩
  have hconv :=
    positiveGenusZeroPolynomial_tendstoLocallyUniformly
      (xiGamma 0) root hsum
  convert hconv using 1
  funext z
  exact hid z

theorem locallyUniformHyperbolicLimit_of_positiveGenusZeroProduct
    {f : ℂ → ℂ} (h : PositiveGenusZeroProductRepresentation f) :
    LocallyUniformHyperbolicLimit f := by
  rcases h with ⟨c, root, hc, hroot, hlim⟩
  refine ⟨positiveGenusZeroPolynomial c root,
    fun N ↦ positiveGenusZeroPolynomial_hyperbolic hc
      (fun i ↦ (hroot i).ne') N, hlim⟩

theorem xiLaguerrePolyaMembership_of_positiveGenusZeroProduct
    (h : PositiveGenusZeroProductRepresentation
      xiJensenGeneratingFunction) :
    XiLaguerrePolyaMembership :=
  locallyUniformHyperbolicLimit_of_positiveGenusZeroProduct h

theorem xiLaguerrePolyaMembership_of_normalizedProductIdentification
    (h : XiNormalizedGenusZeroProductIdentification) :
    XiLaguerrePolyaMembership :=
  xiLaguerrePolyaMembership_of_positiveGenusZeroProduct
    (positiveGenusZeroProductRepresentation_of_identification h)

theorem xiJensenGeneratingFunction_zero :
    xiJensenGeneratingFunction 0 = (xiGamma 0 : ℂ) := by
  unfold xiJensenGeneratingFunction
  rw [tsum_eq_single 0]
  · simp
  · intro n hn
    simp [zero_pow hn]

/-- A centered Hadamard conclusion before evenness removes the affine
exponential factor.  This is the exact output supplied by divisor equality
and order-one quotient rigidity. -/
def XiCenteredAffinePairedProductIdentification : Prop :=
  xiGamma 0 ≠ 0 ∧ ∃ root : ℕ → ℝ,
    (∀ i, 0 < root i) ∧
    Summable (fun i ↦ |(root i)⁻¹|) ∧
    Differentiable ℂ (fun z : ℂ ↦
      ∏' i : ℕ, (1 + ((root i)⁻¹ : ℝ) * z ^ 2)) ∧
    ∃ a b : ℂ, ∀ z : ℂ, xiJensenEntire z =
      Complex.exp (a + b * z) *
        ∏' i : ℕ, (1 + ((root i)⁻¹ : ℝ) * z ^ 2)

/-- The affine centered Hadamard representation automatically has no linear
exponential term, and evaluation at zero fixes its constant to `xiGamma 0`. -/
theorem normalizedProductIdentification_of_centeredAffine
    (h : XiCenteredAffinePairedProductIdentification) :
    XiNormalizedGenusZeroProductIdentification := by
  rcases h with ⟨hgamma0, root, hroot, hsum, hPdiff, a, b, hrep⟩
  let P : ℂ → ℂ := fun z ↦
    ∏' i : ℕ, (1 + ((root i)⁻¹ : ℝ) * z ^ 2)
  have hPeven : ∀ z, P (-z) = P z := by
    intro z
    simp only [P, neg_sq]
  have hP0 : P 0 ≠ 0 := by
    simp [P]
  have hconst : ∀ z, xiJensenEntire z = Complex.exp a * P z :=
    exp_affine_prefactor_eq_const_of_even a b
      differentiable_xiJensenEntire hPdiff xiJensenEntire_neg hPeven hP0 hrep
  have hcenter0 : xiJensenEntire 0 = (xiGamma 0 : ℂ) := by
    rw [← xiJensenGeneratingFunction_sq 0]
    simpa using xiJensenGeneratingFunction_zero
  have hexpa : Complex.exp a = (xiGamma 0 : ℂ) := by
    have h0 := hconst 0
    rw [hcenter0] at h0
    simpa [P] using h0.symm
  refine ⟨hgamma0, root, hroot, hsum, ?_⟩
  intro w
  by_cases hw : w = 0
  · subst w
    rw [xiJensenGeneratingFunction_zero]
    simp
  · let z := Complex.sqrt w
    have hzsq : z ^ 2 = w := by
      rw [pow_two]
      dsimp only [z]
      rw [sqrt_eq_exp hw, ← Complex.exp_add]
      convert Complex.exp_log hw using 1
      ring
    rw [← hzsq, xiJensenGeneratingFunction_sq, hconst, hexpa]

theorem xiLaguerrePolya_zero_reality_of_momentRepresentation
    (hLP : XiLaguerrePolyaMembership)
    (hmoment : XiGammaMomentRepresentation)
    {z : ℂ} (hz : xiJensenGeneratingFunction z = 0) :
    z.im = 0 := by
  apply zero_reality_of_locallyUniformHyperbolicLimit_unconditional hLP
  · refine ⟨0, ?_⟩
    rw [xiJensenGeneratingFunction_zero]
    exact Complex.ofReal_ne_zero.mpr
      (ne_of_gt (xiGamma_strictlyPositive_of_momentRepresentation hmoment 0))
  · exact hz

/-- Exact remaining coefficient nondegeneracy needed to reduce all shifts to
the unshifted xi Jensen family. -/
def XiCoefficientNondegeneracy : Prop :=
  JensenWindowNonzero xiGamma

theorem xiCoefficientNondegeneracy_iff_noAdjacentZeros :
    XiCoefficientNondegeneracy ↔ NoAdjacentZeros xiGamma :=
  jensenWindowNonzero_iff_noAdjacentZeros xiGamma

/-- The source-backed positive-moment identity would close coefficient
nondegeneracy, while allowing the formal route to state that missing analytic
identity separately from the weaker no-adjacent-zero target. -/
theorem xiCoefficientNondegeneracy_of_momentRepresentation
    (h : XiGammaMomentRepresentation) :
    XiCoefficientNondegeneracy := by
  rw [xiCoefficientNondegeneracy_iff_noAdjacentZeros]
  have hpos := xiGamma_strictlyPositive_of_momentRepresentation h
  intro n
  exact Or.inl (ne_of_gt (hpos n))

/-- The Mellin/Phi calculation closes coefficient positivity and
nondegeneracy without any zero-location or RH hypothesis. -/
theorem xiGamma_strictlyPositive : StrictlyPositive xiGamma :=
  xiGamma_strictlyPositive_of_momentRepresentation
    xiGamma_momentRepresentation

theorem xiCoefficientNondegeneracy_unconditional :
    XiCoefficientNondegeneracy :=
  xiCoefficientNondegeneracy_of_momentRepresentation
    xiGamma_momentRepresentation

theorem xiNonzeroJensenFamily : NonzeroJensenFamily xiGamma :=
  (nonzeroJensenFamily_iff_windowNonzero xiGamma).2
    xiCoefficientNondegeneracy_unconditional

/-- Once the coefficient identity is discharged, Laguerre--Pólya membership
is the only hypothesis left in this zero-reality direction. -/
theorem xiLaguerrePolya_zero_reality
    (hLP : XiLaguerrePolyaMembership)
    {z : ℂ} (hz : xiJensenGeneratingFunction z = 0) :
    z.im = 0 :=
  xiLaguerrePolya_zero_reality_of_momentRepresentation
    hLP xiGamma_momentRepresentation hz

theorem xiJensenGeneratingFunction_ofReal_pos {r : ℝ} (hr : 0 ≤ r) :
    0 < (xiJensenGeneratingFunction (r : ℂ)).re := by
  have hs := Complex.hasSum_re (xiJensenGeneratingFunction_hasSum (r : ℂ))
  rw [← hs.tsum_eq]
  refine hs.summable.tsum_pos ?_ 0 ?_
  · intro n
    have hpow : (r : ℂ) ^ n = ((r ^ n : ℝ) : ℂ) :=
      (Complex.ofReal_pow r n).symm
    have hre := congrArg Complex.re hpow
    have him := congrArg Complex.im hpow
    simp only [Complex.ofReal_re] at hre
    simp only [Complex.ofReal_im] at him
    rw [Complex.mul_re, Complex.ofReal_re, Complex.ofReal_im,
      zero_mul, sub_zero, hre]
    exact mul_nonneg
      (div_nonneg (xiGamma_strictlyPositive n).le (by positivity))
      (pow_nonneg hr n)
  · simpa using xiGamma_strictlyPositive 0

theorem xiJensenEntire_ofReal_ne_zero (r : ℝ) :
    xiJensenEntire (r : ℂ) ≠ 0 := by
  intro hz
  have hpos := xiJensenGeneratingFunction_ofReal_pos (sq_nonneg r)
  rw [Complex.ofReal_pow] at hpos
  rw [xiJensenGeneratingFunction_sq, hz] at hpos
  norm_num at hpos

theorem xiJensenEntire_zero_im_ne_zero {z : ℂ}
    (hz : xiJensenEntire z = 0) :
    z.im ≠ 0 := by
  intro hzim
  have hzreal : z = (z.re : ℂ) := by
    apply Complex.ext
    · simp
    · simpa using hzim
  exact xiJensenEntire_ofReal_ne_zero z.re (hzreal ▸ hz)

theorem xiJensenEntire_zero_re_on_critical_axis
    (hJ : AllJensenHyperbolic xiGamma) {z : ℂ}
    (hz : xiJensenEntire z = 0) :
    z.re = 0 := by
  have hgf : xiJensenGeneratingFunction (z ^ 2) = 0 := by
    rw [xiJensenGeneratingFunction_sq, hz]
  have hreal : (z ^ 2).im = 0 :=
    xiLaguerrePolya_zero_reality
      (xiLaguerrePolyaMembership_of_allJensenHyperbolic_unconditional hJ) hgf
  by_contra hzre
  have hzim : z.im = 0 := by
    rw [pow_two, Complex.mul_im] at hreal
    rcases lt_or_gt_of_ne hzre with hzneg | hzpos <;> nlinarith
  have hsq : z ^ 2 = ((z.re ^ 2 : ℝ) : ℂ) := by
    apply Complex.ext
    · simp [pow_two, Complex.mul_re, hzim]
    · simp [pow_two, Complex.mul_im, hzim]
  have hpos :=
    xiJensenGeneratingFunction_ofReal_pos (sq_nonneg z.re)
  rw [← hsq, hgf] at hpos
  norm_num at hpos

theorem riemannXi_eq_zero_of_nontrivial_riemannZeta_zero
    {s : ℂ} (hzeta : riemannZeta s = 0)
    (htrivial : ¬ ∃ n : ℕ, s = -2 * (n + 1))
    (hs1 : s ≠ 1) :
    riemannXi s = 0 := by
  have hs0 : s ≠ 0 := by
    intro hs
    subst s
    rw [riemannZeta_zero] at hzeta
    norm_num at hzeta
  have hgamma : Complex.Gamma (s / 2) ≠ 0 := by
    apply Complex.Gamma_ne_zero
    intro m hm
    cases m with
    | zero =>
        simp only [Nat.cast_zero, neg_zero] at hm
        exact hs0 ((div_eq_zero_iff.mp hm).resolve_right (by norm_num))
    | succ n =>
        apply htrivial
        refine ⟨n, ?_⟩
        calc
          s = 2 * (s / 2) := by ring
          _ = -2 * ((n : ℂ) + 1) := by rw [hm]; push_cast; ring
  have hpow : (Real.pi : ℂ) ^ (-s / 2) ≠ 0 :=
    Complex.cpow_ne_zero_iff.mpr
      (Or.inl (Complex.ofReal_ne_zero.mpr (ne_of_gt Real.pi_pos)))
  rw [riemannZeta_eq_completedRiemannZeta₀ hs0] at hzeta
  have hnum :
      completedRiemannZeta₀ s - 1 / s - 1 / (1 - s) = 0 :=
    (div_eq_zero_iff.mp hzeta).resolve_right (mul_ne_zero hpow hgamma)
  unfold riemannXi
  apply (div_eq_zero_iff.mpr (Or.inl ?_))
  field_simp [hs0, sub_ne_zero.mpr hs1] at hnum
  linear_combination -hnum

theorem riemannHypothesis_of_allJensenHyperbolic
    (hJ : AllJensenHyperbolic xiGamma) :
    KakeyaRiemannHypothesisRoot := by
  rw [← kakeya_rh_expanded_iff_canonical]
  intro s hzeta htrivial hs1
  have hxi :=
    riemannXi_eq_zero_of_nontrivial_riemannZeta_zero
      hzeta htrivial hs1
  have hcenter : xiJensenEntire (s - 1 / 2) = 0 := by
    rw [xiJensenEntire]
    convert mul_eq_zero_of_right (8 : ℂ) hxi using 1
    congr 2
    ring
  have haxis :=
    xiJensenEntire_zero_re_on_critical_axis hJ hcenter
  norm_num [Complex.sub_re] at haxis
  linarith

theorem xiJensenEntire_zero_re_on_critical_axis_of_riemannHypothesis
    (hRH : KakeyaRiemannHypothesisRoot) {z : ℂ}
    (hz : xiJensenEntire z = 0) :
    z.re = 0 := by
  let s : ℂ := 1 / 2 + z
  have hxi : riemannXi s = 0 := by
    rw [xiJensenEntire] at hz
    exact (mul_eq_zero.mp hz).resolve_left (by norm_num)
  have hzim : z.im ≠ 0 := xiJensenEntire_zero_im_ne_zero hz
  have hs_im : s.im = z.im := by simp [s]
  have hs0 : s ≠ 0 := by
    intro hs
    have : s.im = 0 := congrArg Complex.im hs
    exact hzim (hs_im ▸ this)
  have hs1 : s ≠ 1 := by
    intro hs
    have : s.im = 0 := congrArg Complex.im hs
    exact hzim (hs_im ▸ this)
  have hxinumerator :
      1 + s * (s - 1) * completedRiemannZeta₀ s = 0 := by
    unfold riemannXi at hxi
    exact (div_eq_zero_iff.mp hxi).resolve_right (by norm_num)
  have hmellin :
      completedRiemannZeta₀ s - 1 / s - 1 / (1 - s) = 0 := by
    field_simp [hs0, sub_ne_zero.mpr hs1]
    linear_combination -hxinumerator
  have hzeta : riemannZeta s = 0 := by
    rw [riemannZeta_eq_completedRiemannZeta₀ hs0, hmellin, zero_div]
  have htrivial : ¬ ∃ n : ℕ, s = -2 * (n + 1) := by
    rintro ⟨n, hn⟩
    have hsreal : s.im = 0 := by rw [hn]; norm_num
    exact hzim (hs_im ▸ hsreal)
  have hExpanded : KakeyaRiemannHypothesisExpanded :=
    kakeya_rh_expanded_iff_canonical.mpr hRH
  have hsre := hExpanded s hzeta htrivial hs1
  dsimp only [s] at hsre
  norm_num [Complex.add_re] at hsre
  linarith

/-- Exact centered-xi zero statement equivalent to RH in this normalization:
zeros of `z ↦ ξ(1/2+z)` lie on the imaginary axis. -/
def XiCenteredCriticalZeroReality : Prop :=
  ∀ z : ℂ, xiJensenEntire z = 0 → z.re = 0

theorem xiCenteredCriticalZeroReality_of_riemannHypothesis
    (hRH : KakeyaRiemannHypothesisRoot) :
    XiCenteredCriticalZeroReality :=
  fun _ hz ↦ xiJensenEntire_zero_re_on_critical_axis_of_riemannHypothesis hRH hz

theorem riemannHypothesis_of_xiCenteredCriticalZeroReality
    (hzero : XiCenteredCriticalZeroReality) :
    KakeyaRiemannHypothesisRoot := by
  rw [← kakeya_rh_expanded_iff_canonical]
  intro s hzeta htrivial hs1
  have hxi :=
    riemannXi_eq_zero_of_nontrivial_riemannZeta_zero
      hzeta htrivial hs1
  have hcenter : xiJensenEntire (s - 1 / 2) = 0 := by
    rw [xiJensenEntire]
    convert mul_eq_zero_of_right (8 : ℂ) hxi using 1
    congr 2
    ring
  have haxis := hzero (s - 1 / 2) hcenter
  norm_num [Complex.sub_re] at haxis
  linarith

theorem xiCenteredCriticalZeroReality_iff_riemannHypothesis :
    XiCenteredCriticalZeroReality ↔ KakeyaRiemannHypothesisRoot :=
  ⟨riemannHypothesis_of_xiCenteredCriticalZeroReality,
    xiCenteredCriticalZeroReality_of_riemannHypothesis⟩

/-! ### Multiplicity-correct centered-xi zero enumeration -/

def centeredXiZeroOrder (z : ℂ) : ℕ∞ :=
  analyticOrderAt xiJensenEntire z

theorem centeredXiZeroOrder_ne_top (z : ℂ) :
    centeredXiZeroOrder z ≠ ⊤ := by
  unfold centeredXiZeroOrder
  have hxi : AnalyticOnNhd ℂ xiJensenEntire Set.univ :=
    fun w _ ↦ differentiable_xiJensenEntire.analyticAt w
  apply hxi.analyticOrderAt_ne_top_of_isPreconnected
    (x := (0 : ℂ)) (y := z)
  · exact isPreconnected_univ
  · simp
  · simp
  · rw [(differentiable_xiJensenEntire.analyticAt 0).analyticOrderAt_eq_zero.mpr]
    · exact ENat.zero_ne_top
    · rw [← xiJensenGeneratingFunction_sq 0]
      simp only [zero_pow two_ne_zero]
      rw [xiJensenGeneratingFunction_zero]
      exact Complex.ofReal_ne_zero.mpr
        (ne_of_gt (xiGamma_strictlyPositive 0))

def centeredXiZeroMultiplicity (z : ℂ) : ℕ :=
  analyticOrderNatAt xiJensenEntire z

theorem centeredXiZeroMultiplicity_cast (z : ℂ) :
    (centeredXiZeroMultiplicity z : ℕ∞) = centeredXiZeroOrder z :=
  Nat.cast_analyticOrderNatAt (centeredXiZeroOrder_ne_top z)

noncomputable def centeredXiZeroDivisor :=
  MeromorphicOn.divisor xiJensenEntire Set.univ

theorem centeredXiZeroDivisor_support :
    Function.support centeredXiZeroDivisor =
      xiJensenEntire ⁻¹' {0} := by
  have hxi : AnalyticOnNhd ℂ xiJensenEntire Set.univ :=
    fun z _ ↦ differentiable_xiJensenEntire.analyticAt z
  have hfinite : ∀ u : Set.univ,
      meromorphicOrderAt xiJensenEntire u.1 ≠ ⊤ := by
    intro u
    rw [(differentiable_xiJensenEntire.analyticAt u.1).meromorphicOrderAt_eq]
    simpa [centeredXiZeroOrder] using centeredXiZeroOrder_ne_top u.1
  have h := hxi.meromorphicNFOn.zero_set_eq_divisor_support hfinite
  simpa [centeredXiZeroDivisor] using h.symm

theorem centeredXiZeroDivisor_finite_on_compact
    {K : Set ℂ} (hK : IsCompact K) :
    (K ∩ Function.support centeredXiZeroDivisor).Finite := by
  have hlocal : LocallyFiniteSupport
      (fun z ↦ centeredXiZeroDivisor z) :=
    fun z ↦ centeredXiZeroDivisor.supportLocallyFiniteWithinDomain z
      (by trivial)
  exact hlocal.finite_inter_support_of_isCompact hK

theorem centeredXiZeroDivisor_support_countable :
    (Function.support centeredXiZeroDivisor).Countable := by
  letI : DiscreteTopology
      (Function.support centeredXiZeroDivisor) :=
    centeredXiZeroDivisor.discreteSupport.to_subtype
  exact (HereditarilyLindelofSpace.isLindelof
    (Function.support centeredXiZeroDivisor)).countable inferInstance

/-- Every centered-xi zero is repeated exactly its analytic multiplicity. -/
def CenteredXiZeroIndex :=
  Σ z : {w : ℂ // xiJensenEntire w = 0},
    Fin (centeredXiZeroMultiplicity z)

def centeredXiZeroRoot (i : CenteredXiZeroIndex) : ℂ := i.1

noncomputable instance : Encodable CenteredXiZeroIndex := by
  classical
  have hs : (xiJensenEntire ⁻¹' {0}).Countable := by
    rw [← centeredXiZeroDivisor_support]
    exact centeredXiZeroDivisor_support_countable
  letI : Encodable {z : ℂ // xiJensenEntire z = 0} := hs.toEncodable
  unfold CenteredXiZeroIndex
  infer_instance

noncomputable def centeredXiZeroFiber
    (z : ℂ) : Finset CenteredXiZeroIndex := by
  classical
  by_cases hz : xiJensenEntire z = 0
  · exact Finset.univ.map
      ⟨fun k ↦ ⟨⟨z, hz⟩, k⟩, by
        intro a b h
        exact Fin.ext
          (congrArg (fun x : CenteredXiZeroIndex ↦ x.2.val) h)⟩
  · exact ∅

theorem mem_centeredXiZeroFiber (z : ℂ) (i : CenteredXiZeroIndex) :
    i ∈ centeredXiZeroFiber z ↔ centeredXiZeroRoot i = z := by
  classical
  unfold centeredXiZeroFiber
  split_ifs with hz
  · constructor
    · intro hi
      simp only [Finset.mem_map, Finset.mem_univ, true_and] at hi
      obtain ⟨k, rfl⟩ := hi
      rfl
    · intro hi
      subst z
      simp only [Finset.mem_map, Finset.mem_univ, true_and]
      refine ⟨i.2, ?_⟩
      apply Sigma.ext (Subtype.ext rfl)
      simp
  · simp only [Finset.notMem_empty, false_iff]
    intro hi
    apply hz
    rw [← hi]
    exact i.1.property

theorem card_centeredXiZeroFiber (z : ℂ) :
    (centeredXiZeroFiber z).card = centeredXiZeroMultiplicity z := by
  classical
  unfold centeredXiZeroFiber
  split_ifs with hz
  · rw [Finset.card_map, Finset.card_univ, Fintype.card_fin]
  · have hm : centeredXiZeroMultiplicity z = 0 := by
      rw [← Nat.cast_inj (R := ℕ∞), centeredXiZeroMultiplicity_cast]
      unfold centeredXiZeroOrder
      simp only [Nat.cast_zero]
      rw [(differentiable_xiJensenEntire.analyticAt z).analyticOrderAt_eq_zero]
      exact hz
    simp [hm]

theorem centeredXiZeroDivisor_apply (z : ℂ) :
    centeredXiZeroDivisor z =
      (centeredXiZeroMultiplicity z : ℤ) := by
  unfold centeredXiZeroDivisor
  have hA : AnalyticAt ℂ xiJensenEntire z :=
    differentiable_xiJensenEntire.analyticAt z
  have hM : MeromorphicOn xiJensenEntire Set.univ :=
    fun w _ ↦ differentiable_xiJensenEntire.analyticAt w |>.meromorphicAt
  rw [MeromorphicOn.divisor_apply hM (Set.mem_univ z),
    hA.meromorphicOrderAt_eq]
  have ho : analyticOrderAt xiJensenEntire z =
      (centeredXiZeroMultiplicity z : ℕ∞) := by
    simpa [centeredXiZeroOrder] using
      (centeredXiZeroMultiplicity_cast z).symm
  rw [ho]
  simp

theorem centeredXiZeroMultiplicity_pos_iff (z : ℂ) :
    0 < centeredXiZeroMultiplicity z ↔ xiJensenEntire z = 0 := by
  rw [Nat.pos_iff_ne_zero]
  constructor
  · intro hm
    apply (differentiable_xiJensenEntire.analyticAt z).analyticOrderAt_ne_zero.mp
    intro ho
    apply hm
    rw [← Nat.cast_inj (R := ℕ∞), centeredXiZeroMultiplicity_cast]
    simpa [centeredXiZeroOrder] using ho
  · intro hz hm
    have ho :=
      (differentiable_xiJensenEntire.analyticAt z).analyticOrderAt_ne_zero.mpr hz
    apply ho
    change centeredXiZeroOrder z = 0
    rw [← centeredXiZeroMultiplicity_cast]
    simp [hm]

theorem exists_centeredXiZeroRoot_iff (z : ℂ) :
    (∃ i : CenteredXiZeroIndex, centeredXiZeroRoot i = z) ↔
      xiJensenEntire z = 0 := by
  constructor
  · rintro ⟨i, rfl⟩
    exact i.1.property
  · intro hz
    have hcard : 0 < (centeredXiZeroFiber z).card := by
      rw [card_centeredXiZeroFiber, centeredXiZeroMultiplicity_pos_iff]
      exact hz
    obtain ⟨i, hi⟩ := Finset.card_pos.mp hcard
    exact ⟨i, (mem_centeredXiZeroFiber z i).mp hi⟩

theorem centeredXiZeroRoot_ne_zero (i : CenteredXiZeroIndex) :
    centeredXiZeroRoot i ≠ 0 := by
  intro hi
  have hzero : xiJensenEntire (centeredXiZeroRoot i) = 0 := by
    simpa only [centeredXiZeroRoot] using i.1.property
  rw [hi, ← xiJensenGeneratingFunction_sq 0] at hzero
  simp only [zero_pow two_ne_zero] at hzero
  rw [xiJensenGeneratingFunction_zero] at hzero
  exact Complex.ofReal_ne_zero.mpr
    (ne_of_gt (xiGamma_strictlyPositive 0)) hzero

theorem centeredXiZeroIndex_finite_norm_le (R : ℝ) :
    {i : CenteredXiZeroIndex | ‖centeredXiZeroRoot i‖ ≤ R}.Finite := by
  let Z : Set ℂ :=
    Metric.closedBall 0 R ∩ Function.support centeredXiZeroDivisor
  have hZ : Z.Finite :=
    centeredXiZeroDivisor_finite_on_compact
      (ProperSpace.isCompact_closedBall 0 R)
  have hU : (⋃ z ∈ Z, (↑(centeredXiZeroFiber z) :
      Set CenteredXiZeroIndex)).Finite := by
    apply hZ.biUnion
    intro z hz
    exact Finset.finite_toSet _
  apply hU.subset
  intro i hi
  simp only [Set.mem_setOf_eq] at hi
  have hzball : centeredXiZeroRoot i ∈ Metric.closedBall 0 R := by
    simpa [Metric.mem_closedBall, dist_zero_right] using hi
  have hzsupport : centeredXiZeroRoot i ∈
      Function.support centeredXiZeroDivisor := by
    rw [centeredXiZeroDivisor_support]
    exact i.1.property
  apply Set.mem_iUnion_of_mem (centeredXiZeroRoot i)
  apply Set.mem_iUnion_of_mem ⟨hzball, hzsupport⟩
  exact (mem_centeredXiZeroFiber _ _).mpr rfl

theorem centeredXiZeroRoot_escape :
    Filter.Tendsto (fun i : CenteredXiZeroIndex ↦ ‖centeredXiZeroRoot i‖)
      Filter.cofinite Filter.atTop := by
  apply Filter.tendsto_atTop.mpr
  intro R
  change {i : CenteredXiZeroIndex |
    R ≤ ‖centeredXiZeroRoot i‖} ∈ Filter.cofinite
  rw [Filter.mem_cofinite]
  apply (centeredXiZeroIndex_finite_norm_le R).subset
  intro i hi
  simp only [Set.mem_compl_iff, Set.mem_setOf_eq, not_le] at hi ⊢
  exact hi.le

theorem centeredXiZeroRoot_re_eq_zero
    (hzero : XiCenteredCriticalZeroReality) (i : CenteredXiZeroIndex) :
    (centeredXiZeroRoot i).re = 0 :=
  hzero _ i.1.property

theorem centeredXiZeroOrder_neg (z : ℂ) :
    centeredXiZeroOrder (-z) = centeredXiZeroOrder z := by
  let g : ℂ → ℂ := fun w ↦ -w
  have hg : AnalyticAt ℂ g z := by
    dsimp only [g]
    fun_prop
  have hgd : deriv g z ≠ 0 := by
    dsimp only [g]
    simp
  have hcomp := analyticOrderAt_comp_of_deriv_ne_zero
    (f := xiJensenEntire) hg hgd
  have heq : xiJensenEntire ∘ g = xiJensenEntire := by
    funext w
    exact xiJensenEntire_neg w
  unfold centeredXiZeroOrder
  rw [heq] at hcomp
  exact hcomp.symm

theorem centeredXiZeroMultiplicity_neg (z : ℂ) :
    centeredXiZeroMultiplicity (-z) = centeredXiZeroMultiplicity z := by
  rw [← ENat.coe_inj, centeredXiZeroMultiplicity_cast,
    centeredXiZeroMultiplicity_cast, centeredXiZeroOrder_neg]

/-- The fixed-point-free negation involution on the multiplicity enumeration. -/
noncomputable def centeredXiZeroIndexNeg
    (i : CenteredXiZeroIndex) : CenteredXiZeroIndex :=
  ⟨⟨-centeredXiZeroRoot i, by
      rw [xiJensenEntire_neg]
      exact i.1.property⟩,
    Fin.cast (centeredXiZeroMultiplicity_neg
      (centeredXiZeroRoot i)).symm i.2⟩

@[simp]
theorem centeredXiZeroRoot_indexNeg (i : CenteredXiZeroIndex) :
    centeredXiZeroRoot (centeredXiZeroIndexNeg i) =
      -centeredXiZeroRoot i :=
  rfl

theorem centeredXiZeroRoot_im_ne_zero
    (hzero : XiCenteredCriticalZeroReality) (i : CenteredXiZeroIndex) :
    (centeredXiZeroRoot i).im ≠ 0 := by
  intro him
  apply centeredXiZeroRoot_ne_zero i
  apply Complex.ext
  · exact centeredXiZeroRoot_re_eq_zero hzero i
  · exact him

/-- One representative from each `±` pair, retaining every unit of analytic
multiplicity on the positive-imaginary side. -/
abbrev CenteredXiPositiveZeroIndex :=
  {i : CenteredXiZeroIndex // 0 < (centeredXiZeroRoot i).im}

noncomputable instance : Encodable CenteredXiPositiveZeroIndex :=
  (Set.to_countable
    {i : CenteredXiZeroIndex | 0 < (centeredXiZeroRoot i).im}).toEncodable

def centeredXiPositiveSquaredOrdinate
    (i : CenteredXiPositiveZeroIndex) : ℝ :=
  (centeredXiZeroRoot i.1).im ^ 2

theorem centeredXiPositiveSquaredOrdinate_pos
    (i : CenteredXiPositiveZeroIndex) :
    0 < centeredXiPositiveSquaredOrdinate i := by
  unfold centeredXiPositiveSquaredOrdinate
  exact sq_pos_of_pos i.property

theorem centeredXiZeroIndex_positive_or_neg_positive
    (hzero : XiCenteredCriticalZeroReality) (i : CenteredXiZeroIndex) :
    0 < (centeredXiZeroRoot i).im ∨
      0 < (centeredXiZeroRoot (centeredXiZeroIndexNeg i)).im := by
  rcases lt_or_gt_of_ne (centeredXiZeroRoot_im_ne_zero hzero i) with h | h
  · right
    simpa using neg_pos.mpr h
  · exact Or.inl h

/-- The quantitative zero-counting input required to turn the paired RH
enumeration into a normally convergent genus-zero product. -/
def XiCenteredZeroInverseSquareSummability : Prop :=
  Summable (fun i : CenteredXiZeroIndex ↦
    ‖(centeredXiZeroRoot i)⁻¹‖ ^ 2)

theorem summable_dyadic_linear_majorant_jensen (A : ℝ) :
    Summable (fun N : ℕ ↦
      A * (N + 1 : ℝ) * (1 / 2 : ℝ) ^ N) := by
  have hnat : Summable (fun N : ℕ ↦
      (N : ℝ) * (1 / 2 : ℝ) ^ N) := by
    simpa using
      (summable_pow_mul_geometric_of_norm_lt_one (R := ℝ) 1
        (r := (1 / 2 : ℝ)) (by norm_num))
  have hone : Summable (fun N : ℕ ↦ (1 / 2 : ℝ) ^ N) :=
    summable_geometric_of_norm_lt_one (by norm_num)
  apply ((hnat.add hone).mul_left A).congr
  intro N
  ring

/-- An order-one `O(R log R)` dyadic multiplicity count implies
inverse-square summability. -/
theorem summable_norm_inv_sq_of_dyadic_count_jensen
    {ι : Type*} [DecidableEq ι] (root : ι → ℂ)
    (shell : ℕ → Finset ι)
    (hpartition : ∀ i, ∃! N, i ∈ shell N)
    (hlower : ∀ N i, i ∈ shell N →
      (2 : ℝ) ^ N ≤ ‖root i‖)
    (A : ℝ)
    (hcount : ∀ N, ((shell N).card : ℝ) ≤
      A * (2 : ℝ) ^ N * (N + 1 : ℝ)) :
    Summable (fun i ↦ ‖(root i)⁻¹‖ ^ 2) := by
  rw [summable_partition (f := fun i ↦ ‖(root i)⁻¹‖ ^ 2)
    (fun i ↦ sq_nonneg _) (s := fun N ↦ ↑(shell N))
    (by simpa using hpartition)]
  constructor
  · intro N
    apply summable_of_hasFiniteSupport
    exact Set.toFinite _
  · apply (summable_dyadic_linear_majorant_jensen A).of_nonneg_of_le
    · intro N
      exact tsum_nonneg fun _ ↦ sq_nonneg _
    · intro N
      rw [tsum_fintype]
      have hpow : 0 < (2 : ℝ) ^ N := pow_pos (by norm_num) _
      calc
        ∑ i : ↥(↑(shell N) : Set ι), ‖(root i)⁻¹‖ ^ 2
            ≤ ∑ _i : ↥(↑(shell N) : Set ι),
                1 / ((2 : ℝ) ^ N) ^ 2 := by
          apply Finset.sum_le_sum
          intro i hi
          rw [norm_inv, inv_pow]
          have hsquareRoot :
              ((2 : ℝ) ^ N) ^ 2 ≤ ‖root i‖ ^ 2 :=
            (sq_le_sq₀ hpow.le (norm_nonneg _)).mpr
              (hlower N i i.property)
          simpa [one_div] using
            inv_anti₀ (sq_pos_of_pos hpow) hsquareRoot
        _ = ((shell N).card : ℝ) /
            ((2 : ℝ) ^ N) ^ 2 := by
          simp [div_eq_mul_inv]
        _ ≤ (A * (2 : ℝ) ^ N * (N + 1 : ℝ)) /
            ((2 : ℝ) ^ N) ^ 2 := by
          exact div_le_div_of_nonneg_right (hcount N) (sq_nonneg _)
        _ = A * (N + 1 : ℝ) * (1 / 2 : ℝ) ^ N := by
          field_simp
          rw [div_pow]
          norm_num

theorem summable_norm_inv_sq_of_finite_low_dyadic_count_jensen
    {ι : Type*} [DecidableEq ι] (root : ι → ℂ)
    (low : Finset ι)
    (shell : ℕ → Finset {i : ι // i ∉ low})
    (hpartition : ∀ i : {i : ι // i ∉ low},
      ∃! N, i ∈ shell N)
    (hlower : ∀ N i, i ∈ shell N →
      (2 : ℝ) ^ N ≤ ‖root i‖)
    (A : ℝ)
    (hcount : ∀ N, ((shell N).card : ℝ) ≤
      A * (2 : ℝ) ^ N * (N + 1 : ℝ)) :
    Summable (fun i ↦ ‖(root i)⁻¹‖ ^ 2) := by
  have hc : Summable (fun i : {i : ι // i ∉ low} ↦
      ‖(root i)⁻¹‖ ^ 2) :=
    summable_norm_inv_sq_of_dyadic_count_jensen
      (fun i : {i : ι // i ∉ low} ↦ root i)
      shell hpartition hlower A hcount
  exact (low.summable_compl_iff).mp hc

theorem centeredXi_inverseSquareSummability_of_dyadicZeroCount
    (low : Finset CenteredXiZeroIndex)
    (shell : ℕ → Finset {i : CenteredXiZeroIndex // i ∉ low})
    (hpartition : ∀ i : {i : CenteredXiZeroIndex // i ∉ low},
      ∃! N, i ∈ shell N)
    (hlower : ∀ N i, i ∈ shell N →
      (2 : ℝ) ^ N ≤ ‖centeredXiZeroRoot i‖)
    (A : ℝ)
    (hcount : ∀ N, ((shell N).card : ℝ) ≤
      A * (2 : ℝ) ^ N * (N + 1 : ℝ)) :
    XiCenteredZeroInverseSquareSummability := by
  classical
  exact summable_norm_inv_sq_of_finite_low_dyadic_count_jensen
    centeredXiZeroRoot low shell hpartition hlower A hcount

noncomputable def centeredXiZeroIndexWindow
    (r : ℝ) : Finset CenteredXiZeroIndex :=
  (centeredXiZeroIndex_finite_norm_le r).toFinset

@[simp]
theorem mem_centeredXiZeroIndexWindow
    (r : ℝ) (i : CenteredXiZeroIndex) :
    i ∈ centeredXiZeroIndexWindow r ↔
      ‖centeredXiZeroRoot i‖ ≤ r := by
  simp [centeredXiZeroIndexWindow]

theorem existsUnique_dyadic_interval_jensen (x : ℝ) (hx : 1 ≤ x) :
    ∃! N : ℕ,
      (2 : ℝ) ^ N ≤ x ∧ x < (2 : ℝ) ^ (N + 1) := by
  let p : ℕ → Prop := fun N ↦ x < (2 : ℝ) ^ (N + 1)
  have hp : ∃ N, p N := by
    obtain ⟨n, hn⟩ :=
      pow_unbounded_of_one_lt x (by norm_num : (1 : ℝ) < 2)
    exact ⟨n, hn.trans_le
      (pow_le_pow_right₀ (by norm_num) (Nat.le_succ n))⟩
  let N := Nat.find hp
  have hupper : x < (2 : ℝ) ^ (N + 1) :=
    Nat.find_spec hp
  have hlower : (2 : ℝ) ^ N ≤ x := by
    by_contra h
    have hxlt : x < (2 : ℝ) ^ N := lt_of_not_ge h
    by_cases hN : N = 0
    · rw [hN] at hxlt
      norm_num at hxlt
      linarith
    · obtain ⟨k, hk⟩ := Nat.exists_eq_succ_of_ne_zero hN
      have hpk : p k := by
        dsimp [p]
        rw [hk] at hxlt
        simpa [Nat.succ_eq_add_one] using hxlt
      exact (Nat.find_min hp (by omega)) hpk
  refine ⟨N, ⟨hlower, hupper⟩, ?_⟩
  intro M hM
  apply le_antisymm
  · by_contra hle
    have hMN : N < M := lt_of_not_ge hle
    have hpowers : (2 : ℝ) ^ (N + 1) ≤ (2 : ℝ) ^ M :=
      pow_le_pow_right₀ (by norm_num) (by omega)
    linarith [hupper, hM.1]
  · by_contra hle
    have hNM : M < N := lt_of_not_ge hle
    have hpowers : (2 : ℝ) ^ (M + 1) ≤ (2 : ℝ) ^ N :=
      pow_le_pow_right₀ (by norm_num) (by omega)
    linarith [hM.2, hlower]

noncomputable def centeredXiDyadicZeroShell
    (low : Finset CenteredXiZeroIndex) (N : ℕ) :
    Finset {i : CenteredXiZeroIndex // i ∉ low} := by
  classical
  exact Finset.subtype (fun i ↦ i ∉ low)
    ((centeredXiZeroIndexWindow ((2 : ℝ) ^ (N + 1))).filter
      fun i ↦ (2 : ℝ) ^ N ≤ ‖centeredXiZeroRoot i‖ ∧
        ‖centeredXiZeroRoot i‖ < (2 : ℝ) ^ (N + 1))

@[simp]
theorem mem_centeredXiDyadicZeroShell
    (low : Finset CenteredXiZeroIndex) (N : ℕ)
    (i : {i : CenteredXiZeroIndex // i ∉ low}) :
    i ∈ centeredXiDyadicZeroShell low N ↔
      (2 : ℝ) ^ N ≤ ‖centeredXiZeroRoot i‖ ∧
        ‖centeredXiZeroRoot i‖ < (2 : ℝ) ^ (N + 1) := by
  classical
  simp [centeredXiDyadicZeroShell]
  intro _ h
  exact h.le

theorem centeredXiDyadicZeroShell_partition
    (R : ℝ) (hR : 1 ≤ R) :
    ∀ i : {i : CenteredXiZeroIndex //
      i ∉ centeredXiZeroIndexWindow R},
      ∃! N, i ∈
        centeredXiDyadicZeroShell (centeredXiZeroIndexWindow R) N := by
  intro i
  have hiR : R < ‖centeredXiZeroRoot i‖ := by
    simpa using i.property
  obtain ⟨N, hN, huniq⟩ :=
    existsUnique_dyadic_interval_jensen ‖centeredXiZeroRoot i‖
      (hR.trans hiR.le)
  refine ⟨N, (mem_centeredXiDyadicZeroShell _ _ _).mpr hN, ?_⟩
  intro M hM
  exact huniq M ((mem_centeredXiDyadicZeroShell _ _ _).mp hM)

theorem centeredXiDyadicZeroShell_card_le_window
    (low : Finset CenteredXiZeroIndex) (N : ℕ) :
    (centeredXiDyadicZeroShell low N).card ≤
      (centeredXiZeroIndexWindow ((2 : ℝ) ^ (N + 1))).card := by
  classical
  unfold centeredXiDyadicZeroShell
  rw [Finset.card_subtype]
  exact (Finset.card_filter_le _ _).trans
    (Finset.card_filter_le _ _)

/-- Exact multiplicity-counting consequence required from Jensen's formula
and order-one growth.  Pinned Mathlib has no Riemann--von Mangoldt zero-count
theorem from which to derive this bound. -/
def XiCenteredDyadicOrderOneZeroCount : Prop :=
  ∃ R A : ℝ, 1 ≤ R ∧
    ∀ N, ((centeredXiDyadicZeroShell
      (centeredXiZeroIndexWindow R) N).card : ℝ) ≤
        A * (2 : ℝ) ^ N * (N + 1 : ℝ)

/-- A standard Riemann--von Mangoldt/Jensen cumulative upper bound, counting
the canonical centered-xi multiplicity indices. -/
def XiCenteredCumulativeZeroCount : Prop :=
  ∃ A R : ℝ, 0 ≤ A ∧ 1 ≤ R ∧
    ∀ r : ℝ, R ≤ r →
      ((centeredXiZeroIndexWindow r).card : ℝ) ≤
        A * r * Real.log (2 * r + 2)

/-- Finite divisor contributed by the multiplicity indices in one centered
xi radial window. -/
noncomputable def centeredXiZeroIndexWindowDivisor (r : ℝ) :=
  ∑ i ∈ centeredXiZeroIndexWindow r,
    Function.locallyFinsuppWithin.single (centeredXiZeroRoot i) (1 : ℤ)

theorem centeredXiZeroIndexWindowDivisor_le (r : ℝ) :
    centeredXiZeroIndexWindowDivisor r ≤ centeredXiZeroDivisor := by
  classical
  intro z
  rw [centeredXiZeroDivisor_apply]
  unfold centeredXiZeroIndexWindowDivisor
  have heval := congrFun
    (Function.locallyFinsuppWithin.coe_sum
      (s := centeredXiZeroIndexWindow r)
      (F := fun i ↦ Function.locallyFinsuppWithin.single
        (centeredXiZeroRoot i) (1 : ℤ))) z
  rw [heval, Finset.sum_apply]
  simp only [Function.locallyFinsuppWithin.single_apply,
    Finset.sum_ite, Finset.sum_const_zero, add_zero, Finset.sum_const,
    nsmul_eq_mul, mul_one]
  norm_cast
  calc
    {i ∈ centeredXiZeroIndexWindow r |
      z = centeredXiZeroRoot i}.card
        ≤ (centeredXiZeroFiber z).card := by
      apply Finset.card_le_card
      intro i hi
      simp only [Finset.mem_filter] at hi
      exact (mem_centeredXiZeroFiber z i).mpr hi.2.symm
    _ = centeredXiZeroMultiplicity z := card_centeredXiZeroFiber z

theorem centeredXiZeroDivisor_logCounting :
    Function.locallyFinsuppWithin.logCounting centeredXiZeroDivisor =
      ValueDistribution.logCounting xiJensenEntire 0 := by
  rw [ValueDistribution.logCounting_zero]
  have hxi : AnalyticOnNhd ℂ xiJensenEntire Set.univ :=
    fun z _ ↦ differentiable_xiJensenEntire.analyticAt z
  have hnonneg : 0 ≤ MeromorphicOn.divisor xiJensenEntire Set.univ :=
    MeromorphicOn.AnalyticOnNhd.divisor_nonneg hxi
  unfold centeredXiZeroDivisor
  rw [posPart_eq_self.mpr hnonneg]

theorem xiJensenEntire_zero_ne : xiJensenEntire 0 ≠ 0 := by
  rw [← xiJensenGeneratingFunction_sq 0]
  simp only [zero_pow two_ne_zero]
  rw [xiJensenGeneratingFunction_zero]
  exact Complex.ofReal_ne_zero.mpr
    (ne_of_gt (xiGamma_strictlyPositive 0))

theorem centeredXiZeroDivisor_logCounting_eq_circleAverage
    {R : ℝ} (hR : R ≠ 0) :
    Function.locallyFinsuppWithin.logCounting centeredXiZeroDivisor R =
      Real.circleAverage (Real.log ‖xiJensenEntire ·‖) 0 R -
        Real.log ‖xiJensenEntire 0‖ := by
  unfold centeredXiZeroDivisor
  have hmer : Meromorphic xiJensenEntire :=
    fun z ↦ (differentiable_xiJensenEntire.analyticAt z).meromorphicAt
  rw [Function.locallyFinsuppWithin.logCounting_divisor_eq_circleAverage_sub_const
    hmer hR]
  rw [(differentiable_xiJensenEntire.analyticAt 0).meromorphicTrailingCoeffAt_of_ne_zero
    xiJensenEntire_zero_ne]

theorem centeredXiZeroIndexWindow_card_mul_log_two_le_logCounting
    {r : ℝ} (hr : 1 ≤ r) :
    ((centeredXiZeroIndexWindow r).card : ℝ) * Real.log 2 ≤
      Function.locallyFinsuppWithin.logCounting
        centeredXiZeroDivisor (2 * r) := by
  calc
    ((centeredXiZeroIndexWindow r).card : ℝ) * Real.log 2 =
        ∑ _i ∈ centeredXiZeroIndexWindow r, Real.log 2 := by simp
    _ ≤ ∑ i ∈ centeredXiZeroIndexWindow r,
        Function.locallyFinsuppWithin.logCounting
          (Function.locallyFinsuppWithin.single
            (centeredXiZeroRoot i) 1) (2 * r) := by
      apply Finset.sum_le_sum
      intro i hi
      have hir : ‖centeredXiZeroRoot i‖ ≤ r :=
        (mem_centeredXiZeroIndexWindow r i).mp hi
      rw [Function.locallyFinsuppWithin.logCounting_single_eq_log_sub_const
        (hir.trans (by linarith))]
      norm_num
      rw [← Real.log_div (by positivity)
        (norm_ne_zero_iff.mpr (centeredXiZeroRoot_ne_zero i))]
      apply Real.log_le_log (by norm_num)
      rw [le_div_iff₀
        (norm_pos_iff.mpr (centeredXiZeroRoot_ne_zero i))]
      nlinarith
    _ = Function.locallyFinsuppWithin.logCounting
        (centeredXiZeroIndexWindowDivisor r) (2 * r) := by
      unfold centeredXiZeroIndexWindowDivisor
      rw [map_sum, Finset.sum_apply]
    _ ≤ Function.locallyFinsuppWithin.logCounting
        centeredXiZeroDivisor (2 * r) :=
      Function.locallyFinsuppWithin.logCounting_le
        (centeredXiZeroIndexWindowDivisor_le r) (by linarith)

/-- Actual centered-xi cumulative multiplicity bound, reconstructed from the
proved Gamma/Mellin-strip order-one estimate and Jensen's formula. -/
theorem xiCenteredCumulativeZeroCount_actual :
    XiCenteredCumulativeZeroCount := by
  obtain ⟨C, R, hC, hR, hbound⟩ :=
    exists_xiJensenEntire_orderOneGrowthBound
  let K := |Real.log ‖xiJensenEntire 0‖|
  let A := (2 * C + K / Real.log 2) / Real.log 2
  let R' := max 1 (max (R / 2) 2)
  refine ⟨A, R', ?_, le_max_left _ _, ?_⟩
  · exact div_nonneg
      (add_nonneg (mul_nonneg (by norm_num) hC)
        (div_nonneg (abs_nonneg _) (Real.log_nonneg (by norm_num))))
      (Real.log_nonneg (by norm_num))
  intro r hr
  have hr1 : 1 ≤ r := (le_max_left _ _).trans hr
  have hr2 : 2 ≤ r :=
    (le_max_right 1 (max (R / 2) 2)).trans hr |>
      (le_max_right (R / 2) 2).trans
  have hR2r : R ≤ 2 * r := by
    have h : R / 2 ≤ r := (le_max_left (R / 2) 2).trans
      ((le_max_right 1 (max (R / 2) 2)).trans hr)
    linarith
  have hweighted :=
    centeredXiZeroIndexWindow_card_mul_log_two_le_logCounting hr1
  have hcount :
      Function.locallyFinsuppWithin.logCounting
          centeredXiZeroDivisor (2 * r) ≤
        (2 * C + K / Real.log 2) * r * Real.log (2 * r + 2) := by
    rw [centeredXiZeroDivisor_logCounting_eq_circleAverage (by linarith)]
    have havg :
        Real.circleAverage (Real.log ‖xiJensenEntire ·‖) 0 (2 * r) ≤
          C * (2 * r) * Real.log (2 * r + 2) := by
      apply Real.circleAverage_mono_on_of_le_circle
      · have hm : MeromorphicOn xiJensenEntire
            (Metric.sphere 0 |2 * r|) :=
          fun z _ ↦
            (differentiable_xiJensenEntire.analyticAt z).meromorphicAt
        exact hm.circleIntegrable_log_norm
      intro z hz
      have hzr : ‖z‖ = 2 * r := by
        simpa [Metric.mem_sphere, dist_zero_right,
          abs_of_nonneg (by linarith : 0 ≤ 2 * r)] using hz
      have hn := hbound z (by rw [hzr]; exact hR2r)
      rw [hzr] at hn
      by_cases hzero : xiJensenEntire z = 0
      · simp [hzero]
        exact mul_nonneg (mul_nonneg hC (by linarith))
          (Real.log_nonneg (by linarith))
      · exact (Real.log_le_iff_le_exp
          (norm_pos_iff.mpr hzero)).mpr hn
    have hK : -Real.log ‖xiJensenEntire 0‖ ≤ K := by
      exact neg_le_abs _
    calc
      Real.circleAverage (Real.log ‖xiJensenEntire ·‖) 0 (2 * r) -
          Real.log ‖xiJensenEntire 0‖
          ≤ C * (2 * r) * Real.log (2 * r + 2) + K := by linarith
      _ ≤ (2 * C + K / Real.log 2) * r * Real.log (2 * r + 2) := by
        have hlog2 : 0 < Real.log 2 := Real.log_pos (by norm_num)
        have hlog :
            Real.log 2 ≤ r * Real.log (2 * r + 2) := by
          have hmono : Real.log 2 ≤ Real.log (2 * r + 2) :=
            Real.log_le_log (by norm_num) (by linarith)
          have hnon : 0 ≤ Real.log (2 * r + 2) :=
            Real.log_nonneg (by linarith)
          calc
            Real.log 2 ≤ Real.log (2 * r + 2) := hmono
            _ ≤ r * Real.log (2 * r + 2) := by nlinarith
        have hK0 : 0 ≤ K := abs_nonneg _
        have hKscale :
            K ≤ (K / Real.log 2) *
              (r * Real.log (2 * r + 2)) := by
          calc
            K = (K / Real.log 2) * Real.log 2 := by field_simp
            _ ≤ (K / Real.log 2) *
                (r * Real.log (2 * r + 2)) :=
              mul_le_mul_of_nonneg_left hlog
                (div_nonneg hK0 hlog2.le)
        nlinarith
  have hlog2 : 0 < Real.log 2 := Real.log_pos (by norm_num)
  dsimp [A]
  rw [show (2 * C + K / Real.log 2) / Real.log 2 * r *
      Real.log (2 * r + 2) =
      ((2 * C + K / Real.log 2) * r *
        Real.log (2 * r + 2)) / Real.log 2 by ring]
  exact (le_div_iff₀ hlog2).mpr (hweighted.trans hcount)

theorem centeredXiDyadicZeroShell_card_le_of_cumulative
    {A R : ℝ} (hA : 0 ≤ A)
    (hcum : ∀ r : ℝ, R ≤ r →
      ((centeredXiZeroIndexWindow r).card : ℝ) ≤
        A * r * Real.log (2 * r + 2)) :
    ∀ N,
      ((centeredXiDyadicZeroShell
        (centeredXiZeroIndexWindow R) N).card : ℝ) ≤
        (6 * A * Real.log 2) * (2 : ℝ) ^ N *
          (N + 1 : ℝ) := by
  intro N
  let r : ℝ := (2 : ℝ) ^ (N + 1)
  by_cases hr : R ≤ r
  · have hcard :
        ((centeredXiDyadicZeroShell
          (centeredXiZeroIndexWindow R) N).card : ℝ) ≤
          ((centeredXiZeroIndexWindow r).card : ℝ) := by
      dsimp [r]
      exact_mod_cast centeredXiDyadicZeroShell_card_le_window
        (centeredXiZeroIndexWindow R) N
    have hcum' := hcard.trans (hcum r hr)
    have hargpos : 0 < 2 * r + 2 := by positivity
    have harg : 2 * r + 2 ≤ (2 : ℝ) ^ (N + 3) := by
      dsimp [r]
      rw [pow_add, pow_add]
      norm_num
      have hpow : 1 ≤ (2 : ℝ) ^ N :=
        one_le_pow₀ (by norm_num)
      nlinarith
    have hlog : Real.log (2 * r + 2) ≤
        3 * (N + 1 : ℝ) * Real.log 2 := by
      calc
        Real.log (2 * r + 2) ≤
            Real.log ((2 : ℝ) ^ (N + 3)) :=
          Real.log_le_log hargpos harg
        _ = (N + 3 : ℕ) * Real.log 2 := by
          rw [Real.log_pow]
        _ ≤ 3 * (N + 1 : ℝ) * Real.log 2 := by
          have hlog2 : 0 ≤ Real.log 2 :=
            Real.log_nonneg (by norm_num)
          norm_num [Nat.cast_add]
          nlinarith
    calc
      ((centeredXiDyadicZeroShell
        (centeredXiZeroIndexWindow R) N).card : ℝ)
          ≤ A * r * Real.log (2 * r + 2) := hcum'
      _ ≤ A * r * (3 * (N + 1 : ℝ) * Real.log 2) := by
        gcongr
      _ = (6 * A * Real.log 2) * (2 : ℝ) ^ N *
          (N + 1 : ℝ) := by
        dsimp [r]
        rw [pow_succ]
        ring
  · have hempty :
        centeredXiDyadicZeroShell
          (centeredXiZeroIndexWindow R) N = ∅ := by
      apply Finset.eq_empty_iff_forall_notMem.mpr
      intro i hi
      have hiShell :=
        (mem_centeredXiDyadicZeroShell _ _ _).mp hi
      have hiR : R < ‖centeredXiZeroRoot i‖ := by
        simpa using i.property
      exact hr (hiR.le.trans hiShell.2.le)
    rw [hempty]
    simp
    positivity

theorem xiCenteredDyadicOrderOneZeroCount_of_cumulative
    (h : XiCenteredCumulativeZeroCount) :
    XiCenteredDyadicOrderOneZeroCount := by
  rcases h with ⟨A, R, hA, hR, hcum⟩
  exact ⟨R, 6 * A * Real.log 2, hR,
    centeredXiDyadicZeroShell_card_le_of_cumulative hA hcum⟩

theorem centeredXi_inverseSquareSummability_of_orderOneZeroCount
    (hcount : XiCenteredDyadicOrderOneZeroCount) :
    XiCenteredZeroInverseSquareSummability := by
  classical
  rcases hcount with ⟨R, A, hR, hcount⟩
  apply centeredXi_inverseSquareSummability_of_dyadicZeroCount
    (centeredXiZeroIndexWindow R)
    (centeredXiDyadicZeroShell (centeredXiZeroIndexWindow R))
    (centeredXiDyadicZeroShell_partition R hR)
    (A := A)
  · intro N i hi
    exact (mem_centeredXiDyadicZeroShell _ _ _).mp hi |>.1
  · exact hcount

theorem centeredXi_inverseSquareSummability_of_cumulativeZeroCount
    (h : XiCenteredCumulativeZeroCount) :
    XiCenteredZeroInverseSquareSummability :=
  centeredXi_inverseSquareSummability_of_orderOneZeroCount
    (xiCenteredDyadicOrderOneZeroCount_of_cumulative h)

theorem centeredXi_inverseSquareSummability_actual :
    XiCenteredZeroInverseSquareSummability :=
  centeredXi_inverseSquareSummability_of_cumulativeZeroCount
    xiCenteredCumulativeZeroCount_actual

theorem summable_positiveSquaredOrdinate_inv
    (hzero : XiCenteredCriticalZeroReality)
    (hsum : XiCenteredZeroInverseSquareSummability) :
    Summable (fun i : CenteredXiPositiveZeroIndex ↦
      |(centeredXiPositiveSquaredOrdinate i)⁻¹|) := by
  have hsub := hsum.subtype
    (p := fun i : CenteredXiZeroIndex ↦
      0 < (centeredXiZeroRoot i).im)
  apply hsub.congr
  intro i
  have hre : (centeredXiZeroRoot i.1).re = 0 :=
    centeredXiZeroRoot_re_eq_zero hzero i.1
  have hz : centeredXiZeroRoot i.1 =
      ((centeredXiZeroRoot i.1).im : ℂ) * Complex.I := by
    apply Complex.ext
    · simp [hre]
    · simp
  change ‖(centeredXiZeroRoot i.1)⁻¹‖ ^ 2 =
    |(centeredXiPositiveSquaredOrdinate i)⁻¹|
  rw [hz]
  simp only [centeredXiPositiveSquaredOrdinate, norm_inv, norm_mul,
    Complex.norm_real, Complex.norm_I, mul_one, Real.norm_eq_abs,
    abs_of_pos i.property, inv_pow]
  rw [abs_inv, abs_pow, abs_of_pos i.property]

theorem log_norm_one_add_lower_of_norm_le_half_jensen
    {w : ℂ} (hw : ‖w‖ ≤ 1 / 2) :
    -(3 / 2 : ℝ) * ‖w‖ ≤ Real.log ‖1 + w‖ := by
  rw [← Complex.log_re]
  have hlog := Complex.norm_log_one_add_half_le_self hw
  have hre :
      -‖Complex.log (1 + w)‖ ≤ (Complex.log (1 + w)).re :=
    (abs_le.mp (abs_re_le_norm (Complex.log (1 + w)))).1
  simpa only [neg_mul] using (neg_le_neg hlog).trans hre

theorem norm_tprod_one_add_lower_of_norm_le_half_jensen
    {ι : Type*} {f : ι → ℂ}
    (hsum : Summable (fun i ↦ ‖f i‖))
    (hhalf : ∀ i, ‖f i‖ ≤ 1 / 2) :
    Real.exp (-(3 / 2 : ℝ) * ∑' i, ‖f i‖) ≤
      ‖∏' i, (1 + f i)‖ := by
  have hne : ∀ i, 1 + f i ≠ 0 := by
    intro i hi
    have hfi : f i = -1 := by linear_combination hi
    have hh := hhalf i
    rw [hfi] at hh
    norm_num at hh
  have hmult : Multipliable (fun i ↦ 1 + f i) :=
    multipliable_one_add_of_summable hsum
  have hlogsum : Summable (fun i ↦ Real.log ‖1 + f i‖) :=
    hsum.summable_log_norm_one_add
  rw [hmult.norm_tprod,
    ← Real.rexp_tsum_eq_tprod
      (fun i ↦ norm_pos_iff.mpr (hne i)) hlogsum]
  apply Real.exp_le_exp.mpr
  calc
    -(3 / 2 : ℝ) * ∑' i, ‖f i‖ =
        ∑' i, -(3 / 2 : ℝ) * ‖f i‖ := by
      rw [← tsum_mul_left]
    _ ≤ ∑' i, Real.log ‖1 + f i‖ :=
      (hsum.mul_left _).tsum_le_tsum
        (fun i ↦ log_norm_one_add_lower_of_norm_le_half_jensen (hhalf i))
        hlogsum

theorem hasProdLocallyUniformlyOn_centeredXiPositiveFactors
    (hzero : XiCenteredCriticalZeroReality)
    (hsum : XiCenteredZeroInverseSquareSummability) :
    HasProdLocallyUniformlyOn
      (fun (i : CenteredXiPositiveZeroIndex) (w : ℂ) ↦
        1 + ((centeredXiPositiveSquaredOrdinate i)⁻¹ : ℝ) * w)
      (fun w : ℂ ↦ ∏' i : CenteredXiPositiveZeroIndex,
        (1 + ((centeredXiPositiveSquaredOrdinate i)⁻¹ : ℝ) * w))
      Set.univ :=
  hasProdLocallyUniformlyOn_positiveGenusZeroFactors_indexed
    centeredXiPositiveSquaredOrdinate
    (summable_positiveSquaredOrdinate_inv hzero hsum)

noncomputable def centeredXiPairedProduct (z : ℂ) : ℂ :=
  ∏' i : CenteredXiPositiveZeroIndex,
    (1 + ((centeredXiPositiveSquaredOrdinate i)⁻¹ : ℝ) * z ^ 2)

noncomputable def centeredXiPositiveZeroWindow (T : ℝ) :
    Finset CenteredXiPositiveZeroIndex := by
  classical
  exact Finset.subtype
    (fun i : CenteredXiZeroIndex ↦ 0 < (centeredXiZeroRoot i).im)
    (centeredXiZeroIndexWindow T)

@[simp]
theorem mem_centeredXiPositiveZeroWindow
    (T : ℝ) (i : CenteredXiPositiveZeroIndex) :
    i ∈ centeredXiPositiveZeroWindow T ↔
      ‖centeredXiZeroRoot i.1‖ ≤ T := by
  classical
  simp [centeredXiPositiveZeroWindow]

/-- Restricting to positive-imaginary representatives cannot increase the
multiplicity count in a centered radial window. -/
theorem centeredXiPositiveZeroWindow_card_le (T : ℝ) :
    (centeredXiPositiveZeroWindow T).card ≤
      (centeredXiZeroIndexWindow T).card := by
  classical
  unfold centeredXiPositiveZeroWindow
  rw [Finset.card_subtype]
  exact Finset.card_filter_le _ _

/-- A cumulative `O(r log r)` multiplicity bound gives the explicit dyadic
`O(2^j(j+1))` positive-window count used by the Cartan head estimate. -/
theorem exists_centeredXiPositiveZeroWindow_card_le_dyadic
    (hcount : XiCenteredCumulativeZeroCount) :
    ∃ B R : ℝ, 0 ≤ B ∧ 1 ≤ R ∧ ∀ j : ℕ,
      R ≤ (2 : ℝ) ^ (j + 2) →
      ((centeredXiPositiveZeroWindow
        ((2 : ℝ) ^ (j + 2))).card : ℝ) ≤
        B * (2 : ℝ) ^ j * (j + 1 : ℝ) := by
  obtain ⟨A, R, hA, hR, hcum⟩ := hcount
  let B := 16 * A * Real.log 2
  refine ⟨B, R, by positivity, hR, ?_⟩
  intro j hj
  have hc := hcum ((2 : ℝ) ^ (j + 2)) hj
  have hpositive :
      ((centeredXiPositiveZeroWindow
        ((2 : ℝ) ^ (j + 2))).card : ℝ) ≤
      ((centeredXiZeroIndexWindow
        ((2 : ℝ) ^ (j + 2))).card : ℝ) := by
    exact_mod_cast centeredXiPositiveZeroWindow_card_le
      ((2 : ℝ) ^ (j + 2))
  have hj2 : (2 : ℝ) ^ (j + 2) =
      4 * (2 : ℝ) ^ j := by
    rw [pow_add]
    norm_num
    ring
  have hj4 : (2 : ℝ) ^ (j + 4) =
      16 * (2 : ℝ) ^ j := by
    rw [pow_add]
    norm_num
    ring
  have harg : 2 * (2 : ℝ) ^ (j + 2) + 2 ≤
      (2 : ℝ) ^ (j + 4) := by
    rw [hj2, hj4]
    have hp : 1 ≤ (2 : ℝ) ^ j :=
      one_le_pow₀ (by norm_num)
    nlinarith
  have hlog :
      Real.log (2 * (2 : ℝ) ^ (j + 2) + 2) ≤
        4 * (j + 1 : ℝ) * Real.log 2 := by
    calc
      _ ≤ Real.log ((2 : ℝ) ^ (j + 4)) :=
        Real.log_le_log (by positivity) harg
      _ = (j + 4 : ℕ) * Real.log 2 := by
        rw [Real.log_pow]
      _ ≤ 4 * (j + 1 : ℝ) * Real.log 2 := by
        have hl : 0 ≤ Real.log 2 :=
          Real.log_nonneg (by norm_num)
        norm_num [Nat.cast_add]
        nlinarith
  calc
    ((centeredXiPositiveZeroWindow
      ((2 : ℝ) ^ (j + 2))).card : ℝ)
        ≤ ((centeredXiZeroIndexWindow
          ((2 : ℝ) ^ (j + 2))).card : ℝ) := hpositive
    _ ≤ A * (2 : ℝ) ^ (j + 2) *
          Real.log (2 * (2 : ℝ) ^ (j + 2) + 2) := hc
    _ ≤ A * (2 : ℝ) ^ (j + 2) *
        (4 * (j + 1 : ℝ) * Real.log 2) := by
      gcongr
    _ = B * (2 : ℝ) ^ j * (j + 1 : ℝ) := by
      dsimp [B]
      rw [hj2]
      ring

theorem exists_centeredXiPositiveZeroWindow_card_le_dyadic_actual :
    ∃ B R : ℝ, 0 ≤ B ∧ 1 ≤ R ∧ ∀ j : ℕ,
      R ≤ (2 : ℝ) ^ (j + 2) →
      ((centeredXiPositiveZeroWindow
        ((2 : ℝ) ^ (j + 2))).card : ℝ) ≤
        B * (2 : ℝ) ^ j * (j + 1 : ℝ) :=
  exists_centeredXiPositiveZeroWindow_card_le_dyadic
    xiCenteredCumulativeZeroCount_actual

/-- Positive-zero windows increase with the radial cutoff. -/
theorem centeredXiPositiveZeroWindow_mono :
    Monotone centeredXiPositiveZeroWindow := by
  intro T U hTU
  intro i hi
  rw [mem_centeredXiPositiveZeroWindow] at hi ⊢
  exact hi.trans hTU

/-- Dyadic positive-zero windows exhaust the multiplicity index.  This is the
precise cofinality fact needed to turn inverse-square summability into a
vanishing radial tail, without assuming a quantitative local zero count. -/
theorem tendsto_centeredXiPositiveZeroWindow_dyadic :
    Tendsto (fun j : ℕ ↦
      centeredXiPositiveZeroWindow ((2 : ℝ) ^ j)) atTop atTop := by
  apply (centeredXiPositiveZeroWindow_mono.comp
    (monotone_nat_of_le_succ fun j ↦ by
      rw [pow_succ]
      nlinarith [pow_nonneg (by norm_num : (0 : ℝ) ≤ 2) j])).tendsto_atTop_finset
  intro i
  have hpow := tendsto_pow_atTop_atTop_of_one_lt
    (by norm_num : (1 : ℝ) < 2)
  obtain ⟨j, hj⟩ := (hpow.eventually_ge_atTop
    ‖centeredXiZeroRoot i.1‖).exists
  exact ⟨j, (mem_centeredXiPositiveZeroWindow _ i).mpr hj⟩

/-- The finite-head and cardinality logarithmic losses have the common
quadratic-polynomial/geometric scale used below. -/
theorem tendsto_add_one_sq_div_two_pow_jensen :
    Tendsto (fun j : ℕ ↦
      (j + 1 : ℝ) ^ 2 / (2 : ℝ) ^ j) atTop (𝓝 0) := by
  have h2 := tendsto_pow_const_div_const_pow_of_one_lt 2
    (by norm_num : (1 : ℝ) < 2)
  have h1 := tendsto_pow_const_div_const_pow_of_one_lt 1
    (by norm_num : (1 : ℝ) < 2)
  have h0 := tendsto_pow_const_div_const_pow_of_one_lt 0
    (by norm_num : (1 : ℝ) < 2)
  convert (h2.add (h1.const_mul 2)).add h0 using 1
  · funext j
    norm_num
    ring
  · norm_num

theorem tendsto_add_one_div_two_pow_jensen :
    Tendsto (fun j : ℕ ↦
      (j + 1 : ℝ) / (2 : ℝ) ^ j) atTop (𝓝 0) := by
  have h1 := tendsto_pow_const_div_const_pow_of_one_lt 1
    (by norm_num : (1 : ℝ) < 2)
  have h0 := tendsto_pow_const_div_const_pow_of_one_lt 0
    (by norm_num : (1 : ℝ) < 2)
  convert h1.add h0 using 1
  · funext j
    norm_num
    ring
  · norm_num

/-- Crude but sufficient control of the Cartan cardinality logarithm.  The
bound only uses a cumulative cardinality majorant, so no hidden unit-shell
Riemann--von Mangoldt estimate enters the proof. -/
theorem card_mul_log_three_mul_div_sq_le_jensen
    {n B H u : ℝ}
    (hn : 0 < n) (hB : 0 ≤ B) (hH : 0 < H)
    (hu : 0 ≤ u) (hnB : n ≤ B * H * u) :
    n * Real.log (3 * n / H) / H ^ 2 ≤
      3 * B ^ 2 * u ^ 2 / H := by
  have hx : 0 ≤ 3 * n / H := by positivity
  have hlog := Real.log_le_self hx
  have hn2 : n ^ 2 ≤ (B * H * u) ^ 2 :=
    (sq_le_sq₀ hn.le
      (mul_nonneg (mul_nonneg hB hH.le) hu)).mpr hnB
  calc
    n * Real.log (3 * n / H) / H ^ 2
        ≤ n * (3 * n / H) / H ^ 2 := by gcongr
    _ = 3 * n ^ 2 / H ^ 3 := by field_simp
    _ ≤ 3 * (B * H * u) ^ 2 / H ^ 3 := by
      gcongr
    _ = 3 * B ^ 2 * u ^ 2 / H := by
      field_simp

/-- Explicit normalized selected-circle loss.  The first term covers the
finite-head root and cardinality logarithms, the second covers xi's
order-one `r log r` upper bound, and the third is the paired-product tail. -/
noncomputable def CenteredXiDyadicNormalizedLoss
    (Ksq Klin : ℝ)
    (tail : ℕ → ℝ) (j : ℕ) : ℝ :=
  Ksq * ((j + 1 : ℝ) ^ 2 / (2 : ℝ) ^ j) +
    Klin * ((j + 1 : ℝ) / (2 : ℝ) ^ j) + 6 * tail j

/-- All explicit normalized head/card-log, xi-growth, and tail losses are
`o(r²)` on dyadic scales once the inverse-square tail vanishes. -/
theorem tendsto_centeredXiDyadicNormalizedLoss
    (Ksq Klin : ℝ) {tail : ℕ → ℝ}
    (htail : Tendsto tail atTop (𝓝 0)) :
    Tendsto (CenteredXiDyadicNormalizedLoss Ksq Klin tail)
      atTop (𝓝 0) := by
  have hsq := tendsto_add_one_sq_div_two_pow_jensen.const_mul Ksq
  have hlin := tendsto_add_one_div_two_pow_jensen.const_mul Klin
  have ht := htail.const_mul 6
  change Tendsto (fun j : ℕ ↦
    Ksq * ((j + 1 : ℝ) ^ 2 / (2 : ℝ) ^ j) +
      Klin * ((j + 1 : ℝ) / (2 : ℝ) ^ j) + 6 * tail j)
    atTop (𝓝 0)
  convert (hsq.add hlin).add ht using 1
  norm_num

theorem norm_positiveSquaredOrdinate_inv_eq_root_inv_sq
    (hzero : XiCenteredCriticalZeroReality)
    (i : CenteredXiPositiveZeroIndex) :
    ‖(((centeredXiPositiveSquaredOrdinate i)⁻¹ : ℝ) : ℂ)‖ =
      ‖(centeredXiZeroRoot i.1)⁻¹‖ ^ 2 := by
  have hre : (centeredXiZeroRoot i.1).re = 0 :=
    centeredXiZeroRoot_re_eq_zero hzero i.1
  have hz : centeredXiZeroRoot i.1 =
      ((centeredXiZeroRoot i.1).im : ℂ) * Complex.I := by
    apply Complex.ext
    · simp [hre]
    · simp
  rw [hz]
  simp only [centeredXiPositiveSquaredOrdinate, norm_inv, norm_mul,
    Complex.norm_real, Complex.norm_I, mul_one, Real.norm_eq_abs,
    abs_of_pos i.property, inv_pow]
  rw [abs_of_nonneg (sq_nonneg _)]

/-- The inverse-square contribution outside a dyadic positive-zero window
tends to zero.  This is the exact normalized infinite-tail loss appearing in
the paired-product minimum-modulus estimate. -/
theorem tendsto_centeredXiPositiveZeroWindow_invSq_tail_zero
    (hzero : XiCenteredCriticalZeroReality)
    (hsum : XiCenteredZeroInverseSquareSummability) :
    Tendsto (fun j : ℕ ↦
      ∑' i : {i : CenteredXiPositiveZeroIndex //
          i ∉ centeredXiPositiveZeroWindow ((2 : ℝ) ^ j)},
        ‖(((centeredXiPositiveSquaredOrdinate i)⁻¹ : ℝ) : ℂ)‖)
      atTop (𝓝 0) := by
  let f : CenteredXiPositiveZeroIndex → ℝ := fun i ↦
    ‖(((centeredXiPositiveSquaredOrdinate i)⁻¹ : ℝ) : ℂ)‖
  have hf : Summable f := by
    simpa only [f, Complex.norm_real, Real.norm_eq_abs] using
      summable_positiveSquaredOrdinate_inv hzero hsum
  exact (tendsto_tsum_compl_atTop_zero f).comp
    tendsto_centeredXiPositiveZeroWindow_dyadic

/-- Xi-specific instantiation of the complete normalized loss. -/
theorem tendsto_centeredXiPairedProduct_normalizedLoss
    (hzero : XiCenteredCriticalZeroReality)
    (hsum : XiCenteredZeroInverseSquareSummability)
    (Ksq Klin : ℝ) :
    Tendsto
      (CenteredXiDyadicNormalizedLoss Ksq Klin fun j ↦
        ∑' i : {i : CenteredXiPositiveZeroIndex //
            i ∉ centeredXiPositiveZeroWindow ((2 : ℝ) ^ j)},
          ‖(((centeredXiPositiveSquaredOrdinate i)⁻¹ : ℝ) : ℂ)‖)
      atTop (𝓝 0) :=
  tendsto_centeredXiDyadicNormalizedLoss Ksq Klin
    (tendsto_centeredXiPositiveZeroWindow_invSq_tail_zero hzero hsum)

theorem centeredXiPairedProduct_split_window
    (hzero : XiCenteredCriticalZeroReality)
    (hsum : XiCenteredZeroInverseSquareSummability)
    (T : ℝ) (z : ℂ) :
    (∏ i ∈ centeredXiPositiveZeroWindow T,
        (1 + ((centeredXiPositiveSquaredOrdinate i)⁻¹ : ℝ) * z ^ 2)) *
      (∏' i : {i : CenteredXiPositiveZeroIndex //
          i ∉ centeredXiPositiveZeroWindow T},
        (1 + ((centeredXiPositiveSquaredOrdinate i)⁻¹ : ℝ) * z ^ 2)) =
      centeredXiPairedProduct z := by
  let f : CenteredXiPositiveZeroIndex → ℂ := fun i ↦
    1 + ((centeredXiPositiveSquaredOrdinate i)⁻¹ : ℝ) * z ^ 2
  let S : Set CenteredXiPositiveZeroIndex :=
    ↑(centeredXiPositiveZeroWindow T)
  let A : Set CenteredXiPositiveZeroIndex :=
    {i | i ∉ centeredXiPositiveZeroWindow T}
  have hAS : A = Sᶜ := by
    ext i
    simp [A, S]
  let e : A ≃ (Sᶜ : Set CenteredXiPositiveZeroIndex) :=
    Equiv.setCongr hAS
  have hs : Multipliable (f ∘ (↑) : S → ℂ) :=
    (hasProd_fintype _).multipliable
  have hdev : Summable (fun i : {i : CenteredXiPositiveZeroIndex //
      i ∉ centeredXiPositiveZeroWindow T} ↦ ‖f i - 1‖) := by
    have hbase :=
      (summable_positiveSquaredOrdinate_inv hzero hsum).subtype
        (fun i ↦ i ∉ centeredXiPositiveZeroWindow T)
    apply (hbase.mul_right ‖z ^ 2‖).congr
    intro i
    simp [f, norm_mul, Complex.norm_real, Real.norm_eq_abs]
  have hsc0 : Multipliable (fun i : {i : CenteredXiPositiveZeroIndex //
      i ∉ centeredXiPositiveZeroWindow T} ↦ f i) := by
    have h := multipliable_one_add_of_summable
      (f := fun i : {i : CenteredXiPositiveZeroIndex //
        i ∉ centeredXiPositiveZeroWindow T} ↦ f i - 1) hdev
    simpa using h
  change Multipliable (f ∘ (↑) : A → ℂ) at hsc0
  have hsc : Multipliable
      (f ∘ (↑) : (Sᶜ : Set CenteredXiPositiveZeroIndex) → ℂ) :=
    (e.multipliable_iff).mp hsc0
  have hsplit :
      (∏' i : S, f i) *
          (∏' i : (Sᶜ : Set CenteredXiPositiveZeroIndex), f i) =
        ∏' i : CenteredXiPositiveZeroIndex, f i :=
    Multipliable.tprod_mul_tprod_compl (f := f) hs hsc
  have hhead :
      (∏' i : (↑(centeredXiPositiveZeroWindow T) :
        Set CenteredXiPositiveZeroIndex), f i) =
        ∏ i ∈ centeredXiPositiveZeroWindow T, f i :=
    Finset.tprod_subtype _ _
  have htail :
      (∏' i : A, f i) =
        ∏' i : (Sᶜ : Set CenteredXiPositiveZeroIndex), f i :=
    e.tprod_eq (fun i : (Sᶜ : Set CenteredXiPositiveZeroIndex) ↦ f i)
  rw [hhead] at hsplit
  rw [← htail] at hsplit
  simpa [f, A, S, centeredXiPairedProduct] using hsplit

theorem norm_centeredXiPairedProduct_tail_lower
    (hzero : XiCenteredCriticalZeroReality)
    (hsum : XiCenteredZeroInverseSquareSummability)
    {T : ℝ} (hT : 0 < T) {z : ℂ}
    (hz : ‖z‖ ^ 2 ≤ T ^ 2 / 2) :
    Real.exp (-(3 / 2 : ℝ) * ‖z ^ 2‖ *
        ∑' i : {i : CenteredXiPositiveZeroIndex //
          i ∉ centeredXiPositiveZeroWindow T},
          |(centeredXiPositiveSquaredOrdinate i)⁻¹|) ≤
      ‖∏' i : {i : CenteredXiPositiveZeroIndex //
          i ∉ centeredXiPositiveZeroWindow T},
        (1 + ((centeredXiPositiveSquaredOrdinate i)⁻¹ : ℝ) * z ^ 2)‖ := by
  let f : {i : CenteredXiPositiveZeroIndex //
      i ∉ centeredXiPositiveZeroWindow T} → ℂ := fun i ↦
    ((centeredXiPositiveSquaredOrdinate i)⁻¹ : ℝ) * z ^ 2
  have hbase : Summable (fun i : {i : CenteredXiPositiveZeroIndex //
      i ∉ centeredXiPositiveZeroWindow T} ↦
      |(centeredXiPositiveSquaredOrdinate i)⁻¹|) :=
    (summable_positiveSquaredOrdinate_inv hzero hsum).subtype _
  have hsumf : Summable (fun i ↦ ‖f i‖) := by
    apply (hbase.mul_right ‖z ^ 2‖).congr
    intro i
    simp [f, norm_mul, Complex.norm_real, Real.norm_eq_abs]
  have hhalf : ∀ i, ‖f i‖ ≤ 1 / 2 := by
    intro i
    have hiT : T < ‖centeredXiZeroRoot i.1.1‖ := by
      simpa using i.property
    have hroot : 0 < ‖centeredXiZeroRoot i.1.1‖ :=
      norm_pos_iff.mpr (centeredXiZeroRoot_ne_zero i.1.1)
    rw [show ‖f i‖ =
        ‖(centeredXiZeroRoot i.1.1)⁻¹‖ ^ 2 * ‖z‖ ^ 2 by
      simp only [f, norm_mul, norm_pow]
      rw [norm_positiveSquaredOrdinate_inv_eq_root_inv_sq hzero i.1]]
    rw [norm_inv, inv_pow]
    rw [show (‖centeredXiZeroRoot i.1.1‖ ^ 2)⁻¹ * ‖z‖ ^ 2 =
        ‖z‖ ^ 2 / ‖centeredXiZeroRoot i.1.1‖ ^ 2 by
      rw [div_eq_mul_inv]
      ring]
    rw [div_le_iff₀ (sq_pos_of_pos hroot)]
    have hTsq :
        T ^ 2 < ‖centeredXiZeroRoot i.1.1‖ ^ 2 :=
      (sq_lt_sq₀ hT.le hroot.le).mpr hiT
    nlinarith
  have hlower :=
    norm_tprod_one_add_lower_of_norm_le_half_jensen hsumf hhalf
  have htsum : (∑' i, ‖f i‖) =
      (∑' i : {i : CenteredXiPositiveZeroIndex //
        i ∉ centeredXiPositiveZeroWindow T},
        |(centeredXiPositiveSquaredOrdinate i)⁻¹|) * ‖z ^ 2‖ := by
    rw [← tsum_mul_right]
    apply tsum_congr
    intro i
    simp [f, norm_mul, Complex.norm_real, Real.norm_eq_abs]
  rw [htsum] at hlower
  simpa only [f, mul_assoc, mul_left_comm, mul_comm] using hlower

theorem centeredXiPairedProduct_eq_zero_iff
    (hzero : XiCenteredCriticalZeroReality)
    (hsum : XiCenteredZeroInverseSquareSummability) (z : ℂ) :
    centeredXiPairedProduct z = 0 ↔
      ∃ i : CenteredXiPositiveZeroIndex,
        1 + ((centeredXiPositiveSquaredOrdinate i)⁻¹ : ℝ) * z ^ 2 = 0 := by
  let f : CenteredXiPositiveZeroIndex → ℂ := fun i ↦
    ((centeredXiPositiveSquaredOrdinate i)⁻¹ : ℝ) * z ^ 2
  have hu : Summable (fun i ↦ ‖f i‖) := by
    have hs := summable_positiveSquaredOrdinate_inv hzero hsum
    apply (hs.mul_right ‖z ^ 2‖).congr
    intro i
    simp [f, norm_mul, Complex.norm_real, Real.norm_eq_abs]
  constructor
  · intro hp
    by_contra hall
    push Not at hall
    have hne : ∏' i : CenteredXiPositiveZeroIndex, (1 + f i) ≠ 0 :=
      tprod_one_add_ne_zero_of_summable hall hu
    exact hne (by simpa [centeredXiPairedProduct, f] using hp)
  · rintro ⟨i, hi⟩
    unfold centeredXiPairedProduct
    apply tprod_of_exists_eq_zero
    exact ⟨i, hi⟩

theorem centeredXiPositiveFactor_eq_zero_iff
    (hzero : XiCenteredCriticalZeroReality)
    (i : CenteredXiPositiveZeroIndex) (z : ℂ) :
    1 + ((centeredXiPositiveSquaredOrdinate i)⁻¹ : ℝ) * z ^ 2 = 0 ↔
      z = centeredXiZeroRoot i.1 ∨ z = -centeredXiZeroRoot i.1 := by
  let γ : ℝ := (centeredXiZeroRoot i.1).im
  have hγ : γ ≠ 0 := ne_of_gt i.property
  have hroot : centeredXiZeroRoot i.1 = (γ : ℂ) * Complex.I := by
    apply Complex.ext
    · simp [γ, centeredXiZeroRoot_re_eq_zero hzero i.1]
    · simp [γ]
  rw [hroot]
  change 1 + (((γ ^ 2)⁻¹ : ℝ) : ℂ) * z ^ 2 = 0 ↔
    z = (γ : ℂ) * Complex.I ∨ z = -((γ : ℂ) * Complex.I)
  push_cast
  constructor
  · intro hz
    have hquad : z ^ 2 + (γ : ℂ) ^ 2 = 0 := by
      field_simp [hγ] at hz
      linear_combination hz
    have hfac :
        (z - (γ : ℂ) * Complex.I) *
          (z + (γ : ℂ) * Complex.I) = 0 := by
      rw [show
        (z - (γ : ℂ) * Complex.I) * (z + (γ : ℂ) * Complex.I) =
          z ^ 2 + (γ : ℂ) ^ 2 by
            calc
              _ = z ^ 2 - ((γ : ℂ) * Complex.I) ^ 2 := by ring
              _ = z ^ 2 + (γ : ℂ) ^ 2 := by
                rw [mul_pow, Complex.I_sq]
                ring]
      exact hquad
    rcases mul_eq_zero.mp hfac with h | h
    · left
      exact sub_eq_zero.mp h
    · right
      exact eq_neg_of_add_eq_zero_left h
  · rintro (rfl | rfl) <;> field_simp [hγ] <;>
      rw [Complex.I_sq] <;> ring

theorem analyticOrderAt_centeredXiPositiveFactor_of_eq_zero
    (i : CenteredXiPositiveZeroIndex) (z : ℂ)
    (hz : 1 + ((centeredXiPositiveSquaredOrdinate i)⁻¹ : ℝ) *
      z ^ 2 = 0) :
    analyticOrderAt
      (fun s : ℂ ↦
        1 + ((centeredXiPositiveSquaredOrdinate i)⁻¹ : ℝ) * s ^ 2) z = 1 := by
  let a : ℂ := ((centeredXiPositiveSquaredOrdinate i)⁻¹ : ℝ)
  let f : ℂ → ℂ := fun s ↦ 1 + a * s ^ 2
  have ha : a ≠ 0 := by
    dsimp [a]
    exact Complex.ofReal_ne_zero.mpr
      (inv_ne_zero (ne_of_gt (centeredXiPositiveSquaredOrdinate_pos i)))
  have hz0 : z ≠ 0 := by
    intro h
    rw [h] at hz
    norm_num at hz
  have hf : AnalyticAt ℂ f z := by
    dsimp [f]
    fun_prop
  have hfz : f z = 0 := by
    simpa [f, a] using hz
  have hderiv : deriv f z = 2 * a * z := by
    dsimp [f]
    convert HasDerivAt.deriv
      (((hasDerivAt_id z).pow 2).const_mul a |>.const_add 1) using 1 <;>
      simp [Function.id_def] <;> ring
  have hderiv0 : deriv f z ≠ 0 := by
    rw [hderiv]
    exact mul_ne_zero (mul_ne_zero (by norm_num) ha) hz0
  change analyticOrderAt f z = 1
  exact hf.analyticOrderAt_eq_one_of_zero_deriv_ne_zero hfz hderiv0

theorem centeredXiPairedProduct_zero_set_eq
    (hzero : XiCenteredCriticalZeroReality)
    (hsum : XiCenteredZeroInverseSquareSummability) (z : ℂ) :
    centeredXiPairedProduct z = 0 ↔ xiJensenEntire z = 0 := by
  rw [centeredXiPairedProduct_eq_zero_iff hzero hsum]
  constructor
  · rintro ⟨i, hi⟩
    rcases (centeredXiPositiveFactor_eq_zero_iff hzero i z).mp hi with h | h
    · rw [h]
      exact i.1.1.property
    · rw [h, xiJensenEntire_neg]
      exact i.1.1.property
  · intro hz
    obtain ⟨j, hj⟩ := (exists_centeredXiZeroRoot_iff z).mpr hz
    rcases centeredXiZeroIndex_positive_or_neg_positive hzero j with hjpos | hjneg
    · let i : CenteredXiPositiveZeroIndex := ⟨j, hjpos⟩
      refine ⟨i, (centeredXiPositiveFactor_eq_zero_iff hzero i z).mpr ?_⟩
      exact Or.inl hj.symm
    · let i : CenteredXiPositiveZeroIndex :=
        ⟨centeredXiZeroIndexNeg j, hjneg⟩
      refine ⟨i, (centeredXiPositiveFactor_eq_zero_iff hzero i z).mpr ?_⟩
      right
      simp only [i, centeredXiZeroRoot_indexNeg, neg_neg]
      exact hj.symm

noncomputable def centeredXiPairedFiber
    (z : ℂ) : Finset CenteredXiPositiveZeroIndex := by
  classical
  exact Finset.subtype
    (fun i : CenteredXiZeroIndex ↦ 0 < (centeredXiZeroRoot i).im)
    (centeredXiZeroFiber z ∪ centeredXiZeroFiber (-z))

@[simp]
theorem mem_centeredXiPairedFiber
    (z : ℂ) (i : CenteredXiPositiveZeroIndex) :
    i ∈ centeredXiPairedFiber z ↔
      centeredXiZeroRoot i.1 = z ∨
        centeredXiZeroRoot i.1 = -z := by
  classical
  simp [centeredXiPairedFiber, mem_centeredXiZeroFiber]

theorem card_centeredXiPairedFiber
    (hzero : XiCenteredCriticalZeroReality) (z : ℂ) :
    (centeredXiPairedFiber z).card =
      centeredXiZeroMultiplicity z := by
  classical
  by_cases hz : xiJensenEntire z = 0
  · have hzre : z.re = 0 := hzero z hz
    have hzim : z.im ≠ 0 := xiJensenEntire_zero_im_ne_zero hz
    rcases lt_or_gt_of_ne hzim with hneg | hpos
    · have heq : centeredXiPairedFiber z =
          Finset.subtype
            (fun i : CenteredXiZeroIndex ↦
              0 < (centeredXiZeroRoot i).im)
            (centeredXiZeroFiber (-z)) := by
        ext i
        simp only [mem_centeredXiPairedFiber, Finset.mem_subtype,
          mem_centeredXiZeroFiber]
        constructor
        · rintro (hi | hi)
          · exfalso
            have := i.property
            rw [hi] at this
            linarith
          · exact hi
        · exact Or.inr
      rw [heq, Finset.card_subtype]
      have hfilter : (centeredXiZeroFiber (-z)).filter
          (fun i ↦ 0 < (centeredXiZeroRoot i).im) =
          centeredXiZeroFiber (-z) := by
        apply Finset.filter_eq_self.mpr
        intro i hi
        have hir := (mem_centeredXiZeroFiber (-z) i).mp hi
        rw [hir]
        simp only [map_neg, Complex.neg_im]
        linarith
      rw [hfilter, card_centeredXiZeroFiber,
        centeredXiZeroMultiplicity_neg]
    · have heq : centeredXiPairedFiber z =
          Finset.subtype
            (fun i : CenteredXiZeroIndex ↦
              0 < (centeredXiZeroRoot i).im)
            (centeredXiZeroFiber z) := by
        ext i
        simp only [mem_centeredXiPairedFiber, Finset.mem_subtype,
          mem_centeredXiZeroFiber]
        constructor
        · rintro (hi | hi)
          · exact hi
          · exfalso
            have := i.property
            rw [hi] at this
            simp only [map_neg, Complex.neg_im] at this
            linarith
        · exact Or.inl
      rw [heq, Finset.card_subtype]
      have hfilter : (centeredXiZeroFiber z).filter
          (fun i ↦ 0 < (centeredXiZeroRoot i).im) =
          centeredXiZeroFiber z := by
        apply Finset.filter_eq_self.mpr
        intro i hi
        have hir := (mem_centeredXiZeroFiber z i).mp hi
        rw [hir]
        exact hpos
      rw [hfilter, card_centeredXiZeroFiber]
  · have hmult : centeredXiZeroMultiplicity z = 0 :=
      Nat.eq_zero_of_not_pos (by
        intro hp
        exact hz ((centeredXiZeroMultiplicity_pos_iff z).mp hp))
    rw [hmult]
    apply Finset.card_eq_zero.mpr
    rw [Finset.eq_empty_iff_forall_notMem]
    intro i hi
    rcases (mem_centeredXiPairedFiber z i).mp hi with hir | hir
    · apply hz
      rw [← hir]
      exact i.1.1.property
    · apply hz
      have hp := i.1.1.property
      change xiJensenEntire (centeredXiZeroRoot i.1) = 0 at hp
      rw [hir, xiJensenEntire_neg] at hp
      exact hp

set_option maxHeartbeats 800000 in
theorem analyticOrderAt_centeredXiPairedProduct
    (hzero : XiCenteredCriticalZeroReality)
    (hsum : XiCenteredZeroInverseSquareSummability) (z : ℂ) :
    analyticOrderAt centeredXiPairedProduct z =
      (centeredXiPairedFiber z).card := by
  classical
  have hPdiff : Differentiable ℂ centeredXiPairedProduct := by
    let L : ℂ → ℂ := fun w ↦
      ∏' i : CenteredXiPositiveZeroIndex,
        (1 + ((centeredXiPositiveSquaredOrdinate i)⁻¹ : ℝ) * w)
    have hLprod :
        HasProdLocallyUniformlyOn
          (fun (i : CenteredXiPositiveZeroIndex) (w : ℂ) ↦
            1 + ((centeredXiPositiveSquaredOrdinate i)⁻¹ : ℝ) * w)
          L Set.univ :=
      hasProdLocallyUniformlyOn_centeredXiPositiveFactors hzero hsum
    have hLdiff : Differentiable ℂ L := by
      apply differentiableOn_univ.mp
      apply hLprod.differentiableOn
      · filter_upwards [] with t
        apply Differentiable.differentiableOn
        fun_prop
      · exact isOpen_univ
    have hcomp : Differentiable ℂ (fun z : ℂ ↦ L (z ^ 2)) := by
      fun_prop
    unfold centeredXiPairedProduct
    exact hcomp
  let J := centeredXiPairedFiber z
  change analyticOrderAt centeredXiPairedProduct z = J.card
  by_cases hJ : J = ∅
  · rw [hJ]
    simp only [Finset.card_empty, Nat.cast_zero]
    rw [(hPdiff.analyticAt z).analyticOrderAt_eq_zero]
    intro hp
    have hz := (centeredXiPairedProduct_eq_zero_iff hzero hsum z).mp hp
    rcases hz with ⟨i, hi⟩
    have : i ∈ J := by
      rw [mem_centeredXiPairedFiber]
      rcases (centeredXiPositiveFactor_eq_zero_iff hzero i z).mp hi with h | h
      · exact Or.inl h.symm
      · exact Or.inr (by rw [h]; simp)
    simpa [hJ] using this
  · obtain ⟨i0, hi0⟩ := Finset.nonempty_iff_ne_empty.mpr hJ
    let κ := {i : CenteredXiPositiveZeroIndex // i ∉ J}
    let rootC : κ → ℝ := fun i ↦ centeredXiPositiveSquaredOrdinate i
    let L : ℂ → ℂ := fun w ↦
      ∏' i : κ, (1 + ((rootC i)⁻¹ : ℝ) * w)
    let Q : ℂ → ℂ := fun s ↦ L (s ^ 2)
    have hsC : Summable (fun i : κ ↦ |(rootC i)⁻¹|) := by
      simpa [rootC, κ, Function.comp_def] using
        (summable_positiveSquaredOrdinate_inv hzero hsum).subtype
          (fun i ↦ i ∉ J)
    have hLprod : HasProdLocallyUniformlyOn
        (fun (i : κ) (w : ℂ) ↦ 1 + ((rootC i)⁻¹ : ℝ) * w)
        L Set.univ :=
      hasProdLocallyUniformlyOn_positiveGenusZeroFactors_indexed rootC hsC
    have hLdiff : Differentiable ℂ L := by
      apply differentiableOn_univ.mp
      apply hLprod.differentiableOn
      · filter_upwards [] with t
        apply Differentiable.differentiableOn
        fun_prop
      · exact isOpen_univ
    have hQdiff : Differentiable ℂ Q := by
      dsimp [Q]
      fun_prop
    have hQA : AnalyticOnNhd ℂ Q Set.univ :=
      fun s _ ↦ hQdiff.analyticAt s
    have hQz : Q z ≠ 0 := by
      let f : κ → ℂ := fun i ↦ ((rootC i)⁻¹ : ℝ) * z ^ 2
      have hu : Summable (fun i ↦ ‖f i‖) := by
        apply (hsC.mul_right ‖z ^ 2‖).congr
        intro i
        simp [f, norm_mul, Complex.norm_real, Real.norm_eq_abs]
      have hf : ∀ i, 1 + f i ≠ 0 := by
        intro i hiz
        apply i.property
        rw [mem_centeredXiPairedFiber]
        rcases (centeredXiPositiveFactor_eq_zero_iff hzero i.1 z).mp
          (by simpa [f, rootC] using hiz) with h | h
        · exact Or.inl h.symm
        · exact Or.inr (by rw [h]; simp)
      exact tprod_one_add_ne_zero_of_summable hf hu
    let B : ℂ → ℂ := fun s ↦
      1 + ((centeredXiPositiveSquaredOrdinate i0)⁻¹ : ℝ) * s ^ 2
    have hi0z : B z = 0 := by
      apply (centeredXiPositiveFactor_eq_zero_iff hzero i0 z).mpr
      rcases (mem_centeredXiPairedFiber z i0).mp hi0 with h | h
      · exact Or.inl h.symm
      · exact Or.inr (by rw [h]; simp)
    have hz0 : z ≠ 0 := by
      intro hz
      rw [hz] at hi0z
      norm_num [B] at hi0z
    have hfactor : ∀ i ∈ J, (fun s : ℂ ↦
        1 + ((centeredXiPositiveSquaredOrdinate i)⁻¹ : ℝ) * s ^ 2) = B := by
      intro i hi
      have hiz :
          1 + ((centeredXiPositiveSquaredOrdinate i)⁻¹ : ℝ) * z ^ 2 = 0 :=
        (centeredXiPositiveFactor_eq_zero_iff hzero i z).mpr (by
          rcases (mem_centeredXiPairedFiber z i).mp hi with h | h
          · exact Or.inl h.symm
          · exact Or.inr (by rw [h]; simp))
      have hcoef :
          (((centeredXiPositiveSquaredOrdinate i)⁻¹ : ℝ) : ℂ) =
            (((centeredXiPositiveSquaredOrdinate i0)⁻¹ : ℝ) : ℂ) := by
        have hzsq : z ^ 2 ≠ 0 := pow_ne_zero 2 hz0
        apply (mul_right_cancel₀ hzsq)
        linear_combination hiz - hi0z
      funext s
      simp only [B]
      rw [hcoef]
    have hfinite : (fun s : ℂ ↦ ∏ i ∈ J,
        (1 + ((centeredXiPositiveSquaredOrdinate i)⁻¹ : ℝ) * s ^ 2)) =
        B ^ J.card := by
      funext s
      rw [Pi.pow_apply, ← Finset.prod_const]
      apply Finset.prod_congr rfl
      intro i hi
      exact congrFun (hfactor i hi) s
    have hfull : centeredXiPairedProduct =
        (fun s : ℂ ↦ ∏ i ∈ J,
          (1 + ((centeredXiPositiveSquaredOrdinate i)⁻¹ : ℝ) * s ^ 2)) * Q := by
      funext s
      have hsprod := (J.hasProd (fun i ↦
        1 + ((centeredXiPositiveSquaredOrdinate i)⁻¹ : ℝ) * s ^ 2)).mul_compl
          (hLprod.hasProd (Set.mem_univ (s ^ 2)))
      exact hsprod.tprod_eq
    have hBA : AnalyticAt ℂ B z := by
      dsimp [B]
      fun_prop
    have hBorder : analyticOrderAt B z = 1 := by
      exact analyticOrderAt_centeredXiPositiveFactor_of_eq_zero i0 z hi0z
    rw [hfull, hfinite, analyticOrderAt_mul
      (hBA.pow J.card) (hQA z (Set.mem_univ z)),
      analyticOrderAt_pow hBA, hBorder,
      (hQA z (Set.mem_univ z)).analyticOrderAt_eq_zero.mpr hQz]
    simp [J]

/-- Exact divisor statement still required from the infinite-product
multiplicity theorem. -/
def XiCenteredPairedProductDivisorEquality : Prop :=
  MeromorphicOn.divisor centeredXiPairedProduct Set.univ =
    centeredXiZeroDivisor

/-- Exact order-one/minimum-modulus conclusion for the cancelled canonical
quotient. -/
def XiCenteredPairedProductAffineRigidity : Prop :=
  ∃ a b : ℂ, ∀ z : ℂ, xiJensenEntire z =
    Complex.exp (a + b * z) * centeredXiPairedProduct z

/-- The sole remaining Hadamard/minimum-modulus step after exact divisor
matching. -/
def XiCenteredDivisorQuotientRigidity : Prop :=
  XiCenteredPairedProductDivisorEquality →
    XiCenteredPairedProductAffineRigidity

theorem differentiable_centeredXiPairedProduct
    (hzero : XiCenteredCriticalZeroReality)
    (hsum : XiCenteredZeroInverseSquareSummability) :
    Differentiable ℂ centeredXiPairedProduct := by
  let Q : ℂ → ℂ := fun w ↦
    ∏' i : CenteredXiPositiveZeroIndex,
      (1 + ((centeredXiPositiveSquaredOrdinate i)⁻¹ : ℝ) * w)
  have hQprod :
      HasProdLocallyUniformlyOn
        (fun (i : CenteredXiPositiveZeroIndex) (w : ℂ) ↦
          1 + ((centeredXiPositiveSquaredOrdinate i)⁻¹ : ℝ) * w)
        Q Set.univ :=
    hasProdLocallyUniformlyOn_centeredXiPositiveFactors hzero hsum
  have hQdiffOn : DifferentiableOn ℂ Q Set.univ := by
    apply hQprod.differentiableOn
    · filter_upwards [] with s
      apply Differentiable.differentiableOn
      fun_prop
    · exact isOpen_univ
  have hQdiff : Differentiable ℂ Q :=
    differentiableOn_univ.mp hQdiffOn
  have hcomp : Differentiable ℂ (fun z : ℂ ↦ Q (z ^ 2)) := by
    fun_prop
  unfold centeredXiPairedProduct
  exact hcomp

theorem centeredXiPairedProduct_divisor_eq
    (hzero : XiCenteredCriticalZeroReality)
    (hsum : XiCenteredZeroInverseSquareSummability) :
    MeromorphicOn.divisor centeredXiPairedProduct Set.univ =
      centeredXiZeroDivisor := by
  have hPdiff := differentiable_centeredXiPairedProduct hzero hsum
  have hPM : MeromorphicOn centeredXiPairedProduct Set.univ :=
    fun z _ ↦ (hPdiff.analyticAt z).meromorphicAt
  ext z
  rw [MeromorphicOn.divisor_apply hPM (Set.mem_univ z),
    (hPdiff.analyticAt z).meromorphicOrderAt_eq,
    analyticOrderAt_centeredXiPairedProduct hzero hsum z,
    card_centeredXiPairedFiber hzero z,
    centeredXiZeroDivisor_apply]
  simp

theorem xiCenteredPairedProductDivisorEquality
    (hzero : XiCenteredCriticalZeroReality)
    (hsum : XiCenteredZeroInverseSquareSummability) :
    XiCenteredPairedProductDivisorEquality :=
  centeredXiPairedProduct_divisor_eq hzero hsum

noncomputable def centeredXiCancelledQuotient : ℂ → ℂ :=
  toMeromorphicNFOn
    (xiJensenEntire * centeredXiPairedProduct⁻¹) Set.univ

theorem centeredXiCancelledQuotient_analytic_ne_zero
    (hzero : XiCenteredCriticalZeroReality)
    (hsum : XiCenteredZeroInverseSquareSummability) :
    AnalyticOnNhd ℂ centeredXiCancelledQuotient Set.univ ∧
      ∀ z, centeredXiCancelledQuotient z ≠ 0 := by
  let xi := xiJensenEntire
  let P := centeredXiPairedProduct
  let q := xi * P⁻¹
  have hxiA : AnalyticOnNhd ℂ xi Set.univ :=
    fun z _ ↦ differentiable_xiJensenEntire.analyticAt z
  have hPA : AnalyticOnNhd ℂ P Set.univ :=
    fun z _ ↦ (differentiable_centeredXiPairedProduct hzero hsum).analyticAt z
  have hxiM : MeromorphicOn xi Set.univ :=
    fun z hz ↦ (hxiA z hz).meromorphicAt
  have hPM : MeromorphicOn P Set.univ :=
    fun z hz ↦ (hPA z hz).meromorphicAt
  have hxiFinite : ∀ z, meromorphicOrderAt xi z ≠ ⊤ := by
    intro z
    rw [(hxiA z (Set.mem_univ z)).meromorphicOrderAt_eq]
    have ho : analyticOrderAt xi z ≠ ⊤ := by
      change analyticOrderAt xiJensenEntire z ≠ ⊤
      simpa [centeredXiZeroOrder] using centeredXiZeroOrder_ne_top z
    simpa using ho
  have hP0 : P 0 ≠ 0 := by
    simp [P, centeredXiPairedProduct]
  have hPFinite : ∀ z, meromorphicOrderAt P z ≠ ⊤ := by
    intro z
    rw [(hPA z (Set.mem_univ z)).meromorphicOrderAt_eq]
    have ho := hPA.analyticOrderAt_ne_top_of_isPreconnected
      (x := (0 : ℂ)) (y := z) isPreconnected_univ
      (Set.mem_univ 0) (Set.mem_univ z) (by
        rw [(hPA 0 (Set.mem_univ 0)).analyticOrderAt_eq_zero.mpr hP0]
        exact ENat.zero_ne_top)
    simpa using ho
  have hq : MeromorphicOn q Set.univ := hxiM.mul hPM.inv
  have hqdiv : MeromorphicOn.divisor q Set.univ = 0 := by
    dsimp [q]
    rw [hxiM.divisor_mul hPM.inv
      (fun z _ ↦ hxiFinite z)
      (fun z _ ↦ by simpa [meromorphicOrderAt_inv] using hPFinite z),
      MeromorphicOn.divisor_inv]
    change MeromorphicOn.divisor xiJensenEntire Set.univ -
      MeromorphicOn.divisor centeredXiPairedProduct Set.univ = 0
    rw [centeredXiPairedProduct_divisor_eq hzero hsum]
    change centeredXiZeroDivisor - centeredXiZeroDivisor = 0
    abel
  have hnf : MeromorphicNFOn centeredXiCancelledQuotient Set.univ := by
    change MeromorphicNFOn (toMeromorphicNFOn q Set.univ) Set.univ
    exact meromorphicNFOn_toMeromorphicNFOn q Set.univ
  have hqdiv' :
      MeromorphicOn.divisor centeredXiCancelledQuotient Set.univ = 0 := by
    change MeromorphicOn.divisor
      (toMeromorphicNFOn q Set.univ) Set.univ = 0
    rw [hq.divisor_of_toMeromorphicNFOn, hqdiv]
  have han : AnalyticOnNhd ℂ centeredXiCancelledQuotient Set.univ := by
    rw [← hnf.divisor_nonneg_iff_analyticOnNhd, hqdiv']
  refine ⟨han, ?_⟩
  intro z
  rw [← (hnf (Set.mem_univ z)).meromorphicOrderAt_eq_zero_iff]
  have happ := MeromorphicOn.divisor_apply hnf.meromorphicOn
    (Set.mem_univ z)
  rw [hqdiv'] at happ
  have hor :
      meromorphicOrderAt centeredXiCancelledQuotient z = 0 ∨
        meromorphicOrderAt centeredXiCancelledQuotient z = ⊤ := by
    simpa using happ.symm
  apply hor.resolve_right
  change meromorphicOrderAt (toMeromorphicNFOn q Set.univ) z ≠ ⊤
  rw [meromorphicOrderAt_toMeromorphicNFOn hq (Set.mem_univ z),
    meromorphicOrderAt_mul (hxiM z (Set.mem_univ z))
      (hPM.inv z (Set.mem_univ z)), meromorphicOrderAt_inv]
  exact WithTop.add_ne_top.mpr
    ⟨hxiFinite z, by simpa using hPFinite z⟩

theorem centeredXiCancelledQuotient_eq_div_of_product_ne
    (hzero : XiCenteredCriticalZeroReality)
    (hsum : XiCenteredZeroInverseSquareSummability)
    {z : ℂ} (hPz : centeredXiPairedProduct z ≠ 0) :
    centeredXiCancelledQuotient z =
      xiJensenEntire z / centeredXiPairedProduct z := by
  have hxiA := differentiable_xiJensenEntire.analyticAt z
  have hPA :=
    (differentiable_centeredXiPairedProduct hzero hsum).analyticAt z
  have hxiM : MeromorphicOn xiJensenEntire Set.univ :=
    fun w _ ↦ differentiable_xiJensenEntire.analyticAt w |>.meromorphicAt
  have hPM : MeromorphicOn centeredXiPairedProduct Set.univ :=
    fun w _ ↦
      (differentiable_centeredXiPairedProduct hzero hsum).analyticAt w
        |>.meromorphicAt
  have hqM : MeromorphicOn
      (xiJensenEntire * centeredXiPairedProduct⁻¹) Set.univ :=
    hxiM.mul hPM.inv
  have hrawA : AnalyticAt ℂ
      (xiJensenEntire * centeredXiPairedProduct⁻¹) z :=
    hxiA.mul (hPA.inv hPz)
  have hev : centeredXiCancelledQuotient =ᶠ[𝓝 z]
      xiJensenEntire * centeredXiPairedProduct⁻¹ := by
    exact (toMeromorphicNFOn_eq_toMeromorphicNFAt_on_nhds
      hqM (Set.mem_univ z)).trans
        (Filter.Eventually.of_forall (fun w ↦ by
          rw [toMeromorphicNFAt_eq_self.2 hrawA.meromorphicNFAt]))
  simpa [div_eq_mul_inv] using hev.self_of_nhds

theorem norm_centeredXiCancelledQuotient_le_div_of_product_lower
    (hzero : XiCenteredCriticalZeroReality)
    (hsum : XiCenteredZeroInverseSquareSummability)
    {z : ℂ} {L : ℝ} (hL : 0 < L)
    (hlower : L ≤ ‖centeredXiPairedProduct z‖) :
    ‖centeredXiCancelledQuotient z‖ ≤ ‖xiJensenEntire z‖ / L := by
  have hPz : centeredXiPairedProduct z ≠ 0 := by
    intro hp
    rw [hp, norm_zero] at hlower
    linarith
  rw [centeredXiCancelledQuotient_eq_div_of_product_ne
    hzero hsum hPz, norm_div]
  exact div_le_div_of_nonneg_left (norm_nonneg _)
    hL hlower

theorem exists_analytic_log_jensen
    {g : ℂ → ℂ} (hg : AnalyticOnNhd ℂ g Set.univ)
    (hne : ∀ z, g z ≠ 0) :
    ∃ f : ℂ → ℂ, AnalyticOnNhd ℂ f Set.univ ∧
      ∀ z, Complex.exp (f z) = g z := by
  have hUc : IsSimplyConnected (Set.univ : Set ℂ) := by
    change SimplyConnectedSpace (Set.univ : Set ℂ)
    exact (Homeomorph.Set.univ ℂ).toHomotopyEquiv.simplyConnectedSpace
  have hnon : 0 ∉ g '' (Set.univ : Set ℂ) := by
    rintro ⟨z, hz, hzero⟩
    exact hne z hzero
  obtain ⟨f, hfcont, hf⟩ :=
    Complex.exists_continuousOn_eqOn_exp_comp hUc isOpen_univ
      hg.continuousOn hnon
  have hfcont' : Continuous f := continuousOn_univ.mp hfcont
  have hfexp : ∀ z, Complex.exp (f z) = g z := by
    intro z
    simpa [Function.comp_apply] using hf (Set.mem_univ z)
  refine ⟨f, ?_, hfexp⟩
  intro z hz
  have hgz : g z ≠ 0 := by
    rw [← hfexp]
    exact Complex.exp_ne_zero _
  have hratio : AnalyticAt ℂ (fun w ↦ g w / g z) z :=
    (hg z (Set.mem_univ z)).div_const
  have hratio1 : g z / g z = 1 := div_self hgz
  have hrhs : AnalyticAt ℂ
      (fun w ↦ f z + Complex.log (g w / g z)) z :=
    analyticAt_const.add (hratio.clog (by simp [hratio1]))
  have hclose : ∀ᶠ w in 𝓝 z,
      dist (f w) (f z) < Real.pi / 2 :=
    hfcont'.continuousAt.tendsto.eventually
      (Metric.ball_mem_nhds _ (half_pos Real.pi_pos))
  have heq : f =ᶠ[𝓝 z]
      fun w ↦ f z + Complex.log (g w / g z) := by
    filter_upwards [hclose] with w hw
    have hnorm : ‖f w - f z‖ < Real.pi / 2 := by
      simpa [dist_eq_norm] using hw
    have him : |(f w - f z).im| < Real.pi / 2 :=
      (abs_im_le_norm (f w - f z)).trans_lt hnorm
    have hlower : -(Real.pi) < (f w - f z).im := by
      linarith [neg_lt_of_abs_lt him, Real.pi_pos]
    have hupper : (f w - f z).im ≤ Real.pi := by
      linarith [lt_of_abs_lt him, Real.pi_pos]
    have hexpdiff :
        Complex.exp (f w - f z) = g w / g z := by
      rw [Complex.exp_sub, hfexp, hfexp]
    calc
      f w = f z + (f w - f z) := by ring
      _ = f z + Complex.log (Complex.exp (f w - f z)) := by
        rw [Complex.log_exp hlower hupper]
      _ = f z + Complex.log (g w / g z) := by rw [hexpdiff]
  exact hrhs.congr heq.symm

def SubquadraticNormGrowthJensen (h : ℂ → ℂ) : Prop :=
  ∀ ε : ℝ, 0 < ε → ∃ R : ℝ, 0 ≤ R ∧
    ∀ z : ℂ, R ≤ ‖z‖ → ‖h z‖ ≤ ε * ‖z‖ ^ 2

def SubquadraticRealPartGrowthJensen (h : ℂ → ℂ) : Prop :=
  ∀ ε : ℝ, 0 < ε → ∃ R : ℝ, 0 ≤ R ∧
    ∀ r : ℝ, R ≤ r → ∀ z ∈ Metric.ball (0 : ℂ) r,
      (h z).re ≤ ε * r ^ 2

def SubquadraticLogNormGrowthJensen (g : ℂ → ℂ) : Prop :=
  ∀ ε : ℝ, 0 < ε → ∃ R : ℝ, 0 ≤ R ∧
    ∀ r : ℝ, R ≤ r → ∀ z ∈ Metric.ball (0 : ℂ) r,
      Real.log ‖g z‖ ≤ ε * r ^ 2

def SubquadraticBoundaryLogNormGrowthJensen (g : ℂ → ℂ) : Prop :=
  ∀ ε : ℝ, 0 < ε → ∃ r0 : ℝ, 0 ≤ r0 ∧
    ∀ r : ℝ, r0 ≤ r → ∃ R : ℝ, r ≤ R ∧ 0 < R ∧
      ∀ z ∈ Metric.sphere (0 : ℂ) R,
        Real.log ‖g z‖ ≤ ε * r ^ 2

/-- Dyadic selected-circle form naturally produced by the explicit paired
product head/tail decomposition. -/
def DyadicSubquadraticBoundaryLogNormGrowthJensen (g : ℂ → ℂ) : Prop :=
  ∀ ε : ℝ, 0 < ε → ∀ᶠ j : ℕ in atTop,
    ∃ R : ℝ, (2 : ℝ) ^ j ≤ R ∧ 0 < R ∧
      ∀ z ∈ Metric.sphere (0 : ℂ) R,
        Real.log ‖g z‖ ≤ ε * ((2 : ℝ) ^ j) ^ 2

/-- The next dyadic scale is at most twice an arbitrary large radius, so
dyadic selected circles imply the every-radius boundary condition. -/
theorem subquadraticBoundaryLogNormGrowth_of_dyadic_jensen
    {g : ℂ → ℂ}
    (h : DyadicSubquadraticBoundaryLogNormGrowthJensen g) :
    SubquadraticBoundaryLogNormGrowthJensen g := by
  intro ε hε
  have he4 : 0 < ε / 4 := by positivity
  have hev := h (ε / 4) he4
  rw [Filter.eventually_atTop] at hev
  obtain ⟨N, hN⟩ := hev
  let r0 : ℝ := (2 : ℝ) ^ N
  refine ⟨r0, by positivity, ?_⟩
  intro r hr
  have hr1 : 1 ≤ r :=
    (one_le_pow₀ (by norm_num)).trans hr
  obtain ⟨k, hk, _huniq⟩ :=
    existsUnique_dyadic_interval_jensen r hr1
  let j := k + 1
  have hNj : N ≤ j := by
    by_contra hnot
    have hjN : j < N := lt_of_not_ge hnot
    have hp : (2 : ℝ) ^ j ≤ (2 : ℝ) ^ N :=
      pow_le_pow_right₀ (by norm_num) hjN.le
    have hu := hk.2
    dsimp [j] at hp ⊢
    linarith
  obtain ⟨R, hHR, hR, hsphere⟩ := hN j hNj
  refine ⟨R, hk.2.le.trans hHR, hR, ?_⟩
  intro z hz
  have hH2r : (2 : ℝ) ^ j ≤ 2 * r := by
    dsimp [j]
    rw [pow_succ]
    nlinarith [hk.1]
  exact (hsphere z hz).trans (by
    have hsq : ((2 : ℝ) ^ j) ^ 2 ≤ (2 * r) ^ 2 :=
      (sq_le_sq₀ (by positivity) (by positivity)).mpr hH2r
    nlinarith)

/-- Any selected-circle quotient norm estimate controlled by the explicit
normalized head/card-log, xi-growth, and paired-tail loss yields dyadic
subquadratic boundary growth. -/
theorem dyadicSubquadraticBoundaryLogNormGrowth_of_centeredXiLoss
    {g : ℂ → ℂ} (hg0 : ∀ z, g z ≠ 0)
    (hzero : XiCenteredCriticalZeroReality)
    (hsum : XiCenteredZeroInverseSquareSummability)
    (Ksq Klin : ℝ)
    (hcircle : ∀ᶠ j : ℕ in atTop,
      ∃ R : ℝ, (2 : ℝ) ^ j ≤ R ∧ 0 < R ∧
        ∀ z ∈ Metric.sphere (0 : ℂ) R,
          ‖g z‖ ≤ Real.exp (
            CenteredXiDyadicNormalizedLoss Ksq Klin
              (fun j ↦
                ∑' i : {i : CenteredXiPositiveZeroIndex //
                    i ∉ centeredXiPositiveZeroWindow ((2 : ℝ) ^ j)},
                  ‖(((centeredXiPositiveSquaredOrdinate i)⁻¹ : ℝ) : ℂ)‖)
              j * ((2 : ℝ) ^ j) ^ 2)) :
    DyadicSubquadraticBoundaryLogNormGrowthJensen g := by
  intro ε hε
  have hloss :=
    tendsto_centeredXiPairedProduct_normalizedLoss hzero hsum Ksq Klin
  have hevent : ∀ᶠ j : ℕ in atTop,
      CenteredXiDyadicNormalizedLoss Ksq Klin
        (fun j ↦
          ∑' i : {i : CenteredXiPositiveZeroIndex //
              i ∉ centeredXiPositiveZeroWindow ((2 : ℝ) ^ j)},
            ‖(((centeredXiPositiveSquaredOrdinate i)⁻¹ : ℝ) : ℂ)‖)
        j < ε :=
    (tendsto_order.1 hloss).2 ε hε
  filter_upwards [hcircle, hevent] with j hj hmajor
  obtain ⟨R, hHR, hR, hbound⟩ := hj
  refine ⟨R, hHR, hR, ?_⟩
  intro z hz
  calc
    Real.log ‖g z‖ ≤ Real.log (Real.exp (
        CenteredXiDyadicNormalizedLoss Ksq Klin
          (fun j ↦
            ∑' i : {i : CenteredXiPositiveZeroIndex //
                i ∉ centeredXiPositiveZeroWindow ((2 : ℝ) ^ j)},
              ‖(((centeredXiPositiveSquaredOrdinate i)⁻¹ : ℝ) : ℂ)‖)
          j * ((2 : ℝ) ^ j) ^ 2)) :=
      Real.log_le_log (norm_pos_iff.mpr (hg0 z)) (hbound z hz)
    _ = CenteredXiDyadicNormalizedLoss Ksq Klin
          (fun j ↦
            ∑' i : {i : CenteredXiPositiveZeroIndex //
                i ∉ centeredXiPositiveZeroWindow ((2 : ℝ) ^ j)},
              ‖(((centeredXiPositiveSquaredOrdinate i)⁻¹ : ℝ) : ℂ)‖)
          j * ((2 : ℝ) ^ j) ^ 2 := Real.log_exp _
    _ ≤ ε * ((2 : ℝ) ^ j) ^ 2 := by
      gcongr

theorem subquadraticLogNormGrowth_of_boundary_jensen
    {g : ℂ → ℂ} (hgA : AnalyticOnNhd ℂ g Set.univ)
    (hg0 : ∀ z, g z ≠ 0)
    (hboundary : SubquadraticBoundaryLogNormGrowthJensen g) :
    SubquadraticLogNormGrowthJensen g := by
  intro ε hε
  obtain ⟨r0, hr0, hbound⟩ := hboundary ε hε
  refine ⟨r0, hr0, ?_⟩
  intro r hr z hz
  obtain ⟨R, hrR, hR, hsphere⟩ := hbound r hr
  have hboundaryNorm :
      ∀ w ∈ frontier (Metric.ball (0 : ℂ) R),
        ‖g w‖ ≤ Real.exp (ε * r ^ 2) := by
    intro w hw
    have hws : w ∈ Metric.sphere (0 : ℂ) R :=
      Metric.frontier_ball_subset_sphere hw
    rw [← Real.exp_log (norm_pos_iff.mpr (hg0 w))]
    exact Real.exp_le_exp.mpr (hsphere w hws)
  have hzClosure : z ∈ closure (Metric.ball (0 : ℂ) R) := by
    rw [closure_ball _ hR.ne']
    have hzNorm : ‖z‖ < R := by
      have hz' := hz
      simp only [Metric.mem_ball, dist_zero_right] at hz'
      exact hz'.trans_le hrR
    simpa [Metric.mem_closedBall, dist_zero_right] using hzNorm.le
  have hnorm : ‖g z‖ ≤ Real.exp (ε * r ^ 2) :=
    Complex.norm_le_of_forall_mem_frontier_norm_le
      Metric.isBounded_ball
      ((hgA.differentiableOn.mono (Set.subset_univ _)).diffContOnCl)
      hboundaryNorm hzClosure
  calc
    Real.log ‖g z‖ ≤ Real.log (Real.exp (ε * r ^ 2)) :=
      Real.log_le_log (norm_pos_iff.mpr (hg0 z)) hnorm
    _ = ε * r ^ 2 := Real.log_exp _

/-- Finite exceptional disks whose total radial width leaves room for a
centered circle, together with the quotient bound off those disks. -/
def CartanExceptionalDiskLogBoundJensen (g : ℂ → ℂ) : Prop :=
  ∀ ε : ℝ, 0 < ε → ∃ r0 : ℝ, 0 ≤ r0 ∧
    ∀ r : ℝ, r0 ≤ r → ∃ n : ℕ, ∃ center : Fin n → ℂ,
      ∃ radius : Fin n → ℝ,
        (∀ i, 0 ≤ radius i) ∧
        2 * ∑ i, radius i < r ∧
        ∀ z : ℂ, r ≤ ‖z‖ → ‖z‖ ≤ 2 * r →
          (∀ i, z ∉ Metric.closedBall (center i) (radius i)) →
            Real.log ‖g z‖ ≤ ε * r ^ 2

theorem exists_radius_avoiding_exceptionalIntervals_jensen
    {ι : Type*} (s : Finset ι) (center : ι → ℂ)
    (radius : ι → ℝ)
    (hradius : ∀ i ∈ s, 0 ≤ radius i)
    {a b : ℝ}
    (hwidth : 2 * ∑ i ∈ s, radius i < b - a) :
    ∃ R ∈ Set.Icc a b,
      ∀ i ∈ s, R ∉ Set.Icc (‖center i‖ - radius i)
        (‖center i‖ + radius i) := by
  classical
  let U : Set ℝ := ⋃ i ∈ s,
    Set.Icc (‖center i‖ - radius i)
      (‖center i‖ + radius i)
  have hmeasureU : volume U ≤
      ENNReal.ofReal (2 * ∑ i ∈ s, radius i) := by
    calc
      volume U ≤ ∑ i ∈ s,
          volume (Set.Icc (‖center i‖ - radius i)
            (‖center i‖ + radius i)) :=
        measure_biUnion_finset_le s _
      _ = ∑ i ∈ s, ENNReal.ofReal (2 * radius i) := by
        apply Finset.sum_congr rfl
        intro i hi
        rw [Real.volume_Icc]
        congr 1
        ring
      _ = ENNReal.ofReal (2 * ∑ i ∈ s, radius i) := by
        rw [← ENNReal.ofReal_sum_of_nonneg
          (f := fun i ↦ 2 * radius i)]
        · congr 1
          rw [Finset.mul_sum]
        · intro i hi
          exact mul_nonneg (by norm_num) (hradius i hi)
  have hnot : ¬ Set.Icc a b ⊆ U := by
    intro hsubset
    have hvol : ENNReal.ofReal (b - a) ≤
        ENNReal.ofReal (2 * ∑ i ∈ s, radius i) := by
      rw [← Real.volume_Icc]
      exact (measure_mono hsubset).trans hmeasureU
    have hnonneg : 0 ≤ 2 * ∑ i ∈ s, radius i :=
      mul_nonneg (by norm_num)
        (Finset.sum_nonneg fun i hi ↦ hradius i hi)
    have := (ENNReal.ofReal_le_ofReal_iff hnonneg).mp hvol
    linarith
  obtain ⟨R, hR, hRU⟩ := Set.not_subset.mp hnot
  refine ⟨R, hR, ?_⟩
  intro i hi hRi
  apply hRU
  exact Set.mem_iUnion_of_mem i (Set.mem_iUnion_of_mem hi hRi)

theorem norm_centeredXiPositiveFactor_lower_of_radius_not_mem
    (hzero : XiCenteredCriticalZeroReality)
    (i : CenteredXiPositiveZeroIndex)
    {δ R : ℝ} (hδ : 0 ≤ δ) {z : ℂ}
    (hz : z ∈ Metric.sphere (0 : ℂ) R)
    (havoid : R ∉ Set.Icc
      (‖centeredXiZeroRoot i.1‖ - δ)
      (‖centeredXiZeroRoot i.1‖ + δ)) :
    (δ / ‖centeredXiZeroRoot i.1‖) ^ 2 ≤
      ‖1 + ((centeredXiPositiveSquaredOrdinate i)⁻¹ : ℝ) * z ^ 2‖ := by
  let root := centeredXiZeroRoot i.1
  let γ := root.im
  have hγ : 0 < γ := i.property
  have hroot : root = (γ : ℂ) * Complex.I := by
    apply Complex.ext
    · simp [root, γ, centeredXiZeroRoot_re_eq_zero hzero i.1]
    · simp [γ]
  have hzNorm : ‖z‖ = R := by
    simpa [Metric.mem_sphere, dist_zero_right] using hz
  have hrad : δ ≤ |R - ‖root‖| := by
    simp only [Set.mem_Icc, not_and_or, not_le] at havoid
    rcases havoid with hleft | hright
    · rw [abs_of_neg (by linarith)]
      linarith
    · rw [abs_of_pos (by linarith)]
      linarith
  have hd₁ : δ ≤ ‖z - root‖ := by
    exact hrad.trans (by
      simpa [hzNorm] using abs_norm_sub_norm_le z root)
  have hd₂ : δ ≤ ‖z + root‖ := by
    exact hrad.trans (by
      simpa [hzNorm] using abs_norm_sub_norm_le z (-root))
  have hfactor :
      1 + ((centeredXiPositiveSquaredOrdinate i)⁻¹ : ℝ) * z ^ 2 =
        (z - root) * (z + root) / (γ : ℂ) ^ 2 := by
    rw [hroot]
    change 1 + (((γ ^ 2)⁻¹ : ℝ) : ℂ) * z ^ 2 =
      (z - (γ : ℂ) * Complex.I) *
        (z + (γ : ℂ) * Complex.I) / (γ : ℂ) ^ 2
    push_cast
    field_simp [ne_of_gt hγ]
    calc
      (γ : ℂ) ^ 2 + z ^ 2 = z ^ 2 + (γ : ℂ) ^ 2 := by ring
      _ = (z - (γ : ℂ) * Complex.I) *
          (z + (γ : ℂ) * Complex.I) := by
        rw [show
          (z - (γ : ℂ) * Complex.I) *
              (z + (γ : ℂ) * Complex.I) =
            z ^ 2 - ((γ : ℂ) * Complex.I) ^ 2 by ring,
          mul_pow, Complex.I_sq]
        ring
  rw [hfactor, norm_div, norm_mul, norm_pow, Complex.norm_real,
    Real.norm_eq_abs, abs_of_pos hγ, div_pow]
  have hrootNorm : ‖root‖ = γ := by
    rw [hroot, norm_mul, Complex.norm_real, Complex.norm_I,
      Real.norm_eq_abs, abs_of_pos hγ, mul_one]
  change δ ^ 2 / ‖root‖ ^ 2 ≤
    ‖z - root‖ * ‖z + root‖ / γ ^ 2
  rw [hrootNorm]
  rw [div_le_div_iff_of_pos_right (sq_pos_of_pos hγ)]
  rw [sq]
  calc
    δ * δ ≤ ‖z - root‖ * δ :=
      mul_le_mul_of_nonneg_right hd₁ hδ
    _ ≤ ‖z - root‖ * ‖z + root‖ :=
      mul_le_mul_of_nonneg_left hd₂ (norm_nonneg _)

theorem exists_circle_norm_centeredXiPairedProduct_head_lower
    (hzero : XiCenteredCriticalZeroReality)
    {T H a b : ℝ}
    (hs : (centeredXiPositiveZeroWindow T).Nonempty)
    (hH : 0 < H) (hwidth : H ≤ b - a) :
    ∃ R ∈ Set.Icc a b, ∀ z ∈ Metric.sphere (0 : ℂ) R,
      (∏ i ∈ centeredXiPositiveZeroWindow T,
        ((H / (3 * (centeredXiPositiveZeroWindow T).card)) /
          ‖centeredXiZeroRoot i.1‖) ^ 2) ≤
      ‖∏ i ∈ centeredXiPositiveZeroWindow T,
        (1 + ((centeredXiPositiveSquaredOrdinate i)⁻¹ : ℝ) * z ^ 2)‖ := by
  let s := centeredXiPositiveZeroWindow T
  let δ : ℝ := H / (3 * s.card)
  have hδ : 0 ≤ δ := div_nonneg hH.le (by positivity)
  have hsum : 2 * ∑ _i ∈ s, δ < H := by
    simp only [Finset.sum_const, nsmul_eq_mul]
    have hcard : (0 : ℝ) < s.card := by
      exact_mod_cast hs.card_pos
    dsimp [δ]
    field_simp
    nlinarith
  obtain ⟨R, hR, havoid⟩ :=
    exists_radius_avoiding_exceptionalIntervals_jensen
      s (fun i ↦ centeredXiZeroRoot i.1) (fun _ ↦ δ)
      (fun _ _ ↦ hδ) (a := a) (b := b)
      (hsum.trans_le hwidth)
  refine ⟨R, hR, ?_⟩
  intro z hz
  rw [norm_prod]
  apply Finset.prod_le_prod
  · intro i hi
    positivity
  · intro i hi
    change (δ / ‖centeredXiZeroRoot i.1‖) ^ 2 ≤
      ‖1 + ((centeredXiPositiveSquaredOrdinate i)⁻¹ : ℝ) * z ^ 2‖
    exact norm_centeredXiPositiveFactor_lower_of_radius_not_mem
      hzero i hδ hz (havoid i hi)

theorem exists_circle_norm_centeredXiPairedProduct_lower
    (hzero : XiCenteredCriticalZeroReality)
    (hsum : XiCenteredZeroInverseSquareSummability)
    {T H a b : ℝ}
    (hs : (centeredXiPositiveZeroWindow T).Nonempty)
    (hT : 0 < T) (hH : 0 < H) (ha : 0 ≤ a)
    (hwidth : H ≤ b - a) (hbT : b ^ 2 ≤ T ^ 2 / 2) :
    ∃ R ∈ Set.Icc a b, ∀ z ∈ Metric.sphere (0 : ℂ) R,
      (∏ i ∈ centeredXiPositiveZeroWindow T,
          ((H / (3 * (centeredXiPositiveZeroWindow T).card)) /
            ‖centeredXiZeroRoot i.1‖) ^ 2) *
        Real.exp (-(3 / 2 : ℝ) * ‖z ^ 2‖ *
          ∑' i : {i : CenteredXiPositiveZeroIndex //
            i ∉ centeredXiPositiveZeroWindow T},
            |(centeredXiPositiveSquaredOrdinate i)⁻¹|) ≤
        ‖centeredXiPairedProduct z‖ := by
  obtain ⟨R, hR, hhead⟩ :=
    exists_circle_norm_centeredXiPairedProduct_head_lower
      hzero hs hH hwidth
  refine ⟨R, hR, ?_⟩
  intro z hz
  have hzNorm : ‖z‖ = R := by
    simpa [Metric.mem_sphere, dist_zero_right] using hz
  have hR0 : 0 ≤ R := ha.trans hR.1
  have hzT : ‖z‖ ^ 2 ≤ T ^ 2 / 2 := by
    rw [hzNorm]
    have hb0 : 0 ≤ b := hR0.trans hR.2
    exact ((sq_le_sq₀ hR0 hb0).mpr hR.2).trans hbT
  have htail :=
    norm_centeredXiPairedProduct_tail_lower hzero hsum hT hzT
  rw [← centeredXiPairedProduct_split_window hzero hsum T z, norm_mul]
  exact mul_le_mul (hhead z hz) htail
    (Real.exp_pos _).le (norm_nonneg _)

theorem exists_circle_norm_centeredXiPairedProduct_lower_of_cumulative
    (hzero : XiCenteredCriticalZeroReality)
    (hcount : XiCenteredCumulativeZeroCount)
    {T H a b : ℝ}
    (hs : (centeredXiPositiveZeroWindow T).Nonempty)
    (hT : 0 < T) (hH : 0 < H) (ha : 0 ≤ a)
    (hwidth : H ≤ b - a) (hbT : b ^ 2 ≤ T ^ 2 / 2) :
    ∃ R ∈ Set.Icc a b, ∀ z ∈ Metric.sphere (0 : ℂ) R,
      (∏ i ∈ centeredXiPositiveZeroWindow T,
          ((H / (3 * (centeredXiPositiveZeroWindow T).card)) /
            ‖centeredXiZeroRoot i.1‖) ^ 2) *
        Real.exp (-(3 / 2 : ℝ) * ‖z ^ 2‖ *
          ∑' i : {i : CenteredXiPositiveZeroIndex //
            i ∉ centeredXiPositiveZeroWindow T},
            |(centeredXiPositiveSquaredOrdinate i)⁻¹|) ≤
        ‖centeredXiPairedProduct z‖ :=
  exists_circle_norm_centeredXiPairedProduct_lower hzero
    (centeredXi_inverseSquareSummability_of_cumulativeZeroCount hcount)
    hs hT hH ha hwidth hbT

theorem exists_circle_norm_centeredXiPairedProduct_lower_actual
    (hzero : XiCenteredCriticalZeroReality)
    {T H a b : ℝ}
    (hs : (centeredXiPositiveZeroWindow T).Nonempty)
    (hT : 0 < T) (hH : 0 < H) (ha : 0 ≤ a)
    (hwidth : H ≤ b - a) (hbT : b ^ 2 ≤ T ^ 2 / 2) :
    ∃ R ∈ Set.Icc a b, ∀ z ∈ Metric.sphere (0 : ℂ) R,
      (∏ i ∈ centeredXiPositiveZeroWindow T,
          ((H / (3 * (centeredXiPositiveZeroWindow T).card)) /
            ‖centeredXiZeroRoot i.1‖) ^ 2) *
        Real.exp (-(3 / 2 : ℝ) * ‖z ^ 2‖ *
          ∑' i : {i : CenteredXiPositiveZeroIndex //
            i ∉ centeredXiPositiveZeroWindow T},
            |(centeredXiPositiveSquaredOrdinate i)⁻¹|) ≤
        ‖centeredXiPairedProduct z‖ :=
  exists_circle_norm_centeredXiPairedProduct_lower_of_cumulative
    hzero xiCenteredCumulativeZeroCount_actual
    hs hT hH ha hwidth hbT

/-- Uniform version of the finite-head factor in the selected-circle bound.
It is weaker than the root-by-root product but has an exactly computable
logarithm. -/
theorem centeredXiPositiveZeroWindow_uniform_head_le
    {T H : ℝ} (hT : 0 < T) (hH : 0 < H)
    (hs : (centeredXiPositiveZeroWindow T).Nonempty) :
    (H / (3 * (centeredXiPositiveZeroWindow T).card * T)) ^
        (2 * (centeredXiPositiveZeroWindow T).card) ≤
      ∏ i ∈ centeredXiPositiveZeroWindow T,
        ((H / (3 * (centeredXiPositiveZeroWindow T).card)) /
          ‖centeredXiZeroRoot i.1‖) ^ 2 := by
  let s := centeredXiPositiveZeroWindow T
  have hn : (0 : ℝ) < s.card := by exact_mod_cast hs.card_pos
  have hterm : ∀ i ∈ s,
      (H / (3 * s.card * T)) ^ 2 ≤
        ((H / (3 * s.card)) / ‖centeredXiZeroRoot i.1‖) ^ 2 := by
    intro i hi
    have hroot0 : 0 < ‖centeredXiZeroRoot i.1‖ :=
      norm_pos_iff.mpr (centeredXiZeroRoot_ne_zero i.1)
    have hrootT : ‖centeredXiZeroRoot i.1‖ ≤ T :=
      (mem_centeredXiPositiveZeroWindow T i).mp hi
    apply pow_le_pow_left₀ (by positivity)
    rw [div_div]
    apply div_le_div_of_nonneg_left hH.le
      (mul_pos (mul_pos (by norm_num) hn) hroot0)
    exact mul_le_mul_of_nonneg_left hrootT
      (mul_nonneg (by norm_num) hn.le)
  calc
    (H / (3 * s.card * T)) ^ (2 * s.card) =
        ∏ _i ∈ s, (H / (3 * s.card * T)) ^ 2 := by
      simp only [Finset.prod_const, nsmul_eq_mul]
      rw [← pow_mul]
    _ ≤ ∏ i ∈ s,
        ((H / (3 * s.card)) /
          ‖centeredXiZeroRoot i.1‖) ^ 2 :=
      Finset.prod_le_prod (fun _ _ ↦ by positivity) hterm

/-- Explicit quotient exponent on a selected circle. -/
noncomputable def centeredXiQuotientCircleExponent
    (C T H : ℝ) (z : ℂ) : ℝ :=
  C * ‖z‖ * Real.log (‖z‖ + 2) +
    2 * (centeredXiPositiveZeroWindow T).card * Real.log T +
    2 * (centeredXiPositiveZeroWindow T).card *
      Real.log (3 * (centeredXiPositiveZeroWindow T).card / H) +
    (3 / 2 : ℝ) * ‖z ^ 2‖ *
      ∑' i : {i : CenteredXiPositiveZeroIndex //
        i ∉ centeredXiPositiveZeroWindow T},
        |(centeredXiPositiveSquaredOrdinate i)⁻¹|

/-- Global xi growth plus the paired-product minimum modulus gives the
concrete cancelled-quotient estimate with every head, cardinality-log, and
tail loss displayed. -/
theorem exists_circle_norm_centeredXiCancelledQuotient_le_explicit
    (hzero : XiCenteredCriticalZeroReality)
    {C R0 T H a b : ℝ}
    (hxi : ∀ z : ℂ, R0 ≤ ‖z‖ →
      ‖xiJensenEntire z‖ ≤
        Real.exp (C * ‖z‖ * Real.log (‖z‖ + 2)))
    (hs : (centeredXiPositiveZeroWindow T).Nonempty)
    (hT : 0 < T) (hH : 0 < H) (ha : 0 ≤ a)
    (hR0 : R0 ≤ a) (hwidth : H ≤ b - a)
    (hbT : b ^ 2 ≤ T ^ 2 / 2) :
    ∃ R ∈ Set.Icc a b, ∀ z ∈ Metric.sphere (0 : ℂ) R,
      ‖centeredXiCancelledQuotient z‖ ≤
        Real.exp (centeredXiQuotientCircleExponent C T H z) := by
  obtain ⟨R, hR, hproduct⟩ :=
    exists_circle_norm_centeredXiPairedProduct_lower_actual
      hzero hs hT hH ha hwidth hbT
  refine ⟨R, hR, ?_⟩
  intro z hz
  let n : ℕ := (centeredXiPositiveZeroWindow T).card
  let x : ℝ := H / (3 * n * T)
  let tail : ℝ := ∑' i : {i : CenteredXiPositiveZeroIndex //
      i ∉ centeredXiPositiveZeroWindow T},
    |(centeredXiPositiveSquaredOrdinate i)⁻¹|
  have hn : 0 < n := hs.card_pos
  have hx : 0 < x := by dsimp [x, n]; positivity
  have htail0 : 0 ≤ tail := tsum_nonneg fun _ ↦ abs_nonneg _
  have hzNorm : ‖z‖ = R := by
    simpa [Metric.mem_sphere, dist_zero_right] using hz
  have hzR0 : R0 ≤ ‖z‖ := by
    rw [hzNorm]
    exact hR0.trans hR.1
  have hhead :=
    centeredXiPositiveZeroWindow_uniform_head_le hT hH hs
  have hlower :
      x ^ (2 * n) *
          Real.exp (-(3 / 2 : ℝ) * ‖z ^ 2‖ * tail) ≤
        ‖centeredXiPairedProduct z‖ := by
    apply le_trans ?_ (hproduct z hz)
    apply mul_le_mul_of_nonneg_right
    · simpa [x, n] using hhead
    · positivity
  have hL : 0 < x ^ (2 * n) *
      Real.exp (-(3 / 2 : ℝ) * ‖z ^ 2‖ * tail) := by positivity
  have hq := norm_centeredXiCancelledQuotient_le_div_of_product_lower
    hzero centeredXi_inverseSquareSummability_actual hL hlower
  have hupper := hxi z hzR0
  apply hq.trans
  calc
    ‖xiJensenEntire z‖ /
        (x ^ (2 * n) *
          Real.exp (-(3 / 2 : ℝ) * ‖z ^ 2‖ * tail))
        ≤ Real.exp (C * ‖z‖ * Real.log (‖z‖ + 2)) /
            (x ^ (2 * n) *
              Real.exp (-(3 / 2 : ℝ) * ‖z ^ 2‖ * tail)) := by
          exact div_le_div_of_nonneg_right hupper hL.le
    _ = Real.exp (centeredXiQuotientCircleExponent C T H z) := by
      have hlogx :
          Real.log x = -Real.log T -
            Real.log (3 * n / H) := by
        rw [show x = (T * (3 * n / H))⁻¹ by
          dsimp [x]
          field_simp
          <;> ring]
        rw [Real.log_inv, Real.log_mul (ne_of_gt hT) (by positivity)]
        ring
      rw [div_eq_iff hL.ne']
      rw [← Real.exp_log hx, ← Real.exp_nat_mul,
        ← Real.exp_add, ← Real.exp_add]
      dsimp [centeredXiQuotientCircleExponent, n, tail]
      rw [hlogx]
      congr 1
      push_cast
      ring

/-- The explicit circle exponent is dominated by the normalized loss already
proved to vanish. -/
noncomputable def CenteredXiDyadicShiftedLoss
    (Ksq Klin : ℝ) (j : ℕ) : ℝ :=
  Ksq * ((j + 1 : ℝ) ^ 2 / (2 : ℝ) ^ j) +
    Klin * ((j + 1 : ℝ) / (2 : ℝ) ^ j) +
    6 * ∑' i : {i : CenteredXiPositiveZeroIndex //
      i ∉ centeredXiPositiveZeroWindow ((2 : ℝ) ^ (j + 2))},
      ‖(((centeredXiPositiveSquaredOrdinate i)⁻¹ : ℝ) : ℂ)‖

theorem tendsto_centeredXiDyadicShiftedLoss
    (hzero : XiCenteredCriticalZeroReality) (Ksq Klin : ℝ) :
    Tendsto (CenteredXiDyadicShiftedLoss Ksq Klin) atTop (𝓝 0) := by
  have htail :=
    (tendsto_centeredXiPositiveZeroWindow_invSq_tail_zero
      hzero centeredXi_inverseSquareSummability_actual).comp
      (tendsto_add_atTop_nat 2)
  have h :=
    tendsto_centeredXiDyadicNormalizedLoss Ksq Klin htail
  convert h using 1
  funext j
  simp [CenteredXiDyadicShiftedLoss,
    CenteredXiDyadicNormalizedLoss, Function.comp_apply,
    Complex.norm_real, Real.norm_eq_abs]

theorem centeredXiQuotientCircleExponent_le_dyadicLoss
    {B C : ℝ} (hB : 0 ≤ B) (hC : 0 ≤ C) (j : ℕ)
    (hcard :
      ((centeredXiPositiveZeroWindow
        ((2 : ℝ) ^ (j + 2))).card : ℝ) ≤
        B * (2 : ℝ) ^ j * (j + 1 : ℝ))
    (hwindow : (centeredXiPositiveZeroWindow
      ((2 : ℝ) ^ (j + 2))).Nonempty)
    (z : ℂ) (hzlo : (2 : ℝ) ^ j ≤ ‖z‖)
    (hzhi : ‖z‖ ≤ (2 : ℝ) ^ (j + 1)) :
    centeredXiQuotientCircleExponent C
        ((2 : ℝ) ^ (j + 2)) ((2 : ℝ) ^ j) z ≤
      CenteredXiDyadicShiftedLoss
        (4 * B + 6 * B ^ 2) (4 * C) j *
          ((2 : ℝ) ^ j) ^ 2 := by
  let H : ℝ := (2 : ℝ) ^ j
  let T : ℝ := (2 : ℝ) ^ (j + 2)
  let n : ℝ := (centeredXiPositiveZeroWindow T).card
  let tail : ℝ := ∑' i : {i : CenteredXiPositiveZeroIndex //
      i ∉ centeredXiPositiveZeroWindow T},
    |(centeredXiPositiveSquaredOrdinate i)⁻¹|
  have hH : 0 < H := by positivity
  have hn : 0 < n := by
    dsimp [n, T]
    exact_mod_cast hwindow.card_pos
  have htail0 : 0 ≤ tail := tsum_nonneg fun _ ↦ abs_nonneg _
  have hT : T = 4 * H := by
    dsimp [T, H]
    rw [pow_add]
    norm_num
    ring
  have hz2 : ‖z‖ ≤ 2 * H := by
    dsimp [H]
    rw [pow_succ] at hzhi
    nlinarith
  have hcard' : n ≤ B * H * (j + 1 : ℝ) := by
    simpa [n, T, H] using hcard
  have hlogT : Real.log T ≤ 2 * (j + 1 : ℝ) := by
    have hlog2 : Real.log (2 : ℝ) ≤ 1 := by
      have hh := Real.log_le_sub_one_of_pos (x := 2) (by norm_num)
      norm_num at hh ⊢
      exact hh
    calc
      Real.log T = (j + 2 : ℕ) * Real.log 2 := by
        dsimp [T]
        rw [Real.log_pow]
      _ ≤ (j + 2 : ℝ) := by
        have hjnon : 0 ≤ (j + 2 : ℝ) := by positivity
        have hh := mul_le_mul_of_nonneg_left hlog2 hjnon
        norm_num [Nat.cast_add] at hh ⊢
        exact hh
      _ ≤ 2 * (j + 1 : ℝ) := by
        nlinarith [Nat.cast_nonneg (α := ℝ) j]
  have hlogz : Real.log (‖z‖ + 2) ≤ 2 * (j + 1 : ℝ) := by
    have harg : ‖z‖ + 2 ≤ 4 * H := by
      have hH1 : 1 ≤ H := one_le_pow₀ (by norm_num)
      nlinarith [hz2]
    exact (Real.log_le_log (by positivity) harg).trans
      (by simpa [hT] using hlogT)
  have hroot : 2 * n * Real.log T / H ^ 2 ≤
      4 * B * ((j + 1 : ℝ) ^ 2 / H) := by
    have hlogT0 : 0 ≤ Real.log T :=
      Real.log_nonneg (by dsimp [T]; exact one_le_pow₀ (by norm_num))
    rw [div_le_iff₀ (sq_pos_of_pos hH)]
    calc
      2 * n * Real.log T ≤
          2 * (B * H * (j + 1 : ℝ)) *
            (2 * (j + 1 : ℝ)) := by gcongr
      _ = (4 * B * ((j + 1 : ℝ) ^ 2 / H)) * H ^ 2 := by
        field_simp
        ring
  have hcardlog :
      2 * n * Real.log (3 * n / H) / H ^ 2 ≤
        6 * B ^ 2 * (j + 1 : ℝ) ^ 2 / H := by
    have h := card_mul_log_three_mul_div_sq_le_jensen
      hn hB hH (by positivity) hcard'
    calc
      2 * n * Real.log (3 * n / H) / H ^ 2 =
          2 * (n * Real.log (3 * n / H) / H ^ 2) := by ring
      _ ≤ 2 * (3 * B ^ 2 * (j + 1 : ℝ) ^ 2 / H) :=
        mul_le_mul_of_nonneg_left h (by norm_num)
      _ = 6 * B ^ 2 * (j + 1 : ℝ) ^ 2 / H := by ring
  have hxi : C * ‖z‖ * Real.log (‖z‖ + 2) / H ^ 2 ≤
      4 * C * ((j + 1 : ℝ) / H) := by
    have hlogz0 : 0 ≤ Real.log (‖z‖ + 2) :=
      Real.log_nonneg (by linarith [norm_nonneg z])
    rw [div_le_iff₀ (sq_pos_of_pos hH)]
    calc
      C * ‖z‖ * Real.log (‖z‖ + 2) ≤
          C * (2 * H) * Real.log (‖z‖ + 2) := by
            exact mul_le_mul_of_nonneg_right
              (mul_le_mul_of_nonneg_left hz2 hC) hlogz0
      _ ≤ C * (2 * H) * (2 * (j + 1 : ℝ)) := by
            exact mul_le_mul_of_nonneg_left hlogz
              (mul_nonneg hC (by positivity))
      _ = (4 * C * ((j + 1 : ℝ) / H)) * H ^ 2 := by
        field_simp
        ring
  have htail : (3 / 2 : ℝ) * ‖z ^ 2‖ * tail / H ^ 2 ≤
      6 * tail := by
    rw [norm_pow]
    have hzsq : ‖z‖ ^ 2 ≤ (2 * H) ^ 2 :=
      (sq_le_sq₀ (norm_nonneg _) (by positivity)).mpr hz2
    calc
      (3 / 2 : ℝ) * ‖z‖ ^ 2 * tail / H ^ 2
          ≤ (3 / 2 : ℝ) * (2 * H) ^ 2 * tail / H ^ 2 := by gcongr
      _ = 6 * tail := by field_simp; ring
  have hnorm :
      centeredXiQuotientCircleExponent C T H z / H ^ 2 ≤
        CenteredXiDyadicShiftedLoss
          (4 * B + 6 * B ^ 2) (4 * C) j := by
    dsimp [centeredXiQuotientCircleExponent]
    change (C * ‖z‖ * Real.log (‖z‖ + 2) +
        2 * n * Real.log T + 2 * n * Real.log (3 * n / H) +
        (3 / 2 : ℝ) * ‖z ^ 2‖ * tail) / H ^ 2 ≤ _
    calc
      _ = C * ‖z‖ * Real.log (‖z‖ + 2) / H ^ 2 +
          2 * n * Real.log T / H ^ 2 +
          2 * n * Real.log (3 * n / H) / H ^ 2 +
          (3 / 2 : ℝ) * ‖z ^ 2‖ * tail / H ^ 2 := by ring
      _ ≤ 4 * C * ((j + 1 : ℝ) / H) +
          4 * B * ((j + 1 : ℝ) ^ 2 / H) +
          6 * B ^ 2 * ((j + 1 : ℝ) ^ 2 / H) +
          6 * tail := by
            ring_nf at hxi hroot hcardlog htail ⊢
            linarith
      _ = CenteredXiDyadicShiftedLoss
          (4 * B + 6 * B ^ 2) (4 * C) j := by
        unfold CenteredXiDyadicShiftedLoss
        dsimp [H, T, tail]
        simp only [Complex.norm_real, Real.norm_eq_abs]
        ring
  change centeredXiQuotientCircleExponent C T H z ≤
    _ * H ^ 2
  exact (div_le_iff₀ (sq_pos_of_pos hH)).mp hnorm

/-- With at least one positive representative, the actual xi growth and count
estimates instantiate the concrete dyadic quotient-circle bound. -/
theorem exists_eventually_circle_norm_centeredXiCancelledQuotient_le_shiftedLoss
    (hzero : XiCenteredCriticalZeroReality)
    [Nonempty CenteredXiPositiveZeroIndex] :
    ∃ Ksq Klin : ℝ, ∀ᶠ j : ℕ in atTop,
      ∃ R : ℝ, (2 : ℝ) ^ j ≤ R ∧ 0 < R ∧
        ∀ z ∈ Metric.sphere (0 : ℂ) R,
          ‖centeredXiCancelledQuotient z‖ ≤
            Real.exp (CenteredXiDyadicShiftedLoss Ksq Klin j *
              ((2 : ℝ) ^ j) ^ 2) := by
  obtain ⟨C, R0, hC, _hR0, hxi⟩ :=
    exists_xiJensenEntire_orderOneGrowthBound
  obtain ⟨B, RB, hB, _hRB, hcard⟩ :=
    exists_centeredXiPositiveZeroWindow_card_le_dyadic_actual
  let i : CenteredXiPositiveZeroIndex := Classical.choice inferInstance
  refine ⟨4 * B + 6 * B ^ 2, 4 * C, ?_⟩
  have hp := tendsto_pow_atTop_atTop_of_one_lt
    (by norm_num : (1 : ℝ) < 2)
  have hlarge : ∀ᶠ j : ℕ in atTop,
      max (max R0 RB) ‖centeredXiZeroRoot i.1‖ ≤ (2 : ℝ) ^ j :=
    hp.eventually (eventually_ge_atTop _)
  filter_upwards [hlarge] with j hj
  let H : ℝ := (2 : ℝ) ^ j
  let T : ℝ := (2 : ℝ) ^ (j + 2)
  have hH : 0 < H := by positivity
  have hR0H : R0 ≤ H :=
    (le_max_left R0 RB).trans (le_max_left (max R0 RB) _ |>.trans hj)
  have hRBH : RB ≤ H :=
    (le_max_right R0 RB).trans (le_max_left (max R0 RB) _ |>.trans hj)
  have hHT : H ≤ T := by
    dsimp [H, T]
    exact pow_le_pow_right₀ (by norm_num) (by omega)
  have hwindow : (centeredXiPositiveZeroWindow T).Nonempty := by
    refine ⟨i, (mem_centeredXiPositiveZeroWindow T i).mpr ?_⟩
    exact (le_max_right (max R0 RB) _).trans hj |>.trans hHT
  have hcardj :
      ((centeredXiPositiveZeroWindow T).card : ℝ) ≤
        B * H * (j + 1 : ℝ) := by
    simpa [T, H] using hcard j (hRBH.trans hHT)
  have hbT : (2 * H) ^ 2 ≤ T ^ 2 / 2 := by
    have hT : T = 4 * H := by
      dsimp [T, H]
      rw [pow_add]
      norm_num
      ring
    rw [hT]
    nlinarith [sq_nonneg H]
  obtain ⟨R, hR, hq⟩ :=
    exists_circle_norm_centeredXiCancelledQuotient_le_explicit
      hzero hxi hwindow (by positivity) hH hH.le hR0H
      (by linarith) hbT
  refine ⟨R, hR.1, hH.trans_le hR.1, ?_⟩
  intro z hz
  have hzNorm : ‖z‖ = R := by
    simpa [Metric.mem_sphere, dist_zero_right] using hz
  have hzlo : H ≤ ‖z‖ := by rw [hzNorm]; exact hR.1
  have hzhi : ‖z‖ ≤ (2 : ℝ) ^ (j + 1) := by
    rw [hzNorm]
    simpa [H, pow_succ, mul_comm] using hR.2
  exact (hq z hz).trans (Real.exp_le_exp.mpr
    (centeredXiQuotientCircleExponent_le_dyadicLoss
      hB hC j hcardj hwindow z hzlo hzhi))

theorem centeredXiCancelledQuotient_dyadicBoundary_actual_of_nonempty
    (hzero : XiCenteredCriticalZeroReality)
    [Nonempty CenteredXiPositiveZeroIndex] :
    DyadicSubquadraticBoundaryLogNormGrowthJensen
      centeredXiCancelledQuotient := by
  obtain ⟨Ksq, Klin, hcircle⟩ :=
    exists_eventually_circle_norm_centeredXiCancelledQuotient_le_shiftedLoss
      hzero
  obtain ⟨_hqA, hq0⟩ :=
    centeredXiCancelledQuotient_analytic_ne_zero
      hzero centeredXi_inverseSquareSummability_actual
  intro ε hε
  have hloss := tendsto_centeredXiDyadicShiftedLoss hzero Ksq Klin
  have hevent : ∀ᶠ j : ℕ in atTop,
      CenteredXiDyadicShiftedLoss Ksq Klin j < ε :=
    (tendsto_order.1 hloss).2 ε hε
  filter_upwards [hcircle, hevent] with j hj hmajor
  obtain ⟨R, hHR, hR, hbound⟩ := hj
  refine ⟨R, hHR, hR, ?_⟩
  intro z hz
  calc
    Real.log ‖centeredXiCancelledQuotient z‖ ≤
        Real.log (Real.exp
          (CenteredXiDyadicShiftedLoss Ksq Klin j *
            ((2 : ℝ) ^ j) ^ 2)) :=
      Real.log_le_log (norm_pos_iff.mpr (hq0 z)) (hbound z hz)
    _ = CenteredXiDyadicShiftedLoss Ksq Klin j *
          ((2 : ℝ) ^ j) ^ 2 := Real.log_exp _
    _ ≤ ε * ((2 : ℝ) ^ j) ^ 2 := by gcongr

theorem centeredXiCancelledQuotient_dyadicBoundary_actual_of_isEmpty
    (hzero : XiCenteredCriticalZeroReality)
    [IsEmpty CenteredXiPositiveZeroIndex] :
    DyadicSubquadraticBoundaryLogNormGrowthJensen
      centeredXiCancelledQuotient := by
  obtain ⟨C, R0, hC, _hR0, hxi⟩ :=
    exists_xiJensenEntire_orderOneGrowthBound
  obtain ⟨_hqA, hq0⟩ :=
    centeredXiCancelledQuotient_analytic_ne_zero
      hzero centeredXi_inverseSquareSummability_actual
  have hqeq : ∀ z : ℂ,
      centeredXiCancelledQuotient z = xiJensenEntire z := by
    intro z
    rw [centeredXiCancelledQuotient_eq_div_of_product_ne
      hzero centeredXi_inverseSquareSummability_actual
      (by simp [centeredXiPairedProduct])]
    simp [centeredXiPairedProduct]
  intro ε hε
  have hloss := tendsto_centeredXiDyadicShiftedLoss hzero 0 (4 * C)
  have hevent : ∀ᶠ j : ℕ in atTop,
      CenteredXiDyadicShiftedLoss 0 (4 * C) j < ε :=
    (tendsto_order.1 hloss).2 ε hε
  have hp := tendsto_pow_atTop_atTop_of_one_lt
    (by norm_num : (1 : ℝ) < 2)
  have hlarge : ∀ᶠ j : ℕ in atTop, R0 ≤ (2 : ℝ) ^ j :=
    hp.eventually (eventually_ge_atTop R0)
  filter_upwards [hevent, hlarge] with j hj hR0
  let H : ℝ := (2 : ℝ) ^ j
  have hH : 0 < H := by positivity
  refine ⟨H, le_rfl, hH, ?_⟩
  intro z hz
  have hzNorm : ‖z‖ = H := by
    simpa [Metric.mem_sphere, dist_zero_right] using hz
  have hlogT : Real.log (4 * H) ≤ 2 * (j + 1 : ℝ) := by
    rw [show 4 * H = (2 : ℝ) ^ (j + 2) by
      dsimp [H]
      rw [pow_add]
      norm_num
      ring, Real.log_pow]
    have hlog2 : Real.log (2 : ℝ) ≤ 1 := by
      have hh := Real.log_le_sub_one_of_pos (x := 2) (by norm_num)
      norm_num at hh ⊢
      exact hh
    calc
      (j + 2 : ℕ) * Real.log 2 ≤ (j + 2 : ℝ) := by
        have hh := mul_le_mul_of_nonneg_left hlog2
          (show 0 ≤ (j + 2 : ℝ) by positivity)
        norm_num [Nat.cast_add] at hh ⊢
        exact hh
      _ ≤ 2 * (j + 1 : ℝ) := by
        nlinarith [Nat.cast_nonneg (α := ℝ) j]
  have harg : ‖z‖ + 2 ≤ 4 * H := by
    rw [hzNorm]
    have hH1 : 1 ≤ H := one_le_pow₀ (by norm_num)
    linarith
  have hlogz : Real.log (‖z‖ + 2) ≤ 2 * (j + 1 : ℝ) :=
    (Real.log_le_log (by positivity) harg).trans hlogT
  have hexp :
      C * ‖z‖ * Real.log (‖z‖ + 2) ≤
        CenteredXiDyadicShiftedLoss 0 (4 * C) j * H ^ 2 := by
    rw [hzNorm]
    have hlogH : Real.log (H + 2) ≤ 2 * (j + 1 : ℝ) := by
      simpa [hzNorm] using hlogz
    have hmain :
        C * H * Real.log (H + 2) ≤
          C * H * (2 * (j + 1 : ℝ)) := by
      exact mul_le_mul_of_nonneg_left hlogH
        (mul_nonneg hC hH.le)
    calc
      C * H * Real.log (H + 2) ≤
          C * H * (2 * (j + 1 : ℝ)) := hmain
      _ ≤ CenteredXiDyadicShiftedLoss 0 (4 * C) j * H ^ 2 := by
        unfold CenteredXiDyadicShiftedLoss
        simp [H]
        field_simp
        nlinarith [Nat.cast_nonneg (α := ℝ) j]
  rw [hqeq]
  calc
    Real.log ‖xiJensenEntire z‖ ≤
        Real.log (Real.exp
          (C * ‖z‖ * Real.log (‖z‖ + 2))) :=
      Real.log_le_log
        (by simpa [← hqeq] using norm_pos_iff.mpr (hq0 z))
        (hxi z (by simpa [hzNorm, H] using hR0))
    _ = C * ‖z‖ * Real.log (‖z‖ + 2) := Real.log_exp _
    _ ≤ CenteredXiDyadicShiftedLoss 0 (4 * C) j * H ^ 2 := hexp
    _ ≤ ε * H ^ 2 := by gcongr

theorem centeredXiCancelledQuotient_dyadicBoundary_actual
    (hzero : XiCenteredCriticalZeroReality) :
    DyadicSubquadraticBoundaryLogNormGrowthJensen
      centeredXiCancelledQuotient := by
  cases isEmpty_or_nonempty CenteredXiPositiveZeroIndex
  · exact centeredXiCancelledQuotient_dyadicBoundary_actual_of_isEmpty hzero
  · exact centeredXiCancelledQuotient_dyadicBoundary_actual_of_nonempty hzero

theorem sphere_disjoint_closedBall_of_radius_not_mem_jensen
    {c : ℂ} {r R : ℝ}
    (havoid : R ∉ Set.Icc (‖c‖ - r) (‖c‖ + r)) :
    Disjoint (Metric.sphere (0 : ℂ) R)
      (Metric.closedBall c r) := by
  rw [Set.disjoint_left]
  intro z hzSphere hzBall
  apply havoid
  have hzNorm : ‖z‖ = R := by
    simpa [Metric.mem_sphere, dist_zero_right] using hzSphere
  have hzDist : ‖z - c‖ ≤ r := by
    simpa [Metric.mem_closedBall, dist_eq_norm] using hzBall
  have habs := abs_norm_sub_norm_le z c
  rw [hzNorm] at habs
  have hdiff₁ : R - ‖c‖ ≤ r :=
    (le_abs_self _).trans (habs.trans hzDist)
  have hdiff₂ : ‖c‖ - R ≤ r := by
    simpa using
      (neg_le_abs (R - ‖c‖)).trans (habs.trans hzDist)
  constructor <;> linarith

theorem subquadraticBoundaryLogNormGrowth_of_cartanExceptionalDisks_jensen
    {g : ℂ → ℂ} (hcartan : CartanExceptionalDiskLogBoundJensen g) :
    SubquadraticBoundaryLogNormGrowthJensen g := by
  intro ε hε
  obtain ⟨r0, hr0, hdata⟩ := hcartan ε hε
  let r1 := max r0 1
  refine ⟨r1, hr0.trans (le_max_left _ _), ?_⟩
  intro r hr
  have hr0' : r0 ≤ r := (le_max_left _ _).trans hr
  obtain ⟨n, center, radius, hrad, hsum, hbound⟩ :=
    hdata r hr0'
  obtain ⟨R, hR, havoid⟩ :=
    exists_radius_avoiding_exceptionalIntervals_jensen
      Finset.univ center radius (fun i _ ↦ hrad i)
      (a := r) (b := 2 * r) (by
        have hw : 2 * ∑ i, radius i < 2 * r - r := by
          linarith
        simpa using hw)
  refine ⟨R, hR.1, by
    have hr1 : 1 ≤ r := (le_max_right _ _).trans hr
    exact lt_of_lt_of_le zero_lt_one (hr1.trans hR.1), ?_⟩
  intro z hz
  have hzNorm : ‖z‖ = R := by
    simpa [Metric.mem_sphere, dist_zero_right] using hz
  apply hbound z
  · rw [hzNorm]
    exact hR.1
  · rw [hzNorm]
    exact hR.2
  · intro i hiBall
    have hdisjoint :=
      sphere_disjoint_closedBall_of_radius_not_mem_jensen
        (havoid i (Finset.mem_univ i))
    exact Set.disjoint_left.mp hdisjoint hz hiBall

theorem subquadraticRealPartGrowth_of_exp_eq_jensen
    {g h : ℂ → ℂ} (hexp : ∀ z, Complex.exp (h z) = g z)
    (hgrowth : SubquadraticLogNormGrowthJensen g) :
    SubquadraticRealPartGrowthJensen h := by
  intro ε hε
  obtain ⟨R, hR, hbound⟩ := hgrowth ε hε
  refine ⟨R, hR, fun r hr z hz ↦ ?_⟩
  rw [show (h z).re = Real.log ‖g z‖ by
    rw [← hexp, Complex.norm_exp, Real.log_exp]]
  exact hbound r hr z hz

theorem iteratedDeriv_two_eq_zero_of_subquadraticRealPartGrowth_jensen
    (h : ℂ → ℂ) (hh : AnalyticOnNhd ℂ h Set.univ)
    (hgrowth : SubquadraticRealPartGrowthJensen h) :
    ∀ c, iteratedDeriv 2 h c = 0 := by
  intro c
  by_contra hc
  let d := ‖iteratedDeriv 2 h c‖
  have hd : 0 < d := norm_pos_iff.mpr hc
  let ε := d / 256
  have hε : 0 < ε := div_pos hd (by norm_num)
  obtain ⟨R, hR0, hbound⟩ := hgrowth ε hε
  let A := 256 * (‖h 0‖ + 1) / d
  have hA : 0 < A := by dsimp [A]; positivity
  let S := 4 * (R + ‖c‖ + A + 1)
  have hS : 0 < S := by dsimp [S]; positivity
  have hRS : R ≤ S := by dsimp [S]; nlinarith [norm_nonneg c]
  have hcS : ‖c‖ < S / 4 := by dsimp [S]; nlinarith
  have hAS : A < S := by dsimp [S]; nlinarith [norm_nonneg c]
  let q := S / 2
  have hq : 0 < q := half_pos hS
  have hdiff : Differentiable ℂ h :=
    differentiableOn_univ.mp hh.differentiableOn
  have hmaps : Set.MapsTo h (Metric.ball 0 S)
      {z : ℂ | z.re ≤ ε * S ^ 2} := by
    intro z hz
    exact hbound S hRS z hz
  have hM : 0 < ε * S ^ 2 := mul_pos hε (sq_pos_of_pos hS)
  have hsphere : ∀ z ∈ Metric.sphere c q,
      ‖h z‖ ≤ 8 * (ε * S ^ 2) + 8 * ‖h 0‖ := by
    intro z hz
    have hzdist : ‖z - c‖ = q := by
      simpa [Metric.mem_sphere, dist_eq_norm] using hz
    have hznorm : ‖z‖ ≤ q + ‖c‖ := by
      calc
        ‖z‖ = ‖(z - c) + c‖ := by ring_nf
        _ ≤ ‖z - c‖ + ‖c‖ := norm_add_le _ _
        _ = q + ‖c‖ := by rw [hzdist]
    have hzS : ‖z‖ < S := by
      dsimp [q] at hznorm
      linarith
    have hzball : z ∈ Metric.ball (0 : ℂ) S := by
      simpa [Metric.mem_ball] using hzS
    have hBC := Complex.borelCaratheodory hM
      hdiff.differentiableOn hmaps hS hzball
    have hden : S / 4 ≤ S - ‖z‖ := by
      dsimp [q] at hznorm
      linarith
    have hzleS : ‖z‖ ≤ S := hzS.le
    calc
      ‖h z‖ ≤ 2 * (ε * S ^ 2) * ‖z‖ / (S - ‖z‖) +
          ‖h 0‖ * (S + ‖z‖) / (S - ‖z‖) := hBC
      _ ≤ 2 * (ε * S ^ 2) * S / (S / 4) +
          ‖h 0‖ * (2 * S) / (S / 4) := by
        apply add_le_add
        · exact div_le_div₀
            (by positivity) (by gcongr) (by positivity) hden
        · exact div_le_div₀
            (by positivity) (by gcongr; linarith) (by positivity) hden
      _ = 8 * (ε * S ^ 2) + 8 * ‖h 0‖ := by
        field_simp
        ring
  have hcauchy := Complex.norm_iteratedDeriv_le_of_forall_mem_sphere_norm_le
    2 hq hdiff.diffContOnCl hsphere
  have hSsq : 256 * ‖h 0‖ / d ≤ S ^ 2 := by
    have hAle : 256 * ‖h 0‖ / d ≤ A := by
      dsimp [A]
      apply div_le_div_of_nonneg_right _ hd.le
      nlinarith
    have hSone : 1 < S := by
      dsimp [S]
      nlinarith [hR0, norm_nonneg c, hA]
    calc
      256 * ‖h 0‖ / d ≤ A := hAle
      _ ≤ S := hAS.le
      _ ≤ S ^ 2 := by nlinarith
  have hle : d ≤ 64 * ε + 64 * ‖h 0‖ / S ^ 2 := by
    dsimp [d]
    calc
      ‖iteratedDeriv 2 h c‖ ≤
          (2 : ℕ).factorial *
            (8 * (ε * S ^ 2) + 8 * ‖h 0‖) / q ^ 2 := hcauchy
      _ = 64 * ε + 64 * ‖h 0‖ / S ^ 2 := by
        dsimp [q]
        field_simp
        ring
  have hconst : 64 * ‖h 0‖ / S ^ 2 ≤ d / 4 := by
    have hmul : 256 * ‖h 0‖ ≤ S ^ 2 * d :=
      (div_le_iff₀ hd).mp hSsq
    rw [div_le_iff₀ (sq_pos_of_pos hS)]
    nlinarith
  dsimp [ε] at hle
  linarith

theorem eq_affine_of_subquadraticRealPartGrowth_jensen
    (h : ℂ → ℂ) (hh : AnalyticOnNhd ℂ h Set.univ)
    (hgrowth : SubquadraticRealPartGrowthJensen h) :
    ∃ a b : ℂ, h = fun z ↦ a + b * z := by
  let a := h 0
  let b := deriv h 0
  have hsecond :=
    iteratedDeriv_two_eq_zero_of_subquadraticRealPartGrowth_jensen h hh hgrowth
  have hderiv : ∀ z, deriv h z = b := by
    intro z
    apply isOpen_univ.is_const_of_deriv_eq_zero isPreconnected_univ
      hh.deriv.differentiableOn
      (fun w hw ↦ ?_) (Set.mem_univ z) (Set.mem_univ 0)
    simpa [iteratedDeriv_succ, iteratedDeriv_one] using hsecond w
  refine ⟨a, b, ?_⟩
  funext z
  apply isOpen_univ.eqOn_of_deriv_eq isPreconnected_univ
    hh.differentiableOn (by fun_prop) (fun w hw ↦ ?_)
    (Set.mem_univ 0) (by simp [a]) (Set.mem_univ z)
  simpa using hderiv w

theorem iteratedDeriv_two_eq_zero_of_subquadraticNormGrowth_jensen
    (h : ℂ → ℂ) (hh : AnalyticOnNhd ℂ h Set.univ)
    (hgrowth : SubquadraticNormGrowthJensen h) :
    ∀ c, iteratedDeriv 2 h c = 0 := by
  intro c
  by_contra hc
  have hdpos : 0 < ‖iteratedDeriv 2 h c‖ := norm_pos_iff.mpr hc
  let ε : ℝ := ‖iteratedDeriv 2 h c‖ / 16
  have hε : 0 < ε := div_pos hdpos (by norm_num)
  obtain ⟨R, hR0, hbound⟩ := hgrowth ε hε
  let r : ℝ := R + ‖c‖ + 1
  have hr : 0 < r := by dsimp [r]; positivity
  have hC : ∀ z ∈ Metric.sphere c r,
      ‖h z‖ ≤ 4 * ε * r ^ 2 := by
    intro z hz
    have hzdist : ‖z - c‖ = r := by
      simpa [Metric.mem_sphere, dist_eq_norm] using hz
    have hzlower : R ≤ ‖z‖ := by
      have htri : ‖z - c‖ ≤ ‖z‖ + ‖c‖ := norm_sub_le z c
      rw [hzdist] at htri
      dsimp [r] at htri
      linarith
    have hzupper : ‖z‖ ≤ 2 * r := by
      have htri : ‖z‖ ≤ ‖z - c‖ + ‖c‖ := by
        calc
          ‖z‖ = ‖(z - c) + c‖ := by ring_nf
          _ ≤ ‖z - c‖ + ‖c‖ := norm_add_le _ _
      rw [hzdist] at htri
      have hcr : ‖c‖ ≤ r := by dsimp [r]; linarith
      linarith
    calc
      ‖h z‖ ≤ ε * ‖z‖ ^ 2 := hbound z hzlower
      _ ≤ ε * (2 * r) ^ 2 := by gcongr
      _ = 4 * ε * r ^ 2 := by ring
  have hdif : Differentiable ℂ h :=
    differentiableOn_univ.mp hh.differentiableOn
  have hcauchy := Complex.norm_iteratedDeriv_le_of_forall_mem_sphere_norm_le
    2 hr hdif.diffContOnCl hC
  have hle : ‖iteratedDeriv 2 h c‖ ≤ 8 * ε := by
    calc
      ‖iteratedDeriv 2 h c‖ ≤
          (2 : ℕ).factorial * (4 * ε * r ^ 2) / r ^ 2 := hcauchy
      _ = 8 * ε := by
        norm_num [Nat.factorial]
        field_simp
        norm_num
  dsimp [ε] at hle
  linarith

theorem eq_affine_of_subquadraticNormGrowth_jensen
    (h : ℂ → ℂ) (hh : AnalyticOnNhd ℂ h Set.univ)
    (hgrowth : SubquadraticNormGrowthJensen h) :
    ∃ a b : ℂ, h = fun z ↦ a + b * z := by
  let a := h 0
  let b := deriv h 0
  have hsecond :=
    iteratedDeriv_two_eq_zero_of_subquadraticNormGrowth_jensen h hh hgrowth
  have hderiv : ∀ z, deriv h z = b := by
    intro z
    apply isOpen_univ.is_const_of_deriv_eq_zero isPreconnected_univ
      hh.deriv.differentiableOn
      (fun w hw ↦ ?_) (Set.mem_univ z) (Set.mem_univ 0)
    simpa [iteratedDeriv_succ, iteratedDeriv_one] using hsecond w
  refine ⟨a, b, ?_⟩
  funext z
  apply isOpen_univ.eqOn_of_deriv_eq isPreconnected_univ
    hh.differentiableOn (by fun_prop) (fun w hw ↦ ?_)
    (Set.mem_univ 0) (by simp [a]) (Set.mem_univ z)
  simpa using hderiv w

def XiCenteredCancelledQuotientLogGrowth : Prop :=
  ∃ h : ℂ → ℂ,
    AnalyticOnNhd ℂ h Set.univ ∧
    (∀ z, Complex.exp (h z) = centeredXiCancelledQuotient z) ∧
    SubquadraticNormGrowthJensen h

/-- Natural Cartan/Hadamard growth condition on the quotient itself. Unlike
`XiCenteredCancelledQuotientLogGrowth`, this only controls the real part of
an analytic logarithm and is therefore the estimate produced by quotient
norm bounds. -/
def XiCenteredCancelledQuotientLogNormGrowth : Prop :=
  SubquadraticLogNormGrowthJensen centeredXiCancelledQuotient

/-- Selected-circle form of the remaining minimum-modulus obligation. -/
def XiCenteredCancelledQuotientSelectedCircleGrowth : Prop :=
  SubquadraticBoundaryLogNormGrowthJensen centeredXiCancelledQuotient

def XiCenteredCancelledQuotientCartanDiskGrowth : Prop :=
  CartanExceptionalDiskLogBoundJensen centeredXiCancelledQuotient

theorem exists_centeredXiCancelledQuotient_analyticLog
    (hzero : XiCenteredCriticalZeroReality)
    (hsum : XiCenteredZeroInverseSquareSummability) :
    ∃ h : ℂ → ℂ, AnalyticOnNhd ℂ h Set.univ ∧
      ∀ z, Complex.exp (h z) = centeredXiCancelledQuotient z := by
  obtain ⟨hqA, hq0⟩ :=
    centeredXiCancelledQuotient_analytic_ne_zero hzero hsum
  exact exists_analytic_log_jensen hqA hq0

theorem xiCenteredCancelledQuotientLogNormGrowth_of_selectedCircle
    (hzero : XiCenteredCriticalZeroReality)
    (hsum : XiCenteredZeroInverseSquareSummability)
    (hcircle : XiCenteredCancelledQuotientSelectedCircleGrowth) :
    XiCenteredCancelledQuotientLogNormGrowth := by
  obtain ⟨hqA, hq0⟩ :=
    centeredXiCancelledQuotient_analytic_ne_zero hzero hsum
  exact subquadraticLogNormGrowth_of_boundary_jensen hqA hq0 hcircle

theorem xiCenteredCancelledQuotientSelectedCircleGrowth_of_cartan
    (hcartan : XiCenteredCancelledQuotientCartanDiskGrowth) :
    XiCenteredCancelledQuotientSelectedCircleGrowth :=
  subquadraticBoundaryLogNormGrowth_of_cartanExceptionalDisks_jensen hcartan

/-- The explicit normalized-loss estimate is sufficient for the exact
selected-circle growth premise consumed by quotient rigidity. -/
theorem xiCenteredCancelledQuotientSelectedCircleGrowth_of_dyadicLoss
    (hzero : XiCenteredCriticalZeroReality)
    (hsum : XiCenteredZeroInverseSquareSummability)
    (Ksq Klin : ℝ)
    (hcircle : ∀ᶠ j : ℕ in atTop,
      ∃ R : ℝ, (2 : ℝ) ^ j ≤ R ∧ 0 < R ∧
        ∀ z ∈ Metric.sphere (0 : ℂ) R,
          ‖centeredXiCancelledQuotient z‖ ≤ Real.exp (
            CenteredXiDyadicNormalizedLoss Ksq Klin
              (fun j ↦
                ∑' i : {i : CenteredXiPositiveZeroIndex //
                    i ∉ centeredXiPositiveZeroWindow ((2 : ℝ) ^ j)},
                  ‖(((centeredXiPositiveSquaredOrdinate i)⁻¹ : ℝ) : ℂ)‖)
              j * ((2 : ℝ) ^ j) ^ 2)) :
    XiCenteredCancelledQuotientSelectedCircleGrowth := by
  obtain ⟨_hqA, hq0⟩ :=
    centeredXiCancelledQuotient_analytic_ne_zero hzero hsum
  exact subquadraticBoundaryLogNormGrowth_of_dyadic_jensen
    (dyadicSubquadraticBoundaryLogNormGrowth_of_centeredXiLoss
      hq0 hzero hsum Ksq Klin hcircle)

theorem centeredXiPairedProductAffineRigidity_of_logGrowth
    (hzero : XiCenteredCriticalZeroReality)
    (hsum : XiCenteredZeroInverseSquareSummability)
    (hgrowth : XiCenteredCancelledQuotientLogGrowth) :
    XiCenteredPairedProductAffineRigidity := by
  rcases hgrowth with ⟨h, hh, hexp, hgrowth⟩
  obtain ⟨a, b, hab⟩ :=
    eq_affine_of_subquadraticNormGrowth_jensen h hh hgrowth
  refine ⟨a, b, ?_⟩
  have hPdiff :=
    differentiable_centeredXiPairedProduct hzero hsum
  have hP0 : centeredXiPairedProduct 0 ≠ 0 := by
    simp [centeredXiPairedProduct]
  have hlocal : xiJensenEntire =ᶠ[𝓝 0]
      fun z ↦ Complex.exp (a + b * z) * centeredXiPairedProduct z := by
    filter_upwards [(hPdiff 0).continuousAt.eventually_ne hP0] with z hPz
    have hq :=
      centeredXiCancelledQuotient_eq_div_of_product_ne hzero hsum hPz
    have he := hexp z
    rw [hq, hab] at he
    exact (div_eq_iff hPz).mp he.symm
  have hxiA : AnalyticOnNhd ℂ xiJensenEntire Set.univ :=
    fun z _ ↦ differentiable_xiJensenEntire.analyticAt z
  have hrhsA : AnalyticOnNhd ℂ
      (fun z ↦ Complex.exp (a + b * z) * centeredXiPairedProduct z)
      Set.univ := by
    intro z hz
    exact ((analyticAt_const.add
      (analyticAt_const.mul analyticAt_id)).cexp).mul
        (hPdiff.analyticAt z)
  exact fun z ↦ congrFun (hxiA.eq_of_eventuallyEq hrhsA hlocal) z

theorem centeredXiPairedProductAffineRigidity_of_logNormGrowth
    (hzero : XiCenteredCriticalZeroReality)
    (hsum : XiCenteredZeroInverseSquareSummability)
    (hgrowth : XiCenteredCancelledQuotientLogNormGrowth) :
    XiCenteredPairedProductAffineRigidity := by
  obtain ⟨h, hh, hexp⟩ :=
    exists_centeredXiCancelledQuotient_analyticLog hzero hsum
  have hreal : SubquadraticRealPartGrowthJensen h :=
    subquadraticRealPartGrowth_of_exp_eq_jensen hexp hgrowth
  obtain ⟨a, b, hab⟩ :=
    eq_affine_of_subquadraticRealPartGrowth_jensen h hh hreal
  refine ⟨a, b, ?_⟩
  have hPdiff :=
    differentiable_centeredXiPairedProduct hzero hsum
  have hP0 : centeredXiPairedProduct 0 ≠ 0 := by
    simp [centeredXiPairedProduct]
  have hlocal : xiJensenEntire =ᶠ[𝓝 0]
      fun z ↦ Complex.exp (a + b * z) * centeredXiPairedProduct z := by
    filter_upwards [(hPdiff 0).continuousAt.eventually_ne hP0] with z hPz
    have hq :=
      centeredXiCancelledQuotient_eq_div_of_product_ne hzero hsum hPz
    have he := hexp z
    rw [hq, hab] at he
    exact (div_eq_iff hPz).mp he.symm
  have hxiA : AnalyticOnNhd ℂ xiJensenEntire Set.univ :=
    fun z _ ↦ differentiable_xiJensenEntire.analyticAt z
  have hrhsA : AnalyticOnNhd ℂ
      (fun z ↦ Complex.exp (a + b * z) * centeredXiPairedProduct z)
      Set.univ := by
    intro z hz
    exact ((analyticAt_const.add
      (analyticAt_const.mul analyticAt_id)).cexp).mul
        (hPdiff.analyticAt z)
  exact fun z ↦ congrFun (hxiA.eq_of_eventuallyEq hrhsA hlocal) z

theorem centeredXiPairedProduct_divisor_support_eq
    (hzero : XiCenteredCriticalZeroReality)
    (hsum : XiCenteredZeroInverseSquareSummability) :
    Function.support
        (MeromorphicOn.divisor centeredXiPairedProduct Set.univ) =
      Function.support centeredXiZeroDivisor := by
  have hPdiff := differentiable_centeredXiPairedProduct hzero hsum
  have hPA : AnalyticOnNhd ℂ centeredXiPairedProduct Set.univ :=
    fun z _ ↦ hPdiff.analyticAt z
  have hP0 : centeredXiPairedProduct 0 ≠ 0 := by
    simp [centeredXiPairedProduct]
  have hPfinite : ∀ u : Set.univ,
      meromorphicOrderAt centeredXiPairedProduct u.1 ≠ ⊤ := by
    intro u
    rw [(hPdiff.analyticAt u.1).meromorphicOrderAt_eq]
    have ho := hPA.analyticOrderAt_ne_top_of_isPreconnected
      (x := (0 : ℂ)) (y := u.1) isPreconnected_univ
      (Set.mem_univ 0) (Set.mem_univ u.1) (by
        rw [(hPdiff.analyticAt 0).analyticOrderAt_eq_zero.mpr hP0]
        exact ENat.zero_ne_top)
    simpa using ho
  have hPsupport :=
    hPA.meromorphicNFOn.zero_set_eq_divisor_support hPfinite
  rw [centeredXiZeroDivisor_support]
  have hp :
      Function.support
          (MeromorphicOn.divisor centeredXiPairedProduct Set.univ) =
        centeredXiPairedProduct ⁻¹' {0} := by
    simpa only [Set.univ_inter] using hPsupport.symm
  rw [hp]
  ext z
  exact centeredXiPairedProduct_zero_set_eq hzero hsum z

/-- Once divisor/minimum-modulus work has produced the affine representation,
the canonical indexed product is normalized without any further assumptions. -/
theorem centeredXi_eq_normalizedPairedProduct_of_affine
    (hzero : XiCenteredCriticalZeroReality)
    (hsum : XiCenteredZeroInverseSquareSummability)
    (haffine : XiCenteredPairedProductAffineRigidity) :
    ∀ z : ℂ, xiJensenEntire z =
      (xiGamma 0 : ℂ) * centeredXiPairedProduct z := by
  rcases haffine with ⟨a, b, hrep⟩
  have hPeven : ∀ z, centeredXiPairedProduct (-z) =
      centeredXiPairedProduct z := by
    intro z
    simp only [centeredXiPairedProduct, neg_sq]
  have hP0 : centeredXiPairedProduct 0 ≠ 0 := by
    simp [centeredXiPairedProduct]
  have hconst := exp_affine_prefactor_eq_const_of_even a b
    differentiable_xiJensenEntire
    (differentiable_centeredXiPairedProduct hzero hsum)
    xiJensenEntire_neg hPeven hP0 hrep
  have hcenter0 : xiJensenEntire 0 = (xiGamma 0 : ℂ) := by
    rw [← xiJensenGeneratingFunction_sq 0]
    simpa using xiJensenGeneratingFunction_zero
  have hexpa : Complex.exp a = (xiGamma 0 : ℂ) := by
    have h0 := hconst 0
    rw [hcenter0] at h0
    simpa [centeredXiPairedProduct] using h0.symm
  intro z
  rw [hconst, hexpa]

theorem centeredXi_eq_normalizedPairedProduct_of_logGrowth
    (hzero : XiCenteredCriticalZeroReality)
    (hsum : XiCenteredZeroInverseSquareSummability)
    (hgrowth : XiCenteredCancelledQuotientLogGrowth) :
    ∀ z : ℂ, xiJensenEntire z =
      (xiGamma 0 : ℂ) * centeredXiPairedProduct z :=
  centeredXi_eq_normalizedPairedProduct_of_affine hzero hsum
    (centeredXiPairedProductAffineRigidity_of_logGrowth hzero hsum hgrowth)

theorem centeredXi_eq_normalizedPairedProduct_of_logNormGrowth
    (hzero : XiCenteredCriticalZeroReality)
    (hsum : XiCenteredZeroInverseSquareSummability)
    (hgrowth : XiCenteredCancelledQuotientLogNormGrowth) :
    ∀ z : ℂ, xiJensenEntire z =
      (xiGamma 0 : ℂ) * centeredXiPairedProduct z :=
  centeredXi_eq_normalizedPairedProduct_of_affine hzero hsum
    (centeredXiPairedProductAffineRigidity_of_logNormGrowth
      hzero hsum hgrowth)

theorem centeredXi_eq_normalizedPairedProduct_of_selectedCircle
    (hzero : XiCenteredCriticalZeroReality)
    (hsum : XiCenteredZeroInverseSquareSummability)
    (hcircle : XiCenteredCancelledQuotientSelectedCircleGrowth) :
    ∀ z : ℂ, xiJensenEntire z =
      (xiGamma 0 : ℂ) * centeredXiPairedProduct z :=
  centeredXi_eq_normalizedPairedProduct_of_logNormGrowth hzero hsum
    (xiCenteredCancelledQuotientLogNormGrowth_of_selectedCircle
      hzero hsum hcircle)

theorem centeredXi_eq_normalizedPairedProduct_of_count_and_logGrowth
    (hzero : XiCenteredCriticalZeroReality)
    (hcount : XiCenteredCumulativeZeroCount)
    (hgrowth : XiCenteredCancelledQuotientLogGrowth) :
    ∀ z : ℂ, xiJensenEntire z =
      (xiGamma 0 : ℂ) * centeredXiPairedProduct z := by
  exact centeredXi_eq_normalizedPairedProduct_of_logGrowth hzero
    (centeredXi_inverseSquareSummability_of_cumulativeZeroCount hcount)
    hgrowth

theorem centeredXi_eq_normalizedPairedProduct_of_count_and_selectedCircle
    (hzero : XiCenteredCriticalZeroReality)
    (hcount : XiCenteredCumulativeZeroCount)
    (hcircle : XiCenteredCancelledQuotientSelectedCircleGrowth) :
    ∀ z : ℂ, xiJensenEntire z =
      (xiGamma 0 : ℂ) * centeredXiPairedProduct z := by
  exact centeredXi_eq_normalizedPairedProduct_of_selectedCircle hzero
    (centeredXi_inverseSquareSummability_of_cumulativeZeroCount hcount)
    hcircle

/-- End-to-end Hadamard identification from the cumulative multiplicity count
and the explicit dyadic quotient estimate.  All normalized losses in this
statement are now proved to vanish; the remaining analytic task is to supply
the displayed circle norm inequality from xi's order-one bound and the
paired-product minimum modulus. -/
theorem centeredXi_eq_normalizedPairedProduct_of_count_and_dyadicLoss
    (hzero : XiCenteredCriticalZeroReality)
    (hcount : XiCenteredCumulativeZeroCount)
    (Ksq Klin : ℝ)
    (hcircle : ∀ᶠ j : ℕ in atTop,
      ∃ R : ℝ, (2 : ℝ) ^ j ≤ R ∧ 0 < R ∧
        ∀ z ∈ Metric.sphere (0 : ℂ) R,
          ‖centeredXiCancelledQuotient z‖ ≤ Real.exp (
            CenteredXiDyadicNormalizedLoss Ksq Klin
              (fun j ↦
                ∑' i : {i : CenteredXiPositiveZeroIndex //
                    i ∉ centeredXiPositiveZeroWindow ((2 : ℝ) ^ j)},
                  ‖(((centeredXiPositiveSquaredOrdinate i)⁻¹ : ℝ) : ℂ)‖)
              j * ((2 : ℝ) ^ j) ^ 2)) :
    ∀ z : ℂ, xiJensenEntire z =
      (xiGamma 0 : ℂ) * centeredXiPairedProduct z := by
  have hsum :=
    centeredXi_inverseSquareSummability_of_cumulativeZeroCount hcount
  exact centeredXi_eq_normalizedPairedProduct_of_count_and_selectedCircle
    hzero hcount
    (xiCenteredCancelledQuotientSelectedCircleGrowth_of_dyadicLoss
      hzero hsum Ksq Klin hcircle)

/-- Concrete Hadamard identification under RH. The only analytic inputs are
the unconditional xi count and order-one growth estimates above. -/
theorem centeredXi_eq_normalizedPairedProduct_actual
    (hzero : XiCenteredCriticalZeroReality) :
    ∀ z : ℂ, xiJensenEntire z =
      (xiGamma 0 : ℂ) * centeredXiPairedProduct z := by
  exact centeredXi_eq_normalizedPairedProduct_of_selectedCircle
    hzero centeredXi_inverseSquareSummability_actual
    (subquadraticBoundaryLogNormGrowth_of_dyadic_jensen
      (centeredXiCancelledQuotient_dyadicBoundary_actual hzero))

theorem centeredXi_eq_normalizedPairedProduct_of_count_and_cartan
    (hzero : XiCenteredCriticalZeroReality)
    (hcount : XiCenteredCumulativeZeroCount)
    (hcartan : XiCenteredCancelledQuotientCartanDiskGrowth) :
    ∀ z : ℂ, xiJensenEntire z =
      (xiGamma 0 : ℂ) * centeredXiPairedProduct z :=
  centeredXi_eq_normalizedPairedProduct_of_count_and_selectedCircle
    hzero hcount
    (xiCenteredCancelledQuotientSelectedCircleGrowth_of_cartan hcartan)

theorem centeredXi_eq_normalizedPairedProduct_of_cumulativeCount
    (hzero : XiCenteredCriticalZeroReality)
    (hcount : XiCenteredCumulativeZeroCount)
    (hrigid : XiCenteredDivisorQuotientRigidity) :
    ∀ z : ℂ, xiJensenEntire z =
      (xiGamma 0 : ℂ) * centeredXiPairedProduct z := by
  have hsum :=
    centeredXi_inverseSquareSummability_of_cumulativeZeroCount hcount
  exact centeredXi_eq_normalizedPairedProduct_of_affine hzero hsum
    (hrigid (xiCenteredPairedProductDivisorEquality hzero hsum))

/-- The sole remaining classical Pólya--Jensen/Hadamard-product converse:
critical-axis zeros of the centered xi function force hyperbolicity of every
finite Jensen polynomial. -/
def XiRealZeroToJensenHyperbolicity : Prop :=
  XiCenteredCriticalZeroReality → AllJensenHyperbolic xiGamma

/-- Canonical-product half of the remaining converse.  This is exactly where
the Li-branch genus-one divisor/product must be paired under centered evenness
and identified with a genus-zero product in the squared variable. -/
def XiCriticalZerosToPositiveGenusZeroProduct : Prop :=
  XiCenteredCriticalZeroReality →
    PositiveGenusZeroProductRepresentation xiJensenGeneratingFunction

/-- Divisor/order/normalization form of the Hadamard obligation. -/
def XiCriticalZerosToNormalizedProductIdentification : Prop :=
  XiCenteredCriticalZeroReality →
    XiNormalizedGenusZeroProductIdentification

/-- Sharpened Hadamard obligation: construct the paired centered product only
up to the affine exponential factor delivered by order-one rigidity.  The
normalization theorem above removes that factor automatically. -/
def XiCriticalZerosToCenteredAffineProductIdentification : Prop :=
  XiCenteredCriticalZeroReality →
    XiCenteredAffinePairedProductIdentification

/-- Pólya--Jensen half of the remaining converse, in terms of the project's
concrete locally-uniform hyperbolic-limit definition. -/
def XiLaguerrePolyaToAllJensen : Prop :=
  XiLaguerrePolyaMembership → AllJensenHyperbolic xiGamma

/-- Sharp finite-approximation form of the reverse Jensen obligation. The
approximating rows use exactly `binom(d,j) * gamma(j)`; all positive shifts
then follow from the already proved derivative identity and unconditional
xi coefficient nondegeneracy. -/
def XiReverseJensenApproximation : Prop :=
  XiCenteredCriticalZeroReality →
    JensenCoefficientwiseHyperbolicApproximation xiGamma

theorem xiReverseJensenApproximation_of_positiveGenusZero
    (hproduct : XiCriticalZerosToPositiveGenusZeroProduct)
    (hfinite : FinitePositiveGenusZeroJensenHyperbolicity)
    (hcoeff : XiPositiveGenusZeroCoefficientConvergence) :
    XiReverseJensenApproximation := by
  intro hzero
  rcases hproduct hzero with ⟨c, root, hc, hroot, hlim⟩
  refine ⟨fun N ↦
    positiveGenusZeroExponentialCoefficient c root N, ?_, ?_⟩
  · intro N d hd
    exact hfinite c root N d hc hroot hd
  · exact hcoeff c root hc hroot hlim

theorem xiRealZeroToJensenHyperbolicity_of_reverseJensenApproximation
    (hreverse : XiReverseJensenApproximation) :
    XiRealZeroToJensenHyperbolicity := by
  intro hzero
  exact allJensenHyperbolic_of_coefficientwiseApproximation
    xiNonzeroJensenFamily (ne_of_gt (xiGamma_strictlyPositive 0))
    (hreverse hzero)

theorem xiRealZeroToJensenHyperbolicity_of_canonicalProduct
    (hproduct : XiCriticalZerosToPositiveGenusZeroProduct)
    (hJensen : XiLaguerrePolyaToAllJensen) :
    XiRealZeroToJensenHyperbolicity := by
  intro hzero
  exact hJensen
    (xiLaguerrePolyaMembership_of_positiveGenusZeroProduct
      (hproduct hzero))

theorem xiCriticalZerosToPositiveGenusZeroProduct_of_identification
    (hHadamard : XiCriticalZerosToNormalizedProductIdentification) :
    XiCriticalZerosToPositiveGenusZeroProduct := by
  intro hzero
  exact positiveGenusZeroProductRepresentation_of_identification
    (hHadamard hzero)

theorem xiCriticalZerosToNormalizedProduct_of_centeredAffine
    (hHadamard : XiCriticalZerosToCenteredAffineProductIdentification) :
    XiCriticalZerosToNormalizedProductIdentification := by
  intro hzero
  exact normalizedProductIdentification_of_centeredAffine
    (hHadamard hzero)

theorem allJensenHyperbolic_iff_riemannHypothesis
    (hconverse : XiRealZeroToJensenHyperbolicity) :
    AllJensenHyperbolic xiGamma ↔ KakeyaRiemannHypothesisRoot := by
  constructor
  · exact riemannHypothesis_of_allJensenHyperbolic
  · intro hRH
    exact hconverse
      (xiCenteredCriticalZeroReality_of_riemannHypothesis hRH)

theorem allJensenHyperbolic_iff_riemannHypothesis_of_reverseJensenApproximation
    (hreverse : XiReverseJensenApproximation) :
    AllJensenHyperbolic xiGamma ↔ KakeyaRiemannHypothesisRoot :=
  allJensenHyperbolic_iff_riemannHypothesis
    (xiRealZeroToJensenHyperbolicity_of_reverseJensenApproximation hreverse)

theorem allJensenHyperbolic_iff_riemannHypothesis_of_canonicalProduct
    (hproduct : XiCriticalZerosToPositiveGenusZeroProduct)
    (hJensen : XiLaguerrePolyaToAllJensen) :
    AllJensenHyperbolic xiGamma ↔ KakeyaRiemannHypothesisRoot :=
  allJensenHyperbolic_iff_riemannHypothesis
    (xiRealZeroToJensenHyperbolicity_of_canonicalProduct
      hproduct hJensen)

theorem allJensenHyperbolic_iff_riemannHypothesis_of_identification
    (hHadamard : XiCriticalZerosToNormalizedProductIdentification)
    (hJensen : XiLaguerrePolyaToAllJensen) :
    AllJensenHyperbolic xiGamma ↔ KakeyaRiemannHypothesisRoot :=
  allJensenHyperbolic_iff_riemannHypothesis_of_canonicalProduct
    (xiCriticalZerosToPositiveGenusZeroProduct_of_identification hHadamard)
    hJensen

theorem allJensenHyperbolic_iff_riemannHypothesis_of_centeredAffine
    (hHadamard : XiCriticalZerosToCenteredAffineProductIdentification)
    (hJensen : XiLaguerrePolyaToAllJensen) :
    AllJensenHyperbolic xiGamma ↔ KakeyaRiemannHypothesisRoot :=
  allJensenHyperbolic_iff_riemannHypothesis_of_identification
    (xiCriticalZerosToNormalizedProduct_of_centeredAffine hHadamard)
    hJensen

theorem xiJensenPolynomial_degree_two_hyperbolic_of_polyaFrequency
    (hpf : AllJensenPolyaFrequency xiGamma) (n : ℕ) :
    Hyperbolic (jensenPolynomial xiGamma 2 n) := by
  let b := jensenCoefficientSequence xiGamma 2 n
  rw [← finiteCoefficientPolynomial_jensenCoefficientSequence]
  apply nonpositiveRooted_hyperbolic
  apply finiteASW_degree_two b
  · intro j hj
    simp [b, jensenCoefficientSequence, Nat.not_le.mpr (by omega : 2 < j)]
  · exact hpf 2 n (by norm_num)
  · dsimp only [b, jensenCoefficientSequence]
    norm_num
    exact xiGamma_strictlyPositive n
  · dsimp only [b, jensenCoefficientSequence]
    norm_num
    exact xiGamma_strictlyPositive (n + 2)

theorem xiJensenHyperbolicThrough_two_of_polyaFrequency
    (hpf : AllJensenPolyaFrequency xiGamma) :
    JensenHyperbolicThrough xiGamma 2 := by
  intro d n hd hd2
  interval_cases d
  · exact jensenPolynomial_degree_one_hyperbolic xiGamma n
      (ne_of_gt (xiGamma_strictlyPositive (n + 1)))
  · exact xiJensenPolynomial_degree_two_hyperbolic_of_polyaFrequency hpf n

theorem allJensenHyperbolic_xiGamma_of_polyaFrequency
    (hASW : FinitePolyaFrequencyHyperbolicity)
    (hpf : AllJensenPolyaFrequency xiGamma) :
    AllJensenHyperbolic xiGamma :=
  allJensenHyperbolic_of_polyaFrequency hASW hpf xiNonzeroJensenFamily

theorem allJensenHyperbolic_xiGamma_of_finiteASW
    (hASW : FiniteAissenSchoenbergWhitney)
    (hpf : AllJensenPolyaFrequency xiGamma) :
    AllJensenHyperbolic xiGamma :=
  allJensenHyperbolic_xiGamma_of_polyaFrequency
    hASW.toHyperbolicity hpf

theorem riemannHypothesis_of_finiteASW_polyaFrequency
    (hASW : FiniteAissenSchoenbergWhitney)
    (hpf : AllJensenPolyaFrequency xiGamma) :
    KakeyaRiemannHypothesisRoot :=
  riemannHypothesis_of_allJensenHyperbolic
    (allJensenHyperbolic_xiGamma_of_finiteASW hASW hpf)

theorem allJensenHyperbolic_xiGamma_of_polyaFrequencyThrough
    (hASW : ∀ k, FinitePolyaFrequencyHyperbolicityThrough k)
    (hpf : AllJensenPolyaFrequency xiGamma) :
    AllJensenHyperbolic xiGamma :=
  allJensenHyperbolic_of_polyaFrequencyThrough
    hASW hpf xiNonzeroJensenFamily

/-- The all-degree unshifted xi obligation left after derivative closure. -/
def XiUnshiftedHyperbolicity : Prop :=
  ∀ d : ℕ, 1 ≤ d → Hyperbolic (jensenPolynomial xiGamma d 0)

theorem allJensenHyperbolic_xiGamma_iff_unshifted
    (h : XiCoefficientNondegeneracy) :
    AllJensenHyperbolic xiGamma ↔ XiUnshiftedHyperbolicity :=
  allJensenHyperbolic_iff_unshifted_of_windowNonzero xiGamma h

theorem allJensenHyperbolic_xiGamma_iff_unshifted_unconditional :
    AllJensenHyperbolic xiGamma ↔ XiUnshiftedHyperbolicity :=
  allJensenHyperbolic_xiGamma_iff_unshifted
    xiCoefficientNondegeneracy_unconditional

/-- The only remaining bridge interfaces separating the formalized xi Taylor
data and finite Jensen algebra from RH are the two Pólya--Jensen directions. -/
structure BridgeObligations : Prop where
  polyaJensenForward :
    KakeyaRiemannHypothesisRoot → AllJensenHyperbolic xiGamma
  polyaJensenReverse :
    AllJensenHyperbolic xiGamma → KakeyaRiemannHypothesisRoot

/-- The same two bridge directions factored through an actual
local-uniform-limit definition.  This separates the coefficient/Jensen
theorem from the zero-location/RH theorem and prevents either direction from
silently assuming RH. -/
structure AnalyticBridgeObligations : Prop where
  polyaJensenLimit :
    AllJensenHyperbolic xiGamma ↔ XiLaguerrePolyaMembership
  laguerrePolyaRH :
    XiLaguerrePolyaMembership ↔ KakeyaRiemannHypothesisRoot

theorem AnalyticBridgeObligations.toBridgeObligations
    (h : AnalyticBridgeObligations) :
    BridgeObligations where
  polyaJensenForward hRH :=
    h.polyaJensenLimit.mpr (h.laguerrePolyaRH.mpr hRH)
  polyaJensenReverse hJ :=
    h.laguerrePolyaRH.mp (h.polyaJensenLimit.mp hJ)

end Kakeya.RHJensen
