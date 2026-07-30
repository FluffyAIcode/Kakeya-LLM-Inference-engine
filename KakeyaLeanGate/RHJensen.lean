import KakeyaLeanGate.RiemannHypothesisRoot
import Mathlib.Analysis.Complex.TaylorSeries
import Mathlib.Analysis.Calculus.IteratedDeriv.Lemmas

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

open Complex Polynomial

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

theorem riemannXi_one_sub (s : ℂ) : riemannXi (1 - s) = riemannXi s := by
  rw [riemannXi, riemannXi, completedRiemannZeta₀_one_sub]
  ring_nf

/-- The entire function in equation (1) of Griffin--Ono--Rolen--Zagier:
the entire continuation of
`(-1 + 4 z^2) Λ(1/2 + z) = 8 ξ(1/2 + z)`. -/
def xiJensenEntire (z : ℂ) : ℂ :=
  8 * riemannXi (1 / 2 + z)

theorem differentiable_xiJensenEntire : Differentiable ℂ xiJensenEntire := by
  unfold xiJensenEntire
  fun_prop

theorem xiJensenEntire_neg (z : ℂ) :
    xiJensenEntire (-z) = xiJensenEntire z := by
  have harg : (1 / 2 : ℂ) + -z = 1 - (1 / 2 + z) := by ring
  rw [xiJensenEntire, xiJensenEntire, harg, riemannXi_one_sub]

/-- The complex derivative normalization underlying the classical real
coefficient `γ(n)`: `n! ξ_J^(2n)(0) / (2n)!`. -/
def xiGammaComplex (n : ℕ) : ℂ :=
  (n.factorial : ℂ) / ((2 * n).factorial : ℂ) *
    iteratedDeriv (2 * n) xiJensenEntire 0

/-- Real interface to the xi coefficient sequence.  The remaining analytic
bridge must prove that `xiGammaComplex n` is real (and agrees with the sourced
Taylor expansion); taking `re` here does not assume that bridge. -/
def xiGamma (n : ℕ) : ℝ :=
  (xiGammaComplex n).re

theorem xiJensenEntire_taylor (z : ℂ) :
    ∑' n : ℕ, (n.factorial : ℂ)⁻¹ *
        iteratedDeriv n xiJensenEntire 0 * z ^ n =
      xiJensenEntire z := by
  simpa using
    Complex.taylorSeries_eq_of_entire'
      differentiable_xiJensenEntire (c := 0) (z := z)

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

/-- A real polynomial is hyperbolic when all roots of its complex scalar
extension are real. -/
def Hyperbolic (p : ℝ[X]) : Prop :=
  ∀ z : ℂ, (p.map (algebraMap ℝ ℂ)).IsRoot z →
    ∃ x : ℝ, z = (x : ℂ)

/-- The all-degree/all-shift statement occurring in the sourced
Pólya--Jensen criterion.  It is an explicit proof obligation, not a theorem. -/
def AllJensenHyperbolic (a : ℕ → ℝ) : Prop :=
  ∀ d n : ℕ, Hyperbolic (jensenPolynomial a d n)

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

/-- These are the exact still-unproved mathematical interfaces separating the
formalized finite algebra from RH. -/
structure BridgeObligations : Prop where
  coefficientReality : ∀ n, (xiGammaComplex n).im = 0
  coefficientTaylorExpansion :
    ∀ z : ℂ,
      HasSum (fun n : ℕ ↦
        ((xiGamma n / n.factorial : ℝ) : ℂ) * z ^ (2 * n))
        (xiJensenEntire z)
  polyaJensenForward :
    KakeyaRiemannHypothesisRoot → AllJensenHyperbolic xiGamma
  polyaJensenReverse :
    AllJensenHyperbolic xiGamma → KakeyaRiemannHypothesisRoot

end Kakeya.RHJensen
