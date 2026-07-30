import Mathlib

/-!
A small, target-supporting Jensen-polynomial lemma.

This file does not define the Riemann xi function and does not claim any
finite computation proves the Riemann Hypothesis.  It only validates the
quadratic formula for the degree-two Jensen polynomial attached to an
arbitrary real coefficient sequence.
-/

def jensenQuadratic (a : ℕ → ℝ) (n : ℕ) (x : ℝ) : ℝ :=
  a n + 2 * a (n + 1) * x + a (n + 2) * x ^ 2

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
