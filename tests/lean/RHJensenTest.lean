import KakeyaLeanGate.RHJensen

/-! Focused compile-time tests for the Jensen/Laguerre--Pólya route. -/

open Complex Polynomial
open scoped ComplexConjugate

open Kakeya.RHJensen

example : ¬ Hyperbolic (0 : ℝ[X]) :=
  zero_not_hyperbolic

example :
    ¬ (∀ p : ℝ[X], Hyperbolic p → Hyperbolic p.derivative) :=
  not_hyperbolicity_preserved_by_every_derivative

example {p : ℝ[X]} (hp : Hyperbolic p) (hp' : p.derivative ≠ 0) :
    Hyperbolic p.derivative :=
  hyperbolic_derivative hp hp'

example (a : ℕ → ℝ) (ha : PointwiseNonzero a) :
    AllJensenHyperbolic a ↔
      ∀ d : ℕ, 1 ≤ d → Hyperbolic (jensenPolynomial a d 0) :=
  allJensenHyperbolic_iff_unshifted a ha

example (hconj : CompletedZetaConjugation) (n : ℕ) :
    (xiGammaComplex n).im = 0 :=
  xiGammaComplex_im_zero_of_completedZetaConjugation hconj n
