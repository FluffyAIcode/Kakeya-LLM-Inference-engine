import KakeyaLeanGate.LiCriterion

/-! Focused compile-time tests for the Li-coefficient route. -/

example : riemannXiLi 0 = 1 := riemannXiLi_zero

example : riemannXiLi 1 = 1 := riemannXiLi_one

example (s : ℂ) : riemannXiLi (1 - s) = riemannXiLi s :=
  riemannXiLi_one_sub s

example (a : ℕ → ℝ) (h : LiPositive a) (N : ℕ) :
    LiPositiveThrough a N :=
  h.through N

example (N : ℕ) :
    LiPositiveThrough (finitePrefixSpoof N) N ∧
      ¬LiPositive (finitePrefixSpoof N) :=
  ⟨finitePrefixSpoof_positiveThrough N, finitePrefixSpoof_not_positive N⟩

example (rho : ℂ) : liZeroSummand 1 rho = rho⁻¹ :=
  liZeroSummand_one rho

example (a : ℕ → ℝ) (N : ℕ) (x : ℝ) :
    liGeneratingPolynomial a (N + 1) x =
      liGeneratingPolynomial a N x + a N * x ^ N :=
  liGeneratingPolynomial_succ a N x
