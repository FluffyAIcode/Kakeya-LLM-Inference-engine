import KakeyaLeanGate.LiCriterion

/-! Focused compile-time tests for the Li-coefficient route. -/

open Filter
open scoped Topology

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

example (zeros : FiniteZeroMultiset) :
    zeros.liSum 1 = ∑ i, (zeros.root i)⁻¹ :=
  zeros.liSum_one

example (multiplicity n : ℕ) (rho : ℂ) (hρ : rho ≠ 0) :
    (FiniteZeroMultiset.replicate multiplicity rho hρ).liSum n =
      multiplicity * liZeroSummand n rho :=
  FiniteZeroMultiset.liSum_replicate multiplicity n rho hρ

example (zeros : FiniteZeroMultiset)
    (hline : ∀ i, (zeros.root i).re = 1 / 2) (n : ℕ) :
    0 ≤ (zeros.liSum n).re :=
  zeros.liSum_re_nonneg_of_on_criticalLine hline n

example {ι : Type*} [DecidableEq ι]
    (root : ι → ℂ) (cutoff : ℕ → Finset ι) (height : ℕ → ℝ)
    (hheight : Tendsto height atTop atTop)
    (hcutoff : ∀ N i, i ∈ cutoff N ↔ |(root i).im| ≤ height N)
    (hcauchy : HeightSymmetricLiCauchy root cutoff) :
    ∃ coefficient : ℕ → ℂ,
      HeightSymmetricLiLimit root cutoff height coefficient :=
  exists_heightSymmetricLiLimit_of_cauchy
    root cutoff height hheight hcutoff hcauchy

example (zeros : FiniteZeroMultiset) :
    logDeriv zeros.generatingProduct 0 = zeros.liSum 1 :=
  zeros.logDeriv_generatingProduct_zero

example {rho : ℂ} (hρ : rho ≠ 1) :
    riemannZetaZeroOrder rho ≠ 0 ↔ rho ∈ riemannZetaZeros :=
  riemannZetaZeroOrder_ne_zero_iff hρ

example {ι : Type*} {p : Filter ι} [p.NeBot]
    {f : ι → ℂ → ℂ} {g : ℂ → ℂ} {s : Set ℂ} {x : ℂ}
    (hs : IsOpen s) (hx : x ∈ s)
    (hconv : TendstoLocallyUniformlyOn f g p s)
    (hhol : ∀ᶠ i in p, DifferentiableOn ℂ (f i) s)
    (hf0 : ∀ᶠ i in p, ∀ z ∈ s, f i z ≠ 0)
    (hg0 : ∀ z ∈ s, g z ≠ 0) (k : ℕ) :
    Tendsto (fun i ↦ iteratedDeriv k (logDeriv (f i)) x) p
      (𝓝 (iteratedDeriv k (logDeriv g) x)) :=
  iteratedDeriv_logDeriv_tendsto hs hx hconv hhol hf0 hg0 k

example (zeros : FiniteZeroMultiset) :
    zeros.logDerivCoefficient 0 = zeros.liSum 1 :=
  zeros.logDerivCoefficient_zero

example (zeros : ℕ → FiniteZeroMultiset) (g : ℂ → ℂ) (s : Set ℂ)
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
      (𝓝 (iteratedDeriv k (logDeriv g) 0 / (k.factorial : ℂ))) :=
  finiteZeroLiSums_tendsto_of_generatingProducts
    zeros g s hs h0 hconv hhol happrox0 hlimit0 hexact k

example (a : ℕ → ℝ) (N : ℕ) (x : ℝ) :
    liGeneratingPolynomial a (N + 1) x =
      liGeneratingPolynomial a N x + a N * x ^ N :=
  liGeneratingPolynomial_succ a N x
