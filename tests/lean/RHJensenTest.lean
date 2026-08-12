import KakeyaLeanGate.RHJensen

/-! Focused compile-time tests for the Jensen/Laguerre--Pólya route. -/

open Complex Polynomial
open MeasureTheory
open scoped ComplexConjugate Topology

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

example (s : ℂ) :
    completedRiemannZeta₀ (conj s) =
      conj (completedRiemannZeta₀ s) :=
  completedRiemannZeta₀_conj s

example : CompletedZetaConjugation :=
  completedZetaConjugation

example (n : ℕ) :
    xiGammaComplex n = (xiGamma n : ℂ) :=
  xiGammaComplex_eq_ofReal n

example (z : ℂ) :
    HasSum (fun n : ℕ ↦
        ((xiGamma n / n.factorial : ℝ) : ℂ) * z ^ (2 * n))
      (xiJensenEntire z) :=
  xiJensenEntire_hasSum_xiGamma z

example (s : ℂ) :
    completedRiemannZeta₀ s =
      mellin completedZetaMellinKernel (s / 2) / 2 :=
  completedRiemannZeta₀_eq_mellin s

example (n : ℕ) (u : ℝ) :
    0 < riemannPhiTerm n u :=
  riemannPhiTerm_pos n u

example (n : ℕ) {u : ℝ} (hu : 0 ≤ u) :
    riemannPhiTerm n u =
      deriv (deriv (riemannThetaTerm n)) u -
        (1 / 4 : ℝ) * riemannThetaTerm n u :=
  riemannPhiTerm_eq_theta_shifted_second_deriv n hu

example (R : ℝ) :
    Summable (riemannThetaC2Majorant R) :=
  summable_riemannThetaC2Majorant R

example (R : ℝ) :
    TendstoUniformlyOn
      (fun N u ↦ ∑ n ∈ Finset.range N,
        deriv (deriv (riemannThetaTerm n)) u)
      (fun u ↦ ∑' n : ℕ, deriv (deriv (riemannThetaTerm n)) u)
      Filter.atTop (Set.Icc 0 R) :=
  tendstoUniformlyOn_deriv_deriv_riemannThetaTerm_Icc R

example {u : ℝ} (hu : 0 < u) :
    riemannPhi u =
      deriv (deriv riemannThetaTail) u -
        (1 / 4 : ℝ) * riemannThetaTail u :=
  riemannPhi_eq_thetaTail_shifted_second_deriv hu

example {u : ℝ} (hu : 0 < u) :
    (riemannThetaTail u : ℂ) =
      (Real.exp (u / 2) / 2 : ℝ) *
        (HurwitzZeta.hurwitzEvenFEPair 0).f_modif (Real.exp (2 * u)) :=
  riemannThetaTail_eq_f_modif hu

example (u : ℝ) :
    Summable (fun n ↦ riemannPhiTerm n u) :=
  summable_riemannPhiTerm u

example (u : ℝ) :
    riemannPhi (-u) = riemannPhi u ∧ 0 < riemannPhi u :=
  ⟨riemannPhi_even u, riemannPhi_pos u⟩

example : Continuous riemannPhi :=
  continuous_riemannPhi

example (k : ℕ) :
    Integrable (fun u : ℝ ↦ riemannPhi u * u ^ k) :=
  integrable_riemannPhi_mul_pow k

example (h : XiGammaMomentRepresentation) :
    StrictlyPositive xiGamma :=
  xiGamma_strictlyPositive_of_momentRepresentation h

example (a : ℕ → ℝ) (n : ℕ)
    (hc : a (n + 3) ≠ 0)
    (hdisc : 0 ≤ jensenCubicDiscriminant a n) :
    Hyperbolic (jensenPolynomial a 3 n) :=
  jensenPolynomial_degree_three_hyperbolic a n hc hdisc

example :
    ∀ n, Hyperbolic (jensenPolynomial turanOnlySequence 2 n) :=
  turanOnlySequence_degree_two_hyperbolic

example :
    ¬ Hyperbolic (jensenPolynomial turanOnlySequence 3 0) :=
  turanOnlySequence_degree_three_not_hyperbolic

example (a : ℕ → ℝ) :
    NonzeroJensenFamily a ↔ JensenWindowNonzero a :=
  nonzeroJensenFamily_iff_windowNonzero a

example (a : ℕ → ℝ) :
    JensenWindowNonzero a ↔ NoAdjacentZeros a :=
  jensenWindowNonzero_iff_noAdjacentZeros a

example (a : ℕ → ℝ) :
    NonzeroJensenFamily a ↔ NoAdjacentZeros a :=
  nonzeroJensenFamily_iff_noAdjacentZeros a

example {a : ℕ → ℝ} (h : ToeplitzTotallyNonnegative a) :
    (∀ n, 0 ≤ a n) ∧ LogConcave a :=
  ⟨toeplitzTotallyNonnegative_nonnegative h,
    toeplitzTotallyNonnegative_logConcave h⟩

example :
    StrictlyPositive positiveNonPFSequence ∧
      ¬ PolyaFrequency positiveNonPFSequence :=
  ⟨positiveNonPFSequence_strictlyPositive,
    positiveNonPFSequence_not_polyaFrequency⟩

example {a : ℕ → ℝ}
    (hASW : FinitePolyaFrequencyHyperbolicity)
    (hpf : AllJensenPolyaFrequency a)
    (hne : NonzeroJensenFamily a) :
    AllJensenHyperbolic a :=
  allJensenHyperbolic_of_polyaFrequency hASW hpf hne

example (b : ℕ → ℝ)
    (hpf : PolyaFrequency b)
    (hne : finiteCoefficientPolynomial b 1 ≠ 0) :
    NonpositiveRooted (finiteCoefficientPolynomial b 1) :=
  finiteASW_degree_one b hpf hne

example (b : ℕ → ℝ)
    (hpf : PolyaFrequency b)
    (hb0 : 0 < b 0) (hb2 : 0 < b 2)
    (hsharp : 4 * b 0 * b 2 ≤ b 1 ^ 2) :
    NonpositiveRooted (finiteCoefficientPolynomial b 2) :=
  finiteASW_degree_two_of_sharp_bound b hpf hb0 hb2 hsharp

example :
    ¬ (∀ b : ℕ → ℝ, SelectedQuadraticToeplitzMinorConditions b →
      4 * b 0 * b 2 ≤ b 1 ^ 2) :=
  selected_quadratic_minors_do_not_imply_factor_four

example (hdet : QuadraticToeplitzDeterminantPrinciple)
    (b : ℕ → ℝ) (hsupport : ∀ n, 3 ≤ n → b n = 0)
    (hpf : PolyaFrequency b) (hb0 : 0 < b 0) (hb2 : 0 < b 2) :
    NonpositiveRooted (finiteCoefficientPolynomial b 2) :=
  finiteASW_degree_two_of_all_minors hdet b hsupport hpf hb0 hb2

example :
    FinitePolyaFrequencyHyperbolicityThrough 1 :=
  finitePolyaFrequencyHyperbolicityThrough_one

example {a : ℕ → ℝ} {k : ℕ}
    (hASW : FinitePolyaFrequencyHyperbolicityThrough k)
    (hpf : AllJensenPolyaFrequency a)
    (hne : NonzeroJensenFamily a) :
    JensenHyperbolicThrough a k :=
  jensenHyperbolicThrough_of_polyaFrequency hASW hpf hne

example (w : ℂ) :
    HasSum (fun n : ℕ ↦
      ((xiGamma n / n.factorial : ℝ) : ℂ) * w ^ n)
      (xiJensenGeneratingFunction w) :=
  xiJensenGeneratingFunction_hasSum w

example (z : ℂ) :
    xiJensenGeneratingFunction (z ^ 2) = xiJensenEntire z :=
  xiJensenGeneratingFunction_sq z

example :
    xiJensenGeneratingPowerSeries.radius = ⊤ :=
  xiJensenGeneratingPowerSeries_radius

example :
    Differentiable ℂ xiJensenGeneratingFunction :=
  differentiable_xiJensenGeneratingFunction

example :
    TendstoLocallyUniformlyOn
      (fun n z ↦ xiJensenGeneratingPowerSeries.partialSum n z)
      xiJensenGeneratingFunction Filter.atTop Set.univ :=
  xiJensenGeneratingFunction_tendstoLocallyUniformly

example {f : ℂ → ℂ} (h : LocallyUniformHyperbolicLimit f) :
    Differentiable ℂ f :=
  differentiable_of_locallyUniformHyperbolicLimit h

example {f : ℂ → ℂ} (h : LocallyUniformHyperbolicLimit f) (z : ℂ) :
    f (conj z) = conj (f z) :=
  conj_symmetry_of_locallyUniformHyperbolicLimit h z

example (hHurwitz : HurwitzHalfPlaneNonvanishingClosure)
    {f : ℂ → ℂ} (h : LocallyUniformHyperbolicLimit f)
    (hupper : ∃ w : ℂ, 0 < w.im ∧ f w ≠ 0)
    (hlower : ∃ w : ℂ, w.im < 0 ∧ f w ≠ 0)
    {z : ℂ} (hz : f z = 0) :
    z.im = 0 :=
  zero_reality_of_locallyUniformHyperbolicLimit
    hHurwitz h hupper hlower hz

example (hHurwitz : HurwitzHalfPlaneNonvanishingClosure)
    {f : ℂ → ℂ} (h : LocallyUniformHyperbolicLimit f)
    (hnonzero : ∃ w : ℂ, f w ≠ 0)
    {z : ℂ} (hz : f z = 0) :
    z.im = 0 :=
  zero_reality_of_locallyUniformHyperbolicLimit_of_nonzero
    hHurwitz h hnonzero hz

example (q : ℂ) (a b : ℝ) :
    (∫ u in a..b, (2 * Real.exp (2 * u)) •
      (((Real.exp (2 * u) : ℂ) ^ q) *
        (HurwitzZeta.hurwitzEvenFEPair 0).f_modif (Real.exp (2 * u)))) =
      ∫ x in Real.exp (2 * a)..Real.exp (2 * b),
        (x : ℂ) ^ q *
          (HurwitzZeta.hurwitzEvenFEPair 0).f_modif x :=
  intervalIntegral_mellin_exp_two_substitution q a b

example (s : ℂ) {R : ℝ} (hR : 0 ≤ R) :
    (∫ x in (1 : ℝ)..Real.exp (2 * R),
        (x : ℂ) ^ (s / 2 - 1) *
          (HurwitzZeta.hurwitzEvenFEPair 0).f_modif x) =
      ∫ u in (0 : ℝ)..R,
        4 * Complex.exp ((s - 1 / 2) * u) * riemannThetaTail u :=
  intervalIntegral_mellin_f_modif_eq_theta s hR

example {u : ℝ} (hu : 0 ≤ u) :
    ‖riemannThetaTail u‖ ≤
      (∑' n : ℕ, riemannPhiMajorantCoefficient n) *
        Real.exp (-(3 / 2 : ℝ) * u) :=
  norm_riemannThetaTail_le_decay hu

example :
    Filter.Tendsto riemannThetaTail Filter.atTop (𝓝 0) :=
  tendsto_riemannThetaTail_atTop_zero

example :
    Filter.Tendsto (deriv riemannThetaTail) Filter.atTop (𝓝 0) :=
  tendsto_deriv_riemannThetaTail_atTop_zero

example :
    Filter.Tendsto (deriv (deriv riemannThetaTail))
      Filter.atTop (𝓝 0) :=
  tendsto_deriv_deriv_riemannThetaTail_atTop_zero

example (u : ℝ) :
    riemannThetaTail (-u) =
      riemannThetaTail u +
        (Real.exp (u / 2) - Real.exp (-u / 2)) / 2 :=
  riemannThetaTail_neg u

example : deriv riemannThetaTail 0 = -(1 / 4 : ℝ) :=
  deriv_riemannThetaTail_zero

example (r : ℝ) {R : ℝ} (hR : 0 ≤ R) :
    (∫ u in (0 : ℝ)..R, Real.cosh (r * u) * riemannPhi u) =
      thetaCoshBoundary r R + 1 / 4 +
        (r ^ 2 - 1 / 4) *
          ∫ u in (0 : ℝ)..R,
            Real.cosh (r * u) * riemannThetaTail u :=
  intervalIntegral_cosh_mul_riemannPhi_double_ibp r hR

example {r : ℝ} (hr : |r| < 3 / 2) :
    (∫ u : ℝ in Set.Ioi 0, Real.cosh (r * u) * riemannPhi u) =
      1 / 4 + (r ^ 2 - 1 / 4) *
        ∫ u : ℝ in Set.Ioi 0,
          Real.cosh (r * u) * riemannThetaTail u :=
  integral_Ioi_cosh_mul_riemannPhi_double_ibp hr

example (b : ℕ → ℝ) :
    quadraticToeplitzShiftMinor b 2 = b 1 ^ 2 - b 0 * b 2 :=
  quadraticToeplitzShiftMinor_two b

example (b : ℕ → ℝ) (hsupport : ∀ n, 3 ≤ n → b n = 0) :
    quadraticToeplitzShiftMinor b 3 =
      b 1 ^ 3 - 2 * b 0 * b 1 * b 2 :=
  quadraticToeplitzShiftMinor_three b hsupport

example : HurwitzHalfPlaneNonvanishingClosure :=
  hurwitzHalfPlaneNonvanishingClosure

example {f : ℂ → ℂ} (h : LocallyUniformHyperbolicLimit f)
    (hnonzero : ∃ w : ℂ, f w ≠ 0)
    {z : ℂ} (hz : f z = 0) :
    z.im = 0 :=
  zero_reality_of_locallyUniformHyperbolicLimit_unconditional h hnonzero hz

example (h : XiGammaMomentRepresentation) :
    XiCoefficientNondegeneracy :=
  xiCoefficientNondegeneracy_of_momentRepresentation h

example (hLP : XiLaguerrePolyaMembership)
    (hmoment : XiGammaMomentRepresentation)
    {z : ℂ} (hz : xiJensenGeneratingFunction z = 0) :
    z.im = 0 :=
  xiLaguerrePolya_zero_reality_of_momentRepresentation hLP hmoment hz

example (h : AnalyticBridgeObligations) :
    BridgeObligations :=
  h.toBridgeObligations

example (h : XiCoefficientNondegeneracy) :
    AllJensenHyperbolic xiGamma ↔ XiUnshiftedHyperbolicity :=
  allJensenHyperbolic_xiGamma_iff_unshifted h

example (s : ℂ) {R : ℝ} (hR : 0 ≤ R) :
    (∫ x in Real.exp (-2 * R)..Real.exp (2 * R),
        (x : ℂ) ^ (s / 2 - 1) *
          (HurwitzZeta.hurwitzEvenFEPair 0).f_modif x) =
      ∫ u in (0 : ℝ)..R,
        8 * Complex.cosh ((s - 1 / 2) * u) * riemannThetaTail u :=
  intervalIntegral_mellin_f_modif_eq_cosh_theta s hR

example {r : ℝ} (hr : |r| < 3 / 2) :
    completedRiemannZeta₀ (1 / 2 + r : ℝ) =
      (4 * ∫ u : ℝ in Set.Ioi 0,
        Real.cosh (r * u) * riemannThetaTail u : ℝ) :=
  completedRiemannZeta₀_real_eq_cosh_theta hr

example (n : ℕ) :
    iteratedDeriv n xiJensenEntire 0 =
      (8 * ∫ u : ℝ, riemannPhi u * u ^ n : ℝ) :=
  iteratedDeriv_xiJensenEntire_eq_riemannPhi_moment n

example : XiGammaMomentRepresentation :=
  xiGamma_momentRepresentation

example : StrictlyPositive xiGamma :=
  xiGamma_strictlyPositive

example : XiCoefficientNondegeneracy :=
  xiCoefficientNondegeneracy_unconditional

example :
    AllJensenHyperbolic xiGamma ↔ XiUnshiftedHyperbolicity :=
  allJensenHyperbolic_xiGamma_iff_unshifted_unconditional

example
    (hrec : QuadraticToeplitzContinuantRecurrence)
    (hosc : QuadraticContinuantOscillationPrinciple) :
    QuadraticToeplitzDeterminantPrinciple :=
  quadraticToeplitzDeterminantPrinciple_of_recurrence_oscillation hrec hosc

example (a p q : ℝ) (k : ℕ) :
    (toeplitzContinuantMatrix a p q (k + 2)).det =
      a * (toeplitzContinuantMatrix a p q (k + 1)).det -
        p * q * (toeplitzContinuantMatrix a p q k).det :=
  toeplitzContinuantMatrix_det_recurrence a p q k

example : QuadraticToeplitzContinuantRecurrence :=
  quadraticToeplitzContinuantRecurrence

example : QuadraticContinuantOscillationPrinciple :=
  quadraticContinuantOscillationPrinciple

example : QuadraticToeplitzDeterminantPrinciple :=
  quadraticToeplitzDeterminantPrinciple

example (b : ℕ → ℝ) (hsupport : ∀ n, 3 ≤ n → b n = 0)
    (hpf : PolyaFrequency b) (hb0 : 0 < b 0) (hb2 : 0 < b 2) :
    NonpositiveRooted (finiteCoefficientPolynomial b 2) :=
  finiteASW_degree_two b hsupport hpf hb0 hb2

example (hpf : AllJensenPolyaFrequency xiGamma) :
    JensenHyperbolicThrough xiGamma 2 :=
  xiJensenHyperbolicThrough_two_of_polyaFrequency hpf

example (hASW : FinitePolyaFrequencyHyperbolicity)
    (hpf : AllJensenPolyaFrequency xiGamma) :
    AllJensenHyperbolic xiGamma :=
  allJensenHyperbolic_xiGamma_of_polyaFrequency hASW hpf

example (b : ℕ → ℝ) :
    finiteCoefficientPolynomial b 3 =
      Polynomial.C (b 0) + Polynomial.C (b 1) * Polynomial.X +
        Polynomial.C (b 2) * Polynomial.X ^ 2 +
          Polynomial.C (b 3) * Polynomial.X ^ 3 :=
  finiteCoefficientPolynomial_degree_three b

example (hcubic : CubicToeplitzDiscriminantPrinciple)
    (b : ℕ → ℝ) (hsupport : ∀ n, 4 ≤ n → b n = 0)
    (hpf : PolyaFrequency b) (hb0 : 0 < b 0) (hb3 : 0 < b 3) :
    NonpositiveRooted (finiteCoefficientPolynomial b 3) :=
  finiteASW_degree_three hcubic b hsupport hpf hb0 hb3

example (hASW : FiniteAissenSchoenbergWhitney) :
    FinitePolyaFrequencyHyperbolicity :=
  hASW.toHyperbolicity

example (hJ : AllJensenHyperbolic xiGamma) (d : ℕ) :
    Hyperbolic (xiRescaledJensenPolynomial d) :=
  xiRescaledJensenPolynomial_hyperbolic hJ d

example (j : ℕ) :
    Filter.Tendsto
      (fun d : ℕ ↦
        (((d + 1).choose j : ℝ) * (((d + 1 : ℕ) : ℝ)⁻¹) ^ j))
      Filter.atTop (𝓝 ((j.factorial : ℝ)⁻¹)) :=
  tendsto_choose_mul_inv_pow j

example (d j : ℕ) :
    (((d + 1).choose j : ℝ) * (((d + 1 : ℕ) : ℝ)⁻¹) ^ j) ≤
      (j.factorial : ℝ)⁻¹ :=
  choose_mul_inv_pow_le_factorial_inv d j

example (z : ℂ) :
    Filter.Tendsto
      (fun d ↦ complexPolynomialEval (xiRescaledJensenPolynomial d) z)
      Filter.atTop (𝓝 (xiJensenGeneratingFunction z)) :=
  xiRescaledJensenPolynomial_tendsto_pointwise z

example : XiJensenScalingConvergence :=
  xiJensenScalingConvergence

example (s : ℂ) :
    xiJensenGeneratingFunction ((s - 1 / 2) ^ 2) = 8 * riemannXi s :=
  xiJensenGeneratingFunction_centered_square s

example (hconv : XiJensenScalingConvergence)
    (hJ : AllJensenHyperbolic xiGamma) :
    XiLaguerrePolyaMembership :=
  xiLaguerrePolyaMembership_of_allJensenHyperbolic hconv hJ

example (hJ : AllJensenHyperbolic xiGamma) :
    XiLaguerrePolyaMembership :=
  xiLaguerrePolyaMembership_of_allJensenHyperbolic_unconditional hJ

example (hJ : AllJensenHyperbolic xiGamma) {z : ℂ}
    (hz : xiJensenEntire z = 0) :
    z.re = 0 :=
  xiJensenEntire_zero_re_on_critical_axis hJ hz

example (hJ : AllJensenHyperbolic xiGamma) :
    KakeyaRiemannHypothesisRoot :=
  riemannHypothesis_of_allJensenHyperbolic hJ

example (hRH : KakeyaRiemannHypothesisRoot) {z : ℂ}
    (hz : xiJensenEntire z = 0) :
    z.re = 0 :=
  xiJensenEntire_zero_re_on_critical_axis_of_riemannHypothesis hRH hz

example :
    XiCenteredCriticalZeroReality ↔ KakeyaRiemannHypothesisRoot :=
  xiCenteredCriticalZeroReality_iff_riemannHypothesis

example (z : ℂ) :
    centeredXiZeroDivisor z =
      (centeredXiZeroMultiplicity z : ℤ) :=
  centeredXiZeroDivisor_apply z

example (z : ℂ) :
    (∃ i : CenteredXiZeroIndex, centeredXiZeroRoot i = z) ↔
      xiJensenEntire z = 0 :=
  exists_centeredXiZeroRoot_iff z

example (hzero : XiCenteredCriticalZeroReality)
    (hsum : XiCenteredZeroInverseSquareSummability) :
    HasProdLocallyUniformlyOn
      (fun (i : CenteredXiPositiveZeroIndex) (w : ℂ) ↦
        1 + ((centeredXiPositiveSquaredOrdinate i)⁻¹ : ℝ) * w)
      (fun w : ℂ ↦ ∏' i : CenteredXiPositiveZeroIndex,
        (1 + ((centeredXiPositiveSquaredOrdinate i)⁻¹ : ℝ) * w))
      Set.univ :=
  hasProdLocallyUniformlyOn_centeredXiPositiveFactors hzero hsum

example (hcount : XiCenteredDyadicOrderOneZeroCount) :
    XiCenteredZeroInverseSquareSummability :=
  centeredXi_inverseSquareSummability_of_orderOneZeroCount hcount

example (hcount : XiCenteredCumulativeZeroCount) :
    XiCenteredZeroInverseSquareSummability :=
  centeredXi_inverseSquareSummability_of_cumulativeZeroCount hcount

example {ι : Type*} {f : ι → ℂ}
    (hsum : Summable (fun i ↦ ‖f i‖))
    (hhalf : ∀ i, ‖f i‖ ≤ 1 / 2) :
    Real.exp (-(3 / 2 : ℝ) * ∑' i, ‖f i‖) ≤
      ‖∏' i, (1 + f i)‖ :=
  norm_tprod_one_add_lower_of_norm_le_half_jensen hsum hhalf

example (hzero : XiCenteredCriticalZeroReality)
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
  exists_circle_norm_centeredXiPairedProduct_lower_of_cumulative
    hzero hcount hs hT hH ha hwidth hbT

example (hzero : XiCenteredCriticalZeroReality)
    (hsum : XiCenteredZeroInverseSquareSummability) (z : ℂ) :
    centeredXiPairedProduct z = 0 ↔ xiJensenEntire z = 0 :=
  centeredXiPairedProduct_zero_set_eq hzero hsum z

example (hzero : XiCenteredCriticalZeroReality)
    (hsum : XiCenteredZeroInverseSquareSummability) :
    Function.support
        (MeromorphicOn.divisor centeredXiPairedProduct Set.univ) =
      Function.support centeredXiZeroDivisor :=
  centeredXiPairedProduct_divisor_support_eq hzero hsum

example (hzero : XiCenteredCriticalZeroReality)
    (hsum : XiCenteredZeroInverseSquareSummability) (z : ℂ) :
    analyticOrderAt centeredXiPairedProduct z =
      centeredXiZeroMultiplicity z := by
  rw [analyticOrderAt_centeredXiPairedProduct hzero hsum,
    card_centeredXiPairedFiber hzero]

example (hzero : XiCenteredCriticalZeroReality)
    (hsum : XiCenteredZeroInverseSquareSummability) :
    XiCenteredPairedProductDivisorEquality :=
  xiCenteredPairedProductDivisorEquality hzero hsum

example (hzero : XiCenteredCriticalZeroReality)
    (hsum : XiCenteredZeroInverseSquareSummability) :
    AnalyticOnNhd ℂ centeredXiCancelledQuotient Set.univ ∧
      ∀ z, centeredXiCancelledQuotient z ≠ 0 :=
  centeredXiCancelledQuotient_analytic_ne_zero hzero hsum

example (hzero : XiCenteredCriticalZeroReality)
    (hsum : XiCenteredZeroInverseSquareSummability)
    (hgrowth : XiCenteredCancelledQuotientLogGrowth) :
    XiCenteredPairedProductAffineRigidity :=
  centeredXiPairedProductAffineRigidity_of_logGrowth hzero hsum hgrowth

example (hzero : XiCenteredCriticalZeroReality)
    (hsum : XiCenteredZeroInverseSquareSummability)
    (hcircle : XiCenteredCancelledQuotientSelectedCircleGrowth) :
    XiCenteredCancelledQuotientLogNormGrowth :=
  xiCenteredCancelledQuotientLogNormGrowth_of_selectedCircle
    hzero hsum hcircle

example (hcartan : XiCenteredCancelledQuotientCartanDiskGrowth) :
    XiCenteredCancelledQuotientSelectedCircleGrowth :=
  xiCenteredCancelledQuotientSelectedCircleGrowth_of_cartan hcartan

example (hzero : XiCenteredCriticalZeroReality)
    (hcount : XiCenteredCumulativeZeroCount)
    (hcartan : XiCenteredCancelledQuotientCartanDiskGrowth) :
    ∀ z : ℂ, xiJensenEntire z =
      (xiGamma 0 : ℂ) * centeredXiPairedProduct z :=
  centeredXi_eq_normalizedPairedProduct_of_count_and_cartan
    hzero hcount hcartan

example (hcount : XiCenteredCumulativeZeroCount) :
    ∃ B R : ℝ, 0 ≤ B ∧ 1 ≤ R ∧ ∀ j : ℕ,
      R ≤ (2 : ℝ) ^ (j + 2) →
      ((centeredXiPositiveZeroWindow
        ((2 : ℝ) ^ (j + 2))).card : ℝ) ≤
        B * (2 : ℝ) ^ j * (j + 1 : ℝ) :=
  exists_centeredXiPositiveZeroWindow_card_le_dyadic hcount

example : XiCenteredCumulativeZeroCount :=
  xiCenteredCumulativeZeroCount_actual

example : XiCenteredZeroInverseSquareSummability :=
  centeredXi_inverseSquareSummability_actual

example :
    ∃ B R : ℝ, 0 ≤ B ∧ 1 ≤ R ∧ ∀ j : ℕ,
      R ≤ (2 : ℝ) ^ (j + 2) →
      ((centeredXiPositiveZeroWindow
        ((2 : ℝ) ^ (j + 2))).card : ℝ) ≤
        B * (2 : ℝ) ^ j * (j + 1 : ℝ) :=
  exists_centeredXiPositiveZeroWindow_card_le_dyadic_actual

example (hzero : XiCenteredCriticalZeroReality)
    (hsum : XiCenteredZeroInverseSquareSummability)
    (Ksq Klin : ℝ) :
    Filter.Tendsto
      (CenteredXiDyadicNormalizedLoss Ksq Klin fun j ↦
        ∑' i : {i : CenteredXiPositiveZeroIndex //
            i ∉ centeredXiPositiveZeroWindow ((2 : ℝ) ^ j)},
          ‖(((centeredXiPositiveSquaredOrdinate i)⁻¹ : ℝ) : ℂ)‖)
      Filter.atTop (𝓝 0) :=
  tendsto_centeredXiPairedProduct_normalizedLoss hzero hsum Ksq Klin

example (hzero : XiCenteredCriticalZeroReality)
    (hsum : XiCenteredZeroInverseSquareSummability)
    (haffine : XiCenteredPairedProductAffineRigidity) :
    ∀ z : ℂ, xiJensenEntire z =
      (xiGamma 0 : ℂ) * centeredXiPairedProduct z :=
  centeredXi_eq_normalizedPairedProduct_of_affine hzero hsum haffine

example (hzero : XiCenteredCriticalZeroReality) :
    Filter.Tendsto
      (CenteredXiDyadicShiftedLoss 7 11)
      Filter.atTop (𝓝 0) :=
  tendsto_centeredXiDyadicShiftedLoss hzero 7 11

example (hzero : XiCenteredCriticalZeroReality) :
    SubquadraticLogNormGrowthJensen centeredXiCancelledQuotient :=
  xiCenteredCancelledQuotientLogNormGrowth_of_selectedCircle
    hzero centeredXi_inverseSquareSummability_actual
    (subquadraticBoundaryLogNormGrowth_of_dyadic_jensen
      (centeredXiCancelledQuotient_dyadicBoundary_actual hzero))

example (hzero : XiCenteredCriticalZeroReality) :
    ∀ z : ℂ, xiJensenEntire z =
      (xiGamma 0 : ℂ) * centeredXiPairedProduct z :=
  centeredXi_eq_normalizedPairedProduct_actual hzero

example (hconverse : XiRealZeroToJensenHyperbolicity) :
    AllJensenHyperbolic xiGamma ↔ KakeyaRiemannHypothesisRoot :=
  allJensenHyperbolic_iff_riemannHypothesis hconverse

example {c : ℝ} (hc : c ≠ 0) {root : ℕ → ℝ}
    (hroot : ∀ i, root i ≠ 0) (N : ℕ) :
    Hyperbolic (positiveGenusZeroPolynomial c root N) :=
  positiveGenusZeroPolynomial_hyperbolic hc hroot N

example (z : ℂ) {γ : ℝ} (hγ : γ ≠ 0) :
    centeredGenusOnePrimaryFactor (z / (Complex.I * γ)) *
        centeredGenusOnePrimaryFactor (z / (-Complex.I * γ)) =
      1 + z ^ 2 / (γ : ℂ) ^ 2 :=
  centeredGenusOnePrimaryFactor_pair z hγ

example (root : ℕ → ℝ)
    (hsum : Summable (fun i ↦ |(root i)⁻¹|)) :
    HasProdLocallyUniformlyOn
      (fun (i : ℕ) (z : ℂ) ↦ 1 + ((root i)⁻¹ : ℝ) * z)
      (fun z : ℂ ↦ ∏' i : ℕ, (1 + ((root i)⁻¹ : ℝ) * z))
      Set.univ :=
  hasProdLocallyUniformlyOn_positiveGenusZeroFactors root hsum

example (h : XiNormalizedGenusZeroProductIdentification) :
    XiLaguerrePolyaMembership :=
  xiLaguerrePolyaMembership_of_normalizedProductIdentification h

example {f P : ℂ → ℂ} (a b : ℂ)
    (hf : Differentiable ℂ f) (hP : Differentiable ℂ P)
    (hfeven : ∀ z, f (-z) = f z) (hPeven : ∀ z, P (-z) = P z)
    (hP0 : P 0 ≠ 0)
    (hrep : ∀ z, f z = Complex.exp (a + b * z) * P z) :
    ∀ z, f z = Complex.exp a * P z :=
  exp_affine_prefactor_eq_const_of_even a b hf hP hfeven hPeven hP0 hrep

example (h : XiCenteredAffinePairedProductIdentification) :
    XiNormalizedGenusZeroProductIdentification :=
  normalizedProductIdentification_of_centeredAffine h

example (hproduct : XiCriticalZerosToPositiveGenusZeroProduct)
    (hJensen : XiLaguerrePolyaToAllJensen) :
    AllJensenHyperbolic xiGamma ↔ KakeyaRiemannHypothesisRoot :=
  allJensenHyperbolic_iff_riemannHypothesis_of_canonicalProduct
    hproduct hJensen

example {a : ℕ → ℝ} (ha : NonzeroJensenFamily a) (ha0 : a 0 ≠ 0)
    (happrox : JensenCoefficientwiseHyperbolicApproximation a) :
    AllJensenHyperbolic a :=
  allJensenHyperbolic_of_coefficientwiseApproximation ha ha0 happrox

example (hproduct : XiCriticalZerosToPositiveGenusZeroProduct)
    (hfinite : FinitePositiveGenusZeroJensenHyperbolicity)
    (hcoeff : XiPositiveGenusZeroCoefficientConvergence) :
    XiReverseJensenApproximation :=
  xiReverseJensenApproximation_of_positiveGenusZero
    hproduct hfinite hcoeff

example (hreverse : XiReverseJensenApproximation) :
    AllJensenHyperbolic xiGamma ↔ KakeyaRiemannHypothesisRoot :=
  allJensenHyperbolic_iff_riemannHypothesis_of_reverseJensenApproximation
    hreverse

example (n j : ℕ) (p q : ℝ[X]) :
    (schurSzegoComposition n p q).coeff j =
      if j ≤ n then p.coeff j * q.coeff j / (n.choose j : ℝ)
      else 0 :=
  coeff_schurSzegoComposition n p q j

example (c : ℝ) (root : ℕ → ℝ) (N d : ℕ) :
    schurSzegoComposition N (positiveGenusZeroPolynomial c root N)
        (schurSzegoJensenKernel N d) =
      jensenPolynomial
        (positiveGenusZeroExponentialCoefficient c root N) d 0 :=
  schurSzegoComposition_positiveGenusZero_eq_jensenPolynomial c root N d

example (hschur : FiniteSchurSzegoNonpositiveRootedness)
    (hkernel : SchurSzegoJensenKernelNonpositiveRootedness) :
    FinitePositiveGenusZeroJensenHyperbolicity :=
  finitePositiveGenusZeroJensenHyperbolicity_of_schurSzego hschur hkernel

example : ¬ StrictFiniteSchurSzegoNonpositiveRootedness :=
  not_strictFiniteSchurSzegoNonpositiveRootedness

example (d : ℕ) :
    NonpositiveRooted (schurSzegoJensenKernel 0 d) :=
  schurSzegoJensenKernel_nonpositiveRooted_zero_left d

example (N : ℕ) :
    NonpositiveRooted (schurSzegoJensenKernel N 0) :=
  schurSzegoJensenKernel_nonpositiveRooted_zero_right N

example (N : ℕ) :
    NonpositiveRooted (schurSzegoJensenKernel N 1) :=
  schurSzegoJensenKernel_nonpositiveRooted_one_right N

example (hHadamard : XiCriticalZerosToNormalizedProductIdentification)
    (hJensen : XiLaguerrePolyaToAllJensen) :
    AllJensenHyperbolic xiGamma ↔ KakeyaRiemannHypothesisRoot :=
  allJensenHyperbolic_iff_riemannHypothesis_of_identification
    hHadamard hJensen

example (hHadamard : XiCriticalZerosToCenteredAffineProductIdentification)
    (hJensen : XiLaguerrePolyaToAllJensen) :
    AllJensenHyperbolic xiGamma ↔ KakeyaRiemannHypothesisRoot :=
  allJensenHyperbolic_iff_riemannHypothesis_of_centeredAffine
    hHadamard hJensen

example (hASW : FiniteAissenSchoenbergWhitney)
    (hpf : AllJensenPolyaFrequency xiGamma) :
    AllJensenHyperbolic xiGamma :=
  allJensenHyperbolic_xiGamma_of_finiteASW hASW hpf

example (hASW : FiniteAissenSchoenbergWhitney)
    (hpf : AllJensenPolyaFrequency xiGamma) :
    KakeyaRiemannHypothesisRoot :=
  riemannHypothesis_of_finiteASW_polyaFrequency hASW hpf
