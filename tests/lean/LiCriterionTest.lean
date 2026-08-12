import KakeyaLeanGate.LiCriterion

/-! Focused compile-time tests for the Li-coefficient route. -/

open Filter
open scoped Topology

example : riemannXiLi 0 = 1 := riemannXiLi_zero

example : riemannXiLi 1 = 1 := riemannXiLi_one

example (rho : ℂ) :
    riemannXiZeroOrder (1 - rho) = riemannXiZeroOrder rho :=
  riemannXiZeroOrder_one_sub rho

example : Countable RiemannXiZeroIndex := inferInstance

example (z : ℂ) :
    analyticOrderAt riemannXiLi z =
      ((riemannXiZeroFiber z).card : ℕ∞) :=
  analyticOrderAt_riemannXiLi_eq_zeroFiber_card z

example :
    Tendsto (fun i : RiemannXiZeroIndex ↦ ‖riemannXiZeroRoot i‖)
      cofinite atTop :=
  riemannXiZeroRoot_escape

example (rho : ℂ) :
    riemannXiZeroMultiplicity (1 - rho) =
      riemannXiZeroMultiplicity rho :=
  riemannXiZeroMultiplicity_one_sub rho

example :
    Function.support riemannXiZeroDivisor = riemannXiLi ⁻¹' {0} :=
  riemannXiZeroDivisor_support

example {K : Set ℂ} (hK : IsCompact K) :
    (K ∩ Function.support riemannXiZeroDivisor).Finite :=
  riemannXiZeroDivisor_finite_on_compact hK

example :
    Function.locallyFinsuppWithin.logCounting riemannXiZeroDivisor =
      ValueDistribution.logCounting riemannXiLi 0 :=
  riemannXiZeroDivisor_logCounting

example {R : ℝ} (hR : R ≠ 0) :
    Function.locallyFinsuppWithin.logCounting riemannXiZeroDivisor R =
      Real.circleAverage (Real.log ‖riemannXiLi ·‖) 0 R :=
  riemannXiZeroDivisor_logCounting_eq_circleAverage hR

example {K : Set ℂ} (hK : IsCompact K) :
    ∃ C : ℝ, ∀ s ∈ K, ‖riemannXiLi s‖ ≤ C :=
  riemannXiLi_bounded_on_compact hK

example (M : ℝ → ℝ)
    (hbound : ∀ s : ℂ, 1 / 2 ≤ s.re →
      ‖riemannXiLi s‖ ≤ M (max ‖s‖ ‖1 - s‖)) :
    ∀ s : ℂ, ‖riemannXiLi s‖ ≤ M (max ‖s‖ ‖1 - s‖) :=
  riemannXiLi_global_norm_bound_of_rightHalfPlane M hbound

example {s : ℂ} (hs : 2 ≤ s.re) :
    ‖riemannZeta s‖ ≤ ∑' n : ℕ, 1 / (n : ℝ) ^ 2 :=
  norm_riemannZeta_le_tsum_inv_sq_of_two_le_re hs

example (s : ℂ) (hs : 0 < s.re) :
    ‖Complex.Gamma s‖ ≤ Real.Gamma s.re :=
  norm_complexGamma_le_realGamma s hs

example (x : ℝ) (hx : 2 ≤ x) :
    Real.Gamma x ≤ (Nat.ceil x : ℝ) ^ Nat.ceil x :=
  realGamma_le_ceil_pow_self x hx

example {s : ℂ} (hs : 4 ≤ s.re) :
    ‖riemannXiLi s‖ ≤
      ‖s‖ * ‖s - 1‖ *
        ((Nat.ceil (s.re / 2) : ℝ) ^ Nat.ceil (s.re / 2)) *
        (∑' n : ℕ, 1 / (n : ℝ) ^ 2) :=
  norm_riemannXiLi_le_farRight hs

example {s : ℂ} (hs : s.re ≤ -3) :
    ‖riemannXiLi s‖ ≤
      ‖1 - s‖ * ‖s‖ *
        ((Nat.ceil ((1 - s).re / 2) : ℝ) ^
          Nat.ceil ((1 - s).re / 2)) *
        (∑' n : ℕ, 1 / (n : ℝ) ^ 2) :=
  norm_riemannXiLi_le_farLeft hs

example (a b : ℝ) :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ s : ℂ, a ≤ s.re → s.re ≤ b →
      ‖completedRiemannZeta₀ s‖ ≤ C :=
  exists_norm_completedRiemannZeta₀_le_on_verticalStrip a b

example (a b : ℝ) :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ s : ℂ, a ≤ s.re → s.re ≤ b →
      ‖riemannXiLi s‖ ≤ 1 + ‖s‖ * ‖s - 1‖ * C :=
  exists_norm_riemannXiLi_le_on_verticalStrip a b

example : RiemannXiOrderOneGrowthBound :=
  riemannXi_orderOneGrowthBound

example (R : ℝ) :
    ∃ g : ℂ → ℂ, Complex.CanonicalDecomp riemannXiLi g R :=
  riemannXiLi_exists_canonicalDecomp R

example : Nonempty RiemannXiCanonicalDiskExhaustion :=
  exists_riemannXiCanonicalDiskExhaustion

example {rho : ℂ} (h0 : rho ≠ 0) (h1 : rho ≠ 1)
    (hre : 0 < rho.re) :
    riemannXiZeroMultiplicity rho =
      riemannZetaZeroMultiplicity rho :=
  riemannXiZeroMultiplicity_eq_riemannZetaZeroMultiplicity h0 h1 hre

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

example {rho : ℂ} (hρ : rho ≠ 0) :
    ‖1 - rho⁻¹‖ ≤ 1 ↔ 1 / 2 ≤ rho.re :=
  norm_one_sub_inv_le_one_iff_re_ge_half hρ

example (zeros : FiniteZeroMultiset)
    (hsym : zeros.ReflectionSymmetric)
    (hconverse : FiniteLiSpectralConverseStatement zeros)
    (hpositive : ∀ n : ℕ, 0 < n → 0 ≤ (zeros.liSum n).re) :
    ∀ i, (zeros.root i).re = 1 / 2 :=
  zeros.on_criticalLine_of_spectralConverse hsym hconverse hpositive

example {ι : Type*} [DecidableEq ι]
    (root : ι → ℂ) (cutoff : ℕ → Finset ι) (height : ℕ → ℝ)
    (hheight : Tendsto height atTop atTop)
    (hcutoff : ∀ N i, i ∈ cutoff N ↔ |(root i).im| ≤ height N)
    (hcauchy : HeightSymmetricLiCauchy root cutoff) :
    ∃ coefficient : ℕ → ℂ,
      HeightSymmetricLiLimit root cutoff height coefficient :=
  exists_heightSymmetricLiLimit_of_cauchy
    root cutoff height hheight hcutoff hcauchy

example {ι : Type*} [DecidableEq ι]
    {root : ι → ℂ} {cutoff : ℕ → Finset ι} {height : ℕ → ℝ}
    {coefficient : ℕ → ℂ}
    (hroot0 : ∀ i, root i ≠ 0)
    (hsym : ReflectionSymmetricSpectrum root)
    (hsum : BombieriLagariasSummability root)
    (hlimit : HeightSymmetricLiLimit root cutoff height coefficient)
    (hpositive : ∀ n : ℕ, 0 < n → 0 ≤ (coefficient n).re)
    (hconverse : HeightSymmetricLiSpectralConverseStatement
      root cutoff height coefficient) :
    ∀ i, (root i).re = 1 / 2 :=
  on_criticalLine_of_heightSymmetric_spectralConverse
    hroot0 hsym hsum hlimit hpositive hconverse

example {ι : Type*} [Fintype ι] [DecidableEq ι]
    {root : ι → ℂ} {cutoff : ℕ → Finset ι} {height : ℕ → ℝ}
    {coefficient : ℕ → ℂ}
    (hlimit : HeightSymmetricLiLimit root cutoff height coefficient)
    (n : ℕ) :
    coefficient n = ∑ i, liZeroSummand n (root i) :=
  hlimit.coefficient_eq_finite_sum n

example {ι : Type*} [DecidableEq ι]
    {root : ι → ℂ} {cutoff : ℕ → Finset ι} {height : ℕ → ℝ}
    (hexhaust : SymmetricHeightExhaustion root cutoff height)
    (n : ℕ) (hsum : Summable (fun i ↦ liZeroSummand n (root i))) :
    Tendsto (fun N ↦ liZeroPartialSum root (cutoff N) n) atTop
      (𝓝 (∑' i, liZeroSummand n (root i))) :=
  hexhaust.tendsto_liZeroPartialSum n hsum

example {ι : Type*} [DecidableEq ι]
    (root : ι → ℂ) (cutoff : ℕ → Finset ι) (n : ℕ)
    (d : ℕ → ℝ) (hd : Summable d)
    (hbound : ∀ N, dist (liZeroPartialSum root (cutoff N) n)
      (liZeroPartialSum root (cutoff (N + 1)) n) ≤ d N) :
    CauchySeq (fun N ↦ liZeroPartialSum root (cutoff N) n) :=
  cauchySeq_liZeroPartialSum_of_shell_majorant
    root cutoff n d hd hbound

example {ι : Type*} [DecidableEq ι]
    (root : ι → ℂ) (cutoff : ℕ → Finset ι) (C : ℕ → ℝ)
    (hbound : ThreeHalvesLiShellBound root cutoff C) :
    HeightSymmetricLiCauchy root cutoff :=
  heightSymmetricLiCauchy_of_threeHalvesShellBound
    root cutoff C hbound

example {ι : Type*} [DecidableEq ι]
    (root : ι → ℂ) (cutoff : ℕ → Finset ι)
    (hcancel : QuadraticPairedLiShellCancellation root cutoff) :
    HeightSymmetricLiCauchy root cutoff :=
  heightSymmetricLiCauchy_of_quadraticPairedCancellation
    root cutoff hcancel

example (w : ℂ) (hw : ‖w‖ ≤ 1) :
    ‖genusOnePrimaryFactor w - 1‖ ≤ 3 * ‖w‖ ^ 2 :=
  norm_genusOnePrimaryFactor_sub_one_le w hw

example (zeros : FiniteZeroMultiset)
    (hsym : zeros.ReflectionSymmetric) (s : ℂ) :
    (∏ i, genusOnePrimaryFactor (s / zeros.root i)) =
      zeros.genusOneCenteringConstant *
        Complex.exp (s * zeros.genusOneInverseSum) *
          zeros.canonicalProductAtOne s :=
  zeros.genusOneProduct_eq_centered hsym s

example (zeros : FiniteZeroMultiset)
    (hsym : zeros.ReflectionSymmetric) (a b : ℂ)
    (hbalance : zeros.HadamardBalanced b) (k : ℕ) :
    iteratedDeriv k
      (logDeriv (fun z ↦
        zeros.genusOneHadamardApproximation a b ((1 - z)⁻¹))) 0 /
        (k.factorial : ℂ) =
      zeros.liSum (k + 1) :=
  zeros.logDeriv_genusOneHadamardMobius_coefficient
    hsym a b hbalance k

example {ι : Type*} (root : ι → ℂ)
    (hroot0 : ∀ i, root i ≠ 0)
    (hlarge : ∀ i, 1 ≤ ‖root i‖)
    (hsum : BombieriLagariasSummability root)
    (hescape : Tendsto (fun i ↦ ‖root i‖) cofinite atTop) :
    HasProdLocallyUniformlyOn (genusOneCanonicalFactor root)
      (fun s ↦ ∏' i, genusOneCanonicalFactor root i s) Set.univ :=
  hasProdLocallyUniformlyOn_genusOneCanonicalFactor_of_bombieriLagarias
    root hroot0 hlarge hsum hescape

example {ι : Type*} [DecidableEq ι] {root : ι → ℂ} {xi : ℂ → ℂ}
    (hroot0 : ∀ i, root i ≠ 0)
    (hinv : Summable (fun i ↦ ‖(root i)⁻¹‖ ^ 2))
    (hsmall : ∀ K : Set ℂ, IsCompact K →
      ∀ᶠ i in cofinite, ∀ s ∈ K, ‖s / root i‖ ≤ 1)
    (fiber : ℂ → Finset ι)
    (hfiber : ∀ z i, i ∈ fiber z ↔ root i = z)
    (hxi : AnalyticOnNhd ℂ xi Set.univ)
    (hxiOrder : ∀ z, analyticOrderAt xi z = (fiber z).card) :
    MeromorphicOn.divisor
      (fun s ↦ ∏' i, genusOneCanonicalFactor root i s) Set.univ =
        MeromorphicOn.divisor xi Set.univ :=
  genusOneCanonicalProduct_divisor_eq
    hroot0 hinv hsmall fiber hfiber hxi hxiOrder

example
    (hinv : Summable
      (fun i : RiemannXiZeroIndex ↦
        ‖(riemannXiZeroRoot i)⁻¹‖ ^ 2)) :
    MeromorphicOn.divisor
      (fun s ↦ ∏' i : RiemannXiZeroIndex,
        genusOneCanonicalFactor riemannXiZeroRoot i s) Set.univ =
      MeromorphicOn.divisor riemannXiLi Set.univ :=
  riemannXiGenusOneCanonicalProduct_divisor_eq_of_summable hinv

example {ι : Type*} [DecidableEq ι] (root : ι → ℂ)
    (low : Finset ι)
    (shell : ℕ → Finset {i : ι // i ∉ low})
    (hpartition : ∀ i : {i : ι // i ∉ low},
      ∃! N, i ∈ shell N)
    (hlower : ∀ N i, i ∈ shell N →
      ((N + 1 : ℕ) : ℝ) ≤ ‖root i‖)
    (A : ℝ) (hA : 0 ≤ A)
    (hcount : ∀ N, ((shell N).card : ℝ) ≤
      A * Real.log (N + 2)) :
    Summable (fun i ↦ ‖(root i)⁻¹‖ ^ 2) :=
  summable_norm_inv_sq_of_finite_low_and_logarithmic_shell_count
    root low shell hpartition hlower A hA hcount

example {ι : Type*} [DecidableEq ι] (root : ι → ℂ)
    (shell : ℕ → Finset ι)
    (hpartition : ∀ i, ∃! N, i ∈ shell N)
    (hlower : ∀ N i, i ∈ shell N →
      (2 : ℝ) ^ N ≤ ‖root i‖)
    (A : ℝ)
    (hcount : ∀ N, ((shell N).card : ℝ) ≤
      A * (2 : ℝ) ^ N * (N + 1 : ℝ)) :
    Summable (fun i ↦ ‖(root i)⁻¹‖ ^ 2) :=
  summable_norm_inv_sq_of_dyadic_linear_log_shell_count
    root shell hpartition hlower A hcount

example :
    ∃ C R : ℝ, 0 ≤ C ∧ 2 ≤ R ∧ ∀ r : ℝ, R ≤ r →
      Function.locallyFinsuppWithin.logCounting
          riemannXiZeroDivisor r ≤
        C * r * Real.log (r + 2) :=
  exists_riemannXiZeroDivisor_logCounting_le_orderOne

example (z : ℂ) :
    riemannXiZeroDivisor z =
      (riemannXiZeroMultiplicity z : ℤ) :=
  riemannXiZeroDivisor_apply z

example :
    ∃ A R : ℝ, 0 ≤ A ∧ 1 ≤ R ∧ ∀ r : ℝ, R ≤ r →
      ((riemannXiZeroIndexWindow r).card : ℝ) ≤
        A * r * Real.log (2 * r + 2) :=
  exists_riemannXiZeroIndexWindow_card_le_orderOne

example :
    Summable (fun i : RiemannXiZeroIndex ↦
      ‖(riemannXiZeroRoot i)⁻¹‖ ^ 2) :=
  summable_norm_inv_sq_riemannXiZeroRoot

example :
    MeromorphicOn.divisor
      (fun s ↦ ∏' i : RiemannXiZeroIndex,
        genusOneCanonicalFactor riemannXiZeroRoot i s) Set.univ =
      MeromorphicOn.divisor riemannXiLi Set.univ :=
  riemannXiGenusOneCanonicalProduct_divisor_eq_unconditional

example :
    HasProdLocallyUniformlyOn
      (genusOneCanonicalFactor riemannXiZeroRoot)
      (fun s ↦ ∏' i : RiemannXiZeroIndex,
        genusOneCanonicalFactor riemannXiZeroRoot i s) Set.univ :=
  riemannXiGenusOneCanonicalProduct_hasProdLocallyUniformly

example
    (low : Finset RiemannXiZeroIndex)
    (shell : ℕ → Finset
      {i : RiemannXiZeroIndex // i ∉ low})
    (hpartition : ∀ i : {i : RiemannXiZeroIndex // i ∉ low},
      ∃! N, i ∈ shell N)
    (hlower : ∀ N i, i ∈ shell N →
      (2 : ℝ) ^ N ≤ ‖riemannXiZeroRoot i‖)
    (A : ℝ)
    (hcount : ∀ N, ((shell N).card : ℝ) ≤
      A * (2 : ℝ) ^ N * (N + 1 : ℝ)) :
    MeromorphicOn.divisor
      (fun s ↦ ∏' i : RiemannXiZeroIndex,
        genusOneCanonicalFactor riemannXiZeroRoot i s) Set.univ =
      MeromorphicOn.divisor riemannXiLi Set.univ :=
  riemannXiGenusOneCanonicalProduct_divisor_eq_of_dyadicShellCount
    low shell hpartition hlower A hcount

example {ι : Type*} [DecidableEq ι]
    {root : ι → ℂ} {cutoff : ℕ → Finset ι} {height : ℕ → ℝ}
    (hexhaust : SymmetricHeightExhaustion root cutoff height)
    (hprod : HasProdLocallyUniformlyOn (genusOneCanonicalFactor root)
      (fun s ↦ ∏' i, genusOneCanonicalFactor root i s) Set.univ) :
    TendstoLocallyUniformlyOn
      (fun N s ↦ ∏ i ∈ cutoff N, genusOneCanonicalFactor root i s)
      (fun s ↦ ∏' i, genusOneCanonicalFactor root i s)
      atTop Set.univ :=
  hexhaust.tendstoLocallyUniformlyOn_genusOneProduct hprod

example {ι : Type*} [DecidableEq ι]
    {root : ι → ℂ} {cutoff : ℕ → Finset ι} {height : ℕ → ℝ}
    {xi : ℂ → ℂ}
    (hexhaust : SymmetricHeightExhaustion root cutoff height)
    (hprod : HasProdLocallyUniformlyOn (genusOneCanonicalFactor root)
      (fun s ↦ ∏' i, genusOneCanonicalFactor root i s) Set.univ)
    (hrep : GenusOneHadamardRepresentation root xi) :
    ∃ a b : ℂ, TendstoLocallyUniformlyOn
      (fun N s ↦ Complex.exp (a + b * s) *
        ∏ i ∈ cutoff N, genusOneCanonicalFactor root i s)
      xi atTop Set.univ :=
  hrep.exists_tendsto_symmetricApproximants hexhaust hprod

example {ι : Type*} {root : ι → ℂ} {xi : ℂ → ℂ} {a b : ℂ}
    (hrep : ∀ s, xi s =
      Complex.exp (a + b * s) *
        ∏' i, genusOneCanonicalFactor root i s)
    (hxi0 : xi 0 = 1) :
    Complex.exp a = 1 :=
  genusOneHadamard_normalization_zero hrep hxi0

example {ι : Type*} {root : ι → ℂ} {xi : ℂ → ℂ} {a b : ℂ}
    (hrep : ∀ s, xi s =
      Complex.exp (a + b * s) *
        ∏' i, genusOneCanonicalFactor root i s)
    (hfe : ∀ s, xi (1 - s) = xi s) (s : ℂ) :
    Complex.exp (a + b * (1 - s)) *
        ∏' i, genusOneCanonicalFactor root i (1 - s) =
      Complex.exp (a + b * s) *
        ∏' i, genusOneCanonicalFactor root i s :=
  genusOneHadamard_functionalEquation_constraint hrep hfe s

example {P xi : ℂ → ℂ} {a b a' b' : ℂ}
    (hPcont : ContinuousAt P 0) (hP0 : P 0 ≠ 0)
    (hrep : ∀ s, xi s = Complex.exp (a + b * s) * P s)
    (hrep' : ∀ s, xi s = Complex.exp (a' + b' * s) * P s) :
    Complex.exp a = Complex.exp a' ∧ b = b' :=
  hadamardAffineFactor_unique hPcont hP0 hrep hrep'

example {g : ℂ → ℂ} (hg : Continuous g) (hne : ∀ z, g z ≠ 0) :
    ∃ f : ℂ → ℂ, Continuous f ∧
      ∀ z, Complex.exp (f z) = g z :=
  exists_continuous_log_of_continuous_ne_zero hg hne

example {g : ℂ → ℂ} (hg : AnalyticOnNhd ℂ g Set.univ)
    (hne : ∀ z, g z ≠ 0) :
    ∃ f : ℂ → ℂ, AnalyticOnNhd ℂ f Set.univ ∧
      ∀ z, Complex.exp (f z) = g z :=
  exists_analytic_log_of_analytic_ne_zero hg hne

example {g f : ℂ → ℂ}
    (hf : AnalyticOnNhd ℂ f Set.univ)
    (hexp : ∀ z, Complex.exp (f z) = g z) (z : ℂ) :
    deriv f z = logDeriv g z :=
  deriv_analytic_log_eq_logDeriv hf hexp z

example (h : ℂ → ℂ) (hh : AnalyticOnNhd ℂ h Set.univ)
    (hsecond : ∀ z, iteratedDeriv 2 h z = 0) :
    ∃ a b : ℂ, h = fun z ↦ a + b * z :=
  eq_affine_of_iteratedDeriv_two_eq_zero h hh hsecond

example (h : ℂ → ℂ) (hh : AnalyticOnNhd ℂ h Set.univ)
    (hgrowth : SubquadraticNormGrowth h) :
    ∀ z, iteratedDeriv 2 h z = 0 :=
  iteratedDeriv_two_eq_zero_of_subquadraticNormGrowth h hh hgrowth

example (h : ℂ → ℂ) (hh : AnalyticOnNhd ℂ h Set.univ)
    (hgrowth : SubquadraticRealPartGrowth h) :
    ∀ z, iteratedDeriv 2 h z = 0 :=
  iteratedDeriv_two_eq_zero_of_subquadraticRealPartGrowth h hh hgrowth

example {g h : ℂ → ℂ}
    (hh : AnalyticOnNhd ℂ h Set.univ)
    (hexp : ∀ z, Complex.exp (h z) = g z)
    (hgrowth : SubquadraticLogNormGrowth g) :
    ∃ a b : ℂ, ∀ z, g z = Complex.exp (a + b * z) :=
  eq_exp_affine_of_analytic_log_subquadraticLogNormGrowth
    hh hexp hgrowth

example {g : ℂ → ℂ}
    (hgA : AnalyticOnNhd ℂ g Set.univ)
    (hg0 : ∀ z, g z ≠ 0)
    (hboundary : SubquadraticBoundaryLogNormGrowth g) :
    SubquadraticLogNormGrowth g :=
  subquadraticLogNormGrowth_of_boundary hgA hg0 hboundary

example {ι : Type*} (s : Finset ι) (center : ι → ℂ)
    (radius : ι → ℝ)
    (hradius : ∀ i ∈ s, 0 ≤ radius i)
    {a b : ℝ}
    (hwidth : 2 * ∑ i ∈ s, radius i < b - a) :
    ∃ R ∈ Set.Icc a b, ∀ i ∈ s,
      Disjoint (Metric.sphere (0 : ℂ) R)
        (Metric.closedBall (center i) (radius i)) :=
  exists_circle_avoiding_exceptionalDisks
    s center radius hradius hwidth

example {ι : Type*} (s : Finset ι) (root : ι → ℂ)
    (hroot0 : ∀ i, root i ≠ 0)
    (hs : s.Nonempty) {H a b : ℝ} (hH : 0 < H)
    (hwidth : H ≤ b - a) :
    ∃ R ∈ Set.Icc a b, ∀ z ∈ Metric.sphere (0 : ℂ) R,
      (∏ i ∈ s,
        ((H / (3 * s.card)) / ‖root i‖) *
          Real.exp (-‖z‖ * ‖(root i)⁻¹‖)) ≤
        ‖∏ i ∈ s, genusOneCanonicalFactor root i z‖ :=
  exists_circle_norm_finsetProd_genusOneCanonicalFactor_lower
    s root hroot0 hs hH hwidth

example {ι : Type*} (s : Finset ι) (root : ι → ℂ) (z : ℂ)
    (hsmall : ∀ i ∈ s, ‖z / root i‖ ≤ 1) :
    ‖(∏ i ∈ s, genusOneCanonicalFactor root i z) - 1‖ ≤
      Real.exp
        (3 * ‖z‖ ^ 2 *
          ∑ i ∈ s, ‖(root i)⁻¹‖ ^ 2) - 1 :=
  norm_finsetProd_genusOneCanonicalFactor_sub_one_le
    s root z hsmall

example (T : ℝ)
    (hwindow : (riemannXiZeroIndexWindow T).Nonempty)
    {H a b : ℝ} (hH : 0 < H) (hwidth : H ≤ b - a) :
    ∃ R ∈ Set.Icc a b, ∀ z ∈ Metric.sphere (0 : ℂ) R,
      (∏ i ∈ riemannXiZeroIndexWindow T,
        ((H / (3 * (riemannXiZeroIndexWindow T).card)) /
            ‖riemannXiZeroRoot i‖) *
          Real.exp (-‖z‖ * ‖(riemannXiZeroRoot i)⁻¹‖)) ≤
        ‖∏ i ∈ riemannXiZeroIndexWindow T,
          genusOneCanonicalFactor riemannXiZeroRoot i z‖ :=
  exists_circle_norm_riemannXiWindowProduct_lower
    T hwindow hH hwidth

example (T : ℝ) :
    ∑ i ∈ riemannXiZeroIndexWindow T,
        Real.log ‖riemannXiZeroRoot i‖ ≤
      (riemannXiZeroIndexWindow T).card * Real.log T :=
  sum_log_norm_riemannXiZeroRoot_window_le T

example :
    Tendsto (fun R : ℝ ↦
      ∑' i : {i : RiemannXiZeroIndex //
        i ∉ riemannXiZeroIndexWindow R},
          ‖(riemannXiZeroRoot i)⁻¹‖ ^ 2)
      atTop (𝓝 0) :=
  riemannXiZeroRoot_inv_sq_tail_tendsto_zero

example (N : ℕ) :
    (∑' m : ℕ,
      (N + m + 1 : ℝ) / (2 : ℝ) ^ (N + m)) ≤
      4 * (N + 1 : ℝ) / (2 : ℝ) ^ N :=
  tsum_shifted_dyadic_linear_le N

example :
    ∃ A R : ℝ, 0 ≤ A ∧ 1 ≤ R ∧ ∀ N : ℕ,
      (∑' m : ℕ,
        ∑ i ∈ riemannXiDyadicZeroShell
          (riemannXiZeroIndexWindow R) (N + m),
          ‖(riemannXiZeroRoot i)⁻¹‖ ^ 2) ≤
        A * (N + 1 : ℝ) / (2 : ℝ) ^ N :=
  exists_riemannXiDyadicZeroShell_inv_sq_tail_bound

example (N : ℕ) :
    {i : RiemannXiZeroIndex |
      i ∉ riemannXiZeroIndexWindow ((2 : ℝ) ^ N)} =
      ⋃ m : ℕ,
        (↑(riemannXiDyadicZeroTailShell N m) :
          Set RiemannXiZeroIndex) :=
  riemannXiZeroIndexWindow_compl_eq_iUnion_tailShell N

example (N : ℕ) (f : RiemannXiZeroIndex → ℝ)
    (hf : Summable f) :
    (∑' i : {i : RiemannXiZeroIndex //
        i ∉ riemannXiZeroIndexWindow ((2 : ℝ) ^ N)}, f i) =
      ∑' m : ℕ,
        ∑ i ∈ riemannXiDyadicZeroTailShell N m, f i :=
  tsum_riemannXiZeroIndexWindow_compl_eq_tailShells N f hf

example :
    Tendsto (fun j : ℕ ↦
      ((2 : ℝ) ^ j) ^ 2 *
        ((4 * j + 1 : ℝ) / (2 : ℝ) ^ (4 * j)))
      atTop (𝓝 0) :=
  tendsto_radius_sq_mul_four_mul_truncation_tail

example :
    ∃ A R : ℝ, 0 ≤ A ∧ 1 ≤ R ∧ ∀ N : ℕ,
      R ≤ (2 : ℝ) ^ N →
      (∑' i : {i : RiemannXiZeroIndex //
        i ∉ riemannXiZeroIndexWindow ((2 : ℝ) ^ N)},
          ‖(riemannXiZeroRoot i)⁻¹‖ ^ 2) ≤
        A * (N + 1 : ℝ) / (2 : ℝ) ^ N :=
  exists_riemannXiZeroRoot_inv_sq_radial_tail_bound

example :
    Tendsto (fun j : ℕ ↦
      (j + 1 : ℝ) ^ 2 / (2 : ℝ) ^ j) atTop (𝓝 0) :=
  tendsto_add_one_sq_div_two_pow

example :
    ∃ B R : ℝ, 0 ≤ B ∧ 1 ≤ R ∧ ∀ j : ℕ,
      R ≤ (2 : ℝ) ^ (j + 2) →
      ((riemannXiZeroIndexWindow
        ((2 : ℝ) ^ (j + 2))).card : ℝ) ≤
        B * (2 : ℝ) ^ j * (j + 1 : ℝ) :=
  exists_riemannXiZeroIndexWindow_card_le_dyadic

example (K : ℝ) :
    Tendsto (fun j : ℕ ↦
      Real.sqrt (K * ((j + 1 : ℝ) / (2 : ℝ) ^ j)))
      atTop (𝓝 0) :=
  tendsto_sqrt_const_mul_add_one_div_two_pow K

example (Ksq Ksqrt Klin ε : ℝ) (hε : 0 < ε) :
    ∀ᶠ j : ℕ in atTop,
      DyadicCartanLossMajorant Ksq Ksqrt Klin j < ε :=
  eventually_dyadicCartanLossMajorant_lt
    Ksq Ksqrt Klin ε hε

example {g : ℂ → ℂ}
    (h : DyadicSubquadraticBoundaryLogNormGrowth g) :
    SubquadraticBoundaryLogNormGrowth g :=
  subquadraticBoundaryLogNormGrowth_of_dyadic h

example {g : ℂ → ℂ} (hgA : AnalyticOnNhd ℂ g Set.univ)
    (hg0 : ∀ z, g z ≠ 0)
    (hboundary : DyadicSubquadraticBoundaryLogNormGrowth g) :
    SubquadraticLogNormGrowth g :=
  subquadraticLogNormGrowth_of_dyadicBoundary hgA hg0 hboundary

example [Nonempty RiemannXiZeroIndex] :
    DyadicSubquadraticBoundaryLogNormGrowth
      (cancelledEntireQuotient riemannXiLi
        (fun s ↦ ∏' i : RiemannXiZeroIndex,
          genusOneCanonicalFactor riemannXiZeroRoot i s)) :=
  riemannXiCancelledQuotient_dyadicBoundary

example :
    GenusOneHadamardRepresentation riemannXiZeroRoot riemannXiLi :=
  riemannXiGenusOneHadamardRepresentation

example : Nonempty RiemannXiZeroIndex :=
  riemannXiZeroIndex_nonempty

example : riemannXiLi 2 = (Real.pi : ℂ) / 3 :=
  riemannXiLi_two

example :
    ∃ a b : ℂ,
      (∀ z, riemannXiLi z = Complex.exp (a + b * z) *
        ∏' i : RiemannXiZeroIndex,
          genusOneCanonicalFactor riemannXiZeroRoot i z) ∧
      Complex.exp a = 1 ∧
      ∀ z,
        Complex.exp (a + b * (1 - z)) *
            (∏' i : RiemannXiZeroIndex,
              genusOneCanonicalFactor riemannXiZeroRoot i (1 - z)) =
          Complex.exp (a + b * z) *
            ∏' i : RiemannXiZeroIndex,
              genusOneCanonicalFactor riemannXiZeroRoot i z :=
  riemannXiGenusOneHadamardRepresentation_normalized

example (k : ℕ) :
    liDerivativeCoefficient riemannXiLi (k + 1) =
      (∑ i ∈ Finset.range (k + 2),
        ((k + 1).choose i : ℂ) *
          (k.descFactorial i : ℂ) *
          iteratedDeriv (k + 1 - i)
            (fun s ↦ Complex.log (riemannXiLi s)) 1) /
        (k.factorial : ℂ) :=
  liDerivativeCoefficient_eq_leibnizSum
    riemannXiLi (k + 1) (by omega)
      analyticAt_log_riemannXiLi_one.contDiffAt

example :
    LiChangeOfVariablesCoefficientIdentity riemannXiLi ↔
      MobiusFaaDiBrunoCoefficientIdentity riemannXiLi :=
  riemannXiLi_changeOfVariablesCoefficientIdentity_iff_faaDiBruno

example (k : ℕ) (z : ℂ) :
    iteratedDeriv k (fun w : ℂ ↦ (1 - w)⁻¹) z =
      (k.factorial : ℂ) * (1 - z) ^ (-(k + 1 : ℤ)) :=
  iteratedDeriv_one_sub_inv k z

example (k : ℕ) :
    iteratedDeriv k (fun z : ℂ ↦ (1 - z)⁻¹) 0 =
      (k.factorial : ℂ) :=
  iteratedDeriv_one_sub_inv_zero k

example :
    IsMobiusFaaDiBrunoCoefficientRecurrence
      mobiusFaaDiBrunoCoefficient :=
  mobiusFaaDiBrunoCoefficient_recurrence

example {c : ℕ → ℕ → ℕ}
    (hc : IsMobiusFaaDiBrunoCoefficientRecurrence c) :
    c = mobiusFaaDiBrunoCoefficient :=
  hc.eq mobiusFaaDiBrunoCoefficient_recurrence

example (k i : ℕ) (hi : i ≤ k + 1) :
    mobiusFaaDiBrunoCoefficient (k + 1) i =
      mobiusFaaDiBrunoCoefficient k i +
        (if i = 0 then 0 else
          (2 * k + 3 - i) * mobiusFaaDiBrunoCoefficient k (i - 1)) :=
  mobiusFaaDiBrunoCoefficient_succ k i hi

example {L : ℂ → ℂ} {x : ℂ} (hL : AnalyticAt ℂ L x) (m : ℕ) :
    DifferentiableAt ℂ (iteratedDeriv m L) x :=
  hL.differentiableAt_iteratedDeriv m

example {L : ℂ → ℂ} {z : ℂ} (hz : z ≠ 1)
    (hL : AnalyticAt ℂ L ((1 - z)⁻¹)) (p m : ℕ) :
    HasDerivAt
      (fun w ↦ (1 - w)⁻¹ ^ p *
        iteratedDeriv m L ((1 - w)⁻¹))
      (p * ((1 - z)⁻¹ ^ (p + 1) *
          iteratedDeriv m L ((1 - z)⁻¹)) +
        (1 - z)⁻¹ ^ (p + 2) *
          iteratedDeriv (m + 1) L ((1 - z)⁻¹)) z :=
  hasDerivAt_mobiusFaaDiBrunoSummand hz hL p m

example (k : ℕ) (B : ℕ → ℂ) :
    (∑ i ∈ Finset.range (k + 2),
      (mobiusFaaDiBrunoCoefficient k i : ℂ) *
        (((2 * k + 2 - i : ℕ) : ℂ) * B (i + 1) + B i)) =
      ∑ j ∈ Finset.range (k + 3),
        (mobiusFaaDiBrunoCoefficient (k + 1) j : ℂ) * B j :=
  mobiusFaaDiBrunoCoefficientSum_reindex k B

example {L : ℂ → ℂ} (hL : AnalyticAt ℂ L 1) (k : ℕ) :
    iteratedDeriv k
      (fun w ↦ liMobiusArgument w ^ 2 *
        iteratedDeriv 1 L (liMobiusArgument w)) 0 =
      ∑ i ∈ Finset.range (k + 2),
        (mobiusFaaDiBrunoCoefficient k i : ℂ) *
          iteratedDeriv (k + 1 - i) L 1 :=
  iteratedDeriv_mobiusComposition_zero hL k

example : MobiusFaaDiBrunoCoefficientIdentity riemannXiLi :=
  riemannXiLi_mobiusFaaDiBrunoCoefficientIdentity

example : LiChangeOfVariablesCoefficientIdentity riemannXiLi :=
  riemannXiLi_changeOfVariablesCoefficientIdentity

example :
    Tendsto (fun N : ℕ ↦ riemannXiZeroIndexWindow (N : ℝ))
      atTop atTop :=
  riemannXiZeroIndexWindow_nat_tendsto_atTop

example (i : RiemannXiZeroIndex) :
    riemannXiZeroRoot (riemannXiZeroReflection i) =
      1 - riemannXiZeroRoot i :=
  riemannXiZeroRoot_reflection i

example (i : RiemannXiZeroIndex) :
    (riemannXiZeroRoot i)⁻¹ +
        (riemannXiZeroRoot (riemannXiZeroReflection i))⁻¹ =
      (riemannXiZeroRoot i *
        (1 - riemannXiZeroRoot i))⁻¹ :=
  riemannXiZeroRoot_inv_add_reflection i

example (i : RiemannXiZeroIndex) :
    0 < (riemannXiZeroRoot i).re ∧
      (riemannXiZeroRoot i).re < 1 :=
  riemannXiZeroRoot_mem_openCriticalStrip i

example (T : ℝ) (i : RiemannXiZeroIndex) :
    riemannXiZeroReflection i ∈ riemannXiZeroHeightWindow T ↔
      i ∈ riemannXiZeroHeightWindow T :=
  mem_riemannXiZeroHeightWindow_reflection T i

example (T : ℝ) :
    2 * (∑ i ∈ riemannXiZeroHeightWindow T,
      (riemannXiZeroRoot i)⁻¹) =
      ∑ i ∈ riemannXiZeroHeightWindow T,
        (riemannXiZeroRoot i *
          (1 - riemannXiZeroRoot i))⁻¹ :=
  two_mul_sum_inv_riemannXiZeroHeightWindow T

example :
    Tendsto
      (fun N : ℕ ↦ ∑ i ∈ riemannXiZeroHeightWindow (N : ℝ),
        (riemannXiZeroRoot i)⁻¹)
      atTop (𝓝 ((1 / 2 : ℂ) * ∑' i : RiemannXiZeroIndex,
        (riemannXiZeroRoot i *
          (1 - riemannXiZeroRoot i))⁻¹)) :=
  riemannXiInverseRootHeightSums_tendsto

example :
    ∃ a b : ℂ,
      (∀ z, riemannXiLi z = Complex.exp (a + b * z) *
        riemannXiGenusOneCanonicalProduct z) ∧
      Complex.exp a = 1 ∧
      (∑' i : RiemannXiZeroIndex,
        (riemannXiZeroRoot i *
          (1 - riemannXiZeroRoot i))⁻¹) = -2 * b :=
  exists_riemannXiHadamardSlope_eq_pairedInverseRootSum

example :
    ∃ a b : ℂ,
      (∀ z, riemannXiLi z = Complex.exp (a + b * z) *
        riemannXiGenusOneCanonicalProduct z) ∧
      Complex.exp a = 1 ∧
      Tendsto
        (fun N : ℕ ↦ ∑ i ∈ riemannXiZeroHeightWindow (N : ℝ),
          (riemannXiZeroRoot i)⁻¹)
        atTop (𝓝 (-b)) :=
  exists_riemannXiInverseRootHeightSums_tendsto_neg_slope

example (T : ℝ) :
    logDeriv
      (fun s ↦ ∏ i ∈ riemannXiZeroHeightWindow T,
        genusOneCanonicalFactor riemannXiZeroRoot i s) 1 =
      ∑ i ∈ riemannXiZeroHeightWindow T,
        (riemannXiZeroRoot i *
          (1 - riemannXiZeroRoot i))⁻¹ :=
  logDeriv_riemannXiHeightGenusOneProduct_one T

example :
    ∃ a b : ℂ, TendstoLocallyUniformlyOn
      (fun (N : ℕ) s ↦ Complex.exp (a + b * s) *
        ∏ i ∈ riemannXiZeroIndexWindow (N : ℝ),
          genusOneCanonicalFactor riemannXiZeroRoot i s)
      riemannXiLi atTop Set.univ :=
  exists_riemannXiRadialHadamardApproximants_tendstoLocallyUniformly

example {n B H u : ℝ}
    (hn : 0 < n) (hB : 0 ≤ B) (hH : 0 < H)
    (hu : 0 ≤ u) (hnB : n ≤ B * H * u) :
    n * Real.log (3 * n / H) / H ^ 2 ≤
      3 * B ^ 2 * u ^ 2 / H :=
  card_mul_log_three_mul_div_sq_le hn hB hH hu hnB

example {R : ℝ} {z : ℂ} (hz : ‖z‖ ≤ R) :
    ‖(∏' i : {i : RiemannXiZeroIndex //
        i ∉ riemannXiZeroIndexWindow R},
        genusOneCanonicalFactor riemannXiZeroRoot i z) - 1‖ ≤
      Real.exp (3 * R ^ 2 *
        ∑' i : {i : RiemannXiZeroIndex //
          i ∉ riemannXiZeroIndexWindow R},
          ‖(riemannXiZeroRoot i)⁻¹‖ ^ 2) - 1 :=
  norm_riemannXiGenusOneCanonicalProduct_tail_sub_one_le_uniform hz

example {w : ℂ} (hw : ‖w‖ ≤ 1 / 2) :
    Real.exp (-‖w‖ ^ 2) ≤ ‖genusOnePrimaryFactor w‖ :=
  norm_genusOnePrimaryFactor_lower_of_norm_le_half hw

example {T : ℝ} {z : ℂ} (hz : ‖z‖ ≤ T / 2) :
    Real.exp (-‖z‖ ^ 2 *
      ∑' i : {i : RiemannXiZeroIndex //
        i ∉ riemannXiZeroIndexWindow T},
        ‖(riemannXiZeroRoot i)⁻¹‖ ^ 2) ≤
      ‖∏' i : {i : RiemannXiZeroIndex //
        i ∉ riemannXiZeroIndexWindow T},
        genusOneCanonicalFactor riemannXiZeroRoot i z‖ :=
  norm_riemannXiGenusOneCanonicalProduct_tail_lower hz

example (T : ℝ) (z : ℂ) :
    (∏ i ∈ riemannXiZeroIndexWindow T,
        genusOneCanonicalFactor riemannXiZeroRoot i z) *
      (∏' i : {i : RiemannXiZeroIndex //
          i ∉ riemannXiZeroIndexWindow T},
        genusOneCanonicalFactor riemannXiZeroRoot i z) =
      ∏' i : RiemannXiZeroIndex,
        genusOneCanonicalFactor riemannXiZeroRoot i z :=
  riemannXiGenusOneCanonicalProduct_split_window T z

example {T L : ℝ} {z : ℂ}
    (hhead : L ≤ ‖∏ i ∈ riemannXiZeroIndexWindow T,
      genusOneCanonicalFactor riemannXiZeroRoot i z‖)
    (hz : ‖z‖ ≤ T / 2) :
    L * Real.exp (-‖z‖ ^ 2 *
      ∑' i : {i : RiemannXiZeroIndex //
        i ∉ riemannXiZeroIndexWindow T},
        ‖(riemannXiZeroRoot i)⁻¹‖ ^ 2) ≤
      ‖∏' i : RiemannXiZeroIndex,
        genusOneCanonicalFactor riemannXiZeroRoot i z‖ :=
  norm_riemannXiGenusOneCanonicalProduct_lower_of_head hhead hz

example (j : ℕ)
    (hwindow :
      (riemannXiZeroIndexWindow ((2 : ℝ) ^ j)).Nonempty) :
    ∃ R ∈ Set.Icc ((2 : ℝ) ^ j) ((2 : ℝ) ^ (j + 1)),
      ∀ z ∈ Metric.sphere (0 : ℂ) R,
        (∏ i ∈ riemannXiZeroIndexWindow ((2 : ℝ) ^ j),
          ((((2 : ℝ) ^ j) /
              (3 * (riemannXiZeroIndexWindow
                ((2 : ℝ) ^ j)).card)) /
              ‖riemannXiZeroRoot i‖) *
            Real.exp (-‖z‖ *
              ‖(riemannXiZeroRoot i)⁻¹‖)) ≤
          ‖∏ i ∈ riemannXiZeroIndexWindow ((2 : ℝ) ^ j),
            genusOneCanonicalFactor riemannXiZeroRoot i z‖ :=
  exists_circle_norm_riemannXiCompatibleWindowProduct_lower j hwindow

example (T H : ℝ) (hH : 0 < H)
    (hwindow : (riemannXiZeroIndexWindow T).Nonempty)
    (z : ℂ) :
    (∏ i ∈ riemannXiZeroIndexWindow T,
      ((H / (3 * (riemannXiZeroIndexWindow T).card)) /
          ‖riemannXiZeroRoot i‖) *
        Real.exp (-‖z‖ *
          ‖(riemannXiZeroRoot i)⁻¹‖)) =
      Real.exp (-riemannXiFiniteHeadLogLoss T H z) :=
  riemannXiCartanHeadLower_eq_exp_neg_logLoss
    T H hH hwindow z

example (T H : ℝ) (z : ℂ) :
    riemannXiFiniteHeadLogLoss T H z ≤
      ((riemannXiZeroIndexWindow T).card : ℝ) * Real.log T +
        ‖z‖ * Real.sqrt
          (((riemannXiZeroIndexWindow T).card : ℝ) *
            ∑' i : RiemannXiZeroIndex,
              ‖(riemannXiZeroRoot i)⁻¹‖ ^ 2) -
        ((riemannXiZeroIndexWindow T).card : ℝ) *
          Real.log (H /
            (3 * (riemannXiZeroIndexWindow T).card)) :=
  riemannXiFiniteHeadLogLoss_le T H z

example {xi P : ℂ → ℂ} {z : ℂ} {U L : ℝ}
    (hxiM : MeromorphicOn xi Set.univ)
    (hPM : MeromorphicOn P Set.univ)
    (hxi : AnalyticAt ℂ xi z) (hP : AnalyticAt ℂ P z)
    (hxiUpper : ‖xi z‖ ≤ Real.exp U)
    (hPLower : Real.exp (-L) ≤ ‖P z‖) :
    ‖cancelledEntireQuotient xi P z‖ ≤ Real.exp (U + L) :=
  norm_cancelledEntireQuotient_le_exp_add
    hxiM hPM hxi hP hxiUpper hPLower

example {g : ℂ → ℂ}
    (hcartan : CartanExceptionalDiskLogBound g) :
    SubquadraticBoundaryLogNormGrowth g :=
  subquadraticBoundaryLogNormGrowth_of_cartanExceptionalDisks hcartan

example
    (hboundary : SubquadraticBoundaryLogNormGrowth
      (cancelledEntireQuotient riemannXiLi
        (fun s ↦ ∏' i : RiemannXiZeroIndex,
          genusOneCanonicalFactor riemannXiZeroRoot i s))) :
    GenusOneHadamardRepresentation riemannXiZeroRoot riemannXiLi :=
  riemannXiGenusOneHadamardRepresentation_of_cartanBoundary hboundary

example
    (hcartan : CartanExceptionalDiskLogBound
      (cancelledEntireQuotient riemannXiLi
        (fun s ↦ ∏' i : RiemannXiZeroIndex,
          genusOneCanonicalFactor riemannXiZeroRoot i s))) :
    GenusOneHadamardRepresentation riemannXiZeroRoot riemannXiLi :=
  riemannXiGenusOneHadamardRepresentation_of_cartanExceptionalDisks
    hcartan

example {xi P : ℂ → ℂ}
    (hxi : MeromorphicOn xi Set.univ)
    (hP : MeromorphicOn P Set.univ)
    (hxiFinite : ∀ z, meromorphicOrderAt xi z ≠ ⊤)
    (hPFinite : ∀ z, meromorphicOrderAt P z ≠ ⊤)
    (hdiv : MeromorphicOn.divisor xi Set.univ =
      MeromorphicOn.divisor P Set.univ) :
    AnalyticOnNhd ℂ (cancelledEntireQuotient xi P) Set.univ ∧
      ∀ z, cancelledEntireQuotient xi P z ≠ 0 :=
  cancelledEntireQuotient_analytic_ne_zero
    hxi hP hxiFinite hPFinite hdiv

example {xi P : ℂ → ℂ}
    (hxi : AnalyticOnNhd ℂ xi Set.univ)
    (hP : AnalyticOnNhd ℂ P Set.univ)
    (hxiFinite : ∀ z, meromorphicOrderAt xi z ≠ ⊤)
    (hPFinite : ∀ z, meromorphicOrderAt P z ≠ ⊤)
    (hdiv : MeromorphicOn.divisor xi Set.univ =
      MeromorphicOn.divisor P Set.univ)
    (hP0 : P 0 ≠ 0)
    (hgrowth : SubquadraticLogNormGrowth
      (cancelledEntireQuotient xi P)) :
    ∃ a b : ℂ, ∀ z, xi z =
      Complex.exp (a + b * z) * P z :=
  hadamardRepresentation_of_cancelledQuotientGrowth
    hxi hP hxiFinite hPFinite hdiv hP0 hgrowth

example {xi P : ℂ → ℂ}
    (hxi : AnalyticOnNhd ℂ xi Set.univ)
    (hP : AnalyticOnNhd ℂ P Set.univ)
    (hxiFinite : ∀ z, meromorphicOrderAt xi z ≠ ⊤)
    (hPFinite : ∀ z, meromorphicOrderAt P z ≠ ⊤)
    (hdiv : MeromorphicOn.divisor xi Set.univ =
      MeromorphicOn.divisor P Set.univ)
    (hxi0 : xi 0 = 1) (hP0 : P 0 = 1)
    (hfe : ∀ z, xi (1 - z) = xi z)
    (hgrowth : SubquadraticLogNormGrowth
      (cancelledEntireQuotient xi P)) :
    ∃ a b : ℂ,
      (∀ z, xi z = Complex.exp (a + b * z) * P z) ∧
      Complex.exp a = 1 ∧
      ∀ z, Complex.exp (a + b * (1 - z)) * P (1 - z) =
        Complex.exp (a + b * z) * P z :=
  normalized_hadamardRepresentation_of_cancelledQuotientGrowth
    hxi hP hxiFinite hPFinite hdiv hxi0 hP0 hfe hgrowth

example {P xi : ℂ → ℂ} {a b a' b' : ℂ}
    (hPcont : ContinuousAt P 0) (hP0 : P 0 ≠ 0)
    (hrep : ∀ s, xi s = Complex.exp (a + b * s) * P s)
    (hrep' : ∀ s, xi s = Complex.exp (a' + b' * s) * P s) :
    b = b' ∧ ∃ n : ℤ,
      a = a' + n * (2 * (Real.pi : ℂ) * Complex.I) :=
  hadamardAffineFactor_unique_mod_period hPcont hP0 hrep hrep'

example {ι : Type*} {root : ι → ℂ}
    (hroot0 : ∀ i, root i ≠ 0)
    (hsym : ReflectionSymmetricSpectrum root)
    (hunit : LiUnitDiskBound root) :
    ∀ i, (root i).re = 1 / 2 :=
  on_criticalLine_of_reflection_of_liUnitDiskBound hroot0 hsym hunit

example {ι : Type*} [DecidableEq ι]
    {root : ι → ℂ} {cutoff : ℕ → Finset ι} {height : ℕ → ℝ}
    {coefficient : ℕ → ℂ} (hroot0 : ∀ i, root i ≠ 0) :
    HeightSymmetricLiSpectralConverseStatement
        root cutoff height coefficient ↔
      BombieriLagariasPositivityHalfPlaneStatement
        root cutoff height coefficient :=
  heightSymmetricLiSpectralConverse_iff_halfPlane hroot0

example {ι : Type*} [DecidableEq ι]
    (model : RiemannLiSpectralModel ι)
    (hpositive : LiPositive riemannLiCoefficient)
    (hBL : BombieriLagariasPositivityHalfPlaneStatement
      model.root model.cutoff model.height
        (fun n ↦ liDerivativeCoefficient riemannXiLi n)) :
    KakeyaRiemannHypothesisRoot :=
  riemannHypothesis_of_bombieriLagarias model hpositive hBL

example :
    ∃ (root : Bool → ℂ) (coefficient : ℕ → ℂ)
        (_ : ReflectionSymmetricSpectrum root),
      (∀ i, root i ≠ 0) ∧
      (∀ n : ℕ, 0 < n → 0 ≤ (coefficient n).re) ∧
      ∃ i, (root i).re ≠ 1 / 2 :=
  reflection_and_abstract_positivity_insufficient

example :
    ∃ (root : Bool → ℂ) (_ : ReflectionSymmetricSpectrum root),
      (∀ i, root i ≠ 0) ∧ BombieriLagariasSummability root ∧
        ∃ i, (root i).re ≠ 1 / 2 :=
  reflection_and_summability_insufficient

example (zeros : FiniteZeroMultiset) :
    logDeriv zeros.generatingProduct 0 = zeros.liSum 1 :=
  zeros.logDeriv_generatingProduct_zero

example (zeros : FiniteZeroMultiset) (z : ℂ) (hz : z ≠ 1) :
    zeros.generatingProduct z =
      zeros.canonicalProductAtOne ((1 - z)⁻¹) :=
  zeros.generatingProduct_eq_canonical_mobius z hz

example (zeros : FiniteZeroMultiset) (k : ℕ) :
    iteratedDeriv k
      (logDeriv
        (fun z ↦ zeros.canonicalProductAtOne ((1 - z)⁻¹))) 0 /
        (k.factorial : ℂ) =
      zeros.liSum (k + 1) :=
  zeros.logDeriv_canonicalMobius_coefficient k

example {rho : ℂ} (hρ : rho ≠ 1) :
    riemannZetaZeroOrder rho ≠ 0 ↔ rho ∈ riemannZetaZeros :=
  riemannZetaZeroOrder_ne_zero_iff hρ

example {rho : ℂ} (hρ : rho ≠ 1) :
    riemannZetaZeroOrder rho ≠ ⊤ :=
  riemannZetaZeroOrder_ne_top hρ

example {R : ℝ} {rho : ℂ} :
    rho ∈ nontrivialRiemannZetaZeroWindow R ↔
      ‖rho‖ ≤ R ∧ IsNontrivialRiemannZetaZero rho :=
  mem_nontrivialRiemannZetaZeroWindow

example {R S : ℝ} (hRS : R ≤ S) :
    nontrivialRiemannZetaZeroWindow R ⊆
      nontrivialRiemannZetaZeroWindow S :=
  nontrivialRiemannZetaZeroWindow_mono hRS

example {A T : ℝ} {rho : ℂ} :
    rho ∈ symmetricHeightRiemannZetaZeroWindow A T ↔
      IsNontrivialRiemannZetaZero rho ∧
        |rho.re| ≤ A ∧ |rho.im| ≤ T :=
  mem_symmetricHeightRiemannZetaZeroWindow

noncomputable example {A T U : ℝ} (hTU : T ≤ U) :
    RiemannZetaSymmetricHeightIndex A T ↪
      RiemannZetaSymmetricHeightIndex A U :=
  riemannZetaSymmetricHeightReindex hTU

example (hRH : KakeyaRiemannHypothesisRoot) {T : ℝ} {rho : ℂ} :
    rho ∈ symmetricHeightRiemannZetaZeroWindow 1 T ↔
      IsNontrivialRiemannZetaZero rho ∧ |rho.im| ≤ T :=
  mem_symmetricHeightWindow_one_iff_of_rh hRH

example (R : ℝ) :
    FiniteLiTaylorIdentity (riemannZetaZeroWindowMultiset R) :=
  (riemannZetaZeroWindowMultiset R).finiteLiTaylorIdentity

example (hRH : KakeyaRiemannHypothesisRoot) (R : ℝ) (n : ℕ) :
    0 ≤ ((riemannZetaZeroWindowMultiset R).liSum n).re :=
  riemannZetaZeroWindow_liSum_re_nonneg_of_rh hRH R n

example (hRH : KakeyaRiemannHypothesisRoot)
    (hformula : RiemannLiZeroWindowFormula) :
    LiPositive riemannLiCoefficient :=
  liPositive_of_rh_of_zeroWindowFormula hRH hformula

example (hformula : RiemannLiZeroWindowFormula)
    (hconverse : RiemannLiConverseStatement) :
    RiemannLiCriterionStatement :=
  riemannLiCriterion_of_bridges hformula hconverse

example :
    logDeriv (liChangeOfVariables riemannXiLi) 0 =
      liDerivativeCoefficient riemannXiLi 1 :=
  logDeriv_liChangeOfVariables_zero riemannXiLi
    differentiable_riemannXiLi.differentiableAt riemannXiLi_one

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

example (zeros : FiniteZeroMultiset) :
    FiniteLiTaylorIdentity zeros :=
  zeros.finiteLiTaylorIdentity

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

example (zeros : ℕ → FiniteZeroMultiset) (g : ℂ → ℂ) (s : Set ℂ)
    (hs : IsOpen s) (h0 : (0 : ℂ) ∈ s)
    (hconv : TendstoLocallyUniformlyOn
      (fun N ↦ (zeros N).generatingProduct) g atTop s)
    (hhol : ∀ᶠ N in atTop,
      DifferentiableOn ℂ (zeros N).generatingProduct s)
    (happrox0 : ∀ᶠ N in atTop, ∀ z ∈ s,
      (zeros N).generatingProduct z ≠ 0)
    (hlimit0 : ∀ z ∈ s, g z ≠ 0)
    (k : ℕ) :
    Tendsto (fun N ↦ (zeros N).liSum (k + 1)) atTop
      (𝓝 (iteratedDeriv k (logDeriv g) 0 / (k.factorial : ℂ))) :=
  finiteZeroLiSums_tendsto_of_generatingProducts'
    zeros g s hs h0 hconv hhol happrox0 hlimit0 k

example (zeros : ℕ → FiniteZeroMultiset) (g : ℂ → ℂ) (s : Set ℂ)
    (hs : IsOpen s) (h0 : (0 : ℂ) ∈ s)
    (hconv : TendstoLocallyUniformlyOn
      (fun N z ↦ (zeros N).canonicalProductAtOne ((1 - z)⁻¹))
      g atTop s)
    (hhol : ∀ᶠ N in atTop,
      DifferentiableOn ℂ
        (fun z ↦ (zeros N).canonicalProductAtOne ((1 - z)⁻¹)) s)
    (happrox0 : ∀ᶠ N in atTop, ∀ z ∈ s,
      (zeros N).canonicalProductAtOne ((1 - z)⁻¹) ≠ 0)
    (hlimit0 : ∀ z ∈ s, g z ≠ 0)
    (k : ℕ) :
    Tendsto (fun N ↦ (zeros N).liSum (k + 1)) atTop
      (𝓝 (iteratedDeriv k (logDeriv g) 0 / (k.factorial : ℂ))) :=
  finiteZeroLiSums_tendsto_of_canonicalMobiusProducts
    zeros g s hs h0 hconv hhol happrox0 hlimit0 k

example (zeros : ℕ → FiniteZeroMultiset) (xi : ℂ → ℂ) (s : Set ℂ)
    (hs : IsOpen s) (h0 : (0 : ℂ) ∈ s)
    (hconv : TendstoLocallyUniformlyOn
      (fun N ↦ (zeros N).generatingProduct)
      (liChangeOfVariables xi) atTop s)
    (hhol : ∀ᶠ N in atTop,
      DifferentiableOn ℂ (zeros N).generatingProduct s)
    (happrox0 : ∀ᶠ N in atTop, ∀ z ∈ s,
      (zeros N).generatingProduct z ≠ 0)
    (hlimit0 : ∀ z ∈ s, liChangeOfVariables xi z ≠ 0)
    (hcoeff : LiChangeOfVariablesCoefficientIdentity xi)
    (k : ℕ) :
    Tendsto (fun N ↦ (zeros N).liSum (k + 1)) atTop
      (𝓝 (liDerivativeCoefficient xi (k + 1))) :=
  finiteZeroLiSums_tendsto_liDerivativeCoefficient
    zeros xi s hs h0 hconv hhol happrox0 hlimit0 hcoeff k

example (zeros : ℕ → FiniteZeroMultiset) (xi : ℂ → ℂ) (s : Set ℂ)
    (hproduct : XiCanonicalProductApproximation zeros xi s)
    (hcoeff : LiChangeOfVariablesCoefficientIdentity xi)
    (k : ℕ) :
    Tendsto (fun N ↦ (zeros N).liSum (k + 1)) atTop
      (𝓝 (liDerivativeCoefficient xi (k + 1))) :=
  finiteZeroLiSums_tendsto_liDerivativeCoefficient_of_canonicalProduct
    zeros xi s hproduct hcoeff k

example (s : Set ℂ)
    (hproduct : XiCanonicalProductApproximation
      (fun N ↦ riemannZetaZeroWindowMultiset (N : ℝ))
      riemannXiLi s)
    (hcoeff : LiChangeOfVariablesCoefficientIdentity riemannXiLi) :
    RiemannLiZeroWindowFormula :=
  riemannLiZeroWindowFormula_of_canonicalProduct s hproduct hcoeff

example (a : ℕ → ℝ) (N : ℕ) (x : ℝ) :
    liGeneratingPolynomial a (N + 1) x =
      liGeneratingPolynomial a N x + a N * x ^ N :=
  liGeneratingPolynomial_succ a N x
