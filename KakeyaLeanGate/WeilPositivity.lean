import Mathlib.Analysis.Calculus.ContDiff.Convolution
import Mathlib.Analysis.Distribution.TestFunction
import Mathlib.Analysis.Distribution.SchwartzSpace.Fourier
import Mathlib.Analysis.Fourier.Convolution
import Mathlib.Analysis.InnerProductSpace.GramMatrix
import Mathlib.Analysis.MellinTransform
import Mathlib.NumberTheory.LSeries.Dirichlet
import Mathlib.NumberTheory.LSeries.RiemannZeta
import Mathlib.NumberTheory.LSeries.ZetaZeros
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
open scoped ComplexOrder Convolution ContDiff Distributions FourierTransform SchwartzMap Topology

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

/-- The critical-line-centred Mellin transform in logarithmic coordinates. -/
def transform (f : Test) (z : ℂ) : ℂ :=
  ∫ t : ℝ, f t * exp (z * t)

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

/-- A candidate explicit-formula distribution.  The actual zeta distribution
is not constructed here; this structure records only its typed domain. -/
structure Distribution where
  /-- Value of the distribution on a test function. -/
  eval : Test → ℂ
  /-- The zeta distribution is complex-linear on its test space. -/
  map_add : ∀ f g, eval (f + g) = eval f + eval g
  /-- The zeta distribution is complex-linear on its test space. -/
  map_smul : ∀ (c : ℂ) f, eval (c • f) = c * eval f

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

/-- Spectral energy of finitely many ordinates on the critical line. -/
def finiteCriticalEnergy {ι : Type*} [Fintype ι] (ordinate : ι → ℝ) (f : Test) : ℝ :=
  ∑ i, normSq (transform f (ordinate i * I))

theorem finiteCriticalEnergy_nonneg
    {ι : Type*} [Fintype ι] (ordinate : ι → ℝ) (f : Test) :
    0 ≤ finiteCriticalEnergy ordinate f := by
  exact Finset.sum_nonneg fun i _ ↦ normSq_nonneg _

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
