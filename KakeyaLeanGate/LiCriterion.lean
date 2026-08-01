import KakeyaLeanGate.RiemannHypothesisRoot
import Mathlib.Algebra.Order.Chebyshev
import Mathlib.Analysis.SpecialFunctions.Complex.LogBounds
import Mathlib.Analysis.Analytic.Order
import Mathlib.Analysis.Calculus.Deriv.ZPow
import Mathlib.Analysis.Calculus.LogDeriv
import Mathlib.Analysis.Calculus.IteratedDeriv.Lemmas
import Mathlib.Analysis.Complex.LocallyUniformLimit
import Mathlib.Analysis.Complex.CanonicalDecomposition
import Mathlib.Analysis.Complex.BranchLogRoot
import Mathlib.Analysis.Complex.BorelCaratheodory
import Mathlib.Analysis.Complex.ValueDistribution.LogCounting.Basic
import Mathlib.Analysis.Normed.Module.MultipliableUniformlyOn
import Mathlib.Analysis.SpecialFunctions.Complex.Analytic
import Mathlib.Analysis.SpecialFunctions.Gamma.BohrMollerup
import Mathlib.MeasureTheory.Integral.Bochner.Set
import Mathlib.NumberTheory.LSeries.ZetaZeros
import Mathlib.NumberTheory.LSeries.HurwitzZetaValues
import Mathlib.Analysis.Real.Pi.Bounds

/-!
# A typed Li-coefficient route to the Riemann Hypothesis

This file separates definitions and elementary consequences from the two
substantial bridges that are not yet in Mathlib:

* identification of the derivative coefficients with the height-symmetric
  sum over non-trivial zeta zeros, counted with multiplicity;
* Li's theorem identifying non-negativity of every positive-index
  coefficient with `RiemannHypothesis`.

Neither bridge is assumed or asserted as a theorem here.
-/

open Complex Filter Finset MeasureTheory
open scoped Topology MeasureTheory

noncomputable section

/-- Li's normalization of the Riemann xi function.

Mathlib's `completedRiemannZeta` has assigned values at its poles, so the
pointwise expression `s * (s - 1) * completedRiemannZeta s` has the wrong
values at `0` and `1`.  The expression below uses the entire pole-subtracted
completion and is equal to that product away from the two poles.  It is
normalized by `riemannXiLi 0 = riemannXiLi 1 = 1`.
-/
def riemannXiLi (s : ℂ) : ℂ :=
  1 + s * (s - 1) * completedRiemannZeta₀ s

@[simp] theorem riemannXiLi_zero : riemannXiLi 0 = 1 := by
  simp [riemannXiLi]

@[simp] theorem riemannXiLi_one : riemannXiLi 1 = 1 := by
  simp [riemannXiLi]

theorem differentiable_riemannXiLi : Differentiable ℂ riemannXiLi := by
  unfold riemannXiLi
  exact (differentiable_const 1).add
    ((differentiable_id.mul (differentiable_id.sub (differentiable_const 1))).mul
      differentiable_completedZeta₀)

theorem riemannXiLi_one_sub (s : ℂ) :
    riemannXiLi (1 - s) = riemannXiLi s := by
  rw [riemannXiLi, riemannXiLi, completedRiemannZeta₀_one_sub]
  ring

theorem riemannXiLi_eq_mul_completedRiemannZeta
    {s : ℂ} (hs0 : s ≠ 0) (hs1 : s ≠ 1) :
    riemannXiLi s = s * (s - 1) * completedRiemannZeta s := by
  rw [riemannXiLi, completedRiemannZeta_eq]
  field_simp
  ring

/-- Intrinsic zero order of the normalized completed xi function. -/
def riemannXiZeroOrder (rho : ℂ) : ℕ∞ :=
  analyticOrderAt riemannXiLi rho

/-- Xi is entire and nonzero at `0`, so every local zero order is finite. -/
theorem riemannXiZeroOrder_ne_top (rho : ℂ) :
    riemannXiZeroOrder rho ≠ ⊤ := by
  unfold riemannXiZeroOrder
  have hxi : AnalyticOnNhd ℂ riemannXiLi Set.univ :=
    fun z _ ↦ differentiable_riemannXiLi.analyticAt z
  apply hxi.analyticOrderAt_ne_top_of_isPreconnected
    (x := (0 : ℂ)) (y := rho)
  · exact isPreconnected_univ
  · simp
  · simp
  · rw [(differentiable_riemannXiLi.analyticAt 0).analyticOrderAt_eq_zero.mpr
      (by simp)]
    exact ENat.zero_ne_top

/-- Natural-valued multiplicity of an xi zero. -/
def riemannXiZeroMultiplicity (rho : ℂ) : ℕ :=
  analyticOrderNatAt riemannXiLi rho

theorem riemannXiZeroMultiplicity_cast (rho : ℂ) :
    (riemannXiZeroMultiplicity rho : ℕ∞) = riemannXiZeroOrder rho :=
  Nat.cast_analyticOrderNatAt (riemannXiZeroOrder_ne_top rho)

/-- The functional equation preserves xi-zero multiplicity exactly. -/
theorem riemannXiZeroOrder_one_sub (rho : ℂ) :
    riemannXiZeroOrder (1 - rho) = riemannXiZeroOrder rho := by
  let g : ℂ → ℂ := fun z ↦ 1 - z
  have hg : AnalyticAt ℂ g rho := by
    dsimp [g]
    fun_prop
  have hgd : deriv g rho ≠ 0 := by
    dsimp [g]
    simp
  have hcomp := analyticOrderAt_comp_of_deriv_ne_zero
    (f := riemannXiLi) hg hgd
  have heq : riemannXiLi ∘ g = riemannXiLi := by
    funext z
    exact riemannXiLi_one_sub z
  unfold riemannXiZeroOrder
  rw [heq] at hcomp
  exact hcomp.symm

theorem riemannXiZeroMultiplicity_one_sub (rho : ℂ) :
    riemannXiZeroMultiplicity (1 - rho) =
      riemannXiZeroMultiplicity rho := by
  rw [← ENat.coe_inj, riemannXiZeroMultiplicity_cast,
    riemannXiZeroMultiplicity_cast, riemannXiZeroOrder_one_sub]

/-- Mathlib's global meromorphic divisor is a canonical multiplicity carrier
for xi's zeros. -/
noncomputable def riemannXiZeroDivisor :=
  MeromorphicOn.divisor riemannXiLi Set.univ

theorem riemannXiZeroDivisor_support :
    Function.support riemannXiZeroDivisor = riemannXiLi ⁻¹' {0} := by
  have hxi : AnalyticOnNhd ℂ riemannXiLi Set.univ :=
    fun z _ ↦ differentiable_riemannXiLi.analyticAt z
  have hfinite : ∀ u : Set.univ,
      meromorphicOrderAt riemannXiLi u.1 ≠ ⊤ := by
    intro u
    rw [(differentiable_riemannXiLi.analyticAt u.1).meromorphicOrderAt_eq]
    simpa [riemannXiZeroOrder] using riemannXiZeroOrder_ne_top u.1
  have h := hxi.meromorphicNFOn.zero_set_eq_divisor_support hfinite
  simpa [riemannXiZeroDivisor] using h.symm

theorem riemannXiZeroDivisor_finite_on_compact
    {K : Set ℂ} (hK : IsCompact K) :
    (K ∩ Function.support riemannXiZeroDivisor).Finite := by
  have hlocal : LocallyFiniteSupport
      (fun z ↦ riemannXiZeroDivisor z) :=
    fun z ↦
      riemannXiZeroDivisor.supportLocallyFiniteWithinDomain z (by trivial)
  exact hlocal.finite_inter_support_of_isCompact hK

/-- The global xi-zero support is countable: its locally finite divisor
support is discrete, while `ℂ` is hereditarily Lindelöf. -/
theorem riemannXiZeroDivisor_support_countable :
    (Function.support riemannXiZeroDivisor).Countable := by
  letI : DiscreteTopology
      (Function.support riemannXiZeroDivisor) :=
    riemannXiZeroDivisor.discreteSupport.to_subtype
  exact (HereditarilyLindelofSpace.isLindelof
    (Function.support riemannXiZeroDivisor)).countable inferInstance

/-- Canonical countable index type for all xi zeros with multiplicity.
The sigma type handles the zero-free, finite, and infinite cases uniformly. -/
def RiemannXiZeroIndex :=
  Σ rho : {z : ℂ // riemannXiLi z = 0},
    Fin (riemannXiZeroMultiplicity rho)

/-- Root represented by a multiplicity index. -/
def riemannXiZeroRoot (i : RiemannXiZeroIndex) : ℂ := i.1

noncomputable instance : Encodable RiemannXiZeroIndex := by
  classical
  have hs : (riemannXiLi ⁻¹' {0}).Countable := by
    rw [← riemannXiZeroDivisor_support]
    exact riemannXiZeroDivisor_support_countable
  letI : Encodable {z : ℂ // riemannXiLi z = 0} := hs.toEncodable
  unfold RiemannXiZeroIndex
  infer_instance

/-- The finite fiber over `z`, with one index for each unit of analytic
multiplicity. -/
noncomputable def riemannXiZeroFiber
    (z : ℂ) : Finset RiemannXiZeroIndex := by
  classical
  by_cases hz : riemannXiLi z = 0
  · exact Finset.univ.map
      ⟨fun k ↦ ⟨⟨z, hz⟩, k⟩, by
        intro a b h
        exact Fin.ext
          (congrArg (fun x : RiemannXiZeroIndex ↦ x.2.val) h)⟩
  · exact ∅

theorem mem_riemannXiZeroFiber (z : ℂ) (i : RiemannXiZeroIndex) :
    i ∈ riemannXiZeroFiber z ↔ riemannXiZeroRoot i = z := by
  classical
  unfold riemannXiZeroFiber
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

theorem card_riemannXiZeroFiber (z : ℂ) :
    (riemannXiZeroFiber z).card =
      riemannXiZeroMultiplicity z := by
  classical
  unfold riemannXiZeroFiber
  split_ifs with hz
  · rw [Finset.card_map, Finset.card_univ, Fintype.card_fin]
  · have hm : riemannXiZeroMultiplicity z = 0 := by
      rw [← Nat.cast_inj (R := ℕ∞),
        riemannXiZeroMultiplicity_cast]
      unfold riemannXiZeroOrder
      simp only [Nat.cast_zero]
      rw [(differentiable_riemannXiLi.analyticAt z).analyticOrderAt_eq_zero]
      exact hz
    simp [hm]

theorem analyticOrderAt_riemannXiLi_eq_zeroFiber_card (z : ℂ) :
    analyticOrderAt riemannXiLi z =
      ((riemannXiZeroFiber z).card : ℕ∞) := by
  rw [card_riemannXiZeroFiber, riemannXiZeroMultiplicity_cast]
  rfl

/-- The xi divisor value is exactly the natural analytic multiplicity,
embedded in the integer-valued divisor. -/
theorem riemannXiZeroDivisor_apply (z : ℂ) :
    riemannXiZeroDivisor z =
      (riemannXiZeroMultiplicity z : ℤ) := by
  unfold riemannXiZeroDivisor
  have hA : AnalyticAt ℂ riemannXiLi z :=
    differentiable_riemannXiLi.analyticAt z
  have hM : MeromorphicOn riemannXiLi Set.univ :=
    fun w _ ↦
      differentiable_riemannXiLi.analyticAt w |>.meromorphicAt
  rw [MeromorphicOn.divisor_apply hM (Set.mem_univ z),
    hA.meromorphicOrderAt_eq]
  have ho : analyticOrderAt riemannXiLi z =
      (riemannXiZeroMultiplicity z : ℕ∞) := by
    simpa [riemannXiZeroOrder] using
      (riemannXiZeroMultiplicity_cast z).symm
  rw [ho]
  simp

theorem riemannXiZeroRoot_ne_zero (i : RiemannXiZeroIndex) :
    riemannXiZeroRoot i ≠ 0 := by
  intro h
  have hi := i.1.property
  rw [show (i.1 : ℂ) = 0 from h, riemannXiLi_zero] at hi
  exact one_ne_zero hi

/-- Multiplicity-preserving reflection of the canonical xi-zero index. -/
noncomputable def riemannXiZeroReflection
    (i : RiemannXiZeroIndex) : RiemannXiZeroIndex :=
  ⟨⟨1 - riemannXiZeroRoot i, by
      rw [riemannXiLi_one_sub]
      exact i.1.property⟩,
    Fin.cast (riemannXiZeroMultiplicity_one_sub
      (riemannXiZeroRoot i)).symm i.2⟩

@[simp] theorem riemannXiZeroRoot_reflection (i : RiemannXiZeroIndex) :
    riemannXiZeroRoot (riemannXiZeroReflection i) =
      1 - riemannXiZeroRoot i := rfl

@[simp] theorem riemannXiZeroReflection_val (i : RiemannXiZeroIndex) :
    (riemannXiZeroReflection i).2.val = i.2.val := by
  simp [riemannXiZeroReflection]

theorem riemannXiZeroReflection_injective :
    Function.Injective riemannXiZeroReflection := by
  intro i j h
  have hr' := congrArg riemannXiZeroRoot h
  simp only [riemannXiZeroRoot_reflection] at hr'
  have hr : riemannXiZeroRoot i = riemannXiZeroRoot j := by
    linear_combination -hr'
  have e : i.1 = j.1 := Subtype.ext hr
  have hv' := congrArg (fun x : RiemannXiZeroIndex ↦ x.2.val) h
  simp only [riemannXiZeroReflection_val] at hv'
  exact Sigma.ext e <| (Fin.heq_ext_iff (by rw [e])).mpr hv'

noncomputable def riemannXiZeroReflectionEmbedding :
    RiemannXiZeroIndex ↪ RiemannXiZeroIndex :=
  ⟨riemannXiZeroReflection, riemannXiZeroReflection_injective⟩

/-- Every multiplicity occurrence has an explicit reflected occurrence with
the same ordinal in the reflected fiber. -/
theorem exists_riemannXiZeroReflectedMultiplicityIndex
    (i : RiemannXiZeroIndex) :
    ∃ j : RiemannXiZeroIndex,
      riemannXiZeroRoot j = 1 - riemannXiZeroRoot i ∧
        j.2.val = i.2.val :=
  ⟨riemannXiZeroReflection i, riemannXiZeroRoot_reflection i,
    riemannXiZeroReflection_val i⟩

/-- The reflected inverse-root pair is an inverse-quadratic term. -/
theorem riemannXiZeroRoot_inv_add_reflection (i : RiemannXiZeroIndex) :
    (riemannXiZeroRoot i)⁻¹ +
        (riemannXiZeroRoot (riemannXiZeroReflection i))⁻¹ =
      (riemannXiZeroRoot i *
        (1 - riemannXiZeroRoot i))⁻¹ := by
  rw [riemannXiZeroRoot_reflection]
  have h0 := riemannXiZeroRoot_ne_zero i
  have h1 : 1 - riemannXiZeroRoot i ≠ 0 := by
    rw [← riemannXiZeroRoot_reflection]
    exact riemannXiZeroRoot_ne_zero (riemannXiZeroReflection i)
  field_simp
  ring

theorem riemannXiZeroMultiplicity_pos_iff (z : ℂ) :
    0 < riemannXiZeroMultiplicity z ↔ riemannXiLi z = 0 := by
  rw [Nat.pos_iff_ne_zero]
  constructor
  · intro hm
    apply (differentiable_riemannXiLi.analyticAt z).analyticOrderAt_ne_zero.mp
    intro ho
    apply hm
    rw [← Nat.cast_inj (R := ℕ∞),
      riemannXiZeroMultiplicity_cast]
    simpa [riemannXiZeroOrder] using ho
  · intro hz hm
    have ho :=
      (differentiable_riemannXiLi.analyticAt z).analyticOrderAt_ne_zero.mpr hz
    apply ho
    change riemannXiZeroOrder z = 0
    rw [← riemannXiZeroMultiplicity_cast]
    simp [hm]

/-- Completeness of the countable enumeration, including both directions:
an index gives a zero, and every zero has at least one multiplicity index. -/
theorem exists_riemannXiZeroRoot_iff (z : ℂ) :
    (∃ i : RiemannXiZeroIndex, riemannXiZeroRoot i = z) ↔
      riemannXiLi z = 0 := by
  constructor
  · rintro ⟨i, rfl⟩
    exact i.1.property
  · intro hz
    have hcard : 0 < (riemannXiZeroFiber z).card := by
      rw [card_riemannXiZeroFiber,
        riemannXiZeroMultiplicity_pos_iff]
      exact hz
    obtain ⟨i, hi⟩ := Finset.card_pos.mp hcard
    exact ⟨i, (mem_riemannXiZeroFiber z i).mp hi⟩

/-- Only finitely many multiplicity indices lie over a bounded xi-zero
window. -/
theorem riemannXiZeroIndex_finite_norm_le (R : ℝ) :
    {i : RiemannXiZeroIndex |
      ‖riemannXiZeroRoot i‖ ≤ R}.Finite := by
  let Z : Set ℂ :=
    Metric.closedBall 0 R ∩ Function.support riemannXiZeroDivisor
  have hZ : Z.Finite :=
    riemannXiZeroDivisor_finite_on_compact
      (ProperSpace.isCompact_closedBall 0 R)
  have hU : (⋃ z ∈ Z, (↑(riemannXiZeroFiber z) :
      Set RiemannXiZeroIndex)).Finite := by
    apply hZ.biUnion
    intro z hz
    exact Finset.finite_toSet _
  apply hU.subset
  intro i hi
  simp only [Set.mem_setOf_eq] at hi
  have hzball :
      riemannXiZeroRoot i ∈ Metric.closedBall 0 R := by
    simpa [Metric.mem_closedBall, dist_zero_right] using hi
  have hzsupport :
      riemannXiZeroRoot i ∈
        Function.support riemannXiZeroDivisor := by
    rw [riemannXiZeroDivisor_support]
    exact i.1.property
  apply Set.mem_iUnion_of_mem (riemannXiZeroRoot i)
  apply Set.mem_iUnion_of_mem ⟨hzball, hzsupport⟩
  exact (mem_riemannXiZeroFiber _ _).mpr rfl

/-- The finite multiplicity-indexed xi-zero window of radius `r`. -/
noncomputable def riemannXiZeroIndexWindow
    (r : ℝ) : Finset RiemannXiZeroIndex :=
  (riemannXiZeroIndex_finite_norm_le r).toFinset

@[simp]
theorem mem_riemannXiZeroIndexWindow
    (r : ℝ) (i : RiemannXiZeroIndex) :
    i ∈ riemannXiZeroIndexWindow r ↔
      ‖riemannXiZeroRoot i‖ ≤ r := by
  simp [riemannXiZeroIndexWindow]

/-- Radial windows are only reflection-stable after enlarging the radius by
one.  This boundary shift is why inverse-square summability alone does not
justify unpaired radial inverse-root sums. -/
theorem riemannXiZeroReflection_mem_window_add_one
    {R : ℝ} {i : RiemannXiZeroIndex}
    (hi : i ∈ riemannXiZeroIndexWindow R) :
    riemannXiZeroReflection i ∈
      riemannXiZeroIndexWindow (R + 1) := by
  rw [mem_riemannXiZeroIndexWindow] at hi ⊢
  rw [riemannXiZeroRoot_reflection]
  calc
    ‖1 - riemannXiZeroRoot i‖
        ≤ ‖(1 : ℂ)‖ + ‖riemannXiZeroRoot i‖ := norm_sub_le _ _
    _ ≤ 1 + R := by simpa using add_le_add_left hi 1
    _ = R + 1 := by ring

/-- The finite divisor obtained by placing one unit at every multiplicity
index in the bounded xi-zero window. -/
noncomputable def riemannXiZeroIndexWindowDivisor
    (r : ℝ) : Function.locallyFinsupp ℂ ℤ :=
  ∑ i ∈ riemannXiZeroIndexWindow r,
    Function.locallyFinsuppWithin.single
      (riemannXiZeroRoot i) 1

/-- The bounded multiplicity-index divisor is pointwise dominated by xi's
full analytic divisor. -/
theorem riemannXiZeroIndexWindowDivisor_le (r : ℝ) :
    riemannXiZeroIndexWindowDivisor r ≤
      riemannXiZeroDivisor := by
  classical
  intro z
  rw [riemannXiZeroDivisor_apply]
  unfold riemannXiZeroIndexWindowDivisor
  have heval := congrFun
    (Function.locallyFinsuppWithin.coe_sum
      (s := riemannXiZeroIndexWindow r)
      (F := fun i ↦ Function.locallyFinsuppWithin.single
        (riemannXiZeroRoot i) (1 : ℤ))) z
  rw [heval, Finset.sum_apply]
  simp only [Function.locallyFinsuppWithin.single_apply,
    sum_ite, sum_const_zero, add_zero]
  simp only [Finset.sum_const, nsmul_eq_mul, mul_one]
  norm_cast
  calc
    {i ∈ riemannXiZeroIndexWindow r |
        z = riemannXiZeroRoot i}.card
        ≤ (riemannXiZeroFiber z).card := by
      apply Finset.card_le_card
      intro i hi
      simp only [Finset.mem_filter] at hi
      exact (mem_riemannXiZeroFiber z i).mpr hi.2.symm
    _ = riemannXiZeroMultiplicity z :=
      card_riemannXiZeroFiber z

/-- The canonical multiplicity enumeration escapes every bounded set,
including automatically in the zero-free and finite cases. -/
theorem riemannXiZeroRoot_escape :
    Tendsto (fun i : RiemannXiZeroIndex ↦ ‖riemannXiZeroRoot i‖)
      cofinite atTop := by
  apply Filter.tendsto_atTop.mpr
  intro R
  change {i : RiemannXiZeroIndex |
    R ≤ ‖riemannXiZeroRoot i‖} ∈ cofinite
  rw [Filter.mem_cofinite]
  apply (riemannXiZeroIndex_finite_norm_le R).subset
  intro i hi
  simp only [Set.mem_compl_iff, Set.mem_setOf_eq] at hi ⊢
  exact le_of_lt (lt_of_not_ge hi)

/-- Xi's logarithmic zero-counting function is exactly the positive divisor
count from Mathlib's value-distribution API. -/
theorem riemannXiZeroDivisor_logCounting :
    Function.locallyFinsuppWithin.logCounting riemannXiZeroDivisor =
      ValueDistribution.logCounting riemannXiLi 0 := by
  rw [ValueDistribution.logCounting_zero]
  have hxi : AnalyticOnNhd ℂ riemannXiLi Set.univ :=
    fun z _ ↦ differentiable_riemannXiLi.analyticAt z
  have hnonneg : 0 ≤ MeromorphicOn.divisor riemannXiLi Set.univ :=
    MeromorphicOn.AnalyticOnNhd.divisor_nonneg hxi
  unfold riemannXiZeroDivisor
  rw [posPart_eq_self.mpr hnonneg]

/-- Jensen's formula specialized to normalized xi: the trailing-coefficient
constant vanishes because `xi(0)=1`.  Any future global norm bound can feed
directly into this identity to control the zero divisor. -/
theorem riemannXiZeroDivisor_logCounting_eq_circleAverage
    {R : ℝ} (hR : R ≠ 0) :
    Function.locallyFinsuppWithin.logCounting riemannXiZeroDivisor R =
      Real.circleAverage (Real.log ‖riemannXiLi ·‖) 0 R := by
  unfold riemannXiZeroDivisor
  have hmer : Meromorphic riemannXiLi :=
    fun z ↦ (differentiable_riemannXiLi.analyticAt z).meromorphicAt
  rw [Function.locallyFinsuppWithin.logCounting_divisor_eq_circleAverage_sub_const
    hmer hR]
  rw [(differentiable_riemannXiLi.analyticAt 0).meromorphicTrailingCoeffAt_of_ne_zero
    (by simp)]
  simp

/-- Exact conversion from the multiplicity-index window cardinality to
Mathlib's weighted divisor count.  Every root of norm at most `r`
contributes at least `log 2` when the logarithmic count is evaluated at
radius `2r`. -/
theorem riemannXiZeroIndexWindow_card_mul_log_two_le_logCounting
    {r : ℝ} (hr : 1 ≤ r) :
    ((riemannXiZeroIndexWindow r).card : ℝ) * Real.log 2 ≤
      Function.locallyFinsuppWithin.logCounting
        riemannXiZeroDivisor (2 * r) := by
  calc
    ((riemannXiZeroIndexWindow r).card : ℝ) * Real.log 2 =
        ∑ _i ∈ riemannXiZeroIndexWindow r, Real.log 2 := by
      simp
    _ ≤ ∑ i ∈ riemannXiZeroIndexWindow r,
        Function.locallyFinsuppWithin.logCounting
          (Function.locallyFinsuppWithin.single
            (riemannXiZeroRoot i) 1) (2 * r) := by
      apply Finset.sum_le_sum
      intro i hi
      have hir : ‖riemannXiZeroRoot i‖ ≤ r :=
        (mem_riemannXiZeroIndexWindow r i).mp hi
      rw [Function.locallyFinsuppWithin.logCounting_single_eq_log_sub_const
        (hir.trans (by linarith))]
      norm_num
      rw [← Real.log_div (by positivity)
        (norm_ne_zero_iff.mpr (riemannXiZeroRoot_ne_zero i))]
      apply Real.log_le_log (by norm_num)
      rw [le_div_iff₀
        (norm_pos_iff.mpr (riemannXiZeroRoot_ne_zero i))]
      nlinarith
    _ = Function.locallyFinsuppWithin.logCounting
        (riemannXiZeroIndexWindowDivisor r) (2 * r) := by
      unfold riemannXiZeroIndexWindowDivisor
      rw [map_sum, Finset.sum_apply]
    _ ≤ Function.locallyFinsuppWithin.logCounting
        riemannXiZeroDivisor (2 * r) := by
      exact Function.locallyFinsuppWithin.logCounting_le
        (riemannXiZeroIndexWindowDivisor_le r) (by linarith)

/-- The strongest unconditional growth statement currently supplied by
pinned Mathlib: xi is bounded on each compact set.  This is local and is not
a finite-order estimate as the bound has no uniform dependence on `K`. -/
theorem riemannXiLi_bounded_on_compact
    {K : Set ℂ} (hK : IsCompact K) :
    ∃ C : ℝ, ∀ s ∈ K, ‖riemannXiLi s‖ ≤ C :=
  hK.exists_bound_of_continuousOn differentiable_riemannXiLi.continuous.continuousOn

/-- The source-standard global estimate `|xi(s)| ≤ exp(C |s| log(|s|+2))`
that implies order at most one. -/
def RiemannXiOrderOneGrowthBound : Prop :=
  ∃ C R : ℝ, 0 ≤ C ∧ 2 ≤ R ∧ ∀ s : ℂ, R ≤ ‖s‖ →
    ‖riemannXiLi s‖ ≤
      Real.exp (C * ‖s‖ * Real.log (‖s‖ + 2))

/-- The functional equation patches any right-half-plane bound whose
majorant is symmetric in `s` and `1-s` to the whole plane. -/
theorem riemannXiLi_global_norm_bound_of_rightHalfPlane
    (M : ℝ → ℝ)
    (hbound : ∀ s : ℂ, 1 / 2 ≤ s.re →
      ‖riemannXiLi s‖ ≤ M (max ‖s‖ ‖1 - s‖)) :
    ∀ s : ℂ, ‖riemannXiLi s‖ ≤ M (max ‖s‖ ‖1 - s‖) := by
  intro s
  by_cases hs : 1 / 2 ≤ s.re
  · exact hbound s hs
  · rw [← riemannXiLi_one_sub s]
    have hreflect : 1 / 2 ≤ (1 - s).re := by
      simp only [sub_re, one_re]
      linarith
    simpa [max_comm] using hbound (1 - s) hreflect

/-- An actual zeta growth estimate available from the pinned Dirichlet
series API: zeta is uniformly bounded on `re s ≥ 2` by the real
`p = 2` series.  This closes the elementary far-right region, but does
not provide the critical-strip bound needed for xi's order. -/
theorem norm_riemannZeta_le_tsum_inv_sq_of_two_le_re
    {s : ℂ} (hs : 2 ≤ s.re) :
    ‖riemannZeta s‖ ≤ ∑' n : ℕ, 1 / (n : ℝ) ^ 2 := by
  rw [zeta_eq_tsum_one_div_nat_cpow
    (lt_of_lt_of_le (by norm_num) hs)]
  calc
    ‖∑' n : ℕ, 1 / (n : ℂ) ^ s‖ ≤
        ∑' n : ℕ, ‖1 / (n : ℂ) ^ s‖ :=
      norm_tsum_le_tsum_norm
        (Complex.summable_one_div_nat_cpow.mpr
          (lt_of_lt_of_le (by norm_num) hs)).norm
    _ ≤ ∑' n : ℕ, 1 / (n : ℝ) ^ 2 := by
      apply Summable.tsum_le_tsum
      · intro n
        rcases n with _ | n
        · have hs0 : s ≠ 0 := by
            intro h
            subst s
            norm_num at hs
          simp [Complex.zero_cpow hs0]
        · simp only [norm_div, norm_one, ← Complex.ofReal_natCast,
            Complex.norm_cpow_eq_rpow_re_of_pos
              (Nat.cast_pos.mpr (Nat.succ_pos n))]
          rw [div_le_div_iff_of_pos_left (by norm_num)
            (Real.rpow_pos_of_pos
              (Nat.cast_pos.mpr (Nat.succ_pos n)) _)
            (pow_pos (Nat.cast_pos.mpr (Nat.succ_pos n)) _)]
          rw [← Real.rpow_natCast]
          exact Real.rpow_le_rpow_of_exponent_le
            (by norm_num : (1 : ℝ) ≤ (n + 1 : ℕ)) hs
      · exact (Complex.summable_one_div_nat_cpow.mpr
          (lt_of_lt_of_le (by norm_num) hs)).norm
      · exact Real.summable_one_div_nat_pow.mpr (by norm_num)

/-- Euler's integral gives the classical vertical-line Gamma estimate
`|Γ(σ+it)| ≤ Γ(σ)` for `σ > 0`. -/
theorem norm_complexGamma_le_realGamma (s : ℂ) (hs : 0 < s.re) :
    ‖Complex.Gamma s‖ ≤ Real.Gamma s.re := by
  rw [Complex.Gamma_eq_integral hs, Complex.GammaIntegral]
  calc
    ‖∫ x in Set.Ioi (0 : ℝ), ((-x).exp : ℂ) * (x : ℂ) ^ (s - 1)‖ ≤
        ∫ x in Set.Ioi (0 : ℝ),
          ‖((-x).exp : ℂ) * (x : ℂ) ^ (s - 1)‖ :=
      MeasureTheory.norm_integral_le_integral_norm _
    _ = ∫ x in Set.Ioi (0 : ℝ),
          Real.exp (-x) * x ^ (s.re - 1) := by
      apply MeasureTheory.setIntegral_congr_fun measurableSet_Ioi
      intro x hx
      dsimp only
      rw [norm_mul, Complex.norm_of_nonneg (Real.exp_pos _).le,
        Complex.norm_cpow_eq_rpow_re_of_pos hx]
      simp
    _ = Real.Gamma s.re := by
      rw [Real.Gamma_eq_integral hs]

/-- A fully explicit (coarse) real Gamma growth estimate from monotonicity
and `n! ≤ n^n`. -/
theorem realGamma_le_ceil_pow_self (x : ℝ) (hx : 2 ≤ x) :
    Real.Gamma x ≤ (Nat.ceil x : ℝ) ^ Nat.ceil x := by
  let N := Nat.ceil x
  have hxN : x ≤ (N : ℝ) := Nat.le_ceil x
  have hN2 : 2 ≤ (N : ℝ) := hx.trans hxN
  have hmono : Real.Gamma x ≤ Real.Gamma ((N : ℝ) + 1) := by
    apply Real.Gamma_strictMonoOn_Ici.monotoneOn
    · exact hx
    · exact hN2.trans (le_add_of_nonneg_right zero_le_one)
    · exact hxN.trans (le_add_of_nonneg_right zero_le_one)
  rw [Real.Gamma_nat_eq_factorial] at hmono
  exact hmono.trans (by exact_mod_cast Nat.factorial_le_pow N)

theorem norm_complexGamma_le_ceil_pow_re (s : ℂ) (hs : 2 ≤ s.re) :
    ‖Complex.Gamma s‖ ≤ (Nat.ceil s.re : ℝ) ^ Nat.ceil s.re :=
  (norm_complexGamma_le_realGamma s (by linarith)).trans
    (realGamma_le_ceil_pow_self s.re hs)

/-- Explicit completed-xi growth in the far-right half-plane.  The only
non-polynomial term is the transparent ceiling power coming from Gamma. -/
theorem norm_riemannXiLi_le_farRight {s : ℂ} (hs : 4 ≤ s.re) :
    ‖riemannXiLi s‖ ≤
      ‖s‖ * ‖s - 1‖ *
        ((Nat.ceil (s.re / 2) : ℝ) ^ Nat.ceil (s.re / 2)) *
        (∑' n : ℕ, 1 / (n : ℝ) ^ 2) := by
  have hs0 : s ≠ 0 := by
    intro h
    subst s
    norm_num at hs
  have hs1 : s ≠ 1 := by
    intro h
    subst s
    norm_num at hs
  have hsone : 1 < s.re := lt_of_lt_of_le (by norm_num) hs
  rw [riemannXiLi_eq_mul_completedRiemannZeta hs0 hs1,
    completedZeta_eq_tsum_of_one_lt_re hsone,
    ← zeta_eq_tsum_one_div_nat_cpow hsone]
  simp only [norm_mul]
  have hpi : ‖(Real.pi : ℂ) ^ (-s / 2)‖ ≤ 1 := by
    rw [Complex.norm_cpow_eq_rpow_re_of_pos Real.pi_pos]
    apply Real.rpow_le_one_of_one_le_of_nonpos
    · linarith [Real.two_le_pi]
    · norm_num [div_re]
      linarith
  have hgamma : ‖Complex.Gamma (s / 2)‖ ≤
      (Nat.ceil (s.re / 2) : ℝ) ^ Nat.ceil (s.re / 2) := by
    convert norm_complexGamma_le_ceil_pow_re (s / 2)
      (by norm_num [div_re]; linarith) using 1
    all_goals norm_num [div_re]
  have hzeta := norm_riemannZeta_le_tsum_inv_sq_of_two_le_re
    (show 2 ≤ s.re by linarith)
  calc
    ‖s‖ * ‖s - 1‖ *
          (‖(Real.pi : ℂ) ^ (-s / 2)‖ * ‖Complex.Gamma (s / 2)‖ *
            ‖riemannZeta s‖)
        ≤ ‖s‖ * ‖s - 1‖ * (1 *
            ((Nat.ceil (s.re / 2) : ℝ) ^ Nat.ceil (s.re / 2)) *
            (∑' n : ℕ, 1 / (n : ℝ) ^ 2)) := by gcongr
    _ = _ := by ring

/-- The functional equation reflects the explicit far-right estimate to
the far-left half-plane. -/
theorem norm_riemannXiLi_le_farLeft {s : ℂ} (hs : s.re ≤ -3) :
    ‖riemannXiLi s‖ ≤
      ‖1 - s‖ * ‖s‖ *
        ((Nat.ceil ((1 - s).re / 2) : ℝ) ^
          Nat.ceil ((1 - s).re / 2)) *
        (∑' n : ℕ, 1 / (n : ℝ) ^ 2) := by
  rw [← riemannXiLi_one_sub s]
  have hreflect : 4 ≤ (1 - s).re := by
    simp only [sub_re, one_re]
    linarith
  simpa [norm_neg] using
    (norm_riemannXiLi_le_farRight (s := 1 - s) hreflect)

/-- A Mellin transform is uniformly bounded on a closed vertical strip
by the sum of its endpoint norm integrals, provided it converges at every
complex exponent. -/
theorem norm_mellin_le_endpointMajorant
    {E : Type*} [NormedAddCommGroup E] [NormedSpace ℂ E]
    (f : ℝ → E) (hconv : ∀ s : ℂ, MellinConvergent f s)
    {a b : ℝ} (s : ℂ) (ha : a ≤ s.re) (hb : s.re ≤ b) :
    ‖mellin f s‖ ≤
      ∫ t in Set.Ioi (0 : ℝ),
        (‖(t : ℂ) ^ ((a : ℂ) - 1) • f t‖ +
          ‖(t : ℂ) ^ ((b : ℂ) - 1) • f t‖) := by
  rw [mellin]
  calc
    ‖∫ t in Set.Ioi (0 : ℝ), (t : ℂ) ^ (s - 1) • f t‖ ≤
        ∫ t in Set.Ioi (0 : ℝ), ‖(t : ℂ) ^ (s - 1) • f t‖ :=
      MeasureTheory.norm_integral_le_integral_norm _
    _ ≤ ∫ t in Set.Ioi (0 : ℝ),
        (‖(t : ℂ) ^ ((a : ℂ) - 1) • f t‖ +
          ‖(t : ℂ) ^ ((b : ℂ) - 1) • f t‖) := by
      apply MeasureTheory.setIntegral_mono_on
      · exact (hconv s).norm
      · exact ((hconv (a : ℂ)).norm.add (hconv (b : ℂ)).norm)
      · exact measurableSet_Ioi
      · intro t ht
        simp only [norm_smul]
        have ht0 : 0 < t := ht
        simp only [Complex.norm_cpow_eq_rpow_re_of_pos ht0]
        have hpow : t ^ (s.re - 1) ≤
            t ^ (a - 1) + t ^ (b - 1) := by
          by_cases ht1 : t ≤ 1
          · exact (Real.rpow_le_rpow_of_exponent_ge ht0 ht1
                (by linarith)).trans
              (le_add_of_nonneg_right (Real.rpow_nonneg ht0.le _))
          · have h1t : 1 ≤ t := le_of_not_ge ht1
            exact (Real.rpow_le_rpow_of_exponent_le h1t
                (by linarith)).trans
              (le_add_of_nonneg_left (Real.rpow_nonneg ht0.le _))
        simpa [sub_re, add_mul] using
          mul_le_mul_of_nonneg_right hpow (norm_nonneg (f t))

/-- The pole-subtracted completion is uniformly bounded on every closed
vertical strip, directly from Mathlib's all-exponent Mellin representation. -/
theorem exists_norm_completedRiemannZeta₀_le_on_verticalStrip
    (a b : ℝ) :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ s : ℂ, a ≤ s.re → s.re ≤ b →
      ‖completedRiemannZeta₀ s‖ ≤ C := by
  let P := HurwitzZeta.hurwitzEvenFEPair 0
  let C0 : ℝ :=
    ∫ t in Set.Ioi (0 : ℝ),
      (‖(t : ℂ) ^ (((a / 2 : ℝ) : ℂ) - 1) • P.f_modif t‖ +
        ‖(t : ℂ) ^ (((b / 2 : ℝ) : ℂ) - 1) • P.f_modif t‖)
  refine ⟨|C0| / 2, by positivity, ?_⟩
  intro s ha hb
  have hconv : ∀ u : ℂ, MellinConvergent P.f_modif u :=
    fun u ↦ (P.toStrongFEPair.hasMellin u).1
  have hm := norm_mellin_le_endpointMajorant P.f_modif hconv (s / 2)
    (a := a / 2) (b := b / 2)
    (by norm_num [div_re]; linarith) (by norm_num [div_re]; linarith)
  change ‖P.Λ₀ (s / 2) / 2‖ ≤ |C0| / 2
  rw [norm_div]
  norm_num
  exact div_le_div_of_nonneg_right (hm.trans (le_abs_self C0)) (by norm_num)

/-- Consequently normalized xi has only quadratic growth on every fixed
closed vertical strip. -/
theorem exists_norm_riemannXiLi_le_on_verticalStrip (a b : ℝ) :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ s : ℂ, a ≤ s.re → s.re ≤ b →
      ‖riemannXiLi s‖ ≤ 1 + ‖s‖ * ‖s - 1‖ * C := by
  obtain ⟨C, hC0, hC⟩ :=
    exists_norm_completedRiemannZeta₀_le_on_verticalStrip a b
  refine ⟨C, hC0, fun s ha hb ↦ ?_⟩
  rw [riemannXiLi]
  calc
    ‖1 + s * (s - 1) * completedRiemannZeta₀ s‖
        ≤ ‖(1 : ℂ)‖ + ‖s * (s - 1) * completedRiemannZeta₀ s‖ :=
      norm_add_le _ _
    _ = 1 + ‖s‖ * ‖s - 1‖ * ‖completedRiemannZeta₀ s‖ := by
      simp only [norm_one, norm_mul]
    _ ≤ 1 + ‖s‖ * ‖s - 1‖ * C := by
      gcongr
      exact hC s ha hb

/-- The ceiling-power Gamma majorant has the standard order-one
`exp(r log(r+2))` form. -/
theorem ceilPow_le_exp_orderMajorant {x r : ℝ} (hx : 2 ≤ x)
    (hxr : x ≤ r / 2) (_hr : 4 ≤ r) :
    (Nat.ceil x : ℝ) ^ Nat.ceil x ≤
      Real.exp (r * Real.log (r + 2)) := by
  let N := Nat.ceil x
  have hNpos : 0 < (N : ℝ) := by
    exact_mod_cast (Nat.ceil_pos.mpr (by linarith : 0 < x))
  have hNr : (N : ℝ) ≤ r := by
    have hceil : (N : ℝ) < x + 1 :=
      Nat.ceil_lt_add_one (by linarith)
    linarith
  have hlog : Real.log (N : ℝ) ≤ Real.log (r + 2) :=
    Real.log_le_log hNpos (by linarith)
  have hlog0 : 0 ≤ Real.log (r + 2) :=
    Real.log_nonneg (by linarith)
  rw [← Real.rpow_natCast, Real.rpow_def_of_pos hNpos]
  apply Real.exp_le_exp.mpr
  calc
    Real.log (N : ℝ) * (N : ℝ)
        ≤ Real.log (r + 2) * (N : ℝ) :=
      mul_le_mul_of_nonneg_right hlog hNpos.le
    _ ≤ Real.log (r + 2) * r :=
      mul_le_mul_of_nonneg_left hNr hlog0
    _ = r * Real.log (r + 2) := by ring

/-- The explicit far-right estimate has order-one exponential growth. -/
theorem exists_riemannXiLi_orderBound_farRight :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ s : ℂ, 4 ≤ s.re →
      ‖riemannXiLi s‖ ≤
        Real.exp (C * ‖s‖ * Real.log (‖s‖ + 2)) := by
  let Z : ℝ := ∑' n : ℕ, 1 / (n : ℝ) ^ 2
  have hZ0 : 0 ≤ Z := by
    apply tsum_nonneg
    intro n
    positivity
  let L := Real.log 2
  have hL : 0 < L := Real.log_pos (by norm_num)
  let D := Real.log (Z + 1) / L
  have hlogZ0 : 0 ≤ Real.log (Z + 1) :=
    Real.log_nonneg (by linarith)
  have hD0 : 0 ≤ D := div_nonneg hlogZ0 hL.le
  refine ⟨D + 4, by positivity, ?_⟩
  intro s hs
  let r := ‖s‖
  let A := r * Real.log (r + 2)
  have hreNorm : s.re ≤ r :=
    (le_abs_self s.re).trans (Complex.abs_re_le_norm s)
  have hr : 4 ≤ r := hs.trans hreNorm
  have hrpos : 0 < r := by linarith
  have hlog0 : 0 ≤ Real.log (r + 2) :=
    Real.log_nonneg (by linarith)
  have hlogr : Real.log r ≤ A := by
    have hle : Real.log r ≤ Real.log (r + 2) :=
      Real.log_le_log hrpos (by linarith)
    exact hle.trans (by
      calc
        Real.log (r + 2) ≤ r * Real.log (r + 2) := by
          nlinarith
        _ = A := rfl)
  have hrExp : r ≤ Real.exp A := by
    rw [← Real.exp_log hrpos]
    exact Real.exp_le_exp.mpr hlogr
  have hr1 : r + 1 ≤ r ^ 2 := by nlinarith
  have hpoly : r * (r + 1) ≤ Real.exp (3 * A) := by
    calc
      r * (r + 1) ≤ Real.exp A * (Real.exp A) ^ 2 := by
        gcongr
        exact hr1.trans
          ((sq_le_sq₀ hrpos.le (Real.exp_pos A).le).mpr hrExp)
      _ = Real.exp (3 * A) := by
        rw [pow_two, ← Real.exp_add, ← Real.exp_add]
        ring_nf
  have hLA : L ≤ A := by
    have hlog2 : L ≤ Real.log (r + 2) := by
      exact Real.log_le_log (by norm_num) (by linarith)
    exact hlog2.trans (by
      calc
        Real.log (r + 2) ≤ r * Real.log (r + 2) := by
          nlinarith
        _ = A := rfl)
  have hlogZA : Real.log (Z + 1) ≤ D * A := by
    calc
      Real.log (Z + 1) = D * L := by
        dsimp [D]
        field_simp
      _ ≤ D * A := mul_le_mul_of_nonneg_left hLA hD0
  have hZexp : Z ≤ Real.exp (D * A) := by
    calc
      Z ≤ Z + 1 := by linarith
      _ = Real.exp (Real.log (Z + 1)) := by
        rw [Real.exp_log (by linarith)]
      _ ≤ Real.exp (D * A) := Real.exp_le_exp.mpr hlogZA
  have hgamma :
      (Nat.ceil (s.re / 2) : ℝ) ^ Nat.ceil (s.re / 2) ≤
        Real.exp A := by
    exact ceilPow_le_exp_orderMajorant (x := s.re / 2) (r := r)
      (by linarith) (by
        dsimp [r]
        linarith) hr
  calc
    ‖riemannXiLi s‖ ≤
        r * (r + 1) *
          ((Nat.ceil (s.re / 2) : ℝ) ^ Nat.ceil (s.re / 2)) * Z := by
      apply (norm_riemannXiLi_le_farRight hs).trans
      dsimp [r, Z]
      gcongr
      exact norm_sub_le s 1 |>.trans_eq (by simp)
    _ ≤ Real.exp (3 * A) * Real.exp A * Real.exp (D * A) := by
      gcongr
    _ = Real.exp ((D + 4) * r * Real.log (r + 2)) := by
      rw [← Real.exp_add, ← Real.exp_add]
      dsimp [A]
      ring_nf

/-- Any nonnegative quadratic strip estimate is absorbed by an
order-one exponential majorant outside a fixed disk. -/
theorem exists_exp_orderBound_of_quadratic (C : ℝ) (hC : 0 ≤ C) :
    ∃ D : ℝ, 0 ≤ D ∧ ∀ r : ℝ, 4 ≤ r →
      1 + r * (r + 1) * C ≤
        Real.exp (D * r * Real.log (r + 2)) := by
  let L := Real.log 2
  have hL : 0 < L := Real.log_pos (by norm_num)
  let E := Real.log (C + 1) / L
  have hlogC0 : 0 ≤ Real.log (C + 1) :=
    Real.log_nonneg (by linarith)
  have hE0 : 0 ≤ E := div_nonneg hlogC0 hL.le
  refine ⟨E + 4, by positivity, ?_⟩
  intro r hr
  let A := r * Real.log (r + 2)
  have hrpos : 0 < r := by linarith
  have hlog0 : 0 ≤ Real.log (r + 2) :=
    Real.log_nonneg (by linarith)
  have hlogr : Real.log r ≤ A := by
    have hle := Real.log_le_log hrpos (show r ≤ r + 2 by linarith)
    exact hle.trans (by
      calc
        Real.log (r + 2) ≤ r * Real.log (r + 2) := by nlinarith
        _ = A := rfl)
  have hrExp : r ≤ Real.exp A := by
    rw [← Real.exp_log hrpos]
    exact Real.exp_le_exp.mpr hlogr
  have hr1 : r + 1 ≤ r ^ 2 := by nlinarith
  have hr1pow : (r + 1) ^ 2 ≤ Real.exp (4 * A) := by
    calc
      (r + 1) ^ 2 ≤ (r ^ 2) ^ 2 :=
        (sq_le_sq₀ (by linarith) (sq_nonneg r)).mpr hr1
      _ ≤ ((Real.exp A) ^ 2) ^ 2 := by
        gcongr
      _ = Real.exp (4 * A) := by
        rw [pow_two, pow_two, ← Real.exp_add, ← Real.exp_add]
        ring_nf
  have hLA : L ≤ A := by
    have hlog2 : L ≤ Real.log (r + 2) :=
      Real.log_le_log (by norm_num) (by linarith)
    exact hlog2.trans (by
      calc
        Real.log (r + 2) ≤ r * Real.log (r + 2) := by nlinarith
        _ = A := rfl)
  have hlogCA : Real.log (C + 1) ≤ E * A := by
    calc
      Real.log (C + 1) = E * L := by
        dsimp [E]
        field_simp
      _ ≤ E * A := mul_le_mul_of_nonneg_left hLA hE0
  have hCexp : C + 1 ≤ Real.exp (E * A) := by
    rw [← Real.exp_log (by linarith : 0 < C + 1)]
    exact Real.exp_le_exp.mpr hlogCA
  calc
    1 + r * (r + 1) * C ≤ (C + 1) * (r + 1) ^ 2 := by
      nlinarith [sq_nonneg (r + 1)]
    _ ≤ Real.exp (E * A) * Real.exp (4 * A) := by gcongr
    _ = Real.exp ((E + 4) * r * Real.log (r + 2)) := by
      rw [← Real.exp_add]
      dsimp [A]
      ring_nf

/-- The regional estimates give an order-one bound on the right half-plane,
including the transition through the central strip. -/
theorem exists_riemannXiLi_orderBound_rightHalfPlane :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ s : ℂ, 1 / 2 ≤ s.re → 4 ≤ ‖s‖ →
      ‖riemannXiLi s‖ ≤
        Real.exp (C * ‖s‖ * Real.log (‖s‖ + 2)) := by
  obtain ⟨Cf, hCf0, hfar⟩ := exists_riemannXiLi_orderBound_farRight
  obtain ⟨Cs, hCs0, hstrip⟩ :=
    exists_norm_riemannXiLi_le_on_verticalStrip (1 / 2) 4
  obtain ⟨D, hD0, hquad⟩ :=
    exists_exp_orderBound_of_quadratic Cs hCs0
  let C := max Cf D
  refine ⟨C, hCf0.trans (le_max_left _ _), ?_⟩
  intro s hre hr
  let A := ‖s‖ * Real.log (‖s‖ + 2)
  have hA0 : 0 ≤ A := mul_nonneg (norm_nonneg _)
    (Real.log_nonneg (by linarith [norm_nonneg s]))
  by_cases hs4 : 4 ≤ s.re
  · exact (hfar s hs4).trans (Real.exp_le_exp.mpr <| by
      calc
        Cf * ‖s‖ * Real.log (‖s‖ + 2) = Cf * A := by
          dsimp [A]
          ring
        _ ≤ C * A := mul_le_mul_of_nonneg_right (le_max_left _ _) hA0
        _ = C * ‖s‖ * Real.log (‖s‖ + 2) := by
          dsimp [A]
          ring)
  · have hraw := hstrip s hre (le_of_not_ge hs4)
    have hnormsub : ‖s - 1‖ ≤ ‖s‖ + 1 := by
      simpa using norm_sub_le s 1
    have hadjust :
        1 + ‖s‖ * ‖s - 1‖ * Cs ≤
          1 + ‖s‖ * (‖s‖ + 1) * Cs := by gcongr
    exact hraw.trans <| hadjust.trans <| (hquad ‖s‖ hr).trans <|
      Real.exp_le_exp.mpr (by
        calc
          D * ‖s‖ * Real.log (‖s‖ + 2) = D * A := by
            dsimp [A]
            ring
          _ ≤ C * A := mul_le_mul_of_nonneg_right (le_max_right _ _) hA0
          _ = C * ‖s‖ * Real.log (‖s‖ + 2) := by
            dsimp [A]
            ring)

/-- The normalized xi function satisfies the source-standard global
order-at-most-one growth bound. -/
theorem riemannXi_orderOneGrowthBound :
    RiemannXiOrderOneGrowthBound := by
  obtain ⟨C, hC0, hright⟩ :=
    exists_riemannXiLi_orderBound_rightHalfPlane
  refine ⟨3 * C, 5, by positivity, by norm_num, ?_⟩
  intro s hs
  let r := ‖s‖
  let A := r * Real.log (r + 2)
  have hr : 5 ≤ r := hs
  have hlog0 : 0 ≤ Real.log (r + 2) :=
    Real.log_nonneg (by linarith)
  have hA0 : 0 ≤ A := mul_nonneg (norm_nonneg _) hlog0
  by_cases hre : 1 / 2 ≤ s.re
  · exact (hright s hre (by linarith)).trans (Real.exp_le_exp.mpr <| by
      have hbase : 0 ≤ C * ‖s‖ * Real.log (‖s‖ + 2) :=
        mul_nonneg (mul_nonneg hC0 (norm_nonneg s))
          (Real.log_nonneg (by linarith [norm_nonneg s]))
      nlinarith)
  · let t := 1 - s
    have htRe : 1 / 2 ≤ t.re := by
      change 1 / 2 ≤ (1 - s).re
      simp only [sub_re, one_re]
      linarith
    have htrLower : r - 1 ≤ ‖t‖ := by
      have h := norm_sub_le t 1
      have hts : t - 1 = -s := by dsimp [t]; ring
      rw [hts, norm_neg] at h
      norm_num at h
      linarith
    have ht4 : 4 ≤ ‖t‖ := by linarith
    have htUpper : ‖t‖ ≤ r + 1 := by
      dsimp [t, r]
      simpa [norm_neg, sub_eq_add_neg, add_comm] using norm_sub_le 1 s
    have htpos : 0 < ‖t‖ + 2 := by positivity
    have harg : ‖t‖ + 2 ≤ (r + 2) ^ 2 := by
      nlinarith [sq_nonneg (r + 2)]
    have hlogt : Real.log (‖t‖ + 2) ≤ 2 * Real.log (r + 2) := by
      calc
        Real.log (‖t‖ + 2) ≤ Real.log ((r + 2) ^ 2) :=
          Real.log_le_log htpos harg
        _ = 2 * Real.log (r + 2) := by
          rw [Real.log_pow]
          norm_num
    have htA :
        ‖t‖ * Real.log (‖t‖ + 2) ≤ 3 * A := by
      have hlogt0 : 0 ≤ Real.log (‖t‖ + 2) :=
        Real.log_nonneg (by linarith [norm_nonneg t])
      calc
        ‖t‖ * Real.log (‖t‖ + 2)
            ≤ (r + 1) * (2 * Real.log (r + 2)) := by gcongr
        _ ≤ 3 * A := by
          dsimp [A]
          nlinarith
    rw [← riemannXiLi_one_sub s]
    exact (hright t htRe ht4).trans (Real.exp_le_exp.mpr <| by
      calc
        C * ‖t‖ * Real.log (‖t‖ + 2)
            = C * (‖t‖ * Real.log (‖t‖ + 2)) := by ring
        _ ≤ C * (3 * A) := mul_le_mul_of_nonneg_left htA hC0
        _ = 3 * C * r * Real.log (r + 2) := by
          dsimp [A]
          ring)

/-- Jensen's formula and the global order-one bound give the expected
`O(r log r)` logarithmic zero count, with analytic multiplicities.  This
is weaker than the `O(log r)` count in a unit-height shell supplied by
Riemann--von Mangoldt, but is sufficient after dyadic grouping. -/
theorem exists_riemannXiZeroDivisor_logCounting_le_orderOne :
    ∃ C R : ℝ, 0 ≤ C ∧ 2 ≤ R ∧ ∀ r : ℝ, R ≤ r →
      Function.locallyFinsuppWithin.logCounting
          riemannXiZeroDivisor r ≤
        C * r * Real.log (r + 2) := by
  obtain ⟨C, R, hC, hR, hbound⟩ :=
    riemannXi_orderOneGrowthBound
  refine ⟨C, R, hC, hR, ?_⟩
  intro r hr
  rw [riemannXiZeroDivisor_logCounting_eq_circleAverage (by linarith)]
  apply Real.circleAverage_mono_on_of_le_circle
  · have hm : MeromorphicOn riemannXiLi
        (Metric.sphere 0 |r|) :=
      fun z _ ↦
        (differentiable_riemannXiLi.analyticAt z).meromorphicAt
    exact hm.circleIntegrable_log_norm
  · intro z hz
    have hr0 : 0 ≤ r := by linarith
    have hzr : ‖z‖ = r := by
      simpa [Metric.mem_sphere, dist_zero_right,
        abs_of_nonneg hr0] using hz
    have hnorm := hbound z (by rw [hzr]; exact hr)
    rw [hzr] at hnorm
    by_cases hzero : riemannXiLi z = 0
    · simp [hzero]
      exact mul_nonneg (mul_nonneg hC (by linarith))
        (Real.log_nonneg (by linarith))
    · exact
        (Real.log_le_iff_le_exp (norm_pos_iff.mpr hzero)).mpr hnorm

/-- Jensen's weighted divisor estimate gives an explicit cumulative bound
for the canonical multiplicity index.  The factor `1 / log 2` is the exact
cost of comparing the radius-`r` window with the logarithmic count at
radius `2r`. -/
theorem exists_riemannXiZeroIndexWindow_card_le_orderOne :
    ∃ A R : ℝ, 0 ≤ A ∧ 1 ≤ R ∧ ∀ r : ℝ, R ≤ r →
      ((riemannXiZeroIndexWindow r).card : ℝ) ≤
        A * r * Real.log (2 * r + 2) := by
  obtain ⟨C, R, hC, hR, hcount⟩ :=
    exists_riemannXiZeroDivisor_logCounting_le_orderOne
  let R' := max 1 (R / 2)
  let A := 2 * C / Real.log 2
  refine ⟨A, R', ?_, le_max_left _ _, ?_⟩
  · exact div_nonneg (mul_nonneg (by norm_num) hC)
      (Real.log_nonneg (by norm_num))
  · intro r hr
    have hr1 : 1 ≤ r := (le_max_left _ _).trans hr
    have hR2r : R ≤ 2 * r := by
      have := (le_max_right 1 (R / 2)).trans hr
      linarith
    have hweighted :=
      riemannXiZeroIndexWindow_card_mul_log_two_le_logCounting hr1
    have hupper := hcount (2 * r) hR2r
    have hlog2 : 0 < Real.log 2 := Real.log_pos (by norm_num)
    dsimp [A]
    rw [show 2 * C / Real.log 2 * r * Real.log (2 * r + 2) =
        (C * (2 * r) * Real.log (2 * r + 2)) / Real.log 2 by
      field_simp]
    exact (le_div_iff₀ hlog2).mpr (hweighted.trans hupper)

/-- Actual pinned-Mathlib finite canonical decomposition instantiated for xi
on every disk.  This is a finite Blaschke-style decomposition, not the
missing whole-plane genus-one Hadamard factorization. -/
theorem riemannXiLi_exists_canonicalDecomp (R : ℝ) :
    ∃ g : ℂ → ℂ, Complex.CanonicalDecomp riemannXiLi g R := by
  have hxi : AnalyticOnNhd ℂ riemannXiLi Set.univ :=
    fun z _ ↦ differentiable_riemannXiLi.analyticAt z
  apply MeromorphicOn.exists_canonicalDecomp
  · exact fun z hz ↦ hxi.meromorphicOn z (Set.mem_univ z)
  · intro u
    rw [(differentiable_riemannXiLi.analyticAt u).meromorphicOrderAt_eq]
    simpa [riemannXiZeroOrder] using riemannXiZeroOrder_ne_top u

/-- A radius-cofinal family of Mathlib finite canonical decompositions.
No compatibility or normal convergence between the chosen remainders is
asserted. -/
structure RiemannXiCanonicalDiskExhaustion where
  remainder : ℕ → ℂ → ℂ
  decomp : ∀ N,
    Complex.CanonicalDecomp riemannXiLi (remainder N) (N + 1)

theorem exists_riemannXiCanonicalDiskExhaustion :
    Nonempty RiemannXiCanonicalDiskExhaustion := by
  choose g hg using fun N : ℕ ↦
    riemannXiLi_exists_canonicalDecomp (N + 1)
  exact ⟨⟨g, hg⟩⟩

/-- In the positive-real half-plane away from `0,1`, xi and zeta have the
same zeros. -/
theorem riemannXiLi_eq_zero_iff_riemannZeta_eq_zero {rho : ℂ}
    (h0 : rho ≠ 0) (h1 : rho ≠ 1) (hre : 0 < rho.re) :
    riemannXiLi rho = 0 ↔ riemannZeta rho = 0 := by
  rw [riemannXiLi_eq_mul_completedRiemannZeta h0 h1,
    riemannZeta_def_of_ne_zero h0]
  have hgamma : Gammaℝ rho ≠ 0 := Gammaℝ_ne_zero_of_re_pos hre
  constructor
  · intro h
    have hcomp : completedRiemannZeta rho = 0 := by
      rcases mul_eq_zero.mp h with hpre | hcomp
      · exact False.elim ((mul_ne_zero h0 (sub_ne_zero.mpr h1)) hpre)
      · exact hcomp
    simp [hcomp]
  · intro h
    have hcomp : completedRiemannZeta rho = 0 :=
      (div_eq_zero_iff.mp h).resolve_right hgamma
    simp [hcomp]

/-- Every xi zero lies in the open critical strip.  The right boundary uses
Mathlib's zeta nonvanishing theorem on `re s ≥ 1`; the left boundary follows
by the xi functional equation. -/
theorem riemannXiLi_zero_re_lt_one {rho : ℂ}
    (hzero : riemannXiLi rho = 0) : rho.re < 1 := by
  by_contra h
  have hre : 1 ≤ rho.re := le_of_not_gt h
  have h0 : rho ≠ 0 := by
    intro hr
    subst rho
    simp at hzero
  have h1 : rho ≠ 1 := by
    intro hr
    subst rho
    simp at hzero
  have hzeta : riemannZeta rho = 0 :=
    (riemannXiLi_eq_zero_iff_riemannZeta_eq_zero h0 h1
      (lt_of_lt_of_le (by norm_num) hre)).mp hzero
  exact riemannZeta_ne_zero_of_one_le_re hre hzeta

theorem riemannXiLi_zero_re_pos {rho : ℂ}
    (hzero : riemannXiLi rho = 0) : 0 < rho.re := by
  by_contra h
  have hre : rho.re ≤ 0 := le_of_not_gt h
  have hrefzero : riemannXiLi (1 - rho) = 0 := by
    rw [riemannXiLi_one_sub]
    exact hzero
  have href := riemannXiLi_zero_re_lt_one hrefzero
  simp only [sub_re, one_re] at href
  linarith

theorem riemannXiZeroRoot_mem_openCriticalStrip
    (i : RiemannXiZeroIndex) :
    0 < (riemannXiZeroRoot i).re ∧
      (riemannXiZeroRoot i).re < 1 :=
  ⟨riemannXiLi_zero_re_pos i.1.property,
    riemannXiLi_zero_re_lt_one i.1.property⟩

theorem riemannXiZeroRoot_norm_le_one_add_abs_im
    (i : RiemannXiZeroIndex) :
    ‖riemannXiZeroRoot i‖ ≤
      1 + |(riemannXiZeroRoot i).im| := by
  calc
    ‖riemannXiZeroRoot i‖ ≤
        |(riemannXiZeroRoot i).re| +
          |(riemannXiZeroRoot i).im| :=
      Complex.norm_le_abs_re_add_abs_im _
    _ ≤ 1 + |(riemannXiZeroRoot i).im| := by
      gcongr
      rw [abs_of_pos (riemannXiZeroRoot_mem_openCriticalStrip i).1]
      exact (riemannXiZeroRoot_mem_openCriticalStrip i).2.le

/-- Finite xi windows cut out by imaginary height.  The critical-strip bound
embeds this set in a finite radial window. -/
noncomputable def riemannXiZeroHeightWindow
    (T : ℝ) : Finset RiemannXiZeroIndex :=
  (riemannXiZeroIndexWindow (|T| + 1)).filter
    fun i ↦ |(riemannXiZeroRoot i).im| ≤ T

@[simp] theorem mem_riemannXiZeroHeightWindow
    (T : ℝ) (i : RiemannXiZeroIndex) :
    i ∈ riemannXiZeroHeightWindow T ↔
      |(riemannXiZeroRoot i).im| ≤ T := by
  constructor
  · intro hi
    exact (Finset.mem_filter.mp hi).2
  · intro hi
    apply Finset.mem_filter.mpr
    refine ⟨?_, hi⟩
    rw [mem_riemannXiZeroIndexWindow]
    have hT : 0 ≤ T := (abs_nonneg _).trans hi
    calc
      ‖riemannXiZeroRoot i‖ ≤
          1 + |(riemannXiZeroRoot i).im| :=
        riemannXiZeroRoot_norm_le_one_add_abs_im i
      _ ≤ 1 + T := by linarith
      _ = |T| + 1 := by rw [abs_of_nonneg hT]; ring

theorem mem_riemannXiZeroHeightWindow_reflection
    (T : ℝ) (i : RiemannXiZeroIndex) :
    riemannXiZeroReflection i ∈ riemannXiZeroHeightWindow T ↔
      i ∈ riemannXiZeroHeightWindow T := by
  simp only [mem_riemannXiZeroHeightWindow,
    riemannXiZeroRoot_reflection, sub_im, one_im, zero_sub, abs_neg]

/-- Reflection permutes each finite height window exactly, including every
multiplicity occurrence. -/
theorem riemannXiZeroHeightWindow_map_reflection (T : ℝ) :
    (riemannXiZeroHeightWindow T).map
        riemannXiZeroReflectionEmbedding =
      riemannXiZeroHeightWindow T := by
  apply Finset.eq_of_subset_of_card_le
  · intro j hj
    simp only [Finset.mem_map] at hj
    obtain ⟨i, hi, rfl⟩ := hj
    exact (mem_riemannXiZeroHeightWindow_reflection T i).mpr hi
  · simp [riemannXiZeroReflectionEmbedding]

/-- Exact finite paired inverse-root identity on every height window. -/
theorem two_mul_sum_inv_riemannXiZeroHeightWindow (T : ℝ) :
    2 * (∑ i ∈ riemannXiZeroHeightWindow T,
      (riemannXiZeroRoot i)⁻¹) =
      ∑ i ∈ riemannXiZeroHeightWindow T,
        (riemannXiZeroRoot i *
          (1 - riemannXiZeroRoot i))⁻¹ := by
  have href :
      (∑ i ∈ riemannXiZeroHeightWindow T,
        (riemannXiZeroRoot (riemannXiZeroReflection i))⁻¹) =
      ∑ i ∈ riemannXiZeroHeightWindow T,
        (riemannXiZeroRoot i)⁻¹ := by
    calc
      _ = ∑ j ∈ (riemannXiZeroHeightWindow T).map
          riemannXiZeroReflectionEmbedding,
          (riemannXiZeroRoot j)⁻¹ := by
            rw [Finset.sum_map]
            rfl
      _ = _ := by rw [riemannXiZeroHeightWindow_map_reflection]
  rw [show (2 : ℂ) * (∑ i ∈ riemannXiZeroHeightWindow T,
      (riemannXiZeroRoot i)⁻¹) =
      (∑ i ∈ riemannXiZeroHeightWindow T,
        (riemannXiZeroRoot i)⁻¹) +
      ∑ i ∈ riemannXiZeroHeightWindow T,
        (riemannXiZeroRoot (riemannXiZeroReflection i))⁻¹ by
          rw [href]; ring]
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro i hi
  exact riemannXiZeroRoot_inv_add_reflection i

theorem riemannXiZeroIndexWindow_subset_heightWindow (R : ℝ) :
    riemannXiZeroIndexWindow R ⊆ riemannXiZeroHeightWindow R := by
  intro i hi
  rw [mem_riemannXiZeroHeightWindow]
  exact (abs_im_le_norm _).trans
    ((mem_riemannXiZeroIndexWindow R i).mp hi)

theorem riemannXiZeroHeightWindow_subset_indexWindow (T : ℝ) :
    riemannXiZeroHeightWindow T ⊆
      riemannXiZeroIndexWindow (T + 1) := by
  intro i hi
  rw [mem_riemannXiZeroIndexWindow]
  exact (riemannXiZeroRoot_norm_le_one_add_abs_im i).trans
    (by linarith [(mem_riemannXiZeroHeightWindow T i).mp hi])

theorem riemannXiZeroHeightWindow_nat_tendsto_atTop :
    Tendsto (fun N : ℕ ↦ riemannXiZeroHeightWindow (N : ℝ))
      atTop atTop := by
  apply tendsto_atTop.2
  intro F
  let T : ℝ := ∑ i ∈ F, |(riemannXiZeroRoot i).im|
  obtain ⟨n, hn⟩ := exists_nat_ge T
  filter_upwards [eventually_ge_atTop n] with N hN
  intro i hi
  rw [mem_riemannXiZeroHeightWindow]
  calc
    |(riemannXiZeroRoot i).im| ≤ T := by
      dsimp [T]
      exact Finset.single_le_sum
        (s := F) (f := fun j ↦ |(riemannXiZeroRoot j).im|)
        (fun j _ ↦ abs_nonneg _) hi
    _ ≤ n := hn
    _ ≤ N := by exact_mod_cast hN

theorem analyticAt_log_riemannXiLi_one :
    AnalyticAt ℂ (fun s ↦ Complex.log (riemannXiLi s)) 1 :=
  (differentiable_riemannXiLi.analyticAt 1).clog (by simp)

/-- The source-standard derivative expression for the `n`th Li coefficient.

The standard sequence starts at `n = 1`.  We totalize it at `0` by setting
that value to zero.  For positive `n` this is
`1 / (n - 1)! * d^n/ds^n (s^(n-1) log(xi(s)))` at `s = 1`.
-/
def liDerivativeCoefficient (xi : ℂ → ℂ) (n : ℕ) : ℂ :=
  if n = 0 then 0
  else
    iteratedDeriv n (fun s ↦ s ^ (n - 1) * Complex.log (xi s)) 1 /
      ((n - 1).factorial : ℂ)

/-- The concrete real sequence extracted from the derivative formula.

Taking the real part makes the target type appropriate for positivity.
The obligation that the complex derivative is actually real is kept explicit
below rather than silently assumed.
-/
def riemannLiCoefficient (n : ℕ) : ℝ :=
  (liDerivativeCoefficient riemannXiLi n).re

@[simp] theorem liDerivativeCoefficient_zero (xi : ℂ → ℂ) :
    liDerivativeCoefficient xi 0 = 0 := by
  simp [liDerivativeCoefficient]

@[simp] theorem riemannLiCoefficient_zero : riemannLiCoefficient 0 = 0 := by
  simp [riemannLiCoefficient]

theorem liDerivativeCoefficient_one (xi : ℂ → ℂ) :
    liDerivativeCoefficient xi 1 =
      deriv (fun s ↦ Complex.log (xi s)) 1 := by
  simp [liDerivativeCoefficient]

/-- The Möbius change of variables whose logarithmic derivative generates
Li coefficients. -/
def liMobiusArgument (z : ℂ) : ℂ := (1 - z)⁻¹

def liChangeOfVariables (xi : ℂ → ℂ) (z : ℂ) : ℂ :=
  xi ((1 - z)⁻¹)

/-- Closed all-order derivative formula for the Möbius map used in Li's
change of variables, valid at every point (with totalized inverse powers). -/
theorem iteratedDeriv_one_sub_inv (k : ℕ) (z : ℂ) :
    iteratedDeriv k (fun w : ℂ ↦ (1 - w)⁻¹) z =
      (k.factorial : ℂ) * (1 - z) ^ (-(k + 1 : ℤ)) := by
  have h : (fun w : ℂ ↦ (1 - w)⁻¹) =
      fun w ↦ ((-1 : ℂ) * w + 1)⁻¹ := by
    funext w
    congr 1
    ring
  rw [h, iteratedDeriv_eq_iterate, iter_deriv_inv_linear]
  rw [neg_pow]
  ring_nf
  simp [show k * 2 = 2 * k by omega, pow_mul]

@[simp] theorem iteratedDeriv_one_sub_inv_zero (k : ℕ) :
    iteratedDeriv k (fun z : ℂ ↦ (1 - z)⁻¹) 0 =
      (k.factorial : ℂ) := by
  rw [iteratedDeriv_one_sub_inv]
  simp

/-- Scalar iterated derivatives of a complex analytic germ remain
differentiable.  This bridges Mathlib's analytic `iteratedFDeriv` API to the
one-variable recurrence needed by the Li composition formula. -/
theorem AnalyticAt.differentiableAt_iteratedDeriv
    {L : ℂ → ℂ} {x : ℂ} (hL : AnalyticAt ℂ L x) (m : ℕ) :
    DifferentiableAt ℂ (iteratedDeriv m L) x := by
  have hF : AnalyticAt ℂ (iteratedFDeriv ℂ m L) x := by
    have hN : AnalyticOnNhd ℂ L {x} := fun y hy ↦ by
      simp only [Set.mem_singleton_iff] at hy
      subst y
      exact hL
    exact hN.iteratedFDeriv m x (by simp)
  rw [iteratedDeriv_eq_equiv_comp]
  fun_prop

/-- Exact product/chain-rule step for one term in the specialized Möbius
Faà di Bruno sum.  The first term raises the Möbius power by one; the
second raises it by two and advances the outer derivative. -/
theorem hasDerivAt_mobiusFaaDiBrunoSummand
    {L : ℂ → ℂ} {z : ℂ} (hz : z ≠ 1)
    (hL : AnalyticAt ℂ L ((1 - z)⁻¹)) (p m : ℕ) :
    HasDerivAt
      (fun w ↦ (1 - w)⁻¹ ^ p *
        iteratedDeriv m L ((1 - w)⁻¹))
      (p * ((1 - z)⁻¹ ^ (p + 1) *
          iteratedDeriv m L ((1 - z)⁻¹)) +
        (1 - z)⁻¹ ^ (p + 2) *
          iteratedDeriv (m + 1) L ((1 - z)⁻¹)) z := by
  let φ : ℂ → ℂ := fun w ↦ (1 - w)⁻¹
  change HasDerivAt
    (fun w ↦ φ w ^ p * iteratedDeriv m L (φ w))
    (p * (φ z ^ (p + 1) * iteratedDeriv m L (φ z)) +
      φ z ^ (p + 2) * iteratedDeriv (m + 1) L (φ z)) z
  have hφ : HasDerivAt φ (φ z ^ 2) z := by
    have hbase : HasDerivAt (fun w : ℂ ↦ 1 - w) (-1) z := by
      simpa [sub_eq_add_neg] using (hasDerivAt_id z).neg.const_add 1
    have hne : 1 - z ≠ 0 := sub_ne_zero.mpr hz.symm
    dsimp only [φ]
    exact (hbase.fun_inv hne).congr_deriv (by
      rw [pow_two]
      field_simp)
  have hm : HasDerivAt (iteratedDeriv m L)
      (iteratedDeriv (m + 1) L (φ z)) (φ z) := by
    rw [iteratedDeriv_succ]
    exact (hL.differentiableAt_iteratedDeriv m).hasDerivAt
  have h := (hφ.pow p).mul (hm.comp z hφ)
  change HasDerivAt
    (fun w ↦ φ w ^ p * iteratedDeriv m L (φ w))
    (↑p * φ z ^ (p - 1) * φ z ^ 2 * iteratedDeriv m L (φ z) +
      φ z ^ p * (iteratedDeriv (m + 1) L (φ z) * φ z ^ 2)) z at h
  exact h.congr_deriv (by
    cases p with
    | zero => simp; ring
    | succ p =>
      simp only [Nat.cast_add, Nat.cast_one, Nat.succ_sub_one, pow_succ]
      ring)

/-- Differentiating the complete finite specialized sum is termwise valid.
The two displayed summands are exactly the unshifted chain contribution and
the shifted product contribution that must be reindexed in the triangular
recurrence. -/
theorem hasDerivAt_mobiusFaaDiBrunoSum
    {L : ℂ → ℂ} {z : ℂ} (hz : z ≠ 1)
    (hL : AnalyticAt ℂ L ((1 - z)⁻¹)) (c : ℕ → ℂ) (k : ℕ) :
    HasDerivAt
      (∑ i ∈ Finset.range (k + 2), fun w ↦
        c i * ((1 - w)⁻¹ ^ (2 * k + 2 - i) *
          iteratedDeriv (k + 1 - i) L ((1 - w)⁻¹)))
      (∑ i ∈ Finset.range (k + 2),
        c i * (((2 * k + 2 - i : ℕ) : ℂ) *
            ((1 - z)⁻¹ ^ (2 * k + 3 - i) *
              iteratedDeriv (k + 1 - i) L ((1 - z)⁻¹)) +
          (1 - z)⁻¹ ^ (2 * k + 4 - i) *
            iteratedDeriv (k + 2 - i) L ((1 - z)⁻¹))) z := by
  apply HasDerivAt.sum
  intro i hi
  have hi' : i < k + 2 := Finset.mem_range.mp hi
  have hterm := (hasDerivAt_mobiusFaaDiBrunoSummand hz hL
    (2 * k + 2 - i) (k + 1 - i)).const_mul (c i)
  apply hterm.congr_deriv
  congr 4 <;> grind

/-- Exact finite index shift for the two differentiated contributions.
The support assumption removes both artificial top-boundary terms.  The
right side displays the triangular coefficient update before specializing
to the closed Möbius coefficients. -/
theorem mobiusFaaDiBrunoSum_reindex
    (k : ℕ) (c : ℕ → ℂ) (B : ℕ → ℂ)
    (hc : ∀ i, k < i → c i = 0) :
    (∑ i ∈ Finset.range (k + 2),
        c i * (((2 * k + 2 - i : ℕ) : ℂ) * B (i + 1) + B i)) =
      ∑ j ∈ Finset.range (k + 3),
        (c j + if j = 0 then 0 else
          ((2 * k + 3 - j : ℕ) : ℂ) * c (j - 1)) * B j := by
  have hfirst :
      (∑ i ∈ Finset.range (k + 2),
        c i * (((2 * k + 2 - i : ℕ) : ℂ) * B (i + 1))) =
      ∑ i ∈ Finset.range (k + 2),
        (((2 * k + 2 - i : ℕ) : ℂ) * c i) * B (i + 1) := by
    apply Finset.sum_congr rfl
    intro i hi
    ring
  have hsecond :
      (∑ i ∈ Finset.range (k + 2), c i * B i) =
        (∑ i ∈ Finset.range (k + 2), c (i + 1) * B (i + 1)) +
          c 0 * B 0 := by
    rw [Finset.sum_range_succ']
    congr 1
    symm
    rw [Finset.sum_range_succ]
    simp [hc (k + 2) (by omega)]
  have hq :
      (∑ i ∈ Finset.range (k + 2),
        (((2 * k + 2 - i : ℕ) : ℂ) * c i) * B (i + 1)) =
      ∑ i ∈ Finset.range (k + 2),
        (if i + 1 = 0 then 0 else
          ((2 * k + 3 - (i + 1) : ℕ) : ℂ) * c (i + 1 - 1)) *
            B (i + 1) := by
    apply Finset.sum_congr rfl
    intro i hi
    have hi' : i < k + 2 := Finset.mem_range.mp hi
    simp only [Nat.add_eq_zero_iff, one_ne_zero, and_false, if_false,
      Nat.add_sub_cancel]
    have he : 2 * k + 3 - (i + 1) = 2 * k + 2 - i := by grind
    rw [he]
  conv_rhs => rw [Finset.sum_range_succ']
  simp_rw [add_mul]
  rw [Finset.sum_add_distrib]
  simp_rw [mul_add]
  rw [Finset.sum_add_distrib, hfirst, hsecond]
  simp only [Nat.add_one]
  rw [hq]
  simp
  ring

/-- The exact all-order xi coefficient obligation.  Its first coefficient is
proved below; higher orders require the source-standard Möbius
change-of-variables/Faà di Bruno identity. -/
def LiChangeOfVariablesCoefficientIdentity (xi : ℂ → ℂ) : Prop :=
  ∀ k : ℕ,
    iteratedDeriv k (logDeriv (liChangeOfVariables xi)) 0 /
      (k.factorial : ℂ) = liDerivativeCoefficient xi (k + 1)

/-- Exact all-order Leibniz expansion of Li's derivative definition.  This
removes all ambiguity from the `s^(n-1) log xi(s)` side of the remaining
Möbius/Faà di Bruno identity. -/
theorem liDerivativeCoefficient_eq_leibnizSum
    (xi : ℂ → ℂ) (n : ℕ) (hn : 0 < n)
    (hL : ContDiffAt ℂ n (fun s ↦ Complex.log (xi s)) 1) :
    liDerivativeCoefficient xi n =
      (∑ i ∈ Finset.range (n + 1),
        (n.choose i : ℂ) *
          ((n - 1).descFactorial i : ℂ) *
          iteratedDeriv (n - i) (fun s ↦ Complex.log (xi s)) 1) /
        ((n - 1).factorial : ℂ) := by
  rw [liDerivativeCoefficient, if_neg hn.ne']
  change iteratedDeriv n
      ((fun s : ℂ ↦ s ^ (n - 1)) *
        (fun s ↦ Complex.log (xi s))) 1 /
      ((n - 1).factorial : ℂ) = _
  rw [iteratedDeriv_mul (by fun_prop) hL]
  simp_rw [iteratedDeriv_pow]
  simp only [one_pow, mul_one]

/-- The remaining pure all-order composition formula, displayed as an
explicit finite descending-factorial sum rather than hidden behind the Li
coefficient definition. -/
def MobiusFaaDiBrunoCoefficientIdentity (xi : ℂ → ℂ) : Prop :=
  ∀ k : ℕ,
    iteratedDeriv k (logDeriv (liChangeOfVariables xi)) 0 /
        (k.factorial : ℂ) =
      (∑ i ∈ Finset.range (k + 2),
        ((k + 1).choose i : ℂ) *
          (k.descFactorial i : ℂ) *
          iteratedDeriv (k + 1 - i)
            (fun s ↦ Complex.log (xi s)) 1) /
        (k.factorial : ℂ)

/-- Closed specialized Faà di Bruno coefficient for the Möbius map. -/
def mobiusFaaDiBrunoCoefficient (k i : ℕ) : ℕ :=
  (k + 1).choose i * k.descFactorial i

/-- Pascal's identity and the adjacent-binomial identity give the exact
two-term recurrence produced by differentiating the specialized composition.
The `i=0` branch records the boundary term explicitly. -/
theorem mobiusFaaDiBrunoCoefficient_succ (k i : ℕ) (hi : i ≤ k + 1) :
    mobiusFaaDiBrunoCoefficient (k + 1) i =
      mobiusFaaDiBrunoCoefficient k i +
        (if i = 0 then 0 else
          (2 * k + 3 - i) * mobiusFaaDiBrunoCoefficient k (i - 1)) := by
  unfold mobiusFaaDiBrunoCoefficient
  cases i with
  | zero => simp
  | succ i =>
    simp only [Nat.succ_ne_zero, ↓reduceIte, Nat.succ_sub_one]
    rw [Nat.choose, Nat.succ_descFactorial_succ,
      Nat.descFactorial_succ]
    have hi' : i ≤ k := by omega
    have hadj := Nat.choose_succ_right_eq (k + 1) i
    have hsub1 : k + 1 - i = k - i + 1 := by omega
    have hsub2 : 2 * k + 3 - (i + 1) = 2 * k + 2 - i := by omega
    rw [hsub1] at hadj
    rw [hsub2]
    let A := (k + 1).choose i
    let B := (k + 1).choose (i + 1)
    let D := k.descFactorial i
    change (A + B) * ((k + 1) * D) =
      B * ((k - i) * D) + (2 * k + 2 - i) * (A * D)
    change B * (i + 1) = A * (k - i + 1) at hadj
    have hk : k + 1 = (k - i) + (i + 1) := by omega
    have hc : 2 * k + 2 - i = (k + 1) + (k - i + 1) := by omega
    rw [hc, hk]
    have hadjD := congrArg (fun x : ℕ ↦ x * D) hadj
    ring_nf at hadjD ⊢
    omega

/-- Specializing the generic index shift to the closed Möbius coefficients
collects the differentiated sum into the next row exactly.  The apparent
top index is zero by falling-factorial support. -/
theorem mobiusFaaDiBrunoCoefficientSum_reindex
    (k : ℕ) (B : ℕ → ℂ) :
    (∑ i ∈ Finset.range (k + 2),
      (mobiusFaaDiBrunoCoefficient k i : ℂ) *
        (((2 * k + 2 - i : ℕ) : ℂ) * B (i + 1) + B i)) =
      ∑ j ∈ Finset.range (k + 3),
        (mobiusFaaDiBrunoCoefficient (k + 1) j : ℂ) * B j := by
  rw [mobiusFaaDiBrunoSum_reindex k
    (fun i ↦ (mobiusFaaDiBrunoCoefficient k i : ℂ)) B
    (by
      intro i hi
      simp [mobiusFaaDiBrunoCoefficient,
        Nat.descFactorial_eq_zero_iff_lt.mpr hi])]
  apply Finset.sum_congr rfl
  intro j hj
  have hjlt : j < k + 3 := Finset.mem_range.mp hj
  by_cases hjle : j ≤ k + 1
  · have hr := mobiusFaaDiBrunoCoefficient_succ k j hjle
    congr 1
    exact_mod_cast hr.symm
  · have hj : j = k + 2 := by omega
    subst j
    simp [mobiusFaaDiBrunoCoefficient]

/-- The `k`th specialized Möbius composition row as an actual function on
the local analytic domain. -/
def mobiusFaaDiBrunoRow (L : ℂ → ℂ) (k : ℕ) : ℂ → ℂ :=
  ∑ i ∈ Finset.range (k + 2), fun z ↦
    (mobiusFaaDiBrunoCoefficient k i : ℂ) *
      (liMobiusArgument z ^ (2 * k + 2 - i) *
        iteratedDeriv (k + 1 - i) L (liMobiusArgument z))

/-- Local analytic induction for every specialized Möbius/Faà di Bruno row.
Analyticity is pulled back to a neighborhood at each step, so the induction
hypothesis may be differentiated rather than used only pointwise. -/
theorem iteratedDeriv_mobiusComposition_eq_row
    {L : ℂ → ℂ} {z : ℂ} (hz : z ≠ 1)
    (hL : AnalyticAt ℂ L (liMobiusArgument z)) (k : ℕ) :
    iteratedDeriv k
      (fun w ↦ liMobiusArgument w ^ 2 *
        iteratedDeriv 1 L (liMobiusArgument w)) z =
      mobiusFaaDiBrunoRow L k z := by
  induction k generalizing z with
  | zero =>
      simp [mobiusFaaDiBrunoRow, mobiusFaaDiBrunoCoefficient,
        Finset.sum_range_succ, iteratedDeriv_one]
  | succ k ih =>
      have hφc : ContinuousAt liMobiusArgument z := by
        unfold liMobiusArgument
        exact (continuousAt_const.sub continuousAt_id).inv₀
          (sub_ne_zero.mpr hz.symm)
      have hA : ∀ᶠ w in 𝓝 z,
          AnalyticAt ℂ L (liMobiusArgument w) :=
        hφc.tendsto.eventually hL.eventually_analyticAt
      have hne : ∀ᶠ w in 𝓝 z, w ≠ 1 := eventually_ne_nhds hz
      have heq : iteratedDeriv k
          (fun w ↦ liMobiusArgument w ^ 2 *
            iteratedDeriv 1 L (liMobiusArgument w)) =ᶠ[𝓝 z]
          mobiusFaaDiBrunoRow L k := by
        filter_upwards [hne, hA] with w hw hAw
        exact ih hw hAw
      rw [iteratedDeriv_succ, heq.deriv_eq]
      have hd := hasDerivAt_mobiusFaaDiBrunoSum hz hL
        (fun i ↦ (mobiusFaaDiBrunoCoefficient k i : ℂ)) k
      change HasDerivAt (mobiusFaaDiBrunoRow L k)
        (∑ i ∈ Finset.range (k + 2),
          (mobiusFaaDiBrunoCoefficient k i : ℂ) *
            (((2 * k + 2 - i : ℕ) : ℂ) *
                (liMobiusArgument z ^ (2 * k + 3 - i) *
                  iteratedDeriv (k + 1 - i) L (liMobiusArgument z)) +
              liMobiusArgument z ^ (2 * k + 4 - i) *
                iteratedDeriv (k + 2 - i) L
                  (liMobiusArgument z))) z at hd
      change deriv (mobiusFaaDiBrunoRow L k) z =
        mobiusFaaDiBrunoRow L (k + 1) z
      rw [hd.deriv]
      let B : ℕ → ℂ := fun j ↦
        liMobiusArgument z ^ (2 * k + 4 - j) *
          iteratedDeriv (k + 2 - j) L (liMobiusArgument z)
      rw [show mobiusFaaDiBrunoRow L (k + 1) z =
        ∑ j ∈ Finset.range (k + 3),
          (mobiusFaaDiBrunoCoefficient (k + 1) j : ℂ) * B j by
            unfold mobiusFaaDiBrunoRow
            simp only [Finset.sum_apply]
            apply Finset.sum_congr rfl
            intro j hj
            dsimp only [B]
            congr 4 <;> omega]
      convert mobiusFaaDiBrunoCoefficientSum_reindex k B using 1 <;>
        apply Finset.sum_congr rfl <;> intro i hi <;>
        have hi' := Finset.mem_range.mp hi <;>
        simp only [B] <;> congr 4 <;> grind

/-- At the origin all Möbius powers become one, leaving exactly the finite
descending-factorial derivative sum used in the Li coefficient identity. -/
theorem iteratedDeriv_mobiusComposition_zero
    {L : ℂ → ℂ} (hL : AnalyticAt ℂ L 1) (k : ℕ) :
    iteratedDeriv k
      (fun w ↦ liMobiusArgument w ^ 2 *
        iteratedDeriv 1 L (liMobiusArgument w)) 0 =
      ∑ i ∈ Finset.range (k + 2),
        (mobiusFaaDiBrunoCoefficient k i : ℂ) *
          iteratedDeriv (k + 1 - i) L 1 := by
  rw [iteratedDeriv_mobiusComposition_eq_row (by norm_num) (by simpa
    [liMobiusArgument] using hL)]
  unfold mobiusFaaDiBrunoRow
  simp [liMobiusArgument, Finset.sum_apply]

/-- Branch-compatible pointwise logarithmic chain rule for the Möbius
change of variables.  Membership in the slit plane is the exact hypothesis
needed by Mathlib's principal `Complex.log`; it in particular implies local
nonvanishing. -/
theorem logDeriv_liChangeOfVariables_eq_mobiusLogDerivative
    {xi : ℂ → ℂ} {z : ℂ} (hz : z ≠ 1)
    (hxi : DifferentiableAt ℂ xi (liMobiusArgument z))
    (hslit : xi (liMobiusArgument z) ∈ Complex.slitPlane) :
    logDeriv (liChangeOfVariables xi) z =
      liMobiusArgument z ^ 2 *
        iteratedDeriv 1 (fun s ↦ Complex.log (xi s))
          (liMobiusArgument z) := by
  have hbase : HasDerivAt (fun w : ℂ ↦ 1 - w) (-1) z := by
    simpa [sub_eq_add_neg] using (hasDerivAt_id z).neg.const_add 1
  have hne : 1 - z ≠ 0 := sub_ne_zero.mpr hz.symm
  have hφ : HasDerivAt liMobiusArgument
      (liMobiusArgument z ^ 2) z := by
    unfold liMobiusArgument
    exact (hbase.fun_inv hne).congr_deriv (by
      rw [pow_two]
      field_simp)
  have hcomp := hxi.hasDerivAt.comp z hφ
  change HasDerivAt (liChangeOfVariables xi)
    (deriv xi (liMobiusArgument z) * liMobiusArgument z ^ 2) z at hcomp
  have hlog := hxi.hasDerivAt.clog hslit
  have hlog' : iteratedDeriv 1 (fun s ↦ Complex.log (xi s))
      (liMobiusArgument z) =
      deriv xi (liMobiusArgument z) / xi (liMobiusArgument z) := by
    rw [iteratedDeriv_one]
    exact hlog.deriv
  rw [logDeriv_apply, hcomp.deriv, hlog']
  unfold liChangeOfVariables liMobiusArgument
  ring

/-- Since `xi(1)=1` lies in the principal logarithm domain, continuity gives
an entire neighborhood of the transformed origin on which xi is nonzero and
the logarithmic chain rule is branch-compatible. -/
theorem eventually_logDeriv_liChangeOfVariables_riemannXiLi :
    logDeriv (liChangeOfVariables riemannXiLi) =ᶠ[𝓝 0]
      fun w ↦ liMobiusArgument w ^ 2 *
        iteratedDeriv 1 (fun s ↦ Complex.log (riemannXiLi s))
          (liMobiusArgument w) := by
  have hφc : ContinuousAt liMobiusArgument 0 := by
    unfold liMobiusArgument
    exact (continuousAt_const.sub continuousAt_id).inv₀ (by norm_num)
  have hxic : ContinuousAt
      (fun w ↦ riemannXiLi (liMobiusArgument w)) 0 :=
    differentiable_riemannXiLi.continuous.continuousAt.comp hφc
  have hslit : ∀ᶠ w in 𝓝 0,
      riemannXiLi (liMobiusArgument w) ∈ Complex.slitPlane :=
    hxic.tendsto.eventually (Complex.isOpen_slitPlane.mem_nhds (by
      simp [liMobiusArgument]))
  have hne : ∀ᶠ w in 𝓝 (0 : ℂ), w ≠ 1 :=
    eventually_ne_nhds (by norm_num)
  filter_upwards [hne, hslit] with w hw hsw
  exact logDeriv_liChangeOfVariables_eq_mobiusLogDerivative hw
    (differentiable_riemannXiLi.differentiableAt) hsw

/-- Unconditional all-order numerator identity for normalized xi. -/
theorem iteratedDeriv_logDeriv_liChangeOfVariables_riemannXiLi
    (k : ℕ) :
    iteratedDeriv k (logDeriv (liChangeOfVariables riemannXiLi)) 0 =
      ∑ i ∈ Finset.range (k + 2),
        (mobiusFaaDiBrunoCoefficient k i : ℂ) *
          iteratedDeriv (k + 1 - i)
            (fun s ↦ Complex.log (riemannXiLi s)) 1 := by
  rw [eventually_logDeriv_liChangeOfVariables_riemannXiLi.iteratedDeriv_eq k]
  exact iteratedDeriv_mobiusComposition_zero
    analyticAt_log_riemannXiLi_one k

theorem riemannXiLi_mobiusFaaDiBrunoCoefficientIdentity :
    MobiusFaaDiBrunoCoefficientIdentity riemannXiLi := by
  intro k
  rw [iteratedDeriv_logDeriv_liChangeOfVariables_riemannXiLi k]
  simp only [mobiusFaaDiBrunoCoefficient, Nat.cast_mul]

/-- The specialized coefficient recurrence, including base and support
conditions so that it characterizes the triangular coefficient array
uniquely. -/
def IsMobiusFaaDiBrunoCoefficientRecurrence (c : ℕ → ℕ → ℕ) : Prop :=
  (∀ i, c 0 i = if i = 0 then 1 else 0) ∧
  (∀ k i, k < i → c k i = 0) ∧
  ∀ k i, i ≤ k + 1 →
    c (k + 1) i = c k i +
      (if i = 0 then 0 else (2 * k + 3 - i) * c k (i - 1))

theorem mobiusFaaDiBrunoCoefficient_recurrence :
    IsMobiusFaaDiBrunoCoefficientRecurrence
      mobiusFaaDiBrunoCoefficient := by
  refine ⟨?_, ?_, mobiusFaaDiBrunoCoefficient_succ⟩
  · intro i
    cases i <;> simp [mobiusFaaDiBrunoCoefficient]
  · intro k i hki
    simp [mobiusFaaDiBrunoCoefficient,
      Nat.descFactorial_eq_zero_iff_lt.mpr hki]

/-- The recurrence has at most one triangular coefficient array.  Together
with `mobiusFaaDiBrunoCoefficient_recurrence`, this proves that any
coefficient construction obtained by iterating the chain/product rule equals
`choose(k+1,i) * descFactorial(k,i)` at every index and boundary. -/
theorem IsMobiusFaaDiBrunoCoefficientRecurrence.eq
    {c d : ℕ → ℕ → ℕ}
    (hc : IsMobiusFaaDiBrunoCoefficientRecurrence c)
    (hd : IsMobiusFaaDiBrunoCoefficientRecurrence d) :
    c = d := by
  funext k i
  induction k generalizing i with
  | zero =>
      rw [hc.1 i, hd.1 i]
  | succ k ih =>
      by_cases hi : i ≤ k + 1
      · rw [hc.2.2 k i hi, hd.2.2 k i hi, ih i, ih (i - 1)]
      · have hik : k + 1 < i := lt_of_not_ge hi
        rw [hc.2.1 (k + 1) i hik, hd.2.1 (k + 1) i hik]

/-- For normalized xi, the formerly opaque coefficient obligation is exactly
the displayed finite Möbius/Faà di Bruno sum. -/
theorem riemannXiLi_changeOfVariablesCoefficientIdentity_iff_faaDiBruno :
    LiChangeOfVariablesCoefficientIdentity riemannXiLi ↔
      MobiusFaaDiBrunoCoefficientIdentity riemannXiLi := by
  constructor
  · intro h k
    rw [h k]
    exact liDerivativeCoefficient_eq_leibnizSum
      riemannXiLi (k + 1) (by omega)
        analyticAt_log_riemannXiLi_one.contDiffAt
  · intro h k
    rw [h k]
    exact (liDerivativeCoefficient_eq_leibnizSum
      riemannXiLi (k + 1) (by omega)
        analyticAt_log_riemannXiLi_one.contDiffAt).symm

/-- The all-order xi Möbius coefficient transfer is unconditional. -/
theorem riemannXiLi_changeOfVariablesCoefficientIdentity :
    LiChangeOfVariablesCoefficientIdentity riemannXiLi :=
  riemannXiLi_changeOfVariablesCoefficientIdentity_iff_faaDiBruno.mpr
    riemannXiLi_mobiusFaaDiBrunoCoefficientIdentity

/-- The first xi change-of-variables coefficient is unconditional once xi is
differentiable and normalized to `xi 1 = 1`. -/
theorem logDeriv_liChangeOfVariables_zero
    (xi : ℂ → ℂ) (hxi : DifferentiableAt ℂ xi 1) (hxi1 : xi 1 = 1) :
    logDeriv (liChangeOfVariables xi) 0 =
      liDerivativeCoefficient xi 1 := by
  rw [liDerivativeCoefficient_one, logDeriv_apply]
  unfold liChangeOfVariables
  have hinv : HasDerivAt (fun z : ℂ ↦ (1 - z)⁻¹) 1 0 := by
    have hbase : HasDerivAt (fun z : ℂ ↦ 1 - z) (-1) 0 := by
      simpa [sub_eq_add_neg] using
        (hasDerivAt_id (x := (0 : ℂ))).neg.const_add 1
    simpa using hbase.fun_inv (by norm_num)
  have hcomp : HasDerivAt
      (fun z : ℂ ↦ xi ((1 - z)⁻¹)) (deriv xi 1) 0 := by
    have hxi' : HasDerivAt xi (deriv xi 1) ((1 - (0 : ℂ))⁻¹) := by
      simpa using hxi.hasDerivAt
    simpa [Function.comp_def] using hxi'.comp (𝕜 := ℂ) 0 hinv
  rw [hcomp.deriv]
  rw [show (1 - (0 : ℂ))⁻¹ = 1 by norm_num, hxi1]
  rw [(hxi.hasDerivAt.clog (by simp [hxi1])).deriv]
  simp [hxi1]

theorem LiChangeOfVariablesCoefficientIdentity.zero
    {xi : ℂ → ℂ} (h : LiChangeOfVariablesCoefficientIdentity xi) :
    logDeriv (liChangeOfVariables xi) 0 =
      liDerivativeCoefficient xi 1 := by
  simpa [LiChangeOfVariablesCoefficientIdentity] using h 0

/-- Mathlib's analytic order gives the intrinsic multiplicity of a zeta zero.

The value lies in `ℕ∞`: finiteness still requires ruling out local identically
zero behavior.  This definition does not enumerate the zeros and therefore
does not by itself provide the multiset needed by the symmetric Li sum. -/
def riemannZetaZeroOrder (rho : ℂ) : ℕ∞ :=
  analyticOrderAt riemannZeta rho

/-- Away from zeta's pole, nonzero analytic order is exactly membership in
Mathlib's set of zeta zeros. -/
theorem riemannZetaZeroOrder_ne_zero_iff {rho : ℂ} (hρ : rho ≠ 1) :
    riemannZetaZeroOrder rho ≠ 0 ↔ rho ∈ riemannZetaZeros := by
  rw [riemannZetaZeroOrder, analyticOrderAt_ne_zero]
  simp only [mem_riemannZetaZeros]
  exact and_iff_right (analyticOn_riemannZeta rho hρ)

/-- Zeta has finite analytic order at every point away from its pole.

This closes the local multiplicity-finiteness part of zero enumeration.  The
proof propagates finite order from `2`, where zeta is nonzero, through the
connected punctured plane using Mathlib's analytic-order identity theorem. -/
theorem riemannZetaZeroOrder_ne_top {rho : ℂ} (hρ : rho ≠ 1) :
    riemannZetaZeroOrder rho ≠ ⊤ := by
  unfold riemannZetaZeroOrder
  apply analyticOn_riemannZeta.analyticOrderAt_ne_top_of_isPreconnected
    (x := (2 : ℂ)) (y := rho)
  · exact (isConnected_compl_singleton_of_one_lt_rank (by simp) 1).isPreconnected
  · simp
  · simpa
  · have h2 : riemannZeta (2 : ℂ) ≠ 0 :=
      riemannZeta_ne_zero_of_one_le_re (by norm_num)
    rw [(analyticOn_riemannZeta (2 : ℂ) (by norm_num)).analyticOrderAt_eq_zero.mpr h2]
    exact ENat.zero_ne_top

/-- The finite natural multiplicity represented by zeta's analytic order. -/
def riemannZetaZeroMultiplicity (rho : ℂ) : ℕ :=
  analyticOrderNatAt riemannZeta rho

theorem riemannZetaZeroMultiplicity_cast {rho : ℂ} (hρ : rho ≠ 1) :
    (riemannZetaZeroMultiplicity rho : ℕ∞) = riemannZetaZeroOrder rho := by
  exact Nat.cast_analyticOrderNatAt (riemannZetaZeroOrder_ne_top hρ)

/-- Xi and zeta zero multiplicities agree in the positive-real half-plane.
This covers the nontrivial critical strip and avoids Gamma's exceptional
totalized values. -/
theorem riemannXiZeroOrder_eq_riemannZetaZeroOrder {rho : ℂ}
    (h0 : rho ≠ 0) (h1 : rho ≠ 1) (hre : 0 < rho.re) :
    riemannXiZeroOrder rho = riemannZetaZeroOrder rho := by
  let p : ℂ → ℂ := fun s ↦ s * (s - 1)
  let c : ℂ → ℂ := completedRiemannZeta
  let gi : ℂ → ℂ := fun s ↦ (Gammaℝ s)⁻¹
  have hp : AnalyticAt ℂ p rho := by
    dsimp [p]
    fun_prop
  have hc : AnalyticAt ℂ c rho := by
    let U : Set ℂ := ({0} : Set ℂ)ᶜ ∩ ({1} : Set ℂ)ᶜ
    have hUopen : IsOpen U :=
      isOpen_compl_singleton.inter isOpen_compl_singleton
    have hrho : rho ∈ U := by simp [U, h0, h1]
    apply DifferentiableOn.analyticAt (s := U) _ (hUopen.mem_nhds hrho)
    intro z hz
    apply (differentiableAt_completedZeta
      (by simpa [U] using hz.1)
      (by simpa [U] using hz.2)).differentiableWithinAt
  have hgi : AnalyticAt ℂ gi rho :=
    differentiable_Gammaℝ_inv.analyticAt rho
  have hp0 : p rho ≠ 0 := mul_ne_zero h0 (sub_ne_zero.mpr h1)
  have hgi0 : gi rho ≠ 0 := inv_ne_zero (Gammaℝ_ne_zero_of_re_pos hre)
  have hxieq : riemannXiLi =ᶠ[𝓝 rho] p * c := by
    filter_upwards [
      isOpen_compl_singleton.mem_nhds h0,
      isOpen_compl_singleton.mem_nhds h1] with s hs0 hs1
    simpa [p, c, Pi.mul_apply] using
      riemannXiLi_eq_mul_completedRiemannZeta hs0 hs1
  have hzeq : riemannZeta =ᶠ[𝓝 rho] c * gi := by
    filter_upwards [isOpen_compl_singleton.mem_nhds h0] with s hs0
    simpa [c, gi, Pi.mul_apply, div_eq_mul_inv] using
      riemannZeta_def_of_ne_zero hs0
  unfold riemannXiZeroOrder riemannZetaZeroOrder
  rw [analyticOrderAt_congr hxieq, analyticOrderAt_congr hzeq,
    analyticOrderAt_mul hp hc, analyticOrderAt_mul hc hgi,
    hp.analyticOrderAt_eq_zero.mpr hp0,
    hgi.analyticOrderAt_eq_zero.mpr hgi0, zero_add, add_zero]

theorem riemannXiZeroMultiplicity_eq_riemannZetaZeroMultiplicity
    {rho : ℂ} (h0 : rho ≠ 0) (h1 : rho ≠ 1) (hre : 0 < rho.re) :
    riemannXiZeroMultiplicity rho =
      riemannZetaZeroMultiplicity rho := by
  rw [← ENat.coe_inj, riemannXiZeroMultiplicity_cast,
    riemannZetaZeroMultiplicity_cast h1,
    riemannXiZeroOrder_eq_riemannZetaZeroOrder h0 h1 hre]

/-- All zeta zeros in the closed norm ball of radius `R`.

Mathlib proves that a compact set meets `riemannZetaZeros` in a finite set,
so this is an actual `Finset`, not an assumed enumeration. -/
def riemannZetaZeroBall (R : ℝ) : Finset ℂ :=
  ((isCompact_closedBall (0 : ℂ) R).inter_riemannZetaZeros_finite).toFinset

theorem mem_riemannZetaZeroBall {R : ℝ} {rho : ℂ} :
    rho ∈ riemannZetaZeroBall R ↔ ‖rho‖ ≤ R ∧ riemannZeta rho = 0 := by
  simp [riemannZetaZeroBall, Set.Finite.mem_toFinset,
    Metric.mem_closedBall, mem_riemannZetaZeros]

/-- A zeta zero is nontrivial when it is not one of the explicit negative
even zeros.  The impossible pole value `rho = 1` is ruled out separately by
the zero equation whenever needed. -/
def IsNontrivialRiemannZetaZero (rho : ℂ) : Prop :=
  riemannZeta rho = 0 ∧ ¬ ∃ n : ℕ, rho = -2 * (n + 1)

/-- The finite bounded window of nontrivial zeta zeros, without multiplicity. -/
noncomputable def nontrivialRiemannZetaZeroWindow (R : ℝ) : Finset ℂ := by
  classical
  exact (riemannZetaZeroBall R).filter IsNontrivialRiemannZetaZero

theorem mem_nontrivialRiemannZetaZeroWindow {R : ℝ} {rho : ℂ} :
    rho ∈ nontrivialRiemannZetaZeroWindow R ↔
      ‖rho‖ ≤ R ∧ IsNontrivialRiemannZetaZero rho := by
  classical
  rw [nontrivialRiemannZetaZeroWindow, Finset.mem_filter,
    mem_riemannZetaZeroBall]
  simp only [IsNontrivialRiemannZetaZero]
  tauto

theorem nontrivialRiemannZetaZeroWindow_mono {R S : ℝ} (hRS : R ≤ S) :
    nontrivialRiemannZetaZeroWindow R ⊆
      nontrivialRiemannZetaZeroWindow S := by
  intro rho hρ
  rw [mem_nontrivialRiemannZetaZeroWindow] at hρ ⊢
  exact ⟨hρ.1.trans hRS, hρ.2⟩

theorem mem_nontrivialRiemannZetaZeroWindow_self {rho : ℂ}
    (hρ : IsNontrivialRiemannZetaZero rho) :
    rho ∈ nontrivialRiemannZetaZeroWindow ‖rho‖ :=
  mem_nontrivialRiemannZetaZeroWindow.mpr ⟨le_rfl, hρ⟩

/-- A genuinely finite symmetric-height window with an explicit real bound.
Both inequalities are closed, so zeros on either boundary are retained. -/
noncomputable def symmetricHeightRiemannZetaZeroWindow
    (realBound height : ℝ) : Finset ℂ := by
  classical
  exact (nontrivialRiemannZetaZeroWindow (realBound + height)).filter
    (fun rho ↦ |rho.re| ≤ realBound ∧ |rho.im| ≤ height)

theorem mem_symmetricHeightRiemannZetaZeroWindow
    {realBound height : ℝ} {rho : ℂ} :
    rho ∈ symmetricHeightRiemannZetaZeroWindow realBound height ↔
      IsNontrivialRiemannZetaZero rho ∧
        |rho.re| ≤ realBound ∧ |rho.im| ≤ height := by
  classical
  rw [symmetricHeightRiemannZetaZeroWindow, Finset.mem_filter,
    mem_nontrivialRiemannZetaZeroWindow]
  constructor
  · rintro ⟨⟨_, hzero⟩, hre, him⟩
    exact ⟨hzero, hre, him⟩
  · rintro ⟨hzero, hre, him⟩
    exact ⟨⟨(Complex.norm_le_abs_re_add_abs_im rho).trans
      (add_le_add hre him), hzero⟩, hre, him⟩

theorem symmetricHeightRiemannZetaZeroWindow_mono
    {A B T U : ℝ} (hAB : A ≤ B) (hTU : T ≤ U) :
    symmetricHeightRiemannZetaZeroWindow A T ⊆
      symmetricHeightRiemannZetaZeroWindow B U := by
  intro rho hρ
  rw [mem_symmetricHeightRiemannZetaZeroWindow] at hρ ⊢
  exact ⟨hρ.1, hρ.2.1.trans hAB, hρ.2.2.trans hTU⟩

/-- Multiplicity-aware indices in a compact symmetric-height rectangle. -/
def RiemannZetaSymmetricHeightIndex (realBound height : ℝ) :=
  Σ rho : {rho : ℂ //
      rho ∈ symmetricHeightRiemannZetaZeroWindow realBound height},
    Fin (riemannZetaZeroMultiplicity rho)

noncomputable instance (realBound height : ℝ) :
    Fintype (RiemannZetaSymmetricHeightIndex realBound height) := by
  unfold RiemannZetaSymmetricHeightIndex
  infer_instance

/-- Increasing the height gives an injective multiplicity-preserving
reindexing; no boundary zero is lost. -/
noncomputable def riemannZetaSymmetricHeightReindex
    {realBound T U : ℝ} (hTU : T ≤ U) :
    RiemannZetaSymmetricHeightIndex realBound T ↪
      RiemannZetaSymmetricHeightIndex realBound U where
  toFun i := ⟨⟨i.1.1,
    symmetricHeightRiemannZetaZeroWindow_mono le_rfl hTU i.1.2⟩, i.2⟩
  inj' := by
    intro i j hij
    cases i with
    | mk ir im =>
      cases j with
      | mk jr jm =>
        have hir : ir = jr :=
          Subtype.ext (congrArg (fun x ↦ x.1.1) hij)
        subst jr
        have him : im = jm := by
          apply Fin.ext
          simpa using congrArg (fun x ↦ x.2.val) hij
        subst jm
        rfl

@[simp] theorem riemannZetaSymmetricHeightReindex_root
    {realBound T U : ℝ} (hTU : T ≤ U)
    (i : RiemannZetaSymmetricHeightIndex realBound T) :
    (riemannZetaSymmetricHeightReindex hTU i).1.1 = i.1.1 :=
  rfl

/-- An explicit real-strip hypothesis turns the compact rectangles into
pure symmetric-height windows.  This hypothesis is deliberately separate:
pinned Mathlib does not supply the unconditional nontrivial-zero strip
classification needed to instantiate it. -/
def NontrivialZetaZerosInRealStrip (realBound : ℝ) : Prop :=
  ∀ rho, IsNontrivialRiemannZetaZero rho → |rho.re| ≤ realBound

theorem mem_symmetricHeightWindow_iff_of_realStrip
    {realBound height : ℝ}
    (hstrip : NontrivialZetaZerosInRealStrip realBound) {rho : ℂ} :
    rho ∈ symmetricHeightRiemannZetaZeroWindow realBound height ↔
      IsNontrivialRiemannZetaZero rho ∧ |rho.im| ≤ height := by
  rw [mem_symmetricHeightRiemannZetaZeroWindow]
  exact and_congr_right fun hρ ↦
    and_iff_right (hstrip rho hρ)

theorem nontrivialZetaZerosInRealStrip_one_of_rh
    (hRH : KakeyaRiemannHypothesisRoot) :
    NontrivialZetaZerosInRealStrip 1 := by
  intro rho hρ
  have hre : rho.re = 1 / 2 := hRH rho hρ.1 hρ.2
    (fun h1 ↦ riemannZeta_one_ne_zero (h1 ▸ hρ.1))
  rw [hre]
  norm_num

theorem mem_symmetricHeightWindow_one_iff_of_rh
    (hRH : KakeyaRiemannHypothesisRoot) {height : ℝ} {rho : ℂ} :
    rho ∈ symmetricHeightRiemannZetaZeroWindow 1 height ↔
      IsNontrivialRiemannZetaZero rho ∧ |rho.im| ≤ height :=
  mem_symmetricHeightWindow_iff_of_realStrip
    (nontrivialZetaZerosInRealStrip_one_of_rh hRH)

/-- Exact reality obligation for the derivative formula. -/
def LiDerivativeIsReal (xi : ℂ → ℂ) : Prop :=
  ∀ n : ℕ, 0 < n → (liDerivativeCoefficient xi n).im = 0

theorem riemannLiCoefficient_cast_eq
    (hreal : LiDerivativeIsReal riemannXiLi) {n : ℕ} (hn : 0 < n) :
    (riemannLiCoefficient n : ℂ) =
      liDerivativeCoefficient riemannXiLi n := by
  apply Complex.ext
  · simp [riemannLiCoefficient]
  · simp [riemannLiCoefficient, hreal n hn]

/-- Non-negativity at every source-relevant (positive) index. -/
def LiPositive (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, 0 < n → 0 ≤ a n

/-- Finite-prefix evidence through index `N`; this is not an RH statement. -/
def LiPositiveThrough (a : ℕ → ℝ) (N : ℕ) : Prop :=
  ∀ n : ℕ, 0 < n → n ≤ N → 0 ≤ a n

theorem LiPositive.through {a : ℕ → ℝ} (h : LiPositive a) (N : ℕ) :
    LiPositiveThrough a N :=
  fun n hn _ ↦ h n hn

theorem liPositive_iff_all_prefixes (a : ℕ → ℝ) :
    LiPositive a ↔ ∀ N : ℕ, LiPositiveThrough a N := by
  constructor
  · exact fun h N ↦ h.through N
  · intro h n hn
    exact h n n hn le_rfl

theorem LiPositiveThrough.mono {a : ℕ → ℝ} {M N : ℕ}
    (h : LiPositiveThrough a N) (hMN : M ≤ N) :
    LiPositiveThrough a M :=
  fun n hn hnM ↦ h n hn (hnM.trans hMN)

@[simp] theorem liPositiveThrough_zero (a : ℕ → ℝ) :
    LiPositiveThrough a 0 := by
  intro n hn hn0
  omega

/-- A concrete counterexample to the invalid inference from any fixed finite
positive prefix to all-index positivity. -/
def finitePrefixSpoof (N n : ℕ) : ℝ :=
  if n ≤ N then 1 else -1

theorem finitePrefixSpoof_positiveThrough (N : ℕ) :
    LiPositiveThrough (finitePrefixSpoof N) N := by
  intro n _ hn
  simp [finitePrefixSpoof, hn]

theorem finitePrefixSpoof_not_positive (N : ℕ) :
    ¬ LiPositive (finitePrefixSpoof N) := by
  intro h
  have := h (N + 1) (by omega)
  have hbad : (0 : ℝ) ≤ -1 := by
    simpa [finitePrefixSpoof, show ¬N + 1 ≤ N by omega] using this
  norm_num at hbad

/-- The all-index Li criterion, deliberately a proposition rather than a
theorem.  Proving this proposition is the source theorem bridge. -/
def RiemannLiCriterionStatement : Prop :=
  KakeyaRiemannHypothesisRoot ↔ LiPositive riemannLiCoefficient

/-- One term in the zero-sum formulation. -/
def liZeroSummand (n : ℕ) (rho : ℂ) : ℂ :=
  1 - (1 - rho⁻¹) ^ n

@[simp] theorem liZeroSummand_zero (rho : ℂ) :
    liZeroSummand 0 rho = 0 := by
  simp [liZeroSummand]

@[simp] theorem liZeroSummand_one (rho : ℂ) :
    liZeroSummand 1 rho = rho⁻¹ := by
  simp [liZeroSummand]

theorem liZeroSummand_conj (n : ℕ) (rho : ℂ) :
    liZeroSummand n (star rho) = star (liZeroSummand n rho) := by
  simp [liZeroSummand]

/-- A finite truncation of an indexed multiset of zeros.  Indices, rather
than a `Finset ℂ`, preserve repeated roots when multiplicities are supplied. -/
def liZeroPartialSum {ι : Type*} (root : ι → ℂ)
    (cutoff : Finset ι) (n : ℕ) : ℂ :=
  ∑ i ∈ cutoff, liZeroSummand n (root i)

@[simp] theorem liZeroPartialSum_zero {ι : Type*} [DecidableEq ι]
    (root : ι → ℂ) (cutoff : Finset ι) :
    liZeroPartialSum root cutoff 0 = 0 := by
  simp [liZeroPartialSum]

/-- A finite, multiplicity-aware family of nonzero zeros.

Repeated values at distinct `Fin` indices are retained, so this is an
indexed multiset rather than a set.  It is concrete data, not an assumption
about the zeros of zeta. -/
structure FiniteZeroMultiset where
  card : ℕ
  root : Fin card → ℂ
  root_ne_zero : ∀ i, root i ≠ 0

/-- Multiplicity-aware indices for one bounded nontrivial-zeta-zero window.
The second component repeats each root exactly its natural analytic order. -/
def RiemannZetaZeroWindowIndex (R : ℝ) :=
  Σ rho : {rho : ℂ // rho ∈ nontrivialRiemannZetaZeroWindow R},
    Fin (riemannZetaZeroMultiplicity rho)

noncomputable instance (R : ℝ) :
    Fintype (RiemannZetaZeroWindowIndex R) := by
  unfold RiemannZetaZeroWindowIndex
  infer_instance

def riemannZetaZeroWindowRoot {R : ℝ}
    (i : RiemannZetaZeroWindowIndex R) : ℂ :=
  i.1.1

theorem riemannZetaZeroWindowRoot_mem {R : ℝ}
    (i : RiemannZetaZeroWindowIndex R) :
    riemannZetaZeroWindowRoot i ∈ nontrivialRiemannZetaZeroWindow R :=
  i.1.2

theorem riemannZetaZeroWindowRoot_ne_zero {R : ℝ}
    (i : RiemannZetaZeroWindowIndex R) :
    riemannZetaZeroWindowRoot i ≠ 0 := by
  change i.1.1 ≠ 0
  intro hi
  have hz := (mem_nontrivialRiemannZetaZeroWindow.mp i.1.2).2.1
  rw [hi, riemannZeta_zero] at hz
  norm_num at hz

/-- Adapter from the sigma-type enumeration to the route's concrete finite
multiset.  The `Fintype.equivFin` reindexing changes no roots or
multiplicities. -/
noncomputable def riemannZetaZeroWindowMultiset (R : ℝ) :
    FiniteZeroMultiset where
  card := Fintype.card (RiemannZetaZeroWindowIndex R)
  root := fun i ↦ riemannZetaZeroWindowRoot
    ((Fintype.equivFin (RiemannZetaZeroWindowIndex R)).symm i)
  root_ne_zero := fun _ ↦ riemannZetaZeroWindowRoot_ne_zero _

theorem riemannZetaZeroWindowMultiset_root_spec (R : ℝ)
    (i : Fin (riemannZetaZeroWindowMultiset R).card) :
    ‖(riemannZetaZeroWindowMultiset R).root i‖ ≤ R ∧
      IsNontrivialRiemannZetaZero
        ((riemannZetaZeroWindowMultiset R).root i) := by
  let j := (Fintype.equivFin (RiemannZetaZeroWindowIndex R)).symm i
  change ‖riemannZetaZeroWindowRoot j‖ ≤ R ∧
    IsNontrivialRiemannZetaZero (riemannZetaZeroWindowRoot j)
  exact mem_nontrivialRiemannZetaZeroWindow.mp j.1.2

/-- RH places every root in every concrete bounded multiplicity window on
the critical line.  This is a finite-window consequence, not the missing
infinite Li equivalence. -/
theorem riemannZetaZeroWindowMultiset_on_criticalLine
    (hRH : KakeyaRiemannHypothesisRoot) (R : ℝ) :
    ∀ i, ((riemannZetaZeroWindowMultiset R).root i).re = 1 / 2 := by
  intro i
  have hi := riemannZetaZeroWindowMultiset_root_spec R i
  exact hRH _ hi.2.1 hi.2.2 (fun h1 ↦
    riemannZeta_one_ne_zero (h1 ▸ hi.2.1))

/-- One zero repeated with a prescribed positive analytic multiplicity. -/
def FiniteZeroMultiset.replicate (multiplicity : ℕ) (rho : ℂ)
    (hρ : rho ≠ 0) : FiniteZeroMultiset where
  card := multiplicity
  root := fun _ ↦ rho
  root_ne_zero := fun _ ↦ hρ

/-- The exact Li sum of a finite indexed multiset. -/
def FiniteZeroMultiset.liSum (zeros : FiniteZeroMultiset) (n : ℕ) : ℂ :=
  ∑ i, liZeroSummand n (zeros.root i)

@[simp] theorem FiniteZeroMultiset.liSum_replicate
    (multiplicity n : ℕ) (rho : ℂ) (hρ : rho ≠ 0) :
    (FiniteZeroMultiset.replicate multiplicity rho hρ).liSum n =
      multiplicity * liZeroSummand n rho := by
  change (∑ _ : Fin multiplicity, liZeroSummand n rho) =
    multiplicity * liZeroSummand n rho
  rw [Finset.sum_const, Finset.card_univ]
  norm_num [Fintype.card_fin, nsmul_eq_mul]

@[simp] theorem FiniteZeroMultiset.liSum_zero (zeros : FiniteZeroMultiset) :
    zeros.liSum 0 = 0 := by
  simp [FiniteZeroMultiset.liSum]

theorem FiniteZeroMultiset.liSum_one (zeros : FiniteZeroMultiset) :
    zeros.liSum 1 = ∑ i, (zeros.root i)⁻¹ := by
  simp [FiniteZeroMultiset.liSum]

/-- One zero contributes a nonnegative real part whenever its Li transform
`1 - 1 / rho` lies in the closed unit disk. -/
theorem liZeroSummand_re_nonneg_of_norm_le_one
    (n : ℕ) (rho : ℂ) (hρ : ‖1 - rho⁻¹‖ ≤ 1) :
    0 ≤ (liZeroSummand n rho).re := by
  have hp : ((1 - rho⁻¹) ^ n).re ≤ 1 := calc
    ((1 - rho⁻¹) ^ n).re
        ≤ |((1 - rho⁻¹) ^ n).re| := le_abs_self _
    _ ≤ ‖(1 - rho⁻¹) ^ n‖ := abs_re_le_norm _
    _ = ‖1 - rho⁻¹‖ ^ n := norm_pow _ _
    _ ≤ 1 := pow_le_one₀ (norm_nonneg _) hρ
  simpa [liZeroSummand] using sub_nonneg.mpr hp

/-- Exact finite positivity: no convergence or zero-enumeration theorem is
needed for a finite multiplicity-aware family. -/
theorem FiniteZeroMultiset.liSum_re_nonneg
    (zeros : FiniteZeroMultiset)
    (hunit : ∀ i, ‖1 - (zeros.root i)⁻¹‖ ≤ 1)
    (n : ℕ) :
    0 ≤ (zeros.liSum n).re := by
  change 0 ≤ Complex.reCLM (∑ i, liZeroSummand n (zeros.root i))
  rw [map_sum]
  exact Finset.sum_nonneg fun i _ ↦
    liZeroSummand_re_nonneg_of_norm_le_one n (zeros.root i) (hunit i)

/-- On the critical line, the Li transform has norm exactly one. -/
theorem norm_one_sub_inv_eq_one_of_re_eq_half
    {rho : ℂ} (hρ0 : rho ≠ 0) (hline : rho.re = 1 / 2) :
    ‖1 - rho⁻¹‖ = 1 := by
  have hnormSq : ‖rho - 1‖ ^ 2 = ‖rho‖ ^ 2 := by
    rw [Complex.sq_norm, Complex.sq_norm, Complex.normSq_apply, Complex.normSq_apply]
    simp only [sub_re, one_re, sub_im, one_im, sub_zero]
    rw [hline]
    ring
  have hnorm : ‖rho - 1‖ = ‖rho‖ := by
    nlinarith [norm_nonneg (rho - 1), norm_nonneg rho]
  have hid : 1 - rho⁻¹ = (rho - 1) / rho := by
    field_simp
  rw [hid, norm_div, hnorm, div_self]
  exact norm_ne_zero_iff.mpr hρ0

/-- The Li transform is in the closed unit disk exactly when the root is in
the closed right half-plane bounded by the critical line. -/
theorem norm_one_sub_inv_le_one_iff_re_ge_half
    {rho : ℂ} (hρ : rho ≠ 0) :
    ‖1 - rho⁻¹‖ ≤ 1 ↔ 1 / 2 ≤ rho.re := by
  have hid : 1 - rho⁻¹ = (rho - 1) / rho := by
    field_simp
  rw [hid, norm_div]
  have hnorm : 0 < ‖rho‖ := norm_pos_iff.mpr hρ
  rw [div_le_one hnorm]
  have hsquares :
      ‖rho - 1‖ ^ 2 ≤ ‖rho‖ ^ 2 ↔ 1 / 2 ≤ rho.re := by
    rw [Complex.sq_norm, Complex.sq_norm,
      Complex.normSq_apply, Complex.normSq_apply]
    simp only [sub_re, one_re, sub_im, one_im, sub_zero]
    constructor <;> intro h <;> nlinarith
  rw [← hsquares]
  exact (sq_le_sq₀ (norm_nonneg (rho - 1)) (norm_nonneg rho)).symm

theorem re_eq_half_of_reflected_liTransforms_le_one
    {rho : ℂ} (hρ : rho ≠ 0) (h1ρ : 1 - rho ≠ 0)
    (hr : ‖1 - rho⁻¹‖ ≤ 1)
    (hreflect : ‖1 - (1 - rho)⁻¹‖ ≤ 1) :
    rho.re = 1 / 2 := by
  have hright := (norm_one_sub_inv_le_one_iff_re_ge_half hρ).mp hr
  have hleft := (norm_one_sub_inv_le_one_iff_re_ge_half h1ρ).mp hreflect
  simp only [sub_re, one_re] at hleft
  linarith

/-- Finite reflection symmetry, including multiplicities through an
equivalence of indices. -/
structure FiniteZeroMultiset.ReflectionSymmetric
    (zeros : FiniteZeroMultiset) where
  partner : Fin zeros.card ≃ Fin zeros.card
  root_partner : ∀ i, zeros.root (partner i) = 1 - zeros.root i

/-- The exact finite spectral converse needed to recover a unit-disk bound
from all-index aggregate Li positivity.  This is named, not assumed. -/
def FiniteLiSpectralConverseStatement
    (zeros : FiniteZeroMultiset) : Prop :=
  (∀ n : ℕ, 0 < n → 0 ≤ (zeros.liSum n).re) →
    ∀ i, ‖1 - (zeros.root i)⁻¹‖ ≤ 1

/-- For a reflection-symmetric finite multiset, the spectral converse would
force every root onto the critical line. -/
theorem FiniteZeroMultiset.on_criticalLine_of_spectralConverse
    (zeros : FiniteZeroMultiset)
    (hsym : zeros.ReflectionSymmetric)
    (hconverse : FiniteLiSpectralConverseStatement zeros)
    (hpositive : ∀ n : ℕ, 0 < n → 0 ≤ (zeros.liSum n).re) :
    ∀ i, (zeros.root i).re = 1 / 2 := by
  have hunit := hconverse hpositive
  intro i
  apply re_eq_half_of_reflected_liTransforms_le_one
    (zeros.root_ne_zero i)
  · rw [← hsym.root_partner i]
    exact zeros.root_ne_zero (hsym.partner i)
  · exact hunit i
  · rw [← hsym.root_partner i]
    exact hunit (hsym.partner i)

/-- Reflection symmetry for an arbitrary multiplicity-indexed spectrum. -/
structure ReflectionSymmetricSpectrum {ι : Type*} (root : ι → ℂ) where
  partner : ι ≃ ι
  root_partner : ∀ i, root (partner i) = 1 - root i

def LiUnitDiskBound {ι : Type*} (root : ι → ℂ) : Prop :=
  ∀ i, ‖1 - (root i)⁻¹‖ ≤ 1

theorem re_ge_half_of_liUnitDiskBound
    {ι : Type*} {root : ι → ℂ}
    (hroot0 : ∀ i, root i ≠ 0)
    (hunit : LiUnitDiskBound root) :
    ∀ i, 1 / 2 ≤ (root i).re :=
  fun i ↦ (norm_one_sub_inv_le_one_iff_re_ge_half (hroot0 i)).mp (hunit i)

/-- The algebraic half of the infinite Bombieri--Lagarias reduction:
reflection symmetry turns individual unit-disk bounds into critical-line
location.  No convergence or positivity converse is hidden here. -/
theorem on_criticalLine_of_reflection_of_liUnitDiskBound
    {ι : Type*} {root : ι → ℂ}
    (hroot0 : ∀ i, root i ≠ 0)
    (hsym : ReflectionSymmetricSpectrum root)
    (hunit : LiUnitDiskBound root) :
    ∀ i, (root i).re = 1 / 2 := by
  intro i
  apply re_eq_half_of_reflected_liTransforms_le_one (hroot0 i)
  · rw [← hsym.root_partner i]
    exact hroot0 (hsym.partner i)
  · exact hunit i
  · rw [← hsym.root_partner i]
    exact hunit (hsym.partner i)

/-- Reflection symmetry plus positivity of an unrelated coefficient sequence
does not locate roots.  This counterexample records why the zero-sum/product
identity cannot be omitted from an infinite converse. -/
theorem reflection_and_abstract_positivity_insufficient :
    ∃ (root : Bool → ℂ) (coefficient : ℕ → ℂ)
        (_ : ReflectionSymmetricSpectrum root),
      (∀ i, root i ≠ 0) ∧
      (∀ n : ℕ, 0 < n → 0 ≤ (coefficient n).re) ∧
      ∃ i, (root i).re ≠ 1 / 2 := by
  let root : Bool → ℂ := fun b ↦ if b then -1 else 2
  let partner : Bool ≃ Bool := {
    toFun := (!·)
    invFun := (!·)
    left_inv := fun b ↦ by cases b <;> rfl
    right_inv := fun b ↦ by cases b <;> rfl }
  refine ⟨root, fun _ ↦ 0,
    ⟨partner, fun i ↦ by cases i <;> norm_num [root, partner]⟩,
    ?_, ?_, false, ?_⟩
  · intro i
    cases i <;> simp [root]
  · intro n hn
    norm_num
  · norm_num [root]

theorem FiniteZeroMultiset.liSum_re_nonneg_of_on_criticalLine
    (zeros : FiniteZeroMultiset)
    (hline : ∀ i, (zeros.root i).re = 1 / 2)
    (n : ℕ) :
    0 ≤ (zeros.liSum n).re :=
  zeros.liSum_re_nonneg
    (fun i ↦ (norm_one_sub_inv_eq_one_of_re_eq_half
      (zeros.root_ne_zero i) (hline i)).le) n

/-- RH implies nonnegativity of every finite, multiplicity-aware bounded
window.  No limit or finite-to-infinite extrapolation occurs here. -/
theorem riemannZetaZeroWindow_liSum_re_nonneg_of_rh
    (hRH : KakeyaRiemannHypothesisRoot) (R : ℝ) (n : ℕ) :
    0 ≤ ((riemannZetaZeroWindowMultiset R).liSum n).re :=
  (riemannZetaZeroWindowMultiset R).liSum_re_nonneg_of_on_criticalLine
    (riemannZetaZeroWindowMultiset_on_criticalLine hRH R) n

/-- The exact derivative/zero-window formula still required by the Li route.
It asserts convergence at every positive index; no finite prefix is used. -/
def RiemannLiZeroWindowFormula : Prop :=
  ∀ n : ℕ, 0 < n →
    Tendsto (fun N : ℕ ↦
      (riemannZetaZeroWindowMultiset (N : ℝ)).liSum n) atTop
      (𝓝 (liDerivativeCoefficient riemannXiLi n))

/-- The RH-to-Li-positive direction follows once the all-index zero-window
formula is supplied.  This separates finite RH positivity from the missing
global convergence/coefficient theorem. -/
theorem liPositive_of_rh_of_zeroWindowFormula
    (hRH : KakeyaRiemannHypothesisRoot)
    (hformula : RiemannLiZeroWindowFormula) :
    LiPositive riemannLiCoefficient := by
  intro n hn
  change 0 ≤ (liDerivativeCoefficient riemannXiLi n).re
  have hre : Tendsto (fun N : ℕ ↦
      ((riemannZetaZeroWindowMultiset (N : ℝ)).liSum n).re) atTop
      (𝓝 (liDerivativeCoefficient riemannXiLi n).re) :=
    Complex.continuous_re.continuousAt.tendsto.comp (hformula n hn)
  exact le_of_tendsto_of_tendsto tendsto_const_nhds hre
    (Eventually.of_forall fun N ↦
      riemannZetaZeroWindow_liSum_re_nonneg_of_rh hRH (N : ℝ) n)

theorem riemannLiCriterion_forward_of_zeroWindowFormula
    (hformula : RiemannLiZeroWindowFormula) :
    KakeyaRiemannHypothesisRoot →
      LiPositive riemannLiCoefficient :=
  fun hRH ↦ liPositive_of_rh_of_zeroWindowFormula hRH hformula

/-- The converse all-index theorem remains a distinct source bridge. -/
def RiemannLiConverseStatement : Prop :=
  LiPositive riemannLiCoefficient → KakeyaRiemannHypothesisRoot

theorem riemannLiCriterion_of_bridges
    (hformula : RiemannLiZeroWindowFormula)
    (hconverse : RiemannLiConverseStatement) :
    RiemannLiCriterionStatement :=
  ⟨riemannLiCriterion_forward_of_zeroWindowFormula hformula, hconverse⟩

/-- The analytic convergence part of the height-symmetric zero formula.
Completeness and multiplicity-correctness of `root` are separate obligations,
because pinned Mathlib exposes analytic orders and a zero set, but not one
global multiplicity-indexed symmetric-height enumeration. -/
def HeightSymmetricLiLimit {ι : Type*} [DecidableEq ι]
    (root : ι → ℂ) (cutoff : ℕ → Finset ι) (height : ℕ → ℝ)
    (coefficient : ℕ → ℂ) : Prop :=
  Tendsto height atTop atTop ∧
    (∀ N i, i ∈ cutoff N ↔ |(root i).im| ≤ height N) ∧
    ∀ n : ℕ, Tendsto (fun N ↦ liZeroPartialSum root (cutoff N) n)
      atTop (𝓝 (coefficient n))

/-- Source-standard summability hypothesis for the Bombieri--Lagarias
multiset theorem. -/
def BombieriLagariasSummability {ι : Type*} (root : ι → ℂ) : Prop :=
  Summable (fun i ↦
    (1 + |(root i).re|) / (1 + ‖root i‖) ^ 2)

/-- Reflection symmetry and source summability alone do not imply
critical-line location. -/
theorem reflection_and_summability_insufficient :
    ∃ (root : Bool → ℂ) (_ : ReflectionSymmetricSpectrum root),
      (∀ i, root i ≠ 0) ∧ BombieriLagariasSummability root ∧
        ∃ i, (root i).re ≠ 1 / 2 := by
  let root : Bool → ℂ := fun b ↦ if b then -1 else 2
  let partner : Bool ≃ Bool := {
    toFun := (!·)
    invFun := (!·)
    left_inv := fun b ↦ by cases b <;> rfl
    right_inv := fun b ↦ by cases b <;> rfl }
  refine ⟨root,
    ⟨partner, fun i ↦ by cases i <;> norm_num [root, partner]⟩,
    ?_, ?_, false, ?_⟩
  · intro i
    cases i <;> simp [root]
  · apply summable_of_hasFiniteSupport
    exact Set.toFinite _
  · norm_num [root]

/-- Away from the origin, the source-standard Bombieri--Lagarias weight
implies the inverse-square summability required by genus-one products. -/
theorem BombieriLagariasSummability.summable_norm_inv_sq
    {ι : Type*} {root : ι → ℂ}
    (hsum : BombieriLagariasSummability root)
    (hlarge : ∀ i, 1 ≤ ‖root i‖) :
    Summable (fun i ↦ ‖(root i)⁻¹‖ ^ 2) := by
  apply (hsum.mul_left 4).of_nonneg_of_le
  · intro i
    positivity
  · intro i
    rw [norm_inv]
    let r := ‖root i‖
    have hr : 0 < r := lt_of_lt_of_le zero_lt_one (hlarge i)
    have hden : 0 < (1 + r) ^ 2 := by positivity
    calc
      r⁻¹ ^ 2 = 1 / r ^ 2 := by rw [inv_pow, one_div]
      _ ≤ 4 / (1 + r) ^ 2 := by
        rw [div_le_div_iff₀ (by positivity) hden]
        nlinarith [hlarge i]
      _ ≤ 4 * ((1 + |(root i).re|) / (1 + r) ^ 2) := by
        rw [← mul_div_assoc]
        have hnum : (4 : ℝ) ≤ 4 * (1 + |(root i).re|) := by
          nlinarith [abs_nonneg (root i).re]
        exact div_le_div_of_nonneg_right hnum hden.le

/-- Exact infinite spectral implication supplied by the Bombieri--Lagarias
converse after its multiset summability hypothesis is instantiated.  Growth
enters separately when proving `HeightSymmetricLiLimit`. -/
def HeightSymmetricLiSpectralConverseStatement
    {ι : Type*} [DecidableEq ι]
    (root : ι → ℂ) (cutoff : ℕ → Finset ι) (height : ℕ → ℝ)
    (coefficient : ℕ → ℂ) : Prop :=
  BombieriLagariasSummability root →
    HeightSymmetricLiLimit root cutoff height coefficient →
    (∀ n : ℕ, 0 < n → 0 ≤ (coefficient n).re) →
      LiUnitDiskBound root

/-- Source-shaped half-plane form of the Bombieri--Lagarias implication.
For nonzero roots this is exactly the unit-disk formulation above, not an
additional assumption. -/
def BombieriLagariasPositivityHalfPlaneStatement
    {ι : Type*} [DecidableEq ι]
    (root : ι → ℂ) (cutoff : ℕ → Finset ι) (height : ℕ → ℝ)
    (coefficient : ℕ → ℂ) : Prop :=
  BombieriLagariasSummability root →
    HeightSymmetricLiLimit root cutoff height coefficient →
    (∀ n : ℕ, 0 < n → 0 ≤ (coefficient n).re) →
      ∀ i, 1 / 2 ≤ (root i).re

theorem heightSymmetricLiSpectralConverse_iff_halfPlane
    {ι : Type*} [DecidableEq ι]
    {root : ι → ℂ} {cutoff : ℕ → Finset ι} {height : ℕ → ℝ}
    {coefficient : ℕ → ℂ}
    (hroot0 : ∀ i, root i ≠ 0) :
    HeightSymmetricLiSpectralConverseStatement
        root cutoff height coefficient ↔
      BombieriLagariasPositivityHalfPlaneStatement
        root cutoff height coefficient := by
  constructor
  · intro h hsum hlimit hpositive i
    exact (norm_one_sub_inv_le_one_iff_re_ge_half (hroot0 i)).mp
      (h hsum hlimit hpositive i)
  · intro h hsum hlimit hpositive i
    exact (norm_one_sub_inv_le_one_iff_re_ge_half (hroot0 i)).mpr
      (h hsum hlimit hpositive i)

/-- Once the analytic symmetric limit and the infinite spectral implication
are available, reflection symmetry completes the converse algebraically. -/
theorem on_criticalLine_of_heightSymmetric_spectralConverse
    {ι : Type*} [DecidableEq ι]
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
  on_criticalLine_of_reflection_of_liUnitDiskBound
    hroot0 hsym (hconverse hsum hlimit hpositive)

/-- Complete spectral data needed to specialize the abstract
Bombieri--Lagarias implication to the normalized Riemann xi spectrum. -/
structure RiemannLiSpectralModel (ι : Type*) [DecidableEq ι] where
  root : ι → ℂ
  cutoff : ℕ → Finset ι
  height : ℕ → ℝ
  root_ne_zero : ∀ i, root i ≠ 0
  reflection : ReflectionSymmetricSpectrum root
  summability : BombieriLagariasSummability root
  liLimit : HeightSymmetricLiLimit root cutoff height
    (fun n ↦ liDerivativeCoefficient riemannXiLi n)
  complete : ∀ rho, IsNontrivialRiemannZetaZero rho →
    ∃ i, root i = rho

/-- Once a complete xi spectral model and the genuine Bombieri--Lagarias
positivity implication are available, reflection derives RH.  No RH
assumption occurs in this reduction. -/
theorem riemannHypothesis_of_bombieriLagarias
    {ι : Type*} [DecidableEq ι]
    (model : RiemannLiSpectralModel ι)
    (hpositive : LiPositive riemannLiCoefficient)
    (hBL : BombieriLagariasPositivityHalfPlaneStatement
      model.root model.cutoff model.height
        (fun n ↦ liDerivativeCoefficient riemannXiLi n)) :
    KakeyaRiemannHypothesisRoot := by
  have hcoeff : ∀ n : ℕ, 0 < n →
      0 ≤ (liDerivativeCoefficient riemannXiLi n).re := by
    intro n hn
    exact hpositive n hn
  have hright := hBL model.summability model.liLimit hcoeff
  have hunit : LiUnitDiskBound model.root := fun i ↦
    (norm_one_sub_inv_le_one_iff_re_ge_half
      (model.root_ne_zero i)).mpr (hright i)
  have hline := on_criticalLine_of_reflection_of_liUnitDiskBound
    model.root_ne_zero model.reflection hunit
  intro rho hz hnontrivial h1
  obtain ⟨i, hi⟩ := model.complete rho ⟨hz, hnontrivial⟩
  rw [← hi]
  exact hline i

/-- Cauchy form of the analytic convergence obligation.  This does not assert
that zeta zeros satisfy it. -/
def HeightSymmetricLiCauchy {ι : Type*} [DecidableEq ι]
    (root : ι → ℂ) (cutoff : ℕ → Finset ι) : Prop :=
  ∀ n : ℕ, CauchySeq (fun N ↦ liZeroPartialSum root (cutoff N) n)

/-- Exact cofinal exhaustion by symmetric heights, separated from any
coefficient limit. -/
def SymmetricHeightExhaustion {ι : Type*} [DecidableEq ι]
    (root : ι → ℂ) (cutoff : ℕ → Finset ι) (height : ℕ → ℝ) : Prop :=
  Tendsto height atTop atTop ∧
    ∀ N i, i ∈ cutoff N ↔ |(root i).im| ≤ height N

theorem SymmetricHeightExhaustion.eventually_subset
    {ι : Type*} [DecidableEq ι]
    {root : ι → ℂ} {cutoff : ℕ → Finset ι} {height : ℕ → ℝ}
    (h : SymmetricHeightExhaustion root cutoff height)
    (F : Finset ι) :
    ∀ᶠ N in atTop, F ⊆ cutoff N := by
  change ∀ᶠ N in atTop, ∀ i ∈ F, i ∈ cutoff N
  apply (Finset.eventually_all F).2
  intro i _
  filter_upwards [h.1.eventually (eventually_ge_atTop |(root i).im|)] with N hN
  exact (h.2 N i).2 hN

/-- Exact symmetric windows are cofinal in the directed set of all finite
subsets. -/
theorem SymmetricHeightExhaustion.tendsto_cutoff
    {ι : Type*} [DecidableEq ι]
    {root : ι → ℂ} {cutoff : ℕ → Finset ι} {height : ℕ → ℝ}
    (h : SymmetricHeightExhaustion root cutoff height) :
    Tendsto cutoff atTop atTop := by
  apply tendsto_atTop.2
  intro F
  exact h.eventually_subset F

/-- Unconditional summability is sufficient to identify every exact
symmetric-height limit with the corresponding `tsum`. -/
theorem SymmetricHeightExhaustion.tendsto_liZeroPartialSum
    {ι : Type*} [DecidableEq ι]
    {root : ι → ℂ} {cutoff : ℕ → Finset ι} {height : ℕ → ℝ}
    (h : SymmetricHeightExhaustion root cutoff height)
    (n : ℕ) (hsum : Summable (fun i ↦ liZeroSummand n (root i))) :
    Tendsto (fun N ↦ liZeroPartialSum root (cutoff N) n) atTop
      (𝓝 (∑' i, liZeroSummand n (root i))) :=
  hsum.hasSum.comp h.tendsto_cutoff

theorem heightSymmetricLiLimit_tsum_of_summable
    {ι : Type*} [DecidableEq ι]
    (root : ι → ℂ) (cutoff : ℕ → Finset ι) (height : ℕ → ℝ)
    (hexhaust : SymmetricHeightExhaustion root cutoff height)
    (hsum : ∀ n, Summable (fun i ↦ liZeroSummand n (root i))) :
    HeightSymmetricLiLimit root cutoff height
      (fun n ↦ ∑' i, liZeroSummand n (root i)) :=
  ⟨hexhaust.1, hexhaust.2,
    fun n ↦ hexhaust.tendsto_liZeroPartialSum n (hsum n)⟩

/-- A summable bound on successive height shells gives the Cauchy estimate
needed for conditional symmetric sums.  Zero-counting and growth estimates
enter only through `d` and `hbound`. -/
theorem cauchySeq_liZeroPartialSum_of_shell_majorant
    {ι : Type*} [DecidableEq ι]
    (root : ι → ℂ) (cutoff : ℕ → Finset ι) (n : ℕ)
    (d : ℕ → ℝ) (hd : Summable d)
    (hbound : ∀ N, dist (liZeroPartialSum root (cutoff N) n)
      (liZeroPartialSum root (cutoff (N + 1)) n) ≤ d N) :
    CauchySeq (fun N ↦ liZeroPartialSum root (cutoff N) n) := by
  apply cauchySeq_of_summable_dist
  apply hd.of_norm_bounded
  intro N
  simpa [Real.norm_eq_abs, abs_of_nonneg dist_nonneg] using hbound N

theorem heightSymmetricLiCauchy_of_shell_majorants
    {ι : Type*} [DecidableEq ι]
    (root : ι → ℂ) (cutoff : ℕ → Finset ι)
    (d : ℕ → ℕ → ℝ)
    (hd : ∀ n, Summable (d n))
    (hbound : ∀ n N, dist (liZeroPartialSum root (cutoff N) n)
      (liZeroPartialSum root (cutoff (N + 1)) n) ≤ d n N) :
    HeightSymmetricLiCauchy root cutoff :=
  fun n ↦ cauchySeq_liZeroPartialSum_of_shell_majorant
    root cutoff n (d n) (hd n) (hbound n)

/-- The quantitative shell estimate sufficient for symmetric Li convergence.
The exponent `3/2` is deliberately weaker than the source combination
`O(log T)` zero count times `O(T⁻²)` conjugate-pair cancellation. -/
def ThreeHalvesLiShellBound {ι : Type*} [DecidableEq ι]
    (root : ι → ℂ) (cutoff : ℕ → Finset ι) (C : ℕ → ℝ) : Prop :=
  ∀ n N, dist (liZeroPartialSum root (cutoff N) n)
    (liZeroPartialSum root (cutoff (N + 1)) n) ≤
      C n * ((N + 1 : ℕ) : ℝ) ^ (-(3 / 2 : ℝ))

theorem summable_threeHalves_shell_majorant (C : ℝ) :
    Summable
      (fun N : ℕ ↦ C * ((N + 1 : ℕ) : ℝ) ^ (-(3 / 2 : ℝ))) := by
  apply Summable.mul_left
  exact (summable_nat_add_iff 1).2
    (Real.summable_nat_rpow.mpr (by norm_num))

theorem heightSymmetricLiCauchy_of_threeHalvesShellBound
    {ι : Type*} [DecidableEq ι]
    (root : ι → ℂ) (cutoff : ℕ → Finset ι) (C : ℕ → ℝ)
    (hbound : ThreeHalvesLiShellBound root cutoff C) :
    HeightSymmetricLiCauchy root cutoff := by
  apply heightSymmetricLiCauchy_of_shell_majorants root cutoff
    (fun n N ↦ C n * ((N + 1 : ℕ) : ℝ) ^ (-(3 / 2 : ℝ)))
  · exact fun n ↦ summable_threeHalves_shell_majorant (C n)
  · exact hbound

/-- Source-shaped unit-height zero count.  Riemann--von Mangoldt gives this
with `O(log T)`, counting multiplicity; it is a theorem statement here
because pinned Mathlib has no quantitative zeta zero-count theorem. -/
def LogarithmicZeroShellCount {ι : Type*} [DecidableEq ι]
    (cutoff : ℕ → Finset ι) : Prop :=
  ∃ A : ℝ, 0 ≤ A ∧ ∀ N,
    (((cutoff (N + 1) \ cutoff N).card : ℕ) : ℝ) ≤
      A * Real.log (N + 2)

/-- The second estimate needed after zero counting: conjugate/reflected
pairing must improve each unit shell from first-order `O(T⁻¹)` behavior to
quadratic `O(T⁻²)` behavior. -/
def QuadraticPairedLiShellCancellation {ι : Type*} [DecidableEq ι]
    (root : ι → ℂ) (cutoff : ℕ → Finset ι) : Prop :=
  ∀ n, ∃ B : ℝ, 0 ≤ B ∧ ∀ N,
    dist (liZeroPartialSum root (cutoff N) n)
      (liZeroPartialSum root (cutoff (N + 1)) n) ≤
        B * Real.log (N + 2) / (((N + 2 : ℕ) : ℝ) ^ 2)

theorem summable_log_div_square_shell_majorant
    (B : ℝ) (hB : 0 ≤ B) :
    Summable (fun N : ℕ ↦
      B * Real.log (N + 2) / (((N + 2 : ℕ) : ℝ) ^ 2)) := by
  have hbase : Summable
      (fun N : ℕ ↦ 2 * B *
        (((N + 2 : ℕ) : ℝ) ^ (-(3 / 2 : ℝ)))) := by
    apply Summable.mul_left
    exact (summable_nat_add_iff 2).2
      (Real.summable_nat_rpow.mpr (by norm_num))
  apply hbase.of_nonneg_of_le
  · intro N
    exact div_nonneg
      (mul_nonneg hB (Real.log_nonneg (by norm_cast; omega)))
      (sq_nonneg _)
  · intro N
    norm_num [Nat.cast_add]
    let x : ℝ := (N : ℝ) + 2
    have hx : 0 < x := by
      dsimp [x]
      positivity
    change B * Real.log x / x ^ 2 ≤ _
    calc
      B * Real.log x / x ^ 2 ≤
          B * (x ^ (1 / 2 : ℝ) / (1 / 2)) / x ^ 2 := by
        exact div_le_div_of_nonneg_right
          (mul_le_mul_of_nonneg_left
            (Real.log_le_rpow_div (x := x) (ε := (1 / 2 : ℝ))
              hx.le (by norm_num)) hB)
          (sq_nonneg x)
      _ = 2 * B * x ^ (-(3 / 2 : ℝ)) := by
        rw [div_eq_iff (pow_ne_zero 2 hx.ne')]
        rw [mul_assoc, ← Real.rpow_natCast, ← Real.rpow_add hx]
        norm_num
        ring

theorem summable_log_add_two_div_add_one_square
    (A : ℝ) (hA : 0 ≤ A) :
    Summable (fun N : ℕ ↦
      A * Real.log (N + 2) / (((N + 1 : ℕ) : ℝ) ^ 2)) := by
  have hbase :=
    summable_log_div_square_shell_majorant (4 * A) (by positivity)
  apply hbase.of_nonneg_of_le
  · intro N
    exact div_nonneg
      (mul_nonneg hA (Real.log_nonneg (by norm_cast; omega)))
      (sq_nonneg _)
  · intro N
    have hN1 : 0 < ((N + 1 : ℕ) : ℝ) := by positivity
    have hN2 : 0 < ((N + 2 : ℕ) : ℝ) := by positivity
    have hlog : 0 ≤ Real.log ((N + 2 : ℕ) : ℝ) :=
      Real.log_nonneg (by norm_cast; omega)
    norm_num [Nat.cast_add] at *
    rw [div_le_div_iff₀ (sq_pos_of_pos hN1) (sq_pos_of_pos hN2)]
    have hsquare : ((N + 2 : ℕ) : ℝ) ^ 2 ≤
        4 * ((N + 1 : ℕ) : ℝ) ^ 2 := by
      norm_num [Nat.cast_add]
      nlinarith
    have hK : 0 ≤ A * Real.log ((N : ℝ) + 2) :=
      mul_nonneg hA hlog
    calc
      A * Real.log ((N : ℝ) + 2) * ((N : ℝ) + 2) ^ 2
          ≤ A * Real.log ((N : ℝ) + 2) *
              (4 * ((N : ℝ) + 1) ^ 2) := by
            apply mul_le_mul_of_nonneg_left _ hK
            simpa [Nat.cast_add] using hsquare
      _ = 4 * A * Real.log ((N : ℝ) + 2) *
          ((N : ℝ) + 1) ^ 2 := by ring

/-- A sourced `O(log N)` unit-shell zero count implies inverse-square
summability. Multiplicity is represented by shell membership, not erased. -/
theorem summable_norm_inv_sq_of_logarithmic_shell_count
    {ι : Type*} [DecidableEq ι] (root : ι → ℂ)
    (shell : ℕ → Finset ι)
    (hpartition : ∀ i, ∃! N, i ∈ shell N)
    (hlower : ∀ N i, i ∈ shell N →
      ((N + 1 : ℕ) : ℝ) ≤ ‖root i‖)
    (A : ℝ) (hA : 0 ≤ A)
    (hcount : ∀ N, ((shell N).card : ℝ) ≤
      A * Real.log (N + 2)) :
    Summable (fun i ↦ ‖(root i)⁻¹‖ ^ 2) := by
  rw [summable_partition (f := fun i ↦ ‖(root i)⁻¹‖ ^ 2)
    (fun i ↦ sq_nonneg _) (s := fun N ↦ ↑(shell N))
    (by simpa using hpartition)]
  constructor
  · intro N
    apply summable_of_hasFiniteSupport
    exact Set.toFinite _
  · apply (summable_log_add_two_div_add_one_square A hA).of_nonneg_of_le
    · intro N
      exact tsum_nonneg fun _ ↦ sq_nonneg _
    · intro N
      rw [tsum_fintype]
      calc
        ∑ i : ↥(↑(shell N) : Set ι), ‖(root i)⁻¹‖ ^ 2
            ≤ ∑ _i : ↥(↑(shell N) : Set ι),
                1 / (((N + 1 : ℕ) : ℝ) ^ 2) := by
          apply Finset.sum_le_sum
          intro i hi
          rw [norm_inv, inv_pow]
          have hsquareRoot :
              (((N + 1 : ℕ) : ℝ) ^ 2) ≤ ‖root i‖ ^ 2 :=
            (sq_le_sq₀ (by positivity) (norm_nonneg _)).mpr
              (hlower N i i.property)
          simpa [one_div] using
            (inv_anti₀ (by positivity) hsquareRoot)
        _ = ((shell N).card : ℝ) /
            (((N + 1 : ℕ) : ℝ) ^ 2) := by
          simp [div_eq_mul_inv]
        _ ≤ A * Real.log (N + 2) /
            (((N + 1 : ℕ) : ℝ) ^ 2) := by
          exact div_le_div_of_nonneg_right (hcount N) (sq_nonneg _)

/-- Finitely many low zeros never affect inverse-square summability. -/
theorem summable_norm_inv_sq_of_finite_low_and_logarithmic_shell_count
    {ι : Type*} [DecidableEq ι] (root : ι → ℂ)
    (low : Finset ι)
    (shell : ℕ → Finset {i : ι // i ∉ low})
    (hpartition : ∀ i : {i : ι // i ∉ low},
      ∃! N, i ∈ shell N)
    (hlower : ∀ N i, i ∈ shell N →
      ((N + 1 : ℕ) : ℝ) ≤ ‖root i‖)
    (A : ℝ) (hA : 0 ≤ A)
    (hcount : ∀ N, ((shell N).card : ℝ) ≤
      A * Real.log (N + 2)) :
    Summable (fun i ↦ ‖(root i)⁻¹‖ ^ 2) := by
  have hc : Summable (fun i : {i : ι // i ∉ low} ↦
      ‖(root i)⁻¹‖ ^ 2) :=
    summable_norm_inv_sq_of_logarithmic_shell_count
      (fun i : {i : ι // i ∉ low} ↦ root i)
      shell hpartition hlower A hA hcount
  exact (low.summable_compl_iff).mp hc

/-- The dyadic majorant arising from an order-one cumulative zero count. -/
theorem summable_dyadic_linear_majorant (A : ℝ) :
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

theorem tsum_nat_add_one_mul_half_pow :
    (∑' m : ℕ, (m + 1 : ℝ) * (1 / 2 : ℝ) ^ m) = 4 := by
  rw [show (fun m : ℕ ↦ (m + 1 : ℝ) * (1 / 2 : ℝ) ^ m) =
      (fun m : ℕ ↦ (m : ℝ) * (1 / 2 : ℝ) ^ m +
        (1 / 2 : ℝ) ^ m) by
    funext m
    ring]
  have h1 : Summable (fun m : ℕ ↦
      (m : ℝ) * (1 / 2 : ℝ) ^ m) := by
    simpa using
      (summable_pow_mul_geometric_of_norm_lt_one (R := ℝ) 1
        (r := (1 / 2 : ℝ)) (by norm_num))
  have h2 : Summable (fun m : ℕ ↦ (1 / 2 : ℝ) ^ m) :=
    summable_geometric_of_norm_lt_one
      (x := (1 / 2 : ℝ)) (by norm_num)
  rw [h1.tsum_add h2]
  rw [tsum_coe_mul_geometric_of_norm_lt_one (by norm_num),
    tsum_geometric_two]
  norm_num

/-- Explicit tail bound for the weighted dyadic majorant. -/
theorem tsum_shifted_dyadic_linear_le (N : ℕ) :
    (∑' m : ℕ,
      (N + m + 1 : ℝ) / (2 : ℝ) ^ (N + m)) ≤
      4 * (N + 1 : ℝ) / (2 : ℝ) ^ N := by
  have hsum : Summable (fun m : ℕ ↦
      ((N + 1 : ℝ) / (2 : ℝ) ^ N) *
        ((m + 1 : ℝ) * (1 / 2 : ℝ) ^ m)) :=
    ((summable_dyadic_linear_majorant 1).congr
      (fun m ↦ by ring)).mul_left
        ((N + 1 : ℝ) / (2 : ℝ) ^ N)
  have hle : ∀ m : ℕ,
      (N + m + 1 : ℝ) / (2 : ℝ) ^ (N + m) ≤
        ((N + 1 : ℝ) / (2 : ℝ) ^ N) *
          ((m + 1 : ℝ) * (1 / 2 : ℝ) ^ m) := by
    intro m
    rw [show ((N + 1 : ℝ) / (2 : ℝ) ^ N) *
          ((m + 1 : ℝ) * (1 / 2 : ℝ) ^ m) =
        ((N + 1 : ℝ) * (m + 1 : ℝ)) /
          (2 : ℝ) ^ (N + m) by
      rw [pow_add, one_div, inv_pow]
      field_simp]
    apply div_le_div_of_nonneg_right
    · nlinarith
    · positivity
  have hleft : Summable (fun m : ℕ ↦
      (N + m + 1 : ℝ) / (2 : ℝ) ^ (N + m)) :=
    hsum.of_nonneg_of_le
      (fun m ↦ div_nonneg (by positivity) (by positivity)) hle
  calc
    (∑' m : ℕ, (N + m + 1 : ℝ) / (2 : ℝ) ^ (N + m))
        ≤ ∑' m : ℕ, ((N + 1 : ℝ) / (2 : ℝ) ^ N) *
          ((m + 1 : ℝ) * (1 / 2 : ℝ) ^ m) :=
      Summable.tsum_le_tsum hle hleft hsum
    _ = ((N + 1 : ℝ) / (2 : ℝ) ^ N) * 4 := by
      rw [tsum_mul_left, tsum_nat_add_one_mul_half_pow]
    _ = 4 * (N + 1 : ℝ) / (2 : ℝ) ^ N := by ring

/-- Quantitative inverse-square tail estimate from a dyadic
`O(2^k(k+1))` shell count. -/
theorem tsum_norm_inv_sq_dyadic_shell_tail_le
    {ι : Type*} [DecidableEq ι] (root : ι → ℂ)
    (shell : ℕ → Finset ι)
    (hlower : ∀ k i, i ∈ shell k →
      (2 : ℝ) ^ k ≤ ‖root i‖)
    (A : ℝ) (hA : 0 ≤ A)
    (hcount : ∀ k, ((shell k).card : ℝ) ≤
      A * (2 : ℝ) ^ k * (k + 1 : ℝ))
    (N : ℕ) :
    (∑' m : ℕ, ∑ i ∈ shell (N + m),
      ‖(root i)⁻¹‖ ^ 2) ≤
      4 * A * (N + 1 : ℝ) / (2 : ℝ) ^ N := by
  have hterm : ∀ m : ℕ,
      (∑ i ∈ shell (N + m), ‖(root i)⁻¹‖ ^ 2) ≤
        A * ((N + m + 1 : ℝ) / (2 : ℝ) ^ (N + m)) := by
    intro m
    let k := N + m
    have hpow : 0 < (2 : ℝ) ^ k := by positivity
    calc
      (∑ i ∈ shell k, ‖(root i)⁻¹‖ ^ 2)
          ≤ ∑ _i ∈ shell k,
              1 / ((2 : ℝ) ^ k) ^ 2 := by
        apply Finset.sum_le_sum
        intro i hi
        rw [norm_inv, inv_pow]
        have hsquare :
            ((2 : ℝ) ^ k) ^ 2 ≤ ‖root i‖ ^ 2 :=
          (sq_le_sq₀ hpow.le (norm_nonneg _)).mpr
            (hlower k i hi)
        simpa [one_div] using
          inv_anti₀ (sq_pos_of_pos hpow) hsquare
      _ = ((shell k).card : ℝ) /
          ((2 : ℝ) ^ k) ^ 2 := by
        simp [div_eq_mul_inv]
      _ ≤ (A * (2 : ℝ) ^ k * (k + 1 : ℝ)) /
          ((2 : ℝ) ^ k) ^ 2 :=
        div_le_div_of_nonneg_right (hcount k) (sq_nonneg _)
      _ = A * ((k + 1 : ℝ) / (2 : ℝ) ^ k) := by
        field_simp
      _ = A * ((N + m + 1 : ℝ) /
          (2 : ℝ) ^ (N + m)) := by
        simp [k, Nat.cast_add]
  have hbase : Summable (fun m : ℕ ↦
      (N + m + 1 : ℝ) / (2 : ℝ) ^ (N + m)) := by
    have hmajor : Summable (fun m : ℕ ↦
        ((N + 1 : ℝ) / (2 : ℝ) ^ N) *
          ((m + 1 : ℝ) * (1 / 2 : ℝ) ^ m)) :=
      ((summable_dyadic_linear_majorant 1).congr
        (fun m ↦ by ring)).mul_left
          ((N + 1 : ℝ) / (2 : ℝ) ^ N)
    apply hmajor.of_nonneg_of_le
    · intro m
      positivity
    · intro m
      rw [show ((N + 1 : ℝ) / (2 : ℝ) ^ N) *
            ((m + 1 : ℝ) * (1 / 2 : ℝ) ^ m) =
          ((N + 1 : ℝ) * (m + 1 : ℝ)) /
            (2 : ℝ) ^ (N + m) by
        rw [pow_add, one_div, inv_pow]
        field_simp]
      apply div_le_div_of_nonneg_right
      · nlinarith
      · positivity
  have hright : Summable (fun m : ℕ ↦
      A * ((N + m + 1 : ℝ) /
        (2 : ℝ) ^ (N + m))) :=
    hbase.mul_left A
  have hleft : Summable (fun m : ℕ ↦
      ∑ i ∈ shell (N + m), ‖(root i)⁻¹‖ ^ 2) :=
    hright.of_nonneg_of_le
      (fun m ↦ Finset.sum_nonneg fun _ _ ↦ sq_nonneg _) hterm
  calc
    (∑' m : ℕ, ∑ i ∈ shell (N + m),
        ‖(root i)⁻¹‖ ^ 2)
        ≤ ∑' m : ℕ,
          A * ((N + m + 1 : ℝ) /
            (2 : ℝ) ^ (N + m)) :=
      Summable.tsum_le_tsum hterm hleft hright
    _ = A * (∑' m : ℕ,
        (N + m + 1 : ℝ) / (2 : ℝ) ^ (N + m)) := by
      rw [tsum_mul_left]
    _ ≤ A * (4 * (N + 1 : ℝ) / (2 : ℝ) ^ N) := by
      gcongr
      exact tsum_shifted_dyadic_linear_le N
    _ = 4 * A * (N + 1 : ℝ) / (2 : ℝ) ^ N := by
      ring

/-- Every real radius at least one belongs to a unique half-open dyadic
interval.  This is the bookkeeping fact needed to turn a cumulative Jensen
count into disjoint shells without making a false unit-shell estimate. -/
theorem existsUnique_dyadic_interval (x : ℝ) (hx : 1 ≤ x) :
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

/-- Jensen's order-one cumulative bound need not imply an `O(log N)`
unit-shell count.  The correct dyadic consequence still suffices:
`O(2^N (N+1))` roots above radius `2^N` give a summable
`O((N+1)/2^N)` inverse-square contribution. -/
theorem summable_norm_inv_sq_of_dyadic_linear_log_shell_count
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
  · apply (summable_dyadic_linear_majorant A).of_nonneg_of_le
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

/-- Finite low roots can be removed before applying the Jensen-compatible
dyadic shell estimate and restored afterward. -/
theorem summable_norm_inv_sq_of_finite_low_and_dyadic_linear_log_shell_count
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
    summable_norm_inv_sq_of_dyadic_linear_log_shell_count
      (fun i : {i : ι // i ∉ low} ↦ root i)
      shell hpartition hlower A hcount
  exact (low.summable_compl_iff).mp hc

/-- The half-open dyadic shell of xi multiplicity indices, after removing
an arbitrary finite low set. -/
noncomputable def riemannXiDyadicZeroShell
    (low : Finset RiemannXiZeroIndex) (N : ℕ) :
    Finset {i : RiemannXiZeroIndex // i ∉ low} := by
  classical
  exact Finset.subtype (fun i ↦ i ∉ low)
    ((riemannXiZeroIndexWindow ((2 : ℝ) ^ (N + 1))).filter
      fun i ↦ (2 : ℝ) ^ N ≤ ‖riemannXiZeroRoot i‖ ∧
        ‖riemannXiZeroRoot i‖ < (2 : ℝ) ^ (N + 1))

@[simp]
theorem mem_riemannXiDyadicZeroShell
    (low : Finset RiemannXiZeroIndex) (N : ℕ)
    (i : {i : RiemannXiZeroIndex // i ∉ low}) :
    i ∈ riemannXiDyadicZeroShell low N ↔
      (2 : ℝ) ^ N ≤ ‖riemannXiZeroRoot i‖ ∧
        ‖riemannXiZeroRoot i‖ < (2 : ℝ) ^ (N + 1) := by
  classical
  simp [riemannXiDyadicZeroShell]
  intro _ h
  exact h.le

/-- Removing the radius-`R` window leaves a genuine disjoint dyadic
partition, including all multiplicity indices exactly once. -/
theorem riemannXiDyadicZeroShell_partition
    (R : ℝ) (hR : 1 ≤ R) :
    ∀ i : {i : RiemannXiZeroIndex //
      i ∉ riemannXiZeroIndexWindow R},
      ∃! N, i ∈
        riemannXiDyadicZeroShell (riemannXiZeroIndexWindow R) N := by
  intro i
  have hiR : R < ‖riemannXiZeroRoot i‖ := by
    simpa using i.property
  obtain ⟨N, hN, huniq⟩ :=
    existsUnique_dyadic_interval ‖riemannXiZeroRoot i‖
      (hR.trans hiR.le)
  refine ⟨N, (mem_riemannXiDyadicZeroShell _ _ _).mpr hN, ?_⟩
  intro M hM
  exact huniq M ((mem_riemannXiDyadicZeroShell _ _ _).mp hM)

theorem riemannXiDyadicZeroShell_card_le_window
    (low : Finset RiemannXiZeroIndex) (N : ℕ) :
    (riemannXiDyadicZeroShell low N).card ≤
      (riemannXiZeroIndexWindow ((2 : ℝ) ^ (N + 1))).card := by
  classical
  unfold riemannXiDyadicZeroShell
  rw [Finset.card_subtype]
  exact (Finset.card_filter_le _ _).trans
    (Finset.card_filter_le _ _)

/-- Exact dyadic cardinality consequence of a cumulative
`A₀ r log(2r+2)` multiplicity count. -/
theorem riemannXiDyadicZeroShell_card_le_of_cumulative
    {A₀ R : ℝ} (hA₀ : 0 ≤ A₀)
    (hcum : ∀ r : ℝ, R ≤ r →
      ((riemannXiZeroIndexWindow r).card : ℝ) ≤
        A₀ * r * Real.log (2 * r + 2)) :
    ∀ N,
      ((riemannXiDyadicZeroShell
        (riemannXiZeroIndexWindow R) N).card : ℝ) ≤
        (6 * A₀ * Real.log 2) * (2 : ℝ) ^ N *
          (N + 1 : ℝ) := by
  intro N
  let r : ℝ := (2 : ℝ) ^ (N + 1)
  by_cases hr : R ≤ r
  · have hcard :
        ((riemannXiDyadicZeroShell
          (riemannXiZeroIndexWindow R) N).card : ℝ) ≤
          ((riemannXiZeroIndexWindow r).card : ℝ) := by
      dsimp [r]
      exact_mod_cast riemannXiDyadicZeroShell_card_le_window
        (riemannXiZeroIndexWindow R) N
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
      ((riemannXiDyadicZeroShell
        (riemannXiZeroIndexWindow R) N).card : ℝ)
          ≤ A₀ * r * Real.log (2 * r + 2) := hcum'
      _ ≤ A₀ * r * (3 * (N + 1 : ℝ) * Real.log 2) := by
        gcongr
      _ = (6 * A₀ * Real.log 2) * (2 : ℝ) ^ N *
          (N + 1 : ℝ) := by
        dsimp [r]
        rw [pow_succ]
        ring
  · have hempty :
        riemannXiDyadicZeroShell
          (riemannXiZeroIndexWindow R) N = ∅ := by
      apply Finset.eq_empty_iff_forall_notMem.mpr
      intro i hi
      have hiShell :=
        (mem_riemannXiDyadicZeroShell _ _ _).mp hi
      have hiR : R < ‖riemannXiZeroRoot i‖ := by
        simpa using i.property
      exact hr (hiR.le.trans hiShell.2.le)
    rw [hempty]
    simp
    positivity

/-- The xi dyadic shell tail has the quantitative source-standard
`O((N+1)/2^N)` inverse-square bound, with multiplicities and a finite low
window handled explicitly. -/
theorem exists_riemannXiDyadicZeroShell_inv_sq_tail_bound :
    ∃ A R : ℝ, 0 ≤ A ∧ 1 ≤ R ∧ ∀ N : ℕ,
      (∑' m : ℕ,
        ∑ i ∈ riemannXiDyadicZeroShell
          (riemannXiZeroIndexWindow R) (N + m),
          ‖(riemannXiZeroRoot i)⁻¹‖ ^ 2) ≤
        A * (N + 1 : ℝ) / (2 : ℝ) ^ N := by
  classical
  obtain ⟨A₀, R, hA₀, hR, hcum⟩ :=
    exists_riemannXiZeroIndexWindow_card_le_orderOne
  let low := riemannXiZeroIndexWindow R
  let shell := riemannXiDyadicZeroShell low
  let A := 24 * A₀ * Real.log 2
  refine ⟨A, R, ?_, hR, ?_⟩
  · exact mul_nonneg
      (mul_nonneg (by norm_num) hA₀)
      (Real.log_nonneg (by norm_num))
  · intro N
    have htail :=
      tsum_norm_inv_sq_dyadic_shell_tail_le
        (fun i : {i : RiemannXiZeroIndex // i ∉ low} ↦
          riemannXiZeroRoot i)
        shell
        (fun k i hi ↦
          ((mem_riemannXiDyadicZeroShell low k i).mp hi).1)
        (6 * A₀ * Real.log 2)
        (mul_nonneg
          (mul_nonneg (by norm_num) hA₀)
          (Real.log_nonneg (by norm_num)))
        (fun k ↦ by
          dsimp [shell, low]
          exact riemannXiDyadicZeroShell_card_le_of_cumulative
            hA₀ hcum k)
        N
    dsimp [shell, low, A] at *
    convert htail using 1
    ring

/-- Boundary-safe dyadic decomposition of the complement of the closed
radius-`2^N` xi-zero window. The strict first inequality excludes roots on
the truncation boundary, while the dyadic intervals remain half-open. -/
noncomputable def riemannXiDyadicZeroTailShell (N m : ℕ) :
    Finset RiemannXiZeroIndex := by
  classical
  exact
    (riemannXiZeroIndexWindow ((2 : ℝ) ^ (N + m + 1))).filter
      fun i ↦ (2 : ℝ) ^ N < ‖riemannXiZeroRoot i‖ ∧
        (2 : ℝ) ^ (N + m) ≤ ‖riemannXiZeroRoot i‖ ∧
        ‖riemannXiZeroRoot i‖ < (2 : ℝ) ^ (N + m + 1)

@[simp]
theorem mem_riemannXiDyadicZeroTailShell
    (N m : ℕ) (i : RiemannXiZeroIndex) :
    i ∈ riemannXiDyadicZeroTailShell N m ↔
      (2 : ℝ) ^ N < ‖riemannXiZeroRoot i‖ ∧
      (2 : ℝ) ^ (N + m) ≤ ‖riemannXiZeroRoot i‖ ∧
      ‖riemannXiZeroRoot i‖ < (2 : ℝ) ^ (N + m + 1) := by
  classical
  simp [riemannXiDyadicZeroTailShell]
  intro _ _ h
  exact h.le

/-- The complement of the closed radius-`2^N` window is exactly the union
of the boundary-safe shifted dyadic shells. -/
theorem riemannXiZeroIndexWindow_compl_eq_iUnion_tailShell (N : ℕ) :
    {i : RiemannXiZeroIndex |
      i ∉ riemannXiZeroIndexWindow ((2 : ℝ) ^ N)} =
      ⋃ m : ℕ,
        (↑(riemannXiDyadicZeroTailShell N m) :
          Set RiemannXiZeroIndex) := by
  ext i
  simp only [Set.mem_setOf_eq, Set.mem_iUnion, Finset.mem_coe,
    mem_riemannXiDyadicZeroTailShell]
  constructor
  · intro hi
    have hiN : (2 : ℝ) ^ N < ‖riemannXiZeroRoot i‖ := by
      simpa using hi
    obtain ⟨k, hk, huniq⟩ :=
      existsUnique_dyadic_interval ‖riemannXiZeroRoot i‖
        ((one_le_pow₀ (by norm_num)).trans hiN.le)
    have hNk : N ≤ k := by
      by_contra h
      have hkN : k + 1 ≤ N := by omega
      have hp : (2 : ℝ) ^ (k + 1) ≤ (2 : ℝ) ^ N :=
        pow_le_pow_right₀ (by norm_num) hkN
      linarith [hk.2]
    refine ⟨k - N, hiN, ?_, ?_⟩
    · simpa [Nat.add_sub_of_le hNk] using hk.1
    · simpa [Nat.add_sub_of_le hNk] using hk.2
  · rintro ⟨m, hiN, _⟩
    simpa using hiN

/-- Distinct boundary-safe shifted dyadic shells are disjoint. -/
theorem riemannXiDyadicZeroTailShell_disjoint (N : ℕ)
    {m n : ℕ} (hmn : m ≠ n) :
    Disjoint
      (↑(riemannXiDyadicZeroTailShell N m) :
        Set RiemannXiZeroIndex)
      (↑(riemannXiDyadicZeroTailShell N n) :
        Set RiemannXiZeroIndex) := by
  rw [Set.disjoint_left]
  intro i him hin
  have hm := (mem_riemannXiDyadicZeroTailShell N m i).mp him
  have hn := (mem_riemannXiDyadicZeroTailShell N n i).mp hin
  by_cases hmn' : m < n
  · have hp :
        (2 : ℝ) ^ (N + m + 1) ≤ (2 : ℝ) ^ (N + n) :=
      pow_le_pow_right₀ (by norm_num) (by omega)
    linarith
  · have hnm : n < m := by omega
    have hp :
        (2 : ℝ) ^ (N + n + 1) ≤ (2 : ℝ) ^ (N + m) :=
      pow_le_pow_right₀ (by norm_num) (by omega)
    linarith

/-- Exact multiplicity-preserving reindexing of a summable xi-zero tail by
the boundary-safe shifted dyadic shells. -/
theorem tsum_riemannXiZeroIndexWindow_compl_eq_tailShells
    (N : ℕ) (f : RiemannXiZeroIndex → ℝ) (hf : Summable f) :
    (∑' i : {i : RiemannXiZeroIndex //
        i ∉ riemannXiZeroIndexWindow ((2 : ℝ) ^ N)}, f i) =
      ∑' m : ℕ,
        ∑ i ∈ riemannXiDyadicZeroTailShell N m, f i := by
  let Tail := {i : RiemannXiZeroIndex //
    i ∉ riemannXiZeroIndexWindow ((2 : ℝ) ^ N)}
  let t : ℕ → Set Tail :=
    fun m ↦ {i | i.1 ∈ riemannXiDyadicZeroTailShell N m}
  have hpart : ∀ i : Tail, ∃! m, i ∈ t m := by
    intro i
    have hi : i.1 ∈ ⋃ m : ℕ,
        (↑(riemannXiDyadicZeroTailShell N m) :
          Set RiemannXiZeroIndex) := by
      rw [← riemannXiZeroIndexWindow_compl_eq_iUnion_tailShell N]
      exact i.property
    simp only [Set.mem_iUnion, Finset.mem_coe] at hi
    obtain ⟨m, hm⟩ := hi
    refine ⟨m, hm, ?_⟩
    intro n hn
    by_contra hmn
    have hd := riemannXiDyadicZeroTailShell_disjoint N
      (m := m) (n := n) (fun h ↦ hmn h.symm)
    change i.1 ∈ riemannXiDyadicZeroTailShell N m at hm
    change i.1 ∈ riemannXiDyadicZeroTailShell N n at hn
    have hm' : i.1 ∈
        (↑(riemannXiDyadicZeroTailShell N m) :
          Set RiemannXiZeroIndex) := hm
    have hn' : i.1 ∈
        (↑(riemannXiDyadicZeroTailShell N n) :
          Set RiemannXiZeroIndex) := hn
    exact (Set.disjoint_left.mp hd hm') hn'
  let e := Set.sigmaEquiv t hpart
  have hsigma :
      Summable (fun p : (m : ℕ) × t m ↦ f p.2) := by
    change Summable ((fun i : Tail ↦ f i) ∘ e)
    rw [e.summable_iff]
    exact hf.subtype _
  calc
    (∑' i : {i : RiemannXiZeroIndex //
        i ∉ riemannXiZeroIndexWindow ((2 : ℝ) ^ N)}, f i)
        = ∑' p : (m : ℕ) × t m, f p.2 := by
      symm
      exact e.tsum_eq (fun i : Tail ↦ f i)
    _ = ∑' m : ℕ, ∑' i : t m, f i := hsigma.tsum_sigma
    _ = ∑' m : ℕ,
        ∑ i ∈ riemannXiDyadicZeroTailShell N m, f i := by
      apply tsum_congr
      intro m
      let em : t m ≃
          {i // i ∈ riemannXiDyadicZeroTailShell N m} := {
        toFun i := ⟨i.1.1, i.2⟩
        invFun i := ⟨⟨i.1, by
          have hi :=
            (mem_riemannXiDyadicZeroTailShell N m i).mp i.2
          simpa using hi.1⟩, i.2⟩
        left_inv i := by
          apply Subtype.ext
          apply Subtype.ext
          rfl
        right_inv i := by
          apply Subtype.ext
          rfl }
      calc
        (∑' i : t m, f i) =
            ∑' i :
              {i // i ∈ riemannXiDyadicZeroTailShell N m},
              f i := by
          exact em.tsum_eq
            (fun i :
              {i // i ∈ riemannXiDyadicZeroTailShell N m} ↦ f i)
        _ = ∑ i ∈ riemannXiDyadicZeroTailShell N m, f i :=
          Finset.tsum_subtype _ _

/-- The explicit compatible scale `R_j = 2^j`, `N_j = 4j` makes the
radius-squared dyadic tail majorant tend to zero. This is the quantitative
choice required by the uniform canonical-tail estimate on the selected
Cartan circles. -/
theorem tendsto_radius_sq_mul_four_mul_truncation_tail :
    Tendsto (fun j : ℕ ↦
      ((2 : ℝ) ^ j) ^ 2 *
        ((4 * j + 1 : ℝ) / (2 : ℝ) ^ (4 * j)))
      atTop (𝓝 0) := by
  have hself :
      Tendsto (fun j : ℕ ↦ (j : ℝ) * (1 / 4 : ℝ) ^ j)
        atTop (𝓝 0) :=
    tendsto_self_mul_const_pow_of_abs_lt_one (by norm_num)
  have hgeom :
      Tendsto (fun j : ℕ ↦ (1 / 4 : ℝ) ^ j)
        atTop (𝓝 0) :=
    tendsto_pow_atTop_nhds_zero_of_lt_one
      (by norm_num) (by norm_num)
  convert (hself.const_mul 4).add hgeom using 1
  · funext j
    rw [show (4 * j + 1 : ℝ) = 4 * (j : ℝ) + 1 by
      norm_num]
    rw [show ((2 : ℝ) ^ j) ^ 2 = (4 : ℝ) ^ j by
      rw [← pow_mul]
      rw [mul_comm j 2, pow_mul]
      norm_num]
    rw [show ((2 : ℝ) ^ (4 * j)) = (16 : ℝ) ^ j by
      rw [pow_mul]
      norm_num]
    rw [show (4 : ℝ) ^ j * ((4 * (j : ℝ) + 1) /
          (16 : ℝ) ^ j) =
        (4 * (j : ℝ) + 1) *
          ((4 : ℝ) ^ j / (16 : ℝ) ^ j) by ring]
    rw [show (4 : ℝ) ^ j / (16 : ℝ) ^ j =
        (1 / 4 : ℝ) ^ j by
      rw [← div_pow]
      norm_num]
    ring
  · norm_num

/-- Once the Jensen low window lies inside radius `2^N`, each boundary-safe
tail shell injects into the sourced multiplicity shell at index `N+m`. -/
theorem sum_riemannXiDyadicZeroTailShell_le_sourcedShell
    {R : ℝ} {N m : ℕ} (hRN : R ≤ (2 : ℝ) ^ N) :
    (∑ i ∈ riemannXiDyadicZeroTailShell N m,
        ‖(riemannXiZeroRoot i)⁻¹‖ ^ 2) ≤
      ∑ i ∈ riemannXiDyadicZeroShell
        (riemannXiZeroIndexWindow R) (N + m),
        ‖(riemannXiZeroRoot i)⁻¹‖ ^ 2 := by
  classical
  let low := riemannXiZeroIndexWindow R
  let tail := riemannXiDyadicZeroTailShell N m
  let st : Finset {i : RiemannXiZeroIndex // i ∉ low} :=
    Finset.subtype (fun i ↦ i ∉ low) tail
  have hall : ∀ i ∈ tail, i ∉ low := by
    intro i hi
    have hit :=
      (mem_riemannXiDyadicZeroTailShell N m i).mp hi
    rw [mem_riemannXiZeroIndexWindow]
    exact not_le.mpr (hRN.trans_lt hit.1)
  have hst :
      st ⊆ riemannXiDyadicZeroShell low (N + m) := by
    intro i hi
    rw [mem_riemannXiDyadicZeroShell]
    have hi : i.1 ∈ tail := by
      simpa [st, Finset.mem_subtype] using hi
    have hit :=
      (mem_riemannXiDyadicZeroTailShell N m i).mp hi
    simpa [Nat.add_assoc] using hit.2
  have hsum :
      (∑ i ∈ tail, ‖(riemannXiZeroRoot i)⁻¹‖ ^ 2) =
        ∑ i ∈ st, ‖(riemannXiZeroRoot i)⁻¹‖ ^ 2 := by
    apply Finset.sum_bij
      (fun i hi ↦ (⟨i, hall i hi⟩ :
        {i : RiemannXiZeroIndex // i ∉ low}))
    · intro i hi
      simp [st, hi]
    · intro a ha b hb hab
      exact congrArg Subtype.val hab
    · intro b hb
      have hb' : b.val ∈ tail := by
        simpa [st, Finset.mem_subtype] using hb
      exact ⟨b.val, hb', Subtype.ext rfl⟩
    · intro i hi
      rfl
  rw [hsum]
  exact Finset.sum_le_sum_of_subset_of_nonneg hst
    (fun _ _ _ ↦ sq_nonneg _)

/-- Source-backed multiplicity-aware inverse-square estimate for the actual
complement of the closed radius-`2^N` xi window. -/
theorem exists_riemannXiZeroRoot_inv_sq_radial_tail_bound :
    ∃ A R : ℝ, 0 ≤ A ∧ 1 ≤ R ∧ ∀ N : ℕ,
      R ≤ (2 : ℝ) ^ N →
      (∑' i : {i : RiemannXiZeroIndex //
        i ∉ riemannXiZeroIndexWindow ((2 : ℝ) ^ N)},
          ‖(riemannXiZeroRoot i)⁻¹‖ ^ 2) ≤
        A * (N + 1 : ℝ) / (2 : ℝ) ^ N := by
  classical
  obtain ⟨A₀, R, hA₀, hR, hcum⟩ :=
    exists_riemannXiZeroIndexWindow_card_le_orderOne
  let low := riemannXiZeroIndexWindow R
  let shell := riemannXiDyadicZeroShell low
  let A := 24 * A₀ * Real.log 2
  refine ⟨A, R, ?_, hR, ?_⟩
  · exact mul_nonneg
      (mul_nonneg (by norm_num) hA₀)
      (Real.log_nonneg (by norm_num))
  · intro N hRN
    have hglobal : Summable (fun i : RiemannXiZeroIndex ↦
        ‖(riemannXiZeroRoot i)⁻¹‖ ^ 2) := by
      apply
        summable_norm_inv_sq_of_finite_low_and_dyadic_linear_log_shell_count
          riemannXiZeroRoot low shell
          (A := 6 * A₀ * Real.log 2)
      · dsimp [low, shell]
        exact riemannXiDyadicZeroShell_partition R hR
      · intro k i hi
        exact ((mem_riemannXiDyadicZeroShell low k i).mp hi).1
      · intro k
        dsimp [shell, low]
        exact riemannXiDyadicZeroShell_card_le_of_cumulative
          hA₀ hcum k
    have hterm : ∀ m : ℕ,
        (∑ i ∈ riemannXiDyadicZeroTailShell N m,
          ‖(riemannXiZeroRoot i)⁻¹‖ ^ 2) ≤
        ∑ i ∈ shell (N + m),
          ‖(riemannXiZeroRoot i)⁻¹‖ ^ 2 := by
      intro m
      dsimp [shell, low]
      exact sum_riemannXiDyadicZeroTailShell_le_sourcedShell hRN
    have hshellSeries : Summable (fun k : ℕ ↦
        ∑ i ∈ shell k,
          ‖(riemannXiZeroRoot i)⁻¹‖ ^ 2) := by
      have hsub :=
        hglobal.subtype
          (fun i : RiemannXiZeroIndex ↦ i ∉ low)
      have hp := (summable_partition
        (f := fun i :
            {i : RiemannXiZeroIndex // i ∉ low} ↦
          ‖(riemannXiZeroRoot i)⁻¹‖ ^ 2)
        (fun i ↦ sq_nonneg _)
        (s := fun k ↦ ↑(shell k))
        (by
          dsimp [shell, low]
          simpa using
            riemannXiDyadicZeroShell_partition R hR)).mp hsub
      convert hp.2 using 1
      funext k
      exact (Finset.tsum_subtype (shell k)
        (fun i : {i : RiemannXiZeroIndex // i ∉ low} ↦
          ‖(riemannXiZeroRoot i)⁻¹‖ ^ 2)).symm
    have hright : Summable (fun m : ℕ ↦
        ∑ i ∈ shell (N + m),
          ‖(riemannXiZeroRoot i)⁻¹‖ ^ 2) :=
      hshellSeries.comp_injective (fun _ _ h ↦ by omega)
    have hleft : Summable (fun m : ℕ ↦
        ∑ i ∈ riemannXiDyadicZeroTailShell N m,
          ‖(riemannXiZeroRoot i)⁻¹‖ ^ 2) :=
      hright.of_nonneg_of_le
        (fun m ↦ Finset.sum_nonneg fun _ _ ↦ sq_nonneg _) hterm
    rw [tsum_riemannXiZeroIndexWindow_compl_eq_tailShells
      N (fun i ↦ ‖(riemannXiZeroRoot i)⁻¹‖ ^ 2)
      hglobal]
    apply (Summable.tsum_le_tsum hterm hleft hright).trans
    have hbound :=
      tsum_norm_inv_sq_dyadic_shell_tail_le
        (fun i : {i : RiemannXiZeroIndex // i ∉ low} ↦
          riemannXiZeroRoot i)
        shell
        (fun k i hi ↦
          ((mem_riemannXiDyadicZeroShell low k i).mp hi).1)
        (6 * A₀ * Real.log 2)
        (mul_nonneg
          (mul_nonneg (by norm_num) hA₀)
          (Real.log_nonneg (by norm_num)))
        (fun k ↦ by
          dsimp [shell, low]
          exact riemannXiDyadicZeroShell_card_le_of_cumulative
            hA₀ hcum k)
        N
    exact hbound.trans_eq (by
      dsimp [A]
      ring)

/-- The finite-head logarithmic loss at the compatible truncation
`T_j = 2^j` is subquadratic relative to circles of radius `2^j`. -/
theorem tendsto_add_one_sq_div_two_pow :
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

theorem tendsto_add_one_div_two_pow :
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

/-- Square-root form used by the Cauchy--Schwarz finite-head loss. -/
theorem tendsto_sqrt_const_mul_add_one_div_two_pow (K : ℝ) :
    Tendsto (fun j : ℕ ↦
      Real.sqrt (K * ((j + 1 : ℝ) / (2 : ℝ) ^ j)))
      atTop (𝓝 0) := by
  have h := tendsto_add_one_div_two_pow.const_mul K
  have hs := Real.continuous_sqrt.continuousAt.tendsto.comp h
  simpa [Function.comp_def] using hs

theorem tendsto_const_mul_add_one_sq_div_two_pow (K : ℝ) :
    Tendsto (fun j : ℕ ↦
      K * ((j + 1 : ℝ) ^ 2 / (2 : ℝ) ^ j))
      atTop (𝓝 0) := by
  simpa using tendsto_add_one_sq_div_two_pow.const_mul K

/-- Common normalized majorant for all selected-circle losses: root-log and
cardinality-log terms are quadratic-polynomial/geometric, inverse-root loss
is square-root geometric, and xi/tail terms are linear-geometric. -/
def DyadicCartanLossMajorant
    (Ksq Ksqrt Klin : ℝ) (j : ℕ) : ℝ :=
  Ksq * ((j + 1 : ℝ) ^ 2 / (2 : ℝ) ^ j) +
    Real.sqrt (Ksqrt * ((j + 1 : ℝ) / (2 : ℝ) ^ j)) +
    Klin * ((j + 1 : ℝ) / (2 : ℝ) ^ j)

theorem tendsto_dyadicCartanLossMajorant
    (Ksq Ksqrt Klin : ℝ) :
    Tendsto (DyadicCartanLossMajorant Ksq Ksqrt Klin)
      atTop (𝓝 0) := by
  have hsq :=
    tendsto_const_mul_add_one_sq_div_two_pow Ksq
  have hsqrt :=
    tendsto_sqrt_const_mul_add_one_div_two_pow Ksqrt
  have hlin := tendsto_add_one_div_two_pow.const_mul Klin
  change Tendsto (fun j : ℕ ↦
    Ksq * ((j + 1 : ℝ) ^ 2 / (2 : ℝ) ^ j) +
      Real.sqrt (Ksqrt * ((j + 1 : ℝ) / (2 : ℝ) ^ j)) +
      Klin * ((j + 1 : ℝ) / (2 : ℝ) ^ j)) atTop (𝓝 0)
  convert (hsq.add hsqrt).add hlin using 1
  norm_num

/-- Epsilon form consumed by the boundary-growth interface. -/
theorem eventually_dyadicCartanLossMajorant_lt
    (Ksq Ksqrt Klin ε : ℝ) (hε : 0 < ε) :
    ∀ᶠ j : ℕ in atTop,
      DyadicCartanLossMajorant Ksq Ksqrt Klin j < ε := by
  exact (tendsto_order.1
    (tendsto_dyadicCartanLossMajorant Ksq Ksqrt Klin)).2 ε hε

/-- A crude `log x ≤ x` estimate is enough for the Cartan cardinality loss;
it avoids inserting any hidden local zero-count hypothesis. -/
theorem card_mul_log_three_mul_div_sq_le
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

/-- Jensen's cumulative multiplicity count at the compatible cutoff
`2^(j+2)` has the explicit `O(2^j(j+1))` form. -/
theorem exists_riemannXiZeroIndexWindow_card_le_dyadic :
    ∃ B R : ℝ, 0 ≤ B ∧ 1 ≤ R ∧ ∀ j : ℕ,
      R ≤ (2 : ℝ) ^ (j + 2) →
      ((riemannXiZeroIndexWindow
        ((2 : ℝ) ^ (j + 2))).card : ℝ) ≤
        B * (2 : ℝ) ^ j * (j + 1 : ℝ) := by
  obtain ⟨A, R, hA, hR, hcount⟩ :=
    exists_riemannXiZeroIndexWindow_card_le_orderOne
  let B := 16 * A * Real.log 2
  refine ⟨B, R, by positivity, hR, ?_⟩
  intro j hj
  have hc := hcount ((2 : ℝ) ^ (j + 2)) hj
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
    ((riemannXiZeroIndexWindow
      ((2 : ℝ) ^ (j + 2))).card : ℝ)
        ≤ A * (2 : ℝ) ^ (j + 2) *
          Real.log (2 * (2 : ℝ) ^ (j + 2) + 2) := hc
    _ ≤ A * (2 : ℝ) ^ (j + 2) *
        (4 * (j + 1 : ℝ) * Real.log 2) := by
      gcongr
    _ = B * (2 : ℝ) ^ j * (j + 1 : ℝ) := by
      dsimp [B]
      rw [hj2]
      ring

/-- Unconditional inverse-square summability of the canonical
multiplicity enumeration of xi zeros, derived from Jensen's formula and
the proved global order-one estimate. -/
theorem summable_norm_inv_sq_riemannXiZeroRoot :
    Summable (fun i : RiemannXiZeroIndex ↦
      ‖(riemannXiZeroRoot i)⁻¹‖ ^ 2) := by
  classical
  obtain ⟨A₀, R, hA₀, hR, hcum⟩ :=
    exists_riemannXiZeroIndexWindow_card_le_orderOne
  let low := riemannXiZeroIndexWindow R
  let shell := riemannXiDyadicZeroShell low
  apply
    summable_norm_inv_sq_of_finite_low_and_dyadic_linear_log_shell_count
      riemannXiZeroRoot low shell
      (A := 6 * A₀ * Real.log 2)
  · dsimp [low, shell]
    exact riemannXiDyadicZeroShell_partition R hR
  · intro N i hi
    exact ((mem_riemannXiDyadicZeroShell low N i).mp hi).1
  · intro N
    dsimp [shell, low]
    exact riemannXiDyadicZeroShell_card_le_of_cumulative
      hA₀ hcum N

/-- Reflection turns the paired inverse-root term into an absolutely
summable inverse-quadratic term. -/
theorem summable_riemannXiZeroRoot_inv_mul_reflection :
    Summable (fun i : RiemannXiZeroIndex ↦
      (riemannXiZeroRoot i *
        (1 - riemannXiZeroRoot i))⁻¹) := by
  rw [← summable_norm_iff]
  apply Summable.of_nonneg_of_le (fun i ↦ norm_nonneg _) ?_
    ((summable_norm_inv_sq_riemannXiZeroRoot.add
      (summable_norm_inv_sq_riemannXiZeroRoot.comp_injective
        riemannXiZeroReflection_injective)).mul_left (1 / 2))
  intro i
  simp only [Function.comp_apply]
  rw [← riemannXiZeroRoot_reflection, mul_inv, norm_mul, one_div]
  nlinarith [sq_nonneg
    (‖(riemannXiZeroRoot i)⁻¹‖ -
      ‖(riemannXiZeroRoot (riemannXiZeroReflection i))⁻¹‖)]

theorem riemannXiPairedInverseRootHeightSums_tendsto :
    Tendsto
      (fun N : ℕ ↦ ∑ i ∈ riemannXiZeroHeightWindow (N : ℝ),
        (riemannXiZeroRoot i *
          (1 - riemannXiZeroRoot i))⁻¹)
      atTop (𝓝 (∑' i : RiemannXiZeroIndex,
        (riemannXiZeroRoot i *
          (1 - riemannXiZeroRoot i))⁻¹)) :=
  summable_riemannXiZeroRoot_inv_mul_reflection.hasSum.comp
    riemannXiZeroHeightWindow_nat_tendsto_atTop

/-- The conditionally ordered inverse-root sum along exact symmetric-height
windows therefore has an unconditional limit. -/
theorem riemannXiInverseRootHeightSums_tendsto :
    Tendsto
      (fun N : ℕ ↦ ∑ i ∈ riemannXiZeroHeightWindow (N : ℝ),
        (riemannXiZeroRoot i)⁻¹)
      atTop (𝓝 ((1 / 2 : ℂ) * ∑' i : RiemannXiZeroIndex,
        (riemannXiZeroRoot i *
          (1 - riemannXiZeroRoot i))⁻¹)) := by
  convert (tendsto_const_nhds.mul
    riemannXiPairedInverseRootHeightSums_tendsto) using 1
  funext N
  rw [← two_mul_sum_inv_riemannXiZeroHeightWindow]
  ring

theorem heightSymmetricLiCauchy_of_quadraticPairedCancellation
    {ι : Type*} [DecidableEq ι]
    (root : ι → ℂ) (cutoff : ℕ → Finset ι)
    (hcancel : QuadraticPairedLiShellCancellation root cutoff) :
    HeightSymmetricLiCauchy root cutoff := by
  choose B hB hbound using hcancel
  apply heightSymmetricLiCauchy_of_shell_majorants root cutoff
    (fun n N ↦ B n * Real.log (N + 2) /
      (((N + 2 : ℕ) : ℝ) ^ 2))
  · exact fun n ↦ summable_log_div_square_shell_majorant (B n) (hB n)
  · exact hbound

/-- Completeness of `ℂ` transfers explicit Cauchy estimates into a coefficient
sequence and the corresponding symmetric-limit interface. -/
theorem exists_heightSymmetricLiLimit_of_cauchy
    {ι : Type*} [DecidableEq ι]
    (root : ι → ℂ) (cutoff : ℕ → Finset ι) (height : ℕ → ℝ)
    (hheight : Tendsto height atTop atTop)
    (hcutoff : ∀ N i, i ∈ cutoff N ↔ |(root i).im| ≤ height N)
    (hcauchy : HeightSymmetricLiCauchy root cutoff) :
    ∃ coefficient : ℕ → ℂ,
      HeightSymmetricLiLimit root cutoff height coefficient := by
  choose coefficient hcoefficient using
    fun n ↦ cauchySeq_tendsto_of_complete (hcauchy n)
  exact ⟨coefficient, hheight, hcutoff, hcoefficient⟩

/-- Every fixed indexed zero eventually enters an exact symmetric-height
exhaustion. -/
theorem HeightSymmetricLiLimit.eventually_mem
    {ι : Type*} [DecidableEq ι]
    {root : ι → ℂ} {cutoff : ℕ → Finset ι} {height : ℕ → ℝ}
    {coefficient : ℕ → ℂ}
    (hlimit : HeightSymmetricLiLimit root cutoff height coefficient)
    (i : ι) :
    ∀ᶠ N in atTop, i ∈ cutoff N := by
  filter_upwards [hlimit.1.eventually (eventually_ge_atTop |(root i).im|)] with N hN
  exact (hlimit.2.1 N i).2 hN

/-- Every fixed finite family is eventually contained in a symmetric-height
window.  This is the finite-window stabilization interface used before any
infinite product theorem is invoked. -/
theorem HeightSymmetricLiLimit.eventually_subset
    {ι : Type*} [DecidableEq ι]
    {root : ι → ℂ} {cutoff : ℕ → Finset ι} {height : ℕ → ℝ}
    {coefficient : ℕ → ℂ}
    (hlimit : HeightSymmetricLiLimit root cutoff height coefficient)
    (F : Finset ι) :
    ∀ᶠ N in atTop, F ⊆ cutoff N := by
  change ∀ᶠ N in atTop, ∀ i ∈ F, i ∈ cutoff N
  exact (Finset.eventually_all F).2 fun i _ ↦ hlimit.eventually_mem i

/-- For a finite indexed family, an exact symmetric exhaustion eventually
stabilizes at the full finite Li sum, so its declared limit is forced to be
that sum. -/
theorem HeightSymmetricLiLimit.coefficient_eq_finite_sum
    {ι : Type*} [Fintype ι] [DecidableEq ι]
    {root : ι → ℂ} {cutoff : ℕ → Finset ι} {height : ℕ → ℝ}
    {coefficient : ℕ → ℂ}
    (hlimit : HeightSymmetricLiLimit root cutoff height coefficient)
    (n : ℕ) :
    coefficient n = ∑ i, liZeroSummand n (root i) := by
  have hfull : ∀ᶠ N in atTop, cutoff N = Finset.univ := by
    filter_upwards [hlimit.eventually_subset Finset.univ] with N hN
    exact Finset.Subset.antisymm (Finset.subset_univ _) hN
  have hconst : (fun N ↦ liZeroPartialSum root (cutoff N) n) =ᶠ[atTop]
      fun _ ↦ ∑ i, liZeroSummand n (root i) := by
    filter_upwards [hfull] with N hN
    simp [liZeroPartialSum, hN]
  apply tendsto_nhds_unique (hlimit.2.2 n)
  exact tendsto_const_nhds.congr' hconst.symm

/-- Transfer a symmetric zero-sum limit through exact finite approximants.
This isolates the derivative/zero-sum bridge: for xi, one must construct
`approximation` and prove both hypotheses below. -/
theorem HeightSymmetricLiLimit.coefficient_eq_of_approximation
    {ι : Type*} [DecidableEq ι]
    {root : ι → ℂ} {cutoff : ℕ → Finset ι} {height : ℕ → ℝ}
    {coefficient : ℕ → ℂ}
    (hlimit : HeightSymmetricLiLimit root cutoff height coefficient)
    (approximation : ℕ → ℕ → ℂ) (target : ℕ → ℂ)
    (hexact : ∀ N n, approximation N n =
      liZeroPartialSum root (cutoff N) n)
    (htarget : ∀ n, Tendsto (fun N ↦ approximation N n)
      atTop (𝓝 (target n))) :
    coefficient = target := by
  funext n
  apply tendsto_nhds_unique (hlimit.2.2 n)
  simpa only [hexact] using htarget n

/-- Locally uniform convergence of holomorphic functions propagates through
every iterated derivative on an open complex domain.

This is the all-order form of Mathlib's
`TendstoLocallyUniformlyOn.deriv`; the holomorphy needed at later induction
steps follows from complex analyticity. -/
theorem tendstoLocallyUniformlyOn_iteratedDeriv
    {ι : Type*} {p : Filter ι} {f : ι → ℂ → ℂ} {g : ℂ → ℂ}
    {s : Set ℂ} (hs : IsOpen s)
    (hconv : TendstoLocallyUniformlyOn f g p s)
    (hhol : ∀ᶠ i in p, DifferentiableOn ℂ (f i) s)
    (k : ℕ) :
    TendstoLocallyUniformlyOn
      (fun i ↦ iteratedDeriv k (f i)) (iteratedDeriv k g) p s := by
  induction k with
  | zero =>
      simpa only [iteratedDeriv_zero] using hconv
  | succ k ih =>
      rw [iteratedDeriv_succ]
      convert ih.deriv (by
        filter_upwards [hhol] with i hi
        rw [iteratedDeriv_eq_iterate]
        exact ((hi.analyticOnNhd hs).iterated_deriv k).differentiableOn) hs using 1
      ext i z
      simp [Function.comp_apply, iteratedDeriv_succ]

/-- Explicit nonvanishing-neighborhood transfer for logarithmic derivatives.

The denominator is required to be nonzero throughout `s`, not merely at the
coefficient center.  This stronger condition is what permits a locally
uniform quotient, which can then be differentiated to arbitrary order. -/
theorem tendstoLocallyUniformlyOn_logDeriv
    {ι : Type*} {p : Filter ι} [p.NeBot]
    {f : ι → ℂ → ℂ} {g : ℂ → ℂ}
    {s : Set ℂ} (hs : IsOpen s)
    (hconv : TendstoLocallyUniformlyOn f g p s)
    (hhol : ∀ᶠ i in p, DifferentiableOn ℂ (f i) s)
    (hg0 : ∀ z ∈ s, g z ≠ 0) :
    TendstoLocallyUniformlyOn
      (fun i z ↦ logDeriv (f i) z) (fun z ↦ logDeriv g z) p s := by
  have hgDiff : DifferentiableOn ℂ g s :=
    hconv.differentiableOn hhol hs
  have hderiv := hconv.deriv hhol hs
  simp only [logDeriv, Pi.div_apply]
  convert hderiv.div₀ hconv (hgDiff.deriv hs).continuousOn
    hgDiff.continuousOn hg0 using 1
  · ext i z
    rfl
  · ext z
    rfl

/-- All Taylor coefficients of logarithmic derivatives converge under local
uniform convergence, holomorphy, and explicit nonvanishing on a common open
neighborhood. -/
theorem iteratedDeriv_logDeriv_tendsto
    {ι : Type*} {p : Filter ι} [p.NeBot]
    {f : ι → ℂ → ℂ} {g : ℂ → ℂ} {s : Set ℂ} {x : ℂ}
    (hs : IsOpen s) (hx : x ∈ s)
    (hconv : TendstoLocallyUniformlyOn f g p s)
    (hhol : ∀ᶠ i in p, DifferentiableOn ℂ (f i) s)
    (hf0 : ∀ᶠ i in p, ∀ z ∈ s, f i z ≠ 0)
    (hg0 : ∀ z ∈ s, g z ≠ 0) (k : ℕ) :
    Tendsto (fun i ↦ iteratedDeriv k (logDeriv (f i)) x) p
      (𝓝 (iteratedDeriv k (logDeriv g) x)) := by
  have hlog := tendstoLocallyUniformlyOn_logDeriv hs hconv hhol hg0
  have hlogHol : ∀ᶠ i in p, DifferentiableOn ℂ (logDeriv (f i)) s := by
    filter_upwards [hhol, hf0] with i hi hi0
    change DifferentiableOn ℂ (deriv (f i) / f i) s
    exact (hi.deriv hs).div hi hi0
  exact (tendstoLocallyUniformlyOn_iteratedDeriv
    hs hlog hlogHol k).tendsto_at hx

/-- The genus-one Weierstrass primary factor `E₁(w)`. -/
def genusOnePrimaryFactor (w : ℂ) : ℂ :=
  (1 - w) * Complex.exp w

/-- Quadratic cancellation in the genus-one factor.  This is the local
estimate that turns inverse-square zero summability into normal convergence. -/
theorem norm_genusOnePrimaryFactor_sub_one_le
    (w : ℂ) (hw : ‖w‖ ≤ 1) :
    ‖genusOnePrimaryFactor w - 1‖ ≤ 3 * ‖w‖ ^ 2 := by
  have hrem : ‖Complex.exp w - 1 - w‖ ≤ ‖w‖ ^ 2 := by
    simpa using Complex.norm_exp_sub_one_sub_id_le hw
  have hexp : ‖Complex.exp w - 1‖ ≤ 2 * ‖w‖ :=
    Complex.norm_exp_sub_one_le hw
  rw [show genusOnePrimaryFactor w - 1 =
      (Complex.exp w - 1 - w) - w * (Complex.exp w - 1) by
    unfold genusOnePrimaryFactor
    ring]
  calc
    _ ≤ ‖Complex.exp w - 1 - w‖ + ‖w * (Complex.exp w - 1)‖ :=
      norm_sub_le _ _
    _ ≤ ‖w‖ ^ 2 + ‖w‖ * (2 * ‖w‖) := by
      gcongr
      rw [norm_mul]
      gcongr
    _ = 3 * ‖w‖ ^ 2 := by ring

def genusOneCanonicalFactor {ι : Type*}
    (root : ι → ℂ) (i : ι) (s : ℂ) : ℂ :=
  genusOnePrimaryFactor (s / root i)

set_option maxHeartbeats 800000 in
/-- Strong normal-convergence theorem for genus-one canonical products.
Multiplicity is represented by the index type, so repeated roots contribute
repeated factors. -/
theorem hasProdLocallyUniformlyOn_genusOneCanonicalFactor
    {ι : Type*} (root : ι → ℂ)
    (hinv : Summable (fun i ↦ ‖(root i)⁻¹‖ ^ 2))
    (hsmall : ∀ K : Set ℂ, IsCompact K →
      ∀ᶠ i in cofinite, ∀ s ∈ K, ‖s / root i‖ ≤ 1) :
    HasProdLocallyUniformlyOn (genusOneCanonicalFactor root)
      (fun s ↦ ∏' i, genusOneCanonicalFactor root i s) Set.univ := by
  apply hasProdLocallyUniformlyOn_of_forall_compact isOpen_univ
  intro K hKu hK
  obtain ⟨R, hR⟩ := hK.isBounded.exists_norm_le
  have hu : Summable (fun i ↦ 3 * R ^ 2 * ‖(root i)⁻¹‖ ^ 2) :=
    hinv.mul_left (3 * R ^ 2)
  have hp := hu.hasProdUniformlyOn_one_add
    (f := fun i s ↦ genusOneCanonicalFactor root i s - 1) hK
    (by
      filter_upwards [hsmall K hK] with i hi s hs
      have hfactor :=
        norm_genusOnePrimaryFactor_sub_one_le (s / root i) (hi s hs)
      apply hfactor.trans
      rw [norm_div, norm_inv, div_eq_mul_inv, mul_pow]
      have hsquare : ‖s‖ ^ 2 ≤ R ^ 2 :=
        (sq_le_sq₀ (norm_nonneg s)
          ((norm_nonneg s).trans (hR s hs))).2 (hR s hs)
      calc
        3 * (‖s‖ ^ 2 * ‖root i‖⁻¹ ^ 2) ≤
            3 * (R ^ 2 * ‖root i‖⁻¹ ^ 2) := by
          gcongr
        _ = 3 * R ^ 2 * ‖root i‖⁻¹ ^ 2 := by ring)
    (by
      intro i
      unfold genusOneCanonicalFactor genusOnePrimaryFactor
      fun_prop)
  simpa [sub_eq_add_neg, add_assoc] using hp

theorem genusOneCanonicalFactor_ne_zero_iff
    {ι : Type*} {root : ι → ℂ}
    (hroot0 : ∀ i, root i ≠ 0) (i : ι) (s : ℂ) :
    genusOneCanonicalFactor root i s ≠ 0 ↔ s ≠ root i := by
  constructor
  · intro hf hs
    subst s
    apply hf
    unfold genusOneCanonicalFactor genusOnePrimaryFactor
    simp [hroot0 i]
  · intro hs
    unfold genusOneCanonicalFactor genusOnePrimaryFactor
    apply mul_ne_zero
    · rw [sub_ne_zero]
      intro hone
      apply hs
      exact (div_eq_one_iff_eq (hroot0 i)).mp hone.symm
    · exact Complex.exp_ne_zero _

theorem logDeriv_genusOneCanonicalFactor_riemannXi_one
    (i : RiemannXiZeroIndex) :
    logDeriv (genusOneCanonicalFactor riemannXiZeroRoot i) 1 =
      (riemannXiZeroRoot i *
        (1 - riemannXiZeroRoot i))⁻¹ := by
  have hroot := riemannXiZeroRoot_ne_zero i
  have hone : (1 : ℂ) ≠ riemannXiZeroRoot i := by
    intro h
    have := (riemannXiZeroRoot_mem_openCriticalStrip i).2
    rw [← h] at this
    norm_num at this
  unfold genusOneCanonicalFactor genusOnePrimaryFactor
  rw [logDeriv_mul]
  · have hsub : HasDerivAt
        (fun s : ℂ ↦ 1 - s / riemannXiZeroRoot i)
        (-(riemannXiZeroRoot i)⁻¹) 1 := by
      simpa [id_eq] using
        ((hasDerivAt_id (𝕜 := ℂ) (1 : ℂ)).div_const
          (riemannXiZeroRoot i)).const_sub 1
    have hexp : HasDerivAt
        (fun s : ℂ ↦ Complex.exp (s / riemannXiZeroRoot i))
        (Complex.exp (1 / riemannXiZeroRoot i) /
          riemannXiZeroRoot i) 1 := by
      simpa [id_eq, div_eq_mul_inv] using
        ((hasDerivAt_id (𝕜 := ℂ) (1 : ℂ)).div_const
          (riemannXiZeroRoot i)).cexp
    simp only [logDeriv_apply]
    rw [hsub.deriv, hexp.deriv]
    field_simp
    ring
  · simpa [sub_ne_zero] using hone.symm
  · exact Complex.exp_ne_zero _
  · fun_prop
  · fun_prop

theorem logDeriv_riemannXiHeightGenusOneProduct_one (T : ℝ) :
    logDeriv
      (fun s ↦ ∏ i ∈ riemannXiZeroHeightWindow T,
        genusOneCanonicalFactor riemannXiZeroRoot i s) 1 =
      ∑ i ∈ riemannXiZeroHeightWindow T,
        (riemannXiZeroRoot i *
          (1 - riemannXiZeroRoot i))⁻¹ := by
  rw [logDeriv_prod]
  · apply Finset.sum_congr rfl
    intro i hi
    exact logDeriv_genusOneCanonicalFactor_riemannXi_one i
  · intro i hi
    exact (genusOneCanonicalFactor_ne_zero_iff
      riemannXiZeroRoot_ne_zero i 1).mpr (by
        intro h
        have := (riemannXiZeroRoot_mem_openCriticalStrip i).2
        rw [← h] at this
        norm_num at this)
  · intro i hi
    unfold genusOneCanonicalFactor genusOnePrimaryFactor
    fun_prop

/-- The quadratic factor estimate is summable at every point. -/
theorem summable_norm_genusOneCanonicalFactor_sub_one
    {ι : Type*} (root : ι → ℂ)
    (hinv : Summable (fun i ↦ ‖(root i)⁻¹‖ ^ 2))
    (hsmall : ∀ K : Set ℂ, IsCompact K →
      ∀ᶠ i in cofinite, ∀ s ∈ K, ‖s / root i‖ ≤ 1)
    (s : ℂ) :
    Summable (fun i ↦ ‖genusOneCanonicalFactor root i s - 1‖) := by
  have hu : Summable
      (fun i ↦ 3 * ‖s‖ ^ 2 * ‖(root i)⁻¹‖ ^ 2) :=
    hinv.mul_left (3 * ‖s‖ ^ 2)
  apply Summable.of_norm_bounded_eventually hu
  filter_upwards [hsmall {s} isCompact_singleton] with i hi
  have hfactor :=
    norm_genusOnePrimaryFactor_sub_one_le (s / root i)
      (hi s (Set.mem_singleton s))
  rw [Real.norm_eq_abs, abs_of_nonneg (norm_nonneg _)]
  apply hfactor.trans
  rw [norm_div, norm_inv, div_eq_mul_inv, mul_pow]
  ring_nf
  exact le_rfl

/-- The normally convergent genus-one product has no zeros other than its
listed roots. -/
theorem genusOneCanonicalProduct_ne_zero_iff
    {ι : Type*} (root : ι → ℂ)
    (hroot0 : ∀ i, root i ≠ 0)
    (hinv : Summable (fun i ↦ ‖(root i)⁻¹‖ ^ 2))
    (hsmall : ∀ K : Set ℂ, IsCompact K →
      ∀ᶠ i in cofinite, ∀ s ∈ K, ‖s / root i‖ ≤ 1)
    (s : ℂ) :
    (∏' i, genusOneCanonicalFactor root i s) ≠ 0 ↔
      ∀ i, s ≠ root i := by
  constructor
  · intro h i hi
    apply h
    have hz : genusOneCanonicalFactor root i s = 0 :=
      not_ne_iff.mp
        ((genusOneCanonicalFactor_ne_zero_iff hroot0 i s).not.mpr
          (not_ne_iff.mpr hi))
    exact (hasProd_zero_of_exists_eq_zero ⟨i, hz⟩).tprod_eq
  · intro hs
    have hsum :=
      summable_norm_genusOneCanonicalFactor_sub_one
        root hinv hsmall s
    have hne : ∀ i,
        1 + (genusOneCanonicalFactor root i s - 1) ≠ 0 := by
      intro i
      simpa using
        (genusOneCanonicalFactor_ne_zero_iff hroot0 i s).mpr
          (hs i)
    simpa using tprod_one_add_ne_zero_of_summable hne hsum

/-- Normal convergence makes the whole canonical product entire. -/
theorem analyticOnNhd_genusOneCanonicalProduct
    {ι : Type*} {root : ι → ℂ}
    (hprod : HasProdLocallyUniformlyOn
      (genusOneCanonicalFactor root)
      (fun s ↦ ∏' i, genusOneCanonicalFactor root i s) Set.univ) :
    AnalyticOnNhd ℂ
      (fun s ↦ ∏' i, genusOneCanonicalFactor root i s) Set.univ := by
  apply (hprod.differentiableOn (by
    filter_upwards [] with t
    have hd : DifferentiableOn ℂ
        (∏ i ∈ t, genusOneCanonicalFactor root i) Set.univ := by
      apply DifferentiableOn.finsetProd
      intro i hi
      have hdiff : Differentiable ℂ
          (genusOneCanonicalFactor root i) := by
        unfold genusOneCanonicalFactor genusOnePrimaryFactor
        fun_prop
      exact hdiff.differentiableOn
    exact hd.congr (fun z hz ↦ by simp))
    isOpen_univ).analyticOnNhd isOpen_univ

/-- Every individual genus-one factor has a simple zero at its root. -/
theorem analyticOrderAt_genusOneCanonicalFactor_root
    {ι : Type*} {root : ι → ℂ}
    (hroot0 : ∀ i, root i ≠ 0) (i : ι) :
    analyticOrderAt (genusOneCanonicalFactor root i) (root i) = 1 := by
  let f : ℂ → ℂ := fun z ↦ 1 - z / root i
  let g : ℂ → ℂ := fun z ↦ Complex.exp (z / root i)
  have hf : AnalyticAt ℂ f (root i) := by
    dsimp [f]
    fun_prop
  have hg : AnalyticAt ℂ g (root i) := by
    dsimp [g]
    fun_prop
  have hf0 : f (root i) = 0 := by
    dsimp [f]
    rw [div_self (hroot0 i), sub_self]
  have hfd : deriv f (root i) ≠ 0 := by
    have heq : deriv f (root i) = -(root i)⁻¹ := by
      dsimp [f]
      simp [deriv_div_const]
    rw [heq]
    exact neg_ne_zero.mpr (inv_ne_zero (hroot0 i))
  have hfo : analyticOrderAt f (root i) = 1 :=
    hf.analyticOrderAt_eq_one_of_zero_deriv_ne_zero hf0 hfd
  have hgo : analyticOrderAt g (root i) = 0 :=
    hg.analyticOrderAt_eq_zero.mpr (Complex.exp_ne_zero _)
  change analyticOrderAt (f * g) (root i) = 1
  rw [analyticOrderAt_mul hf hg, hfo, hgo, add_zero]

set_option maxHeartbeats 800000 in
private theorem analyticOrderAt_genusOneCanonicalProduct_of_nonempty_fiber
    {ι : Type*} [DecidableEq ι] {root : ι → ℂ}
    (hroot0 : ∀ i, root i ≠ 0)
    (hinv : Summable (fun i ↦ ‖(root i)⁻¹‖ ^ 2))
    (hsmall : ∀ K : Set ℂ, IsCompact K →
      ∀ᶠ i in cofinite, ∀ s ∈ K, ‖s / root i‖ ≤ 1)
    (J : Finset ι) (z : ℂ)
    (hJ : ∀ i, i ∈ J ↔ root i = z)
    (i0 : ι) (hi0 : i0 ∈ J) :
    analyticOrderAt
      (fun s ↦ ∏' i, genusOneCanonicalFactor root i s) z =
        J.card := by
  let κ := {i : ι // i ∉ J}
  let rootC : κ → ℂ := fun i ↦ root i
  let Q : ℂ → ℂ :=
    fun s ↦ ∏' i : κ, genusOneCanonicalFactor rootC i s
  have hrootC0 : ∀ i, rootC i ≠ 0 := fun i ↦ hroot0 i
  have hinvC : Summable (fun i ↦ ‖(rootC i)⁻¹‖ ^ 2) := by
    simpa [rootC, κ, Function.comp_def] using
      hinv.subtype (fun i ↦ i ∉ J)
  have hsmallC : ∀ K : Set ℂ, IsCompact K →
      ∀ᶠ i in cofinite, ∀ s ∈ K, ‖s / rootC i‖ ≤ 1 := by
    intro K hK
    exact Subtype.coe_injective.tendsto_cofinite.eventually
      (hsmall K hK)
  have hprodC : HasProdLocallyUniformlyOn
      (genusOneCanonicalFactor rootC) Q Set.univ :=
    hasProdLocallyUniformlyOn_genusOneCanonicalFactor
      rootC hinvC hsmallC
  have hQA : AnalyticOnNhd ℂ Q Set.univ :=
    analyticOnNhd_genusOneCanonicalProduct hprodC
  have hQz : Q z ≠ 0 := by
    apply (genusOneCanonicalProduct_ne_zero_iff
      rootC hrootC0 hinvC hsmallC z).mpr
    intro i hiz
    apply i.property
    exact (hJ i).mpr hiz.symm
  let B : ℂ → ℂ := fun s ↦ genusOnePrimaryFactor (s / z)
  have hz0 : z ≠ 0 := by
    rw [← (hJ i0).mp hi0]
    exact hroot0 i0
  have hfinite : (fun s ↦ ∏ i ∈ J,
      genusOneCanonicalFactor root i s) = B ^ J.card := by
    funext s
    rw [Pi.pow_apply]
    change (∏ i ∈ J, genusOnePrimaryFactor (s / root i)) =
      genusOnePrimaryFactor (s / z) ^ J.card
    rw [← Finset.prod_const]
    apply Finset.prod_congr rfl
    intro i hi
    rw [(hJ i).mp hi]
  have hfull : (fun s ↦ ∏' i,
      genusOneCanonicalFactor root i s) =
      (fun s ↦ ∏ i ∈ J, genusOneCanonicalFactor root i s) * Q := by
    funext s
    have hs :=
      (J.hasProd (fun i ↦ genusOneCanonicalFactor root i s)).mul_compl
        (hprodC.hasProd (Set.mem_univ s))
    exact hs.tprod_eq
  have hBA : AnalyticAt ℂ B z := by
    dsimp [B, genusOnePrimaryFactor]
    fun_prop
  have hBorder : analyticOrderAt B z = 1 := by
    have hfun : B = genusOneCanonicalFactor root i0 := by
      funext s
      unfold B genusOneCanonicalFactor
      rw [(hJ i0).mp hi0]
    rw [hfun, ← (hJ i0).mp hi0]
    exact analyticOrderAt_genusOneCanonicalFactor_root hroot0 i0
  rw [hfull, hfinite, analyticOrderAt_mul
    (hBA.pow J.card) (hQA z (Set.mem_univ z)),
    analyticOrderAt_pow hBA, hBorder,
    (hQA z (Set.mem_univ z)).analyticOrderAt_eq_zero.mpr hQz]
  simp

set_option maxHeartbeats 800000 in
/-- Exact multiplicity theorem for the infinite canonical product.  The
finite fibers explicitly encode repeated roots. -/
theorem analyticOrderAt_genusOneCanonicalProduct
    {ι : Type*} [DecidableEq ι] {root : ι → ℂ}
    (hroot0 : ∀ i, root i ≠ 0)
    (hinv : Summable (fun i ↦ ‖(root i)⁻¹‖ ^ 2))
    (hsmall : ∀ K : Set ℂ, IsCompact K →
      ∀ᶠ i in cofinite, ∀ s ∈ K, ‖s / root i‖ ≤ 1)
    (fiber : ℂ → Finset ι)
    (hfiber : ∀ z i, i ∈ fiber z ↔ root i = z)
    (z : ℂ) :
    analyticOrderAt
      (fun s ↦ ∏' i, genusOneCanonicalFactor root i s) z =
        (fiber z).card := by
  by_cases hempty : fiber z = ∅
  · have hprod :=
      hasProdLocallyUniformlyOn_genusOneCanonicalFactor
        root hinv hsmall
    have hA := analyticOnNhd_genusOneCanonicalProduct hprod
    rw [hempty]
    simp only [Finset.card_empty, Nat.cast_zero]
    rw [(hA z (Set.mem_univ z)).analyticOrderAt_eq_zero]
    apply (genusOneCanonicalProduct_ne_zero_iff
      root hroot0 hinv hsmall z).mpr
    intro i hiz
    have hi : i ∈ fiber z := (hfiber z i).mpr hiz.symm
    simp [hempty] at hi
  · obtain ⟨i0, hi0⟩ := Finset.nonempty_iff_ne_empty.mpr hempty
    exact analyticOrderAt_genusOneCanonicalProduct_of_nonempty_fiber
      hroot0 hinv hsmall (fiber z) z (hfiber z) i0 hi0

/-- If the indexing fibers enumerate the zeros of an entire function with
their analytic orders, then the infinite genus-one product has exactly the
same global divisor. -/
theorem genusOneCanonicalProduct_divisor_eq
    {ι : Type*} [DecidableEq ι] {root : ι → ℂ} {xi : ℂ → ℂ}
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
        MeromorphicOn.divisor xi Set.univ := by
  have hprod :=
    hasProdLocallyUniformlyOn_genusOneCanonicalFactor
      root hinv hsmall
  have hPA := analyticOnNhd_genusOneCanonicalProduct hprod
  have hPM : MeromorphicOn
      (fun s ↦ ∏' i, genusOneCanonicalFactor root i s) Set.univ :=
    fun z hz ↦ (hPA z hz).meromorphicAt
  have hxiM : MeromorphicOn xi Set.univ :=
    fun z hz ↦ (hxi z hz).meromorphicAt
  ext z
  rw [MeromorphicOn.divisor_apply hPM (Set.mem_univ z),
    MeromorphicOn.divisor_apply hxiM (Set.mem_univ z),
    (hPA z (Set.mem_univ z)).meromorphicOrderAt_eq,
    (hxi z (Set.mem_univ z)).meromorphicOrderAt_eq,
    analyticOrderAt_genusOneCanonicalProduct
      hroot0 hinv hsmall fiber hfiber z,
    hxiOrder z]

/-- Xi-specific divisor matching for the canonical countable multiplicity
enumeration.  Only the quantitative convergence hypotheses remain. -/
theorem riemannXiGenusOneCanonicalProduct_divisor_eq
    (hinv : Summable
      (fun i : RiemannXiZeroIndex ↦ ‖(riemannXiZeroRoot i)⁻¹‖ ^ 2))
    (hsmall : ∀ K : Set ℂ, IsCompact K →
      ∀ᶠ i : RiemannXiZeroIndex in cofinite,
        ∀ s ∈ K, ‖s / riemannXiZeroRoot i‖ ≤ 1) :
    MeromorphicOn.divisor
      (fun s ↦ ∏' i : RiemannXiZeroIndex,
        genusOneCanonicalFactor riemannXiZeroRoot i s) Set.univ =
      MeromorphicOn.divisor riemannXiLi Set.univ := by
  classical
  exact genusOneCanonicalProduct_divisor_eq
    riemannXiZeroRoot_ne_zero hinv hsmall riemannXiZeroFiber
      mem_riemannXiZeroFiber
      (fun z _ ↦ differentiable_riemannXiLi.analyticAt z)
      analyticOrderAt_riemannXiLi_eq_zeroFiber_card

/-- For the canonical xi enumeration, inverse-square summability is the only
remaining hypothesis needed for normal convergence and exact divisor
matching; escape follows from local finiteness. -/
theorem riemannXiGenusOneCanonicalProduct_divisor_eq_of_summable
    (hinv : Summable
      (fun i : RiemannXiZeroIndex ↦
        ‖(riemannXiZeroRoot i)⁻¹‖ ^ 2)) :
    MeromorphicOn.divisor
      (fun s ↦ ∏' i : RiemannXiZeroIndex,
        genusOneCanonicalFactor riemannXiZeroRoot i s) Set.univ =
      MeromorphicOn.divisor riemannXiLi Set.univ := by
  apply riemannXiGenusOneCanonicalProduct_divisor_eq hinv
  intro K hK
  obtain ⟨R, hR⟩ := hK.isBounded.exists_norm_le
  filter_upwards [riemannXiZeroRoot_escape.eventually
    (eventually_ge_atTop R)] with i hi s hs
  rw [norm_div]
  exact (div_le_one
    (norm_pos_iff.mpr (riemannXiZeroRoot_ne_zero i))).2
      ((hR s hs).trans hi)

/-- The genus-one canonical product over the canonical multiplicity
enumeration has exactly xi's global divisor, unconditionally. -/
theorem riemannXiGenusOneCanonicalProduct_divisor_eq_unconditional :
    MeromorphicOn.divisor
      (fun s ↦ ∏' i : RiemannXiZeroIndex,
        genusOneCanonicalFactor riemannXiZeroRoot i s) Set.univ =
      MeromorphicOn.divisor riemannXiLi Set.univ :=
  riemannXiGenusOneCanonicalProduct_divisor_eq_of_summable
    summable_norm_inv_sq_riemannXiZeroRoot

/-- Xi divisor matching from the source-standard `O(log N)` shell count,
with an arbitrary finite low-zero set handled separately. -/
theorem riemannXiGenusOneCanonicalProduct_divisor_eq_of_logarithmicShellCount
    (low : Finset RiemannXiZeroIndex)
    (shell : ℕ → Finset
      {i : RiemannXiZeroIndex // i ∉ low})
    (hpartition : ∀ i : {i : RiemannXiZeroIndex // i ∉ low},
      ∃! N, i ∈ shell N)
    (hlower : ∀ N i, i ∈ shell N →
      ((N + 1 : ℕ) : ℝ) ≤ ‖riemannXiZeroRoot i‖)
    (A : ℝ) (hA : 0 ≤ A)
    (hcount : ∀ N, ((shell N).card : ℝ) ≤
      A * Real.log (N + 2)) :
    MeromorphicOn.divisor
      (fun s ↦ ∏' i : RiemannXiZeroIndex,
        genusOneCanonicalFactor riemannXiZeroRoot i s) Set.univ =
      MeromorphicOn.divisor riemannXiLi Set.univ := by
  classical
  apply riemannXiGenusOneCanonicalProduct_divisor_eq_of_summable
  exact summable_norm_inv_sq_of_finite_low_and_logarithmic_shell_count
    riemannXiZeroRoot low shell hpartition hlower A hA hcount

/-- Xi divisor matching from the weaker dyadic shell count produced by
an order-one Jensen estimate, again allowing arbitrary finite low zeros. -/
theorem riemannXiGenusOneCanonicalProduct_divisor_eq_of_dyadicShellCount
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
      MeromorphicOn.divisor riemannXiLi Set.univ := by
  classical
  apply riemannXiGenusOneCanonicalProduct_divisor_eq_of_summable
  exact
    summable_norm_inv_sq_of_finite_low_and_dyadic_linear_log_shell_count
      riemannXiZeroRoot low shell hpartition hlower A hcount

set_option maxHeartbeats 800000 in
/-- For an escaping zero family, inverse-square summability alone gives the
compact smallness required by the genus-one normal-convergence theorem. -/
theorem hasProdLocallyUniformlyOn_genusOneCanonicalFactor_of_escape
    {ι : Type*} (root : ι → ℂ)
    (hroot0 : ∀ i, root i ≠ 0)
    (hinv : Summable (fun i ↦ ‖(root i)⁻¹‖ ^ 2))
    (hescape : Tendsto (fun i ↦ ‖root i‖) cofinite atTop) :
    HasProdLocallyUniformlyOn (genusOneCanonicalFactor root)
      (fun s ↦ ∏' i, genusOneCanonicalFactor root i s) Set.univ := by
  apply hasProdLocallyUniformlyOn_genusOneCanonicalFactor root hinv
  intro K hK
  obtain ⟨R, hR⟩ := hK.isBounded.exists_norm_le
  filter_upwards [hescape.eventually (eventually_ge_atTop R)] with i hi s hs
  rw [norm_div]
  exact (div_le_one (norm_pos_iff.mpr (hroot0 i))).2 ((hR s hs).trans hi)

/-- The canonical genus-one xi product converges locally uniformly on the
whole plane, with no remaining zero-count hypothesis. -/
theorem riemannXiGenusOneCanonicalProduct_hasProdLocallyUniformly :
    HasProdLocallyUniformlyOn
      (genusOneCanonicalFactor riemannXiZeroRoot)
      (fun s ↦ ∏' i : RiemannXiZeroIndex,
        genusOneCanonicalFactor riemannXiZeroRoot i s) Set.univ :=
  hasProdLocallyUniformlyOn_genusOneCanonicalFactor_of_escape
    riemannXiZeroRoot riemannXiZeroRoot_ne_zero
      summable_norm_inv_sq_riemannXiZeroRoot
      riemannXiZeroRoot_escape

theorem analyticOnNhd_riemannXiGenusOneCanonicalProduct :
    AnalyticOnNhd ℂ
      (fun s ↦ ∏' i : RiemannXiZeroIndex,
        genusOneCanonicalFactor riemannXiZeroRoot i s) Set.univ :=
  analyticOnNhd_genusOneCanonicalProduct
    riemannXiGenusOneCanonicalProduct_hasProdLocallyUniformly

/-- Source-shaped canonical-product convergence: Bombieri--Lagarias
summability, separation from zero, and escape of the roots imply normal
convergence of the multiplicity-indexed genus-one product on `ℂ`. -/
theorem hasProdLocallyUniformlyOn_genusOneCanonicalFactor_of_bombieriLagarias
    {ι : Type*} (root : ι → ℂ)
    (hroot0 : ∀ i, root i ≠ 0)
    (hlarge : ∀ i, 1 ≤ ‖root i‖)
    (hsum : BombieriLagariasSummability root)
    (hescape : Tendsto (fun i ↦ ‖root i‖) cofinite atTop) :
    HasProdLocallyUniformlyOn (genusOneCanonicalFactor root)
      (fun s ↦ ∏' i, genusOneCanonicalFactor root i s) Set.univ :=
  hasProdLocallyUniformlyOn_genusOneCanonicalFactor_of_escape
    root hroot0 (hsum.summable_norm_inv_sq hlarge) hescape

/-- Any exact symmetric-height exhaustion of the index set converges locally
uniformly to the genus-one canonical product. -/
theorem SymmetricHeightExhaustion.tendstoLocallyUniformlyOn_genusOneProduct
    {ι : Type*} [DecidableEq ι]
    {root : ι → ℂ} {cutoff : ℕ → Finset ι} {height : ℕ → ℝ}
    (hexhaust : SymmetricHeightExhaustion root cutoff height)
    (hprod : HasProdLocallyUniformlyOn (genusOneCanonicalFactor root)
      (fun s ↦ ∏' i, genusOneCanonicalFactor root i s) Set.univ) :
    TendstoLocallyUniformlyOn
      (fun N s ↦ ∏ i ∈ cutoff N, genusOneCanonicalFactor root i s)
      (fun s ↦ ∏' i, genusOneCanonicalFactor root i s)
      atTop Set.univ := by
  intro u hu x hx
  obtain ⟨t, ht, hevent⟩ := hprod u hu x hx
  exact ⟨t, ht, hexhaust.tendsto_cutoff.eventually hevent⟩

/-- Closed radial xi windows are cofinal among all finite subsets of the
multiplicity index. -/
theorem riemannXiZeroIndexWindow_nat_tendsto_atTop :
    Tendsto (fun N : ℕ ↦ riemannXiZeroIndexWindow (N : ℝ))
      atTop atTop := by
  apply tendsto_atTop.2
  intro F
  let R : ℝ := ∑ i ∈ F, ‖riemannXiZeroRoot i‖
  obtain ⟨n, hn⟩ := exists_nat_ge R
  filter_upwards [eventually_ge_atTop n] with N hN
  intro i hi
  rw [mem_riemannXiZeroIndexWindow]
  calc
    ‖riemannXiZeroRoot i‖ ≤ R := by
      dsimp [R]
      exact Finset.single_le_sum (fun j _ ↦ norm_nonneg _) hi
    _ ≤ n := hn
    _ ≤ N := by exact_mod_cast hN

/-- Therefore radial finite genus-one products converge locally uniformly to
the unconditional infinite xi canonical product. -/
theorem riemannXiRadialGenusOneProducts_tendstoLocallyUniformly :
    TendstoLocallyUniformlyOn
      (fun (N : ℕ) s ↦
        ∏ i ∈ riemannXiZeroIndexWindow (N : ℝ),
          genusOneCanonicalFactor riemannXiZeroRoot i s)
      (fun s ↦ ∏' i : RiemannXiZeroIndex,
        genusOneCanonicalFactor riemannXiZeroRoot i s)
      atTop Set.univ := by
  intro u hu x hx
  obtain ⟨t, ht, hevent⟩ :=
    riemannXiGenusOneCanonicalProduct_hasProdLocallyUniformly u hu x hx
  exact ⟨t, ht,
    riemannXiZeroIndexWindow_nat_tendsto_atTop.eventually hevent⟩

/-- Exact reflection-stable height windows give a second concrete normal
exhaustion of the xi genus-one product. -/
theorem riemannXiHeightGenusOneProducts_tendstoLocallyUniformly :
    TendstoLocallyUniformlyOn
      (fun (N : ℕ) s ↦
        ∏ i ∈ riemannXiZeroHeightWindow (N : ℝ),
          genusOneCanonicalFactor riemannXiZeroRoot i s)
      (fun s ↦ ∏' i : RiemannXiZeroIndex,
        genusOneCanonicalFactor riemannXiZeroRoot i s)
      atTop Set.univ := by
  intro u hu x hx
  obtain ⟨t, ht, hevent⟩ :=
    riemannXiGenusOneCanonicalProduct_hasProdLocallyUniformly u hu x hx
  exact ⟨t, ht,
    riemannXiZeroHeightWindow_nat_tendsto_atTop.eventually hevent⟩

/-- Hadamard's remaining zeta-specific identification: xi is the normally
convergent genus-one product times an exponential affine prefactor. -/
def GenusOneHadamardRepresentation {ι : Type*}
    (root : ι → ℂ) (xi : ℂ → ℂ) : Prop :=
  ∃ a b : ℂ, ∀ s,
    xi s = Complex.exp (a + b * s) *
      ∏' i, genusOneCanonicalFactor root i s

/-- The simply-connected-domain component of Hadamard factorization:
every continuous zero-free function on the complex plane has a global
continuous logarithm.  Upgrading this lift to an entire logarithm and proving
it affine requires the missing analytic growth theorem. -/
theorem exists_continuous_log_of_continuous_ne_zero
    {g : ℂ → ℂ} (hg : Continuous g) (hne : ∀ z, g z ≠ 0) :
    ∃ f : ℂ → ℂ, Continuous f ∧
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
  refine ⟨f, continuousOn_univ.mp hfcont, ?_⟩
  intro z
  simpa [Function.comp_apply] using hf (Set.mem_univ z)

/-- A continuous exponential lift of an entire zero-free function is itself
entire.  Locally it is the principal logarithm of `g(w) / g(z)`, translated
by the chosen lift value. -/
theorem analyticOnNhd_log_of_continuous_exp
    {g f : ℂ → ℂ}
    (hg : AnalyticOnNhd ℂ g Set.univ)
    (hf : Continuous f)
    (hexp : ∀ z, Complex.exp (f z) = g z) :
    AnalyticOnNhd ℂ f Set.univ := by
  intro z hz
  have hgz : g z ≠ 0 := by
    rw [← hexp]
    exact Complex.exp_ne_zero _
  have hratio : AnalyticAt ℂ (fun w ↦ g w / g z) z :=
    (hg z (Set.mem_univ z)).div_const
  have hratio1 : g z / g z = 1 := div_self hgz
  have hrhs : AnalyticAt ℂ
      (fun w ↦ f z + Complex.log (g w / g z)) z :=
    analyticAt_const.add (hratio.clog (by simp [hratio1]))
  have hclose : ∀ᶠ w in 𝓝 z,
      dist (f w) (f z) < Real.pi / 2 :=
    hf.continuousAt.tendsto.eventually
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
      rw [Complex.exp_sub, hexp, hexp]
    calc
      f w = f z + (f w - f z) := by ring
      _ = f z + Complex.log (Complex.exp (f w - f z)) := by
        rw [Complex.log_exp hlower hupper]
      _ = f z + Complex.log (g w / g z) := by rw [hexpdiff]
  exact hrhs.congr heq.symm

theorem exists_analytic_log_of_analytic_ne_zero
    {g : ℂ → ℂ} (hg : AnalyticOnNhd ℂ g Set.univ)
    (hne : ∀ z, g z ≠ 0) :
    ∃ f : ℂ → ℂ, AnalyticOnNhd ℂ f Set.univ ∧
      ∀ z, Complex.exp (f z) = g z := by
  have hgcont : Continuous g :=
    continuousOn_univ.mp hg.continuousOn
  obtain ⟨f, hfcont, hfexp⟩ :=
    exists_continuous_log_of_continuous_ne_zero hgcont hne
  exact ⟨f, analyticOnNhd_log_of_continuous_exp hg hfcont hfexp, hfexp⟩

theorem deriv_analytic_log_eq_logDeriv
    {g f : ℂ → ℂ}
    (hf : AnalyticOnNhd ℂ f Set.univ)
    (hexp : ∀ z, Complex.exp (f z) = g z) (z : ℂ) :
    deriv f z = logDeriv g z := by
  have hcomp : HasDerivAt (Complex.exp ∘ f)
      (Complex.exp (f z) * deriv f z) z :=
    (Complex.hasDerivAt_exp (f z)).comp z
      (hf z (Set.mem_univ z)).differentiableAt.hasDerivAt
  have heq : Complex.exp ∘ f = g := by
    funext w
    exact hexp w
  have hd : deriv g z = Complex.exp (f z) * deriv f z := by
    rw [← heq]
    exact hcomp.deriv
  rw [logDeriv_apply, hd, ← hexp]
  exact (mul_div_cancel_left₀ _ (Complex.exp_ne_zero (f z))).symm

/-- The removable extension of `xi / P`, obtained by putting the meromorphic
quotient into normal form.  This changes only the values at a discrete set,
precisely allowing common zeros to cancel. -/
noncomputable def cancelledEntireQuotient
    (xi P : ℂ → ℂ) : ℂ → ℂ :=
  toMeromorphicNFOn (xi * P⁻¹) Set.univ

/-- Away from a zero of the denominator, the removable cancelled quotient
agrees pointwise with the ordinary quotient. -/
theorem cancelledEntireQuotient_eq_div_of_ne
    {xi P : ℂ → ℂ} {z : ℂ}
    (hxiM : MeromorphicOn xi Set.univ)
    (hPM : MeromorphicOn P Set.univ)
    (hxi : AnalyticAt ℂ xi z) (hP : AnalyticAt ℂ P z)
    (hPz : P z ≠ 0) :
    cancelledEntireQuotient xi P z = xi z / P z := by
  have hqM : MeromorphicOn (xi * P⁻¹) Set.univ :=
    hxiM.mul hPM.inv
  have hrawA : AnalyticAt ℂ (xi * P⁻¹) z :=
    hxi.mul (hP.inv hPz)
  have hev :
      cancelledEntireQuotient xi P =ᶠ[𝓝 z] xi * P⁻¹ := by
    exact (toMeromorphicNFOn_eq_toMeromorphicNFAt_on_nhds
      hqM (Set.mem_univ z)).trans
        (Filter.Eventually.of_forall (fun w ↦ by
          rw [toMeromorphicNFAt_eq_self.2 hrawA.meromorphicNFAt]))
  simpa [div_eq_mul_inv] using hev.self_of_nhds

/-- An exponential upper bound for the numerator and exponential lower
bound for the denominator transfer additively to the cancelled quotient. -/
theorem norm_cancelledEntireQuotient_le_exp_add
    {xi P : ℂ → ℂ} {z : ℂ} {U L : ℝ}
    (hxiM : MeromorphicOn xi Set.univ)
    (hPM : MeromorphicOn P Set.univ)
    (hxi : AnalyticAt ℂ xi z) (hP : AnalyticAt ℂ P z)
    (hxiUpper : ‖xi z‖ ≤ Real.exp U)
    (hPLower : Real.exp (-L) ≤ ‖P z‖) :
    ‖cancelledEntireQuotient xi P z‖ ≤ Real.exp (U + L) := by
  have hPnorm : 0 < ‖P z‖ :=
    (Real.exp_pos (-L)).trans_le hPLower
  have hPz : P z ≠ 0 := norm_pos_iff.mp hPnorm
  rw [cancelledEntireQuotient_eq_div_of_ne
    hxiM hPM hxi hP hPz, norm_div]
  calc
    ‖xi z‖ / ‖P z‖ ≤ Real.exp U / Real.exp (-L) :=
      div_le_div₀ (Real.exp_pos _).le hxiUpper
        (Real.exp_pos _) hPLower
    _ = Real.exp (U + L) := by
      rw [div_eq_mul_inv, ← Real.exp_neg, neg_neg, ← Real.exp_add]

/-- Equal finite divisors make the normal-form quotient entire and zero-free.
This is the cancellation step that a pointwise reciprocal estimate cannot
replace near common zeros. -/
theorem cancelledEntireQuotient_analytic_ne_zero
    {xi P : ℂ → ℂ}
    (hxi : MeromorphicOn xi Set.univ)
    (hP : MeromorphicOn P Set.univ)
    (hxiFinite : ∀ z, meromorphicOrderAt xi z ≠ ⊤)
    (hPFinite : ∀ z, meromorphicOrderAt P z ≠ ⊤)
    (hdiv : MeromorphicOn.divisor xi Set.univ =
      MeromorphicOn.divisor P Set.univ) :
    AnalyticOnNhd ℂ (cancelledEntireQuotient xi P) Set.univ ∧
      ∀ z, cancelledEntireQuotient xi P z ≠ 0 := by
  let q := xi * P⁻¹
  have hq : MeromorphicOn q Set.univ := hxi.mul hP.inv
  have hqdiv : MeromorphicOn.divisor q Set.univ = 0 := by
    dsimp [q]
    rw [hxi.divisor_mul hP.inv
      (fun z _ ↦ hxiFinite z)
      (fun z _ ↦ by simpa [meromorphicOrderAt_inv] using hPFinite z),
      MeromorphicOn.divisor_inv, hdiv]
    abel
  have hnf : MeromorphicNFOn
      (cancelledEntireQuotient xi P) Set.univ :=
    meromorphicNFOn_toMeromorphicNFOn q Set.univ
  have hqdiv' :
      MeromorphicOn.divisor (cancelledEntireQuotient xi P) Set.univ = 0 := by
    change MeromorphicOn.divisor
      (toMeromorphicNFOn q Set.univ) Set.univ = 0
    rw [hq.divisor_of_toMeromorphicNFOn, hqdiv]
  have han : AnalyticOnNhd ℂ
      (cancelledEntireQuotient xi P) Set.univ := by
    rw [← hnf.divisor_nonneg_iff_analyticOnNhd, hqdiv']
  refine ⟨han, ?_⟩
  intro z
  rw [← (hnf (Set.mem_univ z)).meromorphicOrderAt_eq_zero_iff]
  have happ := MeromorphicOn.divisor_apply hnf.meromorphicOn
    (Set.mem_univ z)
  rw [hqdiv'] at happ
  have hor :
      meromorphicOrderAt (cancelledEntireQuotient xi P) z = 0 ∨
        meromorphicOrderAt (cancelledEntireQuotient xi P) z = ⊤ := by
    simpa using happ.symm
  apply hor.resolve_right
  rw [show cancelledEntireQuotient xi P =
      toMeromorphicNFOn q Set.univ by rfl,
    meromorphicOrderAt_toMeromorphicNFOn hq (Set.mem_univ z),
    meromorphicOrderAt_mul (hxi z (Set.mem_univ z))
      (hP.inv z (Set.mem_univ z)), meromorphicOrderAt_inv]
  exact WithTop.add_ne_top.mpr
    ⟨hxiFinite z, by simpa using hPFinite z⟩

/-- Little-`o(‖z‖²)` growth, stated without asymptotic notation so it can
feed directly into Cauchy's derivative estimate. -/
def SubquadraticNormGrowth (h : ℂ → ℂ) : Prop :=
  ∀ ε : ℝ, 0 < ε → ∃ R : ℝ, 0 ≤ R ∧
    ∀ z : ℂ, R ≤ ‖z‖ → ‖h z‖ ≤ ε * ‖z‖ ^ 2

/-- A real-part version of subquadratic growth.  The bound is uniform on
the disk of radius `r`; this is the exact form used by
Borel--Carathéodory below. -/
def SubquadraticRealPartGrowth (h : ℂ → ℂ) : Prop :=
  ∀ ε : ℝ, 0 < ε → ∃ R : ℝ, 0 ≤ R ∧
    ∀ r : ℝ, R ≤ r → ∀ z ∈ Metric.ball (0 : ℂ) r,
      (h z).re ≤ ε * r ^ 2

/-- Cauchy-estimate rigidity: an entire function of norm growth
`o(‖z‖²)` has zero second derivative. -/
theorem iteratedDeriv_two_eq_zero_of_subquadraticNormGrowth
    (h : ℂ → ℂ) (hh : AnalyticOnNhd ℂ h Set.univ)
    (hgrowth : SubquadraticNormGrowth h) :
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

/-- Borel--Carathéodory plus Cauchy's estimate: a uniform disk bound
`sup_{‖z‖<r} re (h z) = o(r²)` already forces an entire `h` to have
zero second derivative.  This is the form relevant to an analytic
logarithm, since its real part is the logarithm of the quotient norm. -/
theorem iteratedDeriv_two_eq_zero_of_subquadraticRealPartGrowth
    (h : ℂ → ℂ) (hh : AnalyticOnNhd ℂ h Set.univ)
    (hgrowth : SubquadraticRealPartGrowth h) :
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

/-- The real part of an exponential lift is exactly the logarithm of
the norm. -/
theorem re_eq_log_norm_of_exp_eq {g h : ℂ → ℂ}
    (hexp : ∀ z, Complex.exp (h z) = g z) (z : ℂ) :
    (h z).re = Real.log ‖g z‖ := by
  rw [← hexp, Complex.norm_exp, Real.log_exp]

/-- The exact quotient-growth obligation left by the Hadamard route:
the logarithm of the quotient norm is uniformly `o(r²)` on disks. -/
def SubquadraticLogNormGrowth (g : ℂ → ℂ) : Prop :=
  ∀ ε : ℝ, 0 < ε → ∃ R : ℝ, 0 ≤ R ∧
    ∀ r : ℝ, R ≤ r → ∀ z ∈ Metric.ball (0 : ℂ) r,
      Real.log ‖g z‖ ≤ ε * r ^ 2

/-- The exact boundary estimate a Cartan/minimum-modulus argument must
produce before maximum modulus can fill the disks.  The radius may increase,
but the bound remains measured against the requested radius. -/
def SubquadraticBoundaryLogNormGrowth (g : ℂ → ℂ) : Prop :=
  ∀ ε : ℝ, 0 < ε → ∃ r0 : ℝ, 0 ≤ r0 ∧
    ∀ r : ℝ, r0 ≤ r → ∃ R : ℝ, r ≤ R ∧ 0 < R ∧
      ∀ z ∈ Metric.sphere (0 : ℂ) R,
        Real.log ‖g z‖ ≤ ε * r ^ 2

/-- A minimal dyadic cofinal boundary interface. The selected circle itself
may have any larger radius; only its lower dyadic scale and the logarithmic
bound measured at that scale matter. -/
def DyadicSubquadraticBoundaryLogNormGrowth (g : ℂ → ℂ) : Prop :=
  ∀ ε : ℝ, 0 < ε → ∀ᶠ j : ℕ in atTop,
    ∃ R : ℝ, (2 : ℝ) ^ j ≤ R ∧ 0 < R ∧
      ∀ z ∈ Metric.sphere (0 : ℂ) R,
        Real.log ‖g z‖ ≤ ε * ((2 : ℝ) ^ j) ^ 2

/-- Dyadic selected circles suffice for the existing every-disk boundary
interface. Given an arbitrary radius, use the next dyadic scale; its square
costs at most a factor four, absorbed by requesting `ε/4`. -/
theorem subquadraticBoundaryLogNormGrowth_of_dyadic
    {g : ℂ → ℂ}
    (h : DyadicSubquadraticBoundaryLogNormGrowth g) :
    SubquadraticBoundaryLogNormGrowth g := by
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
  obtain ⟨k, hk, huniq⟩ :=
    existsUnique_dyadic_interval r hr1
  let j := k + 1
  have hNj : N ≤ j := by
    by_contra hnot
    have hjN : j < N := lt_of_not_ge hnot
    have hp : (2 : ℝ) ^ j ≤ (2 : ℝ) ^ N :=
      pow_le_pow_right₀ (by norm_num) hjN.le
    have := hk.2
    dsimp [j] at hp ⊢
    linarith
  obtain ⟨R, hHR, hR, hsphere⟩ := hN j hNj
  refine ⟨R, ?_, hR, ?_⟩
  · exact hk.2.le.trans hHR
  · intro z hz
    have hH2r : (2 : ℝ) ^ j ≤ 2 * r := by
      dsimp [j]
      rw [pow_succ]
      nlinarith [hk.1]
    exact (hsphere z hz).trans (by
      have hsq : ((2 : ℝ) ^ j) ^ 2 ≤ (2 * r) ^ 2 :=
        (sq_le_sq₀ (by positivity) (by positivity)).mpr hH2r
      nlinarith)

/-- A dyadic exponential norm estimate dominated by the common Cartan
majorant supplies the minimal dyadic boundary interface. -/
theorem dyadicSubquadraticBoundaryLogNormGrowth_of_majorant
    {g : ℂ → ℂ} (hg0 : ∀ z, g z ≠ 0)
    (Ksq Ksqrt Klin : ℝ)
    (hcircle : ∀ᶠ j : ℕ in atTop,
      ∃ R : ℝ, (2 : ℝ) ^ j ≤ R ∧ 0 < R ∧
        ∀ z ∈ Metric.sphere (0 : ℂ) R,
          ‖g z‖ ≤ Real.exp (
            DyadicCartanLossMajorant Ksq Ksqrt Klin j *
              ((2 : ℝ) ^ j) ^ 2)) :
    DyadicSubquadraticBoundaryLogNormGrowth g := by
  intro ε hε
  filter_upwards [hcircle,
    eventually_dyadicCartanLossMajorant_lt
      Ksq Ksqrt Klin ε hε] with j hj hmajor
  obtain ⟨R, hHR, hR, hbound⟩ := hj
  refine ⟨R, hHR, hR, ?_⟩
  intro z hz
  calc
    Real.log ‖g z‖ ≤ Real.log
        (Real.exp (DyadicCartanLossMajorant
          Ksq Ksqrt Klin j * ((2 : ℝ) ^ j) ^ 2)) :=
      Real.log_le_log (norm_pos_iff.mpr (hg0 z)) (hbound z hz)
    _ = DyadicCartanLossMajorant Ksq Ksqrt Klin j *
        ((2 : ℝ) ^ j) ^ 2 := Real.log_exp _
    _ ≤ ε * ((2 : ℝ) ^ j) ^ 2 := by
      gcongr

/-- Outside equal-radius disks around a finite root family, the norm of the
monic root product is bounded below by the corresponding power of the
radius.  Multiplicity is represented by repeated indices. -/
theorem norm_finsetProd_sub_root_lower
    {ι : Type*} (s : Finset ι) (root : ι → ℂ)
    {δ : ℝ} (hδ : 0 ≤ δ) {z : ℂ}
    (houtside : ∀ i ∈ s,
      z ∉ Metric.closedBall (root i) δ) :
    δ ^ s.card ≤ ‖∏ i ∈ s, (z - root i)‖ := by
  rw [norm_prod]
  calc
    δ ^ s.card = ∏ _i ∈ s, δ := by simp
    _ ≤ ∏ i ∈ s, ‖z - root i‖ := by
      apply Finset.prod_le_prod
      · intro i hi
        exact hδ
      · intro i hi
        have hdist : δ < dist z (root i) := by
          simpa [Metric.mem_closedBall] using houtside i hi
        simpa [dist_eq_norm] using hdist.le

/-- A weak but explicit finite Cartan lemma: put one equal-radius disk around
each indexed root.  The total diameter is less than `H`, and outside the
disks the root product has the displayed lower bound.  Unlike the sharper
classical merging lemma, this construction is elementary and multiplicity
exact. -/
theorem exists_equalRadius_exceptionalDisks_polynomial
    {ι : Type*} (s : Finset ι) (root : ι → ℂ)
    (hs : s.Nonempty) {H : ℝ} (hH : 0 < H) :
    ∃ radius : ι → ℝ,
      (∀ i ∈ s, 0 ≤ radius i) ∧
      2 * ∑ i ∈ s, radius i < H ∧
      ∀ z : ℂ,
        (∀ i ∈ s,
          z ∉ Metric.closedBall (root i) (radius i)) →
        (H / (3 * s.card)) ^ s.card ≤
          ‖∏ i ∈ s, (z - root i)‖ := by
  let δ : ℝ := H / (3 * s.card)
  refine ⟨fun _ ↦ δ, ?_, ?_, ?_⟩
  · intro i hi
    exact div_nonneg hH.le (by positivity)
  · simp only [sum_const, nsmul_eq_mul]
    dsimp [δ]
    have hcard : (0 : ℝ) < s.card := by
      exact_mod_cast hs.card_pos
    field_simp
    nlinarith
  · intro z hout
    simpa [δ] using
      norm_finsetProd_sub_root_lower s root
        (div_nonneg hH.le (by positivity)) hout

/-- Exact norm formula for one genus-one canonical factor. -/
theorem norm_genusOneCanonicalFactor_eq
    {ι : Type*} {root : ι → ℂ}
    (hroot0 : ∀ i, root i ≠ 0) (i : ι) (z : ℂ) :
    ‖genusOneCanonicalFactor root i z‖ =
      (‖z - root i‖ / ‖root i‖) *
        Real.exp ((z / root i).re) := by
  unfold genusOneCanonicalFactor genusOnePrimaryFactor
  rw [norm_mul, Complex.norm_exp]
  congr 1
  rw [show 1 - z / root i = (root i - z) / root i by
    field_simp [hroot0 i]]
  rw [norm_div, norm_sub_rev]

/-- One genus-one factor has an explicit lower bound away from its root
disk.  The exponential loss is bounded solely by `‖z‖/‖root i‖`. -/
theorem norm_genusOneCanonicalFactor_lower_outside
    {ι : Type*} {root : ι → ℂ}
    (hroot0 : ∀ i, root i ≠ 0)
    {δ : ℝ} {i : ι} {z : ℂ}
    (hout : z ∉ Metric.closedBall (root i) δ) :
    (δ / ‖root i‖) *
        Real.exp (-‖z‖ * ‖(root i)⁻¹‖) ≤
      ‖genusOneCanonicalFactor root i z‖ := by
  rw [norm_genusOneCanonicalFactor_eq hroot0]
  have hdist : δ ≤ ‖z - root i‖ := by
    have : δ < dist z (root i) := by
      simpa [Metric.mem_closedBall] using hout
    simpa [dist_eq_norm] using this.le
  have hdiv :
      δ / ‖root i‖ ≤ ‖z - root i‖ / ‖root i‖ :=
    div_le_div_of_nonneg_right hdist (norm_nonneg _)
  have hre :
      -‖z‖ * ‖(root i)⁻¹‖ ≤ (z / root i).re := by
    calc
      -‖z‖ * ‖(root i)⁻¹‖ = -‖z / root i‖ := by
        rw [norm_div, norm_inv, div_eq_mul_inv]
        ring
      _ ≤ (z / root i).re :=
        neg_le_of_abs_le (Complex.abs_re_le_norm _)
  exact mul_le_mul hdiv (Real.exp_le_exp.mpr hre)
    (Real.exp_pos _).le
    (div_nonneg (norm_nonneg _) (norm_nonneg _))

/-- Branch-safe quadratic lower bound for the genus-one primary factor on
the closed half-unit disk. -/
theorem norm_genusOnePrimaryFactor_lower_of_norm_le_half
    {w : ℂ} (hw : ‖w‖ ≤ 1 / 2) :
    Real.exp (-‖w‖ ^ 2) ≤ ‖genusOnePrimaryFactor w‖ := by
  have hwlt : ‖w‖ < 1 := hw.trans_lt (by norm_num)
  have hw1 : w ≠ 1 := by
    intro h
    rw [h, norm_one] at hw
    norm_num at hw
  let u : ℂ := Complex.log (1 - w)⁻¹ - w
  have hu : ‖u‖ ≤ ‖w‖ ^ 2 := by
    apply (Complex.norm_log_one_sub_inv_sub_self_le hwlt).trans
    have hden : (1 - ‖w‖)⁻¹ ≤ 2 := by
      rw [inv_le_iff_one_le_mul₀
        (by linarith [norm_nonneg w])]
      nlinarith
    have hsq : 0 ≤ ‖w‖ ^ 2 := sq_nonneg _
    calc
      ‖w‖ ^ 2 * (1 - ‖w‖)⁻¹ / 2
          ≤ ‖w‖ ^ 2 * 2 / 2 := by gcongr
      _ = ‖w‖ ^ 2 := by ring
  have hexp :
      Complex.exp (-u) = genusOnePrimaryFactor w := by
    dsimp [u, genusOnePrimaryFactor]
    rw [neg_sub, Complex.exp_sub, Complex.exp_log]
    · field_simp
    · exact inv_ne_zero (sub_ne_zero.mpr hw1.symm)
  rw [← hexp, Complex.norm_exp]
  apply Real.exp_le_exp.mpr
  have hre : -‖u‖ ≤ (-u).re := by
    have := neg_le_of_abs_le (Complex.abs_re_le_norm (-u))
    simpa using this
  exact (neg_le_neg hu).trans hre

/-- Product form of the explicit genus-one lower bound outside root disks. -/
theorem norm_finsetProd_genusOneCanonicalFactor_lower
    {ι : Type*} (s : Finset ι) (root : ι → ℂ)
    (hroot0 : ∀ i, root i ≠ 0)
    {δ : ℝ} (hδ : 0 ≤ δ) {z : ℂ}
    (hout : ∀ i ∈ s,
      z ∉ Metric.closedBall (root i) δ) :
    (∏ i ∈ s, (δ / ‖root i‖) *
      Real.exp (-‖z‖ * ‖(root i)⁻¹‖)) ≤
      ‖∏ i ∈ s, genusOneCanonicalFactor root i z‖ := by
  rw [norm_prod]
  apply Finset.prod_le_prod
  · intro i hi
    exact mul_nonneg (div_nonneg hδ (norm_nonneg _))
      (Real.exp_pos _).le
  · intro i hi
    exact norm_genusOneCanonicalFactor_lower_outside
      hroot0 (hout i hi)

/-- Quantitative control of a finite far-zero tail.  Genus-one quadratic
cancellation turns the inverse-square mass into an exponential bound for
the distance of the tail product from one. -/
theorem norm_finsetProd_genusOneCanonicalFactor_sub_one_le
    {ι : Type*} (s : Finset ι) (root : ι → ℂ) (z : ℂ)
    (hsmall : ∀ i ∈ s, ‖z / root i‖ ≤ 1) :
    ‖(∏ i ∈ s, genusOneCanonicalFactor root i z) - 1‖ ≤
      Real.exp
        (3 * ‖z‖ ^ 2 *
          ∑ i ∈ s, ‖(root i)⁻¹‖ ^ 2) - 1 := by
  have hprod := Finset.norm_prod_one_add_sub_one_le s
    (fun i ↦ genusOneCanonicalFactor root i z - 1)
  have hprod' :
      ‖(∏ i ∈ s, genusOneCanonicalFactor root i z) - 1‖ ≤
        Real.exp
          (∑ i ∈ s,
            ‖genusOneCanonicalFactor root i z - 1‖) - 1 := by
    convert hprod using 1
    simp
  apply hprod'.trans
  gcongr
  calc
    ∑ i ∈ s, ‖genusOneCanonicalFactor root i z - 1‖
        ≤ ∑ i ∈ s,
          3 * ‖z‖ ^ 2 * ‖(root i)⁻¹‖ ^ 2 := by
      apply Finset.sum_le_sum
      intro i hi
      have hf := norm_genusOnePrimaryFactor_sub_one_le
        (z / root i) (hsmall i hi)
      unfold genusOneCanonicalFactor
      exact hf.trans_eq (by
        rw [norm_div, norm_inv, div_eq_mul_inv, mul_pow]
        ring)
    _ = 3 * ‖z‖ ^ 2 *
        ∑ i ∈ s, ‖(root i)⁻¹‖ ^ 2 := by
      rw [Finset.mul_sum]

/-- Equal-radius exceptional disks and an explicit minimum-modulus bound for
a finite genus-one canonical product. -/
theorem exists_equalRadius_exceptionalDisks_genusOne
    {ι : Type*} (s : Finset ι) (root : ι → ℂ)
    (hroot0 : ∀ i, root i ≠ 0)
    (hs : s.Nonempty) {H : ℝ} (hH : 0 < H) :
    ∃ radius : ι → ℝ,
      (∀ i ∈ s, 0 ≤ radius i) ∧
      2 * ∑ i ∈ s, radius i < H ∧
      ∀ z : ℂ,
        (∀ i ∈ s,
          z ∉ Metric.closedBall (root i) (radius i)) →
        (∏ i ∈ s,
          ((H / (3 * s.card)) / ‖root i‖) *
            Real.exp (-‖z‖ * ‖(root i)⁻¹‖)) ≤
          ‖∏ i ∈ s,
            genusOneCanonicalFactor root i z‖ := by
  let δ : ℝ := H / (3 * s.card)
  refine ⟨fun _ ↦ δ, ?_, ?_, ?_⟩
  · intro i hi
    exact div_nonneg hH.le (by positivity)
  · simp only [sum_const, nsmul_eq_mul]
    dsimp [δ]
    have hcard : (0 : ℝ) < s.card := by
      exact_mod_cast hs.card_pos
    field_simp
    nlinarith
  · intro z hout
    simpa [δ] using
      norm_finsetProd_genusOneCanonicalFactor_lower
        s root hroot0 (div_nonneg hH.le (by positivity)) hout

/-- A finite family of radial intervals with total width smaller than
`b-a` cannot cover `[a,b]`.  This is the measure-theoretic circle-selection
component of Cartan's exceptional-disk argument, with exact factor `2`. -/
theorem exists_radius_avoiding_exceptionalIntervals
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

/-- Avoiding the radial interval of a closed disk makes the centered circle
disjoint from that disk. -/
theorem sphere_disjoint_closedBall_of_radius_not_mem
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

/-- Explicit finite exceptional-disk circle selection. -/
theorem exists_circle_avoiding_exceptionalDisks
    {ι : Type*} (s : Finset ι) (center : ι → ℂ)
    (radius : ι → ℝ)
    (hradius : ∀ i ∈ s, 0 ≤ radius i)
    {a b : ℝ}
    (hwidth : 2 * ∑ i ∈ s, radius i < b - a) :
    ∃ R ∈ Set.Icc a b, ∀ i ∈ s,
      Disjoint (Metric.sphere (0 : ℂ) R)
        (Metric.closedBall (center i) (radius i)) := by
  obtain ⟨R, hR, havoid⟩ :=
    exists_radius_avoiding_exceptionalIntervals
      s center radius hradius hwidth
  refine ⟨R, hR, ?_⟩
  intro i hi
  exact sphere_disjoint_closedBall_of_radius_not_mem
    (havoid i hi)

/-- Finite genus-one minimum modulus on a selected circle.  This combines
the equal-radius lower-product lemma with the exact exceptional-circle
selection theorem. -/
theorem exists_circle_norm_finsetProd_genusOneCanonicalFactor_lower
    {ι : Type*} (s : Finset ι) (root : ι → ℂ)
    (hroot0 : ∀ i, root i ≠ 0)
    (hs : s.Nonempty) {H a b : ℝ} (hH : 0 < H)
    (hwidth : H ≤ b - a) :
    ∃ R ∈ Set.Icc a b, ∀ z ∈ Metric.sphere (0 : ℂ) R,
      (∏ i ∈ s,
        ((H / (3 * s.card)) / ‖root i‖) *
          Real.exp (-‖z‖ * ‖(root i)⁻¹‖)) ≤
        ‖∏ i ∈ s,
          genusOneCanonicalFactor root i z‖ := by
  obtain ⟨radius, hrad, hsum, hlower⟩ :=
    exists_equalRadius_exceptionalDisks_genusOne
      s root hroot0 hs hH
  obtain ⟨R, hR, havoid⟩ :=
    exists_circle_avoiding_exceptionalDisks
      s root radius hrad (a := a) (b := b)
        (hsum.trans_le hwidth)
  refine ⟨R, hR, ?_⟩
  intro z hz
  apply hlower z
  intro i hi hiBall
  exact Set.disjoint_left.mp (havoid i hi) hz hiBall

/-- Xi-window specialization of the finite selected-circle minimum-modulus
theorem, with analytic multiplicities represented by the canonical index
window. -/
theorem exists_circle_norm_riemannXiWindowProduct_lower
    (T : ℝ)
    (hwindow : (riemannXiZeroIndexWindow T).Nonempty)
    {H a b : ℝ} (hH : 0 < H) (hwidth : H ≤ b - a) :
    ∃ R ∈ Set.Icc a b, ∀ z ∈ Metric.sphere (0 : ℂ) R,
      (∏ i ∈ riemannXiZeroIndexWindow T,
        ((H / (3 * (riemannXiZeroIndexWindow T).card)) /
            ‖riemannXiZeroRoot i‖) *
          Real.exp (-‖z‖ * ‖(riemannXiZeroRoot i)⁻¹‖)) ≤
        ‖∏ i ∈ riemannXiZeroIndexWindow T,
          genusOneCanonicalFactor riemannXiZeroRoot i z‖ :=
  exists_circle_norm_finsetProd_genusOneCanonicalFactor_lower
    (riemannXiZeroIndexWindow T) riemannXiZeroRoot
      riemannXiZeroRoot_ne_zero hwindow hH hwidth

/-- Cartan circle selection at the compatible head scale
`T_j = R_j = 2^j`. The available annulus `[R_j, 2R_j]` has exactly the
width used as the exceptional-disk budget. -/
theorem exists_circle_norm_riemannXiCompatibleWindowProduct_lower
    (j : ℕ)
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
            genusOneCanonicalFactor riemannXiZeroRoot i z‖ := by
  apply exists_circle_norm_riemannXiWindowProduct_lower
    ((2 : ℝ) ^ j) hwindow (H := (2 : ℝ) ^ j)
  · positivity
  · rw [pow_succ]
    ring_nf
    exact le_rfl

/-- Exact logarithmic loss of the finite genus-one Cartan head. -/
def riemannXiFiniteHeadLogLoss (T H : ℝ) (z : ℂ) : ℝ :=
  (∑ i ∈ riemannXiZeroIndexWindow T,
      Real.log ‖riemannXiZeroRoot i‖) +
    ‖z‖ * (∑ i ∈ riemannXiZeroIndexWindow T,
      ‖(riemannXiZeroRoot i)⁻¹‖) -
    ((riemannXiZeroIndexWindow T).card : ℝ) *
      Real.log (H / (3 * (riemannXiZeroIndexWindow T).card))

/-- The finite-head lower-bound expression is exactly the exponential of
the negative aggregate head loss; all logarithm branches and signs are
explicit. -/
theorem riemannXiCartanHeadLower_eq_exp_neg_logLoss
    (T H : ℝ) (hH : 0 < H)
    (hwindow : (riemannXiZeroIndexWindow T).Nonempty)
    (z : ℂ) :
    (∏ i ∈ riemannXiZeroIndexWindow T,
      ((H / (3 * (riemannXiZeroIndexWindow T).card)) /
          ‖riemannXiZeroRoot i‖) *
        Real.exp (-‖z‖ *
          ‖(riemannXiZeroRoot i)⁻¹‖)) =
      Real.exp (-riemannXiFiniteHeadLogLoss T H z) := by
  let s := riemannXiZeroIndexWindow T
  have hcard : (0 : ℝ) < s.card := by
    exact_mod_cast hwindow.card_pos
  have hδ : 0 < H / (3 * (s.card : ℝ)) := by positivity
  rw [show (∏ i ∈ riemannXiZeroIndexWindow T,
      ((H / (3 * (riemannXiZeroIndexWindow T).card)) /
          ‖riemannXiZeroRoot i‖) *
        Real.exp (-‖z‖ * ‖(riemannXiZeroRoot i)⁻¹‖)) =
      ∏ i ∈ riemannXiZeroIndexWindow T,
        Real.exp (Real.log
          (H / (3 * (riemannXiZeroIndexWindow T).card)) -
          Real.log ‖riemannXiZeroRoot i‖ -
          ‖z‖ * ‖(riemannXiZeroRoot i)⁻¹‖) by
    apply Finset.prod_congr rfl
    intro i hi
    rw [show -‖z‖ * ‖(riemannXiZeroRoot i)⁻¹‖ =
      -(‖z‖ * ‖(riemannXiZeroRoot i)⁻¹‖) by ring,
      Real.exp_neg]
    rw [Real.exp_sub, Real.exp_sub, Real.exp_log hδ,
      Real.exp_log
        (norm_pos_iff.mpr (riemannXiZeroRoot_ne_zero i))]
    ring]
  rw [← Real.exp_sum]
  congr 1
  simp only [riemannXiFiniteHeadLogLoss]
  rw [Finset.sum_sub_distrib, Finset.sum_sub_distrib,
    Finset.sum_const, nsmul_eq_mul, Finset.mul_sum]
  ring

/-- The logarithmic root-size loss in a finite xi window is bounded by its
cardinality times the endpoint logarithm. -/
theorem sum_log_norm_riemannXiZeroRoot_window_le (T : ℝ) :
    ∑ i ∈ riemannXiZeroIndexWindow T,
        Real.log ‖riemannXiZeroRoot i‖ ≤
      (riemannXiZeroIndexWindow T).card * Real.log T := by
  simpa using Finset.sum_le_card_nsmul
    (riemannXiZeroIndexWindow T)
    (fun i ↦ Real.log ‖riemannXiZeroRoot i‖)
    (Real.log T) (by
      intro i hi
      apply Real.log_le_log
      · exact norm_pos_iff.mpr (riemannXiZeroRoot_ne_zero i)
      · exact (mem_riemannXiZeroIndexWindow T i).mp hi)

/-- Cauchy--Schwarz controls the finite inverse-root sum by window
cardinality and the already proved global inverse-square mass. -/
theorem sum_norm_inv_riemannXiZeroRoot_window_sq_le (T : ℝ) :
    (∑ i ∈ riemannXiZeroIndexWindow T,
      ‖(riemannXiZeroRoot i)⁻¹‖) ^ 2 ≤
      (riemannXiZeroIndexWindow T).card *
        ∑' i : RiemannXiZeroIndex,
          ‖(riemannXiZeroRoot i)⁻¹‖ ^ 2 := by
  calc
    (∑ i ∈ riemannXiZeroIndexWindow T,
        ‖(riemannXiZeroRoot i)⁻¹‖) ^ 2
        ≤ (riemannXiZeroIndexWindow T).card *
          ∑ i ∈ riemannXiZeroIndexWindow T,
            ‖(riemannXiZeroRoot i)⁻¹‖ ^ 2 :=
      sq_sum_le_card_mul_sum_sq
    _ ≤ (riemannXiZeroIndexWindow T).card *
        ∑' i : RiemannXiZeroIndex,
          ‖(riemannXiZeroRoot i)⁻¹‖ ^ 2 := by
      gcongr
      exact summable_norm_inv_sq_riemannXiZeroRoot.sum_le_tsum _
        (fun _ _ ↦ sq_nonneg _)

/-- Explicit cardinality/root-size/inverse-root upper bound for the finite
Cartan head logarithmic loss. -/
theorem riemannXiFiniteHeadLogLoss_le (T H : ℝ) (z : ℂ) :
    riemannXiFiniteHeadLogLoss T H z ≤
      ((riemannXiZeroIndexWindow T).card : ℝ) * Real.log T +
        ‖z‖ * Real.sqrt
          (((riemannXiZeroIndexWindow T).card : ℝ) *
            ∑' i : RiemannXiZeroIndex,
              ‖(riemannXiZeroRoot i)⁻¹‖ ^ 2) -
        ((riemannXiZeroIndexWindow T).card : ℝ) *
          Real.log (H /
            (3 * (riemannXiZeroIndexWindow T).card)) := by
  have hmass : 0 ≤
      ((riemannXiZeroIndexWindow T).card : ℝ) *
        ∑' i : RiemannXiZeroIndex,
          ‖(riemannXiZeroRoot i)⁻¹‖ ^ 2 :=
    mul_nonneg (Nat.cast_nonneg _)
      (tsum_nonneg fun _ ↦ sq_nonneg _)
  have hinv :
      (∑ i ∈ riemannXiZeroIndexWindow T,
          ‖(riemannXiZeroRoot i)⁻¹‖) ≤
        Real.sqrt
          (((riemannXiZeroIndexWindow T).card : ℝ) *
            ∑' i : RiemannXiZeroIndex,
              ‖(riemannXiZeroRoot i)⁻¹‖ ^ 2) := by
    rw [Real.le_sqrt
      (Finset.sum_nonneg fun _ _ ↦ norm_nonneg _) hmass]
    exact sum_norm_inv_riemannXiZeroRoot_window_sq_le T
  unfold riemannXiFiniteHeadLogLoss
  gcongr
  · exact sum_log_norm_riemannXiZeroRoot_window_le T

/-- Multiplicity-index windows are cofinal among all finite index sets. -/
theorem riemannXiZeroIndexWindow_tendsto_atTop :
    Tendsto riemannXiZeroIndexWindow atTop atTop := by
  rw [tendsto_atTop]
  intro s
  let R : ℝ := ∑ i ∈ s, ‖riemannXiZeroRoot i‖
  filter_upwards [eventually_ge_atTop R] with r hr
  intro i hi
  rw [mem_riemannXiZeroIndexWindow]
  exact (Finset.single_le_sum (fun j _ ↦ norm_nonneg _)
    hi).trans hr

/-- The inverse-square mass outside a growing xi window tends to zero. -/
theorem riemannXiZeroRoot_inv_sq_tail_tendsto_zero :
    Tendsto (fun R : ℝ ↦
      ∑' i : {i : RiemannXiZeroIndex //
        i ∉ riemannXiZeroIndexWindow R},
          ‖(riemannXiZeroRoot i)⁻¹‖ ^ 2)
      atTop (𝓝 0) := by
  exact (tendsto_tsum_compl_atTop_zero
    (fun i : RiemannXiZeroIndex ↦
      ‖(riemannXiZeroRoot i)⁻¹‖ ^ 2)).comp
        riemannXiZeroIndexWindow_tendsto_atTop

/-- At every point, the norms of the genus-one factor deviations are
summable over the complete xi multiplicity enumeration. -/
theorem summable_norm_riemannXiGenusOneCanonicalFactor_sub_one
    (z : ℂ) :
    Summable (fun i : RiemannXiZeroIndex ↦
      ‖genusOneCanonicalFactor riemannXiZeroRoot i z - 1‖) := by
  apply summable_norm_genusOneCanonicalFactor_sub_one
    riemannXiZeroRoot summable_norm_inv_sq_riemannXiZeroRoot
  intro K hK
  obtain ⟨B, hB⟩ := hK.isBounded.exists_norm_le
  filter_upwards [riemannXiZeroRoot_escape.eventually
    (eventually_ge_atTop B)] with i hi s hs
  rw [norm_div]
  exact (div_le_one
    (norm_pos_iff.mpr (riemannXiZeroRoot_ne_zero i))).2
      ((hB s hs).trans hi)

/-- Quantitative infinite-tail estimate.  This is the limit of the finite
tail estimate and makes the dependence on the remaining inverse-square mass
explicit. -/
theorem norm_riemannXiGenusOneCanonicalProduct_tail_sub_one_le
    {R : ℝ} {z : ℂ} (hz : ‖z‖ ≤ R) :
    ‖(∏' i : {i : RiemannXiZeroIndex //
        i ∉ riemannXiZeroIndexWindow R},
        genusOneCanonicalFactor riemannXiZeroRoot i z) - 1‖ ≤
      Real.exp (3 * ‖z‖ ^ 2 *
        ∑' i : {i : RiemannXiZeroIndex //
          i ∉ riemannXiZeroIndexWindow R},
          ‖(riemannXiZeroRoot i)⁻¹‖ ^ 2) - 1 := by
  let κ := {i : RiemannXiZeroIndex //
    i ∉ riemannXiZeroIndexWindow R}
  let f : κ → ℂ := fun i ↦
    genusOneCanonicalFactor riemannXiZeroRoot i z
  have hdev : Summable (fun i : κ ↦ ‖f i - 1‖) := by
    exact
      (summable_norm_riemannXiGenusOneCanonicalFactor_sub_one z).subtype
        (fun i : RiemannXiZeroIndex ↦
          i ∉ riemannXiZeroIndexWindow R)
  have hm : Multipliable f := by
    have h := multipliable_one_add_of_summable
      (f := fun i : κ ↦ f i - 1) hdev
    simpa [f] using h
  have hlim : Tendsto (fun s : Finset κ ↦
      ‖(∏ i ∈ s, f i) - 1‖) atTop
      (𝓝 ‖(∏' i : κ, f i) - 1‖) := by
    have hc : ContinuousAt (fun w : ℂ ↦ ‖w - 1‖)
        (∏' i : κ, f i) := by fun_prop
    exact hc.tendsto.comp hm.hasProd
  apply le_of_tendsto hlim
  filter_upwards [] with s
  have hsmall :
      ∀ i ∈ s, ‖z / riemannXiZeroRoot i‖ ≤ 1 := by
    intro i hi
    rw [norm_div]
    apply (div_le_one
      (norm_pos_iff.mpr (riemannXiZeroRoot_ne_zero i))).2
    have hiR : R < ‖riemannXiZeroRoot i‖ := by
      simpa using i.property
    exact hz.trans hiR.le
  have hfinite :=
    norm_finsetProd_genusOneCanonicalFactor_sub_one_le
      s (fun i : κ ↦ riemannXiZeroRoot i) z hsmall
  dsimp [f] at *
  apply hfinite.trans
  gcongr
  exact (summable_norm_inv_sq_riemannXiZeroRoot.subtype
    (fun i : RiemannXiZeroIndex ↦
      i ∉ riemannXiZeroIndexWindow R)).sum_le_tsum _
        (fun _ _ ↦ sq_nonneg _)

/-- Uniform-on-the-window form of the infinite-tail estimate. -/
theorem norm_riemannXiGenusOneCanonicalProduct_tail_sub_one_le_uniform
    {R : ℝ} {z : ℂ} (hz : ‖z‖ ≤ R) :
    ‖(∏' i : {i : RiemannXiZeroIndex //
        i ∉ riemannXiZeroIndexWindow R},
        genusOneCanonicalFactor riemannXiZeroRoot i z) - 1‖ ≤
      Real.exp (3 * R ^ 2 *
        ∑' i : {i : RiemannXiZeroIndex //
          i ∉ riemannXiZeroIndexWindow R},
          ‖(riemannXiZeroRoot i)⁻¹‖ ^ 2) - 1 := by
  apply
    (norm_riemannXiGenusOneCanonicalProduct_tail_sub_one_le hz).trans
  gcongr

/-- Multiplicative lower bound for the complete genus-one tail. The
half-radius separation selects the principal logarithm safely and gives the
correct negative quadratic exponent, rather than a useless large additive
deviation from `1`. -/
theorem norm_riemannXiGenusOneCanonicalProduct_tail_lower
    {T : ℝ} {z : ℂ} (hz : ‖z‖ ≤ T / 2) :
    Real.exp (-‖z‖ ^ 2 *
      ∑' i : {i : RiemannXiZeroIndex //
        i ∉ riemannXiZeroIndexWindow T},
        ‖(riemannXiZeroRoot i)⁻¹‖ ^ 2) ≤
      ‖∏' i : {i : RiemannXiZeroIndex //
        i ∉ riemannXiZeroIndexWindow T},
        genusOneCanonicalFactor riemannXiZeroRoot i z‖ := by
  let κ := {i : RiemannXiZeroIndex //
    i ∉ riemannXiZeroIndexWindow T}
  let f : κ → ℂ := fun i ↦
    genusOneCanonicalFactor riemannXiZeroRoot i z
  have hdev : Summable (fun i : κ ↦ ‖f i - 1‖) :=
    (summable_norm_riemannXiGenusOneCanonicalFactor_sub_one z).subtype _
  have hm : Multipliable f := by
    have h := multipliable_one_add_of_summable
      (f := fun i : κ ↦ f i - 1) hdev
    simpa [f] using h
  have hlim : Tendsto (fun s : Finset κ ↦
      ‖∏ i ∈ s, f i‖) atTop (𝓝 ‖∏' i : κ, f i‖) := by
    have hc : ContinuousAt (fun w : ℂ ↦ ‖w‖)
        (∏' i : κ, f i) := by fun_prop
    exact hc.tendsto.comp hm.hasProd
  apply ge_of_tendsto hlim
  filter_upwards [] with s
  have hsmall : ∀ i : κ,
      ‖z / riemannXiZeroRoot i‖ ≤ 1 / 2 := by
    intro i
    have hiT : T < ‖riemannXiZeroRoot i‖ := by
      simpa using i.property
    rw [norm_div]
    apply (div_le_iff₀ (norm_pos_iff.mpr
      (riemannXiZeroRoot_ne_zero i))).2
    nlinarith
  rw [norm_prod]
  calc
    Real.exp (-‖z‖ ^ 2 *
        ∑' i : κ, ‖(riemannXiZeroRoot i)⁻¹‖ ^ 2)
        ≤ Real.exp (-‖z‖ ^ 2 *
          ∑ i ∈ s, ‖(riemannXiZeroRoot i)⁻¹‖ ^ 2) := by
      apply Real.exp_le_exp.mpr
      have hsum :=
        (summable_norm_inv_sq_riemannXiZeroRoot.subtype
          (fun i : RiemannXiZeroIndex ↦
            i ∉ riemannXiZeroIndexWindow T)).sum_le_tsum s
          (fun _ _ ↦ sq_nonneg _)
      nlinarith [sq_nonneg ‖z‖]
    _ = ∏ i ∈ s,
        Real.exp (-‖z / riemannXiZeroRoot i‖ ^ 2) := by
      rw [← Real.exp_sum]
      congr 1
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro i hi
      rw [norm_div, norm_inv]
      field_simp [riemannXiZeroRoot_ne_zero i]
    _ ≤ ∏ i ∈ s, ‖f i‖ := by
      apply Finset.prod_le_prod
      · intro i hi
        positivity
      · intro i hi
        exact norm_genusOnePrimaryFactor_lower_of_norm_le_half
          (hsmall i)

/-- Exact finite-head/complement-tail split of the canonical xi product.
The two pieces are proved multipliable separately because `ℂ` is a
commutative monoid with zero, not a commutative group. -/
theorem riemannXiGenusOneCanonicalProduct_split_window
    (T : ℝ) (z : ℂ) :
    (∏ i ∈ riemannXiZeroIndexWindow T,
        genusOneCanonicalFactor riemannXiZeroRoot i z) *
      (∏' i : {i : RiemannXiZeroIndex //
          i ∉ riemannXiZeroIndexWindow T},
        genusOneCanonicalFactor riemannXiZeroRoot i z) =
      ∏' i : RiemannXiZeroIndex,
        genusOneCanonicalFactor riemannXiZeroRoot i z := by
  let f : RiemannXiZeroIndex → ℂ := fun i ↦
    genusOneCanonicalFactor riemannXiZeroRoot i z
  let S : Set RiemannXiZeroIndex :=
    ↑(riemannXiZeroIndexWindow T)
  let A : Set RiemannXiZeroIndex :=
    {i | i ∉ riemannXiZeroIndexWindow T}
  have hAS : A = Sᶜ := by
    ext i
    simp [A, S]
  let e : A ≃ (Sᶜ : Set RiemannXiZeroIndex) :=
    Equiv.setCongr hAS
  have hs : Multipliable (f ∘ (↑) : S → ℂ) :=
    (hasProd_fintype _).multipliable
  have hdev : Summable (fun i : {i : RiemannXiZeroIndex //
      i ∉ riemannXiZeroIndexWindow T} ↦ ‖f i - 1‖) :=
    (summable_norm_riemannXiGenusOneCanonicalFactor_sub_one z).subtype _
  have hsc0 : Multipliable (fun i : {i : RiemannXiZeroIndex //
      i ∉ riemannXiZeroIndexWindow T} ↦ f i) := by
    have h := multipliable_one_add_of_summable
      (f := fun i : {i : RiemannXiZeroIndex //
        i ∉ riemannXiZeroIndexWindow T} ↦ f i - 1) hdev
    simpa using h
  change Multipliable (f ∘ (↑) : A → ℂ) at hsc0
  have hsc : Multipliable
      (f ∘ (↑) : (Sᶜ : Set RiemannXiZeroIndex) → ℂ) :=
    (e.multipliable_iff).mp hsc0
  have hsplit :
      (∏' i : S, f i) *
          (∏' i : (Sᶜ : Set RiemannXiZeroIndex), f i) =
        ∏' i : RiemannXiZeroIndex, f i :=
    Multipliable.tprod_mul_tprod_compl (f := f) hs hsc
  have hhead :
      (∏' i : (↑(riemannXiZeroIndexWindow T) :
        Set RiemannXiZeroIndex), f i) =
        ∏ i ∈ riemannXiZeroIndexWindow T, f i :=
    Finset.tprod_subtype _ _
  have htail :
      (∏' i : A, f i) =
        ∏' i : (Sᶜ : Set RiemannXiZeroIndex), f i :=
    e.tprod_eq (fun i : (Sᶜ : Set RiemannXiZeroIndex) ↦ f i)
  rw [hhead] at hsplit
  rw [← htail] at hsplit
  simpa [f, A, S] using hsplit

/-- Multiplication of any finite-head lower bound with the branch-safe
infinite-tail estimate gives a lower bound for the full canonical product. -/
theorem norm_riemannXiGenusOneCanonicalProduct_lower_of_head
    {T L : ℝ} {z : ℂ}
    (hhead : L ≤ ‖∏ i ∈ riemannXiZeroIndexWindow T,
      genusOneCanonicalFactor riemannXiZeroRoot i z‖)
    (hz : ‖z‖ ≤ T / 2) :
    L * Real.exp (-‖z‖ ^ 2 *
      ∑' i : {i : RiemannXiZeroIndex //
        i ∉ riemannXiZeroIndexWindow T},
        ‖(riemannXiZeroRoot i)⁻¹‖ ^ 2) ≤
      ‖∏' i : RiemannXiZeroIndex,
        genusOneCanonicalFactor riemannXiZeroRoot i z‖ := by
  have htail :=
    norm_riemannXiGenusOneCanonicalProduct_tail_lower hz
  rw [← riemannXiGenusOneCanonicalProduct_split_window T z,
    norm_mul]
  exact mul_le_mul hhead htail (Real.exp_pos _).le (norm_nonneg _)

/-- Full canonical-product minimum modulus on a selected dyadic Cartan
circle. The head cutoff is four times the inner circle scale, so every tail
factor lies in the half-unit logarithm disk. -/
theorem exists_circle_norm_riemannXiGenusOneCanonicalProduct_lower
    (j : ℕ)
    (hwindow :
      (riemannXiZeroIndexWindow
        ((2 : ℝ) ^ (j + 2))).Nonempty) :
    ∃ R ∈ Set.Icc ((2 : ℝ) ^ j) ((2 : ℝ) ^ (j + 1)),
      ∀ z ∈ Metric.sphere (0 : ℂ) R,
        (∏ i ∈
            riemannXiZeroIndexWindow ((2 : ℝ) ^ (j + 2)),
          ((((2 : ℝ) ^ j) /
              (3 * (riemannXiZeroIndexWindow
                ((2 : ℝ) ^ (j + 2))).card)) /
              ‖riemannXiZeroRoot i‖) *
            Real.exp (-‖z‖ *
              ‖(riemannXiZeroRoot i)⁻¹‖)) *
          Real.exp (-‖z‖ ^ 2 *
            ∑' i : {i : RiemannXiZeroIndex //
              i ∉ riemannXiZeroIndexWindow
                ((2 : ℝ) ^ (j + 2))},
              ‖(riemannXiZeroRoot i)⁻¹‖ ^ 2) ≤
        ‖∏' i : RiemannXiZeroIndex,
          genusOneCanonicalFactor riemannXiZeroRoot i z‖ := by
  obtain ⟨R, hR, hhead⟩ :=
    exists_circle_norm_riemannXiWindowProduct_lower
      ((2 : ℝ) ^ (j + 2)) hwindow
      (H := (2 : ℝ) ^ j) (a := (2 : ℝ) ^ j)
      (b := (2 : ℝ) ^ (j + 1)) (by positivity) (by
        rw [pow_succ]
        ring_nf
        exact le_rfl)
  refine ⟨R, hR, ?_⟩
  intro z hz
  apply norm_riemannXiGenusOneCanonicalProduct_lower_of_head
    (hhead z hz)
  have hzNorm : ‖z‖ = R := by
    simpa [Metric.mem_sphere, dist_zero_right] using hz
  rw [hzNorm]
  have hpow : (2 : ℝ) ^ (j + 2) / 2 =
      (2 : ℝ) ^ (j + 1) := by
    rw [show j + 2 = (j + 1) + 1 by omega, pow_succ]
    ring
  rw [hpow]
  exact hR.2

/-- Total explicit logarithmic loss of the selected-circle canonical-product
lower bound. -/
def riemannXiCanonicalCircleLogLoss (T H : ℝ) (z : ℂ) : ℝ :=
  riemannXiFiniteHeadLogLoss T H z +
    ‖z‖ ^ 2 *
      ∑' i : {i : RiemannXiZeroIndex //
        i ∉ riemannXiZeroIndexWindow T},
        ‖(riemannXiZeroRoot i)⁻¹‖ ^ 2

/-- Logarithmic form of the full selected-circle minimum modulus. -/
theorem exists_circle_exp_neg_logLoss_le_norm_riemannXiCanonicalProduct
    (j : ℕ)
    (hwindow :
      (riemannXiZeroIndexWindow
        ((2 : ℝ) ^ (j + 2))).Nonempty) :
    ∃ R ∈ Set.Icc ((2 : ℝ) ^ j) ((2 : ℝ) ^ (j + 1)),
      ∀ z ∈ Metric.sphere (0 : ℂ) R,
        Real.exp (-riemannXiCanonicalCircleLogLoss
          ((2 : ℝ) ^ (j + 2)) ((2 : ℝ) ^ j) z) ≤
          ‖∏' i : RiemannXiZeroIndex,
            genusOneCanonicalFactor riemannXiZeroRoot i z‖ := by
  obtain ⟨R, hR, hbound⟩ :=
    exists_circle_norm_riemannXiGenusOneCanonicalProduct_lower
      j hwindow
  refine ⟨R, hR, ?_⟩
  intro z hz
  have h := hbound z hz
  rw [riemannXiCartanHeadLower_eq_exp_neg_logLoss
    ((2 : ℝ) ^ (j + 2)) ((2 : ℝ) ^ j) (by positivity)
      hwindow z] at h
  rw [← Real.exp_add] at h
  unfold riemannXiCanonicalCircleLogLoss
  convert h using 1
  ring_nf

/-- Xi's global order-one upper bound and the canonical-product minimum
modulus give an explicit selected-circle upper bound for the cancelled
quotient. -/
theorem exists_circle_norm_cancelledRiemannXiQuotient_le_exp :
    ∃ C R0 : ℝ, 0 ≤ C ∧ 0 ≤ R0 ∧
      ∀ j : ℕ, R0 ≤ (2 : ℝ) ^ j →
      (riemannXiZeroIndexWindow
        ((2 : ℝ) ^ (j + 2))).Nonempty →
      ∃ R ∈ Set.Icc ((2 : ℝ) ^ j) ((2 : ℝ) ^ (j + 1)),
        ∀ z ∈ Metric.sphere (0 : ℂ) R,
          ‖cancelledEntireQuotient riemannXiLi
            (fun s ↦ ∏' i : RiemannXiZeroIndex,
              genusOneCanonicalFactor riemannXiZeroRoot i s) z‖ ≤
            Real.exp (
              C * ‖z‖ * Real.log (‖z‖ + 2) +
              riemannXiCanonicalCircleLogLoss
                ((2 : ℝ) ^ (j + 2)) ((2 : ℝ) ^ j) z) := by
  obtain ⟨C, R0, hC, hR0, hxiBound⟩ :=
    riemannXi_orderOneGrowthBound
  refine ⟨C, R0, hC, by linarith, ?_⟩
  intro j hj hwindow
  obtain ⟨R, hR, hPBound⟩ :=
    exists_circle_exp_neg_logLoss_le_norm_riemannXiCanonicalProduct
      j hwindow
  refine ⟨R, hR, ?_⟩
  intro z hz
  have hzNorm : ‖z‖ = R := by
    simpa [Metric.mem_sphere, dist_zero_right] using hz
  have hzLarge : R0 ≤ ‖z‖ := by
    rw [hzNorm]
    exact hj.trans hR.1
  let P : ℂ → ℂ := fun s ↦
    ∏' i : RiemannXiZeroIndex,
      genusOneCanonicalFactor riemannXiZeroRoot i s
  have hPA : AnalyticOnNhd ℂ P Set.univ := by
    dsimp [P]
    exact analyticOnNhd_riemannXiGenusOneCanonicalProduct
  apply norm_cancelledEntireQuotient_le_exp_add
    (xi := riemannXiLi) (P := P)
    (U := C * ‖z‖ * Real.log (‖z‖ + 2))
    (L := riemannXiCanonicalCircleLogLoss
      ((2 : ℝ) ^ (j + 2)) ((2 : ℝ) ^ j) z)
  · exact fun w hw ↦
      (differentiable_riemannXiLi.analyticAt w).meromorphicAt
  · exact fun w hw ↦ (hPA w hw).meromorphicAt
  · exact differentiable_riemannXiLi.analyticAt z
  · exact hPA z (Set.mem_univ z)
  · exact hxiBound z hzLarge
  · exact hPBound z hz

set_option maxHeartbeats 2000000 in
/-- Every term in the selected-circle quotient estimate is bounded by one
fixed dyadic Cartan majorant.  The constants respectively collect the
root/cardinality losses, Cauchy--Schwarz inverse-root loss, and xi/tail
losses. -/
theorem riemannXiCanonicalCircleExponent_le_dyadicMajorant
    (A B C : ℝ) (hA : 0 ≤ A) (hB : 0 ≤ B) (hC : 0 ≤ C)
    (j : ℕ)
    (hcard : ((riemannXiZeroIndexWindow
      ((2 : ℝ) ^ (j + 2))).card : ℝ) ≤
        B * (2 : ℝ) ^ j * (j + 1 : ℝ))
    (htail : (∑' i : {i : RiemannXiZeroIndex //
        i ∉ riemannXiZeroIndexWindow ((2 : ℝ) ^ (j + 2))},
          ‖(riemannXiZeroRoot i)⁻¹‖ ^ 2) ≤
      A * (j + 3 : ℝ) / (2 : ℝ) ^ (j + 2))
    (hwindow : (riemannXiZeroIndexWindow
      ((2 : ℝ) ^ (j + 2))).Nonempty)
    (z : ℂ) (hzlo : (2 : ℝ) ^ j ≤ ‖z‖)
    (hzhi : ‖z‖ ≤ (2 : ℝ) ^ (j + 1)) :
    C * ‖z‖ * Real.log (‖z‖ + 2) +
        riemannXiCanonicalCircleLogLoss
          ((2 : ℝ) ^ (j + 2)) ((2 : ℝ) ^ j) z ≤
      DyadicCartanLossMajorant
        (2 * B + 3 * B ^ 2)
        (4 * B * (∑' i : RiemannXiZeroIndex,
          ‖(riemannXiZeroRoot i)⁻¹‖ ^ 2))
        (4 * C + 3 * A) j * ((2 : ℝ) ^ j) ^ 2 := by
  let H : ℝ := (2 : ℝ) ^ j
  let T : ℝ := (2 : ℝ) ^ (j + 2)
  let n : ℝ := (riemannXiZeroIndexWindow T).card
  let M : ℝ := ∑' i : RiemannXiZeroIndex,
    ‖(riemannXiZeroRoot i)⁻¹‖ ^ 2
  let tail : ℝ := ∑' i : {i : RiemannXiZeroIndex //
      i ∉ riemannXiZeroIndexWindow T},
    ‖(riemannXiZeroRoot i)⁻¹‖ ^ 2
  have hH : 0 < H := by positivity
  have hn : 0 < n := by
    dsimp [n, T]
    exact_mod_cast hwindow.card_pos
  have hM : 0 ≤ M := tsum_nonneg fun _ ↦ sq_nonneg _
  have htail0 : 0 ≤ tail := tsum_nonneg fun _ ↦ sq_nonneg _
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
  have htail' : tail ≤ A * (j + 3 : ℝ) / (4 * H) := by
    calc
      tail ≤ A * (j + 3 : ℝ) / (2 : ℝ) ^ (j + 2) := by
        simpa [tail, T] using htail
      _ = A * (j + 3 : ℝ) / (4 * H) := by
        rw [show (2 : ℝ) ^ (j + 2) = 4 * H by exact hT]
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
  have hhead := riemannXiFiniteHeadLogLoss_le T H z
  have hlogInv :
      -(n * Real.log (H / (3 * n))) =
        n * Real.log (3 * n / H) := by
    calc
      _ = n * (-Real.log (H / (3 * n))) := by ring
      _ = n * Real.log ((H / (3 * n))⁻¹) := by rw [Real.log_inv]
      _ = n * Real.log (3 * n / H) := by
        congr 2
        field_simp
  have hhead' :
      riemannXiFiniteHeadLogLoss T H z ≤
        n * Real.log T + ‖z‖ * Real.sqrt (n * M) +
          n * Real.log (3 * n / H) := by
    change riemannXiFiniteHeadLogLoss T H z ≤
      n * Real.log T + ‖z‖ * Real.sqrt (n * M) -
        n * Real.log (H / (3 * n)) at hhead
    rw [show n * Real.log T + ‖z‖ * Real.sqrt (n * M) -
        n * Real.log (H / (3 * n)) =
      n * Real.log T + ‖z‖ * Real.sqrt (n * M) +
        (-(n * Real.log (H / (3 * n)))) by ring, hlogInv] at hhead
    exact hhead
  have hroot : n * Real.log T / H ^ 2 ≤
      2 * B * ((j + 1 : ℝ) ^ 2 / H) := by
    have hlogT0 : 0 ≤ Real.log T :=
      Real.log_nonneg (by dsimp [T]; exact one_le_pow₀ (by norm_num))
    rw [div_le_iff₀ (sq_pos_of_pos hH)]
    calc
      n * Real.log T ≤ (B * H * (j + 1 : ℝ)) *
          (2 * (j + 1 : ℝ)) := by gcongr
      _ = (2 * B * ((j + 1 : ℝ) ^ 2 / H)) * H ^ 2 := by
        field_simp
  have hcardlog : n * Real.log (3 * n / H) / H ^ 2 ≤
      3 * B ^ 2 * (j + 1 : ℝ) ^ 2 / H :=
    card_mul_log_three_mul_div_sq_le hn hB hH (by positivity) hcard'
  have hsqrt : ‖z‖ * Real.sqrt (n * M) / H ^ 2 ≤
      Real.sqrt (4 * B * M * ((j + 1 : ℝ) / H)) := by
    rw [Real.le_sqrt (by positivity) (by positivity)]
    have hzsq : ‖z‖ ^ 2 ≤ (2 * H) ^ 2 :=
      (sq_le_sq₀ (norm_nonneg _) (by positivity)).mpr hz2
    have hnm : n * M ≤ (B * H * (j + 1 : ℝ)) * M := by gcongr
    rw [div_pow, mul_pow, Real.sq_sqrt (mul_nonneg hn.le hM)]
    calc
      ‖z‖ ^ 2 * (n * M) / (H ^ 2) ^ 2
          ≤ (2 * H) ^ 2 * ((B * H * (j + 1 : ℝ)) * M) /
              (H ^ 2) ^ 2 := by gcongr
      _ = 4 * B * M * ((j + 1 : ℝ) / H) := by
        field_simp
        ring
  have hxi : C * ‖z‖ * Real.log (‖z‖ + 2) / H ^ 2 ≤
      4 * C * ((j + 1 : ℝ) / H) := by
    have hlogz0 : 0 ≤ Real.log (‖z‖ + 2) :=
      Real.log_nonneg (by linarith [norm_nonneg z])
    rw [div_le_iff₀ (sq_pos_of_pos hH)]
    calc
      C * ‖z‖ * Real.log (‖z‖ + 2) ≤
          C * (2 * H) * (2 * (j + 1 : ℝ)) := by gcongr
      _ = (4 * C * ((j + 1 : ℝ) / H)) * H ^ 2 := by
        field_simp
        ring
  have htailLoss : ‖z‖ ^ 2 * tail / H ^ 2 ≤
      3 * A * ((j + 1 : ℝ) / H) := by
    calc
      ‖z‖ ^ 2 * tail / H ^ 2 ≤
          (2 * H) ^ 2 * (A * (j + 3 : ℝ) / (4 * H)) /
            H ^ 2 := by gcongr
      _ = A * (j + 3 : ℝ) / H := by
        field_simp
        ring
      _ ≤ 3 * A * ((j + 1 : ℝ) / H) := by
        rw [show 3 * A * ((j + 1 : ℝ) / H) =
          (3 * A * (j + 1 : ℝ)) / H by ring]
        apply div_le_div_of_nonneg_right _ hH.le
        nlinarith [Nat.cast_nonneg (α := ℝ) j]
  have hnorm :
      (C * ‖z‖ * Real.log (‖z‖ + 2) +
          riemannXiCanonicalCircleLogLoss T H z) / H ^ 2 ≤
        DyadicCartanLossMajorant
          (2 * B + 3 * B ^ 2) (4 * B * M) (4 * C + 3 * A) j := by
    calc
      _ ≤ (C * ‖z‖ * Real.log (‖z‖ + 2) +
              (n * Real.log T + ‖z‖ * Real.sqrt (n * M) +
                n * Real.log (3 * n / H)) +
              ‖z‖ ^ 2 * tail) / H ^ 2 := by
            unfold riemannXiCanonicalCircleLogLoss
            dsimp [tail]
            apply div_le_div_of_nonneg_right _ (sq_nonneg H)
            linarith
      _ = C * ‖z‖ * Real.log (‖z‖ + 2) / H ^ 2 +
            n * Real.log T / H ^ 2 +
            ‖z‖ * Real.sqrt (n * M) / H ^ 2 +
            n * Real.log (3 * n / H) / H ^ 2 +
            ‖z‖ ^ 2 * tail / H ^ 2 := by ring
      _ ≤ 4 * C * ((j + 1 : ℝ) / H) +
            2 * B * ((j + 1 : ℝ) ^ 2 / H) +
            Real.sqrt (4 * B * M * ((j + 1 : ℝ) / H)) +
            3 * B ^ 2 * (j + 1 : ℝ) ^ 2 / H +
            3 * A * ((j + 1 : ℝ) / H) := by gcongr
      _ = DyadicCartanLossMajorant
          (2 * B + 3 * B ^ 2) (4 * B * M) (4 * C + 3 * A) j := by
            unfold DyadicCartanLossMajorant
            dsimp [H]
            ring
  change C * ‖z‖ * Real.log (‖z‖ + 2) +
      riemannXiCanonicalCircleLogLoss T H z ≤
    DyadicCartanLossMajorant
      (2 * B + 3 * B ^ 2) (4 * B * M) (4 * C + 3 * A) j * H ^ 2
  exact (div_le_iff₀ (sq_pos_of_pos hH)).mp hnorm

/-- Once the xi-zero index is inhabited, the sourced count, radial-tail, xi
growth, and selected-circle estimates combine with fixed constants uniformly
for all sufficiently large dyadic scales. -/
theorem exists_eventually_circle_norm_cancelledRiemannXiQuotient_le_majorant
    [Nonempty RiemannXiZeroIndex] :
    ∃ Ksq Ksqrt Klin : ℝ,
      ∀ᶠ j : ℕ in atTop,
        ∃ R : ℝ, (2 : ℝ) ^ j ≤ R ∧ 0 < R ∧
          ∀ z ∈ Metric.sphere (0 : ℂ) R,
            ‖cancelledEntireQuotient riemannXiLi
              (fun s ↦ ∏' i : RiemannXiZeroIndex,
                genusOneCanonicalFactor riemannXiZeroRoot i s) z‖ ≤
              Real.exp (DyadicCartanLossMajorant Ksq Ksqrt Klin j *
                ((2 : ℝ) ^ j) ^ 2) := by
  obtain ⟨A, RA, hA, hRA, htail⟩ :=
    exists_riemannXiZeroRoot_inv_sq_radial_tail_bound
  obtain ⟨B, RB, hB, hRB, hcard⟩ :=
    exists_riemannXiZeroIndexWindow_card_le_dyadic
  obtain ⟨C, RC, hC, hRC, hcircle⟩ :=
    exists_circle_norm_cancelledRiemannXiQuotient_le_exp
  let M : ℝ := ∑' i : RiemannXiZeroIndex,
    ‖(riemannXiZeroRoot i)⁻¹‖ ^ 2
  refine ⟨2 * B + 3 * B ^ 2, 4 * B * M, 4 * C + 3 * A, ?_⟩
  let i : RiemannXiZeroIndex := Classical.choice inferInstance
  have hp : Tendsto (fun j : ℕ ↦ (2 : ℝ) ^ j) atTop atTop :=
    tendsto_pow_atTop_atTop_of_one_lt (by norm_num)
  have hlarge : ∀ᶠ j : ℕ in atTop,
      max (max RA RB) (max RC ‖riemannXiZeroRoot i‖) ≤
        (2 : ℝ) ^ j := hp.eventually
      (eventually_ge_atTop
        (max (max RA RB) (max RC ‖riemannXiZeroRoot i‖)))
  filter_upwards [hlarge] with j hj
  have hjRA : RA ≤ (2 : ℝ) ^ (j + 2) := by
    calc
      RA ≤ (2 : ℝ) ^ j :=
        (le_max_left _ _).trans ((le_max_left _ _).trans hj)
      _ ≤ (2 : ℝ) ^ (j + 2) :=
        pow_le_pow_right₀ (by norm_num) (by omega)
  have hjRB : RB ≤ (2 : ℝ) ^ (j + 2) := by
    calc
      RB ≤ (2 : ℝ) ^ j :=
        (le_max_right RA RB).trans ((le_max_left _ _).trans hj)
      _ ≤ (2 : ℝ) ^ (j + 2) :=
        pow_le_pow_right₀ (by norm_num) (by omega)
  have hjRC : RC ≤ (2 : ℝ) ^ j :=
    (le_max_left RC ‖riemannXiZeroRoot i‖).trans
      ((le_max_right _ _).trans hj)
  have hiNorm : ‖riemannXiZeroRoot i‖ ≤ (2 : ℝ) ^ (j + 2) := by
    calc
      _ ≤ (2 : ℝ) ^ j :=
        (le_max_right RC _).trans ((le_max_right _ _).trans hj)
      _ ≤ (2 : ℝ) ^ (j + 2) :=
        pow_le_pow_right₀ (by norm_num) (by omega)
  have hwindow :
      (riemannXiZeroIndexWindow ((2 : ℝ) ^ (j + 2))).Nonempty :=
    ⟨i, (mem_riemannXiZeroIndexWindow _ _).mpr hiNorm⟩
  have htailj :
      (∑' k : {k : RiemannXiZeroIndex //
          k ∉ riemannXiZeroIndexWindow ((2 : ℝ) ^ (j + 2))},
        ‖(riemannXiZeroRoot k)⁻¹‖ ^ 2) ≤
        A * (j + 3 : ℝ) / (2 : ℝ) ^ (j + 2) := by
    have ht := htail (j + 2) hjRA
    have heq : ((j + 2 : ℕ) : ℝ) + 1 = (j + 3 : ℝ) := by
      push_cast
      ring
    rw [heq] at ht
    exact ht
  obtain ⟨R, hR, hq⟩ := hcircle j hjRC hwindow
  refine ⟨R, hR.1,
    (show 0 < (2 : ℝ) ^ j by positivity).trans_le hR.1, ?_⟩
  intro z hz
  have hzNorm : ‖z‖ = R := by
    simpa [Metric.mem_sphere, dist_zero_right] using hz
  have hdom := riemannXiCanonicalCircleExponent_le_dyadicMajorant
    A B C hA hB hC j (hcard j hjRB) htailj hwindow z
    (by rw [hzNorm]; exact hR.1)
    (by rw [hzNorm]; exact hR.2)
  exact (hq z hz).trans (Real.exp_le_exp.mpr hdom)

/-- The explicit fixed-constant estimate discharges the dyadic boundary
hypothesis whenever the canonical xi-zero enumeration is inhabited. -/
theorem riemannXiCancelledQuotient_dyadicBoundary
    [Nonempty RiemannXiZeroIndex] :
    DyadicSubquadraticBoundaryLogNormGrowth
      (cancelledEntireQuotient riemannXiLi
        (fun s ↦ ∏' i : RiemannXiZeroIndex,
          genusOneCanonicalFactor riemannXiZeroRoot i s)) := by
  obtain ⟨Ksq, Ksqrt, Klin, hcircle⟩ :=
    exists_eventually_circle_norm_cancelledRiemannXiQuotient_le_majorant
  intro ε hε
  filter_upwards [hcircle,
    eventually_dyadicCartanLossMajorant_lt Ksq Ksqrt Klin ε hε]
    with j hj hmajor
  obtain ⟨R, hHR, hR, hbound⟩ := hj
  refine ⟨R, hHR, hR, ?_⟩
  intro z hz
  by_cases hzero :
      cancelledEntireQuotient riemannXiLi
        (fun s ↦ ∏' i : RiemannXiZeroIndex,
          genusOneCanonicalFactor riemannXiZeroRoot i s) z = 0
  · simp [hzero]
    positivity
  · calc
      Real.log ‖cancelledEntireQuotient riemannXiLi
          (fun s ↦ ∏' i : RiemannXiZeroIndex,
            genusOneCanonicalFactor riemannXiZeroRoot i s) z‖
          ≤ Real.log (Real.exp
              (DyadicCartanLossMajorant Ksq Ksqrt Klin j *
                ((2 : ℝ) ^ j) ^ 2)) :=
            Real.log_le_log (norm_pos_iff.mpr hzero) (hbound z hz)
      _ = DyadicCartanLossMajorant Ksq Ksqrt Klin j *
          ((2 : ℝ) ^ j) ^ 2 := Real.log_exp _
      _ ≤ ε * ((2 : ℝ) ^ j) ^ 2 := by gcongr

/-- The weakest Cartan exceptional-disk estimate needed by the xi route.
At each large scale the bad set is covered by finitely many disks whose
total diameters are smaller than the available radial interval; outside the
disks the desired subquadratic logarithmic estimate already holds. -/
def CartanExceptionalDiskLogBound (g : ℂ → ℂ) : Prop :=
  ∀ ε : ℝ, 0 < ε → ∃ r0 : ℝ, 0 ≤ r0 ∧
    ∀ r : ℝ, r0 ≤ r → ∃ n : ℕ, ∃ center : Fin n → ℂ,
      ∃ radius : Fin n → ℝ,
        (∀ i, 0 ≤ radius i) ∧
        2 * ∑ i, radius i < r ∧
        ∀ z : ℂ, r ≤ ‖z‖ → ‖z‖ ≤ 2 * r →
          (∀ i, z ∉ Metric.closedBall (center i) (radius i)) →
            Real.log ‖g z‖ ≤ ε * r ^ 2

/-- The exceptional-disk estimate implies the exact cofinal-circle boundary
condition consumed by maximum modulus. -/
theorem subquadraticBoundaryLogNormGrowth_of_cartanExceptionalDisks
    {g : ℂ → ℂ} (hcartan : CartanExceptionalDiskLogBound g) :
    SubquadraticBoundaryLogNormGrowth g := by
  intro ε hε
  obtain ⟨r0, hr0, hdata⟩ := hcartan ε hε
  let r1 := max r0 1
  refine ⟨r1, hr0.trans (le_max_left _ _), ?_⟩
  intro r hr
  have hr0' : r0 ≤ r := (le_max_left _ _).trans hr
  have hr1 : 1 ≤ r := (le_max_right _ _).trans hr
  obtain ⟨n, center, radius, hrad, hsum, hbound⟩ :=
    hdata r hr0'
  obtain ⟨R, hR, havoid⟩ :=
    exists_circle_avoiding_exceptionalDisks
      Finset.univ center radius (fun i _ ↦ hrad i)
      (a := r) (b := 2 * r) (by linarith)
  refine ⟨R, hR.1, by linarith [hR.1], ?_⟩
  intro z hz
  have hzNorm : ‖z‖ = R := by
    simpa [Metric.mem_sphere, dist_zero_right] using hz
  apply hbound z
  · rw [hzNorm]
    exact hR.1
  · rw [hzNorm]
    exact hR.2
  · intro i hiBall
    exact Set.disjoint_left.mp
      (havoid i (Finset.mem_univ i)) hz hiBall

/-- Maximum modulus upgrades subquadratic logarithmic bounds on a cofinal
family of circles to the required bounds on every large disk. -/
theorem subquadraticLogNormGrowth_of_boundary
    {g : ℂ → ℂ} (hgA : AnalyticOnNhd ℂ g Set.univ)
    (hg0 : ∀ z, g z ≠ 0)
    (hboundary : SubquadraticBoundaryLogNormGrowth g) :
    SubquadraticLogNormGrowth g := by
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

theorem subquadraticLogNormGrowth_of_dyadicBoundary
    {g : ℂ → ℂ} (hgA : AnalyticOnNhd ℂ g Set.univ)
    (hg0 : ∀ z, g z ≠ 0)
    (hboundary : DyadicSubquadraticBoundaryLogNormGrowth g) :
    SubquadraticLogNormGrowth g :=
  subquadraticLogNormGrowth_of_boundary hgA hg0
    (subquadraticBoundaryLogNormGrowth_of_dyadic hboundary)

theorem subquadraticRealPartGrowth_of_exp_eq
    {g h : ℂ → ℂ} (hexp : ∀ z, Complex.exp (h z) = g z)
    (hgrowth : SubquadraticLogNormGrowth g) :
    SubquadraticRealPartGrowth h := by
  intro ε hε
  obtain ⟨R, hR, hbound⟩ := hgrowth ε hε
  refine ⟨R, hR, fun r hr z hz ↦ ?_⟩
  rw [re_eq_log_norm_of_exp_eq hexp z]
  exact hbound r hr z hz

/-- The final algebraic/analytic component of Hadamard rigidity: an entire
function with vanishing second derivative is affine. -/
theorem eq_affine_of_iteratedDeriv_two_eq_zero
    (h : ℂ → ℂ) (hh : AnalyticOnNhd ℂ h Set.univ)
    (hsecond : ∀ z, iteratedDeriv 2 h z = 0) :
    ∃ a b : ℂ, h = fun z ↦ a + b * z := by
  let a := h 0
  let b := deriv h 0
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

theorem eq_exp_affine_of_analytic_log_second_deriv_zero
    {g h : ℂ → ℂ}
    (hh : AnalyticOnNhd ℂ h Set.univ)
    (hexp : ∀ z, Complex.exp (h z) = g z)
    (hsecond : ∀ z, iteratedDeriv 2 h z = 0) :
    ∃ a b : ℂ, ∀ z, g z = Complex.exp (a + b * z) := by
  obtain ⟨a, b, hab⟩ :=
    eq_affine_of_iteratedDeriv_two_eq_zero h hh hsecond
  exact ⟨a, b, fun z ↦ by rw [← hexp, hab]⟩

/-- Hadamard rigidity from the precise remaining growth estimate:
an entire exponential lift of a quotient whose log norm is uniformly
subquadratic on disks is exponential-affine. -/
theorem eq_exp_affine_of_analytic_log_subquadraticLogNormGrowth
    {g h : ℂ → ℂ}
    (hh : AnalyticOnNhd ℂ h Set.univ)
    (hexp : ∀ z, Complex.exp (h z) = g z)
    (hgrowth : SubquadraticLogNormGrowth g) :
    ∃ a b : ℂ, ∀ z, g z = Complex.exp (a + b * z) := by
  apply eq_exp_affine_of_analytic_log_second_deriv_zero hh hexp
  exact iteratedDeriv_two_eq_zero_of_subquadraticRealPartGrowth h hh
    (subquadraticRealPartGrowth_of_exp_eq hexp hgrowth)

/-- Conditional whole-plane Hadamard identification after the two exact
zeta-specific obligations are supplied: equality of the global divisors and
subquadratic log growth of their cancelled quotient. -/
theorem hadamardRepresentation_of_cancelledQuotientGrowth
    {xi P : ℂ → ℂ}
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
      Complex.exp (a + b * z) * P z := by
  have hxiM : MeromorphicOn xi Set.univ :=
    fun z hz ↦ (hxi z hz).meromorphicAt
  have hPM : MeromorphicOn P Set.univ :=
    fun z hz ↦ (hP z hz).meromorphicAt
  obtain ⟨hqA, hq0⟩ :=
    cancelledEntireQuotient_analytic_ne_zero
      hxiM hPM hxiFinite hPFinite hdiv
  obtain ⟨h, hh, hhexp⟩ :=
    exists_analytic_log_of_analytic_ne_zero hqA hq0
  obtain ⟨a, b, hab⟩ :=
    eq_exp_affine_of_analytic_log_subquadraticLogNormGrowth
      hh hhexp hgrowth
  refine ⟨a, b, ?_⟩
  have hqrawA : AnalyticAt ℂ (xi * P⁻¹) 0 :=
    (hxi 0 (Set.mem_univ 0)).mul
      ((hP 0 (Set.mem_univ 0)).inv hP0)
  have hqrawM : MeromorphicOn (xi * P⁻¹) Set.univ :=
    hxiM.mul hPM.inv
  have hcancel_eq :
      cancelledEntireQuotient xi P =ᶠ[𝓝 0] xi * P⁻¹ := by
    exact (toMeromorphicNFOn_eq_toMeromorphicNFAt_on_nhds
      hqrawM (Set.mem_univ 0)).trans
        (Filter.Eventually.of_forall (fun z ↦ by
          rw [toMeromorphicNFAt_eq_self.2 hqrawA.meromorphicNFAt]))
  have hlocal : xi =ᶠ[𝓝 0]
      fun z ↦ Complex.exp (a + b * z) * P z := by
    filter_upwards [hcancel_eq,
      (hP 0 (Set.mem_univ 0)).continuousAt.eventually_ne hP0]
      with z hz hPz
    have hqaff : cancelledEntireQuotient xi P z =
        Complex.exp (a + b * z) := by rw [hab]
    rw [hz, Pi.mul_apply, Pi.inv_apply] at hqaff
    rw [← div_eq_mul_inv] at hqaff
    exact (div_eq_iff hPz).mp hqaff
  have hrhs : AnalyticOnNhd ℂ
      (fun z ↦ Complex.exp (a + b * z) * P z) Set.univ := by
    intro z hz
    exact ((analyticAt_const.add
      (analyticAt_const.mul analyticAt_id)).cexp).mul (hP z hz)
  exact fun z ↦ congrFun (hxi.eq_of_eventuallyEq hrhs hlocal) z

/-- Normalization and the functional equation determine the optimal
constraints on the affine factor.  The raw constant remains defined only
modulo `2πiℤ`, so the valid conclusion is `exp a = 1`. -/
theorem normalized_hadamardRepresentation_of_cancelledQuotientGrowth
    {xi P : ℂ → ℂ}
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
        Complex.exp (a + b * z) * P z := by
  obtain ⟨a, b, hrep⟩ :=
    hadamardRepresentation_of_cancelledQuotientGrowth
      hxi hP hxiFinite hPFinite hdiv (hP0.symm ▸ one_ne_zero)
        hgrowth
  refine ⟨a, b, hrep, ?_, ?_⟩
  · have h := hrep 0
    rw [hxi0, hP0] at h
    simpa using h.symm
  · intro z
    rw [← hrep, ← hrep]
    exact hfe z

set_option maxHeartbeats 800000 in
/-- Complete abstract Hadamard pipeline for a multiplicity-indexed root
enumeration.  The sole remaining global analytic input is now the explicit
boundary estimate on the already-cancelled quotient. -/
theorem genusOneHadamardRepresentation_of_boundaryGrowth
    {ι : Type*} [DecidableEq ι] {root : ι → ℂ} {xi : ℂ → ℂ}
    (hroot0 : ∀ i, root i ≠ 0)
    (hinv : Summable (fun i ↦ ‖(root i)⁻¹‖ ^ 2))
    (hsmall : ∀ K : Set ℂ, IsCompact K →
      ∀ᶠ i in cofinite, ∀ s ∈ K, ‖s / root i‖ ≤ 1)
    (fiber : ℂ → Finset ι)
    (hfiber : ∀ z i, i ∈ fiber z ↔ root i = z)
    (hxi : AnalyticOnNhd ℂ xi Set.univ)
    (hxiOrder : ∀ z, analyticOrderAt xi z = (fiber z).card)
    (hboundary : SubquadraticBoundaryLogNormGrowth
      (cancelledEntireQuotient xi
        (fun s ↦ ∏' i, genusOneCanonicalFactor root i s))) :
    GenusOneHadamardRepresentation root xi := by
  let P : ℂ → ℂ :=
    fun s ↦ ∏' i, genusOneCanonicalFactor root i s
  have hprod :=
    hasProdLocallyUniformlyOn_genusOneCanonicalFactor
      root hinv hsmall
  have hPA : AnalyticOnNhd ℂ P Set.univ :=
    analyticOnNhd_genusOneCanonicalProduct hprod
  have hdiv : MeromorphicOn.divisor P Set.univ =
      MeromorphicOn.divisor xi Set.univ :=
    genusOneCanonicalProduct_divisor_eq
      hroot0 hinv hsmall fiber hfiber hxi hxiOrder
  have hxiFinite : ∀ z, meromorphicOrderAt xi z ≠ ⊤ := by
    intro z
    rw [(hxi z (Set.mem_univ z)).meromorphicOrderAt_eq,
      hxiOrder z]
    simp
  have hPFinite : ∀ z, meromorphicOrderAt P z ≠ ⊤ := by
    intro z
    rw [(hPA z (Set.mem_univ z)).meromorphicOrderAt_eq,
      analyticOrderAt_genusOneCanonicalProduct
        hroot0 hinv hsmall fiber hfiber z]
    simp
  have hP0 : P 0 ≠ 0 := by
    dsimp [P]
    simp [genusOneCanonicalFactor, genusOnePrimaryFactor]
  have hxiM : MeromorphicOn xi Set.univ :=
    fun z hz ↦ (hxi z hz).meromorphicAt
  have hPM : MeromorphicOn P Set.univ :=
    fun z hz ↦ (hPA z hz).meromorphicAt
  obtain ⟨hqA, hq0⟩ :=
    cancelledEntireQuotient_analytic_ne_zero
      hxiM hPM hxiFinite hPFinite hdiv.symm
  have hgrowth : SubquadraticLogNormGrowth
      (cancelledEntireQuotient xi P) :=
    subquadraticLogNormGrowth_of_boundary hqA hq0 hboundary
  simpa [GenusOneHadamardRepresentation, P] using
    hadamardRepresentation_of_cancelledQuotientGrowth
      hxi hPA hxiFinite hPFinite hdiv.symm hP0 hgrowth

/-- Xi specialization: local finiteness already supplies root escape, so
inverse-square summability plus the explicit Cartan boundary estimate close
the whole-plane genus-one Hadamard representation. -/
theorem riemannXiGenusOneHadamardRepresentation_of_boundaryGrowth
    (hinv : Summable
      (fun i : RiemannXiZeroIndex ↦
        ‖(riemannXiZeroRoot i)⁻¹‖ ^ 2))
    (hboundary : SubquadraticBoundaryLogNormGrowth
      (cancelledEntireQuotient riemannXiLi
        (fun s ↦ ∏' i : RiemannXiZeroIndex,
          genusOneCanonicalFactor riemannXiZeroRoot i s))) :
    GenusOneHadamardRepresentation riemannXiZeroRoot riemannXiLi := by
  classical
  have hsmall : ∀ K : Set ℂ, IsCompact K →
      ∀ᶠ i : RiemannXiZeroIndex in cofinite,
        ∀ s ∈ K, ‖s / riemannXiZeroRoot i‖ ≤ 1 := by
    intro K hK
    obtain ⟨R, hR⟩ := hK.isBounded.exists_norm_le
    filter_upwards [riemannXiZeroRoot_escape.eventually
      (eventually_ge_atTop R)] with i hi s hs
    rw [norm_div]
    exact (div_le_one
      (norm_pos_iff.mpr (riemannXiZeroRoot_ne_zero i))).2
        ((hR s hs).trans hi)
  exact genusOneHadamardRepresentation_of_boundaryGrowth
    riemannXiZeroRoot_ne_zero hinv hsmall riemannXiZeroFiber
      mem_riemannXiZeroFiber
      (fun z _ ↦ differentiable_riemannXiLi.analyticAt z)
      analyticOrderAt_riemannXiLi_eq_zeroFiber_card hboundary

/-- After the Jensen argument, the Cartan/minimum-modulus boundary estimate
is the sole remaining hypothesis in the xi Hadamard pipeline. -/
theorem riemannXiGenusOneHadamardRepresentation_of_cartanBoundary
    (hboundary : SubquadraticBoundaryLogNormGrowth
      (cancelledEntireQuotient riemannXiLi
        (fun s ↦ ∏' i : RiemannXiZeroIndex,
          genusOneCanonicalFactor riemannXiZeroRoot i s))) :
    GenusOneHadamardRepresentation riemannXiZeroRoot riemannXiLi :=
  riemannXiGenusOneHadamardRepresentation_of_boundaryGrowth
    summable_norm_inv_sq_riemannXiZeroRoot hboundary

theorem riemannXiGenusOneHadamardRepresentation_of_dyadicBoundary
    (hboundary : DyadicSubquadraticBoundaryLogNormGrowth
      (cancelledEntireQuotient riemannXiLi
        (fun s ↦ ∏' i : RiemannXiZeroIndex,
          genusOneCanonicalFactor riemannXiZeroRoot i s))) :
    GenusOneHadamardRepresentation riemannXiZeroRoot riemannXiLi :=
  riemannXiGenusOneHadamardRepresentation_of_cartanBoundary
    (subquadraticBoundaryLogNormGrowth_of_dyadic hboundary)

/-- Fixed-constant Cartan dominance closes the xi Hadamard representation
when the multiplicity enumeration is inhabited. -/
theorem riemannXiGenusOneHadamardRepresentation_of_nonempty
    [Nonempty RiemannXiZeroIndex] :
    GenusOneHadamardRepresentation riemannXiZeroRoot riemannXiLi :=
  riemannXiGenusOneHadamardRepresentation_of_dyadicBoundary
    riemannXiCancelledQuotient_dyadicBoundary

/-- Xi normalization determines the exponential constant and transports the
functional equation to the affine/product representation.  It does not by
itself identify a unique logarithm `a` or eliminate the slope `b`. -/
theorem riemannXiGenusOneHadamardRepresentation_normalized_of_nonempty
    [Nonempty RiemannXiZeroIndex] :
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
              genusOneCanonicalFactor riemannXiZeroRoot i z := by
  obtain ⟨a, b, hrep⟩ := riemannXiGenusOneHadamardRepresentation_of_nonempty
  refine ⟨a, b, hrep, ?_, ?_⟩
  · have h := hrep 0
    simp [genusOneCanonicalFactor, genusOnePrimaryFactor] at h
    exact h.symm
  · intro z
    rw [← hrep, ← hrep]
    exact riemannXiLi_one_sub z

/-- An explicit nonconstant value used to eliminate the zero-free branch. -/
theorem riemannXiLi_two :
    riemannXiLi 2 = (Real.pi : ℂ) / 3 := by
  rw [riemannXiLi_eq_mul_completedRiemannZeta (by norm_num) (by norm_num)]
  rw [show completedRiemannZeta (2 : ℂ) =
      (Real.pi : ℂ)⁻¹ * riemannZeta 2 by
    have h := riemannZeta_def_of_ne_zero (s := (2 : ℂ)) (by norm_num)
    norm_num [Complex.Gammaℝ_def, Complex.cpow_neg, Complex.cpow_one,
      Complex.Gamma_one] at h ⊢
    field_simp [Real.pi_ne_zero] at h ⊢
    exact h.symm]
  rw [riemannZeta_two]
  field_simp [Real.pi_ne_zero]
  ring

theorem riemannXiLi_two_ne_one : riemannXiLi 2 ≠ 1 := by
  rw [riemannXiLi_two]
  intro h
  have hre := congrArg Complex.re h
  norm_num at hre
  linarith [Real.pi_gt_three]

theorem riemannXiLi_ne_zero_of_isEmpty [IsEmpty RiemannXiZeroIndex]
    (z : ℂ) : riemannXiLi z ≠ 0 := by
  intro hz
  obtain ⟨i, hi⟩ := (exists_riemannXiZeroRoot_iff z).mpr hz
  exact isEmptyElim i

theorem cancelledRiemannXiQuotient_eq_of_isEmpty
    [IsEmpty RiemannXiZeroIndex] (z : ℂ) :
    cancelledEntireQuotient riemannXiLi
      (fun s ↦ ∏' i : RiemannXiZeroIndex,
        genusOneCanonicalFactor riemannXiZeroRoot i s) z = riemannXiLi z := by
  let P : ℂ → ℂ := fun s ↦ ∏' i : RiemannXiZeroIndex,
    genusOneCanonicalFactor riemannXiZeroRoot i s
  have hP : P = fun _ ↦ 1 := by
    funext s
    simp [P]
  have hz := cancelledEntireQuotient_eq_div_of_ne
    (fun w hw ↦ (differentiable_riemannXiLi.analyticAt w).meromorphicAt)
    (show MeromorphicOn P Set.univ by
      rw [hP]
      exact fun w hw ↦ analyticAt_const.meromorphicAt)
    (differentiable_riemannXiLi.analyticAt z)
    (show AnalyticAt ℂ P z by rw [hP]; fun_prop)
    (show P z ≠ 0 by rw [hP]; simp)
  change cancelledEntireQuotient riemannXiLi P z = riemannXiLi z
  simpa [hP] using hz

/-- If the xi divisor were empty, the global xi order bound alone supplies
the dyadic boundary estimate: the canonical product is the empty product. -/
theorem riemannXiCancelledQuotient_dyadicBoundary_of_isEmpty
    [IsEmpty RiemannXiZeroIndex] :
    DyadicSubquadraticBoundaryLogNormGrowth
      (cancelledEntireQuotient riemannXiLi
        (fun s ↦ ∏' i : RiemannXiZeroIndex,
          genusOneCanonicalFactor riemannXiZeroRoot i s)) := by
  obtain ⟨C, R0, hC, hR0, hxi⟩ := riemannXi_orderOneGrowthBound
  intro ε hε
  have hp : Tendsto (fun j : ℕ ↦ (2 : ℝ) ^ j) atTop atTop :=
    tendsto_pow_atTop_atTop_of_one_lt (by norm_num)
  have hlarge : ∀ᶠ j : ℕ in atTop, max R0 1 ≤ (2 : ℝ) ^ j :=
    hp.eventually (eventually_ge_atTop (max R0 1))
  filter_upwards [hlarge,
    eventually_dyadicCartanLossMajorant_lt 0 0 (4 * C) ε hε]
    with j hj hmajor
  let H : ℝ := (2 : ℝ) ^ j
  refine ⟨H, le_rfl, by positivity, ?_⟩
  intro z hz
  have hzNorm : ‖z‖ = H := by
    simpa [Metric.mem_sphere, dist_zero_right] using hz
  have hlog2 : Real.log (2 : ℝ) ≤ 1 := by
    have hh := Real.log_le_sub_one_of_pos (x := 2) (by norm_num)
    norm_num at hh ⊢
    exact hh
  have hlog : Real.log (H + 2) ≤ 2 * (j + 1 : ℝ) := by
    have hH1 : 1 ≤ H := (le_max_right R0 1).trans hj
    have harg : H + 2 ≤ 4 * H := by linarith
    calc
      Real.log (H + 2) ≤ Real.log (4 * H) :=
        Real.log_le_log (by positivity) harg
      _ = (j + 2 : ℕ) * Real.log 2 := by
        dsimp [H]
        rw [show 4 * (2 : ℝ) ^ j = (2 : ℝ) ^ (j + 2) by
          rw [pow_add]; norm_num; ring, Real.log_pow]
      _ ≤ (j + 2 : ℝ) := by
        have hjnon : 0 ≤ (j + 2 : ℝ) := by positivity
        have hh := mul_le_mul_of_nonneg_left hlog2 hjnon
        norm_num [Nat.cast_add] at hh ⊢
        exact hh
      _ ≤ 2 * (j + 1 : ℝ) := by
        nlinarith [Nat.cast_nonneg (α := ℝ) j]
  rw [cancelledRiemannXiQuotient_eq_of_isEmpty]
  calc
    Real.log ‖riemannXiLi z‖ ≤
        Real.log (Real.exp (C * H * Real.log (H + 2))) := by
      apply Real.log_le_log
        (norm_pos_iff.mpr (riemannXiLi_ne_zero_of_isEmpty z))
      rw [← hzNorm]
      apply hxi z
      rw [hzNorm]
      exact (le_max_left R0 1).trans hj
    _ = C * H * Real.log (H + 2) := Real.log_exp _
    _ ≤ 4 * C * ((j + 1 : ℝ) / H) * H ^ 2 := by
      calc
        C * H * Real.log (H + 2) ≤
            C * H * (2 * (j + 1 : ℝ)) := by gcongr
        _ ≤ 4 * C * ((j + 1 : ℝ) / H) * H ^ 2 := by
          have hH : 0 < H := by positivity
          have heq : 4 * C * ((j + 1 : ℝ) / H) * H ^ 2 =
              4 * C * (j + 1 : ℝ) * H := by field_simp
          rw [heq]
          have hu : 0 ≤ (j + 1 : ℝ) := by positivity
          nlinarith [mul_nonneg (mul_nonneg hC hH.le) hu]
    _ = DyadicCartanLossMajorant 0 0 (4 * C) j * H ^ 2 := by
      simp [DyadicCartanLossMajorant, H]
    _ ≤ ε * H ^ 2 := by gcongr

/-- Xi has a nontrivial zero, proved without RH.  If its divisor were empty,
the empty-product Hadamard theorem and functional equation would make xi
constant; the explicit value `xi(2)=π/3≠1` contradicts normalization. -/
theorem riemannXiZeroIndex_nonempty : Nonempty RiemannXiZeroIndex := by
  classical
  cases isEmpty_or_nonempty RiemannXiZeroIndex with
  | inr h => exact h
  | inl h =>
    letI : IsEmpty RiemannXiZeroIndex := h
    obtain ⟨a, b, hab⟩ :=
      riemannXiGenusOneHadamardRepresentation_of_dyadicBoundary
        riemannXiCancelledQuotient_dyadicBoundary_of_isEmpty
    have hab' : ∀ z, riemannXiLi z = Complex.exp (a + b * z) := by
      intro z
      simpa using hab z
    have hfun : (fun z : ℂ ↦ Complex.exp (a + b * (1 - z))) =
        fun z ↦ Complex.exp (a + b * z) := by
      funext z
      rw [← hab', ← hab']
      exact riemannXiLi_one_sub z
    let c : ℂ := 1 / 2
    have hl : HasDerivAt
        (fun z : ℂ ↦ Complex.exp (a + b * (1 - z)))
        ((-b) * Complex.exp (a + b * (1 - c))) c := by
      have hi := (((hasDerivAt_const c 1).sub
        (hasDerivAt_id c)).const_mul b).const_add a
      simpa [mul_comm] using hi.cexp
    have hr : HasDerivAt
        (fun z : ℂ ↦ Complex.exp (a + b * z))
        (b * Complex.exp (a + b * c)) c := by
      have hi := ((hasDerivAt_id c).const_mul b).const_add a
      simpa [mul_comm] using hi.cexp
    have hd := congrArg (fun f : ℂ → ℂ ↦ deriv f c) hfun
    rw [hl.deriv, hr.deriv] at hd
    have heq : Complex.exp (a + b * (1 - c)) =
        Complex.exp (a + b * c) := by
      dsimp [c]
      congr 1
      ring
    rw [heq] at hd
    have hb : b = 0 := by
      have hsame : -(b * Complex.exp (a + b * c)) =
          b * Complex.exp (a + b * c) := by simpa using hd
      have hmul : 2 * (b * Complex.exp (a + b * c)) = 0 := by
        linear_combination -hsame
      have hbe : b * Complex.exp (a + b * c) = 0 :=
        (mul_eq_zero.mp hmul).resolve_left (by norm_num)
      exact (mul_eq_zero.mp hbe).resolve_right (Complex.exp_ne_zero _)
    have ha : Complex.exp a = 1 := by
      have h0 := hab' 0
      simpa using h0.symm
    apply False.elim
    apply riemannXiLi_two_ne_one
    rw [hab', hb]
    simpa using ha

/-- The genus-one Hadamard representation is unconditional across the
empty, finite, and infinite zero-divisor cases. -/
theorem riemannXiGenusOneHadamardRepresentation :
    GenusOneHadamardRepresentation riemannXiZeroRoot riemannXiLi := by
  letI : Nonempty RiemannXiZeroIndex := riemannXiZeroIndex_nonempty
  exact riemannXiGenusOneHadamardRepresentation_of_nonempty

/-- The unconditional Hadamard representation is realized by the concrete
closed radial multiplicity windows, locally uniformly on the whole plane. -/
theorem exists_riemannXiRadialHadamardApproximants_tendstoLocallyUniformly :
    ∃ a b : ℂ, TendstoLocallyUniformlyOn
      (fun (N : ℕ) s ↦ Complex.exp (a + b * s) *
        ∏ i ∈ riemannXiZeroIndexWindow (N : ℝ),
          genusOneCanonicalFactor riemannXiZeroRoot i s)
      riemannXiLi atTop Set.univ := by
  obtain ⟨a, b, hrep⟩ := riemannXiGenusOneHadamardRepresentation
  refine ⟨a, b, ?_⟩
  have hpref : TendstoLocallyUniformlyOn
      (fun _ : ℕ ↦ fun s ↦ Complex.exp (a + b * s))
      (fun s ↦ Complex.exp (a + b * s)) atTop Set.univ := by
    intro u hu x hx
    exact ⟨Set.univ, Filter.univ_mem, Filter.Eventually.of_forall
      (fun _ y hy ↦ refl_mem_uniformity hu)⟩
  have hcanon := riemannXiRadialGenusOneProducts_tendstoLocallyUniformly
  have hcanon_cont : ContinuousOn
      (fun s ↦ ∏' i : RiemannXiZeroIndex,
        genusOneCanonicalFactor riemannXiZeroRoot i s) Set.univ :=
    analyticOnNhd_riemannXiGenusOneCanonicalProduct.continuousOn
  have hmul := hpref.mul₀ hcanon (by fun_prop) hcanon_cont
  have heq : (fun s ↦ Complex.exp (a + b * s)) *
      (fun s ↦ ∏' i : RiemannXiZeroIndex,
        genusOneCanonicalFactor riemannXiZeroRoot i s) = riemannXiLi := by
    funext s
    simpa only [Pi.mul_apply] using (hrep s).symm
  rw [heq] at hmul
  exact hmul

theorem exists_riemannXiHeightHadamardApproximants_tendstoLocallyUniformly :
    ∃ a b : ℂ, TendstoLocallyUniformlyOn
      (fun (N : ℕ) s ↦ Complex.exp (a + b * s) *
        ∏ i ∈ riemannXiZeroHeightWindow (N : ℝ),
          genusOneCanonicalFactor riemannXiZeroRoot i s)
      riemannXiLi atTop Set.univ := by
  obtain ⟨a, b, hrep⟩ := riemannXiGenusOneHadamardRepresentation
  refine ⟨a, b, ?_⟩
  have hpref : TendstoLocallyUniformlyOn
      (fun _ : ℕ ↦ fun s ↦ Complex.exp (a + b * s))
      (fun s ↦ Complex.exp (a + b * s)) atTop Set.univ := by
    intro u hu x hx
    exact ⟨Set.univ, Filter.univ_mem, Filter.Eventually.of_forall
      (fun _ y hy ↦ refl_mem_uniformity hu)⟩
  have hcanon := riemannXiHeightGenusOneProducts_tendstoLocallyUniformly
  have hcanon_cont : ContinuousOn
      (fun s ↦ ∏' i : RiemannXiZeroIndex,
        genusOneCanonicalFactor riemannXiZeroRoot i s) Set.univ :=
    analyticOnNhd_riemannXiGenusOneCanonicalProduct.continuousOn
  have hmul := hpref.mul₀ hcanon (by fun_prop) hcanon_cont
  have heq : (fun s ↦ Complex.exp (a + b * s)) *
      (fun s ↦ ∏' i : RiemannXiZeroIndex,
        genusOneCanonicalFactor riemannXiZeroRoot i s) = riemannXiLi := by
    funext s
    simpa only [Pi.mul_apply] using (hrep s).symm
  rw [heq] at hmul
  exact hmul

theorem riemannXiGenusOneHadamardRepresentation_normalized :
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
              genusOneCanonicalFactor riemannXiZeroRoot i z := by
  letI : Nonempty RiemannXiZeroIndex := riemannXiZeroIndex_nonempty
  exact riemannXiGenusOneHadamardRepresentation_normalized_of_nonempty

noncomputable def riemannXiHeightGenusOneProduct
    (N : ℕ) (s : ℂ) : ℂ :=
  ∏ i ∈ riemannXiZeroHeightWindow (N : ℝ),
    genusOneCanonicalFactor riemannXiZeroRoot i s

noncomputable def riemannXiGenusOneCanonicalProduct (s : ℂ) : ℂ :=
  ∏' i : RiemannXiZeroIndex,
    genusOneCanonicalFactor riemannXiZeroRoot i s

theorem deriv_riemannXiHeightGenusOneProduct_zero (N : ℕ) :
    deriv (riemannXiHeightGenusOneProduct N) 0 = 0 := by
  classical
  unfold riemannXiHeightGenusOneProduct
  have heq : (fun s ↦ ∏ i ∈ riemannXiZeroHeightWindow (N : ℝ),
      genusOneCanonicalFactor riemannXiZeroRoot i s) =
      ∏ i ∈ riemannXiZeroHeightWindow (N : ℝ),
        genusOneCanonicalFactor riemannXiZeroRoot i := by
    funext s
    simp
  rw [heq, deriv_finsetProd (by
    intro i hi
    unfold genusOneCanonicalFactor genusOnePrimaryFactor
    fun_prop)]
  apply Finset.sum_eq_zero
  intro i hi
  have hfactor : HasDerivAt
      (genusOneCanonicalFactor riemannXiZeroRoot i) 0 0 := by
    unfold genusOneCanonicalFactor genusOnePrimaryFactor
    convert (((hasDerivAt_const (x := (0 : ℂ)) (c := (1 : ℂ))).sub
      ((hasDerivAt_id (𝕜 := ℂ) (0 : ℂ)).div_const
        (riemannXiZeroRoot i))).mul
      (((hasDerivAt_id (𝕜 := ℂ) (0 : ℂ)).div_const
        (riemannXiZeroRoot i)).cexp)) using 1 <;>
      first | rfl | (simp [id_eq])
  rw [hfactor.deriv]
  simp

theorem deriv_riemannXiGenusOneCanonicalProduct_zero :
    deriv riemannXiGenusOneCanonicalProduct 0 = 0 := by
  have hd : Tendsto
      (fun N ↦ deriv (riemannXiHeightGenusOneProduct N) 0)
      atTop (𝓝 (deriv riemannXiGenusOneCanonicalProduct 0)) := by
    exact (riemannXiHeightGenusOneProducts_tendstoLocallyUniformly.deriv
      (Filter.Eventually.of_forall (fun N ↦ by
        unfold genusOneCanonicalFactor genusOnePrimaryFactor
        fun_prop))
      isOpen_univ).tendsto_at (Set.mem_univ 0)
  rw [funext deriv_riemannXiHeightGenusOneProduct_zero] at hd
  exact tendsto_nhds_unique hd tendsto_const_nhds

theorem logDeriv_riemannXiGenusOneCanonicalProduct_one :
    logDeriv riemannXiGenusOneCanonicalProduct 1 =
      ∑' i : RiemannXiZeroIndex,
        (riemannXiZeroRoot i *
          (1 - riemannXiZeroRoot i))⁻¹ := by
  have hder : Tendsto
      (fun N ↦ deriv (riemannXiHeightGenusOneProduct N) 1)
      atTop (𝓝 (deriv riemannXiGenusOneCanonicalProduct 1)) := by
    exact (riemannXiHeightGenusOneProducts_tendstoLocallyUniformly.deriv
      (Filter.Eventually.of_forall (fun N ↦ by
        unfold genusOneCanonicalFactor genusOnePrimaryFactor
        fun_prop))
      isOpen_univ).tendsto_at (Set.mem_univ 1)
  have hval : Tendsto
      (fun N ↦ riemannXiHeightGenusOneProduct N 1)
      atTop (𝓝 (riemannXiGenusOneCanonicalProduct 1)) :=
    riemannXiHeightGenusOneProducts_tendstoLocallyUniformly.tendsto_at
      (Set.mem_univ 1)
  obtain ⟨a, b, hrep⟩ := riemannXiGenusOneHadamardRepresentation
  have hP1 : riemannXiGenusOneCanonicalProduct 1 ≠ 0 := by
    intro hp
    have h := hrep 1
    rw [show (∏' i : RiemannXiZeroIndex,
      genusOneCanonicalFactor riemannXiZeroRoot i 1) =
        riemannXiGenusOneCanonicalProduct 1 from rfl, hp] at h
    simp at h
  have hlog : Tendsto
      (fun N ↦ logDeriv (riemannXiHeightGenusOneProduct N) 1)
      atTop (𝓝 (logDeriv riemannXiGenusOneCanonicalProduct 1)) := by
    unfold logDeriv
    apply (hder.div hval hP1).congr'
    exact Filter.Eventually.of_forall (fun N ↦ rfl)
  have heq :
      (fun N : ℕ ↦ logDeriv (riemannXiHeightGenusOneProduct N) 1) =
        fun N : ℕ ↦ ∑ i ∈ riemannXiZeroHeightWindow (N : ℝ),
          (riemannXiZeroRoot i *
            (1 - riemannXiZeroRoot i))⁻¹ := by
    funext N
    exact logDeriv_riemannXiHeightGenusOneProduct_one (N : ℝ)
  rw [heq] at hlog
  exact tendsto_nhds_unique hlog
    riemannXiPairedInverseRootHeightSums_tendsto

/-- The reflection-paired inverse-root `tsum` is exactly minus twice the
affine Hadamard slope. -/
theorem exists_riemannXiHadamardSlope_eq_pairedInverseRootSum :
    ∃ a b : ℂ,
      (∀ z, riemannXiLi z = Complex.exp (a + b * z) *
        riemannXiGenusOneCanonicalProduct z) ∧
      Complex.exp a = 1 ∧
      (∑' i : RiemannXiZeroIndex,
        (riemannXiZeroRoot i *
          (1 - riemannXiZeroRoot i))⁻¹) = -2 * b := by
  obtain ⟨a, b, hrep, ha, hsymm⟩ :=
    riemannXiGenusOneHadamardRepresentation_normalized
  refine ⟨a, b, ?_, ha, ?_⟩
  · exact hrep
  have hP0 : riemannXiGenusOneCanonicalProduct 0 = 1 := by
    unfold riemannXiGenusOneCanonicalProduct
    simp [genusOneCanonicalFactor, genusOnePrimaryFactor]
  have hP0ne : riemannXiGenusOneCanonicalProduct 0 ≠ 0 := by
    rw [hP0]
    norm_num
  have hP1ne : riemannXiGenusOneCanonicalProduct 1 ≠ 0 := by
    intro hp
    have h := hrep 1
    rw [show (∏' i : RiemannXiZeroIndex,
      genusOneCanonicalFactor riemannXiZeroRoot i 1) =
        riemannXiGenusOneCanonicalProduct 1 from rfl, hp] at h
    simp at h
  have hxi0 : logDeriv riemannXiLi 0 = b := by
    have hfun : riemannXiLi =
        fun z ↦ Complex.exp (a + b * z) *
          riemannXiGenusOneCanonicalProduct z := by
      funext z
      exact hrep z
    rw [hfun, logDeriv_mul]
    · rw [show logDeriv riemannXiGenusOneCanonicalProduct 0 = 0 by
        rw [logDeriv_apply,
          deriv_riemannXiGenusOneCanonicalProduct_zero]
        simp]
      simp only [logDeriv_apply]
      have he : HasDerivAt
          (fun z : ℂ ↦ Complex.exp (a + b * z))
          (Complex.exp (a + b * 0) * b) 0 := by
        simpa [id_eq, add_comm] using
          (((hasDerivAt_id (𝕜 := ℂ) (0 : ℂ)).const_mul b).const_add a).cexp
      rw [he.deriv]
      simp
    · exact Complex.exp_ne_zero _
    · exact hP0ne
    · fun_prop
    · exact (analyticOnNhd_riemannXiGenusOneCanonicalProduct
        0 (Set.mem_univ 0)).differentiableAt
  have hxi1 : logDeriv riemannXiLi 1 =
      b + ∑' i : RiemannXiZeroIndex,
        (riemannXiZeroRoot i *
          (1 - riemannXiZeroRoot i))⁻¹ := by
    have hfun : riemannXiLi =
        fun z ↦ Complex.exp (a + b * z) *
          riemannXiGenusOneCanonicalProduct z := by
      funext z
      exact hrep z
    rw [hfun, logDeriv_mul]
    · rw [logDeriv_riemannXiGenusOneCanonicalProduct_one]
      simp only [logDeriv_apply]
      have he : HasDerivAt
          (fun z : ℂ ↦ Complex.exp (a + b * z))
          (Complex.exp (a + b * 1) * b) 1 := by
        simpa [id_eq, add_comm] using
          (((hasDerivAt_id (𝕜 := ℂ) (1 : ℂ)).const_mul b).const_add a).cexp
      rw [he.deriv]
      simp
    · exact Complex.exp_ne_zero _
    · exact hP1ne
    · fun_prop
    · exact (analyticOnNhd_riemannXiGenusOneCanonicalProduct
        1 (Set.mem_univ 1)).differentiableAt
  have hderivsymm : deriv riemannXiLi 0 = -deriv riemannXiLi 1 := by
    have hbase : HasDerivAt (fun z : ℂ ↦ 1 - z) (-1) 0 := by
      simpa [sub_eq_add_neg] using
        (hasDerivAt_id (𝕜 := ℂ) (0 : ℂ)).neg.const_add 1
    have hcomp : HasDerivAt
        (fun z : ℂ ↦ riemannXiLi (1 - z))
        (-deriv riemannXiLi 1) 0 := by
      have houter : HasDerivAt riemannXiLi
          (deriv riemannXiLi 1) 1 :=
        (differentiable_riemannXiLi 1).hasDerivAt
      have houter' : HasDerivAt riemannXiLi
          (deriv riemannXiLi 1) (1 - (0 : ℂ)) := by
        simpa using houter
      convert houter'.comp (𝕜 := ℂ) 0 hbase using 1 <;>
        first | rfl | simp
    have heq : (fun z : ℂ ↦ riemannXiLi (1 - z)) = riemannXiLi := by
      funext z
      exact riemannXiLi_one_sub z
    rw [heq] at hcomp
    exact hcomp.deriv
  have hlogsymm :
      logDeriv riemannXiLi 1 = -logDeriv riemannXiLi 0 := by
    rw [logDeriv_apply, logDeriv_apply]
    simp only [riemannXiLi_one, riemannXiLi_zero, div_one]
    rw [hderivsymm]
    ring
  rw [hxi0, hxi1] at hlogsymm
  linear_combination hlogsymm

theorem exists_riemannXiInverseRootHeightSums_tendsto_neg_slope :
    ∃ a b : ℂ,
      (∀ z, riemannXiLi z = Complex.exp (a + b * z) *
        riemannXiGenusOneCanonicalProduct z) ∧
      Complex.exp a = 1 ∧
      Tendsto
        (fun N : ℕ ↦ ∑ i ∈ riemannXiZeroHeightWindow (N : ℝ),
          (riemannXiZeroRoot i)⁻¹)
        atTop (𝓝 (-b)) := by
  obtain ⟨a, b, hrep, ha, hslope⟩ :=
    exists_riemannXiHadamardSlope_eq_pairedInverseRootSum
  refine ⟨a, b, hrep, ha, ?_⟩
  convert riemannXiInverseRootHeightSums_tendsto using 1
  rw [hslope]
  ring

/-- A Cartan exceptional-disk estimate for the cancelled xi/product
quotient now closes every remaining geometric, maximum-modulus, logarithm,
rigidity, and analytic-continuation step of the Hadamard argument. -/
theorem riemannXiGenusOneHadamardRepresentation_of_cartanExceptionalDisks
    (hcartan : CartanExceptionalDiskLogBound
      (cancelledEntireQuotient riemannXiLi
        (fun s ↦ ∏' i : RiemannXiZeroIndex,
          genusOneCanonicalFactor riemannXiZeroRoot i s))) :
    GenusOneHadamardRepresentation riemannXiZeroRoot riemannXiLi :=
  riemannXiGenusOneHadamardRepresentation_of_cartanBoundary
    (subquadraticBoundaryLogNormGrowth_of_cartanExceptionalDisks
      hcartan)

/-- The affine slope in a Hadamard representation is unique.  The constant
is unique at the level of its exponential, which is optimal without choosing
a logarithm branch. -/
theorem hadamardAffineFactor_unique
    {P xi : ℂ → ℂ} {a b a' b' : ℂ}
    (hPcont : ContinuousAt P 0) (hP0 : P 0 ≠ 0)
    (hrep : ∀ s, xi s = Complex.exp (a + b * s) * P s)
    (hrep' : ∀ s, xi s = Complex.exp (a' + b' * s) * P s) :
    Complex.exp a = Complex.exp a' ∧ b = b' := by
  have heq : (fun s ↦ Complex.exp (a + b * s)) =ᶠ[𝓝 0]
      fun s ↦ Complex.exp (a' + b' * s) := by
    filter_upwards [hPcont.eventually_ne hP0] with s hs
    apply mul_right_cancel₀ hs
    exact (hrep s).symm.trans (hrep' s)
  have hval : Complex.exp a = Complex.exp a' := by
    simpa using heq.self_of_nhds
  have hd : b * Complex.exp a = b' * Complex.exp a' := by
    have hd1 : HasDerivAt (fun s ↦ Complex.exp (a + b * s))
        (b * Complex.exp a) 0 := by
      have hin : HasDerivAt (fun s : ℂ ↦ a + b * s) b 0 := by
        simpa using ((hasDerivAt_id (x := (0 : ℂ))).const_mul b).const_add a
      simpa [Function.comp_def, mul_comm] using
        (Complex.hasDerivAt_exp (a + b * 0)).comp 0 hin
    have hd2 : HasDerivAt (fun s ↦ Complex.exp (a' + b' * s))
        (b' * Complex.exp a') 0 := by
      have hin : HasDerivAt (fun s : ℂ ↦ a' + b' * s) b' 0 := by
        simpa using ((hasDerivAt_id (x := (0 : ℂ))).const_mul b').const_add a'
      simpa [Function.comp_def, mul_comm] using
        (Complex.hasDerivAt_exp (a' + b' * 0)).comp 0 hin
    rw [← hd1.deriv, ← hd2.deriv]
    exact heq.deriv_eq
  refine ⟨hval, ?_⟩
  rw [hval] at hd
  exact mul_right_cancel₀ (Complex.exp_ne_zero a') hd

/-- Full log-branch statement for affine-factor uniqueness: slopes agree and
constants differ by an integral period of the complex exponential. -/
theorem hadamardAffineFactor_unique_mod_period
    {P xi : ℂ → ℂ} {a b a' b' : ℂ}
    (hPcont : ContinuousAt P 0) (hP0 : P 0 ≠ 0)
    (hrep : ∀ s, xi s = Complex.exp (a + b * s) * P s)
    (hrep' : ∀ s, xi s = Complex.exp (a' + b' * s) * P s) :
    b = b' ∧ ∃ n : ℤ,
      a = a' + n * (2 * (Real.pi : ℂ) * Complex.I) := by
  have h := hadamardAffineFactor_unique hPcont hP0 hrep hrep'
  exact ⟨h.2, Complex.exp_eq_exp_iff_exists_int.mp h.1⟩

theorem genusOneHadamard_normalization_zero
    {ι : Type*} {root : ι → ℂ} {xi : ℂ → ℂ} {a b : ℂ}
    (hrep : ∀ s, xi s =
      Complex.exp (a + b * s) *
        ∏' i, genusOneCanonicalFactor root i s)
    (hxi0 : xi 0 = 1) :
    Complex.exp a = 1 := by
  have h := hrep 0
  rw [hxi0] at h
  simpa [genusOneCanonicalFactor, genusOnePrimaryFactor] using h.symm

theorem genusOneHadamard_normalization_zero_period
    {ι : Type*} {root : ι → ℂ} {xi : ℂ → ℂ} {a b : ℂ}
    (hrep : ∀ s, xi s =
      Complex.exp (a + b * s) *
        ∏' i, genusOneCanonicalFactor root i s)
    (hxi0 : xi 0 = 1) :
    ∃ n : ℤ, a = n * (2 * (Real.pi : ℂ) * Complex.I) :=
  Complex.exp_eq_one_iff.mp
    (genusOneHadamard_normalization_zero hrep hxi0)

theorem genusOneHadamard_functionalEquation_constraint
    {ι : Type*} {root : ι → ℂ} {xi : ℂ → ℂ} {a b : ℂ}
    (hrep : ∀ s, xi s =
      Complex.exp (a + b * s) *
        ∏' i, genusOneCanonicalFactor root i s)
    (hfe : ∀ s, xi (1 - s) = xi s) (s : ℂ) :
    Complex.exp (a + b * (1 - s)) *
        ∏' i, genusOneCanonicalFactor root i (1 - s) =
      Complex.exp (a + b * s) *
        ∏' i, genusOneCanonicalFactor root i s := by
  rw [← hrep, ← hrep]
  exact hfe s

theorem tendstoLocallyUniformlyOn_const_index
    {ι : Type*} (f : ℂ → ℂ) (l : Filter ι) (s : Set ℂ) :
    TendstoLocallyUniformlyOn (fun _ ↦ f) f l s := by
  intro u hu x hx
  exact ⟨Set.univ, Filter.univ_mem, Filter.Eventually.of_forall
    (fun _ y hy ↦ refl_mem_uniformity hu)⟩

/-- A Hadamard representation upgrades the symmetric genus-one product
approximants to locally uniform approximants of xi itself. -/
theorem GenusOneHadamardRepresentation.exists_tendsto_symmetricApproximants
    {ι : Type*} [DecidableEq ι]
    {root : ι → ℂ} {cutoff : ℕ → Finset ι} {height : ℕ → ℝ}
    {xi : ℂ → ℂ}
    (hexhaust : SymmetricHeightExhaustion root cutoff height)
    (hprod : HasProdLocallyUniformlyOn (genusOneCanonicalFactor root)
      (fun s ↦ ∏' i, genusOneCanonicalFactor root i s) Set.univ)
    (hrep : GenusOneHadamardRepresentation root xi) :
    ∃ a b : ℂ, TendstoLocallyUniformlyOn
      (fun N s ↦ Complex.exp (a + b * s) *
        ∏ i ∈ cutoff N, genusOneCanonicalFactor root i s)
      xi atTop Set.univ := by
  rcases hrep with ⟨a, b, hrepr⟩
  refine ⟨a, b, ?_⟩
  have hcanon :=
    hexhaust.tendstoLocallyUniformlyOn_genusOneProduct hprod
  have hpref : TendstoLocallyUniformlyOn
      (fun _ : ℕ ↦ fun s ↦ Complex.exp (a + b * s))
      (fun s ↦ Complex.exp (a + b * s)) atTop Set.univ :=
    tendstoLocallyUniformlyOn_const_index _ _ _
  have hcanon_cont : ContinuousOn
      (fun s ↦ ∏' i, genusOneCanonicalFactor root i s) Set.univ := by
    apply hcanon.continuousOn
    exact Filter.Frequently.of_forall fun N ↦ by
      apply Continuous.continuousOn
      apply continuous_finsetProd
      intro i hi
      unfold genusOneCanonicalFactor genusOnePrimaryFactor
      fun_prop
  have hmul := hpref.mul₀ hcanon (by fun_prop) hcanon_cont
  have hlimit : (fun s ↦ Complex.exp (a + b * s)) *
      (fun s ↦ ∏' i, genusOneCanonicalFactor root i s) = xi := by
    funext s
    simpa only [Pi.mul_apply] using (hrepr s).symm
  rw [hlimit] at hmul
  exact hmul

/-- The finite factor whose logarithmic derivative generates one zero's Li
summands. -/
def liFiniteProductFactor (rho z : ℂ) : ℂ :=
  (1 - (1 - rho⁻¹) * z) / (1 - z)

/-- A finite product retaining every indexed occurrence of a repeated zero. -/
def FiniteZeroMultiset.generatingProduct
    (zeros : FiniteZeroMultiset) (z : ℂ) : ℂ :=
  ∏ i, liFiniteProductFactor (zeros.root i) z

/-- The finite canonical product centered at `s = 1`.  Repeated roots remain
separate factors through the `Fin` index. -/
def FiniteZeroMultiset.canonicalProductAtOne
    (zeros : FiniteZeroMultiset) (s : ℂ) : ℂ :=
  ∏ i, (1 + (s - 1) / zeros.root i)

def FiniteZeroMultiset.genusOneCenteringConstant
    (zeros : FiniteZeroMultiset) : ℂ :=
  ∏ i, (-(1 - zeros.root i) / zeros.root i)

def FiniteZeroMultiset.genusOneInverseSum
    (zeros : FiniteZeroMultiset) : ℂ :=
  ∑ i, (zeros.root i)⁻¹

def FiniteZeroMultiset.genusOneHadamardApproximation
    (zeros : FiniteZeroMultiset) (a b s : ℂ) : ℂ :=
  Complex.exp (a + b * s) *
    ∏ i, genusOnePrimaryFactor (s / zeros.root i)

theorem genusOnePrimaryFactor_eq_centered_reflection
    {rho s : ℂ} (hρ : rho ≠ 0) (h1ρ : 1 - rho ≠ 0) :
    genusOnePrimaryFactor (s / rho) =
      (-(1 - rho) / rho) *
        (1 + (s - 1) / (1 - rho)) * Complex.exp (s / rho) := by
  unfold genusOnePrimaryFactor
  congr 1
  field_simp
  ring

/-- Exact multiplicity-aware comparison of the zero-centered genus-one
product with the one-centered Li product.  Reflection symmetry reindexes the
linear factors; the remaining discrepancy is explicit constant and
exponential data. -/
theorem FiniteZeroMultiset.genusOneProduct_eq_centered
    (zeros : FiniteZeroMultiset)
    (hsym : zeros.ReflectionSymmetric) (s : ℂ) :
    (∏ i, genusOnePrimaryFactor (s / zeros.root i)) =
      zeros.genusOneCenteringConstant *
        Complex.exp (s * zeros.genusOneInverseSum) *
          zeros.canonicalProductAtOne s := by
  rw [show (∏ i, genusOnePrimaryFactor (s / zeros.root i)) =
      ∏ i, ((-(1 - zeros.root i) / zeros.root i) *
        (1 + (s - 1) / (1 - zeros.root i)) *
          Complex.exp (s / zeros.root i)) by
    apply Finset.prod_congr rfl
    intro i hi
    apply genusOnePrimaryFactor_eq_centered_reflection
      (zeros.root_ne_zero i)
    rw [← hsym.root_partner i]
    exact zeros.root_ne_zero (hsym.partner i)]
  simp only [Finset.prod_mul_distrib]
  have hreflect :
      (∏ i, (1 + (s - 1) / (1 - zeros.root i))) =
        zeros.canonicalProductAtOne s := by
    unfold FiniteZeroMultiset.canonicalProductAtOne
    rw [← hsym.partner.prod_comp]
    apply Finset.prod_congr rfl
    intro i hi
    rw [hsym.root_partner]
    congr 2
    ring
  have hexp :
      (∏ i, Complex.exp (s / zeros.root i)) =
        Complex.exp (s * ∑ i, (zeros.root i)⁻¹) := by
    rw [← Complex.exp_sum, Finset.mul_sum]
    apply congrArg Complex.exp
    apply Finset.sum_congr rfl
    intro i hi
    rw [div_eq_mul_inv]
  rw [hexp, hreflect]
  unfold FiniteZeroMultiset.genusOneCenteringConstant
    FiniteZeroMultiset.genusOneInverseSum
  ring

theorem FiniteZeroMultiset.genusOneCenteringConstant_ne_zero
    (zeros : FiniteZeroMultiset)
    (hsym : zeros.ReflectionSymmetric) :
    zeros.genusOneCenteringConstant ≠ 0 := by
  unfold FiniteZeroMultiset.genusOneCenteringConstant
  apply Finset.prod_ne_zero_iff.mpr
  intro i hi
  apply div_ne_zero
  · simp only [neg_ne_zero]
    rw [← hsym.root_partner i]
    exact zeros.root_ne_zero (hsym.partner i)
  · exact zeros.root_ne_zero i

/-- The affine Hadamard prefactor is balanced exactly when its slope cancels
the finite inverse-root sum. -/
def FiniteZeroMultiset.HadamardBalanced
    (zeros : FiniteZeroMultiset) (b : ℂ) : Prop :=
  b + zeros.genusOneInverseSum = 0

theorem FiniteZeroMultiset.genusOneHadamardApproximation_eq_centered
    (zeros : FiniteZeroMultiset)
    (hsym : zeros.ReflectionSymmetric) (a b s : ℂ)
    (hbalance : zeros.HadamardBalanced b) :
    zeros.genusOneHadamardApproximation a b s =
      (Complex.exp a * zeros.genusOneCenteringConstant) *
        zeros.canonicalProductAtOne s := by
  unfold FiniteZeroMultiset.genusOneHadamardApproximation
  rw [zeros.genusOneProduct_eq_centered hsym]
  have hexp : Complex.exp (a + b * s) *
      Complex.exp (s * zeros.genusOneInverseSum) = Complex.exp a := by
    rw [← Complex.exp_add]
    congr 1
    rw [show a + b * s + s * zeros.genusOneInverseSum =
        a + (b + zeros.genusOneInverseSum) * s by ring, hbalance]
    ring
  calc
    _ = (Complex.exp (a + b * s) *
        Complex.exp (s * zeros.genusOneInverseSum)) *
          zeros.genusOneCenteringConstant *
            zeros.canonicalProductAtOne s := by ring
    _ = _ := by rw [hexp]

theorem liFiniteProductFactor_eq_canonical_mobius
    (rho z : ℂ) (hρ : rho ≠ 0) (hz : z ≠ 1) :
    liFiniteProductFactor rho z =
      1 + ((1 - z)⁻¹ - 1) / rho := by
  unfold liFiniteProductFactor
  field_simp
  ring

/-- Exact finite Möbius identity, with multiplicity preserved factor by
factor. -/
theorem FiniteZeroMultiset.generatingProduct_eq_canonical_mobius
    (zeros : FiniteZeroMultiset) (z : ℂ) (hz : z ≠ 1) :
    zeros.generatingProduct z =
      zeros.canonicalProductAtOne ((1 - z)⁻¹) := by
  unfold FiniteZeroMultiset.generatingProduct
    FiniteZeroMultiset.canonicalProductAtOne
  apply Finset.prod_congr rfl
  intro i _
  exact liFiniteProductFactor_eq_canonical_mobius
    (zeros.root i) z (zeros.root_ne_zero i) hz

theorem FiniteZeroMultiset.generatingProduct_eventuallyEq_canonical_mobius
    (zeros : FiniteZeroMultiset) :
    zeros.generatingProduct =ᶠ[𝓝 0]
      fun z ↦ zeros.canonicalProductAtOne ((1 - z)⁻¹) := by
  filter_upwards [
    isOpen_compl_singleton.mem_nhds (by norm_num : (0 : ℂ) ≠ 1)] with z hz
  exact zeros.generatingProduct_eq_canonical_mobius z hz

theorem FiniteZeroMultiset.logDeriv_generatingProduct_eventuallyEq_canonical_mobius
    (zeros : FiniteZeroMultiset) :
    logDeriv zeros.generatingProduct =ᶠ[𝓝 0]
      logDeriv (fun z ↦ zeros.canonicalProductAtOne ((1 - z)⁻¹)) := by
  filter_upwards [
    isOpen_compl_singleton.mem_nhds (by norm_num : (0 : ℂ) ≠ 1)] with z hz
  have hlocal : zeros.generatingProduct =ᶠ[𝓝 z]
      fun w ↦ zeros.canonicalProductAtOne ((1 - w)⁻¹) := by
    filter_upwards [isOpen_compl_singleton.mem_nhds hz] with w hw
    exact zeros.generatingProduct_eq_canonical_mobius w hw
  simp only [logDeriv, Pi.div_apply,
    zeros.generatingProduct_eq_canonical_mobius z hz, hlocal.deriv_eq]

/-- The rational generating term for one zero.  Its formal power-series
coefficients are `liZeroSummand (n + 1) rho`. -/
def liZeroGeneratingTerm (rho z : ℂ) : ℂ :=
  1 / (1 - z) - (1 - rho⁻¹) / (1 - (1 - rho⁻¹) * z)

theorem logDeriv_liFiniteProductFactor
    (rho z : ℂ) (hz : 1 - z ≠ 0)
    (hρz : 1 - (1 - rho⁻¹) * z ≠ 0) :
    logDeriv (liFiniteProductFactor rho) z =
      liZeroGeneratingTerm rho z := by
  rw [logDeriv_apply]
  unfold liFiniteProductFactor liZeroGeneratingTerm
  have hnum : HasDerivAt
      (fun w : ℂ ↦ 1 - (1 - rho⁻¹) * w) (-(1 - rho⁻¹)) z := by
    simpa [sub_eq_add_neg] using
      ((hasDerivAt_id z).const_mul (1 - rho⁻¹)).neg.const_add 1
  have hden : HasDerivAt (fun w : ℂ ↦ 1 - w) (-1) z := by
    simpa [sub_eq_add_neg] using (hasDerivAt_id z).neg.const_add 1
  have hρz' : 1 + rho⁻¹ * z - z ≠ 0 := by
    convert hρz using 1
    ring
  rw [(hnum.fun_div hden hz).deriv]
  field_simp [hz, hρz, hρz']
  ring

/-- Exact finite-product logarithmic-derivative identity.  The hypotheses
simply say that the displayed rational factors are defined and nonzero at
`z`; near `z = 0` they hold automatically. -/
theorem FiniteZeroMultiset.logDeriv_generatingProduct
    (zeros : FiniteZeroMultiset) (z : ℂ)
    (hz : 1 - z ≠ 0)
    (hfactor : ∀ i, 1 - (1 - (zeros.root i)⁻¹) * z ≠ 0) :
    logDeriv zeros.generatingProduct z =
      ∑ i, liZeroGeneratingTerm (zeros.root i) z := by
  unfold FiniteZeroMultiset.generatingProduct
  rw [logDeriv_prod]
  · apply Finset.sum_congr rfl
    intro i _
    exact logDeriv_liFiniteProductFactor (zeros.root i) z hz (hfactor i)
  · intro i _
    exact div_ne_zero (hfactor i) hz
  · intro i _
    unfold liFiniteProductFactor
    fun_prop

@[simp] theorem liZeroGeneratingTerm_zero (rho : ℂ) :
    liZeroGeneratingTerm rho 0 = liZeroSummand 1 rho := by
  simp [liZeroGeneratingTerm]

theorem FiniteZeroMultiset.logDeriv_generatingProduct_zero
    (zeros : FiniteZeroMultiset) :
    logDeriv zeros.generatingProduct 0 = zeros.liSum 1 := by
  rw [zeros.logDeriv_generatingProduct 0 (by norm_num) (by simp)]
  simp [FiniteZeroMultiset.liSum]

/-- The normalized Taylor coefficient of the finite product's logarithmic
derivative at the origin. -/
def FiniteZeroMultiset.logDerivCoefficient
    (zeros : FiniteZeroMultiset) (k : ℕ) : ℂ :=
  iteratedDeriv k (logDeriv zeros.generatingProduct) 0 /
    (k.factorial : ℂ)

/-- The exact finite all-order coefficient identity needed by the limit
transfer.  Keeping it named makes clear that this is a finite algebraic
obligation, distinct from every convergence assumption. -/
def FiniteLiTaylorIdentity (zeros : FiniteZeroMultiset) : Prop :=
  ∀ k : ℕ, zeros.logDerivCoefficient k = zeros.liSum (k + 1)

/-- The first finite Taylor coefficient is already unconditional. -/
theorem FiniteZeroMultiset.logDerivCoefficient_zero
    (zeros : FiniteZeroMultiset) :
    zeros.logDerivCoefficient 0 = zeros.liSum 1 := by
  simp [FiniteZeroMultiset.logDerivCoefficient,
    zeros.logDeriv_generatingProduct_zero]

/-- Every Taylor coefficient of one rational generating term is the
corresponding Li zero summand. -/
theorem iteratedDeriv_liZeroGeneratingTerm_zero (rho : ℂ) (k : ℕ) :
    iteratedDeriv k (liZeroGeneratingTerm rho) 0 =
      (k.factorial : ℂ) * liZeroSummand (k + 1) rho := by
  let a : ℂ := 1 - rho⁻¹
  unfold liZeroGeneratingTerm liZeroSummand
  simp only [div_eq_mul_inv, one_mul]
  change iteratedDeriv k
    (fun z : ℂ ↦ (1 - z)⁻¹ - a * (1 - a * z)⁻¹) 0 =
      (k.factorial : ℂ) * (1 - a ^ (k + 1))
  rw [iteratedDeriv_fun_sub
    ((contDiffAt_const.sub contDiffAt_id).inv (by norm_num))
    (contDiffAt_const.mul
      ((contDiffAt_const.sub (contDiffAt_const.mul contDiffAt_id)).inv (by simp)))]
  rw [iteratedDeriv_const_mul_field]
  have h₁ : (fun z : ℂ ↦ (1 - z)⁻¹) =
      fun z ↦ ((-1 : ℂ) * z + 1)⁻¹ := by
    funext z
    congr 1
    ring
  have h₂ : (fun z : ℂ ↦ (1 - a * z)⁻¹) =
      fun z ↦ (-a * z + 1)⁻¹ := by
    funext z
    congr 1
    ring
  rw [h₁, h₂, iteratedDeriv_eq_iterate, iter_deriv_inv_linear,
    iteratedDeriv_eq_iterate, iter_deriv_inv_linear]
  simp only [mul_zero, zero_add, one_zpow, mul_one]
  rw [neg_pow]
  ring_nf
  simp [show k * 2 = 2 * k by omega, pow_mul]
  ring

/-- The all-order finite Taylor identity is unconditional; it follows from
the exact finite logarithmic-derivative formula and Mathlib's closed formula
for iterated derivatives of inverse-linear functions. -/
theorem FiniteZeroMultiset.finiteLiTaylorIdentity
    (zeros : FiniteZeroMultiset) :
    FiniteLiTaylorIdentity zeros := by
  intro k
  have hz : ∀ᶠ z : ℂ in 𝓝 0, 1 - z ≠ 0 :=
    (by fun_prop : ContinuousAt (fun z : ℂ ↦ 1 - z) 0).eventually_ne (by norm_num)
  have hfactor : ∀ᶠ z : ℂ in 𝓝 0,
      ∀ i, 1 - (1 - (zeros.root i)⁻¹) * z ≠ 0 := by
    simpa only [Finset.mem_univ, forall_const] using
      (Finset.eventually_all Finset.univ).2 (fun i _ ↦
        (by fun_prop : ContinuousAt
          (fun z : ℂ ↦ 1 - (1 - (zeros.root i)⁻¹) * z) 0).eventually_ne (by simp))
  have hlog : logDeriv zeros.generatingProduct =ᶠ[𝓝 0]
      fun z ↦ ∑ i, liZeroGeneratingTerm (zeros.root i) z := by
    filter_upwards [hz, hfactor] with z hz' hfactor'
    exact zeros.logDeriv_generatingProduct z hz' hfactor'
  unfold FiniteZeroMultiset.logDerivCoefficient
  rw [hlog.iteratedDeriv_eq k]
  rw [iteratedDeriv_fun_sum]
  · simp_rw [show ∀ i, iteratedDeriv k (liZeroGeneratingTerm (zeros.root i)) 0 =
        (k.factorial : ℂ) * liZeroSummand (k + 1) (zeros.root i) by
      intro i
      exact iteratedDeriv_liZeroGeneratingTerm_zero (zeros.root i) k]
    unfold FiniteZeroMultiset.liSum
    rw [← Finset.mul_sum]
    exact mul_div_cancel_left₀ _ (by exact_mod_cast Nat.factorial_ne_zero k)
  · intro i _
    unfold liZeroGeneratingTerm
    fun_prop (disch := simp)

/-- All-order finite Möbius/binomial coefficient identity.  The normalized
Taylor coefficients of the logarithmic derivative of the transformed
canonical product are exactly the finite Li sums, with every repeated root
counted separately. -/
theorem FiniteZeroMultiset.logDeriv_canonicalMobius_coefficient
    (zeros : FiniteZeroMultiset) (k : ℕ) :
    iteratedDeriv k
      (logDeriv
        (fun z ↦ zeros.canonicalProductAtOne ((1 - z)⁻¹))) 0 /
        (k.factorial : ℂ) =
      zeros.liSum (k + 1) := by
  rw [← zeros.logDeriv_generatingProduct_eventuallyEq_canonical_mobius.iteratedDeriv_eq k]
  exact zeros.finiteLiTaylorIdentity k

/-- All-order coefficient correspondence for a balanced finite genus-one
Hadamard product.  This closes the finite prefactor/reflection bookkeeping:
after the Möbius substitution, its logarithmic-derivative coefficients are
the multiplicity-aware finite Li sums. -/
theorem FiniteZeroMultiset.logDeriv_genusOneHadamardMobius_coefficient
    (zeros : FiniteZeroMultiset)
    (hsym : zeros.ReflectionSymmetric) (a b : ℂ)
    (hbalance : zeros.HadamardBalanced b) (k : ℕ) :
    iteratedDeriv k
      (logDeriv (fun z ↦
        zeros.genusOneHadamardApproximation a b ((1 - z)⁻¹))) 0 /
        (k.factorial : ℂ) =
      zeros.liSum (k + 1) := by
  have hscale : Complex.exp a * zeros.genusOneCenteringConstant ≠ 0 :=
    mul_ne_zero (Complex.exp_ne_zero a)
      (zeros.genusOneCenteringConstant_ne_zero hsym)
  have hfun : (fun z ↦
      zeros.genusOneHadamardApproximation a b ((1 - z)⁻¹)) =
      fun z ↦ (Complex.exp a * zeros.genusOneCenteringConstant) *
        zeros.canonicalProductAtOne ((1 - z)⁻¹) := by
    funext z
    exact zeros.genusOneHadamardApproximation_eq_centered
      hsym a b _ hbalance
  have hlog : logDeriv (fun z ↦
      zeros.genusOneHadamardApproximation a b ((1 - z)⁻¹)) =
      logDeriv (fun z ↦
        zeros.canonicalProductAtOne ((1 - z)⁻¹)) := by
    rw [hfun]
    funext z
    exact logDeriv_const_mul z _ hscale
  rw [hlog]
  exact zeros.logDeriv_canonicalMobius_coefficient k

/-- Rigorous all-index transfer from finite multiplicity-aware zero sums to
the Taylor coefficients of a locally uniform nonvanishing product limit.

The hypotheses separate the four independent obligations:

* local uniform convergence of the finite products;
* holomorphy on one common open neighborhood of `0`;
* nonvanishing of every approximant and the limit there;
* the finite all-order coefficient identity.

No zeta-specific product or convergence claim is hidden in this theorem. -/
theorem finiteZeroLiSums_tendsto_of_generatingProducts
    (zeros : ℕ → FiniteZeroMultiset) (g : ℂ → ℂ) (s : Set ℂ)
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
      (𝓝 (iteratedDeriv k (logDeriv g) 0 / (k.factorial : ℂ))) := by
  have hderiv := iteratedDeriv_logDeriv_tendsto
    hs h0 hconv hhol happrox0 hlimit0 k
  have hnormalized := hderiv.div_const (k.factorial : ℂ)
  simpa only [FiniteLiTaylorIdentity,
    FiniteZeroMultiset.logDerivCoefficient] using
    hnormalized.congr' (Filter.Eventually.of_forall fun N ↦ hexact N k)

/-- All-index coefficient transfer with the finite algebraic identity
discharged.  The remaining assumptions are exactly the analytic normal
convergence, holomorphy, and common nonvanishing obligations. -/
theorem finiteZeroLiSums_tendsto_of_generatingProducts'
    (zeros : ℕ → FiniteZeroMultiset) (g : ℂ → ℂ) (s : Set ℂ)
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
  finiteZeroLiSums_tendsto_of_generatingProducts
    zeros g s hs h0 hconv hhol happrox0 hlimit0
    (fun N ↦ (zeros N).finiteLiTaylorIdentity) k

/-- Direct all-order transfer from finite canonical Möbius products through
locally uniform normal convergence. -/
theorem finiteZeroLiSums_tendsto_of_canonicalMobiusProducts
    (zeros : ℕ → FiniteZeroMultiset) (g : ℂ → ℂ) (s : Set ℂ)
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
      (𝓝 (iteratedDeriv k (logDeriv g) 0 / (k.factorial : ℂ))) := by
  have hderiv := iteratedDeriv_logDeriv_tendsto
    hs h0 hconv hhol happrox0 hlimit0 k
  have hnormalized := hderiv.div_const (k.factorial : ℂ)
  simpa only using hnormalized.congr'
    (Filter.Eventually.of_forall fun N ↦
      (zeros N).logDeriv_canonicalMobius_coefficient k)

/-- The single zeta-product analytic package needed for finite canonical
products to approximate the Möbius transform of xi. -/
def XiCanonicalProductApproximation
    (zeros : ℕ → FiniteZeroMultiset) (xi : ℂ → ℂ) (s : Set ℂ) : Prop :=
  IsOpen s ∧ (0 : ℂ) ∈ s ∧
    TendstoLocallyUniformlyOn
      (fun N z ↦ (zeros N).canonicalProductAtOne ((1 - z)⁻¹))
      (liChangeOfVariables xi) atTop s ∧
    (∀ᶠ N in atTop,
      DifferentiableOn ℂ
        (fun z ↦ (zeros N).canonicalProductAtOne ((1 - z)⁻¹)) s) ∧
    (∀ᶠ N in atTop, ∀ z ∈ s,
      (zeros N).canonicalProductAtOne ((1 - z)⁻¹) ≠ 0) ∧
    ∀ z ∈ s, liChangeOfVariables xi z ≠ 0

/-- All-order xi transfer from exactly two named obligations: the
zeta-specific canonical-product approximation package and the general
Möbius/derivative coefficient identity. -/
theorem finiteZeroLiSums_tendsto_liDerivativeCoefficient_of_canonicalProduct
    (zeros : ℕ → FiniteZeroMultiset) (xi : ℂ → ℂ) (s : Set ℂ)
    (hproduct : XiCanonicalProductApproximation zeros xi s)
    (hcoeff : LiChangeOfVariablesCoefficientIdentity xi)
    (k : ℕ) :
    Tendsto (fun N ↦ (zeros N).liSum (k + 1)) atTop
      (𝓝 (liDerivativeCoefficient xi (k + 1))) := by
  rcases hproduct with ⟨hs, h0, hconv, hhol, happrox0, hlimit0⟩
  simpa only [hcoeff k] using
    finiteZeroLiSums_tendsto_of_canonicalMobiusProducts
      zeros (liChangeOfVariables xi) s hs h0 hconv hhol
      happrox0 hlimit0 k

/-- Instantiation of the all-order product transfer for the normalized
Riemann xi function. -/
theorem riemannXiLi_finiteZeroLiSums_tendsto
    (zeros : ℕ → FiniteZeroMultiset) (s : Set ℂ)
    (hproduct : XiCanonicalProductApproximation zeros riemannXiLi s)
    (hcoeff : LiChangeOfVariablesCoefficientIdentity riemannXiLi)
    (k : ℕ) :
    Tendsto (fun N ↦ (zeros N).liSum (k + 1)) atTop
      (𝓝 (liDerivativeCoefficient riemannXiLi (k + 1))) :=
  finiteZeroLiSums_tendsto_liDerivativeCoefficient_of_canonicalProduct
    zeros riemannXiLi s hproduct hcoeff k

/-- The xi canonical-product package plus the all-order Möbius identity imply
the route's exact multiplicity-aware zeta-window formula. -/
theorem riemannLiZeroWindowFormula_of_canonicalProduct
    (s : Set ℂ)
    (hproduct : XiCanonicalProductApproximation
      (fun N ↦ riemannZetaZeroWindowMultiset (N : ℝ))
      riemannXiLi s)
    (hcoeff : LiChangeOfVariablesCoefficientIdentity riemannXiLi) :
    RiemannLiZeroWindowFormula := by
  intro n hn
  obtain ⟨k, rfl⟩ := Nat.exists_eq_succ_of_ne_zero (Nat.ne_of_gt hn)
  simpa [Nat.add_comm] using
    riemannXiLi_finiteZeroLiSums_tendsto
      (fun N ↦ riemannZetaZeroWindowMultiset (N : ℝ))
      s hproduct hcoeff k

/-- Product-limit transfer all the way to Li's derivative coefficients,
conditional only on the explicit all-order Möbius coefficient identity.
The finite algebraic identity is already discharged. -/
theorem finiteZeroLiSums_tendsto_liDerivativeCoefficient
    (zeros : ℕ → FiniteZeroMultiset) (xi : ℂ → ℂ) (s : Set ℂ)
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
      (𝓝 (liDerivativeCoefficient xi (k + 1))) := by
  simpa only [hcoeff k] using
    finiteZeroLiSums_tendsto_of_generatingProducts'
      zeros (liChangeOfVariables xi) s hs h0 hconv hhol
      happrox0 hlimit0 k

/-- Truncated generating polynomial for any coefficient sequence. -/
def liGeneratingPolynomial (a : ℕ → ℝ) (N : ℕ) (x : ℝ) : ℝ :=
  ∑ n ∈ range N, a n * x ^ n

@[simp] theorem liGeneratingPolynomial_zero (a : ℕ → ℝ) (x : ℝ) :
    liGeneratingPolynomial a 0 x = 0 := by
  simp [liGeneratingPolynomial]

theorem liGeneratingPolynomial_succ (a : ℕ → ℝ) (N : ℕ) (x : ℝ) :
    liGeneratingPolynomial a (N + 1) x =
      liGeneratingPolynomial a N x + a N * x ^ N := by
  simp [liGeneratingPolynomial, sum_range_succ]

end
