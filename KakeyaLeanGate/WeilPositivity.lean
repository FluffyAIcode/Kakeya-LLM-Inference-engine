import Mathlib.Analysis.Calculus.ContDiff.Convolution
import Mathlib.Analysis.Distribution.TestFunction
import Mathlib.Analysis.InnerProductSpace.GramMatrix
import Mathlib.Analysis.MellinTransform
import Mathlib.NumberTheory.LSeries.RiemannZeta

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

open Complex MeasureTheory Set TopologicalSpace
open scoped ComplexOrder Convolution ContDiff Distributions

noncomputable section

namespace WeilPositivity

/-- Smooth compactly-supported complex tests on the logarithmic line. -/
abbrev Test := 𝓓((⊤ : Opens ℝ), ℂ)

/-- The critical-line-centred Mellin transform in logarithmic coordinates. -/
def transform (f : Test) (z : ℂ) : ℂ :=
  ∫ t : ℝ, f t * exp (z * t)

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

/-- Pullback of a distribution along a linear map of test spaces. -/
def Distribution.pullback (W : Distribution) (T : Test →ₗ[ℂ] Test) : Distribution where
  eval f := W.eval (T f)
  map_add f g := by simp [W.map_add]
  map_smul c f := by simp [W.map_smul]

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
