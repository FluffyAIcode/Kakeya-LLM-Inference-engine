import Mathlib.NumberTheory.LSeries.RiemannZeta

/-!
The canonical AutoResearch root is an alias of Mathlib's pinned
`RiemannHypothesis` declaration.  The expanded branch is retained separately
and its relationship to the canonical branch is proved definitionally below.
-/

def KakeyaRiemannHypothesisRoot : Prop := RiemannHypothesis

def KakeyaRiemannHypothesisExpanded : Prop :=
  ∀ (s : ℂ), riemannZeta s = 0 →
    (¬ ∃ n : ℕ, s = -2 * (n + 1)) →
    s ≠ 1 →
    s.re = 1 / 2

theorem kakeya_rh_expanded_iff_canonical :
    KakeyaRiemannHypothesisExpanded ↔ KakeyaRiemannHypothesisRoot := by
  rfl

#check riemannZeta
#check completedRiemannZeta
#check completedRiemannZeta₀
#check riemannZeta_neg_two_mul_nat_add_one
#check RiemannHypothesis
