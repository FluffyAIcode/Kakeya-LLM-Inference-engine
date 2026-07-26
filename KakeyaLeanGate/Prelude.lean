import Mathlib.Analysis.Complex.Basic
import Mathlib.Analysis.Complex.Hadamard
import Mathlib.Analysis.Complex.JensenFormula
import Mathlib.Analysis.Complex.LocallyUniformLimit
import Mathlib.Analysis.Complex.Order
import Mathlib.Analysis.Analytic.Uniqueness

/-!
Minimal import target for AutoResearch theorem-signature validation.

Generated signatures are compiled in temporary files importing this module.
They may use `sorry` while their status is `FORMALIZED`; a proof obligation can
only become `PROVED` after a separate no-sorry/no-axiom proof gate.
-/

open Filter Metric Set

/- Host-owned semantic atoms used by the typed decomposition registry.  These
are definitions, not axioms; generated declarations can mention them, while
proof acceptance still requires the separate no-sorry Lean gate. -/
def polesOutsideDisk (poles : Set ℂ) (center : ℂ) (radius : ℝ) : Prop :=
  Disjoint poles (ball center radius)

def localUniformConvergenceOnDisk
    (terms : ℕ → ℂ → ℂ) (sum : ℂ → ℂ) (center : ℂ) (radius : ℝ) : Prop :=
  TendstoLocallyUniformlyOn terms sum atTop (ball center radius)

def termsHolomorphicOnDisk
    (terms : ℕ → ℂ → ℂ) (center : ℂ) (radius : ℝ) : Prop :=
  ∀ n, DifferentiableOn ℂ (terms n) (ball center radius)

def holomorphicSumOnDisk
    (sum : ℂ → ℂ) (center : ℂ) (radius : ℝ) : Prop :=
  DifferentiableOn ℂ sum (ball center radius)

def agreesWithSimplePoleOnPuncturedDisk
    (sum : ℂ → ℂ) (center residue : ℂ) (radius : ℝ) : Prop :=
  EqOn sum (fun s => residue / (s - center)) (ball center radius \ {center})

def nonzeroComplex (z : ℂ) : Prop := z ≠ 0

def positiveRadius (radius : ℝ) : Prop := 0 < radius
