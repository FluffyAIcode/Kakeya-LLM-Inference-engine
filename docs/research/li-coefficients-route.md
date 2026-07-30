# Li-coefficient route status

## Implemented definitions

`KakeyaLeanGate/LiCriterion.lean` defines:

- `riemannXiLi s = 1 + s * (s - 1) * completedRiemannZeta₀ s`;
- the source-indexed derivative coefficient, totalized by `λ₀ = 0`;
- the real coefficient sequence and its explicit reality obligation;
- all-index non-negativity and finite-prefix non-negativity;
- the Li zero summand and indexed finite truncations;
- the symmetric-height convergence obligation;
- truncated generating polynomials;
- `RiemannLiCriterionStatement`, the unproved source bridge to the canonical
  Mathlib RH root.

The `completedRiemannZeta₀` definition is necessary. Mathlib assigns junk
values to `completedRiemannZeta` at its poles, so direct pointwise
multiplication by `s * (s - 1)` would incorrectly give zero at `s = 1`.

## Lean-proved facts

The route proves without `sorry`, `admit`, or `axiom`:

- `riemannXiLi 0 = riemannXiLi 1 = 1`;
- xi is entire and satisfies `xi(1-s) = xi(s)`;
- away from `0` and `1`, xi equals
  `s * (s - 1) * completedRiemannZeta s`;
- the principal complex logarithm of xi is analytic at `1`;
- the zeroth totalized coefficient is zero and the first coefficient reduces
  to the derivative of `log xi` at `1`;
- all-index positivity implies every finite prefix;
- all-index positivity is equivalent to positivity of every finite prefix
  (an actual universal quantifier, not one chosen computation);
- finite-prefix positivity is monotone under shortening;
- every fixed finite prefix can be positive while the next term is negative;
- zero-summand identities at indices zero and one, compatibility with complex
  conjugation, and the zero partial sum;
- the one-step recurrence for truncated generating polynomials.

No numerical Li value is asserted, because no interval certificate was
introduced.

## Counterexample/Critic result

`finitePrefixSpoof N` equals `1` through `N` and `-1` afterward. Lean proves
both `LiPositiveThrough (finitePrefixSpoof N) N` and failure of
`LiPositive (finitePrefixSpoof N)`. Therefore no persisted lemma treats a
computed finite prefix, monotonic-looking data, or a generating polynomial
pattern as evidence sufficient for RH.

Positivity alone also gives no recurrence or monotonicity law; such laws
would require additional analytic identities. The route persists only the
tautologically valid truncation recurrence.

## Exact remaining bridge

There are two nested blockers:

1. Analytic infrastructure: construct the non-trivial zeta zeros as an
   indexed multiset counted with analytic multiplicity, prove the
   symmetric-height sums converge, prove their equality to the derivatives
   at `1`, and prove those derivatives are real.
2. Mathematical criterion: formalize Li's theorem that all
   `λ_n ≥ 0` for `n ≥ 1` is equivalent to every non-trivial zeta zero lying
   on `re(s)=1/2`, then identify that statement with Mathlib's canonical
   `RiemannHypothesis` (including its explicit exclusions of trivial zeros
   and `s = 1`).

After infrastructure (1), proving all coefficients nonnegative directly
would itself prove RH via (2); it is not a finite or computational next step.
The development therefore stops at the first genuine missing theorem rather
than assuming either bridge.
