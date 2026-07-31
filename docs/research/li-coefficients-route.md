# Li-coefficient route status

## Implemented definitions

`KakeyaLeanGate/LiCriterion.lean` defines:

- `riemannXiLi s = 1 + s * (s - 1) * completedRiemannZeta₀ s`;
- the source-indexed derivative coefficient, totalized by `λ₀ = 0`;
- the real coefficient sequence and its explicit reality obligation;
- all-index non-negativity and finite-prefix non-negativity;
- the Li zero summand and indexed finite truncations;
- a concrete finite indexed multiset with nonzero roots and retained
  multiplicities;
- the symmetric-height convergence obligation;
- the corresponding Cauchy interface and completeness transfer;
- a finite rational product whose logarithmic derivative generates the
  finite zero contributions;
- the intrinsic zeta zero order `riemannZetaZeroOrder : ℂ → ℕ∞`;
- locally uniform all-order derivative and logarithmic-derivative transfer;
- normalized logarithmic-derivative Taylor coefficients of finite products;
- an all-index finite-zero-sum transfer theorem with every analytic
  hypothesis stated explicitly;
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
- exact finite-multiset identities at indices zero and one;
- non-negativity of every finite sum when each transform
  `1 - rho⁻¹` lies in the closed unit disk;
- the algebraic critical-line implication
  `rho.re = 1/2 -> ‖1 - rho⁻¹‖ = 1`, and therefore finite Li-sum
  non-negativity for every multiplicity-aware family on the critical line;
- every explicit Cauchy family of symmetric truncations converges in `ℂ`,
  producing a `HeightSymmetricLiLimit`;
- uniqueness/transfer of that limit through exact finite approximants;
- the exact finite-product logarithmic-derivative identity
  `logDeriv (prod_i ((1-a_i z)/(1-z))) =
   sum_i (1/(1-z) - a_i/(1-a_i z))`, with
  `a_i = 1 - rho_i⁻¹`;
- at `z = 0`, the finite-product logarithmic derivative is exactly the first
  finite Li sum;
- away from `s = 1`, nonzero `analyticOrderAt` for `riemannZeta` is
  equivalent to membership in Mathlib's `riemannZetaZeros`;
- locally uniform limits of holomorphic functions transfer every
  `iteratedDeriv`, by iterating Mathlib's complex Weierstrass theorem;
- on a common open nonvanishing neighborhood, logarithmic derivatives
  converge locally uniformly, and all their normalized Taylor coefficients
  converge;
- if each finite product satisfies the named all-order finite Taylor
  identity, its multiplicity-aware Li sums converge coefficientwise to the
  logarithmic-derivative Taylor coefficients of the product limit;
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

There are two nested blockers. The new finite and convergence-transfer
theorems show precisely where the first one begins:

1. Analytic infrastructure: prove the zeta-specific finite-order and
   enumeration theorem turning `riemannZetaZeroOrder` into an indexed
   multiset of non-trivial zeros; prove symmetric-height Cauchy/normal
   estimates; construct the symmetric xi/Hadamard approximants; prove their
   locally uniform convergence and common nonvanishing neighborhood after
   the Li change of variables; and discharge the finite all-order Taylor
   identity. The abstract theorem now transfers all coefficients once these
   hypotheses are supplied. None of these zeta-specific product/convergence
   facts is currently in pinned Mathlib.
2. Mathematical criterion: formalize Li's theorem that all
   `λ_n ≥ 0` for `n ≥ 1` is equivalent to every non-trivial zeta zero lying
   on `re(s)=1/2`, then identify that statement with Mathlib's canonical
   `RiemannHypothesis` (including its explicit exclusions of trivial zeros
   and `s = 1`).

After infrastructure (1), proving all coefficients nonnegative directly
would itself prove RH via (2); it is not a finite or computational next step.
The development therefore stops at the first genuine missing theorem rather
than assuming either bridge.

The finite logarithmic-derivative formula is not a recurrence in `n`.
Coefficient positivity supplies no relation determining a later coefficient
from earlier ones, and the existing `finitePrefixSpoof` theorem continues to
rule out extrapolation from any fixed prefix.
