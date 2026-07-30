# Jensen polynomial / Laguerre–Pólya route

## Scope and normalization

The Lean route uses the normalization in Griffin–Ono–Rolen–Zagier (2019),
equations (1)–(2).  Mathlib supplies the entire pole-removed completed zeta
`completedRiemannZeta₀`, not a declaration named `riemannXi`.  The project-local
definition

`riemannXi(s) = (1 + s(s-1) completedRiemannZeta₀(s)) / 2`

is the entire continuation of the classical
`ξ(s) = s(s-1) Λ(s)/2`.  Consequently `xiJensenEntire(z) = 8 ξ(1/2+z)`
matches the left side of GORZ equation (1).  `xiGammaComplex(n)` uses the
derivative normalization `n!/(2n)!`; `xiGamma(n)` is its real part.  Taking a
real part is only a typed interface: coefficient reality and identification
with the sourced real Taylor series remain explicit obligations.

## Formal chain and status

1. **Pinned RH root** — `KakeyaRiemannHypothesisRoot` is definitionally
   Mathlib's `RiemannHypothesis`. Proved in
   `KakeyaLeanGate/RiemannHypothesisRoot.lean`.
2. **Entire xi normalization** — `riemannXi`,
   `differentiable_riemannXi`, and `riemannXi_one_sub`. Proved.
3. **Centered generating function and Taylor data** —
   `xiJensenEntire`, its entirety/evenness, `xiGammaComplex`, `xiGamma`, and
   the full Mathlib Taylor expansion. Definitions and listed lemmas proved.
   Reality and the even-only sourced expansion are still bridge obligations.
4. **Jensen family** — `jensenPolynomial`, its coefficient formula, and exact
   degree-zero/one/two formulas. Proved.
5. **Hyperbolicity interface** — `Hyperbolic` means every complex root of the
   scalar extension is real; `AllJensenHyperbolic` quantifies over every degree
   and shift. Definitions elaborate.
6. **Finite RH-relevant algebra** — every nonconstant degree-one Jensen
   polynomial is hyperbolic.  For degree two, the explicit roots exist exactly
   when `a(n+1)^2 - a(n)a(n+2) >= 0` (assuming the quadratic coefficient is
   nonzero), and that condition proves hyperbolicity in the project predicate.
   Proved. These statements do not imply RH.
7. **Classical analytic bridge** — the source says the xi generating function
   is in the Laguerre–Pólya class iff all its Jensen polynomials are
   hyperbolic, and this is equivalent to RH. `BridgeObligations` records the
   exact unproved Lean implications; no theorem asserts them.

## Pinned Mathlib audit

Pinned revision: Mathlib `v4.32.0-rc1`,
`360da6fa66c1273b76b6b2d8c5666fd5ac2e3b56`.

Available and used:

- `RiemannHypothesis`, `riemannZeta`, `completedRiemannZeta`,
  `completedRiemannZeta₀`, `differentiable_completedZeta₀`,
  `completedRiemannZeta₀_one_sub`;
- `iteratedDeriv`, `iteratedDeriv_comp_neg`,
  `Complex.taylorSeries_eq_of_entire'`,
  `Differentiable.hasFPowerSeriesOnBall`;
- `Polynomial`, `Polynomial.IsRoot`, `Polynomial.Splits`,
  `Polynomial.finsetSum_coeff`;
- `TendstoLocallyUniformlyOn`,
  `TendstoLocallyUniformlyOn.differentiableOn`,
  `TendstoLocallyUniformlyOn.deriv`, and power-series partial-sum local
  uniform convergence.

Not found in the pinned library:

- a Riemann xi declaration or theorem identifying its zeros with nontrivial
  zeta zeros;
- a polynomial `IsRealRooted`/hyperbolicity predicate;
- a Laguerre–Pólya class definition;
- Jensen polynomials or the Pólya–Jensen criterion;
- a compact-uniform closure theorem specialized to real-rooted polynomials.

The project-local definitions cover only the standard typed interfaces.
The missing analytic theorems have not been replaced by assumptions.

## Exact blocker

The eventual theorem has quantifier order

`forall d, exists N(d), forall n >= N(d), Hyperbolic(J(d,n))`.

RH requires (equivalently, according to the cited criterion)

`forall d, forall n, Hyperbolic(J(d,n))`.

The first statement leaves a finite but degree-dependent set of shifts
`n < N(d)` for every degree.  Even the all-shifts results for any fixed finite
degree range leave every larger degree untreated.  Neither result implies the
required all-degree/all-shift statement, and compact-uniform convergence to a
Hermite polynomial for fixed degree does not reverse this quantifier gap.

After the coefficient-reality and Laguerre–Pólya interfaces are formalized,
the genuine mathematical blocker is therefore:

`AllJensenHyperbolic xiGamma`

or an independently proved theorem strong enough to imply it (for example,
all-order total positivity with a sourced implication to every Jensen
polynomial).  This statement is equivalent to RH only after the two
`BridgeObligations.polyaJensen*` implications are proved.  It is not a
consequence of known eventual hyperbolicity.

Source metadata and exact locations are in
`docs/research/rh-jensen-source-cards.json`.
