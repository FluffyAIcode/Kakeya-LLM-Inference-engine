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
   degree-zero/one/two/three formulas. Proved.
5. **Hyperbolicity interface** — `Hyperbolic` means every complex root of the
   scalar extension is real; `AllJensenHyperbolic` quantifies over every degree
   `d >= 1` and every shift `n >= 0`, exactly as in the cited criterion.
   Definitions elaborate.
6. **Finite RH-relevant algebra** — every nonconstant degree-one Jensen
   polynomial is hyperbolic.  For degree two, the explicit roots exist exactly
   when `a(n+1)^2 - a(n)a(n+2) >= 0` (assuming the quadratic coefficient is
   nonzero), and that condition proves hyperbolicity in the project predicate.
   Strict positivity plus sequence log-concavity therefore proves every
   degree-two Jensen polynomial hyperbolic. Proved. These statements do not
   imply RH.
7. **Differentiation identity** — O’Sullivan equation (3.1) specializes to

   `derivative (J(a,d+1,n)) = C(d+1) * J(a,d,n+1)`.

   This identity is proved coefficientwise in Lean.  Consequently, conditional
   only on the standard theorem that differentiation preserves real-rootedness,
   all-degree/all-shift hyperbolicity is equivalent to hyperbolicity of
   `J(a,d,0)` for every degree `d`.  Thus the shift quantifier is not the
   essential RH obstruction.
8. **Nested finite obligations** — `JensenSquare a k` asks for hyperbolicity
   only when positive degree and shift are both at most `k`. Lean proves

   `AllJensenHyperbolic a ↔ ∀ k, JensenSquare a k`

   and proves these squares are nested under decreasing `k`. This is a genuine
   finite-obligation reduction, not a decision procedure.
9. **Classical analytic bridge** — the source says the xi generating function
   is in the Laguerre–Pólya class iff all its Jensen polynomials are
   hyperbolic, and this is equivalent to RH. `BridgeObligations` records the
   exact unproved Lean implications; no theorem asserts them.

## Exact source-backed criterion

O’Sullivan (2021), Theorem 3.1, states the Pólya–Schur theorem with no hidden
analytic hypothesis on the formal series: for real `c_j`, the formal
exponential generating series `Φ(z)=Σ c_j z^j/j!` converges compact-uniformly
to a Laguerre–Pólya entire function iff every Jensen polynomial
`g_d(Φ;x)=Σ choose(d,j)c_j x^j` is hyperbolic, for every `d >= 1`.

For `Θ(z)=ξ(1/2+sqrt(z))=Σ γ(m)z^m/m!`, equation (3.2) identifies
`J^{d,n}=g_d(Θ^(n))`. Corollary 3.2 gives
`Θ^(n) hyperbolic ↔ ∀ d>=1, J^{d,n} hyperbolic`, and Theorem 1.1 gives
`RH ↔ ∀ d>=1 ∀ n>=0, J^{d,n} hyperbolic`. These are the exact obligations;
eventual hyperbolicity is not substituted for either direction.

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
  `Polynomial.finsetSum_coeff`, `Polynomial.coeff_derivative`;
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
the unconditional formal reduction now gives the nested family:

`forall k, JensenSquare xiGamma k`.

After importing/proving the classical derivative-preservation theorem, the
same target reduces further to:

`forall d >= 1, Hyperbolic (jensenPolynomial xiGamma d 0)`.

This all-degree unshifted family—not the shift quantifier—is the exact
coefficient-side mathematical blocker. Strict positivity and log-concavity
settle only degree two; ordinary Turán inequalities do not imply all-degree
hyperbolicity. A stronger sourced condition such as the appropriate
all-order total-positivity/Pólya-frequency condition would still require its
full implication to Jensen hyperbolicity to be formalized. The target is
equivalent to RH only after the two `BridgeObligations.polyaJensen*`
implications are proved, and it is not a consequence of known eventual
hyperbolicity.

Source metadata and exact locations are in
`docs/research/rh-jensen-source-cards.json`.
