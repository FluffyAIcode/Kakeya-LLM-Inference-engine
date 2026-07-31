# Source card: Li coefficients and RH

## Primary criterion

- Xian-Jin Li, “The Positivity of a Sequence of Numbers and the Riemann
  Hypothesis,” *Journal of Number Theory* 65 (1997), 325–333.
  DOI: https://doi.org/10.1006/jnth.1997.2137
- Publisher record:
  https://www.sciencedirect.com/science/article/pii/S0022314X97921375

The source-standard normalization used by this route is

\[
\xi(s)=2(s-1)\pi^{-s/2}\Gamma(1+s/2)\zeta(s)
      =s(s-1)\pi^{-s/2}\Gamma(s/2)\zeta(s).
\]

Thus \(\xi(0)=\xi(1)=1\). For each integer \(n\geq 1\),

\[
\lambda_n=\frac1{(n-1)!}
 \left.\frac{d^n}{ds^n}\left(s^{n-1}\log\xi(s)\right)\right|_{s=1}.
\]

Li's criterion is the equivalence between RH and
\(\lambda_n\geq0\) for every integer \(n\geq1\). The Lean route uses
non-negativity, not finite verification and not an inferred all-index claim.
Some summaries state strict positivity; this route does not silently replace
the source theorem by that stronger wording.

## Zero-sum formulation and convergence

- Enrico Bombieri and Jeffrey C. Lagarias, “Complements to Li's Criterion
  for the Riemann Hypothesis,” *Journal of Number Theory* 77 (1999),
  274–287. Institutional abstract:
  https://www.nokia.com/bell-labs/publications-and-media/publications/complements-to-lis-criterion-for-the-riemann-hypothesis/
- Jeffrey C. Lagarias, “Li coefficients for automorphic L-functions,”
  *Annales de l'Institut Fourier* 57 (2007), 1689–1740:
  https://www.numdam.org/item/10.5802/aif.2311.pdf

With non-trivial zeros counted with multiplicity,

\[
\lambda_n=\sum_\rho\left[1-\left(1-\frac1\rho\right)^n\right].
\]

This is not an arbitrary enumeration sum. It is conditionally convergent and
uses the symmetric-height convention

\[
\lim_{T\to\infty}\sum_{|\operatorname{Im}\rho|\leq T}
 \left[1-\left(1-\frac1\rho\right)^n\right].
\]

Bombieri–Lagarias' more general multiset theorem imposes summability
conditions (in one standard formulation,
\(\sum_\rho (1+|\operatorname{Re}\rho|)/(1+|\rho|)^2<\infty\)) and excludes
exceptional points required by the chosen transform. Those hypotheses must
be instantiated, not omitted, before applying the generalized theorem to
zeta zeros.

Lagarias (2007), §1, records a useful two-sided refinement. Under the stated
multiset summability hypotheses, positivity for positive indices controls one
half-plane, positivity for negative indices controls the opposite half-plane,
and invariance under \(\rho\mapsto1-\rho\) identifies the two real parts. This
explains why positive indices suffice for zeta only after the functional
symmetry and convergence hypotheses have been established. It is an
all-index theorem, not a recurrence or a finite-prefix extrapolation.

## Finite-product generating identity

For a finite multiplicity-aware family, set
\(a_\rho=1-\rho^{-1}\). The elementary product

\[
P_F(z)=\prod_{\rho\in F}\frac{1-a_\rho z}{1-z}
\]

satisfies the exact rational identity

\[
\frac{P_F'(z)}{P_F(z)}
=\sum_{\rho\in F}\left(
  \frac1{1-z}-\frac{a_\rho}{1-a_\rho z}\right).
\]

The coefficient of \(z^{n-1}\) in each summand is
\(1-a_\rho^n\). The Lean route proves this finite-product logarithmic
derivative identity (and its value at \(z=0\)) without any convergence
assumption. Passing from these finite products to xi requires the
zeta-specific symmetric Hadamard product and locally strong convergence
sufficient to interchange the limit with logarithmic differentiation/Taylor
coefficient extraction.

## Normalization warning

The often-used Riemann function
\(\xi_{\mathrm{std}}(s)=\tfrac12s(s-1)\Lambda(s)\) differs from Li's
normalization above by the nonzero constant \(1/2\). Positive-order
derivatives of `log` are locally unchanged by this constant, after a
compatible logarithm branch is established. The Lean development avoids
needing that normalization bridge by defining Li's normalization directly.

## Formalization status

Pinned Mathlib v4.32.0-rc1 provides:

- `completedRiemannZeta`, the entire pole-subtracted
  `completedRiemannZeta₀`, and the functional equation;
- `iteratedDeriv` and complex analytic/logarithm machinery;
- `analyticOrderAt : (ℂ → ℂ) → ℂ → ℕ∞`, including the characterization
  `analyticOrderAt_ne_zero` of analytic zeros with intrinsic order;
- `TendstoLocallyUniformlyOn.differentiableOn` and
  `TendstoLocallyUniformlyOn.deriv` for Weierstrass convergence;
- `logDeriv_tendsto` and `logDeriv_tprod_eq_tsum` for logarithmic
  derivatives of locally uniform limits/products;
- `HasProdLocallyUniformlyOn` and `MultipliableLocallyUniformlyOn`;
- `riemannZetaZeros`, discreteness, and finiteness in compact sets;
- the canonical `RiemannHypothesis`.

It does not currently provide:

- a zeta-specific indexed multiset that repeats each non-trivial zero
  according to its finite analytic order;
- a proof, packaged for zeta, that every such order is finite and that the
  indexed family is complete;
- a symmetric Hadamard/canonical product for xi with locally uniform
  convergence on the neighborhood needed for coefficient extraction;
- convergence of the symmetric-height Li zero sum;
- the derivative/zero-sum identity;
- Li's all-index positivity equivalence.

Accordingly, `RiemannLiCriterionStatement` is a typed unproved proposition,
and `HeightSymmetricLiLimit` is only the exact convergence shape for an
already supplied indexed multiset. Neither is inhabited by an axiom.

The route defines `riemannZetaZeroOrder` directly from `analyticOrderAt` and
proves, away from the pole, that nonzero order is exactly membership in
`riemannZetaZeros`. It also proves that locally uniform convergence of
holomorphic, nonvanishing approximants transfers every iterated derivative
of their logarithmic derivatives. Combined with an explicit finite Taylor
identity, this transfers every multiplicity-aware finite Li sum to the
corresponding limit coefficient.

These are functional-analytic interfaces only. They do not claim that zeta
orders have been enumerated, that xi has the required symmetric product, or
that zeta-zero truncations satisfy the stated Cauchy/local-uniform hypotheses.
