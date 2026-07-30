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
- `riemannZetaZeros`, discreteness, and finiteness in compact sets;
- the canonical `RiemannHypothesis`.

It does not currently provide:

- analytic zero orders/multiplicities for the zeta zero set;
- an indexed multiset of all non-trivial zeros with multiplicity;
- convergence of the symmetric-height Li zero sum;
- the derivative/zero-sum identity;
- Li's all-index positivity equivalence.

Accordingly, `RiemannLiCriterionStatement` is a typed unproved proposition,
and `HeightSymmetricLiLimit` is only the exact convergence shape for an
already supplied indexed multiset. Neither is inhabited by an axiom.
