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

## Quantitative zero growth

- Jeffrey C. Lagarias, “Li coefficients for automorphic L-functions,”
  Theorem 2.1(4), gives
  \(N^\pm(T)=\frac{N}{2\pi}T\log T+\frac12 C_0T+O(\log T)\).
  For zeta this specializes to the Riemann--von Mangoldt estimate and implies
  `O(log T)` zeros, counted with multiplicity, in `[T,T+1]`.
- H. Montgomery and R. Vaughan, *Multiplicative Number Theory I*,
  Cambridge Studies in Advanced Mathematics 97 (2010), Chapter 12,
  Lemma 12.1 and the surrounding explicit-formula discussion:
  https://doi.org/10.1017/CBO9780511618314.014
- K. Kedlaya, *Notes on Analytic Number Theory*, Lemma 9.4, states directly
  that the number of zeta zeros in `[T,T+1]` is `O(log T)`:
  https://kskedlaya.org/ant/chap-von-mangoldt.html

Zero counting alone is insufficient for the raw Li summands: their
first-order size is `O(1/T)`, and `sum log(T)/T` diverges. Pairing conjugate
or reflected roots cancels that first-order imaginary contribution when real
parts stay in a bounded strip, leaving the source target
`O(log(T)/T^2)`. This is summable. Lean now uses the weaker sufficient
interface `O(T^(-3/2))`; Mathlib proves its summability and the resulting
Cauchy convergence. Establishing the paired quadratic estimate for actual
zeta windows remains an analytic obligation.

The formal decomposition therefore remains:

1. RH implies each nontrivial zero is on the critical line;
2. on that line, `|1 - 1/rho| = 1`, so every finite multiplicity-aware Li
   sum has nonnegative real part;
3. symmetric convergence identifies the infinite zero sum;
4. the Hadamard/change-of-variables theorem identifies that sum with the
   derivative coefficient;
5. the Bombieri–Lagarias/Li all-index theorem supplies the converse.

Only steps 1–2 are finite. Omitting steps 3–5 is precisely the invalid
finite-to-infinite extrapolation rejected by the Lean counterexample.

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
derivative identity at every Taylor order without any convergence
assumption. It also proves the exact factorwise Möbius identity

\[
P_F(z)=\prod_{\rho\in F}
\left(1+\frac{(1-z)^{-1}-1}{\rho}\right),
\]

so multiplicities are preserved literally as repeated factors. Local normal
convergence, holomorphy, and common nonvanishing then transfer all these
coefficients to the product limit. Passing from these finite products to xi
still requires the
zeta-specific symmetric Hadamard product and locally strong convergence
sufficient to interchange the limit with logarithmic differentiation/Taylor
coefficient extraction.

For the Hadamard direction, the route now uses the genus-one primary factor
\[
E_1(w)=(1-w)e^w.
\]
Lean proves the explicit local estimate
\(\lVert E_1(w)-1\rVert\leq3\lVert w\rVert^2\) for
\(\lVert w\rVert\leq1\). Consequently, for any multiplicity-indexed root
family that escapes compact sets and satisfies
\(\sum_\rho |\rho^{-1}|^2<\infty\), the product
\(\prod_\rho E_1(s/\rho)\) converges locally uniformly on all of
\(\mathbb C\). The Bombieri--Lagarias summability weight implies this
inverse-square condition once the roots are uniformly separated from zero.
Every exact symmetric-height exhaustion then converges to the same product.

The zeta-specific identification is now isolated as
`GenusOneHadamardRepresentation`:
\[
\xi(s)=e^{a+bs}\prod_\rho E_1(s/\rho).
\]
Given it, Lean proves locally uniform convergence of the affine-normalized
symmetric finite products to xi. The normalization `xi(0)=1` forces
`exp(a)=1`, and the functional equation yields the corresponding exact
reflection constraint. The existence of this representation for the
formalized complete zeta-zero multiset is not asserted.

Pinned Mathlib provides `differentiable_completedZeta₀` and
`completedRiemannZeta₀_one_sub`, but the source tree contains no packaged
finite-order theorem, no Hadamard factorization theorem specialized to it,
and no Bombieri--Lagarias theorem. The project now derives the needed
order-one estimate from lower-level Mathlib Gamma, Mellin, and zeta APIs.
The exact zeta-specific missing result is therefore a genus-one product for
normalized xi over the full multiplicity-indexed nontrivial-zero spectrum,
including convergence of symmetric finite products and their affine slopes.

Pinned Mathlib does contain a different, genuinely useful theorem:
`MeromorphicOn.exists_canonicalDecomp` in
`Analysis.Complex.CanonicalDecomposition`. It has now been instantiated for
`riemannXiLi` on every disk. The resulting factors are finite
Blaschke/canonical disk factors, depend on the radius, and provide
codiscrete-within-disk equality. They do not imply finite entire-function
order or the whole-plane genus-one Hadamard product.

Xi's zero data is now tied directly to Mathlib's divisor API:
`riemannXiZeroDivisor` has support exactly `riemannXiLi ⁻¹' {0}`; all local
orders are finite; reflection `ρ ↦ 1-ρ` preserves multiplicity; and in
`0 < re ρ`, away from `0,1`, xi and zeta have equal analytic order. Thus the
remaining divisor issue is global enumeration/exhaustion of this locally
finite divisor, not ambiguity about individual multiplicities.

The global divisor is now proved finite on every compact, and the finite
canonical decompositions are chosen along the cofinal radii `N+1`. This is an
actual exhaustion of available disk theorems, but no compatibility between
the chosen zero-free remainders follows from `CanonicalDecomp`; normal
convergence to a whole-plane genus-one product remains a separate theorem.

The Dirichlet-series API gives the genuine uniform bound
`‖ζ(s)‖ ≤ ∑ n⁻²` on `re s ≥ 2`. The classical Euler-integral estimate
\[
 |\Gamma(\sigma+it)|\leq\Gamma(\sigma)\qquad(\sigma>0)
\]
is now proved directly from Mathlib's `Complex.Gamma_eq_integral`; see
Otto Forster, *Analytic Number Theory*, §9.1,
<https://www.mathematik.uni-muenchen.de/~forster/v/ann/annth_09.pdf>.
Mathlib's real-Gamma monotonicity and `Nat.factorial_le_pow` then give
`Γ(x) ≤ ⌈x⌉^⌈x⌉` for `x ≥ 2`. Together these produce an explicit far-right
xi estimate on `re s ≥ 4`, without asserting an unavailable Stirling API.

For the central strip, bounding zeta itself is the wrong global statement
because of its pole at `s=1`. The route instead proves a generic Mellin
endpoint-majorant theorem and applies Mathlib's all-exponent Mellin
representation of `completedRiemannZeta₀`. The completion is uniformly
bounded on every closed vertical strip, and normalized xi consequently has
quadratic growth there.

The standard quantitative input is now represented by
`RiemannXiOrderOneGrowthBound`:
\[
 |\xi(s)|\le \exp(C|s|\log(|s|+2))
\]
outside a fixed disk. This estimate and the conclusion that xi has order at
most one are stated in K. Kedlaya, *18.785: Order of an entire function and
zeroes of the Riemann zeta function*, pp. 1–2, and the Encyclopedia of
Mathematics entry *Riemann xi-function* records that xi is entire of order
one and gives its Hadamard product. The proposition is not asserted in Lean
upstream, but it is now proved project-locally as
`riemannXi_orderOneGrowthBound`. The proof packages the ceiling-power
far-right estimate, quadratic central-strip bound, and functional-equation
reflection into the exact `exp(C r log(r+2))` interface.

Mathlib's Jensen/value-distribution API is now instantiated unconditionally:
the logarithmic count of `riemannXiZeroDivisor` equals the circle average of
`log ‖riemannXiLi‖`, with no additive normalization term because `xi(0)=1`.
Thus the formal zero-count side of the sourced growth argument is available.

Lean also proves optimal uniqueness for the prefactor over a fixed canonical
product: its slope `b` is unique and `exp(a)` is unique. Equality of the raw
constants `a` is not valid without fixing a logarithm branch; precisely, two
such constants differ by an element of `2πiℤ`.

This is the classical factorization in E. C. Titchmarsh,
*The Theory of the Riemann Zeta-function*, 2nd ed., revised by
D. R. Heath-Brown (1986), §2.12, especially (2.12.5). The Encyclopedia of
Mathematics records the same formula and references Titchmarsh:
https://encyclopediaofmath.org/wiki/Riemann_xi-function

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
- escape of the zero subtype to the cocompact filter;
- the canonical `RiemannHypothesis`.

It does not currently provide:

- a single global zeta-zero enumeration ordered by symmetric height;
- a symmetric Hadamard/canonical product for xi with locally uniform
  convergence on the neighborhood needed for coefficient extraction;
- convergence of the symmetric-height Li zero sum;
- a Riemann--von Mangoldt zero-counting asymptotic or quantitative
  strip/shell-growth bound suitable for a summable majorant;
- the derivative/zero-sum identity;
- Li's all-index positivity equivalence.

Accordingly, `RiemannLiCriterionStatement` is a typed unproved proposition,
and `HeightSymmetricLiLimit` is only the exact convergence shape for an
already supplied indexed multiset. Neither is inhabited by an axiom.

The route defines `riemannZetaZeroOrder` directly from `analyticOrderAt`,
proves away from the pole that nonzero order is exactly membership in
`riemannZetaZeros`, and proves every such order is finite by analytic
continuation from the nonzero value at `s = 2`. Compact-zero finiteness then
produces concrete bounded norm windows, indexed by
`Σ rho, Fin multiplicity`, with exact repetition. It also proves that locally
uniform convergence of
holomorphic, nonvanishing approximants transfers every iterated derivative
of their logarithmic derivatives. The finite Taylor identity is now proved
at every order, so normal convergence transfers every multiplicity-aware
finite Li sum to the corresponding limit coefficient without a separate
finite algebra hypothesis.

The route now also constructs finite closed rectangles
`|Re rho| ≤ A, |Im rho| ≤ T`, including boundary zeros and analytic
multiplicity, and injectively reindexes them as `T` grows. Under the explicit
hypothesis that all nontrivial zeros satisfy `|Re rho| ≤ A`, these are exactly
the source-standard symmetric-height windows. This hypothesis is not silently
asserted: pinned Mathlib does not provide the required unconditional
nontrivial-zero strip classification.

For convergence, the route proves two precise interfaces:

- unconditional summability plus cofinal exact height exhaustion identifies
  the limit with the `tsum`;
- a summable majorant for distances between successive height shells gives
  the required Cauchy sequence. Concretely, `ThreeHalvesLiShellBound` assumes
  `dist(S_N,S_(N+1)) ≤ C_n (N+1)^(-3/2)`; Lean proves this majorant summable
  and obtains `HeightSymmetricLiCauchy`.

`LogarithmicZeroShellCount` records the Riemann--von Mangoldt `O(log T)`
input, while `QuadraticPairedLiShellCancellation` records the combined
`O(log T/T^2)` target. Lean proves this latter majorant summable and derives
`HeightSymmetricLiCauchy`; it is not asserted for zeta without a proof.

The latter is where a zeta zero-count/growth estimate must enter. The route
does not claim such a majorant for zeta. A search of the pinned
`Mathlib/NumberTheory/LSeries` zeta API found only discreteness, compact
finiteness, and cocompact escape for zeros, not the quantitative estimate
needed to discharge it.

Finally, `liChangeOfVariables xi z = xi ((1-z)⁻¹)` is explicit. Lean proves
its zeroth logarithmic-derivative coefficient equals `λ₁` and transfers all
orders from normally convergent nonvanishing finite canonical Möbius
products. The finite all-order Möbius identity is closed; the distinct
general identity equating those limit coefficients with Li's derivative
definition remains named `LiChangeOfVariablesCoefficientIdentity`.

For the converse, Lean proves
`‖1-rho⁻¹‖ ≤ 1 ↔ Re rho ≥ 1/2`. It then isolates the finite Bombieri--Lagarias
step as `FiniteLiSpectralConverseStatement`: aggregate positivity at every
positive index must imply each individual transform lies in the unit disk.
Given this statement and multiplicity-preserving reflection symmetry,
critical-line location follows formally. Proving that spectral implication
under the source summability hypotheses, the zeta-specific symmetric
product, the all-index zero-window formula, and the infinite Li converse
remain genuine source bridges rather than assumptions.

Generic normal convergence is no longer part of that blocker: it follows
from the proved genus-one estimate and inverse-square summability. The
remaining product theorem is specifically the Hadamard identification of
that product with xi (including the affine exponential prefactor), together
with the comparison of those genus-one approximants to the centered finite
Li factors. This comparison is also where reflection reindexing and the
prefactor contribution to the all-order coefficient identity must be
resolved globally.

The finite comparison is now closed exactly. For a reflection-symmetric
multiplicity-indexed finite family,
\[
\prod_\rho E_1(s/\rho)
=C_F\,e^{s\sum_\rho\rho^{-1}}
 \prod_\rho\left(1+\frac{s-1}{\rho}\right).
\]
If the affine Hadamard slope cancels the finite inverse-root sum, the
remaining factor is a nonzero constant, so every Möbius-transformed
logarithmic-derivative coefficient equals the corresponding finite Li sum.
The infinite missing statement is therefore the convergence/balancing of
these affine slopes and inverse-root sums for the zeta exhaustion, not any
further finite binomial algebra.

Mathlib's simply-connected branch-log theorem also supplies a global
continuous logarithm for every continuous zero-free whole-plane quotient.
The project now proves that, for an entire quotient, this lift is entire and
that its derivative equals the quotient's logarithmic derivative. It also
proves that vanishing second derivative makes the logarithm affine and the
quotient exponential-affine.

The remaining Hadamard rigidity theorem is now closed in the exact form
needed. Cauchy's estimate proves rigidity from `o(‖z‖²)` norm growth.
Borel--Carathéodory strengthens this to the disk-real-part condition
`sup_{‖z‖<r} re h(z)=o(r²)`. Since `exp(h)=g` implies
`re h=log ‖g‖`, the zeta-specific obligation is exactly the named
`SubquadraticLogNormGrowth g` estimate for the xi/canonical-product quotient,
not another abstract complex-analysis lemma.

The global xi upper bound alone does not imply this quotient estimate.
Likewise, an upper bound for the canonical product does not bound its
reciprocal near its zeros. Since xi and the product have common zeros, the
regularity comes from cancellation in the analytic quotient. What remains is
a minimum-modulus/cancelled-quotient estimate, or an equivalent finite-order
Hadamard factorization theorem; treating the reciprocal factors separately
would be an invalid argument.

The cancellation step is now formalized independently of that estimate.
Given equal finite global divisors, Mathlib's `toMeromorphicNFOn` changes
`xi * P⁻¹` only at the discrete singular set and produces an entire,
zero-free quotient. If `SubquadraticLogNormGrowth` is supplied for this
normal-form quotient, Lean constructs its entire logarithm, proves it affine,
and analytically continues the resulting identity
`xi(s)=exp(a+bs)P(s)` across all common zeros.

The canonical-product divisor is no longer an assumed product-side fact.
Inverse-square normal convergence gives pointwise summability of
`‖E₁(s/ρ)-1‖`; `tprod_one_add_ne_zero_of_summable` then excludes every
unlisted zero. Removing the finite fiber over a root leaves a normally
convergent nonvanishing product, while
`analyticOrderAt_genusOneCanonicalFactor_root` proves every retained factor
has order one. Consequently a finite fiber contributes exactly its
cardinality, and an order-matching enumeration has the same global divisor
as xi.

Xi's locally finite divisor now supplies that enumeration concretely:
`RiemannXiZeroIndex` is the sigma type of xi zeros paired with
`Fin (riemannXiZeroMultiplicity ρ)`. Hereditary Lindelöfness of `ℂ` makes the
discrete divisor support countable; the construction handles empty, finite,
and infinite zero sets without case-specific choices. Its finite fibers have
exact analytic-order cardinality, every xi zero occurs, and compact local
finiteness proves `riemannXiZeroRoot_escape`.

The shell-to-product implication is now proved. For any multiplicity-indexed
partition with shell cardinality at most `A log(N+2)` and root norm at least
`N+1`, comparison with `log(N+2)/(N+1)^2` proves inverse-square summability.
`summable_norm_inv_sq_of_finite_low_and_logarithmic_shell_count` removes and
restores an arbitrary finite low-zero set, so no unproved lower bound on the
first xi zero is used.

The Jensen route is now partly instantiated rather than merely cited.
`exists_riemannXiZeroDivisor_logCounting_le_orderOne` combines Mathlib's
Jensen formula with `riemannXi_orderOneGrowthBound` to prove
`N_log(r) ≤ C r log(r+2)` for xi's divisor, with analytic multiplicities.
Jensen/order-one growth alone does not prove an `O(log N)` unit-shell count;
that local estimate is a Riemann--von Mangoldt input. Instead,
`summable_norm_inv_sq_of_dyadic_linear_log_shell_count` proves the
mathematically sufficient dyadic route: shell cardinality
`O(2^N(N+1))` yields inverse-square contribution `O((N+1)/2^N)`.
Its finite-low-zero specialization feeds directly into
`riemannXiGenusOneCanonicalProduct_divisor_eq_of_dyadicShellCount`.
The divisor-to-index step is now proved with exact constants:
`riemannXiZeroDivisor_apply` identifies divisor values with analytic
multiplicity, the finite index-window divisor is dominated pointwise by the
full divisor, and evaluation at radius `2r` yields
`card(window r) log 2 ≤ N_log(2r)`. Half-open dyadic intervals then give a
unique shell for every index outside a finite low window. This proves
`summable_norm_inv_sq_riemannXiZeroRoot`,
`riemannXiGenusOneCanonicalProduct_hasProdLocallyUniformly`, and
`riemannXiGenusOneCanonicalProduct_divisor_eq_unconditional`.

For the missing growth theorem, see Paul Garrett,
*Weierstrass and Hadamard products*, Theorem 3.1,
<https://www-users.cse.umn.edu/~garrett/m/complex/notes_2020-21/12_Hadamard_products.pdf>.
The standard finite-order Hadamard proof controls the cancelled quotient,
not the raw reciprocal product. Mathlib's
`Analysis.Complex.ValueDistribution.Cartan` explicitly notes that Cartan's
formula is future work, and the current value-distribution API contains no
replacement minimum-modulus theorem.
The exceptional-disk geometry is now proved. If finite disks have
`2 * sum radii < b-a`, Lebesgue measure supplies a radius in `[a,b]` whose
centered circle is disjoint from every disk; the factor `2` comes from the
exact radial interval width. `CartanExceptionalDiskLogBound` packages only
the still-missing infinite-product analytic lower bound outside these disks.
The finite lower bound is no longer missing: the elementary equal-radius
cover with radius `H/(3n)` has total diameter `2H/3`, gives
`|prod(z-ρ)| ≥ (H/(3n))^n`, and gives the genus-one factorwise bound
`(H/(3n))/|ρ| * exp(-|z|/|ρ|)`. The selected-circle version and its canonical
xi-window specialization are proved. A separate finite-tail estimate bounds
the product's distance from one by
`exp(3|z|² sum |ρ|⁻²)-1`.
This estimate now passes to the actual infinite complementary product:
window cofinality and summability prove that the inverse-square tail mass
tends to zero, and the same exponential bound holds uniformly on
`|z|≤R`. The head terms satisfy
`sum log|ρ| ≤ card(window T) log T`, and Cauchy--Schwarz bounds
`(sum |ρ|⁻¹)²` by cardinality times the global inverse-square mass.
`subquadraticBoundaryLogNormGrowth_of_cartanExceptionalDisks` performs the
cofinal circle selection, and the existing maximum-modulus proof upgrades it
to `SubquadraticLogNormGrowth`. Since inverse-square summability and divisor
matching are unconditional,
`riemannXiGenusOneHadamardRepresentation_of_cartanExceptionalDisks` closes
the analytic log, affine rigidity, and whole-plane identity from that sole
minimum-modulus hypothesis. No minimum-modulus claim is inferred from
separate upper bounds for xi and its product.

A source-level target is Cartan's polynomial lemma: for a monic degree-`n`
polynomial and `h>0`, at most `n` exceptional disks with total radius at most
`2eh` suffice to ensure `|P(z)|>h^n` outside them. See Levin,
*Distribution of Zeros of Entire Functions*, Chapter I, §7, and the statement
reproduced in
<https://doi.org/10.1307/mmj/1070919561>, Lemma 3.1. The sharper disk-merging
constant from that lemma is not yet formalized, but the elementary
equal-radius variant is sufficient in shape for the order-one application.
What remains is a quantitative dyadic decay theorem for the complementary
inverse-square mass (classically `O(log R/R)`) and a joint truncation/circle
choice that preserves the finite-head constants. The weighted geometric
series and xi shell estimate are now formalized:
`sum_{k≥N} shellMass(k) ≤ A(N+1)/2^N`. Exact reindexing from the complement
of the radius-`2^N` window to shifted dyadic shells is now proved using a
strict lower boundary at the truncation
radius and half-open dyadic shells, so boundary roots occur exactly once or
remain in the closed head window. The compatible choice
`R_j=2^j`, `N_j=4j` is also proved to make
`R_j²(N_j+1)/2^N_j` tend to zero. The sourced shell majorant is now
transported through the boundary-safe partition, giving the explicit radial
estimate
`tailMass(2^N) ≤ A(N+1)/2^N`. Constant tracking corrects the truncation
choice to `N_j=j`: the earlier `4j` choice controls the tail but makes the
available finite-head loss too large. At `N_j=j`, both the tail contribution
`O(j·2^j)` and head root-log contribution `O(2^j j²)` are subquadratic
relative to `R_j²=2^(2j)`. Circle selection with radius in
`[2^j,2^(j+1)]` is formalized. The remaining obligation is to combine the
finite head and infinite tail into one canonical-product lower bound. The
tail side is now multiplicative rather than additive: the principal-log
estimate on `‖w‖≤1/2` proves
`exp(-‖w‖²)≤‖(1-w)exp(w)‖`, and normal convergence gives
`exp(-‖z‖² tailMass(T))≤‖tailProduct(z)‖` whenever `‖z‖≤T/2`.
Consequently the head cutoff must be a fixed factor beyond the selected
circle, e.g. `T=2^(j+2)` for circles in `[2^j,2^(j+1)]`; this does not alter
the asymptotic estimates. The previous formal obligation was the
head/complement `tprod` split and multiplication of the two lower bounds.
This is now proved by establishing convergence on the finite and
complementary subtypes separately and applying the commutative-monoid
`HasProd.mul_compl` identity. The resulting selected-circle theorem bounds
the full canonical product below using cutoff `2^(j+2)`. The remaining
obligation is to aggregate the explicit head and tail logarithmic losses and
combine them with xi's order-one upper bound to control the cancelled
quotient. This bridge is now formalized: the finite head is exactly
`exp(-riemannXiFiniteHeadLogLoss)`, its three terms have explicit
cardinality/root-log/Cauchy--Schwarz bounds, and the tail contributes
`‖z‖² tailMass(T)`. Away from product zeros, the normal-form cancelled
quotient equals `xi/P`; hence the selected-circle estimate is
`‖q(z)‖≤exp(C‖z‖log(‖z‖+2)+circleLoss(z))`.
The remaining obligation is the real-asymptotic proof that the displayed
loss is `o(2^(2j))`, especially its square-root and
`card·log(3 card/H)` terms. Jensen's count is now specialized without
strengthening it:
`card(window(2^(j+2)))≤B·2^j(j+1)`. The inequality `log x≤x` reduces the
normalized Cartan cardinal-log loss to
`3B²(j+1)²/2^j`, while Cauchy--Schwarz reduces the normalized inverse-root
loss to a constant multiple of `sqrt((j+1)/2^j)`; both convergence statements
are formalized. What remains is their simultaneous epsilon-level assembly
with the already quantitative root-log and tail terms. The common
polynomial/geometric + square-root + linear/geometric expression is now
packaged as `DyadicCartanLossMajorant`, and its eventual epsilon bound is
proved. The apparent every-real-annulus mismatch is now removed:
`DyadicSubquadraticBoundaryLogNormGrowth` is enough. The next dyadic scale is
at most twice the requested scale, so requesting `ε/4` absorbs the squared
loss; maximum modulus then fills every disk. The corresponding conversion to
global `SubquadraticLogNormGrowth` and the xi Hadamard pipeline are proved.
It remains only to dominate the explicit selected-circle xi quotient bound
by one fixed instance of `DyadicCartanLossMajorant`. This is now formalized
with fixed constants `2B+3B²`, `4BM`, and `4C+3A`; every numerator,
finite-head, inverse-root, cardinal-log, and radial-tail term is matched.
Consequently the dyadic boundary and affine xi Hadamard representation are
proved under `Nonempty RiemannXiZeroIndex`, with normalization `exp a=1`.
The inhabitation is now proved rather than imported: an empty product makes
xi exponential-affine, reflection forces zero slope, and the sourced formulas
`ζ(2)=π²/6` and `π>3` give `xi(2)=π/3≠1`, contradicting constancy.
Accordingly the Hadamard and normalized representations are unconditional.
The separate all-order xi Möbius/Faà di Bruno coefficient identity remains
explicitly open; this is now the exact coefficient-transfer blocker. The
Li-derivative side is nevertheless closed at all orders:
`liDerivativeCoefficient_eq_leibnizSum` expands it into the exact finite
`choose * descFactorial * iteratedDeriv(log xi)` sum.
`MobiusFaaDiBrunoCoefficientIdentity` exposes the matching composition formula
for `(1-z)⁻¹`, and its equivalence to the previous coefficient interface is
proved. The remaining step is therefore purely the all-order composition
identity, not normal-convergence or zero-multiplicity interchange.
The inner Möbius derivatives are now source-free algebra:
`D^k(1-z)⁻¹ = k!(1-z)^(-(k+1))`, globally and hence at zero. The outstanding
Faà di Bruno work is reduced to iterating the two-term product/chain
recurrence and identifying its finite coefficients with
`choose(k+1,i) * descFactorial(k,i)`. This is the genuine remaining
combinatorial blocker. The combinatorial part is now proved:
`IsMobiusFaaDiBrunoCoefficientRecurrence` records base/support/step data,
the closed choose/falling-factorial coefficients satisfy it, and the
recurrence is unique. The remaining obligation is the local analytic bridge
from repeated product/chain differentiation (or Mathlib's ordered-partition
`HasFTaylorSeriesUpToOn.comp`) to that recurrence. Mere
convergence of the tail mass to zero does not control `R² * tailMass(R)`.
The scalar regularity component of this bridge is now proved:
`AnalyticAt.differentiableAt_iteratedDeriv` derives differentiability of
every scalar `iteratedDeriv m` from Mathlib's analytic `iteratedFDeriv`
closure and scalar evaluation equivalence. The exact remaining step is the
finite-sum product/chain-rule reindexing that realizes the triangular
recurrence for the composition.
The analytic side has now advanced further:
`hasDerivAt_mobiusFaaDiBrunoSummand` proves the exact two-contribution
derivative of each specialized basis term, while
`hasDerivAt_mobiusFaaDiBrunoSum` sums those derivatives over the finite
triangle and verifies every natural-subtraction boundary. Only the finite
index shift and coefficient collection remain before recurrence uniqueness
can discharge `MobiusFaaDiBrunoCoefficientIdentity`. Those finite steps are
now proved by `mobiusFaaDiBrunoSum_reindex` and
`mobiusFaaDiBrunoCoefficientSum_reindex`, including support at both endpoints
and the artificial zero top index. The remaining gap is local analytic
induction: upgrading the pointwise row formula to an eventual neighborhood
identity so its derivative can be replaced by the next collected row.
This local induction gap is now closed generically:
`iteratedDeriv_mobiusComposition_eq_row` preserves an eventual neighborhood
through every differentiation step, and
`iteratedDeriv_mobiusComposition_zero` specializes the result to the exact
closed coefficient sum at zero. The remaining xi-specific obligation is to
identify its transformed logarithmic derivative with this generic
composition germ using local nonvanishing and `Complex.log` differentiation.
This final germ identification is now proved. The normalized value
`riemannXiLi 1 = 1`, continuity, and openness of `Complex.slitPlane` yield a
zero-free transformed neighborhood. The exact principal-log chain rule is
`logDeriv_liChangeOfVariables_eq_mobiusLogDerivative`; its eventual xi
specialization transfers all iterated derivatives. Thus both
`riemannXiLi_mobiusFaaDiBrunoCoefficientIdentity` and the original
`riemannXiLi_changeOfVariablesCoefficientIdentity` are unconditional.

The exact remaining forward-criterion obligation is to instantiate
`XiCanonicalProductApproximation` for the concrete xi zero-window
approximants from the proved whole-plane Hadamard representation. Once this
is supplied, the existing zero-window convergence theorem gives
`RiemannLiZeroWindowFormula` and hence RH-to-positivity. The
Bombieri--Lagarias converse remains unproved and is not inferred from the
coefficient identity.
Concrete radial Hadamard approximation is now proved:
`riemannXiZeroIndexWindow_nat_tendsto_atTop` gives cofinality of the
multiplicity windows,
`riemannXiRadialGenusOneProducts_tendstoLocallyUniformly` gives normal
convergence of their genus-one products, and
`exists_riemannXiRadialHadamardApproximants_tendstoLocallyUniformly` restores
the affine factor and converges to xi globally.

This still does not instantiate `XiCanonicalProductApproximation`, whose
finite factors are centered at one and carry no affine prefactor. The precise
missing estimate/identity is convergence of
`b + ∑_{rho in window} rho⁻¹` to zero (with the corresponding centering
constant normalization), plus a multiplicity-preserving comparison between
the canonical xi enumeration windows and the current zeta-divisor windows.
Without these, removing the exponential correction would be an invalid
finite-to-infinite extrapolation.
Multiplicity-exact functional-equation pairing is now available internally:
`riemannXiZeroReflection` transports each finite-fiber ordinal to the
reflected zero, and `riemannXiZeroRoot_inv_add_reflection` reduces the paired
inverse-root term to `(rho(1-rho))⁻¹`. This is precisely the
inverse-quadratic cancellation needed by a symmetric exhaustion.

However, `riemannXiZeroReflection_mem_window_add_one` proves that radial
windows are reflection-compatible only after a radius-one enlargement, not
exactly invariant. The pinned cumulative Jensen estimate `O(R log R)` does
not control this unit boundary shell. Closing the slope correction requires
either a sourced Riemann--von Mangoldt/local shell estimate or symmetric
height windows together with a fixed-real-strip theorem for all xi zeros.
Pinned Mathlib supplies neither latter zeta-specific result, so no slope
limit, `XiCanonicalProductApproximation`, zero-sum formula, or RH positivity
is claimed from the present hypotheses.
The real-strip alternative is now discharged internally. Mathlib's
`riemannZeta_ne_zero_of_one_le_re`, xi/zeta zero compatibility, and the xi
functional equation prove every xi zero lies in `0 < Re rho < 1`. This makes
the new `riemannXiZeroHeightWindow` finite and proves additive-one comparison
with radial windows and cofinality of both exhaustions.

Reflection acts as an exact multiplicity permutation on every height window.
The resulting theorem
`two_mul_sum_inv_riemannXiZeroHeightWindow` converts each finite inverse-root
sum into the inverse-quadratic paired sum. What remains is the analytic
limit/slope identification (and a completed-zeta conjugation theorem for the
requested conjugation symmetry), followed by the multiplicity-exact
xi-window/zeta-divisor comparison. Until those are proved,
`XiCanonicalProductApproximation` and the all-index positivity consequences
remain open.

The paired-limit analytic obstruction has since narrowed further. The paired
terms `1/(rho(1-rho))` are absolutely summable, the exact symmetric-height
inverse-root sums converge to half their `tsum`, and each finite paired sum
is the logarithmic derivative at `1` of its genus-one product. What is not
yet packaged is the derivative/logarithmic-derivative passage through the
locally uniform infinite-product limit needed to identify this value with
the negative affine Hadamard slope. Conjugation and the classification of
all nontrivial zeta zeros into the critical strip also remain unavailable
from pinned Mathlib.

The derivative/slope passage is now formalized. Weierstrass derivative
convergence for the locally uniform height products identifies the infinite
product's logarithmic derivative at `1` with the paired inverse-root `tsum`;
the derivative at `0` vanishes. Combining these facts with the normalized
Hadamard representation and xi functional equation proves that this `tsum`
is `-2b`, so symmetric-height inverse-root sums converge to `-b`.

Pinned Mathlib does contain `Complex.Gamma_conj`, `Complex.conj_tsum`, and
the zeta Dirichlet-series identity on `Re s > 1`, but no packaged analytic
continuation theorem asserting zeta or completed-zeta conjugation. It also
does not classify every zero excluded by `IsNontrivialRiemannZetaZero` into
the open critical strip. These are now the genuine blockers for
conjugation-equivariant multiplicity indices, xi/zeta height-window
identification, and the unconditional canonical-product package.

The functional-equation patch is also closed: a norm estimate on
`re s ≥ 1/2` with a majorant symmetric in `s` and `1-s` automatically extends
globally. Hence the missing sourced estimate can be developed solely in the
right half-plane; no separate left-half-plane Gamma/zeta analysis is needed.

At the theorem-interface level, those two obligations now instantiate the
all-order xi limit directly and imply `RiemannLiZeroWindowFormula` for the
concrete multiplicity-aware zeta windows.

The infinite interface now records the source summability condition as
`BombieriLagariasSummability` and separates the steps:

1. symmetric zero-sum convergence;
2. the Bombieri--Lagarias spectral implication from all-index positivity to
   individual unit-disk bounds;
3. reflection symmetry, which Lean proves converts those bounds to
   `Re rho = 1/2`.

An explicit reflected pair away from the critical line together with the
unrelated zero coefficient sequence is Lean-checked as a counterexample to
omitting step 1/its product identity. It is not a counterexample to the
Bombieri--Lagarias theorem because the coefficients are deliberately not the
multiset's Li sums.

A second Lean counterexample shows that reflection symmetry plus the
Bombieri--Lagarias summability weight alone still permits off-line roots.
Hence the unresolved source theorem is precisely the positivity-to-location
implication for the genuine convergent Li sums; summability and symmetry
cannot replace positivity.

The source-shaped half-plane conclusion has now been separated explicitly:
under the Bombieri--Lagarias summability and genuine symmetric Li limits,
all-index positivity must prove `1/2 ≤ re ρ`. For nonzero roots this is
Lean-equivalent to the existing individual Li-transform unit-disk bound.
Reflection then forces equality. The positivity-to-half-plane implication
itself remains absent from pinned Mathlib.

`RiemannLiSpectralModel` now packages a complete nontrivial-zero enumeration,
reflection, source summability, and the genuine symmetric Li limits. Lean
proves that positivity plus the Bombieri--Lagarias half-plane implication for
such a model yields `KakeyaRiemannHypothesisRoot`. The only unproved spectral
step in that theorem is exactly positivity-to-half-plane location.
