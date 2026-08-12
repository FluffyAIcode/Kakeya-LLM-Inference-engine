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
derivative normalization `n!/(2n)!`; `xiGamma(n)` is its real part. Lean now
proves that `xiGammaComplex(n)` is exactly the complex scalar extension of
`xiGamma(n)` and proves the sourced even Taylor expansion as a `HasSum`.

## Formal chain and status

1. **Pinned RH root** — `KakeyaRiemannHypothesisRoot` is definitionally
   Mathlib's `RiemannHypothesis`. Proved in
   `KakeyaLeanGate/RiemannHypothesisRoot.lean`.
2. **Entire xi normalization** — `riemannXi`,
   `differentiable_riemannXi`, and `riemannXi_one_sub`. Proved.
3. **Centered generating function and Taylor data** —
   `xiJensenEntire`, its entirety/evenness, `xiGammaComplex`, `xiGamma`, and
   the full Mathlib Taylor expansion. `completedRiemannZeta₀_conj` proves
   global completed-zeta conjugation from the Dirichlet series on `Re(s)>1`,
   `Gamma_conj`, `cpow_conj`, and analytic uniqueness. Consequently
   `xiGammaComplex_im_zero`, `xiGammaComplex_eq_ofReal`, odd-derivative
   vanishing, and `xiJensenEntire_hasSum_xiGamma` are proved.  The pinned
   defining integral is now exposed exactly by
   `completedRiemannZeta₀_eq_mellin` and `completedZetaMellinKernel`.
   `riemannPhiTerm` is the authoritative explicit Riemann `Φ` summand from
   Lagarias--Montague, in a factored form. Lean proves its equality to the
   published formula, strict positivity, summability over the positive integer
   index, and the resulting `riemannPhi` kernel's strict positivity and
   symmetry. A summable Weierstrass majorant gives continuity and the global
   bound `Phi(u) <= C exp(-3|u|/2)`; consequently
   `integrable_riemannPhi_mul_pow` proves every polynomial moment integrable.
   A second explicit summable majorant now controls the theta summands and
   their first two derivatives on every bounded positive interval.
   `riemannPhi_eq_thetaTail_shifted_second_deriv` is the resulting summed
   identity. `evenKernel_zero_sub_one_eq_two_mul_thetaSeries` and
   `riemannThetaTail_eq_f_modif` identify that tail with pinned Mathlib's
   `WeakFEPair.f_modif (exp(2u))`; both open pieces and the deliberately zero
   boundary value at `x = 1` are formalized. The finite Mellin substitution
   `x = exp(2u)`, lower-half modular reflection, symmetric truncation, and exact
   complex cosh representation are proved. An `AECover` argument passes these
   truncations to the pinned improper Mellin integral.
   A separate summable estimate proves
   exponential decay of the tail and its first two derivatives. The modular
   theta equation gives the exact reflection formula and forces
   `thetaTail'(0) = -1/4`. The finite and improper cosh-transform double
   integration-by-parts identities are now unconditional on `|r| < 3/2`,
   including weighted integrability and every endpoint term.
   The weighted `riemannPhiMeasure` has a nontrivial exponential-moment strip.
   Mathlib's complex-MGF differentiation theorem, analytic uniqueness, and the
   cosh identity prove every centered derivative moment with exact factor `8`.
   Thus `xiGamma_momentRepresentation`, strict positivity, and coefficient
   nondegeneracy are now unconditional.
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
   degree-two Jensen polynomial hyperbolic. For degree three,
   `cubic_hyperbolic_of_discriminant_nonnegative` proves the standard
   nonnegative-discriminant criterion, specialized by
   `jensenPolynomial_degree_three_hyperbolic`. These statements do not imply
   RH.
7. **Differentiation identity** — O’Sullivan equation (3.1) specializes to

   `derivative (J(a,d+1,n)) = C(d+1) * J(a,d,n+1)`.

   This identity is proved coefficientwise in Lean. Pinned Mathlib's
   `Polynomial.rootSet_derivative_subset_convexHull_rootSet` (Gauss–Lucas) now
   proves `hyperbolic_derivative` whenever the derivative is nonzero. Lean also
   proves that unconditional derivative closure is false for the current
   predicate: `1` is hyperbolic while its zero derivative is not.
   Consequently, under the exact `NonzeroJensenFamily` hypothesis,
   all-degree/all-shift hyperbolicity is equivalent to hyperbolicity of
   `J(a,d,0)` for every degree `d`. `PointwiseNonzero a` is a proved sufficient
   condition. The sharper theorem
   `nonzeroJensenFamily_iff_windowNonzero` identifies the exact coefficient
   condition: every positive-length coefficient window contains a nonzero
   entry.  This is now sharpened by
   `jensenWindowNonzero_iff_noAdjacentZeros`: the exact condition is simply
   that no two consecutive coefficients vanish. Isolated zeros are allowed.
8. **Nested finite obligations** — `JensenSquare a k` asks for hyperbolicity
   only when positive degree and shift are both at most `k`. Lean proves

   `AllJensenHyperbolic a ↔ ∀ k, JensenSquare a k`

   and proves these squares are nested under decreasing `k`. This is a genuine
   finite-obligation reduction, not a decision procedure.
9. **False degree propagation rejected** — `turanOnlySequence` is strictly
   positive and log-concave, so all its quadratic Jensen polynomials are
   hyperbolic, but Lean exhibits an explicit nonreal root of its unshifted
   cubic. Thus ordinary Turán inequalities cannot seed induction in degree.
10. **Classical analytic bridge** — `xiJensenGeneratingFunction` is the
   ordinary exponential generating function in the squared variable, and
   Lean now proves its series converges everywhere,
   `xiJensenGeneratingFunction(z²) = xiJensenEntire(z)`, its formal power
   series has infinite radius, and the function is entire. Its genuine power
   series partial sums converge locally uniformly on all of `ℂ`.
   `LocallyUniformHyperbolicLimit` now gives a literal local-uniform limit of
   real hyperbolic polynomials on `ℂ`. Lean proves such a limit is entire via
   Mathlib's Weierstrass theorem. `AnalyticBridgeObligations` factors the
   remaining theorem into Jensen-to-limit and limit-to-RH equivalences and
   formally composes them into the original `BridgeObligations`.
   The source says the xi generating function
   is in the Laguerre–Pólya class iff all its Jensen polynomials are
   hyperbolic, and this is equivalent to RH. No theorem asserts those source
   bridges.

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
- `Polynomial.rootSet_derivative_subset_convexHull_rootSet` (Gauss–Lucas),
  used to prove nonzero-derivative preservation for the route's exact
  `Hyperbolic` predicate;
- `deriv_conj_conj`, used to prove that conjugation symmetry forces all
  iterated derivatives at zero to be real;
- `zeta_eq_tsum_one_div_nat_add_one_cpow`, `Complex.conj_tsum`,
  `Complex.cpow_conj`, `Complex.Gamma_conj`, and
  `AnalyticOnNhd.eq_of_eventuallyEq`, combined to prove global
  `completedRiemannZeta₀_conj`;
- `TendstoLocallyUniformlyOn`,
  `TendstoLocallyUniformlyOn.differentiableOn`,
  `TendstoLocallyUniformlyOn.deriv`, and power-series partial-sum local
  uniform convergence.
- `AnalyticOnNhd.eqOn_zero_of_preconnected_of_eventuallyEq_zero`, used to
  propagate Hurwitz's half-plane identically-zero alternative to all of `ℂ`;
  consequently one global nonzero witness suffices for zero reality.
- `WeakFEPair.Λ₀`, `WeakFEPair.f_modif`, and `mellin`, which unfold pinned
  `completedRiemannZeta₀` to its exact modified Jacobi-theta Mellin integral;
- finite matrices, submatrices and determinants, used for the project-local
  all-minor Toeplitz/Pólya-frequency predicate.

Not found in the pinned library:

- a Riemann xi declaration or theorem identifying its zeros with nontrivial
  zeta zeros;
- a polynomial `IsRealRooted`/hyperbolicity predicate;
- a Laguerre–Pólya class definition;
- Jensen polynomials or the Pólya–Jensen criterion;
- a compact-uniform closure theorem specialized to real-rooted polynomials;
- the integration-by-parts/change-of-variables theorem identifying the pinned
  modified-theta Mellin integral with moments of the now-formalized explicit
  positive Riemann `Φ` kernel (moment integrability is now proved);
- the finite Aissen–Schoenberg–Whitney theorem beyond degree one (degree one,
  including nonpositive root location, is now proved project-locally);
- a prepackaged conjugation theorem for `completedRiemannZeta₀` (the route now
  derives it from the declarations listed above).

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

The completed-zeta conjugation, coefficient-reality, and sourced even Taylor
expansion gaps are now closed. The unconditional formal reduction gives the
nested family:

`forall k, JensenSquare xiGamma k`.

Under the exact `XiCoefficientNondegeneracy`, equivalently no two consecutive
`xiGamma` coefficients vanishing, the same target reduces further to
`XiUnshiftedHyperbolicity`:

`forall d >= 1, Hyperbolic (jensenPolynomial xiGamma d 0)`.

`XiGammaMomentRepresentation` is now proved. The lower Mellin half is reflected,
the symmetric finite identity is passed to the improper integral, and the
resulting xi/Phi transform is differentiated to every order through a weighted
measure whose exponential-moment strip contains zero. Analytic uniqueness
identifies this transform with `xiJensenEntire`, yielding exact factor `8`.
Consequently `xiGamma` is strictly positive, coefficient nondegeneracy is
unconditional, and all shifts reduce to `XiUnshiftedHyperbolicity`. The
all-degree unshifted family—not Mellin analysis or differentiation—is now the
exact coefficient-side mathematical blocker. Strict positivity
and log-concavity settle only degree two. The Lean theorem
`turanOnlySequence_degree_three_not_hyperbolic` proves this failure
constructively, even while every degree-two member is hyperbolic.

For degree-two ASW, `quadraticToeplitzShiftMinor` is the actual unbounded
family of contiguous Toeplitz continuants supplied by total nonnegativity.
The determinants are now computed formally in orders zero through three:
`D₀=1`, `D₁=b₁`, `D₂=b₁²-b₀b₂`, and under quadratic support
`D₃=b₁³-2b₀b₁b₂`.
`toeplitzContinuantMatrix` is the normalized tridiagonal matrix with diagonal
`b₁`, superdiagonal `b₀`, and subdiagonal `b₂`. Repeated Laplace expansion now
proves its determinant recurrence at every order and an entrywise theorem
identifies it with the shifted Toeplitz minor, including the sign
`D_{k+2}=b₁D_{k+1}-b₀b₂D_k`. The oscillation theorem is also proved: if every
`D_k` is nonnegative and `b₀b₂>0`, positivity of consecutive determinants gives
ratios `r_{k+1}=b₁-b₀b₂/r_k`; a uniform descent under
`b₁²<4b₀b₂` contradicts Archimedean boundedness. Thus the sharp factor-four
bound and degree-two finite ASW from all minors are unconditional.

Pinned Mathlib has no packaged Hurwitz/Rouché theorem, so the project now
proves one from Jensen's circle-average formula.
`exists_zero_closedBall_of_uniform_close` is the quantitative disk theorem,
`hurwitz_nonvanishing_or_eq_zero_on` lifts it to every open preconnected
domain, and `hurwitzHalfPlaneNonvanishingClosure` discharges the former
interface. Thus
`zero_reality_of_locallyUniformHyperbolicLimit_unconditional` needs only one
global nonzero witness; boundary zeros remain allowed and become precisely
the real-axis conclusion.

## Total positivity / PF audit

The Aissen–Schoenberg–Whitney theorem says that a finite nonnegative sequence
`b₀,…,b_d` is a Pólya-frequency sequence—its semi-infinite Toeplitz matrix
`(b_{i-j})` has every minor nonnegative—iff
`Σ b_j X^j` has only real zeros (necessarily nonpositive). Applied with
`b_j = choose(d,j) * xiGamma(j)`, all-order Toeplitz-minor nonnegativity would
be a valid sufficient criterion for each unshifted Jensen polynomial.

Lean now defines `toeplitzMatrix`, `ToeplitzTotallyNonnegative`,
`JensenPolyaFrequency`, and `AllJensenPolyaFrequency`. It proves from actual
one-by-one and two-by-two minors that PF implies coefficient nonnegativity and
ordinary log-concavity. `FinitePolyaFrequencyHyperbolicity` records the exact
missing forward finite ASW theorem, while
`FiniteAissenSchoenbergWhitney` states the exact iff with finite support,
nonzero polynomial, endpoint zeros, and internal zeros handled without hidden
positivity assumptions. Its forward direction implies the former interface.
The theorem
`allJensenHyperbolic_of_polyaFrequency` proves the general Jensen implication
from ASW, all Jensen-row PF conditions, and exact nondegeneracy.
`finiteASW_degree_one` proves the complete forward ASW conclusion in degree
one, including that its root is nonpositive. `finiteASW_degree_two` now proves
the complete nondegenerate degree-two conclusion directly from all minors.
`QuadraticASWCertificate` records the exact sharp factor-four discriminant
condition, and the all-order continuant argument proves that PF supplies it.
Ordinary two-by-two log-concavity alone still supplies only factor one.
`selectedMinorCounterexample` proves that coefficient signs, the contiguous
order-two minor, and the leading order-three Toeplitz minor still do not imply
the factor-four bound.
`positiveNonPFSequence` is a Lean counterexample showing strict positivity
alone does not imply log-concavity or the PF all-minor condition.
`FinitePolyaFrequencyHyperbolicityThrough k` and
`jensenHyperbolicThrough_of_polyaFrequency` make the growing-degree
requirement explicit; proving these bounded statements for every `k` recovers
all Jensen hyperbolicity. For the actual positive xi coefficients,
`xiJensenHyperbolicThrough_two_of_polyaFrequency` closes all shifts through
degree two from PF alone, and unconditional nondegeneracy removes the separate
`NonzeroJensenFamily` assumption from the all-degree PF reduction.

For degree three, `finiteCoefficientPolynomial_degree_three` and
`finiteASW_degree_three_of_discriminant` close every step after the cubic
discriminant inequality, including nonpositive root location. The exact new
blocker is `CubicToeplitzDiscriminantPrinciple`: complete Toeplitz total
nonnegativity must force the full five-term cubic discriminant. A single
Hessenberg continuant family does not encode all minors, so the quadratic ratio
argument cannot be extrapolated without a variation-diminishing or oscillation
matrix theorem.

This does not currently shorten the RH route. It replaces each all-degree
hyperbolicity obligation by all minors of a degree-dependent Toeplitz matrix;
the order-two minors recover log-concavity/Turán, while the cubic
counterexample shows that order two alone is insufficient. No theorem in the
pinned Mathlib snapshot packages the Aissen–Schoenberg–Whitney equivalence.
The project now also proves that locally uniform limits of real hyperbolic
polynomials remain entire and conjugation-symmetric. Although root-location
closure is absent from pinned Mathlib, the project-local Jensen-formula proof
establishes the required half-plane Hurwitz dichotomy. Every zero of a nonzero
locally uniform hyperbolic limit is therefore real.

The Pólya--Jensen forward direction is now reduced to one explicit convergence
statement. `xiRescaledJensenPolynomial d` is
`J^{d+1,0}(X/(d+1))`; scaling is proved to preserve hyperbolicity, and
`XiJensenScalingConvergence` is exactly its locally uniform convergence to the
xi exponential generating function. This statement plus all Jensen
hyperbolicity constructs `XiLaguerrePolyaMembership` directly. The project now
proves the exact coefficient limit
`choose (d+1) j * (d+1)^(-j) -> 1/j!`, the sharp uniform coefficient bound by
`1/j!`, the finite-sum evaluation formula, and pointwise convergence of the
whole triangular array by Tannery dominated convergence. The compact-uniform
upgrade is now also proved: on each compact set the terms are interpreted in
the Banach space of continuous maps, bounded by
`xiGamma j / j! * ‖Z‖^j`, and Tannery's theorem gives uniform convergence.
Thus `XiJensenScalingConvergence` is unconditional.

The final change of variables is also explicit:
`xiJensenGeneratingFunction ((s - 1/2)^2) = 8 * riemannXi s`. Thus the
remaining LP-to-RH work is no longer an unspecified normalization issue; it is
now formalized. Gamma has no zero away from the excluded trivial-zero
parameters, the completed-zeta numerator gives a zero of `riemannXi`, and
positivity of the xi generating series on the nonnegative real axis removes
the spurious real-centered branch. Consequently
`riemannHypothesis_of_allJensenHyperbolic` proves the full reverse implication
`AllJensenHyperbolic xiGamma → KakeyaRiemannHypothesisRoot`.

The centered zero statement itself is now proved equivalent to Mathlib's RH:
`XiCenteredCriticalZeroReality ↔ KakeyaRiemannHypothesisRoot`. The remaining
Jensen/RH direction is isolated as `XiRealZeroToJensenHyperbolicity`, the
classical Hadamard-product/Pólya--Jensen theorem that critical-axis centered-xi
zeros imply hyperbolicity of every finite Jensen polynomial. It is not a
consequence of known eventual hyperbolicity, and pinned Mathlib has no
Laguerre--Pólya canonical-product theorem supplying it.

On the coefficient side,
`riemannHypothesis_of_finiteASW_polyaFrequency` now chains the exact finite ASW
interface and all-window xi PF condition directly to RH. Neither premise is
assumed: cubic complete-minor ASW and actual xi total positivity remain open.

The canonical-product converse is now split into two source-exact obligations.
`PositiveGenusZeroProductRepresentation` is the paired product
`c ∏(1+X/rho_n)` in the squared centered variable, with positive `rho_n` and
local-uniform convergence. Lean proves every finite product hyperbolic and
therefore proves this representation implies `XiLaguerrePolyaMembership`.
`XiCriticalZerosToPositiveGenusZeroProduct` is the remaining Hadamard
identification; `XiLaguerrePolyaToAllJensen` is precisely O'Sullivan's
Pólya--Jensen criterion in the reverse direction. Together they prove the
exact all-Jensen/RH equivalence.

The Li branch's canonical-product work conceptually supplies normal convergence
and exact divisor matching for a multiplicity-indexed genus-one product.
It does not yet supply, in this worktree, the required centered even pairing,
sub-order-one rigidity in the squared variable, or the formal Pólya--Jensen
coefficient theorem. No declarations were copied across worktrees.

The Li-to-Jensen pairing algebra is now formal rather than only conceptual:
`centeredGenusOnePrimaryFactor_pair` proves
`E₁(z/(iγ)) E₁(z/(-iγ)) = 1 + z²/γ²`, including exact cancellation of the
genus-one exponentials. Moreover,
`hasProdLocallyUniformlyOn_positiveGenusZeroFactors` proves normal convergence
of `∏(1+X/rho_n)` from `Σ |rho_n⁻¹| < ∞`; this is exactly the inverse-square
summability of the unpaired centered zeros after `rho_n = γ_n²`.
Consequently, the Hadamard blocker is narrowed to enumerating/pairing every xi
zero with multiplicity and proving the normalized product equals the centered
xi function. Reverse Pólya--Jensen remains independently open.

The normalization/convergence plumbing is now closed once that identity is
known. `XiNormalizedGenusZeroProductIdentification` records the exact formula
with constant `xiGamma 0`, positive paired squared ordinates, and summable
inverse ordinates. Lean proves its partial products converge locally uniformly
as the explicit real hyperbolic polynomials
`positiveGenusZeroPolynomial`, hence the identity implies concrete
`XiLaguerrePolyaMembership`. Thus the Hadamard side no longer includes an
implicit convergence or normalization step: only the global divisor-to-function
equality remains. Exact RH equivalence still additionally requires
`XiLaguerrePolyaToAllJensen`.

The affine-factor normalization at the end of that global equality is now
formalized too. `exp_affine_prefactor_eq_const_of_even` proves that if quotient
rigidity supplies `f(z)=exp(a+bz)P(z)`, with `f` and the paired product `P`
entire/even and `P(0)≠0`, then `b=0`.
`XiCenteredAffinePairedProductIdentification` records exactly this raw
Hadamard output, and
`normalizedProductIdentification_of_centeredAffine` fixes `exp(a)` by the
center value and transfers from `z²` to every complex generating variable.
Therefore the genuine Hadamard blocker is now before normalization: construct
the multiplicity-correct paired divisor and prove that its cancelled quotient
is exponential-affine from order-one growth. The reverse Pólya--Jensen theorem
and general finite ASW remain independently open.

The centered divisor carrier is now concrete and multiplicity-correct.
`CenteredXiZeroIndex` repeats every zero by `analyticOrderNatAt`;
`centeredXiZeroDivisor_apply`, `exists_centeredXiZeroRoot_iff`, bounded-window
finiteness, and `centeredXiZeroRoot_escape` prove exact enumeration behavior.
Negation preserves multiplicity, and under `XiCenteredCriticalZeroReality` the
positive-imaginary subtype selects one representative from each `±` pair.
Assuming the isolated inverse-square summability input, Lean transfers it to
summability of inverse squared ordinates and proves locally uniform convergence
of the canonical paired genus-zero product. The remaining analytic blockers
are now exact: prove inverse-square summability from xi growth, prove this
product has `centeredXiZeroDivisor`, and prove the cancelled quotient is
exponential-affine. Reverse Pólya--Jensen/general ASW remain open.

The inverse-square step is now reduced to the correct order-one quantitative
statement rather than an unjustified unit-shell estimate. Canonical dyadic
shells partition all multiplicity indices outside a finite window, and
`centeredXi_inverseSquareSummability_of_orderOneZeroCount` proves that an
`O(2^N(N+1))` dyadic count implies inverse-square summability. This is the
precise consequence expected from Jensen/Riemann--von Mangoldt growth; pinned
Mathlib still lacks that quantitative xi zero-count theorem. For the resulting
product, `centeredXiPairedProduct_eq_zero_iff` excludes accidental infinite
product zeros, `centeredXiPairedProduct_zero_set_eq` matches centered xi's zero
set under RH, and `centeredXiPairedProduct_divisor_support_eq` matches divisor
supports. Exact divisor equality still requires proving local multiplicities
of the infinite product; quotient rigidity and reverse Pólya--Jensen remain
open.

The local multiplicity blocker is now closed. `centeredXiPairedFiber` records
the positive representative whose quadratic factor vanishes at either `z` or
`-z`; under critical-axis zero reality its cardinality is exactly
`centeredXiZeroMultiplicity z`. Isolating this finite fiber from the normally
convergent product proves
`analyticOrderAt_centeredXiPairedProduct`, hence
`centeredXiPairedProduct_divisor_eq` gives full divisor equality, not merely
support equality. On the counting side, the source-standard cumulative
`O(r log(2r+2))` multiplicity bound is formalized as
`XiCenteredCumulativeZeroCount` and proved sufficient for dyadic counting and
inverse-square summability. Pinned Mathlib still provides no quantitative
argument-principle/Riemann--von Mangoldt theorem establishing this premise.
After divisor equality, the remaining product-identification input is exactly
`XiCenteredDivisorQuotientRigidity`: the minimum-modulus/subquadratic-growth
step forcing the cancelled quotient to be exponential-affine. Reverse
Pólya--Jensen and general ASW remain independent blockers.

The quotient-cancellation and algebraic rigidity layers are now formalized.
`centeredXiCancelledQuotient` is the global meromorphic normal-form extension
of xi divided by the paired product. Exact divisor equality proves that it is
entire and zero-free, and `exists_centeredXiCancelledQuotient_analyticLog`
constructs a whole-plane analytic logarithm. A Cauchy estimate proves that
subquadratic norm growth of this logarithm forces it to be affine;
`centeredXiPairedProductAffineRigidity_of_logGrowth` then recovers the global
xi product identity by analytic continuation, and centered evenness fixes its
normalization exactly. Thus quotient rigidity is reduced to the explicit
`XiCenteredCancelledQuotientLogGrowth` estimate. Establishing that estimate
from order-one xi growth still requires the missing Cartan/minimum-modulus
argument. The actual cumulative RvM premise and reverse Pólya--Jensen/ASW
remain open.

The Cartan-to-rigidity pipeline now uses the natural quantity produced by
minimum-modulus estimates rather than assuming a norm estimate for a chosen
logarithm. `CartanExceptionalDiskLogBoundJensen` records finitely many bad
disks of total diameter smaller than the radial annulus. A measure argument
selects a whole centered circle avoiding them; maximum modulus fills the
enclosed disk; and Borel--Carathéodory turns the resulting
`o(r²)` bound on `log ‖quotient‖` into an `o(r²)` bound strong enough to force
the global analytic logarithm affine. Consequently
`centeredXi_eq_normalizedPairedProduct_of_count_and_cartan` closes the
normalized xi product from exactly two remaining quantitative premises:
`XiCenteredCumulativeZeroCount` and
`XiCenteredCancelledQuotientCartanDiskGrowth`. The geometric
exceptional-disk/selected-circle/maximum-modulus/Borel--Carathéodory chain is
proved; deriving the exceptional-disk estimate for this specific infinite
paired product and proving the actual Riemann--von Mangoldt count remain open.

The xi-specific paired-product minimum-modulus layer is now substantially
formalized. `centeredXiPositiveZeroWindow` gives the finite-head/tail split.
For tail factors with norm at most `1/2`, the complex logarithm estimate
`‖log(1+w)‖ <= 3‖w‖/2` gives an explicit infinite-product lower bound by the
tail inverse-square sum. For the finite head, radial exceptional intervals of
equal total width select one circle on which every quadratic paired factor is
bounded below. `exists_circle_norm_centeredXiPairedProduct_lower` combines
both pieces into a full selected-circle lower bound, and its cumulative-count
corollary instantiates inverse-square convergence automatically.
`norm_centeredXiCancelledQuotient_le_div_of_product_lower` transfers any such
positive product lower bound to the cancelled quotient.

The normalized asymptotic bookkeeping is now formalized. Positive-zero
windows are cofinal, so inverse-square summability makes the complementary
tail tend to zero without a hidden local count. The finite-head root and
cardinality-log losses are bounded by
`O((j+1)^2/2^j)`, while xi's order-one contribution has scale
`O((j+1)/2^j)`. `CenteredXiDyadicNormalizedLoss` packages these terms and
`tendsto_centeredXiPairedProduct_normalizedLoss` proves that their sum tends
to zero. The cumulative multiplicity count also gives the explicit
`O(2^j(j+1))` positive-window cardinality bound.

The dyadic-to-cofinal-circle conversion and the entire
selected-circle/log-norm/affine/normalized-product chain are now instantiated.
`exists_circle_norm_centeredXiCancelledQuotient_le_explicit` combines global
xi growth with the selected-circle product lower bound, displaying separately
the finite-head root log, cardinality log, and inverse-square tail.
`CenteredXiDyadicShiftedLoss` packages the resulting shifted truncation scale
and tends to zero. The empty and nonempty positive-divisor cases are handled
separately, yielding unconditional
`centeredXiCancelledQuotient_dyadicBoundary_actual` under the RH zero-reality
premise.

The actual centered-xi cumulative multiplicity count is now closed.
The clean Li-route Gamma/Mellin-strip machinery has been integrated as a
dependency and gives the global order-one bound for `riemannXiLi`;
`xiJensenEntire_eq_four_mul_riemannXiLi` transfers it through the centered
shift. A centered finite-window divisor is dominated by the full analytic
divisor, and Jensen's circle-average formula then proves
`xiCenteredCumulativeZeroCount_actual`. Consequently inverse-square
summability, dyadic positive-window cardinality, and the previously proved
paired-product selected-circle lower bound are all unconditional (the latter
still assumes RH only for the zero pairing). Combining these results proves
`centeredXi_eq_normalizedPairedProduct_actual`: under RH zero reality,
centered xi equals its exactly normalized multiplicity-correct paired product.
The remaining exact-equivalence blocker is now the independent reverse
Pólya--Jensen step converting this concrete LP/product approximation into
hyperbolicity of every shifted Jensen polynomial; general finite ASW remains
an alternative open route.

The reverse-limit closure has now been formalized without assuming the
finite real-rootedness theorem. Finite coefficient convergence gives locally
uniform convergence of the corresponding Jensen rows, the project-local
Hurwitz theorem preserves hyperbolicity in the nonzero limit, and the existing
derivative identity propagates unshifted hyperbolicity to every shift.
`JensenCoefficientwiseHyperbolicApproximation` is the resulting sharp premise,
and `allJensenHyperbolic_of_coefficientwiseApproximation` proves its
all-degree/all-shift conclusion. Repeated derivatives are also proved to
preserve locally uniform convergence, providing the required coefficient
extraction mechanism. For positive genus-zero approximants the remaining
finite statement is isolated as
`FinitePositiveGenusZeroJensenHyperbolicity`: the weighted polynomials with
coefficients `binom(d,j) * j! * [X^j] P_N` must be real-rooted. This is exactly
the missing Schur--Szegő/weighted-matching theorem (or a consequence of full
finite ASW), rather than a further analytic limit or scaling issue.

The finite normalization has now been formalized exactly.  In binomial
coordinates `schurSzegoComposition N p q` has coefficient
`p_j q_j / binom(N,j)`, and the required second factor is
`schurSzegoJensenKernel N d`, with coefficient
`binom(N,j) * d.descFactorial j`.  Lean proves, without side conditions,
that composing this kernel with a finite positive genus-zero product is
definitionally the current Jensen polynomial.  It also proves that the
standard nonpositive-root Schur--Szegő preservation theorem together with
nonpositive-rootedness of these reversed generalized-Laguerre kernels implies
`FinitePositiveGenusZeroJensenHyperbolicity`, including `N=0`, `d>N`,
`N>d`, repeated roots, and coefficient truncation.  Neither classical root
preservation theorem is currently available in Mathlib; they remain the
genuine finite blocker, so the conditional RH equivalence cannot yet be made
unconditional.

The endpoint audit exposed and fixed one necessary qualification.  The strict
claim that two nonzero nonpositive-rooted inputs always have a nonpositive-
rooted composition is false: at binomial degree one, `X` and `1` have disjoint
coefficient support and compose to the zero polynomial.  Lean now proves this
counterexample and states the exact theorem with the classical alternative
“the composition is zero or has only real nonpositive roots.”  The
positive-genus-zero specialization has nonzero constant coefficient `c`, so
the zero branch is formally excluded there.  Kernel rootedness is also closed
for `N=0`, `d=0`, and all `d=1`; the all-degree generalized-Laguerre
oscillation theorem remains open.

Source metadata and exact locations are in
`docs/research/rh-jensen-source-cards.json`.
