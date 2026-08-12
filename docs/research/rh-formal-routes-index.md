# Formal RH route index

This integration is a common, non-runtime base for three independent formal
research routes. It remains stacked on PR #236; its base branch was fetched at
`dc44e199e45f2abea1e7b4f5f7c168e0309b4a1a` during third-stage integration.
The route history forked from the earlier base head
`d4042140f10e0fd231bdfbcf814d068487b15f82`. None of the routes proves the
Riemann Hypothesis, and no finite calculation is promoted to an all-degree,
all-index, or all-test theorem.

## Integrated source snapshots

The first-stage source worktrees were uncommitted on the earlier
`d4042140f10e0fd231bdfbcf814d068487b15f82` base. The bundle hash is SHA-256
over the sorted intentional route paths, with each relative path and file
content separated by a NUL byte.

| Route | Source branch | Bundle SHA-256 |
| --- | --- | --- |
| Jensen / Laguerre--Pólya | `agent/rh-jensen-laguerre-polya-0730` | `cb9fa16ba89397fdbc2e269bceb7b6fe6559b871ffa36118a9a307ad316fded3` |
| Li criterion | `agent/li-criterion-route-0730` | `ef486b9b9dd69ff3e330bb08cb75a46c16dad1ce5c8686a71f1c3283ac6b7528` |
| Weil positivity | `agent/weil-positivity-rh-0730` | `5ad75565ea9b4f4a25b932ae9f1efadd5283a2693b73e1ce62352f97c7ac21c9` |

The second-stage route commits were then cherry-picked without conflicts:

| Route | Source branch | Source commit |
| --- | --- | --- |
| Jensen / Laguerre--Pólya | `agent/rh-jensen-laguerre-polya-0730-v2` | `b82952e87b05d8e0b77ed0124ca9b9503bd48a7c` |
| Li criterion | `agent/li-zero-multiset-0730` | `9220843f7961b97915dcb86392fe88944bb004a2` |
| Weil positivity | `AgentMemory/rh-weil-approximants-0730` | `d91b02f611b62560b83fb80c79fc3fa005d07db3` |

The third-stage route commits were also cherry-picked without conflicts:

| Route | Source branch | Source commit |
| --- | --- | --- |
| Jensen / Laguerre--Pólya | `agent/rh-jensen-lp-stage3-0731` | `cd47e5727cc4aa1faafdf8bc7c4667212045f20f` |
| Li criterion | `AgentMemory/rh-formal-routes-stage3-0731` | `f8db4fee054ba71e3f106500a7ffd8c69b39eda1` |
| Weil positivity | `agent/weil-positivity-stage3-0731` | `542ce60a773548ee38a62dca119727ca2130b601` |

Exact bibliographic citations, normalizations, source locations, and pinned
Mathlib revision are preserved in:

- `docs/research/rh-jensen-source-cards.json`
- `docs/source-cards/li-criterion.md`
- `docs/weil-positivity-sources.yaml`

## Proven finite support and open bridges

### Jensen / Laguerre--Pólya

Proved: the project xi normalization is entire and symmetric; the centered
generating function is entire and even; global completed-zeta conjugation from
the pinned Dirichlet-series/Gamma/cpow declarations and analytic uniqueness;
real normalized xi coefficients and the sourced even `HasSum` Taylor
expansion; exact
degree-zero through degree-three Jensen formulas; the coefficientwise
derivative identity relating degree `d + 1`, shift `n` to degree `d`, shift
`n + 1`; the equivalence between all Jensen obligations and all nested finite
`JensenSquare` obligations; monotonicity of those squares; the conditional
reduction from all shifts to the unshifted family; Gauss--Lucas derivative
closure whenever the derivative is nonzero; pointwise coefficient
nonvanishing as a sufficient condition for every Jensen polynomial to be
nonzero; exact window nondegeneracy and its equivalence to no adjacent zero
coefficients; the pinned modified-theta Mellin representation; the explicit
Riemann `Phi` summand formula, strict positivity, pointwise summability,
continuity, explicit exponential decay, kernel symmetry/positivity, and
integrability of every polynomial moment; a normalized concrete-`Phi` moment interface
proving strict xi coefficient positivity and nondegeneracy; project-local
termwise theta identity expressing each positive-half-line `Phi` summand as a
shifted second derivative, an explicit common summable majorant through two
derivatives, the summed identity, and its exact identification with
`WeakFEPair.f_modif (exp(2u))` including the two pieces and zero boundary; the
finite Mellin substitution `x = exp(2u)`, lower-half modular transform,
symmetric complex-cosh formula, improper truncation limit, and exact
completed-zeta cosh representation; explicit decay of the theta tail and its first two derivatives,
the modular reflection formula and exact derivative `-1/4` at zero, and
finite/improper cosh-transform double integration by parts on `|r| < 3/2`;
all-minor Toeplitz/PF definitions, their
coefficient-nonnegativity and log-concavity consequences, and the exact
finite-ASW-to-all-Jensen reduction; finite ASW through nondegenerate degree two
with nonpositive root location, an explicit selected-minor insufficiency
counterexample, an all-order tridiagonal determinant recurrence with exact
Toeplitz-minor normalization, and a proved ratio-oscillation theorem yielding
the sharp factor-four bound, plus bounded-degree PF/Jensen reductions; an actual
finite-ASW iff interface with explicit support/nonzero/zero cases, and a
degree-three reduction from all minors to the exact cubic discriminant
principle; an actual
locally-uniform hyperbolic-limit interface whose limits are proved entire and
conjugation-symmetric; a project-local Jensen-formula proof of quantitative
disk zero preservation, the Hurwitz dichotomy on every open preconnected
domain, and the required half-plane closure; unconditional zero reality for
every nonzero locally uniform hyperbolic limit; the
ordinary xi generating function's everywhere-convergent series, squared
relation to centered xi, infinite radius, entirety, and locally uniform Taylor
approximation; degree-one hyperbolicity; the
nondegenerate degree-two Turán/root, hyperbolicity, and positive-log-concave
lemmas; and the nondegenerate cubic discriminant criterion. The unconditional
derivative closure statement is refuted by the constant-polynomial case, and
an explicit positive log-concave sequence refutes propagation from all
quadratics to the cubic.
The explicit rescaled Jensen sequence is proved hyperbolic whenever all Jensen
polynomials are, and its locally uniform convergence is isolated as the exact
remaining Pólya--Jensen approximation theorem.

The exact `xiGamma` moment representation is now proved through a weighted
Phi measure, complex-MGF differentiation, and analytic uniqueness. Coefficient
positivity/nondegeneracy and the reduction from all shifts to the unshifted
family are unconditional.

Open: the cubic all-minor discriminant principle and finite ASW in arbitrary
degree; the isolated real-zero-to-Jensen-hyperbolicity canonical-product
theorem; actual xi-window total positivity; and the all-degree
unshifted family (equivalently, under nondegeneracy,
`AllJensenHyperbolic xiGamma`). Known eventual hyperbolicity, bounded-degree
results, and ordinary Turán inequalities do not prove this all-degree target.

The reverse bridge is closed: compact-uniform convergence of the rescaled
Jensen triangular array, Laguerre--Pólya/Hurwitz zero transfer, exact
completed-zeta factor handling, and positivity on the nonnegative generating
axis now prove `AllJensenHyperbolic xiGamma → RiemannHypothesis`.
Conversely, RH is now proved equivalent to the exact centered-xi
critical-axis zero statement; only the classical canonical-product step from
that zero statement to every Jensen polynomial remains.
That step is now factored into a positive genus-zero product representation
and the reverse Pólya--Jensen criterion. Finite paired products are formally
hyperbolic and their local-uniform limit gives the concrete LP membership.
The exact genus-one pair cancellation and normal convergence of the resulting
genus-zero product from inverse-square zero summability are also formalized.
An exact normalized product identity now automatically constructs the
project's LP membership; the remaining Hadamard obligation is only the global
divisor-to-function equality.
The final affine quotient ambiguity is also discharged: centered evenness
forces the `exp(a+bz)` factor's linear coefficient to vanish, evaluation at
zero fixes the constant, and complex square-root surjectivity transfers the
centered paired product to the generating variable. What remains is the
multiplicity-correct divisor construction and order-one quotient rigidity,
followed independently by reverse Pólya--Jensen (or general finite ASW).
Centered xi now also has a canonical countable zero index carrying exact
analytic multiplicity, a negation pairing, positive-imaginary orbit
representatives under RH, and a normally convergent paired product conditional
only on inverse-square zero summability. Product divisor equality and
exponential-affine quotient rigidity remain the precise Hadamard blockers.
Canonical dyadic shells now prove inverse-square summability from the exact
order-one `O(2^N(N+1))` multiplicity count expected from Jensen/Riemann--von
Mangoldt; that quantitative count is absent from pinned Mathlib. Under RH the
paired product has exactly the centered-xi zero set and matching divisor
support, so only local multiplicity equality—not extra-zero control—remains
before quotient rigidity.
The finite paired fiber is now isolated from its analytic nonvanishing
complement, proving exact local orders and full paired-product divisor equality
under RH plus inverse-square summability. The cumulative
`O(r log(2r+2))` multiplicity count now implies the dyadic premise directly.
The remaining Hadamard blocker is therefore the actual quantitative xi count
and the minimum-modulus/subquadratic quotient-rigidity theorem, not local
multiplicity bookkeeping.
Exact divisor cancellation now yields an entire zero-free quotient with a
global analytic logarithm. Subquadratic norm growth of that logarithm is proved
to force an affine exponential quotient, after which evenness and the center
value normalize the product exactly. The remaining quotient theorem is the
Cartan/minimum-modulus estimate establishing this explicit subquadratic growth
condition.
The quotient route is now sharpened to the natural Cartan output:
finite exceptional disks of sufficiently small total diameter give a selected
centered circle, maximum modulus gives disk bounds for `log ‖quotient‖`, and
Borel--Carathéodory forces its analytic logarithm affine. Thus the normalized
product follows from the cumulative xi count plus one xi-specific
exceptional-disk estimate. Pinned Mathlib contains neither the quantitative
Riemann--von Mangoldt count nor that infinite-product minimum-modulus estimate.
The paired-product estimate itself is now explicit: a finite positive-zero
window is separated from its inverse-square tail, radial interval avoidance
selects a circle with a finite-head lower bound, and summable logarithms give
the infinite-tail lower bound. Their product bounds the full paired product,
and hence the cancelled quotient. The normalized loss analysis is also closed:
cofinal positive-zero windows make the inverse-square tail vanish, and the
head/cardinality and order-one terms are respectively
`O((j+1)^2/2^j)` and `O((j+1)/2^j)`. A dyadic quotient estimate with this
majorant now implies selected-circle growth, global subquadratic log-norm
growth, affine rigidity, and the normalized product. The remaining analytic
gap is to derive that concrete quotient circle estimate from the global xi
upper bound and the explicit product lower bound.
The actual cumulative centered-xi count is now proved: the Li route's
Gamma/Mellin-strip order-one estimate is transferred to the centered
normalization, and Jensen's circle-average formula bounds the
multiplicity-index window by `O(r log(2r+2))`. This unconditionally supplies
the inverse-square and dyadic cardinality premises used above.
The remaining quotient-circle gap is also closed. The explicit product lower
bound and xi upper bound give a selected-circle quotient estimate whose
head/root-log, cardinality-log, and shifted tail losses are all `o(r²)`.
Dyadic-to-cofinal growth, affine rigidity, and normalization therefore prove
the exact centered-xi paired-product identity under RH zero reality. The
remaining converse is reverse Pólya--Jensen (or general finite ASW), not
Hadamard product identification.
The analytic closure portion of reverse Pólya--Jensen is now proved:
coefficientwise limits of hyperbolic Jensen rows converge locally uniformly,
Hurwitz preserves their real zeros, and nondegeneracy propagates the
unshifted result through all shifts. Iterated-derivative convergence supplies
the coefficient-extraction mechanism. The exact residual finite theorem is
the real-rootedness of the weighted Jensen rows of finite positive genus-zero
products, equivalently the needed Schur--Szegő/weighted-matching step; full
finite ASW would also discharge it.

The exact binomial-coordinate composition and Jensen kernel are now
formalized, including a proof that their composition is precisely the
finite-product Jensen row for every relative choice of the product length and
Jensen degree.  The remaining implication is split cleanly into finite
Schur--Szegő preservation for nonpositive roots and nonpositive-rootedness of
the reversed generalized-Laguerre kernel; these finite root-location theorems,
not normalization or limit passage, are the current blocker.

Boundary handling is now exact: `X` and `1` at ambient binomial degree one
give a formal counterexample to the strict nonzero-output formulation, so the
Schur--Szegő interface now concludes “zero or nonpositive-rooted.”  The
genus-zero application excludes zero using its constant coefficient.
Generalized-Laguerre kernel rootedness is proved for either zero parameter and
for Jensen degree one, leaving the genuine all-degree oscillation result.

### Li criterion

Proved: the Li-normalized xi endpoint values, entirety and functional
equation; elementary coefficient identities; finite-prefix positivity
properties; zero-summand and multiplicity-aware finite-multiset identities;
finite Li-sum nonnegativity for roots on the critical line; a
symmetric-height Cauchy/limit interface with uniqueness and approximation
transfer; exact logarithmic-derivative identities for finite zero products;
the intrinsic `analyticOrderAt`-based zeta-zero order and its zero-set
characterization away from the pole; locally uniform transfer of every
derivative and, on a common nonvanishing neighborhood, logarithmic
derivatives; normalized Taylor-coefficient convergence; an abstract
all-index transfer from finite multiplicity-aware Li sums to the limiting
logarithmic derivative; the truncated generating-polynomial recurrence; and a
formal `finitePrefixSpoof` counterexample showing that any fixed positive
prefix need not imply all-index positivity.

Open: construct the actual nontrivial-zeta-zero multiset with analytic
multiplicity; prove the symmetric-height truncations Cauchy; construct
finite xi/Hadamard approximants; prove their locally uniform convergence and
a common nonvanishing neighborhood after the Li change of variables; discharge
the finite all-order Taylor identity needed by the abstract transfer theorem;
thereby identify the analytic coefficients with the multiplicity-aware zero
sums and prove their reality; then formalize Li's all-`n` nonnegativity
equivalence with Mathlib's RH proposition.
`RiemannLiCriterionStatement` remains an unproved proposition.

### Weil positivity

Proved: the smooth compactly supported test interface, involution laws,
convolution closure, the critical-line Fourier normalization, typed finite
zero/prime sums and exact finite-formula stages, nonnegative finite
critical-line energies, transfer of exact explicit-formula identities and
positivity through pointwise, eventual-pointwise, uniform-on-subclass, and
quantified-error limits; bounded-monotone supremum limits; separately typed
zero, prime-power, archimedean, and pole approximants with an algebraic limit
combination theorem; finite distinct-zeta-zero windows from Mathlib's
discreteness theorem; the von Mangoldt \(-\zeta'/\zeta\) identity on
\(\Re s>1\); a counterexample showing finite positivity without convergence
does not control an independently declared limit; finite Gram
positive-semidefinite certificates and their restrictions; and an explicit
Hermitian kernel with a negative direction.

Open: discharge the named `MellinNormalizationObligation`,
`ZeroRegularizationObligation`, `PrimePowerStabilizationObligation`, and
`ArchimedeanLimitObligation`; combine them in `AnalyticBridgeObligations` to
obtain the actual normalized Guinand--Weil formula; and separately prove both
directions in `AllTestPositivityObligation`. `BridgeObligation` is only the
unproved interface; finite stages, finite PSD certificates, and limit-transfer
lemmas without the actual zeta convergence hypotheses cannot discharge it.
Mathlib's zero set has no multiplicities, and its right-half-plane logarithmic
derivative theorem does not supply symmetric zero regularization or the
archimedean explicit-formula term.

## Research priority

1. Build shared multiplicity-aware nontrivial-zero indexing and
   symmetric-height regularization without asserting a criterion.
2. Discharge one named analytic bridge at a time: Jensen actual-family
   nondegeneracy and all-degree unshifted hyperbolicity, the zeta-specific hypotheses of the
   Li/Hadamard coefficient-transfer theorem, or the four finite-to-limit
   Guinand--Weil obligations.
3. Formalize the corresponding source criterion only behind the existing
   explicit propositions: Pólya--Jensen, Li, or all-test Weil positivity.
4. Keep finite computations as falsifiable support tests only. Do not infer an
   all-degree, all-index, or all-test statement from a finite restriction.

The route-specific status documents contain the detailed next obligations:
`docs/research/rh-jensen-route.md`,
`docs/research/li-coefficients-route.md`, and
`docs/weil-positivity-route.md`.
