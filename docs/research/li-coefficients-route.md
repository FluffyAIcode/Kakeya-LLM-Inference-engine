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
- finite norm-ball windows of nontrivial zeta zeros, repeated according to
  their intrinsic analytic multiplicities;
- finite closed symmetric-height rectangles with exact boundary membership,
  multiplicity-aware indices, and injective height reindexing;
- the symmetric-height convergence obligation;
- the corresponding Cauchy interface and completeness transfer;
- unconditional-summability and shell-majorant criteria that discharge the
  symmetric-height Cauchy/limit interfaces;
- the source-shaped logarithmic shell count and quadratic paired-cancellation
  obligations, plus the weaker `O((N+1)^(-3/2))` shell bound sufficient for
  convergence;
- a finite rational product whose logarithmic derivative generates the
  finite zero contributions;
- the finite canonical product centered at `s = 1` and its exact Möbius
  transform, factor by factor and hence with multiplicity;
- the intrinsic zeta zero order `riemannZetaZeroOrder : ℂ → ℕ∞`;
- locally uniform all-order derivative and logarithmic-derivative transfer;
- normalized logarithmic-derivative Taylor coefficients of finite products;
- an all-index finite-zero-sum transfer theorem with every analytic
  hypothesis stated explicitly;
- truncated generating polynomials;
- `RiemannLiCriterionStatement`, the unproved source bridge to the canonical
  Mathlib RH root.
- the Möbius xi transform, its all-order coefficient-identity obligation,
  and the exact zero-window formula/converse decomposition of Li's criterion.
- the finite spectral converse obligation and reflection-symmetry interface
  that isolate the corresponding critical-line implication.
- the infinite reflection-symmetric spectral reduction with the explicit
  Bombieri--Lagarias multiset summability hypothesis;
- a two-obligation xi endpoint: canonical-product approximation and the
  all-order xi Möbius/derivative identity.
- genus-one primary factors, their quadratic remainder estimate, and normal
  convergence of multiplicity-indexed canonical products from inverse-square
  root summability;
- the Hadamard representation obligation and normalization/functional-
  equation constraints on its affine exponential prefactor.
- exact finite comparison between zero-centered genus-one factors and
  one-centered Li factors under multiplicity-preserving reflection symmetry.
- xi's global Mathlib divisor, finite natural zero multiplicities, exact
  reflection invariance, and compatibility with zeta multiplicity throughout
  the positive-real half-plane.
- Mathlib's finite disk canonical decomposition instantiated for xi.
- compact-local finiteness/boundedness and a radius-cofinal sequence of
  finite-disk xi decompositions.
- Jensen/logarithmic-counting identification for xi's divisor and a global
  entire logarithm for every zero-free entire complex-plane quotient.

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
- away from `s = 1`, every zeta analytic order is finite; this follows by
  propagating finite order from `s = 2` across the connected punctured plane;
- every compact norm ball yields a finite `Finset` of nontrivial zeros, and
  the sigma index
  `Σ rho in window, Fin (riemannZetaZeroMultiplicity rho)` repeats each root
  exactly according to that finite order;
- the bounded windows are monotone and exhaust each fixed nontrivial zero;
- compact symmetric-height rectangles are finite with multiplicity, retain
  boundary zeros, and reindex injectively under increasing height;
- under an explicit real-strip hypothesis, rectangle membership is exactly
  nontrivial-zero membership plus `|im rho| ≤ T`;
- RH itself supplies the real bound `|re rho| ≤ 1`, yielding genuine finite
  pure height windows conditionally on RH;
- RH puts every root in every bounded multiplicity window on the critical
  line and hence makes every finite-window Li sum nonnegative;
- locally uniform limits of holomorphic functions transfer every
  `iteratedDeriv`, by iterating Mathlib's complex Weierstrass theorem;
- on a common open nonvanishing neighborhood, logarithmic derivatives
  converge locally uniformly, and all their normalized Taylor coefficients
  converge;
- every finite product satisfies the all-order Taylor identity
  unconditionally: the normalized `k`th logarithmic-derivative coefficient
  is the finite Li sum at index `k + 1`;
- the finite rational generating product is exactly
  `prod (1 + (s - 1) / rho)` after `s = (1-z)⁻¹`; consequently the transformed
  canonical product has the all-order normalized logarithmic-derivative
  coefficient identity, with repeated roots retained as separate factors;
- therefore normal convergence plus holomorphy and common nonvanishing alone
  transfers multiplicity-aware finite Li sums coefficientwise to the
  logarithmic-derivative Taylor coefficients of the product limit;
- this transfer is also proved directly for finite canonical products after
  the Möbius substitution, at every derivative order;
- exact symmetric-height exhaustions eventually contain every fixed finite
  family, and for finite index types their declared coefficients stabilize
  to the full finite sums;
- exact symmetric-height exhaustions are cofinal in all finite subsets, so
  unconditional summability identifies their limit with the `tsum`;
- a summable majorant for successive height shells proves the required
  Cauchy property, exposing the precise target for zero-count/growth bounds;
- an `O((N+1)^(-3/2))` successive-shell estimate is summable and therefore
  proves every symmetric Li partial-sum sequence Cauchy;
- more sharply, the source-shaped `O(log(N+2)/(N+2)^2)` paired-shell estimate
  is proved summable and directly yields the same Cauchy conclusion;
- the genus-one factor satisfies
  `‖E₁(w)-1‖ ≤ 3‖w‖²` for `‖w‖ ≤ 1`;
- inverse-square summability and root escape imply locally uniform convergence
  of the genus-one canonical product on all of `ℂ`; the
  Bombieri--Lagarias summability weight implies inverse-square summability
  when roots are uniformly separated from zero;
- every exact symmetric-height exhaustion converges locally uniformly to
  this product, preserving multiplicity through its index type;
- if xi has the genus-one Hadamard representation
  `exp(a+b*s) * product E₁(s/rho)`, the affine-normalized symmetric finite
  products converge locally uniformly to xi;
- `xi(0)=1` forces `exp(a)=1`, and the functional equation gives the exact
  reflected-product constraint on `a`, `b`, and the canonical product;
- for a fixed nonzero product germ, two affine Hadamard prefactors have the
  same slope and the same exponential constant; the latter cannot be
  strengthened to `a=a'` without selecting a logarithm branch;
- finite reflection symmetry gives the exact identity
  `prod E₁(s/rho) = C * exp(s * sum rho⁻¹) * centeredProduct(s)`;
- when the affine slope balances the finite inverse-root sum, the transformed
  genus-one Hadamard approximation has every normalized
  logarithmic-derivative coefficient equal to the finite Li sum;
- every xi zero has finite analytic order, the functional equation preserves
  that order, and xi/zeta analytic orders agree in the nontrivial strip;
- on every closed disk, xi admits Mathlib's finite canonical decomposition
  into divisor-prescribed disk factors and a nonvanishing remainder;
- affine constants are unique modulo `2πiℤ`, while their slopes are literally
  unique;
- the global xi divisor meets every compact set in a finite set, and finite
  disk decompositions can be chosen along radii `N+1`;
- equal finite divisors of xi and a candidate product produce, via Mathlib's
  meromorphic normal form, an entire zero-free cancelled quotient;
- normal convergence of the infinite genus-one product now proves that its
  only zeros are the indexed roots and that each finite root fiber contributes
  exactly its cardinality to the analytic order; an order-matching zero
  enumeration therefore gives equality with xi's global divisor;
- xi's locally finite divisor now yields a canonical countable sigma-type
  enumeration with one index per unit of analytic multiplicity; it is complete,
  has exact finite fibers, and escapes every bounded set in the zero-free,
  finite, and infinite cases;
- an `O(log N)` multiplicity-counted shell partition now implies
  inverse-square root summability, with an arbitrary finite set of low zeros
  removed and restored rigorously;
- subquadratic log growth of that quotient now conditionally yields the exact
  whole-plane exponential-affine Hadamard identity;
- `xi(0)=P(0)=1` and `xi(1-s)=xi(s)` then force `exp(a)=1` and the exact
  reflected-product constraint, without choosing an invalid logarithm branch;
- the xi canonical-product approximation and all-order Möbius identity now
  directly imply `RiemannLiZeroWindowFormula`;
- xi's divisor log-count equals the circle average of `log ‖xi‖` exactly,
  since the normalization removes Jensen's trailing-coefficient constant;
- every continuous zero-free whole-plane quotient has a global continuous
  logarithm by Mathlib's simply-connected branch-log theorem;
- when the quotient is entire, its continuous logarithm is proved entire and
  its derivative is the quotient's logarithmic derivative;
- an entire logarithm with zero second derivative is affine, including the
  resulting exponential-affine quotient identity;
- Cauchy's estimate proves zero second derivative from `o(‖z‖²)` norm
  growth; Borel--Carathéodory proves the log-relevant version from
  `sup_{‖z‖<r} re h(z) = o(r²)`;
- `exp(h)=g` gives `re h = log ‖g‖`, reducing the quotient step to the
  exact named obligation `SubquadraticLogNormGrowth g`;
- the xi functional equation promotes any symmetric-majorant bound on
  `re s ≥ 1/2` to a global bound;
- the Dirichlet series gives the actual uniform estimate
  `‖ζ(s)‖ ≤ ∑ n⁻²` on `re s ≥ 2`;
- Euler's integral gives `‖Γ(s)‖ ≤ Γ(re s)` for `re s > 0`, while
  real-Gamma monotonicity and `n! ≤ n^n` give the explicit ceiling-power
  growth bound;
- combining those estimates gives an explicit far-right xi bound on
  `re s ≥ 4`;
- the all-exponent Mellin representation of `completedRiemannZeta₀` is
  uniformly bounded on every fixed vertical strip by its two endpoint
  norm integrals, hence xi has only quadratic growth there;
- the ceiling-power, quadratic-strip, and reflected estimates combine into
  the global theorem `RiemannXiOrderOneGrowthBound`;
- the first Möbius xi coefficient satisfies
  `logDeriv (xi ∘ (1-z)⁻¹) 0 = λ₁`;
- normal convergence and common nonvanishing transfer all finite Li sums to
  xi derivative coefficients once the named all-order Möbius coefficient
  identity is supplied;
- RH implies all-index Li positivity once the named all-index zero-window
  formula is supplied; adding the separately named converse recovers the
  full criterion statement;
- `‖1-rho⁻¹‖ ≤ 1` is equivalent to `re rho ≥ 1/2`; for a finite
  reflection-symmetric multiset, the explicitly named spectral converse
  (aggregate all-index positivity implies every transform lies in the unit
  disk) therefore forces every root onto `re rho = 1/2`;
- the same unit-disk/reflection argument works for arbitrary infinite index
  types; the remaining spectral implication is typed with the source
  summability condition
  `sum (1 + |re rho|) / (1 + |rho|)^2 < infinity`;
- reflection symmetry and positivity of an unrelated coefficient sequence
  admit an explicit two-root off-line counterexample, proving that the
  zero-sum/product identity cannot be dropped;
- canonical xi product approximation plus the general all-order xi
  change-of-variables identity imply convergence to every Li derivative
  coefficient, isolating exactly those two obligations;
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

1. Analytic infrastructure: finite order, bounded multiplicity enumeration,
   and compact height rectangles are closed. A pure vertical window follows
   from the explicit real-strip hypothesis, but pinned Mathlib does not yet
   supply that zeta classification. The remaining global work is to derive a
   summable successive-shell majorant from zero counting/growth and identify
   the resulting normally convergent genus-one product with xi, including its
   affine exponential prefactor and a common nonvanishing neighborhood.
   Generic genus-one normal convergence is now closed. Pinned Mathlib's
   zeta-zero API stops at
   discreteness, compact finiteness, and escape to the cocompact filter; no
   zero-counting asymptotic or strip-growth estimate was found. The all-order
   finite canonical-product Möbius identity is closed. The remaining xi
   coefficient bridge is the general all-order change-of-variables identity
   relating `log xi((1-z)⁻¹)` to the source derivative definition.
   Authoritative Riemann--von Mangoldt estimates give `O(log T)` zeros in a
   unit-height shell. Combined with conjugate/reflection cancellation of the
   first-order term, the desired shell size is `O(log T / T^2)`, which is
   stronger than the formal sufficient `O(T^(-3/2))` interface. The
   quantitative cancellation estimate itself remains zeta-specific. The
   single product-identification theorem still missing is the zeta Hadamard
   factorization `GenusOneHadamardRepresentation` for a complete
   multiplicity-indexed nontrivial-zero family.
2. Mathematical criterion: formalize Li's theorem that all
   `λ_n ≥ 0` for `n ≥ 1` is equivalent to every non-trivial zeta zero lying
   on `re(s)=1/2`, then identify that statement with Mathlib's canonical
   `RiemannHypothesis` (including its explicit exclusions of trivial zeros
   and `s = 1`). The finite formalization isolates its hard spectral step:
   infer the individual unit-disk bounds from aggregate positivity for every
   positive index. Reflection symmetry then gives the critical line
   algebraically. The infinite statement now carries the
   Bombieri--Lagarias summability hypothesis and the symmetric zero-sum limit
   explicitly; proving its spectral implication remains the source theorem,
   not a consequence of abstract positivity alone.

The all-order finite Li identity, normal derivative transfer, and finite
genus-one/centered-factor comparison are closed. The remaining infinite step
is to prove that the Hadamard affine slopes of the symmetric zeta
approximants balance their inverse-root sums in the limit, with enough local
nonvanishing to pass logarithmic derivatives. This is now the precise
prefactor theorem needed before the general xi coefficient identity can be
discharged.

The finite disk decomposition is not a substitute for Hadamard
factorization: its canonical factors depend on the disk radius and equality
is codiscrete within that disk. Pinned Mathlib still has no packaged
finite-order Hadamard theorem and no theorem passing these disk
decompositions to a whole-plane genus-one product.
The elementary zeta Dirichlet-series region is closed by a uniform bound on
`re s ≥ 2`. A complex Gamma estimate is now proved directly from Mathlib's
Euler integral, and an explicit ceiling-power real-Gamma estimate replaces
the absent packaged complex Stirling theorem. Rather than bounding zeta
through its pole inside the critical strip, the route uses Mathlib's entire
Mellin representation to prove that `completedRiemannZeta₀` is uniformly
bounded on every fixed vertical strip. Lean now packages the central strip,
far-right estimate, and functional-equation reflection into
`RiemannXiOrderOneGrowthBound`:
`‖xi(s)‖ ≤ exp(C ‖s‖ log(‖s‖+2))` outside a fixed disk. Thus xi's
source-standard order-at-most-one estimate is closed, and Jensen's formula
can consume it directly. The complex-analysis rigidity step is also closed:
Borel--Carathéodory and Cauchy's estimate turn
`SubquadraticLogNormGrowth` into an exponential-affine quotient. The
remaining obligation is the xi/product quotient-growth estimate itself.
The cancellation mechanism itself is now formalized: equality of the two
finite global divisors makes the normal-form extension of `xi * P⁻¹` entire
and zero-free, and `SubquadraticLogNormGrowth` then proves the exact affine
Hadamard representation by analytic continuation from a nonzero product
germ.
The product-side divisor calculation and xi enumeration are now closed.
Local finiteness supplies root escape automatically, so inverse-square
summability alone gives normal convergence and exact xi/product divisor
matching.
Jensen's formula is now combined with the global xi growth theorem to prove
the multiplicity-aware logarithmic counting bound
`N_log(r) ≤ C r log(r+2)`. This does **not** imply an `O(log N)` unit-shell
count; that stronger local statement needs Riemann--von Mangoldt (or an
equivalent local zero-count theorem). The order-one Jensen bound does give
`O(2^N(N+1))` roots in dyadic shells. The new dyadic comparison theorem
turns exactly that bound into a summable `O((N+1)/2^N)` inverse-square
contribution and immediately closes xi/product divisor matching through
`riemannXiGenusOneCanonicalProduct_divisor_eq_of_dyadicShellCount`.
That final bookkeeping is now closed: the finite index-window divisor is
proved pointwise below xi's analytic divisor, evaluating the weighted count
at radius `2r` gives `card(window r) log 2 ≤ N_log(2r)`, and half-open
dyadic shells partition every index outside a finite low window exactly
once. Consequently `summable_norm_inv_sq_riemannXiZeroRoot`, unconditional
normal convergence, and exact xi/product divisor matching are proved.
Separate upper bounds for xi and the canonical product do not control their
quotient near common zeros: a reciprocal-product estimate necessarily blows
up there, while the analytic quotient remains regular by cancellation. The
genuine remaining Hadamard input is therefore a minimum-modulus/cancelled-
quotient estimate, or an equivalent finite-order Hadamard factorization
theorem, sufficient to prove `SubquadraticLogNormGrowth` globally.
The geometric Cartan step is now formalized with exact constants: finitely
many exceptional disks with `2 * sum radii < b-a` leave a radius in `[a,b]`
whose centered circle avoids every disk. The project packages the weakest
remaining analytic statement as `CartanExceptionalDiskLogBound`: outside
such disks the quotient has the required annular log bound.
A finite minimum-modulus theorem is now also proved constructively: put an
equal disk of radius `H/(3n)` around each of `n` indexed roots. Outside these
disks, a monic root product is at least `(H/(3n))^n`, while a finite genus-one
product has the explicit lower bound obtained by multiplying
`(H/(3n))/‖ρ‖ * exp(-‖z‖/‖ρ‖)`. Combining this with circle selection gives
the same bound on a selected circle, and the theorem is instantiated for
canonical xi multiplicity windows. Far finite tails are quantitatively
controlled by
`exp(3‖z‖² * sum ‖ρ⁻¹‖²)-1`.
The finite-to-infinite tail passage is now proved: xi windows are cofinal,
inverse-square mass outside the window tends to zero, factor deviations are
summable, and the infinite tail obeys the same estimate uniformly on
`‖z‖≤R`. Finite head losses are bounded by
`card(window T) log T`, while Cauchy--Schwarz controls the inverse-root sum
by window cardinality and the global inverse-square mass.
Circle selection
proves `SubquadraticBoundaryLogNormGrowth`, and maximum modulus proves the
global disk estimate. Thus the genuine gap is now only the source-standard
passage from these finite head/tail estimates to a uniform subquadratic
outside-disk bound for the infinite cancelled quotient: specifically, a
quantitative dyadic rate such as `tailMass(R)=O(log R/R)` and a compatible
joint choice of truncation and selected radius. The dyadic series estimate is
now proved with constants:
`sum_{k≥N} shellMass(k) ≤ A(N+1)/2^N`, including xi multiplicities and the
finite low window. The remaining bookkeeping is to identify the radial
complementary-product tail with this shifted shell sum and then coordinate
the selected radius with that truncation. The first part is now exact:
boundary-safe half-open tail shells form a disjoint union of the complement
of the closed radius window, and the corresponding summable series reindexes
with multiplicities preserved. The explicit compatible scales
`R_j=2^j`, `N_j=4j` also satisfy
`R_j² (N_j+1)/2^N_j → 0`. What remains is to compare these boundary-safe
shells to the sourced xi shell bound inside the Cartan circle construction
and preserve its finite-head constants. That comparison is now proved, so
the actual closed-window complement satisfies
`tailMass(2^N) ≤ A(N+1)/2^N`. Tracking the finite-head bound exposes an
important scale correction: `N_j=4j` is unnecessarily aggressive and makes
the available head-count loss grow like `2^(4j)`, far beyond the circle
scale. The compatible choice is `N_j=j`: its tail exponent is
`O(j·2^j)=o(2^(2j))`, while the head root-log loss is
`O(2^j(j+1)^2)=o(2^(2j))`. The latter limit and Cartan circle selection on
`[2^j,2^(j+1)]` with disk budget `2^j` are formalized. Remaining work is the
single combined lower bound that multiplies this finite-head estimate by
the infinite complementary product estimate. The crucial signed/branch
piece is now closed: for `‖w‖≤1/2`,
`exp(-‖w‖²)≤‖E₁(w)‖`, proved through the principal complex logarithm, and
hence for `‖z‖≤T/2` the full complementary product is bounded below by
`exp(-‖z‖² tailMass(T))`. Thus the combined construction should use head
cutoff `T=2^(j+2)` on circles in `[2^j,2^(j+1)]`; this constant shift leaves
all subquadratic rates unchanged. The previous formal blocker was the
finite-head/complementary-tail `tprod` split and its multiplication with
the selected-circle lower bound. That split is now exact: the finite head is
multipliable by finiteness, the complement by summable factor deviations,
and `HasProd.mul_compl` gives the whole product without incorrectly requiring
a `CommGroup ℂ` instance. Multiplying both lower bounds yields an explicit
full-product minimum modulus on a circle in `[2^j,2^(j+1)]` with head cutoff
`2^(j+2)`. The remaining blocker is logarithmic aggregation of its head
cardinality/root-log/inverse-root terms with the xi upper-growth estimate,
then transfer to the cancelled quotient on those circles. The aggregation is
now exact: `riemannXiFiniteHeadLogLoss` records the three signed head terms,
the finite Cartan expression equals its negative exponential, and
`riemannXiFiniteHeadLogLoss_le` bounds it by the Jensen cardinality,
root-log, and Cauchy--Schwarz square-root majorants. Adding the tail loss
gives `riemannXiCanonicalCircleLogLoss`; xi's global order-one bound and the
pointwise normal-form quotient identity now yield a selected-circle bound
`‖q(z)‖ ≤ exp(C‖z‖log(‖z‖+2)+circleLoss(z))`.
The remaining genuine step is proving this explicit majorant is
`o(2^(2j))`, including the square-root term and the
`card·log(3 card/H)` contribution, before invoking maximum modulus.
The source count is now in exactly the needed multiplicity-aware form:
`card(window(2^(j+2))) ≤ B·2^j·(j+1)`. No unit-shell estimate is used.
Moreover, `log x≤x` gives
`card·log(3card/H)/H² ≤ 3B²(j+1)²/2^j`, and both this polynomial/geometric
majorant and the Cauchy--Schwarz square-root majorant
`sqrt(K(j+1)/2^j)` are proved to tend to zero. The remaining formal work is
assembling these estimates with the root-log and radial-tail terms into one
eventual epsilon inequality for `riemannXiCanonicalCircleLogLoss`; that
epsilon statement is the final input needed by the existing
maximum-modulus/rigidity pipeline. The three normalized shapes are now
combined into `DyadicCartanLossMajorant`, with a proved
`∀ ε>0, ∀ᶠ j, majorant j<ε` theorem. The remaining dominance proof must
reparameterize an arbitrary real scale `r` by a dyadic `j`. An initially
proposed refinement selected circles directly in `[r,2r]` with cutoff
`2^(j+3)`, but inspection of the maximum-modulus proof shows this is
unnecessary: the boundary interface only requires a circle radius
`R≥r`, with the logarithmic bound measured against `r²`. A new minimal
`DyadicSubquadraticBoundaryLogNormGrowth` interface is now formalized.
Choosing the next dyadic scale costs at most a factor four in the square,
absorbed by `ε/4`; this proves the original boundary interface, global disk
growth, and the complete xi Hadamard pipeline from dyadic boundary data.
`dyadicSubquadraticBoundaryLogNormGrowth_of_majorant` also converts any
eventual exponential circle estimate dominated by
`DyadicCartanLossMajorant` directly into that interface. The sole remaining
step is the concrete inequality placing the already-derived xi quotient
circle bound below one fixed choice of the three majorant constants. This
inequality is now proved explicitly:
`Ksq=2B+3B²`, `Ksqrt=4BM`, and `Klin=4C+3A`, where `A` is the radial-tail
constant, `B` the Jensen count constant, `C` the global xi-growth constant,
and `M` the total inverse-square zero mass. It discharges dyadic boundary
growth and proves the affine genus-one Hadamard representation whenever
`RiemannXiZeroIndex` is inhabited. The normalization gives exactly
`exp a=1` and the product form of the xi functional equation; it does not
canonically select `a` modulo `2πiℤ` or separately eliminate `b`.

The zero-free branch is now eliminated internally, without RH or an imported
zero-existence theorem. For an empty divisor the canonical product is `1`;
the global order-one bound still gives dyadic boundary growth, hence xi is
exponential-affine. Differentiating its functional equation at `1/2` forces
the slope to vanish. The explicit sourced evaluation
`riemannXiLi 2 = π/3 ≠ 1` (from `ζ(2)=π²/6` and `π>3`) then contradicts
normalization. Thus `riemannXiZeroIndex_nonempty`, the affine genus-one
Hadamard representation, and its normalized form are unconditional across
empty/finite/infinite cases.

The all-order finite product identity is proved, but the xi-level
`LiChangeOfVariablesCoefficientIdentity` remains the genuine blocker. It is
the analytic/combinatorial Möbius/Faà di Bruno identity between the derivative
definition at `s=1` and the logarithmic-derivative Taylor coefficients at
`z=0`; Hadamard convergence alone does not identify those two expressions.
The derivative-definition side is now fully expanded for every `n` by
`liDerivativeCoefficient_eq_leibnizSum`, using Mathlib's all-order Leibniz
rule and the closed derivative formula for `s^(n-1)`. The resulting explicit
finite sum has coefficients
`choose(n,i) * descFactorial(n-1,i)` and derivatives
`D^(n-i) log xi(1)`. `MobiusFaaDiBrunoCoefficientIdentity` states exactly
that the transformed logarithmic derivative has these coefficients, and is
proved equivalent to the old xi coefficient obligation. Thus no analytic
interchange remains hidden in the derivative definition; the blocker is the
pure all-order composition formula for `s=(1-z)⁻¹`.
The inner map itself is now completely discharged:
`iteratedDeriv_one_sub_inv` proves
`D^k(1-z)⁻¹ = k! (1-z)^(-(k+1))` at every point, and in particular the
derivative at zero is exactly `k!`. Expanding the outer composition leads to
the specialized recurrence
`D[(1-z)^(-p) D^m L((1-z)⁻¹)] =
p(1-z)^(-(p+1))D^m L + (1-z)^(-(p+2))D^(m+1)L`.
The remaining formal blocker is summing this recurrence and proving its
closed coefficients agree with the already-derived
`choose(k+1,i) * descFactorial(k,i)` expression. This is a pure finite
combinatorial Faà di Bruno step; no zeta estimate or infinite interchange is
left in it.
That finite combinatorial step is now closed. The triangular recurrence is
packaged as `IsMobiusFaaDiBrunoCoefficientRecurrence`, including its base,
support, and boundary cases. Pascal plus
`Nat.choose_succ_right_eq` proves that
`mobiusFaaDiBrunoCoefficient k i = choose(k+1,i) * descFactorial(k,i)`
satisfies the recurrence, and a two-index induction proves uniqueness.

The remaining blocker has consequently moved to the analytic realization of
the recurrence: formalizing, on one neighborhood of zero, repeated
differentiation of
`(1-z)^(-p) D^m(log xi)((1-z)⁻¹)`. Mathlib contains a general
`HasFTaylorSeriesUpToOn.comp` Faà di Bruno theorem, but connecting its ordered
finite partitions to this specialized scalar recurrence (or proving the
local recurrence directly for iterated derivatives of an analytic germ) is
still required. Until that bridge is formalized,
`MobiusFaaDiBrunoCoefficientIdentity`, the all-order zero-sum formula, and
the positivity implications remain conditional.
One important Mathlib API gap in that bridge is now closed project-locally:
`AnalyticAt.differentiableAt_iteratedDeriv` transports analyticity through
Mathlib's `iteratedFDeriv` theorem and the scalar evaluation equivalence,
showing every one-variable `iteratedDeriv m` is differentiable at the germ.
This supplies the outer-derivative step needed in the recurrence. What
remains is reindexing the two finite contributions at each induction step.
The analytic differentiation itself is now explicit:
`hasDerivAt_mobiusFaaDiBrunoSummand` proves the exact product/chain rule for
one basis term, including the Möbius power shifts, and
`hasDerivAt_mobiusFaaDiBrunoSum` differentiates the complete finite sum
termwise with all truncated-subtraction boundaries checked. The remaining
finite algebra is to shift the product contribution from `i` to `i+1` and
identify the resulting coefficients with the proved triangular recurrence.
That finite algebra is now closed:
`mobiusFaaDiBrunoSum_reindex` proves the support-aware generic shift, and
`mobiusFaaDiBrunoCoefficientSum_reindex` specializes it to the closed
choose/falling-factorial array, including the zero top boundary. The
remaining coefficient-transfer obligation is no longer combinatorial: it is
the local analytic induction glue establishing the finite-sum formula as a
neighborhood identity before taking its next derivative.
That neighborhood induction is now formalized by
`iteratedDeriv_mobiusComposition_eq_row`: analyticity is pulled back through
the Möbius map to an eventual neighborhood, the prior row is replaced by an
eventually equal function, and its derivative is the collected next row.
`iteratedDeriv_mobiusComposition_zero` then gives the exact closed finite sum
at zero. The final xi-specific bridge still required is an eventual equality
between `logDeriv (liChangeOfVariables xi)` and the derivative of
`log xi ∘ liMobiusArgument` on the zero neighborhood; this needs explicit
nonvanishing and logarithm-chain bookkeeping.
That xi-specific bridge is now closed. Continuity of
`riemannXiLi ∘ liMobiusArgument` at zero and `xi(1)=1` give an eventual
neighborhood inside `Complex.slitPlane`, hence in particular a zero-free
neighborhood. `logDeriv_liChangeOfVariables_eq_mobiusLogDerivative` proves
the branch-compatible chain rule there, and eventual equality transfers all
iterated derivatives. Consequently
`riemannXiLi_mobiusFaaDiBrunoCoefficientIdentity` and
`riemannXiLi_changeOfVariablesCoefficientIdentity` are unconditional at
every order.

The remaining zero-sum/RH-forward step is downstream rather than
combinatorial: the existing theorem
`riemannXiLi_finiteZeroLiSums_tendsto` now needs only an instantiated
`XiCanonicalProductApproximation` for the concrete zero windows. The
whole-plane Hadamard representation is proved, but that exact approximation
interface has not yet been derived from it. Bombieri--Lagarias
positivity-to-location remains a separate converse theorem.
The first half of that derivation is now concrete. The radial multiplicity
windows are proved cofinal by
`riemannXiZeroIndexWindow_nat_tendsto_atTop`; their finite genus-one products
converge locally uniformly to the infinite product; and
`exists_riemannXiRadialHadamardApproximants_tendstoLocallyUniformly` combines
them with the unconditional affine Hadamard prefactor to converge locally
uniformly to xi on the whole plane.

The remaining mismatch with `XiCanonicalProductApproximation` is exact and
cannot be suppressed: that interface uses unscaled products centered at
`s=1`, while the proved approximants are zero-centered genus-one products
with the fixed affine factor. Passing between them requires proving that the
radial inverse-root correction converges to the negative Hadamard slope and
that the centering constants normalize to one. In addition, the criterion's
current windows are zeta-divisor multisets, whereas the Hadamard approximants
use the canonical xi multiplicity enumeration; their finite-window
multiplicity-preserving identification is not yet formalized.
The reflection algebra and multiplicity transport are now explicit:
`riemannXiZeroReflection` sends every canonical multiplicity occurrence to
the corresponding occurrence over `1-rho`, preserving its fiber ordinal,
and `riemannXiZeroRoot_inv_add_reflection` proves
`rho⁻¹ + (1-rho)⁻¹ = (rho(1-rho))⁻¹`. Thus paired inverse-root corrections
have the desired inverse-quadratic form.

This also exposes why the current radial route cannot yet prove the slope
limit. `riemannXiZeroReflection_mem_window_add_one` only maps the radius-`R`
window into radius `R+1`; radial windows are not exactly reflection
invariant. The available Jensen bound is cumulative `O(R log R)` and gives no
small enough unit-shell boundary count to discard that imbalance. A valid
closure therefore needs either a sourced Riemann--von Mangoldt/unit-shell
estimate, or finite symmetric-height windows. The latter additionally needs
the currently absent unconditional theorem placing every xi zero in a fixed
real strip. Assuming radial symmetry here would be false.
The fixed-strip branch is now closed without RH:
`riemannXiLi_zero_re_pos` and `riemannXiLi_zero_re_lt_one` combine zeta
nonvanishing on `Re s ≥ 1` with xi's functional equation to place every xi
zero in `0 < Re rho < 1`. Consequently
`riemannXiZeroHeightWindow` is finite, radial and height windows contain each
other with the sharp additive-one comparison, and both exhaustions are
cofinal.

Height windows are exactly reflection-stable with multiplicity:
`riemannXiZeroHeightWindow_map_reflection` proves reflection permutes each
finite index set, and `two_mul_sum_inv_riemannXiZeroHeightWindow` proves the
finite paired identity
`2 ∑ rho⁻¹ = ∑ (rho(1-rho))⁻¹`. Their genus-one Hadamard approximants now
also converge locally uniformly to xi after the affine prefactor is restored.
The remaining slope step is to pass this paired finite identity to the limit
and identify it with the Hadamard slope. Conjugation stability is also still
missing because pinned Mathlib exposes no completed-zeta conjugation theorem.
The limit half is now closed:
`summable_riemannXiZeroRoot_inv_mul_reflection` proves absolute summability
of `1/(rho(1-rho))` by AM--GM from the established inverse-square sum, and
`riemannXiInverseRootHeightSums_tendsto` identifies the symmetric-height
inverse-root limit with half that `tsum`. Moreover,
`logDeriv_riemannXiHeightGenusOneProduct_one` identifies every finite paired
sum as the logarithmic derivative at `1` of the corresponding genus-one
product. The exact remaining slope obligation is therefore derivative
passage through the locally uniform product limit, combined with the
functional equation and normalized affine representation.
That derivative obligation is now closed. Locally uniform convergence of the
height-window genus-one products is passed through Mathlib's Weierstrass
derivative theorem at `0` and `1`. The infinite canonical product has
derivative zero at `0`, while its logarithmic derivative at `1` is exactly
the paired inverse-root `tsum`. Differentiating xi's functional equation then
proves
`exists_riemannXiHadamardSlope_eq_pairedInverseRootSum`, namely
`∑ 1/(rho(1-rho)) = -2b`, and hence the symmetric-height sums
`∑ 1/rho` converge to the negative Hadamard slope `-b`.

The remaining route is no longer blocked on slope. It is blocked on
conjugation (Mathlib has `Gamma_conj` and `conj_tsum`, but no packaged
Riemann-zeta/completed-zeta conjugation theorem) and on the global
classification of the definition's nontrivial zeta zeros into the critical
strip. Those are required before the multiplicity-exact xi/zeta height-window
bijection and concrete `XiCanonicalProductApproximation` can be completed.
No all-index Li criterion or Bombieri--Lagarias converse is overclaimed.
Plain tail convergence is
insufficient after multiplication by `R²`. It is not circle selection or any
subsequent quotient, logarithm, rigidity, or continuation step. The
xi-specific pipeline states:
`CartanExceptionalDiskLogBound` alone implies
`GenusOneHadamardRepresentation riemannXiZeroRoot riemannXiLi`.

Reflection symmetry plus Bombieri--Lagarias summability still admits an
explicit finite off-line spectrum. Thus those hypotheses alone do not prove
the critical line; the exact remaining spectral result is the
Bombieri--Lagarias theorem that positivity of the genuine convergent Li sums
forces the individual unit-disk bounds.
The project now records the equivalent source-shaped conclusion
`1/2 ≤ re ρ`; reflection symmetry then supplies the opposite inequality.
It also packages complete spectral enumeration, summability, symmetric Li
limits, and reflection into `RiemannLiSpectralModel`; positivity plus the
actual Bombieri--Lagarias half-plane implication then derives RH without
assuming RH.

After infrastructure (1), proving all coefficients nonnegative directly
would itself prove RH via (2); it is not a finite or computational next step.
The development therefore stops at the first genuine missing theorem rather
than assuming either bridge.

The finite logarithmic-derivative formula is not a recurrence in `n`.
Coefficient positivity supplies no relation determining a later coefficient
from earlier ones, and the existing `finitePrefixSpoof` theorem continues to
rule out extrapolation from any fixed prefix.
