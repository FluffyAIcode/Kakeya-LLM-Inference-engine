# Weil positivity / energy route to RH

Status: **partial, unconditional finite/limit infrastructure**.  Nothing in
this route asserts the full explicit formula or RH.

## Exact classical criterion (multiplicative normalization)

Let

* \(D=C_c^\infty((0,\infty),\mathbb C)\);
* \(\widehat f(s)=\int_0^\infty f(x)x^{s-1}\,dx\);
* \((f*g)(x)=\int_0^\infty f(x/y)g(y)\,dy/y\);
* \(f^\ast(x)=x^{-1}\overline{f(x^{-1})}\).

Then
\[
 \widehat{f^\ast}(s)
 =\overline{\widehat f(1-\overline s)},\qquad
 \widehat{f*f^\ast}(s)
 =\widehat f(s)\overline{\widehat f(1-\overline s)}.
\]

For the Riemann zeta function, with nontrivial zeros counted with
multiplicity in the symmetric explicit-formula limit, Weil's criterion is
\[
  \mathrm{RH}\quad\Longleftrightarrow\quad
  \sum_\rho \widehat{f*f^\ast}(\rho)\ge 0
  \quad\text{for every }f\in D.
\]

Bombieri's refinement states strict positivity for every nonzero complex
\(f\in D\).  The commonly printed real-test version suppresses conjugation:
for real \(f\), the summand is written
\(\widehat f(\rho)\widehat f(1-\rho)\).  These are not different
normalizations.

The associated Guinand--Weil formula in Bombieri's normalization is
\[
\begin{aligned}
\sum_\rho \widehat h(\rho)
={}&\widehat h(1)+\widehat h(0)
-\sum_{n\ge1}\Lambda(n)(h(n)+h^\tau(n))\\
&-(\log(4\pi)+\gamma)h(1)
-\int_1^\infty
  \left(h(x)+h^\tau(x)-\frac{2h(1)}x\right)
  \frac{x\,dx}{x^2-1},
\end{aligned}
\]
where \(h^\tau(x)=x^{-1}h(x^{-1})\).  For the quadratic criterion,
\(h=f*f^\ast\).  Compact support makes the prime-power sum finite, while
smoothness gives rapid vertical decay of the Mellin transform.

## Logarithmic, critical-line-centred normalization

Putting \(x=e^t\), centring \(s=1/2+z\), and setting
\(\phi(t)=e^{t/2}f(e^t)\) identifies the test space with
\(C_c^\infty(\mathbb R,\mathbb C)\).  Exactly,
\[
 \widehat f(1/2+z)=\int_{\mathbb R}\phi(t)e^{zt}\,dt.
\]
The transform and involution used in `KakeyaLeanGate.WeilPositivity` are
\[
 F(z)=\int_\mathbb R f(t)e^{zt}\,dt,\qquad
 f^\star(t)=\overline{f(-t)}.
\]
Additive convolution in \(t\) is multiplicative convolution for \(dx/x\).
RH says exactly that all spectral parameters \(z=\rho-1/2\) are imaginary.
This is also Burnol's shifted-distribution formulation
\(C(f*f^\star)\ge0\).

## Variant and scope distinctions

* **Weil/Bombieri, zeta:** the statements above are equivalent to RH for
  \(\zeta(s)\), not GRH.
* **Nice-test variant (Lagarias):** piecewise \(C^2\), compact support, with
  midpoint values at jumps.  Its spectral functional is
  \(W^{(1)}(h)=\sum_\rho M[h](\rho)\), and RH is equivalent to
  \(W^{(1)}(f*\widetilde{\bar f})\ge0\) for every nice \(f\).
  The smooth compactly-supported class is a sufficient subspace for the
  criterion and is the class formalized here.
* **Barner class:** a broader decay/bounded-variation class used for
  Weil--Barner explicit formulae.  It is useful for numerical zero
  certification, but is not the domain selected for this Lean interface.
* **Number fields / Hecke characters:** Weil's adelic criterion gives the
  corresponding generalized RH only when positivity is required for the
  relevant completed \(L\)-functions.  It must not be described as a proof of
  GRH from the zeta-only functional.
* **Even real additive tests:** positivity only on an improperly weakened
  symmetry class can miss real exceptional zeros; this is why the formal
  domain is complex and the involution includes conjugation.

## Pinned Mathlib audit (`v4.32.0-rc1`)

Available and used:

* `TestFunction`: bundled \(C_c^\infty\) functions, including linear
  operations and compact support;
* `HasCompactSupport.toSchwartzMap`, giving the canonical map
  \(C_c^\infty(\mathbb R)\to\mathcal S(\mathbb R)\);
* Bochner convolution, compact-support closure, and smoothness of
  convolution;
* `mellin`, `MellinConvergent`, and Mellin inversion infrastructure;
* `Real.fourier_eq'`, Schwartz Fourier transforms, Fourier convolution, and
  Plancherel infrastructure (with Mathlib's \(e^{-2\pi i x\xi}\)
  normalization);
* `Matrix.PosSemidef`, Gram matrices, and positive-semidefinite submatrices;
* closed-order limits and `tendsto_atTop_ciSup` for bounded monotone real
  families;
* `riemannZeta`, `completedRiemannZeta`, the functional equation, and the
  proposition `RiemannHypothesis`;
* `riemannZetaZeros`, `isDiscrete_riemannZetaZeros`, and
  `IsCompact.inter_riemannZetaZeros_finite`, giving finite compact windows of
  distinct zeros (without multiplicity);
* `ArithmeticFunction.LSeriesSummable_vonMangoldt` and
  `ArithmeticFunction.LSeries_vonMangoldt_eq_deriv_riemannZeta_div`, giving
  absolute convergence and
  \(\sum_n\Lambda(n)n^{-s}=-\zeta'(s)/\zeta(s)\) for \(\Re s>1\);
* `tendsto_integral_of_dominated_convergence` and its filter variant;
* `TendstoUniformlyOn.tendsto_at`, reducing uniform convergence on a selected
  test subclass to the pointwise convergence positivity needs.

Absent after repository-wide API search:

* a type enumerating nontrivial zeta zeros with multiplicity (the pinned
  library does have the zero set and compact-window finiteness);
* the symmetric/regularized sum over zeros;
* the von-Mangoldt plus archimedean Guinand--Weil explicit formula;
* a completed-zeta zero distribution;
* any theorem connecting distribution positivity to `RiemannHypothesis`.

The Lean file therefore defines the admissible domain, its actual inclusion
into Mathlib's Schwartz space, a multiplicative representative, the actual
`mellin` expression, centred transform, involution, convolution, a typed
distribution interface, its exact quadratic functional and positivity
predicate, and the proposition
`BridgeObligation W := IsPositive W ↔ RiemannHypothesis`.  The multiplicative
change of variables is named `MellinNormalizationObligation`; it is not
silently assumed.  The route supplies no inhabitant of either obligation.
The critical-line Fourier normalization is proved:
\[
 F_f(-2\pi i\xi)=\mathcal F_{\rm Mathlib}(f)(\xi).
\]

The module also exposes `zetaZeroWindowFinset R`, the finite set of distinct
zeta zeros in a closed disk.  This is a source-backed restricted finite object,
not the spectral explicit-formula sum: a `Finset` forgets analytic
multiplicity, and disk truncation does not choose the classical symmetric
regularization.

On the prime side, `vonMangoldt_lseries_eq_neg_logDeriv` and
`vonMangoldt_lseries_summable` wrap Mathlib's checked theorems on \(\Re s>1\).
They are genuine smaller explicit-formula ingredients, but they do not cross
the critical strip or supply zero and archimedean terms.

## Finite approximants proved exactly

`FiniteFormulaData` records a finite multiset of spectral parameters (by a
finite index type, so repetition records multiplicity), finite weighted
prime-power samples in logarithmic coordinates, and pole/archimedean terms.
`FiniteFormulaData.IsExactAt f` is the exact finite identity
\[
 \sum_{\rho\in Z_N}F_f(\rho)
 =
 P_N(f)-\sum_{q\in\mathcal P_N}w_q f(\ell_q)-A_N(f).
\]
It is an explicit hypothesis, not a claim that zeta satisfies the identity.

For ordinates \(\gamma_j\in\mathbb R\), the formalized finite energy is
\[
 E_N(f)=\sum_j\left|F_f(i\gamma_j)\right|^2\ge0.
\]
Under the explicitly stated local factorization hypothesis
\[
 F_{f*f^\star}(i\gamma_j)=|F_f(i\gamma_j)|^2,
\]
Lean proves that the real part of the finite zero sum equals \(E_N(f)\), and
hence is nonnegative.  This is an exact algebraic/positivity result; it does
not assert the Fubini theorem needed to establish factorization for every
test.

`FiniteExplicitStage` packages the same finite identity with all four terms
as linear distributions.  If the spectral, pole, prime-power, and
archimedean distributions converge pointwise, Lean proves that the
explicit-formula identity passes to their limits.

## Limit interfaces and what they do not prove

The route proves:

* a pointwise limit of nonnegative real quadratic forms is nonnegative;
* no uniform convergence hypothesis is needed for this implication;
* eventual positivity may have a test-dependent cutoff;
* uniform convergence on a selected test subclass implies the required
  pointwise convergence there;
* an error envelope tending to zero establishes scalar convergence and
  transfers nonnegativity;
* pointwise convergence of linear distributions gives convergence of their
  quadratic functionals;
* a bounded monotone family converges pointwise to its conditional supremum
  via Mathlib's `tendsto_atTop_ciSup`;
* the resulting supremum is nonnegative when every finite stage is;
* finite-stage nonnegativity alone says nothing about an independently
  declared limiting value.  A formal counterexample uses the constantly
  zero stages and the falsely declared limit \(-1\).

The last item is why zero-sum regularization cannot be omitted or replaced by
numerical positivity at every cutoff.

Thus finite positivity plus an independently declared value is insufficient;
finite positivity plus eventual pointwise convergence for each admissible test
is sufficient.  Uniform convergence is a stronger optional route for
restricted subclasses, not a hidden universal requirement.

## Precise bridge decomposition

The remaining work is split into named propositions:

1. `MellinNormalizationObligation`: prove convergence and the
   \(x=e^t\), \(s=1/2+z\) change of variables for every logarithmic test.
2. `ZeroRegularizationObligation`: define symmetric zero truncations,
   including multiplicity/order, and prove pointwise convergence to a
   distribution.
3. `PrimePowerStabilizationObligation`: define the von Mangoldt
   prime-power stages and prove eventual exactness on each compactly
   supported test.
4. `ArchimedeanLimitObligation`: construct the regularized gamma-factor
   integral and prove the selected truncations converge on every test.
5. `AnalyticBridgeObligations`: combine the four finite distributions and
   their limits; `explicitFormula_passes_to_limit` then yields the limiting
   linear identity.
6. `AllTestPositivityObligation`: separately prove both
   `IsPositive W → RiemannHypothesis` and
   `RiemannHypothesis → IsPositive W`.

The analytic explicit formula does not by itself discharge item 6: the
all-test separation argument and the treatment of off-critical-line zeros
are additional theorems.

`SymmetricZeroApproximants`, `PrimePowerApproximants`,
`ArchimedeanApproximants`, and `PoleApproximants` keep the four sequences typed
separately.  `TypedExplicitApproximants.explicitFormula` combines their limits
from exactly the corresponding regularization, stabilization, and convergence
obligations.  Its stage identities are explicit fields; no zeta instance is
assumed.

## Proved support and terminal blocker

The module proves, without `sorry`, `admit`, or axioms:

* involution is closed, involutive, additive, zero-preserving, and
  conjugate-linear;
* convolution is again a smooth compactly-supported test;
* every logarithmic test canonically gives a Mathlib Schwartz function;
* the centred transform on \(-2\pi i\xi\) equals Mathlib's Schwartz Fourier
  transform at \(\xi\);
* finite zero/prime sums and finite formula stages are typed explicitly;
* finite critical-line spectral energies are nonnegative;
* the finite convolution-square sum equals that energy under an explicit
  factorization hypothesis;
* exact finite explicit-formula identities pass through pointwise limits;
* positivity passes through pointwise and bounded monotone limits;
* every finite Gram kernel is positive semidefinite;
* every reindexed finite Gram restriction remains positive semidefinite;
* rank-one real quadratic certificates are nonnegative;
* the explicit Hermitian matrix
  \(\begin{pmatrix}1&2\\2&1\end{pmatrix}\) has value \(-2\) on
  \((1,-1)\), so Hermitian symmetry alone cannot replace the exact
  arithmetic functional.

The remaining blocker is mathematical: discharge the six named obligations
above for the actual zeta zero and prime-power distributions.  In particular,
Mathlib still has no nontrivial-zero multiset with multiplicity and no
Guinand--Weil theorem from which the finite stages or their regularized limit
could be instantiated.  Its distinct-zero compact windows and
right-half-plane \(-\zeta'/\zeta\) theorem reduce engineering gaps but do not
provide symmetric zero regularization, the gamma-factor limit, or all-test
separation.  Finite Gram/SDP certificates and finite energies prove only
restrictions and cannot discharge universal all-test positivity.
