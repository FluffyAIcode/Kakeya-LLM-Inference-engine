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

### Stage-four right-half-plane fragment

The formalization now exposes Mathlib's exact individual term
\[
  a_n(s)=
  \begin{cases}
  0,&n=0,\\
  \Lambda(n)n^{-s},&n>0,
  \end{cases}
\]
as `vonMangoldtTerm`.  For \(\Re s>1\), Lean proves
`vonMangoldtPartialSum_tendsto_logDeriv`:
\[
  \sum_{n<N} a_n(s)\longrightarrow-\frac{\zeta'(s)}{\zeta(s)}.
\]
This pins both the zero-index convention and the sign of the logarithmic
derivative to actual Mathlib declarations.

`PrimeSmoothingWeight` consists of coefficients \(w(n)\) satisfying
\(|w(n)|\le1\).  Absolute convergence of the von Mangoldt L-series proves
summability of \(\sum_n w(n)a_n(s)\).  If \(w_k(n)\to1\) pointwise, Mathlib's
dominated-convergence theorem for `tsum` proves that these smoothed series
converge to the same negative logarithmic derivative.  This is an honest
bounded/smoothed explicit-formula piece only in \(\Re s>1\); it is not a
Guinand--Weil formula or analytic continuation.

### Actual archimedean factor

The archimedean term is now pinned to Mathlib's Deligne factor
\[
  \Gamma_{\mathbb R}(s)=\pi^{-s/2}\Gamma(s/2).
\]
Lean proves directly from `Complex.Gammaℝ`, `Complex.digamma`, and their
derivative APIs that, for \(\Re s>0\),
\[
  \frac{\Gamma_{\mathbb R}'(s)}{\Gamma_{\mathbb R}(s)}
  =-\frac{\log\pi}{2}+\frac12\psi(s/2).
\]
At \(s=1\), `archimedeanLogDeriv_one` pins the constant to
\[
  -\frac{\gamma+\log(4\pi)}2.
\]
`ArchimedeanGammaApproximation` packages a quantitative truncation and proves
its convergence from an error envelope.  Mathlib's digamma file explicitly
lists Gauss's integral representation as TODO, so the construction of the
classical truncated archimedean integral remains an analytic obligation
rather than an invented identity.

There is nevertheless a genuine source-backed positive-integer
approximation.  Mathlib proves
\[
  \psi(n+1)=-\gamma+H_n
\]
and constructs \(\gamma\) between the harmonic--log sequences.  Lean now
derives the explicit error estimate
\[
  0\le H_n-\log n-\gamma
  \le\log(n+1)-\log n\longrightarrow0
  \qquad(n>0).
\]
This gives `norm_digamma_nat_add_one_sub_log_le`.  At \(s=2\), Gamma
recurrence yields the exact finite `archimedeanAtTwoStage`; its error is zero.

`vonMangoldtPartialSum_add_tail` also splits the actual absolutely convergent
prime series into a finite sum and exact tail.  Combining these gives the
non-circular theorem `finiteCompletedLogDerivAtTwo`:
\[
 \sum_{n<N}\frac{\Lambda(n)}{n^2}
 +\sum_{n\ge0}\frac{\Lambda(n+N)}{(n+N)^2}
 +A_N(2)
 =
 -\frac{\zeta'(2)}{\zeta(2)}
 +\frac{\Gamma_{\mathbb R}'(2)}{\Gamma_{\mathbb R}(2)}.
\]
This is a genuine finite theorem for completed logarithmic-derivative
components.  It is not yet a zero-sum Guinand--Weil formula.

### Multiplicity-aware restricted subclass

`MultiplicityAwareSymmetricZeros` records a symmetry-orbit representative
\(z_n\), an explicit natural multiplicity \(m_n\), and whether the orbit is
fixed by \(z\mapsto-\overline z\).  Its centred term is
\[
  \begin{cases}
    m_nF_f(z_n),&z_n=-\overline{z_n},\\
    m_n\bigl(F_f(z_n)+F_f(-\overline{z_n})\bigr),&\text{otherwise}.
  \end{cases}
\]
The fixed-orbit branch prevents critical-line zeros from being counted twice.
Under an explicit `Summable` hypothesis, the finite range truncations converge
to the `tsum`, and the value is invariant under bijective re-enumeration.
This fixes multiplicity and symmetry at the data level without pretending
that Mathlib supplies an enumeration of zeta zeros.

Under the stronger `ZeroOrbitNormallyConvergent` hypothesis, an arbitrary
`SymmetricZeroWindows` sequence of finite orbit sets may be used.  Every
window scheme exhausting the index type converges to the same regularized
value, and differences between two schemes tend to zero.  This proves window
independence exactly under normal convergence; it does not assert normal
convergence for the classical zeta zero sum.
`ZeroOrbitNormallyConvergent.of_bound` reduces this hypothesis to a summable
majorant.  `ZeroOrbitShells` records a partition into finite height shells,
and `ZeroOrbitNormallyConvergent.of_shell_count_decay` proves normal
convergence from
\[
  \sum_k \#S_k\,d_k<\infty,\qquad
  \|F_f(\rho)\|\le d_k\quad(\rho\in S_k).
\]
The classical Riemann--von Mangoldt estimate supplies the shell-count side,
while rapid vertical transform decay would supply \(d_k\).  Pinned Mathlib
has neither the zero-count/multiplicity theorem nor the strip-uniform decay
estimate, so the actual zeta family still cannot instantiate these fields.

This iteration makes both sides quantitative.  `ZetaZeroShellCountObligation`
states coverage by centered symmetry orbits, the actual zeta-zero equations,
unit-height shells, and the sufficient multiplicity-weighted estimate
\[
  \sum_{i\in S_k}m_i\le C(k+2)^2.
\]
This is a deliberately weakened consequence of Riemann--von Mangoldt's
\(O(T\log T)\), not an asserted Mathlib theorem.  The peer-reviewed explicit
bound of Hasanalizade--Shen--Wong is recorded in the source card.
`summable_quadratic_shell_fourth_power_decay` proves that a
\((k+2)^{-4}\) transform envelope is summable against this count, and
`normalConvergence_of_fourthPower` plus
`symmetricZeroWindowSum_tendsto_of_zetaShellCount` discharge normal
regularization and arbitrary-window convergence.

The source-faithful interface `RiemannVonMangoldtExplicitObligation` records
a multiplicity count connected to every unit shell and the exact estimate
\[
\left|N(T)-\frac{T}{2\pi}\log\frac{T}{2\pi e}\right|
\le0.1038\log T+0.2573\log\log T+9.3675.
\]
The decimal constants are exact rationals in Lean.
`.shell_count_le` derives the precise main-plus-error shell envelope and
`.normalConvergence` proves all subsequent summability consequences.  The
global estimate remains uninhabited because pinned Mathlib has no
Riemann--von Mangoldt theorem.

The local multiplicity count is no longer abstract.  Lean defines
`positiveNontrivialZetaZeroWindow T` using the actual zeta equation and
critical-strip inequalities, proves this set finite from
`isDiscrete_riemannZetaZeros`, and assigns every point Mathlib's genuine
analytic-order multiplicity `zetaZeroMultiplicity`.  The resulting finite sum
`positiveNontrivialZetaZeroCount T` is monotone.  The source-facing
`ActualRiemannVonMangoldtBound` now asks only for the published numerical
inequality on this concrete count; `.shell_count_le` derives the actual
multiplicity mass bound on each unit shell.  What remains external is exactly
the global Riemann--von Mangoldt estimate, not local finiteness or a
placeholder multiplicity function.

For every \(f\in C_c^\infty(\mathbb R,\mathbb C)\) and fixed real \(a\),
`exponentiallyWeightedTest` remains a test function and Lean proves
\[
 F_f(a-2\pi i\xi)
 =\mathcal F(e^{a(\cdot)}f)(\xi),\qquad
 |\xi|^k|F_f(a-2\pi i\xi)|
 \le p_{k,0}\!\left(\mathcal F(e^{a(\cdot)}f)\right).
\]
Thus rapid decay on every fixed vertical line uses Mathlib's actual Schwartz
seminorm theorem.

`StripUniformSchwartzSeminormBound f k C` defines the restricted admissible
class by this concrete Fourier seminorm, uniformly for \(|a|\le1/2\).
`norm_transform_strip_pow_le` proves strip-uniform decay with exactly the same
constant \(C\).

The uniformity gap is now closed directly for every test, without requiring a
parameterized Schwartz-space compactness API.  `stripDerivativeMajorant`
is the explicit finite Leibniz envelope
\[
 \sum_{j=0}^n {n\choose j}2^{-j}e^{|x|/2}
 \left|f^{(n-j)}(x)\right|.
\]
Lean proves its compact support and integrability, bounds every derivative of
\(e^{ax}f(x)\) by it uniformly for \(|a|\le1/2\), and integrates the bound.
Using Mathlib's exact Fourier-derivative normalization then gives
`norm_transform_strip_derivative_le` and
`norm_transform_strip_pow_le_explicit`:
\[
 |\xi|^n |F_f(a-2\pi i\xi)|
 \le \int_{\mathbb R}\operatorname{stripDerivativeMajorant}(f,n,x)\,dx .
\]
This is nonzero, all-test, strip-uniform rapid decay with a concrete constant.
The corollary `norm_transform_centered_strip_im_pow_le_explicit` rewrites the
same estimate directly as
\(|\operatorname{Im}z|^n|F_f(z)|\le C_{f,n}\) on the centered critical strip,
which is the exact form needed for zeta-zero shell estimates.

`RestrictedExplicitFormulaData` is the restricted admissible subclass proved
at this stage.  A datum supplies one compactly supported smooth test, a
bounded prime smoothing, summable zero/pole/archimedean terms, and a checked
finite-stage identity.  Lean then passes all four pieces to the limit, using
the actual Mathlib-normalized prime series.  The finite-stage identity and
analytic summability remain hypotheses to discharge for any concrete class.
`RestrictedGuinandWeilLimitData.explicitFormula` sharpens this by combining
normal zero windows, bounded prime smoothing, a convergent pole stage, and a
Gamma approximation.  Its limit contains the actual targets
\(-\zeta'/\zeta\) and \(\Gamma_{\mathbb R}'/\Gamma_{\mathbb R}\).
`RestrictedGuinandWeilShellData.explicitFormula` instead derives normal zero
convergence from shell count-times-decay summability before combining every
available component limit.  Its finite-stage Guinand--Weil identity remains
an explicit hypothesis.

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
Mathlib's Fourier convolution theorem now proves the factorization
\[
 F_{f*f^\star}(i\gamma_j)=|F_f(i\gamma_j)|^2,
\]
for every compactly supported smooth test.  The proof uses
`transform_involution_imaginary`,
`transform_convolution_involution_imaginary`, and Mathlib's
`Real.fourier_mul_convolution_eq`.  Therefore
`finiteZeroSum_convolution_eq_energy_exact` and
`finiteZeroSum_convolution_nonneg_exact` are unconditional finite
critical-line results.  They still do not identify this spectral sum with
the arithmetic prime/Gamma/pole side; that residue-theoretic finite
Guinand--Weil identity remains unavailable.

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

The dense-subclass route is also now exact.  For any topological space, Lean
proves that nonnegativity of a continuous real functional extends to the
closure of a subclass.  Consequently,
`Distribution.IsPositive.of_dense` requires both:

1. `Dense S` in Mathlib's canonical LF topology on `Test`; and
2. `Distribution.QuadraticContinuous W`.

Positivity merely verified on \(S\) does not imply universal positivity
without both obligations.  In particular, neither pointwise convergence of
distributions nor algebraic linearity of `Distribution.eval` establishes LF
continuity of the quadratic functional.
`dense_nonnegative_counterexample_without_continuity` makes the failure
sharp: the complement of \(0\) is dense in \(\mathbb R\), but the function
equal to \(0\) there and \(-1\) at \(0\) does not preserve nonnegativity.

The LF continuity obligation is now decomposed using Mathlib's actual
universal property.  `Distribution.evalContinuous_iff_supported` proves that
linear evaluation is LF-continuous exactly when every restriction to a fixed
compact support is continuous.  If evaluation is continuous and the nonlinear
map \(f\mapsto f*f^\star\) is LF-continuous, then Lean proves continuity of
the quadratic functional.  The pinned library has no theorem establishing
LF continuity of this convolution-square map and no dense-subclass theorem
for the proposed explicit-formula test classes.  Those are the exact missing
topological results.

Candidate distributions can now instead be introduced as
`ContinuousDistribution`, whose evaluation is a genuine continuous linear
map; conversion to the algebraic interface preserves LF evaluation
continuity.  The nonlinear remainder is split into
`InvolutionContinuous` and joint `ConvolutionContinuous`, and Lean proves
that these imply `ConvolutionSquareContinuous`.  Mathlib provides
pointwise continuity/smoothness of convolution and Schwartz-space continuous
maps, but no corresponding joint continuous map on the LF test-function
space.  The algebraic fixed-support reflection map is now constructed as
`supportedInvolutionLM`, and `involutionContinuous_of_supported` uses the LF
universal property to lift stagewise continuity.  The exact missing local API
is continuity of this precomposition map on every \(\mathcal D_K\).
`SupportedConvolutionContinuous` similarly states fixed-support joint
continuity, but Mathlib's universal property is for linear maps and supplies
no product/final-topology theorem lifting this bilinear map to global joint
LF continuity.
`TestFunction.LFProductUniversalProperty` states that missing theorem exactly;
`convolutionContinuous_of_lfProduct` proves it and fixed-stage joint
continuity suffice.  The one-variable part can be lifted without that missing
product theorem: `convolutionLeftLM` proves fixed-left convolution is
real-linear, `convolutionLeftCLM` applies Mathlib's LF universal property, and
`convolution_separatelyContinuous_left` proves global separate continuity
from `SupportedConvolutionContinuous`.  The remaining jump from separate to
joint continuity is genuine and is not implied by the available API.
Convolution is also bundled as the genuine bilinear map
`convolutionBilinearLM`.  The narrower
`TestFunction.LFBilinearProductUniversalProperty` is now the exact
locally-convex product-lifting obligation, and
`convolutionContinuous_of_lfBilinearProduct` proves it suffices.  This avoids
requiring the overly broad arbitrary-map product property.

The fixed-stage analytic estimate is now explicit.  Lean proves
`iteratedDeriv_convolution_right`, so every derivative may be moved to the
right factor.  For \(f\in\mathcal D_K\), it also proves
`integral_norm_supported_le_measure_mul_seminorm`,
\[
  \int_{\mathbb R}\lVert f(t)\rVert\,dt
  \le \operatorname{vol}(K)\,N_{K,0}(f),
\]
and combines these as `norm_iteratedDeriv_supported_convolution_le`:
\[
 \lVert (f*g)^{(n)}(x)\rVert
 \le \operatorname{vol}(K)\,N_{K,0}(f)\,N_{L,n}(g).
\]
Thus the quantitative fixed-support seminorm estimate is no longer an
obligation.  `convolutionCompact` and `supportedConvolution` now package the
result in the exact target \(\mathcal D_{K+L}\), and
`supportedConvolution_seminorm_le` proves the corresponding target-seminorm
bound.  `supportedConvolution_mono` proves compatibility with every pair of
compact-support inclusions, so the full inductive-system compatibility is
closed.  The bilinear difference identity `supportedConvolution_sub_sub`
turns the estimate into a direct continuity-at-each-point proof:
`supportedConvolution_continuous` establishes joint continuity
\(\mathcal D_K\times\mathcal D_L\to\mathcal D_{K+L}\), and
`supportedConvolutionContinuous` discharges the former fixed-stage
obligation.  Consequently left and right separate LF continuity now hold
unconditionally.  What remains is only the topological theorem turning these
compatible stagewise bilinear maps into joint continuity on the LF product;
Mathlib's linear LF universal property does not state that theorem.

This missing theorem is now reduced to one concrete topology inequality.
`TestFunction.LFPairFinalTopology` is the final topology generated by all
\(\mathcal D_K\times\mathcal D_L\) inclusions.  Lean proves
`lfPairFinalTopology_le_product`: this final topology is no finer than the
ordinary product of the two LF topologies.  The reverse inequality is named
`LFPairProductTopologyObligation`.  Under precisely that inequality,
`continuous_of_lfPairProductTopology` proves the full universal property,
`convolutionContinuous_of_lfPairProductTopology` gives global joint
convolution continuity, and the convolution-square map is continuous.  Thus
the remaining LF blocker is no longer an unspecified “product API”; it is the
single unproved comparison
\[
  \mathcal T_{\mathcal D}\times\mathcal T_{\mathcal D}
  \;\le\;
  \operatorname{Final}_{K,L}
  (\mathcal D_K\times\mathcal D_L).
\]
This reverse inequality is not a formal consequence of separate continuity
or of the one-variable final-topology theorem.

The audit of Mathlib's definition explains the obstruction.  The canonical
test-function topology is not the raw topological final topology:
`TestFunction.topologicalSpace` is the finest locally convex topology coarser
than `TestFunction.originalTop`.  Products need not commute with this
locally-convexification operation.  Lean now proves
`lfPairProductTopologyObligation_iff`: the missing reverse inequality is
literally continuity of the identity from the ordinary LF product into the
pair-stage final topology.  It also proves
`convolutionContinuous_lfPairFinal` unconditionally.  Thus the pair-stage
final topology is a correct domain topology for joint convolution.  Upgrading
that result to the ordinary product requires the identity-continuity theorem.
Pinned Mathlib supplies neither that comparison nor a theorem showing it
fails specifically for this test-function LF space, so this iteration does
not claim a counterexample that has not been formalized.

The comparison now also has a literal neighborhood criterion.
`LFPairProductNeighborhoodCriterion` says that every pair-stage-final
neighborhood of every \((f,g)\) contains a rectangle \(V\times W\) of
ordinary LF neighborhoods of \(f\) and \(g\).
`lfPairProductTopologyObligation_iff_neighborhoods` proves this equivalent to
the reverse topology inequality.  This is the first-principles point at which
a proof or counterexample must act; checking only neighborhoods of zero would
silently assume that the raw pair-final topology is already a topological
group, which has not been established.

The bounded-set statement needed for a safe fallback is now formalized on
every compact-support stage.
`supportedIn_isVonNBounded_iff_seminorm_bounded` identifies von Neumann
bounded subsets of \(\mathcal D_K\) with families uniformly bounded in every
derivative seminorm.  The two
`supportedConvolution_uniform_*_of_isVonNBounded` theorems then give uniform
convolution estimates when either argument ranges over such a bounded set;
this is the concrete fixed-stage hypocontinuity statement.  The predicate
`TestFunction.IsStageVonNBounded` records LF families carried by one bounded
stage, and Lean proves that every such family is LF-bounded.  The converse
(``every LF-bounded subset is contained and bounded in one stage'') is the
regularity theorem for the strict LF space \(\mathcal D\).  It is absent from
the pinned API, so global bornological hypocontinuity is not overclaimed.
`TestFunction.StrictLFBoundedRegularity` now names exactly this converse, and
`strictLFBoundedRegularity_iff` proves that it is equivalent to equality of
LF-bounded and stage-bounded families.
These fixed-stage estimates are already the estimates used by the restricted
Weil positivity results; full continuity of the quadratic map still requires
either the product comparison or the missing strict-LF regularity theorem.

Reflection itself is now closed.  `supportedReflectionPrecompLM` maps
\(\mathcal D_K\) to \(\mathcal D_{-K}\);
`supportedReflectionSeminormBound` proves every defining seminorm decreases,
using Mathlib's iterated-derivative formula for \(x\mapsto-x\).
This yields `supportedReflectionPrecompCLM`,
`supportedInvolutionContinuous`, and finally the unconditional LF theorem
`involutionContinuous`.  Consequently joint convolution continuity is now
the only topological assumption in
`convolutionSquareContinuous_of_convolution`.

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
* the exact finite von Mangoldt truncations converge to
  \(-\zeta'/\zeta\) on \(\Re s>1\);
* every finite von Mangoldt truncation plus its exact tail equals
  \(-\zeta'/\zeta\);
* bounded pointwise-convergent smoothings of those terms converge to the same
  logarithmic derivative by dominated convergence;
* the exact \(\Gamma_{\mathbb R}\) logarithmic derivative equals the normalized
  digamma expression on \(\Re s>0\), including its value at \(s=1\);
* quantitative Gamma-factor approximations converge from their error bounds;
* positive-integer digamma values have explicit harmonic formulas and
  harmonic--log error bounds tending to zero;
* the finite prime tail and exact Gamma stage at \(s=2\) give a genuine
  completed-log-derivative component identity;
* multiplicity-aware symmetric-pair truncations converge and are invariant
  under bijective re-enumeration under an explicit summability hypothesis;
* arbitrary exhausting symmetric zero windows have a common limit under
  normal convergence;
* a summable majorant implies normal zero-orbit convergence;
* finite shell count-times-decay summability implies normal zero-orbit
  convergence;
* a sourced quadratic multiplicity shell-count obligation plus fourth-power
  shell decay implies normal convergence and arbitrary-window regularization;
* bounded positive-height nontrivial-zeta-zero windows are finite, carry
  Mathlib analytic-order multiplicities, and define a monotone actual count;
* the exact explicit Riemann--von Mangoldt estimate on that actual count
  implies a precise unit-shell multiplicity envelope;
* the earlier abstract main/error obligation implies its shell envelope and
  corresponding normal-convergence reduction;
* exponential weighting identifies every fixed vertical transform line with
  a Schwartz Fourier transform and proves decay of every polynomial order;
* explicit integrable derivative majorants prove nonzero strip-uniform rapid
  decay of every polynomial order for every compactly supported smooth test;
* a uniform weighted-Fourier seminorm bound gives strip-uniform decay with
  the same explicit constant;
* the restricted admissible explicit formula passes to the limit from its
  finite identities and summability hypotheses;
* the sharper restricted Guinand--Weil package combines actual prime and
  archimedean targets with zero and pole limits;
* the shell-based restricted package derives its zero convergence from
  explicit count and decay hypotheses before combining all component limits;
* for a convolution-square test whose modeled zero orbits are all fixed,
  `RestrictedGuinandWeilShellData.explicitFormula_re_nonneg` combines the
  shell summability and finite stage identities into nonnegativity of the
  real part of the restricted infinite arithmetic formula;
* the fixed-orbit Boolean hypothesis is unnecessary:
  `symmetricZeroRegularized_convolution_nonneg_of_re_zero` handles either
  branch whenever representatives have zero real part, and the corresponding
  restricted formula theorem uses only that critical-line equation;
* paired zeta-zero equations rule out trivial zeros, so
  `ZetaZeroShellCountObligation.representative_re_zero_of_rh` derives the
  centered critical-line equation directly from Mathlib's
  `RiemannHypothesis`; this yields the restricted RH-to-positivity theorem
  `explicitFormula_re_nonneg_of_rh`;
* finite zero/prime sums and finite formula stages are typed explicitly;
* finite critical-line spectral energies are nonnegative;
* the transform of a Hermitian reflection is the conjugate transform on the
  critical line;
* the Fourier convolution theorem proves the convolution-square transform
  factorization without an extra hypothesis;
* the finite convolution-square sum equals the critical-line energy and is
  nonnegative unconditionally;
* a normally convergent multiplicity-aware family whose symmetry orbits are
  all fixed has nonnegative regularized convolution-square zero sum;
* exact finite explicit-formula identities pass through pointwise limits;
* positivity passes through pointwise and bounded monotone limits;
* every finite Gram kernel is positive semidefinite;
* every reindexed finite Gram restriction remains positive semidefinite;
* rank-one real quadratic certificates are nonnegative;
* positivity extends from a dense subclass exactly when continuity of the
  quadratic functional in the selected topology is supplied;
* LF evaluation continuity reduces to continuity on every fixed-support test
  space, and quadratic continuity follows if convolution-square continuity is
  additionally supplied;
* continuous-linear-map distributions satisfy evaluation continuity by
  construction, while joint convolution and reflection continuity imply
  convolution-square continuity;
* fixed-support reflection is a real-linear map, and its stagewise continuity
  implies global LF involution continuity;
* the required fixed-support reflection seminorm estimate, stagewise
  continuous map, and global LF involution continuity all hold
  unconditionally;
* fixed-left convolution is real-linear and globally continuous whenever the
  fixed-support joint convolution estimates hold;
* convolution is commutative, separately LF-continuous on both sides under
  the fixed-stage estimates, and bundled as a real-bilinear map;
* iterated derivatives of convolution equal convolution with the iterated
  derivative of the right factor;
* fixed-support \(L^1\) norms and every pointwise convolution derivative are
  bounded by explicit volume-times-seminorm products;
* fixed-stage convolution lands in \(\mathcal D_{K+L}\), satisfies the
  defining target seminorm bounds, and commutes with all support inclusions;
* fixed-stage convolution is jointly continuous, so global left and right
  separate LF continuity are unconditional;
* the pair-stage final topology is bounded above by the ordinary LF product
  topology, and the reverse inequality is isolated as the sole topological
  input needed for global joint convolution and convolution-square
  continuity;
* convolution is globally continuous without assumptions from the pair-stage
  final topology, while passage from the ordinary LF product is exactly
  continuity of the identity between those two domain topologies;
* density without continuity is insufficient, witnessed by an explicit
  dense-complement counterexample on \(\mathbb R\);
* the explicit Hermitian matrix
  \(\begin{pmatrix}1&2\\2&1\end{pmatrix}\) has value \(-2\) on
  \((1,-1)\), so Hermitian symmetry alone cannot replace the exact
  arithmetic functional.

The remaining blocker is mathematical: discharge the six named obligations
above for the actual zeta zero and prime-power distributions.  Mathlib's
analytic order now gives an actual finite multiplicity count on every bounded
positive-height window, but Mathlib still has no Riemann--von Mangoldt bound,
global symmetric multiplicity enumeration, or Guinand--Weil theorem from
which the finite stages or their regularized limit could be instantiated.
The right-half-plane \(-\zeta'/\zeta\) theorem reduces engineering gaps but
does not provide symmetric zero regularization, the gamma-factor limit, or
all-test separation.  Finite Gram/SDP certificates and finite energies prove
only restrictions and cannot discharge universal all-test positivity.

Stage four sharpens this blocker: absolute summability is enough for the
formal multiplicity-aware zero truncation and bounded prime smoothing.  The
shell theorem shows precisely how Riemann--von Mangoldt growth plus rapid
test-transform decay would establish it.  Fixed-line transform decay is now
proved, and the exact source estimate and admissible strip-seminorm class are
formalized.  Strip-uniform rapid decay is now proved with an explicit
derivative-integral constant.  The remaining zero-sum analytic input is the
actual global Riemann--von Mangoldt bound; pinned Mathlib exposes neither that
theorem nor the zeta argument-principle contour estimates needed to derive it.
The repository-wide audit also found no argument-principle/winding-number
zero-count theorem to bridge the available zeta functional equation and Gamma
APIs to a global count.  Consequently the sourced
`ActualRiemannVonMangoldtBound` remains an uninhabited proposition, not a
hidden assumption.
Likewise, extending positivity from a restricted class requires an actual
LF-density theorem and continuity of the completed zeta quadratic functional.
No contradiction with RH is found; these are precisely missing analytic
inputs, so no universal positivity--RH implication is claimed.

No nontrivial concrete test subclass can yet instantiate the complete
restricted package: doing so would require at least one genuine finite
Guinand--Weil identity and the source-normalized digamma integral/series
convergence.  Defining pole or zero terms to force the identity would be
circular and is intentionally rejected.

The strongest exact finite arithmetic theorem therefore remains
`finiteCompletedLogDerivAtTwo`: it combines the genuine finite von Mangoldt
sum and tail with the exact Gamma-factor normalization.  Actual finite zero
windows cannot be added to that equality without the missing argument
principle/explicit formula; no tautological residual term is introduced.
The new fixed-stage convolution bounds close the analytic estimate requested
for joint continuity, including target support and inductive-limit
compatibility and genuine stagewise joint continuity, but Mathlib's
one-variable LF universal property still does not provide continuity of
bilinear maps out of the LF product.  Separate continuity alone does not
supply the now-explicit reverse topology inequality.  No global
Riemann--von Mangoldt theorem, contour zero-count theorem, or finite
Guinand--Weil identity appeared in the pinned-source audit, so those analytic
targets remain genuine rather than silently postulated.

The converse direction is unchanged: the new RH-to-restricted-positivity
theorem proves only the easy spectral-energy direction for a modeled,
normally convergent family.  Restricted positivity does not identify all
nontrivial zeros or separate an off-line zero.  A converse still requires the
full explicit formula, complete multiplicity-aware zero coverage, and an
all-test separation theorem; none follows from the restricted package.
The latter is now typed exactly as
`ConvolutionSquaresSeparateOffCriticalOrbits`: an off-critical representative
must admit a normally convergent convolution square for which the *total*
regularized spectral value is negative.  This is stronger than interpolation
at one zero because every other orbit remains in the sum.
`representative_re_zero_of_convolutionSquarePositive` proves that this
separation property plus all-square spectral positivity forces every modeled
representative onto the centered critical line.  Constructing the separating
tests remains a genuine analytic blocker.

There is nevertheless concrete progress on admissible separators.
`normalizedBumpTest` is built from Mathlib's normalized smooth bump and has
integral one.  Complex exponential modulation gives `spectralShiftTest`, with
the exact identity
\[
 \widehat{\operatorname{shift}_z f}(w)=\widehat f(w-z).
\]
Consequently `pointNormalizedTest z` is compactly supported and smooth with
transform exactly one at \(z\).  For finite orbit sets,
`symmetricZeroFinsetSum_re_neg_of_target` proves a robust approximate
separation lemma: a target contribution more negative than a norm bound for
all remaining contributions makes the entire finite spectral sum negative.
`FiniteFormulaData.arithmeticSide_re_neg_of_exact` transfers such negativity
through an exact finite formula, thereby retaining the prime-power, pole, and
Gamma terms rather than dropping them.

Point normalization is not enough by itself.  The current abstract orbit
structure permits multiplicity zero; `zeroMultiplicityOffCriticalModel` and
`pointNormalization_without_positiveMultiplicity_counterexample` exhibit an
off-critical representative with a point-normalized test but identically zero
spectral functional.  The corrected converse condition is
`ConvolutionSquaresSeparatePositiveOffCriticalOrbits`, together with positive
multiplicity of every represented orbit.
`representative_re_zero_of_positiveMultiplicity_separation` proves the
resulting implication.  What remains is simultaneous finite interpolation
and uniform tail control for actual positive-multiplicity zeta zeros.

Finite simultaneous interpolation now has an explicit nonsingular basis.
`translateTest` preserves the test class and satisfies
\[
 \widehat{\tau_a f}(z)=e^{az}\widehat f(z).
\]
For translates by \(0,a,\ldots,(n-1)a\),
`translatedBasisMatrix_eq_diagonal_mul_vandermonde` factors the evaluation
matrix as
\[
 \operatorname{diag}(\widehat f(z_i))\,
 \operatorname{Vandermonde}(e^{az_i}).
\]
`translatedBasisMatrix_det_ne_zero_iff` therefore characterizes the exact
degeneracies: either the common base transform vanishes at one selected point,
or two exponential nodes collide.  Under the complementary hypotheses,
`transform_translatedInterpolatingTest` realizes arbitrary finite transform
data.  The two selection conditions are now discharged:
`exists_test_transform_ne_zero_on_finset` avoids finitely many bad complex
coefficients to construct one common base, while
`collisionFreeTranslationStep_nodes_injective` gives an explicit positive
translation step whose exponential nodes are distinct for every injective
finite point family.  Consequently `exists_translatedInterpolatingTest`
proves unconditional interpolation on arbitrary distinct finite points.
`spectralSymmetryClosure` records simultaneous closure under
\(z\mapsto-\bar z\).

The off-line convolution-square bridge is also no longer an obligation.
`transform_convolution` and `transform_involution` prove for every complex
\(z\)
\[
 \widehat{f*f^\star}(z)
   =\widehat f(z)\overline{\widehat f(-\bar z)}.
\]
For a reflection-closed finite family,
`exists_convolutionSquare_transform_eq` preserves all prescribed values.
In particular, `exists_convolutionSquare_negative_on_offCritical_pair`
constructs a square with value \(-1\) at both members of any non-fixed pair,
and `exists_convolutionSquare_selectedPairTerm_neg` turns positive
multiplicity into a strictly negative selected orbit contribution.
`norm_transform_convolution_involution_strip_pow_le` proves that the same
square retains strip-uniform rapid decay, with doubled power and squared
constant.

The limiting tail step is also explicit.  The quantitative finite lemma
`symmetricZeroFinsetSum_re_lt_of_target` retains a chosen negative margin.
`symmetricZeroRegularized_re_neg_of_tail_domination` proves that a uniform
residual norm bound preserves this margin in the regularized limit.
`residualOrbitNormMass` is the total norm mass away from one selected orbit;
normal convergence bounds every erased finite truncation by this number.
Finally,
`symmetricZeroRegularized_re_neg_of_selected_dominates_majorant` accepts any
summable nonnegative count-times-decay majorant for the residual orbits.
The shell-count step is now quantitative.  `shell_norm_sum_le` bounds each
complete multiplicity-weighted orbit shell by the sourced quadratic
Riemann--von Mangoldt envelope times the transform decay.
`exists_envelope_tail_lt` proves that every summable such envelope has an
arbitrarily small shifted height tail.  Reindexing through the unique shell
partition gives `residualOrbitNormMass_le`, and
`symmetricZeroRegularized_re_neg` combines this bound with a selected negative
orbit into an actual negative regularized spectral value.  The remaining
separation input is to use the finite interpolator on a convolution square so
that its low-shell values vanish and its selected off-line contribution has
the required sign while retaining the rapid-decay envelope.

The source audit still finds no strict-LF bounded-set regularity theorem (the
test-function file explicitly notes that even topological embedding of each
fixed-support stage is not yet in Mathlib), no proved zeta argument-principle
count inhabiting the sourced shell obligation, and no Guinand--Weil identity.
The correct unconditional bornological substitute is now proved:
`TestFunction.IsStageVonNBounded.convolutionSet` sends two families bounded in
fixed compact-support stages to a bounded family in the summed-support stage.
`convolutionSet_isVonNBounded_of_strictLF` shows exactly how the unavailable
strict-LF regularity theorem would upgrade this to all LF-bounded sets.

For actual zeros, `ActualRiemannVonMangoldtBound.shell_weighted_decay_le` and
`.exists_envelope_tail_lt` connect the published main-plus-error envelope to
weighted shell mass and arbitrarily small tails.  The proposition
`ActualRiemannVonMangoldtBound` remains uninhabited because no pinned
argument-principle theorem proves the published estimate.  Even assuming it,
the remaining separation issue is quantitative: interpolation annihilates
every chosen finite low window, but the present determinant construction does
not bound its strip-decay constant uniformly as that window grows.  Such a
bound is needed to ensure the resulting tail is smaller than the fixed
negative selected contribution.  The finite/full Guinand--Weil identity
likewise remains absent.

The interpolation-growth issue is now quantitatively exposed rather than
hidden behind existence.  `translatedInterpolationCoefficients` records the
inverse-Vandermonde coefficients, and `translatedInterpolationWeight` is
\[
 \sum_j |c_j|\exp(|a j|/2).
\]
`norm_transform_translatedInterpolatingTest_strip_pow_le` proves that the
strip-decay constant is at most the base constant times this weight.
`norm_transform_translatedInterpolatorSquare_strip_pow_le` gives the
corresponding square bound with doubled decay power and the explicit constant
\((C_{\rm base}W)^2\).  Thus the exact remaining diagonal estimate is clear:
for growing zero windows, this condition-number-sensitive factor times the
RvM envelope tail must tend to zero.  Distinctness alone gives no uniform
lower spacing or Vandermonde condition-number bound, so such convergence
cannot be inferred from the present abstract orbit assumptions.

`RVMEnvelopeSeparatesPositiveOffCriticalOrbits` packages precisely the
non-circular diagonal domination statement.  Once inhabited,
`ZetaZeroShellCountObligation.convolutionSquaresSeparate` produces genuine
normally convergent infinite separators.  Finally,
`representative_re_zero_of_explicitFormula_and_rvmEnvelope` assembles the
whole converse from that separator, positive multiplicity, a genuine
Guinand--Weil identity, and arithmetic-side positivity.

Strict-LF regularity is not required merely to regard an individual
separator as admissible: `TestFunction.isStageVonNBounded_singleton` and
`convolutionSquare_isStageVonNBounded` place every test and every convolution
square in a concrete bounded compact-support stage.  Strict-LF regularity
remains necessary only for upgrading the bornological statement uniformly to
arbitrary LF-bounded families.

The hoped-for conditioning bound cannot follow from distinctness and an RvM
count alone.  `translatedBasisMatrix_finTwo_det` proves the exact two-point
formula
\[
 \det A=\widehat f(z_0)\widehat f(z_1)
   (e^{az_1}-e^{az_0}).
\]
The distinct pairs `nearCollisionSpectralPair N = {0,i/(N+1)}` then satisfy
`norm_translatedBasisMatrix_nearCollision_det_le`:
\[
 \|\det A_N\|\le C_{f,0}^2\,\frac{2|a|}{N+1}
 \quad (|a|\le N+1).
\]
Thus determinants approach zero already for two distinct points in the
critical strip.  Zero-count upper bounds do not prohibit such near
collisions, and no unconditional lower spacing theorem for zeta zeros is
available.  Any uniform or subexponential inverse-Vandermonde estimate must
therefore assume/prove quantitative spacing (or use a basis with additional
structure); it cannot be extracted from RvM.

On the count side, `actualZetaZeroShellWeightedMass` is the concrete
analytic-order multiplicity mass in an actual positive-height zeta shell.
Assuming the sourced `ActualRiemannVonMangoldtBound`,
`.exists_actual_shell_tail_lt` proves that every summable
main-plus-error-weighted decay envelope gives a summable actual shell tail
smaller than any prescribed positive epsilon.  This completes the
count-to-tail reduction without solving the independent conditioning
problem.

No paradox appears.  The remaining blockers are genuine: the pinned library
contains no argument-principle/Riemann--von Mangoldt theorem, no strict-LF
bounded-set regularity result for test functions, and no residue-theoretic
Guinand--Weil identity.  The near-collision theorem additionally shows that
RvM alone is mathematically insufficient for the requested uniform
conditioning step.

The separator route now avoids global interpolation entirely.
`spectralZeroFactor f a q` is the compact-support finite difference
\[
 f-e^{-aq}\tau_a f,
\qquad
\widehat{\operatorname{spectralZeroFactor}}(z)
 =(1-e^{a(z-q)})\widehat f(z).
\]
Iterating it as `spectralZeroProduct` inserts any finite list of exact
spectral zeros.  `localSpectralSeparator` normalizes the resulting product at
one target; its denominator is the explicit product of local gap factors, so
near collisions are visible rather than hidden in a matrix inverse.

`localPairSeparator` combines two such kernels.  It has transform \(1\) at a
chosen \(z\), \(-1\) at \(-\bar z\), and zero at every listed nearby point.
For every duplicate-free finite target/partner/neighbor list,
`exists_pairLocalSeparatorGap` constructs a positive translation scale with
all required factors nonzero.  Thus multiple zeros cause no extra
conditioning loss: analytic multiplicity remains a scalar weight, while
distinct nearby locations are annihilated once each.

After convolution-square factorization, both selected values are exactly
\(-1\), listed neighboring orbit terms are exactly zero, and
`localPairSeparator_selectedPairTerm` gives the selected contribution
\(-2m_i\).  Distant points retain arbitrary strip-uniform polynomial decay by
`norm_transform_localPairSeparator_square_strip_pow_le`.

This iteration also corrects an overly strong earlier tail package.  Its
shell envelope included the selected term itself, making the intended
strict-domination inequality unusable for an exact negative square.
`residualOrbitNormMass_le_excluding` instead bounds only \(j\ne i\), and
`symmetricZeroRegularized_re_neg_excluding` transfers this corrected bound to
the regularized sum.  The assembled
`localPairSeparator_regularized_re_neg` now requires precisely that the
residual RvM-weighted tail be smaller than \(2m_i\).

The remaining analytic gap is quantitative but local: one must choose a
finite zero neighborhood from actual zeta local discreteness, prove normal
convergence, and show the resulting fixed separator's residual shell envelope
is below \(2m_i\).  Actual RvM, Guinand--Weil, and strict-LF regularity remain
unproved in the pinned library.  No paradox is found.

The actual distant-tail estimate is now closed conditional only on the
published RvM statement itself.  Lean proves the elementary coarse bound
\[
 M(T)+E(T)\le 3T^2\qquad(T\ge3)
\]
for the exact main and error expressions recorded above.  Consequently
`summable_fourthPower_envelope` proves summability against
\(C/(k+2)^4\), and `exists_fourthPower_tail_lt` gives an explicit shifted tail
below every positive epsilon without adding a separate summability
hypothesis.

`actualZetaZeroTransformShellMass` sums
\[
 m(\rho)\,\left|\widehat f(\rho-\tfrac12)\right|
\]
over each actual positive-height zeta shell.  Strip-uniform fourth-power
decay proves `actualZetaZeroTransformShellMass_le`; combining this with actual
analytic-order multiplicities and RvM yields
`exists_actual_transform_tail_lt`.  Its multiplicity corollary makes the
complete distant tail strictly smaller than any selected positive
multiplicity \(m\).

Low zeros are represented concretely by `centeredZetaZeroWindowFinset`.
`localCenteredZetaNeighbors` removes the selected target and partner and lists
every other distinct centered location.  The list is duplicate-free, so the
matrix-free local-gap construction applies and annihilates this complete
finite window.

The local square is unconditionally stage-admissible through
`localPairSeparatorSquare_isStageVonNBounded`; strict-LF bounded-set
regularity is not needed for this individual test.  Finally,
`localPairSeparator_contradicts_formula` packages the exact conditional
contradiction between its negative regularized spectral value and a genuine
Guinand--Weil identity with nonnegative arithmetic side.

The remaining bridge is not tail analysis: it is formal identification of
the abstract symmetric orbit enumeration with the actual centered zero
windows (including reflection and multiplicity), an inhabitant of
`ActualRiemannVonMangoldtBound`, and the genuine residue-theoretic
Guinand--Weil formula.  Global strict-LF regularity remains absent but no
longer blocks admissibility of the constructed separator.

The actual symmetry and location bridge is now closed.  In the open strip,
`riemannZeta_eq_zero_iff_completedRiemannZeta_eq_zero_of_strip` identifies
zeta and completed-zeta zeros using nonvanishing of `GammaR`, and
`riemannZeta_one_sub_eq_zero_of_strip` proves the functional-equation
reflection.  In centered coordinates this is `z ↦ -z`.
`nontrivialRiemannZetaZeros_countable` and
`actualZetaZeroCenteringEquiv` give an exact countable equivalence between
actual strip zeros and their centered locations, while
`actualCenteredZetaMultiplicity_centering` preserves analytic-order
multiplicity by definition.

`riemannZeta_conj` analytically continues the real-coefficient Dirichlet
series identity from `Re(s)>1` across `ℂ \ {1}`.  Iterated derivative
conjugation then proves `zetaZeroMultiplicity_star`.  The completed-zeta
factorization through the entire inverse `GammaR` factor and the affine
functional equation prove both conjugation and `s ↦ 1-s` multiplicity
invariance.  Therefore `actualCenteredNontrivialZetaZeros_neg_star` and
`actualCenteredZetaMultiplicity_neg_star` establish the complete centered
quartet symmetry `z ↦ -conj z`.  Partner membership and partner multiplicity
are now theorems derived from `ExactActualZetaOrbitModel`, not fields assumed
by it.

`nontrivialZetaZerosInCriticalStrip` also discharges the global strip
classification.  The right boundary uses Mathlib's closed-half-plane
nonvanishing.  On the left, completed-zeta reflection makes
`Λ(1-s)` nonzero; a hypothetical zeta zero then forces `GammaR(s)=0`, whose
exact zero classification leaves only the excluded negative even integers.
Thus `ExactActualZetaOrbitModel.riemannHypothesis` no longer needs a separate
strip assumption.

The actual orbit index is now constructed rather than postulated.
`ActualPositiveCenteredZetaZero` restricts to positive height, and
`actualPositiveCenteredPartner` is the exact involution `z ↦ -conj z`.
Its quotient `ActualZetaZeroOrbit` is countable.  Fixed points are exactly
the critical-line zeros; nonfixed classes are the two positive-height
members of an off-line quartet.  `actualZetaZeroOrbitMultiplicity` descends
analytic multiplicity to this quotient.  The partial natural enumeration
`actualZetaZeroOrbitDecode` is complete and injective on used codes, so it
does not silently duplicate an orbit if the zero type is finite.
`actualZetaZeroOrbitWindow_finite` proves local finiteness directly from
Mathlib's finite positive zero windows.

The argument-principle input has now been reduced to a precise quantitative
boundary estimate.  `riemannXi` is the pole-cancelled entire function
`s(s-1)Λ₀(s)+1`; it satisfies `riemannXi_one_sub` and agrees with
`s(s-1)Λ(s)` away from `0,1`.  Open-strip zero locations and analytic
multiplicities are transported exactly by
`riemannXi_eq_zero_iff_riemannZeta_eq_zero_of_strip` and
`riemannXiZeroMultiplicity_eq_zeta`.  Analytic uniqueness plus the explicit
nonzero value at `2` proves every xi analytic order finite.

Two exact contour precursors are closed.  First,
`riemannXi_logDeriv_local` proves the local logarithmic-derivative normal form
with residue equal to analytic multiplicity.  Second,
`riemannXi_circleAverage_log_norm_eq_divisor_sum` instantiates Mathlib's
Jensen formula for xi, giving an exact finite divisor/count identity on every
nonzero-radius circle.  This avoids postulating an argument principle, but a
quantitative RvM estimate still needs a complex-Gamma Stirling bound and
vertical zeta growth estimate for the boundary average.

The puncture contribution is now globalized over every finite zero window.
`XiPunctureData` extracts a positive closed disk at each xi point on which
the regular factor is analytic and nonvanishing, xi has no other zero, and
the logarithmic-derivative decomposition holds uniformly.  For every smaller
circle, `XiPunctureData.circleIntegral_logDeriv_eq` computes the integral as
`2πi` times multiplicity.  The weighted theorem
`XiPunctureData.circleIntegral_mul_logDeriv_eq` inserts any transform analytic
on the disk and returns `2πi mρ h(ρ)`.  The two `riemannXi_finset_sum_*`
theorems sum these identities over independently chosen punctures.  Thus
puncture shrinking and the finite weighted local residue sum no longer
require a limiting argument.

The quantitative count step downstream of such a boundary estimate is now
formalized.  `riemannXi_sum_divisor_le` applies Mathlib's Jensen inequality:
an explicit outer-circle bound `‖ξ(z)‖ ≤ M` gives an explicit upper bound for
the inner multiplicity-weighted divisor count.  `riemannXi_divisor_apply`
proves every xi divisor coefficient equals `riemannXiZeroMultiplicity`, and
`riemannXi_finset_divisor_sum_eq_multiplicity` identifies finite divisor sums
with finite multiplicity sums.  Thus no argument-principle theorem remains
between a valid boundary growth bound and a quantitative xi zero count.

The converse endpoint is correspondingly strengthened:
`riemannHypothesis_of_exactActualOrbits_formula_rvm` combines exact orbit
completeness, the shell/RvM separator, a genuine formula, and nonnegative
arithmetic side to produce Mathlib's `RiemannHypothesis`.

No paradox was found.  The exact xi divisor identity is now available, but
the pinned library has factorial Stirling asymptotics rather than the needed
uniform complex-Gamma boundary asymptotic, and supplies no suitable vertical
zeta bound.  Repository-wide source inspection found no Li/Weil growth lemma
that supplies either estimate.  Thus the proved Jensen inequality cannot yet
be instantiated with the RvM-scale `M`, and it does not yield an inhabitant of
`ActualRiemannVonMangoldtBound`.  The complete finite weighted sum of inner
puncture integrals is proved, but Mathlib has no rectangle/punctured-domain
residue theorem identifying the outer contour with the sum of those inner
circles.  That outer-minus-inner contour decomposition, followed by the
prime/Gamma side evaluation, still blocks finite Guinand--Weil.
Consequently the existing separator/tail implication cannot yet be
instantiated on the actual orbits.
