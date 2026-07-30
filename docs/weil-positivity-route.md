# Weil positivity / energy route to RH

Status: **partial, unconditional infrastructure only**.  Nothing in this route
asserts the explicit formula or RH.

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
* Bochner convolution, compact-support closure, and smoothness of
  convolution;
* `mellin`, `MellinConvergent`, and Mellin inversion infrastructure;
* Fourier transforms and Fourier transforms of Schwartz functions;
* `Matrix.PosSemidef`, Gram matrices, and positive-semidefinite submatrices;
* `riemannZeta`, `completedRiemannZeta`, the functional equation, and the
  proposition `RiemannHypothesis`.

Absent after repository-wide API search:

* a type enumerating nontrivial zeta zeros with multiplicity;
* the symmetric/regularized sum over zeros;
* the von-Mangoldt plus archimedean Guinand--Weil explicit formula;
* a completed-zeta zero distribution;
* any theorem connecting distribution positivity to `RiemannHypothesis`.

The Lean file therefore defines the admissible domain, centred transform,
involution, convolution, a typed distribution interface, its exact quadratic
functional and positivity predicate, and the proposition
`BridgeObligation W := IsPositive W ↔ RiemannHypothesis`.  It supplies no
inhabitant of that proposition.

## Proved support and terminal blocker

The module proves, without `sorry`, `admit`, or axioms:

* involution is closed, involutive, additive, zero-preserving, and
  conjugate-linear;
* convolution is again a smooth compactly-supported test;
* every finite Gram kernel is positive semidefinite;
* every reindexed finite Gram restriction remains positive semidefinite;
* rank-one real quadratic certificates are nonnegative;
* the explicit Hermitian matrix
  \(\begin{pmatrix}1&2\\2&1\end{pmatrix}\) has value \(-2\) on
  \((1,-1)\), so Hermitian symmetry alone cannot replace the exact
  arithmetic functional.

The remaining blocker is mathematical: construct the zeta Weil distribution
with the displayed normalization, formalize and prove the full explicit
formula, control the zero sum, and prove its positivity is equivalent to
Mathlib's `RiemannHypothesis` for **all** admissible tests.  Finite Gram/SDP
certificates prove only restrictions and cannot discharge that universal
bridge.
