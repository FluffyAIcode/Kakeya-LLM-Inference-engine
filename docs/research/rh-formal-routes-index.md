# Formal RH route index

This integration is a common, non-runtime base for three independent formal
research routes. It is stacked on PR #236 at
`d4042140f10e0fd231bdfbcf814d068487b15f82`. None of the routes proves the
Riemann Hypothesis, and no finite calculation is promoted to an all-index or
all-test theorem.

## Integrated source snapshots

The first-stage source worktrees were uncommitted on the same base commit
above. The bundle hash is SHA-256 over the sorted intentional route paths,
with each relative path and file content separated by a NUL byte.

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

Exact bibliographic citations, normalizations, source locations, and pinned
Mathlib revision are preserved in:

- `docs/research/rh-jensen-source-cards.json`
- `docs/source-cards/li-criterion.md`
- `docs/weil-positivity-sources.yaml`

## Proven finite support and open bridges

### Jensen / Laguerre--Pólya

Proved: the project xi normalization is entire and symmetric; the centered
generating function is entire and even; its Mathlib Taylor expansion; exact
degree-zero through degree-three Jensen formulas; the coefficientwise
derivative identity relating degree `d + 1`, shift `n` to degree `d`, shift
`n + 1`; the equivalence between all Jensen obligations and all nested finite
`JensenSquare` obligations; monotonicity of those squares; the conditional
reduction from all shifts to the unshifted family under derivative
preservation; degree-one hyperbolicity; and the nondegenerate degree-two
Turán/root, hyperbolicity, and positive-log-concave lemmas.

Open: coefficient reality and identification with the sourced even Taylor
series; the classical theorem that differentiation preserves real-rootedness
in the project predicate; both Pólya--Jensen implications in
`BridgeObligations`; and the all-degree unshifted family (equivalently, after
derivative preservation, `AllJensenHyperbolic xiGamma`). Known eventual
hyperbolicity, bounded-degree results, and ordinary Turán inequalities do not
prove this all-degree target.

### Li criterion

Proved: the Li-normalized xi endpoint values, entirety and functional
equation; elementary coefficient identities; finite-prefix positivity
properties; zero-summand and multiplicity-aware finite-multiset identities;
finite Li-sum nonnegativity for roots on the critical line; a
symmetric-height Cauchy/limit interface with uniqueness and approximation
transfer; exact logarithmic-derivative identities for finite zero products;
the truncated generating-polynomial recurrence; and a formal
`finitePrefixSpoof` counterexample showing that any fixed positive prefix need
not imply all-index positivity.

Open: construct the actual nontrivial-zeta-zero multiset with analytic
multiplicity; prove the symmetric-height truncations Cauchy; construct
finite xi/Hadamard approximants and transfer their logarithmic derivatives
and all Taylor coefficients to the analytic xi expression; thereby prove
coefficient reality and the derivative/zero-sum identity; then formalize Li's
all-`n` nonnegativity equivalence with Mathlib's RH proposition.
`RiemannLiCriterionStatement` remains an unproved proposition.

### Weil positivity

Proved: the smooth compactly supported test interface, involution laws,
convolution closure, the critical-line Fourier normalization, typed finite
zero/prime sums and exact finite-formula stages, nonnegative finite
critical-line energies, transfer of exact explicit-formula identities and
positivity through pointwise limits, bounded-monotone supremum limits, a
counterexample showing finite positivity without convergence does not control
an independently declared limit, finite Gram positive-semidefinite
certificates and their restrictions, and an explicit Hermitian kernel with a
negative direction.

Open: discharge the named `MellinNormalizationObligation`,
`ZeroRegularizationObligation`, `PrimePowerStabilizationObligation`, and
`ArchimedeanLimitObligation`; combine them in `AnalyticBridgeObligations` to
obtain the actual normalized Guinand--Weil formula; and separately prove both
directions in `AllTestPositivityObligation`. `BridgeObligation` is only the
unproved interface; finite stages, finite PSD certificates, and limit-transfer
lemmas without the actual zeta convergence hypotheses cannot discharge it.

## Research priority

1. Build shared multiplicity-aware nontrivial-zero indexing and
   symmetric-height regularization without asserting a criterion.
2. Discharge one named analytic bridge at a time: Jensen coefficient reality
   and derivative preservation, Li/Hadamard coefficient transfer, or the four
   finite-to-limit Guinand--Weil obligations.
3. Formalize the corresponding source criterion only behind the existing
   explicit propositions: Pólya--Jensen, Li, or all-test Weil positivity.
4. Keep finite computations as falsifiable support tests only. Do not infer an
   all-degree, all-index, or all-test statement from a finite restriction.

The route-specific status documents contain the detailed next obligations:
`docs/research/rh-jensen-route.md`,
`docs/research/li-coefficients-route.md`, and
`docs/weil-positivity-route.md`.
