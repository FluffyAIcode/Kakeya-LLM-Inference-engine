# Formal RH route index

This integration is a common, non-runtime base for three independent formal
research routes. It is stacked on PR #236 at
`d4042140f10e0fd231bdfbcf814d068487b15f82`. None of the routes proves the
Riemann Hypothesis, and no finite calculation is promoted to an all-index or
all-test theorem.

## Integrated source snapshots

Each source worktree was uncommitted on the same base commit above. The bundle
hash is SHA-256 over the sorted intentional route paths, with each relative
path and file content separated by a NUL byte.

| Route | Source branch | Bundle SHA-256 |
| --- | --- | --- |
| Jensen / Laguerre--Pólya | `agent/rh-jensen-laguerre-polya-0730` | `cb9fa16ba89397fdbc2e269bceb7b6fe6559b871ffa36118a9a307ad316fded3` |
| Li criterion | `agent/li-criterion-route-0730` | `ef486b9b9dd69ff3e330bb08cb75a46c16dad1ce5c8686a71f1c3283ac6b7528` |
| Weil positivity | `agent/weil-positivity-rh-0730` | `5ad75565ea9b4f4a25b932ae9f1efadd5283a2693b73e1ce62352f97c7ac21c9` |

Exact bibliographic citations, normalizations, source locations, and pinned
Mathlib revision are preserved in:

- `docs/research/rh-jensen-source-cards.json`
- `docs/source-cards/li-criterion.md`
- `docs/weil-positivity-sources.yaml`

## Proven finite support and open bridges

### Jensen / Laguerre--Pólya

Proved: the project xi normalization is entire and symmetric; the centered
generating function is entire and even; its Mathlib Taylor expansion; exact
degree-zero, degree-one, and degree-two Jensen formulas; degree-one
hyperbolicity; and the nondegenerate degree-two Turán/root and hyperbolicity
lemmas.

Open: coefficient reality and identification with the sourced even Taylor
series, the Pólya--Jensen implications, and
`AllJensenHyperbolic xiGamma`. Known eventual hyperbolicity and bounded-degree
results do not prove the required all-degree/all-shift statement.

### Li criterion

Proved: the Li-normalized xi endpoint values, entirety and functional
equation; elementary coefficient identities; finite-prefix positivity
properties; zero-summand and finite-sum identities; and a formal
`finitePrefixSpoof` counterexample showing that any fixed positive prefix need
not imply all-index positivity.

Open: coefficient reality, multiplicity-correct zero enumeration,
symmetric-height convergence, the derivative/zero-sum identity, and Li's
all-`n` nonnegativity equivalence with Mathlib's RH proposition.
`RiemannLiCriterionStatement` remains an unproved proposition.

### Weil positivity

Proved: the smooth compactly supported test interface, involution laws,
convolution closure, finite Gram positive-semidefinite certificates and their
restrictions, and an explicit Hermitian kernel with a negative direction.

Open: construction of the zeta distribution, the normalized
Guinand--Weil explicit formula and zero-sum control, and positivity for every
admissible test iff RH. `BridgeObligation` is only the unproved interface;
finite PSD certificates cannot discharge it.

## Research priority

1. Build shared analytic infrastructure without asserting a criterion:
   normalization comparison, real-valued coefficient lemmas, and
   multiplicity-aware nontrivial-zero indexing.
2. Formalize one source theorem at a time behind the existing explicit bridge
   propositions: Pólya--Jensen, Li, or Guinand--Weil.
3. Keep finite computations as falsifiable support tests only. Do not infer an
   all-degree, all-index, or all-test statement from a finite restriction.

The route-specific status documents contain the detailed next obligations:
`docs/research/rh-jensen-route.md`,
`docs/research/li-coefficients-route.md`, and
`docs/weil-positivity-route.md`.
