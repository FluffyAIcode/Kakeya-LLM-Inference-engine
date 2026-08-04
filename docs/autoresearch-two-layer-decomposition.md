# Architecture 9 two-layer decomposition

Architecture 9 separates decomposition into two trust layers.

## Exploration layer

`DECOMPOSE_TO_SUBPROBLEMS` opens a bounded
`DecompositionExplorationContract`. The existing Decomposer produces 8–16
independent natural-mathematics memos across fixed information-gain categories.
The memos are private, content-addressed, and advisory. A Host-owned static
analyzer may consume them, but free text is never copied into an authoritative
artifact. Public state contains only short candidate IDs, memo hashes,
registered metadata IDs, rankings, and typed rejection codes.

The Host prefilter requires no Lean proof. It rejects duplicate, no-go,
disconnected, incomplete, or assumption-bearing metadata before selecting a
ranked formalization queue. Exploration cannot mutate the proof ledger or enter
Proof Search.

## Certification layer

Each queued candidate must map through a registered Host mapper to a Typed
Candidate Intent. The Host deterministically renders a statement-only Lean
declaration and elaborates it while the candidate remains provisional.

An elaborated candidate becomes eligible for the existing proof pipeline only
after a separate child-to-parent reduction theorem is verified. Public
assumptions must match, strict reduction and non-circularity checks must pass,
and Critic and Judge must accept. Only then may the existing idempotent commit
path add the child to the ledger. OProver remains exclusive and on-demand for
the reduction proof; it is never loaded for exploration or statement
formalization.

## Candidate representation analysis

An unmappable candidate enters `CANDIDATE_REPRESENTATION_ANALYSIS` before
rejection. The Host persists a content-addressed `RepresentationGapReport`
bound to the candidate, target, context, mapper capability registry, and Lean
environment. Reports contain only closed registry IDs and safe relation codes;
private memo prose and model-proposed registry entries are forbidden.

Representation gaps route independently of mathematical budgets:

- a missing mapper pattern uses `MAPPER_EXTENSION_REQUIRED`;
- an available trusted symbol uses `REGISTRY_RESOLUTION`;
- a genuinely absent concept uses autonomous definition resolution;
- a hidden assumption, target restatement, stronger parent, or disconnected
  analogue receives an evidenced semantic rejection;
- an uninterpretable candidate uses `REPRESENTATION_EXHAUSTED`.

A candidate may be retried once only after its mapper or environment hash
changes. Representation analysis cannot elaborate, invoke OProver, or mutate
the ledger. If every candidate is rejected, the Host persists a
content-addressed representation-exhaustion certificate and performs a typed
backjump.
