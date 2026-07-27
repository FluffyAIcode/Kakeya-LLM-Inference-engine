# Architecture 9 two-layer decomposition

Architecture 9 separates decomposition into two trust layers.

## Exploration layer

`DECOMPOSE_TO_SUBPROBLEMS` opens a bounded
`DecompositionExplorationContract`. The existing Decomposer produces 8–16
independent natural-mathematics memos across fixed information-gain categories.
The memos are private, content-addressed, advisory, and never parsed into an
authoritative artifact. Public state contains only short candidate IDs, memo
hashes, registered metadata IDs, rankings, and typed rejection codes.

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

If every candidate is rejected, the Host persists a content-addressed
exploration-exhaustion certificate and performs a typed Strategy backjump.
