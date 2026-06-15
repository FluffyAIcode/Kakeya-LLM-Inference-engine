# Architecture Decision Records

This directory contains Architecture Decision Records (ADRs) for the
DLM-proposer + AR-verifier project. Each ADR captures a single architectural
decision, the context that led to it, the alternatives considered, and the
consequences of choosing one path over another.

## Why ADRs

Code shows *what* we built. ADRs show *why* — and just as importantly, *what
we deliberately rejected and on what grounds*. Without ADRs, every new
contributor (human or agent) re-derives the same decision tree from scratch
and either burns time or re-opens settled debates.

We follow a lightweight variant of the [Michael Nygard format][nygard]:
**Context → Decision → Consequences**, plus an explicit **Alternatives
considered** section because most of the value comes from showing the
reader what was *not* chosen.

[nygard]: https://github.com/joelparkerhenderson/architecture-decision-record

## Conventions

- File name: `NNNN-kebab-case-title.md` where `NNNN` is a four-digit zero-padded
  monotonically increasing number.
- Status: `Proposed` / `Accepted` / `Superseded by NNNN` / `Deprecated`.
- Once `Accepted`, an ADR is immutable except for the `Status` field.
  Disagreements are resolved by writing a new ADR that supersedes it.
- Length: aim for ≤ 5 pages of rendered markdown. If longer, split.

## Index

| #    | Title                                                           | Status   |
| ---- | --------------------------------------------------------------- | -------- |
| 0001 | [Proposer sizing, alignment, and verifier decoupling](0001-proposer-sizing-and-alignment.md) | Accepted |
| 0002 | [Verifier selection, quantization, and the open-vs-closed-weight constraint](0002-verifier-selection-and-quantization.md) | Accepted |
| 0003 | [Verifier ↔ slab pool integration: deferred refactor + intermediate step](0003-verifier-slab-pool-integration.md) | Accepted |
| 0004 | [Alignment training data preparation policy (Nemotron-informed)](0004-alignment-training-data-preparation-policy.md) | Accepted |
| 0005 | Personal layer / personal data store                            | Planned  |
| 0006 | [Project positioning as local agent infrastructure](0006-local-agent-infrastructure-positioning.md) | Accepted |
| 0007 | [Cross-request KV cache reuse for long sessions](0007-cross-request-kv-reuse.md) | Superseded by 0008 |
| 0008 | [Session-bound runtime + gRPC protocol](0008-session-bound-runtime-and-grpc-protocol.md) | Accepted |
| 0014 | [Agent-connection capacity & cross-host proposer/verifier topology: test plan & results](0014-agent-connection-capacity-and-cross-host-topology-tests.md) | Accepted |
| 0015 | [Kakeya Attention as an attention algorithm + engine substrate roadmap](0015-kakeya-attention-and-engine-substrate.md) | Accepted |

Note: ADR numbering is monotonically increasing; in-flight or
planned numbers (0005) appear in the index so readers can
see the planned shape of the decision tree even before those ADRs
are written. When an ADR moves from "Planned" to "In flight" it
gets a PR link; when it merges, the row updates to "Accepted"
with a file link.
