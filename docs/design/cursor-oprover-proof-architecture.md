# Cursor Strategy + OProver Proof Architecture

Architecture/checkpoint version 9 is a direct cutover. Legacy model Strategy
and proof Generator states are audit-only and have no executable transition.

## Trust boundaries

1. `CursorStrategyAdapter` receives a sanitized evidence snapshot in a
   disposable, read-only directory. Its private memo is not persisted.
2. The Host Intent Compiler accepts exactly one registered plan ID from the
   memo. The host owns the typed Strategy/Synthesis/Research Contract.
3. `OProverProofAdvisor` receives only an elaborated theorem ID, Lean
   goal/context, theorem-card refs, and optional OProofs refs.
4. Every OProver candidate runs in an isolated temporary Lean environment.
   Only verified hashes and registered action IDs enter `ProofAdvice`.
5. The host stepwise controller and Lean remain authoritative. Gemma is
   restored only after OProver unload and acts as an advisory critic.
6. Adversarial/Host/Judge may commit only host-validated artifacts.

No model response may directly mutate the proof ledger, production worktree,
assumptions, Lean source, or content-addressed artifact store.

## Required configuration

Set secrets outside the repository:

```sh
export KAKEYA_CURSOR_STRATEGY_MODEL='<ID returned by Cursor.models.list()>'
export KAKEYA_OPROVER_MODEL_DIR="$HOME/kakeya-models/oprover-8b-mlx-q4-cd9ffd"
export KAKEYA_OPROVER_REVISION='cd9ffd383b584d95bf00e04b88b35b05928b211c'
export KAKEYA_OPROVER_QUANT='q4'
```

Store the Cursor SDK credential in macOS Keychain as a generic-password item
with service `ai.kakeya.cursor-sdk` and account equal to the macOS user. The
adapter retrieves it over a captured pipe; it is never placed in a process
argument, status payload, or log. Model discovery must run before setting
`KAKEYA_CURSOR_STRATEGY_MODEL`.

The model ID is deliberately not prescribed. Runtime discovery must confirm
that the configured account can use it. Missing Cursor configuration produces
`STRATEGY_PROVIDER_UNAVAILABLE` and no Gemma fallback.

## Pinned OProver source

- Repository: `m-a-p/OProver-8B` (final Round 3 checkpoint)
- Revision: `cd9ffd383b584d95bf00e04b88b35b05928b211c`
- License: Apache-2.0
- Architecture: `Qwen3ForCausalLM`
- Parameters: 8,190,735,360 BF16
- Hub source storage: 16,393,509,991 bytes

Download must pin the revision and preserve a checksum manifest. Prepare MLX
Q5 first only when production preflight shows enough wired-memory/Metal
headroom; otherwise prepare Q4. OProofs is optional and remains
`UNAVAILABLE` until separately installed.

The local acceptance run prepared Q5 first, then observed repeated isolated
Lean-candidate failures. The measured fallback Q4 bundle completed the official
prompt-template smoke and produced a candidate accepted by isolated Lean.

## Production-only residency sequence

`GEMMA_SERVING → QUIESCE_SNAPSHOT → GEMMA_UNLOAD → HEADROOM_CHECK →
OPROVER_LOAD → OPROVER_ADVISE → OPROVER_UNLOAD → GEMMA_RESTORE →
GEMMA_HEALTH_VERIFY → GEMMA_SERVING`

The scheduler uses an exclusive file lock and fsynced journal. Gemma and
OProver cache namespaces must differ. Allens remains Gemma-only and receives
no OProver KV. Ordinary CI mocks process management; it never performs a
destructive model swap.
