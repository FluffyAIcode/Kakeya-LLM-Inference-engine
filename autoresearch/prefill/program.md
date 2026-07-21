# Prefill AutoResearch Program

You are optimizing the two-Mac full-context RH proof research system.

## Ownership

- You may edit only `candidate.py`.
- Never edit `prepare.py`, benchmark reports, tests, or production metrics.
- The human owns this file.

## Objective

Use a lexicographic objective:

1. Close an unresolved Proof Obligation Ledger item when an experiment supports
   a proof step.
2. Otherwise, require a novel falsified hypothesis or a novel, strictly smaller
   proof frontier. Rewording an existing obligation is not progress.
3. Subject to (1)-(2), minimize `metric_cold_critic_prefill_s`.

## Hard constraints

- Every compute segment must remain at or below 300 seconds.
- Critic must receive the complete Generator response.
- `critic_omitted_tokens` must equal zero.
- Protocol must be `goal_anchored_recursive_gan_v3`.
- Snapshot mode must remain `final_only`.
- No fallback, local Primary Prefill, failed remote job, sampling, summary, or
  semantic simplification is allowed.
- Primary remains decode-only and allens remains Prefill-only.

## Experiment loop

1. Read `candidate.py` and `results.tsv`.
2. Let the host deterministically select the deepest unresolved leaf.
3. Modify only `candidate.py`.
4. Verify Primary and allens health without restarting either service.
5. Preserve all KV caches across decomposition iterations.
6. Run the fixed full-context acceptance workload.
7. Run `prepare.py` against the resulting report.
8. Keep the candidate only if every hard constraint passes and it closes an
   obligation, falsifies a novel hypothesis, or creates a novel smaller
   frontier. Otherwise restore the previous candidate.
9. Append the result and repeat.

Every candidate must target exactly one current unresolved leaf obligation and
contain a falsifiable hypothesis plus distinct Generator and Critic directives.
Normal iterations use the deterministic host candidate and do not call the
Strategy model. Invoke Strategy Gemma only after three consecutive
non-progressing runs, after a falsified branch, or via an explicit CLI/trigger
file request.
An unresolved Critic verdict must isolate one strictly smaller missing lemma;
the host records that lemma as a deduplicated child obligation. Completed GAN
runs, transcripts, checkpoints, and ledger updates remain durable even when the
candidate strategy is reverted or fixed evaluation fails.

Worker lifecycle and cache policy belong to the inference serving plane, not
the proof experiment. `prefill_compute_chunk_tokens` is immutable during this
loop. Never invoke launchctl, restart Primary/allens, clear KV, or run a cold
benchmark here. Model/tokenizer/quantization/rope/window/cache-format changes
must be deployed outside this supervisor. Cold benchmarks are explicit,
separate invocations of `scripts/benchmark_prefill_architecture.py`.

Prefill budgets are hard admission limits, never truncation instructions.
Strategy input must fit 8448 tokens by carrying the complete active leaf
ancestry and its exact experiment records. Generator and Critic inputs must fit
6144 tokens; the Critic always receives the complete current Generator output.
Repeated Strategy strings are interned once in `text_by_id`; `_ref` fields
losslessly reference that exact text.
If any complete semantic unit exceeds its budget, reject it before remote
Prefill and preserve the checkpoint. Never slice, sample, summarize, or drop
the tail of an over-budget input.

Obligation IDs are host-owned. Treat IDs emitted by Generator/Critic as
untrusted labels and bind verdicts to the exact current target. Never persist a
model-invented ID. Reject a proposed child when it duplicates an existing
statement or lemma name, is highly similar to an ancestor, or is too vague to
be falsifiable. A rejected cyclic frontier is `INCONCLUSIVE`, not progress.

`DECOMPOSED` is keepable only when the host actually persisted at least one
new child that passed the ID, novelty, cycle, and falsifiability gates.
Generator coverage must be complete. Before persistence, alpha-normalize
mathematical variables, extract premise/conclusion structure, compare the child
against every ancestor in both entailment directions, and require an explicit
new assumption, narrower domain, or falsifiable conclusion. Existing semantic
duplicates and all descendants beneath them are retained for audit but marked
`REJECTED_DUPLICATE`; they are not pending leaves and do not reset stagnation.

Every proposed minimum leaf must include one safe Lean theorem signature with
explicit typed variables, hypotheses, and conclusion. The host compiles it
against pinned Lean/mathlib before persistence and records `FORMALIZED` plus a
signature hash. Missing, unsafe, ill-typed, or duplicate signatures reject the
child. `FORMALIZED` is not `PROVED`: closure still requires a separate proof
with no `sorry` and no added axioms.
The supervisor prewarms Lean. Signature checks use a 45-second first attempt;
on timeout the entire Lean process group is killed, the environment is warmed
again, and one 120-second retry is allowed. Distinguish `TYPECHECK_FAILED`,
`TYPECHECK_TIMEOUT`, `UNSAFE_REJECTED`, and `ENVIRONMENT_FAILED`.

Generator/Critic decode must also make semantic progress. Three consecutive
chunks containing only whitespace or empty decoded text terminate the turn as
`semantic_stall`; never accept an unterminated partial Lean block.

Do not optimize output wording, scores, prizes, or other proof-irrelevant
content. Prefill performance is a tertiary objective after mathematical
decomposition progress, while preserving the complete semantic contract.
