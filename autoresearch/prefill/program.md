# Prefill AutoResearch Program

You are optimizing the two-Mac retained-interface RH proof research system.

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
6. Run the fixed retained-capacity certified-interface acceptance workload.
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
file request. A valid premise invalidation is also an immediate event-driven
Strategy trigger. It is informative recovery, but does not reset mathematical
stagnation as proof progress.
If an optional Strategy replan exceeds its exact-interface retained budget, defer the
replan and continue immediately with the deterministic host candidate. Never
truncate the Strategy prompt and never stop the GAN proof loop for this
control-plane admission failure.
An unresolved Critic verdict must isolate one strictly smaller missing lemma;
that free-form text is only a trigger for certified decomposition and never
creates a child directly. Completed GAN runs, transcripts, checkpoints, and
ledger updates remain durable even when the candidate strategy is reverted or
fixed evaluation fails.

## Event-driven orchestration and recovery

The persisted proof machine, not the outer loop counter, selects the next
role. Its states include `NEEDS_STRATEGY`, `GENERATOR`, `CRITIC`,
`DEFINITION_AUDITOR`, `COUNTEREXAMPLE_WORKER`, `SYNTHESIS`, `REFRAME`,
`DECOMPOSER`, the typed Math IR/Host/Lean gates, `PROOF_SEARCH`,
`ADVERSARIAL_REVIEW`, `JUDGE`, `COMMIT`, `APPROACH_FAILED`, `PREMISE_AUDIT`,
`BLOCKED`, and `IDLE`. Every committed transition records the
exact target, candidate/strategy hashes, parent statement/signature hashes,
root-goal hash, ledger identity/version, validated artifact hashes and
dependencies, retry counters, source run IDs, transition reason, resume
origin, timestamps, and whether Strategy was reused. The checkpoint and its
content-addressed artifacts are atomic private files with mode `0600`.

Protocol, schema, JSON, EOS, output-budget, and transient role failures retry
the same role within a bounded budget. Decomposer contract failures resume
Decomposer. Lean signature/elaboration failures resume Formalizer; Prover
proof failures resume Prover unless the typed error invalidates Formalizer.
Host/Judge rejection returns to the earliest invalid artifact. Premise
suspicion enters independent `PREMISE_AUDIT`. Only confirmed approach failure,
audited premise invalidation or failed rescue, target/branch invalidation,
explicit operator request, or configured mathematical stagnation may enter
`NEEDS_STRATEGY`. Infrastructure failures retain the separate circuit breaker
and never count as mathematical stagnation. Exhausted role budgets enter
`BLOCKED`; they do not silently burn another Strategy call.

Each validated certified-role artifact is reusable only when its schema/hash,
ordered dependency hashes, target, candidate, parent statement/signature,
root goal, and ledger identity/version still match. Partial or rejected
artifacts are audit-only. A restart resumes the first unresolved role and
reports its exact state; a candidate remains active until an explicit Strategy
trigger. `COMMIT` is idempotent: the deterministic child ID and certificate
hash prevent duplicate ledger children after a crash.

## Certified decomposition

Certified decomposition is the authoritative and only child-persistence path.
For an unresolved frontier, seven isolated fresh-session workers run in order:
Definition Auditor, Counterexample Worker, Decomposer, Formalizer, Prover,
Adversarial Proponent, and Judge. Every role uses allens Prefill and Primary
decode with no shared session/KV state; only explicit host-packaged artifacts
and hashes flow forward. Raw transcripts and parsed artifacts are persisted in
one private atomic manifest. Any timeout, malformed binding, or role failure
fails open without ledger mutation.

Every role artifact binds the exact target ID, parent statement hash, immutable
root-goal hash, producer role/run ID, and all upstream artifact hashes.
Decomposer labels are temporary: only the host assigns a persistent ID after
the complete certificate passes. The current artifact schema contains exactly
one `child` object and one complete `reduction_contract` object; it has no
`children`, dependency-edge, child-label-list, or reduction-label-list fields.
The contract binds the exact child label, exact parent statement, exact public
assumptions, and a substantive child-to-parent derivation. If several missing
definitions are inseparable, Decomposer must preserve all of them inside one
bundled `DEFINITION` child and bind every source definition label. Deeper
graphs are discovered recursively across later certified iterations.

Formalizer must preserve an existing parent Lean signature/hash exactly, or
propose a new parent signature only for an `UNFORMALIZED` parent. Parent and
all child signatures must pass the pinned signature gate. The reduction
signature may assume the child propositions and declared public assumptions,
but Prover must provide a complete proof of that exact reduction theorem.
`sorry`, `admit`, axioms, unsafe commands, placeholders, signature changes,
disconnected children, semantic duplicates, and vague glossary tasks reject
the entire bundle. Judge receives only the host-generated verification
manifest and cannot override a failed host gate.

Lean certification proves only that the formal child propositions and public
assumptions suffice for the exact formal parent proposition in the accepted
reduction theorem. The mapping from mathematical prose to Math IR and Lean
propositions remains model-authored semantic translation; host hashes,
typechecking, and adversarial review make that translation explicit but do not
prove it faithfully represents the intended informal mathematics.

Worker lifecycle and cache policy belong to the inference serving plane, not
the proof experiment. `prefill_compute_chunk_tokens` is immutable during this
loop. Never invoke launchctl, restart Primary/allens, clear KV, or run a cold
benchmark here. Model/tokenizer/quantization/rope/window/cache-format changes
must be deployed outside this supervisor. Cold benchmarks are explicit,
separate invocations of `scripts/benchmark_prefill_architecture.py`.

## Creative decomposition v3

Decomposer and Synthesis may first reason in an isolated private scratchpad
using prose or LaTeX. This consumes research budget but is untrusted,
audit-only, never parsed, and never enters Math IR, Lean, a gate, or the
ledger. The Host subsequently generates a finite menu of versioned typed moves
with scoped operands, preconditions, strict complexity metrics, theorem-card
support, and content hashes. Model transport contains candidate IDs and typed
reason codes only.

The registry includes case split, domain restriction, irrelevant-assumption
removal, holomorphic extension, singularity contradiction, and retained
definition/local-convergence/growth/bridge moves. Exact or alpha-equivalent
ancestors, disconnected tasks, unsupported symbols, unverified premises,
duplicates, and non-simplifying moves are excluded before ranking. Exactly one
child may continue through the existing Host and Lean gates.

`SYNTHESIS`/`REFRAME` is event-driven after semantic stagnation, repeated no
move, or sufficiently broad cross-role evidence. Counterexamples remain
advisory unless independently verified. Reframing may change a case partition
or viewpoint; only a typed whole-approach failure returns to global Strategy.

The local holomorphicity candidate is explicitly a special-case lemma: poles
lie outside an open disk, the terms converge locally uniformly there, the sum
is holomorphic, and agreement with a nonzero simple-pole expression produces
the proposed contradiction. It never claims to prove the parent; a separate
parent case-split reduction is mandatory.

Theorem cards are bounded deterministic records from the pinned local Mathlib
checkout. Each contains an actual declaration name and type, import, source
and environment hashes, tags, and required hypotheses. CI elaborates all card
names with `#check` and rejects stale hashes.

Retained KV capacity—not nominal Prefill admission—is the hard model-call
limit. The deployed default is sink 4 + window 2048 = 2052 tokens. Every
Strategy, Generator, Critic, premise, and certified-role chat template is
counted before append and must fit `min(configured prefill,
max_retained_tokens)`, including explicit control/decode reserve where needed.
No call may rely on evicted middle tokens.

Every active model call receives one exact `ProofStepInterface`: immutable root
hash, exact target statement, exact formal target/parent certificate interface,
public assumptions, immediate dependency interface, relevant active no-go
premises, current target evidence, and archive hashes. Eleven-node prose
ancestry and historical evidence are not active context. They remain durable
and hash-addressed; this is state selection, not an LLM summary and not a claim
of arbitrary natural-language full-attention equivalence.

Strategy proposes exactly one next proof step/question. Generator emits exactly
one bounded ISSUE_RESPONSE. Before Generator decode, the host reserves enough
retained capacity for Critic's fixed package plus the complete Generator
output; Critic receives that output byte-for-byte with the same exact
ProofStepInterface. Certified Decomposer proposes exactly one structurally
singular child per certificate; recursive later iterations perform deeper
decomposition. A complete multi-child response receives one bounded fresh
protocol-repair call with exact host validation errors and all rejected
obligations, requiring one bundled child. An output-cap/no-EOS response
receives one bounded fresh compact retry from the original certified package,
never a JSON continuation or splice. A second violation fails closed before
Formalizer, Prover, or Judge.

If an exact statement, structured artifact field, ISSUE block, dependency
node, or Lean source is indivisible and too large, fail closed with
`SEMANTIC_UNIT_TOO_LARGE`. Never slice tokens or strings, drop tails, or call a
model summary lossless. Recursive decomposition preserves exact certified
interfaces and reduction semantics, not arbitrary prose-history equivalence.
Structured-role output is budgeted from actual retained availability after the
tokenized prompt and control reserve. Decomposer must retain at least 512
output tokens within the 2052-token window; otherwise it fails before decode
with `STRUCTURED_RESPONSE_BUDGET_TOO_SMALL`. No-EOS and incomplete JSON remain
audit-only and never enter formal review or persistence.

Structured Artifact closure is a transport contract, not a conversational
convention. Models commonly append prose, an extra brace, or a second attempted
Artifact. The former implementation checked role-specific semantic validity
after tokens had already been generated; a syntactically closed but
schema-invalid first object therefore failed to stop transport, and divergent
repair prompts let the same defect recur in later roles. Every structured role
now uses one registered, string/escape-aware closure scanner. Generation stops
at the first balanced top-level object after the exact heading/marker with
`semantic_complete`; normal strict JSON, schema, binding, Lean, and host gates
then accept or reject that first object. They never scan ahead to a replacement.
Historical output with an extra brace, prose, Markdown fence, or second object
remains strictly invalid. The deterministic structured-output contract suite is
a mandatory local and GitHub CI gate for every registered role.

The concise stable `STRATEGY_CONTRACT` in `supervisor.py` is the authoritative
deterministic projection of this human-owned program for Strategy inference.
It includes the objective, event triggers, obligation/no-go/premise recovery
rules, exact-target requirement, immutable candidate/runtime constraints, and
the no-fallback/no-truncation contract. The full program remains authoritative
for host behavior but is not serialized into every Strategy prompt.

Obligation IDs are host-owned. Treat IDs emitted by Generator/Critic as
untrusted labels and bind verdicts to the exact current target. Never persist a
model-invented ID. Reject a proposed child when it duplicates an existing
statement or lemma name, is highly similar to an ancestor, or is too vague to
be falsifiable. Also reject any child that canonically or semantically repeats
a persisted no-go premise. A rejected cyclic frontier is `INCONCLUSIVE`, not
progress.

Every `DISPROVED` ISSUE_VERDICT distinguishes
`Invalidation: APPROACH|PREMISE_SUSPECTED`. Missing invalidation is legacy
`APPROACH`; legacy transcript value `PREMISE` is only a suspicion. A suspicion
must name `Premise refuted`, identify one evidence type, and provide a concrete
one-line JSON artifact. It never directly closes or quarantines a branch.

The worker automatically runs two fresh, ordered inference roles for every
structurally valid suspicion. The isolated Premise Auditor returns
`PREMISE_AUDIT` with `CONFIRMED|NOT_CONFIRMED|INCONCLUSIVE`, evidence
type/source, confidence, artifact, and analysis. The isolated Adversarial
Proponent then receives only the immutable goal, host-packaged suspicion, and
complete Auditor output and returns `PREMISE_DEFENSE` with
`RESCUED|NOT_RESCUED|INCONCLUSIVE`, exact correction/failure, and evidence.
Each inference uses a distinct session on allens-prefill/Primary-decode; only
explicit text crosses role boundaries. Both complete outputs and parsed
artifacts are durably persisted. Any timeout, malformed output, or worker
failure is an `INCONCLUSIVE` review and can never invalidate a premise.

The host upgrades to `PREMISE_INVALIDATED` only on a structurally valid
suspicion, Auditor `CONFIRMED` at confidence >= 0.8, Proponent `NOT_RESCUED`,
and a deterministically verified artifact. Arithmetic evidence uses one
host-normalized `FOR_ALL` claim object (`variables`, domain, lhs, claimed
relation, rhs). The host hashes this exact schema at suspicion time. The
Auditor must preserve the schema/hash, bind every quantified variable exactly
once, and provide a finite witness for which the Critic's claimed relation
evaluates false. Unknown variables, true-but-unrelated arithmetic, missing
bindings, schema changes, or hash changes are rejected.

Lean evidence is bound to the target's host-recorded `lean_signature_hash` and
must claim `NEGATION_OF_TARGET_SIGNATURE`. The current stored Lean signature
format does not permit the host to safely synthesize an exact negation wrapper,
so Lean artifacts are presently recorded but fail open to `INCONCLUSIVE` even
if their standalone theorem compiles. Pinned theorem references are likewise
untrusted until a local exact-assumption registry exists.

Before upgrade, descendants retain their statuses and only reversible
`SUSPECTED`/temporary-quarantine metadata is recorded. After upgrade, the host
marks the target `DISPROVED/PREMISE_INVALIDATED`, quarantines descendants,
stores one bound-schema-hash no-go lesson with confidence/evidence/worker-run
provenance, and backjumps to the nearest sound unresolved ancestor. A later
Auditor `NOT_CONFIRMED` or Proponent `RESCUED` deterministically restores prior
statuses and marks quarantine/no-go records `REVERSED`. `APPROACH` failure
never quarantines descendants or siblings.

After a structurally valid suspicion completes review without verified
upgrade (`NOT_CONFIRMED`, `RESCUED`, or `INCONCLUSIVE`), the host clears all
temporary quarantine, restores descendants, preserves audit provenance, and
closes only the attempted leaf as `DISPROVED/APPROACH_FAILED`. This fallback
is allowed only when the original Critic DISPROVED evidence passed the existing
strong closure gate. It creates no no-go lesson or premise quarantine.

Only a host-upgraded invalidation triggers supervisor `premise-invalidated`.
Suspicion initiates worker audit but is neither proof progress nor permanent
falsification. The exact Strategy interface carries only active relevant no-go
records. New candidates must not assume, rename, or reconstruct them. Lean
signature typechecking remains merely `FORMALIZED` and does not establish
premise truth; only the separate complete-proof gate can validate Lean
evidence.

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
