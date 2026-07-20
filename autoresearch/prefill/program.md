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
2. State one concrete mathematical hypothesis for one unresolved leaf.
3. Modify only `candidate.py`.
4. Deploy the candidate to allens.
5. Clear Primary and allens caches.
6. Run the fixed full-context acceptance workload.
7. Run `prepare.py` against the resulting report.
8. Keep the candidate only if every hard constraint passes and it closes an
   obligation, falsifies a novel hypothesis, or creates a novel smaller
   frontier. Otherwise restore the previous candidate.
9. Append the result and repeat.

Every candidate must target exactly one current unresolved leaf obligation and
contain a falsifiable hypothesis plus distinct Generator and Critic directives.
An unresolved Critic verdict must isolate one strictly smaller missing lemma;
the host records that lemma as a deduplicated child obligation. Completed GAN
runs, transcripts, checkpoints, and ledger updates remain durable even when the
candidate strategy is reverted or fixed evaluation fails.

Do not optimize output wording, scores, prizes, or other proof-irrelevant
content. Prefill performance is a tertiary objective after mathematical
decomposition progress, while preserving the complete semantic contract.
