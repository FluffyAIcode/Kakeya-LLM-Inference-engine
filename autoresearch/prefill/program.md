# Prefill AutoResearch Program

You are optimizing the two-Mac full-context Prefill system.

## Ownership

- You may edit only `candidate.py`.
- Never edit `prepare.py`, benchmark reports, tests, or production metrics.
- The human owns this file.

## Objective

Minimize `metric_cold_critic_prefill_s`. Lower is better.

## Hard constraints

- Every compute segment must remain at or below 300 seconds.
- Critic must receive the complete Generator response.
- `critic_omitted_tokens` must equal zero.
- Protocol must be `recursive_proof_decomposition_v2`.
- Snapshot mode must remain `final_only`.
- No fallback, local Primary Prefill, failed remote job, sampling, summary, or
  semantic simplification is allowed.
- Primary remains decode-only and allens remains Prefill-only.

## Experiment loop

1. Read `candidate.py` and `results.tsv`.
2. State one concrete performance hypothesis.
3. Modify only `candidate.py`.
4. Deploy the candidate to allens.
5. Clear Primary and allens caches.
6. Run the fixed full-context acceptance workload.
7. Run `prepare.py` against the resulting report.
8. Keep the candidate only if every hard constraint passes and cold Critic
   Prefill time improves. Otherwise restore the previous candidate.
9. Append the result and repeat.

Do not optimize output wording, scores, prizes, or other proof-irrelevant
content. Optimize only measured Prefill execution while preserving the complete
semantic contract.
