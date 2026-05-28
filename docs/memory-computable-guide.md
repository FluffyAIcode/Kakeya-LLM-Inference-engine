# Memory Computable Guide

This guide captures the current design baseline for memory computability in
the Kakeya inference engine. It deliberately starts from the three laws of
attention and memory computation, then translates them into a trainable memory
operator. It does not assume proposer cross-training.

## Three laws

### 1. Alignment object law

Attention aligns to the output token distribution; memory aligns to stored
records.

- Attention computation is constrained by generation targets such as
  `P(y_t | x, y_<t)`.
- Memory computation is constrained by already stored records such as user
  history, interaction traces, documents, logs, or persistent system state.

In short:

```text
attention alignment = distribution alignment
memory alignment    = record alignment
```

### 2. Sparsification law

Both attention and memory need sparse classification, but under different
constraints.

- Attention sparsifies under a generation-distribution constraint and is a
  natural fit for autoregressive token generation.
- Memory sparsifies by topic under an existing-record constraint and is a
  natural fit for diffusion-style reconstruction.

The important distinction is not "short term vs long term". It is whether the
sparse choice is judged against a token distribution or against stored records.

### 3. Mutual embedding law

Attention contains memory storage; memory contains attention forward. They are
not opposing modules, but mutually embedded computation processes.

For this guide, "memory contains attention forward" means that the current
Transformer computation predicts a memory support distribution. It is not a
claim that attention forward has already read a verified memory fact.

## Memory computation definition

Memory computation is the extension of attention computation from output
distribution alignment to stored-record alignment:

```text
current Transformer state
  -> memory attention forward
  -> memory support prediction
  -> record-aligned support objective
  -> sparse topic support
  -> diffusion-style denoising convergence
  -> record-aligned memory state
```

The core operator is:

```text
M(q, R) -> z_M
```

where:

- `q` is the current Transformer computation state.
- `R = {r_i}` is the set of stored records.
- `M` is the memory computation operator.
- `z_M` is the resulting memory state.

The required constraint is:

```text
z_M ~= E(R_relevant)
```

where `R_relevant` is the stored-record support that should be relevant for
the current computation.

## Operator boundaries

### Memory attention forward predicts support

Memory attention forward produces a support prediction:

```text
pi = P(record/topic | q, R)
```

This is a prediction over records or topic clusters. It is not a verified read.
The support prediction must be aligned to stored records during training.

### Record alignment is the memory requirement

Record alignment is not an optional external check. It is the core training
requirement that makes the support prediction a memory computation rather than
a generic retrieval or generation bias.

The model must learn:

```text
predicted support ~= relevant stored records
```

If this alignment fails, later denoising can stabilize or amplify the wrong
topic.

### Diffusion denoising converges within support

Diffusion-style reconstruction does not change the predicted topic support.
It only denoises and converges inside the support produced by memory attention
forward.

```text
z_T -> z_0 ~= E(R_relevant)
```

Therefore diffusion is useful only after the support prediction is record
aligned. If the support is noisy or wrong, diffusion can make that wrong memory
state more stable rather than correcting it.

## Minimal neural operators

### 1. Transformer state adapter

Maps the current Transformer computation state into a memory query:

```text
q_M = Phi(q)
```

The adapter can consume hidden states, residual states, attention summaries,
or other current-forward state summaries.

### 2. Record encoder

Maps stored records into the memory space:

```text
r_i -> e_i
```

The output can be grouped into record embeddings, topic embeddings, or
record-topic blocks.

### 3. Memory attention forward

Predicts a sparse support distribution over records or topics:

```text
score_i = sim(q_M, e_i)
pi = sparsemax(score)  # or entmax / top-k routing
```

### 4. Support alignment head

Trains `pi` against the known relevant support:

```text
L_support = CE(pi, support_label)
```

or a contrastive support objective:

```text
L_support = -log exp(sim(q_M, e_pos) / tau)
                  / sum_j exp(sim(q_M, e_j) / tau)
```

### 5. Diffusion memory denoiser

Denoises a memory latent inside the predicted support:

```text
L_denoise = || eps - eps_theta(z_t, q_M, pi, R, t) ||^2
```

The denoiser should be evaluated only with support-quality metrics attached,
because denoising quality is meaningless if the support is wrong.

## Training objective

The memory operator can be trained with:

```text
L_memory =
    L_support_alignment
  + lambda_record * L_record_alignment
  + lambda_sparse * L_sparse_topic
  + lambda_denoise * L_diffusion_reconstruction
  + lambda_cf * L_counterfactual_record
```

Where:

- `L_support_alignment` trains memory attention forward to predict the correct
  record or topic support.
- `L_record_alignment` trains the final memory state to align with the stored
  records.
- `L_sparse_topic` encourages a small active topic set.
- `L_diffusion_reconstruction` trains denoising convergence inside the support.
- `L_counterfactual_record` verifies that changing stored records changes the
  predicted support and memory state.

## H200 training target

The first H200 training milestone should validate the memory operator itself,
not proposer integration.

Recommended sample shape:

```text
(q, R_positive, R_negative, R_counterfactual, support_label)
```

The run should report both model-quality and system-efficiency metrics.

### Quality metrics

- Support precision: predicted support records are relevant.
- Support recall: relevant records are covered by the predicted support.
- Record alignment accuracy: `z_M` matches the correct stored record state.
- Sparse topic entropy: routing remains sparse rather than diffuse.
- Denoising gain on clean support: diffusion improves a correct support state.
- Noise amplification rate: diffusion does not stabilize wrong support.
- Counterfactual flip rate: replacing records changes support and `z_M`.
- Update/delete consistency: changed or deleted records stop influencing
  memory output.

### H200 system metrics

- Samples per second.
- Records per second.
- HBM usage.
- Routing latency.
- Denoising latency.
- End-to-end memory-operator latency.

## Acceptance criteria

The memory operator should be considered valid only if:

1. Support prediction aligns to stored records with high precision and recall.
2. Diffusion denoising improves record-aligned states on clean support.
3. Diffusion does not amplify noisy or wrong support beyond an allowed budget.
4. Counterfactual record changes cause corresponding memory state changes.
5. Record update/delete tests stop stale records from influencing output.
6. The H200 latency and memory overhead are small enough to justify later
   inference-engine integration.

## Design summary

Memory computability is not proven by training on historical data. It is proven
when the current computation predicts a sparse memory support, that support is
aligned to stored records, and diffusion-style denoising converges only inside
the aligned support.
