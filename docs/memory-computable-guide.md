# Memory Computable Guide

本文档只定义 Memory Computation Operator 的设计基线。

## 保留的三定律

### 对齐对象定律

注意力对齐输出分布，记忆力对齐存储记录。

### 稀疏化定律

注意力和记忆力都需要稀疏化分类；注意力在生成分布约束下稀疏化，更适合自回归；记忆力在既存记录约束下主题稀疏化，更适合 diffusion 式重构。

### 相互内嵌定律

注意力中有记忆存储，记忆中有注意力 forward；二者不是对立模块，而是相互内嵌的计算过程。

## 新设计起点

### 目标

设计一个 Memory Computation Operator，它不是普通检索，也不是输出分布拟合，而是：

> 在 Transformer 框架中，把注意力计算扩展为面向存储记录的可微分记忆力计算。

### 核心定义

记忆力计算可定义为：

> 由当前计算状态触发，通过 attention forward 预测相关记忆支撑，并在既存存储记录约束下完成主题稀疏化与 diffusion 式重构的过程。

形式化：

```text
M(q, R) -> z_M
```

其中：

- `q`：当前 Transformer 计算状态；
- `R`：既存存储记录；
- `M`：记忆力计算算子；
- `z_M`：重构后的 memory state。

### 最小架构

```text
Transformer Current State
        ↓
Memory Attention Forward
        ↓
Memory Support Prediction
        ↓
Record Alignment Objective
        ↓
Sparse Topic Support
        ↓
Diffusion-style Memory Reconstruction
        ↓
Record-aligned Memory State
```

## 三个关键算子

### 1. Memory Attention Forward

作用不是“读取事实”，而是：

> 预测当前计算需要哪些记忆支撑。

输出：

```text
π = P(record/topic | q, R)
```

即对存储记录或主题簇的稀疏预测分布。

### 2. Record Alignment

这是记忆力计算的核心约束：

```text
π, z_M ≈ R_relevant
```

目标是让 memory support prediction 和 memory state 都对齐既存记录。

### 3. Diffusion Memory Reconstruction

Diffusion 不负责改变主题，只负责：

> 在已对齐的 memory support 内进行去噪、收敛和重构。

即：

```text
z_T -> z_0 ≈ E(R_relevant)
```

## 训练目标

```text
L_memory =
  L_support_alignment
+ λ L_record_alignment
+ μ L_sparse_topic
+ ν L_diffusion_reconstruction
```

其中：

- `L_support_alignment`：预测的 memory support 是否对应真实相关记录；
- `L_record_alignment`：最终 memory state 是否对齐存储记录；
- `L_sparse_topic`：主题选择是否足够稀疏；
- `L_diffusion_reconstruction`：是否能在正确支撑内完成去噪重构。

## 判断是否真的有记忆力计算

核心不是看模型是否“用过历史数据”，而是看：

> 当前 memory prediction 和 memory state 是否被既存存储记录约束。

关键指标：

- support precision；
- support recall；
- record alignment accuracy；
- sparse topic entropy；
- reconstruction quality；
- counterfactual record sensitivity；
- update/delete consistency。

## 新设计一句话

记忆力计算是注意力计算从“输出分布对齐”向“存储记录对齐”的扩展：它通过 memory attention forward 预测记忆支撑，通过记录对齐约束校正支撑分布，再通过 diffusion 式重构形成可用的 memory state。
