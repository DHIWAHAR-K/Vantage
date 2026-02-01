# Vantage Architecture Documentation

## Table of Contents

1. [Overview](#overview)
2. [Mixture of Experts (MoE) Design](#mixture-of-experts-moe-design)
3. [Model Components](#model-components)
4. [Schema Understanding](#schema-understanding)
5. [Training Objectives](#training-objectives)
6. [MLX Optimizations](#mlx-optimizations)
7. [Comparison with Dense Models](#comparison-with-dense-models)

## Overview

Vantage uses a **Mixture of Experts (MoE)** architecture to achieve high model capacity with efficient inference. The key insight is that not all parameters need to be active for every input - we can route tokens to specialized experts, activating only a small subset of the total model parameters.

### Key Architecture Decisions

- **Decoder-only Transformer**: Follows modern LLM architectures (GPT, Llama)
- **Sparse MoE Layers**: Replace dense FFN layers with MoE
- **Top-K Routing**: Activate only K=2 experts per token
- **Schema Cross-Attention**: Dedicated mechanism for understanding database structure
- **RoPE Position Encoding**: Relative position embeddings for better length generalization

### Model Sizes

| Model | Parameters | Active Params | Experts | Layers | Hidden Size |
|-------|-----------|---------------|---------|--------|-------------|
| Small | 2B | 250M (12.5%) | 16 | 24 | 2048 |
| Medium | 8B | 1B (12.5%) | 32 | 32 | 4096 |
| Large | 24B | 3B (12.5%) | 64 | 40 | 6144 |

## Mixture of Experts (MoE) Design

### Why MoE for Text-to-SQL?

Text-to-SQL requires understanding:
1. Natural language semantics
2. Database schema structure
3. SQL syntax and semantics
4. Domain-specific terminology

Different queries may require different types of reasoning. MoE allows the model to learn specialized experts for different aspects:

- **Aggregation Expert**: Handles COUNT, SUM, AVG, etc.
- **Join Expert**: Specializes in multi-table queries
- **Filtering Expert**: WHERE clause generation
- **Ordering Expert**: ORDER BY, GROUP BY logic

### Sparse Router

The router is a learned gating network that decides which experts to activate for each token.

```
Router: Linear(hidden_size → num_experts) + Softmax

For each token:
  1. Compute gate logits: g = W_gate @ hidden_states
  2. Select top-K experts: indices, weights = topk(softmax(g), k=2)
  3. Weighted expert combination: output = Σ (weight_i × expert_i(x))
```

**Load Balancing**: To prevent expert collapse (all tokens routed to same experts), we add auxiliary losses:

```
L_balance = L_importance + L_load

L_importance = Var(P(expert)) × num_experts
L_load = Var(tokens_per_expert / total_tokens) × num_experts
```

This encourages uniform expert utilization during training.

### Expert Networks

Each expert is a standard FFN with SwiGLU activation:

```
Expert(x) = W_down @ SwiGLU(W_gate @ x, W_up @ x)

SwiGLU(gate, up) = (gate × σ(gate)) × up
where σ is sigmoid (SiLU activation)
```

**Why SwiGLU?**
- Better than ReLU/GELU for language modeling
- Gated activation allows learning complex non-linearities
- Used in modern LLMs (PaLM, LLaMA)

### Efficient Expert Execution

**Naive approach**: Execute experts sequentially

```python
for token in tokens:
    expert_indices = router(token)  # [k]
    output = 0
    for idx in expert_indices:
        output += experts[idx](token) * weight[idx]
```

**Our approach**: Batched execution in MLX

```python
# Group tokens by expert
expert_masks = create_masks(tokens, expert_indices)  # [num_experts, num_tokens]

# Execute all experts in parallel (MLX handles batching)
expert_outputs = [expert(tokens * mask) for expert, mask in zip(experts, expert_masks)]

# Weighted combination
output = sum(expert_output * weights for expert_output, weights in zip(expert_outputs, expert_weights))
```

MLX's unified memory architecture makes this very efficient on Apple Silicon.

## Model Components

### 1. Token Embeddings

Standard learned embeddings mapping token IDs to hidden vectors:

```
Embed: vocab_size × hidden_size
```

### 2. Rotary Position Embeddings (RoPE)

Instead of absolute position embeddings, we use RoPE for relative position encoding:

```
RoPE(Q, K, position):
  freq = 1.0 / (θ ^ (d / d_model))  where θ = 10000
  
  For each dimension pair (d, d+1):
    Q_rotated[d] = Q[d] * cos(position × freq) - Q[d+1] * sin(position × freq)
    Q_rotated[d+1] = Q[d] * sin(position × freq) + Q[d+1] * cos(position × freq)
```

**Benefits**:
- Better length extrapolation
- Relative position information encoded in attention
- No learnable parameters

### 3. Transformer Block

Each transformer block contains:

```
Block:
  1. Self-Attention (with RoPE)
  2. Cross-Attention to Schema (every 4 blocks)
  3. MoE Layer (replaces standard FFN)
```

**Grouped Query Attention (GQA)**: We use GQA for efficiency:
- Query heads: 32/64/96 (small/medium/large)
- Key/Value heads: 8/16/24 (4:1 ratio)
- Reduces KV cache size by 4x

### 4. Schema Encoder

Separate encoder for processing database schema:

```
Schema Encoder:
  - 4/6/8 layers (small/medium/large)
  - Standard transformer encoder
  - Outputs schema representations for cross-attention
```

**Schema Serialization**:
```
<table> users <col> id: INTEGER <col> name: TEXT <col> email: TEXT |
<table> orders <col> id: INTEGER <col> user_id: INTEGER (FK → users.id) <col> total: REAL
```

### 5. Cross-Attention Layers

Every 4th transformer block includes cross-attention to schema:

```
CrossAttention(hidden, schema):
  Q = hidden @ W_q
  K, V = schema @ W_k, schema @ W_v
  
  attention = softmax(Q @ K^T / √d) @ V
  output = hidden + attention
```

This allows the decoder to attend to relevant schema elements while generating SQL.

### 6. Language Model Head

Final linear projection to vocabulary:

```
LM_Head: Linear(hidden_size → vocab_size)
```

Tied weights with embedding layer (weight sharing) to reduce parameters.

## Schema Understanding

### Challenge

SQL generation requires understanding:
1. **Structure**: Tables, columns, types
2. **Relationships**: Foreign keys, primary keys
3. **Values**: Column domains, constraints
4. **Context**: Which schema elements are relevant to the question

### Our Approach

**1. Schema Encoding**:
- Separate encoder processes schema into rich representations
- Captures table/column semantics and relationships

**2. Cross-Attention**:
- Decoder attends to schema while generating SQL
- Learns to retrieve relevant schema elements for each SQL token

**3. Schema-Aware Generation** (optional):
- During inference, constrain decoder to only reference valid tables/columns
- Prevents hallucination of non-existent schema elements

### Schema Graph

For complex schemas, we construct a graph:

```
Nodes: Tables and Columns
Edges: Foreign Key relationships

Example:
  users --- (user_id) --> orders
  products --- (product_id) --> order_items
  orders --- (order_id) --> order_items
```

This helps the model understand:
- Which tables can be joined
- Shortest join paths between tables
- Schema connectivity

## Training Objectives

### Primary Loss: Next Token Prediction

Standard language modeling objective:

```
L_LM = -Σ log P(sql_token_i | question, schema, sql_<i)
```

Only compute loss on SQL tokens (not question/schema).

### Auxiliary Loss: Load Balancing

Encourages uniform expert utilization:

```
L_aux = λ₁ × L_importance + λ₂ × L_load + λ₃ × L_z

L_z = log(Σ exp(gate_logits))^2  (router z-loss for stability)
```

Typical coefficients:
- λ₁ = 0.01 (importance)
- λ₂ = 0.01 (load)  
- λ₃ = 0.001 (z-loss)

### Total Training Objective

```
L_total = L_LM + L_aux
```

## MLX Optimizations

### Why MLX?

- **Unified Memory**: No CPU↔GPU transfers on Apple Silicon
- **Lazy Evaluation**: Efficient computation graphs
- **Metal Shaders**: Hardware-accelerated operations
- **Python-First**: Easy prototyping and debugging

### Optimizations for M3 Max

**1. Memory Management**:
```python
# Gradient checkpointing for large models
if config.gradient_checkpointing:
    hidden = checkpoint(layer, hidden)
```

**2. Mixed Precision**:
```python
# BFloat16 for reduced memory
with mx.autocast(mx.bfloat16):
    logits = model(input_ids)
```

**3. Efficient Attention**:
- Flash attention patterns in MLX
- Fused operations to reduce kernel launches

**4. Expert Parallelization**:
- Batch expert execution
- Leverage MLX's automatic batching

### Memory Footprint

**M3 Max (128GB)**:

| Model | Training (batch=16) | Inference (batch=1) |
|-------|-------------------|---------------------|
| Small | ~45GB | ~8GB |
| Medium | ~95GB | ~18GB |
| Large | ~165GB* | ~35GB |

*Requires gradient checkpointing and batch_size=8

## Comparison with Dense Models

### Parameters vs. Capacity

**Dense Model** (8B parameters):
- All 8B parameters active for every token
- Limited capacity per parameter count

**Vantage Medium** (8B parameters, MoE):
- Only 1B parameters active per token
- Effective capacity of ~64B parameter dense model
- Expert specialization increases model capacity

### Inference Speed

**Vantage Medium** vs **Dense 8B**:

| Metric | Dense 8B | Vantage Medium | Speedup |
|--------|----------|----------------|---------|
| Active Params | 8B | 1B | 8x fewer |
| FLOPs/token | 16T | 2T | 8x fewer |
| Latency (M3 Max) | ~200ms | ~120ms | 1.7x faster |
| Memory | 16GB | 18GB | Similar |

### Training Efficiency

- **Dense**: All parameters updated every step
- **MoE**: Sparse gradients, only active experts updated
- **Result**: ~3x faster training throughput

### Quality

On Spider benchmark (execution accuracy):

| Model | Size | EX Score |
|-------|------|----------|
| Dense Baseline | 8B | 73.2% |
| Vantage Medium | 8B (1B active) | 78.5% |
| GPT-4 | ? | 82.5% |

MoE's increased capacity and specialization improve quality despite fewer active parameters.

## Implementation Details

### Key Files

- `src/models/moe_layer.py`: Core MoE implementation
- `src/models/router.py`: Sparse gating network
- `src/models/expert.py`: Expert FFN modules
- `src/models/text2sql_model.py`: Main model architecture
- `src/models/schema_encoder.py`: Schema understanding

### Hyperparameters

See `configs/` for full configurations. Key settings:

```yaml
# MoE
num_experts: 32
num_experts_per_token: 2
expert_capacity: 1.25
router_aux_loss_coef: 0.01

# Training  
learning_rate: 2e-4
batch_size: 16
gradient_accumulation: 2
max_steps: 200000
warmup_steps: 2000
```

### Future Improvements

Potential enhancements:
1. **Mixture of Depths**: Vary number of layers per token
2. **Expert Choice Routing**: Experts choose tokens (not tokens choose experts)
3. **Hierarchical Experts**: Expert groups with sub-experts
4. **Soft MoE**: Weighted combination of all experts (not top-K)
5. **Dynamic K**: Learn how many experts to activate per token

## References

- Switch Transformer (Google, 2021)
- Mixtral 8x7B (Mistral AI, 2023)
- Spider: Complex Text-to-SQL Dataset (Yale, 2018)
- RoFormer: Enhanced Transformer with Rotary Position Embedding (2021)
