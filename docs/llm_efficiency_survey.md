# Speed Always Wins: A Survey on Efficient Architectures for Large Language Models

## Executive Summary

This comprehensive survey examines innovative architectural designs that address the computational inefficiencies of Transformer-based Large Language Models (LLMs). The paper categorizes recent advances into seven main areas, providing both theoretical foundations and practical solutions for scaling LLMs efficiently.

## 1. Introduction & Motivation

### The Efficiency Problem
Modern LLMs face critical computational bottlenecks:
- **Quadratic attention complexity**: Standard self-attention scales as O(N²) with sequence length N
- **Growing memory requirements**: KV cache and parameter storage become prohibitive
- **Long-context challenges**: RAG, agentic patterns, reasoning chains, and multimodal inputs demand longer sequences

### Survey Categories
The paper organizes efficient architectures into seven categories:
1. **Linear Sequence Modeling** - Reducing attention to O(N) complexity
2. **Sparse Sequence Modeling** - Selective attention computation
3. **Efficient Full Attention** - Optimizing standard attention without approximation
4. **Sparse Mixture-of-Experts** - Conditional computation paradigms
5. **Hybrid Architectures** - Combining efficient and standard components
6. **Diffusion LLMs** - Non-autoregressive generation alternatives
7. **Cross-Modal Applications** - Extensions beyond text

## 2. Linear Sequence Modeling

### 2.1 Linear Attention

**Core Principle**: Replace softmax attention with kernel-based approximations to achieve linear complexity.

**Standard Attention**:
```
qt, kt, vt = xtWQ, xtWK, xtWV
ot = Σᵢ₌₁ᵗ exp(qtᵀki)vi / Σᵢ₌₁ᵗ exp(qtᵀki)
```

**Linear Attention**:
```
ot = φ(qt)Σᵢ₌₁ᵗ φ(ki)ᵀvi / φ(qt)Σᵢ₌₁ᵗ φ(ki)ᵀ
```

**Recurrent Formulation**:
```
St = St-1 + φ(kt)ᵀvt
zt = zt-1 + φ(kt)ᵀ
ot = φ(qt)St / φ(qt)zt
```

**Key Innovations**:
- **Feature Mappings**: elu(x) + 1, random features, polynomial kernels
- **Gating Mechanisms**: Data-dependent gates for selective memory management
- **Delta Learning**: Memory updates based on prediction errors

### 2.2 Linear RNNs

**Motivation**: Combine RNN efficiency with parallel training capability.

**Gated Linear RNN (GLRU)**:
```
gt = σ(Wgxt + bg)    # Forget gate
it = τ(Wixt + bi)    # Input gate  
ot = σ(Woxt + bo)    # Output gate
ht = gt ⊙ ht-1 + (1 - gt) ⊙ it
yt = ht ⊙ ot
```

**Matrix Memory Extension**:
```
ht = ht-1 · Diag{ft} + it ⊗ (1 - ft) ∈ ℝᵈˣᵈ
```

**Notable Models**: HGRN, RWKV, xLSTM, GateLoop

### 2.3 State Space Models (SSMs)

**Mathematical Foundation**: Continuous-time state space representation.

**Continuous SSM**:
```
x'(t) = Ax(t) + Bu(t)
y(t) = Cx(t) + Du(t)
```

**Discretization** (Zero-Order Hold):
```
A = exp(ΔA)
B = (ΔA)⁻¹(exp(ΔA) - I) · ΔB
```

**Convolution Form**:
```
K = (CB, CAB, ..., CAᴸB)
y = x * K
```

**Evolution Path**:
- **S4**: HiPPO initialization, diagonal + low-rank structure
- **Mamba**: Data-dependent parameters, selective SSMs
- **Mamba2**: Scalar state transitions, blockwise computation

### 2.4 Test-Time-Training RNNs

**Concept**: Treat memory as trainable parameters updated via gradient descent during inference.

**General Update Rule**:
```
St = αtSt-1 - ηt∇SL(St-1; kt, vt)
```

**Objective Functions**:
- **Local L2**: βt||vt - Stkt||² + ½Tr(SᵀΛS)
- **Global L2**: Σᵢ₌₁ᵗ γi||vi - Ski||² + Tr(SᵀΛS)

### 2.5 Unified Framework

**Memory Perspective**: All methods can be viewed as key-value associative memory systems.

**Linear Update**:
```
St = (αt, βt)@(St-1, ktᵀvt)ᵀ    # Write
ot = Stqt                        # Read
```

**Bilinear Update** (with interactions):
```
St = (αt, βt)@(St-1, ktᵀvt)ᵀ + γtSt-1kt
```

### 2.6 Linearization

**Goal**: Convert pre-trained Transformers to linear models with minimal retraining.

**Approaches**:
- **Finetuning-based**: Direct replacement + architecture adaptation
- **Distillation-based**: Knowledge transfer from teacher to student model

## 3. Sparse Sequence Modeling

### 3.1 Static Sparse Attention

**Concept**: Predefined, fixed sparsity patterns that remain constant during training and inference.

**Common Patterns**:
- **Local/Sliding Window**: Attend to nearby tokens
- **Dilated**: Exponentially increasing attention spans
- **Random**: Randomly selected attention positions
- **Global**: Special tokens that attend to all positions

**Representative Models**:
- **Sparse Transformer**: Strided and dilated patterns
- **Longformer**: Sliding window + global tokens
- **BigBird**: Local + global + random connections

### 3.2 Dynamic Sparse Attention

**Concept**: Attention patterns determined adaptively based on input content.

**Strategies**:
- **Clustering**: Group semantically similar tokens (LSH, k-means)
- **Retrieval**: Use external memory with k-NN lookup
- **Learned Routing**: Neural networks determine importance

**Key Models**:
- **Reformer**: Locality-sensitive hashing for token bucketing
- **Memorizing Transformers**: External kNN-based memory
- **Native Sparse Attention**: Hardware-aligned hierarchical selection

### 3.3 Training-Free Sparse Attention

**Applications**:
- **Prefill Acceleration**: Reduce computation during initial prompt processing
- **Decoding Optimization**: Manage KV cache growth during generation

**Techniques**:
- **Attention Sinks**: Preserve initial tokens that attract high attention
- **Heavy Hitters**: Retain tokens with highest attention scores
- **Dynamic Eviction**: Remove least important tokens from cache

## 4. Efficient Full Attention

### 4.1 IO-Aware Attention (FlashAttention)

**Problem**: Memory bandwidth bottleneck, not computation.

**Solution**: Optimize memory access patterns through tiling and online computation.

**FlashAttention Algorithm**:
1. Divide Q, K, V into blocks that fit in SRAM
2. Compute attention scores block-wise
3. Use online softmax to avoid storing full attention matrix
4. Recompute attention weights during backward pass

**Improvements**:
- **FlashAttention-2**: Better work partitioning, query-outer loop
- **FlashAttention-3**: Producer-consumer asynchrony, FP8 quantization

### 4.2 Grouped Attention

**Multi-Query Attention (MQA)**:
```
Multiple query heads share single key/value head
Memory: O(d) instead of O(Hd) for KV cache
```

**Grouped-Query Attention (GQA)**:
```
Middle ground: Group queries, each group shares K/V
Trade-off between quality and efficiency
```

**Multi-Head Latent Attention (MLA)**:
```
Compress KV into low-rank latent representation
Further reduce memory requirements
```

### 4.3 Mixture of Attention

**Concept**: Different attention heads use different patterns or mechanisms.

**Approaches**:
- **Pattern Mixing**: Heads use different sparsity patterns
- **Head Selection**: Route tokens to specific attention heads
- **Dynamic Allocation**: Adaptively determine attention computation

### 4.4 Quantized Attention

**Goal**: Reduce precision while maintaining accuracy.

**Strategies**:
- **Post-Training**: Quantize pre-trained models (INT8, FP8)
- **Quantization-Aware Training**: Train with low precision
- **Mixed Precision**: Use different precisions for different operations

## 5. Sparse Mixture-of-Experts (MoE)

### 5.1 Routing Mechanisms

**Basic Formulation**:
```
P = Softmax(XWg + bg)     # Router probabilities
Y = Σk Top-k(P) · Ek(X)   # Expert outputs
```

**Routing Strategies**:
- **Token-Choice**: Each token selects top-k experts
- **Expert-Choice**: Each expert selects top-k tokens
- **Adaptive**: Dynamic number of experts per token

**Load Balancing Loss**:
```
Laux = (1/N) Σᵢ₌₁ᴺ Di(X) · Gi(X)
```

### 5.2 Expert Architectures

**Design Variations**:
- **Fine-grained**: Many small experts for more combinations
- **Shared Experts**: Fixed experts always activated
- **Specialized**: Domain-specific or task-specific experts

### 5.3 MoE Conversion

**Dense-to-MoE Strategies**:
- **Splitting**: Partition existing FFN layers
- **Copying**: Replicate and specialize layers
- **Merging**: Combine multiple dense models

## 6. Hybrid Architectures

### 6.1 Inter-layer Hybrid

**Concept**: Alternate between efficient and standard attention layers.

**Representative Models**:
- **Zamba**: Mamba backbone with periodic attention layers
- **Jamba**: Mamba + Attention + MoE combination
- **YOCO**: Sliding window + standard attention with shared KV cache

### 6.2 Intra-layer Hybrid

**Approaches**:
- **Head-wise Split**: Different heads use different mechanisms
- **Sequence-wise Split**: Different positions use different attention types

**Examples**:
- **Hymba**: Mamba and attention heads within same layer
- **LoLCATs**: Linear attention for distant tokens, softmax for recent tokens

## 7. Diffusion Large Language Models

### 7.1 Non-Autoregressive Generation

**Key Advantage**: Parallel token generation instead of sequential.

**LLaDA Framework**:
```
Forward: Progressively mask tokens
Reverse: Jointly predict all masked tokens
Objective: L(θ) = -Et,x₀,xt[1/t Σᵢ₌₁ᴸ 1[xᵢt = M] log pθ(xᵢ₀|xt)]
```

**Benefits**:
- **Parallel Decoding**: Generate multiple tokens per step
- **Controllability**: Better adherence to constraints
- **Bidirectional Context**: Access full sequence during generation

### 7.2 Hybrid Approaches

**BD3-LMs**: Autoregressive across blocks, diffusion within blocks
```
log pθ(x) = Σᵦ₌₁ᴮ log pθ(x⁽ᵦ⁾|x⁽<ᵦ⁾)
```

### 7.3 Multimodal Extensions

**Applications**: Vision-language models using diffusion paradigm
**Advantages**: Unified generation across modalities

## 8. Applications to Other Modalities

### 8.1 Vision
- **Classification**: Mamba-based backbones for image recognition
- **Detection**: SSM backbones in YOLO-style frameworks
- **Segmentation**: U-Net architectures with linear attention
- **Generation**: Diffusion models with Mamba backbones

### 8.2 Audio
- **Understanding**: Audio classification and tagging
- **Enhancement**: Speech separation and noise reduction
- **Generation**: Symbolic music and raw audio synthesis

### 8.3 Multimodality
- **Alignment**: Cross-modal fusion using linear models
- **Scaling**: MoE for large multimodal models
- **Generation**: Unified frameworks for text, image, audio

## 9. Technical Comparison & Analysis

### Complexity Comparison
| Method | Training | Inference | Memory |
|--------|----------|-----------|---------|
| Standard Attention | O(N²d) | O(N²d) | O(N²) |
| Linear Attention | O(Nd²) | O(d²) | O(d²) |
| Sparse Attention | O(sNd) | O(sNd) | O(sN) |
| State Space | O(Nd) | O(d) | O(d) |

### Performance Trade-offs
- **Linear Models**: High efficiency, some quality loss on recall tasks
- **Sparse Models**: Good efficiency-quality balance, pattern-dependent
- **Hybrid Models**: Best of both worlds, increased complexity
- **MoE**: Scalable capacity, routing challenges

## 10. Future Directions

### Architectural Innovations
- **Algorithm-System-Hardware Co-design**: Joint optimization across all levels
- **Adaptive Mechanisms**: Dynamic adjustment based on input/hardware
- **Hierarchical Memory**: Multi-tiered memory architectures
- **Non-Autoregressive Models**: Parallel generation paradigms

### Application Frontiers
- **Infinite Context**: Handling unbounded sequence lengths
- **Efficient Agents**: Real-time tool usage and planning
- **Multimodal Reasoning**: Unified processing across modalities
- **Continual Learning**: Adaptation without catastrophic forgetting

## Key Takeaways

1. **No Silver Bullet**: Different approaches excel in different scenarios
2. **Efficiency-Quality Trade-off**: Most methods involve some performance compromise
3. **Hardware Matters**: Implementation details crucial for real-world gains
4. **Hybrid Solutions**: Combining approaches often yields best results
5. **Domain Specificity**: Optimal architectures vary by application

## Notation Summary

### Common Symbols
- **N**: Sequence length
- **d**: Hidden dimension  
- **H**: Number of attention heads
- **qt, kt, vt**: Query, key, value at position t
- **St**: Hidden state/memory at time t
- **φ(·)**: Feature mapping function
- **⊙**: Element-wise multiplication
- **⊗**: Outer product

### Key Equations
- **Attention**: `Attention(Q,K,V) = softmax(QKᵀ/√d)V`
- **Linear Attention**: `Output = φ(Q)(φ(K)ᵀV)`
- **SSM**: `x'(t) = Ax(t) + Bu(t), y(t) = Cx(t)`
- **MoE**: `Y = Σk Top-k(Router(x)) · Expertk(x)`

This comprehensive survey provides both theoretical foundations and practical guidance for developing efficient LLM architectures, highlighting the diverse approaches available and their respective trade-offs in the pursuit of scalable artificial intelligence.