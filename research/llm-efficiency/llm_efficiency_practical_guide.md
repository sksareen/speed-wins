# LLM Efficiency Survey: Practical Implementation Guide

## Overview
This guide provides hands-on code examples and exercises to help you implement and understand the key concepts from the LLM efficiency survey. Each section includes working code, visualizations, and practical exercises.

## Setup and Environment

### Required Libraries
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List
import math
```

### Basic Utilities
```python
def visualize_attention(attention_matrix, title="Attention Weights"):
    """Visualize attention patterns"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(attention_matrix.detach().numpy(), 
                cmap='viridis', cbar=True, square=True)
    plt.title(title)
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.show()

def measure_memory_usage(model, input_size):
    """Measure memory usage of a model"""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    x = torch.randn(input_size).cuda()
    with torch.no_grad():
        _ = model(x)
    
    memory_used = torch.cuda.max_memory_allocated() / 1024**3  # GB
    return memory_used
```

## 1. Standard Attention Implementation

### Basic Self-Attention
```python
class StandardAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        # Project to Q, K, V
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        return self.W_o(context), attention_weights

# Exercise 1: Analyze attention complexity
def analyze_attention_complexity():
    """Demonstrate quadratic complexity of standard attention"""
    seq_lengths = [100, 200, 400, 800, 1600]
    d_model = 512
    num_heads = 8
    
    model = StandardAttention(d_model, num_heads)
    times = []
    
    for seq_len in seq_lengths:
        x = torch.randn(1, seq_len, d_model)
        
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        with torch.no_grad():
            _ = model(x)
        end_time.record()
        
        torch.cuda.synchronize()
        elapsed_time = start_time.elapsed_time(end_time)
        times.append(elapsed_time)
    
    # Plot complexity
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(seq_lengths, times, 'o-', label='Actual Time')
    plt.xlabel('Sequence Length')
    plt.ylabel('Time (ms)')
    plt.title('Attention Computation Time')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    theoretical_complexity = [n**2 for n in seq_lengths]
    plt.plot(seq_lengths, theoretical_complexity, 'o-', label='O(N²)')
    plt.xlabel('Sequence Length')
    plt.ylabel('Theoretical Complexity')
    plt.title('Quadratic Complexity')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
```

## 2. Linear Attention Implementation

### Feature Mapping Functions
```python
def elu_plus_one(x: torch.Tensor) -> torch.Tensor:
    """ELU + 1 feature mapping for linear attention"""
    return F.elu(x) + 1

def random_features(x: torch.Tensor, num_features: int = 64) -> torch.Tensor:
    """Random feature approximation for softmax kernel"""
    # Generate random projection matrix
    W = torch.randn(x.shape[-1], num_features, device=x.device) / math.sqrt(num_features)
    return torch.cos(x @ W)

class LinearAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 8, feature_map='elu'):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.feature_map = feature_map
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        # Project to Q, K, V
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply feature mapping
        if self.feature_map == 'elu':
            Q_mapped = elu_plus_one(Q)
            K_mapped = elu_plus_one(K)
        elif self.feature_map == 'random':
            Q_mapped = random_features(Q)
            K_mapped = random_features(K)
        
        # Linear attention computation
        KV = torch.matmul(K_mapped.transpose(-2, -1), V)  # (batch, heads, head_dim, head_dim)
        QKV = torch.matmul(Q_mapped, KV)  # (batch, heads, seq_len, head_dim)
        
        # Normalization
        K_sum = K_mapped.sum(dim=-2, keepdim=True)  # (batch, heads, 1, head_dim)
        QK_sum = torch.matmul(Q_mapped, K_sum.transpose(-2, -1))  # (batch, heads, seq_len, 1)
        output = QKV / (QK_sum + 1e-8)
        
        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.W_o(output)

# Exercise 2: Compare standard vs linear attention
def compare_attention_methods():
    """Compare standard and linear attention performance"""
    seq_lengths = [100, 200, 400, 800, 1600]
    d_model = 512
    
    standard_model = StandardAttention(d_model)
    linear_model = LinearAttention(d_model)
    
    standard_times = []
    linear_times = []
    
    for seq_len in seq_lengths:
        x = torch.randn(1, seq_len, d_model)
        
        # Standard attention
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        with torch.no_grad():
            _ = standard_model(x)
        end_time.record()
        torch.cuda.synchronize()
        standard_times.append(start_time.elapsed_time(end_time))
        
        # Linear attention
        start_time.record()
        with torch.no_grad():
            _ = linear_model(x)
        end_time.record()
        torch.cuda.synchronize()
        linear_times.append(start_time.elapsed_time(end_time))
    
    # Plot comparison
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(seq_lengths, standard_times, 'o-', label='Standard Attention')
    plt.plot(seq_lengths, linear_times, 's-', label='Linear Attention')
    plt.xlabel('Sequence Length')
    plt.ylabel('Time (ms)')
    plt.title('Computation Time Comparison')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    speedup = [s/l for s, l in zip(standard_times, linear_times)]
    plt.plot(seq_lengths, speedup, 'o-', label='Speedup')
    plt.xlabel('Sequence Length')
    plt.ylabel('Speedup Factor')
    plt.title('Linear Attention Speedup')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    theoretical_speedup = [n for n in seq_lengths]
    plt.plot(seq_lengths, theoretical_speedup, 'o-', label='Theoretical O(N)')
    plt.xlabel('Sequence Length')
    plt.ylabel('Theoretical Speedup')
    plt.title('Theoretical vs Actual Speedup')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
```

## 3. State Space Models (SSMs)

### Basic SSM Implementation
```python
class StateSpaceModel(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # State space parameters
        self.A = nn.Parameter(torch.randn(d_state, d_state) / math.sqrt(d_state))
        self.B = nn.Parameter(torch.randn(d_state, d_model))
        self.C = nn.Parameter(torch.randn(d_model, d_state))
        self.D = nn.Parameter(torch.randn(d_model, d_model))
        
        # Discretization parameters
        self.delta = nn.Parameter(torch.ones(1))
        
    def discretize(self):
        """Discretize continuous SSM using zero-order hold"""
        delta = self.delta
        A_cont = self.A
        B_cont = self.B
        
        # Discretization
        A_disc = torch.matrix_exp(delta * A_cont)
        B_disc = torch.linalg.solve(A_cont, (A_disc - torch.eye(self.d_state, device=A_cont.device))) @ B_cont
        
        return A_disc, B_disc
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        # Discretize
        A, B = self.discretize()
        
        # Initialize state
        h = torch.zeros(batch_size, self.d_state, device=x.device)
        outputs = []
        
        # Recurrent computation
        for t in range(seq_len):
            h = torch.matmul(h, A.T) + torch.matmul(x[:, t, :], B.T)
            y_t = torch.matmul(h, self.C.T) + torch.matmul(x[:, t, :], self.D.T)
            outputs.append(y_t)
        
        return torch.stack(outputs, dim=1)

# Exercise 3: SSM vs Attention comparison
def compare_ssm_attention():
    """Compare SSM and attention on long sequences"""
    seq_lengths = [100, 500, 1000, 2000, 4000]
    d_model = 256
    
    ssm_model = StateSpaceModel(d_model)
    attention_model = StandardAttention(d_model)
    
    ssm_times = []
    attention_times = []
    
    for seq_len in seq_lengths:
        x = torch.randn(1, seq_len, d_model)
        
        # SSM
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        with torch.no_grad():
            _ = ssm_model(x)
        end_time.record()
        torch.cuda.synchronize()
        ssm_times.append(start_time.elapsed_time(end_time))
        
        # Attention (only for shorter sequences due to memory)
        if seq_len <= 1000:
            start_time.record()
            with torch.no_grad():
                _ = attention_model(x)
            end_time.record()
            torch.cuda.synchronize()
            attention_times.append(start_time.elapsed_time(end_time))
        else:
            attention_times.append(float('inf'))
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.plot(seq_lengths, ssm_times, 'o-', label='SSM (O(N))')
    plt.plot(seq_lengths[:len(attention_times)], 
             [t for t in attention_times if t != float('inf')], 
             's-', label='Attention (O(N²))')
    plt.xlabel('Sequence Length')
    plt.ylabel('Time (ms)')
    plt.title('SSM vs Attention Scaling')
    plt.legend()
    plt.yscale('log')
    plt.show()
```

## 4. Sparse Attention Implementation

### Local Attention
```python
class LocalAttention(nn.Module):
    def __init__(self, d_model: int, window_size: int = 64):
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        # Project to Q, K, V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        outputs = []
        
        for i in range(seq_len):
            # Define local window
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            
            # Extract local context
            q_i = Q[:, i:i+1, :]  # (batch, 1, d_model)
            k_local = K[:, start:end, :]  # (batch, window_size, d_model)
            v_local = V[:, start:end, :]  # (batch, window_size, d_model)
            
            # Compute local attention
            scores = torch.matmul(q_i, k_local.transpose(-2, -1)) / math.sqrt(d_model)
            attention_weights = F.softmax(scores, dim=-1)
            context = torch.matmul(attention_weights, v_local)
            
            outputs.append(context)
        
        output = torch.cat(outputs, dim=1)
        return self.W_o(output)

### Dilated Attention
```python
class DilatedAttention(nn.Module):
    def __init__(self, d_model: int, dilation: int = 2):
        super().__init__()
        self.d_model = d_model
        self.dilation = dilation
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        # Project to Q, K, V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        outputs = []
        
        for i in range(seq_len):
            # Define dilated positions
            positions = []
            pos = i
            while pos >= 0:
                positions.append(pos)
                pos -= self.dilation
            positions = positions[::-1]  # Reverse to get chronological order
            
            if positions:
                q_i = Q[:, i:i+1, :]
                k_dilated = K[:, positions, :]
                v_dilated = V[:, positions, :]
                
                # Compute dilated attention
                scores = torch.matmul(q_i, k_dilated.transpose(-2, -1)) / math.sqrt(d_model)
                attention_weights = F.softmax(scores, dim=-1)
                context = torch.matmul(attention_weights, v_dilated)
                
                outputs.append(context)
            else:
                outputs.append(torch.zeros_like(Q[:, i:i+1, :]))
        
        output = torch.cat(outputs, dim=1)
        return self.W_o(output)

# Exercise 4: Visualize sparse attention patterns
def visualize_sparse_patterns():
    """Visualize different sparse attention patterns"""
    seq_len = 64
    d_model = 256
    
    # Create models
    local_attn = LocalAttention(d_model, window_size=16)
    dilated_attn = DilatedAttention(d_model, dilation=4)
    
    x = torch.randn(1, seq_len, d_model)
    
    # Get attention patterns (simplified)
    patterns = {}
    
    # Local attention pattern
    local_pattern = torch.zeros(seq_len, seq_len)
    for i in range(seq_len):
        start = max(0, i - 8)
        end = min(seq_len, i + 8)
        local_pattern[i, start:end] = 1
    
    # Dilated attention pattern
    dilated_pattern = torch.zeros(seq_len, seq_len)
    for i in range(seq_len):
        pos = i
        while pos >= 0:
            dilated_pattern[i, pos] = 1
            pos -= 4
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.heatmap(local_pattern.numpy(), ax=axes[0], cmap='viridis', cbar=True)
    axes[0].set_title('Local Attention Pattern')
    axes[0].set_xlabel('Key Position')
    axes[0].set_ylabel('Query Position')
    
    sns.heatmap(dilated_pattern.numpy(), ax=axes[1], cmap='viridis', cbar=True)
    axes[1].set_title('Dilated Attention Pattern')
    axes[1].set_xlabel('Key Position')
    axes[1].set_ylabel('Query Position')
    
    plt.tight_layout()
    plt.show()
```

## 5. Mixture of Experts (MoE)

### Basic MoE Implementation
```python
class Expert(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.activation(self.linear1(x)))

class MoELayer(nn.Module):
    def __init__(self, d_model: int, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Router
        self.router = nn.Linear(d_model, num_experts)
        
        # Experts
        self.experts = nn.ModuleList([
            Expert(d_model, d_model * 4) for _ in range(num_experts)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        # Get routing probabilities
        router_logits = self.router(x)  # (batch, seq_len, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Initialize output
        output = torch.zeros_like(x)
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens that use this expert
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)  # (batch, seq_len)
            
            if expert_mask.any():
                # Get tokens for this expert
                expert_tokens = x[expert_mask]  # (num_tokens, d_model)
                
                # Get routing weights for this expert
                expert_weights = top_k_probs[expert_mask]  # (num_tokens, top_k)
                expert_weights = expert_weights[:, (top_k_indices[expert_mask] == expert_idx).any(dim=-1)]
                
                # Apply expert
                expert_output = self.experts[expert_idx](expert_tokens)  # (num_tokens, d_model)
                
                # Weight by routing probability
                expert_output = expert_output * expert_weights.unsqueeze(-1)
                
                # Add to output
                output[expert_mask] += expert_output
        
        return output

# Exercise 5: Analyze MoE routing
def analyze_moe_routing():
    """Analyze MoE routing patterns and load balancing"""
    d_model = 256
    num_experts = 8
    seq_len = 128
    batch_size = 4
    
    moe_layer = MoELayer(d_model, num_experts, top_k=2)
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Get routing statistics
    router_logits = moe_layer.router(x)
    router_probs = F.softmax(router_logits, dim=-1)
    
    # Analyze expert usage
    top_k_probs, top_k_indices = torch.topk(router_probs, 2, dim=-1)
    
    # Count expert usage
    expert_usage = torch.zeros(num_experts)
    for i in range(num_experts):
        expert_usage[i] = (top_k_indices == i).sum().item()
    
    # Visualize
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.bar(range(num_experts), expert_usage)
    plt.xlabel('Expert Index')
    plt.ylabel('Number of Tokens')
    plt.title('Expert Usage Distribution')
    
    plt.subplot(1, 3, 2)
    plt.imshow(router_probs[0].T.detach().numpy(), cmap='viridis', aspect='auto')
    plt.xlabel('Token Position')
    plt.ylabel('Expert Index')
    plt.title('Routing Probabilities')
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    plt.imshow(top_k_indices[0].detach().numpy(), cmap='viridis', aspect='auto')
    plt.xlabel('Token Position')
    plt.ylabel('Top-k Rank')
    plt.title('Selected Experts (Top-2)')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()
```

## 6. Hybrid Architectures

### Mamba + Attention Hybrid
```python
class HybridMambaAttention(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, num_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # Mamba components
        self.mamba = StateSpaceModel(d_model, d_state)
        
        # Attention components
        self.attention = StandardAttention(d_model, num_heads)
        
        # Gating mechanism
        self.gate = nn.Linear(d_model * 2, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply both Mamba and Attention
        mamba_out = self.mamba(x)
        attn_out, _ = self.attention(x)
        
        # Combine with gating
        combined = torch.cat([mamba_out, attn_out], dim=-1)
        gate_weights = torch.sigmoid(self.gate(combined))
        
        output = gate_weights * mamba_out + (1 - gate_weights) * attn_out
        return output

# Exercise 6: Compare hybrid vs individual methods
def compare_hybrid_methods():
    """Compare hybrid architecture with individual methods"""
    seq_lengths = [100, 200, 400, 800]
    d_model = 256
    
    hybrid_model = HybridMambaAttention(d_model)
    mamba_model = StateSpaceModel(d_model)
    attention_model = StandardAttention(d_model)
    
    hybrid_times = []
    mamba_times = []
    attention_times = []
    
    for seq_len in seq_lengths:
        x = torch.randn(1, seq_len, d_model)
        
        # Hybrid
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        with torch.no_grad():
            _ = hybrid_model(x)
        end_time.record()
        torch.cuda.synchronize()
        hybrid_times.append(start_time.elapsed_time(end_time))
        
        # Mamba
        start_time.record()
        with torch.no_grad():
            _ = mamba_model(x)
        end_time.record()
        torch.cuda.synchronize()
        mamba_times.append(start_time.elapsed_time(end_time))
        
        # Attention
        start_time.record()
        with torch.no_grad():
            _ = attention_model(x)
        end_time.record()
        torch.cuda.synchronize()
        attention_times.append(start_time.elapsed_time(end_time))
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.plot(seq_lengths, hybrid_times, 'o-', label='Hybrid')
    plt.plot(seq_lengths, mamba_times, 's-', label='Mamba')
    plt.plot(seq_lengths, attention_times, '^-', label='Attention')
    plt.xlabel('Sequence Length')
    plt.ylabel('Time (ms)')
    plt.title('Hybrid vs Individual Methods')
    plt.legend()
    plt.show()
```

## 7. Performance Benchmarking

### Comprehensive Benchmark
```python
def run_comprehensive_benchmark():
    """Run comprehensive benchmark of all methods"""
    methods = {
        'Standard Attention': StandardAttention(256),
        'Linear Attention': LinearAttention(256),
        'Local Attention': LocalAttention(256, window_size=64),
        'Dilated Attention': DilatedAttention(256, dilation=4),
        'SSM': StateSpaceModel(256),
        'MoE': MoELayer(256, num_experts=8),
        'Hybrid': HybridMambaAttention(256)
    }
    
    seq_lengths = [100, 200, 400, 800]
    results = {}
    
    for method_name, model in methods.items():
        times = []
        memory_usage = []
        
        for seq_len in seq_lengths:
            x = torch.randn(1, seq_len, 256)
            
            # Measure time
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            with torch.no_grad():
                _ = model(x)
            end_time.record()
            torch.cuda.synchronize()
            times.append(start_time.elapsed_time(end_time))
            
            # Measure memory
            memory = measure_memory_usage(model, (1, seq_len, 256))
            memory_usage.append(memory)
        
        results[method_name] = {'times': times, 'memory': memory_usage}
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Time comparison
    for method_name, data in results.items():
        axes[0].plot(seq_lengths, data['times'], 'o-', label=method_name)
    axes[0].set_xlabel('Sequence Length')
    axes[0].set_ylabel('Time (ms)')
    axes[0].set_title('Computation Time Comparison')
    axes[0].legend()
    axes[0].set_yscale('log')
    
    # Memory comparison
    for method_name, data in results.items():
        axes[1].plot(seq_lengths, data['memory'], 'o-', label=method_name)
    axes[1].set_xlabel('Sequence Length')
    axes[1].set_ylabel('Memory Usage (GB)')
    axes[1].set_title('Memory Usage Comparison')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
    
    return results

# Run the benchmark
if __name__ == "__main__":
    print("Running comprehensive benchmark...")
    results = run_comprehensive_benchmark()
    print("Benchmark completed!")
```

## 8. Practical Exercises

### Exercise 1: Implement Your Own Attention Variant
```python
def implement_custom_attention():
    """Implement a custom attention variant"""
    # TODO: Implement your own attention mechanism
    # Consider: hierarchical attention, multi-scale attention, etc.
    pass
```

### Exercise 2: Optimize for Specific Use Case
```python
def optimize_for_use_case():
    """Optimize architecture for specific use case"""
    # TODO: Design architecture for:
    # - Real-time chat (low latency)
    # - Document processing (long sequences)
    # - Resource-constrained devices
    pass
```

### Exercise 3: Create Visualization Tools
```python
def create_visualization_tools():
    """Create tools for visualizing attention patterns"""
    # TODO: Implement:
    # - Attention head visualization
    # - Routing pattern visualization
    # - Memory usage over time
    pass
```

This practical guide provides hands-on experience with all the key concepts from the LLM efficiency survey. Work through each section, run the exercises, and experiment with the code to build deep understanding of these efficient architectures.
