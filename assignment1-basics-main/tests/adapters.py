from __future__ import annotations

import os
import numpy as np
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    # 给定一个线性层的权重，计算批处理输入的线性变换。
    # 参数:
    #     d_in (int): 输入维度的大小
    #     d_out (int): 输出维度的大小
    #     weights (Float[Tensor, "d_out d_in"]): 要使用的线性层权重
    #     in_features (Float[Tensor, "... d_in"]): 要应用函数的输入张量
    # 返回:
    #     Float[Tensor, "... d_out"]: 线性模块变换后的输出。
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """

    return in_features @ weights.T


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    # 给定一个嵌入层的权重，获取一批 token id 对应的嵌入向量。
    # 参数:
    #     vocab_size (int): 词汇表中嵌入向量的数量
    #     d_model (int): 嵌入维度的大小
    #     weights (Float[Tensor, "vocab_size d_model"]): 要从中获取的嵌入向量
    #     token_ids (Int[Tensor, "..."]): 要从嵌入层获取的 token id 集合
    # 返回:
    #     Float[Tensor, "... d_model"]: 嵌入层返回的批处理嵌入向量。
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """

    return weights[token_ids]


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    # 给定 SwiGLU 网络的权重，返回使用这些权重的 SwiGLU 实现输出。
    # 参数:
    #     d_model (int): 前馈层输入和输出的维度。
    #     d_ff (int): SwiGLU 内部上投影的维度。
    #     w1_weight (Float[Tensor, "d_ff d_model"]): W1 的存储权重
    #     w2_weight (Float[Tensor, "d_model d_ff"]): W2 的存储权重
    #     w3_weight (Float[Tensor, "d_ff d_model"]): W3 的存储权重
    #     in_features (Float[Tensor, "... d_model"]): 前馈层的输入嵌入。
    # 返回:
    #     Float[Tensor, "... d_model"]: 与输入嵌入形状相同的输出嵌入。
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # 示例：
    # 如果你的 state dict 键名匹配，可以使用 `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # 你也可以手动赋值权重
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    
    return (run_silu(in_features @ w1_weight.T) * (in_features @ w3_weight.T)) @ w2_weight.T


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    # 给定 key(K)、query(Q) 和 value(V) 张量，返回缩放点积注意力的实现输出。
    # 参数:
    #     Q (Float[Tensor, " ... queries d_k"]): 查询张量
    #     K (Float[Tensor, " ... keys d_k"]): 键张量
    #     V (Float[Tensor, " ... keys d_v"]): 值张量
    #     mask (Bool[Tensor, " ... queries keys"] | None): 掩码张量，False 的位置要遮罩（变为 -inf）
    # 返回:
    #     Float[Tensor, " ... queries d_v"]: 缩放点积注意力的输出
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... keys d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    d_k = Q.shape[-1]

    scores = Q @ K.transpose(-1, -2) / d_k ** 0.5
    if mask is not None:
        assert mask.shape == scores.shape
        scores = torch.where(~mask, float('-inf'), scores) 

    return run_softmax(scores, dim=-1) @ V


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
) -> Float[Tensor, " ... sequence_length d_model"]:
    # 给定一个朴素（非批处理）多头注意力实现的 Q/K/V 投影权重，
    # 返回一个优化后的批处理实现输出。
    # 该实现应使用单次矩阵乘法处理所有注意力头的 Q/K/V 投影。
    # 此函数不应使用 RoPE。
    # 参见 Vaswani et al., 2017 第 3.2.2 节。
    # 参数:
    #     d_model (int): 前馈层输入和输出的维度。
    #     num_heads (int): 多头注意力中使用的头数。
    #     max_seq_len (int): 最大序列长度，如果你的实现需要预缓存的话。
    #     q_proj_weight (Float[Tensor, "d_model d_model"]): Q 投影的权重
    #     k_proj_weight (Float[Tensor, "d_model d_model"]): K 投影的权重
    #     v_proj_weight (Float[Tensor, "d_model d_model"]): V 投影的权重
    #     o_proj_weight (Float[Tensor, "d_model d_model"]): 输出投影的权重
    #     in_features (Float[Tensor, "... sequence_length d_model"]): 要运行实现的输入张量。
    # 返回:
    #     Float[Tensor, " ... sequence_length d_model"]: 使用给定 QKV 投影权重和输入特征，
    #     运行优化后批处理多头注意力实现得到的输出张量。
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_model"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_model"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """

    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
    d_k = d_model // num_heads
    batch_dims = in_features.shape[:-2]
    seq_len = in_features.shape[-2]

    Q = (in_features @ q_proj_weight.T).reshape(*batch_dims, seq_len, num_heads, d_k).transpose(-3, -2)
    K = (in_features @ k_proj_weight.T).reshape(*batch_dims, seq_len, num_heads, d_k).transpose(-3, -2)
    V = (in_features @ v_proj_weight.T).reshape(*batch_dims, seq_len, num_heads, d_k).transpose(-3, -2)

    result = run_scaled_dot_product_attention(Q=Q, K=K, V=V).transpose(-3, -2).reshape(*batch_dims, seq_len, d_model)

    return result @ o_proj_weight.T

def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_model"]:
    # 给定一个朴素（非批处理）多头注意力实现的 Q/K/V 投影权重，
    # 返回一个优化后的批处理实现输出。
    # 该实现应使用单次矩阵乘法处理所有注意力头的 Q/K/V 投影。
    # 此版本的 MHA 应包含 RoPE。
    # 在此情况下，RoPE 的嵌入维度必须是每个头的嵌入维度 (d_model // num_heads)。
    # 参见 Vaswani et al., 2017 第 3.2.2 节。
    # 参数:
    #     d_model (int): 前馈层输入和输出的维度。
    #     num_heads (int): 多头注意力中使用的头数。
    #     max_seq_len (int): 最大序列长度，如果你的实现需要预缓存的话。
    #     theta (float): RoPE 参数。
    #     q_proj_weight (Float[Tensor, "d_model d_model"]): Q 投影的权重
    #     k_proj_weight (Float[Tensor, "d_model d_model"]): K 投影的权重
    #     v_proj_weight (Float[Tensor, "d_model d_model"]): V 投影的权重
    #     o_proj_weight (Float[Tensor, "d_model d_model"]): 输出投影的权重
    #     in_features (Float[Tensor, "... sequence_length d_model"]): 要运行实现的输入张量。
    #     token_positions (Int[Tensor, " ... sequence_length"] | None): 可选，token 位置张量
    # 返回:
    #     Float[Tensor, " ... sequence_length d_model"]: 使用给定 QKV 投影权重和输入特征，
    #     运行优化后批处理多头注意力（含 RoPE）实现得到的输出张量。
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_model"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_model"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """

    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
    d_k = d_model // num_heads
    batch_dims = in_features.shape[:-2]
    seq_len = in_features.shape[-2]

    if token_positions is None:
        token_positions = torch.arange(seq_len, device=in_features.device)

    Q = (in_features @ q_proj_weight.T).reshape(*batch_dims, seq_len, num_heads, d_k).transpose(-3, -2)
    Q = run_rope(d_k=d_k, theta=theta, max_seq_len=max_seq_len, in_query_or_key=Q, token_positions=token_positions)
    K = (in_features @ k_proj_weight.T).reshape(*batch_dims, seq_len, num_heads, d_k).transpose(-3, -2)
    K = run_rope(d_k=d_k, theta=theta, max_seq_len=max_seq_len, in_query_or_key=K, token_positions=token_positions)
    V = (in_features @ v_proj_weight.T).reshape(*batch_dims, seq_len, num_heads, d_k).transpose(-3, -2)

    result = run_scaled_dot_product_attention(Q=Q, K=K, V=V).transpose(-3, -2).reshape(*batch_dims, seq_len, d_model)

    return result @ o_proj_weight.T


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    # 对给定输入张量运行 RoPE（旋转位置编码）。
    # 参数:
    #     d_k (int): query 或 key 张量的嵌入维度大小。
    #     theta (float): RoPE 参数。
    #     max_seq_len (int): 最大序列长度，如果你的实现需要预缓存的话。
    #     in_query_or_key (Float[Tensor, "... sequence_length d_k"]): 要运行 RoPE 的输入张量。
    #     token_positions (Int[Tensor, "... sequence_length"]): 形状为 (batch_size, sequence_length) 的 token 位置张量
    # 返回:
    #     Float[Tensor, " ... sequence_length d_k"]: 经过 RoPE 处理后的张量。
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, d_k, step=2) / d_k))

    angles = token_positions.unsqueeze(-1) * freqs

    cos_val = torch.cos(angles)
    sin_val = torch.sin(angles)

    x_even = in_query_or_key[..., 0::2]
    x_odd = in_query_or_key[..., 1::2]

    rotated_even = x_even * cos_val - x_odd * sin_val
    rotated_odd = x_even * sin_val + x_odd * cos_val

    stacked = torch.stack([rotated_even, rotated_odd], dim=-1)
    output = stacked.flatten(start_dim=-2)
    
    return output

def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    # 给定一个 pre-norm Transformer block 的权重和输入特征，
    # 返回在该输入特征上运行 Transformer block 的输出。
    # 此函数应使用 RoPE。
    # 根据你的实现方式，你可能只需要将相关参数传给 TransformerBlock 构造函数，
    # 也可能需要初始化自己的 RoPE 类并传入。
    # 参数:
    #     d_model (int): Transformer block 输入的维度。
    #     num_heads (int): 多头注意力中使用的头数。`d_model` 必须能被 `num_heads` 整除。
    #     d_ff (int): 前馈层内部层的维度。
    #     max_seq_len (int): 最大序列长度，如果你的实现需要预缓存的话。
    #     theta (float): RoPE 参数。
    #     weights (dict[str, Tensor]):
    #         参考实现的 state dict。
    #         此字典的键如下：
    #         - `attn.q_proj.weight`
    #             所有 `num_heads` 个注意力头的 query 投影。
    #             形状为 (d_model, d_model)。
    #             行按 (num_heads, d_k) 的矩阵顺序排列，
    #             即 `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`。
    #         - `attn.k_proj.weight`
    #             所有 `num_heads` 个注意力头的 key 投影。
    #             形状为 (d_model, d_model)。
    #             行按 (num_heads, d_k) 的矩阵顺序排列，
    #             即 `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`。
    #         - `attn.v_proj.weight`
    #             所有 `num_heads` 个注意力头的 value 投影。
    #             形状为 (d_model, d_model)。
    #             行按 (num_heads, d_v) 的矩阵顺序排列，
    #             即 `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`。
    #         - `attn.output_proj.weight`
    #             多头自注意力输出投影的权重
    #             形状为 (d_model, d_model)。
    #         - `ln1.weight`
    #             Transformer block 中第一个 RMSNorm 的仿射变换权重。
    #             形状为 (d_model,)。
    #         - `ffn.w1.weight`
    #             FFN 中第一个线性变换的权重。
    #             形状为 (d_ff, d_model)。
    #         - `ffn.w2.weight`
    #             FFN 中第二个线性变换的权重。
    #             形状为 (d_model, d_ff)。
    #         - `ffn.w3.weight`
    #             FFN 中第三个线性变换的权重。
    #             形状为 (d_ff, d_model)。
    #         - `ln2.weight`
    #             Transformer block 中第二个 RMSNorm 的仿射变换权重。
    #             形状为 (d_model,)。
    #     in_features (Float[Tensor, "batch sequence_length d_model"]): 要运行实现的输入张量。
    # 返回:
    #     Float[Tensor, "batch sequence_length d_model"]: 在输入特征上运行 Transformer block
    #     （使用 RoPE）得到的输出张量。
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    
    norm1 = run_rmsnorm(d_model=d_model, eps=1e-5, weights=weights["ln1.weight"], in_features=in_features)
    attn_out = run_multihead_self_attention_with_rope(d_model=d_model, num_heads=num_heads, max_seq_len=max_seq_len, theta=theta, q_proj_weight=weights["attn.q_proj.weight"], k_proj_weight=weights["attn.k_proj.weight"], v_proj_weight=weights["attn.v_proj.weight"], o_proj_weight=weights["attn.output_proj.weight"], in_features=norm1)
    y = attn_out + in_features

    norm2 = run_rmsnorm(d_model=d_model, eps=1e-5, weights=weights["ln2.weight"], in_features=y)
    acti_out = run_swiglu(d_model=d_model, d_ff=d_ff, w1_weight=weights["ffn.w1.weight"], w2_weight=weights["ffn.w2.weight"], w3_weight=weights["ffn.w3.weight"], in_features=norm2)
    tran_out = acti_out + y

    return tran_out


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    # 给定 Transformer 语言模型的权重和输入索引，
    # 返回在输入索引上运行前向传播的输出。
    # 此函数应使用 RoPE。
    # 参数:
    #     vocab_size (int): 要预测的输出词汇表中唯一项的数量。
    #     context_length (int): 一次处理的最大 token 数量。
    #     d_model (int): 模型嵌入和子层输出的维度。
    #     num_layers (int): 要使用的 Transformer 层数。
    #     num_heads (int): 多头注意力中使用的头数。`d_model` 必须能被 `num_heads` 整除。
    #     d_ff (int): 前馈层内部层的维度（第 3.3 节）。
    #     rope_theta (float): RoPE 的 $\\Theta$ 参数。
    #     weights (dict[str, Tensor]):
    #         参考实现的 state dict。{num_layers} 表示 0 到 num_layers-1 之间的整数（层索引）。
    #         此字典的键如下：
    #         - `token_embeddings.weight`
    #             Token 嵌入矩阵。形状为 (vocab_size, d_model)。
    #         - `layers.{num_layers}.attn.q_proj.weight`
    #             所有 `num_heads` 个注意力头的 query 投影。
    #             形状为 (num_heads * (d_model / num_heads), d_model)。
    #             行按 (num_heads, d_k) 的矩阵顺序排列，
    #             即 `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`。
    #         - `layers.{num_layers}.attn.k_proj.weight`
    #             所有 `num_heads` 个注意力头的 key 投影。
    #             形状为 (num_heads * (d_model / num_heads), d_model)。
    #             行按 (num_heads, d_k) 的矩阵顺序排列，
    #             即 `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`。
    #         - `layers.{num_layers}.attn.v_proj.weight`
    #             所有 `num_heads` 个注意力头的 value 投影。
    #             形状为 (num_heads * (d_model / num_heads), d_model)。
    #             行按 (num_heads, d_v) 的矩阵顺序排列，
    #             即 `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`。
    #         - `layers.{num_layers}.attn.output_proj.weight`
    #             多头自注意力输出投影的权重
    #             形状为 ((d_model / num_heads) * num_heads, d_model)。
    #         - `layers.{num_layers}.ln1.weight`
    #             Transformer block 中第一个 RMSNorm 的仿射变换权重。
    #             形状为 (d_model,)。
    #         - `layers.{num_layers}.ffn.w1.weight`
    #             FFN 中第一个线性变换的权重。
    #             形状为 (d_ff, d_model)。
    #         - `layers.{num_layers}.ffn.w2.weight`
    #             FFN 中第二个线性变换的权重。
    #             形状为 (d_model, d_ff)。
    #         - `layers.{num_layers}.ffn.w3.weight`
    #             FFN 中第三个线性变换的权重。
    #             形状为 (d_ff, d_model)。
    #         - `layers.{num_layers}.ln2.weight`
    #             Transformer block 中第二个 RMSNorm 的仿射变换权重。
    #             形状为 (d_model,)。
    #         - `ln_final.weight`
    #             应用于最后一个 Transformer block 输出的 RMSNorm 仿射变换权重。
    #             形状为 (d_model,)。
    #         - `lm_head.weight`
    #             语言模型输出嵌入的权重。
    #             形状为 (vocab_size, d_model)。
    #     in_indices (Int[Tensor, "batch_size sequence_length"]): 要在语言模型上运行的输入索引张量。
    #         形状为 (batch_size, sequence_length)，其中 `sequence_length` 最多为 `context_length`。
    # 返回:
    #     Float[Tensor, "batch_size sequence_length vocab_size"]: 每个 token 的预测
    #     未归一化下一个词分布的张量。
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """

    x = run_embedding(vocab_size=vocab_size, d_model=d_model, weights=weights["token_embeddings.weight"], token_ids=in_indices)

    for i in range(num_layers):
        prefix = f"layers.{i}."
        block_weights = {
            k.replace(prefix, ""): v
            for k, v in weights.items()
            if k.startswith(prefix)
        }
        x = run_transformer_block(d_model=d_model, num_heads=num_heads, d_ff=d_ff, max_seq_len=context_length, theta=rope_theta, weights=block_weights, in_features=x)

    # for k,v in weight.item():
    #     if k.startswith(prefix):
    #         k.replace(prefix, ""): v

    x = run_rmsnorm(d_model=d_model, eps=1e-5, weights=weights["ln_final.weight"], in_features=x)
    x = run_linear(d_in=d_model, d_out=vocab_size, weights=weights["lm_head.weight"], in_features=x)

    return x

def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    # 给定 RMSNorm 仿射变换的权重，返回在输入特征上运行 RMSNorm 的输出。
    # 参数:
    #     d_model (int): RMSNorm 输入的维度。
    #     eps: (float): 为数值稳定性而加到分母上的值。
    #     weights (Float[Tensor, "d_model"]): RMSNorm 权重。
    #     in_features (Float[Tensor, "... d_model"]): 要运行 RMSNorm 的输入特征。可以有任意前导维度。
    # 返回:
    #     Float[Tensor,"... d_model"]: 与 `in_features` 形状相同的张量，包含对 `in_features` 运行 RMSNorm 的结果。
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    xx = torch.mean(in_features * in_features, dim=-1, keepdim=True)
    return in_features * weights / torch.sqrt(xx + eps)

def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:

    # 给定一个输入张量，返回对每个元素应用 SiLU 的结果。
    # 参数:
    #     in_features(Float[Tensor, "..."]): 要运行 SiLU 的输入特征。形状任意。
    # 返回:
    #     Float[Tensor,"..."]: 与 `in_features` 形状相同的张量，包含对每个元素应用 SiLU 的结果。
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    return in_features * torch.sigmoid(in_features)

def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    # 给定数据集（一个 1D 的整数 numpy 数组）、期望的批大小和上下文长度，
    # 从数据集中采样语言建模的输入序列及其对应的标签。
    # 参数:
    #     dataset (np.array): 数据集中整数 token ID 的 1D numpy 数组。
    #     batch_size (int): 要采样的期望批大小。
    #     context_length (int): 每个采样样本的期望上下文长度。
    #     device (str): PyTorch 设备字符串（例如 'cpu' 或 'cuda:0'），
    #         指定将采样的输入序列和标签放置在哪个设备上。
    # 返回:
    #     两个形状为 (batch_size, context_length) 的 torch.LongTensor 元组。
    #     第一个元组项是采样的输入序列，第二个元组项是对应的语言建模标签。
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """

    data_len = len(dataset)
    start_indices = np.random.randint(0, data_len - context_length, size=batch_size)

    x = np.stack([dataset[i : i + context_length] for i in start_indices])
    y = np.stack([dataset[i + 1 : i + context_length + 1] for i in start_indices])

    return torch.tensor(x, dtype=torch.long, device=device), torch.tensor(y, dtype=torch.long, device=device)
    
def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    # 给定一个输入张量，返回对输入的指定 `dim` 维度做 softmax 的结果。
    # 参数:
    #     in_features (Float[Tensor, "..."]): 要做 softmax 的输入特征。形状任意。
    #     dim (int): 对 `in_features` 的哪个维度应用 softmax。
    # 返回:
    #     Float[Tensor, "..."]: 与 `in_features` 形状相同的张量，包含对指定 `dim` 做 softmax 归一化的结果。
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    max_val = torch.max(in_features, dim=dim, keepdim=True).values

    shifted = in_features - max_val

    exp_result = torch.exp(shifted)
    sum_result = torch.sum(exp_result, dim=dim, keepdim=True) 

    return exp_result / sum_result


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    # 给定输入张量和目标张量，计算所有样本的平均交叉熵损失。
    # 参数:
    #     inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] 是第 i 个样本的第 j 类的未归一化 logit。
    #     targets (Int[Tensor, "batch_size"]): 形状为 (batch_size,) 的张量，包含正确类别的索引。
    #         每个值必须在 0 到 `num_classes - 1` 之间。
    # 返回:
    #     Float[Tensor, ""]: 所有样本的平均交叉熵损失。
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    max_val = torch.max(inputs, dim=-1, keepdim=True).values
    shifted = inputs - max_val

    exp_result = torch.exp(shifted)
    sum_result = torch.sum(exp_result, dim=-1, keepdim=True)

    log_result = torch.log(sum_result)
    log_softmax = shifted - log_result

    return -(log_softmax[torch.arange(inputs.shape[0]), targets.long()]).mean()


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    # 给定一组参数，将其合并的梯度裁剪为 L2 范数不超过 max_l2_norm。
    # 参数:
    #     parameters (Iterable[torch.nn.Parameter]): 可训练参数的集合。
    #     max_l2_norm (float): 包含最大 L2 范数的正值。
    # 参数的梯度 (parameter.grad) 应原地修改。
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """

    total_norm = 0

    for p in parameters:
        if p.grad is not None:
            total_norm += torch.norm(p.grad) ** 2

    total_norm = torch.sqrt(total_norm)

    if total_norm > max_l2_norm:
        scale = max_l2_norm / total_norm
        for p in parameters:
            if p.grad is not None:
                p.grad.mul_(scale)


def get_adamw_cls() -> Any:
    # 返回一个实现 AdamW 的 torch.optim.Optimizer 类。
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    class AdamW(torch.optim.Optimizer):
        def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
            defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
            super().__init__(params, defaults)

        @torch.no_grad()
        def step(self, closure=None):

            loss = None

            # 如果传了 closure，用它再算一次 loss（多数情况不用）
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()

            for group in self.param_groups:
                beta1, beta2 = group["betas"]
                lr = group["lr"]
                weight_decay = group["weight_decay"]
                eps = group["eps"]

                for param in group["params"]:
                    if param.grad is None:
                        continue

                    # 1. 获取/初始化该参数的状态
                    state = self.state[param]
                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(param)
                        state["exp_avg_sq"] = torch.zeros_like(param)

                    # 2. 步数 +1
                    state["step"] += 1

                    # 3. 权重衰减（AdamW 关键：直接减权重，不进动量）
                    param.mul_(1 - lr * weight_decay)

                    grad = param.grad
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]

                    # 4. 更新一阶矩（惯性）
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                    # 5. 更新二阶矩（步长调节）
                    exp_avg_sq.mul_(beta2).add_(grad ** 2, alpha=1 - beta2)

                    # 6. 偏差修正
                    t = state["step"]
                    bias_corr1 = 1 - beta1 ** t
                    bias_corr2 = 1 - beta2 ** t

                    # 7. 更新参数
                    denom = (exp_avg_sq / bias_corr2).sqrt() + eps
                    step_size = lr / bias_corr1
                    param -= step_size * exp_avg / denom

            return loss  # 如果有 closure，返回 loss；否则 None

    return AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    # 给定余弦学习率衰减调度（含线性预热）的参数和一个迭代次数，
    # 返回在指定调度下该迭代次数的学习率。
    # 参数:
    #     it (int): 要获取学习率的迭代次数。
    #     max_learning_rate (float): alpha_max，余弦学习率调度的最大学习率（含预热）。
    #     min_learning_rate (float): alpha_min，余弦学习率调度的最小/最终学习率（含预热）。
    #     warmup_iters (int): T_w，线性预热学习率的迭代次数。
    #     cosine_cycle_iters (int): T_c，余弦退火的迭代次数。
    # 返回:
    #     指定调度下该迭代次数的学习率。
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    if it < warmup_iters:
        return max_learning_rate * (it / warmup_iters)
    elif warmup_iters <= it < cosine_cycle_iters:
        rate = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (1 + np.cos(np.pi * rate))
    else:
        return min_learning_rate

def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    # 给定模型、优化器和迭代次数，将它们序列化到磁盘。
    # 参数:
    #     model (torch.nn.Module): 序列化此模型的状态。
    #     optimizer (torch.optim.Optimizer): 序列化此优化器的状态。
    #     iteration (int): 序列化此值，表示我们已完成的训练迭代次数。
    #     out (str | os.PathLike | BinaryIO | IO[bytes]): 要将模型、优化器和迭代次数序列化到的路径或类文件对象。
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    pack = {
        "model_state" : model.state_dict(),
        "optimizer_state" : optimizer.state_dict(),
        "iteration" : iteration
    }

    torch.save(pack, out)

def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    # 给定一个序列化的 checkpoint（路径或类文件对象），
    # 将序列化状态恢复到给定的模型和优化器中。
    # 返回之前在 checkpoint 中序列化的迭代次数。
    # 参数:
    #     src (str | os.PathLike | BinaryIO | IO[bytes]): 序列化 checkpoint 的路径或类文件对象。
    #     model (torch.nn.Module): 恢复此模型的状态。
    #     optimizer (torch.optim.Optimizer): 恢复此优化器的状态。
    # 返回:
    #     int: 之前序列化的迭代次数。
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    pack = torch.load(src, map_location="cpu")
    model.load_state_dict(pack["model_state"])
    optimizer.load_state_dict(pack["optimizer_state"])

    return pack["iteration"]


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    # 给定一个词表、一个 merges 列表和一个特殊 token 列表，
    # 返回一个使用所提供的 vocab、merges 和特殊 token 的 BPE tokenizer。
    # 参数:
    #     vocab (dict[int, bytes]): tokenizer 词表，从 int（词表中的 token ID）到 bytes（token 字节）的映射
    #     merges (list[tuple[bytes, bytes]]): BPE merges。每个列表项是一个 bytes 元组 (<token1>, <token2>)，
    #         表示 <token1> 与 <token2> 合并。Merges 按创建顺序排列。
    #     special_tokens (list[str] | None): tokenizer 的字符串特殊 token 列表。这些字符串永远不会
    #         被拆分为多个 token，始终作为单个 token 保留。
    # 返回:
    #     一个使用所提供的 vocab、merges 和特殊 token 的 BPE tokenizer。
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    raise NotImplementedError


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 给定输入语料文件的路径，训练一个 BPE tokenizer 并输出其词表和 merges。
    # 参数:
    #     input_path (str | os.PathLike): BPE tokenizer 训练数据的路径。
    #     vocab_size (int): tokenizer 词表中的总项数（包括特殊 token）。
    #     special_tokens (list[str]): 要添加到 tokenizer 词表中的字符串特殊 token 列表。
    #         这些字符串永远不会被拆分为多个 token，始终作为单个 token 保留。
    #         如果这些特殊 token 出现在 `input_path` 中，它们被视为普通字符串处理。
    # 返回:
    #     tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    #         vocab:
    #             训练好的 tokenizer 词表，从 int（词表中的 token ID）到 bytes（token 字节）的映射
    #         merges:
    #             BPE merges。每个列表项是一个 bytes 元组 (<token1>, <token2>)，
    #             表示 <token1> 与 <token2> 合并。
    #             Merges 按创建顺序排列。
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    raise NotImplementedError
