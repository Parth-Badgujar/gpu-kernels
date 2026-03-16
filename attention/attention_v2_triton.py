import os 
os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
import triton
import triton.language as tl
import torch
import torch.nn as nn
import torch.nn.functional as F

@triton.autotune(
    configs = [
        triton.Config({"BLOCK_Q" : 32, "BLOCK_KV" : 64}, num_warps = 4),
        triton.Config({"BLOCK_Q" : 64, "BLOCK_KV" : 32}, num_warps = 4),
        triton.Config({"BLOCK_Q" : 32, "BLOCK_KV" : 64}, num_warps = 8),
        triton.Config({"BLOCK_Q" : 64, "BLOCK_KV" : 32}, num_warps = 8),
        triton.Config({"BLOCK_Q" : 32, "BLOCK_KV" : 64}, num_warps = 16),
        triton.Config({"BLOCK_Q" : 64, "BLOCK_KV" : 32}, num_warps = 16),
    ],
    key = ['q_len', 'kv_len']
)
@triton.jit
def _flash_attention_v2(
    Q, K, V, O,
    bs, head, q_len, kv_len,
    stride_B, stride_H, stride_N, stride_D,
    DIM: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    b_id  = tl.program_id(0)
    h_id  = tl.program_id(1)
    qb_id = tl.program_id(2)  # query block id

    Q = Q + b_id * stride_B + h_id * stride_H
    K = K + b_id * stride_B + h_id * stride_H
    V = V + b_id * stride_B + h_id * stride_H
    O = O + b_id * stride_B + h_id * stride_H

    q_start = qb_id * BLOCK_Q
    q_idx   = q_start + tl.arange(0, BLOCK_Q)
    d_idx   = tl.arange(0, DIM)

    q_mask = q_idx < q_len
    q_ptrs = Q + q_idx[:, None] * stride_N + d_idx[None, :] * stride_D
    o_ptrs = O + q_idx[:, None] * stride_N + d_idx[None, :] * stride_D

    q_block = tl.load(q_ptrs, mask=q_mask[:, None], other=0)
    scale = tl.rsqrt(tl.cast(DIM, tl.float32))

    m_i = tl.full((BLOCK_Q,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_Q,), tl.float32)
    o_i = tl.zeros((BLOCK_Q, DIM), tl.float32)
    
    for kv_start in tl.range(0, kv_len, BLOCK_KV, num_stages = 2):
        kv_idx  = kv_start + tl.arange(0, BLOCK_KV)
        kv_mask = kv_idx < kv_len

        k_ptrs = K + kv_idx[:, None] * stride_N + d_idx[None, :] * stride_D
        v_ptrs = V + kv_idx[:, None] * stride_N + d_idx[None, :] * stride_D

        k_block = tl.load(k_ptrs, mask=kv_mask[:, None], other=0)
        v_block = tl.load(v_ptrs, mask=kv_mask[:, None], other=0)


        S = tl.dot(q_block, tl.trans(k_block)) * scale
        m_block = tl.max(S, axis=1)
                
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)

        P = tl.exp(S - m_new[:, None]) * q_mask[:, None] 
        P = P.to(tl.bfloat16)
        l_i = alpha * l_i + tl.sum(P, axis=1)
        o_i = alpha[:, None] * o_i + tl.dot(P, v_block)
        m_i = m_new

    out = o_i / l_i[:, None]
    tl.store(o_ptrs, out.to(tl.bfloat16), mask=q_mask[:, None])

def flash_attention_v2_triton(q, k, v):
    B, H, q_len, D = q.shape
    stride_B, stride_H, stride_N, stride_D = q.stride()
    kv_len = k.size(2)
    o = torch.empty_like(q)
    grid = lambda meta: (B, H, triton.cdiv(q_len, meta["BLOCK_Q"]))
    _flash_attention_v2[grid](q, k, v, o, B, H, q_len, kv_len, stride_B, stride_H, stride_N, stride_D, DIM = D)
    return o


if __name__ == "__main__":
    B, H, N, D = 16, 12, 1024, 128
    q = torch.randn(B, H, N, D, dtype = torch.bfloat16, device = "cuda")
    k = torch.randn(B, H, N, D, dtype = torch.bfloat16, device = "cuda")
    v = torch.randn(B, H, N, D, dtype = torch.bfloat16, device = "cuda")
    sdpa_time   = triton.testing.do_bench(lambda : F.scaled_dot_product_attention(q, k, v), warmup = 10, rep = 1000)
    triton_time = triton.testing.do_bench(lambda : flash_attention_v2_triton(q, k, v), warmup = 10, rep = 1000)
    print("SDPA time : ", sdpa_time)
    print("Triton time : ", triton_time)
