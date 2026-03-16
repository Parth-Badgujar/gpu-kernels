import os 
os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
import triton
import triton.language as tl
import torch
import torch.nn.functional as f

@triton.autotune(
    configs = [
        triton.Config({"BLOCK_SIZE" : bs}, num_warps = nw, num_stages = ns) 
        for bs in [128, 256, 512, 1024, 2048]
        for nw in [4, 8, 16]
        for ns in [1, 2, 3]
        ],
    key = ['F', 'D1', 'D2']
)
@triton.jit
def _layer_norm_simple(
    src_ptr,
    dst_ptr,
    gamma_ptr,
    beta_ptr,
    stride_b,
    B, F, D1, D2,
    BLOCK_SIZE: tl.constexpr,
):
    bid = tl.program_id(0)
    src_ptr = src_ptr + bid * stride_b
    dst_ptr = dst_ptr + bid * stride_b
    features = F * D1 * D2

    idx = tl.arange(0, BLOCK_SIZE)
    mean = tl.zeros((BLOCK_SIZE, ), dtype = tl.float32)
    var = tl.zeros((BLOCK_SIZE, ), dtype = tl.float32)
    for i in range(0, features, BLOCK_SIZE):
        mask = (idx + i) < features
        data = tl.load(src_ptr + idx + i, mask = mask)
        mean += data
        var += (data * data)

    mean = tl.sum(mean, axis = 0)
    var = tl.sum(var, axis = 0)
    mean /= features
    var = var / features - (mean * mean) + 1e-5
    var = tl.sqrt(var + 1e-5)
    
    
    for i in range(0, features, BLOCK_SIZE):
        mask = idx + i < features
        gamma = tl.load(gamma_ptr + idx + i, mask = mask)
        beta = tl.load(beta_ptr + idx + i, mask = mask)
        data = tl.load(src_ptr + idx + i, mask = mask)
        res = (data - mean) / var * gamma + beta
        tl.store(dst_ptr + idx + i, res, mask = mask)


def layer_norm_triton(x, gamma, beta):
    y = torch.empty_like(x)
    B, F, D1, D2 = x.shape
    stride_b = F * D1 * D2
    grid = lambda meta: (B,)
    _layer_norm_simple[grid](x, y, gamma, beta, stride_b, B, F, D1, D2)
    return y

@torch.compile
def layer_norm_compiled(x, gamma, beta):
    B, F, D1, D2 = x.shape
    return f.layer_norm(x, (F, D1, D2), gamma, beta, eps = 1e-5)

if __name__ == "__main__":
    B = 32
    F = 128
    D1 = 64
    D2 = 64

    x = torch.randn(B, F, D1, D2, device = 'cuda', dtype = torch.float32)
    gamma = torch.randn(F, D1, D2, device = 'cuda', dtype = torch.float32)
    beta = torch.randn(F, D1, D2, device = 'cuda', dtype = torch.float32)

    result = layer_norm_triton(x, gamma, beta)
    result_ref = f.layer_norm(x, (F, D1, D2), gamma, beta, eps = 1e-5)

    # print(torch.allclose(result, result_ref, atol = 1e-3))

    triton_time = triton.testing.do_bench(lambda : layer_norm_triton(x, gamma, beta), warmup = 10, rep = 1000)
    torch_time = triton.testing.do_bench(lambda : f.layer_norm(x, (F, D1, D2), gamma, beta, eps = 1e-5), warmup = 10, rep = 1000)
    compiled_time = triton.testing.do_bench(lambda : layer_norm_compiled(x, gamma, beta), warmup = 10, rep = 1000)


    print("torch GFLOPs :", (B * (F * D1 * D2 * 7 + 6) / torch_time) / 1e6)
    print("triton GFLOPs :",( B * (F * D1 * D2 * 7 + 6) / triton_time) / 1e6)
    print("compiled GFLOPs :",( B * (F * D1 * D2 * 7 + 6) / compiled_time) / 1e6)
    
    


    
    
    
    


