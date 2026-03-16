import torch

def create_tensors(M, N, K, sf_vec_size = 16, seed = 42):
    generator = torch.Generator()
    generator.manual_seed(seed)
    a = torch.randint(-128, 127, (M, K // 2), device = "cuda", generator = generator).to(torch.float4_e2m1fn_x2)
    b = torch.randint(-128, 127, (N, K // 2), device = "cuda", generator = generator).to(torch.float4_e2m1fn_x2)
    sfa = torch.randint(0, 4, (M, K // sf_vec_size), device = "cuda", generator = generator).to(torch.float8_e4m3fn)
    sfb = torch.randint(0, 4, (N, K // sf_vec_size), device = "cuda", generator = generator).to(torch.float8_e4m3fn)
    rest_k = K // 4
    rest_m = M // 128
    rest_n = N // 128
    sfa = sfa.reshape(rest_m, 4, 32, rest_k, 4).permute(0, 3, 2, 1, 4)
    sfb = sfb.reshape(rest_n, 4, 32, rest_k, 4).permute(0, 3, 2, 1, 4)
    sfa = sfa.contiguous()
    sfb = sfb.contiguous()
    return (a, b, sfa, sfb)

def reference_kernel(a, b, sfa, sfb):
    res = torch._scaled_mm(
        a,
        b.transpose(0, 1),
        sfa,
        sfb,
        bias = None,
        out_dtype = torch.float16,
    )
    return res