import torch
import triton

def create_tensors(M, N, K, sf_vec_size = 16, seed = 42):
    generator = torch.Generator(device = "cuda")
    generator.manual_seed(seed)
    a = torch.randint(-128, 127, (M, K // 2), dtype = torch.int8, device = "cuda", generator = generator).view(torch.float4_e2m1fn_x2)
    b = torch.randint(-128, 127, (N, K // 2), dtype = torch.int8, device = "cuda", generator = generator).view(torch.float4_e2m1fn_x2)
    sfa = torch.randint(0, 4, (M, K // sf_vec_size), dtype = torch.int8, device = "cuda", generator = generator).view(torch.float8_e4m3fn)
    sfb = torch.randint(0, 4, (N, K // sf_vec_size), dtype = torch.int8, device = "cuda", generator = generator).view(torch.float8_e4m3fn)
    rest_k = K // (sf_vec_size * 4)
    rest_m = M // 128
    rest_n = N // 128
    sfa = sfa.reshape(rest_m, 4, 32, rest_k, 4).permute(0, 3, 2, 1, 4)
    sfb = sfb.reshape(rest_n, 4, 32, rest_k, 4).permute(0, 3, 2, 1, 4)
    sfa = sfa.contiguous().flatten()
    sfb = sfb.contiguous().flatten()
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

def save_tensors(a, b, sfa, sfb):
    with open("a_mat.bin", "wb") as file:
        file.write(a.cpu().view(torch.int8).numpy().tobytes())
    with open("b_mat.bin", "wb") as file:
        file.write(b.cpu().view(torch.int8).numpy().tobytes())
    with open("a_scales.bin", "wb") as file:
        file.write(sfa.cpu().view(torch.int8).numpy().tobytes())
    with open("b_scales.bin", "wb") as file:
        file.write(sfb.cpu().view(torch.int8).numpy().tobytes())

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-M", type = int, default = 1024)
    parser.add_argument("-N", type = int, default = 1024)
    parser.add_argument("-K", type = int, default = 256)
    args = parser.parse_args()
    M = args.M
    N = args.N
    K = args.K
    a, b, sfa, sfb = create_tensors(M, N, K)
    ref = reference_kernel(a, b, sfa, sfb)
    warmup = 500
    rep = 100000
    time_taken = triton.testing.do_bench(lambda: reference_kernel(a, b, sfa, sfb), warmup = warmup, rep = rep)
    print("Reference (TFLOPS) :", (2 * M * N * K) / (time_taken * 1e9))
