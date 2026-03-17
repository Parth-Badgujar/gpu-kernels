import os
import modal
import subprocess
import torch
import triton
import torch.nn as nn
import torch.nn.functional as F

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

app = modal.App("blackwell")
image = (
    modal.Image.from_registry("nvidia/cuda:13.1.1-devel-ubuntu24.04", add_python="3.11")
    .run_commands("apt update && apt install -y git")
    .uv_pip_install("torch", "nvidia-cutlass-dsl", "jupyter")
    .run_commands("git clone https://github.com/NVIDIA/cutlass")
    .workdir("/data")
)

vol = modal.Volume.from_name("blackwell-volume", create_if_missing=True)

def compile_locally():
    os.system("cmake -S . -B build")
    os.system("cmake --build build")

def verify_output(filename, atol, rtol):
    data_ref = open("c_ref.bin", "rb").read()
    data = open(filename, "rb").read()
    ref = torch.frombuffer(data_ref, dtype = torch.float16)
    out = torch.frombuffer(data, dtype = torch.float16)
    return torch.allclose(ref, out, atol = atol, rtol = rtol)

@app.cls(image=image, volumes={"/data": vol}, gpu="B200", scaledown_window = 2)
class BlackwellRunner():
    @modal.enter()
    def setup(self):
        self.root = "/data"
        self.prev_refsize = [-1, -1, -1]
        print("Runner init ✅")

    @modal.method()
    def run_verify(self, m, n, k):
        if self.prev_refsize != [m, n, k]:
            print("[*] Run reference before CUDA")
            return
        os.system("chmod +x ./out")
        os.system(f"./out {m} {n} {k} verify")
        vol.commit()
        print("Verify")
        print("v4 : ", verify_output("c_out_v4.bin", 1e-3, 1e-3))
        print("v3 : ", verify_output("c_out_v3.bin", 1e-3, 1e-3))
        print("v2 : ", verify_output("c_out_v2.bin", 1e-3, 1e-3))

    @modal.method()
    def run_benchmark(self, m, n, k):
        if self.prev_refsize != [m, n, k]:
            print("[*] Run reference before CUDA")
            return
        output = subprocess.run(f"./out {m} {n} {k} benchmark", shell = True, capture_output = True)
        time_taken = list(map(float, output.stdout.decode().split()))
        return time_taken

    @modal.method()
    def run_ref(self, m, n, k, bench = True):
        self.prev_refsize = [m, n, k]
        a, b, sfa, sfb = create_tensors(M=m, N=n, K=k)
        save_tensors(a, b, sfa, sfb)
        c_ref = reference_kernel(a, b, sfa, sfb)
        with open("c_ref.bin", "wb") as file:
            file.write(c_ref.cpu().numpy().tobytes())
        vol.commit()
        if bench:
            time_taken = triton.testing.do_bench(lambda : reference_kernel(a, b, sfa, sfb), warmup = 10, rep = 1000)
            return time_taken
        return None

@app.local_entrypoint()
def main():
    mnk = [
        (1024, 1024, 1024),
        (1024, 1024, 2048),
        (1024, 1024, 4096),
        (2048, 2048, 2048),
        (2048, 2048, 4096),
        (2048, 2048, 8192),
        (4096, 4096, 4096),
        (4096, 4096, 8192),
        (4096, 4096, 16384),
        (8192, 8192, 8192),
        (8192, 8192, 4096),
        (8192, 8192, 16384),
    ]
    compile_locally()
    with vol.batch_upload(force = True) as batch:
        batch.put_file("./build/nvfp4_gemm", "out")
    runner = BlackwellRunner()
    for M, N, K in mnk:
        print("M = {} N = {} K = {}".format(M, N, K))
        runner.run_ref.remote(M, N, K, bench = False)
        runner.run_verify.remote(M, N, K)
        cuda_time_v4, cuda_time_v3, cuda_time_v2 = runner.run_benchmark.remote(M, N, K)
        torch_time = runner.run_ref.remote(M, N, K, bench = True)
        print("Torch (TFLOPS):", 2 * M * N * K / (1e9 * torch_time))
        print("CUDA v4 (TFLOPS):", 2 * M * N * K / (1e9 * cuda_time_v4))
        print("CUDA v3 (TFLOPS):", 2 * M * N * K / (1e9 * cuda_time_v3))
        print("CUDA v2 (TFLOPS):", 2 * M * N * K / (1e9 * cuda_time_v2))
        print("---------------------------------------")
        