import os
import modal
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
from reference_implementation import generate_input, ref_kernel, to_blocked

app = modal.App("blackwell")
image = (
    modal.Image.from_registry("nvidia/cuda:13.1.1-devel-ubuntu22.04", add_python="3.11")
    .run_commands("apt update && apt install -y git")
    .uv_pip_install("torch", "nvidia-cutlass-dsl", "jupyter")
    .run_commands("git clone https://github.com/NVIDIA/cutlass")
    .workdir("/data")
)

vol = modal.Volume.from_name("blackwell-volume", create_if_missing=True)


def compile_locally():
    nvcc_res = subprocess.run(
        [
            "nvcc", "nvfp4_v4.cu", "-o", "out", "-lcuda", "-cudart", "static",
            "-gencode", "arch=compute_100a,code=sm_100a"
        ],
        capture_output=True,
        text=True,
    )
    print("nvcc return code:", nvcc_res.returncode)
    print("nvcc stdout:", nvcc_res.stdout)
    print("nvcc stderr:", nvcc_res.stderr)


@app.cls(image=image, volumes={"/data": vol}, gpu="B200", scaledown_window = 2)
class BlackwellRunner():
    @modal.enter()
    def setup(self):
        self.root = "/data"
        print("Runner init ✅")

    @modal.method()
    def compile(self):
        nvcc_res = subprocess.run(
            [
                "nvcc", "nvfp4_v4.cu", "-o", "out", "-lcuda",
                "-gencode", "arch=compute_100a,code=sm_100a"
            ],
            capture_output=True,
            text=True,
        )
        print("nvcc return code:", nvcc_res.returncode)
        print("nvcc stdout:", nvcc_res.stdout)
        print("nvcc stderr:", nvcc_res.stderr)
        vol.commit()

    @modal.method()
    def run(self):
        os.system("chmod +x out")
        os.system("./out 2>&1 | tee out.log")
        vol.commit()

    @modal.method()
    def run_ref(self, m, n, k):
        data = generate_input(m=m, n=n, k=k, l=1, seed=0)
        a_ref, b_ref, sfa_ref_cpu, sfb_ref_cpu, sfa_ref_permuted, sfb_ref_permuted, c = data
        with open("a_mat.bin", "wb") as file:
            file.write(a_ref.cpu().view(torch.int8).numpy().tobytes())
        with open("b_mat.bin", "wb") as file:
            file.write(b_ref.cpu().view(torch.int8).numpy().tobytes())
        with open("a_scales.bin", "wb") as file:
            file.write(to_blocked(sfa_ref_cpu[:, :, 0]).cpu().view(torch.int8).numpy().tobytes())
        with open("b_scales.bin", "wb") as file:
            file.write(to_blocked(sfb_ref_cpu[:, :, 0]).cpu().view(torch.int8).numpy().tobytes())
        c_ref = ref_kernel(data)
        with open("c_ref.bin", "wb") as file:
            file.write(c_ref.view(torch.float16).cpu().numpy().tobytes())
        vol.commit()

@app.local_entrypoint()
def main():
    M = 4096
    N = 4096
    K = 4096
    compile_locally()
    with vol.batch_upload(force = True) as batch:
        batch.put_file("out", "out")
    runner = BlackwellRunner()
    runner.run_ref.remote(M, N, K)
    runner.run.remote()