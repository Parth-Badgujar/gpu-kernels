import torch
import os

M = 4096
N = 4096
os.system("modal volume get --force blackwell-volume c_ref.bin c_ref.bin")
os.system("modal volume get --force blackwell-volume c_out.bin c_out.bin")
ref_tensor = torch.frombuffer(open("c_ref.bin", "rb").read(), dtype = torch.float16).view(M, N)
out_tensor = torch.frombuffer(open("c_out.bin", "rb").read(), dtype = torch.float16).view(M, N)
print(ref_tensor)
print(out_tensor)
print("Check :", torch.allclose(ref_tensor, out_tensor, atol = 1e-3))