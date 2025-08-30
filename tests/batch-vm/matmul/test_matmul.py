import torch
from kvm_matmul import kvm_matmul

M, N, K = 8192, 8192, 16384

a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") / K**.25
b = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / K**.25
d = torch.zeros((M, N), dtype=torch.bfloat16, device="cuda")

def make_instructions(M,N,K):
    instructions = []
    for super_m in range(0, M, 4096):
        for n in range(N//256):
            for m256 in range(super_m, min(super_m+4096, M), 256):
                instructions.append([1,m256//256,n,K//64]+[0]*28)
    while len(instructions) % 148 != 0:
        instructions.append([0]*32)
    instructions = torch.tensor(instructions, dtype=torch.int32, device="cuda").reshape(-1,148,32).transpose(0,1).contiguous()
    timings = torch.zeros((148, instructions.shape[1], 128), dtype=torch.int32, device=instructions.device)
    return instructions, timings

instructions, timings = make_instructions(M,N,K)

print(instructions.shape)
print(timings.shape)

# warmup

kvm_matmul(instructions, timings, a, b, d)
torch.cuda.synchronize()

# do another

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
kvm_matmul(instructions, timings, a, b, d)
end_event.record()
torch.cuda.synchronize()

time_us = start_event.elapsed_time(end_event)*1000
print(f"Matmul runtime: {time_us:.1f} us")
torch.cuda.synchronize()

print(f"TFLOPs: {2*M*N*K/(time_us*1e6):.1f}")

ref = a@b.T
print(d)
print(ref)
print((d-ref).abs().max())