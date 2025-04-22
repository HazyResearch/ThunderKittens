import sys
import time

import torch
from make_instructions import make_instructions
from silu_mlp import silu_mlp as silu_mlp_tk
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from einops import einsum


torch.manual_seed(0)

hidden_size = 2048
intermediate_size = 2048


################# CREATE PYTORCH REFERENCE ######################

def matvec(mat: Tensor, vec: Tensor):
    out = einsum(mat, vec, "o i, i -> o")
    return out


def silu_mlp_reference(inp: Tensor, up_proj: nn.Linear, down_proj: nn.Linear, gate_proj: nn.Linear) -> Tensor:

    up_matvec = matvec(
        mat=up_proj, 
        vec=inp,
    )

    gate_matvec = matvec(
        mat=gate_proj,
        vec=inp,
    )

    post_silu = F.silu(gate_matvec) * up_matvec

    out = matvec(
        mat=down_proj, 
        vec=post_silu,
    )

    return out


################## CREATE INPUTS ######################

DEPTH = 1

# Create input and output tensors
UP_PROJ_W = (
    torch.randn((intermediate_size, hidden_size), device=0, dtype=torch.float32) / 2048**0.25
).to(torch.bfloat16)

GATE_PROJ_W = (
    torch.randn((intermediate_size, hidden_size), device=0, dtype=torch.float32) / 2048**0.25
).to(torch.bfloat16)

DOWN_PROJ_W = (
    torch.randn((hidden_size, intermediate_size), device=0, dtype=torch.float32) / 2048**0.25
).to(torch.bfloat16)

INP = (
    torch.randn((hidden_size), device=0, dtype=torch.float32) / 2048**0.25
).to(torch.bfloat16)

O = torch.zeros(
    (1, hidden_size), device=0, dtype=torch.bfloat16
)  # noqa: E741

Bar = torch.ones((1, 6, 32), device=0, dtype=torch.int32)


print("Input tensors created, of shapes", f"{INP.shape=}\n{UP_PROJ_W.shape=}\n{DOWN_PROJ_W.shape=}\n{GATE_PROJ_W.shape}\n{O.shape}\n\n")

sys.stdout.flush()


# Create instruction and timing tensors
instructions, timings = make_instructions(DEPTH)

print(
    f"Instruction and timing tensors created, of shapes {instructions.shape} and {timings.shape}"
)
print(instructions.float().mean())


############# CORE FUNCTIONS ##################

# Run the matvec kernel
def go():
    silu_mlp_tk(
        instructions, 
        timings, 
        UP_PROJ_W, 
        DOWN_PROJ_W, 
        GATE_PROJ_W, 
        INP, 
        O,
        Bar
    )


def reference_go():
    O2 = silu_mlp_reference(
        INP,
        UP_PROJ_W,
        DOWN_PROJ_W,
        GATE_PROJ_W,
    )
    return O2


############## EVALUATE TIMINGS ######################

print("Starting test...")

torch.cuda.synchronize()

go()
print("Kernel launched")

torch.cuda.synchronize()


print("Starting timing loop...")

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()
for i in range(5):
    go()
torch.cuda.synchronize()
end_event.record()
torch.cuda.synchronize()
elapsed_time = start_event.elapsed_time(end_event)
t1 = time.time()  # Keep this for compatibility with the time_per_iter calculation
t0 = t1 - (elapsed_time / 1000)  # Convert ms to seconds
time_per_iter = ((t1 - t0) * 1e6) / 5
print(f"Time per iter: {time_per_iter} us")
print(f"GB/s: {(2 * 2048 * 2048 * 1e-9) / (time_per_iter * 1e-6)}")


print("Test completed successfully!")

O_numpy = O.float().cpu().numpy()
print(O_numpy.shape)
print(O_numpy)

O2 = reference_go()
O2_numpy = O2.float().cpu().numpy()

diff = O - O2
adiff = diff.abs()
rdiff = 2 * adiff / (O.abs() + O2.abs())

print("\nStats:")
print("ADIFFS:", adiff.max(), adiff.min(), adiff.mean())
print("RDIFFS:", rdiff.max(), rdiff.min(), rdiff.mean())

breakpoint()


print(diff.shape)

# print("TIMINGS")
# for i in range(128):
#     print(
#         f"event {i:3d}: {', '.join([f'{timings[0, j, i]:6d}' for j in range(DEPTH)])}"
#     )

# Create histogram of differences
import matplotlib.pyplot as plt

differences = (O - O2).flatten()
plt.figure(figsize=(10, 6))
plt.hist(differences.float().cpu(), bins=50, alpha=0.7)
plt.title("Histogram of Differences Between Matrices")
plt.xlabel("Difference Value")
plt.ylabel("Frequency")
plt.yscale("log")  # Set y-axis to logarithmic scale
plt.grid(True, alpha=0.3)
plt.savefig("matrix_diff_histogram.png")
plt.close()
print(f"Histogram saved to matrix_diff_histogram.png")


