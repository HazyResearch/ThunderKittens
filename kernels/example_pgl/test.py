import torch

import example_pgl

input_tensors = [
    (torch.randn(1024 * 1024 * 2, dtype=torch.float32, device=i) * 100).to(torch.int32)
    for i in range(4)
]
output_tensors = [
    torch.zeros(1024 * 1024 * 2, dtype=torch.int32, device=i)
    for i in range(4)
]

device_ids = [0, 1, 2, 3]
example_pgl.enable_all_p2p_access(device_ids)
club = example_pgl.KittensClub(device_ids)
kernel_globals = example_pgl.Globals(input_tensors, output_tensors, 1024 * 1024 * 2)

example_pgl.example_pgl(kernel_globals, club)

print(output_tensors[0] - input_tensors[0])
print(output_tensors[1] - input_tensors[1])
print(output_tensors[2] - input_tensors[2])
print(output_tensors[3] - input_tensors[3])
