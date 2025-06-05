import torch

import example_pgl

test_tensors = [
    torch.randn(1024 * 1024, dtype=torch.float32, device=i)
    for i in range(4)
]
input_tensors = [
    torch.randn(1024 * 1024, dtype=torch.float32, device=i)
    for i in range(4)
]
output_tensors = [
    torch.zeros(1024 * 1024, dtype=torch.float32, device=i)
    for i in range(4)
]

device_ids = [0, 1, 2, 3]
example_pgl.enable_all_p2p_access(device_ids)
club = example_pgl.KittensClub(device_ids)
kernel_globals = example_pgl.make_globals(test_tensors, input_tensors, output_tensors, 1024 * 1024 * 2)

example_pgl.example_pgl(club, *kernel_globals)
for i in range(4): 
    torch.cuda.synchronize(i)

print(output_tensors[0] - input_tensors[0])
print(output_tensors[1] - input_tensors[1])
print(output_tensors[2] - input_tensors[2])
print(output_tensors[3] - input_tensors[3])
