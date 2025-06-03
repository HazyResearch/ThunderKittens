import torch

import example_pgl

input_tensors = [
    (torch.randn(32, dtype=torch.float32, device=i) * 100).to(torch.int32)
    for i in range(4)
]
output_tensors = [
    torch.zeros(32, dtype=torch.int32, device=i)
    for i in range(4)
]
club = example_pgl.KittensClub([0, 1, 2, 3])

example_pgl.test(input_tensors, output_tensors, club)
print(output_tensors[0] - input_tensors[0])
print(output_tensors[1] - input_tensors[1])
print(output_tensors[2] - input_tensors[2])
print(output_tensors[3] - input_tensors[3])
