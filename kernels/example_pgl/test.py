import torch

import example_pgl

input_tensor = (torch.randn(32, dtype=torch.float32, device=0) * 100).to(torch.int32)
output_tensor = torch.zeros(32, dtype=torch.int32, device=0)
club = example_pgl.KittensClub([0])

example_pgl.test(input_tensor, output_tensor, club)
print(output_tensor - input_tensor)
