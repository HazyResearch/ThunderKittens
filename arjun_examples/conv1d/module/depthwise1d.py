import torch
import sys

# sys.path.append('../../../../')
# from src.common.pyutils.test_build_utils import __eq
# sys.path.append('build/lib.linux-x86_64-cpython-311')

sys.path.append("../kernel")
import conv1d_tk as mod
  
class TKDepthWiseConv1d(torch.nn.Module):
    def __init__(self, channels, kernel_size, padding, weights, bias, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TKDepthWiseConv1d, self).__init__()
        self.padding = padding
        # Convert weights to format expected by kernel
        self.weights = weights.squeeze(1)
        self.bias = bias
    
    def forward(self, input):
        # Receives (B, D, L)
        # Converts to (D, L, B)
        transposed = input.transpose(0, 2).transpose(0, 1).contiguous()
        mod.conv1d_tk(transposed, self.weights, self.bias, self.padding)