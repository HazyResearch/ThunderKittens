import torch
from tqdm import trange
import numpy as np
import scipy
import sys

KERNEL_SIZE = 17 #33, 49, etc
INPUT_LEN = 1024
# KERNEL_SIZE + INPUT_LEN -1 % 16 == 0 and INPUT_LEN % 16 == 0 so
# KERNEL_SIZE % 16 == 1
NUM_CHANNELS = 16
# BATCH_SIZE has to be multiple of 16
BATCH_SIZE = 32
# TILE_DIM = 16


conv = torch.nn.Conv1d(
    in_channels = NUM_CHANNELS,
    out_channels = NUM_CHANNELS,
    kernel_size=KERNEL_SIZE,
    # Depthwise conv
    groups=NUM_CHANNELS,
    padding=KERNEL_SIZE - 1,
    bias=False
)

TESTNAME = sys.argv[1]

if TESTNAME in ['ones_all']:
    kernel = (torch.ones((NUM_CHANNELS, KERNEL_SIZE), dtype=torch.bfloat16, device='cpu')).to(torch.float32) 
    inp = (torch.ones((NUM_CHANNELS, INPUT_LEN, BATCH_SIZE), dtype=torch.bfloat16, device='cpu')).to(torch.float32) 
    
elif TESTNAME in ['randn_all']:
    torch.random.manual_seed(42)
    kernel = (torch.randn((NUM_CHANNELS, KERNEL_SIZE), dtype=torch.bfloat16, device='cpu')).to(torch.float32)
    inp = (torch.randn((NUM_CHANNELS, INPUT_LEN, BATCH_SIZE), dtype=torch.bfloat16, device='cpu')).to(torch.float32)
else:
    print('Invalid test name')
    sys.exit(0)

def pytorch_test(kernel, inp, TESTNAME='all'):
    
    # Toeplitz matrix (per channel) for the convolution
    # "Full" mode analogous to torch conv1d w/ input padding on both sides
    conv_tensor = np.empty((NUM_CHANNELS, INPUT_LEN + KERNEL_SIZE - 1, INPUT_LEN))
    # Stack toeplitz matrices along channel dim
    for i in range(NUM_CHANNELS):
        toeplitz = scipy.linalg.convolution_matrix(kernel[i, :], n=INPUT_LEN, mode='full')
        conv_tensor[i, :, :] = toeplitz[:, :]

    H = torch.from_numpy(conv_tensor)

    print(H.shape)
    print(inp.shape)

    # Torch uses cross-correlation (not actual convolution) so we need to reverse the filter
    kern_tensor = torch.flip(kernel, (1,)).unsqueeze(1)
    conv.weight = torch.nn.Parameter(kern_tensor, requires_grad=False)
    out = conv(inp.transpose(0, 2).transpose(1, 2)).transpose(0, 2).transpose(0, 1)

    # Pad output so it's same as kernel output

    # H = H.to(dtype=torch.bfloat16)
    # x = inp.to(dtype=torch.bfloat16)
    # out = out.to(dtype=torch.bfloat16)
    print(out.shape)

    return H, inp, out


H, x, o = pytorch_test(kernel, inp, TESTNAME=TESTNAME)

with open(f'{TESTNAME}_dlb.txt', 'w') as f:
    Hf = H.to(torch.float32).flatten().cpu().numpy()
    xf = x.to(torch.float32).flatten().cpu().numpy()
    of = o.to(torch.float32).flatten().cpu().numpy()
    
    for i in trange(Hf.shape[0]):
        f.write(repr(Hf[i]))
        f.write(' ')
    for i in trange(xf.shape[0]):
        f.write(repr(xf[i]))
        f.write(' ')
    for i in trange(of.shape[0]):
        f.write(repr(of[i]))
        f.write(' ')

    # TODO output kernel to file so we can check Toeplitz matrix formation from C++