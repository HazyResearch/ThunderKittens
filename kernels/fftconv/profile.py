import torch
#from flashfftconv import FlashFFTConv

import sys
sys.path.append("./module")
from fftconv import TKFFTConv
torch.set_grad_enabled(False)


B = 64
H = 768

torch.random.manual_seed(42)
dtype = torch.bfloat16
device = 'cuda'

#sudo /usr/local/cuda-12/bin/ncu -f -o /home/bfs/arjun/ThunderKittens/arjun_examples/fftconv/profile.ncu-rep --set full ~/miniconda3/envs/sim/bin/python profile.py

N = 1024
u = torch.randn((B, H, N), dtype=dtype).to(device)
# Needs to be fp32 so we can take its FFT
k = torch.randn((H, N), dtype=torch.float32).to(device)
# Only profile our kernel
conv_tk = TKFFTConv(N, dtype=dtype).to(device)
#conv_flashfft = FlashFFTConv(N, dtype=dtype).to(device)

#y_flash = conv_flashfft(u, k)
y_tk = conv_tk(u, k)

