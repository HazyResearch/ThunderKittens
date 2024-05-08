import torch
from tqdm import trange
import numpy as np
import sys
from einops import rearrange, repeat
import math

# only generate a single batch/head of data, which makes file loading much faster.
# it does mean we'll have to check batch/head behavior separately later, but that should be much easier to debug.
B = 1
H = 1
N = 2048
D = 64

NUM_WORKERS = 16
TILE_DIM = 16

def forward_causal(Q, K, V, O):
    # Q, K, V: (B, H, N, D)
    # O: (B, H, N, D)
    B, H, N, D = Q.shape
    
    # Divide Q into (64 x 64) blocks
    Q = Q.reshape(B, H, N//64, 64, D)
    
    # Divide K and V into (64 x 64) blocks
    
    
    