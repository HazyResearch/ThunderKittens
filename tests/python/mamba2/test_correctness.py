import torch 
import torch.nn.functional as F
from einops import rearrange

from baselines.ssd_minimal import ssd_minimal_discrete 

try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    print(f"Successfully imported mamba_chunk_scan_combined")
except ImportError:
    print(f"Failed to import mamba_chunk_scan_combined; Please 'pip install mamba_ssm'.")
    mamba_chunk_scan_combined = None


try:
    import thunderkittens as tk 
    print(f"Successfully imported thunderkittens")
except ImportError:
    print(f'Failed to import thunderkittens; Please "pip setup.py install".')
    tk = None





