import pdb

import torch
import sys
sys.path.append("../../")
import copy
from src.common.pyutils.test_build_utils import  __eq
import tk_batch_add


def pt_batch_add(XA, XB):
    return XA + XB


def simple_test(dt, use_ones=False):
    torch.manual_seed(0)
    input_dtype = torch.bfloat16
    BS, NH, HF, = 2, 2, 64
    CS = 16

    XA = torch.randn(BS * NH, CS, HF, device='cuda', dtype=input_dtype) * 0.02
    XA_pt = XA.clone()
    XB = torch.randn(BS * NH, CS, HF, device='cuda', dtype=input_dtype) * 0.02
    XB_pt = XB.clone()
    XC = torch.empty_like(XA)

    tk_batch_add.batch_add(XA, XB, XC)
    XC_pt = pt_batch_add(XA_pt, XB_pt)

    print(f'\n========== {match_module} Matching  ============')
    print('Pytorch v.s TK')
    diff = XC - XC_pt
    print(f'Output diff: max={diff.max()}, median={diff.median()}\n')


profile = False
simple_test(torch.bfloat16)
