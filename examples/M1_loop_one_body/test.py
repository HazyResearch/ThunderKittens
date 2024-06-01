import pdb

import torch
import sys
sys.path.append("../../")
import copy
from src.common.pyutils.test_build_utils import  __eq
import tk_m1_prefill


def pt_prefill(W1, XA, XB, XC):
    Z1 = XB @ W1
    dl_dZ1 = Z1 - XA
    Attn1 = XC @ XB.transpose(-1,-2)
    Attn1 = torch.tril(Attn1)
    Z1_bar_term_1 = XC @ W1
    Z1_bar_term_2 = Attn1 @ dl_dZ1
    Z1_bar = Z1_bar_term_1 - Z1_bar_term_2

    output = Z1_bar

    return output


def simple_test(dt, use_ones=False):
    torch.manual_seed(0)
    input_dtype = torch.bfloat16
    BS, NH, HF, = 2, 2, 64
    CS = 16

    match_module = 'M1'
    HF_prime = 4 * HF if match_module == 'M2' else HF

    original_state_dict = {
        'W1': torch.randn(BS * NH, HF, HF_prime, device='cuda', dtype=input_dtype) * 0.2,
    }
    original_input_dict = {
        'XA': torch.randn(BS * NH, CS, HF, device='cuda', dtype=input_dtype) * 0.2,
        'XB': torch.randn(BS * NH, CS, HF, device='cuda', dtype=input_dtype) * 0.2,
        'XC': torch.randn(BS * NH, CS, HF, device='cuda', dtype=input_dtype) * 0.2,
    }

    cuda_state_dict = copy.deepcopy(original_state_dict)
    cuda_input_dict = copy.deepcopy(original_input_dict)
    cuda_output  = torch.zeros_like(cuda_input_dict['XA'])
    tk_m1_prefill.prefill(cuda_state_dict['W1'],
                          cuda_input_dict['XA'], cuda_input_dict['XB'], cuda_input_dict['XC'],
                          cuda_output)

    pt_state_dict = copy.deepcopy(original_state_dict)
    pt_input_dict = copy.deepcopy(original_input_dict)
    pt_output = pt_prefill(pt_state_dict['W1'], pt_input_dict['XA'], pt_input_dict['XB'], pt_input_dict['XC'])

    print('============ TK ============')
    print(cuda_output[0,0,:4])
    print('============ PT ============')
    print(pt_output[0,0,:4])
    print((cuda_output[0] != pt_output[0]).sum())

    print('============ TK ============')
    print(cuda_output[1,:,:2])
    # print(cuda_output[0,:4,:4])
    print('============ PT ============')
    print(pt_output[1,:,:2])
    # print(pt_output[0,:4,:4])


    print(f'\n========== {match_module} Matching  ============')
    print('Pytorch v.s TK')
    for k in original_state_dict.keys():
        diff = torch.abs(pt_state_dict[k] - cuda_state_dict[k])
        print(f'{k} diff: max={diff.max()}, median={diff.median()}')
    diff = torch.abs(pt_output - cuda_output)
    print(f'Output diff: max={diff.max()}, median={diff.median()}\n')


profile = False
simple_test(torch.bfloat16)
