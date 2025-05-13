import sys
from pathlib import Path

import torch
from torch import Tensor
from tqdm import tqdm

from kvm_unity.llama import LlamaForCausalLM
from kvm_unity.model_types import BatchState
from kvm_unity.scheduler import Schedule


def get_kvm_func(kvm_dir: Path):
    sys.path.append(str(kvm_dir.expanduser().absolute()))
    from kvm_llama import kvm_llama  # type: ignore

    return kvm_llama


class KVM_Interpreter:
    def __init__(self, kvm_dir: Path):
        self.kvm_func = get_kvm_func(kvm_dir)

    def interpret(self, globs):
        raise NotImplementedError
