import sys
from pathlib import Path


def get_kvm_func(kvm_dir: Path):
    sys.path.append(str(kvm_dir.expanduser().absolute()))
    from kvm_llama import kvm_llama  # type: ignore

    return kvm_llama


class KVM_Interpreter:
    def __init__(self, kvm_dir: Path):
        self.kvm_func = get_kvm_func(kvm_dir)

    def interpret(self, globs):
        raise NotImplementedError
