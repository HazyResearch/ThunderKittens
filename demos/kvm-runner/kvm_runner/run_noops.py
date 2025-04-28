import time
from pathlib import Path

import pydra
import torch
from kvm_runner.instructions import NoOp
from kvm_runner.kvm import get_kvm_func, interpret_with_kvm
from kvm_runner.llama import ExtraModelConfig, LlamaForCausalLM
from kvm_runner.scheduler import (
    make_globals,
    tensorize_instructions,
)


class ScriptConfig(pydra.Config):
    kvm_path: Path = (
        Path(__file__).parent.parent.parent.parent / "tests" / "vm" / "llama_official"
    )
    model: str = "meta-llama/Llama-3.2-1B-Instruct"
    device: str = "cuda:0"
    instruction_reps: int = 1


def main(config: ScriptConfig):
    torch.manual_seed(0)

    kvm_func = get_kvm_func(config.kvm_path)
    torch.cuda.set_device(config.device)

    extra_config = ExtraModelConfig(
        interleave_rope=True,
    )
    model = LlamaForCausalLM.from_pretrained(
        config.model, extra_config=extra_config, device=config.device
    )

    globs = make_globals(
        model,
    )

    sm_count = globs.sm_count()

    instructions = [NoOp()] * (sm_count * config.instruction_reps)
    tensorize_instructions(globs, instructions)

    start = time.time()
    interpret_with_kvm(globs, kvm_func)
    torch.cuda.synchronize()
    end = time.time()
    print(f"Time taken: {end - start} seconds")


if __name__ == "__main__":
    pydra.run(main)
