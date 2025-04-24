import time
from pathlib import Path

import pydra
import torch
from kvm_runner.kvm import get_kvm_func, interpret_with_kvm
from kvm_runner.llama import ExtraModelConfig, LlamaForCausalLM
from kvm_runner.python_vm import interpret_with_pyvm
from kvm_runner.scheduler import (
    make_globals,
    schedule_layer,
    schedule_model,
    tensorize_instructions,
)
from torch import Tensor


class ScriptConfig(pydra.Config):
    kvm_path: Path = (
        Path(__file__).parent.parent.parent.parent / "tests" / "vm" / "llama_official"
    )
    model: str = "meta-llama/Llama-3.2-1B-Instruct"
    device: str = "cuda:0"
    prompt_len: int = 10
    ntok: int = 10
    full_model: bool = False
    stop_after_op: str | None = None
    start_after_op: str | None = None


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

    globs_for_pyvm = make_globals(
        model=model,
    )
    globs_for_kvm = make_globals(
        model=model,
    )

    globs_for_pyvm.pos_id = config.prompt_len + config.ntok
    globs_for_kvm.pos_id = config.prompt_len + config.ntok

    globs_for_pyvm.hidden_states = torch.randn_like(globs_for_pyvm.hidden_states)
    globs_for_kvm.hidden_states = globs_for_pyvm.hidden_states.clone()

    globs_for_pyvm.k_cache = torch.randn_like(globs_for_pyvm.k_cache)
    globs_for_kvm.k_cache = globs_for_pyvm.k_cache.clone()

    globs_for_pyvm.v_cache = torch.randn_like(globs_for_pyvm.v_cache)
    globs_for_kvm.v_cache = globs_for_pyvm.v_cache.clone()

    if config.full_model:
        instructions = schedule_model(
            globals=globs_for_pyvm,
            prompt_len=config.prompt_len,
            ntok=config.ntok,
        )
        starting_instructions = []
    else:
        instructions = schedule_layer(
            globals=globs_for_pyvm,
            layer_idx=0,
            prompt_len=config.prompt_len,
            ntok=config.ntok,
            stop_after_op=config.stop_after_op,
        )

        if config.start_after_op is not None:
            starting_instructions = schedule_layer(
                globals=globs_for_pyvm,
                layer_idx=0,
                prompt_len=config.prompt_len,
                ntok=config.ntok,
                stop_after_op=config.start_after_op,
            )

            assert len(starting_instructions) < len(instructions)
            for i, i2 in zip(starting_instructions, instructions):
                assert i == i2

            instructions = instructions[len(starting_instructions) :]
        else:
            starting_instructions = []

    tensorize_instructions(globs_for_kvm, instructions)
    tensorize_instructions(globs_for_pyvm, instructions)

    if len(starting_instructions) > 0:
        print("running starting instructions...")

        # run all the starting instructions with pyvm
        start = time.time()
        interpret_with_pyvm(globs_for_pyvm, starting_instructions)
        interpret_with_pyvm(globs_for_kvm, starting_instructions)
        torch.cuda.synchronize()
        end = time.time()
        print(f"starting instructions time: {end - start}")

    print("interpreting with pyvm...")
    start = time.time()
    interpret_with_pyvm(globs_for_pyvm, instructions)
    torch.cuda.synchronize()
    end = time.time()
    print(f"pyvm time: {end - start}")

    print("interpreting with kvm...")
    start = time.time()
    interpret_with_kvm(globs_for_kvm, kvm_func)
    torch.cuda.synchronize()
    end = time.time()
    print(f"kvm time: {end - start}")

    print("done! diffing tensors:")

    def test_tensors(a: Tensor, b: Tensor, name: str):
        diff = a - b
        adiff = diff.abs()
        rdiff = 2 * adiff / (a.abs() + b.abs() + 1e-6)
        print(f"{name}: max adiff: {adiff.max()}, mean rdiff: {rdiff.mean()}")

    test_tensors(
        globs_for_pyvm.hidden_states, globs_for_kvm.hidden_states, "hidden_states"
    )
    test_tensors(
        globs_for_pyvm.post_ln_rope_q, globs_for_kvm.post_ln_rope_q, "post_ln_rope_q"
    )
    test_tensors(
        globs_for_pyvm.attn_lse_intermediates,
        globs_for_kvm.attn_lse_intermediates,
        "attn_lse_intermediates",
    )
    test_tensors(
        globs_for_pyvm.attn_out_intermediates,
        globs_for_kvm.attn_out_intermediates,
        "attn_out_intermediates",
    )
    test_tensors(globs_for_pyvm.attn_out, globs_for_kvm.attn_out, "attn_out")
    test_tensors(globs_for_pyvm.silu_out, globs_for_kvm.silu_out, "silu_out")
    test_tensors(globs_for_pyvm.k_cache, globs_for_kvm.k_cache, "k_cache")
    test_tensors(globs_for_pyvm.v_cache, globs_for_kvm.v_cache, "v_cache")
    test_tensors(globs_for_pyvm.barriers, globs_for_kvm.barriers, "barriers")

    print("attn out", globs_for_pyvm.attn_out[:128])
    print("attn out kvm", globs_for_kvm.attn_out[:128])
    breakpoint()


if __name__ == "__main__":
    pydra.run(main)
