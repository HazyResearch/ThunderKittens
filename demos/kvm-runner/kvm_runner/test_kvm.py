import time
from pathlib import Path

import pydra
import torch
from kvm_runner.kvm import get_kvm_func, interpret_with_kvm
from kvm_runner.llama import ExtraModelConfig, LlamaForCausalLM
from kvm_runner.python_vm import interpret_with_pyvm
from kvm_runner.scheduler import (
    Globals,
    make_globals,
    schedule_model,
    tensorize_instructions,
)
from torch import Tensor
from torch.nn.init import normal_
from tqdm import tqdm


class ScriptConfig(pydra.Config):
    kvm_path: Path = (
        Path(__file__).parent.parent.parent.parent / "tests" / "vm" / "llama_official"
    )
    model: str = "meta-llama/Llama-3.2-1B-Instruct"
    device: str = "cuda:0"
    prompt_len: int = 10
    ntok: int = 10
    stop_after_op: str | None = None
    start_after_op: str | None = None
    layer_limit: int | None = 1
    skip_pyvm: bool = False
    instruction_reps: int = 1
    exec_reps: int = 1
    skip_starting_instructions: bool = False
    barrier_init_val: int = 0


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

    pos_id = config.prompt_len + config.ntok

    globs_for_pyvm.pos_id = pos_id
    globs_for_kvm.pos_id = pos_id

    normal_(globs_for_pyvm.hidden_states)
    globs_for_kvm.hidden_states.copy_(globs_for_pyvm.hidden_states)
    print("hidden states sum:", globs_for_pyvm.hidden_states.float().sum())

    print("HACK LOW MEM NO KV CACHE GOODNESS")

    # NOTE: important to clone the KV caches since these originally come from the model
    # and so are the same tensor.
    normal_(globs_for_pyvm.k_cache)
    # globs_for_kvm.k_cache = globs_for_pyvm.k_cache.clone()

    normal_(globs_for_pyvm.v_cache)
    # globs_for_kvm.v_cache = globs_for_pyvm.v_cache.clone()

    _, instructions = schedule_model(
        prompt_len=config.prompt_len,
        ntok=config.ntok,
        globs=globs_for_pyvm,
        stop_after_op=config.stop_after_op,
        layer_limit=config.layer_limit,
    )

    if config.start_after_op is not None:
        _, starting_instructions = schedule_model(
            prompt_len=config.prompt_len,
            ntok=config.ntok,
            globs=globs_for_pyvm,
            stop_after_op=config.start_after_op,
            layer_limit=config.layer_limit,
        )

        assert len(starting_instructions) < len(instructions)
        for i, i2 in zip(starting_instructions, instructions):
            assert i == i2

        instructions = instructions[len(starting_instructions) :]
    else:
        starting_instructions = []

    if config.instruction_reps > 1:
        print(f"repeating instructions {config.instruction_reps} times")
        instructions = instructions * config.instruction_reps

    tensorize_instructions(
        globs_for_kvm, instructions, barrier_init_val=config.barrier_init_val
    )
    tensorize_instructions(
        globs_for_pyvm, instructions, barrier_init_val=config.barrier_init_val
    )

    for _ in tqdm(range(config.exec_reps)):
        if len(starting_instructions) > 0 and not config.skip_starting_instructions:
            print("running starting instructions...")

            # run all the starting instructions with pyvm
            start = time.time()
            interpret_with_pyvm(globs_for_pyvm, starting_instructions)
            interpret_with_pyvm(globs_for_kvm, starting_instructions)
            torch.cuda.synchronize()
            end = time.time()
            print(f"starting instructions time: {end - start}")

        def summarize_caches(globs: Globals, name: str):
            k_cache_summary = globs.k_cache[:, pos_id].float().sum(-1).sum(-1)
            print(f"{name} k_cache_summary:", k_cache_summary)
            v_cache_summary = globs.v_cache[:, pos_id].float().sum(-1).sum(-1)
            print(f"{name} v_cache_summary:", v_cache_summary)

        # summarize_caches(globs_for_pyvm, "pyvm")
        # summarize_caches(globs_for_kvm, "kvm")

        if not config.skip_pyvm:
            print("interpreting with pyvm...")
            start = time.time()
            interpret_with_pyvm(globs_for_pyvm, instructions)
            torch.cuda.synchronize()
            end = time.time()
            print(f"pyvm time: {end - start}")

        # summarize_caches(globs_for_pyvm, "pyvm")
        # summarize_caches(globs_for_kvm, "kvm")

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
            globs_for_pyvm.post_ln_rope_q,
            globs_for_kvm.post_ln_rope_q,
            "post_ln_rope_q",
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
        test_tensors(globs_for_pyvm.barriers, globs_for_kvm.barriers, "barriers")

        # test_tensors(globs_for_pyvm.k_cache, globs_for_kvm.k_cache, "k_cache")
        # test_tensors(globs_for_pyvm.v_cache, globs_for_kvm.v_cache, "v_cache")

        print("kvm hidden states sum:", globs_for_kvm.hidden_states.float().sum())
        print("pyvm hidden states sum:", globs_for_pyvm.hidden_states.float().sum())

        # print("pyvm", globs_for_pyvm.attn_out_intermediates[0].view(-1)[:128])
        # print("kvm", globs_for_kvm.attn_out_intermediates[0].view(-1)[:128])

        # summarize_caches(globs_for_pyvm, "pyvm")
        # summarize_caches(globs_for_kvm, "kvm")


if __name__ == "__main__":
    pydra.run(main)
