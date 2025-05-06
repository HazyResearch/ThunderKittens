import pickle
import time
from pathlib import Path

import pydra
import torch
from kvm_runner.kvm import get_kvm_func, interpret_with_kvm
from kvm_runner.llama import ExtraModelConfig, LlamaForCausalLM
from kvm_runner.python_vm import interpret_with_pyvm
from kvm_runner.scheduler import (
    Globals,
    assign_to_sms,
    schedule,
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
    truncate_instructions: int | None = None
    bp: bool = False
    outfile: Path | None = None
    noops: bool = False
    max_len_override: int | None = 16384
    sched: str = "smart"

    def full(self):
        self.layer_limit = None


def main(config: ScriptConfig):
    torch.manual_seed(0)

    kvm_func = get_kvm_func(config.kvm_path)
    torch.cuda.set_device(config.device)

    extra_config = ExtraModelConfig(
        interleave_rope=True,
        max_len_override=config.max_len_override,
    )

    model = LlamaForCausalLM.from_pretrained(
        config.model, extra_config=extra_config, device=config.device
    )

    spy = schedule(
        prompt_len=config.prompt_len,
        ntok=config.ntok,
        model=model,
        layer_limit=config.layer_limit,
        stop_after_op=config.stop_after_op,
    )

    skvm = spy.with_new_globals(model)

    gpy = spy.globs
    gkvm = skvm.globs

    pos_id = config.prompt_len + config.ntok

    gpy.pos_id = pos_id
    gkvm.pos_id = pos_id

    normal_(gpy.hidden_states)
    gkvm.hidden_states.copy_(gpy.hidden_states)
    print("hidden states sum:", gpy.hidden_states.float().sum())

    print("HACK LOW MEM NO KV CACHE GOODNESS")

    # NOTE: important to clone the KV caches since these originally come from the model
    # and so are the same tensor.
    normal_(gpy.k_cache)
    # skvm.globs.k_cache = spy.globs.k_cache.clone()

    normal_(gpy.v_cache)
    # skvm.globs.v_cache = spy.globs.v_cache.clone()

    instructions = spy.get_linear_instructions()

    if config.start_after_op is not None:
        start_schedule = schedule(
            prompt_len=config.prompt_len,
            ntok=config.ntok,
            model=model,
            stop_after_op=config.start_after_op,
            layer_limit=config.layer_limit,
        )

        starting_instructions = start_schedule.get_linear_instructions()

        assert len(starting_instructions) < len(instructions)
        for i, i2 in zip(starting_instructions, instructions):
            assert i == i2

        instructions = instructions[len(starting_instructions) :]

        if config.instruction_reps > 1:
            print(f"repeating instructions {config.instruction_reps} times")
            instructions = instructions * config.instruction_reps

        if config.truncate_instructions is not None:
            print(f"truncating instructions to {config.truncate_instructions}")
            instructions = instructions[: config.truncate_instructions]

        assigned_to_sms = assign_to_sms(
            config.sched,
            instructions=instructions,
            sm_count=spy.globs.sm_count(),
        )

    else:
        starting_instructions = []

        start = time.time()
        print(f"assigning to sms with mode {config.sched}...")
        assigned_to_sms = assign_to_sms(
            mode=config.sched,
            schedule=skvm,
        )
        end = time.time()
        print(f"assign time: {end - start}")

    cost_per_sm = []
    for sm_queue in assigned_to_sms:
        cost = 0
        for instruction in sm_queue:
            cost += instruction.cost(gpy)
        cost_per_sm.append(cost)

    cost_tensor = torch.tensor(cost_per_sm)
    relative_cost_tensor = cost_tensor / cost_tensor.max()

    print(
        f"cost per sm: min={relative_cost_tensor.min():.2f}, mean={relative_cost_tensor.mean():.2f}"
    )

    tensorize_instructions(
        gpy, assigned_to_sms, barrier_init_val=config.barrier_init_val
    )
    tensorize_instructions(
        gkvm, assigned_to_sms, barrier_init_val=config.barrier_init_val
    )

    if config.noops:
        gpy.instructions.zero_()
        gkvm.instructions.zero_()

    for _ in tqdm(range(config.exec_reps)):
        if len(starting_instructions) > 0 and not config.skip_starting_instructions:
            print("running starting instructions...")

            # run all the starting instructions with pyvm
            start = time.time()
            interpret_with_pyvm(gpy, starting_instructions)
            interpret_with_pyvm(gkvm, starting_instructions)
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
            interpret_with_pyvm(gpy, instructions)
            torch.cuda.synchronize()
            end = time.time()
            print(f"pyvm time: {end - start}")

        # summarize_caches(globs_for_pyvm, "pyvm")
        # summarize_caches(globs_for_kvm, "kvm")

        print("interpreting with kvm...")
        start = time.time()
        interpret_with_kvm(gkvm, kvm_func)
        torch.cuda.synchronize()
        end = time.time()
        print(f"kvm time: {end - start}")

        print("done! diffing tensors:")

        def test_tensors(a: Tensor, b: Tensor, name: str):
            a = a.float()
            b = b.float()

            diff = a - b
            adiff = diff.abs()
            rdiff = 2 * adiff / (a.abs() + b.abs() + 1e-6)
            print(f"{name}: max adiff: {adiff.max()}, mean rdiff: {rdiff.mean()}")
            return diff, adiff, rdiff

        d, a, r = test_tensors(gpy.hidden_states, gkvm.hidden_states, "hidden_states")
        test_tensors(
            gpy.post_ln_rope_q,
            gkvm.post_ln_rope_q,
            "post_ln_rope_q",
        )
        test_tensors(
            gpy.attn_lse_intermediates,
            gkvm.attn_lse_intermediates,
            "attn_lse_intermediates",
        )
        test_tensors(
            gpy.attn_out_intermediates,
            gkvm.attn_out_intermediates,
            "attn_out_intermediates",
        )
        test_tensors(gpy.attn_out, gkvm.attn_out, "attn_out")
        test_tensors(gpy.silu_out, gkvm.silu_out, "silu_out")
        test_tensors(gpy.barriers, gkvm.barriers, "barriers")

        test_tensors(gpy.logits, gkvm.logits, "logits")

        # test_tensors(globs_for_pyvm.k_cache, globs_for_kvm.k_cache, "k_cache")
        # test_tensors(globs_for_pyvm.v_cache, globs_for_kvm.v_cache, "v_cache")

        print("kvm hidden states sum:", gkvm.hidden_states.float().sum())
        print("pyvm hidden states sum:", gpy.hidden_states.float().sum())

        # print("pyvm", globs_for_pyvm.attn_out_intermediates[0].view(-1)[:128])
        # print("kvm", globs_for_kvm.attn_out_intermediates[0].view(-1)[:128])

        # summarize_caches(globs_for_pyvm, "pyvm")
        # summarize_caches(globs_for_kvm, "kvm")

    if config.bp:
        breakpoint()

    if config.outfile is not None:
        outdata = {
            "timings": gkvm.timings.cpu(),
            "instructions": assigned_to_sms,
            "tensor_instructions": gkvm.instructions.cpu(),
        }

        with open(config.outfile, "wb") as f:
            pickle.dump(outdata, f)


if __name__ == "__main__":
    pydra.run(main)
