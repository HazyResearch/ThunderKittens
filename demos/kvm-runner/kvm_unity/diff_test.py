import pickle
import time
from pathlib import Path

import pydra
import torch
from torch.nn.init import normal_
from tqdm import tqdm

from kvm_unity.dispatch import (
    make_kvm_interpreter,
    make_pyvm_interpreter,
    make_schedule_builder,
)
from kvm_unity.llama import ExtraModelConfig, LlamaForCausalLM
from kvm_unity.scheduler import (
    assign_to_sms,
    tensorize_instructions,
)


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
    sched: str = "rr"
    setting: str = "latency"
    batch_size: int = 1
    skip_cost: bool = False
    interleave_rope: bool = True

    def full(self):
        self.layer_limit = None

    def th(self, bs=1024, sl=128):
        self.setting = "throughput"
        self.kvm_path = (
            Path(__file__).parent.parent.parent.parent
            / "tests"
            / "batch-vm"
            / "llama_official"
        )
        self.batch_size = bs
        self.skip_cost = True
        self.max_len_override = sl
        self.interleave_rope = False
        self.l8()

    def l1(self):
        self.model = "meta-llama/Llama-3.2-1B-Instruct"

    def l8(self):
        self.model = "meta-llama/Llama-3.1-8B-Instruct"


def main(config: ScriptConfig):
    torch.manual_seed(0)
    torch.cuda.set_device(config.device)

    extra_config = ExtraModelConfig(
        interleave_rope=config.interleave_rope,
        max_len_override=config.max_len_override,
        max_batch_size=config.batch_size,
    )

    model = LlamaForCausalLM.from_pretrained(
        config.model, extra_config=extra_config, device=config.device
    )

    builder = make_schedule_builder(config.setting)
    kvm_interpreter = make_kvm_interpreter(config.setting, config.kvm_path)
    pyvm_interpreter = make_pyvm_interpreter(config.setting)

    spy = builder.build(
        model=model,
        layer_limit=config.layer_limit,
        stop_after_op=config.stop_after_op,
    )

    skvm = builder.with_new_globals(spy, model)

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
        start_schedule = builder.build(
            model=model,
            stop_after_op=config.start_after_op,
            layer_limit=config.layer_limit,
        )

        starting_instructions = start_schedule.get_linear_instructions()

        assert len(starting_instructions) < len(instructions), (
            f"num starting instructions {len(starting_instructions)} should be less than num total instructions {len(instructions)}"
        )
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

    queue_lengths = [len(q) for q in assigned_to_sms]
    print(
        f"sm queue lengths: min={min(queue_lengths)}, max={max(queue_lengths)}, mean={sum(queue_lengths) / len(queue_lengths)}"
    )

    if not config.skip_cost:
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
    else:
        cost_per_sm = None

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
            pyvm_interpreter.interpret(gpy, starting_instructions)
            pyvm_interpreter.interpret(gkvm, starting_instructions)
            torch.cuda.synchronize()
            end = time.time()
            print(f"starting instructions time: {end - start}")

        if not config.skip_pyvm:
            print("interpreting with pyvm...")
            start = time.time()
            pyvm_interpreter.interpret(gpy, instructions)
            torch.cuda.synchronize()
            end = time.time()
            print(f"pyvm time: {end - start}")

        print("interpreting with kvm...")
        start = time.time()
        kvm_interpreter.interpret(gkvm)
        torch.cuda.synchronize()
        end = time.time()
        print(f"kvm time: {end - start}")

        print("done! diffing tensors:")

        gpy.diff(gkvm)

    if config.bp:
        breakpoint()

    if config.outfile is not None:
        outdata = {
            "timings": gkvm.timings.cpu(),
            "instructions": gkvm.instructions.cpu(),
            "python_instructions": assigned_to_sms,
            "cost_per_sm": cost_per_sm,
        }

        print(f"Saving to {config.outfile}")

        with open(config.outfile, "wb") as f:
            pickle.dump(outdata, f)


if __name__ == "__main__":
    pydra.run(main)
