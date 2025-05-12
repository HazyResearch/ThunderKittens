import time
from pathlib import Path

import pydra
import torch
from tabulate import tabulate
from tqdm import tqdm
from transformers import AutoTokenizer

from kvm_unity.dispatch import (
    make_kvm_interpreter,
    make_pyvm_interpreter,
    make_schedule_builder,
)
from kvm_unity.generators import (
    KVM_Generator,
    PyTorchGenerator,
    PyVM_Generator,
)
from kvm_unity.llama import LlamaForCausalLM
from kvm_unity.model_types import BatchState, ExtraModelConfig
from kvm_unity.scheduler import (
    assign_to_sms,
    tensorize_instructions,
)


class ScriptConfig(pydra.Config):
    model: str = "meta-llama/Llama-3.2-1B-Instruct"
    device: str = "cuda:0"
    prompt: str = "tell me a funny joke about cookies"
    chat: bool = False
    ntok: int = 100
    mode: str = "model"
    interleave_rope: bool = True
    kvm_dir: Path = (
        Path(__file__).parent.parent.parent.parent / "tests" / "vm" / "llama_official"
    )
    token_details: bool = False
    tokens: bool = True
    num_warmup: int = 5
    num_iters: int = 10
    barrier_fill_val: int = 0
    max_len_override: int | None = 16384
    noops: bool = False
    skip_kvm: bool = False
    skip_rest: bool = False
    sched: str = "rr"
    setting: str = "latency"

    def finalize(self):
        if self.mode in ["kvm", "pyvm"]:
            assert self.interleave_rope, "interleave_rope must be True for kvm mode"

    def once(self):
        self.num_warmup = 0
        self.num_iters = 1

    def th(self):
        self.setting = "throughput"
        self.kvm_dir = (
            Path(__file__).parent.parent.parent.parent
            / "tests"
            / "batch_vm"
            / "llama_official"
        )


@torch.inference_mode()
def main(config: ScriptConfig):
    torch.cuda.set_device(config.device)

    tokenizer = AutoTokenizer.from_pretrained(config.model)
    extra_config = ExtraModelConfig(
        interleave_rope=config.interleave_rope,
        max_len_override=config.max_len_override,
    )
    model = LlamaForCausalLM.from_pretrained(
        config.model, device=config.device, extra_config=extra_config
    )

    if config.chat:
        tok_inp = tokenizer.apply_chat_template(
            config.prompt, tokenize=False, add_generation_prompt=True
        )
    else:
        tok_inp = config.prompt

    input_ids = tokenizer(tok_inp, return_tensors="pt")["input_ids"][0].to(model.device)
    prompt_len = input_ids.shape[0]

    print(f"Prompt length: {prompt_len}")

    position_ids = torch.arange(prompt_len).to(model.device)

    prefill_inp = BatchState(
        input_ids=input_ids,
        position_ids=position_ids,
    )

    prefill_output: BatchState = model(prefill_inp)
    assert prefill_output.output_ids is not None
    new_input_token = prefill_output.output_ids[:, -1:]

    output_tokens = torch.zeros(config.ntok, device=model.device, dtype=torch.long)
    output_tokens[0] = new_input_token

    schedule_builder = make_schedule_builder(config.setting)
    schedule = schedule_builder.build(model)

    match config.mode:
        case "torch":
            gen = PyTorchGenerator(model)
        case "pyvm":
            interpreter = make_pyvm_interpreter(config.setting)
            gen = PyVM_Generator(model, interpreter, schedule)
        case "kvm":
            interpreter = make_kvm_interpreter(config.setting, config.kvm_dir)
            assigned_to_sms = assign_to_sms(config.sched, schedule=schedule)
            tensorize_instructions(schedule.globs, assigned_to_sms)
            gen = KVM_Generator(
                model,
                interpreter,
                schedule,
                barrier_fill_val=config.barrier_fill_val,
                skip_kvm=config.skip_kvm,
                skip_rest=config.skip_rest,
            )
            if config.noops:
                gen.replace_with_noops()
        case _:
            raise ValueError(f"Invalid mode: {config.mode}")

    times = []
    cpu_times = []
    for _ in tqdm(range(config.num_warmup + config.num_iters)):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        cpu_start = time.time()
        gen.generate(output_tokens, prompt_len, config.ntok)
        cpu_end = time.time()
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event) / 1000)
        cpu_times.append(cpu_end - cpu_start)

    non_warmup_times = times[config.num_warmup :]
    non_warmup_cpu_times = cpu_times[config.num_warmup :]
    elapsed = sum(non_warmup_times) / len(non_warmup_times)
    elapsed_cpu = sum(non_warmup_cpu_times) / len(non_warmup_cpu_times)
    print(f"Average time: {(elapsed * 1000):.2f}ms (CPU: {(elapsed_cpu * 1000):.2f}ms)")

    if config.tokens:
        to_cpu = output_tokens.cpu()
        print("Output ids: ", to_cpu)
        print("Output text: ", tokenizer.decode(to_cpu))

    if config.token_details:
        ids_list = to_cpu.tolist()
        tokens = tokenizer.convert_ids_to_tokens(ids_list)

        table = []
        for i, token in enumerate(tokens):
            pos_id = i + prompt_len
            table.append([i, pos_id, token])

        print("More detailed output:")
        print(tabulate(table, headers=["output id", "position id", "token"]))

    tokens_per_second = (config.ntok - 1) / elapsed
    print(f"Tokens per second: {tokens_per_second:.2f}")


if __name__ == "__main__":
    pydra.run(main)
