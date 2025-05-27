import time
from pathlib import Path

import pydra
import torch
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
from tabulate import tabulate
from tqdm import tqdm
from transformers import AutoTokenizer


class ScriptConfig(pydra.Config):
    model: str = "meta-llama/Llama-3.2-1B-Instruct"
    device: str = "cuda:0"
    prompt: str = "tell me a funny joke about cookies"
    chat: bool = False
    ntok: int = 100
    mode: str = "model"
    interleave_rope: bool = True
    kvm_dir: Path = (
        Path(__file__).parent.parent.parent.parent.parent
        / "tests"
        / "vm"
        / "llama_official"
    )
    token_details: bool = False
    tokens: bool = True
    num_warmup: int = 5
    num_iters: int = 10
    barrier_fill_val: int = 0
    batch_size: int = 1
    max_len_override: int | None = 16384
    noops: bool = False
    skip_kvm: bool = False
    skip_rest: bool = False
    sched: str = "rr"
    setting: str = "latency"
    memory_fraction: float | None = None

    def finalize(self):
        if self.setting == "latency" and self.mode in ["kvm", "pyvm"]:
            assert self.interleave_rope, "interleave_rope must be True for kvm mode"

    def once(self):
        self.num_warmup = 0
        self.num_iters = 1

    def th(self, bs=1024, sl=128):
        self.setting = "throughput"
        self.kvm_dir = (
            Path(__file__).parent.parent.parent.parent
            / "tests"
            / "batch-vm"
            / "llama_official"
        )
        self.batch_size = bs
        self.max_len_override = sl
        self.interleave_rope = False
        self.l8()

        if self.mode == "kvm":
            assert self.batch_size == 1024, (
                "must recompile the kernel with new BATCH_SIZE"
            )

    def l1(self):
        self.model = "meta-llama/Llama-3.2-1B-Instruct"

    def l8(self):
        self.model = "meta-llama/Llama-3.1-8B-Instruct"


@torch.inference_mode()
def main(config: ScriptConfig):
    torch.cuda.set_device(config.device)

    tokenizer = AutoTokenizer.from_pretrained(config.model)
    extra_config = ExtraModelConfig(
        interleave_rope=config.interleave_rope,
        max_len_override=config.max_len_override,
        max_batch_size=config.batch_size,
    )
    model = LlamaForCausalLM.from_pretrained(
        config.model, device=config.device, extra_config=extra_config
    )

    messages = []

    schedule_builder = make_schedule_builder(config.setting)
    schedule = schedule_builder.build(model)
    assigned_to_sms = assign_to_sms(
        config.sched, schedule=schedule, memory_fraction=config.memory_fraction
    )
    tensorize_instructions(schedule.globs, assigned_to_sms)

    interpreter = make_kvm_interpreter(config.setting, config.kvm_dir)
    gen = KVM_Generator(
        model,
        interpreter,
        schedule,
        barrier_fill_val=config.barrier_fill_val,
        skip_kvm=config.skip_kvm,
        skip_rest=config.skip_rest,
    )

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    while True:
        user_input = input(">>> ")
        messages.append({"role": "user", "content": user_input})

        tok_inp = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        input_ids = tokenizer(tok_inp, return_tensors="pt")["input_ids"].to(
            model.device
        )
        prompt_len = input_ids.shape[-1]

        print(f"Input ids shape: {input_ids.shape}")

        position_ids = torch.arange(prompt_len).to(model.device)

        prefill_inp = BatchState(
            input_ids=input_ids,
            position_ids=position_ids,
        )

        prefill_output: BatchState = model(prefill_inp)
        assert prefill_output.output_ids is not None
        new_input_token = prefill_output.output_ids[:, -1:]

        output_tokens = torch.zeros(
            1, config.ntok, device=model.device, dtype=torch.long
        )
        output_tokens[:, 0] = new_input_token

        start_event.record()
        gen.generate(output_tokens, prompt_len, config.ntok)
        end_event.record()
        torch.cuda.synchronize()
        elapsed = start_event.elapsed_time(end_event) / 1000

        tokens_per_second = config.ntok / elapsed

        to_cpu = output_tokens.cpu()
        output_text = tokenizer.batch_decode(to_cpu, skip_special_tokens=True)
        print("Response: ", output_text[0])
        print(f"Speed: {tokens_per_second:.2f} tokens/s")


if __name__ == "__main__":
    pydra.run(main)
