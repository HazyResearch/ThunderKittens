import time
from pathlib import Path

import pydra
import torch
from kvm_runner.kvm import KVM_Runner
from kvm_runner.llama import LlamaForCausalLM
from kvm_runner.model_types import BatchState, ExtraModelConfig
from kvm_runner.python_vm import PyVM_Runner
from kvm_runner.scheduler import PrintInfo
from tabulate import tabulate
from torch import Tensor
from tqdm import tqdm
from transformers import AutoTokenizer


class ScriptConfig(pydra.Config):
    model: str = "meta-llama/Llama-3.2-1B-Instruct"
    device: str = "cuda:0"
    prompt: str = "tell me a funny joke about cookies"
    chat: bool = False
    ntok: int = 100
    mode: str = "model"
    add_print_instructions: bool = False
    print_layer_filter: list[int] | None = None
    print_name_filter: list[str] | None = None
    print_state_filter: list[str] | None = None
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

    def finalize(self):
        if self.mode in ["kvm", "pyvm"]:
            assert self.interleave_rope, "interleave_rope must be True for kvm mode"

    def once(self):
        self.num_warmup = 0
        self.num_iters = 1


class Runner:
    def go():
        raise NotImplementedError


class PyTorchRunner(Runner):
    def __init__(
        self,
        config: ScriptConfig,
        model: LlamaForCausalLM,
        output_tokens: Tensor,
        prompt_len: int,
    ):
        self.config = config
        self.model = model
        self.output_tokens = output_tokens
        self.prompt_len = prompt_len
        self.start_position_ids = torch.ones(
            1, dtype=torch.long, device=model.device
        ) * (prompt_len)

    def go(self):
        for i in tqdm(range(1, self.config.ntok)):
            position_ids = self.start_position_ids + i
            decode_inp = BatchState(
                input_ids=self.output_tokens[i - 1 : i],
                position_ids=position_ids,
                seq_len=self.prompt_len + i + 1,
            )
            decode_output: BatchState = self.model(decode_inp)
            assert decode_output.output_ids is not None
            self.output_tokens[i] = decode_output.output_ids


class PyVMRunner(Runner):
    def __init__(
        self,
        config: ScriptConfig,
        model: LlamaForCausalLM,
        output_tokens: Tensor,
        prompt_len: int,
    ):
        self.config = config
        self.model = model
        self.output_tokens = output_tokens
        self.prompt_len = prompt_len

        if config.add_print_instructions:
            print_info = PrintInfo(
                layer_filter=config.print_layer_filter,
                name_filter=config.print_name_filter,
                state_filter=config.print_state_filter,
            )
        else:
            print_info = None

        runner = PyVM_Runner(
            model,
            print_info=print_info,
            prompt_len=prompt_len,
            ntok=config.ntok,
        )

        self.runner = runner

    def go(self):
        for i in tqdm(range(1, self.config.ntok)):
            input_ids = self.output_tokens[i - 1 : i]
            output_ids = self.runner.run(input_ids, pos_id=self.prompt_len + i)
            self.output_tokens[i] = output_ids


class KVMRunner(Runner):
    def __init__(
        self,
        config: ScriptConfig,
        model: LlamaForCausalLM,
        output_tokens: Tensor,
        prompt_len: int,
    ):
        self.config = config
        self.model = model
        self.output_tokens = output_tokens
        self.prompt_len = prompt_len

        assert config.kvm_dir is not None
        torch.cuda.set_device(config.device)

        runner = KVM_Runner(
            model,
            kvm_dir=config.kvm_dir,
            prompt_len=prompt_len,
            ntok=config.ntok,
            barrier_fill_val=config.barrier_fill_val,
            skip_kvm=config.skip_kvm,
            skip_rest=config.skip_rest,
        )

        self.runner = runner

    def go(self):
        for i in tqdm(range(1, self.config.ntok)):
            input_ids = self.output_tokens[i - 1 : i]
            output_ids = self.runner.run(input_ids, pos_id=self.prompt_len + i)
            self.output_tokens[i] = output_ids


class GraphedKVMRunner(KVMRunner):
    def record_graph(self):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):  # type: ignore
            # warmup
            for _ in tqdm(range(3), desc="CUDA Graph Warmup"):
                super().go()

        torch.cuda.current_stream().wait_stream(s)

        torch.cuda.synchronize()

        print("Recording CUDA Graph")
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            super().go()

        torch.cuda.synchronize()
        print("CUDA Graph recorded")

        self.graph = g

    def go(self):
        self.graph.replay()


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

    position_ids = torch.arange(prompt_len).to(model.device)

    prefill_inp = BatchState(
        input_ids=input_ids,
        position_ids=position_ids,
    )

    prefill_output: BatchState = model(prefill_inp)
    assert prefill_output.output_ids is not None
    new_input_token = prefill_output.output_ids[-1:]

    output_tokens = torch.zeros(config.ntok, device=model.device, dtype=torch.long)
    output_tokens[0] = new_input_token

    match config.mode:
        case "model":
            model = PyTorchRunner(config, model, output_tokens, prompt_len)
        case "pyvm":
            model = PyVMRunner(config, model, output_tokens, prompt_len)
        case "kvm":
            model = KVMRunner(config, model, output_tokens, prompt_len)
            if config.noops:
                model.runner.globals.instructions.zero_()
        case "gkvm":
            model = GraphedKVMRunner(config, model, output_tokens, prompt_len)
            if config.noops:
                model.runner.globals.instructions.zero_()
            model.record_graph()
        case _:
            raise ValueError(f"Invalid mode: {config.mode}")

    times = []
    cpu_times = []
    for _ in tqdm(range(config.num_warmup + config.num_iters)):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        cpu_start = time.time()
        model.go()
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
