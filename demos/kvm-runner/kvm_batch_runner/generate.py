from pathlib import Path

import pydra
import torch
from kvm_batch_runner.kvm import KVM_Runner
from kvm_batch_runner.llama import LlamaForCausalLM
from kvm_batch_runner.model_types import BatchState, ExtraModelConfig
from kvm_batch_runner.python_vm import PyVM_Runner
from kvm_batch_runner.scheduler import PrintInfo
from tabulate import tabulate
from torch import Tensor
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import Optional, Dict

class ScriptConfig(pydra.Config):
    model: str = "meta-llama/Llama-3.1-70B-Instruct"
    model_cache_dir: Path = ( "/data/bfs/" )
    device: str = "cuda:0"
    prompt: str = "tell me a funny joke about cookies"
    chat: bool = False
    ntok: int = 100
    # batch_size: int = 512
    batch_size: int = 128
    mode: str = "model"
    add_print_instructions: bool = False
    print_layer_filter: list[int] | None = None
    print_name_filter: list[str] | None = None
    print_state_filter: list[str] | None = None
    interleave_rope: bool = True
    kvm_dir: Path = (
        Path(__file__).parent.parent.parent.parent / "tests" / "batch-vm" / "llama_official"
    )
    token_details: bool = False
    tokens: bool = True
    num_warmup: int = 0
    num_iters: int = 1
    barrier_fill_val: int = 0
    max_len_override: int | None = 32
    noops: bool = False
    skip_kvm: bool = False
    skip_rest: bool = False

    def finalize(self):
        if self.mode in ["kvm", "pyvm"]:
            assert self.interleave_rope, "interleave_rope must be True for kvm mode"


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
        debug_outputs: Optional[Dict[str, Tensor]] = None,
    ):
        self.config = config
        self.model = model
        self.output_tokens = output_tokens
        self.prompt_len = prompt_len
        self.start_position_ids = torch.ones(
            self.model.extra_config.max_batch_size, 1, dtype=torch.long, device=model.device
        ) * (prompt_len)
        self.debug_outputs = debug_outputs
    
    def go(self):
        for i in tqdm(range(1, self.config.ntok)):
            position_ids = self.start_position_ids + i
            decode_inp = BatchState(
                input_ids=self.output_tokens[:, i - 1 : i],
                position_ids=position_ids,
                seq_len=self.prompt_len + i + 1,
            )
            decode_output: BatchState = self.model(decode_inp, debug_outputs=self.debug_outputs)
            assert decode_output.output_ids is not None
            self.output_tokens[:, i] = decode_output.output_ids.squeeze(1)


class PyVMRunner(Runner):
    def __init__(
        self,
        config: ScriptConfig,
        model: LlamaForCausalLM,
        output_tokens: Tensor,
        prompt_len: int,
        debug_outputs: Optional[Dict[str, Tensor]] = None,
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
        self.debug_outputs = debug_outputs
    
    def go(self):
        for i in tqdm(range(1, self.config.ntok)):
            input_ids = self.output_tokens[:, i - 1 : i]
            output_ids = self.runner.run(
                input_ids,
                pos_id=self.prompt_len + i,
                debug_outputs=self.debug_outputs,
            )
            self.output_tokens[:, i] = output_ids


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
            input_ids = self.output_tokens[:, i - 1 : i]
            output_ids = self.runner.run(input_ids, pos_id=self.prompt_len + i)
            self.output_tokens[:, i] = output_ids.squeeze(1)



def main(config: ScriptConfig):

    torch.cuda.set_device(config.device)

    BATCH_SIZE = config.batch_size

    tokenizer = AutoTokenizer.from_pretrained(config.model)
    extra_config = ExtraModelConfig(
        interleave_rope=config.interleave_rope,
        max_len_override=config.max_len_override,
        max_batch_size=BATCH_SIZE,
    )

    model = LlamaForCausalLM.from_pretrained(
        config.model, device=config.device, extra_config=extra_config, 
        cache_dir=config.model_cache_dir, dtype=torch.bfloat16
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

    # Expand to batch size artificially
    input_ids = input_ids.unsqueeze(0).expand(BATCH_SIZE, -1)
    position_ids = position_ids.unsqueeze(0).expand(BATCH_SIZE, -1)
    prefill_inp = BatchState(input_ids=input_ids, position_ids=position_ids,)

    prefill_output: BatchState = model(prefill_inp)
    assert prefill_output.output_ids is not None
    new_input_token = prefill_output.output_ids[:, -1:]

    output_tokens = torch.zeros(BATCH_SIZE, config.ntok, device=model.device, dtype=torch.long)
    output_tokens[:, :1]= new_input_token

    # match config.mode:
    #     case "model":
    #         model = PyTorchRunner(config, model, output_tokens, prompt_len)
    #     case "pyvm":
    #         model = PyVMRunner(config, model, output_tokens, prompt_len)
    #     case "kvm":
    #         model = KVMRunner(config, model, output_tokens, prompt_len)
    #         if config.noops:
    #             model.runner.globals.instructions.zero_()
    #     case _:
    #         raise ValueError(f"Invalid mode: {config.mode}")


    times = []
    pytorch_debug_outputs = {}
    pytorch_model = PyTorchRunner(config, model, output_tokens, prompt_len, pytorch_debug_outputs)
    print("Running PyTorch model")
    for _ in tqdm(range(config.num_warmup + config.num_iters)):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        pytorch_model.go()
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event) / 1000)
    for key, tensor in pytorch_model.debug_outputs.items():
        #Only print if key name starts with L0
        if key.startswith("L0"):
            print(f"{key}: {tensor.shape}")
        elif key == "LlamaModel_Out":
            print(f"{key}: {tensor.shape}")
        elif key == "pre_lm_head_rms":
            print(f"{key}: {tensor.shape}")
        elif key == "lm_head_logits":
            print(f"{key}: {tensor.shape}")
    
    print("Running PyVM model")
    pyvm_debug_outputs = {}
    pyvm_model = PyVMRunner(config, model, output_tokens, prompt_len, pyvm_debug_outputs)
    for _ in tqdm(range(config.num_warmup + config.num_iters)):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        pyvm_model.go()
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event) / 1000)

    # Diff check same size tensors
    # layer_to_check = 0
    # name_to_check = f"L{layer_to_check}_pre_attn_ln"
    # name_to_check = f"L{layer_to_check}_Q_rope"
    # name_to_check = f"L{layer_to_check}_o_proj_residual"
    # name_to_check = f"L{layer_to_check}_pre_mlp_layer_norm"
    # name_to_check = f"L{layer_to_check}_gate_silu"
    # name_to_check = f"L{layer_to_check}_up_matmul"
    # name_to_check = f"L{layer_to_check}_down_proj_residual"
    # name_to_check = "pre_lm_head_rms"
    # print(f"Diff check {name_to_check}")
    # diff = pytorch_debug_outputs[name_to_check] - pyvm_debug_outputs[name_to_check]
    # print("PyVM shape: ", pyvm_debug_outputs[name_to_check].shape)
    # print("Pytorch shape: ", pytorch_debug_outputs[name_to_check].shape)
    # print(f"Layer {layer_to_check} diff shape: {diff.shape}")
    # print(f"Layer {layer_to_check} max absolute difference: {torch.max(torch.abs(diff))}")
    # print(f"Layer {layer_to_check} mean absolute difference: {torch.mean(torch.abs(diff))}")

    # Diff check across all layers
    for layer_idx in range(80):
        name_to_check = f"L{layer_idx}_down_proj_residual"
        diff = pytorch_debug_outputs[name_to_check] - pyvm_debug_outputs[name_to_check]
        print(f"Layer {layer_idx} diff shape: {diff.shape}")
        print(f"Layer {layer_idx} max absolute difference: {torch.max(torch.abs(diff))}")
        print(f"Layer {layer_idx} mean absolute difference: {torch.mean(torch.abs(diff))}")


    # Diff check multiple names in same layer
    # layer_to_check = 0
    # names_to_check = [
    #     f"L{layer_to_check}_o_proj_residual",
    #     f"L{layer_to_check}_pre_mlp_layer_norm",
    #     f"L{layer_to_check}_gate_silu",
    #     f"L{layer_to_check}_up_matmul",
    #     f"L{layer_to_check}_down_proj_residual",
    # ]
    # for name_to_check in names_to_check:
    #     print(f"Diff check {name_to_check}")
    #     diff = pytorch_debug_outputs[name_to_check] - pyvm_debug_outputs[name_to_check]
    #     print(f"Layer {layer_to_check} {name_to_check} diff shape: {diff.shape}")
    #     print(f"Layer {layer_to_check} {name_to_check} max absolute difference: {torch.max(torch.abs(diff))}")
    #     print(f"Layer {layer_to_check} {name_to_check} mean absolute difference: {torch.mean(torch.abs(diff))}")


    # Diff check different size tensors
    # pytorch_name_to_check = f"L{layer_to_check}_attn_out"
    # pytorch_name_to_check = f"L{layer_to_check}_attn_out"
    # for i in range(128):
    #     pyvm_name_to_check = f"L{layer_to_check}_B{i}_attn_out"
    #     pyvm_tensor = pyvm_debug_outputs[pyvm_name_to_check]
    #     pytorch_tensor = pytorch_debug_outputs[pytorch_name_to_check][i]
    #     print(f"PyVM shape: {pyvm_tensor.shape}")
    #     print(f"PyTorch shape: {pytorch_tensor.shape}")
    #     print(f"PyVM: {pyvm_tensor}")
    #     print(f"PyTorch: {pytorch_tensor}")
    #     diff = pytorch_tensor - pyvm_tensor
    #     print(f"Layer {layer_to_check} B{i} diff shape: {diff.shape}")
    #     print(f"Layer {layer_to_check} B{i} max absolute difference: {torch.max(torch.abs(diff))}")
    #     print(f"Layer {layer_to_check} B{i} mean absolute difference: {torch.mean(torch.abs(diff))}")

    non_warmup_times = times[config.num_warmup :]
    elapsed = sum(non_warmup_times) / len(non_warmup_times)
    print(f"Average time: {elapsed:.2f}s")

    if config.tokens:
        # breakpoint()
        to_cpu_0 = output_tokens.cpu()[0]
        print("Output ids: ", to_cpu_0)
        print("Output text: ", tokenizer.batch_decode(to_cpu_0, skip_special_tokens=True))

        print("----"*10)

        to_cpu_2 = output_tokens.cpu()[2]
        print("Output ids: ", to_cpu_2)
        print("Output text: ", tokenizer.batch_decode(to_cpu_2, skip_special_tokens=True))

    if config.token_details:
        ids_list = to_cpu_0.tolist()
        tokens = tokenizer.convert_ids_to_tokens(ids_list)

        table = []
        for i, token in enumerate(tokens):
            pos_id = i + prompt_len
            table.append([i, pos_id, token])

        print("More detailed output:")
        print(tabulate(table, headers=["output id", "position id", "token"]))

    tokens_per_second = (BATCH_SIZE * config.ntok) / elapsed
    print(f"Tokens per second: {tokens_per_second:.2f}")


if __name__ == "__main__":
    pydra.run(main)

kvm_batch_runner/generate.py kvm_batch_runner/instructions.py kvm_batch_runner/llama.py kvm_batch_runner/python_vm.py kvm_batch_runner/scheduler.p 