import pydra
import torch
from tabulate import tabulate
from torch import Tensor
from tqdm import tqdm
from transformers import AutoTokenizer

from kvm_runner.llama import LlamaForCausalLM
from kvm_runner.python_vm import PyVM_Runner
from kvm_runner.scheduler import PrintInfo
from kvm_runner.types import BatchState


class ScriptConfig(pydra.Config):
    model: str = "meta-llama/Llama-3.2-1B-Instruct"
    device: str = "cuda:0"
    prompt: str = "tell me a funny joke about cookies"
    chat: bool = False
    ntok: int = 100
    mode: str = "model"
    num_attention_partitions: int = 2
    add_print_instructions: bool = False
    print_layer_filter: list[int] | None = None
    print_name_filter: list[str] | None = None
    print_state_filter: list[str] | None = None


def pytorch_model_generate(
    config: ScriptConfig,
    model: LlamaForCausalLM,
    output_tokens: Tensor,
    prompt_len: int,
):
    start_position_ids = torch.ones(1, dtype=torch.long, device=model.device) * (
        prompt_len
    )
    for i in tqdm(range(1, config.ntok)):
        position_ids = start_position_ids + i
        decode_inp = BatchState(
            input_ids=output_tokens[i - 1 : i],
            position_ids=position_ids,
            seq_len=prompt_len + i + 1,
        )
        decode_output: BatchState = model(decode_inp)
        assert decode_output.output_ids is not None
        output_tokens[i] = decode_output.output_ids


def pyvm_generate(
    config: ScriptConfig,
    model: LlamaForCausalLM,
    output_tokens: Tensor,
    prompt_len: int,
):
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
        num_attention_partitions=config.num_attention_partitions,
        print_info=print_info,
    )

    for i in tqdm(range(1, config.ntok)):
        input_ids = output_tokens[i - 1 : i]
        output_ids = runner.run(input_ids, pos_id=prompt_len + i)
        output_tokens[i] = output_ids


def kvm_interpreter_generate(
    config: ScriptConfig,
    model: LlamaForCausalLM,
    output_tokens: Tensor,
    prompt_len: int,
):
    """
    The magic goes here!
    """
    raise NotImplementedError


def main(config: ScriptConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    model = LlamaForCausalLM.from_pretrained(config.model, device=config.device)

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
            pytorch_model_generate(config, model, output_tokens, prompt_len)
        case "pyvm":
            pyvm_generate(config, model, output_tokens, prompt_len)
        case "kvm":
            kvm_interpreter_generate(config, model, output_tokens, prompt_len)
        case _:
            raise ValueError(f"Invalid mode: {config.mode}")

    to_cpu = output_tokens.cpu()
    print("Output ids: ", to_cpu)
    print("Output text: ", tokenizer.decode(to_cpu))

    ids_list = to_cpu.tolist()
    tokens = tokenizer.convert_ids_to_tokens(ids_list)

    table = []
    for i, token in enumerate(tokens):
        pos_id = i + prompt_len
        table.append([i, pos_id, token])

    print("More detailed output:")
    print(tabulate(table, headers=["output id", "position id", "token"]))


if __name__ == "__main__":
    pydra.run(main)
