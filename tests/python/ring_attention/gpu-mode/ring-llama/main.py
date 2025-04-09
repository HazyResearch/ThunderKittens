import os
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer, AutoModelForCausalLM
from transformers.modeling_utils import no_init_weights
from transformers.utils.generic import ContextManagers
from configuration_llama import LlamaConfig
from modeling_llama import LlamaForCausalLM
from tokenization_llama_fast import LlamaTokenizerFast
import torch.distributed as dist

from ring_flash_attn import ring_flash_attn_qkvpacked_func


def load_model(
    model_name: str,
    cache_dir: str,
    torch_dtype: torch.dtype,
    device: torch.DeviceObjType,
    skip_load: bool = False,
    no_weight_init: bool = False,
):
    tokenizer = LlamaTokenizerFast.from_pretrained(model_name, cache_dir=cache_dir)

    if skip_load:
        config = LlamaConfig.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        config._attn_implementation = "flash_attention_2"
        config = LlamaForCausalLM._autoset_attn_implementation(
            config, torch_dtype=torch_dtype
        )
        print("using llama config:", config)
        init_contexts = [no_init_weights(_enable=no_weight_init)]
        with ContextManagers(init_contexts):
            LlamaForCausalLM._set_default_torch_dtype(torch_dtype)
            model = LlamaForCausalLM(config)
    else:
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2",
            device_map=device,
        )
    return model, tokenizer


# source https://stackoverflow.com/a/1094933
def sizeof_fmt(num, suffix="B"):
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


@torch.inference_mode()
def main():
    dtype = torch.float16
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{rank}")
    else:
        rank = 0
        world_size = 1
        device = torch.device(f"cuda:0")

    print(f"world_size: {world_size}, device: {device}")

    skip_load = False
    model, tokenizer = load_model(
        "meta-llama/Llama-2-7b-chat-hf",
        cache_dir="/workspace/hf_home/",
        torch_dtype=dtype,
        device=device,
        skip_load=skip_load,
        no_weight_init=skip_load,
    )
    model.eval()
    model.to(device)

    # x = tokenizer("Hello I am the llama, test", return_tensors="pt")
    # tokenized_input = x.input_ids

    torch.cuda.reset_peak_memory_stats()
    max_mem_allocated_before = torch.cuda.max_memory_allocated(device)

    # temporarily use dummy input (ensure shape is same for both devices)
    tokenized_input = torch.arange(32).unsqueeze(0)
    position_ids = torch.arange(32).unsqueeze(0)
    # print("tokenized_input", tokenized_input.shape)

    input_chunks = tokenized_input.chunk(chunks=world_size, dim=1)
    position_ids = position_ids.chunk(chunks=world_size, dim=1)
    # print("input_chunks", input_chunks)

    x = input_chunks[rank]
    x_pos_ids = position_ids[rank]
    print(f"{rank=} {x.shape=}")

    x = x.to(device)
    x_pos_ids = x_pos_ids.to(device)
    print(f"model input x for rank: {rank}: {x} (position_ids: {x_pos_ids})")

    y = model(x, position_ids=x_pos_ids).logits

    print(f"output logits for rank: {rank}:", y.shape, y.dtype, y.device)

    vocab_size = y.size(-1)
    print("y", y[0, 1, 0:10])
    a = torch.zeros(x.size(0), x.size(1), vocab_size, dtype=y.dtype, device=device)
    b = torch.zeros(x.size(0), x.size(1), vocab_size, dtype=y.dtype, device=device)

    torch.distributed.all_gather([a, b], y, group=None, async_op=False)
    print("ok", a[0, 1, 0:10])

    max_mem_allocated_after = torch.cuda.max_memory_allocated(device)
    print(
        f"{device} delta: {sizeof_fmt(max_mem_allocated_after-max_mem_allocated_before)}"
    )

    # if rank == 0:
    #     torch.save(torch.cat([a,b], dim=1).squeeze(), "ring_attn_output.pt")


if __name__ == "__main__":
    main()
