import torch
from transformers import AutoTokenizer, PreTrainedTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM


def main():
    dtype = torch.float16
    device = torch.device(f"cuda:0")

    model = LlamaForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf",
            cache_dir="/workspace/hf_home/",
            torch_dtype=dtype,
            attn_implementation="flash_attention_2",
            device_map=device,
        )

    model.eval()
    model.to(device)

    x = torch.arange(32).unsqueeze(0)

    x = x.to(device)

    y = model(x).logits

    print(f"output logits:", y.shape, y.dtype, y.device)

    torch.save(y, "original_attn_output.pt")


if __name__ == "__main__":
    main()
