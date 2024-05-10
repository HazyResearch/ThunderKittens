
import torch
from transformers import AutoTokenizer
from based.models.gpt import GPTLMHeadModel

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = GPTLMHeadModel.from_pretrained_hf("hazyresearch/based-360m", device="cuda")

# Inputs
input_text = "The capital of California is Sacramento. The capital of Italy is Rome. The capital of France is" 

context_length = 36
generation_length = 2
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
inputs = tokenizer.batch_encode_plus(
    [input_text], return_tensors="pt", padding=True, truncation=True, max_length=context_length
).input_ids.to("cuda")

limit = inputs.shape[-1] + generation_length
start = inputs.shape[-1]
print(f"{start=}, {limit=}")

# Generate
model.eval()
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    with torch.no_grad():

        fn = model.generate
        generations = fn(
            input_ids=inputs,
            max_length=limit,
            temperature=0.1,
            top_k=1,
            top_p=1.0,
        )
        preds = generations[:, start:]
        pred_ids =  preds[0].tolist()
        pred = tokenizer.decode(pred_ids)
        input_text = tokenizer.decode(inputs[0].tolist())  

print(f"{input_text} -> {pred}")



