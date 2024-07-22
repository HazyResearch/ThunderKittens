
import torch
import time
from transformers import AutoTokenizer
from tqdm import tqdm

torch.manual_seed(0)
torch.cuda.manual_seed(0)

# batch size
bs = 1

# Load pretrained models
print("\nLoading pretrained models...")
from train.src.models.gpt import GPTLMHeadModel as BasedGPTLMHeadModel
from based.models.mamba import MambaLMHeadModel
from based.models.transformer.gpt import GPTLMHeadModel
based_model = BasedGPTLMHeadModel.from_pretrained_hf(
    "hazyresearch/based-360m", 
    device="cuda", 
    implementation='tk',                # choices are [default, tk]
    swa_inference_mode="fast_rotary",   # choices [default, default_rotary, fast_rotary]
    silent=True,                        # will print more info during inference if set to False
    inference_bs=bs,
).to(torch.bfloat16)
# mamba_model = MambaLMHeadModel.from_pretrained_hf("hazyresearch/mamba-360m").to("cuda").to(torch.bfloat16)
# attn_model = GPTLMHeadModel.from_pretrained_hf("hazyresearch/attn-360m").to("cuda").to(torch.bfloat16)

# Setup tokenizer
context_length, generation_length = 1000, 48
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.padding_side = "left"
tokenizer.pad_token = " "
tokenizer.pad_token_id = 220

from datasets import load_dataset
ds = load_dataset("hazyresearch/based-squad")

model_name = "based"
model = based_model

results = []
total_time = []
preds_list = []
values_list = []

for i in tqdm(range(0, len(ds['validation']), bs)):
    if i > 100: break
    batch = ds['validation'][i:i+bs]
    
    input_texts = [b.strip() for b in batch['text']]
    values = batch['value']
    keys = batch['question']
    inputs = tokenizer.batch_encode_plus(
        input_texts, return_tensors="pt", padding=True, truncation=True, max_length=context_length
    ).input_ids.to("cuda")
    
    limit = inputs.shape[-1] + generation_length
    start = inputs.shape[-1]

    model.eval()
    fn = model.generate
    if 'based' in model_name:
        with torch.no_grad():
            torch.cuda.synchronize()
            start_time = time.time()
            generations = fn(
                input_ids=inputs,
                max_length=limit,
                temperature=0.1,
                top_k=1,
                top_p=1.0,
                implementation="tk"
            )
            torch.cuda.synchronize()
            end_time = time.time()
    else:
        with torch.no_grad():
            torch.cuda.synchronize()
            start_time = time.time()
            generations = fn(
                input_ids=inputs,
                max_length=limit,
                temperature=0.1,
                top_k=1,
                top_p=1.0
            )
            torch.cuda.synchronize()
            end_time = time.time()

    preds = generations[:, start:]
    preds = preds.tolist()
    preds = tokenizer.batch_decode(preds)
    seen_inputs = tokenizer.batch_decode(inputs)

    for inp, p, k, v in zip(seen_inputs, preds, keys, values):
        correct = v.lower() in p.lower()
        results.append(correct)
        preds_list.append(p.split("\n")[0])
        values_list.append(v.split("\n")[0])
    total_time.append(end_time-start_time)

print(f"{model_name=}: score={sum(results)/len(results)}; total_time={sum(total_time)} seconds")

output_file = f"{model_name}_predictions_and_values.txt"
with open(output_file, 'w') as f:
    for pred, value in zip(preds_list, values_list):
        f.write(f"Prediction: {pred}\tValue: {value}\n")
print(f"Predictions and values saved to {output_file}")

