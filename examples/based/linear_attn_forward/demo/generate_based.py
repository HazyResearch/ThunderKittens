
import torch
from transformers import AutoTokenizer
from train.src.models.gpt import GPTLMHeadModel

# Load pretrained model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = GPTLMHeadModel.from_pretrained_hf(
    "hazyresearch/based-360m", 
    device="cuda", 
    implementation='tk',  # choices are [default, tk]
    silent=True           # prints info during inference if False
)

# Inputs
sample_inputs = [ 
    "The capital of California is Sacramento. The capital of Italy is Rome. The capital of France is Paris and capital of New York is",
    "After going to the movies,",
    "1, 2, 3, 4,"
]
context_length = 36
generation_length = 4
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

for input_text in sample_inputs:
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
                implementation="tk"
            )
            preds = generations[:, start:]
            pred_ids =  preds[0].tolist()
            pred = tokenizer.decode(pred_ids)
            input_text = tokenizer.decode(inputs[0].tolist())  

    print(f"{input_text} -> {pred}\n")

