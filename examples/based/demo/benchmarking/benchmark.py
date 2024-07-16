
import torch
import time
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np

# tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# models
from train.src.models.gpt import GPTLMHeadModel as BasedGPTLMHeadModel
from based.models.mamba import MambaLMHeadModel
from based.models.transformer.gpt import GPTLMHeadModel

def get_model(model_name, impl, batch_size):
    if model_name == 'based': 
        return BasedGPTLMHeadModel.from_pretrained_hf(
            "hazyresearch/based-360m", 
            device="cuda", 
            implementation=impl,  # choices are [default, tk]
            silent=True,          # prints info during inference if False
            inference_bs=batch_size
        )
    elif model_name == 'mamba':
        return MambaLMHeadModel.from_pretrained_hf("hazyresearch/mamba-360m").to("cuda")
    elif model_name == "attn": 
        return GPTLMHeadModel.from_pretrained_hf("hazyresearch/attn-360m").to("cuda")
    else:
        assert 0, print("Unknown model.")

# benchmark
NUM_ITERS = 2
WARMUP_ITERS = 1
assert NUM_ITERS > WARMUP_ITERS, print("Not enough iters.")
toks_per_sec = []

input_len, output_len = 1024, 1024
benchmark_dims = [ 
    ('based', 'tk', 1, input_len, output_len), 
    ('based', 'tk', 4, input_len, output_len), 
    ('based', 'tk', 16, input_len, output_len), 
    ('based', 'tk', 32, input_len, output_len),
    ('based', 'tk', 64, input_len, output_len), 

    ('based', 'default', 1, input_len, output_len), 
    ('based', 'default', 4, input_len, output_len), 
    ('based', 'default', 16, input_len, output_len), 
    ('based', 'default', 32, input_len, output_len),
    ('based', 'default', 64, input_len, output_len), 

    ('mamba', 'default', 1, input_len, output_len), 
    ('mamba', 'default', 4, input_len, output_len), 
    ('mamba', 'default', 16, input_len, output_len), 
    ('mamba', 'default', 32, input_len, output_len),
    ('mamba', 'default', 64, input_len, output_len), 

    ('attn', 'default', 1, input_len, output_len), 
    ('attn', 'default', 4, input_len, output_len), 
    ('attn', 'default', 16, input_len, output_len), 
    ('attn', 'default', 32, input_len, output_len),
    ('attn', 'default', 64, input_len, output_len), 
]
for model_name, impl, batch_size, input_len, output_len in benchmark_dims:

    model = get_model(model_name, impl, batch_size)

    inputs = torch.randint(
        low=0,
        high=len(tokenizer),
        size=(batch_size, input_len),
        device="cuda"
    )
    limit = inputs.shape[-1] + output_len
    start = inputs.shape[-1]
    print(f"{start=}, {limit=}")

    tps_iters = []

    # Generate
    model.eval()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            
            try:
                for i in range(NUM_ITERS): 
                    fn = model.generate
                    torch.cuda.synchronize()
                    start_t = time.time()

                    generations = fn(
                        input_ids=inputs,
                        max_length=limit,
                        temperature=0.1,
                        top_k=1,
                        top_p=1.0,
                        implementation=impl
                    )

                    torch.cuda.synchronize()
                    end_t = time.time()
                    tps = ((input_len + output_len) / (end_t-start_t))
                    if i >= WARMUP_ITERS: tps_iters.append(tps)

                toks_per_sec.append({
                    'model': model_name,
                    'impl': impl,
                    'bs': batch_size, 
                    'input_len': input_len, 
                    'output_len': output_len, 
                    'time': sum(tps_iters)/len(tps_iters)
                })
            except Exception as e:
                print(e)

# plot
def plot_results(toks_per_sec):
    # Prepare data for plotting
    batch_size_2_data = {}
    
    for value in toks_per_sec:
        model = value['model']
        impl = value['impl'] 
        time = value['time']
        bs = value['bs']
        key = f"{impl} {model}"
        if bs not in batch_size_2_data:
            batch_size_2_data[bs] = {}
        batch_size_2_data[bs][key] = time
        input_len = value['input_len']
        output_len = value['output_len']
    
    # Extract unique batch sizes and sort them
    batch_sizes = sorted(batch_size_2_data.keys())
    
    # Get unique implementation-model combinations
    impl_models = sorted(set(key for bs_data in batch_size_2_data.values() for key in bs_data.keys()))
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Set the width of each bar and positions of the bars
    num_impl_models = len(impl_models)
    width = 0.8 / num_impl_models
    x = np.arange(len(batch_sizes))
    
    # Create the bars
    for i, impl_model in enumerate(impl_models):
        data = [batch_size_2_data[bs].get(impl_model, 0) for bs in batch_sizes]
        offset = (i - (num_impl_models - 1) / 2) * width
        bars = ax.bar(x + offset, data, width, label=impl_model, alpha=0.7)
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.0f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', rotation=90)
    
    # Customize the plot
    ax.set_ylabel('Tokens per Second')
    ax.set_xlabel('Batch Size')
    ax.set_title(f'Performance Comparison (Input: {input_len}, Output: {output_len})')
    ax.set_xticks(x)
    ax.set_xticklabels(batch_sizes)
    ax.legend(title='Implementation Model', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.savefig(f'benchmarking/plots/performance_comparison_input{input_len}_output{output_len}.png', dpi=300, bbox_inches='tight')
    plt.close()


plot_results(toks_per_sec)
print(toks_per_sec)

