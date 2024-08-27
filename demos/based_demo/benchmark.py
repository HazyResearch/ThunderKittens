import torch
import time
import os
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np


def get_model(model_name, impl, batch_size, seqlen):
    from based.models.baselines.mamba_model import MambaLMHeadModel
    from train.src.models.gpt import GPTLMHeadModel as BasedGPTLMHeadModel

    if model_name == 'based': 
        return BasedGPTLMHeadModel.from_pretrained_hf(
                 "hazyresearch/my-awesome-model", 
                device="cuda", 
                implementation=impl,  # choices are [default, tk]
                silent=True,          # prints info during inference if False
                inference_bs=batch_size,
                override_seqlen=seqlen,
                swa_inference_mode="fast_rotary",
            ).to(dtype=torch.bfloat16)
    elif model_name == 'mamba':
        return MambaLMHeadModel.from_pretrained_hf(
            "hazyresearch/mamba-360m",
            override_seqlen=seqlen,
        ).to("cuda").to(dtype=torch.bfloat16)
    elif model_name == "mamba2": 
        return MambaLMHeadModel.from_pretrained_hf(
            "state-spaces/mamba2-370m",
            override_seqlen=seqlen,
        ).to("cuda").to(torch.float16)
    elif model_name == "attn": 
        return BasedGPTLMHeadModel.from_pretrained_hf(
            "hazyresearch/attn-360m",
            override_seqlen=seqlen,
        ).to("cuda").to(dtype=torch.bfloat16)
    else:
        assert 0, print("Unknown model.")


def benchmark_models():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    NUM_ITERS = 5
    WARMUP_ITERS = 1
    assert NUM_ITERS > WARMUP_ITERS, print("Not enough iters.")
    toks_per_sec = []

    context_len, input_len, output_len, cg = 8192, 8000, 1, True
    assert context_len % 64 == 0, print("Context length must be divisible by 64.")
    benchmark_dims = [ 
        ('based', 'tk', [4, 16, 32, 64, 128], input_len, output_len), 
        ('based', 'fla_parallel', [4, 16, 32, 64, 128], input_len, output_len), 
        ('mamba', 'default', [4, 16, 32, 64, 128], input_len, output_len),  # benchmarks CUDA kernel from Mamba
        ('mamba2', 'default', [4, 16, 32, 64, 128], input_len, output_len), # benchmarks CUDA kernel from Mamba-2
        ('attn', 'default', [4, 16, 32, 64, 128], input_len, output_len),   # benchmarks Flash Attention 
    ]
    for model_name, impl, batch_sizes, input_len, output_len in benchmark_dims:
        print(f"Running {model_name} with {impl}...")

        for batch_size in batch_sizes:
            try:
                model = get_model(model_name, impl, batch_size, context_len)
            except Exception as e:
                print(e)
                continue
            inputs = torch.randint(low=0, high=len(tokenizer), size=(batch_size, input_len), device="cuda")
            limit = inputs.shape[-1] + output_len
            start = inputs.shape[-1]

            times_iters = []
            model.eval()
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
                            implementation=impl,
                            cg=cg,
                        )

                        torch.cuda.synchronize()
                        end_t = time.time()
                        if i >= WARMUP_ITERS: 
                            times_iters.append((end_t-start_t)*1000)

                    toks_per_sec.append({
                        'model': model_name,
                        'impl': impl,
                        'bs': batch_size, 
                        'input_len': input_len, 
                        'output_len': output_len, 
                        'time': sum(times_iters)/len(times_iters),
                        'cg': cg
                    })
                except Exception as e:
                    print(e)
                    pass
            import gc
            try:
                del model;gc.collect();torch.cuda.empty_cache()
            except:
                pass
    return toks_per_sec


def plot_results(toks_per_sec):
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
    
    batch_sizes = sorted(batch_size_2_data.keys())    
    impl_models = sorted(set(key for bs_data in batch_size_2_data.values() for key in bs_data.keys()))
    
    # Set up the plot
    # Increase font size globally
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(figsize=(15, 8))
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
                        xytext=(0, 3),  
                        textcoords="offset points",
                        ha='center', va='bottom', rotation=90)
    
    # Customize the plot
    ax.set_ylabel('Time (ms)')
    ax.set_xlabel('Batch Size')
    ax.set_title(f'Performance (Prefill Length: {input_len}, Decode Length: {output_len})')
    ax.set_xticks(x)
    ax.set_xticklabels(batch_sizes)
    ax.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')

    ax.text(4, 8, 'Empty means approach failed.', color='red', fontsize=14, fontweight='bold')

    # Adjust layout and display the plot
    if not os.path.exists("plots/"): os.makedirs("plots/")
    plt.tight_layout()
    plt.savefig(f'plots/benchmark_input{input_len}_output{output_len}.png', dpi=300, bbox_inches='tight')
    plt.close()


toks_per_sec = benchmark_models()
plot_results(toks_per_sec)
