import os 
import matplotlib.pyplot as plt
import numpy as np
import pickle

FA_COLOR = "#9467bd"  
TK_COLOR = "#67bbbd"
FLA_COLOR = "#4EAE4E" #"#2ba02b"
PYTORCH_COLOR = "#ED7D66" # "#ee4b2b"
CU_COLOR = "#F29441" #ff7f0f"

metadatas = [
    {   
        "Tag": "conv",
        "Title": "Long Convolution (B=16, D=1024)",
        "method_remap": {
            "fftconv_pytorch": "PyTorch", 
            "fftconv_cutlass": "FlashFFTConv (CUDA)",
            "fftconv_tk": "ThunderKittens"
        },
        "method_color": {
            "fftconv_pytorch": PYTORCH_COLOR,
            "fftconv_cutlass": FA_COLOR,
            "fftconv_tk": TK_COLOR,
        },
        "filename": "fftconv_method2tflops.pkl",
        "include_x": False,
        "is_short": True,
    },
    {
        "Tag": "rotary",
        "Title": "Rotary Encoding (B=16, H=8, D=128)",
        "method_remap": {
            "rotary_torch": "PyTorch",
            "rotary_flash": "FlashRotary Triton",
            "rotary_tk": "ThunderKittens"
        },
        "method_color": {
            "rotary_torch": PYTORCH_COLOR,
            "rotary_flash": FA_COLOR,
            "rotary_tk": TK_COLOR,
        },
        "filename": "rotary_method2tflops.pkl",
        "include_x": False,
        "is_short": True,
    },
    {
        "Tag": "fusednorm",
        "Title": "Fused dropout-residual-norm (B=16, D=1024)",
        "method_remap": {
            "layernorm_pytorch": "PyTorch",
            "layernorm_triton": "FlashNorm Triton",
            "layernorm_tk": "ThunderKittens"
        },
        "method_color": {
            "layernorm_pytorch": PYTORCH_COLOR,
            "layernorm_tk": TK_COLOR,
            "layernorm_triton": FA_COLOR
        },
        "data": {
            "layernorm_pytorch": {
                1024: 0.18, 
                2048: 0.35,
                4096: 0.68,
                8192: 0.91,
                16384: 1.57
            },
            "layernorm_triton": {
                1024: 0.39,
                2048: 0.49,
                4096: 0.49,
                8192: 0.80,
                16384: 1.11
            },
            "layernorm_tk": {
                1024: 0.84,
                2048: 1.00,
                4096: 1.11,
                8192: 1.18,
                16384: 1.22
            }
        },
        "filename": None,
        "include_x": True,
        "is_short": True,
    },
    {
        "Tag": "based_fwd",
        "Title": "Based Linear Attention (B=16, H=16, D=64)",
        "method_remap": {
            "based_tk": "ThunderKittens",
            "based_fla": "FLA Triton",
            "based_torch": "PyTorch"
        },
        "method_color": {
            "based_tk": TK_COLOR,
            "based_fla": FLA_COLOR,
            "based_torch": PYTORCH_COLOR
        },
        'data': {
            'based_tk': {
                1024: 192,
                2048: 203,
                4096: 210,
                8192: 214,
                16384: 217 
            },
            'based_fla': {
                1024: 51.70,
                2048: 51.07,
                4096: 38.24,
                8192: 24.77,
                16384: 14.98 
            },
            "based_torch": {
                1024: 0.42,
                2048: 0.42,
                4096: 0,
                8192: 0,
                16384: 0 
            }
        },
        "filename": None,
        "include_x": True,
        "is_short": True,
    },
    {
        "Tag": "hedgehog_fwd",
        "Title": "Hedghog Linear Attention (B=16, H=8, D=128)",
        "method_remap": {
            "hedgehog_tk": "ThunderKittens",
            "hedgehog_fla": "FLA Triton",
            "hedgehog_pytorch": "PyTorch"
        },
        "method_color": {
            "hedgehog_tk": TK_COLOR,
            "hedgehog_fla": FLA_COLOR,
            "hedgehog_pytorch": PYTORCH_COLOR
        },
        "filename": "hedgehog_method2tflops.pkl",
        "include_x": False,
        "is_short": True,
    },
    {
        "Tag": "attn_nc_fwd", 
        "Title": "Attention Inference (B=16, H=16, D=128)",
        "method_remap": {
            "attn_fwd_nc_fa3": "FlashAttention-3",
            "attn_fwd_nc_tk": "ThunderKittens"
        },
        "method_color": {
            "attn_fwd_nc_fa3": FA_COLOR,
            "attn_fwd_nc_tk": TK_COLOR
        },
        'data': {
            'attn_fwd_nc_fa3': {
                768: 444,
                1536: 616,
                3072: 670,
                6144: 668,
                12288: 692
            },
            'attn_fwd_nc_tk': {
                768: 508,
                1536: 612,
                3072: 667,
                6144: 635,
                12288: 653
            }
        },
        "filename": None,
        "include_x": True,
        "is_short": True,
    },
    {
        "Tag": "attn_nc_bwd",
        "Title": "Attention Backwards (B=16, H=16, D=128)",
        "method_remap": {
            "attn_bwd_nc_fa3": "FlashAttention-3",
            "attn_bwd_nc_tk": "ThunderKittens"
        },
        "method_color": {
            "attn_bwd_nc_fa3": FA_COLOR,
            "attn_bwd_nc_tk": TK_COLOR
        },
        'data': {
            'attn_bwd_nc_fa3': {
                768: 253,
                1536: 379,
                3072: 451,
                6144: 495,
                12288: 500
            },
            'attn_bwd_nc_tk': {
                768: 364,
                1536: 493,
                3072: 547,
                6144: 568,
                12288: 553
            }
        },
        "filename": None,
        "include_x": False,
        "is_short": True,
    },
    {
        "Tag": "gemm",
        "Title": "Matrix Multiplication (M=N=K)",
        "method_remap": {
            "cublas": "CuBLAS",
            "tk": "ThunderKittens"
        },
        "method_color": {
            "cublas": CU_COLOR,
            "tk": TK_COLOR
        },
        'data' : {
            'cublas': {
                1024: 176,
                2048: 562,
                4096: 757,
                8192: 855,
                16384: 804
            },
            'tk': {
                1024: 247,
                2048: 658,
                4096: 771,
                8192: 810,
                16384: 793
            }
        },
        "filename": None,
        "include_x": True,
        "is_short": True,
    },
    {
        "Tag": "mamba-2", 
        "Title": "Mamba-2 (B=16, H=32, D=64, D-State 64)",
        "method_remap": {
            "mamba_tk": "ThunderKittens",
            "mamba_triton": "Mamba-2 Triton",
        },
        "method_color": {
            "mamba_tk": TK_COLOR,
            "mamba_triton": FA_COLOR,
        },
        'data': {
            'mamba_tk': {
                1024: 128,
                2048: 134,
                4096: 138,
                8192: 139,
                16384: 140
            },
            'mamba_triton': {
                1024: 34,
                2048: 44,
                4096: 41,
                8192: 40,
                16384: 40 
            }
        },
        "filename": None,
        "include_x": True,
        "is_short": True,
    }, 
]



for metadata in metadatas:
    include_x = metadata['include_x']
    is_short = metadata['is_short']

    # Load the data from the pickle file
    # sample data: {'TK': {1024: 50000000}}, where 1024 is the sequence length and 50000000 is the speed in TFLOPs/s
    filen = metadata['filename']
    if metadata['filename'] is not None:
        if not os.path.exists(filen): continue
        with open(filen, 'rb') as f:
            flopdata = pickle.load(f)
    elif 'data' in metadata:
        flopdata = metadata['data']
    else:
        continue
    sequence_lengths = [] 
    for method, dic in flopdata.items():
        for k, v in dic.items():
            if k not in sequence_lengths:
                sequence_lengths.append(k)
    sequence_lengths.sort()
    methods = flopdata.keys()

    print(f"Plotting {metadata['Title']}")

    # Set up the plot
    x = np.arange(len(sequence_lengths))  # the label locations

    if len(methods) == 2:
        width = 0.35
    else:
        width = 0.3

    if is_short:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig, ax = plt.subplots()

    # set larger font size
    plt.rcParams.update({'font.size': 14})

    # put TK method last in sorted order
    tk_name = [n for n in methods if 'tk' in n][0]
    method_order = list(methods)
    method_order.remove(tk_name)
    method_order.append(tk_name)

    # Plot each method's flops for each sequence length
    max_flops = 0
    for i, method in enumerate(method_order):

        color = metadata['method_color'][method]

        flops = [] 
        for seq_len in sequence_lengths:
            if seq_len not in flopdata[method]:
                flops.append(0)
            else:
                flops.append(flopdata[method][seq_len])
        max_flop = max([f for f in flops if f != "X"])
        if max_flop > max_flops:
            max_flops = max_flop

        if 'method_remap' in metadata:
            method_name = metadata['method_remap'].get(method, method)
            ax.bar(x + i * width, flops, width, label=method_name, color=color)

            # Add text labels to the bars
            for j, flops in enumerate(flops):
                # round flops up to the nearest int
                if flops != "OOM":
                    if flops > 5:
                        print_flop = flops+0.03
                        import math 
                        flops = math.ceil(flops)
                        
                    else:
                        print_flop = flops+0.01
                        flops = round(flops, 1)
                        
                else:
                    print_flop = 0
                if flops == 0: str_flops = "X"
                else: str_flops = str(flops)
                # use bold font
                ax.text(x[j] + i * width, print_flop, str_flops, ha='center', va='bottom', fontsize=14)

    # y_max 
    if max_flops > 800: 
        push = 90
    elif max_flops > 600:
        push = 200 
    elif max_flops > 400:
        push = 150
    elif max_flops < 2: 
        push = 0.3
    elif max_flops < 80:
        push = 10
    else:
        push = 30
    ax.set_ylim(0, max_flops + push)

    # Add labels, title, and custom x-axis tick labels
    ax.set_xlabel('Size')
    if include_x:
        ax.set_ylabel('Speed (TFLOPs/s)')
    ax.set_title(metadata['Title'])
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(sequence_lengths)

    ax.legend() 

    # Show the plot
    plt.tight_layout()
    tag = metadata['Tag']
    
    if not os.path.exists('paper_plots'):
        os.makedirs('paper_plots')
    plt.savefig(f'paper_plots/{tag}_len2tflops.png', dpi=300)
    plt.show()



