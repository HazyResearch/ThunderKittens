import pytest
import torch

from based.models.gpt import GPTLMHeadModel, GPT2MixerConfig


BASE_CONFIG = {
    "n_embd": 256,
    "special_initializer": True,
    "n_head": 16,
    "n_layer": 2,
    "rms_norm": True,
    "fused_mlp": False,
    "attn_pdrop": 0,
    "embd_pdrop": 0,
    "n_positions": 2048,
    "resid_pdrop": 0,
    "mlp_fc1_bias": False,
    "mlp_fc2_bias": False,
    "fused_bias_fc": True,
    "out_proj_bias": False,
    "qkv_proj_bias": False,
    "use_flash_attn": False,
    "residual_in_fp32": True,
    "activation_function": "geglu",    # flagging,
    # rotary_emb_fraction: 1        # flagging -- 0.5 for the other model.
    "fused_dropout_add_ln": True,
    "max_position_embeddings": 2048,   # flagging -- not RoPE,
    "pad_vocab_size_multiple": 8,
    "reorder_and_upcast_attn": False,
    "scale_attn_by_inverse_layer_idx": False,
    "n_inner": 1408 * 2,
    "mlp_type": "alt"                 
}

CONFIGS = {
    "conv": GPT2MixerConfig(
        **{
            "mixer": {
                "_target_": "based.models.mixers.convolution.ShortConvolution",
                "kernel_size": 3,
                "use_cuda": False
            }, **BASE_CONFIG
        }
    ),
    "base_conv": GPT2MixerConfig(
        **{
            "mixer": {
                "_target_": "based.models.mixers.convolution.BaseConv",
                "kernel_size": 3,
                "l_max": 2048
            }, **BASE_CONFIG
        }
    ),
    "linear_attn": GPT2MixerConfig(
        **{
            "feature_dim": 16,
            "mixer": {
                "_target_": "based.models.mixers.linear_attention.LinearAttention",
                "feature_map": {
                    "_target_": "based.models.mixers.linear_attention.TaylorExp",
                    "input_dim": 16,
                },
                "num_heads": 16
            }, 
            **BASE_CONFIG
        }
    ),
    "sliding": GPT2MixerConfig(
        **{
            "mixer": {
                "_target_": "based.models.mixers.slide_attention.SlidingAttention",  
                "window_size": 128,
                # "embed_dim": BASE_CONFIG["n_embd"],
                "num_heads": 16,
                "causal": True,
                # "do_update": True
            }, **BASE_CONFIG
        }
    ),
    "mha": GPT2MixerConfig(
        **{
            "mixer": {
                "_target_": "based.models.mixers.mha.MHA",  
                "window_size": (64, -1),
                # "embed_dim": BASE_CONFIG["n_embd"],
                "num_heads": 16,
                "causal": True,
                "use_flash_attn": True
                # "do_update": True
            }, **BASE_CONFIG
        }
    ),
}

CONFIGS_TO_TEST = [
    # "conv",
    # "base_conv",
    # "linear_attn",
    # "sliding",
    "mha"
]

# SE (02/26): borrowing these tolerances from Mamba's test_selective_state_update
# https://github.com/state-spaces/mamba/blob/ce59daea3a090d011d6476c6e5b97f6d58ddad8b/tests/ops/triton/test_selective_state_update.py#L24C1-L26C32
DTYPE_TO_ATOL = {
    torch.float32: 1e-3,
    torch.float16: 1e-2,
    torch.bfloat16: 5e-2,
}
DTYPE_TO_RTOL = {
    torch.float32: 3e-4,
    torch.float16: 5e-4,
    torch.bfloat16: 1e-2,
}

@pytest.mark.parametrize("config", CONFIGS_TO_TEST)
@pytest.mark.parametrize("prefill_size", [1, 128])
@pytest.mark.parametrize("cache_graph", [False])
@pytest.mark.parametrize("naive_generation", [False])
@pytest.mark.parametrize("dtype", [
    torch.float32, 
    torch.float16, torch.bfloat16])
def test_generation(
    config: GPT2MixerConfig, 
    prefill_size: int, 
    cache_graph: bool, 
    naive_generation: bool,
    dtype: torch.dtype,
):
    if config in ["mha", "sliding"] and dtype == torch.torch.float32:
        # SE: MHA is not implemented for float32
        return 

    config = CONFIGS[config]
    batch_size = 4
    n_generated_tokens = 64
    device = "cuda"


    model = GPTLMHeadModel(config).to(device=device, dtype=dtype)
    model.eval()
    torch.manual_seed(0)
    input_ids = torch.randint(1, 1000, (batch_size, prefill_size), dtype=torch.long, device=device)

    fn = model.generate_naive if naive_generation else model.generate
    out = fn(
        input_ids=input_ids, 
        max_length=prefill_size + n_generated_tokens, 
        return_dict_in_generate=True, 
        output_scores=True, 
        eos_token_id=None,  # ensure this is None so that we test full output length
        top_k=1, # enforces that we take the top token
        cg=cache_graph 
    )
    print("done with generation")

    # SE: need to clone because of "RuntimeError: Inference tensors cannot be saved for 
    # backward. To work around you can make a clone to get a normal tensor and use it 
    # in autograd.
    scores = torch.stack(out.scores, dim=1)
    out = out.sequences.clone()

    # pick a tolerance based on dtype -- for bfloat16 and float16 we have to be more lenient
    atol, rtol = DTYPE_TO_ATOL[dtype], DTYPE_TO_RTOL[dtype]

    # get reference output by repeatedly using the parallel view of the model
    # (e.g. with a transformer this is like generating without a kv cache)
    for i in range(n_generated_tokens):
        scores_ref = model(input_ids=out[:, :prefill_size + i]).logits
        out_ref = scores_ref.argmax(dim=-1)
        assert torch.allclose(scores_ref[:, -1], scores[:, i], atol=atol, rtol=rtol)
        assert torch.allclose(out[:, prefill_size + i], out_ref[:, -1])
