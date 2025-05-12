import torch
from einops import rearrange
from kvm_unity.kvm import KVM_Interpreter
from kvm_unity.latency.instructions import Globals


def interpret_with_kvm(
    globs: Globals,
    kvm_func,
):
    fourD_k_cache = rearrange(globs.k_cache, "l b t h d -> (l b) t h d")
    fourD_v_cache = rearrange(globs.v_cache, "l b t h d -> (l b) t h d")

    kvm_func(
        # vm stuff
        globs.barriers,
        globs.instructions,
        globs.timings,
        # weights
        globs.qkv_proj,
        globs.attn_ln_weight,
        globs.o_proj,
        globs.mlp_ln_weight,
        globs.up_proj,
        globs.gate_proj,
        globs.down_proj,
        globs.lm_head_norm_weights.data,
        globs.lm_head_weights.data,
        fourD_k_cache,
        fourD_v_cache,
        # rope
        globs.rope_cos,
        globs.rope_sin,
        # activations
        globs.hidden_states,
        globs.post_ln_rope_q,
        globs.attn_out,
        globs.attn_lse_intermediates,
        globs.attn_out_intermediates,
        globs.silu_out,
        globs.logits,
        # scalars
        globs.pos_id,
        globs.attn_scale,
        globs.rms_norm_eps,
        globs.skip_attn_reduction,
        stream=torch.cuda.current_stream(),
    )


class LatencyKVM_Interpreter(KVM_Interpreter):
    def interpret(self, globs: Globals):
        interpret_with_kvm(globs, self.kvm_func)
