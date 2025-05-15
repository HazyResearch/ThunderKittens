from einops import rearrange
from kvm_unity.kvm import KVM_Interpreter
from kvm_unity.throughput.instructions import Globals


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
        globs.qkv_proj_weights,
        globs.attn_ln_weights,
        globs.o_proj_weights,
        globs.mlp_ln_weights,

        globs.up_proj_weights,
        globs.gate_proj_weights,
        globs.down_proj_weights,

        globs.lm_head_norm_weights.data,
        globs.lm_head_weights.data,

        fourD_k_cache,
        fourD_v_cache,
        # rope
        globs.rope_cos,
        globs.rope_sin,
        # activations
        globs.hidden_states,
        globs.rms_rope_intermediates,
        globs.rms_gate_intermediates,
        
        globs.post_ln_rope_q,
        globs.attn_out,
        globs.silu_out,

        globs.rms_lm_head_intermediates,
        globs.logits,
        globs.pos_id,
        globs.attn_scale,
        globs.rms_norm_eps,
        globs.batch_size,
    )

    # kvm_llama(
    #     Bar,
    #     instructions,
    #     timings,
    #     qkv_weights,
    #     attn_norm_weights,
    #     o_weights,
    #     mlp_norm_weights,
    #     up_weights,
    #     gate_weights,
    #     down_weights,
    #     lm_head_norm_weights,
    #     lm_head_weights,
    #     k_cache,
    #     v_cache,
    #     rope_cos,
    #     rope_sin,
    #     hidden_states,
    #     rms_rope_intermediates,
    #     rms_gate_intermediates,
    #     gate_silu_intermediates,
    #     q_post_rope,
    #     attn_out,
    #     silu_out,
    #     logits,
    #     pos_id,
    #     attn_scale,
    #     rms_norm_eps
    # )

    # kvm_func(
    #     # vm stuff
    #     globs.barriers,
    #     globs.instructions,
    #     globs.timings,
    #     # weights
    #     globs.qkv_proj,
    #     globs.attn_ln_weight,
    #     globs.o_proj,
    #     globs.mlp_ln_weight,
    #     globs.up_proj,
    #     globs.gate_proj,
    #     globs.down_proj,
    #     globs.lm_head_norm_weights.data,
    #     globs.lm_head_weights.data,
    #     globs.k_cache,
    #     globs.v_cache,
    #     # rope
    #     globs.rope_cos,
    #     globs.rope_sin,
    #     # activations
    #     globs.hidden_states,
    #     globs.post_ln_rope_q,
    #     globs.attn_out,
    #     globs.silu_out,
    #     globs.logits,
    #     # scalars
    #     globs.pos_id,
    #     globs.attn_scale,
    #     globs.rms_norm_eps,
    # )


class ThroughputKVM_Interpreter(KVM_Interpreter):
    def interpret(self, globs: Globals):
        interpret_with_kvm(globs, self.kvm_func)
