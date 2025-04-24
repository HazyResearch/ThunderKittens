import sys
from pathlib import Path

from kvm_runner.llama import BatchState, LlamaForCausalLM
from kvm_runner.scheduler import Globals, schedule_model, tensorize_instructions
import torch
from torch import Tensor


def get_kvm_func(kvm_dir: Path):
    sys.path.append(str(kvm_dir.expanduser().absolute()))
    from kvm_llama import kvm_llama  # type: ignore

    return kvm_llama


def interpret_with_kvm(
    globs: Globals,
    kvm_func,
):
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
        globs.k_cache,
        globs.v_cache,
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
        # scalars
        globs.pos_id,
        globs.attn_scale,
        globs.rms_norm_eps,
    )
    torch.cuda.synchronize()


class KVM_Runner:
    def __init__(
        self,
        model: LlamaForCausalLM,
        kvm_dir: Path,
        prompt_len: int,
        ntok: int,
    ):
        sys.path.append(str(kvm_dir.expanduser().absolute()))
        from kvm_llama import kvm_llama  # type: ignore

        self.kvm_func = kvm_llama

        self.model = model

        self.globals, self.instructions = schedule_model(
            model=self.model,
            prompt_len=prompt_len,
            ntok=ntok,
        )
        tensorize_instructions(self.globals, self.instructions)

    def run(self, input_ids: Tensor, pos_id: int):
        batch_state = BatchState(
            input_ids=input_ids,
        )

        post_embedding: BatchState = self.model.model.embed_tokens(batch_state)
        hiddens = post_embedding.hidden_states
        assert hiddens is not None
        self.globals.hidden_states[:] = hiddens
        self.globals.pos_id = pos_id

        self.globals.barriers.zero_()

        interpret_with_kvm(self.globals, self.kvm_func)

        output_hiddens = self.globals.hidden_states

        post_embedding.hidden_states = output_hiddens

        post_lm_head: BatchState = self.model.lm_head(post_embedding)

        output_ids = post_lm_head.output_ids
        assert output_ids is not None
        return output_ids
