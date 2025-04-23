import sys
from pathlib import Path

from kvm_runner.llama import BatchState, LlamaForCausalLM
from kvm_runner.scheduler import schedule_model
from torch import Tensor


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
            self.model,
            prompt_len=prompt_len,
            ntok=ntok,
        )

    def invoke(self):
        self.kvm_func(
            # vm stuff
            self.globals.barriers,
            self.globals.instructions,
            self.globals.timings,
            # weights
            self.globals.qkv_proj,
            self.globals.attn_ln_weight,
            self.globals.o_proj,
            self.globals.mlp_ln_weight,
            self.globals.up_proj,
            self.globals.gate_proj,
            self.globals.down_proj,
            # rope
            self.globals.rope_cos,
            self.globals.rope_sin,
            # activations
            self.globals.hidden_states,
            self.globals.post_ln_rope_q,
            self.globals.attn_out,
            self.globals.attn_lse_intermediates,
            self.globals.attn_out_intermediates,
            self.globals.silu_out,
            # scalars
            self.globals.pos_id,
            self.globals.attn_scale,
            self.globals.rms_norm_eps,
        )

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

        self.invoke()

        output_hiddens = self.globals.hidden_states

        post_embedding.hidden_states = output_hiddens

        post_lm_head: BatchState = self.model.lm_head(post_embedding)

        output_ids = post_lm_head.output_ids
        assert output_ids is not None
        return output_ids
