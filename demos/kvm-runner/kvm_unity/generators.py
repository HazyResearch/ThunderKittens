import torch
from torch import Tensor
from tqdm import tqdm

from kvm_unity.kvm import KVM_Interpreter
from kvm_unity.llama import LlamaForCausalLM
from kvm_unity.model_types import BatchState
from kvm_unity.python_vm import PyVM_Interpreter
from kvm_unity.scheduler import Schedule


class Generator:
    def generate(self, output_tokens: Tensor, prompt_len: int, ntok: int):
        raise NotImplementedError


class PyTorchGenerator(Generator):
    def __init__(
        self,
        model: LlamaForCausalLM,
    ):
        self.model = model

    def generate(self, output_tokens: Tensor, prompt_len: int, ntok: int):
        start_position_ids = torch.ones(
            1, dtype=torch.long, device=self.model.device
        ) * (prompt_len)

        for i in tqdm(range(1, ntok)):
            position_ids = start_position_ids + i
            decode_inp = BatchState(
                input_ids=output_tokens[i - 1 : i],
                position_ids=position_ids,
                seq_len=prompt_len + i + 1,
            )
            decode_output: BatchState = self.model(decode_inp)
            assert decode_output.output_ids is not None
            output_tokens[i] = decode_output.output_ids


class KVM_Generator(Generator):
    def __init__(
        self,
        model: LlamaForCausalLM,
        interpreter: KVM_Interpreter,
        schedule: Schedule,
        barrier_fill_val: int = 0,
        skip_kvm: bool = False,
        skip_rest: bool = False,
    ):
        self.model = model
        self.interpreter = interpreter
        self.schedule = schedule

        self.barrier_fill_val = barrier_fill_val
        self.skip_kvm = skip_kvm
        self.skip_rest = skip_rest

        self.fill()

    def fill(self):
        self.schedule.globs.barriers.fill_(self.barrier_fill_val)

    def replace_with_noops(self):
        self.schedule.globs.instructions.zero_()

    def run(self, input_ids: Tensor, pos_id: int):
        if not self.skip_rest:
            batch_state = BatchState(
                input_ids=input_ids,
            )

            post_embedding: BatchState = self.model.model.embed_tokens(batch_state)
            hiddens = post_embedding.hidden_states
            assert hiddens is not None
            self.schedule.globs.hidden_states[:] = hiddens

        self.fill()
        self.schedule.globs.pos_id = pos_id
        if not self.skip_kvm:
            self.interpreter.interpret(self.schedule.globs)

        if self.skip_rest:
            return input_ids

        logits = self.schedule.globs.logits
        output_ids = torch.argmax(logits, dim=-1)

        return output_ids

    def generate(self, output_tokens: Tensor, prompt_len: int, ntok: int):
        for i in tqdm(range(1, ntok)):
            input_ids = output_tokens[i - 1 : i]
            output_ids = self.run(input_ids, pos_id=prompt_len + i)
            output_tokens[i] = output_ids


class PyVM_Generator(Generator):
    def __init__(
        self,
        model: LlamaForCausalLM,
        interpreter: PyVM_Interpreter,
        schedule: Schedule,
    ):
        self.model = model
        self.interpreter = interpreter
        self.schedule = schedule

        self.instructions = self.schedule.get_linear_instructions()

    def run(self, input_ids: Tensor, pos_id: int):
        batch_state = BatchState(
            input_ids=input_ids,
        )

        post_embedding: BatchState = self.model.model.embed_tokens(batch_state)
        hiddens = post_embedding.hidden_states
        assert hiddens is not None
        self.schedule.globs.hidden_states[:] = hiddens
        self.schedule.globs.barriers.zero_()
        self.schedule.globs.pos_id = pos_id

        self.interpreter.interpret(self.schedule.globs, self.instructions)

        output_hiddens = self.schedule.globs.hidden_states

        post_embedding.hidden_states = output_hiddens

        post_lm_head: BatchState = self.model.lm_head(post_embedding)

        output_ids = post_lm_head.output_ids
        assert output_ids is not None
        return output_ids

    def generate(self, output_tokens: Tensor, prompt_len: int, ntok: int):
        for i in tqdm(range(1, ntok)):
            input_ids = output_tokens[i - 1 : i]
            output_ids = self.run(input_ids, pos_id=prompt_len + i)
            output_tokens[i] = output_ids
