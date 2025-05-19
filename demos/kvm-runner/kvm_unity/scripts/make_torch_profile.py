import pydra
import torch
from kvm_unity.llama import LlamaForCausalLM
from kvm_unity.model_types import BatchState, ExtraModelConfig
from tqdm import tqdm


class ScriptConfig(pydra.Config):
    outfile: str = "proj.json"
    model: str = "meta-llama/Llama-3.2-1B-Instruct"
    device: str = "cuda:0"
    compile: bool = True
    prompt: str = "Hello, world!"
    fullgraph: bool = True
    dynamic: bool = False
    bs: int = 1
    seq_len: int = 1
    num_warmup: int = 3
    num_iters: int = 10
    num_profile_repeat: int = 3
    prof: bool = True


def main(config: ScriptConfig):
    model = LlamaForCausalLM.from_pretrained(
        config.model,
        device=config.device,
        extra_config=ExtraModelConfig(
            torch_compile=config.compile,
            max_batch_size=config.bs,
            max_len_override=config.seq_len,
        ),
    )

    if config.compile:
        print("Compiling model...")
        model = torch.compile(model, fullgraph=config.fullgraph, dynamic=config.dynamic)

    input_ids = torch.ones(
        config.bs, config.seq_len, dtype=torch.long, device=config.device
    )

    position_ids = (
        torch.arange(config.seq_len, device=config.device)
        .unsqueeze(0)
        .expand(config.bs, -1)
    )
    batch_state = BatchState(input_ids=input_ids, position_ids=position_ids)

    num_iters = config.num_profile_repeat * (config.num_iters + config.num_warmup + 1)

    if not config.prof:
        for _ in tqdm(range(num_iters)):
            model(batch_state)

        return

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=config.num_warmup,
            active=config.num_iters,
            repeat=config.num_profile_repeat,
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for _ in tqdm(
            range(
                config.num_profile_repeat * (config.num_iters + config.num_warmup + 1),
            ),
        ):
            model(batch_state)
            prof.step()

    prof.export_chrome_trace(config.outfile)


if __name__ == "__main__":
    pydra.run(main)
