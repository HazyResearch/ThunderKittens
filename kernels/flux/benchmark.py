import os
import re
import time
from dataclasses import dataclass
from glob import iglob

import torch
from einops import rearrange
from fire import Fire
from PIL import ExifTags, Image

from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from flux.util import (load_flow_model)

NSFW_THRESHOLD = 0.85

@dataclass
class SamplingOptions:
    prompt: str
    width: int
    height: int
    num_steps: int
    guidance: float
    seed: int | None

@torch.inference_mode()
def main(
    name: str = "flux-dev",
    width: int = 1024,
    height: int = 768,
    seed: int | None = None,
    prompt: str = (
        "a photo of a forest with mist swirling around the tree trunks. The word "
        '"FLUX" is painted over it in big, red brush strokes with visible texture'
    ),
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_steps: int | None = None,
    loop: bool = False,
    guidance: float = 3.5,
    offload: bool = False,
    output_dir: str = "output",
    add_sampling_metadata: bool = True,

    optimized=False,
    use_tk=False,
):
    """
    Sample the flux model. Either interactively (set `--loop`) or run for a
    single image.
    """
    num_steps = 28
    height = 16 * (height // 16)
    width = 16 * (width // 16)
    print(f"{height=}, {width=}")
    rng = torch.Generator(device="cpu")
    opts = SamplingOptions(
        prompt=prompt,
        width=width,
        height=height,
        num_steps=num_steps,
        guidance=guidance,
        seed=0,
    )

    model = load_flow_model(name, device=torch.device(device), use_tk=use_tk)
    if optimized:
        print("Torch compiling model")
        model = torch.compile(model)

    # denoise initial noise
    high = 100
    in_channels, vec_in_dim, context_in_dim = 64, 768, 4096
    b, img_in_dim, txt_in_dim = 1, 3072, 512
    img = torch.randn(b, img_in_dim, in_channels, dtype=torch.bfloat16, device='cuda')
    txt = torch.randn(b, txt_in_dim, context_in_dim, dtype=torch.bfloat16, device='cuda')
    vec = torch.randn(b, vec_in_dim, dtype=torch.bfloat16, device='cuda')
    img_ids = torch.randint(0, high, size=(b, img_in_dim, 3), device='cuda')
    txt_ids = torch.randint(0, high, size=(b, txt_in_dim, 3), device='cuda')

    inp = {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
    }

    timesteps = get_schedule(opts.num_steps, inp["img"].shape[1], shift=(name != "flux-schnell"))

    n_warmup_iters = 2
    print(f"Warmup iters:")
    for i in range(n_warmup_iters): 
        t0 = time.perf_counter()
        x = denoise(model, **inp, timesteps=timesteps, guidance=opts.guidance)
        t1 = time.perf_counter()
        print(f"Warmup done in {t1 - t0:.1f}s")

    print(f"Real iters:")
    n_iters = 3
    for i in range(n_iters):
        t0 = time.perf_counter()
        x = denoise(model, **inp, timesteps=timesteps, guidance=opts.guidance)
        t1 = time.perf_counter()
        print(f"Done in {t1 - t0:.1f}s")

if __name__ == "__main__":
    main()

