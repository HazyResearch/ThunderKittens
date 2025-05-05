import json
from pathlib import Path
from typing import List

import torch
from kvm_batch_runner.model_types import DeviceType
from safetensors import safe_open
from torch import Tensor
from tqdm import tqdm


def compute_shard_bounds(
    tensor_shape: List[int], dim: int, num_shards: int, shard_index: int
):
    dim_size = tensor_shape[dim]
    base_shard_size = dim_size // num_shards
    remainder = dim_size % num_shards

    start_idx = shard_index * base_shard_size + min(shard_index, remainder)

    if shard_index < remainder:
        end_idx = start_idx + base_shard_size + 1
    else:
        end_idx = start_idx + base_shard_size

    return slice(start_idx, end_idx)


def load_safetensors_repo(
    repo_path: Path,
    include_parameters: set[str],
    device: DeviceType,
    tp_rank: int = 0,
    tp_size: int = 1,
    tp_map: dict[str, int] | None = None,
):
    if tp_map is None:
        tp_map = {}

    single_file = repo_path / "model.safetensors"
    if single_file.exists():
        files_to_load = [single_file]

    else:
        safetensors_index = repo_path / "model.safetensors.index.json"

        if not safetensors_index.exists():
            raise FileNotFoundError(
                f"Could not find model.safetensors or model.safetensors.index.json in {repo_path}"
            )

        with open(safetensors_index, "r") as f:
            index = json.load(f)

        param_to_path = index["weight_map"]

        files_to_load_set = set()

        for param_name, path in param_to_path.items():
            if param_name in include_parameters:
                files_to_load_set.add(repo_path / path)

        files_to_load = list(sorted(files_to_load_set))

    state_dict = {}

    for file in tqdm(
        files_to_load,
        desc="Loading safetensors files",
    ):
        with safe_open(file, framework="pt", device=device) as f:
            for k in f.keys():
                if k in include_parameters:
                    if tp_size > 1 and (split_dim := tp_map.get(k)) is not None:
                        tensor_slice = f.get_slice(k)
                        shard_bounds = compute_shard_bounds(
                            tensor_slice.get_shape(), split_dim, tp_size, tp_rank
                        )
                        # TODO: there's gotta be a better way to do this
                        match split_dim:
                            case 0:
                                state_dict[k] = tensor_slice[shard_bounds]
                            case 1:
                                state_dict[k] = tensor_slice[:, shard_bounds]
                            case _:
                                raise ValueError(
                                    f"Unsupported split dimension: {split_dim}"
                                )
                    else:
                        state_dict[k] = f.get_tensor(k)

    return state_dict


def trepr(t: Tensor):
    """
    Tensor representation.
    """
    return f"shape={t.shape}, sum={t.sum()}, vals={t}"


def get_sm_count(device: str) -> int:
    device_props = torch.cuda.get_device_properties(device)
    return device_props.multi_processor_count
