import torch
import torch.nn.functional as F

"""
Contains various sampling strategies for logits

"""


def top_p_sampling(logits, p=0.9):
    """Applies top-p (nucleus) sampling to logits."""
    # Sort logits in descending order and compute their probabilities
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probabilities = torch.nn.functional.softmax(sorted_logits, dim=-1)

    # Compute cumulative probabilities
    cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)

    # Remove tokens with a cumulative probability above the threshold p
    indices_to_remove = cumulative_probabilities > p
    # Shift the indices to the right to keep the first token above p
    indices_to_remove[..., 1:] = indices_to_remove[..., :-1].clone()
    indices_to_remove[..., 0] = False

    # Set the logits for the removed indices to negative infinity
    sorted_indices_to_remove = sorted_indices[indices_to_remove]
    logits[sorted_indices_to_remove] = float("-inf")

    return logits


# source: https://github.com/andreaskoepf/laion_idle_cap/blob/main/docker/sampling.py
def top_k_top_p_filtering_batch(logits, top_k=0, top_p=0.0, filter_value=float("-inf")):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (vocabulary size)
        top_k >0: keep only top k tokens with highest probability (top-k filtering).
        top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    batch_size = logits.size(0)
    num_logits = logits.size(-1)
    device = logits.device
    # print('top_k', type(top_k), top_k)
    if type(top_k) == float:
        if top_k > 0 and top_k < 1:
            top_k = max(1, int(top_k * num_logits))
        else:
            top_k = int(top_k)
    # Remove all tokens with a probability less than the last token of the top-k
    if type(top_k) == int:
        if top_k > 0:
            cutoff = torch.topk(logits, k=top_k, largest=True).values[:, -1:]
            indices_to_remove = logits < cutoff
            logits[indices_to_remove] = filter_value
    elif torch.any(top_k > 0):
        assert top_k.size(0) == batch_size
        top_k = top_k.clamp_max(num_logits)
        for i in range(batch_size):
            k = top_k[i]
            if k <= 0:
                continue
            if k < 1:
                k = max(1, int(k * num_logits))
            cutoff = torch.topk(logits[i], k=k, largest=True).values[-1]
            indices_to_remove = logits[i] < cutoff
            logits[i][indices_to_remove] = filter_value
    if type(top_p) == float and top_p > 0.0 or torch.any(top_p > 0):
        if type(top_p) == torch.Tensor and top_p.size(-1) != 1:
            top_p = top_p.unsqueeze(-1)
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False
        # convert sorted indices into flat indices
        row_starts = (
            torch.arange(sorted_indices.shape[0], device=device).unsqueeze(1)
            * sorted_indices.shape[1]
        )
        sorted_indices_flat = sorted_indices + row_starts
        indices_to_remove = sorted_indices_flat[sorted_indices_to_remove]
        logits = logits.contiguous()
        logits.view(-1)[indices_to_remove] = filter_value
    return logits


def greedy(logits, filter_value=float("-inf")):
    probabilities = F.softmax(logits, dim=-1)
    sampled_indices = torch.argmax(probabilities, dim=-1)
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask.scatter_(dim=-1, index=sampled_indices.unsqueeze(-1), value=False)
    logits.masked_fill_(mask, filter_value)
    return logits


def sample_from_logitsV2(gathered_logits, strategy="top-k", k=5, p=0.9):
    # should the logits be concatenated and then sampled or sampled and concatenated ???
    sampled_logits_list = []
    for logits in gathered_logits:
        sampled_logits = None
        if strategy == "greedy":
            sampled_logits = greedy(logits)
        elif strategy == "top-k" or strategy == "top-p":
            sampled_logits = top_k_top_p_filtering_batch(logits, top_k=k, top_p=p)
        sampled_logits_list.append(sampled_logits)

    # concatenated_logits = torch.cat(sampled_logits_list)
    return sampled_logits_list


def sample_from_logitsV1(next_token_logits, strategy="top-k", k=5, p=0.9):

    sampled_token = None

    if strategy == "greedy" or strategy == "top-k":
        probabilities = F.softmax(next_token_logits, dim=-1)
        if strategy == "greedy":
            # Greedy sampling: select the token with the highest probability at each step
            sampled_token = torch.argmax(probabilities, dim=-1)
            print(f"sample_indices shape : {sampled_token.shape}")
        elif strategy == "top-k":
            probabilities = F.softmax(next_token_logits, dim=-1)
            print(f"probabilities : {probabilities.shape}")
            topk_vals, topk_indices = torch.topk(probabilities, k=k, dim=-1)
            print(f"topk_vals: {topk_vals.shape}, topk_indices : {topk_indices.shape}")
            # Ensuring topk_vals is 2D: [batch_size, k]
            if topk_vals.dim() > 2:
                topk_vals = topk_vals.view(
                    -1, k
                )  # Reshape for safety, though it should already be [batch_size, k]
            print(f"topk_vals after: {topk_vals.shape}")

            # Sampling from the top-k values for each item in the batch
            # topk_vals is now guaranteed to be [batch_size, k], suitable for torch.multinomial
            sampled_from_topk = torch.multinomial(
                topk_vals, 1
            )  # [sequence, 1], samples one index per batch item

            # Gathering the actual token indices corresponding to the sampled positions
            # Use torch.gather or advanced indexing to map back to original token indices
            batch_size = topk_indices.size(0)
            batch_indices = (
                torch.arange(batch_size).unsqueeze(-1).to(topk_indices.device)
            )
            sampled_token = topk_indices[batch_indices, sampled_from_topk].squeeze(
                -1
            )  # Remove singleton dimension
    elif strategy == "top-p":
        # Apply top-p sampling to logits and then sample
        sampled_token = torch.empty(
            next_token_logits.size(0),
            # next_token_logits.size(1),
            dtype=torch.long,
            device=next_token_logits.device,
        )
        for i in range(next_token_logits.shape[0]):  # Iterate through sequence
            logits = next_token_logits[i, :]
            filtered_logits = top_p_sampling(logits, p=p)
            probs = F.softmax(filtered_logits, dim=-1)
            # Use torch.multinomial to sample from the filtered distribution
            next_token_samples = torch.multinomial(
                probs, 1
            )  # Sample 1 token per sequence
            sampled_token[i] = next_token_samples.squeeze(-1)

    return sampled_token
