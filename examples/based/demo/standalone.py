
# Step 0. Allocate KV cache in allocate_inference_cache()
return torch.empty(batch_size,self.window_size,2,self.num_heads_kv,self.head_dim,dtype=dtype,device=device,)



# Step 1. Init the kv cache in _update_kv_cache()
assert window_size is not None, print("Implementing this for SWA.")
 cache_size = min(inference_params.max_seqlen, window_size + 1) 
# Pre-allocate memory for key-values for inference.
num_heads, head_dim = kv.shape[-2:]
if layer_idx not in inference_params.key_value_memory_dict:
    kv_cache = torch.empty(
        inference_params.max_batch_size,
        cache_size,    # SA: Change 1
        2,
        num_heads,
        head_dim,
        dtype=kv.dtype,
        device=kv.device,
    )
    inference_params.key_value_memory_dict[layer_idx] = kv_cache
else:
    kv_cache = inference_params.key_value_memory_dict[layer_idx]

# Adjust key and value for inference
batch_start = inference_params.batch_size_offset
batch_end = batch_start + kv.shape[0]
sequence_start = inference_params.seqlen_offset
sequence_end = sequence_start + kv.shape[1]

assert batch_end <= (kv_cache.shape[0] if kv_cache is not None else kv_cache.shape[0])
# assert sequence_end <= (kv_cache.shape[1] if kv_cache is not None else kv_cache.shape[2]) # SA: Change 2
assert kv_cache is not None
kv_cache[batch_start:batch_end, 0:cache_size, ...] = kv[:, -cache_size:,:,:,:] # SA: Change 3]
return kv_cache[batch_start:batch_end, :sequence_end, ...], cache_size, kv


# Step 2. Perform prefill.
kv_cache, cur_kv = self._update_kv_cache(kv, inference_params)
return self.inner_cross_attn(q, cur_kv) # applies to the full prefill, whereas kv_cache just has window_size entries

# Step 3. Decoding.
context = flash_attn_with_kvcache(
    q,
    kv_cache[:, :, 0].to(q.dtype),  # has size 65 if self.window is 64
    kv_cache[:, :, 1].to(q.dtype),
    kv[:, :, 0],
    kv[:, :, 1],
    rotary_cos=rotary_cos[(inference_params.seqlen_offset-self.cache_size):inference_params.seqlen_offset], # pick the 65 rotary_cos for this window
    rotary_sin=rotary_sin[(inference_params.seqlen_offset-self.cache_size):inference_params.seqlen_offset],
    cache_seqlens=self.cache_size,
    softmax_scale=self.inner_cross_attn.softmax_scale,
    causal=self.inner_cross_attn.causal,
    rotary_interleaved=self.rotary_emb.interleaved if self.rotary_emb_dim > 0 else False,
    window_size=self.window,
).to(in_dtype)

# Step 3.5 adjust KV cache for circular buffer.
kv_cache = torch.roll(kv_cache, -1, dims=1) # make space at the end for the next KV cache entry



