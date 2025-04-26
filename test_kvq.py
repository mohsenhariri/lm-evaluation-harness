import torch
from kvq import KVQ, KVQCacheConfig

config = KVQCacheConfig(
    nbits_k=4,
    nbits_v=2,
    axis_key=0,
    axis_value=0,
    q_group_size=64,
    residual_length=128,
    compute_dtype=torch.bfloat16,
    backend="quanto",
    # device=model.device,
)
kvq = KVQ(config)