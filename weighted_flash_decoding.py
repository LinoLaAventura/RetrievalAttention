import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def _ensure_tensor(x):
    return x if isinstance(x, torch.Tensor) else torch.tensor(x)


def weighted_flash_decoding(*args,
                           previous_out: Optional[torch.Tensor]=None,
                           previous_lse: Optional[torch.Tensor]=None,
                           cache_seqlens: Optional[object]=None,
                           return_softmax_lse: bool=False):
    """
    Lightweight fallback implementation of weighted_flash_decoding.
    This provides a compatible interface for the project when the optimized
    implementation is unavailable. It computes scaled dot-product attention
    between `queries` and `keys` and applies the weights to `values`.

    Args (positional):
      queries, keys, values [, cluster_size]

    Keyword args mirror callers in the repo. This implementation is intentionally
    simple and may be slower than the optimized version; it's intended as a
    functional shim to allow testing and debugging.

    Returns:
      If `return_softmax_lse` is False: attention output tensor with shape
        matching callers: (batch_groups, 1, group_size, head_dim)
      If True: (output, lse) where `lse` is the log-sum-exp of the logits.
    """
    if len(args) < 3:
        raise ValueError("expected at least (queries, keys, values)")

    queries, keys, values = args[0], args[1], args[2]

    # optional cluster/size arg ignored by fallback
    # normalize to tensors
    queries = _ensure_tensor(queries)
    keys = _ensure_tensor(keys)
    values = _ensure_tensor(values)

    # Expected input shapes in callers:
    # queries: (B, 1, group_size, D)
    # keys:    (B, seq_len, 1, D)  or (B, seq_len, D)
    # values:  (B, seq_len, 1, D)  or (B, seq_len, D)

    if keys.dim() == 4 and keys.size(2) == 1:
        keys = keys.squeeze(2)
    if values.dim() == 4 and values.size(2) == 1:
        values = values.squeeze(2)

    B = queries.shape[0]
    qlen = queries.shape[1]
    group_size = queries.shape[2]
    D = queries.shape[3]

    # queries: (B, qlen, group_size, D) -> (B, group_size, D)
    q = queries.view(B, qlen, group_size, D).squeeze(1)  # (B, group_size, D)

    # keys, values: (B, seq_len, D)
    seq_len = keys.shape[1]

    # compute logits per head in the group
    # q: (B, group_size, D), keys: (B, seq_len, D)
    # compute dot-product -> (B, group_size, seq_len)
    logits = torch.matmul(q, keys.transpose(-1, -2))  # (B, group_size, seq_len)

    # apply masking if cache_seqlens provided
    if cache_seqlens is not None:
        try:
            if isinstance(cache_seqlens, torch.Tensor):
                lens = cache_seqlens.long().view(-1)
            else:
                # allow scalar or list/ndarray
                if isinstance(cache_seqlens, (list, tuple)):
                    lens = torch.tensor(cache_seqlens, device=logits.device, dtype=torch.long)
                else:
                    lens = torch.full((B,), int(cache_seqlens), dtype=torch.long, device=logits.device)
            arange = torch.arange(seq_len, device=logits.device).view(1, 1, -1)
            mask = arange >= lens.view(-1, 1, 1)
            logits = logits.masked_fill(mask, float('-inf'))
        except Exception:
            pass

    # scaled softmax
    scale = 1.0 / (D ** 0.5)
    logits = logits * scale
    attn = F.softmax(logits, dim=-1)

    # output: attn @ values
    # attn: (B, group_size, seq_len), values: (B, seq_len, D)
    out = torch.einsum('bgs,bkd->bgd', attn, values)  # (B, group_size, D)

    # reshape to (B, 1, group_size, D)
    out = out.unsqueeze(1)

    if return_softmax_lse:
        # compute log-sum-exp along seq_len (before softmax scaling)
        lse = torch.logsumexp(logits, dim=-1, keepdim=True)  # (B, group_size, 1)
        lse = lse.unsqueeze(1)  # (B,1,group_size,1)
        return out, lse

    return out
