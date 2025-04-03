import torch
from torch import Tensor


@torch.compile
def densify_w_padding(x: Tensor, pad_token_id: int) -> Tensor:
    """Converts sparse one-hot tensors to token ids with padding."""
    x = x.coalesce()
    x_dense = torch.zeros(x.shape, dtype=torch.int64, device=x.device)
    x_dense[:, :, pad_token_id] = 1
    nz_idx = x.indices().detach().clone()
    nz_idx[-1] = pad_token_id
    x_dense[nz_idx.tolist()] = 0
    x_dense[x.indices().tolist()] = x.values().long()
    x_tok_ids = x_dense.argmax(dim=-1)

    return x_tok_ids
