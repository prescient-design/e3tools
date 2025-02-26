import torch
from torch import Tensor


def broadcast(src: Tensor, other: Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src


def scatter(src, index, dim, dim_size: int | None = None, reduce="sum"):
    in_shape = src.shape

    if dim < 0:
        dim = src.dim() + dim

    if dim_size is None:
        if index.numel() == 0:
            dim_size = 0
        else:
            dim_size = int(index.max()) + 1

    index = broadcast(index, src, dim)

    assert src.ndim == index.ndim, f"{src.ndim=}, {index.ndim=}"

    out_shape = (*in_shape[:dim], dim_size, *in_shape[dim + 1 :])
    out = torch.zeros(*out_shape, dtype=src.dtype, device=src.device)

    assert out.ndim == index.ndim, (
        f"{out.ndim=}, {index.ndim=} {out_shape=}, {in_shape=}, {dim=}"
    )
    return torch.scatter_reduce(out, dim, index, src, reduce, include_self=False)
