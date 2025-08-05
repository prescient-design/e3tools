import torch


def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int) -> torch.Tensor:
    """Broadcasts `src` to match `other`."""
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src


def scatter(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int,
    dim_size: int | None = None,
    reduce: str = "sum",
) -> torch.Tensor:
    """Collects elements at the indices `index` of a source tensor `src`."""
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


def scatter_softmax(
    src: torch.Tensor, index: torch.Tensor, dim: int, dim_size: int | None = None
):
    index = broadcast(index, src, dim)

    segment_max = scatter(src, index, dim, dim_size, reduce="amax")
    segment_max = torch.gather(input=segment_max, dim=dim, index=index)
    segment_max = torch.where(
        segment_max == float("-inf"), torch.zeros_like(segment_max), segment_max
    )

    scores = (src - segment_max).exp()
    z = scatter(scores, index, dim, dim_size, reduce="sum")
    z = torch.gather(input=z, dim=dim, index=index)
    z = torch.where(z == 0, torch.ones_like(z), z)
    scores = scores / z

    return scores
