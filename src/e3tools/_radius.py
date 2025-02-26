import math

import torch
from torch import Tensor

from jaxtyping import Float, Int64


# ref https://github.com/rusty1s/pytorch_cluster/blob/master/torch_cluster/radius.py
def radius(
    x: Float[Tensor, "N D"],
    y: Float[Tensor, "M D"],
    r: float,
    batch_x: Int64[Tensor, " N"] | None = None,
    batch_y: Int64[Tensor, " M"] | None = None,
    ignore_same_index: bool = True,
    chunk_size: int | None = None,
) -> Int64[Tensor, "2 E"]:
    """For each element in `y` find all points in `x` within distance `r`"""
    N, _ = x.shape
    M, _ = y.shape

    if chunk_size is None:
        chunk_size = N + 1

    if batch_x is None:
        batch_x = torch.zeros(N, dtype=torch.int64, device=x.device)

    if batch_y is None:
        batch_y = torch.zeros(N, dtype=torch.int64, device=x.device)

    if x.numel() == 0 or y.numel() == 0:
        return torch.empty(2, 0, dtype=torch.long, device=x.device)

    x = x.view(-1, 1) if x.dim() == 1 else x
    y = y.view(-1, 1) if y.dim() == 1 else y
    x, y = x.contiguous(), y.contiguous()

    batch_size = int(batch_x.max()) + 1
    batch_size = max(batch_size, int(batch_y.max()) + 1)
    assert batch_size > 0

    r2 = torch.as_tensor(r * r, dtype=x.dtype, device=x.device)

    n_chunks = math.ceil(N / chunk_size)

    rows = []
    cols = []

    for y_chunk, batch_y_chunk, index_y_chunk in zip(
        torch.chunk(y, n_chunks),
        torch.chunk(batch_y, n_chunks),
        torch.chunk(torch.arange(M, device=x.device), n_chunks),
    ):
        pdist = (x[:, None] - y_chunk).pow(2).sum(dim=-1)
        same_batch = batch_x[:, None] == batch_y_chunk
        same_index = torch.arange(N, device=x.device)[:, None] == index_y_chunk

        connected = (pdist <= r2) & same_batch
        if ignore_same_index:
            connected = connected & ~same_index

        row, col = torch.nonzero(connected, as_tuple=True)
        cols.append(col + index_y_chunk[0])
        rows.append(row)

    row = torch.cat(rows, dim=0)
    col = torch.cat(cols, dim=0)

    return torch.stack((col, row), dim=0)


def radius_graph(
    x: Float[Tensor, "N D"],
    r: float,
    batch: Int64[Tensor, " N"] | None = None,
    chunk_size: int | None = None,
) -> Int64[Tensor, "2 E"]:
    return radius(x, x, r, batch, batch, ignore_same_index=True, chunk_size=chunk_size)
