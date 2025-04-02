import torch
from typing import Iterator, Tuple
import e3nn.o3


def unpack_irreps(
    x: torch.Tensor, irreps: e3nn.o3.Irreps
) -> Iterator[Tuple[int, e3nn.o3.Irrep, torch.Tensor]]:
    """
    Given a packed irreps vector of dimension [..., irreps.dim]
    yield tuples (mul, ir, field) where field has dimension [..., mul, 2*l+1]
    for each irrep in irreps
    """
    assert x.shape[-1] == irreps.dim, (
        f"last dimension of x (shape {x.shape}) does not match irreps.dim ({irreps} with dim {irreps.dim})"
    )
    ix = 0
    for mul, ir in irreps:
        field = x.narrow(-1, ix, mul * ir.dim).reshape(*x.shape[:-1], mul, ir.dim)
        ix += mul * ir.dim
        yield mul, ir, field

    assert ix == irreps.dim


def pack_irreps(
    unpacked_tuples: Iterator[Tuple[int, e3nn.o3.Irrep, torch.Tensor]],
) -> torch.Tensor:
    """Pack fields into a single tensor."""
    fields = []
    for mul, ir, field in unpacked_tuples:
        fields.append(field.reshape(*field.shape[:-2], mul * ir.dim))
    return torch.cat(fields, dim=-1)
