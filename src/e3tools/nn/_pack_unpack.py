from typing import Iterator, Tuple

import e3nn.o3
import torch


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


def factor_tuples(
    unpacked_tuples: Iterator[Tuple[int, e3nn.o3.Irrep, torch.Tensor]], factor: int
) -> Iterator[Tuple[int, e3nn.o3.Irrep, torch.Tensor]]:
    """Factor the fields in each tuple by a factor."""
    for mul, ir, field in unpacked_tuples:
        if mul % factor != 0:
            raise ValueError(
                f"irrep multiplicity {mul} is not divisible by factor {factor}"
            )
        new_mul = mul // factor
        new_field = field.reshape(*field.shape[:-2], factor, mul // factor, ir.dim)
        yield new_mul, ir, new_field


def undo_factor_tuples(
    factored_tuples: Iterator[Tuple[int, e3nn.o3.Irrep, torch.Tensor]], factor: int
) -> Iterator[Tuple[int, e3nn.o3.Irrep, torch.Tensor]]:
    """Undo the factorization of the fields in each tuple."""
    for mul, ir, field in factored_tuples:
        new_mul = mul * factor
        new_field = field.reshape(*field.shape[:-3], new_mul, ir.dim)
        yield new_mul, ir, new_field


def pack_irreps(
    unpacked_tuples: Iterator[Tuple[int, e3nn.o3.Irrep, torch.Tensor]],
) -> torch.Tensor:
    """Pack fields into a single tensor."""
    fields = []
    for mul, ir, field in unpacked_tuples:
        fields.append(field.reshape(*field.shape[:-2], mul * ir.dim))
    return torch.cat(fields, dim=-1)


def mul_to_axis(
    x: torch.Tensor, irreps: e3nn.o3.Irreps, *, factor: int
) -> Tuple[torch.Tensor, e3nn.o3.Irreps]:
    """Adds a new axis by factoring out irreps.

    If x has shape [..., irreps.dim], the output will have shape [..., factor, irreps.dim // factor].
    """
    x_factored = pack_irreps(factor_tuples(unpack_irreps(x, irreps), factor))
    irreps_factored = e3nn.o3.Irreps([(mul // factor, ir) for mul, ir in irreps])
    return x_factored, irreps_factored


def axis_to_mul(
    x: torch.Tensor, irreps: e3nn.o3.Irreps
) -> Tuple[torch.Tensor, e3nn.o3.Irreps]:
    """Collapses the second-last axis by flattening the irreps.

    If x has shape [..., factor, irreps.dim // factor], the output will have shape [..., irreps.dim].
    """
    factor = x.shape[-2]
    x_multiplied = pack_irreps(
        undo_factor_tuples(unpack_irreps(x, irreps), factor=factor)
    )
    irreps_multiplied = e3nn.o3.Irreps([(mul * factor, ir) for mul, ir in irreps])
    return x_multiplied, irreps_multiplied


class MulToAxis(torch.nn.Module):
    """Adds a new axis by factoring out irreps."""

    def __init__(self, irreps_in: e3nn.o3.Irreps, factor: int):
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_out = e3nn.o3.Irreps([(mul // factor, ir) for mul, ir in irreps_in])
        self.factor = factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds a new axis by factoring out irreps.

        Parameters:
            x: torch.Tensor of shape [..., irreps.dim]

        Returns:
            torch.Tensor of shape [..., factor, irreps.dim // factor]
        """

        return mul_to_axis(x, self.irreps_in, factor=self.factor)[0]


class AxisToMul(torch.nn.Module):
    """Collapses the second-last axis by flattening the irreps."""

    def __init__(self, irreps_in: e3nn.o3.Irreps, factor: int):
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_out = e3nn.o3.Irreps([(mul * factor, ir) for mul, ir in irreps_in])
        self.factor = factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Collapses the second-last axis by flattening the irreps.

        Parameters:
            x: torch.Tensor of shape [..., factor, irreps.dim // factor]

        Returns:
            torch.Tensor of shape [..., irreps.dim]
        """
        return axis_to_mul(x, self.irreps_in)[0]
