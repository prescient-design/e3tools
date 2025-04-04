from typing import Iterator, Tuple

import e3nn
import e3nn.o3
import torch
from torch import nn

from e3tools import pack_irreps, unpack_irreps


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


class MulToAxis(nn.Module):
    """Adds a new axis by factoring out irreps. Compatible with torch.compile."""

    def __init__(self, irreps_in: e3nn.o3.Irreps, factor: int):
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_out = e3nn.o3.Irreps([(mul // factor, ir) for mul, ir in irreps_in])
        self.factor = factor

        # Pre-compute indices and shapes for reshaping operations
        # This avoids dynamic operations during forward pass
        self._setup_indices_and_shapes()

    def _setup_indices_and_shapes(self):
        """Pre-compute indices and shapes for reshaping operations."""
        irreps = self.irreps_in
        factor = self.factor

        # Store the start indices for each irrep
        self.start_indices = []
        # Store the sizes (mul * ir.dim) for each irrep
        self.sizes = []
        # Store the output shapes for each irrep
        self.out_shapes = []

        idx = 0
        for mul, ir in irreps:
            if mul % factor != 0:
                raise ValueError(
                    f"irrep multiplicity {mul} is not divisible by factor {factor}"
                )

            size = mul * ir.dim
            self.start_indices.append(idx)
            self.sizes.append(size)

            # New shape after factoring
            new_mul = mul // factor
            self.out_shapes.append((factor, new_mul, ir.dim))

            idx += size

        # Register these as buffers so they're saved/loaded with the model
        self.register_buffer(
            "_start_indices", torch.tensor(self.start_indices, dtype=torch.long)
        )
        self.register_buffer("_sizes", torch.tensor(self.sizes, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds a new axis by factoring out irreps.

        Parameters:
            x: torch.Tensor of shape [..., irreps_in.dim]

        Returns:
            torch.Tensor of shape [..., factor, irreps_out.dim]
        """

        # Get the batch dimensions (everything except the last dim)
        batch_dims = x.shape[:-1]
        factor = self.factor

        # Create output tensor
        output_list = []

        # Process each irrep separately and concatenate at the end
        for i, (start_idx, size, out_shape) in enumerate(
            zip(self.start_indices, self.sizes, self.out_shapes)
        ):
            # Extract the field
            field = x.narrow(-1, start_idx, size)

            # Reshape to factor out the multiplicity
            field_reshaped = field.reshape(*batch_dims, *out_shape)

            # Reshape to match the expected output format
            field_out = field_reshaped.reshape(
                *batch_dims, factor, out_shape[1] * out_shape[2]
            )

            output_list.append(field_out)

        # Concatenate along the last dimension
        return torch.cat(output_list, dim=-1)


class AxisToMul(nn.Module):
    """Collapses the second-last axis by flattening the irreps. Compatible with torch.compile."""

    def __init__(self, irreps_in: e3nn.o3.Irreps, factor: int):
        super().__init__()
        self.irreps_in = e3nn.o3.Irreps(irreps_in)
        self.irreps_out = e3nn.o3.Irreps([(mul * factor, ir) for mul, ir in irreps_in])
        self.factor = factor
        self.irreps_in_dim = self.irreps_in.dim

        # Pre-compute indices and shapes for reshaping operations
        self._setup_indices_and_shapes()

    def _setup_indices_and_shapes(self):
        """Pre-compute indices and shapes for reshaping operations."""
        irreps = self.irreps_in
        factor = self.factor

        # Store the start indices for each irrep
        self.start_indices = []
        # Store the sizes (mul * ir.dim) for each irrep
        self.sizes = []
        # Store intermediate shapes for each irrep
        self.int_shapes = []
        # Store output shapes for each irrep
        self.out_shapes = []

        idx = 0
        for mul, ir in irreps:
            size = mul * ir.dim
            self.start_indices.append(idx)
            self.sizes.append(size)

            # Intermediate shape (factored)
            self.int_shapes.append((mul, ir.dim))

            # Output shape (after collapsing the axis)
            new_mul = mul * factor
            self.out_shapes.append((new_mul, ir.dim))

            idx += size

        # Register these as buffers so they're saved/loaded with the model
        self.register_buffer(
            "_start_indices", torch.tensor(self.start_indices, dtype=torch.long)
        )
        self.register_buffer("_sizes", torch.tensor(self.sizes, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Collapses the second-last axis by flattening the irreps.

        Parameters:
            x: torch.Tensor of shape [..., factor, irreps_in.dim]

        Returns:
            torch.Tensor of shape [..., irreps_out.dim]
        """
        # Check input shape
        assert x.shape[-1] == self.irreps_in_dim, (
            f"Last dimension of x (shape {x.shape}) doesn't match irreps_in.dim "
            f"({self.irreps_in} with dim {self.irreps_in_dim})"
        )
        assert x.shape[-2] == self.factor, (
            f"Second-last dimension of x (shape {x.shape}) doesn't match factor {self.factor}"
        )

        # Get the batch dimensions (everything except the last two dims)
        batch_dims = x.shape[:-2]
        factor = self.factor

        # Create output tensor
        output_list = []

        # Process each irrep separately and concatenate at the end
        for i, (start_idx, size, int_shape, out_shape) in enumerate(
            zip(self.start_indices, self.sizes, self.int_shapes, self.out_shapes)
        ):
            # Extract the field
            field = x.narrow(-1, start_idx, size)

            # Reshape to intermediate format
            field_int = field.reshape(*batch_dims, factor, *int_shape)

            # Reshape to output format
            field_out = field_int.reshape(*batch_dims, out_shape[0], out_shape[1])

            # Flatten the dimensions for concatenation
            field_flat = field_out.reshape(*batch_dims, out_shape[0] * out_shape[1])

            output_list.append(field_flat)

        # Concatenate along the last dimension
        return torch.cat(output_list, dim=-1)
