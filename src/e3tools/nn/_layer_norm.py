import e3nn
import torch
import torch.nn.functional as F
from e3nn import o3

from ._pack_unpack import unpack_irreps


class LayerNormCompiled(torch.nn.Module):
    """
    Equivariant layer normalization compatible with torch.compile.

    Each irrep is normalized independently.

    ref: https://github.com/atomicarchitects/equiformer/blob/master/nets/fast_layer_norm.py
    """

    def __init__(self, irreps: e3nn.o3.Irreps, eps: float = 1e-6):
        """
        Parameters
        ----------
        irreps: e3nn.o3.Irreps
            Input/output irreps
        eps: float = 1e-6
            softening factor
        """
        super().__init__()
        self.irreps_in = o3.Irreps(irreps)
        self.irreps_out = o3.Irreps(irreps)
        self.eps = eps

        # Pre-compute indices and shapes for reshaping operations
        self._setup_indices_and_shapes()

    def _setup_indices_and_shapes(self):
        """Pre-compute indices and shapes for reshaping operations."""
        irreps = self.irreps_in

        # Lists to store information about each irrep
        self.start_indices = []  # Start index in the flattened tensor
        self.sizes = []  # Size (mul * ir.dim) for each irrep
        self.muls = []  # Multiplicity of each irrep
        self.dims = []  # Dimension (2*l+1) of each irrep
        self.ls = []  # Angular momentum (l) of each irrep
        self.ps = []  # Parity (p) of each irrep

        idx = 0
        for mul, ir in irreps:
            size = mul * ir.dim

            self.start_indices.append(idx)
            self.sizes.append(size)
            self.muls.append(mul)
            self.dims.append(ir.dim)
            self.ls.append(ir.l)
            self.ps.append(ir.p)

            idx += size

        # Register these as buffers so they're saved/loaded with the model
        # Use long tensors for indices, booleans for flags
        self.register_buffer(
            "_start_indices", torch.tensor(self.start_indices, dtype=torch.long)
        )
        self.register_buffer("_sizes", torch.tensor(self.sizes, dtype=torch.long))
        self.register_buffer("_muls", torch.tensor(self.muls, dtype=torch.long))
        self.register_buffer("_dims", torch.tensor(self.dims, dtype=torch.long))
        self.register_buffer("_ls", torch.tensor(self.ls, dtype=torch.long))
        self.register_buffer("_ps", torch.tensor(self.ps, dtype=torch.long))

        # Create a mask for scalar (l=0, p=1) irreps for faster processing
        self.scalar_masks = [(l == 0 and p == 1) for l, p in zip(self.ls, self.ps)]
        self.register_buffer(
            "_scalar_masks", torch.tensor(self.scalar_masks, dtype=torch.bool)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization to input tensor.
        Each irrep is normalized independently.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape [..., self.irreps_in.dim]

        Returns
        -------
        torch.Tensor
            Normalized tensor of shape [..., self.irreps_out.dim]
        """
        # Check input shape
        assert x.shape[-1] == self.irreps_in.dim, (
            f"Last dimension of x (shape {x.shape}) doesn't match irreps_in.dim "
            f"({self.irreps_in} with dim {self.irreps_in.dim})"
        )

        # Get batch dimensions (everything except the last dim)
        batch_shape = x.shape[:-1]

        # Process each irrep and collect outputs
        output_fields = []

        for i, (start_idx, size, mul, dim, is_scalar) in enumerate(
            zip(self.start_indices, self.sizes, self.muls, self.dims, self.scalar_masks)
        ):
            # Extract the field for this irrep
            field = x.narrow(-1, start_idx, size)

            # Reshape to [..., mul, dim]
            field = field.reshape(*batch_shape, mul, dim)

            if False or is_scalar:
                # For scalar irreps (l=0, p=1), use standard layer norm
                # F.layer_norm expects normalized_shape as a list of dimensions to normalize over
                field = F.layer_norm(field, [mul, 1], None, None, self.eps)
                output_fields.append(field.reshape(*batch_shape, size))
            else:
                # For non-scalar irreps, normalize by the L2 norm
                # Compute squared L2 norm along the last dimension
                norm2 = field.pow(2).sum(-1)  # [..., mul]

                # Compute RMS of the norm across multiplicity
                field_norm = (norm2.mean(dim=-1) + self.eps).pow(-0.5)  # [...]

                # Reshape for broadcasting
                field_norm = field_norm.reshape(*batch_shape, 1, 1)

                # Apply normalization
                field = field * field_norm

                # Reshape back to original format
                output_fields.append(field.reshape(*batch_shape, size))

        # Concatenate all fields
        return torch.cat(output_fields, dim=-1)


class LayerNorm(torch.nn.Module):
    """
    Equivariant layer normalization.

    ref: https://github.com/atomicarchitects/equiformer/blob/master/nets/fast_layer_norm.py
    """

    def __init__(self, irreps: e3nn.o3.Irreps, eps: float = 1e-6):
        """
        Parameters
        ----------
        irreps: e3nn.o3.Irreps
            Input/output irreps
        eps: float = 1e-6
            softening factor
        """
        super().__init__()
        self.irreps_in = o3.Irreps(irreps)
        self.irreps_out = o3.Irreps(irreps)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization to input tensor.
        Each irrep is normalized independently.
        """
        # x: [..., self.irreps.dim]
        fields = []
        for mul, ir, field in unpack_irreps(x, self.irreps_in):
            # field: [..., mul, 2*l+1]
            if ir.l == 0 and ir.p == 1:
                field = F.layer_norm(field, (mul, 1), None, None, self.eps)
                fields.append(field.reshape(-1, mul))
                continue

            norm2 = field.pow(2).sum(-1)  # [..., mul] (squared L2 norm of l-reprs)
            field_norm = (norm2.mean(dim=-1) + self.eps).pow(
                -0.5
            )  # [...] (1/RMS(norm))
            field = field * field_norm.reshape(-1, 1, 1)
            fields.append(field.reshape(-1, mul * ir.dim))

        output = torch.cat(fields, dim=-1)
        return output
