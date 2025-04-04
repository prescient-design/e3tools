import e3nn
import e3nn.o3
import torch
from torch import nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
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
        self.irreps_in = e3nn.o3.Irreps(irreps)
        self.irreps_out = e3nn.o3.Irreps(irreps)
        self.eps = eps
        self.irreps_in_dim = self.irreps_in.dim

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
        self.scalar_masks = [(l == 0 and p == 1) for l, p in zip(self.ls, self.ps)]  # noqa: E741
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
        assert x.shape[-1] == self.irreps_in_dim, (
            f"Last dimension of x (shape {x.shape}) doesn't match irreps_in.dim "
            f"({self.irreps_in} with dim {self.irreps_in_dim})"
        )

        # Get batch dimensions (everything except the last dim)
        batch_dims = x.shape[:-1]

        # Process each irrep and collect outputs
        output_fields = []

        for i in range(len(self.start_indices)):
            start_idx = self.start_indices[i]
            size = self.sizes[i]
            mul = self.muls[i]
            dim = self.dims[i]
            is_scalar = self.scalar_masks[i]

            # Extract the field for this irrep
            field = x.narrow(-1, start_idx, size)

            # Reshape to [*batch_dims, mul, dim]
            # Using view instead of reshape for better traceability
            field_view = field.view(*batch_dims, mul, dim)

            if is_scalar:
                # For scalar irreps (l=0, p=1), use standard layer norm
                field_norm = F.layer_norm(field_view, [mul, dim], None, None, self.eps)
                # Flatten back for concatenation
                field_out = field_norm.reshape(*batch_dims, size)
            else:
                # For non-scalar irreps, normalize by the L2 norm
                # Compute squared L2 norm along the last dimension
                norm2 = torch.sum(field_view.pow(2), dim=-1)  # [*batch_dims, mul]

                # Compute RMS of the norm across multiplicity
                mean_norm2 = torch.mean(norm2, dim=-1, keepdim=True)  # [*batch_dims, 1]
                field_norm = torch.rsqrt(mean_norm2 + self.eps)  # [*batch_dims, 1]

                # Add an extra dimension for broadcasting
                field_norm = field_norm.unsqueeze(-1)  # [*batch_dims, 1, 1]

                # Apply normalization
                field_norm = field_view * field_norm

                # Flatten back for concatenation
                field_out = field_norm.reshape(*batch_dims, size)

            output_fields.append(field_out)

        # Concatenate all fields
        return torch.cat(output_fields, dim=-1)
