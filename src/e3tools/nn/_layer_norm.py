import e3nn
import torch
import torch.nn.functional as F
from e3nn import o3

from ._pack_unpack import unpack_irreps


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
