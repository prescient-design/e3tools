import torch
from torch import nn
import e3nn.o3


class Repeat(nn.Module):
    """Repeat the irreps along the last axis."""

    def __init__(self, irreps_in: e3nn.o3.Irreps, repeats: int):
        super().__init__()
        self.irreps_in = e3nn.o3.Irreps(irreps_in)
        self.repeats = repeats
        self.irreps_out = e3nn.o3.Irreps([(mul * repeats, ir) for mul, ir in irreps_in])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat the features along the last axis.

        If `x` has shape `[..., irreps_in.dim]`, the output will have shape
        `[..., repeats * irreps_in.dim]`.
        """
        return x.repeat_interleave(self.repeats, dim=-1)
