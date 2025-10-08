from typing import Union

import e3nn
import e3nn.o3
import torch
from torch import nn

from ._linear import Linear


class DepthwiseTensorProduct(nn.Module):
    """
    Depthwise tensor product

    ref: https://arxiv.org/abs/2206.11990
    ref: https://github.com/atomicarchitects/equiformer/blob/a4360ada2d213ba7b4d884335d3dc54a92b7a371/nets/graph_attention_transformer.py#L157
    """

    def __init__(
        self,
        irreps_in1: Union[str, e3nn.o3.Irreps],
        irreps_in2: Union[str, e3nn.o3.Irreps],
        irreps_out: Union[str, e3nn.o3.Irreps],
    ):
        super().__init__()
        self.irreps_in1 = e3nn.o3.Irreps(irreps_in1)
        self.irreps_in2 = e3nn.o3.Irreps(irreps_in2)
        irreps_out = e3nn.o3.Irreps(irreps_out)

        irreps_out_dtp = []
        instructions_dtp = []

        for i, (mul, ir_in1) in enumerate(self.irreps_in1):
            for j, (_, ir_in2) in enumerate(self.irreps_in2):
                for ir_out in ir_in1 * ir_in2:
                    if ir_out in irreps_out or ir_out == e3nn.o3.Irrep(0, 1):
                        k = len(irreps_out_dtp)
                        irreps_out_dtp.append((mul, ir_out))
                        instructions_dtp.append((i, j, k, "uvu", True))

        irreps_out_dtp = e3nn.o3.Irreps(irreps_out_dtp)

        self.tp = e3nn.o3.TensorProduct(
            irreps_in1,
            irreps_in2,
            irreps_out_dtp,
            instructions_dtp,
            internal_weights=False,
            shared_weights=False,
        )

        # For book-keeping.
        self.irreps_out = self.tp.irreps_out
        self.weight_numel = self.tp.weight_numel
        self.instructions = self.tp.instructions

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        out = self.tp(x, y, weight)
        return out


class SeparableTensorProduct(nn.Module):
    """
    Tensor product factored into depthwise and pointwise components

    ref: https://arxiv.org/abs/2206.11990
    ref: https://github.com/atomicarchitects/equiformer/blob/a4360ada2d213ba7b4d884335d3dc54a92b7a371/nets/graph_attention_transformer.py#L157
    """

    def __init__(
        self,
        irreps_in1: Union[str, e3nn.o3.Irreps],
        irreps_in2: Union[str, e3nn.o3.Irreps],
        irreps_out: Union[str, e3nn.o3.Irreps],
    ):
        super().__init__()
        self.irreps_in1 = e3nn.o3.Irreps(irreps_in1)
        self.irreps_in2 = e3nn.o3.Irreps(irreps_in2)
        self.irreps_out = e3nn.o3.Irreps(irreps_out)

        # Depthwise and pointwise
        self.dtp = DepthwiseTensorProduct(
            self.irreps_in1, self.irreps_in2, self.irreps_out
        )
        self.lin = Linear(self.dtp.irreps_out, self.irreps_out)

        # For book-keeping.
        self.weight_numel = self.dtp.weight_numel

    def forward(self, x, y, weight):
        out = self.dtp(x, y, weight)
        out = self.lin(out)
        return out
