from typing import Union

import e3nn
import torch
from e3nn import o3


class SeparableTensorProduct(torch.nn.Module):
    """
    Tensor product factored into depthwise and point wise components

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
        self.irreps_in1 = o3.Irreps(irreps_in1)
        self.irreps_in2 = o3.Irreps(irreps_in2)
        self.irreps_out = o3.Irreps(irreps_out)

        irreps_out_dtp = []
        instructions_dtp = []

        for i, (mul, ir_in1) in enumerate(self.irreps_in1):
            for j, (_, ir_in2) in enumerate(self.irreps_in2):
                for ir_out in ir_in1 * ir_in2:
                    if ir_out in self.irreps_out or ir_out == o3.Irrep(0, 1):
                        k = len(irreps_out_dtp)
                        irreps_out_dtp.append((mul, ir_out))
                        instructions_dtp.append((i, j, k, "uvu", True))

        irreps_out_dtp = o3.Irreps(irreps_out_dtp)

        # depth wise
        self.dtp = o3.TensorProduct(
            irreps_in1,
            irreps_in2,
            irreps_out_dtp,
            instructions_dtp,
            internal_weights=False,
            shared_weights=False,
        )

        # point wise
        self.lin = o3.Linear(irreps_out_dtp, self.irreps_out)

        self.weight_numel = self.dtp.weight_numel

    def forward(self, x, y, weight):
        out = self.dtp(x, y, weight)
        out = self.lin(out)
        return out


class ExperimentalTensorProduct(torch.nn.Module):
    """
    Compileable tensor product
    """

    def __init__(
        self,
        irreps_in1: Union[str, e3nn.o3.Irreps],
        irreps_in2: Union[str, e3nn.o3.Irreps],
        irreps_out: Union[str, e3nn.o3.Irreps],
    ):
        super().__init__()
        self.irreps_in1 = o3.Irreps(irreps_in1)
        self.irreps_in2 = o3.Irreps(irreps_in2)
        self.irreps_out = o3.Irreps(irreps_out)

        self.tp = o3.FullTensorProductv2(self.irreps_in1, self.irreps_in2)

        self.lin = o3.Linear(
            self.tp.irreps_out,
            self.irreps_out,
            internal_weights=False,
            shared_weights=False,
        )

        self.weight_numel = self.lin.weight_numel

    def forward(self, x, y, weight):
        out = self.tp(x, y)
        out = self.lin(out, weight)
        return out
