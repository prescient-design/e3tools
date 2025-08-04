from typing import Callable, Mapping, Optional

import e3nn
import e3nn.o3
from torch import nn

from ._gate import Gate
from ._linear import Linear


class ScalarMLP(nn.Sequential):
    """A multi-layer perceptron for scalar inputs and outputs."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: list[int],
        activation_layer: Callable[..., nn.Module] = nn.ReLU,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout=0.0,
        bias=True,
    ):
        layers = []
        in_dim = in_features
        for hidden_dim in hidden_features:
            layers.append(nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(activation_layer())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, out_features, bias=bias))
        layers.append(nn.Dropout(dropout))

        super().__init__(*layers)


class EquivariantMLPBlock(nn.Module):
    """
    Equivariant linear layer followed by optional norm and gated non-linearity
    """

    def __init__(
        self,
        irreps_in: e3nn.o3.Irreps,
        irreps_out: e3nn.o3.Irreps,
        act: Optional[Mapping[int, nn.Module]] = None,
        act_gates: Optional[Mapping[int, nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        """
        Parameters
        ----------
        irreps_in: e3nn.o3.Irreps
            Input irreps
        irreps_out: e3nn.o3.Irreps
            Output irreps
        act: Optional[Mapping[int, nn.Module]]
            Mapping from parity to activation module.
            If `None` defaults to `{1 : nn.LeakyReLU(), -1: nn.Tanh()}`
        act_gates: Optional[Mapping[int, nn.Module]]
            Mapping from parity to activation module.
            If `None` defaults to `{1 : nn.Sigmoid(), -1: nn.Tanh()}`
        """
        super().__init__()
        self.irreps_in = e3nn.o3.Irreps(irreps_in)
        self.irreps_out = e3nn.o3.Irreps(irreps_out)

        self.gate = Gate(self.irreps_out, act=act, act_gates=act_gates)
        self.lin = Linear(self.irreps_in, self.gate.irreps_in)

        if norm_layer:
            self.norm = norm_layer(self.lin.irreps_out)
        else:
            self.norm = None

    def forward(self, x):
        x = self.lin(x)
        if self.norm:
            x = self.norm(x)
        x = self.gate(x)
        return x


class EquivariantMLP(nn.Sequential):
    """An equivariant multi-layer perceptron with gated non-linearities."""

    def __init__(
        self,
        irreps_in: e3nn.o3.Irreps,
        irreps_out: e3nn.o3.Irreps,
        irreps_hidden_list: list[e3nn.o3.Irreps],
        act: Optional[Mapping[int, nn.Module]] = None,
        act_gates: Optional[Mapping[int, nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        layers = []

        irreps = irreps_in
        for irreps_hidden in irreps_hidden_list:
            layers.append(
                EquivariantMLPBlock(
                    irreps,
                    irreps_hidden,
                    act=act,
                    act_gates=act_gates,
                    norm_layer=norm_layer,
                )
            )
            irreps = irreps_hidden

        layers.append(Linear(irreps, irreps_out))

        super().__init__(*layers)

        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
