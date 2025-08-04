from typing import Callable, Mapping, Optional, Union

import e3nn
import e3nn.o3
import e3nn.nn
import torch
from torch import nn

from ._linear import Linear


class Gate(nn.Module):
    """
    Equivariant non-linear gate

    Parameters
    ----------
    irreps_out: e3nn.o3.Irreps
        output feature irreps
        (input irreps are inferred from output irreps)
    act: Mapping[int, nn.Module]
        Mapping from parity to activation module.
        If `None` defaults to `{1 : torch.nn.LeakyReLU(), -1: torch.nn.Tanh()}`
    act_gates: Mapping[int, nn.Module]
        Mapping from parity to activation module.
        If `None` defaults to `{1 : torch.nn.Sigmoid(), -1: torch.nn.Tanh()}`
    """

    def __init__(
        self,
        irreps_out: Union[str, e3nn.o3.Irreps],
        act: Optional[Mapping[int, nn.Module]] = None,
        act_gates: Optional[Mapping[int, nn.Module]] = None,
    ):
        super().__init__()

        self.irreps_out = e3nn.o3.Irreps(irreps_out)

        if act is None:
            act = {
                1: nn.LeakyReLU(),
                -1: nn.Tanh(),
            }

        if act_gates is None:
            act_gates = {
                1: nn.Sigmoid(),
                -1: nn.Tanh(),
            }

        irreps_scalars = e3nn.o3.Irreps(
            [(mul, ir) for mul, ir in irreps_out if ir.l == 0]
        )
        irreps_gated = e3nn.o3.Irreps([(mul, ir) for mul, ir in irreps_out if ir.l > 0])
        irreps_gates = e3nn.o3.Irreps([(mul, "0e") for mul, _ in irreps_gated])

        self.gate = e3nn.nn.Gate(
            irreps_scalars,
            [act[ir.p] for _, ir in irreps_scalars],
            irreps_gates,
            [act_gates[ir.p] for _, ir in irreps_gates],
            irreps_gated,
        )

        self.irreps_in = self.gate.irreps_in

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the gate to the input tensor."""
        return self.gate(x)


class Gated(nn.Module):
    """Wraps another layer with an equivariant gate."""

    def __init__(
        self,
        layer: Callable[..., nn.Module],
        irreps_in: Union[str, e3nn.o3.Irreps],
        irreps_out: Union[str, e3nn.o3.Irreps],
        act: Optional[Mapping[int, nn.Module]] = None,
        act_gates: Optional[Mapping[int, nn.Module]] = None,
    ):
        """
        Wraps another layer with an equivariant gate.

        Parameters
        ----------
        layer: Callable[..., nn.Module]
            factory function for wrapped layer.
            Should be callable as `layer(irreps_in=irreps_in, irreps_out=gate.irreps_in)`
        irreps_in: Union[str, e3nn.o3.Irreps]
            input feature irreps
        irreps_out: Union[str, e3nn.o3.Irreps]
            output feature irreps
        act: Mapping[int, nn.Module]
            Mapping from parity to activation module.
            If `None` defaults to `{1 : torch.nn.LeakyReLU(), -1: torch.nn.Tanh()}`
        act_gates: Mapping[int, nn.Module]
            Mapping from parity to activation module.
            If `None` defaults to `{1 : torch.nn.Sigmoid(), -1: torch.nn.Tanh()}`
        """
        super().__init__()

        self.irreps_in = e3nn.o3.Irreps(irreps_in)
        self.irreps_out = e3nn.o3.Irreps(irreps_out)

        self.gate = Gate(self.irreps_out, act=act, act_gates=act_gates)

        self.f = layer(irreps_in=self.irreps_in, irreps_out=self.gate.irreps_in)
        self.irreps_sh = self.f.irreps_sh

    def forward(self, *args, **kwargs):
        """Apply the layer and then the gate to the input tensor."""
        out = self.f(*args, **kwargs)
        out = self.gate(out)
        return out


class GateWrapper(nn.Module):
    """Applies a linear transformation before and after the gate."""

    def __init__(
        self,
        irreps_in: e3nn.o3.Irreps,
        irreps_out: e3nn.o3.Irreps,
        irreps_gate: e3nn.o3.Irreps,
    ):
        """Applies a linear transformation before and after the gate."""
        super().__init__()
        self.irreps_in = e3nn.o3.Irreps(irreps_in)
        self.irreps_out = e3nn.o3.Irreps(irreps_out)
        self.irreps_gate = e3nn.o3.Irreps(irreps_gate)

        self.gate = Gate(irreps_out)
        self.pre_gate = Linear(self.irreps_in, self.gate.irreps_in)
        self.post_gate = Linear(self.gate.irreps_out, self.irreps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the pre-gate, gate, and post-gate transformations."""
        x = self.pre_gate(x)
        x = self.gate(x)
        x = self.post_gate(x)
        return x
