import pytest

import e3nn
import e3nn.util.test
import torch

from e3tools.nn import LayerNorm
from e3tools.nn._pack_unpack import unpack_irreps


@pytest.mark.parametrize("irreps_in", ["0e + 1o", "0e + 1o + 2e", "3x1o + 2x2o"])
def test_equivariance(irreps_in: str):
    irreps_in = e3nn.o3.Irreps(irreps_in)
    layer = LayerNorm(irreps_in)
    e3nn.util.test.assert_equivariant(
        layer,
        irreps_in=layer.irreps_in,
        irreps_out=layer.irreps_out,
    )


@pytest.mark.parametrize("irreps_in", ["0e + 1o", "0e + 1o + 2e", "3x1o + 2x2o"])
def test_layer_norm(irreps_in: str):
    irreps_in = e3nn.o3.Irreps(irreps_in)
    layer = LayerNorm(irreps_in)
    assert layer.irreps_in == irreps_in
    assert layer.irreps_out == irreps_in

    input = irreps_in.randn(-1)
    output = layer(input)

    for _, ir, field in unpack_irreps(output, irreps_in):
        sq_norms = field.norm(dim=-1, keepdim=True).pow(2).sum(dim=-1).mean(dim=-1)
        if ir.l == 0 and ir.p == 1:
            assert torch.allclose(sq_norms, torch.as_tensor([0.0]))
        else:
            assert torch.allclose(sq_norms, torch.as_tensor([1.0]))
