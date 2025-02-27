import pytest

import e3nn
import e3nn.util.test
import torch

from e3tools.nn import ScaleIrreps


@pytest.mark.parametrize("irreps_in", ["0e + 1o", "0e + 1o + 2e", "3x1o + 2x2o"])
def test_equivariance(irreps_in: str):
    irreps_in = e3nn.o3.Irreps(irreps_in)
    layer = ScaleIrreps(irreps_in)
    irreps_weight = e3nn.o3.Irreps(f"{layer.irreps_in.num_irreps}x0e")
    e3nn.util.test.assert_equivariant(
        layer,
        irreps_in=[layer.irreps_in, irreps_weight],
        irreps_out=layer.irreps_out,
    )


@pytest.mark.parametrize("irreps_in", ["0e + 1o", "0e + 1o + 2e", "3x1o + 2x2o"])
def test_scale_irreps_by_one(irreps_in: str):
    irreps_in = e3nn.o3.Irreps(irreps_in)
    layer = ScaleIrreps(irreps_in)
    assert layer.irreps_in == irreps_in
    assert layer.irreps_out == irreps_in

    input = irreps_in.randn(-1)
    weight = torch.ones(irreps_in.num_irreps)
    output = layer(input, weight)

    assert torch.allclose(input, output)


@pytest.mark.parametrize("irreps_in", ["0e + 1o", "0e + 1o + 2e", "3x1o + 2x2o"])
def test_scale_irreps_random(irreps_in: str):
    irreps_in = e3nn.o3.Irreps(irreps_in)
    layer = ScaleIrreps(irreps_in)
    assert layer.irreps_in == irreps_in
    assert layer.irreps_out == irreps_in

    input = irreps_in.randn(-1)
    weight = torch.randn(irreps_in.num_irreps)
    output = layer(input, weight)

    norm = e3nn.o3.Norm(irreps_in)
    factor = norm(output) / norm(input)
    assert torch.allclose(factor, torch.abs(weight))
