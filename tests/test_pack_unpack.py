import pytest

import e3nn
import e3nn.util.test
import torch

from e3tools.nn import AxisToMul, MulToAxis


@pytest.mark.parametrize(
    "irreps_in, factor",
    zip(
        ["0e + 1o", "8x0e + 8x1o + 8x2e", "8x0e + 8x1o + 8x2e", "3x1o + 3x2o"],
        [1, 2, 4, 3],
    ),
)
def test_axis_to_mul_equivariance(irreps_in: str, factor: int, batch_size: int = 5):
    irreps_in = e3nn.o3.Irreps(irreps_in)
    layer = AxisToMul(irreps_in, factor)
    args_in = irreps_in.randn(batch_size, factor, -1)
    e3nn.util.test.assert_equivariant(
        layer,
        args_in=[args_in],
        irreps_in=layer.irreps_in,
        irreps_out=layer.irreps_out,
    )


@pytest.mark.parametrize(
    "irreps_in, factor",
    zip(
        ["0e + 1o", "8x0e + 8x1o + 8x2e", "8x0e + 8x1o + 8x2e", "3x1o + 3x2o"],
        [1, 2, 4, 3],
    ),
)
def test_mul_to_axis_equivariance(irreps_in: str, factor: int, batch_size: int = 5):
    irreps_in = e3nn.o3.Irreps(irreps_in)
    layer = MulToAxis(irreps_in, factor)
    args_in = irreps_in.randn(batch_size, -1)
    e3nn.util.test.assert_equivariant(
        layer,
        args_in=[args_in],
        irreps_in=layer.irreps_in,
        irreps_out=layer.irreps_out,
    )


@pytest.mark.parametrize(
    "irreps_in, factor",
    zip(
        ["0e + 1o", "8x0e + 8x1o + 8x2e", "8x0e + 8x1o + 8x2e", "3x1o + 3x2o"],
        [1, 2, 4, 3],
    ),
)
def test_axis_to_mul_shape(irreps_in: str, factor: int, batch_size: int = 5):
    irreps_in = e3nn.o3.Irreps(irreps_in)
    layer = AxisToMul(irreps_in, factor)
    assert layer.irreps_in == irreps_in

    input = irreps_in.randn(batch_size, factor, -1)
    output = layer(input)

    assert output.shape == (batch_size, factor * irreps_in.dim)


@pytest.mark.parametrize(
    "irreps_in, factor",
    zip(
        ["0e + 1o", "8x0e + 8x1o + 8x2e", "8x0e + 8x1o + 8x2e", "3x1o + 3x2o"],
        [1, 2, 4, 3],
    ),
)
def test_mul_to_axis_shape(irreps_in: str, factor: int, batch_size: int = 5):
    irreps_in = e3nn.o3.Irreps(irreps_in)
    layer = MulToAxis(irreps_in, factor)
    assert layer.irreps_in == irreps_in

    input = irreps_in.randn(batch_size, -1)
    output = layer(input)

    assert output.shape == (batch_size, factor, irreps_in.dim // factor)


@pytest.mark.parametrize(
    "irreps_in, factor",
    zip(
        ["0e + 1o", "8x0e + 8x1o + 8x2e", "8x0e + 8x1o + 8x2e", "3x1o + 3x2o"],
        [1, 2, 4, 3],
    ),
)
def test_inverse(irreps_in: str, factor: int, batch_size: int = 5):
    irreps_in = e3nn.o3.Irreps(irreps_in)
    layer = MulToAxis(irreps_in, factor)
    inv_layer = AxisToMul(layer.irreps_out, factor)

    assert layer.irreps_in == irreps_in
    assert inv_layer.irreps_out == irreps_in

    input = irreps_in.randn(batch_size, -1)
    output = layer(input)
    recovered = inv_layer(output)

    assert torch.allclose(input, recovered)
