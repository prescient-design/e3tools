import pytest

import e3nn
import e3nn.util.test
import torch

from e3tools.nn import ExtractIrreps


@pytest.mark.parametrize("irreps_in", ["0e + 1o + 2e", "2x0e + 1o + 2x2e + 1x1o"])
def test_equivariance(irreps_in: str):
    irreps_in = e3nn.o3.Irreps(irreps_in)
    layer = ExtractIrreps(irreps_in, "0e")
    e3nn.util.test.assert_equivariant(
        layer,
        irreps_in=layer.irreps_in,
        irreps_out=layer.irreps_out,
    )

    layer = ExtractIrreps(irreps_in, "1o")
    e3nn.util.test.assert_equivariant(
        layer,
        irreps_in=layer.irreps_in,
        irreps_out=layer.irreps_out,
    )


def test_extract_irreps_simple():
    irreps_in = e3nn.o3.Irreps("0e + 1o + 2e")
    input = torch.as_tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    assert input.shape[-1] == irreps_in.dim

    layer = ExtractIrreps(irreps_in, "0e")
    output = layer(input)
    assert torch.allclose(output, torch.as_tensor([1.0]))

    layer = ExtractIrreps(irreps_in, "1o")
    output = layer(input)
    assert torch.allclose(output, torch.as_tensor([2.0, 3.0, 4.0]))

    layer = ExtractIrreps(irreps_in, "2e")
    output = layer(input)
    assert torch.allclose(output, torch.as_tensor([5.0, 6.0, 7.0, 8.0, 9.0]))


def test_extract_irreps_multiplicity():
    irreps_in = e3nn.o3.Irreps("0e + 1o + 2x0e + 1o")
    input = torch.as_tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    assert input.shape[-1] == irreps_in.dim

    layer = ExtractIrreps(irreps_in, "0e")
    output = layer(input)
    assert torch.allclose(output, torch.as_tensor([1.0, 5.0, 6.0]))

    layer = ExtractIrreps(irreps_in, "1o")
    output = layer(input)
    assert torch.allclose(output, torch.as_tensor([2.0, 3.0, 4.0, 7.0, 8.0, 9.0]))
