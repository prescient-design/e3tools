import pytest

import e3nn
import torch

from e3tools.nn import ExtractIrreps


def test_extract_irreps():
    irreps_in = e3nn.o3.Irreps("0e + 1o + 2e")
    input = torch.as_tensor([1., 2., 3., 4., 5., 6., 7., 8., 9.])
    assert input.shape[-1] == irreps_in.dim

    layer = ExtractIrreps(irreps_in, "0e")
    output = layer(input)
    assert torch.allclose(output, torch.as_tensor([1.]))

    layer = ExtractIrreps(irreps_in, "1o")
    output = layer(input)
    assert torch.allclose(output, torch.as_tensor([2., 3., 4.]))

    layer = ExtractIrreps(irreps_in, "2e")
    output = layer(input)
    assert torch.allclose(output, torch.as_tensor([5., 6., 7., 8., 9.]))

