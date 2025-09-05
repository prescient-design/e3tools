from typing import Tuple
import functools

import pytest
import torch
import e3nn

from e3tools.nn import Conv, FusedConv
from e3tools import radius_graph


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fused_conv(seed):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    torch.manual_seed(seed)
    torch.set_default_device("cuda")

    N = 20
    edge_attr_dim = 1
    max_radius = 100
    irreps_in = e3nn.o3.Irreps("1x0e + 1x1o + 1x2e + 1x3o")
    irreps_sh = irreps_in.spherical_harmonics(2)

    def fake_nn(edge_attr_dim, num_elements):
        def fn(edge_attr):
            return torch.ones((edge_attr.shape[0], num_elements), device=edge_attr.device)
        return fn

    layer = Conv(irreps_in=irreps_in, irreps_out=irreps_in, irreps_sh=irreps_sh, radial_nn=fake_nn, edge_attr_dim=edge_attr_dim)
    fused_layer = FusedConv(irreps_in=irreps_in, irreps_out=irreps_in, irreps_sh=irreps_sh, radial_nn=fake_nn, edge_attr_dim=edge_attr_dim)

    pos = torch.randn(N, 3)
    node_attr = layer.irreps_in.randn(N, -1)

    edge_index = radius_graph(pos, max_radius)
    edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
    edge_length = (edge_vec).norm(dim=1)
    edge_attr = e3nn.math.soft_one_hot_linspace(
        edge_length,
        start=0.0,
        end=max_radius,
        number=edge_attr_dim,
        basis="smooth_finite",
        cutoff=True,
    )
    edge_sh = e3nn.o3.spherical_harmonics(
        layer.irreps_sh, edge_vec, True, normalization="component"
    )

    out = layer(node_attr, edge_index, edge_attr, edge_sh)
    out_fused = fused_layer(node_attr, edge_index, edge_attr, edge_sh)
    print(out[0] / out_fused[0])
    print(((out[0, 0] / out_fused[0, 0]) / (out[1, 1] / out_fused[1, 1])) ** 4)
    print(((out[0, 0] / out_fused[0, 0]) / (out[4, 4] / out_fused[4, 4])) ** 4)
    print(((out[0, 0] / out_fused[0, 0]) / (out[9, 9] / out_fused[9, 9])) ** 4)
    assert torch.allclose(out, out_fused, atol=1e-10)

