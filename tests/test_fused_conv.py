from typing import Tuple
import functools

import pytest
import torch
import e3nn

from e3tools.nn import Conv, FusedConv
from e3tools import radius_graph

torch.set_default_dtype(torch.float64)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fused_conv(seed):
    torch.manual_seed(seed)

    N = 20
    edge_attr_dim = 10
    max_radius = 1.3
    irreps_in = e3nn.o3.Irreps("10x0e + 10x1o + 10x2e")
    irreps_sh = irreps_in.spherical_harmonics(2)

    layer = Conv(irreps_in, irreps_in, irreps_sh, edge_attr_dim=edge_attr_dim)
    fused_layer = FusedConv(irreps_in, irreps_in, irreps_sh, edge_attr_dim=edge_attr_dim)

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
    assert torch.allclose(out, out_fused, atol=1e-10)

