import functools

import pytest
import torch
from torch import nn
import e3nn

from e3tools.nn import (
    Conv,
    FusedConv,
    DepthwiseTensorProduct,
    ScalarMLP,
    SeparableConv,
    FusedSeparableConv,
    SeparableTensorProduct,
)
from e3tools import radius_graph


@pytest.mark.parametrize(
    "tensor_product_type",
    [
        "default",
        "depthwise",
        "separable",
    ],
)
@pytest.mark.parametrize("seed", [0, 1])
def test_fused_conv(tensor_product_type: str, seed: int):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    if tensor_product_type == "default":
        tensor_product = functools.partial(
            e3nn.o3.FullyConnectedTensorProduct,
            shared_weights=False,
            internal_weights=False,
        )

    elif tensor_product_type == "depthwise":
        tensor_product = DepthwiseTensorProduct

    elif tensor_product_type == "separable":
        tensor_product = SeparableTensorProduct

    torch.manual_seed(seed)
    torch.set_default_device("cuda")

    N = 20
    edge_attr_dim = 10
    max_radius = 1.0
    irreps_in = e3nn.o3.Irreps("10x0e + 4x1o + 1x2e")
    irreps_sh = irreps_in.spherical_harmonics(2)

    layer = Conv(
        irreps_in=irreps_in,
        irreps_out=irreps_in,
        irreps_sh=irreps_sh,
        edge_attr_dim=edge_attr_dim,
        tensor_product=tensor_product,
    )
    fused_layer = FusedConv(
        irreps_in=irreps_in,
        irreps_out=irreps_in,
        irreps_sh=irreps_sh,
        edge_attr_dim=edge_attr_dim,
        tensor_product=tensor_product,
    )

    # Copy weights.
    fused_layer.load_state_dict(layer.state_dict())

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

    torch.testing.assert_close(out, out_fused, rtol=1e-3, atol=1e-5)
