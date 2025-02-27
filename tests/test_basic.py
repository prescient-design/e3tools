import functools

import pytest
import torch
import e3nn
from e3nn import o3
from e3tools.nn import (
    Attention,
    Conv,
    ConvBlock,
    EquivariantMLP,
    ExperimentalConv,
    Gated,
    LayerNorm,
    MultiheadAttention,
    SeparableConv,
    TransformerBlock,
)
from e3tools import radius_graph

torch.set_default_dtype(torch.float64)

CONV_LAYERS = [Conv, SeparableConv, ExperimentalConv]


def apply_layer_rot(layer):
    N = 20
    edge_attr_dim = 10
    max_radius = 1.3

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

    edge_sh = o3.spherical_harmonics(
        layer.irreps_sh, edge_vec, True, normalization="component"
    )

    rot = o3.rand_matrix()

    D_node_attr = layer.irreps_in.D_from_matrix(rot)
    D_edge_sh = layer.irreps_sh.D_from_matrix(rot)

    D_out = layer.irreps_out.D_from_matrix(rot)

    out_1 = layer(
        node_attr @ D_node_attr.T, edge_index, edge_attr, edge_sh @ D_edge_sh.T
    )
    out_2 = layer(node_attr, edge_index, edge_attr, edge_sh) @ D_out.T

    return out_1, out_2


@pytest.mark.parametrize("conv", CONV_LAYERS)
def test_conv(conv):
    irreps_in = o3.Irreps("10x0e + 10x1o + 10x2e")
    irreps_sh = irreps_in.spherical_harmonics(2)
    edge_attr_dim = 10

    layer = conv(irreps_in, irreps_in, irreps_sh, edge_attr_dim=edge_attr_dim)

    out_1, out_2 = apply_layer_rot(layer)
    assert torch.allclose(out_1, out_2, atol=1e-10)


@pytest.mark.parametrize("conv", CONV_LAYERS)
def test_gated_conv(conv):
    irreps_in = o3.Irreps("10x0e + 10x1o + 10x2e")
    irreps_sh = irreps_in.spherical_harmonics(2)
    edge_attr_dim = 10

    wrapped = functools.partial(conv, irreps_sh=irreps_sh, edge_attr_dim=edge_attr_dim)

    layer = Gated(wrapped, irreps_in=irreps_in, irreps_out=irreps_in)

    out_1, out_2 = apply_layer_rot(layer)
    assert torch.allclose(out_1, out_2, atol=1e-10)


@pytest.mark.parametrize("conv", CONV_LAYERS)
def test_conv_block(conv):
    irreps_in = o3.Irreps("10x0e + 10x1o + 10x2e")
    irreps_sh = irreps_in.spherical_harmonics(2)
    edge_attr_dim = 10

    layer = ConvBlock(
        irreps_in=irreps_in,
        irreps_out=irreps_in,
        irreps_sh=irreps_sh,
        edge_attr_dim=edge_attr_dim,
        conv=conv,
    )

    out_1, out_2 = apply_layer_rot(layer)
    assert torch.allclose(out_1, out_2, atol=1e-10)


@pytest.mark.parametrize("conv", CONV_LAYERS)
def test_attention(conv):
    irreps_in = o3.Irreps("10x0e + 10x1o + 10x2e")
    irreps_out = irreps_in
    irreps_sh = irreps_in.spherical_harmonics(2)
    irreps_key = irreps_in
    irreps_query = irreps_in
    edge_attr_dim = 10

    layer = Attention(
        irreps_in,
        irreps_out,
        irreps_sh,
        irreps_query,
        irreps_key,
        edge_attr_dim,
        conv=conv,
    )

    out_1, out_2 = apply_layer_rot(layer)
    assert torch.allclose(out_1, out_2, atol=1e-10)


@pytest.mark.parametrize("conv", [Conv, SeparableConv])
def test_multihead_attention(conv):
    irreps_in = o3.Irreps("10x0e + 10x1o + 10x2e")
    irreps_out = irreps_in
    irreps_sh = irreps_in.spherical_harmonics(2)
    irreps_key = irreps_in
    irreps_query = irreps_in
    edge_attr_dim = 10
    n_head = 2

    layer = MultiheadAttention(
        irreps_in,
        irreps_out,
        irreps_sh,
        irreps_query,
        irreps_key,
        edge_attr_dim,
        n_head,
        conv=conv,
    )

    out_1, out_2 = apply_layer_rot(layer)
    assert torch.allclose(out_1, out_2, atol=1e-10)


def test_layer_norm():
    irreps = o3.Irreps("10x0e + 10x1o + 10x2e")

    layer = LayerNorm(irreps)
    rot = o3.rand_matrix()
    D = irreps.D_from_matrix(rot)

    x = irreps.randn(10, -1)

    out_1 = layer(x @ D.T)
    out_2 = layer(x) @ D.T

    assert torch.allclose(out_1, out_2, atol=1e-10)


def test_equivariant_mlp():
    irreps = o3.Irreps("10x0e + 10x1o + 10x2e")
    irreps_hidden = o3.Irreps([(4 * mul, ir) for mul, ir in irreps])

    layer = EquivariantMLP(
        irreps, irreps, [irreps_hidden, irreps_hidden], norm_layer=LayerNorm
    )

    rot = o3.rand_matrix()
    D = irreps.D_from_matrix(rot)

    x = irreps.randn(10, -1)

    out_1 = layer(x @ D.T)
    out_2 = layer(x) @ D.T

    assert torch.allclose(out_1, out_2, atol=1e-10)


def test_transformer():
    irreps_in = o3.Irreps("10x0e + 10x1o + 10x2e")
    irreps_out = irreps_in
    irreps_sh = irreps_in.spherical_harmonics(2)
    edge_attr_dim = 10
    n_head = 2

    layer = TransformerBlock(
        irreps_in=irreps_in,
        irreps_out=irreps_out,
        irreps_sh=irreps_sh,
        edge_attr_dim=edge_attr_dim,
        n_head=n_head,
    )

    out_1, out_2 = apply_layer_rot(layer)
    assert torch.allclose(out_1, out_2, atol=1e-10)
