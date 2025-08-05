import pytest
import torch
import e3nn

e3nn.set_optimization_defaults(jit_script_fx=False)

import e3nn.o3
import copy

from e3tools.nn import Attention, MultiheadAttention


@pytest.fixture
def irreps_in():
    return e3nn.o3.Irreps("2x0e + 2x1o + 2x2e")


@pytest.fixture
def irreps_out():
    return e3nn.o3.Irreps("2x0e + 2x1o + 2x2e")


@pytest.fixture
def irreps_sh():
    return e3nn.o3.Irreps.spherical_harmonics(2)


@pytest.fixture
def irreps_key(irreps_in):
    return irreps_in


@pytest.fixture
def irreps_query(irreps_in):
    return irreps_in


@pytest.fixture
def edge_attr_dim():
    return 10


@pytest.fixture
def singlehead_attention(
    irreps_in, irreps_out, irreps_sh, irreps_key, irreps_query, edge_attr_dim
):
    return Attention(
        irreps_in=irreps_in,
        irreps_out=irreps_out,
        irreps_sh=irreps_sh,
        irreps_query=irreps_query,
        irreps_key=irreps_key,
        edge_attr_dim=edge_attr_dim,
        conv=None,
    )


@pytest.fixture
def multihead_attention(
    irreps_in, irreps_out, irreps_sh, irreps_key, irreps_query, edge_attr_dim
):
    return MultiheadAttention(
        irreps_in=irreps_in,
        irreps_out=irreps_out,
        irreps_sh=irreps_sh,
        irreps_query=irreps_query,
        irreps_key=irreps_key,
        edge_attr_dim=edge_attr_dim,
        num_heads=2,
        conv=None,
    )


@pytest.mark.parametrize("model_name", ["singlehead_attention", "multihead_attention"])
@pytest.mark.parametrize("causal", [True, False])
def test_compile(model_name, causal, request, irreps_in, irreps_sh, edge_attr_dim):
    model = request.getfixturevalue(model_name)
    compiled_model = torch.compile(model, fullgraph=True)

    node_attr = torch.randn(5, irreps_in.dim)
    edge_attr = torch.randn(4, edge_attr_dim)
    edge_sh = torch.randn(4, irreps_sh.dim)
    edge_index = torch.tensor([[1, 2, 3, 4], [0, 1, 2, 3]])
    if causal:
        mask = edge_index[0] <= edge_index[1]
    else:
        mask = None

    out = model(node_attr, edge_index, edge_attr, edge_sh, mask=mask)
    compiled_out = compiled_model(node_attr, edge_index, edge_attr, edge_sh, mask=mask)

    assert torch.allclose(out, compiled_out)


@pytest.mark.parametrize("model_name", ["singlehead_attention", "multihead_attention"])
@pytest.mark.parametrize("causal", [True, False])
def test_no_nans(model_name, causal, request, irreps_in, irreps_sh, edge_attr_dim):
    model = request.getfixturevalue(model_name)
    node_attr = torch.randn(5, irreps_in.dim)
    edge_attr = torch.randn(4, edge_attr_dim)
    edge_sh = torch.randn(4, irreps_sh.dim)

    edge_index = torch.tensor([[1, 2, 3, 4], [0, 1, 2, 3]])
    if causal:
        mask = edge_index[0] <= edge_index[1]
    else:
        mask = None
    print(f"Mask: {mask}")

    out = model(node_attr, edge_index, edge_attr, edge_sh, mask=mask)

    assert torch.isfinite(out).all()


@pytest.mark.parametrize("model_name", ["singlehead_attention", "multihead_attention"])
def test_causal_vs_non_causal_attention(
    model_name, request, irreps_in, irreps_sh, edge_attr_dim
):
    model = request.getfixturevalue(model_name)
    node_attr = torch.randn(5, irreps_in.dim)
    edge_attr = torch.randn(4, edge_attr_dim)
    edge_sh = torch.randn(4, irreps_sh.dim)

    # Check that the outputs are the same when all edges are causal.
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    causal_mask = edge_index[0] <= edge_index[1]
    non_causal_out = model(node_attr, edge_index, edge_attr, edge_sh, mask=None)
    causal_out = model(node_attr, edge_index, edge_attr, edge_sh, mask=causal_mask)
    assert torch.allclose(non_causal_out, causal_out)

    # Check that the outputs are the same for the nodes that do not have any causal edges.
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    causal_mask = edge_index[0] <= edge_index[1]
    non_causal_out = model(node_attr, edge_index, edge_attr, edge_sh, mask=None)
    causal_out = model(node_attr, edge_index, edge_attr, edge_sh, mask=causal_mask)
    assert not torch.allclose(non_causal_out[:1], causal_out[:1])
    assert torch.allclose(non_causal_out[1:], causal_out[1:])

    # Check that the outputs are the same for the nodes that do not have any causal edges.
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 4]])
    causal_mask = edge_index[0] <= edge_index[1]
    non_causal_out = model(node_attr, edge_index, edge_attr, edge_sh, mask=None)
    causal_out = model(node_attr, edge_index, edge_attr, edge_sh, mask=causal_mask)
    assert not torch.allclose(non_causal_out[:2], causal_out[:2])
    assert torch.allclose(non_causal_out[2:], causal_out[2:])
