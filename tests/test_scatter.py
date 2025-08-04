import einops
import torch

from e3tools import scatter, scatter_softmax


def test_scatter_basic():
    B = 4
    D = 7
    x = torch.randn(B, D)
    ref = torch.sum(x, dim=1)

    index = einops.repeat(torch.arange(B), "b -> (b d)", d=D)

    out = scatter(x.view(-1), index, dim=0, dim_size=B, reduce="sum")

    torch.testing.assert_close(ref, out)


def test_scatter_softmax():
    B = 4
    D = 7
    x = torch.randn(B, D)
    ref = torch.nn.functional.softmax(x, dim=1)

    index = einops.repeat(torch.arange(B), "b -> (b d)", d=D)

    out = scatter_softmax(x.view(-1), index, dim=0, dim_size=B)

    torch.testing.assert_close(ref.view(-1), out)


def test_scatter_softmax_overflow():
    B = 4
    D = 7
    x = torch.randn(B, D) + 100.0
    ref = torch.nn.functional.softmax(x, dim=1)

    index = einops.repeat(torch.arange(B), "b -> (b d)", d=D)

    out = scatter_softmax(x.view(-1), index, dim=0, dim_size=B)

    torch.testing.assert_close(ref.view(-1), out)
