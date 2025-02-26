import itertools
from typing import Callable, Optional

import e3nn
import torch
from e3nn import o3

from e3tools import scatter

from ._conv import Conv
from ._interaction import LinearSelfInteraction
from ._layer_norm import LayerNorm
from ._mlp import EquivariantMLP


def split_irreps(irreps: e3nn.o3.Irreps, n_head: int):
    for mul, ir in irreps:
        assert mul % n_head == 0

    irreps_per_head = o3.Irreps([(mul // n_head, ir) for mul, ir in irreps])
    irreps_split = sum(itertools.repeat(irreps_per_head, n_head), start=o3.Irreps())

    assert irreps.dim == irreps_split.dim

    return irreps_split, irreps_per_head


class Attention(torch.nn.Module):
    """
    Equivariant attention layer

    ref: https://arxiv.org/abs/2006.10503
    """

    def __init__(
        self,
        irreps_in: e3nn.o3.Irreps,
        irreps_out: e3nn.o3.Irreps,
        irreps_sh: e3nn.o3.Irreps,
        irreps_query: e3nn.o3.Irreps,
        irreps_key: e3nn.o3.Irreps,
        edge_attr_dim,
        conv: Optional[Callable[..., torch.nn.Module]] = None,
        return_attention: bool = False,
    ):
        """
        Parameters
        ----------
        irreps_in: e3nn.o3.Irreps
            Input node feature irreps
        irreps_out: e3nn.o3.Irreps
            Ouput node feature irreps
        irreps_sh: e3nn.o3.Irreps
            Edge spherical harmonic irreps
        irreps_query: e3nn.o3.Irreps
            Attention query irreps
        irreps_key: e3nn.o3.Irreps
            Attention key irreps
        edge_attr_dim: int
            Dimension of scalar edge attributes to be passed to radial_nn
        conv: Optional[Callable[..., torch.nn.Module]] = None
            Factory function for convolution layer used for computing keys and values
        return_attention: bool = False
            Whether to return attn or not
        """
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.irreps_query = o3.Irreps(irreps_query)
        self.irreps_key = o3.Irreps(irreps_key)
        self.return_attention = return_attention

        self.h_q = o3.Linear(irreps_in, irreps_query)

        if conv is None:
            conv = Conv

        self.h_k = conv(
            irreps_in=self.irreps_in,
            irreps_out=self.irreps_key,
            irreps_sh=irreps_sh,
            edge_attr_dim=edge_attr_dim,
        )
        self.h_v = conv(
            irreps_in=self.irreps_in,
            irreps_out=self.irreps_out,
            irreps_sh=irreps_sh,
            edge_attr_dim=edge_attr_dim,
        )

        self.dot = o3.FullyConnectedTensorProduct(irreps_query, irreps_key, "0e")

    def forward(self, node_attr, edge_index, edge_attr, edge_sh):
        """
        Computes the forward pass of the equivariant graph attention

        Let N be the number of nodes, and E be the number of edges

        Parameters
        ----------
        node_attr: [N, irreps_in.dim]
        edge_index: [2, E]
        edge_attr: [E, edge_attr_dim]
        edge_sh: [E, irreps_sh.dim]

        Returns
        -------
        out: [N, irreps_out.dim]
        """

        N = node_attr.shape[0]

        src, dst = edge_index

        # compute queries (per node), keys (per edge) and values (per edge)
        q = self.h_q(node_attr)
        k = self.h_k.apply_per_edge(node_attr[src], edge_attr, edge_sh)
        v = self.h_v.apply_per_edge(node_attr[src], edge_attr, edge_sh)

        # compute softmax
        exp = self.dot(q[dst], k).exp()
        z = scatter(exp, dst, dim=0, dim_size=N, reduce="mean")
        alpha = exp / z[dst]

        attn = alpha.relu().sqrt()

        out = scatter(attn * v, dst, dim=0, dim_size=N, reduce="sum")

        if self.return_attention:
            return out, attn
        else:
            return out


class MultiheadAttention(torch.nn.Module):
    """
    Equivariant attention layer with multiple heads

    ref: https://arxiv.org/abs/2006.10503
    """

    def __init__(
        self,
        irreps_in: e3nn.o3.Irreps,
        irreps_out: e3nn.o3.Irreps,
        irreps_sh: e3nn.o3.Irreps,
        irreps_query: e3nn.o3.Irreps,
        irreps_key: e3nn.o3.Irreps,
        edge_attr_dim: int,
        n_head: int,
        conv: Optional[Callable[..., torch.nn.Module]] = None,
        return_attention: bool = False,
    ):
        """
        Parameters
        ----------
        irreps_in: e3nn.o3.Irreps
            Input node feature irreps
        irreps_out: e3nn.o3.Irreps
            Ouput node feature irreps
        irreps_sh: e3nn.o3.Irreps
            Edge spherical harmonic irreps
        irreps_query: e3nn.o3.Irreps
            Attention query irreps
        irreps_key: e3nn.o3.Irreps
            Attention key irreps
        edge_attr_dim: int
            Dimension of scalar edge attributes to be passed to radial_nn
        n_head: int
            Number of attention heads
        conv: Optional[Callable[..., torch.nn.Module]] = None
            Factory function for convolution layer used for computing keys and values
        return_attention: bool = False
            Whether to return attn or not
        """
        super().__init__()

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.irreps_query = o3.Irreps(irreps_query)
        self.irreps_key = o3.Irreps(irreps_key)
        self.n_head = n_head
        self.return_attention = return_attention

        irreps_in_split, irreps_in_per_head = split_irreps(irreps_in, n_head)
        irreps_out_split, irreps_out_per_head = split_irreps(irreps_out, n_head)
        irreps_query_split, irreps_query_per_head = split_irreps(irreps_query, n_head)
        irreps_key_split, irreps_key_per_head = split_irreps(irreps_key, n_head)

        if conv is None:
            conv = Conv

        self.h_q = o3.Linear(irreps_in, irreps_query_split)

        self.h_k = conv(
            irreps_in=self.irreps_in,
            irreps_out=irreps_key_split,
            irreps_sh=irreps_sh,
            edge_attr_dim=edge_attr_dim,
        )
        self.h_v = conv(
            irreps_in=self.irreps_in,
            irreps_out=irreps_out_split,
            irreps_sh=irreps_sh,
            edge_attr_dim=edge_attr_dim,
        )

        self.dot = o3.FullyConnectedTensorProduct(
            irreps_query_per_head, irreps_key_per_head, "0e"
        )

        self.lin_out = o3.Linear(irreps_out_split, irreps_out)

    def forward(self, node_attr, edge_index, edge_attr, edge_sh):
        """
        Computes the forward pass of equivariant graph attention

        Let N be the number of nodes, and E be the number of edges

        Parameters
        ----------
        node_attr: [N, irreps_in.dim]
        edge_index: [2, E]
        edge_attr: [E, edge_attr_dim]
        edge_sh: [E, irreps_sh.dim]

        Returns
        -------
        out: [N, irreps_out.dim]
        """
        src, dst = edge_index

        N = node_attr.shape[0]
        E = src.shape[0]

        # compute queries (per node), keys (per edge) and values (per edge)
        q = self.h_q(node_attr)
        k = self.h_k.apply_per_edge(node_attr[src], edge_attr, edge_sh)
        v = self.h_v.apply_per_edge(node_attr[src], edge_attr, edge_sh)

        # create head index as batch-like dimension
        q = q.view(N, self.n_head, -1)
        k = k.view(E, self.n_head, -1)
        v = v.view(E, self.n_head, -1)

        # compute softmax
        exp = self.dot(q[dst], k).exp()
        z = scatter(exp, dst, dim=0, dim_size=N, reduce="mean")
        alpha = exp / z[dst]

        attn = alpha.relu().sqrt()

        out = scatter(attn * v, dst, dim=0, dim_size=N, reduce="sum").view(N, -1)

        # use linear layer to transform back into original irreps
        out = self.lin_out(out)

        if self.return_attention:
            return out, attn
        else:
            return out


class TransformerBlock(torch.nn.Module):
    """
    Equivariant transformer block
    """

    def __init__(
        self,
        irreps_in: e3nn.o3.Irreps,
        irreps_out: e3nn.o3.Irreps,
        irreps_sh: e3nn.o3.Irreps,
        edge_attr_dim: int,
        n_head: int = 1,
        irreps_query: Optional[e3nn.o3.Irreps] = None,
        irreps_key: Optional[e3nn.o3.Irreps] = None,
        irreps_ff_hidden_list: Optional[list[e3nn.o3.Irreps]] = None,
        conv: Optional[Callable[..., torch.nn.Module]] = None,
    ):
        """
        Parameters
        ----------
        irreps_in: e3nn.o3.Irreps
            Input node feature irreps
        irreps_out: e3nn.o3.Irreps
            Ouput node feature irreps
        irreps_sh: e3nn.o3.Irreps
            Edge spherical harmonic irreps
        edge_attr_dim: int
            Dimension of scalar edge attributes to be passed to radial_nn
        n_head: int
            Number of attention heads
        irreps_query: Optional[e3nn.o3.Irreps]
            Attention query irreps. If `None` use `irreps_in`.
        irreps_key: Optional[e3nn.o3.Irreps]
            Attention key irreps. If `None` use `irreps_in`.
        irreps_ff_hidden_list: Optional[list[e3nn.o3.Irreps]] = None
            list of irreps for hidden layers used in feedforward network.
            If `None` then single hidden layer with multiplicity of each irrep
            blown up 4x.
        conv: Optional[Callable[..., torch.nn.Module]] = None
            Factory function for convolution layer used for computing keys and values
        """
        super().__init__()

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_sh = o3.Irreps(irreps_sh)

        if irreps_query is None:
            irreps_query = irreps_in

        if irreps_key is None:
            irreps_key = irreps_in

        if irreps_ff_hidden_list is None:
            irreps_ff_hidden = o3.Irreps([(4 * mul, ir) for mul, ir in self.irreps_out])
            irreps_ff_hidden_list = [irreps_ff_hidden]

        if conv is None:
            conv = Conv

        attn = MultiheadAttention(
            irreps_in,
            irreps_out,
            irreps_sh,
            irreps_query,
            irreps_key,
            edge_attr_dim,
            n_head,
            conv=conv,
            return_attention=False,
        )

        self.attn = LinearSelfInteraction(attn)
        self.norm1 = LayerNorm(self.attn.irreps_out)

        ff = EquivariantMLP(
            irreps_out, irreps_out, irreps_ff_hidden_list, norm_layer=None
        )
        self.ff = LinearSelfInteraction(ff)

        self.norm2 = LayerNorm(self.ff.irreps_out)

    def forward(self, node_attr, edge_index, edge_attr, edge_sh):
        """
        Computes the forward pass of equivariant graph attention

        Let N be the number of nodes, and E be the number of edges

        Parameters
        ----------
        node_attr: [N, irreps_in.dim]
        edge_index: [2, E]
        edge_attr: [E, edge_attr_dim]
        edge_sh: [E, irreps_sh.dim]

        Returns
        -------
        out: [N, irreps_out.dim]
        """

        x = self.attn(node_attr, edge_index, edge_attr, edge_sh)
        x = self.norm1(x)
        x = self.ff(x)
        x = self.norm2(x)
        return x
