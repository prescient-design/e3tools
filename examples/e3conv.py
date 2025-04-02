from typing import Dict, Union

import e3nn
import torch
import torch.nn as nn
from e3nn import o3

from e3tools.nn import ConvBlock, EquivariantMLP


class E3Conv(nn.Module):
    """A simple E(3)-equivariant convolutional neural network, similar to NequIP."""

    def __init__(
        self,
        irreps_out: Union[str, e3nn.o3.Irreps],
        irreps_hidden: Union[str, e3nn.o3.Irreps],
        irreps_sh: Union[str, e3nn.o3.Irreps],
        num_layers: int,
        edge_attr_dim: int,
        atom_type_embedding_dim: int,
        num_atom_types: int,
        max_radius: float,
    ):
        super().__init__()

        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_hidden = o3.Irreps(irreps_hidden)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.num_layers = num_layers
        self.edge_attr_dim = edge_attr_dim
        self.max_radius = max_radius

        self.sh = o3.SphericalHarmonics(
            irreps_out=self.irreps_sh, normalize=True, normalization="component"
        )
        self.bonded_edge_attr_dim, self.radial_edge_attr_dim = (
            self.edge_attr_dim // 2,
            (self.edge_attr_dim + 1) // 2,
        )
        self.embed_bondedness = nn.Embedding(2, self.bonded_edge_attr_dim)

        self.atom_embedder = nn.Embedding(
            num_embeddings=num_atom_types,
            embedding_dim=atom_type_embedding_dim,
        )
        self.initial_linear = o3.Linear(
            f"{atom_type_embedding_dim}x0e", self.irreps_hidden
        )

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                ConvBlock(
                    irreps_in=self.irreps_hidden,
                    irreps_out=self.irreps_hidden,
                    irreps_sh=self.irreps_sh,
                    edge_attr_dim=self.edge_attr_dim,
                )
            )
        self.output_head = EquivariantMLP(
            irreps_in=self.irreps_hidden, irreps_out=self.irreps_out
        )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the E3Conv model."""
        # Extract edge attributes.
        pos = data["pos"]
        edge_index = data["edge_index"]
        bond_mask = data["bond_mask"]

        src, dst = edge_index
        edge_vec = pos[src] - pos[dst]
        edge_sh = self.sh(edge_vec)

        # Compute edge attributes.
        bonded_edge_attr = self.embed_bondedness(bond_mask)
        radial_edge_attr = e3nn.math.soft_one_hot_linspace(
            edge_vec.norm(dim=1),
            0.0,
            self.max_radius,
            self.radial_edge_attr_dim,
            basis="gaussian",
            cutoff=True,
        )
        edge_attr = torch.cat((bonded_edge_attr, radial_edge_attr), dim=-1)

        # Compute node attributes.
        node_attr = self.atom_embedder(data)
        node_attr = self.initial_linear(node_attr)

        # Perform message passing.
        for layer in self.layers:
            node_attr = layer(node_attr, edge_index, edge_attr, edge_sh)
        node_attr = self.output_head(node_attr)

        data["pred"] = node_attr
        return data
