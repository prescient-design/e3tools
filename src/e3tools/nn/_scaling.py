import torch
import e3nn


class ScaleIrreps(torch.nn.Module):
    """Scales each irrep by a weight."""

    def __init__(self, irreps_in: torch.Tensor):
        super().__init__()

        self.irreps_in = e3nn.o3.Irreps(irreps_in)
        self.irreps_out = e3nn.o3.Irreps(irreps_in)
        self.tp = e3nn.o3.ElementwiseTensorProduct(
            self.irreps_in,
            f"{self.irreps_in.num_irreps}x0e",
        )

    def forward(self, data: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Scales each irrep by a weight.

        Parameters:
            data: torch.Tensor of shape [..., irreps_in.dim]
            weights: torch.Tensor of shape [..., irreps_in.num_irreps]

        Returns:
            torch.Tensor of shape [..., irreps_in.dim]
        """
        return self.tp(data, weights)
