import torch
import e3nn.o3


class ExtractIrreps(torch.nn.Module):
    """Extracts specific irreps from a e3nn tensor."""

    def __init__(self, irreps_in: e3nn.o3.Irreps, irreps_extract: e3nn.o3.Irrep):
        super().__init__()

        self.irreps_in = e3nn.o3.Irreps(irreps_in)
        self.irreps_extract = e3nn.o3.Irrep(irreps_extract)

        irreps_out = e3nn.o3.Irreps()
        slices = []
        for (mul, ir), ir_slice in zip(self.irreps_in, self.irreps_in.slices()):
            if ir.l == self.irreps_extract.l and ir.p == self.irreps_extract.p:
                slices.append(ir_slice)
                irreps_out += e3nn.o3.Irreps(f"{mul}x{ir}")

        if len(slices) == 0:
            raise ValueError(f"irreps {irreps_extract} not found in {irreps_in}")

        self.slices = slices
        self.irreps_out = irreps_out

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return torch.cat([data[..., s] for s in self.slices], dim=-1)
