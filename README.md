# e3tools

A repository of building blocks in PyTorch for E(3)/SE(3)-equivariant neural networks, built on top of [e3nn](https://github.com/e3nn/e3nn):
- Equivariant Convolution: [`e3tools.nn.Conv`](https://github.com/prescient-design/e3tools/blob/main/src/e3tools/nn/_conv.py#L16)
- Equivariant Multi-Layer Perceptrons (MLPs): [`e3tools.nn.EquivariantMLP`](https://github.com/prescient-design/e3tools/blob/main/src/e3tools/nn/_mlp.py#L86)
- Equivariant Layer Norm: [`e3tools.nn.LayerNorm`](https://github.com/prescient-design/e3tools/blob/main/src/e3tools/nn/_layer_norm.py#L9)
- Equivariant Activations: [`e3tools.nn.Gate`](https://github.com/prescient-design/e3tools/blob/main/src/e3tools/nn/_gate.py#L10) and [`e3tools.nn.Gated`](https://github.com/prescient-design/e3tools/blob/main/src/e3tools/nn/_gate.py#L68)
- Separable Equivariant Tensor Products: [`e3tools.nn.SeparableTensorProduct`](https://github.com/prescient-design/e3tools/blob/main/src/e3tools/nn/_tensor_product.py#L8)
- Extracting Irreps: [`e3tools.nn.ExtractIrreps`](https://github.com/prescient-design/e3tools/blob/main/src/e3tools/nn/_extract_irreps.py#L5)
- Self-Interactions: [`e3tools.nn.LinearSelfInteraction`](https://github.com/prescient-design/e3tools/blob/main/src/e3tools/nn/_interaction.py#L5)

All modules are compatible with PyTorch 2.0's `torch.compile` for JIT compilation!

## Installation

```bash
pip install e3tools
```

## Examples

See [examples/e3conv.py](https://github.com/prescient-design/e3tools/blob/main/examples/e3conv.py) for an example
of a simple E(3)-equivariant message passing network built with `e3tools`.
