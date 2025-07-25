# e3tools

A repository of building blocks in PyTorch 2.0 for E(3)/SE(3)-equivariant neural networks, built on top of [e3nn](https://github.com/e3nn/e3nn):
- Equivariant Convolution: [`e3tools.nn.Conv`](https://github.com/prescient-design/e3tools/blob/main/src/e3tools/nn/_conv.py#L16) and [`e3tools.nn.SeparableConv`](https://github.com/prescient-design/e3tools/blob/main/src/e3tools/nn/_conv.py#L124)
- Equivariant Multi-Layer Perceptrons (MLPs): [`e3tools.nn.EquivariantMLP`](https://github.com/prescient-design/e3tools/blob/main/src/e3tools/nn/_mlp.py#L86)
- Equivariant Layer Norm: [`e3tools.nn.LayerNorm`](https://github.com/prescient-design/e3tools/blob/main/src/e3tools/nn/_layer_norm.py#L9)
- Equivariant Activations: [`e3tools.nn.Gate`](https://github.com/prescient-design/e3tools/blob/main/src/e3tools/nn/_gate.py#L10), [`e3tools.nn.GateWrapper`](https://github.com/prescient-design/e3tools/blob/main/src/e3tools/nn/_gate.py#L117) and [`e3tools.nn.Gated`](https://github.com/prescient-design/e3tools/blob/main/src/e3tools/nn/_gate.py#L68)
- Separable Equivariant Tensor Products: [`e3tools.nn.SeparableTensorProduct`](https://github.com/prescient-design/e3tools/blob/main/src/e3tools/nn/_tensor_product.py#L8)
- Extracting Irreps: [`e3tools.nn.ExtractIrreps`](https://github.com/prescient-design/e3tools/blob/main/src/e3tools/nn/_extract_irreps.py#L5)
- Self-Interactions: [`e3tools.nn.LinearSelfInteraction`](https://github.com/prescient-design/e3tools/blob/main/src/e3tools/nn/_interaction.py#L5)

All modules are compatible with `torch.compile` for JIT compilation.

## Installation

Install from PyPI:

```bash
pip install e3tools
```

or get the latest development version from GitHub:
```bash
pip install git+https://github.com/prescient-design/e3tools.git
```

## Examples

We provide examples of a [convolution-based](https://github.com/prescient-design/e3tools/blob/main/examples/models/conv.py) and [attention-based](https://github.com/prescient-design/e3tools/blob/main/examples/models/transformer.py) E(3)-equivariant message passing networks built with `e3tools`. We also provide an [example training script on QM9](https://github.com/prescient-design/e3tools/blob/main/examples/train_qm9.py):
```bash
python examples/train_qm9.py --model conv
```

We see an approximate 2.5x improvement in training speed with `torch.compile`.
