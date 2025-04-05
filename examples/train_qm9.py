import os
import argparse
import numpy as np
import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric.datasets
import torch_geometric.loader
import torch_geometric.transforms
import e3nn
import e3tools

from models import E3ConvNet, E3Transformer

e3nn.set_optimization_defaults(jit_script_fx=False)
torch.set_float32_matmul_precision("high")

# Set up command line arguments
parser = argparse.ArgumentParser(description="Train on QM9 dataset")
parser.add_argument(
    "--target", type=int, default=0, help="QM9 target property to predict (0-12)"
)
parser.add_argument(
    "--batch_size", type=int, default=32, help="Batch size for training"
)
parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument(
    "--max_radius", type=float, default=5.0, help="Maximum radius for edge connections"
)
parser.add_argument(
    "--hidden_irreps", type=str, default="64x0e + 16x1o", help="Hidden irreps"
)
parser.add_argument(
    "--sh_irreps", type=str, default="1x0e + 1x1o", help="Spherical harmonics irreps"
)
parser.add_argument(
    "--num_layers", type=int, default=4, help="Number of message passing layers"
)
parser.add_argument(
    "--edge_attr_dim", type=int, default=64, help="Edge attribute dimension"
)
parser.add_argument(
    "--atom_embedding_dim", type=int, default=64, help="Atom embedding dimension"
)
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument(
    "--device", type=str, default="cuda", help="Device to use (cuda or cpu)"
)
parser.add_argument(
    "--save_dir",
    type=str,
    default="./checkpoints",
    help="Directory to save checkpoints",
)
parser.add_argument(
    "--model",
    type=str,
    choices=["conv", "transformer"],
    default="transformer",
    help="Model type to use (E3ConvNet or E3Transformer)",
)
args = parser.parse_args()

# Set random seed for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available() and args.device == "cuda":
    torch.cuda.manual_seed(args.seed)
device = torch.device(
    args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
)

# Create directory for saving checkpoints
os.makedirs(args.save_dir, exist_ok=True)


# Define transforms for the QM9 dataset
class RadiusGraph(torch_geometric.transforms.BaseTransform):
    def __init__(self, r):
        self.r = r

    def __call__(self, data):
        # Create edges based on radius
        data.edge_index = e3tools.radius_graph(data.pos, self.r, data.batch)
        return data


class SelectTargetProperty(torch_geometric.transforms.BaseTransform):
    def __init__(self, target):
        self.target = target

    def __call__(self, data):
        # Select the target property and add it as 'y'
        data.y = data.y[:, self.target].unsqueeze(1)
        return data


class SetupBondMask(torch_geometric.transforms.BaseTransform):
    def __init__(self):
        pass

    def __call__(self, data):
        # Create a binary bond mask (1 for bonded, 0 for non-bonded)
        edge_index = data.edge_index
        bond_mask = torch.zeros(edge_index.shape[1], dtype=torch.long)

        # Set edges in the original molecular graph to 1 (bonded)
        if hasattr(data, "edge_attr"):
            num_bonds = data.edge_attr.shape[0]
            bond_mask[:num_bonds] = 1

        data.bond_mask = bond_mask
        return data


# Define a transform to normalize the target property
class NormalizeTarget(torch_geometric.transforms.BaseTransform):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        data.y = (data.y - self.mean) / self.std
        return data


# Define training function
def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    """Train the model for one epoch."""
    model.train()

    with tqdm.tqdm(
        loader,
        total=len(loader.dataset) // args.batch_size,
        desc="Training",
        leave=False,
    ) as pbar:
        for data in pbar:
            data = data.to(device)
            optimizer.zero_grad()

            # Forward pass
            output = model(data)
            loss = F.mse_loss(output[:, 0], data.y[:, args.target])

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=loss.item(), refresh=False)
            pbar.update(1)


# Define evaluation function
def evaluate(
    model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device
) -> float:
    """Evaluate the model on the validation/test set."""
    total_loss = 0

    with torch.inference_mode():
        with tqdm.tqdm(
            loader,
            total=len(loader.dataset) // args.batch_size,
            desc="Evaluating",
            leave=False,
            miniters=10,
        ) as pbar:
            for data in pbar:
                data = data.to(device)

                # Forward pass
                output = model(data)
                loss = F.mse_loss(output[:, 0], data.y[:, args.target])

                total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)


# Define dataset transformations
transforms = torch_geometric.transforms.Compose(
    [
        RadiusGraph(r=args.max_radius),  # Create radius graph
        SetupBondMask(),  # Setup bond mask
        SelectTargetProperty(args.target),  # Select target property
    ]
)

if __name__ == "__main__":
    # Load QM9 dataset
    path = os.path.join(os.getcwd(), "data", "QM9")
    dataset = torch_geometric.datasets.QM9(path, transform=transforms)

    # Get the mean and standard deviation of the target property for normalization
    target_mean = dataset.y.mean(dim=0)
    target_std = dataset.y.std(dim=0)

    # Apply normalization transform
    normalize_transform = torch_geometric.transforms.Compose(
        [transforms, NormalizeTarget(target_mean, target_std)]
    )
    dataset = torch_geometric.datasets.QM9(path, transform=normalize_transform)

    # Split the dataset into training, validation, and test sets
    torch.manual_seed(args.seed)
    num_samples = len(dataset)
    indices = torch.randperm(num_samples)
    train_idx = indices[: int(0.8 * num_samples)]
    val_idx = indices[int(0.8 * num_samples) : int(0.9 * num_samples)]
    test_idx = indices[int(0.9 * num_samples) :]

    train_dataset = dataset[train_idx]
    val_dataset = dataset[val_idx]
    test_dataset = dataset[test_idx]

    # Create data loaders
    train_loader = torch_geometric.loader.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = torch_geometric.loader.DataLoader(
        val_dataset, batch_size=args.batch_size
    )
    test_loader = torch_geometric.loader.DataLoader(
        test_dataset, batch_size=args.batch_size
    )

    # Define the model
    if args.model == "conv":
        model = E3ConvNet(
            irreps_out="1x0e",  # Scalar output for regression
            irreps_hidden=args.hidden_irreps,
            irreps_sh=args.sh_irreps,
            num_layers=args.num_layers,
            edge_attr_dim=args.edge_attr_dim,
            atom_type_embedding_dim=args.atom_embedding_dim,
            num_atom_types=dataset.z.max().item() + 1,
            max_radius=args.max_radius,
        )
    elif args.model == "transformer":
        model = E3Transformer(
            irreps_out="1x0e",  # Scalar output for regression
            irreps_hidden=args.hidden_irreps,
            irreps_sh=args.sh_irreps,
            num_layers=args.num_layers,
            edge_attr_dim=args.edge_attr_dim,
            atom_type_embedding_dim=args.atom_embedding_dim,
            num_atom_types=dataset.z.max().item() + 1,
            max_radius=args.max_radius,
            num_attention_heads=1,
        )

    # Compile the model
    model = torch.compile(model, fullgraph=True, dynamic=True)

    # Move model to device
    model.to(device)

    # Call the model to avoid compilation time during training
    dummy_data = next(iter(train_loader)).to(device)
    _ = model(dummy_data)

    # Print model summary
    print(
        model,
        "with parameters:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    # Define loss function and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # QM9 target properties for reference
    target_names = [
        "mu",
        "alpha",
        "homo",
        "lumo",
        "gap",
        "r2",
        "zpve",
        "U0",
        "U",
        "H",
        "G",
        "Cv",
        "U0_atom",
        "U_atom",
        "H_atom",
        "G_atom",
        "A",
        "B",
        "C",
    ]

    print(
        f"Target property: '{target_names[args.target]}' with mean {target_mean[args.target]:0.2f} ± {target_std[args.target]:0.2f}"
    )

    # Training loop
    best_val_loss = float("inf")
    with tqdm.trange(args.epochs, desc="Epochs") as pbar:
        for epoch in pbar:
            train_epoch(model, train_loader, optimizer, device)
            val_loss = evaluate(model, val_loader, device)

            # Save checkpoint if validation loss improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "target_mean": target_mean,
                    "target_std": target_std,
                }
                torch.save(
                    checkpoint,
                    os.path.join(args.save_dir, f"best_model_target_{args.target}.pt"),
                )

            pbar.set_postfix(val_loss=val_loss)

    # Test the best model
    checkpoint = torch.load(
        os.path.join(args.save_dir, f"best_model_target_{args.target}.pt")
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss = evaluate(model, test_loader, device)

    # Convert normalized test loss back to the original scale
    test_loss_unnormalized = test_loss * (target_std.item() ** 2)
    test_rmse = np.sqrt(test_loss_unnormalized)

    print(f"Test Loss (MSE): {test_loss:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
