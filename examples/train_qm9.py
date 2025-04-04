import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose, Distance, RadiusGraph
from e3conv_net import E3ConvNet  # Assuming E3ConvNet is in a file called e3conv_net.py

# Set up command line arguments
parser = argparse.ArgumentParser(description='Train E3ConvNet on QM9 dataset')
parser.add_argument('--target', type=int, default=0, help='QM9 target property to predict (0-12)')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--max_radius', type=float, default=5.0, help='Maximum radius for edge connections')
parser.add_argument('--hidden_irreps', type=str, default='32x0e + 16x1o + 8x2e', help='Hidden irreps')
parser.add_argument('--sh_irreps', type=str, default='1x0e + 1x1o + 1x2e', help='Spherical harmonics irreps')
parser.add_argument('--num_layers', type=int, default=4, help='Number of message passing layers')
parser.add_argument('--edge_attr_dim', type=int, default=16, help='Edge attribute dimension')
parser.add_argument('--atom_embedding_dim', type=int, default=64, help='Atom embedding dimension')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
args = parser.parse_args()

# Set random seed for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available() and args.device == 'cuda':
    torch.cuda.manual_seed(args.seed)
device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')

# Create directory for saving checkpoints
os.makedirs(args.save_dir, exist_ok=True)

# Define transforms for the QM9 dataset
class SelectTargetProperty(object):
    def __init__(self, target):
        self.target = target
    
    def __call__(self, data):
        # Select the target property and add it as 'y'
        data.y = data.y[:, self.target].unsqueeze(1)
        return data

class SetupBondMask(object):
    def __init__(self):
        pass
    
    def __call__(self, data):
        # Create a binary bond mask (1 for bonded, 0 for non-bonded)
        edge_index = data.edge_index
        bond_mask = torch.zeros(edge_index.shape[1], dtype=torch.long)
        
        # Set edges in the original molecular graph to 1 (bonded)
        if hasattr(data, 'edge_attr') and hasattr(data, 'edge_index'):
            num_bonds = data.edge_attr.shape[0]
            bond_mask[:num_bonds] = 1
            
        data.bond_mask = bond_mask
        return data

# Define dataset transformations
transforms = Compose([
    Distance(),  # Compute pairwise distances
    RadiusGraph(args.max_radius),  # Create edges within max_radius
    SetupBondMask(),  # Setup bond mask
    SelectTargetProperty(args.target)  # Select target property
])

# Load QM9 dataset
path = os.path.join(os.getcwd(), 'data', 'QM9')
dataset = QM9(path, transform=transforms)

# Get the mean and standard deviation of the target property for normalization
target_mean = dataset.data.y.mean(dim=0, keepdim=True)
target_std = dataset.data.y.std(dim=0, keepdim=True)

# Define a transform to normalize the target property
class NormalizeTarget(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, data):
        data.y = (data.y - self.mean) / self.std
        return data

# Apply normalization transform
normalize_transform = Compose([transforms, NormalizeTarget(target_mean, target_std)])
dataset = QM9(path, transform=normalize_transform)

# Split the dataset into training, validation, and test sets
torch.manual_seed(args.seed)
num_samples = len(dataset)
indices = torch.randperm(num_samples)
train_idx = indices[:int(0.8 * num_samples)]
val_idx = indices[int(0.8 * num_samples):int(0.9 * num_samples)]
test_idx = indices[int(0.9 * num_samples):]

train_dataset = dataset[train_idx]
val_dataset = dataset[val_idx]
test_dataset = dataset[test_idx]

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

# Define the model
model = E3ConvNet(
    irreps_out='1x0e',  # Scalar output for regression
    irreps_hidden=args.hidden_irreps,
    irreps_sh=args.sh_irreps,
    num_layers=args.num_layers,
    edge_attr_dim=args.edge_attr_dim,
    atom_type_embedding_dim=args.atom_embedding_dim,
    num_atom_types=dataset.num_atom_types,
    max_radius=args.max_radius
)

# Compile the model
model = torch.compile(model, fullgraph=True, dynamic=True)

# Move model to device
model.to(device)



# Define loss function and optimizer
def loss_fn(pred, target):
    return F.mse_loss(pred, target)

optimizer = Adam(model.parameters(), lr=args.lr)

# Define training function
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        # Prepare data dictionary for the model
        data_dict = {
            'pos': data.pos,
            'edge_index': data.edge_index,
            'bond_mask': data.bond_mask,
            'z': data.z  # Atom types
        }
        
        # Forward pass
        output = model(data_dict)
        loss = loss_fn(output['pred'], data.y)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
    
    return total_loss / len(loader.dataset)

# Define evaluation function
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            
            # Prepare data dictionary for the model
            data_dict = {
                'pos': data.pos,
                'edge_index': data.edge_index,
                'bond_mask': data.bond_mask,
                'z': data.z  # Atom types
            }
            
            # Forward pass
            output = model(data_dict)
            loss = loss_fn(output['pred'], data.y)
            
            total_loss += loss.item() * data.num_graphs
    
    return total_loss / len(loader.dataset)

# Training loop
best_val_loss = float('inf')
for epoch in range(args.epochs):
    # Train
    train_loss = train_epoch(model, train_loader, optimizer, device)
    
    # Validate
    val_loss = evaluate(model, val_loader, device)
    
    # Print progress
    print(f'Epoch: {epoch+1:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    # Save checkpoint if validation loss improves
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'target_mean': target_mean,
            'target_std': target_std,
        }
        torch.save(checkpoint, os.path.join(args.save_dir, f'best_model_target_{args.target}.pt'))

# Test the best model
checkpoint = torch.load(os.path.join(args.save_dir, f'best_model_target_{args.target}.pt'))
model.load_state_dict(checkpoint['model_state_dict'])
test_loss = evaluate(model, test_loader, device)

# Convert normalized test loss back to the original scale
test_loss_unnormalized = test_loss * (target_std.item() ** 2)
test_rmse = np.sqrt(test_loss_unnormalized)

print(f'Test Loss (MSE): {test_loss:.4f}')
print(f'Test RMSE: {test_rmse:.4f}')

# QM9 target properties for reference
target_names = [
    'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve',
    'U0', 'U', 'H', 'G', 'Cv', 'U0_atom', 'U_atom', 'H_atom',
    'G_atom', 'A', 'B', 'C'
]

if args.target < len(target_names):
    print(f'Target property: {target_names[args.target]}')