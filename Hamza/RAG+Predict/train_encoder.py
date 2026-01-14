"""
Train the graph encoder using contrastive learning.
Uses InfoNCE (NT-Xent) loss to learn embeddings where similar molecules
(based on text descriptions) are closer in embedding space.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch_geometric.data import Batch
from sentence_transformers import SentenceTransformer

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from models.graph_encoder import MoleculeGINE
import config


class ContrastiveGraphDataset(Dataset):
    """Dataset for contrastive learning on molecular graphs."""
    
    def __init__(self, graph_path, text_model_name='all-MiniLM-L6-v2'):
        """
        Args:
            graph_path: Path to pickle file with graphs
            text_model_name: Sentence transformer model for text embeddings
        """
        print(f"Loading graphs from {graph_path}")
        with open(graph_path, 'rb') as f:
            data = pickle.load(f)
        
        # Support both list-of-graphs and dict format
        if isinstance(data, dict):
            self.graphs = data.get('graphs', [])
            # Prefer provided descriptions if present; else derive from graphs
            self.descriptions = data.get('descriptions', [getattr(g, 'description', '') for g in self.graphs])
        else:
            self.graphs = data
            self.descriptions = [getattr(g, 'description', '') for g in self.graphs]
        
        print(f"Loaded {len(self.graphs)} graphs")
        
        # Load text encoder for computing text similarity
        print(f"Loading text encoder: {text_model_name}")
        self.text_encoder = SentenceTransformer(text_model_name)
        
        # Precompute text embeddings for similarity computation
        print("Computing text embeddings for all descriptions...")
        self.text_embeddings = self.text_encoder.encode(
            self.descriptions,
            show_progress_bar=True,
            convert_to_tensor=True,
            device='cpu'
        )
        print(f"Text embeddings shape: {self.text_embeddings.shape}")
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return {
            'graph': self.graphs[idx],
            'text_embedding': self.text_embeddings[idx],
            'description': self.descriptions[idx]
        }


def collate_contrastive_batch(batch):
    """Collate function for batching graphs and text embeddings."""
    graphs = [item['graph'] for item in batch]
    text_embeddings = torch.stack([item['text_embedding'] for item in batch])
    
    # Batch graphs using PyG
    batched_graphs = Batch.from_data_list(graphs)
    
    return {
        'graphs': batched_graphs,
        'text_embeddings': text_embeddings
    }


class InfoNCELoss(nn.Module):
    """
    InfoNCE (NT-Xent) contrastive loss.
    Pulls together graph embeddings with similar text descriptions,
    pushes apart those with dissimilar descriptions.
    """
    
    def __init__(self, temperature=0.07, text_dim=384, graph_dim=256):
        super().__init__()
        self.temperature = temperature
        # Project text embeddings to graph embedding space
        self.text_projection = nn.Linear(text_dim, graph_dim)
    
    def forward(self, graph_embeddings, text_embeddings):
        """
        Args:
            graph_embeddings: [batch_size, embed_dim]
            text_embeddings: [batch_size, text_embed_dim]
        
        Returns:
            loss: InfoNCE loss value
        """
        batch_size = graph_embeddings.size(0)
        
        # Project text embeddings to same dimension as graph embeddings
        text_embeddings = self.text_projection(text_embeddings)
        
        # Normalize embeddings
        graph_embeddings = F.normalize(graph_embeddings, dim=1)
        text_embeddings = F.normalize(text_embeddings, dim=1)
        
        # Compute similarity matrix between graphs and texts
        # [batch_size, batch_size]
        similarity_matrix = torch.matmul(graph_embeddings, text_embeddings.T) / self.temperature
        
        # Labels: diagonal elements are positive pairs
        labels = torch.arange(batch_size, device=similarity_matrix.device)
        
        # Symmetric loss: graph->text and text->graph
        loss_g2t = F.cross_entropy(similarity_matrix, labels)
        loss_t2g = F.cross_entropy(similarity_matrix.T, labels)
        
        loss = (loss_g2t + loss_t2g) / 2
        
        return loss


class TripletMarginLoss(nn.Module):
    """
    Triplet loss: anchor, positive, negative.
    Positive = similar description, Negative = dissimilar description.
    """
    
    def __init__(self, margin=0.5, graph_dim=256, text_dim=384):
        super().__init__()
        self.margin = margin
        # Project graph embeddings to text embedding space
        self.graph_projector = nn.Linear(graph_dim, text_dim)
    
    def forward(self, graph_embeddings, text_embeddings):
        """
        Args:
            graph_embeddings: [batch_size, embed_dim]
            text_embeddings: [batch_size, text_embed_dim] 
        
        Returns:
            loss: Triplet loss value
        """
        batch_size = graph_embeddings.size(0)
        
        # Project graph embeddings to text space
        graph_embeddings = self.graph_projector(graph_embeddings)
        
        # Normalize embeddings
        graph_embeddings = F.normalize(graph_embeddings, dim=1)
        text_embeddings = F.normalize(text_embeddings, dim=1)
        
        # Compute pairwise distances in text space
        text_sim = torch.matmul(text_embeddings, text_embeddings.T)
        
        # For each anchor, find hardest positive and negative
        losses = []
        for i in range(batch_size):
            anchor = graph_embeddings[i]
            
            # Positive: most similar text (excluding self)
            pos_idx = torch.argmax(text_sim[i] * (torch.arange(batch_size, device=text_sim.device) != i).float())
            positive = graph_embeddings[pos_idx]
            
            # Negative: least similar text
            neg_idx = torch.argmin(text_sim[i] + (torch.arange(batch_size, device=text_sim.device) == i).float() * 1e9)
            negative = graph_embeddings[neg_idx]
            
            # Compute distances
            pos_dist = torch.sum((anchor - positive) ** 2)
            neg_dist = torch.sum((anchor - negative) ** 2)
            
            # Triplet loss
            loss = F.relu(pos_dist - neg_dist + self.margin)
            losses.append(loss)
        
        return torch.stack(losses).mean()


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        graphs = batch['graphs'].to(device)
        text_embeddings = batch['text_embeddings'].to(device)
        
        # Forward pass
        graph_embeddings = model(
            graphs.x,
            graphs.edge_index,
            graphs.edge_attr,
            graphs.batch
        )
        
        # Compute loss
        loss = criterion(graph_embeddings, text_embeddings)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        graphs = batch['graphs'].to(device)
        text_embeddings = batch['text_embeddings'].to(device)
        
        # Forward pass
        graph_embeddings = model(
            graphs.x,
            graphs.edge_index,
            graphs.edge_attr,
            graphs.batch
        )
        
        # Compute loss
        loss = criterion(graph_embeddings, text_embeddings)
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f"Checkpoint saved to {path}")


def main(graph_path=None, epochs=30, batch_size=32, lr=1e-4, save_dir=None):
    print("="*60)
    print("Contrastive Graph Encoder Training")
    print("="*60)
    
    # Use config defaults if not provided
    if graph_path is None:
        graph_path = config.TRAIN_GRAPHS
    if save_dir is None:
        save_dir = config.EMBEDDINGS_DIR
    
    # Hyperparameters
    BATCH_SIZE = batch_size
    EPOCHS = epochs
    LR = lr
    TEMPERATURE = 0.07  # For InfoNCE
    LOSS_TYPE = "infonce"  # Options: "infonce", "triplet"
    
    print(f"\nHyperparameters:")
    print(f"  Graph path: {graph_path}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning rate: {LR}")
    print(f"  Loss type: {LOSS_TYPE}")
    print(f"  Device: {config.DEVICE}")
    
    # Load datasets
    train_dataset = ContrastiveGraphDataset(graph_path)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_contrastive_batch,
        num_workers=0
    )
    
    # Get dimensions from first graph
    sample_graph = train_dataset[0]['graph']
    node_dim = sample_graph.x.size(1)
    edge_dim = sample_graph.edge_attr.size(1)
    text_dim = train_dataset.text_embeddings.size(1)
    
    print(f"\nGraph dimensions:")
    print(f"  Node features: {node_dim}")
    print(f"  Edge features: {edge_dim}")
    print(f"  Text embedding dim: {text_dim}")
    
    # Initialize model
    model = MoleculeGINE(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden=config.GRAPH_HIDDEN_DIM,
        layers=config.GRAPH_LAYERS,
        out_dim=config.GRAPH_HIDDEN_DIM
    ).to(config.DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")
    
    # Loss function
    if LOSS_TYPE == "infonce":
        criterion = InfoNCELoss(
            temperature=TEMPERATURE,
            text_dim=text_dim,
            graph_dim=config.GRAPH_HIDDEN_DIM
        ).to(config.DEVICE)
    elif LOSS_TYPE == "triplet":
        criterion = TripletMarginLoss(margin=0.5)
    else:
        raise ValueError(f"Unknown loss type: {LOSS_TYPE}")
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    
    # Training loop
    best_loss = float('inf')
    checkpoint_dir = os.path.join(save_dir, "encoder_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{EPOCHS}")
        print(f"{'='*60}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config.DEVICE, epoch)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Save best model
        if train_loss < best_loss:
            best_loss = train_loss
            checkpoint_path = os.path.join(checkpoint_dir, "best_encoder.pt")
            save_checkpoint(model, optimizer, epoch, train_loss, checkpoint_path)
        
        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"encoder_epoch_{epoch}.pt")
            save_checkpoint(model, optimizer, epoch, train_loss, checkpoint_path)
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved in: {checkpoint_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
