import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import argparse
import os
from tqdm import tqdm
from src.models.mlp import MLPClassifier

def load_embeddings(path):
    data = torch.load(path)
    return data['embeddings'], data['labels']

def train(args):
    device = args.device
    
    # Load Embeddings
    print(f"Loading embeddings from {args.embedding_dir}...")
    train_emb, train_labels = load_embeddings(os.path.join(args.embedding_dir, "train.pt"))
    test_emb, test_labels = load_embeddings(os.path.join(args.embedding_dir, "test.pt"))
    
    # Create DataLoaders
    train_ds = TensorDataset(train_emb, train_labels)
    test_ds = TensorDataset(test_emb, test_labels)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    
    # Initialize Model
    input_dim = train_emb.shape[1]
    num_classes = len(torch.unique(train_labels)) # Should be 200
    print(f"Input Dim: {input_dim}, Classes: {num_classes}")
    
    model = MLPClassifier(input_dim=input_dim, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        acc = correct / total
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {running_loss/len(train_loader):.4f} - Test Acc: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            os.makedirs("models/mlp", exist_ok=True)
            save_name = os.path.basename(args.embedding_dir) # e.g. dinov2
            torch.save(model.state_dict(), f"models/mlp/best_mlp_{save_name}.pt")
            
    print(f"Best Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_dir", required=True, help="Directory containing train.pt and test.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    train(args)

