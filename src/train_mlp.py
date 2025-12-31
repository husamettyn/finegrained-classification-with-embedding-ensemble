import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import argparse
import os
from tqdm import tqdm
from models.mlp import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from datetime import datetime
import numpy as np

def load_embeddings(path):
    data = torch.load(path)
    return data['embeddings'], data['labels']

def evaluate_metrics(model, loader, device, class_names=None):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    # Calculate metrics
    acc = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    
    report = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True, zero_division=0)
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'report': report,
        'predictions': all_preds,
        'targets': all_targets
    }

def plot_confusion_matrix(targets, predictions, save_path, class_names=None):
    cm = confusion_matrix(targets, predictions)
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.close()

def plot_training_history(history, save_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss Plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_plot.png'))
    plt.close()
    
    # Accuracy Plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'], label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'accuracy_plot.png'))
    plt.close()

def train(args):
    device = args.device
    
    # Setup Output Directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    embedding_name = os.path.basename(args.embedding_dir)
    if not embedding_name: # Handle trailing slash
        embedding_name = os.path.basename(os.path.dirname(args.embedding_dir))
        
    output_dir = os.path.join("models", "mlp", embedding_name, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Save training config
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
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
    num_classes = len(torch.unique(train_labels)) 
    print(f"Input Dim: {input_dim}, Classes: {num_classes}")
    
    model = MLPClassifier(input_dim=input_dim, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    best_val_f1 = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        
        # Validation
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_targets = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
                
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_targets.extend(targets.cpu().numpy())
        
        val_loss = val_running_loss / len(test_loader)
        val_acc = val_correct / val_total
        val_f1 = f1_score(all_val_targets, all_val_preds, average='weighted', zero_division=0)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
            
    # Final Evaluation with Best Model
    print("\nEvaluating best model...")
    model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pt")))
    
    metrics = evaluate_metrics(model, test_loader, device)
    
    print(f"\nTest Results:")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    
    # Save metrics
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        # Convert NumPy types to Python types for JSON serialization
        serializable_metrics = {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'report': metrics['report']
        }
        json.dump(serializable_metrics, f, indent=4)
        
    # Plots
    plot_training_history(history, output_dir)
    plot_confusion_matrix(metrics['targets'], metrics['predictions'], 
                         os.path.join(output_dir, 'confusion_matrix.png'))
    
    print(f"\nAll results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_dir", required=True, help="Directory containing train.pt and test.pt")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    train(args)

