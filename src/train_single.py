import argparse
import torch
import os
from src.models.encoders import CLIPEncoder, ViTEncoder, ResNetEncoder
from src.models.ensemble import EnsembleClassifier
from src.data.loader import get_dataloaders
from src.training.trainer import train_model

def main():
    parser = argparse.ArgumentParser(description="Train a single encoder model")
    parser.add_argument("--encoder", type=str, required=True, choices=['clip', 'vit', 'resnet'], help="Encoder type")
    parser.add_argument("--dataset", type=str, default="beans", help="HuggingFace dataset name")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    # Initialize Encoder
    print(f"Initializing {args.encoder} encoder...")
    if args.encoder == 'clip':
        encoder = CLIPEncoder(device=args.device)
    elif args.encoder == 'vit':
        encoder = ViTEncoder(device=args.device)
    elif args.encoder == 'resnet':
        encoder = ResNetEncoder(device=args.device)
        
    encoders_dict = {args.encoder: encoder}
    
    # Load Data
    print(f"Loading dataset {args.dataset}...")
    train_loader, val_loader, num_classes = get_dataloaders(args.dataset, encoders_dict, batch_size=args.batch_size)
    
    if train_loader is None:
        return
        
    print(f"Number of classes: {num_classes}")
    
    # Create Model (Single encoder ensemble)
    model = EnsembleClassifier(encoders_dict, num_classes=num_classes)
    
    # Train
    save_path = f"models/{args.encoder}_{args.dataset}_best.pt"
    train_model(model, train_loader, val_loader, args.epochs, args.lr, args.device, save_path)

if __name__ == "__main__":
    main()

