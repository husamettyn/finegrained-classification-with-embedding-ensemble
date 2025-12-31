import argparse
import torch
from src.models.encoders import CLIPEncoder, ViTEncoder, ResNetEncoder
from src.models.ensemble import EnsembleClassifier
from src.data.loader import get_dataloaders
from src.training.trainer import train_model

def main():
    parser = argparse.ArgumentParser(description="Train ensemble model")
    parser.add_argument("--dataset", type=str, default="beans", help="HuggingFace dataset name")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16) # Smaller batch size due to multiple models
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--freeze_encoders", action="store_true", help="Freeze encoder weights")
    args = parser.parse_args()
    
    # Initialize Encoders
    print("Initializing encoders...")
    clip = CLIPEncoder(device=args.device)
    vit = ViTEncoder(device=args.device)
    resnet = ResNetEncoder(device=args.device)
    
    encoders_dict = {
        'clip': clip,
        'vit': vit,
        'resnet': resnet
    }
    
    # Freeze encoders if requested
    if args.freeze_encoders:
        for name, encoder in encoders_dict.items():
            for param in encoder.parameters():
                param.requires_grad = False
            print(f"Freezed {name}")
    
    # Load Data
    print(f"Loading dataset {args.dataset}...")
    train_loader, val_loader, num_classes = get_dataloaders(args.dataset, encoders_dict, batch_size=args.batch_size)
    
    if train_loader is None:
        return

    print(f"Number of classes: {num_classes}")
    
    # Create Ensemble Model
    model = EnsembleClassifier(encoders_dict, num_classes=num_classes)
    
    # Train
    save_path = f"models/ensemble_{args.dataset}_best.pt"
    train_model(model, train_loader, val_loader, args.epochs, args.lr, args.device, save_path)

if __name__ == "__main__":
    main()

