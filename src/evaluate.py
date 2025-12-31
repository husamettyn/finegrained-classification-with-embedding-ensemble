import argparse
import torch
from src.models.encoders import CLIPEncoder, ViTEncoder, ResNetEncoder
from src.models.ensemble import EnsembleClassifier
from src.data.loader import get_dataloaders
from src.training.trainer import evaluate_model

def main():
    parser = argparse.ArgumentParser(description="Evaluate model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model")
    parser.add_argument("--dataset", type=str, default="beans", help="HuggingFace dataset name")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Flags to determine architecture
    parser.add_argument("--ensemble", action="store_true", help="Is it the ensemble model?")
    parser.add_argument("--encoder", type=str, help="If single model, which encoder? (clip, vit, resnet)")
    
    args = parser.parse_args()
    
    if not args.ensemble and not args.encoder:
        print("Please specify either --ensemble or --encoder <type>")
        return

    # Recreate Encoders
    print("Initializing encoders...")
    encoders_dict = {}
    
    if args.ensemble:
        encoders_dict['clip'] = CLIPEncoder(device=args.device)
        encoders_dict['vit'] = ViTEncoder(device=args.device)
        encoders_dict['resnet'] = ResNetEncoder(device=args.device)
    else:
        if args.encoder == 'clip':
            encoders_dict['clip'] = CLIPEncoder(device=args.device)
        elif args.encoder == 'vit':
            encoders_dict['vit'] = ViTEncoder(device=args.device)
        elif args.encoder == 'resnet':
            encoders_dict['resnet'] = ResNetEncoder(device=args.device)

    # Load Data (we need test loader)
    # Note: get_dataloaders returns train, test/val. We need the second one.
    print(f"Loading dataset {args.dataset}...")
    _, test_loader, num_classes = get_dataloaders(args.dataset, encoders_dict, batch_size=args.batch_size)
    
    if test_loader is None:
        return

    # Recreate Model
    model = EnsembleClassifier(encoders_dict, num_classes=num_classes)
    
    # Load Weights
    print(f"Loading weights from {args.model_path}...")
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    except Exception as e:
        print(f"Error loading weights: {e}")
        return
        
    model.to(args.device)
    
    # Evaluate
    acc = evaluate_model(model, test_loader, args.device)
    print(f"Test Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()

