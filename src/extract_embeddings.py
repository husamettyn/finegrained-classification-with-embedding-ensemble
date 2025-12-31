import torch
import os
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from data.dataset import CUB200Dataset
from models.extractors import get_extractor

def extract_and_save(extractor, loader, save_path, device):
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"Extracting features"):
            images = images.to(device)
            features = extractor(images)
            
            all_embeddings.append(features.cpu())
            all_labels.append(labels)
            
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({'embeddings': all_embeddings, 'labels': all_labels}, save_path)
    print(f"Saved {all_embeddings.shape[0]} embeddings to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="dataset", help="Path to dataset root")
    parser.add_argument("--model", required=True, choices=['dinov2', 'dinov3', 'openclip', 'siglip', 'convnext'], help="Model to use")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    print(f"Loading model {args.model}...")
    extractor = get_extractor(args.model, args.device)
    
    # Transforms
    transform = extractor.get_transform()
    
    # Datasets
    print("Loading datasets...")
    train_ds = CUB200Dataset(args.dataset_dir, split='train', transform=transform)
    test_ds = CUB200Dataset(args.dataset_dir, split='test', transform=transform)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Paths
    save_dir = os.path.join("embeddings", args.model)
    
    print("Extracting train embeddings...")
    extract_and_save(extractor, train_loader, os.path.join(save_dir, "train.pt"), args.device)
    
    print("Extracting test embeddings...")
    extract_and_save(extractor, test_loader, os.path.join(save_dir, "test.pt"), args.device)

if __name__ == "__main__":
    main()

