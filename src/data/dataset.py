import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch

class CUB200Dataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (string): Root directory of the CUB_200_2011 dataset.
            split (string): 'train' or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = os.path.join(root_dir, 'CUB_200_2011')
        self.transform = transform
        self.split = split
        
        if not os.path.exists(self.root_dir):
            # Fallback if the folder structure is slightly different (e.g. directly in root)
            if os.path.exists(os.path.join(root_dir, 'images')):
                 self.root_dir = root_dir
            else:
                raise FileNotFoundError(f"CUB dataset not found at {self.root_dir}")

        # Load metadata
        self.images = pd.read_csv(os.path.join(self.root_dir, 'images.txt'), sep=' ', names=['img_id', 'filepath'])
        self.image_class_labels = pd.read_csv(os.path.join(self.root_dir, 'image_class_labels.txt'), sep=' ', names=['img_id', 'target'])
        self.train_test_split = pd.read_csv(os.path.join(self.root_dir, 'train_test_split.txt'), sep=' ', names=['img_id', 'is_training_image'])

        # Merge
        self.data = self.images.merge(self.image_class_labels, on='img_id')
        self.data = self.data.merge(self.train_test_split, on='img_id')

        # Filter by split
        if self.split == 'train':
            self.data = self.data[self.data['is_training_image'] == 1]
        elif self.split == 'test':
            self.data = self.data[self.data['is_training_image'] == 0]
        
        # Targets are 1-indexed in file, make them 0-indexed
        self.data['target'] = self.data['target'] - 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.root_dir, 'images', row['filepath'])
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image or handle gracefully? 
            # For now, let's assume valid data or crash
            raise e

        label = row['target']

        if self.transform:
            image = self.transform(image)

        return image, label

