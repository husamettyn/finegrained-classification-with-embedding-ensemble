import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

def get_dataloaders(dataset_name, encoders_dict, batch_size=32, num_workers=4, split_ratios={'train': 0.8, 'val': 0.1, 'test': 0.1}):
    """
    dataset_name: Name of HuggingFace dataset (e.g. 'beans', 'cifar10')
    encoders_dict: Dict of {name: encoder_instance} to get transforms from
    """
    
    # Load dataset
    # Handling typical dataset structures (train/test splits)
    # If dataset only has 'train', we split it.
    try:
        ds = load_dataset(dataset_name)
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        return None, None, None

    # Helper to get image column name
    # Common names: 'image', 'img', 'file'
    img_key = 'image'
    for k in ds['train'].features.keys():
        if k in ['image', 'img']:
            img_key = k
            break
            
    label_key = 'label'
    if 'labels' in ds['train'].features:
        label_key = 'labels'

    # Get transforms for each encoder
    transforms = {}
    for name, encoder in encoders_dict.items():
        transforms[name] = encoder.get_transform()

    def transform_fn(examples):
        outputs = {}
        images = [x.convert("RGB") for x in examples[img_key]]
        
        for name, trans in transforms.items():
            # Handle different types of transforms
            # 1. HuggingFace Processor (callable, returns dict with pixel_values)
            # 2. Torchvision Transform (callable, returns Tensor)
            
            try:
                # Check if it's a HF processor by looking for 'preprocess' or similar, 
                # but simplest is to try calling it on list of images
                # HF processors usually handle list of PIL images
                
                # We need to distinguish between torchvision transform (single img) and HF processor (batch)
                # But here we are inside a batch map function.
                
                # Heuristic: HF processors usually have 'feature_extractor_type' or similar attributes, 
                # or we check if it is a torchvision transform (nn.Module or function).
                # Let's assume wrappers handle this? No, wrappers return the raw processor/transform.
                
                # Let's apply individually for robustness if unsure, though slower for HF processors.
                # BETTER: Check name/type
                
                if hasattr(trans, 'preprocess') or hasattr(trans, '__call__'):
                     # Try batch processing first if it looks like a HF processor
                    if hasattr(trans, 'image_mean'): # Likely HF feature extractor
                         # HF processors return BatchFeature, we want the tensor 'pixel_values'
                         encoded = trans(images=images, return_tensors="pt")
                         outputs[name] = encoded['pixel_values']
                    else:
                        # Likely torchvision or simple callable
                        # Apply to each image and stack
                        processed = [trans(img) for img in images]
                        outputs[name] = torch.stack(processed)
                        
            except Exception:
                # Fallback
                processed = [trans(img) for img in images]
                outputs[name] = torch.stack(processed)

        outputs['labels'] = torch.tensor(examples[label_key])
        return outputs

    # Combine splits if needed or respect existing splits
    # Simplified: Use 'train' for train, split 'test' or 'validation' if exists
    
    # Note: set_transform is applied on the fly
    
    if 'train' in ds:
        train_ds = ds['train']
    else:
        # Fallback if no train split ??
        train_ds = ds[list(ds.keys())[0]]

    if 'test' in ds:
        test_ds = ds['test']
    elif 'validation' in ds:
        test_ds = ds['validation']
    else:
        # Split train
        splits = train_ds.train_test_split(test_size=0.2)
        train_ds = splits['train']
        test_ds = splits['test']

    # Apply transforms
    train_ds.set_transform(transform_fn)
    test_ds.set_transform(transform_fn)

    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader, ds['train'].features[label_key].num_classes

