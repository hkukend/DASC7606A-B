#!/usr/bin/env python3
"""
Compute mean and std for a dataset to use in normalization.
"""
import argparse
import sys
from pathlib import Path
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

def compute_mean_std(data_dir, num_workers=4, batch_size=256):
    """Compute mean and std of dataset."""
    # Load without normalization
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to [0, 1]
    ])
    
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    print(f"Computing statistics for {len(dataset)} images...")
    
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_images = 0
    
    for images, _ in tqdm(loader, desc="Computing mean"):
        batch_size = images.size(0)
        images = images.view(batch_size, images.size(1), -1)  # (B, C, H*W)
        mean += images.mean(2).sum(0)
        total_images += batch_size
    
    mean /= total_images
    
    for images, _ in tqdm(loader, desc="Computing std"):
        batch_size = images.size(0)
        images = images.view(batch_size, images.size(1), -1)
        std += ((images - mean.view(1, 3, 1)) ** 2).mean(2).sum(0)
    
    std = torch.sqrt(std / total_images)
    
    return mean, std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()
    
    mean, std = compute_mean_std(args.data_dir, args.num_workers, args.batch_size)
    
    print(f"\n{'='*80}")
    print("NORMALIZATION STATISTICS")
    print(f"{'='*80}")
    print(f"Mean: {tuple(mean.tolist())}")
    print(f"Std:  {tuple(std.tolist())}")
    print(f"\nUse these in scripts/train_utils.py:")
    print(f"DATASET_MEAN = {tuple(float(f'{x:.4f}') for x in mean.tolist())}")
    print(f"DATASET_STD = {tuple(float(f'{x:.4f}') for x in std.tolist())}")
    

if __name__ == "__main__":
    sys.exit(main())
