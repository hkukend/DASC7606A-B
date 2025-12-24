#!/usr/bin/env python3
"""
Debug validation data to identify why val accuracy is 0%.
"""
import argparse
import sys
from pathlib import Path
import torch
import numpy as np

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_utils import load_data, load_transforms
from scripts.model_architectures import create_model


def check_batch_stats(loader, name="Loader"):
    """Check statistics of batches from a loader."""
    print(f"\n{'='*80}")
    print(f"{name} Statistics")
    print(f"{'='*80}")
    
    batch = next(iter(loader))
    inputs, labels = batch
    
    print(f"Batch shape: {inputs.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Input dtype: {inputs.dtype}")
    print(f"Labels dtype: {labels.dtype}")
    print(f"\nInput statistics:")
    print(f"  Mean: {inputs.mean().item():.4f}")
    print(f"  Std:  {inputs.std().item():.4f}")
    print(f"  Min:  {inputs.min().item():.4f}")
    print(f"  Max:  {inputs.max().item():.4f}")
    print(f"\nLabel statistics:")
    print(f"  Unique labels: {torch.unique(labels).tolist()[:20]}...")
    print(f"  Min label: {labels.min().item()}")
    print(f"  Max label: {labels.max().item()}")
    print(f"  Label counts: {len(torch.unique(labels))} unique in batch")
    
    return inputs, labels


def test_model_predictions(model, inputs, labels, device):
    """Test model predictions on a batch."""
    print(f"\n{'='*80}")
    print("Model Prediction Test")
    print(f"{'='*80}")
    
    model.eval()
    with torch.no_grad():
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)
        
        print(f"Output shape: {outputs.shape}")
        print(f"Output statistics:")
        print(f"  Mean: {outputs.mean().item():.4f}")
        print(f"  Std:  {outputs.std().item():.4f}")
        print(f"  Min:  {outputs.min().item():.4f}")
        print(f"  Max:  {outputs.max().item():.4f}")
        
        print(f"\nPredictions (first 20): {preds[:20].cpu().tolist()}")
        print(f"True labels (first 20): {labels[:20].cpu().tolist()}")
        
        correct = (preds == labels).sum().item()
        total = labels.size(0)
        accuracy = 100.0 * correct / total
        
        print(f"\nBatch accuracy: {accuracy:.2f}% ({correct}/{total})")
        
        # Check if model is just predicting one class
        unique_preds = torch.unique(preds)
        print(f"Unique predictions: {len(unique_preds)} classes predicted")
        if len(unique_preds) < 10:
            print(f"  WARNING: Model only predicting {len(unique_preds)} classes!")
            print(f"  Classes: {unique_preds.cpu().tolist()}")
        
        # Check confidence
        max_probs = probs.max(dim=1)[0]
        print(f"\nPrediction confidence:")
        print(f"  Mean max prob: {max_probs.mean().item():.4f}")
        print(f"  Min max prob:  {max_probs.min().item():.4f}")
        print(f"  Max max prob:  {max_probs.max().item():.4f}")


def compare_loaders(train_loader, val_loader, model, device):
    """Compare train and val loaders side by side."""
    print(f"\n{'='*80}")
    print("COMPARISON: Train vs Val")
    print(f"{'='*80}")
    
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    
    t_inputs, t_labels = train_batch
    v_inputs, v_labels = val_batch
    
    print(f"\nInput Statistics Comparison:")
    print(f"  Train - Mean: {t_inputs.mean().item():.4f}, Std: {t_inputs.std().item():.4f}")
    print(f"  Val   - Mean: {v_inputs.mean().item():.4f}, Std: {v_inputs.std().item():.4f}")
    print(f"  Difference: {abs(t_inputs.mean().item() - v_inputs.mean().item()):.4f}")
    
    if abs(t_inputs.mean().item() - v_inputs.mean().item()) > 0.5:
        print("  ⚠️  WARNING: Large difference in input means!")
    
    print(f"\nLabel Range Comparison:")
    print(f"  Train - Min: {t_labels.min().item()}, Max: {t_labels.max().item()}")
    print(f"  Val   - Min: {v_labels.min().item()}, Max: {v_labels.max().item()}")
    
    # Test model on both
    model.eval()
    with torch.no_grad():
        t_outputs = model(t_inputs.to(device))
        v_outputs = model(v_inputs.to(device))
        
        t_preds = t_outputs.argmax(dim=1)
        v_preds = v_outputs.argmax(dim=1)
        
        t_correct = (t_preds == t_labels.to(device)).sum().item()
        v_correct = (v_preds == v_labels.to(device)).sum().item()
        
        print(f"\nModel Performance Comparison:")
        print(f"  Train batch accuracy: {100.0 * t_correct / len(t_labels):.2f}%")
        print(f"  Val batch accuracy:   {100.0 * v_correct / len(v_labels):.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Debug validation data issues")
    parser.add_argument("--data_dir", type=str, default="data/augmented/train", help="Data directory")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model_path", type=str, default=None, help="Path to trained model checkpoint")
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("VALIDATION DEBUGGING")
    print(f"{'='*80}")
    print(f"Data directory: {args.data_dir}")
    print(f"Device: {args.device}")
    
    # Load data
    print("\n[1/4] Loading data...")
    train_loader, val_loader = load_data(args.data_dir, args.batch_size, num_workers=0)
    
    # Check train loader
    print("\n[2/4] Checking train loader...")
    train_inputs, train_labels = check_batch_stats(train_loader, "Train Loader")
    
    # Check val loader
    print("\n[3/4] Checking val loader...")
    val_inputs, val_labels = check_batch_stats(val_loader, "Val Loader")
    
    # Load model
    print("\n[4/4] Testing with model...")
    if args.model_path and Path(args.model_path).exists():
        print(f"Loading model from {args.model_path}")
        model = create_model(num_classes=100, device=args.device)
        checkpoint = torch.load(args.model_path, map_location=args.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print("Creating fresh model (random weights)")
        model = create_model(num_classes=100, device=args.device)
    
    # Test predictions on val
    test_model_predictions(model, val_inputs, val_labels, args.device)
    
    # Compare train vs val
    compare_loaders(train_loader, val_loader, model, args.device)
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print("\nIf val accuracy is 0%:")
    print("  1. Check if 'Input Statistics Comparison' shows large difference")
    print("  2. Check if model is only predicting few classes")
    print("  3. Verify label ranges match between train/val")
    print("  4. Check if normalization stats are appropriate for your data")
    print("\nNext steps:")
    print("  - If means differ significantly: normalization issue")
    print("  - If model predicts only 1-2 classes on val: severe distribution mismatch")
    print("  - If labels are out of range: data loading bug")
    

if __name__ == "__main__":
    sys.exit(main())
