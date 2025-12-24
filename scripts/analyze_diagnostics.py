#!/usr/bin/env python3
"""
Analyze training diagnostics to identify issues causing plateauing or poor convergence.
"""
import argparse
import json
from pathlib import Path
import sys


def load_diagnostics(file_path):
    """Load diagnostics from JSONL file."""
    records = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def analyze_diagnostics(records):
    """Analyze diagnostic records and provide insights."""
    if not records:
        print("[ERROR] No diagnostic records found.")
        return
    
    print(f"\n{'='*80}")
    print("TRAINING DIAGNOSTICS ANALYSIS")
    print(f"{'='*80}\n")
    
    print(f"Total epochs recorded: {len(records)}")
    
    # Learning rate analysis
    print(f"\n{'Learning Rate Analysis':-^80}")
    lrs = [r['learning_rate'] for r in records]
    print(f"  Initial LR: {lrs[0]:.6f}")
    print(f"  Final LR:   {lrs[-1]:.6f}")
    
    # Find LR reductions
    reductions = []
    for i in range(1, len(lrs)):
        if lrs[i] < lrs[i-1] * 0.99:  # More than 1% reduction
            reductions.append((i, lrs[i-1], lrs[i]))
    
    if reductions:
        print(f"  LR reductions detected: {len(reductions)}")
        for epoch, old_lr, new_lr in reductions:
            print(f"    Epoch {epoch}: {old_lr:.6f} → {new_lr:.6f} ({new_lr/old_lr:.2%} of previous)")
    else:
        print("  No LR reductions detected")
    
    # Loss analysis
    print(f"\n{'Loss Analysis':-^80}")
    losses = [r['loss'] for r in records]
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss:   {losses[-1]:.4f}")
    print(f"  Best loss:    {min(losses):.4f} (epoch {losses.index(min(losses))})")
    
    # Check for plateauing
    if len(losses) >= 5:
        recent_losses = losses[-5:]
        loss_std = sum((l - sum(recent_losses)/5)**2 for l in recent_losses)**0.5 / 5
        if loss_std < 0.01:
            print(f"  ⚠️  WARNING: Loss plateaued in last 5 epochs (std: {loss_std:.6f})")
    
    # Accuracy analysis
    print(f"\n{'Accuracy Analysis':-^80}")
    accs = [r['accuracy'] for r in records]
    print(f"  Initial accuracy: {accs[0]:.2f}%")
    print(f"  Final accuracy:   {accs[-1]:.2f}%")
    print(f"  Best accuracy:    {max(accs):.2f}% (epoch {accs.index(max(accs))})")
    
    # Gradient norm analysis
    print(f"\n{'Gradient Norm Analysis':-^80}")
    grad_norms = [r['mean_grad_norm'] for r in records]
    print(f"  Initial mean grad norm: {grad_norms[0]:.6f}")
    print(f"  Final mean grad norm:   {grad_norms[-1]:.6f}")
    print(f"  Max grad norm seen:     {max(grad_norms):.6f} (epoch {grad_norms.index(max(grad_norms))})")
    print(f"  Min grad norm seen:     {min(grad_norms):.6f} (epoch {grad_norms.index(min(grad_norms))})")
    
    # Check for vanishing gradients
    if grad_norms[-1] < 0.001:
        print("  ⚠️  WARNING: Very small gradients detected - possible vanishing gradient problem")
    
    # Weight change analysis
    print(f"\n{'Weight Update Analysis':-^80}")
    weight_changes = [r['max_weight_change'] for r in records]
    print(f"  Initial max weight change: {weight_changes[0]:.6f}")
    print(f"  Final max weight change:   {weight_changes[-1]:.6f}")
    print(f"  Largest weight change:     {max(weight_changes):.6f} (epoch {weight_changes.index(max(weight_changes))})")
    
    # Check for stalled learning
    if len(weight_changes) >= 3:
        recent_changes = weight_changes[-3:]
        if all(c < 1e-6 for c in recent_changes):
            print("  ⚠️  WARNING: Weights barely changing - learning may have stalled")
    
    # Diagnosis
    print(f"\n{'Diagnosis & Recommendations':-^80}")
    
    issues = []
    
    # Check for aggressive LR decay
    if len(reductions) > 2 and reductions[0][0] < 10:
        issues.append("LR reduced too early and frequently")
        print("  🔍 Issue: Learning rate reduced too early and/or too frequently")
        print("     → Consider increasing scheduler patience or using a different schedule")
        print("     → Try CosineAnnealingLR or OneCycleLR instead of ReduceLROnPlateau")
    
    # Check for plateau
    if len(losses) >= 10:
        mid_loss = sum(losses[5:10]) / 5 if len(losses) > 10 else sum(losses[5:]) / (len(losses) - 5)
        final_loss = sum(losses[-5:]) / 5
        if abs(mid_loss - final_loss) < 0.05:
            issues.append("Loss plateaued early")
            print("  🔍 Issue: Loss plateaued after initial improvement")
            print("     → Model may be underfitting - consider:")
            print("        - Reducing regularization (weight decay, label smoothing)")
            print("        - Reducing augmentation strength")
            print("        - Using a larger model")
    
    # Check for low final accuracy
    if accs[-1] < 20.0:
        issues.append("Very low accuracy")
        print("  🔍 Issue: Final accuracy extremely low")
        print("     → Possible causes:")
        print("        - Label mismatch or data corruption")
        print("        - Wrong number of output classes")
        print("        - Severe overfitting on augmentation")
    
    # Check for vanishing gradients
    if grad_norms[-1] < 0.01:
        issues.append("Vanishing gradients")
        print("  🔍 Issue: Gradient norms very small")
        print("     → Consider:")
        print("        - Increasing learning rate")
        print("        - Reducing gradient clipping threshold")
        print("        - Checking for ReLU death or poor initialization")
    
    if not issues:
        print("  ✅ No major issues detected in diagnostics")
        print("     → Training appears to be progressing normally")
        print("     → If still not achieving target performance, consider:")
        print("        - Training for more epochs")
        print("        - Tuning hyperparameters (LR, batch size, weight decay)")
        print("        - Trying different augmentation strategies")
    
    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze training diagnostics")
    parser.add_argument(
        "--file",
        type=str,
        default="results/diagnostics/train_diagnostics.jsonl",
        help="Path to diagnostics JSONL file"
    )
    args = parser.parse_args()
    
    diag_file = Path(args.file)
    if not diag_file.exists():
        print(f"[ERROR] Diagnostics file not found: {diag_file}")
        print("Make sure training has run and produced diagnostics.")
        return 1
    
    records = load_diagnostics(diag_file)
    analyze_diagnostics(records)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
