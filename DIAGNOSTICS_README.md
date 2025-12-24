# Training Diagnostics System

This codebase includes comprehensive training diagnostics to help identify and debug training issues like plateauing loss, vanishing gradients, or premature convergence.

## Features

### Automatic Diagnostics Collection
Training diagnostics are **automatically saved** to `results/diagnostics/` during training. No changes to `main.py` are needed.

**What's tracked:**
- **Per-epoch metrics**:
  - Training loss & accuracy
  - Validation loss & accuracy
  - Learning rate
  - Gradient norms (mean, min, max)
  - Weight update magnitudes
  - Number of batches processed

**Output files:**
- `results/diagnostics/train_diagnostics.jsonl` - Training metrics
- `results/diagnostics/val_diagnostics.jsonl` - Validation metrics

Each line is a JSON object representing one epoch.

## Usage

### 1. Run Training
Training automatically collects diagnostics:
```bash
python main.py --dataset cifar100 --device cuda --batch_size 512 --num_epochs 120
```

Diagnostics are saved to `results/diagnostics/` by default. To change location:
```bash
export DIAGNOSTICS_DIR=my_custom_diagnostics_dir
python main.py ...
```

### 2. Analyze Diagnostics
After training (or during, if you stop early), analyze the diagnostics:

```bash
bash scripts/bash/analyze_training.sh
```

Or directly:
```bash
python -m scripts.analyze_diagnostics --file results/diagnostics/train_diagnostics.jsonl
```

### 3. Interpret Results

The analyzer will print:
- **Learning Rate Analysis**: Shows if/when LR was reduced and by how much
- **Loss Analysis**: Initial, final, best loss; checks for plateauing
- **Accuracy Analysis**: Progress over time
- **Gradient Norm Analysis**: Detects vanishing/exploding gradients
- **Weight Update Analysis**: Confirms learning is happening
- **Diagnosis & Recommendations**: Actionable suggestions

## Common Issues Detected

### Issue: Learning rate reduced too early
**Symptoms:**
- LR drops within first 5-10 epochs
- Loss plateaus shortly after

**Recommendations:**
- Increase `ReduceLROnPlateau` patience (currently 3)
- Switch to `CosineAnnealingLR` or `OneCycleLR`
- Increase initial learning rate

**Fix:**
```python
# In scripts/train_utils.py, define_loss_and_optimizer():
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
```

### Issue: Loss plateaued early
**Symptoms:**
- Loss improves for 3-6 epochs then stops
- Accuracy remains very low (< 20%)

**Possible causes:**
1. **Over-regularization**: Label smoothing + weight decay + strong augmentation may be too much
2. **Data issues**: Transforms not separated (run `scripts/check_transforms.py`)
3. **LR too low**: Especially after aggressive early reductions

**Recommendations:**
- Reduce label smoothing: `0.1 → 0.05` or `0.0`
- Reduce weight decay: `5e-4 → 1e-4`
- Reduce augmentation: RandAugment magnitude `11 → 9`
- Use a learning rate schedule without early reductions

### Issue: Vanishing gradients
**Symptoms:**
- Gradient norms < 0.01
- Weight changes < 1e-6

**Recommendations:**
- Increase learning rate
- Reduce gradient clipping threshold (currently 1.0)
- Check model initialization
- Verify activation functions (ReLU death)

### Issue: Very low accuracy (< 5%)
**Symptoms:**
- Accuracy stuck near random guessing
- Loss around 4.605 (ln(100) for CIFAR-100)

**Likely causes:**
1. **Label mismatch**: Class indices don't match dataset
2. **Transform bug**: Train/val using same transforms (run `scripts/check_transforms.py`)
3. **Data corruption**: Wrong dataset loaded

**Debug steps:**
```bash
# Verify transforms differ
bash scripts/bash/check_transforms.sh

# Check diagnostics for grad flow
bash scripts/bash/analyze_training.sh
```

## Critical Fixes Already Applied

### ✅ AMP Gradient Clipping Fix
**Issue**: Gradients were clipped before unscaling, effectively over-clipping and stalling learning.

**Fix applied** in `scripts/train_utils.py`:
```python
scaler.scale(loss).backward()
scaler.unscale_(optimizer)  # ← CRITICAL: Unscale before clipping
grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
```

### ✅ Transform Separation Fix
**Issue**: Train and validation used the same transforms due to shared `ImageFolder` instance.

**Fix applied** in `scripts/train_utils.py`:
```python
# Separate ImageFolder instances for train/val
ds_train_full = datasets.ImageFolder(root=root_dir, transform=load_transforms("train", policy))
ds_val_full = datasets.ImageFolder(root=root_dir, transform=load_transforms("test", policy))
# Apply same index split to each
train_dataset = Subset(ds_train_full, train_indices)
val_dataset = Subset(ds_val_full, val_indices)
```

## Manual Inspection of Diagnostics

If you prefer to inspect the raw data:

```bash
# View first 5 training records
head -5 results/diagnostics/train_diagnostics.jsonl | jq .

# View last 5 training records
tail -5 results/diagnostics/train_diagnostics.jsonl | jq .

# Extract learning rates
jq '.learning_rate' results/diagnostics/train_diagnostics.jsonl

# Extract losses
jq '.loss' results/diagnostics/train_diagnostics.jsonl

# Extract gradient norms
jq '.mean_grad_norm' results/diagnostics/train_diagnostics.jsonl
```

## Quick Reference

| Script | Purpose |
|--------|---------|
| `scripts/train_utils.py` | Collects diagnostics automatically during training |
| `scripts/analyze_diagnostics.py` | Analyzes diagnostics and provides recommendations |
| `scripts/check_transforms.py` | Verifies train/val transforms are separated |
| `scripts/bash/analyze_training.sh` | Convenience wrapper for analyzer |
| `scripts/bash/check_transforms.sh` | Convenience wrapper for transform check |

## Next Steps After Diagnosis

Based on the analyzer output, consider:

1. **Adjust learning rate schedule**: Switch from ReduceLROnPlateau to Cosine or OneCycle
2. **Tune regularization**: Balance label smoothing, weight decay, and augmentation
3. **Verify data pipeline**: Run transform and label checks
4. **Try different model**: ResNet34 instead of ResNet18, or add capacity
5. **Increase training time**: Some models need more epochs to converge

## Support

If diagnostics show unexpected behavior not covered here, share the analyzer output for targeted debugging.
