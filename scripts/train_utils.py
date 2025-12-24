import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
import os
import numpy as np
import json
from pathlib import Path
from typing import Tuple, Optional


# CIFAR-100 normalization stats
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

def load_transforms(split: str = "test", policy: str = "randaugment"):
    """
    Load the data transformations for a given split.
    Default to 'test' to remain compatible with main.py's evaluation call.
    """
    if split == "train":
        aug = []
        aug.append(transforms.RandomCrop(32, padding=4))
        aug.append(transforms.RandomHorizontalFlip())
        if policy == "autoaugment":
            try:
                aug.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10))
            except AttributeError:
                pass
        else:
            try:
                aug.append(transforms.RandAugment(num_ops=2, magnitude=11))
            except AttributeError:
                pass
        aug.extend([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
            transforms.RandomErasing(p=0.4),
        ])
        return transforms.Compose(aug)
    # test/val
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

def load_data(root_dir: str, batch_size: int, num_workers: int = None, policy: str = "randaugment") -> Tuple[DataLoader, DataLoader]:
    """
    Load data for training/validation.
    - If 'root_dir' points to an ImageFolder root, split 80/20 and use on-the-fly transforms.
    - Otherwise, fallback to TorchVision CIFAR-100 datasets under 'root_dir'.

    Args:
        root_dir: Either ImageFolder root (augmented) or TorchVision base directory
        batch_size: The batch size to use for the data loaders
        num_workers: DataLoader workers (defaults to half of CPUs)
        policy: Augment policy ("randaugment" | "autoaugment")
    Returns:
        train_loader, val_loader
    """
    if num_workers is None:
        try:
            num_workers = max(2, os.cpu_count() // 2)
        except Exception:
            num_workers = 2

    pin = torch.cuda.is_available()

    # Determine if root_dir is an ImageFolder directory (classes as subfolders)
    if os.path.isdir(root_dir):
        # CRITICAL FIX: Use separate ImageFolder instances for train/val to avoid transform override bug
        ds_train_full = datasets.ImageFolder(root=root_dir, transform=load_transforms("train", policy))
        ds_val_full = datasets.ImageFolder(root=root_dir, transform=load_transforms("test", policy))
        
        # Consistent split indices
        total_len = len(ds_train_full)
        train_size = int(0.8 * total_len)
        val_size = total_len - train_size
        generator = torch.Generator().manual_seed(42)
        indices = list(range(total_len))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Apply indices to respective datasets
        train_dataset = Subset(ds_train_full, train_indices)
        val_dataset = Subset(ds_val_full, val_indices)
        
        # DIAGNOSTIC: Verify transforms are different
        print("\n[DIAGNOSTIC] Checking transforms...")
        print(f"  Train transform: {ds_train_full.transform}")
        print(f"  Val transform: {ds_val_full.transform}")
        assert ds_train_full.transform != ds_val_full.transform, "BUG: Train and val transforms are identical!"
    else:
        # Fallback to TorchVision CIFAR-100 datasets
        train_dataset = datasets.CIFAR100(root=root_dir, train=True, download=True, transform=load_transforms("train", policy))
        val_dataset = datasets.CIFAR100(root=root_dir, train=False, download=True, transform=load_transforms("test", policy))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=pin,
    )

    print(f"CIFAR-100 loaded under: {root_dir}")
    try:
        tr_len = len(train_dataset)
    except Exception:
        tr_len = "unknown"
    try:
        va_len = len(val_dataset)
    except Exception:
        va_len = "unknown"
    print(f"Training set size: {tr_len} | Validation set size: {va_len}")
    
    # DIAGNOSTIC: Check label distribution in first batch
    print("\n[DIAGNOSTIC] Sampling labels from train loader...")
    try:
        sample_batch = next(iter(train_loader))
        sample_labels = sample_batch[1].numpy()
        print(f"  Sample batch labels (first 20): {sample_labels[:20]}")
        print(f"  Unique labels in batch: {len(np.unique(sample_labels))}")
        print(f"  Min label: {sample_labels.min()}, Max label: {sample_labels.max()}")
        assert sample_labels.min() >= 0, "BUG: Negative labels detected!"
        assert sample_labels.max() < 100, "BUG: Labels exceed num_classes!"
    except Exception as e:
        print(f"  Warning: Could not verify labels: {e}")

    return train_loader, val_loader


def define_loss_and_optimizer(model: nn.Module, lr: float, weight_decay: float, t_max: int = 100):
    """
    Define the loss function and optimizer
    Args:
        model: The model to train
        lr: Learning rate
        weight_decay: Weight decay
        t_max: Unused; kept for backward compatibility if needed
    Returns:
        criterion: The loss function
        optimizer: The optimizer
        scheduler: The scheduler
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    # Keep ReduceLROnPlateau for compatibility with main.py's scheduler.step(val_loss)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    return criterion, optimizer, scheduler


def train_epoch(model, dataloader, criterion, optimizer, device, epoch_num=None, diagnostics_dir: Optional[str] = None):
    """
    Train the model for one epoch
    Args:
        model: The model to train
        dataloader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch_num: Current epoch number for diagnostics (optional, for backward compat)
        diagnostics_dir: Directory to save diagnostic logs (optional, defaults to 'results/diagnostics')
    Returns:
        Average loss and accuracy for the epoch
    """
    # Auto-detect if this is first epoch by checking if optimizer has state
    if epoch_num is None:
        try:
            epoch_num = 0 if len(optimizer.state) == 0 else 1
        except Exception:
            epoch_num = 0
    
    # Auto-enable diagnostics if not specified
    if diagnostics_dir is None:
        diagnostics_dir = os.environ.get('DIAGNOSTICS_DIR', 'results/diagnostics')
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    use_cuda = torch.cuda.is_available() and device == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_cuda else None
    
    # Track diagnostics throughout training
    grad_norms = []
    batch_losses = []
    
    # Capture initial weights to measure change
    try:
        first_param = next(model.parameters())
        initial_weights = first_param.data.clone()
    except Exception:
        initial_weights = None

    for inputs, labels in progress_bar:
        if use_cuda:
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                inputs = inputs.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
                labels = labels.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            # CRITICAL FIX: Unscale gradients before clipping when using AMP
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            grad_norms.append(float(grad_norm.item()) if hasattr(grad_norm, 'item') else float(grad_norm))
        else:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            grad_norms.append(float(grad_norm.item()) if hasattr(grad_norm, 'item') else float(grad_norm))

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        batch_losses.append(loss.item())
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        progress_bar.set_postfix(
            {"Loss": f"{loss.item():.4f}", "Acc": f"{100.0 * correct / total:.2f}%"}
        )

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    
    # Compute weight change
    weight_change = 0.0
    if initial_weights is not None:
        try:
            first_param = next(model.parameters())
            weight_change = (first_param.data - initial_weights).abs().max().item()
        except Exception:
            pass
    
    # Get current learning rate
    current_lr = optimizer.param_groups[0]['lr']
    
    # Save diagnostics to file
    if diagnostics_dir is not None:
        diag_path = Path(diagnostics_dir)
        diag_path.mkdir(parents=True, exist_ok=True)
        diag_file = diag_path / "train_diagnostics.jsonl"
        
        diag_record = {
            "epoch": epoch_num,
            "loss": float(epoch_loss),
            "accuracy": float(epoch_acc),
            "learning_rate": float(current_lr),
            "mean_grad_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
            "max_grad_norm": float(np.max(grad_norms)) if grad_norms else 0.0,
            "min_grad_norm": float(np.min(grad_norms)) if grad_norms else 0.0,
            "mean_batch_loss": float(np.mean(batch_losses)) if batch_losses else 0.0,
            "max_weight_change": float(weight_change),
            "num_batches": len(batch_losses),
        }
        
        with open(diag_file, 'a') as f:
            f.write(json.dumps(diag_record) + '\n')
    
    # Print summary for important epochs
    if epoch_num < 3 or epoch_num % 10 == 0:
        print(f"\n[DIAGNOSTIC] Epoch {epoch_num}:")
        print(f"  LR: {current_lr:.6f}")
        print(f"  Mean grad norm: {np.mean(grad_norms):.6f}" if grad_norms else "  No grad norms")
        print(f"  Max weight change: {weight_change:.6f}")

    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device, epoch_num=None, diagnostics_dir: Optional[str] = None):
    """
    Validate the model
    Args:
        model: The model to validate
        dataloader: DataLoader for validation data
        criterion: Loss function
        device: Device to use
        epoch_num: Current epoch number for diagnostics (optional)
        diagnostics_dir: Directory to save diagnostic logs (optional)
    Returns:
        Average loss and accuracy for the validation set
    """
    # Auto-enable diagnostics if not specified
    if diagnostics_dir is None:
        diagnostics_dir = os.environ.get('DIAGNOSTICS_DIR', 'results/diagnostics')
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation", leave=False)

        for inputs, labels in progress_bar:
            if torch.cuda.is_available() and device == "cuda":
                with torch.cuda.amp.autocast():
                    inputs = inputs.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
                    labels = labels.to(device, non_blocking=True)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
            else:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            progress_bar.set_postfix(
                {"Loss": f"{loss.item():.4f}", "Acc": f"{100.0 * correct / total:.2f}%"}
            )

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    
    # Save validation diagnostics to file
    if diagnostics_dir is not None and epoch_num is not None:
        diag_path = Path(diagnostics_dir)
        diag_path.mkdir(parents=True, exist_ok=True)
        diag_file = diag_path / "val_diagnostics.jsonl"
        
        diag_record = {
            "epoch": epoch_num,
            "loss": float(epoch_loss),
            "accuracy": float(epoch_acc),
        }
        
        with open(diag_file, 'a') as f:
            f.write(json.dumps(diag_record) + '\n')

    return epoch_loss, epoch_acc


def save_checkpoint(state, filename):
    """
    Save model checkpoint
    Args:
        state: Checkpoint state
        filename: Path to save checkpoint
    """
    torch.save(state, filename)


def load_checkpoint(filename, model, optimizer=None, scheduler=None):
    """
    Load model checkpoint
    Args:
        filename: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
    Returns:
        Checkpoint state
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Checkpoint file {filename} not found")

    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["state_dict"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint

def save_metrics(metrics: str, filename: str = "training_metrics.txt"):
    """
    Save training metrics to a file
    Args:
        metrics: Metrics string to save
        filename: Path to save metrics
    """
    with open(filename, 'w') as f:
        f.write(metrics)
