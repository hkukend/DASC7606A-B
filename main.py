#!/usr/bin/env python3
# cifar_pipeline.py - Complete pipeline for CIFAR-10/100 data preparation, augmentation, training and evaluation

import argparse
import logging
import os
import random
import numpy as np

import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from torchvision import datasets

# Import our custom modules
from scripts.data_download import download_and_extract_cifar10_data, download_and_extract_cifar100_data
from scripts.data_augmentation import augment_dataset
from scripts.model_architectures import create_model
from scripts.train_utils import (
    save_metrics,
    train_epoch,
    validate_epoch,
    save_checkpoint,
    define_loss_and_optimizer,
    load_data,
    load_transforms,
)
from scripts.evaluation_metrics import (
    evaluate_model,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("cifar_pipeline.log")
    ]
)
logger = logging.getLogger(__name__)

def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="CIFAR-10/100 Training Pipeline")

    # Dataset selection
    parser.add_argument("--dataset", type=str, choices=["cifar10", "cifar100"], default="cifar10",
                        help="Dataset to use (cifar10 or cifar100)")

    # Data paths
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Base directory for data storage")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save results")

    # Data augmentation
    parser.add_argument("--aug_count", type=int, default=3,
                        help="Number of augmentations per image")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay (L2 penalty)")

    # Checkpointing
    parser.add_argument("--save_freq", type=int, default=1,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--early_stopping_patience", type=int, default=10,
                        help="Early stopping patience")

    # Hardware
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training (cuda/cpu)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")

    # Random seeds
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    return parser.parse_args()

def collect_data(args):
    """Collect data"""
    logger.info(f"Collecting {args.dataset} dataset...")

    # Create the directory for our raw data if it doesn't already exist
    print("Preparing data directory...")
    os.makedirs(args.data_dir + "/raw", exist_ok=True)
    print("Setup complete.")

    if args.dataset == "cifar10":
        train_dataset, test_dataset = download_and_extract_cifar10_data(
            root_dir=args.data_dir + "/raw",
        )
    else:
        train_dataset, test_dataset = download_and_extract_cifar100_data(
            root_dir=args.data_dir + "/raw",
        )

def augment_data(args):
    """Prepare and augment data"""
    logger.info(f"Augmenting {args.dataset} dataset...")

    raw_data_dir = args.data_dir + '/raw/train/'
    augmented_data_dir = args.data_dir + '/augmented/train/'
    augmentations_per_image = args.aug_count

    # --- Path Validation ---
    # Check if the raw data directory exists before proceeding.
    if not os.path.exists(raw_data_dir):
        print(f"❌ Error: Raw data directory '{raw_data_dir}' not found.")
        print("Please ensure you have run 'collect_data' first.")
    else:
        print(f"✅ Found raw data at: {raw_data_dir}")
        print(f"   Augmented data will be saved to: {augmented_data_dir}")
        print(f"   Number of augmentations per image: {augmentations_per_image}")

    # Ensure the raw data directory exists before running
    if os.path.exists(raw_data_dir):
        print("🚀 Starting data augmentation...")
        augment_dataset(
            input_dir=raw_data_dir,
            output_dir=augmented_data_dir,
            augmentations_per_image=augmentations_per_image
        )
        print("\n🎉 Data augmentation completed successfully!")
    else:
        print("Skipping augmentation process due to missing raw data directory.")

    return augmented_data_dir

def build_model(args):
    """Build the model"""
    if args.dataset == "cifar10":
        num_classes = 10
    else:
        num_classes = 100
    logger.info(f"Creating model with {num_classes} classes, {args.device} device...")
    model = create_model(num_classes=num_classes, device=args.device)
    return model

def train(args, model: nn.Module):
    # Define loss and optimizer
    criterion, optimizer, scheduler = define_loss_and_optimizer(model, args.lr, args.weight_decay)

    # Initialize tracking variables
    best_val_loss = float("inf")
    patience_counter = 0

    # Lists to store training history for later plotting
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Create directories for saving models and results if they don't exist
    os.makedirs(args.output_dir + "/models", exist_ok=True)
    os.makedirs(args.output_dir + "/results", exist_ok=True)

    print(f"Training configured for {args.num_epochs} epochs with early stopping patience of {args.early_stopping_patience}.")

    # Load data
    train_loader, val_loader = load_data(args.data_dir + "/augmented/train", args.batch_size)

    print("Starting training...")
    for epoch in range(args.num_epochs):
        # Train for one epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, args.device
        )

        # Validate the model
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, args.device)

        # Update learning rate based on validation loss
        scheduler.step(val_loss)

        # Store metrics for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        # Print epoch summary
        print(f"Epoch {epoch + 1}/{args.num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Check for improvement and save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_val_loss": best_val_loss,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                args.output_dir + "/models/best_model.pth",
            )
            print("  ↳ Validation loss improved. Saving best model!")
        else:
            patience_counter += 1
            print(
                f"  ↳ No improvement. Early stopping counter: {patience_counter}/{args.early_stopping_patience}"
            )

        # Check for early stopping
        if patience_counter >= args.early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs!")
            break

    print("\nTraining completed!")

    # Load the best model checkpoint saved during training
    checkpoint = torch.load(args.output_dir + "/models/best_model.pth")
    model.load_state_dict(checkpoint["state_dict"])

    # Retrieve details from the checkpoint
    best_epoch = checkpoint["epoch"]
    best_val_loss_loaded = checkpoint["best_val_loss"]

    print(f"Loaded best model from epoch {best_epoch} with validation loss {best_val_loss_loaded:.4f}")

    # Save the final model's state_dict for easy use in evaluation/inference
    torch.save(model.state_dict(), args.output_dir + "/models/final_model.pth")
    print(f"Final model state_dict saved to '{args.output_dir}/models/final_model.pth'.")

    return model, best_val_loss

def evaluate(args, model: nn.Module):
    """Evaluate the model on test data"""
    # Load the test dataset from the specified directory
    test_data_dir = args.data_dir + "/raw/test"
    test_dataset = datasets.ImageFolder(root=test_data_dir, transform=load_transforms())
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Set the model to evaluation mode
    model.eval()

    # Define loss function
    criterion, _, _ = define_loss_and_optimizer(model, args.lr, args.weight_decay)

    # Evaluate the model
    test_loss, test_accuracy, all_preds, all_labels, all_probs = evaluate_model(
        model, test_loader, criterion, args.device
    )
    metrics_str = classification_report(all_labels, all_preds, target_names=test_dataset.classes)

    save_metrics(metrics_str)

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()

    # Set random seeds
    set_random_seeds(args.seed)

    # Print configuration
    logger.info("Starting CIFAR pipeline with configuration:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")

    # Collect data
    collect_data(args)
    # Augment data
    augment_data(args)
    # Build model
    model = build_model(args)
    # Train
    train(args, model)
    # Evaluate
    evaluate(args, model)

if __name__ == "__main__":
    main()