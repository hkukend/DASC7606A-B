import argparse
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch

# We reuse the transform definitions from train_utils
from scripts.train_utils import load_transforms


def analyze_image(img_path, train_tf, test_tf):
    img = Image.open(img_path).convert("RGB")
    t_train = train_tf(img)
    t_test = test_tf(img)

    def stats(t):
        # t: CxHxW tensor in [-inf, inf] after Normalize
        return {
            "shape": tuple(t.shape),
            "mean": float(t.mean().item()),
            "std": float(t.std().item()),
            "min": float(t.min().item()),
            "max": float(t.max().item()),
        }

    s_train = stats(t_train)
    s_test = stats(t_test)

    # L2 difference to confirm they are not identical
    diff = float(torch.norm(t_train - t_test).item())

    return s_train, s_test, diff


def main():
    parser = argparse.ArgumentParser(description="Check if train/test transforms are separated on ImageFolder data.")
    parser.add_argument("--root", type=str, required=True, help="Root folder of images (e.g., data/augmented/train)")
    parser.add_argument("--num_samples", type=int, default=8, help="Number of images to sample")
    parser.add_argument("--policy", type=str, default="randaugment", choices=["randaugment", "autoaugment"], help="Augmentation policy for train transform")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"[ERROR] Root folder not found: {root}")
        return 1

    # Build transforms (train has strong aug, test is deterministic normalize only)
    train_tf = load_transforms("train", policy=args.policy)
    test_tf = load_transforms("test", policy=args.policy)

    print("[INFO] Transform objects:")
    print("  Train:", train_tf)
    print("  Test: ", test_tf)
    if repr(train_tf) == repr(test_tf):
        print("[FAIL] Train and Test transforms appear identical by repr().")
    else:
        print("[OK] Train and Test transforms differ by repr().")

    # Collect image paths (png/jpg/jpeg)
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    image_paths = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if Path(f).suffix.lower() in exts:
                image_paths.append(Path(dirpath) / f)

    if len(image_paths) == 0:
        print(f"[ERROR] No images found under {root}")
        return 1

    random.seed(args.seed)
    sample_paths = random.sample(image_paths, min(args.num_samples, len(image_paths)))

    diffs = []
    for i, p in enumerate(sample_paths, 1):
        try:
            s_train, s_test, diff = analyze_image(p, train_tf, test_tf)
            diffs.append(diff)
            print(f"\n[Sample {i}] {p}")
            print("  Train stats:", s_train)
            print("  Test  stats:", s_test)
            print(f"  L2 difference between train/test outputs: {diff:.6f}")
        except Exception as e:
            print(f"  [WARN] Failed to process {p}: {e}")

    diffs = np.array(diffs, dtype=float) if diffs else np.array([0.0])
    print("\n[SUMMARY]")
    print(f"  Samples processed: {len(diffs)}")
    print(f"  Mean L2 difference (train vs test): {diffs.mean():.6f}")
    print(f"  Min/Max L2 difference: {diffs.min():.6f} / {diffs.max():.6f}")

    if diffs.mean() < 1e-6:
        print("[FAIL] Train and Test transforms produce nearly identical outputs on sampled images.")
        return 2
    else:
        print("[OK] Train/Test transforms appear separated.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
