import logging
import random
from pathlib import Path
from typing import List, Optional, Tuple

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100
import json

# Configure module-level logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CIFAR10Downloader:
    """
    A configurable class to download and prepare the CIFAR-10 dataset.

    Supports custom root directories, transforms, and verbose reporting.
    Also supports saving images to organized train/test/classname folders.
    """

    CLASS_NAMES = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    def __init__(
        self,
        root_dir: str = "data/raw",
        transform: Optional[transforms.Compose] = None,
        download: bool = True,
        log_stats: bool = True,
    ):
        """
        Initialize the CIFAR-10 downloader.

        Args:
            root_dir: Base directory to store downloaded dataset and extracted images.
            transform: Optional torchvision transform pipeline. If None, uses default.
            download: Whether to download if not present.
            log_stats: Whether to log dataset statistics after loading.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform or self._get_default_transform()
        self.download = download
        self.log_stats = log_stats

        # Ensure base directory exists
        self.root_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _get_default_transform() -> transforms.Compose:
        """Return default transform: ToTensor + Normalize to [-1, 1]."""
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def load_datasets(self) -> Tuple[CIFAR10, CIFAR10]:
        """
        Load CIFAR-10 train and test datasets.

        Returns:
            Tuple of (train_dataset, test_dataset)

        Raises:
            RuntimeError: If dataset fails to download or load.
        """
        try:
            # If dataset already present, avoid re-downloading
            try:
                _ = CIFAR10(root=str(self.root_dir), train=True, download=False)
                _ = CIFAR10(root=str(self.root_dir), train=False, download=False)
                effective_download = False
                logger.info("CIFAR-10 detected locally. Skipping download.")
            except Exception:
                effective_download = self.download

            logger.info("Loading CIFAR-10 training dataset...")
            train_dataset = CIFAR10(
                root=str(self.root_dir),
                train=True,
                download=effective_download,
                transform=self.transform,
            )

            logger.info("Loading CIFAR-10 test dataset...")
            test_dataset = CIFAR10(
                root=str(self.root_dir),
                train=False,
                download=effective_download,
                transform=self.transform,
            )

        except Exception as e:
            raise RuntimeError(f"Failed to load CIFAR-10 dataset: {e}") from e

        if self.log_stats:
            self._log_dataset_stats(train_dataset, test_dataset)

        return train_dataset, test_dataset

    def _log_dataset_stats(self, train_dataset: CIFAR10, test_dataset: CIFAR10) -> None:
        """Log formatted dataset statistics."""
        total = len(train_dataset) + len(test_dataset)
        separator = "=" * 50

        logger.info("\n" + separator)
        logger.info("✅ CIFAR-10 Dataset Loaded Successfully!")
        logger.info(separator)
        logger.info("📊 Dataset Statistics:")
        logger.info(f"   • Training samples: {len(train_dataset):,}")
        logger.info(f"   • Test samples: {len(test_dataset):,}")
        logger.info(f"   • Total samples: {total:,}")
        logger.info("   • Image size: 32x32 pixels")
        logger.info("   • Color channels: 3 (RGB)")
        logger.info(f"   • Number of classes: {len(self.CLASS_NAMES)}")
        logger.info(f"   • Classes: {', '.join(self.CLASS_NAMES)}")
        logger.info(separator)

    def save_images_to_folders(self, train_dataset: CIFAR10, test_dataset: CIFAR10, val_split: float = 0.2) -> None:
        """
        Save CIFAR-10 images as PNG files into organized folders:
        - raw/train/<class_name>/
        - raw/val/<class_name>/
        - raw/test/<class_name>/

        Args:
            train_dataset: The loaded CIFAR-10 training dataset.
            test_dataset: The loaded CIFAR-10 test dataset.
            val_split: Fraction of training data to move to validation (default: 0.2).
        """
        # Define output directories
        train_image_dir = self.root_dir / "train"
        val_image_dir = self.root_dir / "val"
        test_image_dir = self.root_dir / "test"

        for class_name in self.CLASS_NAMES:
            (train_image_dir / class_name).mkdir(parents=True, exist_ok=True)
            (val_image_dir / class_name).mkdir(parents=True, exist_ok=True)
            (test_image_dir / class_name).mkdir(parents=True, exist_ok=True)

        # Split training data into train and validation
        train_indices, val_indices = self._split_train_dataset(train_dataset, val_split)

        logger.info("Saving training images...")
        self._save_dataset_images(train_dataset, train_image_dir, train_indices)

        logger.info("Saving validation images...")
        self._save_dataset_images(train_dataset, val_image_dir, val_indices)

        logger.info("Saving test images...")
        self._save_dataset_images(test_dataset, test_image_dir)

        logger.info("✅ All images saved successfully!")

    def _split_train_dataset(self, train_dataset: CIFAR10, val_split: float) -> Tuple[List[int], List[int]]:
        """
        Split training dataset indices into train and validation sets.

        Args:
            train_dataset: The training dataset to split.
            val_split: Fraction of data to allocate to validation.

        Returns:
            Tuple of (train_indices, val_indices)
        """
        total_samples = len(train_dataset)
        val_size = int(total_samples * val_split)
        train_size = total_samples - val_size

        # Create indices and shuffle them
        indices = list(range(total_samples))
        random.shuffle(indices)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        logger.info(f"Split training data: {len(train_indices)} train, {len(val_indices)} validation samples")

        return train_indices, val_indices

    def _save_dataset_images(self, dataset: CIFAR10, base_dir: Path, indices: Optional[List[int]] = None) -> None:
        """
        Helper to save images from a dataset into class-named subfolders.

        Args:
            dataset: The CIFAR-10 dataset (train, val, or test).
            base_dir: Root directory to save images under.
            indices: Optional list of indices to save. If None, saves all images.
        """
        if indices is None:
            indices = list(range(len(dataset)))

        for idx in indices:
            image, label = dataset[idx]
            # If transform includes ToTensor, convert back to PIL for saving
            if hasattr(image, "min") and image.min() < 0:  # Assume normalized to [-1, 1]
                image = image * 0.5 + 0.5
            if hasattr(image, "cpu"):
                image = image.cpu()
            if hasattr(image, "numpy"):
                import numpy as np

                image = image.numpy()
                if image.shape[0] == 3:  # CHW format
                    image = image.transpose(1, 2, 0)  # Convert to HWC
                image = (image * 255).astype(np.uint8)
                from PIL import Image

                image = Image.fromarray(image)
            elif hasattr(image, "permute"):
                # Handle tensor format
                image = transforms.functional.to_pil_image(image)

            class_name = self.CLASS_NAMES[label]
            file_path = base_dir / class_name / f"{idx}.png"
            image.save(file_path)


class CIFAR100Downloader:
    """
    A configurable class to download and prepare the CIFAR-100 dataset.

    Supports custom root directories, transforms, and verbose reporting.
    Also supports saving images to organized train/test/classname folders.
    """

    # CIFAR-100 has 100 fine-grained classes
    CLASS_NAMES = [
        "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
        "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
        "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
        "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
        "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
        "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
        "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
        "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
        "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
        "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea",
        "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
        "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank",
        "telephone", "television", "tiger", "tractor", "train", "trout", "tulip",
        "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"
    ]
    def __init__(self, root_dir: str = "data/raw", transform: Optional[transforms.Compose] = None, download: bool = True, log_stats: bool = True):
        self.root_dir = Path(root_dir)
        self.transform = transform or self._get_default_transform()
        self.download = download
        self.log_stats = log_stats

        # Ensure base directory exists
        self.root_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _get_default_transform() -> transforms.Compose:
        """Return default transform: ToTensor + Normalize to [-1, 1]."""
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def load_datasets(self) -> Tuple[CIFAR100, CIFAR100]:
        """
        Load CIFAR-100 train and test datasets.

        Returns:
            Tuple of (train_dataset, test_dataset)

        Raises:
            RuntimeError: If dataset fails to download or load.
        """
        try:
            # If dataset already present, avoid re-downloading
            try:
                _ = CIFAR100(root=str(self.root_dir), train=True, download=False)
                _ = CIFAR100(root=str(self.root_dir), train=False, download=False)
                effective_download = False
                logger.info("CIFAR-100 detected locally. Skipping download.")
            except Exception:
                effective_download = self.download

            logger.info("Loading CIFAR-100 training dataset...")
            train_dataset = CIFAR100(
                root=str(self.root_dir),
                train=True,
                download=effective_download,
                transform=self.transform,
            )

            logger.info("Loading CIFAR-100 test dataset...")
            test_dataset = CIFAR100(
                root=str(self.root_dir),
                train=False,
                download=effective_download,
                transform=self.transform,
            )

        except Exception as e:
            raise RuntimeError(f"Failed to load CIFAR-100 dataset: {e}") from e

        if self.log_stats:
            self._log_dataset_stats(train_dataset, test_dataset)

        return train_dataset, test_dataset

    def _log_dataset_stats(self, train_dataset: CIFAR100, test_dataset: CIFAR100) -> None:
        """Log formatted dataset statistics."""
        total = len(train_dataset) + len(test_dataset)
        separator = "=" * 50

        logger.info("\n" + separator)
        logger.info("✅ CIFAR-100 Dataset Loaded Successfully!")
        logger.info(separator)
        logger.info("📊 Dataset Statistics:")
        logger.info(f"   • Training samples: {len(train_dataset):,}")
        logger.info(f"   • Test samples: {len(test_dataset):,}")
        logger.info(f"   • Total samples: {total:,}")
        logger.info("   • Image size: 32x32 pixels")
        logger.info("   • Color channels: 3 (RGB)")
        logger.info(f"   • Number of classes: {len(self.CLASS_NAMES)}")
        logger.info("   • Classes: Too many to list (100 classes)")
        logger.info(separator)

    def save_images_to_folders(self, train_dataset: CIFAR100, test_dataset: CIFAR100, val_split: float = 0.2) -> None:
        """
        Save CIFAR-100 images as PNG files into organized folders:
        - raw/train/<class_name>/
        - raw/val/<class_name>/
        - raw/test/<class_name>/

        Args:
            train_dataset: The loaded CIFAR-100 training dataset.
            test_dataset: The loaded CIFAR-100 test dataset.
            val_split: Fraction of training data to move to validation (default: 0.2).
        """
        # Define output directories
        train_image_dir = self.root_dir / "train"
        val_image_dir = self.root_dir / "val"
        test_image_dir = self.root_dir / "test"

        for class_name in self.CLASS_NAMES:
            (train_image_dir / class_name).mkdir(parents=True, exist_ok=True)
            (val_image_dir / class_name).mkdir(parents=True, exist_ok=True)
            (test_image_dir / class_name).mkdir(parents=True, exist_ok=True)

        # Split training data into train and validation
        train_indices, val_indices = self._split_train_dataset(train_dataset, val_split)

        logger.info("Saving training images...")
        self._save_dataset_images(train_dataset, train_image_dir, train_indices)

        logger.info("Saving validation images...")
        self._save_dataset_images(train_dataset, val_image_dir, val_indices)

        logger.info("Saving test images...")
        self._save_dataset_images(test_dataset, test_image_dir)

        logger.info("✅ All images saved successfully!")

    def _split_train_dataset(self, train_dataset: CIFAR100, val_split: float) -> Tuple[List[int], List[int]]:
        """
        Split training dataset indices into train and validation sets.

        Args:
            train_dataset: The training dataset to split.
            val_split: Fraction of data to allocate to validation.

        Returns:
            Tuple of (train_indices, val_indices)
        """
        total_samples = len(train_dataset)
        val_size = int(total_samples * val_split)
        train_size = total_samples - val_size

        # Create indices and shuffle them
        indices = list(range(total_samples))
        random.shuffle(indices)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        logger.info(f"Split training data: {len(train_indices)} train, {len(val_indices)} validation samples")

        return train_indices, val_indices

    def _save_dataset_images(self, dataset: CIFAR100, base_dir: Path, indices: Optional[List[int]] = None) -> None:
        """
        Helper to save images from a dataset into class-named subfolders.

        Args:
            dataset: The CIFAR-100 dataset (train, val, or test).
            base_dir: Root directory to save images under.
            indices: Optional list of indices to save. If None, saves all images.
        """
        if indices is None:
            indices = list(range(len(dataset)))

        for idx in indices:
            image, label = dataset[idx]
            # If transform includes ToTensor, convert back to PIL for saving
            if hasattr(image, "min") and image.min() < 0:  # Assume normalized to [-1, 1]
                image = image * 0.5 + 0.5
            if hasattr(image, "cpu"):
                image = image.cpu()
            if hasattr(image, "numpy"):
                import numpy as np

                image = image.numpy()
                if image.shape[0] == 3:  # CHW format
                    image = image.transpose(1, 2, 0)  # Convert to HWC
                image = (image * 255).astype(np.uint8)
                from PIL import Image

                image = Image.fromarray(image)
            elif hasattr(image, "permute"):
                # Handle tensor format
                image = transforms.functional.to_pil_image(image)

            class_name = self.CLASS_NAMES[label]
            file_path = base_dir / class_name / f"{idx}.png"
            image.save(file_path)


def download_and_extract_cifar10_data(
    root_dir: str = "data/raw",
    transform: Optional[transforms.Compose] = None,
    download: bool = True,
    log_stats: bool = True,
    val_split: float = 0.2,
) -> Tuple[CIFAR10, CIFAR10]:
    """
    Convenience function to download and extract CIFAR-10 dataset.

    Optionally saves images to organized train/val/test folders.

    Args:
        root_dir: Directory to store dataset.
        transform: Optional transform pipeline.
        download: Whether to download if not present.
        log_stats: Whether to print dataset statistics.
        val_split: Fraction of training data to move to validation (default: 0.2).

    Returns:
        Tuple of (train_dataset, test_dataset)

    Example:
        >>> train_data, test_data = download_and_extract_cifar10_data()
        >>> print(f"Train: {len(train_data)}, Test: {len(test_data)}")
    """
    downloader = CIFAR10Downloader(
        root_dir=root_dir,
        transform=transform,
        download=download,
        log_stats=log_stats,
    )
    train_dataset, test_dataset = downloader.load_datasets()

    if save_images:
        # Idempotency marker for exported images
        marker = Path(root_dir) / ".images_export_done_cifar10.json"
        if marker.exists():
            try:
                info = json.loads(marker.read_text())
                logger.info(
                    f"Image export previously completed (marker: {info}). Skipping export."
                )
            except Exception:
                logger.warning("Existing CIFAR-10 image export marker unreadable. Re-exporting.")
                downloader.save_images_to_folders(train_dataset, test_dataset)
        else:
            downloader.save_images_to_folders(train_dataset, test_dataset)
            try:
                marker.write_text(json.dumps({"root_dir": str(root_dir)}, indent=2))
            except Exception as e:
                logger.warning(f"Failed to write CIFAR-10 export marker: {e}")

    return train_dataset, test_dataset


def download_and_extract_cifar100_data(
    root_dir: str = "data/raw",
    transform: Optional[transforms.Compose] = None,
    download: bool = True,
    log_stats: bool = True,
    val_split: float = 0.2,
) -> Tuple[CIFAR100, CIFAR100]:
    """
    Convenience function to download and extract CIFAR-100 dataset.

    Optionally saves images to organized train/val/test folders.

    Args:
        root_dir: Directory to store dataset.
        transform: Optional transform pipeline.
        download: Whether to download if not present.
        log_stats: Whether to print dataset statistics.
        val_split: Fraction of training data to move to validation (default: 0.2).

    Returns:
        Tuple of (train_dataset, test_dataset)

    Example:
        >>> train_data, test_data = download_and_extract_cifar100_data()
        >>> print(f"Train: {len(train_data)}, Test: {len(test_data)}")
    """
    downloader = CIFAR100Downloader(
        root_dir=root_dir,
        transform=transform,
        download=download,
        log_stats=log_stats,
    )
    train_dataset, test_dataset = downloader.load_datasets()

    if save_images:
        # Idempotency marker for exported images
        marker = Path(root_dir) / ".images_export_done_cifar100.json"
        if marker.exists():
            try:
                info = json.loads(marker.read_text())
                logger.info(
                    f"Image export previously completed (marker: {info}). Skipping export."
                )
            except Exception:
                logger.warning("Existing CIFAR-100 image export marker unreadable. Re-exporting.")
                downloader.save_images_to_folders(train_dataset, test_dataset)
        else:
            downloader.save_images_to_folders(train_dataset, test_dataset)
            try:
                marker.write_text(json.dumps({"root_dir": str(root_dir)}, indent=2))
            except Exception as e:
                logger.warning(f"Failed to write CIFAR-100 export marker: {e}")

    return train_dataset, test_dataset
