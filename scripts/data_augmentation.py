import logging
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import albumentations as A
from PIL import Image
from tqdm import tqdm
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageAugmenter:
    """Class to handle image augmentation operations using Albumentations."""

    def __init__(
        self,
        augmentations_per_image: int = 5,
        seed: int = 42,
        save_original: bool = True,
        image_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
        output_format: str = "jpg",
        jpeg_quality: int = 90,
    ):
        """
        Initialize the ImageAugmenter.

        Args:
            augmentations_per_image: Number of augmented versions per original image.
            seed: Random seed for reproducibility.
            save_original: Whether to save the original image with prefix 'orig_'.
            image_extensions: Tuple of valid image file extensions.
            output_format: Image format to save augmented files ('jpg' or 'png').
            jpeg_quality: JPEG quality (only used when output_format='jpg').
        """
        self.augmentations_per_image = augmentations_per_image
        self.seed = seed
        self.save_original = save_original
        self.image_extensions = image_extensions
        self.output_format = output_format.lower()
        self.jpeg_quality = jpeg_quality

        self._set_seed()

        # Define Albumentations pipeline
        self.transform = A.Compose(
            [
                A.Rotate(limit=15, p=0.8),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=0,
                    p=0.8,
                    border_mode=0,  # cv2.BORDER_CONSTANT
                ),
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8
                ),
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                        A.MotionBlur(blur_limit=7, p=0.5),
                    ],
                    p=0.3,
                ),
                A.RandomBrightnessContrast(p=0.2),
            ]
        )

    def _set_seed(self):
        """Set random seeds for reproducibility."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def augment_image(self, image: Image.Image) -> Image.Image:
        """
        Apply augmentation transforms to a single image using Albumentations.

        Args:
            image: PIL Image to augment.

        Returns:
            Augmented PIL Image.
        """
        # Convert PIL to NumPy array (RGB)
        image_np = np.array(image)

        # Apply Albumentations transform
        augmented = self.transform(image=image_np)
        augmented_image_np = augmented["image"]

        # Convert back to PIL Image
        return Image.fromarray(augmented_image_np.astype(np.uint8))

    def process_directory(self, input_dir: str, output_dir: str) -> None:
        """
        Augment all images in input directory and save to output directory.

        Preserves folder structure. Skips files that fail to load.

        Args:
            input_dir: Path to input directory with class subfolders.
            output_dir: Path to output directory for augmented images.
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        count = 0

        # Idempotency: if a completion marker exists, skip processing
        marker = output_path / ".augmentation_done.json"
        if marker.exists():
            try:
                info = json.loads(marker.read_text())
                logger.info(
                    "Augmentation already completed previously. "
                    f"Found marker with settings: {info}. Skipping."
                )
                return
            except Exception:
                # If marker is unreadable, fall through and regenerate
                logger.warning("Existing augmentation marker unreadable. Re-running augmentation.")

        image_files = self._find_image_files(input_path)

        logger.info(f"Found {len(image_files)} images to augment.")

        pbar = tqdm(image_files, desc="Augmenting images", unit="img", leave=True)
        for img_path in pbar:
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                logger.warning(f"Failed to load image {img_path}: {e}")
                continue

            # Determine output subdirectory
            rel_dir = img_path.parent.relative_to(input_path)
            target_dir = output_path / rel_dir
            if not target_dir.exists():
                target_dir.mkdir(parents=True, exist_ok=True)

            # Determine output extension and save params
            base = img_path.stem
            if self.output_format in ("jpg", "jpeg"):
                out_ext = ".jpg"
                save_kwargs = {"format": "JPEG", "quality": self.jpeg_quality, "subsampling": 2}
            else:
                out_ext = ".png"
                save_kwargs = {"format": "PNG", "optimize": True}

            # Save original if requested
            if self.save_original:
                orig_name = f"orig_{base}{out_ext}"
                image.save(target_dir / orig_name, **save_kwargs)
                count += 1
                pbar.set_postfix({"generated": count})

            # Generate and save augmented versions
            for i in range(self.augmentations_per_image):
                augmented = self.augment_image(image.copy())
                aug_name = f"aug_{i}_{base}{out_ext}"
                augmented.save(target_dir / aug_name, **save_kwargs)
                count += 1
                pbar.set_postfix({"generated": count})

        # Write completion marker for idempotency
        try:
            marker.write_text(
                json.dumps(
                    {
                        "augmentations_per_image": self.augmentations_per_image,
                        "save_original": self.save_original,
                        "output_format": getattr(self, "output_format", "jpg"),
                        "jpeg_quality": getattr(self, "jpeg_quality", 90),
                        "seed": self.seed,
                    },
                    indent=2,
                )
            )
        except Exception as e:
            logger.warning(f"Failed to write augmentation marker: {e}")

        logger.info(
            f"Augmentation of {count} images completed. Output saved to: {output_dir}"
        )

    def _find_image_files(self, root: Path) -> List[Path]:
        """
        Recursively find all image files in directory.

        Args:
            root: Root directory path.

        Returns:
            List of image file paths.
        """
        files = []
        for ext in self.image_extensions:
            files.extend(root.rglob(f"*{ext}"))
        return files


def augment_dataset(
    input_dir: str,
    output_dir: str,
    augmentations_per_image: int = 5,
    seed: int = 42,
) -> None:
    """
    Backward-compatible wrapper for legacy code.

    Args:
        input_dir: Directory containing cleaned images (organized by class).
        output_dir: Directory to save augmented images.
        augmentations_per_image: Number of augmented versions per original image.
        seed: Random seed for reproducibility.
    """
    augmenter = ImageAugmenter(
        augmentations_per_image=augmentations_per_image, seed=seed, save_original=True
    )
    augmenter.process_directory(input_dir, output_dir)
