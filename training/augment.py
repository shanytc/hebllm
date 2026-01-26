#!/usr/bin/env python3
"""
Data Augmentation for HebLLM Training

Implements augmentations designed for document OCR:
- Rotation, perspective, and scaling transforms
- Color and contrast adjustments
- Noise and blur (simulating scan artifacts)
- Compression artifacts
"""

import random
from typing import Optional, Callable, Tuple

import torch
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps


class DocumentAugmentation:
    """
    Augmentation pipeline for document images.

    Designed to simulate real-world document variations:
    - Scanning artifacts
    - Camera capture distortions
    - Lighting variations
    - Paper quality differences
    """

    def __init__(self,
                 rotation_range: Tuple[float, float] = (-3, 3),
                 scale_range: Tuple[float, float] = (0.95, 1.05),
                 brightness_range: Tuple[float, float] = (0.9, 1.1),
                 contrast_range: Tuple[float, float] = (0.9, 1.1),
                 blur_prob: float = 0.1,
                 noise_prob: float = 0.1,
                 jpeg_prob: float = 0.2,
                 perspective_prob: float = 0.1,
                 enabled: bool = True):
        """
        Initialize augmentation pipeline.

        Args:
            rotation_range: Min/max rotation in degrees
            scale_range: Min/max scale factor
            brightness_range: Min/max brightness multiplier
            contrast_range: Min/max contrast multiplier
            blur_prob: Probability of applying blur
            noise_prob: Probability of adding noise
            jpeg_prob: Probability of JPEG compression artifacts
            perspective_prob: Probability of perspective transform
            enabled: Whether augmentation is active
        """
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.blur_prob = blur_prob
        self.noise_prob = noise_prob
        self.jpeg_prob = jpeg_prob
        self.perspective_prob = perspective_prob
        self.enabled = enabled

    def __call__(self, image: Image.Image) -> Image.Image:
        """Apply augmentations to image."""
        if not self.enabled:
            return image

        # Rotation
        if self.rotation_range[1] > self.rotation_range[0]:
            angle = random.uniform(*self.rotation_range)
            image = image.rotate(angle, fillcolor=(255, 255, 255), expand=False)

        # Scale (resize then crop/pad back to original size)
        if self.scale_range[1] > self.scale_range[0]:
            scale = random.uniform(*self.scale_range)
            if scale != 1.0:
                orig_size = image.size
                new_size = (int(orig_size[0] * scale), int(orig_size[1] * scale))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                image = self._crop_or_pad(image, orig_size)

        # Brightness
        if self.brightness_range[1] > self.brightness_range[0]:
            factor = random.uniform(*self.brightness_range)
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(factor)

        # Contrast
        if self.contrast_range[1] > self.contrast_range[0]:
            factor = random.uniform(*self.contrast_range)
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(factor)

        # Blur
        if random.random() < self.blur_prob:
            radius = random.uniform(0.5, 1.5)
            image = image.filter(ImageFilter.GaussianBlur(radius))

        # Noise
        if random.random() < self.noise_prob:
            image = self._add_noise(image)

        # JPEG compression artifacts
        if random.random() < self.jpeg_prob:
            image = self._jpeg_compress(image)

        # Perspective (simulate camera angle)
        if random.random() < self.perspective_prob:
            image = self._perspective_transform(image)

        return image

    def _crop_or_pad(self, image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """Crop or pad image to target size."""
        w, h = image.size
        tw, th = target_size

        if w == tw and h == th:
            return image

        # Create new image with white background
        result = Image.new("RGB", target_size, (255, 255, 255))

        # Calculate paste position (center)
        paste_x = (tw - w) // 2
        paste_y = (th - h) // 2

        # Crop source if larger than target
        src_x = max(0, -paste_x)
        src_y = max(0, -paste_y)
        src_w = min(w, tw)
        src_h = min(h, th)

        # Paste position adjustment
        paste_x = max(0, paste_x)
        paste_y = max(0, paste_y)

        # Crop and paste
        cropped = image.crop((src_x, src_y, src_x + src_w, src_y + src_h))
        result.paste(cropped, (paste_x, paste_y))

        return result

    def _add_noise(self, image: Image.Image, intensity: float = 0.02) -> Image.Image:
        """Add Gaussian noise to image."""
        arr = np.array(image).astype(np.float32)
        noise = np.random.normal(0, intensity * 255, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def _jpeg_compress(self, image: Image.Image, quality_range: Tuple[int, int] = (60, 90)) -> Image.Image:
        """Apply JPEG compression artifacts."""
        import io

        quality = random.randint(*quality_range)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    def _perspective_transform(self, image: Image.Image, magnitude: float = 0.05) -> Image.Image:
        """Apply perspective transform."""
        w, h = image.size

        # Random corner offsets
        m = magnitude * min(w, h)

        # Source corners (original)
        src_corners = [(0, 0), (w, 0), (w, h), (0, h)]

        # Destination corners (with random offsets)
        dst_corners = [
            (random.uniform(0, m), random.uniform(0, m)),
            (w - random.uniform(0, m), random.uniform(0, m)),
            (w - random.uniform(0, m), h - random.uniform(0, m)),
            (random.uniform(0, m), h - random.uniform(0, m))
        ]

        # Compute perspective transform coefficients
        coeffs = self._find_perspective_coeffs(dst_corners, src_corners)

        return image.transform(image.size, Image.Transform.PERSPECTIVE, coeffs,
                               Image.Resampling.BICUBIC, fillcolor=(255, 255, 255))

    @staticmethod
    def _find_perspective_coeffs(src_coords, dst_coords):
        """Calculate perspective transform coefficients."""
        matrix = []
        for s, d in zip(src_coords, dst_coords):
            matrix.append([d[0], d[1], 1, 0, 0, 0, -s[0]*d[0], -s[0]*d[1]])
            matrix.append([0, 0, 0, d[0], d[1], 1, -s[1]*d[0], -s[1]*d[1]])

        A = np.array(matrix, dtype=np.float64)
        B = np.array([s for src in src_coords for s in src], dtype=np.float64)

        res = np.linalg.lstsq(A, B, rcond=None)[0]
        return tuple(res.tolist())


class OCRAugmentation:
    """
    Augmentation specifically designed for OCR training.

    More conservative than general document augmentation,
    preserving text readability while adding variation.
    """

    def __init__(self,
                 mode: str = "light",  # light, medium, heavy
                 target_size: Tuple[int, int] = None):
        """
        Initialize OCR augmentation.

        Args:
            mode: Augmentation intensity
            target_size: Optional target size for resize
        """
        self.mode = mode
        self.target_size = target_size

        # Mode-specific settings
        if mode == "light":
            self.aug = DocumentAugmentation(
                rotation_range=(-1, 1),
                scale_range=(0.98, 1.02),
                brightness_range=(0.95, 1.05),
                contrast_range=(0.95, 1.05),
                blur_prob=0.05,
                noise_prob=0.05,
                jpeg_prob=0.1,
                perspective_prob=0.02
            )
        elif mode == "medium":
            self.aug = DocumentAugmentation(
                rotation_range=(-2, 2),
                scale_range=(0.95, 1.05),
                brightness_range=(0.9, 1.1),
                contrast_range=(0.9, 1.1),
                blur_prob=0.1,
                noise_prob=0.1,
                jpeg_prob=0.2,
                perspective_prob=0.05
            )
        else:  # heavy
            self.aug = DocumentAugmentation(
                rotation_range=(-3, 3),
                scale_range=(0.9, 1.1),
                brightness_range=(0.8, 1.2),
                contrast_range=(0.85, 1.15),
                blur_prob=0.15,
                noise_prob=0.15,
                jpeg_prob=0.3,
                perspective_prob=0.1
            )

    def __call__(self, image: Image.Image) -> Image.Image:
        """Apply OCR-safe augmentations."""
        # Apply augmentations
        image = self.aug(image)

        # Resize if needed
        if self.target_size:
            image = image.resize(self.target_size, Image.Resampling.LANCZOS)

        return image


class ToTensor:
    """Convert PIL Image to PyTorch tensor."""

    def __init__(self, normalize: bool = True):
        """
        Initialize converter.

        Args:
            normalize: Whether to normalize to [0, 1]
        """
        self.normalize = normalize

    def __call__(self, image: Image.Image) -> torch.Tensor:
        """Convert image to tensor."""
        arr = np.array(image)

        # Handle grayscale
        if len(arr.shape) == 2:
            arr = np.stack([arr] * 3, axis=-1)

        # HWC to CHW
        tensor = torch.from_numpy(arr).permute(2, 0, 1)

        if self.normalize:
            tensor = tensor.float() / 255.0

        return tensor


class Compose:
    """Compose multiple transforms."""

    def __init__(self, transforms: list):
        """
        Initialize composition.

        Args:
            transforms: List of transform callables
        """
        self.transforms = transforms

    def __call__(self, image: Image.Image):
        """Apply all transforms in sequence."""
        for t in self.transforms:
            image = t(image)
        return image


def get_training_transforms(
    image_size: Tuple[int, int] = (768, 768),
    augmentation_mode: str = "medium"
) -> Callable:
    """
    Get standard training transforms.

    Args:
        image_size: Target image size (width, height)
        augmentation_mode: Augmentation intensity

    Returns:
        Composed transform function
    """
    return Compose([
        OCRAugmentation(mode=augmentation_mode, target_size=image_size),
        ToTensor(normalize=True)
    ])


def get_validation_transforms(
    image_size: Tuple[int, int] = (768, 768)
) -> Callable:
    """
    Get validation transforms (no augmentation).

    Args:
        image_size: Target image size

    Returns:
        Composed transform function
    """
    class Resize:
        def __init__(self, size):
            self.size = size
        def __call__(self, img):
            return img.resize(self.size, Image.Resampling.LANCZOS)

    return Compose([
        Resize(image_size),
        ToTensor(normalize=True)
    ])


if __name__ == "__main__":
    # Test augmentations
    print("Testing augmentations...")

    # Create test image
    test_img = Image.new("RGB", (800, 600), color="white")

    # Apply augmentations
    aug = DocumentAugmentation()
    augmented = aug(test_img)
    print(f"Original size: {test_img.size}, Augmented size: {augmented.size}")

    # Test training transforms
    transforms = get_training_transforms((768, 768), "medium")
    tensor = transforms(test_img)
    print(f"Output tensor shape: {tensor.shape}, dtype: {tensor.dtype}")
