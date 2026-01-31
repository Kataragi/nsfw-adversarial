"""Utility functions for image-based adversarial attacks."""

import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tensorflow as tf
import yaml

logger = logging.getLogger(__name__)

IMAGE_SIZE = 320


def setup_logging(level: str = "INFO", log_format: str | None = None) -> None:
    """Configure logging for the application."""
    if log_format is None:
        log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=getattr(logging, level.upper()), format=log_format)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info("Random seed set to %d", seed)


def load_config(config_path: str = "config/attack_config.yaml") -> dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info("Configuration loaded from %s", config_path)
    return config


# ── Image I/O ──────────────────────────────────────────────────────


def load_image(path: str, size: int = IMAGE_SIZE) -> np.ndarray:
    """Load and preprocess a single image.

    Args:
        path: Path to the image file.
        size: Target size (square).

    Returns:
        Image array (H, W, 3) float32 in [0, 1].
    """
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    return (img.astype(np.float32) / 255.0)


def load_images(
    image_dir: str, size: int = IMAGE_SIZE, max_images: int | None = None
) -> tuple[np.ndarray, list[str]]:
    """Load all images from a directory.

    Args:
        image_dir: Directory containing image files.
        size: Target size (square).
        max_images: Optional cap on number of images.

    Returns:
        Tuple of (images array (N,H,W,3), list of file paths).
    """
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_dir_path = Path(image_dir)
    files = sorted(
        f for f in image_dir_path.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    )
    if not files:
        raise FileNotFoundError(f"No images found in {image_dir}")

    if max_images is not None:
        files = files[:max_images]

    images = []
    valid_paths: list[str] = []
    for f in files:
        try:
            img = load_image(str(f), size)
            images.append(img)
            valid_paths.append(str(f))
        except Exception as e:
            logger.warning("Skipping %s: %s", f.name, e)

    result = np.stack(images, axis=0)
    logger.info(
        "Loaded %d images from %s  shape=%s", len(result), image_dir, result.shape
    )
    return result, valid_paths


def save_adversarial_images(
    adversarial: np.ndarray,
    output_dir: str,
    filenames: list[str] | None = None,
    prefix: str = "adv",
) -> list[str]:
    """Save adversarial images as PNG files.

    Args:
        adversarial: (N, H, W, 3) float32 in [0, 1].
        output_dir: Output directory.
        filenames: Original filenames (stems used for naming).
        prefix: Filename prefix when filenames is None.

    Returns:
        List of saved file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved: list[str] = []
    for i, img in enumerate(adversarial):
        if filenames is not None:
            stem = Path(filenames[i]).stem
            name = f"{prefix}_{stem}.png"
        else:
            name = f"{prefix}_{i:06d}.png"
        path = os.path.join(output_dir, name)
        img_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, img_bgr)
        saved.append(path)
    logger.info("Saved %d adversarial images to %s", len(saved), output_dir)
    return saved


def save_noise_images(
    noise: np.ndarray,
    output_dir: str,
    amplify: float = 10.0,
    prefix: str = "noise",
) -> None:
    """Save noise visualisations (amplified for visibility).

    Args:
        noise: (N, H, W, 3) float32 perturbation.
        output_dir: Output directory.
        amplify: Amplification factor for visualisation.
        prefix: Filename prefix.
    """
    os.makedirs(output_dir, exist_ok=True)
    for i, n in enumerate(noise):
        vis = np.clip(0.5 + n * amplify, 0, 1)
        vis_uint8 = (vis * 255).astype(np.uint8)
        vis_bgr = cv2.cvtColor(vis_uint8, cv2.COLOR_RGB2BGR)
        path = os.path.join(output_dir, f"{prefix}_{i:06d}.png")
        cv2.imwrite(path, vis_bgr)


# ── Metrics ────────────────────────────────────────────────────────


def compute_metrics(
    original_probs: np.ndarray,
    adversarial_probs: np.ndarray,
    noise: np.ndarray,
    thresholds: list[float] | None = None,
) -> dict[str, Any]:
    """Compute evaluation metrics for adversarial attacks.

    Args:
        original_probs: Original NSFW probabilities (N,).
        adversarial_probs: Post-attack NSFW probabilities (N,).
        noise: Perturbation array. (N, H, W, 3) for images.
        thresholds: Threshold values for success rate calculation.

    Returns:
        Dictionary of metrics.
    """
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5]

    n_samples = len(original_probs)
    nsfw_mask = original_probs >= 0.5
    nsfw_original = original_probs[nsfw_mask]
    nsfw_adversarial = adversarial_probs[nsfw_mask]
    n_nsfw = int(nsfw_mask.sum())

    metrics: dict[str, Any] = {
        "total_samples": n_samples,
        "nsfw_samples": n_nsfw,
    }

    if n_nsfw > 0:
        for t in thresholds:
            success = (nsfw_adversarial < t).sum()
            metrics[f"success_rate_{t}"] = float(success / n_nsfw)

        metrics["avg_prob_reduction"] = float(
            (nsfw_original - nsfw_adversarial).mean()
        )
        metrics["avg_original_prob"] = float(nsfw_original.mean())
        metrics["avg_adversarial_prob"] = float(nsfw_adversarial.mean())

    # Noise norms – flatten spatial dims per sample
    if noise.ndim >= 2:
        flat = noise.reshape(n_samples, -1)
        l2_norms = np.linalg.norm(flat, axis=1)
        linf_norms = np.abs(flat).max(axis=1)
        metrics["avg_l2_norm"] = float(l2_norms.mean())
        metrics["avg_linf_norm"] = float(linf_norms.mean())
        metrics["max_l2_norm"] = float(l2_norms.max())
        metrics["max_linf_norm"] = float(linf_norms.max())
        # Per-pixel metrics (more intuitive for images)
        metrics["avg_linf_norm_pixel"] = float(linf_norms.mean() * 255)

    return metrics


# ── Result I/O ─────────────────────────────────────────────────────


def save_results(
    results: dict[str, Any],
    output_dir: str,
    prefix: str = "attack_results",
) -> str:
    """Save results as JSON."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"{prefix}_{timestamp}.json")

    def convert(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=convert)

    logger.info("Results saved to %s", filepath)
    return filepath


# ── TensorBoard / GPU ──────────────────────────────────────────────


def setup_tensorboard(log_dir: str) -> tf.summary.SummaryWriter | None:
    """Set up TensorBoard writer."""
    os.makedirs(log_dir, exist_ok=True)
    run_dir = os.path.join(log_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    writer = tf.summary.create_file_writer(run_dir)
    logger.info("TensorBoard logging to %s", run_dir)
    return writer


def detect_gpu() -> bool:
    """Detect and log GPU availability."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        logger.info("GPU detected: %s", [g.name for g in gpus])
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        return True
    logger.info("No GPU detected, using CPU")
    return False
