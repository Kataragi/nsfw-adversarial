"""Utility functions for adversarial attacks."""

import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
import yaml

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO", log_format: str | None = None) -> None:
    """Configure logging for the application.

    Args:
        level: Logging level string.
        log_format: Optional format string.
    """
    if log_format is None:
        log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=getattr(logging, level.upper()), format=log_format)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info("Random seed set to %d", seed)


def load_config(config_path: str = "config/attack_config.yaml") -> dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Configuration dictionary.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info("Configuration loaded from %s", config_path)
    return config


def load_target_model(model_path: str) -> tf.keras.Model:
    """Load the target NSFW classifier model.

    Args:
        model_path: Path to the .keras model file.

    Returns:
        Loaded Keras model.
    """
    model = tf.keras.models.load_model(model_path)
    logger.info("Target model loaded from %s", model_path)
    logger.info("Model summary: %d parameters", model.count_params())
    return model


def load_embeddings(embeddings_dir: str) -> np.ndarray:
    """Load embedding vectors from a directory.

    Supports .npy and .npz files. All embeddings are concatenated
    into a single array.

    Args:
        embeddings_dir: Directory containing embedding files.

    Returns:
        Array of shape (N, 256) containing all embeddings.
    """
    embeddings_path = Path(embeddings_dir)
    all_embeddings = []

    # Load .npy files
    for npy_file in sorted(embeddings_path.glob("*.npy")):
        data = np.load(str(npy_file))
        if data.ndim == 1:
            data = data.reshape(1, -1)
        all_embeddings.append(data)
        logger.debug("Loaded %d embeddings from %s", len(data), npy_file.name)

    # Load .npz files
    for npz_file in sorted(embeddings_path.glob("*.npz")):
        data = np.load(str(npz_file))
        for key in data.files:
            arr = data[key]
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            all_embeddings.append(arr)
        logger.debug("Loaded embeddings from %s", npz_file.name)

    if not all_embeddings:
        raise FileNotFoundError(f"No embedding files found in {embeddings_dir}")

    result = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    logger.info("Loaded %d embeddings with dimension %d", result.shape[0], result.shape[1])
    return result


def predict_batch(
    model: tf.keras.Model,
    embeddings: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    """Run batch predictions on embeddings.

    Args:
        model: Target classifier model.
        embeddings: Input embeddings of shape (N, 256).
        batch_size: Batch size for prediction.

    Returns:
        Array of NSFW probabilities of shape (N,).
    """
    predictions = model.predict(embeddings, batch_size=batch_size, verbose=0)
    return predictions.flatten()


def compute_metrics(
    original_probs: np.ndarray,
    adversarial_probs: np.ndarray,
    noise: np.ndarray,
    thresholds: list[float] | None = None,
) -> dict[str, Any]:
    """Compute evaluation metrics for adversarial attacks.

    Args:
        original_probs: Original NSFW probabilities.
        adversarial_probs: Post-attack NSFW probabilities.
        noise: Adversarial perturbation vectors.
        thresholds: Threshold values for success rate calculation.

    Returns:
        Dictionary containing all metrics.
    """
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5]

    n_samples = len(original_probs)

    # Filter only NSFW samples (original prob >= 0.5)
    nsfw_mask = original_probs >= 0.5
    nsfw_original = original_probs[nsfw_mask]
    nsfw_adversarial = adversarial_probs[nsfw_mask]
    nsfw_noise = noise[nsfw_mask] if noise.ndim > 1 else noise

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

    if noise.ndim == 2:
        l2_norms = np.linalg.norm(noise, axis=1)
        linf_norms = np.abs(noise).max(axis=1)
        metrics["avg_l2_norm"] = float(l2_norms.mean())
        metrics["avg_linf_norm"] = float(linf_norms.mean())
        metrics["max_l2_norm"] = float(l2_norms.max())
        metrics["max_linf_norm"] = float(linf_norms.max())

    return metrics


def save_results(
    results: dict[str, Any],
    output_dir: str,
    prefix: str = "attack_results",
) -> str:
    """Save attack results to a JSON file.

    Args:
        results: Results dictionary.
        output_dir: Output directory path.
        prefix: Filename prefix.

    Returns:
        Path to the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    # Convert numpy types for JSON serialization
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


def save_adversarial_embeddings(
    adversarial: np.ndarray,
    noise: np.ndarray,
    output_dir: str,
    prefix: str = "adversarial",
) -> None:
    """Save adversarial embeddings and noise vectors.

    Args:
        adversarial: Adversarial embedding vectors.
        noise: Noise vectors.
        output_dir: Output directory.
        prefix: Filename prefix.
    """
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f"{prefix}_embeddings.npy"), adversarial)
    np.save(os.path.join(output_dir, f"{prefix}_noise.npy"), noise)
    logger.info("Adversarial embeddings saved to %s", output_dir)


def setup_tensorboard(log_dir: str) -> tf.summary.SummaryWriter | None:
    """Set up TensorBoard writer.

    Args:
        log_dir: TensorBoard log directory.

    Returns:
        TensorBoard SummaryWriter or None if setup fails.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(log_dir, timestamp)
    writer = tf.summary.create_file_writer(run_dir)
    logger.info("TensorBoard logging to %s", run_dir)
    return writer


def detect_gpu() -> bool:
    """Detect and log GPU availability.

    Returns:
        True if GPU is available.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        logger.info("GPU detected: %s", [g.name for g in gpus])
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        return True
    logger.info("No GPU detected, using CPU")
    return False
