"""FGSM (Fast Gradient Sign Method) attack on images.

Single-step gradient-based attack that perturbs each pixel by
±ε in the direction that decreases the NSFW probability.

Reference:
    Goodfellow et al., "Explaining and Harnessing Adversarial Examples", 2015
"""

import logging
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class FGSMConfig:
    """Configuration for FGSM attack.

    Attributes:
        epsilon: L-inf perturbation budget in [0, 1] pixel scale.
            Common values: 4/255, 8/255, 16/255.
        targeted: If True, push prediction toward *target_label*.
        target_label: Target label for the attack (0.0 = SFW).
        batch_size: Batch size for processing.
    """

    epsilon: float = 8 / 255
    targeted: bool = True
    target_label: float = 0.0
    batch_size: int = 8


class FGSMAttack:
    """FGSM attack on images through the end-to-end pipeline.

    Args:
        pipeline: :class:`~src.pipeline.EndToEndModel` instance.
        config: FGSM configuration.
    """

    def __init__(
        self,
        pipeline: tf.keras.Model,
        config: FGSMConfig | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.config = config or FGSMConfig()
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    def _compute_gradient(
        self, images: tf.Tensor, target_labels: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Compute loss gradient w.r.t. input images.

        Args:
            images: ``(N, H, W, 3)`` float32 in ``[0, 1]``.
            target_labels: ``(N, 1)`` target labels.

        Returns:
            ``(loss, gradient)`` tensors.
        """
        with tf.GradientTape() as tape:
            tape.watch(images)
            preds = self.pipeline(images, training=False)
            loss = self.loss_fn(target_labels, preds)

        gradient = tape.gradient(loss, images)
        return loss, gradient

    def attack(
        self,
        images: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run FGSM attack on a batch of images.

        Args:
            images: ``(N, H, W, 3)`` float32 in ``[0, 1]``.

        Returns:
            ``(adversarial_images, noise)`` both ``(N, H, W, 3)``.
        """
        cfg = self.config
        n = len(images)

        logger.info(
            "Running FGSM attack: ε=%.4f (%.1f/255), samples=%d",
            cfg.epsilon, cfg.epsilon * 255, n,
        )

        all_adv = np.zeros_like(images)
        all_noise = np.zeros_like(images)

        n_batches = (n + cfg.batch_size - 1) // cfg.batch_size

        for i in tqdm(range(n_batches), desc="FGSM Attack", unit="batch"):
            start = i * cfg.batch_size
            end = min(start + cfg.batch_size, n)
            batch = tf.constant(images[start:end], dtype=tf.float32)
            targets = tf.fill([end - start, 1], cfg.target_label)

            _, grad = self._compute_gradient(batch, targets)

            # Targeted attack -> minimise loss -> subtract sign(grad)
            if cfg.targeted:
                perturbation = -cfg.epsilon * tf.sign(grad)
            else:
                perturbation = cfg.epsilon * tf.sign(grad)

            adv = tf.clip_by_value(batch + perturbation, 0.0, 1.0)
            noise = adv - batch

            all_adv[start:end] = adv.numpy()
            all_noise[start:end] = noise.numpy()

        logger.info("FGSM attack complete")
        return all_adv, all_noise
