"""Fast Gradient Sign Method (FGSM) attack implementation.

FGSM is a single-step gradient-based adversarial attack that perturbs
inputs in the direction of the gradient of the loss with respect to
the input, scaled by epsilon.

Reference:
    Goodfellow et al., "Explaining and Harnessing Adversarial Examples", 2015
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import tensorflow as tf
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class FGSMConfig:
    """Configuration for FGSM attack.

    Attributes:
        epsilon: Maximum perturbation magnitude (L-inf norm).
        targeted: Whether to perform targeted attack.
        target_label: Target label for targeted attack (0.0 = SFW).
        batch_size: Batch size for processing.
        clip_min: Minimum value for clipping adversarial embeddings.
        clip_max: Maximum value for clipping adversarial embeddings.
    """

    epsilon: float = 0.05
    targeted: bool = True
    target_label: float = 0.0
    batch_size: int = 256
    clip_min: float | None = None
    clip_max: float | None = None


class FGSMAttack:
    """Fast Gradient Sign Method attack.

    Generates adversarial perturbations using a single gradient step
    to cause misclassification of NSFW embeddings.

    Args:
        model: Target classifier model.
        config: FGSM configuration.
    """

    def __init__(self, model: tf.keras.Model, config: FGSMConfig | None = None) -> None:
        self.model = model
        self.config = config or FGSMConfig()
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    def _compute_gradient(
        self, embeddings: tf.Tensor, target_labels: tf.Tensor
    ) -> tf.Tensor:
        """Compute gradient of loss with respect to input embeddings.

        Args:
            embeddings: Input embeddings tensor.
            target_labels: Target labels for the attack.

        Returns:
            Gradient tensor.
        """
        with tf.GradientTape() as tape:
            tape.watch(embeddings)
            predictions = self.model(embeddings, training=False)
            loss = self.loss_fn(target_labels, predictions)

        gradient = tape.gradient(loss, embeddings)
        return gradient

    def attack(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run FGSM attack on embeddings.

        Args:
            embeddings: Original embeddings of shape (N, 256).

        Returns:
            Tuple of (adversarial_embeddings, noise) arrays.
        """
        n_samples = len(embeddings)
        adversarial = np.copy(embeddings)
        noise = np.zeros_like(embeddings)

        target_val = self.config.target_label
        batch_size = self.config.batch_size
        epsilon = self.config.epsilon

        logger.info(
            "Running FGSM attack: epsilon=%.4f, samples=%d",
            epsilon, n_samples,
        )

        n_batches = (n_samples + batch_size - 1) // batch_size

        for i in tqdm(range(n_batches), desc="FGSM Attack", unit="batch"):
            start = i * batch_size
            end = min(start + batch_size, n_samples)
            batch = tf.constant(embeddings[start:end], dtype=tf.float32)
            targets = tf.fill([end - start, 1], target_val)

            gradient = self._compute_gradient(batch, targets)

            # FGSM: perturb in the direction that minimizes loss toward target
            if self.config.targeted:
                perturbation = -epsilon * tf.sign(gradient)
            else:
                perturbation = epsilon * tf.sign(gradient)

            adv_batch = batch + perturbation

            # Clip if bounds are specified
            if self.config.clip_min is not None and self.config.clip_max is not None:
                adv_batch = tf.clip_by_value(
                    adv_batch, self.config.clip_min, self.config.clip_max
                )

            adversarial[start:end] = adv_batch.numpy()
            noise[start:end] = (adv_batch - batch).numpy()

        logger.info("FGSM attack complete")
        return adversarial, noise

    def attack_single(self, embedding: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run FGSM attack on a single embedding.

        Args:
            embedding: Single embedding of shape (256,).

        Returns:
            Tuple of (adversarial_embedding, noise).
        """
        emb = embedding.reshape(1, -1)
        adv, n = self.attack(emb)
        return adv.flatten(), n.flatten()
