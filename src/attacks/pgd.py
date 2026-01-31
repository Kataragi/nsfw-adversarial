"""Projected Gradient Descent (PGD) attack implementation.

PGD is an iterative gradient-based adversarial attack that applies
multiple small perturbation steps, projecting back onto the epsilon-ball
after each step.

Reference:
    Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks", 2018
"""

import logging
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class PGDConfig:
    """Configuration for PGD attack.

    Attributes:
        epsilon: Maximum perturbation magnitude (L-inf norm).
        alpha: Step size per iteration.
        iterations: Number of attack iterations.
        random_start: Whether to start from a random point within the epsilon-ball.
        targeted: Whether to perform targeted attack.
        target_label: Target label for targeted attack.
        batch_size: Batch size for processing.
        clip_min: Minimum value for clipping.
        clip_max: Maximum value for clipping.
        early_stop: Stop early if all samples are successfully attacked.
        early_stop_threshold: Threshold for early stopping.
    """

    epsilon: float = 0.05
    alpha: float = 0.01
    iterations: int = 20
    random_start: bool = True
    targeted: bool = True
    target_label: float = 0.0
    batch_size: int = 256
    clip_min: float | None = None
    clip_max: float | None = None
    early_stop: bool = True
    early_stop_threshold: float = 0.3


class PGDAttack:
    """Projected Gradient Descent attack.

    Iteratively perturbs embeddings using gradient descent with
    projection onto the L-inf epsilon-ball.

    Args:
        model: Target classifier model.
        config: PGD configuration.
    """

    def __init__(self, model: tf.keras.Model, config: PGDConfig | None = None) -> None:
        self.model = model
        self.config = config or PGDConfig()
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    def _project(
        self, adversarial: tf.Tensor, original: tf.Tensor, epsilon: float
    ) -> tf.Tensor:
        """Project adversarial examples back onto the L-inf epsilon-ball.

        Args:
            adversarial: Current adversarial embeddings.
            original: Original clean embeddings.
            epsilon: Maximum perturbation magnitude.

        Returns:
            Projected adversarial embeddings.
        """
        perturbation = adversarial - original
        perturbation = tf.clip_by_value(perturbation, -epsilon, epsilon)
        projected = original + perturbation

        if self.config.clip_min is not None and self.config.clip_max is not None:
            projected = tf.clip_by_value(
                projected, self.config.clip_min, self.config.clip_max
            )

        return projected

    def _attack_batch(
        self,
        embeddings: np.ndarray,
        tb_writer: tf.summary.SummaryWriter | None = None,
        global_step: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run PGD attack on a batch of embeddings.

        Args:
            embeddings: Batch of embeddings.
            tb_writer: Optional TensorBoard writer.
            global_step: Global step for TensorBoard logging.

        Returns:
            Tuple of (adversarial, noise, iterations_used).
        """
        batch_size = len(embeddings)
        original = tf.constant(embeddings, dtype=tf.float32)
        target_labels = tf.fill([batch_size, 1], self.config.target_label)
        epsilon = self.config.epsilon
        alpha = self.config.alpha

        # Initialize with random start
        if self.config.random_start:
            random_noise = tf.random.uniform(
                original.shape, -epsilon, epsilon, dtype=tf.float32
            )
            adv = original + random_noise
            adv = self._project(adv, original, epsilon)
        else:
            adv = tf.identity(original)

        iterations_used = np.full(batch_size, self.config.iterations, dtype=np.int32)
        best_adv = adv.numpy()
        best_probs = np.ones(batch_size, dtype=np.float32)

        for step in range(self.config.iterations):
            adv_var = tf.Variable(adv)

            with tf.GradientTape() as tape:
                predictions = self.model(adv_var, training=False)
                loss = self.loss_fn(target_labels, predictions)

            gradient = tape.gradient(loss, adv_var)

            # Update step
            if self.config.targeted:
                adv = adv_var - alpha * tf.sign(gradient)
            else:
                adv = adv_var + alpha * tf.sign(gradient)

            # Project back onto epsilon-ball
            adv = self._project(adv, original, epsilon)

            # Track best adversarial examples
            current_probs = self.model(adv, training=False).numpy().flatten()
            improved = current_probs < best_probs
            best_probs = np.where(improved, current_probs, best_probs)
            best_adv = np.where(
                improved[:, np.newaxis], adv.numpy(), best_adv
            )

            # Track first successful iteration
            success_mask = (current_probs < self.config.early_stop_threshold) & (
                iterations_used == self.config.iterations
            )
            iterations_used[success_mask] = step + 1

            # TensorBoard logging
            if tb_writer is not None:
                with tb_writer.as_default():
                    tf.summary.scalar(
                        "pgd/avg_prob", np.mean(current_probs),
                        step=global_step + step,
                    )
                    tf.summary.scalar(
                        "pgd/loss", float(loss), step=global_step + step,
                    )
                    tf.summary.scalar(
                        "pgd/success_rate",
                        float((current_probs < 0.5).mean()),
                        step=global_step + step,
                    )

            # Early stopping
            if self.config.early_stop and np.all(
                current_probs < self.config.early_stop_threshold
            ):
                logger.debug("Early stop at iteration %d", step + 1)
                break

        noise = best_adv - embeddings
        return best_adv, noise, iterations_used

    def attack(
        self,
        embeddings: np.ndarray,
        tb_writer: tf.summary.SummaryWriter | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run PGD attack on all embeddings.

        Args:
            embeddings: Original embeddings of shape (N, 256).
            tb_writer: Optional TensorBoard writer.

        Returns:
            Tuple of (adversarial_embeddings, noise, iterations_per_sample).
        """
        n_samples = len(embeddings)
        batch_size = self.config.batch_size
        cfg = self.config

        logger.info(
            "Running PGD attack: epsilon=%.4f, alpha=%.4f, iterations=%d, samples=%d",
            cfg.epsilon, cfg.alpha, cfg.iterations, n_samples,
        )

        all_adversarial = np.zeros_like(embeddings)
        all_noise = np.zeros_like(embeddings)
        all_iterations = np.zeros(n_samples, dtype=np.int32)

        n_batches = (n_samples + batch_size - 1) // batch_size

        for i in tqdm(range(n_batches), desc="PGD Attack", unit="batch"):
            start = i * batch_size
            end = min(start + batch_size, n_samples)
            batch = embeddings[start:end]

            adv, noise, iters = self._attack_batch(
                batch, tb_writer=tb_writer, global_step=i * cfg.iterations
            )

            all_adversarial[start:end] = adv
            all_noise[start:end] = noise
            all_iterations[start:end] = iters

        logger.info("PGD attack complete")
        return all_adversarial, all_noise, all_iterations

    def attack_single(self, embedding: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        """Run PGD attack on a single embedding.

        Args:
            embedding: Single embedding of shape (256,).

        Returns:
            Tuple of (adversarial_embedding, noise, iterations_used).
        """
        emb = embedding.reshape(1, -1)
        adv, noise, iters = self.attack(emb)
        return adv.flatten(), noise.flatten(), int(iters[0])
