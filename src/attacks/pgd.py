"""PGD (Projected Gradient Descent) attack on images.

Iterative gradient-based attack that applies multiple small perturbation
steps, projecting back onto the epsilon-ball after each step.

Reference:
    Madry et al., "Towards Deep Learning Models Resistant to Adversarial
    Attacks", 2018
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
        epsilon: L-inf perturbation budget in [0, 1] pixel scale.
        alpha: Step size per iteration.
        iterations: Number of attack iterations.
        random_start: Start from a random point within the epsilon-ball.
        targeted: Perform targeted attack.
        target_label: Target label (0.0 = SFW).
        batch_size: Batch size for processing.
        early_stop: Stop early if all samples are misclassified.
        early_stop_threshold: Probability threshold for early stopping.
    """

    epsilon: float = 8 / 255
    alpha: float = 2 / 255
    iterations: int = 20
    random_start: bool = True
    targeted: bool = True
    target_label: float = 0.0
    batch_size: int = 8
    early_stop: bool = True
    early_stop_threshold: float = 0.3


class PGDAttack:
    """PGD attack on images through the end-to-end pipeline.

    Args:
        pipeline: EndToEndModel instance.
        config: PGD configuration.
    """

    def __init__(
        self,
        pipeline: tf.keras.Model,
        config: PGDConfig | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.config = config or PGDConfig()
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    def _project(
        self, adversarial: tf.Tensor, original: tf.Tensor, epsilon: float
    ) -> tf.Tensor:
        """Project onto the L-inf epsilon-ball centred at original, clipped to [0,1]."""
        perturbation = tf.clip_by_value(adversarial - original, -epsilon, epsilon)
        return tf.clip_by_value(original + perturbation, 0.0, 1.0)

    def _attack_batch(
        self,
        images: np.ndarray,
        tb_writer: tf.summary.SummaryWriter | None = None,
        global_step: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run PGD on a single batch of images.

        Returns:
            (adversarial, noise, iterations_used)
        """
        batch_size = len(images)
        cfg = self.config
        original = tf.constant(images, dtype=tf.float32)
        target_labels = tf.fill([batch_size, 1], cfg.target_label)

        # Random start
        if cfg.random_start:
            noise = tf.random.uniform(original.shape, -cfg.epsilon, cfg.epsilon)
            adv = tf.clip_by_value(original + noise, 0.0, 1.0)
        else:
            adv = tf.identity(original)

        iterations_used = np.full(batch_size, cfg.iterations, dtype=np.int32)
        best_adv = adv.numpy()
        best_probs = np.ones(batch_size, dtype=np.float32)

        for step in range(cfg.iterations):
            adv_var = tf.Variable(adv)

            with tf.GradientTape() as tape:
                preds = self.pipeline(adv_var, training=False)
                loss = self.loss_fn(target_labels, preds)

            grad = tape.gradient(loss, adv_var)

            if cfg.targeted:
                adv = adv_var - cfg.alpha * tf.sign(grad)
            else:
                adv = adv_var + cfg.alpha * tf.sign(grad)

            adv = self._project(adv, original, cfg.epsilon)

            current_probs = self.pipeline(adv, training=False).numpy().flatten()
            improved = current_probs < best_probs
            best_probs = np.where(improved, current_probs, best_probs)
            best_adv = np.where(
                improved[:, np.newaxis, np.newaxis, np.newaxis],
                adv.numpy(),
                best_adv,
            )

            success_mask = (current_probs < cfg.early_stop_threshold) & (
                iterations_used == cfg.iterations
            )
            iterations_used[success_mask] = step + 1

            if tb_writer is not None:
                with tb_writer.as_default():
                    s = global_step + step
                    tf.summary.scalar(
                        "pgd/avg_prob", float(current_probs.mean()), step=s
                    )
                    tf.summary.scalar("pgd/loss", float(loss), step=s)
                    tf.summary.scalar(
                        "pgd/success_rate",
                        float((current_probs < 0.5).mean()),
                        step=s,
                    )

            if cfg.early_stop and np.all(current_probs < cfg.early_stop_threshold):
                logger.debug("Early stop at iteration %d", step + 1)
                break

        noise = best_adv - images
        return best_adv, noise, iterations_used

    def attack(
        self,
        images: np.ndarray,
        tb_writer: tf.summary.SummaryWriter | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run PGD attack on all images.

        Args:
            images: (N, H, W, 3) float32 in [0, 1].
            tb_writer: Optional TensorBoard writer.

        Returns:
            (adversarial_images, noise, iterations_per_sample)
        """
        n = len(images)
        cfg = self.config

        logger.info(
            "Running PGD attack: eps=%.4f (%.1f/255), alpha=%.4f, iters=%d, samples=%d",
            cfg.epsilon, cfg.epsilon * 255, cfg.alpha, cfg.iterations, n,
        )

        all_adv = np.zeros_like(images)
        all_noise = np.zeros_like(images)
        all_iters = np.zeros(n, dtype=np.int32)

        n_batches = (n + cfg.batch_size - 1) // cfg.batch_size

        for i in tqdm(range(n_batches), desc="PGD Attack", unit="batch"):
            start = i * cfg.batch_size
            end = min(start + cfg.batch_size, n)
            adv, noise, iters = self._attack_batch(
                images[start:end],
                tb_writer=tb_writer,
                global_step=i * cfg.iterations,
            )
            all_adv[start:end] = adv
            all_noise[start:end] = noise
            all_iters[start:end] = iters

        logger.info("PGD attack complete")
        return all_adv, all_noise, all_iters
