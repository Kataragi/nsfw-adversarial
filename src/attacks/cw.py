"""Carlini & Wagner (C&W) L2 attack on images.

Optimization-based attack that minimises the L2 perturbation while
ensuring the classifier prediction drops below the decision boundary.

Reference:
    Carlini & Wagner, "Towards Evaluating the Robustness of Neural
    Networks", 2017
"""

import logging
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class CWConfig:
    """Configuration for C&W attack.

    Attributes:
        c: Trade-off constant between perturbation and misclassification.
        kappa: Confidence margin.
        iterations: Optimisation iterations per binary-search step.
        learning_rate: Adam learning rate.
        binary_search_steps: Number of binary search steps for c.
        batch_size: Batch size for processing.
        abort_early: Abort if loss plateaus.
        target_label: Target label (0.0 = SFW).
    """

    c: float = 1.0
    kappa: float = 0.0
    iterations: int = 500
    learning_rate: float = 0.01
    binary_search_steps: int = 9
    batch_size: int = 4
    abort_early: bool = True
    target_label: float = 0.0


class CWAttack:
    """C&W L2 attack on images through the end-to-end pipeline.

    Args:
        pipeline: EndToEndModel instance.
        config: C&W configuration.
    """

    def __init__(
        self, pipeline: tf.keras.Model, config: CWConfig | None = None
    ) -> None:
        self.pipeline = pipeline
        self.config = config or CWConfig()

    @staticmethod
    def _cw_loss(
        predictions: tf.Tensor, target_label: float, kappa: float
    ) -> tf.Tensor:
        """C&W classification loss: max(pred - target - kappa, 0)."""
        return tf.maximum(predictions - target_label - kappa, 0.0)

    def _attack_batch(
        self,
        images: np.ndarray,
        tb_writer: tf.summary.SummaryWriter | None = None,
        global_step: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run C&W on one batch.

        Returns:
            (adversarial, noise, iterations_used)
        """
        batch_size = len(images)
        cfg = self.config
        original = tf.constant(images, dtype=tf.float32)

        best_adv = np.copy(images)
        best_l2 = np.full(batch_size, 1e10, dtype=np.float32)
        iterations_used = np.full(batch_size, cfg.iterations, dtype=np.int32)

        c_lower = np.zeros(batch_size, dtype=np.float32)
        c_upper = np.full(batch_size, cfg.c * 10, dtype=np.float32)
        c_current = np.full(batch_size, cfg.c, dtype=np.float32)

        for search_step in range(cfg.binary_search_steps):
            # w parameterises unconstrained perturbation (tanh space)
            w = tf.Variable(tf.zeros_like(original))
            optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)
            c_tensor = tf.constant(
                c_current.reshape(-1, 1, 1, 1), dtype=tf.float32
            )

            prev_loss = 1e10

            for step in range(cfg.iterations):
                with tf.GradientTape() as tape:
                    perturbation = tf.tanh(w) * 0.5
                    adv = tf.clip_by_value(original + perturbation, 0.0, 1.0)

                    predictions = self.pipeline(adv, training=False)

                    # L2 over spatial dims
                    l2_dist = tf.reduce_sum(
                        tf.square(perturbation),
                        axis=[1, 2, 3],
                        keepdims=True,
                    )

                    cls_loss = self._cw_loss(
                        predictions, cfg.target_label, cfg.kappa
                    )
                    # cls_loss is (N,1); broadcast c
                    c_broad = tf.reshape(
                        tf.constant(c_current, dtype=tf.float32), [-1, 1]
                    )
                    total_loss = tf.squeeze(l2_dist, axis=[2, 3]) + c_broad * cls_loss
                    loss = tf.reduce_sum(total_loss)

                grad = tape.gradient(loss, w)
                optimizer.apply_gradients([(grad, w)])

                current_preds = predictions.numpy().flatten()
                current_l2 = np.sqrt(
                    l2_dist.numpy().reshape(batch_size, -1).sum(axis=1)
                )

                success = current_preds < 0.5
                improved = success & (current_l2 < best_l2)
                for j in range(batch_size):
                    if improved[j]:
                        best_adv[j] = adv[j].numpy()
                        best_l2[j] = current_l2[j]
                        iterations_used[j] = (
                            search_step * cfg.iterations + step + 1
                        )

                if tb_writer is not None and step % 50 == 0:
                    with tb_writer.as_default():
                        s = global_step + search_step * cfg.iterations + step
                        tf.summary.scalar("cw/loss", float(loss), step=s)
                        tf.summary.scalar(
                            "cw/avg_pred",
                            float(current_preds.mean()),
                            step=s,
                        )

                if cfg.abort_early and step % 100 == 0:
                    current_loss_val = float(loss)
                    if current_loss_val > prev_loss * 0.9999:
                        break
                    prev_loss = current_loss_val

            # Binary search update
            final_preds = (
                self.pipeline(
                    tf.constant(best_adv, dtype=tf.float32), training=False
                )
                .numpy()
                .flatten()
            )
            for j in range(batch_size):
                if final_preds[j] < 0.5:
                    c_upper[j] = min(c_upper[j], c_current[j])
                else:
                    c_lower[j] = max(c_lower[j], c_current[j])
                c_current[j] = (c_lower[j] + c_upper[j]) / 2

        noise = best_adv - images
        return best_adv, noise, iterations_used

    def attack(
        self,
        images: np.ndarray,
        tb_writer: tf.summary.SummaryWriter | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run C&W attack on all images.

        Args:
            images: (N, H, W, 3) float32 in [0, 1].
            tb_writer: Optional TensorBoard writer.

        Returns:
            (adversarial_images, noise, iterations_per_sample)
        """
        n = len(images)
        cfg = self.config

        logger.info(
            "Running C&W attack: c=%.4f, kappa=%.1f, iters=%d, samples=%d",
            cfg.c, cfg.kappa, cfg.iterations, n,
        )

        all_adv = np.zeros_like(images)
        all_noise = np.zeros_like(images)
        all_iters = np.zeros(n, dtype=np.int32)

        n_batches = (n + cfg.batch_size - 1) // cfg.batch_size

        for i in tqdm(range(n_batches), desc="C&W Attack", unit="batch"):
            start = i * cfg.batch_size
            end = min(start + cfg.batch_size, n)
            adv, noise, iters = self._attack_batch(
                images[start:end],
                tb_writer=tb_writer,
                global_step=i * cfg.binary_search_steps * cfg.iterations,
            )
            all_adv[start:end] = adv
            all_noise[start:end] = noise
            all_iters[start:end] = iters

        logger.info("C&W attack complete")
        return all_adv, all_noise, all_iters
