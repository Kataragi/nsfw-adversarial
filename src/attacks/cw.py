"""Carlini & Wagner (C&W) attack implementation.

C&W is an optimization-based adversarial attack that minimizes
the perturbation norm while ensuring misclassification.

Reference:
    Carlini & Wagner, "Towards Evaluating the Robustness of Neural Networks", 2017
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
        kappa: Confidence margin for misclassification.
        iterations: Number of optimization iterations.
        learning_rate: Adam optimizer learning rate.
        binary_search_steps: Number of binary search steps for c.
        batch_size: Batch size for processing.
        abort_early: Whether to abort if loss stops decreasing.
        target_label: Target label for the attack.
    """

    c: float = 1.0
    kappa: float = 0.0
    iterations: int = 500
    learning_rate: float = 0.01
    binary_search_steps: int = 9
    batch_size: int = 64
    abort_early: bool = True
    target_label: float = 0.0


class CWAttack:
    """Carlini & Wagner L2 attack.

    Finds minimal L2 perturbation that causes misclassification
    using optimization with binary search over the trade-off constant.

    Args:
        model: Target classifier model.
        config: C&W configuration.
    """

    def __init__(self, model: tf.keras.Model, config: CWConfig | None = None) -> None:
        self.model = model
        self.config = config or CWConfig()

    def _cw_loss(
        self,
        predictions: tf.Tensor,
        target_label: float,
        kappa: float,
    ) -> tf.Tensor:
        """Compute C&W loss for binary classification.

        For targeted attack toward SFW (label=0), we want predictions
        to be low (close to 0).

        Args:
            predictions: Model predictions (NSFW probability).
            target_label: Target label value.
            kappa: Confidence margin.

        Returns:
            C&W loss tensor.
        """
        # For binary sigmoid output targeting label 0:
        # We want to maximize (threshold - prediction), i.e., minimize prediction
        # loss = max(prediction - threshold + kappa, 0)
        return tf.maximum(predictions - target_label - kappa, 0.0)

    def _attack_batch(
        self,
        embeddings: np.ndarray,
        tb_writer: tf.summary.SummaryWriter | None = None,
        global_step: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run C&W attack on a batch.

        Args:
            embeddings: Batch of embeddings.
            tb_writer: Optional TensorBoard writer.
            global_step: Step counter for TensorBoard.

        Returns:
            Tuple of (adversarial, noise, iterations_used).
        """
        batch_size = len(embeddings)
        cfg = self.config
        original = tf.constant(embeddings, dtype=tf.float32)

        best_adv = np.copy(embeddings)
        best_l2 = np.full(batch_size, 1e10, dtype=np.float32)
        iterations_used = np.full(batch_size, cfg.iterations, dtype=np.int32)

        # Binary search over c
        c_lower = np.zeros(batch_size, dtype=np.float32)
        c_upper = np.full(batch_size, cfg.c * 10, dtype=np.float32)
        c_current = np.full(batch_size, cfg.c, dtype=np.float32)

        for search_step in range(cfg.binary_search_steps):
            # Initialize perturbation variable (in tanh space for unconstrained opt)
            w = tf.Variable(tf.zeros_like(original))
            optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)

            c_tensor = tf.constant(c_current.reshape(-1, 1), dtype=tf.float32)

            prev_loss = 1e10

            for step in range(cfg.iterations):
                with tf.GradientTape() as tape:
                    # Perturbation via tanh transformation
                    perturbation = tf.tanh(w) * 0.5
                    adv = original + perturbation

                    predictions = self.model(adv, training=False)

                    # L2 distance
                    l2_dist = tf.reduce_sum(tf.square(perturbation), axis=1, keepdims=True)

                    # Classification loss
                    cls_loss = self._cw_loss(predictions, cfg.target_label, cfg.kappa)

                    # Total loss
                    total_loss = l2_dist + c_tensor * cls_loss
                    loss = tf.reduce_sum(total_loss)

                optimizer.apply_gradients([(tape.gradient(loss, w), w)])

                # Track progress
                current_preds = predictions.numpy().flatten()
                current_l2 = np.sqrt(l2_dist.numpy().flatten())

                # Update best results
                success = current_preds < 0.5
                improved = success & (current_l2 < best_l2)
                for j in range(batch_size):
                    if improved[j]:
                        best_adv[j] = adv[j].numpy()
                        best_l2[j] = current_l2[j]
                        iterations_used[j] = (
                            search_step * cfg.iterations + step + 1
                        )

                # TensorBoard logging
                if tb_writer is not None and step % 50 == 0:
                    with tb_writer.as_default():
                        s = global_step + search_step * cfg.iterations + step
                        tf.summary.scalar("cw/loss", float(loss), step=s)
                        tf.summary.scalar(
                            "cw/avg_pred", float(np.mean(current_preds)), step=s
                        )
                        tf.summary.scalar(
                            "cw/avg_l2", float(np.mean(current_l2)), step=s
                        )

                # Abort early if loss plateaus
                if cfg.abort_early and step % 100 == 0:
                    current_loss_val = float(loss)
                    if current_loss_val > prev_loss * 0.9999:
                        logger.debug(
                            "Early abort at search=%d, step=%d",
                            search_step, step,
                        )
                        break
                    prev_loss = current_loss_val

            # Update c via binary search
            final_preds = self.model(
                tf.constant(best_adv, dtype=tf.float32), training=False
            ).numpy().flatten()

            for j in range(batch_size):
                if final_preds[j] < 0.5:  # Attack succeeded
                    c_upper[j] = min(c_upper[j], c_current[j])
                    c_current[j] = (c_lower[j] + c_upper[j]) / 2
                else:  # Attack failed
                    c_lower[j] = max(c_lower[j], c_current[j])
                    c_current[j] = (c_lower[j] + c_upper[j]) / 2

        noise = best_adv - embeddings
        return best_adv, noise, iterations_used

    def attack(
        self,
        embeddings: np.ndarray,
        tb_writer: tf.summary.SummaryWriter | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run C&W attack on all embeddings.

        Args:
            embeddings: Original embeddings of shape (N, 256).
            tb_writer: Optional TensorBoard writer.

        Returns:
            Tuple of (adversarial_embeddings, noise, iterations_per_sample).
        """
        n_samples = len(embeddings)
        batch_size = self.config.batch_size

        logger.info(
            "Running C&W attack: c=%.4f, kappa=%.1f, iterations=%d, samples=%d",
            self.config.c, self.config.kappa, self.config.iterations, n_samples,
        )

        all_adversarial = np.zeros_like(embeddings)
        all_noise = np.zeros_like(embeddings)
        all_iterations = np.zeros(n_samples, dtype=np.int32)

        n_batches = (n_samples + batch_size - 1) // batch_size

        for i in tqdm(range(n_batches), desc="C&W Attack", unit="batch"):
            start = i * batch_size
            end = min(start + batch_size, n_samples)
            batch = embeddings[start:end]

            adv, noise, iters = self._attack_batch(
                batch, tb_writer=tb_writer,
                global_step=i * self.config.binary_search_steps * self.config.iterations,
            )

            all_adversarial[start:end] = adv
            all_noise[start:end] = noise
            all_iterations[start:end] = iters

        logger.info("C&W attack complete")
        return all_adversarial, all_noise, all_iterations
