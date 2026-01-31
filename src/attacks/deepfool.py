"""DeepFool attack implementation.

DeepFool iteratively finds the minimal perturbation needed to cross
the decision boundary.

Reference:
    Moosavi-Dezfooli et al., "DeepFool: a simple and accurate method to fool
    deep neural networks", 2016
"""

import logging
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class DeepFoolConfig:
    """Configuration for DeepFool attack.

    Attributes:
        max_iterations: Maximum number of iterations.
        overshoot: Overshoot parameter to ensure crossing the boundary.
        batch_size: Batch size for processing.
        target_threshold: Target prediction threshold for success.
    """

    max_iterations: int = 100
    overshoot: float = 0.02
    batch_size: int = 128
    target_threshold: float = 0.5


class DeepFoolAttack:
    """DeepFool attack for binary classification.

    Finds the minimal perturbation that moves the prediction across
    the decision boundary (0.5) for a sigmoid-output binary classifier.

    Args:
        model: Target classifier model.
        config: DeepFool configuration.
    """

    def __init__(
        self, model: tf.keras.Model, config: DeepFoolConfig | None = None
    ) -> None:
        self.model = model
        self.config = config or DeepFoolConfig()

    def _attack_single(self, embedding: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        """Run DeepFool on a single embedding.

        For binary classification with sigmoid output, the decision
        boundary is at f(x) = 0.5. We linearize and find the minimal
        perturbation to cross this boundary.

        Args:
            embedding: Single embedding of shape (256,).

        Returns:
            Tuple of (adversarial_embedding, noise, iterations_used).
        """
        cfg = self.config
        x = tf.Variable(embedding.reshape(1, -1).astype(np.float32))
        original = embedding.copy()

        for i in range(cfg.max_iterations):
            with tf.GradientTape() as tape:
                pred = self.model(x, training=False)

            # Check if already misclassified
            pred_val = float(pred.numpy().flatten()[0])
            if pred_val < cfg.target_threshold:
                noise = x.numpy().flatten() - original
                return x.numpy().flatten(), noise, i

            gradient = tape.gradient(pred, x)
            grad = gradient.numpy().flatten()
            grad_norm_sq = np.dot(grad, grad)

            if grad_norm_sq < 1e-20:
                logger.debug("Gradient vanished at iteration %d", i)
                break

            # Distance to boundary: |f(x) - 0.5| / ||grad||^2
            f_val = pred_val - cfg.target_threshold
            perturbation = (abs(f_val) / grad_norm_sq) * grad

            # Move toward the boundary (decrease prediction)
            x.assign(x - (1 + cfg.overshoot) * perturbation.reshape(1, -1))

        noise = x.numpy().flatten() - original
        return x.numpy().flatten(), noise, cfg.max_iterations

    def attack(
        self,
        embeddings: np.ndarray,
        tb_writer: tf.summary.SummaryWriter | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run DeepFool attack on all embeddings.

        Args:
            embeddings: Original embeddings of shape (N, 256).
            tb_writer: Optional TensorBoard writer.

        Returns:
            Tuple of (adversarial_embeddings, noise, iterations_per_sample).
        """
        n_samples = len(embeddings)

        logger.info(
            "Running DeepFool attack: max_iter=%d, overshoot=%.3f, samples=%d",
            self.config.max_iterations, self.config.overshoot, n_samples,
        )

        all_adversarial = np.zeros_like(embeddings)
        all_noise = np.zeros_like(embeddings)
        all_iterations = np.zeros(n_samples, dtype=np.int32)

        for i in tqdm(range(n_samples), desc="DeepFool Attack", unit="sample"):
            adv, noise, iters = self._attack_single(embeddings[i])
            all_adversarial[i] = adv
            all_noise[i] = noise
            all_iterations[i] = iters

            # TensorBoard logging
            if tb_writer is not None and i % 100 == 0:
                pred = self.model(
                    tf.constant(adv.reshape(1, -1), dtype=tf.float32),
                    training=False,
                ).numpy().flatten()[0]
                with tb_writer.as_default():
                    tf.summary.scalar("deepfool/pred", pred, step=i)
                    tf.summary.scalar(
                        "deepfool/l2_norm",
                        float(np.linalg.norm(noise)),
                        step=i,
                    )

        logger.info("DeepFool attack complete")
        return all_adversarial, all_noise, all_iterations
