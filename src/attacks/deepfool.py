"""DeepFool attack on images.

Iteratively finds the minimal perturbation needed to cross the
decision boundary.

Reference:
    Moosavi-Dezfooli et al., "DeepFool: a simple and accurate method to
    fool deep neural networks", 2016
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
        max_iterations: Maximum iterations per sample.
        overshoot: Overshoot factor to ensure crossing the boundary.
        target_threshold: Decision boundary threshold.
    """

    max_iterations: int = 100
    overshoot: float = 0.02
    target_threshold: float = 0.5


class DeepFoolAttack:
    """DeepFool attack on images through the end-to-end pipeline.

    Args:
        pipeline: EndToEndModel instance.
        config: DeepFool configuration.
    """

    def __init__(
        self,
        pipeline: tf.keras.Model,
        config: DeepFoolConfig | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.config = config or DeepFoolConfig()

    def _attack_single(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Run DeepFool on a single image.

        Args:
            image: Single image (H, W, 3) float32 in [0, 1].

        Returns:
            (adversarial_image, noise, iterations_used)
        """
        cfg = self.config
        x = tf.Variable(image[np.newaxis].astype(np.float32))  # (1,H,W,3)
        original = image.copy()

        for i in range(cfg.max_iterations):
            with tf.GradientTape() as tape:
                pred = self.pipeline(x, training=False)

            pred_val = float(pred.numpy().flatten()[0])
            if pred_val < cfg.target_threshold:
                noise = x.numpy()[0] - original
                return x.numpy()[0], noise, i

            gradient = tape.gradient(pred, x)
            grad = gradient.numpy().flatten()
            grad_norm_sq = np.dot(grad, grad)

            if grad_norm_sq < 1e-20:
                logger.debug("Gradient vanished at iteration %d", i)
                break

            f_val = pred_val - cfg.target_threshold
            # Minimal perturbation along the gradient direction
            r = (abs(f_val) / grad_norm_sq) * grad
            r = r.reshape(x.shape)

            x.assign(x - (1.0 + cfg.overshoot) * r)
            # Clip to valid pixel range
            x.assign(tf.clip_by_value(x, 0.0, 1.0))

        noise = x.numpy()[0] - original
        return x.numpy()[0], noise, cfg.max_iterations

    def attack(
        self,
        images: np.ndarray,
        tb_writer: tf.summary.SummaryWriter | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run DeepFool attack on all images.

        Args:
            images: (N, H, W, 3) float32 in [0, 1].
            tb_writer: Optional TensorBoard writer.

        Returns:
            (adversarial_images, noise, iterations_per_sample)
        """
        n = len(images)

        logger.info(
            "Running DeepFool attack: max_iter=%d, overshoot=%.3f, samples=%d",
            self.config.max_iterations, self.config.overshoot, n,
        )

        all_adv = np.zeros_like(images)
        all_noise = np.zeros_like(images)
        all_iters = np.zeros(n, dtype=np.int32)

        for i in tqdm(range(n), desc="DeepFool Attack", unit="sample"):
            adv, noise, iters = self._attack_single(images[i])
            all_adv[i] = adv
            all_noise[i] = noise
            all_iters[i] = iters

            if tb_writer is not None and i % 50 == 0:
                pred_val = float(
                    self.pipeline(
                        tf.constant(adv[np.newaxis], dtype=tf.float32),
                        training=False,
                    ).numpy().flatten()[0]
                )
                with tb_writer.as_default():
                    tf.summary.scalar("deepfool/pred", pred_val, step=i)
                    tf.summary.scalar(
                        "deepfool/l2_norm",
                        float(np.linalg.norm(noise)),
                        step=i,
                    )

        logger.info("DeepFool attack complete")
        return all_adv, all_noise, all_iters
