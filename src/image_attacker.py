"""High-level image attacker that orchestrates adversarial attacks.

Provides a unified interface for running different attack methods
on images and collecting results.
"""

import logging
import os
from datetime import datetime
from typing import Any

import numpy as np
import tensorflow as tf

from src.attacks.cw import CWAttack, CWConfig
from src.attacks.deepfool import DeepFoolAttack, DeepFoolConfig
from src.attacks.fgsm import FGSMAttack, FGSMConfig
from src.attacks.pgd import PGDAttack, PGDConfig
from src.utils import (
    compute_metrics,
    save_adversarial_images,
    save_noise_images,
    save_results,
    setup_tensorboard,
)

logger = logging.getLogger(__name__)


class ImageAttacker:
    """Orchestrates adversarial attacks on images.

    Args:
        pipeline: EndToEndModel (image -> NSFW probability).
        config: Optional configuration dictionary.
    """

    ATTACK_METHODS = {"fgsm", "pgd", "cw", "deepfool"}

    def __init__(
        self,
        pipeline: tf.keras.Model,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.config = config or {}
        self.tb_writer: tf.summary.SummaryWriter | None = None

        tb_cfg = self.config.get("tensorboard", {})
        if tb_cfg.get("enabled", False):
            self.tb_writer = setup_tensorboard(tb_cfg.get("log_dir", "logs"))

    def run_attack(
        self,
        method: str,
        images: np.ndarray,
        output_dir: str | None = None,
        filenames: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run the specified attack on images.

        Args:
            method: Attack name ('fgsm', 'pgd', 'cw', 'deepfool').
            images: (N, H, W, 3) float32 in [0, 1].
            output_dir: Directory to save results and images.
            filenames: Original filenames for naming adversarial images.
            **kwargs: Attack-specific parameters.

        Returns:
            Results dictionary.
        """
        if method not in self.ATTACK_METHODS:
            raise ValueError(f"Unknown method: {method}")

        logger.info("Starting %s attack on %d images", method.upper(), len(images))

        # Original predictions
        original_probs = self.pipeline.predict_images(images)
        logger.info(
            "Original predictions -- mean=%.4f, NSFW(>=0.5)=%d/%d",
            original_probs.mean(),
            (original_probs >= 0.5).sum(),
            len(original_probs),
        )

        # Run attack
        attack_fn = getattr(self, f"_run_{method}")
        adversarial, noise, iterations = attack_fn(images, **kwargs)

        # Adversarial predictions
        adversarial_probs = self.pipeline.predict_images(adversarial)

        # Metrics
        thresholds = self.config.get("evaluation", {}).get(
            "thresholds", [0.3, 0.4, 0.5]
        )
        metrics = compute_metrics(
            original_probs, adversarial_probs, noise, thresholds
        )
        if iterations is not None:
            metrics["avg_iterations"] = float(iterations.mean())

        results: dict[str, Any] = {
            "attack_method": method.upper(),
            "parameters": kwargs,
            "timestamp": datetime.now().isoformat(),
            "results": metrics,
        }

        logger.info(
            "Results: %s",
            {
                k: f"{v:.4f}" if isinstance(v, float) else v
                for k, v in metrics.items()
            },
        )

        # Save
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            save_results(results, output_dir, prefix=f"{method}_results")

            if self.config.get("output", {}).get("save_adversarial_images", True):
                save_adversarial_images(
                    adversarial, os.path.join(output_dir, "images"),
                    filenames=filenames, prefix=method,
                )
            if self.config.get("output", {}).get("save_noise_images", False):
                save_noise_images(
                    noise, os.path.join(output_dir, "noise"), prefix=method,
                )

        return results

    # ── per-method dispatchers ─────────────────────────────────────

    def _run_fgsm(
        self, images: np.ndarray, **kwargs: Any
    ) -> tuple[np.ndarray, np.ndarray, None]:
        cfg = FGSMConfig(
            **{k: v for k, v in kwargs.items() if k in FGSMConfig.__dataclass_fields__}
        )
        attack = FGSMAttack(self.pipeline, cfg)
        adv, noise = attack.attack(images)
        return adv, noise, None

    def _run_pgd(
        self, images: np.ndarray, **kwargs: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg = PGDConfig(
            **{k: v for k, v in kwargs.items() if k in PGDConfig.__dataclass_fields__}
        )
        attack = PGDAttack(self.pipeline, cfg)
        adv, noise, iters = attack.attack(images, tb_writer=self.tb_writer)
        return adv, noise, iters

    def _run_cw(
        self, images: np.ndarray, **kwargs: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg = CWConfig(
            **{k: v for k, v in kwargs.items() if k in CWConfig.__dataclass_fields__}
        )
        attack = CWAttack(self.pipeline, cfg)
        adv, noise, iters = attack.attack(images, tb_writer=self.tb_writer)
        return adv, noise, iters

    def _run_deepfool(
        self, images: np.ndarray, **kwargs: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg = DeepFoolConfig(
            **{
                k: v
                for k, v in kwargs.items()
                if k in DeepFoolConfig.__dataclass_fields__
            }
        )
        attack = DeepFoolAttack(self.pipeline, cfg)
        adv, noise, iters = attack.attack(images, tb_writer=self.tb_writer)
        return adv, noise, iters

    def compare_attacks(
        self,
        images: np.ndarray,
        methods: list[str] | None = None,
        output_dir: str = "experiments/attack_results",
        filenames: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, dict[str, Any]]:
        """Run multiple attacks and compare."""
        if methods is None:
            methods = list(self.ATTACK_METHODS)

        all_results: dict[str, dict[str, Any]] = {}
        for method in methods:
            method_dir = os.path.join(output_dir, method)
            try:
                r = self.run_attack(
                    method, images, output_dir=method_dir,
                    filenames=filenames, **kwargs,
                )
                all_results[method] = r
            except Exception:
                logger.exception("Failed: %s", method)
                all_results[method] = {"error": True}
        return all_results
