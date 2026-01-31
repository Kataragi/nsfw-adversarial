"""High-level embedding attacker that orchestrates adversarial attacks.

Provides a unified interface for running different attack methods
on embedding vectors and collecting results.
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
    predict_batch,
    save_adversarial_embeddings,
    save_results,
    setup_tensorboard,
)

logger = logging.getLogger(__name__)


class EmbeddingAttacker:
    """Orchestrates adversarial attacks on embedding vectors.

    Supports FGSM, PGD, C&W, and DeepFool attacks with unified
    result collection and evaluation.

    Args:
        model: Target classifier model.
        config: Optional configuration dictionary.
    """

    ATTACK_METHODS = {"fgsm", "pgd", "cw", "deepfool"}

    def __init__(
        self,
        model: tf.keras.Model,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.config = config or {}
        self.tb_writer: tf.summary.SummaryWriter | None = None

        tb_config = self.config.get("tensorboard", {})
        if tb_config.get("enabled", False):
            self.tb_writer = setup_tensorboard(
                tb_config.get("log_dir", "logs")
            )

    def run_attack(
        self,
        method: str,
        embeddings: np.ndarray,
        output_dir: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run a specified attack method on embeddings.

        Args:
            method: Attack method name ('fgsm', 'pgd', 'cw', 'deepfool').
            embeddings: Input embeddings of shape (N, 256).
            output_dir: Optional directory to save results.
            **kwargs: Attack-specific parameters.

        Returns:
            Results dictionary with metrics and per-sample data.

        Raises:
            ValueError: If the attack method is unknown.
        """
        if method not in self.ATTACK_METHODS:
            raise ValueError(
                f"Unknown attack method: {method}. "
                f"Supported: {self.ATTACK_METHODS}"
            )

        logger.info("Starting %s attack on %d embeddings", method.upper(), len(embeddings))

        # Get original predictions
        original_probs = predict_batch(self.model, embeddings)
        logger.info(
            "Original predictions - mean: %.4f, NSFW (>=0.5): %d/%d",
            original_probs.mean(),
            (original_probs >= 0.5).sum(),
            len(original_probs),
        )

        # Run attack
        attack_fn = getattr(self, f"_run_{method}")
        adversarial, noise, iterations = attack_fn(embeddings, **kwargs)

        # Get adversarial predictions
        adversarial_probs = predict_batch(self.model, adversarial)

        # Compute metrics
        thresholds = self.config.get("evaluation", {}).get(
            "thresholds", [0.3, 0.4, 0.5]
        )
        metrics = compute_metrics(original_probs, adversarial_probs, noise, thresholds)

        if iterations is not None:
            metrics["avg_iterations"] = float(iterations.mean())

        # Build results
        results: dict[str, Any] = {
            "attack_method": method.upper(),
            "parameters": kwargs,
            "timestamp": datetime.now().isoformat(),
            "results": metrics,
        }

        logger.info("Attack results: %s", {
            k: f"{v:.4f}" if isinstance(v, float) else v
            for k, v in metrics.items()
        })

        # Save results
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            save_results(results, output_dir, prefix=f"{method}_results")

            if self.config.get("output", {}).get("save_adversarial_embeddings", True):
                save_adversarial_embeddings(adversarial, noise, output_dir, prefix=method)

        return results

    def _run_fgsm(
        self, embeddings: np.ndarray, **kwargs: Any
    ) -> tuple[np.ndarray, np.ndarray, None]:
        """Run FGSM attack.

        Args:
            embeddings: Input embeddings.
            **kwargs: FGSM parameters.

        Returns:
            Tuple of (adversarial, noise, None).
        """
        config = FGSMConfig(**{
            k: v for k, v in kwargs.items()
            if k in FGSMConfig.__dataclass_fields__
        })
        attack = FGSMAttack(self.model, config)
        adversarial, noise = attack.attack(embeddings)
        return adversarial, noise, None

    def _run_pgd(
        self, embeddings: np.ndarray, **kwargs: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run PGD attack.

        Args:
            embeddings: Input embeddings.
            **kwargs: PGD parameters.

        Returns:
            Tuple of (adversarial, noise, iterations).
        """
        config = PGDConfig(**{
            k: v for k, v in kwargs.items()
            if k in PGDConfig.__dataclass_fields__
        })
        attack = PGDAttack(self.model, config)
        adversarial, noise, iterations = attack.attack(
            embeddings, tb_writer=self.tb_writer
        )
        return adversarial, noise, iterations

    def _run_cw(
        self, embeddings: np.ndarray, **kwargs: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run C&W attack.

        Args:
            embeddings: Input embeddings.
            **kwargs: C&W parameters.

        Returns:
            Tuple of (adversarial, noise, iterations).
        """
        config = CWConfig(**{
            k: v for k, v in kwargs.items()
            if k in CWConfig.__dataclass_fields__
        })
        attack = CWAttack(self.model, config)
        adversarial, noise, iterations = attack.attack(
            embeddings, tb_writer=self.tb_writer
        )
        return adversarial, noise, iterations

    def _run_deepfool(
        self, embeddings: np.ndarray, **kwargs: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run DeepFool attack.

        Args:
            embeddings: Input embeddings.
            **kwargs: DeepFool parameters.

        Returns:
            Tuple of (adversarial, noise, iterations).
        """
        config = DeepFoolConfig(**{
            k: v for k, v in kwargs.items()
            if k in DeepFoolConfig.__dataclass_fields__
        })
        attack = DeepFoolAttack(self.model, config)
        adversarial, noise, iterations = attack.attack(
            embeddings, tb_writer=self.tb_writer
        )
        return adversarial, noise, iterations

    def compare_attacks(
        self,
        embeddings: np.ndarray,
        methods: list[str] | None = None,
        output_dir: str = "experiments/attack_results",
        **kwargs: Any,
    ) -> dict[str, dict[str, Any]]:
        """Run multiple attacks and compare results.

        Args:
            embeddings: Input embeddings.
            methods: List of attack methods to compare.
            output_dir: Base output directory.
            **kwargs: Parameters passed to each attack.

        Returns:
            Dictionary mapping method names to their results.
        """
        if methods is None:
            methods = list(self.ATTACK_METHODS)

        all_results: dict[str, dict[str, Any]] = {}

        for method in methods:
            method_dir = os.path.join(output_dir, method)
            try:
                results = self.run_attack(
                    method, embeddings, output_dir=method_dir, **kwargs
                )
                all_results[method] = results
            except Exception:
                logger.exception("Failed to run %s attack", method)
                all_results[method] = {"error": True}

        return all_results
