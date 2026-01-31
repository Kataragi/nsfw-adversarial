"""Robustness evaluation script.

Runs multiple attack methods with varying parameters and generates
comprehensive evaluation reports with visualizations.

Usage:
    python src/evaluate_robustness.py \
        --target-model MODEL \
        --embeddings-dir DIR \
        --methods fgsm pgd cw deepfool \
        --epsilon-range 0.01 0.05 0.1
"""

import argparse
import json
import logging
import os
from datetime import datetime
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tqdm import tqdm

from src.embedding_attacker import EmbeddingAttacker
from src.utils import (
    detect_gpu,
    load_config,
    load_embeddings,
    load_target_model,
    predict_batch,
    save_results,
    set_seed,
    setup_logging,
)

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Robustness Evaluation for NSFW Classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--target-model", type=str, required=True,
        help="Path to target classifier model",
    )
    parser.add_argument(
        "--embeddings-dir", type=str, required=True,
        help="Directory containing embeddings",
    )
    parser.add_argument(
        "--methods", type=str, nargs="+",
        default=["fgsm", "pgd", "cw", "deepfool"],
        help="Attack methods to evaluate",
    )
    parser.add_argument(
        "--epsilon-range", type=float, nargs="+",
        default=[0.01, 0.03, 0.05, 0.07, 0.1],
        help="Epsilon values to test",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="experiments/robustness_analysis",
        help="Output directory for results",
    )
    parser.add_argument(
        "--config", type=str, default="config/attack_config.yaml",
        help="Configuration file path",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256, help="Batch size",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Maximum number of samples to evaluate (for quick tests)",
    )
    return parser.parse_args(argv)


def run_epsilon_sweep(
    attacker: EmbeddingAttacker,
    embeddings: np.ndarray,
    method: str,
    epsilons: list[float],
    output_dir: str,
    batch_size: int = 256,
) -> list[dict[str, Any]]:
    """Run attack with multiple epsilon values.

    Args:
        attacker: EmbeddingAttacker instance.
        embeddings: Input embeddings.
        method: Attack method name.
        epsilons: List of epsilon values.
        output_dir: Output directory.
        batch_size: Batch size.

    Returns:
        List of result dictionaries for each epsilon.
    """
    results_list = []

    for eps in tqdm(epsilons, desc=f"{method.upper()} epsilon sweep"):
        kwargs: dict[str, Any] = {"epsilon": eps, "batch_size": batch_size}

        if method == "pgd":
            kwargs["alpha"] = eps / 4
            kwargs["iterations"] = 20

        method_dir = os.path.join(output_dir, method, f"eps_{eps:.4f}")

        try:
            results = attacker.run_attack(
                method=method,
                embeddings=embeddings,
                output_dir=method_dir,
                **kwargs,
            )
            results["epsilon"] = eps
            results_list.append(results)
        except Exception:
            logger.exception("Failed: %s eps=%.4f", method, eps)

    return results_list


def run_deepfool_sweep(
    attacker: EmbeddingAttacker,
    embeddings: np.ndarray,
    output_dir: str,
) -> list[dict[str, Any]]:
    """Run DeepFool with varying overshoot values.

    Args:
        attacker: EmbeddingAttacker instance.
        embeddings: Input embeddings.
        output_dir: Output directory.

    Returns:
        List of result dictionaries.
    """
    overshoots = [0.02, 0.05, 0.1]
    results_list = []

    for ov in tqdm(overshoots, desc="DeepFool overshoot sweep"):
        method_dir = os.path.join(output_dir, "deepfool", f"overshoot_{ov:.3f}")
        try:
            results = attacker.run_attack(
                method="deepfool",
                embeddings=embeddings,
                output_dir=method_dir,
                overshoot=ov,
                max_iterations=100,
            )
            results["overshoot"] = ov
            results_list.append(results)
        except Exception:
            logger.exception("Failed: deepfool overshoot=%.3f", ov)

    return results_list


def run_cw_sweep(
    attacker: EmbeddingAttacker,
    embeddings: np.ndarray,
    output_dir: str,
    batch_size: int = 64,
) -> list[dict[str, Any]]:
    """Run C&W with varying c values.

    Args:
        attacker: EmbeddingAttacker instance.
        embeddings: Input embeddings.
        output_dir: Output directory.
        batch_size: Batch size.

    Returns:
        List of result dictionaries.
    """
    c_values = [0.1, 1.0, 10.0]
    results_list = []

    for c in tqdm(c_values, desc="C&W c sweep"):
        method_dir = os.path.join(output_dir, "cw", f"c_{c:.2f}")
        try:
            results = attacker.run_attack(
                method="cw",
                embeddings=embeddings,
                output_dir=method_dir,
                c=c,
                iterations=500,
                batch_size=batch_size,
            )
            results["c"] = c
            results_list.append(results)
        except Exception:
            logger.exception("Failed: cw c=%.2f", c)

    return results_list


def plot_epsilon_vs_success_rate(
    all_results: dict[str, list[dict[str, Any]]],
    output_dir: str,
) -> None:
    """Plot success rate vs epsilon for each attack method.

    Args:
        all_results: Results grouped by method.
        output_dir: Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    thresholds = [0.5, 0.4, 0.3]

    for ax, threshold in zip(axes, thresholds):
        for method, results_list in all_results.items():
            if method in ("deepfool", "cw"):
                continue
            epsilons = [r.get("epsilon", 0) for r in results_list]
            rates = [
                r.get("results", {}).get(f"success_rate_{threshold}", 0)
                for r in results_list
            ]
            ax.plot(epsilons, rates, "o-", label=method.upper(), linewidth=2)

        ax.set_xlabel("Epsilon (ε)", fontsize=12)
        ax.set_ylabel("Success Rate", fontsize=12)
        ax.set_title(f"Threshold = {threshold}", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    plt.suptitle("Attack Success Rate vs Perturbation Budget", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "epsilon_vs_success_rate.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()
    logger.info("Saved epsilon_vs_success_rate.png")


def plot_probability_distributions(
    model: tf.keras.Model,
    embeddings: np.ndarray,
    all_results: dict[str, list[dict[str, Any]]],
    output_dir: str,
) -> None:
    """Plot before/after probability distributions.

    Args:
        model: Target model.
        embeddings: Original embeddings.
        all_results: Attack results.
        output_dir: Output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    original_probs = predict_batch(model, embeddings)

    # Find best epsilon result for each method
    methods_to_plot = []
    for method, results_list in all_results.items():
        if not results_list:
            continue
        # Pick the result with highest success rate at 0.5
        best = max(
            results_list,
            key=lambda r: r.get("results", {}).get("success_rate_0.5", 0),
        )
        methods_to_plot.append((method, best))

    n_plots = len(methods_to_plot) + 1
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    # Original distribution
    axes[0].hist(original_probs, bins=50, alpha=0.7, color="blue", edgecolor="black")
    axes[0].axvline(x=0.5, color="red", linestyle="--", label="Threshold")
    axes[0].set_title("Original", fontsize=12)
    axes[0].set_xlabel("NSFW Probability")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    for idx, (method, best_result) in enumerate(methods_to_plot, 1):
        params = best_result.get("parameters", {})
        # Load adversarial embeddings if available
        eps = best_result.get("epsilon", params.get("epsilon", "?"))
        method_dir = os.path.join(
            output_dir, "..", method, f"eps_{eps:.4f}" if isinstance(eps, float) else method
        )
        adv_path = os.path.join(method_dir, f"{method}_embeddings.npy")

        if os.path.exists(adv_path):
            adv_embeddings = np.load(adv_path)
            adv_probs = predict_batch(model, adv_embeddings)
        else:
            # Use metrics from results
            adv_probs = None

        if adv_probs is not None:
            axes[idx].hist(
                adv_probs, bins=50, alpha=0.7, color="orange", edgecolor="black"
            )

        axes[idx].axvline(x=0.5, color="red", linestyle="--", label="Threshold")
        sr = best_result.get("results", {}).get("success_rate_0.5", "N/A")
        axes[idx].set_title(f"{method.upper()} (SR={sr:.2%})" if isinstance(sr, float) else f"{method.upper()}", fontsize=12)
        axes[idx].set_xlabel("NSFW Probability")
        axes[idx].legend()

    plt.suptitle("Probability Distribution: Before vs After Attack", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "probability_distributions.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()
    logger.info("Saved probability_distributions.png")


def plot_noise_distributions(
    all_results: dict[str, list[dict[str, Any]]],
    output_dir: str,
) -> None:
    """Plot noise norm distributions across methods.

    Args:
        all_results: Attack results.
        output_dir: Output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    methods = []
    l2_norms = []
    linf_norms = []

    for method, results_list in all_results.items():
        for r in results_list:
            metrics = r.get("results", {})
            if "avg_l2_norm" in metrics:
                methods.append(method.upper())
                l2_norms.append(metrics["avg_l2_norm"])
                linf_norms.append(metrics.get("avg_linf_norm", 0))

    if not methods:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(range(len(methods)), l2_norms, tick_label=methods, color="steelblue")
    ax1.set_title("Average L2 Norm of Perturbation", fontsize=12)
    ax1.set_ylabel("L2 Norm")
    ax1.tick_params(axis="x", rotation=45)

    ax2.bar(range(len(methods)), linf_norms, tick_label=methods, color="coral")
    ax2.set_title("Average L-inf Norm of Perturbation", fontsize=12)
    ax2.set_ylabel("L-inf Norm")
    ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "noise_distributions.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()
    logger.info("Saved noise_distributions.png")


def plot_comparison_table(
    all_results: dict[str, list[dict[str, Any]]],
    output_dir: str,
) -> None:
    """Generate a comparison summary table as an image.

    Args:
        all_results: Attack results.
        output_dir: Output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    for method, results_list in all_results.items():
        for r in results_list:
            metrics = r.get("results", {})
            params = r.get("parameters", {})
            eps = r.get("epsilon", params.get("epsilon", "-"))
            rows.append([
                method.upper(),
                f"{eps}" if isinstance(eps, (int, float)) else eps,
                f"{metrics.get('success_rate_0.5', 0):.2%}",
                f"{metrics.get('success_rate_0.3', 0):.2%}",
                f"{metrics.get('avg_prob_reduction', 0):.4f}",
                f"{metrics.get('avg_l2_norm', 0):.6f}",
            ])

    if not rows:
        return

    fig, ax = plt.subplots(figsize=(14, max(3, len(rows) * 0.4 + 1)))
    ax.axis("off")

    headers = ["Method", "ε", "SR@0.5", "SR@0.3", "Avg ΔP", "Avg L2"]
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Style header
    for j in range(len(headers)):
        table[(0, j)].set_facecolor("#4472C4")
        table[(0, j)].set_text_props(color="white", weight="bold")

    plt.title("Attack Comparison Summary", fontsize=14, pad=20)
    plt.savefig(
        os.path.join(output_dir, "comparison_table.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()
    logger.info("Saved comparison_table.png")


def main(argv: list[str] | None = None) -> None:
    """Main entry point for robustness evaluation."""
    args = parse_args(argv)

    setup_logging(args.log_level)
    set_seed(args.seed)
    detect_gpu()

    logger.info("=" * 60)
    logger.info("Robustness Evaluation")
    logger.info("Methods: %s", args.methods)
    logger.info("Epsilon range: %s", args.epsilon_range)
    logger.info("=" * 60)

    # Load
    model = load_target_model(args.target_model)
    embeddings = load_embeddings(args.embeddings_dir)

    if args.max_samples and len(embeddings) > args.max_samples:
        logger.info("Limiting to %d samples", args.max_samples)
        embeddings = embeddings[:args.max_samples]

    config = {}
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        logger.warning("Config not found, using defaults")

    attacker = EmbeddingAttacker(model, config)
    vis_dir = os.path.join(args.output_dir, "visualizations")
    all_results: dict[str, list[dict[str, Any]]] = {}

    # Run epsilon sweeps for gradient methods
    for method in args.methods:
        if method in ("fgsm", "pgd"):
            results = run_epsilon_sweep(
                attacker, embeddings, method,
                args.epsilon_range, args.output_dir, args.batch_size,
            )
            all_results[method] = results
        elif method == "cw":
            results = run_cw_sweep(
                attacker, embeddings, args.output_dir,
                batch_size=min(64, args.batch_size),
            )
            all_results["cw"] = results
        elif method == "deepfool":
            results = run_deepfool_sweep(
                attacker, embeddings, args.output_dir,
            )
            all_results["deepfool"] = results

    # Generate visualizations
    logger.info("Generating visualizations...")
    plot_epsilon_vs_success_rate(all_results, vis_dir)
    plot_probability_distributions(model, embeddings, all_results, vis_dir)
    plot_noise_distributions(all_results, vis_dir)
    plot_comparison_table(all_results, vis_dir)

    # Save combined results
    combined = {
        method: results_list
        for method, results_list in all_results.items()
    }
    save_results(
        combined, args.output_dir, prefix="robustness_analysis"
    )

    # Print summary
    print("\n" + "=" * 60)
    print("ROBUSTNESS EVALUATION SUMMARY")
    print("=" * 60)
    for method, results_list in all_results.items():
        print(f"\n--- {method.upper()} ---")
        for r in results_list:
            metrics = r.get("results", {})
            eps = r.get("epsilon", r.get("overshoot", r.get("c", "?")))
            sr = metrics.get("success_rate_0.5", 0)
            print(f"  param={eps}: SR@0.5={sr:.2%}")

    print(f"\nResults saved to: {args.output_dir}")
    print(f"Visualizations saved to: {vis_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
