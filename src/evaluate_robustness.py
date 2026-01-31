"""Robustness evaluation script (image-based).

Runs multiple attack methods with varying parameters and generates
comprehensive evaluation reports with visualisations.

Usage:
    python src/evaluate_robustness.py \
        --images-dir dataset/nsfw_images \
        --methods fgsm pgd cw deepfool \
        --epsilon-range 0.01 0.02 0.031 0.063
"""

import argparse
import logging
import os
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from src.image_attacker import ImageAttacker
from src.pipeline import build_pipeline
from src.utils import (
    detect_gpu,
    load_config,
    load_images,
    save_results,
    set_seed,
    setup_logging,
)

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Robustness Evaluation (image-based)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--images-dir", type=str, required=True)
    parser.add_argument(
        "--classifier-model", type=str,
        default="models/target_classifier/pnsfwmedia_classifier.keras",
    )
    parser.add_argument(
        "--projection-path", type=str,
        default="models/nudenet_projection.npy",
    )
    parser.add_argument("--nudenet-onnx", type=str, default=None)
    parser.add_argument("--backbone-cache", type=str, default="models/tf_backbone")
    parser.add_argument(
        "--methods", type=str, nargs="+",
        default=["fgsm", "pgd", "cw", "deepfool"],
    )
    parser.add_argument(
        "--epsilon-range", type=float, nargs="+",
        default=[4 / 255, 8 / 255, 16 / 255, 32 / 255],
    )
    parser.add_argument(
        "--output-dir", type=str, default="experiments/robustness_analysis",
    )
    parser.add_argument("--config", type=str, default="config/attack_config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument("--max-images", type=int, default=None)
    return parser.parse_args(argv)


# ── sweeps ─────────────────────────────────────────────────────────


def run_epsilon_sweep(
    attacker: ImageAttacker,
    images: np.ndarray,
    method: str,
    epsilons: list[float],
    output_dir: str,
    batch_size: int = 8,
    filenames: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Sweep epsilon values for FGSM/PGD."""
    results_list = []

    for eps in tqdm(epsilons, desc=f"{method.upper()} epsilon sweep"):
        kwargs: dict[str, Any] = {"epsilon": eps, "batch_size": batch_size}
        if method == "pgd":
            kwargs["alpha"] = eps / 4
            kwargs["iterations"] = 20

        method_dir = os.path.join(output_dir, method, f"eps_{eps:.4f}")
        try:
            r = attacker.run_attack(
                method=method, images=images, output_dir=method_dir,
                filenames=filenames, **kwargs,
            )
            r["epsilon"] = eps
            results_list.append(r)
        except Exception:
            logger.exception("Failed: %s eps=%.4f", method, eps)

    return results_list


def run_cw_sweep(
    attacker: ImageAttacker,
    images: np.ndarray,
    output_dir: str,
    batch_size: int = 4,
    filenames: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Sweep c values for C&W."""
    c_values = [0.1, 1.0, 10.0]
    results_list = []
    for c in tqdm(c_values, desc="C&W c sweep"):
        method_dir = os.path.join(output_dir, "cw", f"c_{c:.2f}")
        try:
            r = attacker.run_attack(
                method="cw", images=images, output_dir=method_dir,
                filenames=filenames, c=c, iterations=500, batch_size=batch_size,
            )
            r["c"] = c
            results_list.append(r)
        except Exception:
            logger.exception("Failed: cw c=%.2f", c)
    return results_list


def run_deepfool_sweep(
    attacker: ImageAttacker,
    images: np.ndarray,
    output_dir: str,
    filenames: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Sweep overshoot values for DeepFool."""
    overshoots = [0.02, 0.05, 0.1]
    results_list = []
    for ov in tqdm(overshoots, desc="DeepFool overshoot sweep"):
        method_dir = os.path.join(output_dir, "deepfool", f"overshoot_{ov:.3f}")
        try:
            r = attacker.run_attack(
                method="deepfool", images=images, output_dir=method_dir,
                filenames=filenames, overshoot=ov, max_iterations=100,
            )
            r["overshoot"] = ov
            results_list.append(r)
        except Exception:
            logger.exception("Failed: deepfool ov=%.3f", ov)
    return results_list


# ── visualisation ──────────────────────────────────────────────────


def plot_epsilon_vs_success_rate(
    all_results: dict[str, list[dict[str, Any]]], output_dir: str
) -> None:
    """Success rate vs epsilon for FGSM/PGD."""
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, threshold in zip(axes, [0.5, 0.4, 0.3]):
        for method, rlist in all_results.items():
            if method in ("deepfool", "cw"):
                continue
            epsilons = [r.get("epsilon", 0) for r in rlist]
            rates = [
                r.get("results", {}).get(f"success_rate_{threshold}", 0)
                for r in rlist
            ]
            eps_px = [e * 255 for e in epsilons]
            ax.plot(eps_px, rates, "o-", label=method.upper(), linewidth=2)

        ax.set_xlabel("Epsilon (pixel /255)", fontsize=12)
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
    pipeline: tf.keras.Model,
    images: np.ndarray,
    all_results: dict[str, list[dict[str, Any]]],
    output_dir: str,
) -> None:
    """Before/after probability distributions."""
    os.makedirs(output_dir, exist_ok=True)
    original_probs = pipeline.predict_images(images)

    methods_to_plot = []
    for method, rlist in all_results.items():
        if not rlist:
            continue
        best = max(rlist, key=lambda r: r.get("results", {}).get("success_rate_0.5", 0))
        methods_to_plot.append((method, best))

    n_plots = len(methods_to_plot) + 1
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    axes[0].hist(original_probs, bins=50, alpha=0.7, color="blue", edgecolor="black")
    axes[0].axvline(x=0.5, color="red", linestyle="--", label="Threshold")
    axes[0].set_title("Original", fontsize=12)
    axes[0].set_xlabel("NSFW Probability")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    for idx, (method, best_result) in enumerate(methods_to_plot, 1):
        sr = best_result.get("results", {}).get("success_rate_0.5", "N/A")
        title = (
            f"{method.upper()} (SR={sr:.2%})"
            if isinstance(sr, float)
            else f"{method.upper()}"
        )
        axes[idx].axvline(x=0.5, color="red", linestyle="--", label="Threshold")
        axes[idx].set_title(title, fontsize=12)
        axes[idx].set_xlabel("NSFW Probability")
        axes[idx].legend()

    plt.suptitle("Probability Distribution: Before vs After", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "probability_distributions.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()
    logger.info("Saved probability_distributions.png")


def plot_comparison_table(
    all_results: dict[str, list[dict[str, Any]]], output_dir: str
) -> None:
    """Comparison summary table as image."""
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for method, rlist in all_results.items():
        for r in rlist:
            m = r.get("results", {})
            eps = r.get("epsilon", r.get("c", r.get("overshoot", "-")))
            eps_str = f"{eps * 255:.1f}/255" if isinstance(eps, float) and eps < 1 else str(eps)
            rows.append([
                method.upper(),
                eps_str,
                f"{m.get('success_rate_0.5', 0):.2%}",
                f"{m.get('success_rate_0.3', 0):.2%}",
                f"{m.get('avg_prob_reduction', 0):.4f}",
                f"{m.get('avg_linf_norm_pixel', 0):.2f}",
            ])

    if not rows:
        return

    fig, ax = plt.subplots(figsize=(14, max(3, len(rows) * 0.4 + 1)))
    ax.axis("off")
    headers = ["Method", "Param", "SR@0.5", "SR@0.3", "Avg dP", "L-inf(px)"]
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
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


# ── main ───────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    """Main entry point."""
    args = parse_args(argv)

    setup_logging(args.log_level)
    set_seed(args.seed)
    detect_gpu()

    logger.info("=" * 60)
    logger.info("Robustness Evaluation (image-based)")
    logger.info("Methods: %s", args.methods)
    logger.info("Epsilon range: %s", args.epsilon_range)
    logger.info("=" * 60)

    # Build pipeline
    pipeline = build_pipeline(
        nudenet_onnx_path=args.nudenet_onnx,
        projection_path=args.projection_path,
        classifier_path=args.classifier_model,
        backbone_cache_dir=args.backbone_cache,
    )

    # Load images
    images, filenames = load_images(args.images_dir, max_images=args.max_images)

    config: dict = {}
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        logger.warning("Config not found, using defaults")

    attacker = ImageAttacker(pipeline, config)
    vis_dir = os.path.join(args.output_dir, "visualizations")
    all_results: dict[str, list[dict[str, Any]]] = {}

    for method in args.methods:
        if method in ("fgsm", "pgd"):
            all_results[method] = run_epsilon_sweep(
                attacker, images, method, args.epsilon_range,
                args.output_dir, args.batch_size, filenames,
            )
        elif method == "cw":
            all_results["cw"] = run_cw_sweep(
                attacker, images, args.output_dir,
                batch_size=min(4, args.batch_size), filenames=filenames,
            )
        elif method == "deepfool":
            all_results["deepfool"] = run_deepfool_sweep(
                attacker, images, args.output_dir, filenames=filenames,
            )

    # Visualisations
    logger.info("Generating visualisations...")
    plot_epsilon_vs_success_rate(all_results, vis_dir)
    plot_probability_distributions(pipeline, images, all_results, vis_dir)
    plot_comparison_table(all_results, vis_dir)

    save_results(
        {m: rl for m, rl in all_results.items()},
        args.output_dir, prefix="robustness_analysis",
    )

    # Summary
    print("\n" + "=" * 60)
    print("ROBUSTNESS EVALUATION SUMMARY")
    print("=" * 60)
    for method, rlist in all_results.items():
        print(f"\n--- {method.upper()} ---")
        for r in rlist:
            m = r.get("results", {})
            eps = r.get("epsilon", r.get("overshoot", r.get("c", "?")))
            sr = m.get("success_rate_0.5", 0)
            if isinstance(eps, float) and eps < 1:
                print(f"  eps={eps * 255:.1f}/255: SR@0.5={sr:.2%}")
            else:
                print(f"  param={eps}: SR@0.5={sr:.2%}")

    print(f"\nResults saved to: {args.output_dir}")
    print(f"Visualisations: {vis_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
