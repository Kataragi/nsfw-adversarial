"""Command-line interface for adversarial noise generation on images.

Usage:
    python src/noise_generator.py --method fgsm --images-dir dataset/nsfw_images
"""

import argparse
import logging

from src.image_attacker import ImageAttacker
from src.pipeline import build_pipeline
from src.utils import (
    detect_gpu,
    load_config,
    load_images,
    set_seed,
    setup_logging,
)

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Adversarial Noise Generator for NSFW Classifier (image-based)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument(
        "--method", type=str, required=True,
        choices=["fgsm", "pgd", "cw", "deepfool"],
        help="Attack method",
    )
    parser.add_argument(
        "--images-dir", type=str, required=True,
        help="Directory containing NSFW images to attack",
    )

    # Pipeline paths
    parser.add_argument(
        "--classifier-model", type=str,
        default="models/target_classifier/pnsfwmedia_classifier.keras",
        help="Path to pNSFWMedia classifier (.keras)",
    )
    parser.add_argument(
        "--projection-path", type=str,
        default="models/nudenet_projection.npy",
        help="Path to projection matrix (.npy)",
    )
    parser.add_argument(
        "--nudenet-onnx", type=str, default=None,
        help="Path to NudeNet ONNX model (auto-detect if omitted)",
    )
    parser.add_argument(
        "--backbone-cache", type=str, default="models/tf_backbone",
        help="Directory to cache converted TF backbone",
    )

    # Output
    parser.add_argument(
        "--output-dir", type=str, default="experiments/attack_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--config", type=str, default="config/attack_config.yaml",
        help="YAML configuration file",
    )

    # Common parameters
    parser.add_argument(
        "--epsilon", type=float, default=8 / 255,
        help="L-inf perturbation budget (pixel scale [0,1])",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-images", type=int, default=None,
        help="Limit number of images (for quick tests)",
    )

    # PGD
    parser.add_argument("--alpha", type=float, default=2 / 255, help="PGD step size")
    parser.add_argument("--iterations", type=int, default=20, help="PGD/CW iterations")
    parser.add_argument("--random-start", action="store_true", default=True)

    # C&W
    parser.add_argument("--cw-c", type=float, default=1.0, help="C&W trade-off")
    parser.add_argument("--cw-kappa", type=float, default=0.0, help="C&W confidence")
    parser.add_argument("--cw-lr", type=float, default=0.01, help="C&W learning rate")
    parser.add_argument("--binary-search-steps", type=int, default=9)

    # DeepFool
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--overshoot", type=float, default=0.02)

    # TensorBoard
    parser.add_argument("--tensorboard", action="store_true", default=False)
    parser.add_argument("--tb-log-dir", type=str, default="logs")

    # Logging
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    return parser.parse_args(argv)


def build_attack_kwargs(args: argparse.Namespace) -> dict:
    """Map CLI args to attack-specific kwargs."""
    method = args.method
    kwargs: dict = {"batch_size": args.batch_size}

    if method == "fgsm":
        kwargs["epsilon"] = args.epsilon

    elif method == "pgd":
        kwargs["epsilon"] = args.epsilon
        kwargs["alpha"] = args.alpha
        kwargs["iterations"] = args.iterations
        kwargs["random_start"] = args.random_start

    elif method == "cw":
        kwargs["c"] = args.cw_c
        kwargs["kappa"] = args.cw_kappa
        kwargs["iterations"] = args.iterations
        kwargs["learning_rate"] = args.cw_lr
        kwargs["binary_search_steps"] = args.binary_search_steps

    elif method == "deepfool":
        kwargs["max_iterations"] = args.max_iterations
        kwargs["overshoot"] = args.overshoot

    return kwargs


def main(argv: list[str] | None = None) -> None:
    """Main entry point."""
    args = parse_args(argv)

    setup_logging(args.log_level)
    set_seed(args.seed)
    detect_gpu()

    logger.info("=" * 60)
    logger.info("Adversarial Noise Generator (image-based)")
    logger.info("Method: %s", args.method.upper())
    logger.info("Images: %s", args.images_dir)
    logger.info("=" * 60)

    # Build end-to-end pipeline
    pipeline = build_pipeline(
        nudenet_onnx_path=args.nudenet_onnx,
        projection_path=args.projection_path,
        classifier_path=args.classifier_model,
        backbone_cache_dir=args.backbone_cache,
    )

    # Load images
    images, filenames = load_images(
        args.images_dir, max_images=args.max_images,
    )

    # Config
    config: dict = {}
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        logger.warning("Config not found: %s, using defaults", args.config)

    if args.tensorboard:
        config.setdefault("tensorboard", {})["enabled"] = True
        config["tensorboard"]["log_dir"] = args.tb_log_dir

    # Run attack
    attacker = ImageAttacker(pipeline, config)
    attack_kwargs = build_attack_kwargs(args)

    results = attacker.run_attack(
        method=args.method,
        images=images,
        output_dir=args.output_dir,
        filenames=filenames,
        **attack_kwargs,
    )

    # Print summary
    metrics = results.get("results", {})
    print("\n" + "=" * 60)
    print(f"Attack: {args.method.upper()}")
    print(f"Parameters: {results.get('parameters', {})}")
    print("-" * 60)
    print(f"Total samples: {metrics.get('total_samples', 'N/A')}")
    print(f"NSFW samples:  {metrics.get('nsfw_samples', 'N/A')}")

    for t in [0.5, 0.4, 0.3]:
        key = f"success_rate_{t}"
        if key in metrics:
            print(f"Success rate (threshold={t}): {metrics[key]:.4f}")

    if "avg_prob_reduction" in metrics:
        print(f"Avg probability reduction: {metrics['avg_prob_reduction']:.4f}")
    if "avg_l2_norm" in metrics:
        print(f"Avg L2 norm:    {metrics['avg_l2_norm']:.6f}")
    if "avg_linf_norm" in metrics:
        print(f"Avg L-inf norm: {metrics['avg_linf_norm']:.6f}")
    if "avg_linf_norm_pixel" in metrics:
        print(f"Avg L-inf (px): {metrics['avg_linf_norm_pixel']:.2f}/255")
    if "avg_iterations" in metrics:
        print(f"Avg iterations: {metrics['avg_iterations']:.1f}")

    print(f"\nResults saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
