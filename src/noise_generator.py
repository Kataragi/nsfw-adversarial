"""Command-line interface for adversarial noise generation.

Usage:
    python src/noise_generator.py --method fgsm --target-model MODEL --embeddings-dir DIR
"""

import argparse
import logging
import sys

from src.embedding_attacker import EmbeddingAttacker
from src.utils import (
    detect_gpu,
    load_config,
    load_embeddings,
    load_target_model,
    set_seed,
    setup_logging,
)

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (defaults to sys.argv).

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Adversarial Noise Generator for NSFW Classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["fgsm", "pgd", "cw", "deepfool"],
        help="Attack method to use",
    )
    parser.add_argument(
        "--target-model",
        type=str,
        required=True,
        help="Path to target classifier model (.keras)",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        required=True,
        help="Directory containing embedding files (.npy/.npz)",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/attack_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/attack_config.yaml",
        help="Path to configuration file",
    )

    # Common attack parameters
    parser.add_argument(
        "--epsilon", type=float, default=0.05,
        help="Maximum perturbation (L-inf norm)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )

    # PGD-specific
    parser.add_argument(
        "--alpha", type=float, default=0.01,
        help="PGD step size per iteration",
    )
    parser.add_argument(
        "--iterations", type=int, default=20,
        help="Number of attack iterations (PGD/C&W)",
    )
    parser.add_argument(
        "--random-start", action="store_true", default=True,
        help="PGD: random initialization within epsilon-ball",
    )

    # C&W-specific
    parser.add_argument(
        "--cw-c", type=float, default=1.0,
        help="C&W: trade-off constant",
    )
    parser.add_argument(
        "--cw-kappa", type=float, default=0.0,
        help="C&W: confidence margin",
    )
    parser.add_argument(
        "--cw-lr", type=float, default=0.01,
        help="C&W: optimizer learning rate",
    )
    parser.add_argument(
        "--binary-search-steps", type=int, default=9,
        help="C&W: binary search steps for c",
    )

    # DeepFool-specific
    parser.add_argument(
        "--max-iterations", type=int, default=100,
        help="DeepFool: maximum iterations",
    )
    parser.add_argument(
        "--overshoot", type=float, default=0.02,
        help="DeepFool: overshoot parameter",
    )

    # TensorBoard
    parser.add_argument(
        "--tensorboard", action="store_true", default=False,
        help="Enable TensorBoard logging",
    )
    parser.add_argument(
        "--tb-log-dir", type=str, default="logs",
        help="TensorBoard log directory",
    )

    # Logging
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    return parser.parse_args(argv)


def build_attack_kwargs(args: argparse.Namespace) -> dict:
    """Build attack-specific keyword arguments from parsed args.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Dictionary of attack parameters.
    """
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
    """Main entry point for the noise generator CLI.

    Args:
        argv: Optional argument list.
    """
    args = parse_args(argv)

    # Setup
    setup_logging(args.log_level)
    set_seed(args.seed)
    detect_gpu()

    logger.info("=" * 60)
    logger.info("Adversarial Noise Generator")
    logger.info("Method: %s", args.method.upper())
    logger.info("Target model: %s", args.target_model)
    logger.info("Embeddings: %s", args.embeddings_dir)
    logger.info("=" * 60)

    # Load model and embeddings
    model = load_target_model(args.target_model)
    embeddings = load_embeddings(args.embeddings_dir)

    # Configuration
    config = {}
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        logger.warning("Config file not found: %s, using defaults", args.config)

    if args.tensorboard:
        config.setdefault("tensorboard", {})["enabled"] = True
        config["tensorboard"]["log_dir"] = args.tb_log_dir

    # Run attack
    attacker = EmbeddingAttacker(model, config)
    attack_kwargs = build_attack_kwargs(args)

    results = attacker.run_attack(
        method=args.method,
        embeddings=embeddings,
        output_dir=args.output_dir,
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
        print(f"Avg L2 norm:  {metrics['avg_l2_norm']:.6f}")
    if "avg_linf_norm" in metrics:
        print(f"Avg L-inf norm: {metrics['avg_linf_norm']:.6f}")
    if "avg_iterations" in metrics:
        print(f"Avg iterations: {metrics['avg_iterations']:.1f}")

    print(f"\nResults saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
