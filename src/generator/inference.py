"""ジェネレータベース敵対的摂動の推論スクリプト。

訓練済みのAdversarialGeneratorを使用して、NSFW画像に摂動を適用し、
分類器を回避する攻撃を実行する。

機能:
    - 訓練済みジェネレータのロード
    - 画像ディレクトリからのバッチ推論
    - 摂動画像・摂動マップの保存
    - メトリクス計算 (PSNR, SSIM, L2, L-inf)
    - 分類器での攻撃成功率の検証

使用例:
    # 基本的な使用方法
    python -m src.generator.inference \\
        --generator checkpoints/generator/generator_final.pt \\
        --input_dir data/nsfw_images \\
        --output_dir results/perturbed

    # 分類器での検証も実行
    python -m src.generator.inference \\
        --generator checkpoints/generator/generator_final.pt \\
        --input_dir data/nsfw_images \\
        --output_dir results/perturbed \\
        --verify \\
        --classifier models/target_classifier/pnsfwmedia_classifier.keras
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from .model import AdversarialGenerator

# 分類器検証用（オプション）
try:
    from .classifier_wrapper import build_frozen_classifier
    CLASSIFIER_AVAILABLE = True
except ImportError:
    CLASSIFIER_AVAILABLE = False
    logging.warning("分類器ラッパーのインポートに失敗しました。--verify は使用できません。")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

IMAGE_SIZE = 320


def load_generator(
    checkpoint_path: str,
    device: str = "cuda",
    max_perturbation: float = 16 / 255,
) -> AdversarialGenerator:
    """訓練済みジェネレータをロードする。

    Args:
        checkpoint_path: .pt ファイルのパス。
        device: 実行デバイス ("cuda" or "cpu")。
        max_perturbation: 摂動の最大値（デフォルト 16/255）。

    Returns:
        ロードされた AdversarialGenerator モデル。
    """
    logger.info("ジェネレータをロード中: %s", checkpoint_path)

    # チェックポイントファイルの読み込み
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # state_dict の取得（チェックポイント形式に対応）
    if isinstance(checkpoint, dict):
        if "generator_state_dict" in checkpoint:
            state_dict = checkpoint["generator_state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            # チェックポイント自体が state_dict の可能性
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # モデルの構築
    model = AdversarialGenerator(
        in_channels=3,
        base_channels=32,
        depth=4,
        max_perturbation=max_perturbation,
        use_attention=True,
    )

    # 重みのロード
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    logger.info("ジェネレータのロード完了")
    return model


def load_images_from_directory(
    image_dir: str,
    image_size: int = IMAGE_SIZE,
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp"),
) -> tuple[list[torch.Tensor], list[str], list[tuple[int, int]]]:
    """ディレクトリから画像を読み込む。

    Args:
        image_dir: 画像ディレクトリのパス。
        image_size: リサイズ後の画像サイズ（推論用に320x320に統一）。
        extensions: 対象とする拡張子。

    Returns:
        (images, filenames, original_sizes) のタプル。
        images: [(3, H, W), ...] のテンソルリスト [0, 1]。
        filenames: ファイル名のリスト。
        original_sizes: [(width, height), ...] の元サイズリスト。
    """
    logger.info("画像を読み込み中: %s", image_dir)

    image_paths = []
    for ext in extensions:
        image_paths.extend(Path(image_dir).glob(f"*{ext}"))
        image_paths.extend(Path(image_dir).glob(f"*{ext.upper()}"))

    if not image_paths:
        raise ValueError(f"{image_dir} に画像が見つかりません")

    images = []
    filenames = []
    original_sizes = []

    for img_path in tqdm(sorted(image_paths), desc="画像読み込み"):
        try:
            img = Image.open(img_path).convert("RGB")
            # 元のサイズを記録
            original_sizes.append(img.size)  # (width, height)

            # 推論用に320x320にリサイズ
            img_resized = img.resize((image_size, image_size), Image.LANCZOS)
            img_array = np.array(img_resized, dtype=np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # (3, H, W)
            images.append(img_tensor)
            filenames.append(img_path.name)
        except Exception as e:
            logger.warning("画像の読み込みに失敗: %s (%s)", img_path, e)
            continue

    logger.info("%d 枚の画像を読み込みました", len(images))
    return images, filenames, original_sizes


def compute_metrics(
    original: torch.Tensor,
    perturbed: torch.Tensor,
) -> dict[str, float]:
    """画像間のメトリクスを計算する。

    Args:
        original: (3, H, W) 元画像 [0, 1]。
        perturbed: (3, H, W) 摂動画像 [0, 1]。

    Returns:
        メトリクスの辞書 (PSNR, SSIM, L2, L-inf)。
    """
    # MSE と PSNR
    mse = F.mse_loss(perturbed, original).item()
    psnr = 10 * np.log10(1.0 / (mse + 1e-10)) if mse > 0 else float("inf")

    # L2 距離
    l2_dist = torch.norm(perturbed - original, p=2).item()

    # L-inf 距離
    linf_dist = torch.max(torch.abs(perturbed - original)).item()

    # SSIM（簡易版: 完全な実装は skimage を使用）
    # ここでは計算コストを抑えるため構造類似度の簡易近似を使用
    # より正確な SSIM が必要な場合は skimage.metrics.structural_similarity を使用
    try:
        from skimage.metrics import structural_similarity as ssim
        orig_np = original.permute(1, 2, 0).cpu().numpy()
        pert_np = perturbed.permute(1, 2, 0).cpu().numpy()
        ssim_value = ssim(
            orig_np,
            pert_np,
            data_range=1.0,
            channel_axis=2,
        )
    except ImportError:
        # skimage がない場合は None
        ssim_value = None

    return {
        "psnr": psnr,
        "ssim": ssim_value,
        "l2_distance": l2_dist,
        "linf_distance": linf_dist,
        "mse": mse,
    }


def save_image(
    tensor: torch.Tensor,
    output_path: str,
) -> None:
    """テンソルを画像ファイルとして保存する。

    Args:
        tensor: (3, H, W) [0, 1] の画像テンソル。
        output_path: 保存先パス。
    """
    img_array = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    img.save(output_path)


def save_perturbation_map(
    perturbation: torch.Tensor,
    output_path: str,
    scale: float = 10.0,
) -> None:
    """摂動マップを可視化して保存する。

    Args:
        perturbation: (3, H, W) 摂動テンソル（元画像との差分）。
        output_path: 保存先パス。
        scale: 可視化のためのスケール係数。
    """
    # 摂動を [-1, 1] 範囲に正規化してスケール
    pert_scaled = torch.clamp(perturbation * scale + 0.5, 0, 1)
    save_image(pert_scaled, output_path)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """テンソルを PIL Image に変換する。

    Args:
        tensor: (3, H, W) [0, 1] の画像テンソル。

    Returns:
        PIL Image オブジェクト。
    """
    img_array = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(img_array)


def run_inference(
    generator: AdversarialGenerator,
    images: list[torch.Tensor],
    filenames: list[str],
    original_sizes: list[tuple[int, int]],
    output_dir: str,
    batch_size: int = 16,
    device: str = "cuda",
    save_perturbation_maps: bool = True,
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    """バッチ推論を実行し、結果を保存する。

    Args:
        generator: AdversarialGenerator モデル。
        images: 画像テンソルのリスト（320x320にリサイズ済み）。
        filenames: ファイル名のリスト。
        original_sizes: [(width, height), ...] の元サイズリスト。
        output_dir: 出力ディレクトリ。
        batch_size: バッチサイズ。
        device: 実行デバイス。
        save_perturbation_maps: 摂動マップも保存するか。

    Returns:
        (perturbed_images, metrics) のタプル。
    """
    os.makedirs(output_dir, exist_ok=True)
    perturbed_dir = os.path.join(output_dir, "perturbed")
    os.makedirs(perturbed_dir, exist_ok=True)

    if save_perturbation_maps:
        perturbation_dir = os.path.join(output_dir, "perturbations")
        os.makedirs(perturbation_dir, exist_ok=True)

    perturbed_images = []
    all_metrics = []

    logger.info("推論を実行中...")

    with torch.no_grad():
        for i in tqdm(range(0, len(images), batch_size), desc="バッチ推論"):
            batch_images = images[i : i + batch_size]
            batch_filenames = filenames[i : i + batch_size]
            batch_sizes = original_sizes[i : i + batch_size]

            # バッチテンソルの作成
            batch_tensor = torch.stack(batch_images).to(device)

            # ジェネレータで摂動を生成
            perturbation_batch = generator(batch_tensor)
            # 元画像に摂動を加算
            perturbed_batch = torch.clamp(batch_tensor + perturbation_batch, 0.0, 1.0)

            # 各画像を保存してメトリクスを計算
            for j, (orig, pert, perturbation, fname, orig_size) in enumerate(
                zip(batch_tensor, perturbed_batch, perturbation_batch, batch_filenames, batch_sizes)
            ):
                # 摂動画像を元のサイズにリサイズして保存
                pert_pil = tensor_to_pil(pert.cpu())
                pert_pil_resized = pert_pil.resize(orig_size, Image.LANCZOS)
                perturbed_path = os.path.join(perturbed_dir, fname)
                pert_pil_resized.save(perturbed_path)

                # 320x320 の摂動画像を記録（分類器検証用）
                perturbed_images.append(pert.cpu())

                # 摂動マップの保存（320x320で保存）
                if save_perturbation_maps:
                    pert_map_path = os.path.join(
                        perturbation_dir,
                        f"pert_{fname}",
                    )
                    save_perturbation_map(perturbation.cpu(), pert_map_path)

                # メトリクスの計算（320x320で計算）
                metrics = compute_metrics(orig.cpu(), pert.cpu())
                metrics["filename"] = fname
                metrics["original_size"] = orig_size
                all_metrics.append(metrics)

    # 平均メトリクスの計算
    avg_metrics = {
        "psnr": np.mean([m["psnr"] for m in all_metrics if np.isfinite(m["psnr"])]),
        "l2_distance": np.mean([m["l2_distance"] for m in all_metrics]),
        "linf_distance": np.mean([m["linf_distance"] for m in all_metrics]),
        "mse": np.mean([m["mse"] for m in all_metrics]),
    }

    if all_metrics[0]["ssim"] is not None:
        avg_metrics["ssim"] = np.mean([m["ssim"] for m in all_metrics])

    summary = {
        "total_images": len(images),
        "average_metrics": avg_metrics,
        "per_image_metrics": all_metrics,
    }

    logger.info("推論完了: %d 枚の画像を処理しました", len(images))
    logger.info("平均メトリクス: PSNR=%.2f, L2=%.4f, L-inf=%.4f",
                avg_metrics["psnr"],
                avg_metrics["l2_distance"],
                avg_metrics["linf_distance"])

    return perturbed_images, summary


def verify_with_classifier(
    original_images: list[torch.Tensor],
    perturbed_images: list[torch.Tensor],
    classifier_path: str,
    device: str = "cuda",
    threshold: float = 0.5,
) -> dict[str, Any]:
    """分類器を使用して攻撃成功率を検証する。

    Args:
        original_images: 元画像のリスト。
        perturbed_images: 摂動画像のリスト。
        classifier_path: 分類器の .keras ファイルパス。
        device: 実行デバイス。
        threshold: NSFW判定の閾値。

    Returns:
        検証結果の辞書。
    """
    if not CLASSIFIER_AVAILABLE:
        logger.error("分類器ラッパーが利用できません")
        return {}

    logger.info("分類器で攻撃成功率を検証中...")

    # 分類器のロード
    classifier = build_frozen_classifier(
        classifier_path=classifier_path,
        device=device,
    )

    original_predictions = []
    perturbed_predictions = []

    with torch.no_grad():
        # 元画像の予測
        for img in tqdm(original_images, desc="元画像の分類"):
            img_batch = img.unsqueeze(0).to(device)  # (1, 3, H, W)
            pred = classifier(img_batch).cpu().item()
            original_predictions.append(pred)

        # 摂動画像の予測
        for img in tqdm(perturbed_images, desc="摂動画像の分類"):
            img_batch = img.unsqueeze(0).to(device)  # (1, 3, H, W)
            pred = classifier(img_batch).cpu().item()
            perturbed_predictions.append(pred)

    # 攻撃成功率の計算
    original_nsfw = [p >= threshold for p in original_predictions]
    perturbed_nsfw = [p >= threshold for p in perturbed_predictions]

    # 元々NSFWと判定された画像のうち、摂動後にSFWになった割合
    nsfw_count = sum(original_nsfw)
    if nsfw_count > 0:
        success_count = sum(
            orig and not pert
            for orig, pert in zip(original_nsfw, perturbed_nsfw)
        )
        attack_success_rate = success_count / nsfw_count
    else:
        attack_success_rate = 0.0

    # 平均スコアの変化
    avg_original_score = np.mean(original_predictions)
    avg_perturbed_score = np.mean(perturbed_predictions)
    score_reduction = avg_original_score - avg_perturbed_score

    results = {
        "threshold": threshold,
        "original_nsfw_count": nsfw_count,
        "perturbed_nsfw_count": sum(perturbed_nsfw),
        "attack_success_count": success_count if nsfw_count > 0 else 0,
        "attack_success_rate": attack_success_rate,
        "avg_original_score": avg_original_score,
        "avg_perturbed_score": avg_perturbed_score,
        "score_reduction": score_reduction,
        "predictions": {
            "original": original_predictions,
            "perturbed": perturbed_predictions,
        },
    }

    logger.info("攻撃成功率: %.2f%% (%d/%d)",
                attack_success_rate * 100,
                results["attack_success_count"],
                nsfw_count)
    logger.info("スコア変化: %.4f -> %.4f (減少: %.4f)",
                avg_original_score,
                avg_perturbed_score,
                score_reduction)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ジェネレータベース敵対的摂動の推論",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--generator",
        type=str,
        required=True,
        help="訓練済みジェネレータの .pt ファイルパス",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="入力画像ディレクトリ",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="出力ディレクトリ",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="バッチサイズ (デフォルト: 16)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="実行デバイス (デフォルト: cuda)",
    )
    parser.add_argument(
        "--max_perturbation",
        type=float,
        default=16 / 255,
        help="摂動の最大値 (デフォルト: 16/255)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="分類器で攻撃成功率を検証する",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default="models/target_classifier/pnsfwmedia_classifier.keras",
        help="分類器の .keras ファイルパス (デフォルト: models/target_classifier/pnsfwmedia_classifier.keras)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="NSFW判定の閾値 (デフォルト: 0.5)",
    )
    parser.add_argument(
        "--no_perturbation_maps",
        action="store_true",
        help="摂動マップを保存しない",
    )

    args = parser.parse_args()

    # ジェネレータのロード
    generator = load_generator(
        args.generator,
        device=args.device,
        max_perturbation=args.max_perturbation,
    )

    # 画像の読み込み（元サイズも記録）
    images, filenames, original_sizes = load_images_from_directory(args.input_dir)

    # 推論の実行
    perturbed_images, metrics_summary = run_inference(
        generator=generator,
        images=images,
        filenames=filenames,
        original_sizes=original_sizes,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
        save_perturbation_maps=not args.no_perturbation_maps,
    )

    # 分類器での検証（オプション）
    if args.verify:
        verification_results = verify_with_classifier(
            original_images=images,
            perturbed_images=perturbed_images,
            classifier_path=args.classifier,
            device=args.device,
            threshold=args.threshold,
        )
        metrics_summary["verification"] = verification_results

    # 結果をJSONとして保存
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(metrics_summary, f, indent=2, ensure_ascii=False)

    logger.info("結果を保存しました: %s", results_path)
    logger.info("摂動画像: %s", os.path.join(args.output_dir, "perturbed"))
    if not args.no_perturbation_maps:
        logger.info("摂動マップ: %s", os.path.join(args.output_dir, "perturbations"))


if __name__ == "__main__":
    main()
