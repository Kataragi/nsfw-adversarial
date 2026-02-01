"""敵対的摂動ジェネレータの学習スクリプト。

pNSFWMedia分類器を凍結した状態で、ジェネレータ G を学習する。
G はNSFW画像を入力として、分類器がSFWと誤判定する摂動済み画像を出力する。

損失関数:
    L_total = λ_cls * L_cls + λ_l2 * L_l2

    L_cls: 分類損失 (BCE) - 分類器がSFWと判定するよう誘導
    L_l2:  摂動ペナルティ - 元画像との L2 距離を最小化

使い方:
    python -m src.generator.train --config config/generator_config.yaml

    # カスタムパラメータ指定
    python -m src.generator.train \\
        --config config/generator_config.yaml \\
        --train-dir dataset/nsfw_images/train \\
        --val-dir dataset/nsfw_images/val \\
        --epochs 100 \\
        --batch-size 8 \\
        --lr 0.0002

TensorBoard:
    tensorboard --logdir logs/generator
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.generator.classifier_wrapper import build_frozen_classifier
from src.generator.dataset import create_dataloaders
from src.generator.model import AdversarialGenerator

logger = logging.getLogger(__name__)


# ── 設定 ────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """コマンドライン引数をパースする。"""
    parser = argparse.ArgumentParser(
        description="敵対的摂動ジェネレータの学習",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 設定ファイル
    parser.add_argument(
        "--config",
        type=str,
        default="config/generator_config.yaml",
        help="設定ファイルパス",
    )

    # データセット
    parser.add_argument("--train-dir", type=str, help="学習画像ディレクトリ")
    parser.add_argument("--val-dir", type=str, help="検証画像ディレクトリ")

    # 分類器パス
    parser.add_argument("--classifier-model", type=str, help="分類器モデルパス (.keras)")
    parser.add_argument("--projection-path", type=str, help="射影行列パス (.npy)")
    parser.add_argument("--nudenet-onnx", type=str, help="NudeNet ONNXパス")
    parser.add_argument("--backbone-cache", type=str, help="PyTorchバックボーンのキャッシュ")

    # 学習パラメータ
    parser.add_argument("--epochs", type=int, help="学習エポック数")
    parser.add_argument("--batch-size", type=int, help="バッチサイズ")
    parser.add_argument("--lr", type=float, help="学習率")
    parser.add_argument("--lambda-cls", type=float, help="分類損失の重み")
    parser.add_argument("--lambda-l2", type=float, help="L2摂動ペナルティの重み")
    parser.add_argument("--max-perturbation", type=float, help="最大摂動量 (L-inf)")

    # ジェネレータ設定
    parser.add_argument("--base-channels", type=int, help="UNetの基本チャネル数")
    parser.add_argument("--depth", type=int, help="UNetの段数")

    # 出力
    parser.add_argument("--checkpoint-dir", type=str, help="チェックポイント保存先")
    parser.add_argument("--log-dir", type=str, help="TensorBoardログ保存先")

    # その他
    parser.add_argument("--resume", type=str, help="チェックポイントから学習を再開")
    parser.add_argument("--seed", type=int, help="乱数シード")
    parser.add_argument("--device", type=str, help="使用デバイス (cpu/cuda)")

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """YAML設定ファイルを読み込む。"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def merge_config(config: dict, args: argparse.Namespace) -> dict:
    """コマンドライン引数で設定を上書きする。CLIの値が優先される。"""
    # データセット
    if args.train_dir:
        config.setdefault("dataset", {})["train_dir"] = args.train_dir
    if args.val_dir:
        config.setdefault("dataset", {})["val_dir"] = args.val_dir

    # 分類器パス
    if args.classifier_model:
        config.setdefault("pipeline", {})["classifier_path"] = args.classifier_model
    if args.projection_path:
        config.setdefault("pipeline", {})["projection_path"] = args.projection_path
    if args.nudenet_onnx:
        config.setdefault("pipeline", {})["backbone_onnx_path"] = args.nudenet_onnx
    if args.backbone_cache:
        config.setdefault("pipeline", {})["backbone_cache_dir"] = args.backbone_cache

    # 学習パラメータ
    training = config.setdefault("training", {})
    if args.epochs is not None:
        training["epochs"] = args.epochs
    if args.batch_size is not None:
        training["batch_size"] = args.batch_size
    if args.lr is not None:
        training["learning_rate"] = args.lr
    if args.lambda_cls is not None:
        training["lambda_cls"] = args.lambda_cls
    if args.lambda_l2 is not None:
        training["lambda_l2"] = args.lambda_l2

    # ジェネレータ設定
    gen = config.setdefault("generator", {})
    if args.max_perturbation is not None:
        gen["max_perturbation"] = args.max_perturbation
    if args.base_channels is not None:
        gen["base_channels"] = args.base_channels
    if args.depth is not None:
        gen["depth"] = args.depth

    # 出力
    if args.checkpoint_dir:
        config.setdefault("checkpoint", {})["save_dir"] = args.checkpoint_dir
    if args.log_dir:
        config.setdefault("tensorboard", {})["log_dir"] = args.log_dir
    if args.seed is not None:
        config["seed"] = args.seed

    return config


# ── 学習率スケジューラ ──────────────────────────────────────────────


def build_scheduler(
    optimizer: torch.optim.Optimizer, config: dict, steps_per_epoch: int
) -> torch.optim.lr_scheduler._LRScheduler | None:
    """学習率スケジューラを構築する。

    ウォームアップ付きのコサイン/ステップスケジューラをサポート。
    """
    sched_config = config.get("training", {}).get("scheduler", {})
    sched_type = sched_config.get("type", "cosine")

    if sched_type == "none":
        return None

    epochs = config.get("training", {}).get("epochs", 100)
    warmup_epochs = sched_config.get("warmup_epochs", 5)
    min_lr = sched_config.get("min_lr", 1e-5)
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    if sched_type == "cosine":
        # ウォームアップ + コサインアニーリング
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                # 線形ウォームアップ
                return max(step / max(warmup_steps, 1), 0.01)
            # コサインアニーリング
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            base_lr = config.get("training", {}).get("learning_rate", 2e-4)
            cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
            return max(cosine_factor, min_lr / base_lr)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif sched_type == "step":
        # ステップ減衰（30エポックごとに0.5倍）
        step_size = sched_config.get("step_size", 30) * steps_per_epoch
        gamma = sched_config.get("gamma", 0.5)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    else:
        logger.warning("未知のスケジューラタイプ: %s。スケジューラなしで実行します。", sched_type)
        return None

    logger.info(
        "スケジューラ: %s (ウォームアップ=%dエポック, 最小LR=%.2e)",
        sched_type,
        warmup_epochs,
        min_lr,
    )
    return scheduler


# ── 損失関数 ────────────────────────────────────────────────────────


class AdversarialLoss(nn.Module):
    """敵対的摂動ジェネレータの損失関数。

    L_total = λ_cls * L_cls + λ_l2 * L_l2

    L_cls: 二値交差エントロピー損失
           分類器の出力をターゲット確率（SFW=0）に近づける
    L_l2:  摂動の L2 ノルム
           元画像との差分を最小化し知覚品質を維持する
    """

    def __init__(
        self,
        lambda_cls: float = 1.0,
        lambda_l2: float = 10.0,
        target_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_l2 = lambda_l2
        self.target_prob = target_prob

    def forward(
        self,
        nsfw_prob: torch.Tensor,
        original: torch.Tensor,
        perturbed: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """損失を計算する。

        Args:
            nsfw_prob: 分類器の出力 (N, 1) [0, 1]。
            original: 元画像 (N, 3, H, W) [0, 1]。
            perturbed: 摂動済み画像 (N, 3, H, W) [0, 1]。

        Returns:
            (total_loss, loss_dict) のタプル。
            loss_dict: 各損失の値を辞書で返す（ロギング用）。
        """
        batch_size = nsfw_prob.shape[0]

        # === 分類損失 (BCE) ===
        # ターゲット: 全サンプルをSFW (target_prob) に分類させる
        target = torch.full_like(nsfw_prob, self.target_prob)
        # 数値安定性のためクリッピング
        nsfw_prob_clamped = torch.clamp(nsfw_prob, 1e-7, 1 - 1e-7)
        loss_cls = nn.functional.binary_cross_entropy(
            nsfw_prob_clamped, target
        )

        # === L2 摂動ペナルティ ===
        # ピクセル単位の差分の L2 ノルム（バッチ平均）
        diff = perturbed - original
        loss_l2 = torch.mean(torch.norm(diff.view(batch_size, -1), p=2, dim=1))

        # === 合計損失 ===
        total_loss = self.lambda_cls * loss_cls + self.lambda_l2 * loss_l2

        # ロギング用の辞書
        loss_dict = {
            "loss_total": total_loss.item(),
            "loss_cls": loss_cls.item(),
            "loss_l2": loss_l2.item(),
        }

        return total_loss, loss_dict


# ── 評価 ────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate(
    generator: nn.Module,
    classifier: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: AdversarialLoss,
    device: torch.device,
    thresholds: list[float] | None = None,
) -> dict[str, float]:
    """検証データセットでジェネレータを評価する。

    Args:
        generator: ジェネレータモデル。
        classifier: 凍結済み分類器。
        dataloader: 検証用DataLoader。
        loss_fn: 損失関数。
        device: 使用デバイス。
        thresholds: 成功率を計算する閾値一覧。

    Returns:
        評価メトリクスの辞書。
    """
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5]

    generator.eval()

    total_loss = 0.0
    total_loss_cls = 0.0
    total_loss_l2 = 0.0
    total_samples = 0

    all_orig_probs = []
    all_adv_probs = []
    all_l2_norms = []
    all_linf_norms = []

    for images, _paths in dataloader:
        images = images.to(device)
        batch_size = images.shape[0]

        # ジェネレータで摂動済み画像を生成
        perturbed = generator(images)

        # 分類器で判定
        orig_probs = classifier(images)
        adv_probs = classifier(perturbed)

        # 損失計算
        loss, loss_dict = loss_fn(adv_probs, images, perturbed)

        total_loss += loss_dict["loss_total"] * batch_size
        total_loss_cls += loss_dict["loss_cls"] * batch_size
        total_loss_l2 += loss_dict["loss_l2"] * batch_size
        total_samples += batch_size

        # メトリクス収集
        all_orig_probs.append(orig_probs.cpu().numpy().flatten())
        all_adv_probs.append(adv_probs.cpu().numpy().flatten())

        # 摂動のノルム
        diff = (perturbed - images).view(batch_size, -1)
        all_l2_norms.append(torch.norm(diff, p=2, dim=1).cpu().numpy())
        all_linf_norms.append(
            torch.max(torch.abs(diff), dim=1)[0].cpu().numpy()
        )

    generator.train()

    # メトリクスを集約
    orig_probs = np.concatenate(all_orig_probs)
    adv_probs = np.concatenate(all_adv_probs)
    l2_norms = np.concatenate(all_l2_norms)
    linf_norms = np.concatenate(all_linf_norms)

    # NSFWサンプルのみを対象にした成功率
    nsfw_mask = orig_probs >= 0.5
    nsfw_count = nsfw_mask.sum()

    metrics = {
        "val_loss": total_loss / total_samples,
        "val_loss_cls": total_loss_cls / total_samples,
        "val_loss_l2": total_loss_l2 / total_samples,
        "val_total_samples": total_samples,
        "val_nsfw_samples": int(nsfw_count),
        "val_avg_orig_prob": float(orig_probs.mean()),
        "val_avg_adv_prob": float(adv_probs.mean()),
        "val_avg_prob_reduction": float((orig_probs - adv_probs).mean()),
        "val_avg_l2_norm": float(l2_norms.mean()),
        "val_avg_linf_norm": float(linf_norms.mean()),
    }

    # 各閾値での成功率（NSFWサンプル中、摂動後にSFWと判定された割合）
    for th in thresholds:
        if nsfw_count > 0:
            success = (adv_probs[nsfw_mask] < th).sum()
            metrics[f"val_success_rate_{th}"] = float(success / nsfw_count)
        else:
            metrics[f"val_success_rate_{th}"] = 0.0

    return metrics


# ── TensorBoard可視化 ───────────────────────────────────────────────


def log_sample_images(
    writer: SummaryWriter,
    generator: nn.Module,
    images: torch.Tensor,
    classifier: nn.Module,
    epoch: int,
    num_samples: int = 8,
) -> None:
    """サンプル画像をTensorBoardに記録する。

    元画像、摂動済み画像、摂動マップ（増幅済み）の3列を並べて表示。
    """
    generator.eval()
    with torch.no_grad():
        samples = images[:num_samples]
        perturbed = generator(samples)
        perturbation = perturbed - samples

        # 元画像の分類確率
        orig_probs = classifier(samples).cpu().numpy().flatten()
        adv_probs = classifier(perturbed).cpu().numpy().flatten()

        # 摂動の可視化（10倍に増幅して0.5を中心にシフト）
        noise_vis = perturbation * 10 + 0.5
        noise_vis = torch.clamp(noise_vis, 0, 1)

        # 3行を縦に結合: [元画像, 摂動済み, 摂動マップ]
        combined = torch.cat([samples, perturbed, noise_vis], dim=0)

        # TensorBoardに記録
        from torchvision.utils import make_grid

        grid = make_grid(combined, nrow=num_samples, padding=2, normalize=False)
        writer.add_image("samples/orig_perturbed_noise", grid, epoch)

        # 各サンプルの確率をテキストで記録
        text_lines = []
        for i in range(min(num_samples, len(orig_probs))):
            text_lines.append(
                f"Sample {i}: {orig_probs[i]:.3f} -> {adv_probs[i]:.3f}"
            )
        writer.add_text("samples/probabilities", "  \n".join(text_lines), epoch)

    generator.train()


# ── 学習ループ ──────────────────────────────────────────────────────


def train(config: dict, device_name: str | None = None) -> str:
    """ジェネレータの学習を実行する。

    Args:
        config: 設定辞書（YAML由来）。
        device_name: 使用デバイス名。Noneの場合は自動検出。

    Returns:
        最良モデルのチェックポイントパス。
    """
    # ── デバイス設定 ──
    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("使用デバイス: %s", device)

    # ── シード設定 ──
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    logger.info("乱数シード: %d", seed)

    # ── 設定値の取得 ──
    pipeline_cfg = config.get("pipeline", {})
    dataset_cfg = config.get("dataset", {})
    gen_cfg = config.get("generator", {})
    train_cfg = config.get("training", {})
    ckpt_cfg = config.get("checkpoint", {})
    tb_cfg = config.get("tensorboard", {})
    eval_cfg = config.get("evaluation", {})

    # ── 凍結済み分類器の構築 ──
    logger.info("凍結済み分類器を構築中 ...")
    classifier = build_frozen_classifier(
        nudenet_onnx_path=pipeline_cfg.get("backbone_onnx_path"),
        projection_path=pipeline_cfg.get("projection_path", "models/nudenet_projection.npy"),
        classifier_path=pipeline_cfg.get(
            "classifier_path", "models/target_classifier/pnsfwmedia_classifier.keras"
        ),
        backbone_cache_dir=pipeline_cfg.get("backbone_cache_dir", "models/torch_backbone"),
        device=str(device),
    )
    # 勾配が分類器を通じて逆伝播できるようにeval()のまま維持
    classifier.eval()

    # ── DataLoader の作成 ──
    train_loader, val_loader = create_dataloaders(
        train_dir=dataset_cfg.get("train_dir", "dataset/nsfw_images/train"),
        val_dir=dataset_cfg.get("val_dir"),
        image_size=pipeline_cfg.get("image_size", 320),
        batch_size=train_cfg.get("batch_size", 8),
        num_workers=dataset_cfg.get("num_workers", 4),
        pin_memory=dataset_cfg.get("pin_memory", True),
    )

    # ── ジェネレータモデルの構築 ──
    generator = AdversarialGenerator(
        in_channels=3,
        base_channels=gen_cfg.get("base_channels", 32),
        depth=gen_cfg.get("depth", 4),
        max_perturbation=gen_cfg.get("max_perturbation", 16 / 255),
        use_attention=gen_cfg.get("use_attention", True),
    ).to(device)

    total_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    logger.info("ジェネレータパラメータ数: %s", f"{total_params:,}")

    # ── 損失関数 ──
    loss_fn = AdversarialLoss(
        lambda_cls=train_cfg.get("lambda_cls", 1.0),
        lambda_l2=train_cfg.get("lambda_l2", 10.0),
        target_prob=train_cfg.get("target_prob", 0.0),
    )

    # ── オプティマイザ ──
    optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=train_cfg.get("learning_rate", 2e-4),
        betas=tuple(train_cfg.get("betas", [0.5, 0.999])),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )

    # ── スケジューラ ──
    scheduler = build_scheduler(optimizer, config, len(train_loader))

    # ── チェックポイントから復元 ──
    start_epoch = 0
    best_metric = 0.0
    resume_path = config.get("_resume_path")
    if resume_path and os.path.exists(resume_path):
        logger.info("チェックポイントから復元: %s", resume_path)
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        generator.load_state_dict(ckpt["generator_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_metric = ckpt.get("best_metric", 0.0)
        if scheduler and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        logger.info("エポック %d から学習を再開", start_epoch)

    # ── TensorBoard ──
    writer = None
    if tb_cfg.get("enabled", True):
        log_dir = tb_cfg.get("log_dir", "logs/generator")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(log_dir, timestamp)
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        logger.info("TensorBoardログ: %s", log_dir)

    # ── チェックポイントディレクトリ ──
    ckpt_dir = ckpt_cfg.get("save_dir", "checkpoints/generator")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── 学習設定 ──
    epochs = train_cfg.get("epochs", 100)
    grad_clip_norm = train_cfg.get("grad_clip_norm", 1.0)
    eval_interval = eval_cfg.get("eval_interval", 1)
    save_interval = ckpt_cfg.get("save_interval", 5)
    save_best = ckpt_cfg.get("save_best", True)
    thresholds = eval_cfg.get("thresholds", [0.3, 0.4, 0.5])
    num_visualize = eval_cfg.get("num_visualize", 8)

    # 早期停止の設定
    es_cfg = train_cfg.get("early_stopping", {})
    es_enabled = es_cfg.get("enabled", True)
    es_patience = es_cfg.get("patience", 15)
    es_metric = es_cfg.get("metric", "val_success_rate")
    no_improve_count = 0

    # 可視化用の固定サンプルを取得
    vis_samples = None
    for images, _paths in train_loader:
        vis_samples = images[:num_visualize].to(device)
        break

    best_ckpt_path = os.path.join(ckpt_dir, "best_generator.pt")

    logger.info("=" * 60)
    logger.info("学習開始: %d エポック, バッチサイズ=%d", epochs, train_cfg.get("batch_size", 8))
    logger.info(
        "損失の重み: λ_cls=%.2f, λ_l2=%.2f",
        train_cfg.get("lambda_cls", 1.0),
        train_cfg.get("lambda_l2", 10.0),
    )
    logger.info("最大摂動量 (L-inf): %.4f (≈ %.1f/255)", gen_cfg.get("max_perturbation", 16 / 255), gen_cfg.get("max_perturbation", 16 / 255) * 255)
    logger.info("=" * 60)

    # ── メイン学習ループ ──
    global_step = start_epoch * len(train_loader)

    for epoch in range(start_epoch, epochs):
        generator.train()

        epoch_loss = 0.0
        epoch_loss_cls = 0.0
        epoch_loss_l2 = 0.0
        epoch_samples = 0
        epoch_success = 0  # 攻撃成功数（NSFW -> SFW）

        # tqdm プログレスバー（行を上書き更新、改行なし）
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1:3d}/{epochs}",
            leave=False,       # エポック完了後に行を消去して上書き
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        )

        for batch_idx, (images, _paths) in enumerate(pbar):
            images = images.to(device)
            batch_size = images.shape[0]

            # === 順伝播 ===
            # ジェネレータで摂動済み画像を生成
            perturbed = generator(images)

            # 凍結済み分類器で判定（勾配はジェネレータの出力を通じて逆伝播）
            nsfw_probs = classifier(perturbed)

            # 損失計算
            loss, loss_dict = loss_fn(nsfw_probs, images, perturbed)

            # === 逆伝播 ===
            optimizer.zero_grad()
            loss.backward()

            # 勾配クリッピング
            if grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(generator.parameters(), grad_clip_norm)

            optimizer.step()

            # スケジューラ更新（ステップ単位）
            if scheduler is not None:
                scheduler.step()

            global_step += 1

            # === メトリクス集計 ===
            epoch_loss += loss_dict["loss_total"] * batch_size
            epoch_loss_cls += loss_dict["loss_cls"] * batch_size
            epoch_loss_l2 += loss_dict["loss_l2"] * batch_size
            epoch_samples += batch_size

            # 攻撃成功判定
            with torch.no_grad():
                orig_probs = classifier(images)
                nsfw_mask = orig_probs.flatten() >= 0.5
                if nsfw_mask.any():
                    epoch_success += (
                        (nsfw_probs.flatten()[nsfw_mask] < 0.5).sum().item()
                    )

            # プログレスバー更新
            avg_prob = nsfw_probs.mean().item()
            pbar.set_postfix(
                loss=f"{loss_dict['loss_total']:.4f}",
                cls=f"{loss_dict['loss_cls']:.4f}",
                l2=f"{loss_dict['loss_l2']:.4f}",
                prob=f"{avg_prob:.3f}",
                ordered=True,
            )

            # TensorBoard: バッチ単位のロギング
            if writer and batch_idx % 10 == 0:
                writer.add_scalar("batch/loss_total", loss_dict["loss_total"], global_step)
                writer.add_scalar("batch/loss_cls", loss_dict["loss_cls"], global_step)
                writer.add_scalar("batch/loss_l2", loss_dict["loss_l2"], global_step)
                writer.add_scalar("batch/avg_nsfw_prob", avg_prob, global_step)
                writer.add_scalar(
                    "batch/learning_rate",
                    optimizer.param_groups[0]["lr"],
                    global_step,
                )

        pbar.close()

        # ── エポック集計 ──
        avg_loss = epoch_loss / max(epoch_samples, 1)
        avg_loss_cls = epoch_loss_cls / max(epoch_samples, 1)
        avg_loss_l2 = epoch_loss_l2 / max(epoch_samples, 1)

        # エポック結果を1行で表示（上書き形式）
        epoch_msg = (
            f"Epoch {epoch + 1:3d}/{epochs} | "
            f"loss={avg_loss:.4f} (cls={avg_loss_cls:.4f}, l2={avg_loss_l2:.4f}) | "
            f"LR={optimizer.param_groups[0]['lr']:.2e}"
        )

        # TensorBoard: エポック単位のロギング
        if writer:
            writer.add_scalar("epoch/loss_total", avg_loss, epoch)
            writer.add_scalar("epoch/loss_cls", avg_loss_cls, epoch)
            writer.add_scalar("epoch/loss_l2", avg_loss_l2, epoch)
            writer.add_scalar(
                "epoch/learning_rate", optimizer.param_groups[0]["lr"], epoch
            )

        # ── 検証 ──
        val_metrics = None
        if val_loader and (epoch + 1) % eval_interval == 0:
            val_metrics = evaluate(
                generator=generator,
                classifier=classifier,
                dataloader=val_loader,
                loss_fn=loss_fn,
                device=device,
                thresholds=thresholds,
            )

            sr_05 = val_metrics.get("val_success_rate_0.5", 0.0)
            epoch_msg += (
                f" | val_loss={val_metrics['val_loss']:.4f}"
                f" | SR@0.5={sr_05:.1%}"
                f" | avg_prob={val_metrics['val_avg_adv_prob']:.3f}"
            )

            # TensorBoard: 検証メトリクス
            if writer:
                for key, value in val_metrics.items():
                    writer.add_scalar(f"val/{key}", value, epoch)

        # サンプル画像の可視化
        if writer and vis_samples is not None and (epoch + 1) % eval_interval == 0:
            log_sample_images(
                writer, generator, vis_samples, classifier, epoch, num_visualize
            )

        logger.info(epoch_msg)

        # ── チェックポイント保存 ──
        ckpt_data = {
            "epoch": epoch,
            "generator_state_dict": generator.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric": best_metric,
            "config": config,
        }
        if scheduler:
            ckpt_data["scheduler_state_dict"] = scheduler.state_dict()

        # 定期保存
        if (epoch + 1) % save_interval == 0:
            path = os.path.join(ckpt_dir, f"generator_epoch{epoch + 1:04d}.pt")
            torch.save(ckpt_data, path)
            logger.info("チェックポイント保存: %s", path)

        # 最良モデルの保存
        if save_best and val_metrics:
            current_metric = val_metrics.get("val_success_rate_0.5", 0.0)
            if current_metric > best_metric:
                best_metric = current_metric
                ckpt_data["best_metric"] = best_metric
                torch.save(ckpt_data, best_ckpt_path)
                logger.info(
                    "最良モデルを更新: SR@0.5=%.1%% -> %s",
                    best_metric * 100,
                    best_ckpt_path,
                )
                no_improve_count = 0
            else:
                no_improve_count += 1
        elif not val_loader:
            # 検証なしの場合は損失ベースで保存
            if avg_loss < best_metric or best_metric == 0.0:
                best_metric = avg_loss
                ckpt_data["best_metric"] = best_metric
                torch.save(ckpt_data, best_ckpt_path)

        # ── 早期停止判定 ──
        if es_enabled and val_loader and no_improve_count >= es_patience:
            logger.info(
                "早期停止: %d エポック連続で改善なし (patience=%d)",
                no_improve_count,
                es_patience,
            )
            break

    # ── 学習完了 ──
    # 最終モデルの保存
    final_path = os.path.join(ckpt_dir, "generator_final.pt")
    torch.save(
        {
            "epoch": epoch,
            "generator_state_dict": generator.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric": best_metric,
            "config": config,
        },
        final_path,
    )
    logger.info("最終モデルを保存: %s", final_path)

    if writer:
        writer.close()

    logger.info("=" * 60)
    logger.info("学習完了!")
    logger.info("最良モデル: %s (SR@0.5=%.1%%)", best_ckpt_path, best_metric * 100)
    logger.info("最終モデル: %s", final_path)
    logger.info("=" * 60)

    return best_ckpt_path


# ── エントリポイント ────────────────────────────────────────────────


def main() -> None:
    """メインエントリポイント。"""
    # ロギング設定
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args()

    # 設定ファイルの読み込み
    config = {}
    if os.path.exists(args.config):
        config = load_config(args.config)
        logger.info("設定ファイルを読み込み: %s", args.config)
    else:
        logger.warning("設定ファイルが見つかりません: %s。デフォルト値で実行します。", args.config)

    # CLIの引数で上書き
    config = merge_config(config, args)

    # 学習再開パスの設定
    if args.resume:
        config["_resume_path"] = args.resume

    # デバイスの設定
    device_name = args.device

    # 学習実行
    best_path = train(config, device_name=device_name)
    logger.info("完了。最良モデル: %s", best_path)


if __name__ == "__main__":
    main()
