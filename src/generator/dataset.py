"""NSFW画像データセット。

指定ディレクトリから画像を読み込み、PyTorch DataLoader で使用可能な
Dataset クラスを提供する。

画像の前処理:
    1. 指定サイズ (320x320) にリサイズ
    2. [0, 255] -> [0, 1] に正規化
    3. HWC -> CHW に変換（PyTorchの入力形式）
    4. 学習時はランダムな水平反転でデータ拡張
"""

import logging
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# サポートする画像拡張子
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class NSFWImageDataset(Dataset):
    """NSFW画像データセット。

    指定ディレクトリから画像ファイルを検索し、
    リサイズ・正規化した画像テンソルを返す。

    Args:
        image_dir: 画像ファイルが格納されたディレクトリパス。
        image_size: リサイズ先の画像サイズ（正方形）。
        extensions: 読み込む画像の拡張子一覧。
        augment: データ拡張を適用するか（学習時はTrue）。
    """

    def __init__(
        self,
        image_dir: str,
        image_size: int = 320,
        extensions: set[str] | None = None,
        augment: bool = False,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.augment = augment

        if extensions is None:
            extensions = SUPPORTED_EXTENSIONS

        # ディレクトリ内の画像ファイルパスを収集
        self.image_paths = self._collect_image_paths(image_dir, extensions)
        logger.info(
            "データセット初期化: %d 枚の画像を %s から読み込み",
            len(self.image_paths),
            image_dir,
        )

    @staticmethod
    def _collect_image_paths(
        image_dir: str, extensions: set[str]
    ) -> list[str]:
        """ディレクトリから画像ファイルのパスを再帰的に収集する。"""
        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"画像ディレクトリが見つかりません: {image_dir}")

        paths = []
        for root, _dirs, files in os.walk(image_dir):
            for fname in sorted(files):
                if Path(fname).suffix.lower() in extensions:
                    paths.append(os.path.join(root, fname))

        if not paths:
            raise FileNotFoundError(
                f"画像ファイルが見つかりません: {image_dir} "
                f"(対応拡張子: {extensions})"
            )

        return paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        """1枚の画像を読み込んでテンソルに変換する。

        Args:
            idx: データセットのインデックス。

        Returns:
            (image_tensor, file_path) のタプル。
            image_tensor: (3, H, W) float32 [0, 1]
            file_path: 元画像のファイルパス
        """
        path = self.image_paths[idx]

        # OpenCVで画像を読み込み (BGR形式)
        img = cv2.imread(path)
        if img is None:
            logger.warning("画像の読み込みに失敗: %s (スキップ)", path)
            # 読み込み失敗時は黒画像を返す
            return (
                torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32),
                path,
            )

        # BGR -> RGB変換
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # リサイズ
        img = cv2.resize(
            img,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )

        # データ拡張（学習時のみ）
        if self.augment:
            # ランダム水平反転
            if np.random.random() > 0.5:
                img = np.fliplr(img).copy()

        # [0, 255] -> [0, 1] に正規化
        img = img.astype(np.float32) / 255.0

        # HWC -> CHW に変換
        img = np.transpose(img, (2, 0, 1))

        # PyTorch テンソルに変換
        tensor = torch.from_numpy(img)

        return tensor, path


def create_dataloaders(
    data_dir: str,
    image_size: int = 320,
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """学習用・検証用 DataLoader を作成する。

    単一ディレクトリの画像を指定比率で train/val に自動分割する。
    分割はシード値で決定論的に行われるため、再実行でも同じ分割になる。

    Args:
        data_dir: 画像ファイルが格納されたディレクトリパス。
        image_size: リサイズ先の画像サイズ。
        batch_size: バッチサイズ。
        num_workers: データ読み込みのワーカー数。
        pin_memory: CUDAピンメモリを使用するか。
        val_ratio: 検証用データの割合（デフォルト: 0.15 = 15%）。
        seed: 分割のシード値（再現性のため）。

    Returns:
        (train_loader, val_loader) のタプル。
    """
    # 全画像パスを収集
    all_paths = NSFWImageDataset._collect_image_paths(data_dir, SUPPORTED_EXTENSIONS)
    total = len(all_paths)

    # シード固定で決定論的にシャッフル→分割
    rng = np.random.default_rng(seed)
    indices = np.arange(total)
    rng.shuffle(indices)

    val_count = max(1, int(total * val_ratio))
    val_indices = set(indices[:val_count].tolist())
    train_paths = [p for i, p in enumerate(all_paths) if i not in val_indices]
    val_paths = [p for i, p in enumerate(all_paths) if i in val_indices]

    logger.info(
        "データ分割: 全 %d 枚 → 学習 %d 枚 (%.0f%%) / 検証 %d 枚 (%.0f%%)",
        total, len(train_paths), (1 - val_ratio) * 100,
        len(val_paths), val_ratio * 100,
    )

    # 学習用データセット（データ拡張あり）
    train_dataset = NSFWImageDataset(
        image_dir=data_dir, image_size=image_size, augment=True,
    )
    train_dataset.image_paths = train_paths

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    logger.info("学習用DataLoader: %d 枚, バッチサイズ=%d", len(train_dataset), batch_size)

    # 検証用データセット（データ拡張なし）
    val_dataset = NSFWImageDataset(
        image_dir=data_dir, image_size=image_size, augment=False,
    )
    val_dataset.image_paths = val_paths

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    logger.info("検証用DataLoader: %d 枚, バッチサイズ=%d", len(val_dataset), batch_size)

    return train_loader, val_loader
