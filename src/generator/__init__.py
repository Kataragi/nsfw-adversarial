"""敵対的摂動ジェネレータモジュール。

PyTorchベースのUNetジェネレータを学習し、
NSFW画像に知覚的に微小な摂動を加えて分類器をSFWと誤判定させる。
"""

from src.generator.classifier_wrapper import (
    FrozenNSFWClassifier,
    build_frozen_classifier,
)
from src.generator.dataset import NSFWImageDataset, create_dataloaders
from src.generator.model import AdversarialGenerator

__all__ = [
    "AdversarialGenerator",
    "FrozenNSFWClassifier",
    "NSFWImageDataset",
    "build_frozen_classifier",
    "create_dataloaders",
]
