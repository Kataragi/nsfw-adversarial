"""pNSFWMedia分類器のPyTorchラッパー。

TensorFlowで構築された既存パイプライン（NudeNetバックボーン + 射影 + 分類器）を
PyTorchのnn.Moduleとしてラップし、ジェネレータ学習時に勾配が逆伝播できるようにする。

方式:
    1. NudeNetバックボーン (ONNX) -> onnx2torch で PyTorch化
    2. 射影行列 (numpy) -> nn.Linear として読み込み
    3. pNSFWMedia分類器 (Keras) -> 重みを抽出し PyTorch MLP に変換
    4. 全パラメータを凍結 (requires_grad=False)

パイプライン:
    画像 (N, 3, 320, 320) [0, 1]
    -> NudeNet backbone (PyTorch)
    -> Global Average Pooling
    -> 直交射影 (C -> 256)
    -> L2正規化
    -> pNSFWMedia MLP分類器
    -> NSFW確率 (N, 1)
"""

import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

IMAGE_SIZE = 320


def find_nudenet_onnx() -> str:
    """NudeNet ONNXモデルファイルの場所を特定する。"""
    try:
        import nudenet

        model_path = Path(nudenet.__file__).parent / "best.onnx"
        if model_path.exists():
            return str(model_path)
    except ImportError:
        pass

    candidates = [
        Path.home() / ".nudenet" / "best.onnx",
        Path("models") / "best.onnx",
    ]
    for c in candidates:
        if c.exists():
            return str(c)

    raise FileNotFoundError(
        "NudeNet ONNXモデルが見つかりません。nudenetをインストールしてください: pip install nudenet"
    )


def _find_backbone_node(onnx_model) -> str:
    """YOLOv8 ONNXモデルからバックボーンの特徴ノードを探す。"""
    consumers: dict[str, list[str]] = {}
    for node in onnx_model.graph.node:
        for inp in node.input:
            consumers.setdefault(inp, []).append(node.op_type)

    candidates = []
    for node in onnx_model.graph.node:
        if node.op_type == "Conv" and node.output:
            out = node.output[0]
            if out in consumers and "Concat" in consumers[out]:
                candidates.append(out)

    if candidates:
        return candidates[-1]

    for node in reversed(onnx_model.graph.node):
        if node.op_type == "Conv" and node.output:
            return node.output[0]

    raise RuntimeError("ONNXモデル内にバックボーン特徴ノードが見つかりませんでした")


def _extract_backbone_onnx(onnx_path: str, output_path: str) -> str:
    """NudeNet ONNXモデルからバックボーンサブグラフを抽出する。"""
    import onnx

    model = onnx.load(onnx_path)
    input_name = model.graph.input[0].name
    backbone_node = _find_backbone_node(model)

    logger.info("バックボーン特徴ノード: %s", backbone_node)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    onnx.utils.extract_model(
        onnx_path,
        output_path,
        input_names=[input_name],
        output_names=[backbone_node],
    )
    logger.info("バックボーンONNXを保存: %s", output_path)
    return output_path


def _get_backbone_feature_dim(onnx_path: str, node_name: str) -> tuple[int, ...]:
    """ダミー入力でバックボーンの出力形状を取得する。"""
    import onnx
    import onnxruntime as ort

    model = onnx.load(onnx_path)
    intermediate = onnx.helper.make_tensor_value_info(
        node_name, onnx.TensorProto.FLOAT, None
    )
    model.graph.output.append(intermediate)
    modified_bytes = model.SerializeToString()

    sess = ort.InferenceSession(
        modified_bytes, providers=["CPUExecutionProvider"]
    )
    input_info = sess.get_inputs()[0]
    h = input_info.shape[2] if isinstance(input_info.shape[2], int) else IMAGE_SIZE
    w = input_info.shape[3] if isinstance(input_info.shape[3], int) else IMAGE_SIZE
    dummy = np.zeros((1, 3, h, w), dtype=np.float32)

    out = sess.run([node_name], {input_info.name: dummy})[0]
    return out.shape  # (1, C, H, W)


def convert_backbone_to_pytorch(
    nudenet_onnx_path: str | None = None,
    cache_dir: str = "models/torch_backbone",
) -> tuple[nn.Module, int]:
    """NudeNetバックボーンをONNXからPyTorchモデルに変換する。

    Args:
        nudenet_onnx_path: NudeNet best.onnx のパス。Noneの場合は自動検出。
        cache_dir: 変換済みモデルのキャッシュディレクトリ。

    Returns:
        (backbone_module, feature_channels) のタプル。
    """
    import onnx

    # onnx2torch はインポート時にのみ必要
    from onnx2torch import convert

    os.makedirs(cache_dir, exist_ok=True)
    meta_path = os.path.join(cache_dir, "backbone_meta.json")
    pt_path = os.path.join(cache_dir, "backbone.pt")

    # キャッシュが存在する場合はそれを使用
    if os.path.exists(pt_path) and os.path.exists(meta_path):
        logger.info("キャッシュ済みPyTorchバックボーンを使用: %s", cache_dir)
        with open(meta_path, "r") as f:
            meta = json.load(f)
        backbone = torch.load(pt_path, map_location="cpu", weights_only=False)
        backbone.eval()
        return backbone, meta["feature_channels"]

    if nudenet_onnx_path is None:
        nudenet_onnx_path = find_nudenet_onnx()

    # バックボーンノードと特徴次元を取得
    onnx_model = onnx.load(nudenet_onnx_path)
    backbone_node = _find_backbone_node(onnx_model)
    feat_shape = _get_backbone_feature_dim(nudenet_onnx_path, backbone_node)
    feature_channels = int(feat_shape[1])
    logger.info(
        "バックボーン特徴形状 (NCHW): %s  チャネル数=%d",
        feat_shape,
        feature_channels,
    )

    # バックボーンONNXを抽出
    backbone_onnx = os.path.join(cache_dir, "backbone.onnx")
    _extract_backbone_onnx(nudenet_onnx_path, backbone_onnx)

    # ONNX -> PyTorch変換
    logger.info("バックボーンONNX -> PyTorch変換中 ...")
    onnx_model = onnx.load(backbone_onnx)
    backbone = convert(onnx_model)
    backbone.eval()

    # キャッシュに保存
    torch.save(backbone, pt_path)
    meta = {
        "backbone_node": backbone_node,
        "feature_channels": feature_channels,
        "feature_shape_nchw": list(feat_shape),
        "image_size": IMAGE_SIZE,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("バックボーン変換完了 -> %s", cache_dir)
    return backbone, feature_channels


def _load_keras_classifier_weights(
    classifier_path: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Keras分類器からDense層の重みとバイアスを抽出する。

    Args:
        classifier_path: .keras ファイルのパス。

    Returns:
        [(weight, bias), ...] のリスト。各weightは (in, out) 形状。
    """
    # TensorFlowのインポートはここでのみ必要
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")

    classifier = tf.keras.models.load_model(classifier_path)
    layers = []

    for layer in classifier.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            w, b = layer.get_weights()
            layers.append((w, b))
            logger.info(
                "  Dense層を抽出: 入力=%d, 出力=%d, 活性化=%s",
                w.shape[0],
                w.shape[1],
                layer.get_config().get("activation", "none"),
            )

    if not layers:
        raise ValueError(
            f"分類器 {classifier_path} にDense層が見つかりませんでした"
        )

    return layers


class PyTorchClassifierMLP(nn.Module):
    """pNSFWMedia分類器のPyTorch再実装。

    Kerasモデルの重みをPyTorchのLinear層に読み込む。
    典型的な構造: Dense(256->128, ReLU) -> Dense(128->1, Sigmoid)
    """

    def __init__(self, keras_weights: list[tuple[np.ndarray, np.ndarray]]) -> None:
        super().__init__()
        layers = []
        for i, (w, b) in enumerate(keras_weights):
            linear = nn.Linear(w.shape[0], w.shape[1])
            # Kerasは(in, out)形状、PyTorchは(out, in)形状
            linear.weight.data = torch.from_numpy(w.T.copy())
            linear.bias.data = torch.from_numpy(b.copy())
            layers.append(linear)

            # 最終層以外はReLUを追加（Keras分類器の一般的な構造）
            if i < len(keras_weights) - 1:
                layers.append(nn.ReLU())

        # 最終層にSigmoidを追加（NSFW確率出力）
        layers.append(nn.Sigmoid())

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class FrozenNSFWClassifier(nn.Module):
    """pNSFWMedia分類パイプライン全体のPyTorchラッパー（凍結済み）。

    パイプライン:
        入力画像 (N, 3, 320, 320) [0, 1]
        -> NudeNet backbone
        -> Global Average Pooling (空間軸を平均)
        -> 直交射影 (C -> 256)
        -> L2正規化
        -> MLP分類器
        -> NSFW確率 (N, 1)

    全パラメータは凍結され、学習時に更新されない。
    """

    def __init__(
        self,
        backbone: nn.Module,
        projection_matrix: np.ndarray,
        classifier_mlp: nn.Module,
        feature_channels: int,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier_mlp = classifier_mlp
        self.feature_channels = feature_channels

        # 射影行列を nn.Linear として登録（バイアスなし）
        proj_in, proj_out = projection_matrix.shape
        self.projection = nn.Linear(proj_in, proj_out, bias=False)
        self.projection.weight.data = torch.from_numpy(
            projection_matrix.T.copy().astype(np.float32)
        )

        # 全パラメータを凍結
        self._freeze_all()

    def _freeze_all(self) -> None:
        """全パラメータの勾配計算を無効化する。"""
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def train(self, mode: bool = True) -> "FrozenNSFWClassifier":
        """常にeval()モードを維持する（凍結モデルのため）。"""
        # バックボーンとMLPは常にevalモード
        super().train(False)
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """順伝播。

        Args:
            images: (N, 3, 320, 320) float32 [0, 1]。

        Returns:
            NSFW確率 (N, 1)。
        """
        # バックボーン特徴抽出 (NCHW出力)
        features = self.backbone(images)

        # Global Average Pooling (空間軸: H, W)
        if features.dim() == 4:
            pooled = features.mean(dim=[2, 3])  # (N, C)
        else:
            pooled = features

        # 直交射影 (C -> 256)
        projected = self.projection(pooled)  # (N, 256)

        # L2正規化
        normalized = F.normalize(projected, p=2, dim=1)

        # 分類器でNSFW確率を出力
        output = self.classifier_mlp(normalized)  # (N, 1)
        return output


def build_frozen_classifier(
    nudenet_onnx_path: str | None = None,
    projection_path: str = "models/nudenet_projection.npy",
    classifier_path: str = "models/target_classifier/pnsfwmedia_classifier.keras",
    backbone_cache_dir: str = "models/torch_backbone",
    device: str = "cpu",
) -> FrozenNSFWClassifier:
    """凍結済み分類パイプラインを構築する。

    Args:
        nudenet_onnx_path: NudeNet best.onnx のパス（自動検出可）。
        projection_path: 射影行列 (.npy) のパス。
        classifier_path: pNSFWMedia分類器 (.keras) のパス。
        backbone_cache_dir: PyTorchバックボーンのキャッシュディレクトリ。
        device: 使用デバイス ("cpu" or "cuda")。

    Returns:
        凍結済み FrozenNSFWClassifier インスタンス。
    """
    logger.info("=== 凍結済み分類パイプラインを構築中 ===")

    # 1. バックボーンの変換・読み込み
    logger.info("[1/3] NudeNetバックボーンを読み込み中 ...")
    backbone, feature_channels = convert_backbone_to_pytorch(
        nudenet_onnx_path=nudenet_onnx_path,
        cache_dir=backbone_cache_dir,
    )

    # 2. 射影行列の読み込み
    logger.info("[2/3] 射影行列を読み込み中 ...")
    projection = np.load(projection_path).astype(np.float32)
    assert projection.shape[0] == feature_channels, (
        f"射影行列の第1次元 ({projection.shape[0]}) != "
        f"バックボーンチャネル数 ({feature_channels})"
    )
    logger.info("射影行列の形状: %s", projection.shape)

    # 3. Keras分類器の重みをPyTorchに変換
    logger.info("[3/3] pNSFWMedia分類器を読み込み中 ...")
    keras_weights = _load_keras_classifier_weights(classifier_path)
    classifier_mlp = PyTorchClassifierMLP(keras_weights)

    # パイプライン組み立て
    model = FrozenNSFWClassifier(
        backbone=backbone,
        projection_matrix=projection,
        classifier_mlp=classifier_mlp,
        feature_channels=feature_channels,
    )
    model = model.to(device)

    # ダミー入力で動作確認
    dummy = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    with torch.no_grad():
        out = model(dummy)
    logger.info("パイプライン検証完了 - ダミー出力形状: %s", out.shape)
    logger.info("=== 凍結済み分類パイプラインの構築完了 ===")

    return model
