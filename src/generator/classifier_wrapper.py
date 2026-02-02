"""pNSFWMedia分類器のPyTorchラッパー。

TensorFlowで構築された既存パイプライン（NudeNetバックボーン + 射影 + 分類器）を
PyTorchのnn.Moduleとしてラップし、ジェネレータ学習時に勾配が逆伝播できるようにする。

方式:
    1. NudeNetバックボーン (ONNX) -> onnx2torch で PyTorch化
    2. 射影行列 -> ファイルがあれば読み込み、なければ QR 分解 (seed=42) で再現生成
    3. pNSFWMedia分類器 (Keras) -> 重みを抽出し PyTorch MLP に変換
    4. 全パラメータを凍結 (requires_grad=False)

射影行列について:
    pNSFWMedia の extract_embeddings_nudenet.py で生成される直交射影行列は
    numpy.random.default_rng(seed=42) + QR 分解で決定論的に求まる。
    したがって nudenet_projection.npy が存在しなくても
    バックボーンのチャネル数さえ判明すれば同一の行列を再現できる。

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
    """NudeNet ONNXモデルファイルの場所を特定する。

    nudenet パッケージ内の .onnx ファイルを動的に検出する。
    best.onnx があればそちらを優先し、なければ見つかったものを使用する。
    これにより nudenet のバージョンでファイル名が変わっても動作する。
    """
    try:
        import nudenet

        nudenet_dir = Path(nudenet.__file__).parent
        # パッケージ内の .onnx を全て収集し、best.onnx があればそちらを優先
        onnx_files = sorted(nudenet_dir.glob("*.onnx"))
        if onnx_files:
            for f in onnx_files:
                if f.name == "best.onnx":
                    return str(f)
            # best.onnx がなければ最初に見つかったものを使用
            return str(onnx_files[0])
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
    """YOLOv8 ONNXモデルからバックボーンの特徴ノードを探す。

    ヒューリスティック:
        1. 優先: SPPF ブロックの出力 (MaxPool->Concat パターン)
        2. フォールバック: 最初の Conv->Concat 境界
        3. 最終: 最後の Conv 出力

    FPN/Neck 内部のノードを誤選択しないよう、最初の境界を使用する。
    """
    # --- 1. 優先: SPPF ブロックの出力を探す ---
    for node in onnx_model.graph.node:
        if node.op_type == "Concat":
            input_ops = set()
            for inp in node.input:
                for n2 in onnx_model.graph.node:
                    if n2.output and n2.output[0] == inp:
                        input_ops.add(n2.op_type)
            if "MaxPool" in input_ops and node.output:
                concat_out = node.output[0]
                for n2 in onnx_model.graph.node:
                    if n2.op_type == "Conv" and concat_out in n2.input and n2.output:
                        logger.info("SPPF 後の Conv ノードを検出: %s", n2.output[0])
                        return n2.output[0]
                logger.info("SPPF Concat ノードを検出: %s", concat_out)
                return concat_out

    # --- 2. フォールバック: 最初の Conv->Concat 境界 ---
    consumers: dict[str, list[str]] = {}
    for node in onnx_model.graph.node:
        for inp in node.input:
            consumers.setdefault(inp, []).append(node.op_type)

    for node in onnx_model.graph.node:
        if node.op_type == "Conv" and node.output:
            out = node.output[0]
            if out in consumers and "Concat" in consumers[out]:
                logger.info("Conv->Concat 境界ノードを検出: %s", out)
                return out

    # --- 3. 最終フォールバック: 最後の Conv 出力 ---
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


def _get_backbone_feature_dim_from_extracted(
    backbone_onnx_path: str,
) -> tuple[int, ...]:
    """抽出済みバックボーンONNXからダミー推論で出力形状を取得する。

    フルモデル上で中間ノードを取得する方式だと FPN の Concat ノードで
    形状不一致エラーが発生するため、先に抽出したサブグラフを使用する。
    """
    import onnxruntime as ort

    sess = ort.InferenceSession(
        backbone_onnx_path, providers=["CPUExecutionProvider"]
    )
    input_info = sess.get_inputs()[0]
    output_info = sess.get_outputs()[0]

    h = input_info.shape[2] if isinstance(input_info.shape[2], int) else IMAGE_SIZE
    w = input_info.shape[3] if isinstance(input_info.shape[3], int) else IMAGE_SIZE
    dummy = np.zeros((1, 3, h, w), dtype=np.float32)

    out = sess.run([output_info.name], {input_info.name: dummy})[0]
    return out.shape  # (1, C, H, W)


def create_orthogonal_projection(
    feature_dim: int, output_dim: int = 256, seed: int = 42
) -> np.ndarray:
    """pNSFWMedia と同一の直交射影行列を決定論的に生成する。

    pNSFWMedia/src/extract_embeddings_nudenet.py の
    _create_orthogonal_projection() と完全に同じアルゴリズム。
    QR 分解を用いるため seed さえ同じなら常に同一の行列が得られる。

    Args:
        feature_dim: バックボーン出力のチャネル数 (C)。
        output_dim: 射影先の次元数（デフォルト 256）。
        seed: 乱数シード（pNSFWMedia のデフォルトは 42）。

    Returns:
        (feature_dim, output_dim) 形状の float32 直交行列。
    """
    rng = np.random.default_rng(seed=seed)
    random_matrix = rng.standard_normal((feature_dim, output_dim))
    q, _ = np.linalg.qr(random_matrix)
    if q.shape[1] < output_dim:
        pad = rng.standard_normal((feature_dim, output_dim - q.shape[1]))
        q = np.hstack([q, pad])
    return q[:, :output_dim].astype(np.float32)


def load_or_create_projection(
    feature_channels: int,
    projection_path: str | None = None,
    output_dim: int = 256,
) -> np.ndarray:
    """射影行列をファイルから読み込むか、存在しなければ自動生成する。

    Args:
        feature_channels: バックボーン出力チャネル数。
        projection_path: .npy ファイルのパス。None または存在しない場合は自動生成。
        output_dim: 射影先の次元数。

    Returns:
        (feature_channels, output_dim) 形状の float32 行列。
    """
    if projection_path and os.path.exists(projection_path):
        projection = np.load(projection_path).astype(np.float32)
        if projection.shape[0] != feature_channels:
            raise ValueError(
                f"射影行列の第1次元 ({projection.shape[0]}) != "
                f"バックボーンチャネル数 ({feature_channels})"
            )
        logger.info("射影行列をファイルから読み込み: %s %s", projection_path, projection.shape)
        return projection

    # ファイルが存在しない場合は pNSFWMedia と同一アルゴリズムで生成
    logger.info(
        "射影行列ファイルが見つかりません。QR分解 (seed=42) で自動生成します "
        "(feature_dim=%d -> %d)",
        feature_channels,
        output_dim,
    )
    projection = create_orthogonal_projection(feature_channels, output_dim)

    # 生成した行列をキャッシュとして保存
    save_path = projection_path or "models/nudenet_projection.npy"
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    np.save(save_path, projection)
    logger.info("生成した射影行列を保存: %s", save_path)

    return projection


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

    # 1. バックボーンONNXを先に抽出（フルモデル推論の Concat エラーを回避）
    backbone_onnx = os.path.join(cache_dir, "backbone.onnx")
    _extract_backbone_onnx(nudenet_onnx_path, backbone_onnx)

    # 2. 抽出済みバックボーンから特徴次元を取得
    feat_shape = _get_backbone_feature_dim_from_extracted(backbone_onnx)
    feature_channels = int(feat_shape[1])
    logger.info(
        "バックボーン特徴形状 (NCHW): %s  チャネル数=%d",
        feat_shape,
        feature_channels,
    )

    # ONNX -> PyTorch変換
    logger.info("バックボーンONNX -> PyTorch変換中 ...")
    onnx_model = onnx.load(backbone_onnx)
    backbone = convert(onnx_model)
    backbone.eval()

    # キャッシュに保存
    torch.save(backbone, pt_path)
    onnx_full = onnx.load(nudenet_onnx_path)
    backbone_node = _find_backbone_node(onnx_full)
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


def _count_dense_layers_from_config(config_data: dict) -> int:
    """config.json から Dense 層の数を再帰的に数える。"""
    count = 0

    def _walk(obj: object) -> None:
        nonlocal count
        if isinstance(obj, dict):
            if obj.get("class_name") == "Dense":
                count += 1
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for v in obj:
                _walk(v)

    _walk(config_data)
    return count


def _load_keras_classifier_weights(
    classifier_path: str,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[str]]:
    """Keras分類器からDense層の重みとバイアスを抽出する。

    .keras ファイル（ZIP アーカイブ）内の model.weights.h5 を
    h5py で直接読み込み、TensorFlow に依存せずに重みを取得する。

    抽出戦略:
        1. config.json を解析して Dense 層の構造（層名、units、activation）を特定
        2. HDF5 から層名に基づいて直接 kernel/bias を取得
        3. 重複を排除し、config.json の順序に従って正確に抽出

    Args:
        classifier_path: .keras ファイルのパス。

    Returns:
        ([(weight, bias), ...], [activation, ...]) のタプル。
        各weightは (in, out) 形状。activations は各層の活性化関数名。
    """
    import tempfile
    import zipfile

    import h5py

    with zipfile.ZipFile(classifier_path, "r") as zf:
        names = zf.namelist()
        logger.info(".keras ZIP 内容: %s", names)

        # Step 1: config.json から Dense 層の構造を取得
        if "config.json" not in names:
            raise ValueError(f"{classifier_path} に config.json が含まれていません")

        with zf.open("config.json") as cf:
            config_data = json.loads(cf.read())

        # Sequential モデルの layers から Dense 層のみ抽出
        expected_dense_layers = []
        layers_config = config_data.get("config", {}).get("layers", [])

        for layer_config in layers_config:
            if layer_config.get("class_name") == "Dense":
                layer_name = layer_config.get("config", {}).get("name", "")
                units = layer_config.get("config", {}).get("units", 0)
                activation = layer_config.get("config", {}).get("activation", "linear")
                expected_dense_layers.append({
                    "name": layer_name,
                    "units": units,
                    "activation": activation,
                })

        expected_count = len(expected_dense_layers)
        logger.info("config.json から Dense=%d 層を検出", expected_count)
        for i, info in enumerate(expected_dense_layers):
            logger.info(
                "  Layer %d: name=%s, units=%d, activation=%s",
                i, info["name"], info["units"], info["activation"]
            )

        # Step 2: weights ファイルを探す
        weight_file = None
        for name in names:
            if name.endswith(".h5"):
                weight_file = name
                break
        if weight_file is None:
            raise ValueError(
                f"{classifier_path} 内に .h5 ウェイトファイルが見つかりません。"
                f" 含まれるファイル: {names}"
            )

        # Step 3: HDF5 から層名に基づいて重みを抽出
        with tempfile.TemporaryDirectory() as tmpdir:
            extracted = zf.extract(weight_file, tmpdir)

            with h5py.File(extracted, "r") as f:
                # デバッグ用: HDF5 内の全キーを出力
                all_keys = []
                def _list_keys(name: str, obj: object) -> None:
                    all_keys.append(name)
                f.visititems(_list_keys)
                logger.info("HDF5 内の全キー (%d 個):", len(all_keys))
                for key in all_keys[:20]:  # 最初の20個のみ表示
                    logger.info("  %s", key)
                if len(all_keys) > 20:
                    logger.info("  ... (残り %d 個)", len(all_keys) - 20)

                layers = []
                activations = []

                # config.json の順序で重みを取得
                for layer_info in expected_dense_layers:
                    layer_name = layer_info["name"]

                    # HDF5 内での層の探索パス候補
                    # Keras 3.x では以下のパターンが考えられる：
                    # - vars/<layer_name>/0/kernel:0, vars/<layer_name>/0/bias:0
                    # - vars/<layer_name>/kernel:0, vars/<layer_name>/bias:0
                    # - <layer_name>/kernel:0, <layer_name>/bias:0
                    kernel_paths = [
                        f"vars/{layer_name}/0/kernel:0",
                        f"vars/{layer_name}/kernel:0",
                        f"{layer_name}/kernel:0",
                        f"vars/{layer_name}/0/0",  # flat-numbered layout
                        f"vars/{layer_name}/0",
                    ]
                    bias_paths = [
                        f"vars/{layer_name}/0/bias:0",
                        f"vars/{layer_name}/bias:0",
                        f"{layer_name}/bias:0",
                        f"vars/{layer_name}/1/0",  # flat-numbered layout
                        f"vars/{layer_name}/1",
                    ]

                    kernel = None
                    bias = None

                    # kernel を探す
                    for kpath in kernel_paths:
                        if kpath in f:
                            kernel = np.array(f[kpath], dtype=np.float32)
                            logger.info("  kernel found at: %s, shape=%s", kpath, kernel.shape)
                            break

                    # bias を探す
                    for bpath in bias_paths:
                        if bpath in f:
                            bias = np.array(f[bpath], dtype=np.float32)
                            logger.info("  bias found at: %s, shape=%s", bpath, bias.shape)
                            break

                    if kernel is None or bias is None:
                        # パスが見つからない場合、層名を含むパスを探す
                        logger.warning(
                            "Layer '%s' の重みが標準パスで見つかりませんでした。"
                            "層名を含むパスを検索します...",
                            layer_name
                        )

                        for key in all_keys:
                            if layer_name in key:
                                if "kernel" in key and kernel is None:
                                    kernel = np.array(f[key], dtype=np.float32)
                                    logger.info("  kernel found at: %s, shape=%s", key, kernel.shape)
                                elif "bias" in key and bias is None:
                                    bias = np.array(f[key], dtype=np.float32)
                                    logger.info("  bias found at: %s, shape=%s", key, bias.shape)

                    if kernel is None or bias is None:
                        raise ValueError(
                            f"Layer '{layer_name}' の重みが見つかりませんでした。\n"
                            f"HDF5内のキー: {all_keys[:10]}..."
                        )

                    layers.append((kernel, bias))
                    activations.append(layer_info["activation"])
                    logger.info(
                        "  Dense層を抽出: %s, 入力=%d, 出力=%d, 活性化=%s",
                        layer_name, kernel.shape[0], kernel.shape[1],
                        layer_info["activation"]
                    )

    if len(layers) != expected_count:
        raise ValueError(
            f"期待される Dense 層数 {expected_count} と "
            f"抽出された層数 {len(layers)} が一致しません"
        )

    logger.info("分類器から %d 個の Dense 層を抽出完了", len(layers))
    return layers, activations


class PyTorchClassifierMLP(nn.Module):
    """pNSFWMedia分類器のPyTorch再実装。

    Kerasモデルの重みをPyTorchのLinear層に読み込む。
    各層の活性化関数は config.json から取得した情報に基づいて構築される。
    """

    def __init__(
        self,
        keras_weights: list[tuple[np.ndarray, np.ndarray]],
        activations: list[str],
    ) -> None:
        """
        Args:
            keras_weights: [(weight, bias), ...] のリスト。
            activations: 各層の活性化関数名 ["tanh", "sigmoid", ...] のリスト。
        """
        super().__init__()

        if len(keras_weights) != len(activations):
            raise ValueError(
                f"重みの数 ({len(keras_weights)}) と活性化関数の数 ({len(activations)}) "
                f"が一致しません"
            )

        layers = []
        for i, ((w, b), act) in enumerate(zip(keras_weights, activations)):
            linear = nn.Linear(w.shape[0], w.shape[1])
            # Kerasは(in, out)形状、PyTorchは(out, in)形状
            linear.weight.data = torch.from_numpy(w.T.copy())
            linear.bias.data = torch.from_numpy(b.copy())
            layers.append(linear)

            # config.json の activation に従って活性化関数を追加
            if act == "tanh":
                layers.append(nn.Tanh())
            elif act == "relu":
                layers.append(nn.ReLU())
            elif act == "sigmoid":
                layers.append(nn.Sigmoid())
            elif act == "linear":
                # linear の場合は何も追加しない（恒等関数）
                pass
            else:
                logger.warning(
                    "未知の活性化関数 '%s' が Layer %d に指定されています。"
                    "スキップします。",
                    act, i
                )

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

    @staticmethod
    def _extract_tensor(raw_output: object) -> torch.Tensor:
        """バックボーン出力からテンソルを抽出する。

        onnx2torch はモデルによって Tensor / tuple / list / OrderedDict
        のいずれかを返す可能性がある。最も大きい 4D テンソルを返す。
        """
        if isinstance(raw_output, torch.Tensor):
            return raw_output

        # tuple / list の場合
        if isinstance(raw_output, (tuple, list)):
            # 4Dテンソルを優先、なければ最大要素数のテンソルを選択
            tensors = [t for t in raw_output if isinstance(t, torch.Tensor)]
            four_d = [t for t in tensors if t.dim() == 4]
            if four_d:
                return max(four_d, key=lambda t: t.numel())
            if tensors:
                return max(tensors, key=lambda t: t.numel())

        # dict / OrderedDict の場合
        if isinstance(raw_output, dict):
            tensors = [v for v in raw_output.values() if isinstance(v, torch.Tensor)]
            four_d = [t for t in tensors if t.dim() == 4]
            if four_d:
                return max(four_d, key=lambda t: t.numel())
            if tensors:
                return max(tensors, key=lambda t: t.numel())

        raise TypeError(
            f"バックボーン出力の型が不正です: {type(raw_output)}。"
            f" Tensor / tuple / list / dict のいずれかが必要です。"
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """順伝播。

        Args:
            images: (N, 3, 320, 320) float32 [0, 1]。

        Returns:
            NSFW確率 (N, 1)。
        """
        # バックボーン特徴抽出 (NCHW出力)
        raw = self.backbone(images)
        features = self._extract_tensor(raw)

        # Global Average Pooling (空間軸: H, W)
        if features.dim() == 4:
            pooled = features.mean(dim=[2, 3])  # (N, C)
        elif features.dim() == 3:
            pooled = features.mean(dim=2)  # (N, C)
        else:
            pooled = features

        # pooledの形状を確認
        if pooled.dim() != 2:
            # 2次元でない場合、flattenしてから適切な形状に変換
            pooled = pooled.flatten(1)
            # feature_channelsと一致するように調整
            if pooled.shape[1] != self.feature_channels:
                if pooled.shape[1] > self.feature_channels:
                    pooled = pooled[:, :self.feature_channels]
                else:
                    # パディングが必要な場合
                    padding = torch.zeros(
                        pooled.shape[0],
                        self.feature_channels - pooled.shape[1],
                        device=pooled.device,
                        dtype=pooled.dtype
                    )
                    pooled = torch.cat([pooled, padding], dim=1)

        # 直交射影 (C -> 256)
        projected = self.projection(pooled)  # (N, 256)

        # L2正規化
        normalized = F.normalize(projected, p=2, dim=1)

        # 形状の検証と修正
        if normalized.dim() != 2 or normalized.shape[-1] != 256:
            normalized = normalized.contiguous().view(-1, 256)

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

    # 2. 射影行列の読み込み（ファイルが無ければ自動生成）
    logger.info("[2/3] 射影行列を準備中 ...")
    projection = load_or_create_projection(
        feature_channels=feature_channels,
        projection_path=projection_path,
    )
    logger.info("射影行列の形状: %s", projection.shape)

    # 3. Keras分類器の重みをPyTorchに変換
    logger.info("[3/3] pNSFWMedia分類器を読み込み中 ...")
    keras_weights, activations = _load_keras_classifier_weights(classifier_path)
    classifier_mlp = PyTorchClassifierMLP(keras_weights, activations)

    # パイプライン組み立て
    model = FrozenNSFWClassifier(
        backbone=backbone,
        projection_matrix=projection,
        classifier_mlp=classifier_mlp,
        feature_channels=feature_channels,
    )
    model = model.to(device)

    # ダミー入力でステップごとに動作確認
    dummy = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    with torch.no_grad():
        # 1. バックボーン
        raw = backbone(dummy.to(device))
        logger.info("バックボーン出力型: %s", type(raw))
        if isinstance(raw, torch.Tensor):
            logger.info("バックボーン出力形状: %s", raw.shape)
        elif isinstance(raw, (tuple, list)):
            for idx, t in enumerate(raw):
                if isinstance(t, torch.Tensor):
                    logger.info("バックボーン出力[%d]: %s", idx, t.shape)
        elif isinstance(raw, dict):
            for k, v in raw.items():
                if isinstance(v, torch.Tensor):
                    logger.info("バックボーン出力[%s]: %s", k, v.shape)

        features = FrozenNSFWClassifier._extract_tensor(raw)
        logger.info("抽出後の特徴形状: %s", features.shape)

        # 2. GAP
        if features.dim() == 4:
            pooled = features.mean(dim=[2, 3])
        elif features.dim() == 3:
            pooled = features.mean(dim=2)
        else:
            pooled = features
        logger.info("GAP後の形状: %s (期待: (1, %d))", pooled.shape, feature_channels)

        # pooledの形状を確認して修正
        if pooled.dim() != 2:
            logger.warning("GAP後の次元が2ではありません: %d。flattenします。", pooled.dim())
            pooled = pooled.flatten(1)

        if pooled.shape[1] != feature_channels:
            logger.warning(
                "GAP後のチャネル数が期待と異なります: %d (期待: %d)",
                pooled.shape[1], feature_channels
            )
            # 形状を調整
            if pooled.shape[1] > feature_channels:
                pooled = pooled[:, :feature_channels]
            else:
                # パディング
                padding = torch.zeros(
                    pooled.shape[0],
                    feature_channels - pooled.shape[1],
                    device=pooled.device,
                    dtype=pooled.dtype
                )
                pooled = torch.cat([pooled, padding], dim=1)
            logger.info("形状を調整しました: %s", pooled.shape)

        # 3. 射影
        projected = model.projection(pooled)
        logger.info("射影後の形状: %s (期待: (1, 256))", projected.shape)

        # 4. 分類器
        normalized = F.normalize(projected, p=2, dim=1)
        logger.info("正規化後の形状: %s", normalized.shape)

        # 形状を確認して、2次元テンソルで最後の次元が256であることを保証
        if normalized.dim() != 2 or normalized.shape[-1] != 256:
            logger.warning(
                "正規化後の形状が不正です: %s。reshapeを試みます。",
                normalized.shape
            )
            # contiguous()を呼んで、view可能な状態にする
            normalized = normalized.contiguous().view(-1, 256)
            logger.info("reshape後の形状: %s", normalized.shape)

        out = model.classifier_mlp(normalized)
        logger.info("分類器出力形状: %s", out.shape)

    logger.info("パイプライン検証完了 - ダミー出力形状: %s", out.shape)
    logger.info("=== 凍結済み分類パイプラインの構築完了 ===")

    return model
