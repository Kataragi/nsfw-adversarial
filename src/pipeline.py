"""End-to-end differentiable pipeline: Image -> NSFW probability.

Converts the NudeNet ONNX backbone to TensorFlow, then chains it with
the orthogonal projection and pNSFWMedia classifier so that
tf.GradientTape can compute image-level gradients for adversarial attacks.

Projection matrix:
    The (C, 256) orthogonal projection is deterministically generated
    using QR decomposition with numpy.random.default_rng(seed=42),
    matching pNSFWMedia/src/extract_embeddings_nudenet.py.
    If the .npy file does not exist it is regenerated automatically.

Pipeline:
    Image (N,320,320,3) [0,1]
    -> NudeNet backbone (TF, converted from ONNX)
    -> Global Average Pooling
    -> Orthogonal projection (C -> 256)
    -> L2 normalization
    -> pNSFWMedia classifier
    -> NSFW probability (N,1)
"""

import json
import logging
import os
from pathlib import Path

import numpy as np
import onnx
import tensorflow as tf

logger = logging.getLogger(__name__)

IMAGE_SIZE = 320


# ── NudeNet ONNX helpers ──────────────────────────────────────────


def find_nudenet_onnx() -> str:
    """Locate the NudeNet ONNX model file.

    Searches the ``nudenet`` package directory for any ``.onnx`` file,
    then falls back to common paths.

    Returns:
        Absolute path to the ONNX model.

    Raises:
        FileNotFoundError: If the model cannot be found.
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
        "NudeNet ONNX model not found. Install nudenet: pip install nudenet"
    )


def _find_backbone_node(onnx_model: onnx.ModelProto) -> str:
    """Find the backbone feature node in a YOLOv8 ONNX model.

    Heuristic: look for the SPPF output (MaxPool -> Concat pattern unique
    to the SPPF block at the end of the YOLOv8 backbone).  If that fails,
    fall back to the *first* Conv output that feeds a Concat – which is
    the deepest backbone feature at the backbone-FPN boundary.

    Using the first Conv->Concat candidate avoids accidentally selecting
    nodes inside the FPN/neck that have shape mismatches.
    """
    # --- 1. 優先: SPPF ブロックの出力を探す ---
    # SPPF は MaxPool -> Concat パターン。Concat の出力がバックボーン最終特徴。
    for node in onnx_model.graph.node:
        if node.op_type == "Concat":
            # Concat の入力がすべて MaxPool (SPPF 構造) か確認
            input_ops = set()
            for inp in node.input:
                for n2 in onnx_model.graph.node:
                    if n2.output and n2.output[0] == inp:
                        input_ops.add(n2.op_type)
            if "MaxPool" in input_ops and node.output:
                # この Concat の出力を通す次の Conv を探す
                concat_out = node.output[0]
                for n2 in onnx_model.graph.node:
                    if n2.op_type == "Conv" and concat_out in n2.input and n2.output:
                        logger.info("SPPF 後の Conv ノードを検出: %s", n2.output[0])
                        return n2.output[0]
                # Conv が見つからなければ Concat 出力自体を使用
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

    raise RuntimeError("Could not find backbone feature node in ONNX model")


def _get_backbone_feature_dim_from_extracted(
    backbone_onnx_path: str,
) -> tuple[int, ...]:
    """Run a dummy forward pass on the *extracted* backbone ONNX to get
    the output shape.

    Unlike the previous approach that ran the full NudeNet model (which
    could fail at FPN Concat nodes), this runs only the backbone sub-graph.
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


# ── ONNX -> TF conversion ─────────────────────────────────────────


def _extract_backbone_onnx(onnx_path: str, output_path: str) -> str:
    """Extract backbone sub-graph from the full NudeNet ONNX model."""
    model = onnx.load(onnx_path)
    input_name = model.graph.input[0].name
    backbone_node = _find_backbone_node(model)

    logger.info("Backbone feature node: %s", backbone_node)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    onnx.utils.extract_model(
        onnx_path,
        output_path,
        input_names=[input_name],
        output_names=[backbone_node],
    )
    logger.info("Backbone ONNX saved to %s", output_path)
    return output_path


def _sanitize_onnx_names(onnx_path: str) -> None:
    """Sanitize ONNX node/output names so they comply with TF naming rules.

    TensorFlow SavedModel requires names matching ``^[A-Za-z0-9.][A-Za-z0-9_./>-]*$``.
    NudeNet ONNX models contain names like ``/model.22/cv3.2/Conv_output_0/``
    which have leading/trailing slashes.  This function rewrites names in-place.
    """
    import re

    model = onnx.load(onnx_path)
    _TF_NAME_RE = re.compile(r"^[A-Za-z0-9.][A-Za-z0-9_./>-]*$")

    def _clean(name: str) -> str:
        if not name or _TF_NAME_RE.match(name):
            return name
        # 先頭・末尾のスラッシュを除去し、残ったスラッシュをドットに変換
        cleaned = name.strip("/").replace("/", ".")
        # 先頭が不正な文字の場合にプレフィックスを付与
        if cleaned and not re.match(r"^[A-Za-z0-9.]", cleaned):
            cleaned = "n." + cleaned
        return cleaned or "unnamed"

    # 名前のマッピングテーブルを構築
    rename_map: dict[str, str] = {}

    for node in model.graph.node:
        if node.name:
            new_name = _clean(node.name)
            if new_name != node.name:
                rename_map[node.name] = new_name
                node.name = new_name
        for i, out in enumerate(node.output):
            new_out = _clean(out)
            if new_out != out:
                rename_map[out] = new_out
                node.output[i] = new_out
        for i, inp in enumerate(node.input):
            if inp in rename_map:
                node.input[i] = rename_map[inp]

    # グラフの入出力名も更新
    for io in list(model.graph.input) + list(model.graph.output):
        if io.name in rename_map:
            io.name = rename_map[io.name]

    # initializer 名も更新
    for init in model.graph.initializer:
        if init.name in rename_map:
            init.name = rename_map[init.name]

    # value_info 名も更新
    for vi in model.graph.value_info:
        if vi.name in rename_map:
            vi.name = rename_map[vi.name]

    if rename_map:
        logger.info(
            "Sanitized %d ONNX names for TF compatibility", len(rename_map)
        )

    onnx.save(model, onnx_path)


def convert_backbone(
    nudenet_onnx_path: str | None = None,
    cache_dir: str = "models/tf_backbone",
    force: bool = False,
) -> str:
    """Convert NudeNet ONNX backbone to a TF SavedModel (cached).

    Steps:
        1. Find the backbone feature node (SPPF output or Conv->Concat boundary)
        2. Extract backbone-only ONNX sub-graph
        3. Get feature dimensions from the extracted (not full) model
        4. Sanitize ONNX node names for TF compatibility
        5. Convert to TF SavedModel via onnx2tf

    Args:
        nudenet_onnx_path: Path to NudeNet ``best.onnx``.  Auto-detected
            if *None*.
        cache_dir: Directory to store the converted model and metadata.
        force: Re-convert even if a cached model exists.

    Returns:
        Path to the TF SavedModel directory.
    """
    import onnx2tf

    saved_model_path = os.path.join(cache_dir, "saved_model.pb")
    meta_path = os.path.join(cache_dir, "backbone_meta.json")

    if os.path.exists(saved_model_path) and os.path.exists(meta_path) and not force:
        logger.info("Using cached TF backbone from %s", cache_dir)
        return cache_dir

    if nudenet_onnx_path is None:
        nudenet_onnx_path = find_nudenet_onnx()

    # 1. Extract backbone-only ONNX (before running inference on full model)
    os.makedirs(cache_dir, exist_ok=True)
    backbone_onnx = os.path.join(cache_dir, "backbone.onnx")
    _extract_backbone_onnx(nudenet_onnx_path, backbone_onnx)

    # 2. Get feature dimensions from the *extracted* backbone
    #    (avoids Concat shape errors in the full model's FPN)
    feat_shape = _get_backbone_feature_dim_from_extracted(backbone_onnx)
    feature_channels = int(feat_shape[1])  # NCHW
    logger.info(
        "Backbone feature shape (NCHW): %s  channels=%d",
        feat_shape,
        feature_channels,
    )

    # 3. Sanitize ONNX node names for TF naming rules
    _sanitize_onnx_names(backbone_onnx)

    # 4. Convert to TF SavedModel
    logger.info("Converting backbone ONNX -> TF SavedModel ...")
    onnx2tf.convert(
        input_onnx_file_path=backbone_onnx,
        output_folder_path=cache_dir,
        non_verbose=True,
        copy_onnx_input_output_names_to_tflite=False,
        output_signaturedefs=True,
    )

    # 5. Save metadata
    onnx_model = onnx.load(nudenet_onnx_path)
    backbone_node = _find_backbone_node(onnx_model)
    meta = {
        "backbone_node": backbone_node,
        "feature_channels": feature_channels,
        "feature_shape_nchw": list(feat_shape),
        "image_size": IMAGE_SIZE,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Backbone conversion complete -> %s", cache_dir)
    return cache_dir


# ── Projection matrix ─────────────────────────────────────────────


def create_orthogonal_projection(
    feature_dim: int, output_dim: int = 256, seed: int = 42
) -> np.ndarray:
    """Deterministically create the same orthogonal projection as pNSFWMedia.

    Reproduces pNSFWMedia/src/extract_embeddings_nudenet.py
    ``_create_orthogonal_projection()`` exactly.  Uses QR decomposition
    with ``numpy.random.default_rng(seed=42)`` so the result is always
    identical given the same *feature_dim*.

    Args:
        feature_dim: Backbone output channels (C).
        output_dim: Target dimensionality (default 256).
        seed: Random seed (pNSFWMedia default is 42).

    Returns:
        ``(feature_dim, output_dim)`` float32 orthogonal matrix.
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
    """Load the projection matrix from file, or auto-generate if missing.

    When the ``.npy`` file does not exist the matrix is regenerated using
    QR decomposition (seed=42), which is identical to the one produced by
    pNSFWMedia.  The generated matrix is saved for future runs.

    Args:
        feature_channels: Backbone output channels.
        projection_path: Path to ``.npy`` file (may be *None* or non-existent).
        output_dim: Target dimensionality.

    Returns:
        ``(feature_channels, output_dim)`` float32 matrix.
    """
    if projection_path and os.path.exists(projection_path):
        projection = np.load(projection_path).astype(np.float32)
        if projection.shape[0] != feature_channels:
            raise ValueError(
                f"Projection matrix first dim ({projection.shape[0]}) != "
                f"backbone channels ({feature_channels})"
            )
        logger.info("Projection matrix loaded: %s %s", projection_path, projection.shape)
        return projection

    logger.info(
        "Projection file not found – auto-generating via QR (seed=42) "
        "(feature_dim=%d -> %d)",
        feature_channels,
        output_dim,
    )
    projection = create_orthogonal_projection(feature_channels, output_dim)

    save_path = projection_path or "models/nudenet_projection.npy"
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    np.save(save_path, projection)
    logger.info("Saved generated projection to %s", save_path)

    return projection


# ── End-to-end differentiable model ───────────────────────────────


class EndToEndModel(tf.keras.Model):
    """End-to-end differentiable model: Image -> NSFW probability.

    All intermediate operations are native TensorFlow so that
    ``tf.GradientTape`` can compute gradients w.r.t. the input image.

    Args:
        backbone: Loaded TF backbone (Keras model or SavedModel).
        projection_matrix: Orthogonal projection ``(C, 256)``.
        classifier: pNSFWMedia Keras classifier.
        backbone_output_is_nhwc: Whether backbone output is NHWC format.
    """

    def __init__(
        self,
        backbone: tf.keras.Model,
        projection_matrix: np.ndarray,
        classifier: tf.keras.Model,
        backbone_output_is_nhwc: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.backbone.trainable = False
        self.projection = tf.Variable(
            projection_matrix.astype(np.float32),
            trainable=False,
            name="projection",
        )
        self.nsfw_classifier = classifier
        self.nsfw_classifier.trainable = False
        self._output_nhwc = backbone_output_is_nhwc

    def call(self, images: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass.

        Args:
            images: ``(N, 320, 320, 3)`` float32 in ``[0, 1]``.
            training: Unused (kept for Keras API compatibility).

        Returns:
            NSFW probabilities ``(N, 1)``.
        """
        # onnx2tf converts to NHWC by default so input is already (N,H,W,C)
        out = self.backbone(images, training=False)
        # TFSMLayer returns a dict keyed by output name – extract the tensor
        if isinstance(out, dict):
            features = next(iter(out.values()))
        else:
            features = out

        # GAP – spatial axes depend on data format
        if self._output_nhwc:
            pooled = tf.reduce_mean(features, axis=[1, 2])  # (N, C)
        else:
            pooled = tf.reduce_mean(features, axis=[2, 3])  # (N, C)

        # Project to 256-d
        projected = tf.matmul(pooled, self.projection)

        # L2 normalise
        norms = tf.maximum(
            tf.norm(projected, axis=1, keepdims=True), 1e-8
        )
        normalized = projected / norms

        # Classify
        output = self.nsfw_classifier(normalized, training=False)
        return output

    # convenience helpers ------------------------------------------------

    def predict_images(
        self, images: np.ndarray, batch_size: int = 16
    ) -> np.ndarray:
        """Batch prediction returning numpy NSFW probabilities ``(N,)``."""
        preds = []
        for i in range(0, len(images), batch_size):
            batch = tf.constant(images[i : i + batch_size], dtype=tf.float32)
            preds.append(self(batch, training=False).numpy().flatten())
        return np.concatenate(preds)

    def compute_gradient(
        self, images: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Compute gradient of NSFW probability w.r.t. input images.

        Args:
            images: ``(N, 320, 320, 3)`` float32 in ``[0, 1]``.

        Returns:
            ``(predictions, gradients)`` – both same shape as *images*.
        """
        images = tf.cast(images, tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(images)
            preds = self(images, training=False)
            loss = tf.reduce_sum(preds)
        grads = tape.gradient(loss, images)
        return preds, grads


# ── Builder ────────────────────────────────────────────────────────


def _detect_backbone_format(
    backbone: tf.keras.Model, expected_channels: int
) -> bool:
    """Return True if the backbone outputs NHWC."""
    dummy = tf.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=tf.float32)
    raw = backbone(dummy, training=False)
    # TFSMLayer returns a dict – extract the tensor
    out = next(iter(raw.values())) if isinstance(raw, dict) else raw
    shape = out.shape  # (1, ?, ?, ?)
    if shape[-1] == expected_channels:
        return True  # NHWC
    if len(shape) == 4 and shape[1] == expected_channels:
        return False  # NCHW
    # Ambiguous – default to NHWC (onnx2tf default)
    logger.warning(
        "Could not determine backbone output format from shape %s; assuming NHWC",
        shape,
    )
    return True


def build_pipeline(
    nudenet_onnx_path: str | None = None,
    projection_path: str = "models/nudenet_projection.npy",
    classifier_path: str = "models/target_classifier/pnsfwmedia_classifier.keras",
    backbone_cache_dir: str = "models/tf_backbone",
    force_convert: bool = False,
) -> EndToEndModel:
    """Build the end-to-end differentiable pipeline.

    On first invocation the NudeNet ONNX backbone is converted to TF
    (result is cached in *backbone_cache_dir*).

    Args:
        nudenet_onnx_path: Path to ``best.onnx`` (auto-detected if *None*).
        projection_path: ``.npy`` file with the ``(C, 256)`` projection matrix.
        classifier_path: ``.keras`` pNSFWMedia classifier.
        backbone_cache_dir: Cache directory for the converted backbone.
        force_convert: Force re-conversion even if cache exists.

    Returns:
        Fully-differentiable :class:`EndToEndModel`.
    """
    # Convert backbone (no-op if cached)
    convert_backbone(
        nudenet_onnx_path=nudenet_onnx_path,
        cache_dir=backbone_cache_dir,
        force=force_convert,
    )

    # Load metadata
    meta_path = os.path.join(backbone_cache_dir, "backbone_meta.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    feature_channels = meta["feature_channels"]

    # Load TF backbone (SavedModel → TFSMLayer for Keras 3 compatibility)
    backbone = tf.keras.layers.TFSMLayer(
        backbone_cache_dir, call_endpoint="serving_default"
    )
    logger.info("TF backbone loaded from %s", backbone_cache_dir)

    # Detect output format
    is_nhwc = _detect_backbone_format(backbone, feature_channels)
    logger.info("Backbone output format: %s", "NHWC" if is_nhwc else "NCHW")

    # Load projection matrix (auto-generate if file is missing)
    projection = load_or_create_projection(
        feature_channels=feature_channels,
        projection_path=projection_path,
    )
    logger.info("Projection matrix: %s", projection.shape)

    # Load classifier
    classifier = tf.keras.models.load_model(classifier_path)
    logger.info("Classifier loaded: %d params", classifier.count_params())

    # Assemble
    model = EndToEndModel(backbone, projection, classifier, is_nhwc)

    # Verify with dummy input
    dummy = tf.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=tf.float32)
    out = model(dummy, training=False)
    logger.info("Pipeline verified – dummy output shape: %s", out.shape)

    return model
