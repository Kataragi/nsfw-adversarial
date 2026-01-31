"""End-to-end differentiable pipeline: Image -> NSFW probability.

Converts the NudeNet ONNX backbone to TensorFlow, then chains it with
the orthogonal projection and pNSFWMedia classifier so that
tf.GradientTape can compute image-level gradients for adversarial attacks.

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

    Searches the ``nudenet`` package directory and common fallback paths.

    Returns:
        Absolute path to ``best.onnx``.

    Raises:
        FileNotFoundError: If the model cannot be found.
    """
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
        "NudeNet ONNX model not found. Install nudenet: pip install nudenet"
    )


def _find_backbone_node(onnx_model: onnx.ModelProto) -> str:
    """Find the backbone feature node in a YOLOv8 ONNX model.

    Heuristic: the last Conv output that feeds into a Concat node
    (backbone-FPN boundary).
    """
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

    # Fallback: last Conv output
    for node in reversed(onnx_model.graph.node):
        if node.op_type == "Conv" and node.output:
            return node.output[0]

    raise RuntimeError("Could not find backbone feature node in ONNX model")


def _get_backbone_feature_dim(onnx_path: str, node_name: str) -> tuple[int, ...]:
    """Run a dummy forward pass to discover the backbone output shape."""
    import onnxruntime as ort

    model = onnx.load(onnx_path)

    # Temporarily add the intermediate node as an output
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


def convert_backbone(
    nudenet_onnx_path: str | None = None,
    cache_dir: str = "models/tf_backbone",
    force: bool = False,
) -> str:
    """Convert NudeNet ONNX backbone to a TF SavedModel (cached).

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

    # 1. Discover backbone node and feature dimension
    onnx_model = onnx.load(nudenet_onnx_path)
    backbone_node = _find_backbone_node(onnx_model)
    feat_shape = _get_backbone_feature_dim(nudenet_onnx_path, backbone_node)
    feature_channels = int(feat_shape[1])  # NCHW
    logger.info(
        "Backbone feature shape (NCHW): %s  channels=%d",
        feat_shape,
        feature_channels,
    )

    # 2. Extract backbone-only ONNX
    backbone_onnx = os.path.join(cache_dir, "backbone.onnx")
    _extract_backbone_onnx(nudenet_onnx_path, backbone_onnx)

    # 3. Convert to TF
    logger.info("Converting backbone ONNX -> TF SavedModel ...")
    onnx2tf.convert(
        input_onnx_file_path=backbone_onnx,
        output_folder_path=cache_dir,
        non_verbose=True,
        copy_onnx_input_output_names_to_tflite=False,
    )

    # 4. Save metadata
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
        features = self.backbone(images, training=False)

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
    out = backbone(dummy, training=False)
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

    # Load TF backbone
    backbone = tf.keras.models.load_model(backbone_cache_dir)
    logger.info("TF backbone loaded (%d params)", backbone.count_params())

    # Detect output format
    is_nhwc = _detect_backbone_format(backbone, feature_channels)
    logger.info("Backbone output format: %s", "NHWC" if is_nhwc else "NCHW")

    # Load projection matrix
    projection = np.load(projection_path).astype(np.float32)
    assert projection.shape[0] == feature_channels, (
        f"Projection matrix first dim ({projection.shape[0]}) != "
        f"backbone channels ({feature_channels})"
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
