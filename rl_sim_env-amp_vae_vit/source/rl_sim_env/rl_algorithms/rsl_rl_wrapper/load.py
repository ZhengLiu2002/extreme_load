import importlib
from typing import Optional

_onnx_import_error: Optional[Exception] = None
_onnx = None
_ort = None


def _ensure_onnx_runtime():
    """Lazily import onnx / onnxruntime so training can run without them."""
    global _onnx, _ort, _onnx_import_error
    if _onnx is not None and _ort is not None:
        return
    try:
        _onnx = importlib.import_module("onnx")
        _ort = importlib.import_module("onnxruntime")
    except ImportError as e:
        _onnx_import_error = e
        raise ImportError(
            "onnxruntime (and onnx) is required for ONNX export/inference. "
            "Install with `pip install onnx onnxruntime-gpu` (or onnxruntime)."
        ) from e


def verify_onnx_model(model_path, name):
    _ensure_onnx_runtime()
    onnx = _onnx
    try:
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        print(f"{name} ONNX validation passed!")
    except onnx.checker.ValidationError as e:
        print(f"{name} ONNX validation failed: {e}")


def load_onnx_model(model_path):
    _ensure_onnx_runtime()
    session_options = _ort.SessionOptions()
    session_options.graph_optimization_level = _ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = _ort.InferenceSession(model_path, sess_options=session_options, providers=["CPUExecutionProvider"])
    return session


def onnx_run_inference(session, actor_obs, vae_obs):
    inputs = {
        "actor_obs": actor_obs,
        "vae_obs": vae_obs,
    }
    outputs = session.run(None, inputs)

    return outputs
