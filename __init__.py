"""ComfyUI Optional Asset Loader

Custom nodes for handling optional/nullable image, video, and audio assets in ComfyUI workflows.
Designed for cloud APIs (Graydient, RunPod, etc.) and models with nullable input slots (MiniMax, Qwen 2.1, etc.).
"""

try:
    from .nodes import (
        BooleanLogic,
        LoadImageOptional,
        LoadVideoOptional,
        NullableConstant,
        OptionalDimensionFilter,
        OptionalImageGate,
        OptionalVideoGate,
    )
except (ImportError, ValueError):
    from nodes import (
        BooleanLogic,
        LoadImageOptional,
        LoadVideoOptional,
        NullableConstant,
        OptionalDimensionFilter,
        OptionalImageGate,
        OptionalVideoGate,
    )

NODE_CLASS_MAPPINGS = {
    "LoadImageOptional": LoadImageOptional,
    "LoadVideoOptional": LoadVideoOptional,
    "OptionalDimensionFilter": OptionalDimensionFilter,
    "OptionalImageGate": OptionalImageGate,
    "OptionalVideoGate": OptionalVideoGate,
    "NullableConstant": NullableConstant,
    "BooleanLogic": BooleanLogic,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageOptional": "Load Image (Optional / Fallback)",
    "LoadVideoOptional": "Load Video (Optional / Fallback)",
    "OptionalDimensionFilter": "Optional Dimension Filter",
    "OptionalImageGate": "Optional Image Gate",
    "OptionalVideoGate": "Optional Video Gate",
    "NullableConstant": "Nullable Constant (None)",
    "BooleanLogic": "Boolean Logic",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
