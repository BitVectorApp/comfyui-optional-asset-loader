"""ComfyUI custom node implementations for optional asset loading and conditional routing.
"""

from typing import Any, Dict, Tuple

try:
    from .utils import (
        create_blank_image_tensor,
        create_blank_mask_tensor,
        is_fallback_asset,
        load_image_to_tensor,
        load_video_to_tensor,
        resolve_file_path,
    )
except (ImportError, ValueError):
    from utils import (
        create_blank_image_tensor,
        create_blank_mask_tensor,
        is_fallback_asset,
        load_image_to_tensor,
        load_video_to_tensor,
        resolve_file_path,
    )


class LoadImageOptional:
    """Loads an image if provided and valid.
    
    If the file is not provided (empty string), matches a fallback placeholder name
    (e.g., 'blank_start.png'), or does not exist on disk, this node safely outputs
    Python None (or a blank tensor), preventing workflow execution crashes.
    """

    @classmethod
    def INPUT_TYPES(s) -> Dict[str, Any]:
        return {
            "required": {
                "image_path": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "fallback_patterns": (
                    "STRING",
                    {
                        "default": "blank_start.png, none, null, empty, undefined",
                        "multiline": False,
                    },
                ),
                "missing_behavior": (
                    ["Return None", "Return Blank Tensor", "Raise Error"],
                    {"default": "Return None"},
                ),
                "blank_width": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 8}),
                "blank_height": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 8}),
                "blank_color": (["black", "white", "gray"], {"default": "black"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BOOLEAN", "STRING")
    RETURN_NAMES = ("IMAGE", "MASK", "has_image", "resolved_path")
    FUNCTION = "load"
    CATEGORY = "optional_loaders"

    def load(
        self,
        image_path: str,
        fallback_patterns: str = "blank_start.png, none, null, empty, undefined",
        missing_behavior: str = "Return None",
        blank_width: int = 512,
        blank_height: int = 512,
        blank_color: str = "black",
    ) -> Tuple[Any, Any, bool, str]:
        if is_fallback_asset(image_path, fallback_patterns):
            return self._handle_missing(
                image_path,
                missing_behavior,
                blank_width,
                blank_height,
                blank_color,
                reason="Path matched fallback pattern or was empty",
            )

        resolved = resolve_file_path(image_path, fallback_patterns)
        if not resolved:
            return self._handle_missing(
                image_path,
                missing_behavior,
                blank_width,
                blank_height,
                blank_color,
                reason=f"File '{image_path}' could not be found on disk",
            )

        try:
            image_tensor, mask_tensor = load_image_to_tensor(resolved)
            return (image_tensor, mask_tensor, True, resolved)
        except Exception as e:
            if missing_behavior == "Raise Error":
                raise e
            return self._handle_missing(
                image_path,
                missing_behavior,
                blank_width,
                blank_height,
                blank_color,
                reason=f"Failed to decode image '{resolved}': {e}",
            )

    def _handle_missing(
        self,
        image_path: str,
        behavior: str,
        width: int,
        height: int,
        color: str,
        reason: str,
    ) -> Tuple[Any, Any, bool, str]:
        if behavior == "Raise Error":
            raise FileNotFoundError(f"LoadImageOptional: {reason} (input='{image_path}')")

        if behavior == "Return Blank Tensor":
            blank_img = create_blank_image_tensor(width, height, color=color)
            blank_mask = create_blank_mask_tensor(width, height)
            return (blank_img, blank_mask, False, "")

        # Default: Return None
        return (None, None, False, "")


class LoadVideoOptional:
    """Loads video frames and audio if provided and valid.
    
    If the video file is missing, empty, or matches a fallback placeholder name
    (e.g., 'blank_start.mp4'), this node safely returns Python None (or a blank tensor),
    avoiding graph crashes or ZeroDivisionError in downstream nodes.
    """

    @classmethod
    def INPUT_TYPES(s) -> Dict[str, Any]:
        return {
            "required": {
                "video_path": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "fallback_patterns": (
                    "STRING",
                    {
                        "default": "blank_start.mp4, none, null, empty, undefined",
                        "multiline": False,
                    },
                ),
                "missing_behavior": (
                    ["Return None", "Return Blank Tensor", "Raise Error"],
                    {"default": "Return None"},
                ),
                "frame_load_cap": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "skip_first_frames": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
                "trim_last_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.1}),
                "blank_width": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 8}),
                "blank_height": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 8}),
                "blank_frames": ("INT", {"default": 64, "min": 1, "max": 1000, "step": 1}),
                "default_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "BOOLEAN", "INT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("IMAGE", "AUDIO", "has_video", "frame_count", "fps", "duration", "resolved_path")
    FUNCTION = "load"
    CATEGORY = "optional_loaders"

    def load(
        self,
        video_path: str,
        fallback_patterns: str = "blank_start.mp4, none, null, empty, undefined",
        missing_behavior: str = "Return None",
        frame_load_cap: int = 0,
        skip_first_frames: int = 0,
        select_every_nth: int = 1,
        trim_last_seconds: float = 0.0,
        blank_width: int = 512,
        blank_height: int = 512,
        blank_frames: int = 64,
        default_fps: float = 24.0,
    ) -> Tuple[Any, Any, bool, int, float, float, str]:
        if is_fallback_asset(video_path, fallback_patterns):
            return self._handle_missing(
                video_path,
                missing_behavior,
                blank_width,
                blank_height,
                blank_frames,
                default_fps,
                reason="Video path matched fallback pattern or was empty",
            )

        resolved = resolve_file_path(video_path, fallback_patterns)
        if not resolved:
            return self._handle_missing(
                video_path,
                missing_behavior,
                blank_width,
                blank_height,
                blank_frames,
                default_fps,
                reason=f"Video file '{video_path}' could not be found on disk",
            )

        try:
            frames, audio, frame_count, fps, duration = load_video_to_tensor(
                resolved,
                frame_load_cap=frame_load_cap,
                skip_first_frames=skip_first_frames,
                select_every_nth=select_every_nth,
                trim_last_seconds=trim_last_seconds,
            )
            return (frames, audio, True, frame_count, fps, duration, resolved)
        except Exception as e:
            if missing_behavior == "Raise Error":
                raise e
            return self._handle_missing(
                video_path,
                missing_behavior,
                blank_width,
                blank_height,
                blank_frames,
                default_fps,
                reason=f"Failed to load video '{resolved}': {e}",
            )

    def _handle_missing(
        self,
        video_path: str,
        behavior: str,
        width: int,
        height: int,
        frames: int,
        fps: float,
        reason: str,
    ) -> Tuple[Any, Any, bool, int, float, float, str]:
        if behavior == "Raise Error":
            raise FileNotFoundError(f"LoadVideoOptional: {reason} (input='{video_path}')")

        safe_fps = max(float(fps), 1.0)
        if behavior == "Return Blank Tensor":
            blank_frames = create_blank_image_tensor(width, height, color="black", batch_size=frames)
            duration = float(frames) / safe_fps
            return (blank_frames, None, False, frames, safe_fps, duration, "")

        # Default: Return None (safe fps provided to prevent ZeroDivisionError if fps output is connected)
        return (None, None, False, 0, safe_fps, 0.0, "")


class OptionalDimensionFilter:
    """Filters an incoming IMAGE by dimension threshold.
    
    Acts as a direct replacement for workflows checking placeholder image dimensions
    (such as width <= 91 or video width <= 16). If the image matches the placeholder
    dimension condition, this node drops it to Python None (or a blank tensor).
    """

    @classmethod
    def INPUT_TYPES(s) -> Dict[str, Any]:
        return {
            "required": {
                "dimension_target": (
                    ["width", "height", "either_dimension", "both_dimensions"],
                    {"default": "width"},
                ),
                "comparison": (
                    [
                        "greater_than",
                        "greater_or_equal",
                        "less_than",
                        "less_or_equal",
                        "equal",
                        "not_equal",
                    ],
                    {"default": "greater_than"},
                ),
                "threshold": ("INT", {"default": 91, "min": 1, "max": 16384, "step": 1}),
                "fallback_behavior": (
                    ["Return None", "Return Blank Tensor"],
                    {"default": "Return None"},
                ),
            },
            "optional": {
                "image": ("IMAGE",),
                "blank_width": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 8}),
                "blank_height": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 8}),
            },
        }

    RETURN_TYPES = ("IMAGE", "BOOLEAN", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "passed", "width", "height")
    FUNCTION = "filter"
    CATEGORY = "optional_loaders"

    def filter(
        self,
        dimension_target: str,
        comparison: str,
        threshold: int,
        fallback_behavior: str = "Return None",
        image: Any = None,
        blank_width: int = 512,
        blank_height: int = 512,
    ) -> Tuple[Any, bool, int, int]:
        if image is None:
            return (None, False, 0, 0)

        # Determine dimensions from tensor shape [B, H, W, C] or [H, W, C]
        if hasattr(image, "shape"):
            shape = image.shape
            if len(shape) == 4:
                h, w = int(shape[1]), int(shape[2])
            elif len(shape) == 3:
                h, w = int(shape[0]), int(shape[1])
            else:
                h, w = 0, 0
        else:
            return (None, False, 0, 0)

        def _eval_op(val: int) -> bool:
            if comparison == "greater_than":
                return val > threshold
            if comparison == "greater_or_equal":
                return val >= threshold
            if comparison == "less_than":
                return val < threshold
            if comparison == "less_or_equal":
                return val <= threshold
            if comparison == "equal":
                return val == threshold
            if comparison == "not_equal":
                return val != threshold
            return False

        if dimension_target == "width":
            passed = _eval_op(w)
        elif dimension_target == "height":
            passed = _eval_op(h)
        elif dimension_target == "either_dimension":
            passed = _eval_op(w) or _eval_op(h)
        elif dimension_target == "both_dimensions":
            passed = _eval_op(w) and _eval_op(h)
        else:
            passed = True

        if passed:
            return (image, True, w, h)

        if fallback_behavior == "Return Blank Tensor":
            blank = create_blank_image_tensor(blank_width, blank_height)
            return (blank, False, w, h)

        return (None, False, w, h)


class OptionalImageGate:
    """Passes an IMAGE (and MASK) through if enabled, otherwise returns Python None or blank."""

    @classmethod
    def INPUT_TYPES(s) -> Dict[str, Any]:
        return {
            "required": {
                "is_active": ("BOOLEAN", {"default": True}),
                "fallback_behavior": (
                    ["Return None", "Return Blank Tensor"],
                    {"default": "Return None"},
                ),
            },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "blank_width": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 8}),
                "blank_height": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 8}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BOOLEAN")
    RETURN_NAMES = ("IMAGE", "MASK", "has_image")
    FUNCTION = "gate"
    CATEGORY = "optional_loaders"

    def gate(
        self,
        is_active: bool,
        fallback_behavior: str = "Return None",
        image: Any = None,
        mask: Any = None,
        blank_width: int = 512,
        blank_height: int = 512,
    ) -> Tuple[Any, Any, bool]:
        if is_active and image is not None:
            return (image, mask, True)

        if fallback_behavior == "Return Blank Tensor":
            blank_img = create_blank_image_tensor(blank_width, blank_height)
            blank_mask = create_blank_mask_tensor(blank_width, blank_height)
            return (blank_img, blank_mask, False)

        return (None, None, False)


class OptionalVideoGate:
    """Passes video frames (IMAGE) and AUDIO through if enabled, otherwise returns Python None."""

    @classmethod
    def INPUT_TYPES(s) -> Dict[str, Any]:
        return {
            "required": {
                "is_active": ("BOOLEAN", {"default": True}),
                "fallback_behavior": (
                    ["Return None", "Return Blank Tensor"],
                    {"default": "Return None"},
                ),
            },
            "optional": {
                "video_frames": ("IMAGE",),
                "audio": ("AUDIO",),
                "blank_width": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 8}),
                "blank_height": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 8}),
                "blank_frames": ("INT", {"default": 1, "min": 1, "max": 1000, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "BOOLEAN")
    RETURN_NAMES = ("IMAGE", "AUDIO", "has_video")
    FUNCTION = "gate"
    CATEGORY = "optional_loaders"

    def gate(
        self,
        is_active: bool,
        fallback_behavior: str = "Return None",
        video_frames: Any = None,
        audio: Any = None,
        blank_width: int = 512,
        blank_height: int = 512,
        blank_frames: int = 1,
    ) -> Tuple[Any, Any, bool]:
        if is_active and video_frames is not None:
            return (video_frames, audio, True)

        if fallback_behavior == "Return Blank Tensor":
            blank_vid = create_blank_image_tensor(blank_width, blank_height, batch_size=blank_frames)
            return (blank_vid, None, False)

        return (None, None, False)


class NullableConstant:
    """Outputs a constant Python None value for any ComfyUI connection slot.
    
    Eliminates the need for dummy unconnected Context or Switch nodes.
    """

    @classmethod
    def INPUT_TYPES(s) -> Dict[str, Any]:
        return {
            "optional": {
                "label": ("STRING", {"default": "None (Null)", "multiline": False}),
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("none_value",)
    FUNCTION = "get_none"
    CATEGORY = "optional_loaders"

    def get_none(self, label: str = "None (Null)") -> Tuple[Any]:
        return (None,)


class BooleanLogic:
    """Performs boolean operations (AND, OR, NOT, XOR) on boolean inputs.
    
    Useful for combining conditional workflow flags (e.g., has_video AND should_combine).
    """

    @classmethod
    def INPUT_TYPES(s) -> Dict[str, Any]:
        return {
            "required": {
                "operation": (["AND", "OR", "XOR", "NOT_A"], {"default": "AND"}),
            },
            "optional": {
                "a": ("BOOLEAN", {"default": False}),
                "b": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("result",)
    FUNCTION = "op"
    CATEGORY = "optional_loaders"

    def op(self, operation: str = "AND", a: bool = False, b: bool = False) -> Tuple[bool]:
        if operation == "AND":
            return (bool(a and b),)
        elif operation == "OR":
            return (bool(a or b),)
        elif operation == "XOR":
            return (bool(a != b),)
        elif operation == "NOT_A":
            return (bool(not a),)
        return (False,)
