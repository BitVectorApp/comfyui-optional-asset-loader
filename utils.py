"""Utility functions for comfyui-optional-asset-loader.

Handles path resolution, fallback detection, tensor creation, and file loading
with robust fallbacks for ComfyUI environments and standalone execution.
"""

import fnmatch
import os
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import folder_paths
except ImportError:
    folder_paths = None

try:
    import torch
except ImportError:
    torch = None

import numpy as np
from PIL import Image, ImageOps, ImageSequence


DEFAULT_FALLBACK_PATTERNS = "blank_start.png, blank_start.mp4, none, null, empty, undefined, false"


def parse_fallback_patterns(patterns_str: Optional[str]) -> List[str]:
    """Parse comma-separated fallback patterns into a normalized list of lower-case tokens."""
    if not patterns_str:
        patterns_str = DEFAULT_FALLBACK_PATTERNS
    tokens = [t.strip().lower() for t in patterns_str.split(",") if t.strip()]
    return tokens


def is_fallback_asset(path_or_filename: Optional[str], fallback_patterns: Optional[str] = None) -> bool:
    """Check if the provided path or filename should be treated as an unprovided/fallback asset.
    
    Returns True if:
    - path_or_filename is None or empty/whitespace string
    - filename or basename matches any pattern in fallback_patterns
    """
    if path_or_filename is None:
        return True

    clean_str = str(path_or_filename).strip()
    if not clean_str or clean_str.lower() in ("''", '""'):
        return True

    patterns = parse_fallback_patterns(fallback_patterns)
    lower_path = clean_str.lower()
    basename = os.path.basename(clean_str).lower()

    for pat in patterns:
        if lower_path == pat or basename == pat:
            return True
        # Glob match (e.g. *blank*, blank_*)
        if any(c in pat for c in ("*", "?", "[", "]")):
            if fnmatch.fnmatch(lower_path, pat) or fnmatch.fnmatch(basename, pat):
                return True

    return False


def resolve_file_path(path_or_filename: Optional[str], fallback_patterns: Optional[str] = None) -> Optional[str]:
    """Resolve an incoming path/filename string to an absolute existing file path.
    
    Checks in order:
    1. If matches fallback pattern or is empty -> returns None.
    2. Absolute path if it exists.
    3. ComfyUI input directory (via folder_paths.get_input_directory() or folder_paths.get_annotated_filepath).
    4. Relative to current working directory (with and without './input/' / 'input/' prefixes).
    5. ComfyUI models directory (e.g. for fallback models/assets configured in Graydient).
    """
    if is_fallback_asset(path_or_filename, fallback_patterns):
        return None

    clean_path = str(path_or_filename).strip().strip("'\"")
    if not clean_path:
        return None

    # 1. Absolute path check
    if os.path.isabs(clean_path) and os.path.isfile(clean_path):
        return os.path.abspath(clean_path)

    candidate_paths = []

    # 2. ComfyUI input folder
    if folder_paths is not None:
        try:
            if hasattr(folder_paths, "get_annotated_filepath"):
                annotated = folder_paths.get_annotated_filepath(clean_path)
                if annotated and os.path.isfile(annotated):
                    return os.path.abspath(annotated)
        except Exception:
            pass

        try:
            if hasattr(folder_paths, "get_input_directory"):
                input_dir = folder_paths.get_input_directory()
                if input_dir:
                    candidate_paths.append(os.path.join(input_dir, clean_path))
                    # If clean_path starts with input/ or ./input/, strip it
                    sub = clean_path.replace("\\", "/")
                    if sub.startswith("./input/"):
                        candidate_paths.append(os.path.join(input_dir, sub[8:]))
                    elif sub.startswith("input/"):
                        candidate_paths.append(os.path.join(input_dir, sub[6:]))
        except Exception:
            pass

    # 3. Working directory relative paths
    candidate_paths.append(os.path.abspath(clean_path))
    sub = clean_path.replace("\\", "/")
    if sub.startswith("./input/"):
        candidate_paths.append(os.path.abspath(sub[2:]))
    elif not sub.startswith("input/"):
        candidate_paths.append(os.path.abspath(os.path.join("input", clean_path)))

    # 4. Check folder_paths.models_dir (e.g. /models/fallback/...)
    if folder_paths is not None and hasattr(folder_paths, "models_dir"):
        candidate_paths.append(os.path.join(folder_paths.models_dir, clean_path))
        candidate_paths.append(os.path.join(folder_paths.models_dir, "fallback", os.path.basename(clean_path)))

    # Test all candidates
    for cand in candidate_paths:
        if os.path.isfile(cand):
            return os.path.abspath(cand)

    return None


def create_blank_image_tensor(width: int = 512, height: int = 512, color: str = "black", batch_size: int = 1):
    """Create a blank ComfyUI IMAGE tensor [B, H, W, 3] in float32 [0.0, 1.0]."""
    c_val = 0.0
    if color == "white":
        c_val = 1.0
    elif color == "gray":
        c_val = 0.5

    arr = np.full((batch_size, height, width, 3), c_val, dtype=np.float32)
    if torch is not None:
        return torch.from_numpy(arr)
    return arr


def create_blank_mask_tensor(width: int = 512, height: int = 512, batch_size: int = 1):
    """Create a blank ComfyUI MASK tensor [B, H, W] in float32 [0.0, 1.0] (0.0 = unmasked)."""
    arr = np.zeros((batch_size, height, width), dtype=np.float32)
    if torch is not None:
        return torch.from_numpy(arr)
    return arr


def load_image_to_tensor(file_path: str) -> Tuple[Any, Any]:
    """Load an image file into ComfyUI IMAGE [B, H, W, 3] and MASK [B, H, W] tensors.
    
    Handles orientation (EXIF transpose), grayscale scaling, alpha channel extraction,
    and multi-frame animated images (GIF/APNG/WebP) identical to standard ComfyUI LoadImage.
    """
    img = Image.open(file_path)
    output_images = []
    output_masks = []
    w, h = None, None

    excluded_formats = ["MPO"]
    for i in ImageSequence.Iterator(img):
        i = ImageOps.exif_transpose(i)
        if i.mode == "I":
            i = i.point(lambda p: p * (1 / 255.0))
        image = i.convert("RGB")
        if w is None:
            w, h = image.size
        if image.size != (w, h):
            continue

        image_np = np.array(image).astype(np.float32) / 255.0

        if torch is not None:
            image_tensor = torch.from_numpy(image_np).unsqueeze(0)
            if "A" in i.getbands():
                mask_np = np.array(i.getchannel("A")).astype(np.float32) / 255.0
                mask_tensor = 1.0 - torch.from_numpy(mask_np)
            else:
                mask_tensor = torch.zeros((h, w), dtype=torch.float32)
            output_images.append(image_tensor)
            output_masks.append(mask_tensor.unsqueeze(0))
        else:
            image_tensor = np.expand_dims(image_np, axis=0)
            if "A" in i.getbands():
                mask_np = np.array(i.getchannel("A")).astype(np.float32) / 255.0
                mask_tensor = 1.0 - mask_np
            else:
                mask_tensor = np.zeros((h, w), dtype=np.float32)
            output_images.append(image_tensor)
            output_masks.append(np.expand_dims(mask_tensor, axis=0))

    if not output_images:
        raise ValueError(f"No valid image frames decoded from {file_path}")

    if torch is not None:
        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]
    else:
        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = np.concatenate(output_images, axis=0)
            output_mask = np.concatenate(output_masks, axis=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

    return output_image, output_mask


def load_video_to_tensor(
    file_path: str,
    frame_load_cap: int = 0,
    skip_first_frames: int = 0,
    select_every_nth: int = 1,
    trim_last_seconds: float = 0.0,
) -> Tuple[Any, Optional[Dict[str, Any]], int, float, float]:
    """Load video frames into an IMAGE tensor [N, H, W, 3] and extract AUDIO if available.
    
    Returns (video_frames_tensor, audio_dict_or_none, frame_count, fps, duration).
    Supports PyAV (av), OpenCV (cv2), or imageio based on what is installed.
    """
    frames = []
    fps = 24.0
    audio_dict = None

    # Try PyAV (best quality & audio support)
    try:
        import av

        container = av.open(file_path)
        video_stream = container.streams.video[0] if container.streams.video else None
        if video_stream is not None:
            if video_stream.average_rate:
                fps = float(video_stream.average_rate)

            frame_idx = 0
            loaded_count = 0
            for frame in container.decode(video=0):
                if frame_idx < skip_first_frames:
                    frame_idx += 1
                    continue
                if (frame_idx - skip_first_frames) % max(1, select_every_nth) != 0:
                    frame_idx += 1
                    continue

                img = frame.to_image().convert("RGB")
                img_np = np.array(img).astype(np.float32) / 255.0
                frames.append(img_np)
                loaded_count += 1
                frame_idx += 1

                if frame_load_cap > 0 and loaded_count >= frame_load_cap:
                    break

        # Check audio stream
        if container.streams.audio:
            try:
                container.seek(0)
                audio_frames = []
                sample_rate = 44100
                for a_frame in container.decode(audio=0):
                    sample_rate = a_frame.rate
                    # shape: [channels, samples]
                    audio_frames.append(a_frame.to_ndarray())
                if audio_frames:
                    audio_arr = np.concatenate(audio_frames, axis=-1).astype(np.float32)
                    # Normalize if int16
                    if audio_arr.dtype == np.int16:
                        audio_arr = audio_arr / 32768.0
                    if torch is not None:
                        audio_tensor = torch.from_numpy(audio_arr)
                        if audio_tensor.ndim == 1:
                            audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
                        elif audio_tensor.ndim == 2:
                            audio_tensor = audio_tensor.unsqueeze(0)
                        audio_dict = {"waveform": audio_tensor, "sample_rate": sample_rate}
                    else:
                        audio_dict = {"waveform": audio_arr, "sample_rate": sample_rate}
            except Exception:
                audio_dict = None

        container.close()

    except ImportError:
        # Fallback to OpenCV
        try:
            import cv2

            cap = cv2.VideoCapture(file_path)
            fps_val = cap.get(cv2.CAP_PROP_FPS)
            if fps_val > 0:
                fps = float(fps_val)

            frame_idx = 0
            loaded_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx < skip_first_frames:
                    frame_idx += 1
                    continue
                if (frame_idx - skip_first_frames) % max(1, select_every_nth) != 0:
                    frame_idx += 1
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                frames.append(frame_rgb)
                loaded_count += 1
                frame_idx += 1

                if frame_load_cap > 0 and loaded_count >= frame_load_cap:
                    break
            cap.release()
        except ImportError:
            # Fallback to imageio
            try:
                import imageio

                reader = imageio.get_reader(file_path)
                meta = reader.get_meta_data()
                fps = float(meta.get("fps", 24.0))
                frame_idx = 0
                loaded_count = 0
                for img in reader:
                    if frame_idx < skip_first_frames:
                        frame_idx += 1
                        continue
                    if (frame_idx - skip_first_frames) % max(1, select_every_nth) != 0:
                        frame_idx += 1
                        continue
                    frames.append(np.array(img).astype(np.float32) / 255.0)
                    loaded_count += 1
                    frame_idx += 1
                    if frame_load_cap > 0 and loaded_count >= frame_load_cap:
                        break
                reader.close()
            except Exception as e:
                raise RuntimeError(
                    f"Could not load video '{file_path}'. Ensure av (PyAV), opencv-python, or imageio is installed. Error: {e}"
                )

    if not frames:
        raise ValueError(f"No frames could be extracted from video file {file_path}")

    # Calculate duration
    duration = float(len(frames)) / max(fps, 0.001)

    # Optional trim last X seconds
    if trim_last_seconds > 0.0 and duration > trim_last_seconds:
        frames_to_keep = max(1, int(round(fps * trim_last_seconds)))
        frames = frames[-frames_to_keep:]
        duration = float(len(frames)) / max(fps, 0.001)

        if audio_dict and "waveform" in audio_dict:
            wf = audio_dict["waveform"]
            sr = audio_dict.get("sample_rate", 44100)
            samples_to_keep = int(sr * trim_last_seconds)
            if hasattr(wf, "shape") and wf.shape[-1] > samples_to_keep:
                audio_dict["waveform"] = wf[..., -samples_to_keep:]

    frames_np = np.stack(frames, axis=0)
    if torch is not None:
        frames_tensor = torch.from_numpy(frames_np)
    else:
        frames_tensor = frames_np

    return frames_tensor, audio_dict, len(frames), fps, duration
