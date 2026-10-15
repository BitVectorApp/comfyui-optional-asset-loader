# ComfyUI Optional Asset Loader

A lightweight suite of custom ComfyUI nodes for handling **optional and nullable image, video, and audio assets** without workflow crashes or sprawling spaghetti workarounds.

Designed specifically for cloud platforms ([**Graydient**](https://graydient.ai), RunPod, ComfyUI API) and models that support nullable input slots (such as **MiniMax**, **Qwen Image 2.1**, and custom diffusion models).

**NOTE:**

This will only work for ComfyUI nodes that can take *optional* inputs, such as MiniMax's Reference-to-Video node, or Qwen 2.1 Image's Text Encoder node. These are usually represented with the "connector" dots having a dark inner middle to them - and don't cause a workflow execution error when the node isn't connected.

These nodes will output blank tensors to connected nodes when the input isn't found, which might allow Comfy to "execute" the workflow, but if the input node isn't meant to take one, it could cause runtime issues.

### AI disclaimer:

This ComfyUI node pack was fully created via the usage of AI (Gemini 3.8 Flash), by taking a workflow I created to solve this issue (and the mess of spaghetti it required) into a condensed set of nodes.

Feel free to use in commercial contexts, personal, or any other context. It is MIT licensed, at least to the extent that fully AI derived code can be licensed.

---

## The Problem

On platforms like Graydient or web-based ComfyUI APIs, workflows often need to accept a variable number of reference images or videos (e.g. 0 to 4 images, optional reference video). However:

1. Standard loaders (`LoadImage`, `VHS_LoadVideoPath`) throw exceptions and halt execution if a file path is empty (`""`), missing, or invalid.
2. To work around this, authors historically had to create an elaborate "hack":
   - Download a placeholder dummy file (e.g. `blank_start.png`, `blank_start.mp4`) with magic dimensions (e.g. width `91px` or `16px`).
   - Prepend folder paths with `StringConcatenate` and `StringCompare`.
   - Load the dummy file through WAS Suite `Image Load`.
   - Inspect the dimension with `GetImageSize`.
   - Check `width > 91` with `easy compare`.
   - Route through `ComfySwitchNode`.
   - Feed `on_false` from an unconnected slot on an empty `Context (rgthree)` node just to produce a Python `None`!

**This required ~10 nodes per reference input (~50 nodes for 4 images + 1 video).**

---

## The Solution

In ComfyUI, nodes can simply return Python `None`. When an output of `None` is connected to a downstream model slot (like MiniMax `ref_image_0` or Qwen reference inputs), the model cleanly skips that reference asset!

With **ComfyUI Optional Asset Loader**, you replace that entire 10-node chain with a single node:

```
[Graydient Input String] ──► [Load Image (Optional)] ──► MiniMax / Qwen (ref_image_0)
                                      │
                                      └──► has_image (BOOLEAN)
```

If the user provides an image, it loads normally. If the user leaves it blank or passes a fallback name (like `blank_start.png`, `none`, or empty string `""`), the node safely outputs `None` (or a blank tensor), and `has_image = False`.

---

## Included Nodes

### 1. `Load Image (Optional / Fallback)` (`LoadImageOptional`)
Loads an image file if available; otherwise returns `None` (or a blank tensor) safely.

* **Category:** `optional_loaders`
* **Inputs:**
  * `image_path` (`STRING`): Filename (auto-checked in ComfyUI's `input/` folder), relative path (`input/foo.png`), absolute path, or empty string.
  * `fallback_patterns` (`STRING`, default: `"blank_start.png, none, null, empty, undefined"`): Comma-separated list of strings or wildcard patterns (e.g. `*blank*`) to treat as "no asset provided".
  * `missing_behavior` (`COMBO`):
    * `Return None` (default): Outputs Python `None` on the `IMAGE` and `MASK` ports. Downstream models that accept optional inputs cleanly ignore it.
    * `Return Blank Tensor`: Outputs a valid zero/blank tensor of size `[1, blank_height, blank_width, 3]` and blank mask (for downstream nodes that strictly require a tensor).
    * `Raise Error`: Re-raises `FileNotFoundError` if strict validation is desired.
  * `blank_width` / `blank_height` (`INT`, default: `512`): Dimensions for the blank tensor if `Return Blank Tensor` is selected.
  * `blank_color` (`COMBO`: `["black", "white", "gray"]`): Fill color for the blank tensor.
* **Outputs:**
  * `IMAGE` (`IMAGE`): The loaded image tensor `[B, H, W, 3]`, `None`, or blank tensor.
  * `MASK` (`MASK`): Alpha mask `[B, H, W]`, `None`, or blank mask.
  * `has_image` (`BOOLEAN`): `True` if a valid user image was loaded, `False` if fallback or missing.
  * `resolved_path` (`STRING`): Absolute file path on disk, or `""` if missing.

---

### 2. `Load Video (Optional / Fallback)` (`LoadVideoOptional`)
Loads video frames and audio if available; otherwise returns `None` safely.

* **Category:** `optional_loaders`
* **Inputs:**
  * `video_path` (`STRING`): Video filename or path.
  * `fallback_patterns` (`STRING`, default: `"blank_start.mp4, none, null, empty, undefined"`).
  * `missing_behavior` (`COMBO`: `["Return None", "Return Blank Tensor", "Raise Error"]`).
  * `frame_load_cap` (`INT`, default: `0`): Maximum frames to load (`0` = all).
  * `skip_first_frames` (`INT`, default: `0`): Skip initial frames.
  * `select_every_nth` (`INT`, default: `1`): Frame sampling rate.
  * `trim_last_seconds` (`FLOAT`, default: `0.0`): Automatically trim the video and audio to the last X seconds directly in memory (`0.0` = disabled).
  * `blank_width` / `blank_height` / `blank_frames` (`INT`): Dimensions for blank video tensor mode.
  * `default_fps` (`FLOAT`, default: `24.0`): Fallback frame rate returned when blank or missing to prevent division by zero in downstream video math nodes.
* **Outputs:**
  * `IMAGE` (`IMAGE`): Video frames tensor `[N, H, W, 3]` or `None`.
  * `AUDIO` (`AUDIO`): Audio dictionary `{"waveform": ..., "sample_rate": ...}` or `None`.
  * `has_video` (`BOOLEAN`): `True` if valid video loaded, `False` otherwise.
  * `frame_count` (`INT`): Number of frames loaded (or `blank_frames` when blank tensor mode).
  * `fps` (`FLOAT`): Frame rate (or `default_fps`).
  * `duration` (`FLOAT`): Duration in seconds (`frame_count / fps`).
  * `resolved_path` (`STRING`): Absolute file path on disk, or `""`.

---

### 3. `Optional Dimension Filter` (`OptionalDimensionFilter`)
Drop-in replacement for existing workflows using the dimension comparison trick (`GetImageSize` -> check width `> 91` -> `ComfySwitchNode`).

* **Category:** `optional_loaders`
* **Inputs:**
  * `image` (`IMAGE`): Image tensor to check.
  * `dimension_target` (`COMBO`: `["width", "height", "either_dimension", "both_dimensions"]`).
  * `comparison` (`COMBO`: `["greater_than", "greater_or_equal", "less_than", "less_or_equal", "equal", "not_equal"]`).
  * `threshold` (`INT`, default: `91`): The cutoff value.
  * `fallback_behavior` (`COMBO`: `["Return None", "Return Blank Tensor"]`).
* **Outputs:**
  * `IMAGE` (`IMAGE`): Kept image, or `None` / blank tensor if it matched the placeholder dimension.
  * `passed` (`BOOLEAN`): `True` if it passed the threshold condition, `False` if dropped.
  * `width` / `height` (`INT`): Dimensions of the image.

---

### 4. `Optional Image Gate` (`OptionalImageGate`)
Passes an existing image and mask through if `is_active` is True; otherwise drops them to `None` (or blank).

* **Inputs:** `is_active` (`BOOLEAN`), `image` (`IMAGE`, optional), `mask` (`MASK`, optional), `fallback_behavior`.
* **Outputs:** `IMAGE`, `MASK`, `has_image` (`BOOLEAN`).

---

### 5. `Optional Video Gate` (`OptionalVideoGate`)
Passes video frames (`IMAGE`) and `AUDIO` through if `is_active` is True; otherwise drops them to `None` (or blank).

* **Inputs:** `is_active` (`BOOLEAN`), `video_frames` (`IMAGE`, optional), `audio` (`AUDIO`, optional), `fallback_behavior`.
* **Outputs:** `IMAGE`, `AUDIO`, `has_video` (`BOOLEAN`).

---

### 6. `Nullable Constant (None)` (`NullableConstant`)
Outputs a pure Python `None` value for wildcard `*` connections. Completely replaces the trick of using an unconnected slot from an empty `Context (rgthree)` node.

* **Outputs:** `none_value` (`*` = `None`).

---

### 7. `Boolean Logic` (`BooleanLogic`)
Performs boolean logic (`AND`, `OR`, `XOR`, `NOT_A`) on two inputs. Ideal for combining workflow triggers (e.g. `has_video AND should_combine`).

* **Inputs:** `operation` (`COMBO`), `a` (`BOOLEAN`), `b` (`BOOLEAN`).
* **Outputs:** `result` (`BOOLEAN`).

---

## Graydient Integration Guide

When using Graydient:
1. In Graydient's parameter settings, map the input field to the `image_path` string of the `Load Image (Optional)` node (or right-click `image_path` -> *Convert to Input* to attach a `PrimitiveString`).
2. You can leave the default value in Graydient as `""` (empty) or keep `blank_start.png`. Because `blank_start.png` is in the node's `fallback_patterns` list, the node automatically recognizes that no image was submitted!
3. You **no longer need** to download any placeholder image into Graydient's `Models` tab.
4. Wire the `IMAGE` output of `Load Image (Optional)` directly into your model's reference slot (e.g. MiniMax `ref_images.ref_image_0` or Qwen reference inputs).
5. If your workflow does branching based on whether a video was submitted, connect the `has_video` output directly into your switch or execution bypass node.

---

## Installation

### Method 1: Git Clone
Clone this repository into your ComfyUI `custom_nodes` folder:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/BitVectorApp/comfyui-optional-asset-loader.git
```

### Dependencies
All required dependencies (`torch`, `numpy`, `pillow`) are already included in any standard ComfyUI environment.

Optional for video loading:
```bash
pip install av
# or
pip install opencv-python
```

---

## Running Tests

This repository includes a full test suite:

```bash
python3 -m pytest -v
```

---

## License

MIT License. Feel free to use in personal, community, and commercial workflows!
