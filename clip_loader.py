import os
import torch
import folder_paths
import comfy.sd
import comfy.model_management as model_management


def get_available_compute_devices():
    devices = ["auto", "cpu"]

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            devices.append(f"cuda:{i}")

    has_xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
    if has_xpu:
        for i in range(torch.xpu.device_count()):
            devices.append(f"xpu:{i}")

    return devices


def resolve_torch_device(device_selection):
    if device_selection == "auto":
        return model_management.text_encoder_device()
    return torch.device(device_selection)


def get_clip_filenames():
    files = set()
    for folder in ["clip", "text_encoders", "checkpoints"]:
        try:
            for f in folder_paths.get_filename_list(folder):
                files.add(f)
        except Exception:
            pass
    return sorted(list(files))


def resolve_file_path(filename, search_folders):
    for folder in search_folders:
        try:
            p = folder_paths.get_full_path(folder, filename)
            if p and os.path.exists(p):
                return p
        except Exception:
            pass
    if os.path.exists(filename):
        return filename
    raise FileNotFoundError(f"File not found: {filename}")


def resolve_clip_type(clip_type_name):
    type_map = {
        "ltxv": getattr(comfy.sd.CLIPType, "LTXV", comfy.sd.CLIPType.SD3),
        "sd3": getattr(comfy.sd.CLIPType, "SD3", comfy.sd.CLIPType.SD3),
        "flux": getattr(comfy.sd.CLIPType, "FLUX", comfy.sd.CLIPType.STABLE_DIFFUSION),
        "stable_diffusion": comfy.sd.CLIPType.STABLE_DIFFUSION,
    }
    return type_map.get(clip_type_name, comfy.sd.CLIPType.STABLE_DIFFUSION)


class LTX_Gemma_CLIP_Loader:
    @classmethod
    def INPUT_TYPES(s):
        clip_list = get_clip_filenames()
        align_list = ["none"] + clip_list
        return {
            "required": {
                # 放在同一個 required 字典內，保證 UI 順序由上而下
                "clip_name": (clip_list, {"tooltip": "主文字編碼器 (e.g. Gemma, T5, etc.)"}),
                "align_connector": (align_list, {"default": "none", "tooltip": "可選的 LTX 投影對齊模型 (Connector / Alignment Weights)"}),
                "device": (get_available_compute_devices(), {"default": "auto", "tooltip": "運算裝置 (Auto / CPU / CUDA / XPU)"}),
                "clip_type": (["ltxv", "sd3", "flux", "stable_diffusion"], {"default": "ltxv"}),
            }
        }

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "load_clip"
    CATEGORY = "loaders/ltx_gguf"
    TITLE = "LTX / Gemma CLIP Loader (XPU/GPU)"

    def load_clip(self, clip_name, align_connector="none", device="auto", clip_type="ltxv"):
        clip_path = resolve_file_path(clip_name, ["clip", "text_encoders", "checkpoints"])
        selected_clip_type = resolve_clip_type(clip_type)

        ckpt_paths = [clip_path]
        if align_connector and align_connector != "none":
            connector_path = resolve_file_path(align_connector, ["clip", "text_encoders", "checkpoints"])
            ckpt_paths.append(connector_path)

        # 載入並合併主模型與 connector 權重至同一模型實例
        clip = comfy.sd.load_clip(
            ckpt_paths=ckpt_paths,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=selected_clip_type,
        )

        # 正確綁定動態調度裝置
        if device != "auto":
            target_device = resolve_torch_device(device)
            clip.load_device = target_device
            if hasattr(clip, "patcher") and clip.patcher is not None:
                clip.patcher.load_device = target_device
                # 當手動指定運算裝置時，確保 offload 回到 CPU 以便釋放該顯存
                clip.patcher.offload_device = model_management.text_encoder_offload_device()

        return (clip,)
