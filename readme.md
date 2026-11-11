## About

為 ComfyUI 提供支援 LTX-Video / GGUF 量化擴展模型的載入節點。支援動態裝置分配（CUDA / Intel XPU / CPU）以及自訂反量化精度。

---

## 🌟 特色功能 (Features)

* **廣泛格式相容**：支援標準 `.safetensors` Diffusion 模型與 `.gguf` 量化模型。
* **硬體跨平台支援**：
* 自動偵測系統硬體（`auto`）。
* 支援 NVIDIA 多 GPU 指定（`cuda:0`, `cuda:1`, ...）。
* 支援 Intel Arc / Data Center GPU（`xpu:0`, `xpu:1`, ...）。
* 支援純 CPU 模式（`cpu`）。


* **進階量化控制**：支援自訂 Dequant 與 Patch 精度（`default`, `target`, `float16`, `bfloat16`, `float32`）。
* **LTX-Video 架構修補**：自動適配 LTX 模型的 AdaLN、Connector 與 Gated Attention 結構。


## ⚙️ 節點參數說明 (Parameters)

| 參數名稱 (`Input`) | 類型 | 說明 |
| --- | --- | --- |
| **`unet_name`** | Combo | 選擇要載入的模型（支援 `diffusion_models`、`unet`、`checkpoints`、`unet_gguf` 目錄）。 |
| **`device`** | Combo | 選擇運算裝置：`auto`、`cpu`、`cuda:X` 或 `xpu:X`。 |
| **`dequant_dtype`** | Combo | 反量化計算精度（`default` / `target` / `float32` / `float16` / `bfloat16`）。 |
| **`patch_dtype`** | Combo | 權重補丁 (Patch) 運算精度。 |
| **`patch_on_device`** | Boolean | 是否直接在目標裝置上套用 LoRA / Patches（開啟可節省 Host RAM，但佔用 VRAM）。 |

---

## 🖥️ Intel XPU 使用提示 (Intel Arc Notes)

若要啟用 Intel XPU 支援，請確保：

1. 已安裝適用於 Intel GPU 的 PyTorch 擴展（PyTorch 2.4+ 已原生整合 `torch.xpu` 或需安裝 Intel Extension for PyTorch `ipex`）。
2. 在終端機執行下列指令驗證：
```bash
python -c "import torch; print(hasattr(torch, 'xpu') and torch.xpu.is_available())"

```


顯示 `True` 即可在節點的 `device` 下拉選單中看到 `xpu:0` 等選項。

---

# ComfyUI-LTX-GGUF-Loader

A flexible diffusion model loader for ComfyUI with support for LTX-Video, GGUF quants, multi-GPU selection, and Intel XPU acceleration.

## Features

* **Dual Support**: Seamlessly loads both `.safetensors` and `.gguf` diffusion checkpoints.
* **Hardware Agnostic**: Supports NVIDIA CUDA, Intel XPU, and CPU execution.
* **Device Selection**: Dynamic detection of available devices (`cuda:0`, `cuda:1`, `xpu:0`, etc.).
* **Fine-grained Precision Control**: Configurable dequantization and patching dtypes.

## Requirements

* ComfyUI
* PyTorch with CUDA or XPU support

## Node Category

* **Category**: `loaders/ltx_gguf`
* **Node Title**: `LTX / GGUF Diffusion Model Loader`
