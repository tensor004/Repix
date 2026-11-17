# DreamLite-Mobile ONNX export spike

This is a desktop export and CPU-parity harness, not Android integration. It loads only the official DreamLite-Mobile UNet and TinyVAE subfolders, then exports them as separate ONNX graphs. It deliberately does not load the Qwen3-VL encoder. FP32 is the correctness baseline before FP16 conversion or Qualcomm QNN compilation.

## Environment

Use Python 3.10 or 3.11. Clone the official DreamLite source repository and install its requirements plus the export dependencies:

\`\`\`bash
git clone https://github.com/ByteVisionLab/DreamLite.git
git clone -b feature/dreamlite-android-integration https://github.com/tensor004/Repix.git
cd DreamLite
mkdir -p tools/dreamlite_onnx
cp -r ../Repix/tools/dreamlite_onnx/. tools/dreamlite_onnx/
pip install -r requirements.txt
pip install onnx onnxruntime huggingface_hub
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="carlofkl/DreamLite-mobile",
    revision="main",
    allow_patterns=["unet/*", "vae/*"],
    local_dir="models/DreamLite-mobile",
)
PY
\`\`\`

Review DreamLite's WEIGHTS_LICENSE and the Hugging Face model terms before downloading or using weights.

## Run

From the DreamLite source repository root, copy the RePix exporter folder into tools/dreamlite_onnx/ (or run it from a checkout where that path exists):

\`\`\`bash
python tools/dreamlite_onnx/export_mobile.py --model-id models/DreamLite-mobile --output-dir artifacts/dreamlite-onnx --device cpu --validate
\`\`\`

For a CUDA desktop, use --device cuda. Export still uses FP32.

## Graph contracts

| Graph | Input | Expected output |
|---|---|---|
| VAE encoder | [1,3,1024,1024] | [1,4,128,128] |
| VAE decoder | [1,4,128,128] | [1,3,1024,1024] |
| UNet | sample [1,4,128,256], timestep [1], hidden [1,L,2048], mask [1,L], time IDs [1,2] | [1,4,128,256] |

The UNet sequence axis is dynamic. L=200 is only the export/parity example input.

## Limitations

- The text/image encoder is not loaded or exported. Editing still requires Qwen3-VL's image-conditioned final hidden-state tensor, not generated chat text. This is intentionally the next independent export/runtime task.
- This validates exported graphs against PyTorch on the same desktop. It does not establish QNN support, Android performance, memory use, or Nothing Phone 3a Pro compatibility.
- Zero-valued smoke-test tensors are used for export. A later parity stage must test representative non-zero images/latents, preprocessing, latent scaling, and scheduler behavior.
- FP32 is a correctness baseline. QNN deployment needs a separate FP16/quantization and operator-support pass.
- No model weights are included in RePix.

## Phone-friendly export option

A Google Colab notebook is included for the real weight export. Open it in Colab, select a T4 GPU runtime, run the cells in order, and keep the resulting ZIP private:

[Open RePix DreamLite ONNX export notebook in Google Colab](https://colab.research.google.com/github/tensor004/Repix/blob/feature/dreamlite-android-integration/tools/dreamlite_onnx/RePix_DreamLite_ONNX_Export.ipynb)

Colab is used only for this one-time model conversion and validation. The exported ONNX graphs are intended for later offline Android deployment. A free GPU runtime is not guaranteed; if unavailable, the export must be run on another CUDA-capable computer.
