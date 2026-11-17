# DreamLite-Mobile ONNX export spike

This is a desktop export and CPU-parity harness, not Android integration. It exports the official DreamLite-Mobile UNet and TinyVAE as separate ONNX graphs. FP32 is deliberate: establish correctness before attempting FP16 conversion or Qualcomm QNN compilation.

## Environment

Use Python 3.10 and the DreamLite repository's main implementation/weight branch. Install its requirements, then:

\`\`\`bash
pip install onnx onnxruntime
hf download carlofkl/DreamLite-mobile --revision main --local-dir models/DreamLite-mobile
\`\`\`

Review DreamLite's WEIGHTS_LICENSE and the Hugging Face model terms before downloading or using weights.

## Run

From the DreamLite source repository root, after placing this folder under tools/dreamlite_onnx/:

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

- The text/image encoder is not exported. Editing requires Qwen3-VL's image-conditioned final hidden-state tensor, not generated chat text.
- This validates exported graphs against PyTorch on the same desktop. It does not establish QNN support, Android performance, memory use, or Nothing Phone 3a Pro compatibility.
- Zero-valued smoke-test tensors are used for export. A later parity stage must test representative non-zero images/latents, preprocessing, latent scaling, and scheduler behavior.
- FP32 is a correctness baseline. QNN deployment needs a separate FP16/quantization and operator-support pass.
- No model weights are included in RePix.
