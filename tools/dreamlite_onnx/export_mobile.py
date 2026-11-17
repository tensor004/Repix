#!/usr/bin/env python3
"""Export DreamLite-Mobile UNet and TinyVAE to ONNX for Android feasibility tests.

This is an export/parity spike, not an Android runtime. It intentionally exports
FP32 first so numerical parity can be checked before trying FP16/QNN.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from torch import nn

class VAEEncoder(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae
    def forward(self, image):
        return self.vae.encode(image, return_dict=False)[0]

class VAEDecoder(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae
    def forward(self, latent):
        return self.vae.decode(latent, return_dict=False)[0]

class UNetWrapper(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
    def forward(self, sample, timestep, encoder_hidden_states,
                encoder_attention_mask, time_ids):
        return self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            added_cond_kwargs={"time_ids": time_ids},
            return_dict=False,
        )[0]

def export_one(module, args, path, input_names, output_name, dynamic_axes=None):
    module.eval()
    with torch.inference_mode():
        expected = module(*args).detach().cpu()
        torch.onnx.export(
            module, args, str(path), export_params=True, opset_version=17,
            do_constant_folding=True, input_names=input_names,
            output_names=[output_name], dynamic_axes=dynamic_axes, dynamo=False,
        )
    return expected

def validate_onnx(path, inputs, expected, atol, rtol):
    import onnx
    import onnxruntime as ort
    onnx.checker.check_model(str(path))
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feed = {meta.name: tensor.detach().cpu().numpy()
            for meta, tensor in zip(session.get_inputs(), inputs)}
    actual = torch.from_numpy(session.run(None, feed)[0])
    error = (actual - expected).abs()
    return {
        "passed": bool(torch.allclose(actual, expected, atol=atol, rtol=rtol)),
        "max_abs_error": float(error.max()),
        "mean_abs_error": float(error.mean()),
        "output_shape": list(actual.shape),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="models/DreamLite-mobile")
    parser.add_argument("--output-dir", default="artifacts/dreamlite-onnx")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--atol", type=float, default=2e-3)
    parser.add_argument("--rtol", type=float, default=2e-3)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable. Use --device cpu.")
    device = torch.device(args.device)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    from diffusers.models import AutoencoderTiny
    from dreamlite.models import DreamLiteUNetModel
    print(f"Loading only VAE and UNet from {args.model_id} on {device} (FP32 baseline)...", flush=True)
    # Load the two components directly. The Qwen3-VL encoder is not needed for
    # these exports, so it is never downloaded, instantiated, or placed in RAM.
    vae = AutoencoderTiny.from_pretrained(
        args.model_id, subfolder="vae", torch_dtype=torch.float32
    ).to(device).eval()
    unet = DreamLiteUNetModel.from_pretrained(
        args.model_id, subfolder="unet", torch_dtype=torch.float32
    ).to(device).eval()
    for model in (vae, unet):
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    report = {
        "model_id": args.model_id, "device": str(device), "opset": 17,
        "export_dtype": "float32", "contracts": {}, "validation": {},
        "note": "Encoder hidden-state export is separate; these graphs alone cannot perform edits.",
    }

    image = torch.zeros((1, 3, 1024, 1024), dtype=torch.float32, device=device)
    enc, enc_path = VAEEncoder(vae).to(device).eval(), out / "vae_encoder.onnx"
    enc_args = (image,)
    enc_expected = export_one(enc, enc_args, enc_path, ["image"], "latent")
    report["contracts"]["vae_encoder"] = {
        "inputs": {"image": [1, 3, 1024, 1024]},
        "output": list(enc_expected.shape), "file": enc_path.name,
    }

    latent = torch.zeros((1, 4, 128, 128), dtype=torch.float32, device=device)
    dec, dec_path = VAEDecoder(vae).to(device).eval(), out / "vae_decoder.onnx"
    dec_args = (latent,)
    dec_expected = export_one(dec, dec_args, dec_path, ["latent"], "image")
    report["contracts"]["vae_decoder"] = {
        "inputs": {"latent": [1, 4, 128, 128]},
        "output": list(dec_expected.shape), "file": dec_path.name,
    }

    sample = torch.zeros((1, 4, 128, 256), dtype=torch.float32, device=device)
    timestep = torch.tensor([500.0], dtype=torch.float32, device=device)
    seq = 200
    hidden = torch.zeros((1, seq, 2048), dtype=torch.float32, device=device)
    mask = torch.ones((1, seq), dtype=torch.long, device=device)
    time_ids = torch.tensor([[1024.0, 1024.0]], dtype=torch.float32, device=device)
    unet, unet_path = UNetWrapper(unet).to(device).eval(), out / "unet.onnx"
    unet_args = (sample, timestep, hidden, mask, time_ids)
    unet_expected = export_one(
        unet, unet_args, unet_path,
        ["sample", "timestep", "encoder_hidden_states",
         "encoder_attention_mask", "time_ids"],
        "noise_pred",
        dynamic_axes={
            "encoder_hidden_states": {1: "sequence_length"},
            "encoder_attention_mask": {1: "sequence_length"},
        },
    )
    report["contracts"]["unet"] = {
        "inputs": {
            "sample": [1, 4, 128, 256], "timestep": [1],
            "encoder_hidden_states": [1, "sequence_length", 2048],
            "encoder_attention_mask": [1, "sequence_length"],
            "time_ids": [1, 2],
        },
        "output": list(unet_expected.shape), "file": unet_path.name,
    }

    if args.validate:
        print("Validating exported graphs with ONNX Runtime CPU...", flush=True)
        report["validation"]["vae_encoder"] = validate_onnx(
            enc_path, enc_args, enc_expected, args.atol, args.rtol)
        report["validation"]["vae_decoder"] = validate_onnx(
            dec_path, dec_args, dec_expected, args.atol, args.rtol)
        report["validation"]["unet"] = validate_onnx(
            unet_path, unet_args, unet_expected, args.atol, args.rtol)

    (out / "export_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"\nExport artifacts: {out.resolve()}")

if __name__ == "__main__":
    main()
