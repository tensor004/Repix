"""Fast unit tests for DreamLite ONNX wrappers and export parity helpers."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

import torch

SCRIPT = Path(__file__).with_name("export_mobile.py")
SPEC = importlib.util.spec_from_file_location("export_mobile", SCRIPT)
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)


class DummyVAE:
    def encode(self, image, return_dict=True):
        assert return_dict is False
        return (image[:, :1, ::8, ::8].repeat(1, 4, 1, 1),)

    def decode(self, latent, return_dict=True):
        assert return_dict is False
        return (latent[:, :3].repeat_interleave(8, 2).repeat_interleave(8, 3),)


class DummyUNet:
    def __call__(self, **kwargs):
        assert kwargs["return_dict"] is False
        value = (
            kwargs["encoder_hidden_states"].mean()
            + kwargs["encoder_attention_mask"].float().mean()
            + kwargs["timestep"].mean()
            + kwargs["added_cond_kwargs"]["time_ids"].mean()
        )
        return (kwargs["sample"] * 0.25 + value,)


class ExportWrapperTests(unittest.TestCase):
    def test_vae_encoder_contract(self):
        model = EXPORT.VAEEncoder(DummyVAE())
        latent = model(torch.zeros(1, 3, 64, 64))
        self.assertEqual(tuple(latent.shape), (1, 4, 8, 8))

    def test_vae_decoder_contract(self):
        model = EXPORT.VAEDecoder(DummyVAE())
        image = model(torch.zeros(1, 4, 8, 8))
        self.assertEqual(tuple(image.shape), (1, 3, 64, 64))

    def test_unet_contract_and_conditioning_forward(self):
        model = EXPORT.UNetWrapper(DummyUNet())
        sample = torch.zeros(1, 4, 8, 16)
        output = model(
            sample, torch.tensor([1.0]), torch.zeros(1, 23, 2048),
            torch.ones(1, 23, dtype=torch.long), torch.tensor([[1024.0, 1024.0]]),
        )
        self.assertEqual(tuple(output.shape), (1, 4, 8, 16))
        self.assertTrue(torch.isfinite(output).all())

    def test_onnx_roundtrip_and_dynamic_sequence(self):
        # Exercise the same export and ORT comparison helpers without model weights.
        model = EXPORT.UNetWrapper(DummyUNet()).eval()
        sample = torch.zeros(1, 4, 8, 16)
        timestep = torch.tensor([2.0])
        hidden = torch.ones(1, 12, 2048)
        mask = torch.ones(1, 12, dtype=torch.long)
        time_ids = torch.tensor([[1024.0, 1024.0]])
        args = (sample, timestep, hidden, mask, time_ids)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny_unet.onnx"
            expected = EXPORT.export_one(
                model, args, path,
                ["sample", "timestep", "encoder_hidden_states",
                 "encoder_attention_mask", "time_ids"],
                "noise_pred",
                dynamic_axes={
                    "encoder_hidden_states": {1: "sequence_length"},
                    "encoder_attention_mask": {1: "sequence_length"},
                },
            )
            result = EXPORT.validate_onnx(path, args, expected, 1e-4, 1e-4)
            self.assertTrue(result["passed"], result)

            # A different sequence length verifies that the exported axis is dynamic.
            hidden_2 = torch.ones(1, 7, 2048) * 0.5
            mask_2 = torch.ones(1, 7, dtype=torch.long)
            args_2 = (sample, timestep, hidden_2, mask_2, time_ids)
            expected_2 = model(*args_2).detach().cpu()
            result_2 = EXPORT.validate_onnx(path, args_2, expected_2, 1e-4, 1e-4)
            self.assertTrue(result_2["passed"], result_2)


if __name__ == "__main__":
    unittest.main()
