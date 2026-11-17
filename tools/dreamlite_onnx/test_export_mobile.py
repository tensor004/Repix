"""Fast unit tests for the DreamLite ONNX wrapper contracts."""
import importlib.util
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
        assert kwargs["added_cond_kwargs"]["time_ids"].shape == (1, 2)
        return (kwargs["sample"] * 0.25,)


class ExportWrapperTests(unittest.TestCase):
    def test_vae_encoder_contract(self):
        model = EXPORT.VAEEncoder(DummyVAE())
        image = torch.zeros(1, 3, 64, 64)
        latent = model(image)
        self.assertEqual(tuple(latent.shape), (1, 4, 8, 8))

    def test_vae_decoder_contract(self):
        model = EXPORT.VAEDecoder(DummyVAE())
        latent = torch.zeros(1, 4, 8, 8)
        image = model(latent)
        self.assertEqual(tuple(image.shape), (1, 3, 64, 64))

    def test_unet_contract_and_conditioning_forward(self):
        model = EXPORT.UNetWrapper(DummyUNet())
        sample = torch.zeros(1, 4, 8, 16)
        timestep = torch.tensor([1.0])
        hidden = torch.zeros(1, 23, 2048)
        mask = torch.ones(1, 23, dtype=torch.long)
        time_ids = torch.tensor([[1024.0, 1024.0]])
        output = model(sample, timestep, hidden, mask, time_ids)
        self.assertEqual(tuple(output.shape), (1, 4, 8, 16))
        self.assertTrue(torch.equal(output, sample * 0.25))


if __name__ == "__main__":
    unittest.main()
