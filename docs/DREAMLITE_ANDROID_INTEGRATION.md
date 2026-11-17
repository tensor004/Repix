# DreamLite Android Integration: Technical Plan

Status: feasibility and architecture investigation
Target branch: `feature/dreamlite-android-integration`
Default branch: `main` (untouched)

## Goal

Add a separate, fully local image-to-image editing engine to PrivateLM/RePix. The user selects a photo, enters a natural-language instruction, receives an edited image, can compare before/after, and saves the result. Existing chat inference and Stable Diffusion text-to-image features must remain unchanged.

## Verified upstream facts

- The official DreamLite project describes a 0.39B unified diffusion network for text-to-image and text-guided image editing. Its mobile variant is distilled for four denoising steps at 1024x1024.
- The official inference path accepts an optional source image. In edit mode, the image is encoded to a latent and supplied as conditioning alongside the prompt embeddings.
- The official mobile deployment guide is an iOS implementation: Core ML FP16 UNet/VAE plus a 4-bit Qwen3-VL encoder running through MLX. It requires a macOS export/quantization workflow and lists iPhone 16 Pro / Pro Max or later with 8GB+ RAM.
- The official code is Apache-2.0, but the model weights have a separate CC BY-NC 4.0 license and additional research/non-commercial and use restrictions. Code licensing does not grant broader rights to the weights.
- The community `xay2001/dreamlite-android` ONNX package is explicitly a text-to-image experiment. Its model card says the 2B text/image encoder is not included, prompt conditioning is fixed in its test APK, and it does not provide arbitrary prompt entry or photo editing. Its CPU checks do not establish Android GPU/NPU compatibility or speed.

## RePix integration points

- RePix is a Flutter application with a native Android host in `android/app/src/main/kotlin/com/orailnoor/privatelm/MainActivity.kt`.
- Existing image generation is provided by `local_plugins/sd_flutter_android` and `LocalImageService`. That path is Stable-Diffusion-specific and should not be modified to pretend it supports DreamLite.
- Add DreamLite as an independent engine behind a Flutter service and a dedicated Android platform channel/native module. Keep model files out of the APK; download and validate them into app-private storage only when the user chooses to install the model.

## Proposed architecture

```
Flutter editor UI
  ├── image picker / prompt / progress
  ├── before-after preview / save
  └── DreamLiteEditingService
        └── MethodChannel: privatelm/dreamlite
              └── Android DreamLite runtime
                    ├── model manifest + download / checksum
                    ├── image preprocessing + VAE encoder
                    ├── Qwen3-VL image+instruction encoder
                    ├── DreamLite Mobile UNet + scheduler
                    └── VAE decoder + output image
```

The Android runtime must be a distinct implementation. Candidate inference paths to evaluate are ONNX Runtime with a supported Android execution provider, Qualcomm QNN where the exact operator graph can be compiled, or another Android-native runtime proven to support the exported operators. Do not select a runtime based only on general GPU support claims.

## Android feasibility gates

1. **Encoder gate:** demonstrate Qwen3-VL image-plus-instruction embeddings on Android. This is the largest unknown and must be solved before claiming general editing.
2. **Diffusion export gate:** export the mobile UNet and VAE encoder/decoder to Android-consumable graphs; validate tensor shapes, precision, operators, and numerical parity against upstream.
3. **Device gate:** run the complete pipeline on a physical Android device with measured peak RAM, latency, thermal behavior, and output quality. The first target is the Nothing Phone (3a Pro, 8GB RAM, Snapdragon 7s Gen 3), but compatibility is not assumed.
4. **Product gate:** only after the above pass, connect the Flutter editor UI and model download manager.

## First end-to-end acceptance test

On the target Android phone, select a local photograph and submit: “Change the sky to a warm sunset while preserving the buildings and people.” The device must produce the edited result offline, with no API key, network inference, or cloud upload. Also test object removal/replacement and portrait edits.

## Licensing and distribution

- Do not commit model weights or derived weight files to this repository.
- Before bundling, downloading, redistributing, or shipping weights in an app, review the current upstream `WEIGHTS_LICENSE` and model access/use terms. The current terms describe non-commercial research use and restrict public redistribution.
- Keep the runtime/code license separate from the model-weight license in all documentation and release notes.

## Current status

- [x] Create isolated feature branch.
- [x] Inspect RePix Flutter, Android host, and existing image-generation service.
- [x] Inspect official DreamLite inference and iOS deployment references.
- [x] Identify why the 1.57GB community ONNX package is not an image editor.
- [ ] Prove Android Qwen3-VL encoder execution.
- [ ] Export and validate UNet/VAE for Android.
- [ ] Run a full offline edit on the target phone.
- [ ] Add production Flutter UI, model management, and tests.

No functional DreamLite inference code is claimed at this stage. The next implementation commit should follow a successful Android encoder/runtime spike, not introduce a nonfunctional placeholder.
