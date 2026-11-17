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

The Android runtime must be a distinct implementation. Two useful leads exist, but neither proves DreamLite compatibility yet:\n\n- ONNX Runtime now documents Android ARM64 support for its Qualcomm QNN execution provider through a Maven artifact. This gives us a real Snapdragon NPU route to investigate, but every exported DreamLite graph must still pass QNN operator/shape compilation and device validation.\n- Qualcomm's Android GenieX examples list Qwen3-VL 2B VLM support on Snapdragon devices, and Qwen3-VL 2B can also run through llama.cpp on Android CPU. However, DreamLite needs the encoder's image-conditioned hidden-state tensor, not a normal chat completion. The iOS reference explicitly modifies the VLM implementation to expose hidden states. We must prove an Android runtime can expose equivalent tensors before treating this as a solution.\n\nThe preferred spike is therefore: first test whether an Android Qwen3-VL implementation can return the exact image+instruction hidden-state tensor DreamLite expects; in parallel, export a minimal DreamLite UNet/VAE graph and test ONNX Runtime/QNN compilation. Do not select a runtime based only on general GPU support claims.

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

## Follow-up source-level deployment audit (2026-11-17)

A closer audit of the official `deploy/` sources narrows the Android porting work:

- The upstream deployment guide explicitly targets **iOS / Swift / Core ML**. It is not a portable mobile runtime package.
- The provided UNet exporter is Core ML-specific and traces a wrapper with inputs:
  - `sample [1, 4, 128, 256]`
  - `timestep [1]`
  - `encoder_hidden_states [1, 77, 2048]` (77 is only the export example/default)
  - `encoder_attention_mask [1, 77]`
  - `time_ids [1, 2]`
  The exporter allows sequence lengths 10–512 in Core ML, so an Android export must preserve dynamic sequence handling or deliberately prove a safe fixed-length strategy.
- The VAE exporter contracts are explicit: image `[1,3,1024,1024]` to latent `[1,4,128,128]`, and the reverse for decoding. These are useful parity targets for an ONNX export.
- The official Qwen3-VL patch adds `getHiddenStatesWithImage(inputIds:pixelValues:imageGridTHW:)`. It runs the vision model, merges image features into token embeddings, then runs the language layers and returns hidden states. This confirms the critical encoder output is not a standard generated answer; it is a tensor produced by a custom model-internal path.
- The official UNet exporter uses Core ML conversion with FP16 and an iOS deployment target. It does not provide an ONNX/QNN export recipe.

### Updated engineering decision

Do **not** add a pretend Android inference class or wire a nonfunctional editor UI yet. The next useful implementation artifact is an export/parity spike outside the Android app:

1. Reproduce the official PyTorch outputs for VAE encoder, VAE decoder, and UNet with saved test tensors.
2. Export those modules to ONNX with named inputs and dynamic axes where required.
3. Run ONNX Runtime CPU validation first, then attempt Qualcomm QNN compilation on a supported Snapdragon device.
4. Separately port the exact Qwen3-VL image-conditioned hidden-state method to an Android-capable runtime. A text-only embedding or chat completion is not an acceptable substitute.
5. Compare tensor shapes and numerical error against the official Python pipeline before connecting Flutter.

### Current implementation status

- [x] Inspected the exact upstream Core ML export scripts and their tensor contracts.
- [x] Located the custom Qwen3-VL image-conditioned hidden-state method and confirmed why ordinary chat inference is insufficient.
- [ ] Android ONNX export/parity harness.
- [ ] QNN compilation and device execution.
- [ ] Android Qwen3-VL hidden-state extraction.
- [ ] End-to-end offline edit on the Nothing Phone (3a Pro).

This audit is source inspection only. No Android model has been exported, compiled, or run yet.
