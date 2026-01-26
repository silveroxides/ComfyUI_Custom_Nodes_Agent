# Audit Plan: Custom Node Documentation & Agent Rules

This plan outlines the steps for auditing `custom_node.examples.md` and `AGENTS.md` against the current state of the ComfyUI repository. The goal is to ensure all code examples, API references, and development rules are accurate and up-to-date.

## Execution Protocol
- **Domain Focus:** Audit one interval at a time.
- **Summary & Notes:** At the end of each interval, provide a summary of findings and specific "Audit Notes" that might be relevant for the next interval (e.g., changes in a core utility that affects multiple domains).
- **Reset:** After providing the summary, the agent's context for the specific code details will be "reset" for the next domain.

---

## Audit Intervals

### Interval 1: V3 API & Core Node Architecture
**Target Sections:**
- `custom_node.examples.md`: V3 API Examples, V1 API Examples
- `AGENTS.md`: Section 6 (ComfyUI Node Requirements), Section 4 (API Version Preference)

**Reference Files:**
- `comfy_api/latest/` (Schema definitions)
- `nodes.py` (V1 examples)
- `comfy_api/latest/_io.py` (Input/Output types)

**Audit Checks:**
- [x] Verify `io.Schema` fields (check for new `search_aliases` field).
- [x] Verify `io.*.Input()` parameters (check for new `advanced` parameter).
- [x] Verify `execute()` signature requirements (check for `kwargs` support).
- [x] Ensure V1 `INPUT_TYPES` pattern still matches `nodes.py`.

**Interval Summary & Audit Notes:**
> Verified that `io.Schema` includes `search_aliases` and `accept_all_inputs`. V3 `Input()` now supports the `advanced` parameter. The `execute()` method for V3 nodes now formally supports `**kwargs` for flexibility. V1 nodes in `nodes.py` use `SEARCH_ALIASES`, `DESCRIPTION`, and `OUTPUT_TOOLTIPS`, which have been added to the guidelines. Expanded the list of "Common IO Types" to include `VIDEO`, `SVG`, `MESH`, `VOXEL`, and more model types.

---

### Interval 2: Operations & Quantization
**Target Sections:**
- `custom_node.examples.md`: Operations & Model Loading, Quantization
- `AGENTS.md`: Section 7 (Advanced Operations), Section 9 (Quantization Operations)

**Reference Files:**
- `comfy/ops.py`
- `comfy/quant_ops.py`
- `comfy/model_management.py` (FP8/FP4 logic)

**Audit Checks:**
- [x] Verify `CastWeightBiasOp` attributes and logic.
- [x] Check if `manual_cast` or `disable_weight_init` patterns have changed.
- [x] Update Quantization section with references to `nvfp4` or new FP8 optimizations if applicable.
- [x] Verify `QuantizedTensor` and layout registration patterns.

**Interval Summary & Audit Notes:**
> Verified `CastWeightBiasOp` attributes. The modern pattern for `cast_bias_weight` now includes `offloadable=True` and mandatory call to `uncast_bias_weight` for async offload performance. `RMSNorm` has been added to the core operations hierarchy and is now a required nested class for custom ops. Quantization has been significantly updated with the introduction of `nvfp4` (NVIDIA 4-bit) and the `comfy_kitchen` library for optimized kernels. `QuantizedTensor` now supports layouts like `TensorCoreNVFP4Layout` with per-block scales. Added `disabled` parameter to `mixed_precision_ops`. New quantization layouts mean ModelPatchers in Interval 3 should be aware of `QuantizedTensor` when applying patches.

---

### Interval 3: Model Loading, Patching & LoRA
**Target Sections:**
- `custom_node.examples.md`: Model Patching & LoRA
- `AGENTS.md`: Section 14 (Model Patching & LoRA)

**Reference Files:**
- `comfy/sd.py`
- `comfy/model_patcher.py`
- `comfy/lora.py`

**Audit Checks:**
- [x] Verify `load_lora_for_models` signature.
- [x] Check `ModelPatcher.add_patches` and object patching logic.
- [x] Audit CFG and Post-CFG function customization patterns.
- [x] Check for new weight-adapter or trainer-specific patching logic.

**Interval Summary & Audit Notes:**
> Verified `load_lora_for_models` and `load_bypass_lora_for_models`. `ModelPatcher` has been significantly expanded with `attachments`, `additional_models`, `callbacks`, `wrappers`, `injections`, and `hooks`. These provide powerful extension points. `add_patches` now uses two strength parameters: `strength_patch` and `strength_model`. Introduced `WeightAdapterBase` for generalized patching. CFG customization now includes `pre_cfg` and `post_cfg` functions that are additive, while `cfg_function` replaces the logic. `add_object_patch` is the standard for replacing model components.

---

### Interval 4: Device & Memory Management
**Target Sections:**
- `custom_node.examples.md`: Device & Memory Management
- `AGENTS.md`: Section 8 (Device & Memory Management)

**Reference Files:**
- `comfy/model_management.py`

**Audit Checks:**
- [x] Verify device utility functions: `get_torch_device()`, `unet_offload_device()`, etc.
- [x] Check memory management functions: `free_memory()`, `load_models_gpu()`.
- [x] Verify `unet_dtype` and `unet_manual_cast` logic.

**Interval Summary & Audit Notes:**
> Verified device utility functions (`get_torch_device`, `unet_offload_device`, etc.). Memory management via `free_memory` and `load_models_gpu` remains the standard for custom nodes. Centralized attention backend checks (`sage_attention`, `flash_attention`, `xformers`, `pytorch`) added. Explicit support for new compute devices including Intel XPU, Ascend NPU, Cambricon MLU, and Moore Threads IXUCA. Added `supports_fp8_compute` and `supports_nvfp4_compute` as official capability checks. Introduced Async Weight Offloading via `get_offload_stream` and pinned memory management (`pin_memory`, `unpin_memory`) for high-performance transfers. `unet_dtype` and `unet_manual_cast` are the authoritative helpers for inference precision.

---

### Interval 5: CLIP, Conditioning & Hooks
**Target Sections:**
- `custom_node.examples.md`: Text Encoder (CLIP), Conditioning, Hooks
- `AGENTS.md`: Section 10 (CLIP Handling), Section 11 (Conditioning Manipulation), Section 15 (Hooks System)

**Reference Files:**
- `comfy/sd.py` (CLIP class)
- `comfy/hooks.py`
- `node_helpers.py` (Conditioning utils)

**Audit Checks:**
- [x] Verify `clip.tokenize` and `clip.encode_from_tokens_scheduled`.
- [x] Check `conditioning_set_values` usage in `node_helpers.py`.
- [x] Verify `WeightHook` and `HookGroup` implementation in `comfy/hooks.py`.
- [x] Audit keyframe scheduling logic for hooks.

**Interval Summary & Audit Notes:**
> Verified CLIP Handling, Conditioning Manipulation, and Hooks System. `CLIP.encode_from_tokens_scheduled` is the standard for handling scheduled hooks during encoding, returning multiple conditioning entries with `clip_start_percent` and `clip_end_percent`. The conditioning metadata structure was expanded to include `mask_strength`, `set_area_to_bounds`, and specific CLIP scheduling percentages. The Hooks system is highly extensible; `WeightHook` for LoRAs, `TransformerOptionsHook` for attention/model logic, and `AdditionalModelsHook` for loading extra models. `HookKeyframe`'s `guarantee_steps` ensures stability in high-step sampling. These hook scheduling details are critical for Samplers in Interval 6 as they dictate which patches are active at each step.

---

### Interval 6: Samplers & Schedulers
**Target Sections:**
- `custom_node.examples.md`: Samplers, Schedulers
- `AGENTS.md`: Section 12 (Custom Sampler Creation), Section 13 (Custom Scheduler Creation)

**Reference Files:**
- `comfy/samplers.py`
- `comfy/k_diffusion/sampling.py`

**Audit Checks:**
- [x] Verify `KSAMPLER` class and factory functions.
- [x] Check scheduler handlers in `SCHEDULER_HANDLERS`.
- [x] Verify sigma schedule generation and the "append 0.0" rule.

**Interval Summary & Audit Notes:**
> Verified `KSAMPLER` class and `ksampler()` factory functions. `KSAMPLER_NAMES` has been significantly expanded with many new research-based samplers (e.g., `heunpp2`, `ipndm`, `deis`, `res_multistep`, `sa_solver`). `SCHEDULER_HANDLERS` now includes `sgm_uniform`, `linear_quadratic`, and `kl_optimal`. The "append 0.0" rule for sigma schedules remains a critical requirement and is consistently implemented across all built-in schedulers. High-level `KSampler` wrapper (different from `KSAMPLER` object) handles steps, denoise, and sigma calculation logic. Sampler functions now consistently receive `model_options` and `seed` via `extra_args`. Inpainting logic is handled via `KSamplerX0Inpaint`. These changes don't fundamentally change Latent/VAE handling but reinforce the need for consistent `model_options` propagation.

---

### Interval 7: Images, Latents, VAE & General Patterns
**Target Sections:**
- `custom_node.examples.md`: VAE, Latents, Images, Common Patterns
- `AGENTS.md`: Sections 17-20

**Reference Files:**
- `comfy/utils.py` (Upscaling, tensor utils)
- `nodes.py` (VAE nodes)
- `comfy/sd.py` (VAE class)

**Audit Checks:**
- [x] Verify VAE `decode` and `encode` signatures.
- [x] Check latent dictionary format (samples, noise_mask).
- [x] Verify Image tensor format (B,H,W,C) and conversion logic.
- [x] Review "Common Patterns & Gotchas" for any new critical rules.

**Interval Summary & Final Audit Notes:**
> Verified VAE Handling, Latents, and Image processing. Images strictly follow the `[B, H, W, C]` format with 0-1 values. VAE `decode` and `encode` methods handle dimension conversion and include built-in support for tiled processing. Latent dictionary format was updated to include `downscale_ratio_spacial` (default 8) and support for 3D video latents `[B, C, T, H, W]`. Introduced `io.LatentOperation` as a standard V3 IO type for modular latent transformations, following a functional pattern: `def op(latent, **kwargs)`. Common patterns now emphasize `comfy.utils.common_upscale` for resolution changes and explicit handling of `is_nested` tensors for PyTorch compatibility. This concludes the full audit plan.

---

## Interval Tracking
- [x] Interval 1
- [x] Interval 2
- [x] Interval 3
- [x] Interval 4
- [x] Interval 5
- [x] Interval 6
- [x] Interval 7
