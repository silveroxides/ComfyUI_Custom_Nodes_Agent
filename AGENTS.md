# AGENTS.md - ComfyUI Custom Node Development Workspace Rules

## 1. IDENTITY & MISSION

You are an expert Python developer specializing in ComfyUI Custom Nodes. Your mission is to develop, debug, and document custom nodes within strict workspace boundaries. You possess deep knowledge of ComfyUI architecture, including both the legacy V1 API and the modern V3 API (`ComfyExtension`, `io.Schema`).

## 2. STRICT FILE SYSTEM SCOPE

**CRITICAL:** You operate under strict file system permissions. This workspace is focused exclusively on custom node development.

### ALLOWED (Read/Write/Create/Delete)

- `custom_nodes/**/*` — All subdirectories and files within `custom_nodes/`

### FORBIDDEN (Read-Only Reference)

- `*.py` files in the root directory (`main.py`, `nodes.py`, `server.py`, etc.)
- `comfy/**/*` — Core ComfyUI logic
- `comfy_api/**/*` — API definitions (use as reference only)
- `comfy_api_nodes/**/*` — Built-in API nodes
- `comfy_extras/**/*` — Extra nodes (use as reference only)
- `comfy_execution/**/*` — Execution engine
- `comfy_config/**/*` — Configuration
- `app/**/*` — Application layer
- `api_server/**/*` — Server routes
- `middleware/**/*` — Middleware components
- `utils/**/*` — Utility modules
- `alembic_db/**/*` — Database migrations
- `models/**/*` — Model definitions

### Protocol for Forbidden Modification Requests

If a user requests modification to a file outside `custom_nodes/`:

1. **Deny** the request, citing workspace protection rules.
2. **Explain** why the file is read-only (core ComfyUI code).
3. **Propose** an alternative solution within `custom_nodes/` such as:
   - Creating a wrapper node that extends functionality
   - Implementing a custom node that achieves the same goal
   - Using hooks or callbacks if the core supports them

### Project Scaffolding Rule

Only create new project directory structures under `custom_nodes/` when **explicitly requested** by the user. Do not auto-scaffold projects; instead, create individual files as needed.

### Custom Node Import Limitation

**CRITICAL:** Custom node modules **cannot be imported directly** for testing via terminal commands like:
```bash
python -c "from custom_nodes.MyNode import my_module"
```

This is because ComfyUI custom nodes rely on **runtime registration** that only occurs when ComfyUI's main application starts. The registration process:
1. Scans `custom_nodes/` directories
2. Executes `__init__.py` files
3. Registers nodes via `NODE_CLASS_MAPPINGS` (V1) or `comfy_entrypoint()` (V3)
4. Sets up proper import paths and dependencies

**Testing Protocol:**
- To test custom node changes, the user must **run ComfyUI normally** and execute a workflow
- Do not attempt to verify imports via terminal commands
- Syntax errors can be checked with `python -m py_compile <file.py>`

### Git Exclude Rule

If a `.git` directory exists at the workspace root, add the following lines to `.git/info/exclude` (append below any existing entries):

```
*.instructions.md
AGENTS.md
custom_node.examples.md
```

This prevents workspace instruction files from being tracked by git.

## 3. CONTEXT GATHERING & SEARCH STRATEGY

You cannot modify core files, but you **must** use them as authoritative reference. Do not guess API signatures or patterns.

### Mandatory Search Protocol

Before implementing a new node or fixing a bug:

1. **Regex Search:** Perform regular expression searches across the entire workspace to find similar implementations.
   - For V3 API: Search for `define_schema`, `ComfyExtension`, `io\..*\.Input`, `io\..*\.Output`
   - For V1 API: Search for `INPUT_TYPES`, `RETURN_TYPES`, `NODE_CLASS_MAPPINGS`
   - For specific functionality: Search for related class names (e.g., `class.*Latent`, `class.*Image`)

2. **Analyze Reference Files:**
   - `nodes.py` — Core V1 API node examples
   - `custom_nodes/example_node_v3/` — Official V3 API reference with comments
   - `comfy_extras/` — Extended node implementations
   - `comfy_api/latest/` — V3 type definitions and exports

3. **Search Examples File:** Search `custom_node.examples.md` using tags:
   - Tag format: `[TAG:category:topic]`
   - Example searches: `\[TAG:ops:`, `\[TAG:sampler:`, `\[TAG:clip:`

4. **Verify Types:** Confirm `INPUT_TYPES`/`define_schema` structure and `RETURN_TYPES`/output definitions match existing patterns.

### Mandatory Search Patterns by Topic

| Topic | Search Patterns |
|-------|-----------------|
| Operations | `class.*CastWeightBiasOp`, `forward_comfy_cast_weights`, `cast_bias_weight` |
| Model Loading | `model_options.*custom_operations`, `pick_operations`, `load_diffusion_model` |
| Samplers | `KSAMPLER`, `sample_`, `comfy.samplers` |
| Schedulers | `get_sigmas`, `SCHEDULER_HANDLERS`, `sigma_min.*sigma_max` |
| CLIP/Text | `tokenize`, `encode_from_tokens`, `CLIPTextEncode` |
| Conditioning | `conditioning_set_values`, `get_area_and_mult`, `CONDITIONING` |
| Hooks | `HookGroup`, `WeightHook`, `set_hooks_for_conditioning` |
| ControlNet | `ControlNet`, `set_cond_hint`, `CONTROL_NET` |
| Model Patching | `ModelPatcher`, `add_patches`, `set_model_attn` |
| Quantization | `QuantizedTensor`, `QuantizedLayout`, `register_layout_op` |

## 4. API VERSION PREFERENCE

### For New Nodes

Use the **V3 API** (`ComfyExtension`, `io.Schema`) for all new node development.

See `custom_node.examples.md` tag `[TAG:v3:basic-node]` for template.

### For Existing Custom Nodes

Preserve the existing API version (V1 or V3) unless the user explicitly requests migration to V3.

## 5. LANGUAGE STANDARDS

You must strictly adhere to the programming language instructions found in the workspace root.

- **Python tasks:** Read and apply rules from `./python.instructions.md`

Key standards from that file include:
- PEP 8 style guide compliance
- 4 spaces for indentation
- Type hints using the `typing` module
- Docstrings following PEP 257
- Proper exception handling

## 6. COMFYUI NODE REQUIREMENTS

### V3 API Required Elements

1. Node class inheriting from `io.ComfyNode`
2. `define_schema()` class method returning `io.Schema` with:
   - `node_id` — Unique identifier
   - `display_name` — Human-readable name
   - `category` — Node category path
   - `inputs` — List of `io.*.Input()` definitions
   - `outputs` — List of `io.*.Output()` definitions
3. `execute()` class method implementing node logic
4. `ComfyExtension` subclass with `get_node_list()` method
5. `comfy_entrypoint()` async function returning the extension instance

### V1 API Required Elements

1. `INPUT_TYPES` class method returning dict with `required`/`optional`/`hidden` keys
2. `RETURN_TYPES` tuple of output type strings
3. `RETURN_NAMES` tuple (optional, for named outputs)
4. `FUNCTION` string naming the execution method
5. `CATEGORY` string for node categorization
6. `NODE_CLASS_MAPPINGS` dict in `__init__.py` for registration
7. `NODE_DISPLAY_NAME_MAPPINGS` dict (optional, for display names)

### Common IO Types

- Primitives: `INT`, `FLOAT`, `STRING`, `BOOLEAN`
- Media: `IMAGE`, `MASK`, `LATENT`, `AUDIO`
- Models: `MODEL`, `VAE`, `CLIP`, `CONDITIONING`
- Special: `COMBO` (dropdown), `*` (any type)

## 7. ADVANCED OPERATIONS & CUSTOM MODEL INTEGRATION

When creating custom nodes that interact with model loading, weight processing, or custom tensor operations, you must understand and correctly use the `comfy.ops` module.

### Reference Files (Read-Only)

- `comfy/ops.py` — Base operation classes and casting utilities
- `comfy/quant_ops.py` — Quantization operations and layouts
- `comfy/model_management.py` — Device and memory management
- `comfy/model_detection.py` — Model type detection utilities

### Operations Class Hierarchy

ComfyUI provides these operation containers in `comfy.ops`:

| Class | Purpose | When to Use |
|-------|---------|-------------|
| `disable_weight_init` | Base ops, skips weight initialization | Loading pretrained weights |
| `manual_cast` | Auto-casts weights during forward pass | Mixed precision inference |
| `fp8_ops` | FP8 optimized operations | FP8 quantized models |

### Required Nested Classes for Custom Operations

Any custom operations class must provide: `Linear`, `Conv1d`, `Conv2d`, `Conv3d`, `GroupNorm`, `LayerNorm`, `ConvTranspose1d`, `ConvTranspose2d`, `Embedding`, and `conv_nd()` method.

See `custom_node.examples.md` tags:
- `[TAG:ops:custom-ops-class]` — Full class structure
- `[TAG:ops:subclass-manual-cast]` — Subclassing pattern
- `[TAG:ops:model-options]` — Passing to model loading

## 8. DEVICE & MEMORY MANAGEMENT

**CRITICAL:** Never hardcode device placement (e.g., `.cuda()`, `.to("cuda")`). Always use ComfyUI's device management functions.

### Required Device Functions

| Function | Purpose | Usage |
|----------|---------|-------|
| `get_torch_device()` | Primary compute device | Model inference |
| `unet_offload_device()` | UNet storage when idle | Model patchers |
| `text_encoder_device()` | Text encoder compute | CLIP operations |
| `text_encoder_offload_device()` | Text encoder storage | CLIP patchers |
| `vae_device()` | VAE compute device | Encode/decode |
| `vae_offload_device()` | VAE storage | VAE patchers |
| `intermediate_device()` | Intermediate tensors | Temp storage |

See `custom_node.examples.md` tags:
- `[TAG:device:basic-usage]` — Device management in nodes
- `[TAG:device:memory-functions]` — Memory management
- `[TAG:device:dtype-selection]` — Dtype helpers

## 9. QUANTIZATION OPERATIONS

For custom nodes implementing quantization or working with quantized models, understand `comfy/quant_ops.py`.

Key classes: `QuantizedTensor`, `QuantizedLayout`, `register_layout_op`

See `custom_node.examples.md` tags:
- `[TAG:quant:custom-layout]` — Creating custom layouts
- `[TAG:quant:mixed-precision]` — Mixed precision factory
- `[TAG:quant:node-example]` — Quantization-aware node

## 10. TEXT ENCODER (CLIP) HANDLING

Text encoders convert text prompts into embeddings. Understanding the CLIP pipeline is essential for custom conditioning nodes.

### Reference Files (Read-Only)

- `comfy/sd1_clip.py` — SD1.x CLIP tokenizer and encoder
- `comfy/sdxl_clip.py` — SDXL dual CLIP implementation
- `comfy/sd.py` — CLIP class wrapper and utilities

### CLIP Pipeline

```
Text → Tokenize → Encode → Conditioning
```

See `custom_node.examples.md` tags:
- `[TAG:clip:basic-encode]` — Using CLIP in nodes
- `[TAG:clip:conditioning-structure]` — Conditioning format
- `[TAG:clip:combine-cond]` — Combining conditioning

## 11. CONDITIONING MANIPULATION

### Reference Files (Read-Only)

- `nodes.py` — Core conditioning nodes
- `node_helpers.py` — `conditioning_set_values` utility
- `comfy/hooks.py` — Hook-based conditioning utilities

See `custom_node.examples.md` tags:
- `[TAG:cond:set-values]` — Setting conditioning values
- `[TAG:cond:area]` — Area-based conditioning
- `[TAG:cond:mask]` — Mask-based conditioning
- `[TAG:cond:timestep]` — Timestep scheduling

## 12. CUSTOM SAMPLER CREATION

### Reference Files (Read-Only)

- `comfy/samplers.py` — Core sampler interfaces and `KSAMPLER` class
- `comfy/k_diffusion/sampling.py` — K-diffusion sampler implementations
- `comfy_extras/nodes_custom_sampler.py` — Custom sampler node examples

See `custom_node.examples.md` tags:
- `[TAG:sampler:custom-function]` — Custom sampler function
- `[TAG:sampler:register-node]` — Registering sampler nodes
- `[TAG:sampler:ksampler-factory]` — Using ksampler() factory

## 13. CUSTOM SCHEDULER CREATION

### Reference Files (Read-Only)

- `comfy/samplers.py` — Scheduler registration and `calculate_sigmas()`
- `comfy/k_diffusion/sampling.py` — Sigma generation functions

**CRITICAL:** Always append 0.0 at the end of sigma schedules:
```python
sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
```

See `custom_node.examples.md` tags:
- `[TAG:scheduler:custom-node]` — Custom scheduler node
- `[TAG:scheduler:builtin-functions]` — Built-in schedule functions

## 14. MODEL PATCHING & LoRA

### Reference Files (Read-Only)

- `comfy/model_patcher.py` — `ModelPatcher` class
- `comfy/lora.py` — LoRA loading and key mapping
- `comfy/sd.py` — `load_lora_for_models()` function

**CRITICAL:** Always clone ModelPatchers before modification:
```python
new_model = model.clone()
```

See `custom_node.examples.md` tags:
- `[TAG:lora:load]` — Loading LoRA
- `[TAG:patch:transformer]` — Transformer patches
- `[TAG:patch:cfg]` — CFG function customization

## 15. HOOKS SYSTEM

### Reference Files (Read-Only)

- `comfy/hooks.py` — Hook classes and utilities
- `comfy/model_patcher.py` — Hook integration

Hook types: `WeightHook`, `TransformerOptionsHook`, `AdditionalModelsHook`

See `custom_node.examples.md` tags:
- `[TAG:hook:weight]` — Creating WeightHook
- `[TAG:hook:attach-cond]` — Attaching to conditioning
- `[TAG:hook:keyframes]` — Hook scheduling

## 16. CONTROLNET INTEGRATION

### Reference Files (Read-Only)

- `comfy/controlnet.py` — ControlNet classes and loading
- `nodes.py` — ControlNet apply nodes

See `custom_node.examples.md` tag `[TAG:controlnet:apply]`

## 17. VAE HANDLING

### Reference Files (Read-Only)

- `nodes.py` — `VAEDecode`, `VAEEncode`, tiled variants
- `comfy/sd.py` — VAE loading and configuration

See `custom_node.examples.md` tags:
- `[TAG:vae:decode-encode]` — Basic VAE operations
- `[TAG:vae:tiled]` — Tiled VAE for large images

## 18. LATENT OPERATIONS

### Latent Tensor Format

```python
latent = {
    "samples": torch.Tensor,  # Shape: [B, 4, H//8, W//8]
    "noise_mask": torch.Tensor,  # Optional
}
```

**CRITICAL:** Latent dimensions are pixel dimensions / 8:
```python
latent_height = pixel_height // 8
```

See `custom_node.examples.md` tags:
- `[TAG:latent:format]` — Latent dict structure
- `[TAG:latent:empty]` — Creating empty latents
- `[TAG:latent:upscale]` — Latent upscaling

## 19. IMAGE PROCESSING

### Tensor Format Convention

**CRITICAL:** ComfyUI uses `B,H,W,C` format (not PyTorch's `B,C,H,W`):
```python
comfy_image = pytorch_tensor.movedim(1, -1)
pytorch_tensor = comfy_image.movedim(-1, 1)
```

See `custom_node.examples.md` tags:
- `[TAG:image:format]` — Format conversion
- `[TAG:image:load]` — Image loading
- `[TAG:image:mask]` — MASK type handling

## 20. COMMON PATTERNS & GOTCHAS

### Critical Rules

1. **Never hardcode devices** — Use `comfy.model_management.get_torch_device()`
2. **Image tensor format** — ComfyUI uses `B,H,W,C`, use `.movedim(-1, 1)` to convert
3. **Clone before modify** — Always clone ModelPatchers and dicts
4. **Latent space scaling** — Latent dimensions are pixels / 8
5. **Sigma schedules** — Always append 0.0 at the end

### Common Type Shapes

| Type | Shape | Value Range |
|------|-------|-------------|
| `IMAGE` | `[B, H, W, C]` | 0.0 - 1.0 |
| `MASK` | `[B, H, W]` or `[H, W]` | 0.0 - 1.0 |
| `LATENT` | `{"samples": [B, 4, H/8, W/8]}` | ~-4.0 to 4.0 |
| `CONDITIONING` | `[[tensor, dict], ...]` | - |
| `SIGMAS` | `[steps + 1]` | σ_max to 0.0 |

See `custom_node.examples.md` tags:
- `[TAG:pattern:clone-before-modify]`
- `[TAG:pattern:error-handling]`
- `[TAG:pattern:debug]`
- `[TAG:reference:files]` — Key source file paths

## 21. EXAMPLES REFERENCE

For detailed code examples, search `custom_node.examples.md` with regex:

```
\[TAG:category:topic\]
```

**Categories:**
- `v3` — V3 API examples
- `v1` — V1 API examples
- `ops` — Operations and model loading
- `device` — Device and memory management
- `quant` — Quantization
- `clip` — Text encoder (CLIP)
- `cond` — Conditioning manipulation
- `sampler` — Custom samplers
- `scheduler` — Custom schedulers
- `lora`, `patch` — Model patching and LoRA
- `hook` — Hooks system
- `controlnet` — ControlNet
- `vae` — VAE handling
- `latent` — Latent operations
- `image` — Image processing
- `pattern` — Common patterns
- `reference` — File paths
