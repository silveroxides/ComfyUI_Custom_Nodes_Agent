# ComfyUI Custom Node Examples Reference

This file contains code examples and patterns for custom node development. Use regex to search for specific tags.

**Tag Format:** `[TAG:category:topic]`

**Search Examples:**
- `\[TAG:ops:` — All operations-related examples
- `\[TAG:sampler:` — Sampler creation examples
- `\[TAG:clip:` — Text encoder examples
- `\[TAG:v3:` — V3 API examples

---

## Table of Contents

1. [V3 API Examples](#v3-api-examples)
2. [V1 API Examples](#v1-api-examples)
3. [Operations & Model Loading](#operations--model-loading)
4. [Device & Memory Management](#device--memory-management)
5. [Quantization](#quantization)
6. [Text Encoder (CLIP)](#text-encoder-clip)
7. [Conditioning](#conditioning)
8. [Samplers](#samplers)
9. [Schedulers](#schedulers)
10. [Model Patching & LoRA](#model-patching--lora)
11. [Hooks](#hooks)
12. [ControlNet](#controlnet)
13. [VAE](#vae)
14. [Latents](#latents)
15. [Images](#images)
16. [Common Patterns](#common-patterns)

---

## V3 API Examples

### [TAG:v3:basic-node]
Basic V3 node structure with `ComfyExtension`:

```python
from comfy_api.latest import ComfyExtension, io

class MyNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MyNode",
            display_name="My Node",
            category="custom/category",
            inputs=[...],
            outputs=[...],
            search_aliases=["alias1", "alias2"],
            description="My Node Description",
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        ...

class MyExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [MyNode]

async def comfy_entrypoint() -> MyExtension:
    return MyExtension()
```

### [TAG:v3:full-example]
Complete V3 node with all common input/output types:

```python
from comfy_api.latest import ComfyExtension, io

class MyNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UniqueNodeID",
            display_name="My Node",
            category="custom/category",
            inputs=[
                io.Model.Input("model"),
                io.Image.Input("image"),
                io.Int.Input("steps", default=20, min=1, max=100),
                io.Float.Input("strength", default=1.0, advanced=True),
                io.String.Input("text", multiline=True),
                io.Combo.Input("mode", options=["option1", "option2"]),
                io.Boolean.Input("enabled", default=True),
            ],
            outputs=[
                io.Image.Output(),
                io.Latent.Output(display_name="latent_out"),
            ],
            search_aliases=["UniqueNodeID", "MyNode"],
            description="My Node Description",
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        # Access inputs via kwargs
        model = kwargs.get("model")
        image = kwargs.get("image")
        # Process...
        return io.NodeOutput(result_image, result_latent)

class MyExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [MyNode]

async def comfy_entrypoint() -> MyExtension:
    return MyExtension()
```

---

## V1 API Examples

### [TAG:v1:basic-node]
Basic V1 node structure:

```python
class MyNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("output_image",)
    OUTPUT_TOOLTIPS = ("The processed image.",)
    FUNCTION = "process"
    CATEGORY = "custom/category"
    DESCRIPTION = "My Custom Node Description"
    SEARCH_ALIASES = ["alias1", "alias2"]

    def process(self, image, strength, mask=None):
        # Process...
        return (result,)

# In __init__.py:
NODE_CLASS_MAPPINGS = {
    "MyNode": MyNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MyNode": "My Custom Node",
}
```

---

## Operations & Model Loading

### [TAG:ops:cast-weight-bias]
CastWeightBiasOp mixin attributes:

```python
class CastWeightBiasOp:
    comfy_cast_weights = False  # Set True to enable casting in forward
    weight_function = []        # Functions applied to weights
    bias_function = []          # Functions applied to biases
```

### [TAG:ops:custom-ops-class]
Required nested classes for custom operations:

```python
class MyCustomOps:
    class Linear(torch.nn.Linear, CastWeightBiasOp): ...
    class Conv1d(torch.nn.Conv1d, CastWeightBiasOp): ...
    class Conv2d(torch.nn.Conv2d, CastWeightBiasOp): ...
    class Conv3d(torch.nn.Conv3d, CastWeightBiasOp): ...
    class GroupNorm(torch.nn.GroupNorm, CastWeightBiasOp): ...
    class LayerNorm(torch.nn.LayerNorm, CastWeightBiasOp): ...
    class RMSNorm(comfy.rmsnorm.RMSNorm, CastWeightBiasOp): ...
    class ConvTranspose1d(torch.nn.ConvTranspose1d, CastWeightBiasOp): ...
    class ConvTranspose2d(torch.nn.ConvTranspose2d, CastWeightBiasOp): ...
    class Embedding(torch.nn.Embedding, CastWeightBiasOp): ...

    @classmethod
    def conv_nd(cls, dims, *args, **kwargs):
        if dims == 2:
            return cls.Conv2d(*args, **kwargs)
        elif dims == 3:
            return cls.Conv3d(*args, **kwargs)
```

### [TAG:ops:subclass-manual-cast]
Basic pattern — subclass `manual_cast`:

```python
import comfy.ops

class MyCustomOps(comfy.ops.manual_cast):
    class Linear(comfy.ops.manual_cast.Linear):
        def forward_comfy_cast_weights(self, input):
            # Cast weights to input dtype/device
            weight, bias, offload_stream = comfy.ops.cast_bias_weight(
                self, input, offloadable=True
            )

            # Your custom logic here
            result = torch.nn.functional.linear(input, weight, bias)

            # Clean up (important for memory management)
            comfy.ops.uncast_bias_weight(self, weight, bias, offload_stream)
            return result
```

### [TAG:ops:custom-init]
Advanced pattern — custom initialization:

```python
import comfy.ops

class MyCustomOps(comfy.ops.disable_weight_init):
    class Linear(comfy.ops.disable_weight_init.Linear):
        comfy_cast_weights = True  # Enable casting

        def reset_parameters(self):
            # Custom initialization (called when layer is created)
            self.my_custom_scale = None
            return None  # Skip default init

        def forward_comfy_cast_weights(self, input):
            weight, bias, offload_stream = comfy.ops.cast_bias_weight(
                self, input, offloadable=True
            )

            # Apply custom scaling
            if self.my_custom_scale is not None:
                weight = weight * self.my_custom_scale

            result = torch.nn.functional.linear(input, weight, bias)
            comfy.ops.uncast_bias_weight(self, weight, bias, offload_stream)
            return result
```

### [TAG:ops:model-options]
Passing custom operations to model loading:

```python
import comfy.sd
import comfy.ops

# Basic usage
model_options = {"custom_operations": MyCustomOps}
model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)

# With additional options
model_options = {
    "custom_operations": MyCustomOps,
    "dtype": torch.float16,
    "fp8_optimizations": True,
}
model_patcher, clip, vae, clipvision = comfy.sd.load_checkpoint(
    ckpt_path,
    model_options=model_options
)
```

### [TAG:ops:loader-node]
Custom node loading model with operations:

```python
import comfy.sd
import comfy.ops
import comfy.model_management
import comfy.utils

class LoadModelWithCustomOps:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_path": ("STRING", {"default": ""}),
                "use_custom_ops": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "custom/loaders"

    def load_model(self, model_path, use_custom_ops):
        model_options = {}
        if use_custom_ops:
            model_options["custom_operations"] = comfy.ops.manual_cast

        model = comfy.sd.load_diffusion_model(
            model_path,
            model_options=model_options
        )
        return (model,)
```

---

## Device & Memory Management

### [TAG:device:basic-usage]
Device management in a node:

```python
import comfy.model_management

class MyModelNode:
    def process(self, model, image):
        # Get appropriate devices
        device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()

        # Move tensor to compute device
        image = image.to(device)

        # Process...
        result = model(image)

        # Move result to intermediate device if needed
        return result.to(comfy.model_management.intermediate_device())
```

### [TAG:device:memory-functions]
Memory management functions:

```python
import comfy.model_management

# Check available memory before heavy operations
free_mem = comfy.model_management.get_free_memory(device)
total_mem = comfy.model_management.get_total_memory(device)

# Free memory by unloading models
comfy.model_management.free_memory(
    memory_required,  # Bytes needed
    device,
    keep_loaded=[]    # Models to keep
)

# Load models to GPU with smart memory management
comfy.model_management.load_models_gpu(
    models,                      # List of model patchers
    memory_required=0,
    force_patch_weights=False,
    minimum_memory_required=None,
    force_full_load=False
)

# Emergency: unload everything
comfy.model_management.unload_all_models()
```

### [TAG:device:dtype-selection]
Dtype selection helpers:

```python
import comfy.model_management

# Get optimal dtype for UNet
dtype = comfy.model_management.unet_dtype(
    device=None,
    model_params=0,
    supported_dtypes=[torch.float16, torch.bfloat16, torch.float32],
    weight_dtype=None
)

# Check if manual casting is needed
manual_cast_dtype = comfy.model_management.unet_manual_cast(
    weight_dtype,
    inference_device,
    supported_dtypes=[torch.float16, torch.bfloat16, torch.float32]
)

# Check device capabilities
if comfy.model_management.should_use_fp16(device):
    dtype = torch.float16
elif comfy.model_management.should_use_bf16(device):
    dtype = torch.bfloat16

# Check FP8 support
if comfy.model_management.supports_fp8_compute(device):
    # Can use FP8 operations
    pass
```

### [TAG:device:tensor-casting]
Tensor casting utility:

```python
import comfy.model_management

# Cast tensor with proper non-blocking support
tensor = comfy.model_management.cast_to(
    weight,
    dtype=torch.float16,
    device=device,
    non_blocking=True,  # Async transfer if supported
    copy=False
)

# Check if device supports non-blocking
if comfy.model_management.device_supports_non_blocking(device):
    # Use async transfers
    pass
```


---

## Quantization

### [TAG:quant:quantized-tensor]
QuantizedTensor class structure:

```python
from comfy.quant_ops import QuantizedTensor

class QuantizedTensor(torch.Tensor):
    _qdata: torch.Tensor      # Quantized data storage
    _layout_type: str         # Layout identifier (e.g., "TensorCoreFP8Layout", "TensorCoreNVFP4Layout")
    _layout_params: dict      # Scale, block_scale (for NVFP4), orig_dtype, etc.

    @classmethod
    def from_float(cls, tensor, layout_type, **kwargs):
        """Create quantized tensor from float tensor."""
        pass

    def dequantize(self):
        """Convert back to original dtype."""
        pass
```

### [TAG:quant:custom-layout]
Creating custom quantization layouts:

```python
from comfy.quant_ops import QuantizedLayout, register_layout_op
import torch

class MyCustomLayout(QuantizedLayout):
    """Custom quantization layout for specific use case."""

    @classmethod
    def quantize(cls, tensor, scale=None, dtype=torch.float8_e4m3fn, **kwargs):
        """
        Quantize a float tensor.

        Args:
            tensor: Input float tensor
            scale: Quantization scale (computed if None)
            dtype: Target quantized dtype

        Returns:
            Tuple of (quantized_data, layout_params_dict)
        """
        if scale is None:
            scale = tensor.abs().max() / torch.finfo(dtype).max

        qdata = (tensor / scale).to(dtype)
        layout_params = {
            "scale": scale,
            "orig_dtype": tensor.dtype,
        }
        return qdata, layout_params

    @staticmethod
    def dequantize(qdata, scale, orig_dtype, **kwargs):
        """Dequantize back to original dtype."""
        return qdata.to(orig_dtype) * scale


# Register custom operation handler for your layout
@register_layout_op(torch.ops.aten.linear.default, "MyCustomLayout")
def my_custom_linear(func, args, kwargs):
    """
    Custom linear operation for MyCustomLayout tensors.

    Args:
        func: Original torch function
        args: Positional arguments (input, weight, bias)
        kwargs: Keyword arguments
    """
    input_tensor, weight, bias = args[0], args[1], args[2] if len(args) > 2 else None

    # Dequantize weight if needed
    if isinstance(weight, QuantizedTensor):
        weight = weight.dequantize()

    # Perform operation
    return torch.nn.functional.linear(input_tensor, weight, bias)
```

### [TAG:quant:mixed-precision]
Mixed precision operations factory:

```python
from comfy.ops import mixed_precision_ops

# Create ops class with quantization config
quant_config = {
    "layer_name.weight": {"dtype": torch.float8_e4m3fn, "scale": 0.1},
    # ... per-layer configs
}

CustomQuantOps = mixed_precision_ops(
    quant_config=quant_config,
    compute_dtype=torch.bfloat16,
    full_precision_mm=False,
    disabled=[] # Optional: list of formats to disable (e.g., ["nvfp4"])
)

# Use in model loading
model_options = {"custom_operations": CustomQuantOps}
```

### [TAG:quant:node-example]
Quantization-aware custom node:

```python
import torch
import comfy.ops
import comfy.model_management
from comfy.quant_ops import QuantizedTensor

class QuantizeModelWeights:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "quantize_dtype": (["fp8_e4m3fn", "fp8_e5m2", "nvfp4", "int8"],),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "quantize"
    CATEGORY = "custom/quantization"

    def quantize(self, model, quantize_dtype):
        dtype_map = {
            "fp8_e4m3fn": torch.float8_e4m3fn,
            "fp8_e5m2": torch.float8_e5m2,
            "nvfp4": torch.uint8, # NVFP4 stored as uint8
            "int8": torch.int8,
        }
        target_dtype = dtype_map[quantize_dtype]

        # Clone model to avoid modifying original
        model = model.clone()

        # Apply quantization via model patcher
        def quantize_weight(weight, **kwargs):
            if weight.dtype in [torch.float16, torch.float32, torch.bfloat16]:
                scale = weight.abs().max() / torch.finfo(target_dtype).max
                return (weight / scale).to(target_dtype), {"scale": scale}
            return weight, {}

        # Add weight hook
        model.add_weight_hook(quantize_weight)

        return (model,)
```

### [TAG:quant:fp8-ops]
FP8 operations reference:

```python
import comfy.ops

# Check if FP8 is available
if comfy.model_management.supports_fp8_compute(device):
    # Use FP8 optimized operations
    model_options = {
        "custom_operations": comfy.ops.fp8_ops,
        "fp8_optimizations": True,
    }
```

---

## Text Encoder (CLIP)

### [TAG:clip:basic-encode]
Using CLIP in custom nodes:

```python
class MyTextEncoder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "clip": ("CLIP",),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "conditioning"

    def encode(self, clip, text):
        # Validate CLIP input
        if clip is None:
            raise RuntimeError("clip input is invalid: None")

        # Tokenize text (handles prompt weights like (word:1.2))
        tokens = clip.tokenize(text)

        # Encode and return conditioning
        # encode_from_tokens_scheduled handles hook schedules.
        # It returns a list of [embedding_tensor, metadata_dict] entries.
        # If hooks are scheduled, it produces entries with clip_start_percent/clip_end_percent.
        return (clip.encode_from_tokens_scheduled(tokens), )
```

### [TAG:clip:tokenize]
Token weight syntax:

```python
# ComfyUI supports prompt weighting using parentheses:
# (word) — Increase weight by 1.1x
# (word:1.5) — Set explicit weight
# ((word)) — Nested increases (1.1 * 1.1 = 1.21x)

# tokens is a dict keyed by clip name, containing list of (token_id, weight) tuples
tokens = clip.tokenize("a (beautiful:1.2) sunset")
```

### [TAG:clip:config]
CLIP configuration options:

```python
# Set which layer to use for embeddings
clip.clip_layer(-2)  # Use penultimate layer (common for SD2.x)

# Set tokenizer options
clip.set_tokenizer_option("disable_weights", True)  # Ignore () weights
```

### [TAG:clip:conditioning-structure]
Conditioning structure:

```python
# Single conditioning entry structure:
conditioning = [
    [
        torch.Tensor(...),  # Shape: [1, seq_len, embed_dim]
        {
            "pooled_output": torch.Tensor(...),  # Shape: [1, pooled_dim]
            # Optional metadata:
            "area": (h_latent, w_latent, y_latent, x_latent),  # Pixels / 8
            "strength": 1.0,
            "mask": torch.Tensor(...),          # [B, H_latent, W_latent]
            "mask_strength": 1.0,
            "set_area_to_bounds": False,
            "start_percent": 0.0,               # Sampling start (0.0=start)
            "end_percent": 1.0,                 # Sampling end (1.0=end)
            "clip_start_percent": 0.0,          # Specifically for scheduled CLIP
            "clip_end_percent": 1.0,            # Specifically for scheduled CLIP
            "hooks": HookGroup(...),
        }
    ]
]
```

### [TAG:clip:combine-cond]
Combining conditioning:

```python
# Simple combination (both apply)
combined = conditioning_1 + conditioning_2

# Weighted average
def weighted_average(cond_to, cond_from, weight):
    # cond_to[i][0] is the embedding tensor
    # cond_to[i][1] is the metadata dict
    out = []
    for i in range(len(cond_to)):
        t1 = cond_to[i][0]
        t0 = cond_from[0][0][:, :t1.shape[1]]
        tw = t1 * weight + t0 * (1.0 - weight)
        out.append([tw, cond_to[i][1].copy()])
    return out
```

---

## Conditioning

### [TAG:cond:set-values]
Setting conditioning values:

```python
import node_helpers

class ConditioningSetStrength:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "set_strength"

    def set_strength(self, conditioning, strength):
        c = node_helpers.conditioning_set_values(
            conditioning,
            {"strength": strength}
        )
        return (c,)
```

### [TAG:cond:area]
Area-based conditioning:

```python
def set_area(conditioning, width, height, x, y, strength):
    # Area values are in latent space (pixels / 8)
    return node_helpers.conditioning_set_values(conditioning, {
        "area": (height // 8, width // 8, y // 8, x // 8),
        "strength": strength,
        "set_area_to_bounds": False
    })
```

### [TAG:cond:mask]
Mask-based conditioning:

```python
def set_mask(conditioning, mask, strength, set_cond_area="default"):
    set_area_to_bounds = (set_cond_area != "default")
    if len(mask.shape) < 3:
        mask = mask.unsqueeze(0)

    return node_helpers.conditioning_set_values(conditioning, {
        "mask": mask,
        "set_area_to_bounds": set_area_to_bounds,
        "mask_strength": strength
    })
```

### [TAG:cond:timestep]
Timestep scheduling:

```python
def set_timestep_range(conditioning, start_percent, end_percent):
    # start_percent=0.0 means beginning (high noise)
    # end_percent=1.0 means end (low noise)
    return node_helpers.conditioning_set_values(conditioning, {
        "start_percent": start_percent,
        "end_percent": end_percent
    })
```

---

## Samplers

### [TAG:sampler:custom-function]
Creating a custom sampler function:

```python
import torch
from comfy.k_diffusion.sampling import to_d, get_ancestral_step

@torch.no_grad()
def sample_my_sampler(model, x, sigmas, extra_args=None, callback=None, disable=None):
    """Custom sampler implementation."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    for i in range(len(sigmas) - 1):
        # Get model prediction (denoised image estimate)
        denoised = model(x, sigmas[i] * s_in, **extra_args)

        # Calculate derivative
        d = to_d(x, sigmas[i], denoised)

        # Euler step
        dt = sigmas[i + 1] - sigmas[i]
        x = x + d * dt

        # Callback for preview
        if callback is not None:
            callback({'i': i, 'denoised': denoised, 'x': x})

    return x
```

### [TAG:sampler:register-node]
Registering a custom sampler node:

```python
import comfy.samplers

class MySamplerNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "my_param": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "get_sampler"
    CATEGORY = "sampling/custom_sampling/samplers"

    def get_sampler(self, my_param):
        # Create KSAMPLER with custom function and options
        sampler = comfy.samplers.KSAMPLER(
            sample_my_sampler,
            extra_options={"my_param": my_param}
        )
        return (sampler,)
```

### [TAG:sampler:ksampler-factory]
Using `ksampler()` factory:

```python
def get_sampler(self, eta, s_noise):
    # ksampler() wraps standard k-diffusion samplers
    sampler = comfy.samplers.ksampler(
        "euler_ancestral",  # Base sampler name
        extra_options={"eta": eta, "s_noise": s_noise}
    )
    return (sampler,)
```

### [TAG:sampler:names]
Available sampler names:

```python
comfy.samplers.KSAMPLER_NAMES = [
    "euler", "euler_cfg_pp", "euler_ancestral", "euler_ancestral_cfg_pp", "heun", "heunpp2",
    "exp_heun_2_x0", "exp_heun_2_x0_sde", "dpm_2", "dpm_2_ancestral", "lms", "dpm_fast",
    "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_2s_ancestral_cfg_pp", "dpmpp_sde", "dpmpp_sde_gpu",
    "dpmpp_2m", "dpmpp_2m_cfg_pp", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_2m_sde_heun",
    "dpmpp_2m_sde_heun_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm", "ipndm", "ipndm_v",
    "deis", "res_multistep", "res_multistep_cfg_pp", "res_multistep_ancestral",
    "res_multistep_ancestral_cfg_pp", "gradient_estimation", "gradient_estimation_cfg_pp",
    "er_sde", "seeds_2", "seeds_3", "sa_solver", "sa_solver_pece"
]
```

---

## Schedulers

### [TAG:scheduler:basics]
Scheduler basics — sigma tensor format:

```python
# Sigmas tensor: [σ_0, σ_1, ..., σ_n, 0.0]
# σ_0 is highest (most noise), decreasing to 0 (no noise)
sigmas = torch.FloatTensor([14.6, 10.0, 5.0, 2.0, 1.0, 0.5, 0.0])
```

### [TAG:scheduler:handlers]
Built-in scheduler handlers:

```python
comfy.samplers.SCHEDULER_HANDLERS = {
    "simple": simple_scheduler,
    "sgm_uniform": partial(normal_scheduler, sgm=True),
    "karras": k_diffusion_sampling.get_sigmas_karras,
    "exponential": k_diffusion_sampling.get_sigmas_exponential,
    "ddim_uniform": ddim_scheduler,
    "beta": beta_scheduler,
    "normal": normal_scheduler,
    "linear_quadratic": linear_quadratic_schedule,
    "kl_optimal": kl_optimal_scheduler,
}
```

### [TAG:scheduler:custom-node]
Creating a custom scheduler node:

```python
import torch
import comfy.samplers
from comfy.k_diffusion import sampling as k_diffusion_sampling

class MyScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "my_param": ("FLOAT", {"default": 1.0}),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = "sampling/custom_sampling/schedulers"

    def get_sigmas(self, model, steps, my_param):
        # Get model's sigma range
        model_sampling = model.get_model_object("model_sampling")
        sigma_min = float(model_sampling.sigma_min)
        sigma_max = float(model_sampling.sigma_max)

        # Generate custom sigma schedule
        sigmas = my_custom_schedule(steps, sigma_min, sigma_max, my_param)

        # Always append zero at the end
        sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
        return (sigmas,)

def my_custom_schedule(n, sigma_min, sigma_max, param):
    """Custom sigma schedule function."""
    # Example: linear with parameter adjustment
    ramp = torch.linspace(0, 1, n)
    sigmas = sigma_max * (sigma_min / sigma_max) ** (ramp ** param)
    return sigmas
```

### [TAG:scheduler:builtin-functions]
Using built-in schedule functions:

```python
from comfy.k_diffusion import sampling as k_diffusion_sampling

# Karras schedule (good default)
sigmas = k_diffusion_sampling.get_sigmas_karras(
    n=steps, sigma_min=0.029, sigma_max=14.6, rho=7.0
)

# Exponential schedule
sigmas = k_diffusion_sampling.get_sigmas_exponential(
    n=steps, sigma_min=0.029, sigma_max=14.6
)
```

---

## Model Patching & LoRA

### [TAG:lora:load]
Loading LoRA for models (Standard and Bypass):

```python
import comfy.lora
import comfy.sd

class MyLoRALoader:
    def load_lora(self, model, clip, lora_path, strength_model, strength_clip, bypass_mode=False):
        # Load LoRA weights
        lora = comfy.utils.load_torch_file(lora_path)

        if bypass_mode:
            # Bypass mode injects LoRA into forward pass (output = base + lora)
            # Useful for training or when models are offloaded
            model_lora, clip_lora = comfy.sd.load_bypass_lora_for_models(
                model, clip, lora, strength_model, strength_clip
            )
        else:
            # Standard mode patches the weights directly
            model_lora, clip_lora = comfy.sd.load_lora_for_models(
                model, clip, lora, strength_model, strength_clip
            )
        return (model_lora, clip_lora)
```

### [TAG:lora:direct-patch]
Direct patch addition with strengths:

```python
# Clone the model patcher (don't modify original)
new_model = model.clone()

# Add patches with strength
# patches: Dict[str, tuple] or Dict[str, WeightAdapterBase]
# strength_patch: multiper for the patch itself (e.g. LoRA strength)
# strength_model: multiplier for the base model weight before patching
patches = {...}
applied_keys = new_model.add_patches(
    patches,
    strength_patch=1.0,
    strength_model=1.0
)
```

### [TAG:patch:object]
Object patches for replacing core components:

```python
# Add object patch (affects model.get_model_object())
model.add_object_patch("manual_cast_dtype", torch.float16)

# Set custom operations
model.add_object_patch("custom_operations", my_ops_class)

# Replace model sampling logic
model.add_object_patch("model_sampling", MyNewSampling(model.model.model_config))
```

### [TAG:patch:transformer]
Transformer options patches and forward wrappers:

```python
# Add attention patches
model.set_model_attn1_patch(my_self_attn_patch)
model.set_model_attn2_patch(my_cross_attn_patch)

# Wrap the entire UNet forward pass
# Signature: wrapper(model_function, params)
model.set_model_unet_function_wrapper(my_unet_wrapper)

# Add block patches
model.set_model_input_block_patch(my_input_patch)
model.set_model_output_block_patch(my_output_patch)

# Replace attention entirely for specific blocks
model.set_model_attn1_replace(my_attn_func, "input", block_number, transformer_index=None)
```

### [TAG:patch:cfg]
CFG and Post-CFG function customization:

```python
# Custom CFG function (replaces entire CFG logic)
def my_cfg_function(args):
    cond = args["cond"]
    uncond = args["uncond"]
    cond_scale = args["cond_scale"]
    # ... logic ...
    return uncond + (cond - uncond) * cond_scale

model.set_model_sampler_cfg_function(my_cfg_function)

# Pre-CFG and Post-CFG modifications (additive)
def pre_cfg(args):
    # Modify cond/uncond before CFG
    return args

def post_cfg(args):
    denoised = args["denoised"]
    # Modify denoised result after CFG
    return denoised

model.set_model_sampler_pre_cfg_function(pre_cfg)
model.set_model_sampler_post_cfg_function(post_cfg)
```

### [TAG:patch:injections]
Low-level model injections:

```python
# Injections are used to wrap or modify specific model layers
# Often used by Bypass LoRA or advanced extension nodes
from comfy.patcher_extension import PatcherInjection

injections = [PatcherInjection(target_key, wrapper_func)]
model.set_injections("my_extension_id", injections)
```

---

## Hooks

### [TAG:hook:weight]
Creating a WeightHook:

```python
import comfy.hooks

def create_lora_hook(lora_weights, strength_model, strength_clip):
    hook_group = comfy.hooks.HookGroup()

    hook = comfy.hooks.WeightHook(
        strength_model=strength_model,
        strength_clip=strength_clip
    )
    hook.weights = lora_weights
    hook_group.add(hook)

    return hook_group
```

### [TAG:hook:attach-cond]
Attaching hooks to conditioning:

```python
import comfy.hooks

def attach_hook_to_cond(conditioning, hooks):
    return comfy.hooks.set_hooks_for_conditioning(
        conditioning,
        hooks,
        append_hooks=True  # Combine with existing hooks
    )
```

### [TAG:hook:keyframes]
Hook keyframes (scheduling):

```python
# Create keyframe for hook strength over time
hook_kf = comfy.hooks.HookKeyframeGroup()

# Full strength for first 50% of sampling
# guarantee_steps=1 ensures it runs for at least 1 step even if next KF starts immediately
hook_kf.add(comfy.hooks.HookKeyframe(strength=1.0, start_percent=0.0, guarantee_steps=1))

# Fade out from 50% to 100%
hook_kf.add(comfy.hooks.HookKeyframe(strength=0.0, start_percent=0.5))

# Apply to hooks
hooks.set_keyframes_on_hooks(hook_kf)
```

### [TAG:hook:transformer-options]
Creating a TransformerOptionsHook for attention manipulation:

```python
import comfy.hooks

def create_attn_hook(self_attn_patch, cross_attn_patch):
    # Dictionary structure matching ModelPatcher.transformer_options
    transformers_dict = {
        "patches": {
            "self_attn": [self_attn_patch],
            "cross_attn": [cross_attn_patch],
        }
    }

    hook = comfy.hooks.TransformerOptionsHook(
        transformers_dict=transformers_dict,
        hook_scope=comfy.hooks.EnumHookScope.HookedOnly
    )

    hook_group = comfy.hooks.HookGroup()
    hook_group.add(hook)
    return hook_group
```

---

## ControlNet

### [TAG:controlnet:apply]
Applying ControlNet in nodes:

```python
class ApplyControlNet:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "control_net": ("CONTROL_NET",),
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)

    def apply(self, conditioning, control_net, image, strength):
        if strength == 0:
            return (conditioning,)

        # Clone controlnet
        c_net = control_net.copy()

        # Set control hint (image) and strength
        # Image must be in B,H,W,C format, convert to B,C,H,W
        c_net.set_cond_hint(image.movedim(-1, 1), strength)

        # Add to conditioning
        out = []
        for t in conditioning:
            n = [t[0], t[1].copy()]
            # Chain with existing controlnet if present
            c = c_net.copy()
            prev_cnet = n[1].get('control', None)
            if prev_cnet is not None:
                c.set_previous_controlnet(prev_cnet)
            n[1]['control'] = c
            out.append(n)

        return (out,)
```

---

## VAE

### [TAG:vae:decode-encode]
Basic VAE operations:

```python
class VAEDecode:
    def decode(self, vae, samples):
        # samples is {"samples": latent_tensor}
        # VAE.decode handles is_nested and returns B,H,W,C
        images = vae.decode(samples["samples"])
        return (images,)

class VAEEncode:
    def encode(self, vae, pixels):
        # pixels is IMAGE tensor in B,H,W,C format
        # Returns dict with "samples"
        latent = vae.encode(pixels)
        return ({"samples": latent},)
```

### [TAG:vae:tiled]
Tiled VAE for large images and video:

```python
class VAEDecodeTiled:
    def decode(self, vae, samples, tile_size=512, overlap=64, temporal_size=64, temporal_overlap=8):
        compression = vae.spacial_compression_decode()
        # temporal_compression_decode returns None if not video VAE
        temporal_compression = vae.temporal_compression_decode()

        images = vae.decode_tiled(
            samples["samples"],
            tile_x=tile_size // compression,
            tile_y=tile_size // compression,
            overlap=overlap // compression,
            tile_t=temporal_size // (temporal_compression or 1) if temporal_compression else None,
            overlap_t=temporal_overlap // (temporal_compression or 1) if temporal_compression else None
        )
        return (images,)
```

### [TAG:vae:device]
VAE device management:

```python
import comfy.model_management

# Get appropriate device
vae_device = comfy.model_management.vae_device()
offload_device = comfy.model_management.vae_offload_device()

# Load VAE to device before use
comfy.model_management.load_models_gpu([vae.patcher])
```

---

## Latents

### [TAG:latent:format]
Latent tensor format:

```python
# Latent dict structure
latent = {
    "samples": torch.Tensor,  # Shape: [B, C, (T), H, W]
    "noise_mask": torch.Tensor,  # Optional: [B, 1, (T), H, W]
    "batch_index": list[int],  # Optional: indices for batched operations
    "downscale_ratio_spacial": 8, # Optional: metadata for scaling
}
```

### [TAG:latent:operation]
Using `io.LatentOperation` for modular transformations:

```python
class LatentOperationSharpen(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LatentSharpen",
            inputs=[io.Float.Input("alpha", default=0.1)],
            outputs=[io.LatentOperation.Output()],
        )

    @classmethod
    def execute(cls, alpha) -> io.NodeOutput:
        def sharpen(latent: torch.Tensor, **kwargs) -> torch.Tensor:
            # Process latent...
            return processed_latent
        return io.NodeOutput(sharpen)

class LatentApplyOperation(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LatentApplyOp",
            inputs=[io.Latent.Input("samples"), io.LatentOperation.Input("operation")],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, samples, operation) -> io.NodeOutput:
        s = samples.copy()
        s["samples"] = operation(latent=s["samples"])
        return io.NodeOutput(s)
```

### [TAG:latent:empty]
Creating empty latents:

```python
class EmptyLatentImage:
    def __init__(self):
        # Use intermediate device for latents
        self.device = comfy.model_management.intermediate_device()

    def generate(self, width, height, batch_size=1):
        # SD uses 4 channels, dimensions / 8
        latent = torch.zeros(
            [batch_size, 4, height // 8, width // 8],
            device=self.device
        )
        return ({"samples": latent},)
```

### [TAG:latent:upscale]
Latent upscaling:

```python
import comfy.utils

class LatentUpscale:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]

    def upscale(self, samples, upscale_method, width, height):
        s = samples.copy()
        # Upscale in latent space (/ 8)
        s["samples"] = comfy.utils.common_upscale(
            samples["samples"],
            width // 8,
            height // 8,
            upscale_method,
            "disabled"  # crop mode
        )
        return (s,)
```

### [TAG:latent:composite]
Latent compositing:

```python
def composite_latents(samples_to, samples_from, x, y, feather=0):
    # Convert pixel coords to latent space
    x = x // 8
    y = y // 8
    feather = feather // 8

    s = samples_to["samples"].clone()
    s_from = samples_from["samples"]

    # Simple paste (no feather)
    if feather == 0:
        s[:, :, y:y+s_from.shape[2], x:x+s_from.shape[3]] = s_from
    else:
        # Apply feathered blending
        # ... feather mask calculation
        pass

    return {"samples": s}
```

---

## Images

### [TAG:image:format]
Tensor format convention:

```python
# IMAGE type: [Batch, Height, Width, Channels] - values 0-1
# Channels: RGB (3) or RGBA (4)

# Converting from PyTorch format (B,C,H,W) to ComfyUI (B,H,W,C):
comfy_image = pytorch_tensor.movedim(1, -1)  # or .permute(0, 2, 3, 1)

# Converting from ComfyUI (B,H,W,C) to PyTorch (B,C,H,W):
pytorch_tensor = comfy_image.movedim(-1, 1)  # or .permute(0, 3, 1, 2)
```

### [TAG:image:load]
Image loading and processing:

```python
import torch
from PIL import Image
import numpy as np

class LoadImage:
    def load(self, image_path):
        # Load with PIL
        img = Image.open(image_path)
        img = img.convert("RGB")

        # Convert to tensor: [H, W, C] -> [1, H, W, C], normalized to 0-1
        img_tensor = torch.from_numpy(np.array(img).astype(np.float32) / 255.0)
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

        return (img_tensor,)
```

### [TAG:image:upscale]
Image upscaling:

```python
import comfy.utils

def upscale_image(image, width, height, method="bilinear"):
    # Convert B,H,W,C -> B,C,H,W for processing
    samples = image.movedim(-1, 1)

    # Upscale
    samples = comfy.utils.common_upscale(samples, width, height, method, "disabled")

    # Convert back to B,H,W,C
    return samples.movedim(1, -1)
```

### [TAG:image:batch]
Batch handling:

```python
def process_batch(images):
    # images shape: [B, H, W, C]
    batch_size = images.shape[0]

    results = []
    for i in range(batch_size):
        single = images[i:i+1]  # Keep batch dimension
        processed = process_single(single)
        results.append(processed)

    return torch.cat(results, dim=0)
```

### [TAG:image:mask]
MASK type:

```python
# MASK type: [B, H, W] or [H, W] - values 0-1
# 1.0 = affected, 0.0 = unaffected

def create_mask(height, width, batch_size=1):
    return torch.ones([batch_size, height, width])

# Ensure mask has batch dimension
if len(mask.shape) < 3:
    mask = mask.unsqueeze(0)
```

---

## Common Patterns

### [TAG:pattern:clone-before-modify]
Clone before modify:

```python
new_model = model.clone()
new_dict = old_dict.copy()
new_cond = [c.copy() for c in conditioning]
```

### [TAG:pattern:type-shapes]
Common type shapes:

| Type | Shape | Value Range |
|------|-------|-------------|
| `IMAGE` | `[B, H, W, C]` | 0.0 - 1.0 |
| `MASK` | `[B, H, W]` or `[H, W]` | 0.0 - 1.0 |
| `LATENT` | `{"samples": [B, C, (T), H, W]}` | ~-4.0 to 4.0 |
| `CONDITIONING` | `[[tensor, dict], ...]` | - |
| `SIGMAS` | `[steps + 1]` | σ_max to 0.0 |

### [TAG:pattern:memory]
Memory management best practices:

```python
import comfy.model_management

# Before heavy operations
free_mem = comfy.model_management.get_free_memory(device)

# Ensure models are loaded
comfy.model_management.load_models_gpu([model_patcher])

# After operations, allow cleanup
comfy.model_management.soft_empty_cache()
```

### [TAG:pattern:error-handling]
Error handling patterns:

```python
class MyNode:
    def execute(self, model, image):
        # Validate inputs
        if model is None:
            raise RuntimeError("Model input is None")

        if image.shape[0] == 0:
            raise ValueError("Empty image batch")

        # Check compatibility
        if image.shape[-1] not in [3, 4]:
            raise ValueError(f"Expected RGB or RGBA image, got {image.shape[-1]} channels")

        # ... process
```

### [TAG:pattern:debug]
Debugging tips:

```python
# Print tensor shapes - most bugs come from shape mismatches
print(f"Image shape: {image.shape}")  # Should be [B, H, W, C]
print(f"Latent shape: {samples['samples'].shape}")  # Should be [B, 4, H/8, W/8]

# Check devices - tensors must be on the same device
print(f"Tensor device: {tensor.device}")
print(f"Model device: {model.device}")

# Verify dtypes - mixed precision can cause issues
print(f"Weight dtype: {weight.dtype}")
print(f"Input dtype: {input.dtype}")
```

---

## File Paths Reference

### [TAG:reference:files]
Key source file paths:

| Purpose | Path |
|---------|------|
| Core nodes | `nodes.py` |
| Extra nodes | `comfy_extras/` |
| Sampling | `comfy/samplers.py`, `comfy/k_diffusion/sampling.py` |
| Model loading | `comfy/sd.py` |
| Model patching | `comfy/model_patcher.py` |
| LoRA | `comfy/lora.py` |
| Device management | `comfy/model_management.py` |
| Operations | `comfy/ops.py` |
| Hooks | `comfy/hooks.py` |
| ControlNet | `comfy/controlnet.py` |
| CLIP/Text encoders | `comfy/sd1_clip.py`, `comfy/sdxl_clip.py` |
| Conditioning | `comfy/conds.py` |
| Utilities | `comfy/utils.py`, `node_helpers.py` |
| V3 API types | `comfy_api/latest/` |
