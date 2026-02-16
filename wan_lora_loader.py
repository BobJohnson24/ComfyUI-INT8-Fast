from collections import OrderedDict
import os
import copy
import torch
import torch.nn.functional as F
import folder_paths
import comfy.utils
import gc
import threading
import hashlib
import math
from .lora_utils import parse_wan_lora

_object_patches_lock = threading.RLock()
_tensor_hash_lock = threading.Lock()

def _parse_env_int(name, default):
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        print(f"Warning: Invalid value for {name}, using default: {default}")
        return default

def _parse_env_bool(name, default="0"):
    val = os.environ.get(name, default)
    return val == "1"

_ENABLE_CUDA_SYNC = _parse_env_bool("INT8_ENABLE_CUDA_SYNC", "0")
_CLEAR_CACHE_STRATEGY = os.environ.get("INT8_CLEAR_CACHE", "auto")
_MAX_LORA_CACHE_SIZE = _parse_env_int("INT8_LORA_CACHE_SIZE", 8)

def _aggressive_memory_cleanup():
    """Aggressive memory cleanup for OOM situations."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

_MAX_FULL_HASH_ELEMENTS = 64
_HASH_SAMPLE_COUNT = 16

def _should_clear_cache():
    if _CLEAR_CACHE_STRATEGY == "always":
        return True
    if _CLEAR_CACHE_STRATEGY == "never":
        return False
    if not torch.cuda.is_available():
        return False
    try:
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        if reserved > 0:
            utilization = allocated / reserved
            return utilization < 0.8
        if allocated > 0:
            return False
    except Exception:
        pass
    return True


def _get_tensor_content_hash(t):
    """Get a stable hash for tensor identity based on content, not memory address."""
    if not isinstance(t, torch.Tensor):
        return (id(t),)
    
    # Use shape, dtype, device as base identity (stable across moves)
    base_id = (t.shape, t.dtype, str(t.device))
    
    # For small tensors, hash a sample of values for stronger identity
    # This prevents different tensors with same shape from colliding
    if t.numel() <= _HASH_SAMPLE_COUNT:
        # Hash all values for small tensors
        try:
            value_hash = hash(t.detach().cpu().numpy().tobytes())
            return base_id + (value_hash,)
        except:
            pass
    elif t.numel() > 0:
        # Sample a few values for larger tensors
        try:
            sample_indices = [0, t.numel() // 4, t.numel() // 2, 3 * t.numel() // 4, t.numel() - 1]
            sample_indices = [i for i in sample_indices if i < t.numel()]
            samples = t.flatten()[sample_indices].detach().cpu().numpy().tobytes()
            value_hash = hash(samples)
            return base_id + (value_hash,)
        except:
            pass
    
    return base_id


def _get_patch_identity(patches):
    """Get a stable identity for patches list that avoids id() reuse issues."""
    if not patches:
        return ()
    
    try:
        patches_snapshot = tuple(patches)
    except (RuntimeError, TypeError):
        import time
        return ("unstable", id(patches), time.time())
    
    identities = []
    for patch in patches_snapshot:
        if isinstance(patch, (list, tuple)) and len(patch) >= 2:
            d, u = patch[0], patch[1]
            patch_id = (
                _get_tensor_content_hash(d),
                _get_tensor_content_hash(u),
                len(patch)
            )
            if len(patch) >= 3:
                alpha = patch[2]
                if isinstance(alpha, (int, float)):
                    patch_id = patch_id + (alpha,)
                elif isinstance(alpha, torch.Tensor):
                    patch_id = patch_id + (float(alpha.item()),)
            identities.append(patch_id)
        else:
            identities.append((type(patch).__name__, id(patch)))
    return tuple(identities)


def _is_valid_alpha(a):
    """Check if alpha value is valid (not None, NaN, or zero)."""
    if a is None:
        return False
    
    if isinstance(a, torch.Tensor):
        if a.numel() == 0:
            return False
        try:
            a_val = a.item()
            return not (math.isnan(a_val) or a_val == 0)
        except:
            return False
    
    if isinstance(a, (int, float)):
        return not (a != a or a == 0)
    
    return False


def _is_zero_alpha(a):
    """Check if alpha is exactly zero (for warning suppression)."""
    if a is None:
        return False
    
    if isinstance(a, torch.Tensor):
        try:
            return a.item() == 0
        except:
            return False
    
    return a == 0


def _parse_patch(patch):
    """Normalize LoRA patch tuple parsing."""
    if not isinstance(patch, (list, tuple)):
        return None

    if len(patch) == 3:
        d, u, a = patch
        return d, u, a, None, None, 0, 0
    if len(patch) == 5:
        d, u, a, d_scale, u_scale = patch
        return d, u, a, d_scale, u_scale, 0, 0
    if len(patch) == 7:
        return patch
    return None


def _identify_qkv_component(key: str) -> tuple:
    """Identify which QKV component a LoRA key targets using word boundary-aware matching."""
    key_lower = key.lower()
    
    # Check for fused QKV patterns first (highest priority)
    fused_patterns = ['.qkv.', '_qkv.', '.to_qkv.', '_to_qkv', 'qkv_proj']
    for pattern in fused_patterns:
        if pattern in key_lower:
            return (None, True)
    
    # Use more specific patterns with word boundaries to avoid false matches
    # Order matters: check more specific patterns first
    
    # Q patterns - from most specific to least specific
    q_patterns = [
        '.to_q.',      # Standard ComfyUI/Wan pattern
        '_to_q_',      # Alternative separator
        '.q_proj.',    # HuggingFace style
        '_q_proj_',    # Alternative separator
        '_attn_q',     # Short form attention
        '/q/',         # Path style
    ]
    
    k_patterns = [
        '.to_k.',
        '_to_k_',
        '.k_proj.',
        '_k_proj_',
        '_attn_k',
        '/k/',
    ]
    
    v_patterns = [
        '.to_v.',
        '_to_v_',
        '.v_proj.',
        '_v_proj_',
        '_attn_v',
        '/v/',
    ]
    
    # Check for Q (Query)
    for pattern in q_patterns:
        if pattern in key_lower:
            # Verify it's not a substring of a longer word (like 'query' containing 'q')
            idx = key_lower.find(pattern)
            if idx >= 0:
                # Check if preceded by non-alphanumeric (or start of string)
                before = key_lower[idx-1] if idx > 0 else None
                if before is None or not before.isalnum():
                    return ('q', False)
    
    # Check for K (Key)
    for pattern in k_patterns:
        if pattern in key_lower:
            idx = key_lower.find(pattern)
            if idx >= 0:
                before = key_lower[idx-1] if idx > 0 else None
                if before is None or not before.isalnum():
                    return ('k', False)
    
    # Check for V (Value)
    for pattern in v_patterns:
        if pattern in key_lower:
            idx = key_lower.find(pattern)
            if idx >= 0:
                before = key_lower[idx-1] if idx > 0 else None
                if before is None or not before.isalnum():
                    return ('v', False)
    
    return (None, False)


def _validate_lora_bounds(offset, size, out_dim, key=""):
    """Validate and adjust LoRA patch bounds to fit output tensor."""
    if offset < 0:
        print(f"Warning: Negative offset {offset} for {key}, clamping to 0")
        offset = 0
    
    if size < 0:
        print(f"Warning: Negative size {size} for {key}, treating as full size")
        size = 0
    
    if size == 0:
        size = out_dim - offset
    
    if offset >= out_dim:
        print(f"Warning: Offset {offset} >= output dim {out_dim} for {key}, skipping")
        return 0, 0, False
    
    if offset + size > out_dim:
        old_size = size
        size = out_dim - offset
        print(f"Warning: Clamping size from {old_size} to {size} for {key}")
    
    if size <= 0:
        return 0, 0, False
    
    return offset, size, True


class LoRAWeightCache:
    """Thread-safe LRU cache for LoRA weights with memory awareness."""

    def __init__(self, max_size=_MAX_LORA_CACHE_SIZE):
        self.max_size = max_size
        self._cache = OrderedDict()
        self._lock = threading.RLock()
        self._ref_counts = {}

    def _check_memory_pressure(self):
        if not torch.cuda.is_available():
            return False
        try:
            free_mem, total_mem = torch.cuda.mem_get_info()
            return free_mem < total_mem * 0.15
        except Exception:
            return False

    def _release_tensor(self, value):
        if isinstance(value, dict):
            for v in value.values():
                if isinstance(v, torch.Tensor):
                    del v
        elif isinstance(value, torch.Tensor):
            del value

    def _evict_one(self):
        for key in list(self._cache.keys()):
            if self._ref_counts.get(key, 0) == 0:
                value = self._cache.pop(key, None)
                if value is not None:
                    self._release_tensor(value)
                self._ref_counts.pop(key, None)
                return True
        return False

    def _evict_for_memory(self):
        evicted = 0
        while self._check_memory_pressure():
            if not self._evict_one():
                break
            evicted += 1
        if evicted > 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return evicted

    def get_or_create(self, key, factory_fn):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._ref_counts[key] = self._ref_counts.get(key, 0) + 1
                return self._cache[key], True  # cached

            if len(self._cache) >= self.max_size:
                evicted = self._evict_one()
                if not evicted:
                    # Return uncached value to avoid unbounded growth
                    return factory_fn(), False

            value = factory_fn()
            self._cache[key] = value
            self._ref_counts[key] = 1
            return value, True  # cached

    def release(self, key):
        with self._lock:
            if key in self._ref_counts:
                self._ref_counts[key] -= 1
                if self._ref_counts[key] <= 0:
                    del self._ref_counts[key]
                    if self._check_memory_pressure():
                        value = self._cache.pop(key, None)
                        if value is not None:
                            self._release_tensor(value)

    def clear(self):
        with self._lock:
            for key in list(self._cache.keys()):
                if self._ref_counts.get(key, 0) == 0:
                    value = self._cache.pop(key, None)
                    if value is not None:
                        self._release_tensor(value)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class LoRAWrapperModule(torch.nn.Module):
    def __init__(self, wrapped_module, lora_patches_list):
        super().__init__()
        if lora_patches_list is not None and not isinstance(lora_patches_list, (list, tuple)):
            raise TypeError(f"lora_patches_list must be a list or tuple, got {type(lora_patches_list).__name__}")
        
        self._getting_attr = threading.local()
        self._getattr_lock = threading.Lock()
        
        self._wrapped_module = wrapped_module
        self.lora_patches = lora_patches_list if lora_patches_list is not None else []
        self._lora_weight_cache = LoRAWeightCache(_MAX_LORA_CACHE_SIZE)
        self._last_patch_ids = None
        self._patch_ids_lock = threading.Lock()

    def __setstate__(self, state):
        """Restore object state from pickle, handling backwards compatibility."""
        torch.nn.Module.__init__(self)
        
        if 'wrapped_module' in state and '_wrapped_module' not in state:
            state['_wrapped_module'] = state.pop('wrapped_module')
        
        if '_wrapped_module' not in state:
            state['_wrapped_module'] = None
        
        wrapped = state.pop('_wrapped_module', None)
        lora_patches = state.pop('lora_patches', [])
        
        state.pop('_getting_attr', None)
        state.pop('_getattr_lock', None)
        state.pop('_patch_ids_lock', None)
        state.pop('_lora_weight_cache', None)
        state.pop('_last_patch_ids', None)
        
        self.__dict__.update(state)
        self.wrapped_module = wrapped
        self.lora_patches = lora_patches
        
        self._getting_attr = threading.local()
        self._getattr_lock = threading.Lock()
        self._patch_ids_lock = threading.Lock()
        self._lora_weight_cache = LoRAWeightCache(_MAX_LORA_CACHE_SIZE)
        self._last_patch_ids = None

    @property
    def wrapped_module(self):
        if '_wrapped_module' in self.__dict__:
            return self.__dict__['_wrapped_module']
        if 'wrapped_module' in self.__dict__:
            return self.__dict__['wrapped_module']
        raise AttributeError("wrapped_module not set")
    
    @wrapped_module.setter
    def wrapped_module(self, value):
        self._wrapped_module = value

    @property
    def weight(self):
        """Expose weight for ComfyUI compatibility."""
        wrapped = self.wrapped_module
        if wrapped is not None and hasattr(wrapped, 'weight'):
            return wrapped.weight
        raise AttributeError("wrapped module has no weight")

    @property
    def bias(self):
        """Expose bias for ComfyUI compatibility."""
        wrapped = self.wrapped_module
        if wrapped is not None and hasattr(wrapped, 'bias'):
            return wrapped.bias
        return None

    def __deepcopy__(self, memo):
        """Custom deepcopy that handles threading.local() and locks."""
        new_instance = object.__new__(self.__class__)
        memo[id(self)] = new_instance
        
        torch.nn.Module.__init__(new_instance)
        
        wrapped = self.wrapped_module
        if wrapped is not None:
            new_instance._wrapped_module = copy.deepcopy(wrapped, memo)
        else:
            new_instance._wrapped_module = None
        
        new_instance.lora_patches = copy.deepcopy(self.lora_patches, memo)
        
        new_instance._getting_attr = threading.local()
        new_instance._getattr_lock = threading.Lock()
        new_instance._patch_ids_lock = threading.Lock()
        
        new_instance._lora_weight_cache = LoRAWeightCache(_MAX_LORA_CACHE_SIZE)
        new_instance._last_patch_ids = None
        
        return new_instance

    def __reduce__(self):
        """Support proper pickling by providing reconstruction info."""
        wrapped = self.wrapped_module
        patches = self.lora_patches
        return (
            self.__class__,
            (wrapped, patches),
        )

    def _is_getting_attr(self, name):
        """Check if attribute is being accessed (per-thread recursion guard)."""
        if not hasattr(self._getting_attr, 'set'):
            self._getting_attr.set = set()
            return False
        return name in self._getting_attr.set

    def _add_getting_attr(self, name):
        """Add attribute to recursion guard set."""
        try:
            self._getting_attr.set.add(name)
        except AttributeError:
            self._getting_attr.set = {name}

    def _remove_getting_attr(self, name):
        """Remove attribute from recursion guard set."""
        try:
            self._getting_attr.set.discard(name)
        except AttributeError:
            pass

    def __getattr__(self, name):
        if name in ('_wrapped_module', 'wrapped_module'):
            if '_wrapped_module' in self.__dict__:
                return self.__dict__['_wrapped_module']
            if 'wrapped_module' in self.__dict__:
                return self.__dict__['wrapped_module']
            if '_modules' in self.__dict__:
                if 'wrapped_module' in self._modules:
                    return self._modules['wrapped_module']
                if '_wrapped_module' in self._modules:
                    return self._modules['_wrapped_module']
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        try:
            getting_attr = object.__getattribute__(self, '_getting_attr')
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        if self._is_getting_attr(name):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}' (recursion detected)")
        
        self._add_getting_attr(name)
        try:
            wrapped = None
            if '_wrapped_module' in self.__dict__:
                wrapped = self.__dict__['_wrapped_module']
            elif 'wrapped_module' in self.__dict__:
                wrapped = self.__dict__['wrapped_module']
            
            if wrapped is not None:
                return getattr(wrapped, name)
            
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        finally:
            self._remove_getting_attr(name)

    def __delattr__(self, name):
        """Safe attribute deletion with protected attributes."""
        _protected_attrs = {'_getting_attr', '_getattr_lock', '_patch_ids_lock',
                            '_wrapped_module', 'lora_patches', '_lora_weight_cache'}
        
        if name in _protected_attrs:
            raise AttributeError(f"Cannot delete protected attribute '{name}'")
        
        if name in self.__dict__:
            object.__delattr__(self, name)
            return
        
        if name in self._modules:
            del self._modules[name]
            return
        if name in self._parameters:
            del self._parameters[name]
            return
        if name in self._buffers:
            del self._buffers[name]
            return
        
        try:
            wrapped = self.wrapped_module
        except AttributeError:
            wrapped = None
        if wrapped is not None:
            try:
                delattr(wrapped, name)
                return
            except AttributeError:
                pass
        
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def forward(self, x):
        wrapped = self.wrapped_module
        original_out = wrapped(x)

        try:
            from .int8_quant import chunked_int8_lora_forward, chunked_lora_forward
        except ImportError as e:
            print(f"ERROR: Failed to import int8_quant module: {e}")
            return original_out

        with self._patch_ids_lock:
            patches = list(self.lora_patches)
            if not patches:
                return original_out

            current_patch_ids = _get_patch_identity(patches)
            if self._last_patch_ids != current_patch_ids:
                self._lora_weight_cache.clear()
            self._last_patch_ids = current_patch_ids

        x_2d = x.reshape(-1, x.shape[-1])
        if not x_2d.is_contiguous():
            x_2d = x_2d.contiguous()

        out = original_out.clone()
        keys_to_release = set()
        any_patch_applied = False

        try:
            for patch_data in patches:
                parsed = _parse_patch(patch_data)
                if parsed is None:
                    continue

                d, u, a, d_scale, u_scale, offset, size = parsed
                if not all(isinstance(t, torch.Tensor) for t in (d, u)):
                    continue
                if not _is_valid_alpha(a):
                    continue

                expected_input_dim = d.shape[1]
                if expected_input_dim != x_2d.shape[-1]:
                    continue

                offset, size, is_valid = _validate_lora_bounds(
                    offset, size, original_out.shape[-1],
                    key=f"patch_{id(patch_data)}"
                )
                if not is_valid:
                    continue

                is_int8 = d.dtype == torch.int8 and u.dtype == torch.int8
                if is_int8 and d_scale is not None and u_scale is not None:
                    # Use stable content-based hash to avoid cache collisions from id() reuse
                    def _get_scale_key(scale, name="scale"):
                        if scale is None:
                            return None
                        if isinstance(scale, torch.Tensor):
                            if scale.numel() == 1:
                                return float(scale.item())
                            # Use stable content hash instead of id()
                            return _get_tensor_content_hash(scale)
                        return float(scale)
                    
                    # Also include alpha in cache key for consistency
                    alpha_key = float(a) if isinstance(a, (int, float)) else hash(str(a))
                    
                    cache_key = (
                        "int8",
                        d.shape, d.dtype, u.shape, u.dtype,
                        _get_scale_key(d_scale, "down_scale"),
                        _get_scale_key(u_scale, "up_scale"),
                        offset, size, original_out.shape[-1],
                        alpha_key,  # Include alpha to prevent collision
                    )

                    def create_int8_cache(d=d, u=u, d_scale=d_scale, u_scale=u_scale, device=original_out.device):
                        return {
                            "d": d.to(device=device, non_blocking=True),
                            "u": u.to(device=device, non_blocking=True),
                            "d_scale": d_scale.to(device=device, non_blocking=True) if isinstance(d_scale, torch.Tensor) else d_scale,
                            "u_scale": u_scale.to(device=device, non_blocking=True) if isinstance(u_scale, torch.Tensor) else u_scale,
                        }

                    cached, is_cached = self._lora_weight_cache.get_or_create(cache_key, create_int8_cache)
                    if is_cached:
                        keys_to_release.add(cache_key)

                    try:
                        chunked_int8_lora_forward(
                            x_2d, cached["d"], cached["u"],
                            cached["d_scale"], cached["u_scale"],
                            a, out, offset=offset, size=size
                        )
                        any_patch_applied = True
                    except Exception as e:
                        # Release cache key on error to prevent reference count leak
                        if cache_key in keys_to_release:
                            self._lora_weight_cache.release(cache_key)
                            keys_to_release.discard(cache_key)
                        raise
                else:
                    curr_x = x_2d if x_2d.dtype == original_out.dtype else x_2d.to(dtype=original_out.dtype)

                    # Include alpha in cache key to prevent collision between patches with different alphas
                    alpha_key = float(a) if isinstance(a, (int, float)) else hash(str(a))
                    
                    cache_key = (
                        "float",
                        d.shape, d.dtype, u.shape, u.dtype,
                        original_out.dtype, offset, size, original_out.shape[-1],
                        alpha_key,  # Now includes alpha to prevent collision
                    )

                    def create_float_cache(d=d, u=u, device=original_out.device, dtype=original_out.dtype):
                        return {
                            "d": d.to(device=device, dtype=dtype, non_blocking=True),
                            "u": u.to(device=device, dtype=dtype, non_blocking=True),
                        }

                    cached, is_cached = self._lora_weight_cache.get_or_create(cache_key, create_float_cache)
                    if is_cached:
                        keys_to_release.add(cache_key)

                    try:
                        chunked_lora_forward(
                            curr_x, cached["d"], cached["u"],
                            a, out, offset=offset, size=size
                        )
                        any_patch_applied = True
                    except Exception as e:
                        # Release cache key on error to prevent reference count leak
                        if cache_key in keys_to_release:
                            self._lora_weight_cache.release(cache_key)
                            keys_to_release.discard(cache_key)
                        raise

            if any_patch_applied:
                return out

            del out
            return original_out

        finally:
            for key in keys_to_release:
                try:
                    self._lora_weight_cache.release(key)
                except Exception:
                    pass  # Ignore release errors in finally block


class WanLoRALoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "offload_to_cpu": (["enable", "disable"], {"default": "disable"}),
            },
            "optional": {
                "debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lora"
    CATEGORY = "WanVideo/INT8"

    def load_lora(self, model, lora_name, strength, offload_to_cpu="disable", debug=False):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            free_mem, total_mem = torch.cuda.mem_get_info()
            if free_mem < total_mem * 0.1:
                print(f"Warning: Low GPU memory ({free_mem / 1e9:.2f}GB free). "
                      f"Consider reducing resolution or batch size.")

        if strength == 0:
            return (model,)

        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not lora_path:
            print(f"LoRA not found: {lora_name}")
            return (model,)

        print(f"Loading LoRA: {lora_name}")

        lora_state_dict = None
        lora_weights = None
        modules = None
        
        try:
            lora_state_dict = comfy.utils.load_torch_file(lora_path)
            lora_weights = parse_wan_lora(lora_state_dict, strength, debug=debug)
            
            if lora_state_dict is not None:
                del lora_state_dict
                lora_state_dict = None
            gc.collect()

            if lora_weights is None:
                print(f"Failed to parse LoRA: {lora_name}")
                return (model,)

            new_model = model.clone()
            
            # Clear stale patches from INT8 modules to ensure clean state
            torch_model_for_cleanup = new_model.model
            if hasattr(torch_model_for_cleanup, 'named_modules'):
                for name, mod in torch_model_for_cleanup.named_modules():
                    if hasattr(mod, 'lora_patches') and hasattr(mod, '_is_quantized'):
                        mod.lora_patches = []
                        if hasattr(mod, '_lora_weight_cache'):
                            mod._lora_weight_cache.clear()
            del torch_model_for_cleanup
            
            with _object_patches_lock:
                if not hasattr(new_model, "object_patches"):
                    new_model.object_patches = {}
                else:
                    original_patches = new_model.object_patches
                    new_object_patches = {}
                    
                    for key, value in list(original_patches.items()):
                        if isinstance(value, LoRAWrapperModule):
                            value._lora_weight_cache.clear()

                            try:
                                wrapped = value.wrapped_module
                                patches = list(value.lora_patches) if value.lora_patches else []
                                new_object_patches[key] = LoRAWrapperModule(wrapped, patches)
                            except Exception as e:
                                print(f"Warning: Rebuilding patch {key} failed: {e}")
                                new_object_patches[key] = value
                        else:
                            try:
                                new_object_patches[key] = copy.deepcopy(value)
                            except Exception as e:
                                print(f"Warning: Failed to deepcopy patch {key}: {e}")
                                new_object_patches[key] = value
                    
                    new_model.object_patches = new_object_patches

            torch_model = new_model.model
            modules = dict(torch_model.named_modules())

            patched_count = 0
            patches_to_apply = []

            for key in lora_weights.weights:
                target_module = None
                target_key = None

                candidates = [key]
                
                if key.startswith("diffusion_model."):
                    candidates.append(key[len("diffusion_model."):])
                else:
                    candidates.append("diffusion_model." + key)

                var_candidates = []
                for c in candidates:
                    if ".self_attn." in c:
                        var_candidates.append(c.replace(".self_attn.", ".attn."))
                    elif ".attn." in c:
                        var_candidates.append(c.replace(".attn.", ".self_attn."))
                    if ".to_out.0" in c:
                        var_candidates.append(c.replace(".to_out.0", ".to_out"))
                    elif ".to_out" in c and ".to_out.0" not in c:
                        var_candidates.append(c.replace(".to_out", ".to_out.0"))

                candidates = candidates + var_candidates

                replacements = [
                    (".to_q", ".q"), (".to_k", ".k"), (".to_v", ".v"), (".to_out", ".o"),
                    (".q_proj", ".qkv"), (".k_proj", ".qkv"), (".v_proj", ".qkv"),
                    (".to_q", ".qkv"), (".to_k", ".qkv"), (".to_v", ".qkv")
                ]

                final_pass_candidates = list(candidates)
                for c in candidates:
                    for old, new in replacements:
                        if old in c:
                            final_pass_candidates.append(c.replace(old, new))

                candidates = final_pass_candidates

                patch_offset = 0
                patch_size = 0
                qkv_matched = False

                for cand in candidates:
                    if cand in modules:
                        target_module = modules[cand]
                        target_key = cand

                        if ".qkv" in cand and isinstance(target_module, torch.nn.Linear):
                            total_out = target_module.weight.shape[0]
                            
                            if total_out % 3 != 0:
                                print(f"Warning: Cannot apply LoRA to QKV module {cand}: "
                                      f"output dimension {total_out} is not divisible by 3. "
                                      f"This LoRA patch will be skipped for key: {key}")
                                target_module = None
                                target_key = None
                                break
                            
                            head_dim = total_out // 3
                            
                            is_int8 = lora_weights.is_int8.get(key, False)
                            if is_int8:
                                down, up, down_scale, up_scale, alpha = lora_weights.weights[key]
                            else:
                                down, up, alpha = lora_weights.weights[key]
                            
                            lora_out_dim = up.shape[0] if up is not None and hasattr(up, 'shape') else 0
                            
                            if lora_out_dim == total_out:
                                patch_offset = 0
                                patch_size = 0
                                qkv_matched = True
                                if debug:
                                    print(f"Full QKV LoRA detected for {key}: out_dim={lora_out_dim}")
                            elif lora_out_dim == head_dim:
                                component, is_fused = _identify_qkv_component(key)
                                
                                if component == 'q':
                                    patch_offset = 0
                                    patch_size = head_dim
                                    qkv_matched = True
                                elif component == 'k':
                                    patch_offset = head_dim
                                    patch_size = head_dim
                                    qkv_matched = True
                                elif component == 'v':
                                    patch_offset = head_dim * 2
                                    patch_size = head_dim
                                    qkv_matched = True
                                else:
                                    print(f"Warning: Could not determine Q/K/V type for partial LoRA key: {key}")
                                    target_module = None
                                    target_key = None
                            elif lora_out_dim > 0:
                                print(f"Warning: LoRA output dimension {lora_out_dim} doesn't match "
                                      f"expected full QKV ({total_out}) or single head ({head_dim}) for {key}")
                                target_module = None
                                target_key = None
                            break

                if target_module is not None and isinstance(target_module, torch.nn.Linear):
                    is_int8 = lora_weights.is_int8.get(key, False)
                    if is_int8:
                        down, up, down_scale, up_scale, alpha = lora_weights.weights[key]
                    else:
                        down, up, alpha = lora_weights.weights[key]
                        down_scale = None
                        up_scale = None

                    if hasattr(target_module, "weight"):
                        mod_out, mod_in = target_module.weight.shape

                        if patch_size > 0:
                            if patch_offset + patch_size > mod_out:
                                if debug:
                                    print(f"Warning: clamping patch for {key} from offset={patch_offset}, size={patch_size} to fit mod_out={mod_out}")
                                patch_size = max(0, mod_out - patch_offset)
                            if patch_size <= 0:
                                if debug:
                                    print(f"Skipping {key}: Invalid patch size after clamping")
                                continue

                        if mod_in != down.shape[1]:
                            if debug:
                                print(f"Skipping {key}: Input mismatch {mod_in} vs {down.shape[1]}")
                            continue
                        
                        # Validate output dimension matches
                        expected_out = patch_size if patch_size > 0 else mod_out
                        if up.shape[0] != expected_out:
                            if debug:
                                print(f"Warning: {key} output dim mismatch: up.shape[0]={up.shape[0]}, expected={expected_out}")
                            # Try to handle gracefully
                            if up.shape[0] < expected_out:
                                print(f"Skipping {key}: LoRA output {up.shape[0]} < expected {expected_out}")
                                continue

                    patch_tuple = (down, up, alpha, down_scale, up_scale, patch_offset, patch_size)
                    patches_to_apply.append((target_key, target_module, patch_tuple))

            def _is_int8_module(module):
                """Check if module is an INT8 quantized linear layer."""
                try:
                    from .int8_quant import Int8TensorwiseOps
                    return (hasattr(module, '_is_quantized') and
                            module._is_quantized and
                            type(module).__name__ == 'Linear' and
                            hasattr(Int8TensorwiseOps, 'Linear') and
                            isinstance(module, Int8TensorwiseOps.Linear))
                except ImportError:
                    return False

            wrappers_to_add = {}
            int8_direct_patches = {}
            
            for target_key, target_module, patch_tuple in patches_to_apply:
                current_patches = []
                if target_key in new_model.object_patches:
                    existing_obj = new_model.object_patches[target_key]
                    if hasattr(existing_obj, "lora_patches"):
                        current_patches = list(existing_obj.lora_patches)

                new_patch_list = current_patches + [patch_tuple]

                raw_module = target_module
                
                if isinstance(raw_module, LoRAWrapperModule):
                    raw_module = raw_module.wrapped_module
                
                if target_key in new_model.object_patches:
                    existing = new_model.object_patches[target_key]
                    if isinstance(existing, LoRAWrapperModule):
                        raw_module = existing.wrapped_module
                    elif isinstance(existing, torch.nn.Linear):
                        raw_module = existing
                
                while isinstance(raw_module, LoRAWrapperModule):
                    raw_module = raw_module.wrapped_module

                if _is_int8_module(raw_module):
                    module_id = id(raw_module)
                    if module_id not in int8_direct_patches:
                        int8_direct_patches[module_id] = (target_key, raw_module, [])
                    int8_direct_patches[module_id][2].append(patch_tuple)
                    if debug:
                        print(f"[LoRA] Direct INT8 patch for {target_key} (accumulated {len(int8_direct_patches[module_id][2])} patches)")
                else:
                    if target_key not in wrappers_to_add:
                        wrappers_to_add[target_key] = (raw_module, [])
                    wrappers_to_add[target_key][1].append(patch_tuple)
                    if debug:
                        print(f"[LoRA] Wrapper patch for {target_key} (accumulated {len(wrappers_to_add[target_key][1])} patches)")

            with _object_patches_lock:
                for module_id, (target_key, int8_module, patch_list) in int8_direct_patches.items():
                    int8_module.lora_patches = patch_list
                    patched_count += len(patch_list)
                    if target_key not in new_model.object_patches:
                        new_model.add_object_patch(target_key, int8_module)
                
                for target_key, (raw_module, patch_list) in wrappers_to_add.items():
                    wrapper = LoRAWrapperModule(raw_module, patch_list)
                    new_model.add_object_patch(target_key, wrapper)
                    patched_count += len(patch_list)

            int8_count = len(int8_direct_patches)
            wrapper_count = len(wrappers_to_add)
            if int8_count > 0:
                print(f"Applied {patched_count} LoRA patches ({int8_count} direct INT8, {wrapper_count} wrapped).")
            else:
                print(f"Applied {patched_count} LoRA patches.")
            return (new_model,)
            
        except Exception as e:
            print(f"Error loading LoRA {lora_name}: {e}")
            raise
            
        finally:
            if lora_state_dict is not None:
                del lora_state_dict
            if lora_weights is not None:
                del lora_weights
            if modules is not None:
                del modules
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
