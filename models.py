import logging
from VisionTransformer import VisionTransformer
from jt_extension import resize_pos_embed
import jittor as jt
import numpy as np

_logger = logging.getLogger(__name__)

DEFAULT_CROP_PCT = 0.875
DEFAULT_CROP_MODE = 'center'
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)
IMAGENET_DPN_MEAN = (124 / 255, 117 / 255, 104 / 255)
IMAGENET_DPN_STD = tuple([1 / (.0167 * 255)] * 3)
OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

def _cfg(url='', **kwargs):
    return {
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'vit_tiny_patch16_224': _cfg(
    ),
    'vit_tiny_patch16_384': _cfg(
        input_size=(3, 384, 384), crop_pct=1.0
    ),
    'vit_small_patch16_224': _cfg(
    ),
    'vit_base_patch16_224': _cfg(
    ),
     'vit_base_patch16_384': _cfg(
        input_size=(3, 384, 384), crop_pct=1.0
    ),
    'vit_large_patch16_224': _cfg(
    ),
    'vit_large_patch16_384': _cfg(
        input_size=(3, 384, 384), crop_pct=1.0
    ),

    'vit_base_patch16_224_in21k': _cfg(
        num_classes=21843,
        mean=IMAGENET_INCEPTION_MEAN,
        std=IMAGENET_INCEPTION_STD,
    ),
    'vit_base_patch16_224_dino': _cfg(
        num_classes=0,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    ),
}

model_arch_configs = {
    'vit_tiny_patch16_224': dict(patch_size=16, embed_dim=192, depth=12, num_heads=3),
    'vit_tiny_patch16_384': dict(patch_size=16, embed_dim=192, depth=12, num_heads=3),
    'vit_small_patch16_224': dict(patch_size=16, embed_dim=384, depth=12, num_heads=6),
    'vit_base_patch16_224': dict(patch_size=16, embed_dim=768, depth=12, num_heads=12),
    'vit_base_patch16_384': dict(patch_size=16, embed_dim=768, depth=12, num_heads=12),
    'vit_large_patch16_224': dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16),
    'vit_large_patch16_384': dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16),
    'vit_base_patch16_224_in21k': dict(patch_size=16, embed_dim=768, depth=12, num_heads=12),
    'vit_base_patch16_224_dino': dict(patch_size=16, embed_dim=768, depth=12, num_heads=12),
}

def load_pretrained_weights(model: 'VisionTransformer', state_dict: dict):
    model_state_dict = model.state_dict()
    loaded_keys = []
    skipped_keys_mismatch = []
    skipped_keys_missing_in_model = [] # Keys in state_dict but not in model
    missing_keys_in_checkpoint = [] # Keys in model but not in state_dict

    # Process keys present in the loaded state_dict
    for name, param_loaded in state_dict.items():
        if name not in model_state_dict:
            # This weight exists in the file but not in the current model structure
            _logger.warning(f"Weight '{name}' found in checkpoint but not in model. Skipping.")
            skipped_keys_missing_in_model.append(name)
            continue

        param_model = model_state_dict[name]

        if param_model.shape != param_loaded.shape:
            # --- Handle Shape Mismatches ---
            if name == 'pos_embed':
                _logger.info(f"Attempting to resize positional embedding '{name}' from {param_loaded.shape} to {param_model.shape}...")
                try:
                    # Ensure patch_embed and grid_size exist before accessing
                    if hasattr(model, 'patch_embed') and hasattr(model.patch_embed, 'grid_size'):
                        gs_new = model.patch_embed.grid_size
                    else:
                        _logger.warning(f"Cannot determine grid size for resizing '{name}'. Skipping resize.")
                        skipped_keys_mismatch.append((name, param_loaded.shape, param_model.shape))
                        continue # Skip to next parameter

                    if isinstance(param_loaded, np.ndarray):
                        param_loaded_jt = jt.array(param_loaded)
                    else:
                        param_loaded_jt = param_loaded

                    param_model_jt = param_model

                    param_resized = resize_pos_embed(
                        param_loaded_jt,
                        param_model_jt,
                        num_prefix=getattr(model, 'num_prefix_tokens', 1),
                        new_grid_size=gs_new
                    )

                    # Check if resize was successful and shapes match *after* resize
                    if param_resized.shape == param_model.shape:
                        param_model.assign(param_resized)
                        _logger.info(f"Successfully resized and loaded '{name}'.")
                        loaded_keys.append(name)
                    else:
                        _logger.warning(f"Skipping '{name}' due to resize failure or final shape mismatch "
                                        f"after resize ({param_resized.shape} vs {param_model.shape}).")
                        skipped_keys_mismatch.append((name, param_loaded.shape, param_model.shape))

                except Exception as e:
                    _logger.error(f"Error during resizing or assigning pos_embed '{name}': {e}. Skipping.")
                    skipped_keys_mismatch.append((name, param_loaded.shape, param_model.shape))

            elif name.startswith('head.'): # Handle classification head mismatch
                _logger.warning(f"Skipping '{name}' due to shape mismatch (likely different num_classes). "
                                f"Checkpoint: {param_loaded.shape}, Model: {param_model.shape}. Model head remains initialized.")
                skipped_keys_mismatch.append((name, param_loaded.shape, param_model.shape))

            elif name == 'cls_token' and not model.class_token:
                 # Handle case where checkpoint has cls_token but model doesn't
                 _logger.warning(f"Skipping 'cls_token' from checkpoint as model does not use class token (model.class_token=False).")
                 skipped_keys_mismatch.append((name, param_loaded.shape, 'None in model'))

            else: # General shape mismatch
                _logger.warning(f"Skipping '{name}' due to shape mismatch. "
                                f"Checkpoint: {param_loaded.shape}, Model: {param_model.shape}")
                skipped_keys_mismatch.append((name, param_loaded.shape, param_model.shape))
        else:
            # --- Shapes Match: Load the weight ---
            try:
                param_model.assign(param_loaded)
                loaded_keys.append(name)
            except Exception as e:
                 _logger.error(f"Error assigning matching weights for '{name}': {e}")
                 # Treat as mismatch if assign fails
                 skipped_keys_mismatch.append((name, param_loaded.shape, param_model.shape))

    # Check for keys present in the model but missing in the checkpoint
    for name in model_state_dict.keys():
        if name not in state_dict:
            # Prompt parameters are expected to be missing if added newly
            if not name.startswith('prompt.'):
                _logger.warning(f"Parameter '{name}' found in model but not in checkpoint. Weights remain initialized.")
            missing_keys_in_checkpoint.append(name)

    # --- Final Logging ---
    _logger.info("-" * 50)
    _logger.info(f"Weight Loading Summary:")
    _logger.info(f"  Successfully loaded: {len(loaded_keys)} parameters.")
    if skipped_keys_mismatch:
        _logger.warning(f"  Skipped due to shape mismatch: {len(skipped_keys_mismatch)} parameters.")
        for name, shape_loaded, shape_model in skipped_keys_mismatch:
            _logger.warning(f"    - {name}: Checkpoint {shape_loaded}, Model {shape_model}")
    if skipped_keys_missing_in_model:
         _logger.warning(f"  Skipped (in checkpoint, not in model): {len(skipped_keys_missing_in_model)} parameters.")
    if missing_keys_in_checkpoint:
        _logger.warning(f"  Missing in checkpoint (in model, not in ckpt): {len(missing_keys_in_checkpoint)} parameters (may include newly added params like prompts).")
    _logger.info("-" * 50)


def create_vision_transformer(variant_name, pretrained=False, **kwargs):
    if variant_name not in model_arch_configs:
        raise ValueError(f"Unknown ViT variant architecture: {variant_name}")

    # Priority: User kwargs > cfg > defaults
    cfg = default_cfgs.get(variant_name, {})

    model_kwargs = dict()
    model_kwargs.update(model_arch_configs[variant_name])

    model_kwargs['num_classes'] = cfg['num_classes']
    model_kwargs['img_size'] = cfg['input_size'][1]
    model_kwargs.update(kwargs)

    model = VisionTransformer(**model_kwargs)

    if pretrained:
            weights_path = f"./params/{variant_name}_jittor_weights.jt"

            _logger.info(f"Attempting to load pretrained weights from: {weights_path}")
            try:
                state_dict = jt.load(weights_path)
                _logger.info(f"Successfully loaded weights file into dictionary.")

                load_pretrained_weights(model, state_dict)

            except FileNotFoundError:
                _logger.error(f"*** Pretrained weights file not found at '{weights_path}'. "
                              f"Model '{variant_name}' initialized with random weights. ***")
            except Exception as e:
                _logger.error(f"*** Failed to load or process pretrained weights from '{weights_path}': {e}. "
                              f"Model '{variant_name}' initialized with random weights. ***")

    return model


def vit_base_patch16_224_jittor(pretrained=False, **kwargs):
    model = create_vision_transformer('vit_base_patch16_224', pretrained=pretrained, **kwargs)
    return model
