# Copyright (c) 2025 TeleAI-infra Team. All rights reserved.

import os
import hashlib
import safetensors
from contextlib import contextmanager

import torch
import numpy as np
from einops import rearrange

from torchvision.transforms.functional import to_pil_image


def encode_prompt(prompter,prompt, positive=True):
    prompt_emb = prompter.encode_prompt(
        prompt, positive=positive, device=torch.cuda.current_device()
    )
    return {"context": prompt_emb}

def encode_image(
    vae,
    image_encoder,
    image,
    num_frames,
    height,
    width,
    tiled=False,
    tile_size=(34, 34),
    tile_stride=(18, 16),
):
    image = preprocess_image(image.resize((width, height))).to(torch.cuda.current_device())
    clip_context = image_encoder.encode_image([image])
    msk = torch.ones(1, num_frames, height // 8, width // 8, device=torch.cuda.current_device())
    
    msk[:, 1:] = 0 # 1, 1:81, 56, 98
    msk = torch.concat(
        [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1
    ) # 1, 4, 56, 98; # 1, 80, 56, 98 => 1, 84, 56, 98
    
    msk = msk.view(1, msk.shape[1] // 4, 4, height // 8, width // 8) # 1, 21, 4, 56, 98
    msk = msk.transpose(1, 2)[0]
    vae_input = torch.concat(
        [image.transpose(0, 1), torch.zeros((3, num_frames - 1, height, width), device=image.device)],
        dim=1,
    )
    y = vae.encode(
        [vae_input.to(dtype=torch.bfloat16, device=torch.cuda.current_device())],
        device=torch.cuda.current_device(),
        tiled=tiled,
        tile_size=tile_size,
        tile_stride=tile_stride,
    )[0]
    y = y.to(dtype=torch.bfloat16, device=torch.cuda.current_device())
    y = torch.concat([msk, y])
    y = y.unsqueeze(0)
    clip_context = clip_context.to(dtype=torch.bfloat16, device=torch.cuda.current_device())
    y = y.to(dtype=torch.bfloat16, device=torch.cuda.current_device())
    return {"clip_feature": clip_context, "y": y}

def encode_image_with_mask(
        vae,
        image_encoder, 
        image, 
        num_frames,
        height, 
        width, 
        msk, 
        ref_images, 
        tiled=False, 
        tile_size=(34, 34), 
        tile_stride=(18, 16)
    ):
    image = preprocess_image(image.resize((width, height))).to(torch.cuda.current_device())
    clip_context = image_encoder.encode_image([image])
    ref_images = rearrange(ref_images, 'b t c h w -> b c t h w')
    y = encode_video(
        vae, 
        ref_images.to(dtype=torch.bfloat16, device=torch.cuda.current_device()),
        tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
    y = y.unsqueeze(0)
    y = y.to(dtype=torch.bfloat16, device=torch.cuda.current_device())
    msk = msk.transpose(1, 2).to(torch.cuda.current_device())
    y = torch.concat([msk, y], dim=1)
    clip_context = clip_context.to(dtype=torch.bfloat16, device=torch.cuda.current_device())
    y = y.to(dtype=torch.bfloat16, device=torch.cuda.current_device())
    return {"clip_feature": clip_context, "y": y}


def encode_first_last_image(
    vae,
    image_encoder,
    pil_first_image,
    pil_last_image,
    num_frames,
    height,
    width,
    tiled=False,
    tile_size=(34, 34),
    tile_stride=(18, 16),
):
    first_image = preprocess_image(pil_first_image.resize((width, height))).to(
        torch.cuda.current_device()
    )
    last_image = preprocess_image(pil_last_image.resize((width, height))).to(
        torch.cuda.current_device()
    )

    clip_context = torch.cat(
        [
            image_encoder.encode_image([first_image]),
            image_encoder.encode_image([last_image]),
        ],
        dim=1,
    )
    msk = torch.ones(1, num_frames, height // 8, width // 8, device=torch.cuda.current_device())
    msk[:, 1:-1] = 0
    msk = torch.concat(
        [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1
    )
    msk = msk.view(1, msk.shape[1] // 4, 4, height // 8, width // 8)
    msk = msk.transpose(1, 2)[0]
    vae_input = torch.concat(
        [
            first_image.transpose(0, 1),
            torch.zeros(3, num_frames - 2, height, width).to(first_image.device),
            last_image.transpose(0, 1),
        ],
        dim=1,
    )
    y = vae.encode(
        [vae_input.to(dtype=torch.bfloat16, device=torch.cuda.current_device())],
        device=torch.cuda.current_device(),
        tiled=tiled,
        tile_size=tile_size,
        tile_stride=tile_stride,
    )[0]
    y = y.to(dtype=torch.bfloat16, device=torch.cuda.current_device())
    y = torch.concat([msk, y])
    y = y.unsqueeze(0)
    clip_context = clip_context.to(dtype=torch.bfloat16, device=torch.cuda.current_device())
    y = y.to(dtype=torch.bfloat16, device=torch.cuda.current_device())
    return {"clip_feature": clip_context, "y": y}

    
def encode_video(vae, input_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
    latents = vae.encode(
        input_video,
        device=torch.cuda.current_device(),
        tiled=tiled,
        tile_size=tile_size,
        tile_stride=tile_stride,
    )
    return latents



def preprocess_image(image):
    image = (
        torch.Tensor(np.array(image, dtype=np.float32) * (2 / 255) - 1)
        .permute(2, 0, 1)
        .unsqueeze(0)
    )
    return image


def get_encoder_features(batch, prompter, vae, tiler_kwargs, image_encoder):
    dtype_wan = torch.bfloat16
    with torch.no_grad():
        prompt_emb = encode_prompt(prompter,batch["dense_prompt"][0])
        latents = encode_video(vae,
            rearrange(batch["images"], "b t c h w -> b c t h w").to(
                dtype=dtype_wan, device=torch.cuda.current_device()
            ),
            **tiler_kwargs,
        )[0]
        _, num_frames, _, height, width = batch["images"].shape
        # print("images: ",height, width )
        if 'raw_last_image' in batch:
            raw_first_image = batch["raw_first_image"]
            pil_first_image = to_pil_image(
                raw_first_image[0][0].cpu().permute(1, 2, 0).numpy().astype(np.uint8)
            )
            raw_last_image = batch['raw_last_image']
            pil_last_image = to_pil_image(
                raw_last_image[0][0].cpu().permute(1, 2, 0).numpy().astype(np.uint8)
            )
            image_emb = encode_first_last_image(
                vae,image_encoder, pil_first_image, pil_last_image, num_frames, height, width
            )
        elif 'raw_first_image' in batch:
            raw_first_image = batch["raw_first_image"]
            pil_image = to_pil_image(
                raw_first_image[0][0].cpu().permute(1, 2, 0).numpy().astype(np.uint8)
            )
            image_emb = encode_image(vae, image_encoder, pil_image, num_frames, height, width)
        elif 'ref_images' in batch:
            first_image = (batch['ref_images'] + 1) / 2 * 255
            ref_mask = batch["ref_mask"]
            ref_images = batch["ref_images"]
            pil_image = to_pil_image(first_image[0][0].cpu().permute(1,2,0).numpy().astype(np.uint8))
            image_emb = encode_image_with_mask(vae, image_encoder, pil_image, num_frames,
                                                height, width, ref_mask, ref_images)
        
        
        latents = latents.unsqueeze(0).to(dtype=dtype_wan, device=torch.cuda.current_device())

        # Data
        prompt_emb["context"] = prompt_emb["context"][0].to(
            dtype=dtype_wan, device=torch.cuda.current_device()
        )
        prompt_emb["context"] = prompt_emb["context"].unsqueeze(0)

        if "clip_feature" in image_emb:
            image_emb["clip_feature"] = (
                image_emb["clip_feature"][0]
                .to(dtype=dtype_wan, device=torch.cuda.current_device())
                .unsqueeze(0)
            )
        if "y" in image_emb:
            image_emb["y"] = (
                image_emb["y"][0]
                .to(dtype=dtype_wan, device=torch.cuda.current_device())
                .unsqueeze(0)
            )
    return prompt_emb, image_emb, latents




@contextmanager
def init_weights_on_device(device = torch.device("meta"), include_buffers :bool = False):
    
    old_register_parameter = torch.nn.Module.register_parameter
    if include_buffers:
        old_register_buffer = torch.nn.Module.register_buffer
    
    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)

    def register_empty_buffer(module, name, buffer, persistent=True):
        old_register_buffer(module, name, buffer, persistent=persistent)
        if buffer is not None:
            module._buffers[name] = module._buffers[name].to(device)
            
    def patch_tensor_constructor(fn):
        def wrapper(*args, **kwargs):
            kwargs["device"] = device
            return fn(*args, **kwargs)

        return wrapper
    
    if include_buffers:
        tensor_constructors_to_patch = {
            torch_function_name: getattr(torch, torch_function_name)
            for torch_function_name in ["empty", "zeros", "ones", "full"]
        }
    else:
        tensor_constructors_to_patch = {}
    
    try:
        torch.nn.Module.register_parameter = register_empty_parameter
        if include_buffers:
            torch.nn.Module.register_buffer = register_empty_buffer
        for torch_function_name in tensor_constructors_to_patch.keys():
            setattr(torch, torch_function_name, patch_tensor_constructor(getattr(torch, torch_function_name)))
        yield
    finally:
        torch.nn.Module.register_parameter = old_register_parameter
        if include_buffers:
            torch.nn.Module.register_buffer = old_register_buffer
        for torch_function_name, old_torch_function in tensor_constructors_to_patch.items():
            setattr(torch, torch_function_name, old_torch_function)


def wrap_call(func):
    def f(*args, **kwargs):
        while True:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(e)
                continue

    return f


def get_device():
    if torch.cuda.is_available():
        device = torch.device(
            "cuda:{}".format(int(os.environ.get("VAST_MODELS_DEFAULT_DEVICE", "0")))
        )
    else:
        device = torch.device("cpu")
    return device

def save_state_dict(state_dict, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if save_path.endswith(".safetensors"):
        safetensors.torch.save_file(state_dict, save_path)
    else:
        torch.save(state_dict, save_path)


def convert_state_dict_keys_to_single_str(state_dict, with_shape=True):
    keys = []
    for key, value in state_dict.items():
        if isinstance(key, str):
            if isinstance(value, torch.Tensor):
                if with_shape:
                    shape = "_".join(map(str, list(value.shape)))
                    keys.append(key + ":" + shape)
                keys.append(key)
            elif isinstance(value, dict):
                keys.append(key + "|" + convert_state_dict_keys_to_single_str(value, with_shape=with_shape))
    keys.sort()
    keys_str = ",".join(keys)
    return keys_str


def split_state_dict_with_prefix(state_dict):
    keys = sorted([key for key in state_dict if isinstance(key, str)])
    prefix_dict = {}
    for key in  keys:
        prefix = key if "." not in key else key.split(".")[0]
        if prefix not in prefix_dict:
            prefix_dict[prefix] = []
        prefix_dict[prefix].append(key)
    state_dicts = []
    for prefix, keys in prefix_dict.items():
        sub_state_dict = {key: state_dict[key] for key in keys}
        state_dicts.append(sub_state_dict)
    return state_dicts


def hash_state_dict_keys(state_dict, with_shape=True):
    keys_str = convert_state_dict_keys_to_single_str(state_dict, with_shape=with_shape)
    keys_str = keys_str.encode(encoding="UTF-8")
    # breakpoint()
    return hashlib.md5(keys_str).hexdigest()
