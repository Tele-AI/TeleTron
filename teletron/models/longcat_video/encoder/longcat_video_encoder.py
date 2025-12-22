# Copyright (c) 2025 TeleAI-infra Team. All rights reserved.

import torch
from typing import Dict, Any, List, Union

from teletron.core.distributed.base_encoder import BaseEncoder
from teletron.models.wan.encoder.wan_prompter import WanPrompter
from teletron.models.wan.encoder.wan_video_vae import WanVideoVAE
from teletron.models.wan.encoder.wan_video_text_encoder import WanTextEncoder
from teletron.models.wan.encoder.wan_video_image_encoder import WanImageEncoder

from teletron.models.teleai.teleai_encoder_utils import (
    get_context,
    get_img_clip_feature,
    get_img_emb_y,
    get_latents,
)
from teletron.utils import get_args, set_config

class LongcatVideoEncoder(BaseEncoder):
    """LongcatVideo models concrete encoder implementation."""
    
    _OUTPUT_MOE_SCHEMA = ['context', 'img_clip_feature', 'img_emb_y', 'latents', 'noise']
    _OUTPUT_SCHEMA = ['context', 'img_clip_feature', 'img_emb_y', 'latents']

    @staticmethod
    def get_output_schema() -> List[str]:
        """Returns the fixed names and order of the output tensors for this encoder."""
        cfg_encoder = set_config().get("model_config", None).get("encoder", None)
        if cfg_encoder and cfg_encoder.get("encoder_schema", None):
            return cfg_encoder.get("encoder_schema")
        args = get_args()
        is_moe = (args.consumer_models_num > 1)
        if is_moe is True:
            return LongcatVideoEncoder._OUTPUT_MOE_SCHEMA
        return LongcatVideoEncoder._OUTPUT_SCHEMA

    def __init__(self, device: torch.device):
        super().__init__(device)

        encoder_model_config = set_config().get("model_config", None).get("encoder", None)
        if encoder_model_config is None:
            raise ValueError("Encoder model config not found.")

        self.vae_path = encoder_model_config.get("vae", None).get("path", None)
        self.tiler_kwargs = encoder_model_config.get("vae", None).get("tiler_kwargs", {})
        if self.tiler_kwargs is None:
            self.tiler_kwargs = dict(
                tiled=False,
                tile_size=(34, 34),
                tile_stride=(18, 16),
            )
        self.text_encoder_path = encoder_model_config.get("text_encoder", None).get("path", None)
        self.tokenizer_path = encoder_model_config.get("text_encoder", None).get("tokenizer_path", None)

        if encoder_model_config.get("image_encoder", None) is not None:
            self.image_encoder_path = encoder_model_config.get("image_encoder", None).get("path", None)
        else:
            self.image_encoder_path = None

        if not self.vae_path or not self.text_encoder_path or not self.tokenizer_path:
            raise ValueError("LongcatVideoEncoder requires 'text_encoder_path' and 'tokenizer_path' parameters.")

        # Initialize model components to None, they will be loaded in setup()
        self.text_encoder = None
        self.image_encoder = None
        self.vae = None
        self.prompter = None
        self.work_fn = {
            'context': get_context,
            'img_clip_feature': get_img_clip_feature,
            'img_emb_y': get_img_emb_y,
            'latents': get_latents,
        }
        self.compression = (4, 8, 8)

    def setup(self) -> None:
        """Load all required LongcatVideo model components to the specified device."""
        print(f"Setting up LongcatVideoEncoder on device {self.device}...")
        
        print(f"Loading VAE model... {self.vae_path}")
        self.vae = WanVideoVAE().to(device=self.device, dtype=torch.bfloat16).eval().requires_grad_(False)
        self.vae.model.load_state_dict(torch.load(self.vae_path, map_location='cpu'), strict=True)

        print(f"Loading Text Encoder model... {self.text_encoder_path}")
        self.text_encoder = WanTextEncoder().to(device=self.device, dtype=torch.bfloat16)
        self.text_encoder.load_state_dict(torch.load(self.text_encoder_path, map_location='cpu', weights_only=False), strict=True)
        self.prompter = WanPrompter()
        self.prompter.fetch_models(self.text_encoder)
        self.prompter.fetch_tokenizer(self.tokenizer_path)


        if self.image_encoder_path is not None:
            print(f"Loading Image Encoder model... {self.image_encoder_path}")
            self.image_encoder = WanImageEncoder().to(device=self.device, dtype=torch.bfloat16).eval().requires_grad_(False)
            self.image_encoder.model.load_state_dict(torch.load(self.image_encoder_path, map_location='cpu', weights_only=False), strict=False)

        print("LongcatVideoEncoder setup complete.")

        for key, val in self.work_fn.items():
            self.work_fn[key] = self.prepare_work_fn(key, val)

    def prepare_work_fn(self, target, work_fn):
        if target == 'context':
            return lambda batch: work_fn(batch=batch, prompter=self.prompter, dtype=torch.bfloat16)
        elif target == 'img_clip_feature':
            return lambda batch: work_fn(batch=batch, image_encoder=self.image_encoder, dtype=torch.bfloat16)
        elif target == 'img_emb_y':
            return lambda batch: work_fn(batch=batch, vae=self.vae, dtype=torch.bfloat16, compression=self.compression, tiler_kwargs=self.tiler_kwargs)
        elif target == 'latents':
            return lambda batch: work_fn(batch=batch, vae=self.vae, dtype=torch.bfloat16, tiler_kwargs=self.tiler_kwargs)
        else:
            return work_fn

    def encode(self, raw_batch: Union[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Encode data batch using LongcatVideo model.
        """
        batch = dict(raw_batch)
        schema = self.get_output_schema()
        outputs_map: Dict[str, torch.Tensor] = {}
        cache_latents = None
        for data_to_produce in schema:
            if data_to_produce == 'noise':
                if cache_latents is None:
                    cache_latents = self.work_fn['latents'](batch)
                outputs_map['noise'] = torch.randn_like(cache_latents, device=self.device, dtype=torch.bfloat16)
            else:
                tensor = self.work_fn[data_to_produce](batch)
                if data_to_produce == 'latents':
                    cache_latents = tensor
                outputs_map[data_to_produce] = tensor

        return outputs_map
