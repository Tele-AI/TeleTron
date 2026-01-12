# Copyright (c) 2025 TeleAI-infra Team. All rights reserved.

import os
import torch
from typing import Dict, Any, Tuple, List

from teletron.core.distributed.base_encoder import BaseEncoder
from .wan_prompter import WanPrompter
from .wan_video_vae import WanVideoVAE
from .wan_video_text_encoder import WanTextEncoder
from .wan_video_image_encoder import WanImageEncoder

from .wan_encoder_utils import get_encoder_features
from teletron.utils import get_args, set_config

def get_encoder_model_paths(path):
    filenames = [
        "models_t5_umt5-xxl-enc-bf16.pth",
        "Wan2.1_VAE.pth",
        "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
    ]
    return [os.path.join(path, f) for f in filenames]

class WanVideoEncoder(BaseEncoder):
    """WAN视频模型的具体编码器实现。"""
    
    _OUTPUT_MOE_SCHEMA = ['context', 'img_clip_feature', 'img_emb_y', 'latents', 'noise']
    _OUTPUT_SCHEMA = ['context', 'img_clip_feature', 'img_emb_y', 'latents']

    @staticmethod
    def get_output_schema() -> List[str]:
        """返回此编码器输出张量的固定名称和顺序。"""
        args = get_args()
        is_moe = (args.consumer_models_num > 1)
        if is_moe is True:
            return WanVideoEncoder._OUTPUT_MOE_SCHEMA
        return WanVideoEncoder._OUTPUT_SCHEMA

    def __init__(self, device: torch.device):
        super().__init__(device)

        encoder_model_config = set_config().get("model_config", None).get("encoder", None)
        if encoder_model_config is None:
            raise ValueError("未找到encoder模型配置。")

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
            raise ValueError("TeleaiEncoder需要 'text_encoder_path' 和 'tokenizer_path' 参数。")

        # 将模型组件初始化为None，它们将在setup()中被加载
        self.text_encoder = None
        self.image_encoder = None
        self.vae = None
        self.prompter = None

    def setup(self) -> None:
        """加载所有必需的WAN模型组件到指定设备。"""
        print(f"在设备 {self.device} 上设置 WanVideoEncoder...")
        
        print(f"加载 VAE 模型... {self.vae_path}")
        self.vae = WanVideoVAE().to(device=self.device, dtype=torch.bfloat16).eval().requires_grad_(False)
        self.vae.model.load_state_dict(torch.load(self.vae_path, map_location='cpu'), strict=True)

        print(f"加载 Text Encoder 模型... {self.text_encoder_path}")
        self.text_encoder = WanTextEncoder().to(device=self.device, dtype=torch.bfloat16)
        self.text_encoder.load_state_dict(torch.load(self.text_encoder_path, map_location='cpu', weights_only=False), strict=True)
        self.prompter = WanPrompter()
        self.prompter.fetch_models(self.text_encoder)
        self.prompter.fetch_tokenizer(self.tokenizer_path)


        if self.image_encoder_path is not None:
            print(f"加载 Image Encoder 模型... {self.image_encoder_path}")
            self.image_encoder = WanImageEncoder().to(device=self.device, dtype=torch.bfloat16).eval().requires_grad_(False)
            self.image_encoder.model.load_state_dict(torch.load(self.image_encoder_path, map_location='cpu', weights_only=False), strict=False)

        print("WanVideoEncoder 设置完成。")


    def encode(self, raw_batch: Dict[str, Any]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        使用WAN模型对数据批次进行编码。
        """
        batch = dict(raw_batch)

        prompt_emb, image_emb, latents = get_encoder_features(
            batch, self.prompter, self.vae, self.tiler_kwargs, self.image_encoder
        )
        
        
        context = prompt_emb['context']
        img_clip_feature = image_emb["clip_feature"]
        img_emb_y = image_emb["y"]

        if self.moe is True:
            noise = torch.randn_like(latents, device=self.device)
            tensors_to_send = [context, img_clip_feature, img_emb_y, latents, noise]
        else:
            tensors_to_send = [context, img_clip_feature, img_emb_y, latents]

        size_info_tensor = self._get_tensors_size(tensors_to_send, device=self.device)

        return tensors_to_send, size_info_tensor