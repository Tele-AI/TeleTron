from teletron.utils.prompt import clean_prompt
import random
import logging
from teleai_data_tool.schema.clip import Clip, ImageWithCaption
from teletron.train.checkpoint import get_model_path
from .text_encoder import PromptEncoder
from .clip_transform import CLIPTextTransform

logger = logging.getLogger(__name__)


class PromptGenerator:
    def __init__(
        self,
        default_prompt="",
        default_prompt_prob=0.2,
        clean_prompt=False,
        field_weights={
            'dense_caption': 1.0,
            'subject': 0.0,
            'background': 0.0,
            'style': 0.0,
            'shot_type': 0.0,
            'lighting': 0.0,
            'atmosphere': 0.0,
        },
    ) -> None:
        self.default_prompt = default_prompt
        self.default_prompt_prob = default_prompt_prob
        self.clean_prompt = clean_prompt
        self.field_weights = field_weights

    def _process_caption(self, caption, data_dict):
        """处理单个caption（列表或字符串）"""
        if isinstance(caption, list):
            if data_dict.get("slice_index") is not None:
                return caption[data_dict["slice_index"]]
            if len(caption) > 0:
                return random.choice(caption)
        return caption

    def _build_structured_prompt(self, clip_caption):
        """构建结构化prompt（MLLM输入）"""
        fields = []
        
        # 动态字段处理（带权重随机丢弃）
        for field, weight in self.field_weights.items():
            if random.random() < weight:
                field_value = getattr(clip_caption, field, None)
                if field_value:
                    processed_value = self._process_caption(field_value, {})
                    fields.append(f"{field.replace('_', ' ')}: {processed_value}")

        # 随机排列字段顺序增强鲁棒性
        random.shuffle(fields)
        
        return "; ".join(fields)

    def __call__(self, data_dict):
        # 处理默认提示逻辑
        if random.random() < self.default_prompt_prob:
            short_prompt = self.default_prompt
            dense_prompt = self.default_prompt
            struct_prompt = self.default_prompt
        else:
            clip: Clip | ImageWithCaption = data_dict["clip_info"]
            if isinstance(clip, Clip):
                short_prompt = self._process_caption(clip.caption.short_caption, data_dict)
                dense_prompt = self._process_caption(clip.caption.dense_caption, data_dict)
                struct_prompt = self._build_structured_prompt(clip.caption)
            else:
                short_prompt = self._process_caption(clip.caption_en, data_dict)
                dense_prompt = self._process_caption(clip.caption_en, data_dict)
                struct_prompt = self._process_caption(clip.caption_en, data_dict)
        # 文本清理
        if self.clean_prompt:
            short_prompt = clean_prompt(clean_prompt(short_prompt))
            dense_prompt = clean_prompt(clean_prompt(dense_prompt))
            struct_prompt = clean_prompt(clean_prompt(struct_prompt))

        # 存储到不同字段
        data_dict["short_prompt"] = short_prompt
        data_dict["dense_prompt"] = dense_prompt
        data_dict["struct_prompt"] = struct_prompt
        return data_dict


class PromptToClipEmbedding:
    def __init__(self, model_path, dtype=None) -> None:
        self.clip_transform = CLIPTextTransform(
            get_model_path(model_path), dtype=dtype
        )

    def __call__(self, data_dict):
        if data_dict["short_prompt"] != '':  # 判断short caption是否存在
            prompt = data_dict["short_prompt"]
        else:
            prompt = data_dict["dense_prompt"]
        clip_text_embed = self.clip_transform(
            prompt, mode="after_pool", to_numpy=False
        )[0]

        data_dict["clip_text_embed"] = clip_text_embed
        return data_dict


class PromptToTransformerEmbedding:
    """
    使用struct prompt生成transformer嵌入
    """

    def __init__(
        self,
        model_name,
        model_path,
        max_length=None,
        with_attention_mask=False,
        padding="max_length",
    ):
        self.prompt_encoder = PromptEncoder(
            model_name, get_model_path(model_path)#, device="cpu",
        )
        self.max_length = max_length
        self.with_attention_mask = with_attention_mask
        self.padding = padding

    def __call__(self, data_dict):
        prompt = data_dict["struct_prompt"]  # 使用struct prompt
        prompt_embeds, prompt_masks = self.prompt_encoder(
            prompt,
            max_length=self.max_length,
            with_attention_mask=self.with_attention_mask,
            padding=self.padding,
        )
        data_dict["prompt_embeds"] = prompt_embeds[0]
        if prompt_masks is not None:
            data_dict["prompt_masks"] = prompt_masks[0]
        return data_dict
