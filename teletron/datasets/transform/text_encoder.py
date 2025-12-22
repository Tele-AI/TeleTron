import torch
from transformers import (
    LlamaModel,
    LlamaTokenizerFast,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    UMT5EncoderModel,
    T5Tokenizer,
    AutoTokenizer
)
import os
from teleai_data_tool.logger import logger


class PromptEncoder:
    def __init__(self, mode, model_path, device=None, dtype=None):
        if dtype is None:
            if device is not None and "cuda" in device:
                dtype = torch.float16
            else:
                dtype = torch.float32
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        self.mode = mode
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.tokenizers = None
        self.text_encoders = None

    def load_model(self):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 或 "true" 
        if self.text_encoders is None:
            logger.info(f"Loading tokenzier {self.mode} from {self.model_path}")
            self.tokenizers = load_tokenizers(self.mode, self.model_path)
            logger.info(f"Loading text_encoders {self.mode} from {self.model_path}")
            self.text_encoders = load_text_encoders(
                self.mode, self.model_path, self.device, self.dtype
            )

    @torch.no_grad()
    def __call__(self, prompt, max_length=None, with_attention_mask=False, padding="max_length"):

        self.load_model()
        prompt_ids, prompt_masks = forward_tokenizers(self.tokenizers, prompt, max_length, padding)
        prompt_embeds = forward_text_encoders(self.text_encoders, self.mode, prompt_ids, prompt_masks)
        if len(prompt_masks) == 1:
            prompt_masks = prompt_masks[0]
        if not with_attention_mask:
            prompt_masks = None
        return prompt_embeds, prompt_masks


class PromptTokenizer:
    def __init__(self, mode, model_path):
        self.mode = mode
        self.model_path = model_path
        self.tokenizers = None

    def load_model(self):
        if self.tokenizers is None:
            self.tokenizers = load_tokenizers(self.mode, self.model_path)

    def __call__(self, prompt, max_length=None):
        self.load_model()
        prompt_ids, prompt_masks = forward_tokenizers(
            self.tokenizers, prompt, max_length
        )
        if len(prompt_ids) == 1:
            return prompt_ids[0], prompt_masks[0]
        else:
            return prompt_ids, prompt_masks


class PromptEncoderTransform:
    def __init__(self, mode, model_path, device=None, dtype=None):
        if dtype is None:
            if device is not None and "cuda" in device:
                dtype = torch.float16
            else:
                dtype = torch.float32
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        self.mode = mode
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.text_encoders = None

    def load_model(self):
        if self.text_encoders is None:
            logger.info(f"Loading text_encoders {self.mode} from {self.model_path}")
            self.text_encoders = load_text_encoders(
                self.mode, self.model_path, self.device, self.dtype
            )

    @torch.no_grad()
    def __call__(self, prompt_ids, prompt_masks=None):
        self.load_model()
        prompt_embeds = forward_text_encoders(
            self.text_encoders, self.mode, prompt_ids, prompt_masks
        )
        return prompt_embeds


def load_tokenizers(mode, model_path):
    tokenizer_model_paths = []
    for tokenizer_name in ["tokenizer", "tokenizer_2"]:
        tokenizer_model_path = os.path.join(model_path, tokenizer_name)
        if os.path.exists(tokenizer_model_path):
            tokenizer_model_paths.append(tokenizer_model_path)
    if len(tokenizer_model_paths) == 0:
        tokenizer_model_paths = [model_path]
    logger.info(f"loading prompt encoder tokenlizer from {tokenizer_model_paths}")
    if "clip" in mode:
        tokenizers = [CLIPTokenizer.from_pretrained(_) for _ in tokenizer_model_paths]
    elif mode == "t5":
        tokenizers = [T5Tokenizer.from_pretrained(_) for _ in tokenizer_model_paths]
    elif mode == "umt5":
        tokenizers = [AutoTokenizer.from_pretrained(_) for _ in tokenizer_model_paths]
    elif mode == "llama":
        tokenizers = [
            LlamaTokenizerFast.from_pretrained(_) for _ in tokenizer_model_paths
        ]
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return tokenizers


def load_text_encoders(mode, model_path, device, dtype):
    text_encoder_model_paths = []
    for text_encoder_name in ["text_encoder", "text_encoder_2"]:
        text_encoder_model_path = os.path.join(model_path, text_encoder_name)
        if os.path.exists(text_encoder_model_path):
            text_encoder_model_paths.append(text_encoder_model_path)
    if len(text_encoder_model_paths) == 0:
        text_encoder_model_paths = [model_path]
    logger.info(f"loading prompt encoder transformer from {text_encoder_model_paths}")
    if mode == "clip_text":
        text_encoders = [
            CLIPTextModel.from_pretrained(
                text_encoder_model_paths[0], torch_dtype=dtype
            )
        ]
    elif mode == "clip_text_proj":
        text_encoders = [
            CLIPTextModelWithProjection.from_pretrained(
                text_encoder_model_paths[0], torch_dtype=dtype
            )
        ]
    elif mode == "clip_text_and_proj":
        text_encoders = [
            CLIPTextModel.from_pretrained(
                text_encoder_model_paths[0], torch_dtype=dtype
            ),
            CLIPTextModelWithProjection.from_pretrained(
                text_encoder_model_paths[1], torch_dtype=dtype
            ),
        ]
    elif mode == "t5":
        logger.info(f"loading t5 from {text_encoder_model_paths[0]}")
        text_encoders = [
            T5EncoderModel.from_pretrained(
                text_encoder_model_paths[0], torch_dtype=dtype
            )
        ]
    elif mode == "umt5":
        logger.info(f"loading UMT5 from {text_encoder_model_paths[0]}")
        text_encoders = [
            UMT5EncoderModel.from_pretrained(
                text_encoder_model_paths[0], torch_dtype=dtype
            )
        ]
    elif mode == "llama":
        text_encoders = [
            LlamaModel.from_pretrained(text_encoder_model_paths[0], torch_dtype=dtype)
        ]
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    
    for text_encoder in text_encoders:
        text_encoder.requires_grad_(False)
        if device is not None:
            text_encoder.to(device)
            logger.info(f"text_encoder={text_encoder}, device={device}")
    return text_encoders


def forward_tokenizers(tokenizers, prompt, max_length=None, padding="max_length"):
    prompt_ids = []
    prompt_masks = []
    for tokenizer in tokenizers:
        max_length_i = max_length or tokenizer.model_max_length
        text_inputs = tokenizer(
            prompt, 
            padding=padding, 
            max_length=max_length_i, 
            truncation=True, 
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask
        prompt_ids.append(input_ids)
        prompt_masks.append(attention_mask)
    return prompt_ids, prompt_masks


def forward_text_encoders(text_encoders, mode, prompt_ids, prompt_masks=None):
    device = text_encoders[0].device
    if mode in ("clip_text", "clip_text_proj", "t5", "umt5"):
        if isinstance(prompt_ids, list):
            prompt_ids = prompt_ids[0].to(device)
        else:
            prompt_ids = prompt_ids.to(device)
        if prompt_masks is not None:
            if isinstance(prompt_masks, list):
                prompt_masks = prompt_masks[0].to(device)
            else:
                prompt_masks = prompt_masks.to(device)
        with torch.no_grad():
            prompt_embeds = text_encoders[0](prompt_ids, attention_mask=prompt_masks)
        if mode in ("clip_text", "t5", "umt5"):
            prompt_embeds = prompt_embeds[0]
        else:
            prompt_embeds = prompt_embeds[0].unsqueeze(1)

    elif mode == "clip_text_and_proj":
        prompt_embeds_list = []
        for i, text_encoder in enumerate(text_encoders):
            prompt_ids_i = prompt_ids[i].to(device)
            prompt_masks_i = (
                prompt_masks[i].to(device) if prompt_masks is not None else None
            )
            with torch.no_grad():
                prompt_embeds = text_encoder(
                    prompt_ids_i,
                    attention_mask=prompt_masks_i,
                    output_hidden_states=True,
                )
            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            prompt_embeds_list.append(prompt_embeds)
        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        prompt_embeds = (prompt_embeds, pooled_prompt_embeds)
    elif mode == "llama":
        prompt_ids = prompt_ids[0].to(device)
        prompt_masks = prompt_masks[0].to(device) if prompt_masks is not None else None
        with torch.no_grad():
            prompt_embeds = text_encoders[0](prompt_ids, attention_mask=prompt_masks)
        prompt_embeds = prompt_embeds.last_hidden_state
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return prompt_embeds


def truncate_prompt(prompt_embeds, prompt_masks):
    assert len(prompt_embeds) == len(prompt_masks)
    new_prompt_embeds = []
    for i in range(len(prompt_embeds)):
        keep_index = prompt_masks[i].sum().item()
        prompt_embed = prompt_embeds[i][:keep_index]
        new_prompt_embeds.append(prompt_embed)
    return new_prompt_embeds[0] if len(new_prompt_embeds) == 1 else new_prompt_embeds


def pad_prompt(prompt_embeds, max_length, prompt_masks=None):
    cur_length = prompt_embeds.shape[0]
    assert cur_length <= max_length
    if prompt_masks is None:
        prompt_masks = torch.ones(
            (cur_length,), device=prompt_embeds.device, dtype=torch.int64
        )
    if cur_length == max_length:
        return prompt_embeds, prompt_masks
    new_shape = list(prompt_embeds.shape)
    new_shape[0] = max_length
    new_prompt_embeds = torch.zeros(
        new_shape, device=prompt_embeds.device, dtype=prompt_embeds.dtype
    )
    new_prompt_masks = torch.zeros(
        (max_length,), device=prompt_masks.device, dtype=prompt_masks.dtype
    )
    new_prompt_embeds[:cur_length] = prompt_embeds
    new_prompt_masks[:cur_length] = prompt_masks
    return new_prompt_embeds, new_prompt_masks
