import os
import math

dst_size = (480, 480) # Adjusted from 832 to be divisible by 48 (CP=3 * Patch=2 * Downsample=8)
_env_dst_size = os.environ.get("LONGCAT_DST_SIZE")
if _env_dst_size:
    _parts = [p.strip() for p in _env_dst_size.split(",") if p.strip() != ""]
    if len(_parts) == 2:
        try:
            _h = int(_parts[0])
            _w = int(_parts[1])
        except ValueError:
            _h, _w = dst_size
        if _h > 0 and _w > 0:
            dst_size = (_h, _w)
dst_fps = 16
dst_num_frames = 33

# Get debug params and derive safe aligned params at top-level
depth = int(os.environ.get("DEBUG_DEPTH", 48))
raw_hidden = int(os.environ.get("DEBUG_HIDDEN_SIZE", 4096))
raw_heads = int(os.environ.get("DEBUG_NUM_HEADS", 32))

# CP & patch & VAE spatial settings
PATCH_SIZE = (1, 2, 2)
def _parse_cp_split_hw(s: str):
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    if len(parts) != 2:
        return None
    try:
        h = int(parts[0])
        w = int(parts[1])
    except ValueError:
        return None
    if h <= 0 or w <= 0:
        return None
    return (h, w)

_env_cp_split_hw = os.environ.get("LONGCAT_CP_SPLIT_HW")
_parsed_cp_split_hw = _parse_cp_split_hw(_env_cp_split_hw) if _env_cp_split_hw else None
CP_SPLIT_HW = _parsed_cp_split_hw if _parsed_cp_split_hw is not None else (3, 1)
SPATIAL_DOWNSAMPLE = 8
def _lcm(a, b):
    return abs(a * b) // math.gcd(a, b)

# ensure dataset spatial buckets match CP H/W split
_MULTIPLE_H = CP_SPLIT_HW[0] * PATCH_SIZE[1] * SPATIAL_DOWNSAMPLE
_MULTIPLE_W = CP_SPLIT_HW[1] * PATCH_SIZE[2] * SPATIAL_DOWNSAMPLE
MULTIPLE_ALIGN = _lcm(_MULTIPLE_H, _MULTIPLE_W)

# align num_heads to CP group size for ulysses head all-to-all
CP_SIZE = CP_SPLIT_HW[0] * CP_SPLIT_HW[1]
NUM_HEADS_ALIGN = raw_heads if (raw_heads % CP_SIZE == 0) else ((raw_heads // CP_SIZE) + 1) * CP_SIZE
# align hidden_size to satisfy both RoPE (×8) and attention dim % heads == 0
DIM_UNIT = _lcm(8, NUM_HEADS_ALIGN)
HIDDEN_ALIGN = ((raw_hidden + DIM_UNIT - 1) // DIM_UNIT) * DIM_UNIT

config = dict(
    dataset=dict(
        type="ClipDataset",
        serialize_data=False,
        data_path_list=[
            "/nvfile-heatstorage/AIGC_H100/basemodel_exp/dataset/istock/istock_0.json",
        ],
        filter_cfg=dict(
            dst_size=dst_size,
            dst_num_frames=dst_num_frames,
            dst_fps=dst_fps,
            multiple=MULTIPLE_ALIGN,
            min_area=dst_size[0] * dst_size[1],
            optical_flow_th=1.5,
            aesthetic_th=5.0,
            bucket_size_th=4,
            motion_th=0,
            clearity_th=0.9,
            laplacian_th=30,
            training_suitability_th=5.0,
            area_th=dst_size[0] * dst_size[1],
        ),
        transforms=[
            dict(
                type="SampleImages",
                num_frames=dst_num_frames,
            ),
            dict(
                type="PromptGenerator",
                clean_prompt=True,
                default_prompt_prob=0.1,
            ),
            dict(
                type="PackInputs",
                deterministic=True,
                image_keys=[
                    "images",
                ],
            ),
        ],
    ),
    eval=dict(
        data_path_list=[
             "/nvfile-heatstorage/AIGC_H100/basemodel_exp/dataset/istock/istock_0.json",
        ],
    ),
    sampler=dict(
        type="DefaultSampler",
        shuffle=False,
        seed=42,
        drop_last=True,
        infinite=True,
    ),
    train=dict(
        resume=True,
        checkpoint_save_optimizer=True,
        max_epochs=10,
        gradient_accumulation_steps=1,
        mixed_precision="fp16",  # fp16, bf16
        checkpoint_interval=100,
        checkpoint_total_limit=-1,
        log_with="tensorboard",
        log_interval=1,
        with_ema=False,
        activation_checkpointing=True,
        activation_class_names=[
            "LongCatSingleStreamBlock",
        ],
    ),
    test=dict(
        no_load_weights=True,
        single_gpu=True,
        model_config=dict(
            dit=dict(
                config=dict(
                    depth=1,
                    hidden_size=HIDDEN_ALIGN,
                    num_heads=NUM_HEADS_ALIGN,
                    cp_split_hw=CP_SPLIT_HW,
                )
            )
        )
    ),
    model_config=dict(
        dit=dict(
            type="ParallelLongCatModel", # ParallelTeleaiModel
            config=dict(
                patch_size=list(PATCH_SIZE),
                in_channels=16,
                out_channels=16,
                hidden_size=HIDDEN_ALIGN,
                depth=depth,
                num_heads=NUM_HEADS_ALIGN,
                num_layers=depth,
                cp_split_hw=list(CP_SPLIT_HW),
                dim=HIDDEN_ALIGN,
                ffn_dim=HIDDEN_ALIGN * 4,
                mlp_ratio=4,
                adaln_tembed_dim=512,
                frequency_embedding_size=256,
                enable_flashattn3=False,
                enable_flashattn2=True,
                enable_xformers=False,
                enable_bsa=False,
                bsa_params=None,
                text_tokens_zero_pad=False,
            ),
        ),
        encoder=dict(
            type="longcat_encoder", # teleai_encoder
            encoder_schema=['context', 'latents'],
            vae=dict(
                path="/nvfile-heatstorage/model_zoo/Wan2___1-I2V-14B-480P/Wan2.1_VAE.pth",
                tiler_kwargs=dict(
                    tiled=False,
                    tile_size=(34, 34),
                    tile_stride=(18, 16),
                ),
            ),
            text_encoder=dict(
                path="/nvfile-heatstorage/model_zoo/Wan2___1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth",
                tokenizer_path="/nvfile-heatstorage/model_zoo/Wan2___1-I2V-14B-480P/google/umt5-xxl",
            ),
        ),
    ),
)
