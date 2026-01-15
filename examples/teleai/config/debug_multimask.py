import os
import math

dst_size = (832, 480)
dst_fps = 16
dst_num_frames = 81

config = dict(
    dataset=dict(
        type="ClipDataset",
        serialize_data=False,
        data_path_list=[
             "/nvfile-heatstorage/AIGC_H100/basemodel_exp/dataset/istock/istock_0.json",
        ],
        filter_cfg=dict(
            dst_size=dst_size,
            dst_num_frames = dst_num_frames,
            dst_fps = dst_fps,
            multiple=16,
            min_area=dst_size[0] * dst_size[1] * (2 if dst_size[1] == 480 else 1),
            optical_flow_th=1.5,
            aesthetic_th=5,
            bucket_size_th=4,
            motion_th=0,
            clearity_th=0.9,
            laplacian_th=30,
            training_suitability_th=5.0,
            area_th=dst_size[0] * dst_size[1] * (2 if dst_size[1] == 480 else 1),
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
                type="GenerateRawFirstRefImage",
            ),
            dict(
                type="GenerateRefImagesWithMask",
                mask_cfg={
                    "i2v": 0.75,
                    # "transition": 0.5, # first and last frame
                    "f1fn2v": 0.25, # first frame and random frames
                },
                min_clear_ratio=0,
                max_clear_ratio=0.25,
            ),
            dict(
                type="PackInputs",
                deterministic=True,
                image_keys=[
                    "images",
                ],
                embedding_keys=[
                    "raw_first_image",
                    "ref_images",
                    "ref_mask"
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
        shuffle=True,
        seed=42,
        drop_last=True,
        infinite=True,
    ),
    model_config=dict(
        dit=dict(
            type="ParallelTeleaiModel", # ParallelTeleaiModel
            config=dict(
                has_image_input=True, # t2v:False i2v:True i2v Wan2.2:False
                patch_size=[1, 2, 2],
                in_dim=36, # t2v:16 i2v:36
                dim=1536, # 1.3B:1536 10B:5120 14B:5120
                ffn_dim=8960, # 1.3B:8960 10B:13824 14B:13824
                freq_dim=256,
                text_dim=4096,
                out_dim=16,
                num_heads=12, # 1.3B:12 10B:40 14B:40
                num_layers=30, # 1.3B:30 10B:30 14B:40
                eps=1e-6,
                has_image_pos_emb=False, 
            ),
        ),
        encoder=dict(
            type="teleai_encoder", # teleai_encoder
            encoder_schema=['context', 'img_clip_feature', 'img_emb_y', 'latents'],
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
            image_encoder=dict(
                path="/nvfile-heatstorage/model_zoo/Wan2___1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            ),
            depth_model=dict(
                path="/nvfile-heatstorage/ai_infra/ckpts/lit117/qiuyang/video_depth_anything_vitl.pth",
            ),
        ),
    ),
)
