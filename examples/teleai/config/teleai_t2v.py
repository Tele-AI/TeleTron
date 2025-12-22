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
            "/nvfile-heatstorage/cjf/share/export_to_clipdataset/istock/istock_0.json",
            "/nvfile-heatstorage/cjf/share/export_to_clipdataset/istock/istock_1.json",
        ],
        filter_cfg=dict(
            dst_size=dst_size,
            dst_num_frames = dst_num_frames,
            dst_fps = dst_fps,
            multiple=32,
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
            "/nvfile-heatstorage/cjf/share/export_to_clipdataset/istock/istock_0.json",
        ],
        eval_time_steps=[200,400,600,800,1000]
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
            type="ParallelTeleaiModel", # ParallelWanModel
            config=dict(
                has_image_input=False, # t2v:False i2v:True i2v Wan2.2:False
                patch_size=[1, 2, 2],
                in_dim=48, # t2v:16 i2v:36 # 5B 48
                dim=3072, # 1.3B:1536 10B:5120 14B:5120 5B:3072
                ffn_dim=14336, # 1.3B:8960 10B:13824 14B:13824
                freq_dim=256,
                text_dim=4096,
                out_dim=48, # 5B:48
                num_heads=24, # 1.3B:12 10B:40 14B:40 5B:24
                num_layers=30, # 1.3B:30 10B:30 14B:40
                eps=1e-6,
                has_image_pos_emb=False, 
            ),
        ),
        encoder=dict(
            type="teleai_encoder", # wan_encoder
            encoder_schema=['context', 'latents'],
            vae=dict(
                type="TeleaiVideoVAE_2_1", # TeleaiVideoVAE_2_2
                # path="/nvfile-heatstorage/model_zoo/Wan2___1-I2V-14B-480P/Wan2.1_VAE.pth",
                path="/nvfile-heatstorage/model_zoo/modelscope/Wan2.2-TI2V-5B/Wan2.2_VAE.pth",
                tiler_kwargs=dict(
                    tiled=False,
                    tile_size=(34, 34),
                    tile_stride=(18, 16),
                ),
                torch_compile=False,
            ),
            text_encoder=dict(
                path="/nvfile-heatstorage/model_zoo/Wan2___1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth",
                tokenizer_path="/nvfile-heatstorage/model_zoo/Wan2___1-I2V-14B-480P/google/umt5-xxl",
            )
        ),
    ),
)
