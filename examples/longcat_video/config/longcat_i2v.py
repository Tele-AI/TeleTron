import os

dst_size = (832, 480)
dst_fps = 16
dst_num_frames = 81

# Get debug params from env
depth = int(os.environ.get("DEBUG_DEPTH", 48))
hidden_size = int(os.environ.get("DEBUG_HIDDEN_SIZE", 4096))
num_heads = int(os.environ.get("DEBUG_NUM_HEADS", 32))

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
            multiple=32,
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
            dict(
                type="GenerateRefImagesWithMask",
                mask_cfg={
                    "t2v": 0.0,
                    "i2v": 0.4,
                    "clear": 0.0,
                    "continuation": 0.2,
                    "random": 0.0,
                    "transition": 0.4
                },
                min_clear_ratio=0.0,
                max_clear_ratio=1.0,
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
        mixed_precision="bf16",  # fp16, bf16
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
                    num_layers=1,
                    dim=128,
                    num_heads=4,
                    cp_split_hw=[1,1],
                )
            )
        )
    ),
    model_config=dict(
        dit=dict(
            type="ParallelLongCatModel",
            config=dict(
                patch_size=[1, 2, 2],
                in_channels=16,
                out_channels=16,
                hidden_size=int(os.getenv("DEBUG_HIDDEN_SIZE", 2048)),
                depth=int(os.getenv("DEBUG_DEPTH", 25)),
                num_heads=int(os.getenv("DEBUG_NUM_HEADS", 16)),
                num_layers=int(os.getenv("DEBUG_DEPTH", 25)),
                dim=int(os.getenv("DEBUG_HIDDEN_SIZE", 2048)),
                ffn_dim=int(os.getenv("DEBUG_HIDDEN_SIZE", 2048)) * 4,
                mlp_ratio=4,
                adaln_tembed_dim=512,
                frequency_embedding_size=256,
                enable_flashattn3=False,
                enable_flashattn2=True,
                enable_xformers=False,
                enable_bsa=False,
                bsa_params=None,
                cp_split_hw=[1, 1],
                text_tokens_zero_pad=False,
                has_image_input=True,
                has_image_pos_emb=True,
            ),
        ),
        encoder=dict(
            type="longcat_encoder",
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
        ),
    ),
)
