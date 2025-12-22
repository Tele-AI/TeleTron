import os

# dst_size = (720, 480)
dst_fps = 24
dst_num_frames = 81
# Temporary code for quick debugging
debug = False # open
if debug:
    GPU_IDS = [0]
    NUM_WORKERS = 1
    import logging

    logging.basicConfig(level=logging.DEBUG)
else:
    GPU_IDS = [0, 1, 2, 3, 4, 5, 6, 7]
    NUM_WORKERS = 1


config = dict(
    runners=["projects.wan.adaptors.WanI2VTrainer"],
    launch=dict(
        gpu_ids=GPU_IDS,
        distributed_type="DEEPSPEED",
        deepspeed_config=dict(
            deepspeed_config_file=os.path.join(
                os.getcwd(), "configs/accelerate_configs/zero2.json"
            ),
        ),
        num_machines=os.environ.get("WORLD_SIZE", 1),
        until_completion=True,
    ),

    dataloaders=dict(
        train=dict(
            dataset=dict(
                type="ClipDataset",
                data_path_list=[
                    "/nvfile-heatstorage/Text2Video/annotations/200w/pack_zwzx_1.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/200w/pack_zwzx_2.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/200w/pack_zwzx_3.json",

                    # # "/nvfile-heatstorage/Text2Video/annotations/200w_nobody/pack_zwzx_1_slice_new_0.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/150w/pexels_v0.0.8.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/150w/mixkit_v0.0.7.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/150w/pixapay_v0.0.7.json",
                ],

                filter_cfg=dict(
                    dst_num_frames=dst_num_frames,
                    dst_fps=dst_fps,
                    multiple=16,
                    optical_flow_th=3,
                    aesthetic_th=3.5,
                    bucket_size_th=4,
                    motion_th=0,
                    clearity_th=0.9,
                    laplacian_th=200,
                    training_suitability_th=4.4,
                    area_th=1280 * 720,
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
                        type="GenerateRawFirstLastRefImage",
                    ),
                    dict(
                        type="PackInputs",
                        image_keys=[
                            "images",
                        ],
                        embedding_keys=[
                            "raw_first_image", 
                            "raw_last_image"
                        ],  
                    ),
                ],
            ),
            batch_size_per_gpu=1,
            num_workers=NUM_WORKERS,
            sampler=dict(
                type="DefaultSampler",
            ),
            collator=dict(
                is_equal=True,
            ),
        ),
    ),
    models=dict(
        text_encoder_path="/workspace/Wan2___1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth",
        vae_path="/workspace/Wan2___1-I2V-14B-480P/Wan2.1_VAE.pth",
        image_encoder_path="/workspace/Wan2___1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        dit_path="/workspace/Wan2___1-FLF2V-14B-480P-init", 
        tiled=True, 
        tile_size=(34, 34),
        tile_stride=(18, 16), 
    ),
    ### 优化器optimizer配置
    optimizers=dict(
        type="AdamW",
        lr=1e-6,
    ),
    ### 学习率scheduler配置
    schedulers=dict(
        type="ConstantScheduler",
    ),
    ### 训练过程train配置
    train=dict(
        resume=False,
        checkpoint_save_optimizer=False,
        max_epochs=10,
        gradient_accumulation_steps=1,
        mixed_precision="bf16",  # fp16, bf16
        checkpoint_interval=100, # 200, 只存lora
        checkpoint_total_limit=-1,
        log_with="tensorboard",
        log_interval=1,
        with_ema=False,
        activation_checkpointing=False,
        activation_class_names=[
            "DiTBlock",
        ],
    ),
    test=dict(),
)
