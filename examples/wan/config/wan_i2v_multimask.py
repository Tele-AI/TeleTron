import os

dst_size = (720, 480)
dst_fps = 15
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
                    "/nvfile-heatstorage/cjf/share/export_to_clipdataset/istock/istock_0.json",
                ],

                filter_cfg=dict(
                    dst_size=dst_size,
                    dst_num_frames=dst_num_frames,
                    dst_fps=dst_fps,
                    multiple=16,
                    min_area=dst_size[0] * dst_size[1],
                    optical_flow_th=4,
                    aesthetic_th=4.5,
                    bucket_size_th=4,
                    motion_th=0,
                    clearity_th=0.9,
                    laplacian_th=200,
                    training_suitability_th=5.0,
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
                        type="PackInputs",
                        image_keys=[
                            "images",
                        ],
                        #dst_size=dst_size,
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
        dit_path="/workspace/Wan2___1-I2V-14B-480P", 
        tiled=True, 
        tile_size=(34, 34),
        tile_stride=(18, 16), 
    ),
    ### 优化器optimizer配置
    optimizers=dict(
        type="AdamW",
        lr=1e-5,
    ),
    ### 学习率scheduler配置
    schedulers=dict(
        type="ConstantScheduler",
    ),
    ### 训练过程train配置
    train=dict(
        resume=True,
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
