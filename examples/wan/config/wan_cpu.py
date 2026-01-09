import os

# 数据w h t
dst_size = (720, 480)
dst_fps = 15
dst_num_frames = 49
NUM_WORKER = 1
# 训练配置
config = dict(
    ## log&ckpts路径
    runners=["projects.wan.adaptors.WanTrainerCPU"],
    ## 分布式配置for luancher
    launch=dict(
        gpu_ids=[0,1,2,3,4,5,6,7],
        distributed_type="DEEPSPEED",
        deepspeed_config=dict(
            deepspeed_config_file=os.path.join(
                os.getcwd(), "configs/accelerate_configs/zero2.json"
            ),
        ),
        num_machines=os.environ.get("WORLD_SIZE", 1),
        until_completion=True,
    ),
    ## 训练配置for runner
    ### dataloader配置
    dataloaders=dict(
        #### dataloader train配置
        train=dict(
            dataset=dict(
                type="ClipDataset",
                data_path_list=[
                    "/nvfile-heatstorage/Text2Video/annotations/200w/pack_zwzx_1.json",
                    "/nvfile-heatstorage/Text2Video/annotations/200w/pack_zwzx_2.json",
                    "/nvfile-heatstorage/Text2Video/annotations/200w/pack_zwzx_3.json",
                ],
                filter_cfg=dict(
                    dst_size=dst_size,
                    dst_num_frames=dst_num_frames,
                    dst_fps=dst_fps,
                    multiple=16,
                    min_area=dst_size[0] * dst_size[1],
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
                        type="PromptToTransformerEmbedding",
                        model_name="umt5",
                        model_path="/data02/Wan2.1-I2V-14B-720P-Diffusers",
                        max_length=256,
                        with_attention_mask=True,
                    ),
                    dict(
                        type="PackInputs",
                        image_keys=["images"],
                        embedding_keys=[
                            "prompt_embeds",
                            "prompt_masks",
                        ],
                        dst_size=dst_size,
                    ),
                    dict(
                        type="GenerateRefImagesWithMask",
                        mask_cfg={
                            "t2v": 0.0,
                            "i2v": 0.6,
                            "clear": 0.0,
                            "continuation": 0.1,
                            "random": 0.0,
                            "transition": 0.3
                        },
                        min_clear_ratio=0.0,
                        max_clear_ratio=1.0,
                    ),
                ],
            ),
            batch_size_per_gpu=1,
            num_workers=NUM_WORKER,
            sampler=dict(
                type="DefaultSampler",
            ),
            collator=dict(
                is_equal=True,
            ),
        ),
        #### dataloader eval配置
        eval=None,
    ),
    ### 模型model配置
    models=dict(
        pretrained="/data02/Wan2.1-I2V-14B-720P-Diffusers",
        transformer_pretrained="/data02/Wan2.1-I2V-14B-720P-Diffusers/transformer",
        text_encoder=dict(
            max_length=256,
        ),
        transformer=dict(
            in_channels=36,  # with ref images 16->32 / 33 / 36, with ref and cn_images 16->48
        ),
        loss=dict(),
        # flow matching schdule
        scheduler=dict(
            flow_resolution_shifting=False,
            flow_base_image_seq_len=256,
            flow_max_image_seq_len=4096,
            flow_base_shift=0.5,
            flow_max_shift=1.15,
            flow_shift=5.0,
            flow_weighting_scheme="none",
            flow_logit_mean=0.0,
            flow_logit_std=1.0,
            flow_mode_scale=1.29,
        ),
    ),
    ### 优化器optimizer配置
    optimizers=dict(
        type="AdamW",
        lr=1e-5,
        weight_decay=1e-2,
    ),
    ### 学习率scheduler配置
    schedulers=dict(
        type="ConstantScheduler",
    ),
    ### 训练过程train配置
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
            "WanTransformerBlock",
        ],
    ),
    ### 测试过程test配置
    test=dict(),
)
