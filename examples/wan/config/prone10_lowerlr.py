import os

dst_size = (720, 480)
dst_fps = 15
dst_num_frames = 81


config = dict(
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
                                
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-20.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-21.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-22.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-23.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-24.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-25.json",

            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-26-1.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-26-2.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-26-3.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-26-4.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-26-5.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-27-1.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-27-2.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-27-3.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-27-4.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-27-5.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-28-1.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-28-2.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-28-3.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-28-4.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-28-5.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-28-6.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-28-7.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-28-8.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-29-1.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-29-2.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-29-3.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-29-4.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-29-5.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-29-6.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-29-7.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-29-8.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-29-9.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-29-10.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-29-11.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-29-12.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-29-13.json",
            
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-30-1.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-30-2.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-30-3.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-30-4.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-30-5.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-30-6.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-30-7.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-30-8.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-30-9.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-31-1.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-31-2.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-31-3.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-31-4.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-31-5.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-31-6.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-31-7.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-31-8.json",

            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-01-1.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-01-2.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-01-3.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-01-4.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-01-5.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-01-6.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-02-1.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-02-2.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-02-3.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-03-1.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-03-2.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-04-1.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-04-2.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-04-3.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-05-1.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-05-2.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-05-3.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-05-4.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-05-5.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-05-6.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-05-7.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-06-1.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-06-2.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-06-3.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-07-1.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-07-2.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-07-3.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-07-4.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-07-5.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-07-6.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-07-7.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-07-8.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-07-9.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-08-1.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-08-2.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-08-3.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-09-1.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-09-2.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-09-3.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-09-4.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-09-5.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-10-1.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-10-2.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-10-3.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-10-4.json",
            # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-11-1.json",
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
                type="GenerateRawFirstRefImage",
            ),
            dict(
                type="PackInputs",
                image_keys=[
                    "images",
                ],
                embedding_keys=[
                    "raw_first_image", 
                ],  
            ),
        ],
    ),
    eval=dict(
        data_path_list=[
            "/nvfile-heatstorage/Text2Video/annotations/200w/pack_zwzx_1.json",
        ],
    ),
        sampler=dict(
        type="DefaultSampler",
        shuffle=False,
        seed=42,
        drop_last=True,
        infinite=True,
    ),
    model_config=dict(
        dit=dict(
            type="CausalDiffusion", # ParallelWanModel
            config=dict(
                has_image_input=False, # t2v:False i2v:True i2v Wan2.2:False
                patch_size=[1, 2, 2],
                in_dim=16, # t2v:16 i2v:36 # 5B 48
                dim=5120, # 1.3B:1536 10B:5120 14B:5120 5B:3072
                ffn_dim=13824, # 1.3B:8960 10B:13824 14B:13824
                freq_dim=256,
                text_dim=4096,
                out_dim=16, # 5B:48
                num_heads=40, # 1.3B:12 10B:40 14B:40 5B:24
                num_layers=40, # 1.3B:30 10B:30 14B:40
                eps=1e-6,
                has_image_pos_emb=False, 
            ),
        ),
        encoder=dict(
            type="teleai_encoder", # wan_encoder
            encoder_schema=['prompt_emb','unprompt_emb','latents'],
            vae=dict(
                type="TeleaiVideoVAE_2_1",
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
            )
        ),
    ),
)
