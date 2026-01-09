import os
import torch
from teletron.train import parse_args
from teletron.train.trainer import Trainer
from teletron.train.utils import average_losses_across_data_parallel_group, get_args
import debugpy

def causal_loss_func(output_tensor):
    """Loss function."""
    loss = output_tensor[0].mean()
    averaged_loss = average_losses_across_data_parallel_group([loss])
    loss = loss.unsqueeze(0)

    loss_wo_w = output_tensor[1].mean()
    averaged_loss_wo_w  = average_losses_across_data_parallel_group([loss_wo_w])
    loss_wo_w = loss_wo_w.unsqueeze(0)

    return loss, {"loss": averaged_loss[0], "loss_wo_w": averaged_loss_wo_w[0]}

def extra_args(parser):
    group = parser.add_argument_group(title='customized args')
    # follow this format to add
    group.add_argument("--no_save", action="store_false")
    group.add_argument("--load_raw_video", action="store_false")
    group.add_argument("--gradient-checkpointing", action="store_false")
    group.add_argument("--real-name", type=str, default="Wan2.1-T2V")
    group.add_argument("--negative_prompt",type=str,
                       default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")
    group.add_argument("--timestep_shift",default=5.0)
    group.add_argument("--model_kwargs",default={"timestep_shift": 5.0})
    group.add_argument("--guidance_scale",default=5.0)
    group.add_argument("--mixed_precision",default=True)
    group.add_argument("--image_or_video_shape",default=[1,21,16,60,104])
    group.add_argument("--num_frame_per_block",default=3)
    group.add_argument("--num_train_timestep",default=1000)
    
    group2 = parser.add_argument_group(title='encoder args')
    group2.add_argument("--encoder_model_path", type=str, nargs = '+',default=
                       ['/workspace/Wan2___1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth', 
                        '/workspace/Wan2___1-I2V-14B-480P/Wan2.1_VAE.pth', 
                        '/workspace/Wan2___1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth']
                       )
    group2.add_argument("--encoder_tokenizer_path", type=str, default=
                       "/workspace/Wan2___1-I2V-14B-480P/google/umt5-xxl")
    group.add_argument("--depth-model-path", type=str, default=
                        "/nvfile-heatstorage/ai_infra/ckpts/lit117/qiuyang/video_depth_anything_vitl.pth")

    return parser

def wait_for_debugger(rank_to_debug=0, port=5678):
    rank = int(os.environ.get("RANK", "0"))
    # All ranks pause here before debugger
    if rank == rank_to_debug:
        print(f"[Rank {rank}] Waiting for debugger on port {port}...")
        debugpy.listen(("0.0.0.0", port))
        debugpy.wait_for_client()
        print(f"[Rank {rank}] Debugger attached.")

def forward_step(data_iterator, ddpmodel):
    batch = next(data_iterator)
    args = get_args()
    device = torch.cuda.current_device()
    dtype = torch.bfloat16 if args.mixed_precision else torch.float32
    accumulation_steps = getattr(ddpmodel, "accumulation_steps", 1)
    
    clean_latent = batch["latents"].to(device, dtype)
    clean_latent = clean_latent.permute(0, 2, 1, 3, 4)
    image_latent = clean_latent[:, 0:1, ]
    image_or_video_shape = args.image_or_video_shape

    from teletron.utils.aux_func import get_attr_wrapped_model
    with torch.no_grad():
        conditional_dict = {'prompt_embeds':batch["prompt_emb"]}
        if not getattr(ddpmodel, "unconditional_dict", None):
            unconditional_dict = {'prompt_embeds':batch["unprompt_emb"]}
        else:
            unconditional_dict = ddpmodel.unconditional_dict

    generator_loss = get_attr_wrapped_model(ddpmodel,"generator_loss")
    generator_loss, generator_loss_w, log_dict = generator_loss(
        image_or_video_shape=image_or_video_shape,
        conditional_dict=conditional_dict,
        unconditional_dict=unconditional_dict,
        clean_latent=clean_latent,
        initial_latent=image_latent,
    )
    generator_loss = generator_loss / accumulation_steps
    generator_loss_w = generator_loss_w / accumulation_steps
    
    return [generator_loss, generator_loss_w], causal_loss_func


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    args = parse_args(extra_args=extra_args)
    trainer = Trainer(args)
    trainer.pretrain(forward_step_func=forward_step)
