import torch
from teletron.train import Trainer, parse_args
import torch.distributed as dist
from megatron.core import mpu
from teletron.models.teleai.schedulers.flow_match import FlowMatchScheduler
from teletron.train.utils import get_batch, sr_loss_func
from teletron.utils import get_timers


def extra_args(parser):
    group = parser.add_argument_group(title='customized args')
    # follow this format to add
    # group.add_argument("--test_valid", type=str, default="")
    group.add_argument("--moe-step-factor-list", type=float, action='append')
    
    return parser

def forward_step(data_iterator, model, time_step=None):
    prompt_emb = {}

    timers = get_timers()
    timers.start_timer('get-data-time')
    batch = next(data_iterator)
    timers.stop_timer('get-data-time')

    latents = batch["latents"]
    fake_latents = batch["fake_latents"]
    noise = torch.randn_like(latents) if "noise" not in batch else batch["noise"]
    end_sigma = 0.5

    src_latents = (1 - end_sigma) * fake_latents + end_sigma * noise
    sr_sigma = 1- torch.rand((1,))
    rescaled_timestep = sr_sigma * 1000 * end_sigma
    sr_sigma = sr_sigma.to(
        dtype=torch.bfloat16, device=torch.cuda.current_device()
    )
    rescaled_timestep = rescaled_timestep.to(
        dtype=torch.bfloat16, device=torch.cuda.current_device()
    )

    def broadcast_timesteps(input: torch.Tensor):
        tp_cp_src_rank = mpu.get_tensor_context_parallel_src_rank()
        if mpu.get_tensor_context_parallel_world_size() > 1:
            dist.broadcast(input, tp_cp_src_rank, group=mpu.get_tensor_context_parallel_group())

    if time_step is not None:
        rescaled_timestep = torch.tensor([time_step], dtype=torch.bfloat16, device=torch.cuda.current_device())
    broadcast_timesteps(rescaled_timestep)
    broadcast_timesteps(noise)
    prompt_emb["context"] = batch["context"]
    training_target = src_latents - latents
    image_emb = {}
    image_emb["y"] = batch["img_emb_y"]
    #print('y shape', image_emb['y'].shape)
    noisy_latents = (1 - sr_sigma) * latents + sr_sigma * src_latents
    #print('x shape', latents.shape)
    image_emb["clip_feature"] = batch["img_clip_feature"]

    #print("batch[clip_feature].shape", batch["clip_feature"].shape)
    #print("noisy_latents.shape", noisy_latents.shape)

    output_tensor_list = model(x=noisy_latents, 
                               timestep=rescaled_timestep, 
                               context=prompt_emb["context"],
                               clip_feature=image_emb["clip_feature"],
                               y=image_emb["y"])

    loss = torch.nn.functional.mse_loss(
        output_tensor_list.float(), training_target.float()
    )
    
    return [loss], sr_loss_func



if __name__ == "__main__":
    args = parse_args(extra_args=extra_args)
    trainer = Trainer(args)
    trainer.pretrain(forward_step_func=forward_step)
