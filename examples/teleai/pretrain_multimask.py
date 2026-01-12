import torch
from teletron.train import Trainer, parse_args
import torch.distributed as dist
from megatron.core import mpu
from teletron.models.flow_match import FlowMatchScheduler
from teletron.train.utils import get_batch, loss_func, flow_loss_func
from teletron.utils import get_timers


def extra_args(parser):
    group = parser.add_argument_group(title='customized args')
    # follow this format to add
    # group.add_argument("--test_valid", type=str, default="")
    group.add_argument("--moe-step-factor-list", type=float, action='append')
    
    return parser

def forward_step(data_iterator, model, time_step=None):
    flow_scheduler = FlowMatchScheduler(shift=1, sigma_min=0.0, extra_one_step=True)
    flow_scheduler.set_timesteps(1000, training=True)
    
    timers = get_timers()
    timers.start_timer('get-data-time')
    batch = next(data_iterator)
    timers.stop_timer('get-data-time')

    latents = batch["latents"]
    noise = torch.randn_like(latents) 
    timestep_range = [0, flow_scheduler.num_train_timesteps]

    timestep_id = torch.randint(timestep_range[0], timestep_range[1], (1,))
    
    timestep = flow_scheduler.timesteps[timestep_id].to(
        dtype=torch.bfloat16, device=torch.cuda.current_device()
    )
    def broadcast_timesteps(input: torch.Tensor):
        tp_cp_src_rank = mpu.get_tensor_context_parallel_src_rank()
        if mpu.get_tensor_context_parallel_world_size() > 1:
            dist.broadcast(input, tp_cp_src_rank, group=mpu.get_tensor_context_parallel_group())

    if time_step is not None:
        timestep = torch.tensor([time_step], dtype=torch.bfloat16, device=torch.cuda.current_device())
    broadcast_timesteps(timestep)
    broadcast_timesteps(noise)

    training_target = flow_scheduler.training_target(latents, noise, timestep)    
    noisy_latents = flow_scheduler.add_noise(latents, noise, timestep)
    output_tensor_list = model(x=noisy_latents, 
                               timestep=timestep, 
                               context=batch['context'],
                               clip_feature=batch['img_clip_feature'],
                               y=batch['img_emb_y'])

    loss = torch.nn.functional.mse_loss(
        output_tensor_list.float(), training_target.float()
    )
    loss_wo_w = loss
    loss = loss * flow_scheduler.training_weight(timestep)

    return [loss, loss_wo_w], flow_loss_func



if __name__ == "__main__":
    args = parse_args(extra_args=extra_args)
    trainer = Trainer(args)
    trainer.pretrain(forward_step_func=forward_step)
