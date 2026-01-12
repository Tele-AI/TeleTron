import torch
import torch.distributed as dist
from megatron.core import mpu

from teletron.models.flow_match import FlowMatchScheduler
from teletron.train import Trainer, parse_args
from teletron.train.utils import average_losses_across_data_parallel_group

def loss_func(output_tensor):
    """Loss function."""
    loss = output_tensor[0].mean()
    averaged_loss = average_losses_across_data_parallel_group([loss])
    loss = loss.unsqueeze(0)
    return loss, {"loss": averaged_loss[0]}

def extra_args(parser):
    group = parser.add_argument_group(title='customized args')
    # follow this format to add
    # group.add_argument("--test_valid", type=str, default="")
    group.add_argument("--moe-step-factor-list", type=float, action='append')
    group = parser.add_argument_group(title='encoder args')
    group.add_argument("--encoder-model-path", type=str, default=None)
    group.add_argument("--encoder-tokenizer-path", type=str, default=None)
    return parser

def forward_step(data_iterator, model):
    flow_scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
    flow_scheduler.set_timesteps(1000, training=True)
    prompt_emb = {}
    batch = next(data_iterator)
    latents = batch["latents"]
    noise = torch.randn_like(latents) if "noise" not in batch else batch["noise"]

    timestep_id = torch.randint(0, flow_scheduler.num_train_timesteps, (1,))
    timestep = flow_scheduler.timesteps[timestep_id].to(
        dtype=torch.bfloat16, device=torch.cuda.current_device()
    )
    def broadcast_timesteps(input: torch.Tensor):
        tp_cp_src_rank = mpu.get_tensor_context_parallel_src_rank()
        if mpu.get_tensor_context_parallel_world_size() > 1:
            dist.broadcast(input, tp_cp_src_rank, group=mpu.get_tensor_context_parallel_group())

    broadcast_timesteps(timestep)
    broadcast_timesteps(noise)
    prompt_emb["context"] = batch["context"]
    training_target = flow_scheduler.training_target(latents, noise, timestep)
    image_emb = {}
    image_emb["y"] = batch["img_emb_y"]
    noisy_latents = flow_scheduler.add_noise(latents, noise, timestep)

    output_tensor_list = model(x=noisy_latents, 
                               timestep=timestep, 
                               context=prompt_emb["context"],
                               y=image_emb["y"])
    
    loss = torch.nn.functional.mse_loss(
        output_tensor_list.float(), training_target.float()
    )
    loss = loss * flow_scheduler.training_weight(timestep)
    return [loss], loss_func



if __name__ == "__main__":
    args = parse_args(extra_args=extra_args)
    trainer = Trainer(args)
    trainer.pretrain(forward_step_func=forward_step)