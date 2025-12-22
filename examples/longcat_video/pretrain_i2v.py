import torch
import torch.distributed as dist
from megatron.core import mpu
from teletron.train import Trainer, parse_args
from teletron.models.flow_match import FlowMatchScheduler
from teletron.train.utils import flow_loss_func
from teletron.utils import get_timers


def extra_args(parser):
    group = parser.add_argument_group(title='customized args')
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
    loss_weight = flow_scheduler.training_weight(timestep)

    context = batch["context"]
    if context.dim() == 3:
        context = context.unsqueeze(1)
        
    clip_feature = batch.get("img_clip_feature", None)
    img_emb_y = batch.get("img_emb_y", None)

    if img_emb_y is not None:
        assert img_emb_y.dim() == 5 and noisy_latents.dim() == 5
        assert img_emb_y.shape[:1] == noisy_latents.shape[:1]
        if img_emb_y.shape[1] != noisy_latents.shape[1]:
            img_emb_y = img_emb_y[:, -noisy_latents.shape[1]:, :, :, :]
        noisy_latents = torch.cat([img_emb_y, noisy_latents], dim=2)
        num_cond_latents = img_emb_y.shape[2]
    else:
        num_cond_latents = 0
    
    import inspect
    forward_params = inspect.signature(model.forward).parameters
    call_kwargs = dict(
        hidden_states=noisy_latents,
        timestep=timestep,
        encoder_hidden_states=context,
        num_cond_latents=num_cond_latents,
    )
    if "clip_feature" in forward_params and clip_feature is not None:
        call_kwargs["clip_feature"] = clip_feature
    
    pred = model(**call_kwargs)
    if num_cond_latents > 0:
        pred = pred[:, :, num_cond_latents:, :, :]
    
    loss_wo_w = torch.nn.functional.mse_loss(
        pred.float(), training_target.float()
    )
    loss = loss_wo_w * loss_weight
    return [loss, loss_wo_w], flow_loss_func


if __name__ == "__main__":
    args = parse_args(extra_args=extra_args)
    trainer = Trainer(args)
    trainer.pretrain(forward_step_func=forward_step)
