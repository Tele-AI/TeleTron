import torch
from teletron.train import Trainer, parse_args
import torch.distributed as dist
from megatron.core import mpu
from teletron.models.flow_match import FlowMatchScheduler
from teletron.train.utils import get_batch, loss_func
from teletron.utils import get_timers


def extra_args(parser):
    group = parser.add_argument_group(title='customized args')
    # follow this format to add
    # group.add_argument("--test_valid", type=str, default="")
    group.add_argument("--moe-step-factor-list", type=float, action='append')
    group.add_argument("--test-with-pseudo-data", action="store_true")
    group.add_argument("--test-resolution", type=str, default="360")
    
    return parser

def forward_step(data_iterator, model,time_step=None):
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
    if args.test_with_pseudo_data:
        dp_rank = mpu.get_data_parallel_rank() % 2
        curr_iter = args.curr_iteration % 1000
        input_dict = torch.load(f"../test_data/saved_inputs_{args.test_resolution}/input_dict_iter{curr_iter}_rank{dp_rank}.pt", weights_only=False, map_location="cpu")
        # print(input_dict['noisy_latents'].shape)
        output_tensor_list = model(x=input_dict['noisy_latents'].cuda(),
                                timestep=input_dict['timestep'].cuda(),
                                context=input_dict['prompt_emb']['context'].cuda(),
                                clip_feature=input_dict['image_emb']['clip_feature'].cuda(),
                                y=input_dict['image_emb']['y'].cuda())
        training_target = input_dict['training_target'].cuda()
        loss_weight = input_dict['loss_weight'].cuda()
    else:
        output_tensor_list = model(x=noisy_latents, 
                                timestep=timestep, 
                                context=batch['context'],
                                clip_feature=batch['img_clip_feature'],
                                y=batch['img_emb_y'])

    loss = torch.nn.functional.mse_loss(
        output_tensor_list.float(), training_target.float()
    )
    loss_wo_w = loss
    loss = loss * loss_weight

    first_frame_pred = output_tensor_list[:, :, :1, :, :]
    first_frame_target = training_target[:, :, :1, :, :]
    assert first_frame_pred.shape[1] == 16
    first_frame_loss = torch.nn.functional.mse_loss(
        first_frame_pred.float(), first_frame_target.float()
    )
    loss += first_frame_loss

    # print("loss", loss)
    return [loss, loss_wo_w, first_frame_loss], loss_func



if __name__ == "__main__":
    args = parse_args(extra_args=extra_args)
    trainer = Trainer(args)
    trainer.pretrain(forward_step_func=forward_step)
