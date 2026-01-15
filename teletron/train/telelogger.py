
import torch
from megatron.core import mpu
from teletron.utils import (
    get_args,
    print_rank_last,
    get_num_microbatches
)
from datetime import datetime
from teletron.train.utils import report_memory, report_theoretical_memory
from teletron.utils.global_vars import get_timers, get_tensorboard_writer, get_wandb_writer

NUM_BYTES_IN_MEGABYTE = 1024 * 1024


class TeleLoggerMixin:

    def log_training_infos(
        self,
        loss_dict,
        total_loss_dict,
        learning_rate,
        decoupled_learning_rate,
        iteration,
        loss_scale,
        report_memory_flag,
        skipped_iter,
        grad_norm,
        params_norm,
        num_zeros_in_grad,
        dit_time,
    ):
        """Log training information such as losses, timing, ...."""
        args = get_args()
        timers = get_timers()

        writer = get_tensorboard_writer()
        wandb_writer = get_wandb_writer()
        # one_logger = get_one_logger()

        # Advanced, skipped, and Nan iterations.
        advanced_iters_key = 'advanced iterations'
        skipped_iters_key = 'skipped iterations'
        nan_iters_key = 'nan iterations'
        # Advanced iterations.
        if not skipped_iter:
            total_loss_dict[advanced_iters_key] = total_loss_dict.get(
                advanced_iters_key, 0) + 1
        else:
            if advanced_iters_key not in total_loss_dict:
                total_loss_dict[advanced_iters_key] = 0
        # Skipped iterations.
        total_loss_dict[skipped_iters_key] = total_loss_dict.get(
            skipped_iters_key, 0) + skipped_iter
        # Update losses and set nan iterations
        got_nan = False
        for key in loss_dict:
            if not skipped_iter:
                total_loss_dict[key] = total_loss_dict.get(
                    key, torch.tensor([0.0], dtype=torch.float, device='cuda')) + loss_dict[key]
            else:
                value = loss_dict[key].float().sum().item()
                is_nan = value == float('inf') or \
                        value == -float('inf') or \
                        value != value
                got_nan = got_nan or is_nan
        total_loss_dict[nan_iters_key] = total_loss_dict.get(
            nan_iters_key, 0) + int(got_nan)
        # breakpoint()

        # Logging.
        timers_to_log = [
            'forward-backward',
            'forward-compute',
            'backward-compute',
            'batch-generator',
            'forward-recv',
            'forward-send',
            'backward-recv',
            'backward-send',
            'forward-send-forward-recv',
            'forward-send-backward-recv',
            'backward-send-forward-recv',
            'backward-send-backward-recv',
            'forward-backward-send-forward-backward-recv',
            'layernorm-grads-all-reduce',
            'embedding-grads-all-reduce',
            'all-grads-sync',
            'params-all-gather',
            'optimizer-copy-to-main-grad',
            'optimizer-unscale-and-check-inf',
            'optimizer-clip-main-grad',
            'optimizer-count-zeros',
            'optimizer-inner-step',
            'optimizer-copy-main-to-model-params',
            'optimizer']

        # Calculate batch size.
        batch_size = args.micro_batch_size * args.data_parallel_size * \
            get_num_microbatches()

        # Track app tag & app tag ID
        # if one_logger:
        #     job_name = os.environ.get('SLURM_JOB_NAME', None)
        #     current_app_tag = f'{job_name}_{batch_size}_{args.world_size}'
        #     one_logger.log_app_tag(current_app_tag)

        total_iterations = total_loss_dict[advanced_iters_key] + \
                           total_loss_dict[skipped_iters_key]

        # Tensorboard values.
        # Timer requires all the ranks to call.
        if args.log_timers_to_tensorboard and \
           (iteration % args.tensorboard_log_interval == 0):
            timers.write(timers_to_log, writer, iteration,
                         normalizer=total_iterations)
        if writer and (iteration % args.tensorboard_log_interval == 0):
            if wandb_writer:
                wandb_writer.log({'samples vs steps': args.consumed_train_samples},
                                 iteration)
            if args.log_learning_rate_to_tensorboard:
                writer.add_scalar('learning-rate', learning_rate, iteration)
                if args.decoupled_lr is not None:
                    writer.add_scalar('decoupled-learning-rate', decoupled_learning_rate, iteration)
                writer.add_scalar('learning-rate vs samples', learning_rate,
                                  args.consumed_train_samples)
                if wandb_writer:
                    wandb_writer.log({'learning-rate': learning_rate}, iteration)
            if args.log_batch_size_to_tensorboard:
                writer.add_scalar('batch-size', batch_size, iteration)
                writer.add_scalar('batch-size vs samples', batch_size,
                                  args.consumed_train_samples)
                if wandb_writer:
                    wandb_writer.log({'batch-size': batch_size}, iteration)
            for key in loss_dict:
                writer.add_scalar(key , loss_dict[key], iteration)
                writer.add_scalar(key + ' vs samples', loss_dict[key],
                                  args.consumed_train_samples)
                if wandb_writer:
                    wandb_writer.log({key: loss_dict[key]}, iteration)
            if args.log_loss_scale_to_tensorboard:
                writer.add_scalar('loss-scale', loss_scale, iteration)
                writer.add_scalar('loss-scale vs samples', loss_scale,
                                  args.consumed_train_samples)
                if wandb_writer:
                    wandb_writer.log({'loss-scale': loss_scale}, iteration)
            if args.log_world_size_to_tensorboard:
                writer.add_scalar('world-size', args.world_size, iteration)
                writer.add_scalar('world-size vs samples', args.world_size,
                                  args.consumed_train_samples)
                if wandb_writer:
                    wandb_writer.log({'world-size': args.world_size}, iteration)
            if grad_norm is not None:
                writer.add_scalar('grad-norm', grad_norm, iteration)
                writer.add_scalar('grad-norm vs samples', grad_norm,
                                  args.consumed_train_samples)
                if wandb_writer:
                    wandb_writer.log({'grad-norm': grad_norm}, iteration)
            if num_zeros_in_grad is not None:
                writer.add_scalar('num-zeros', num_zeros_in_grad, iteration)
                writer.add_scalar('num-zeros vs samples', num_zeros_in_grad,
                                  args.consumed_train_samples)
                if wandb_writer:
                    wandb_writer.log({'num-zeros': num_zeros_in_grad}, iteration)
            if params_norm is not None:
                writer.add_scalar('params-norm', params_norm, iteration)
                writer.add_scalar('params-norm vs samples', params_norm,
                                  args.consumed_train_samples)
                if wandb_writer:
                    wandb_writer.log({'params-norm': params_norm}, iteration)
            if args.log_memory_to_tensorboard:
                mem_stats = torch.cuda.memory_stats()
                writer.add_scalar(
                    "mem-reserved-bytes",
                    mem_stats["reserved_bytes.all.current"],
                    iteration,
                )
                writer.add_scalar(
                    "mem-allocated-bytes",
                    mem_stats["allocated_bytes.all.current"],
                    iteration,
                )
                writer.add_scalar(
                    "mem-allocated-count",
                    mem_stats["allocation.all.current"],
                    iteration,
                )
        # if args.num_experts is not None:
        #     moe_loss_scale = 1 / get_num_microbatches()
        #     track_moe_metrics(moe_loss_scale, iteration, writer, None, total_loss_dict, args.moe_per_layer_logging)

        if iteration % args.log_interval == 0:
            # elapsed_time = timers('interval-time').elapsed(barrier=True)
            # elapsed_time_per_iteration = elapsed_time / total_iterations

            # throughput = num_floating_point_operations(args, batch_size) / (
            #     elapsed_time_per_iteration * 10**12 * args.world_size)
            log_string = f" [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
            log_string += ' iteration {:8d}/{:8d} |'.format(
                iteration, args.train_iters)
            log_string += ' consumed samples: {:12d} |'.format(
                args.consumed_train_samples)
            log_string += ' dit_time (ms): {:.2f} |'.format(dit_time * 1000)
            # log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(
            #     elapsed_time_per_iteration * 1000.0)
            # if args.log_throughput:
            #     log_string += f' throughput per GPU (TFLOP/s/GPU): {throughput:.1f} |'
                # if args.log_timers_to_tensorboard:
            assert learning_rate is not None
            # Decoupled_learning_rate should be not None only on first and last pipeline stage.
            log_string += ' learning rate: {:.6E} |'.format(learning_rate)
            if args.decoupled_lr is not None and (mpu.is_pipeline_first_stage(ignore_virtual=True) or
                                                mpu.is_pipeline_last_stage(ignore_virtual=True)):
                assert decoupled_learning_rate is not None
                log_string += ' decoupled learning rate: {:.6E} |'.format(decoupled_learning_rate)
            else:
                assert decoupled_learning_rate is None
            # log_string += ' global batch size: {:5d} |'.format(batch_size)
            for key in total_loss_dict:
                if key not in [advanced_iters_key, skipped_iters_key,
                            nan_iters_key]:
                    avg = total_loss_dict[key].item() / \
                        float(max(1, total_loss_dict[advanced_iters_key]))
                    if avg > 0.0:
                        log_string += ' {}: {:.4f} |'.format(key, avg)
                    total_loss_dict[key] = torch.tensor([0.0], dtype=torch.float, device='cuda')
            log_string += ' loss scale: {:.1f} |'.format(loss_scale)
            if grad_norm is not None:
                log_string += ' grad norm: {:.3f} |'.format(grad_norm)
            if num_zeros_in_grad is not None:
                log_string += ' num zeros: {:.1f} |'.format(num_zeros_in_grad)
            if params_norm is not None:
                log_string += ' params norm: {:.3f} |'.format(params_norm)
            log_string += ' number of skipped iterations: {:3d} |'.format(
                total_loss_dict[skipped_iters_key])
            log_string += ' number of nan iterations: {:3d} |'.format(
                total_loss_dict[nan_iters_key])
            loss = loss_dict["loss"].item()
            # breakpoint()
            # log_string += 'loss: {:.4f} |'.format(loss)
            total_loss_dict[advanced_iters_key] = 0
            total_loss_dict[skipped_iters_key] = 0
            total_loss_dict[nan_iters_key] = 0
            print_rank_last(log_string)
            if report_memory_flag and learning_rate > 0.:
                # Report memory after optimizer state has been initialized.
                if torch.distributed.get_rank() == 0:
                    num_microbatches = get_num_microbatches()
                    report_theoretical_memory(args, num_microbatches=num_microbatches, verbose=True)
                report_memory('(after {} iterations)'.format(iteration))
                report_memory_flag = False
            # timers.log(timers_to_log, normalizer=args.log_interval)

        return report_memory_flag

    def log_validation_infos(self, loss_dict, iteration, eval_time_steps=None):
        args = get_args()
        writer = get_tensorboard_writer()
        if writer:
            if eval_time_steps:
                for time_step in loss_dict:
                    # writer.add_scalar(f"time step: {time_step}")
                    for key in loss_dict[time_step]:
                        writer.add_scalar(f"validation/time step: {time_step} {key}" , loss_dict[time_step][key], iteration)
                        writer.add_scalar(f"validation vs samples/time step: {time_step} {key}" + ' ', loss_dict[time_step][key],
                                    args.consumed_train_samples)
            else:
                for key in loss_dict:
                    writer.add_scalar(f"validation: {key}" , loss_dict[key], iteration)
                    writer.add_scalar(f"validation: {key}" + ' vs samples', loss_dict[key],
                                    args.consumed_train_samples)