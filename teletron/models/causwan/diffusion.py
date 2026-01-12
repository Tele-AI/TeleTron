from typing import Tuple
import torch
from teletron.utils import get_args
from .base import BaseModel
from teletron.models.causwan.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
import torch.distributed as dist
from tqdm import tqdm
import os
from safetensors.torch import load_file
import torch.distributed as dist

def load_sharded_generator_distributed(model, shard_dir):
    shard_files = sorted(f for f in os.listdir(shard_dir) if f.endswith(".safetensors"))

    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    model.to(device)
    loaded = 0

    if rank == 0:
        print(f"🔄 [rank 0] Loading {len(shard_files)} shards from '{shard_dir}'")
        for shard_file in tqdm(shard_files, desc="🔧 Loading safetensor shards"):
            shard_path = os.path.join(shard_dir, shard_file)
            shard = load_file(shard_path, device="cpu")
            for name, param in model.named_parameters():
                if name in shard:
                    param.data.copy_(shard[name])
                    loaded += 1
            del shard
            torch.cuda.empty_cache()
        print(f"✅ [rank 0] Loaded {loaded} generator parameters from safetensors.")

    if is_dist:
        for name, param in model.named_parameters():
            dist.broadcast(param.data, src=0)


class CausalDiffusion(BaseModel):
    def __init__(self, config, device=torch.cuda.current_device()):
        """
        Initialize the Diffusion loss module.
        """
        super().__init__(config, device)
        args = get_args()
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 3)
        self.dtype = torch.bfloat16 if args.mixed_precision else torch.float32
        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block
        self.independent_first_frame = getattr(args, "independent_first_frame", False)

        if hasattr(args, 'gradient_checkpointing'):
            self.generator.enable_gradient_checkpointing()

        # Step 2: Initialize all hyperparameters
        self.num_train_timestep = args.num_train_timestep
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        self.guidance_scale = args.guidance_scale
        self.timestep_shift = getattr(args, "timestep_shift", 5.0)
        self.teacher_forcing = True
        self.save_index = 0
        self.global_rank = dist.get_rank() if dist.is_initialized() else 0
        # Noise augmentation in teacher forcing, we add small noise to clean context latents
        self.noise_augmentation_max_timestep = getattr(args, "noise_augmentation_max_timestep", 0)
        self.config = config
        self.device = torch.cuda.current_device()

    def _initialize_models(self, args, device):
        self.generator = WanDiffusionWrapper(**getattr(args, "model_kwargs", {}), is_causal=True)
        self.generator.model.requires_grad_(True)
        
        for param in self.generator.model.parameters():
            param.requires_grad = False

        for block in self.generator.model.blocks:
            for name, param in block.self_attn.named_parameters():
                param.requires_grad = True

        total_params = sum(p.numel() for p in self.generator.model.parameters())
        trainable_params = sum(p.numel() for p in self.generator.model.parameters() if p.requires_grad)

        total_params_b = total_params / 1e9
        trainable_params_b = trainable_params / 1e9
        trainable_ratio = trainable_params / total_params

        print(f"\n✅ Total parameters in the model: {total_params_b:.3f}B")
        print(f"✅ Trainable parameters: {trainable_params_b:.3f}B")
        print(f"✅ Percentage trainable: {trainable_ratio:.2%}\n")

        # Load checkpoint with memory optimization - no distributed loading
        # load_sharded_generator_distributed(self.generator, "/gemini/space/xxz/Self-Forcing/tmp_shard")

        # self.text_encoder = WanTextEncoder()
        # self.text_encoder.requires_grad_(False)

        # self.vae = WanVAEWrapper()
        # self.vae.requires_grad_(False)

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(torch.cuda.current_device())
        
    def forward(
        self,
        noisy_latents,
        timestep,
        conditional_dict: dict,
        clean_latent_aug: torch.Tensor,
        timestep_clean_aug: torch.Tensor,
        unconditional_dict: dict = None
    ) -> Tuple[torch.Tensor, dict]:     
        guidance_drop_prob = 0.1
        if torch.rand(1).item() > guidance_drop_prob:
            flow_pred, x0_pred = self.generator(
                noisy_image_or_video=noisy_latents,
                conditional_dict=conditional_dict,
                timestep=timestep,
                clean_x=clean_latent_aug if self.teacher_forcing else None,
                aug_t=timestep_clean_aug if self.teacher_forcing else None
            )
        else:
            flow_pred, x0_pred = self.generator(
                noisy_image_or_video=noisy_latents,
                conditional_dict=unconditional_dict,
                timestep=timestep,
                clean_x=clean_latent_aug if self.teacher_forcing else None,
                aug_t=timestep_clean_aug if self.teacher_forcing else None
            )
        return flow_pred, x0_pred 

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and compute the DMD loss.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - generator_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        noise = torch.randn_like(clean_latent)
        batch_size, num_frame = image_or_video_shape[:2]

        # Step 2: Randomly sample a timestep and add noise to denoiser inputs
        index = self._get_timestep(
            0,
            self.scheduler.num_train_timesteps,
            image_or_video_shape[0],
            image_or_video_shape[1],
            self.num_frame_per_block,
            uniform_timestep=False
        )

        timestep = self.scheduler.timesteps[index].to(dtype=self.dtype, device=self.device)
        noisy_latents = self.scheduler.add_noise(
            clean_latent.flatten(0, 1),
            noise.flatten(0, 1),
            timestep.flatten(0, 1)
        ).unflatten(0, (batch_size, num_frame))
        training_target = self.scheduler.training_target(clean_latent, noise, timestep)

        # Step 3: Noise augmentation, also add small noise to clean context latents
        if self.noise_augmentation_max_timestep > 0:
            index_clean_aug = self._get_timestep(
                0,
                self.noise_augmentation_max_timestep,
                image_or_video_shape[0],
                image_or_video_shape[1],
                self.num_frame_per_block,
                uniform_timestep=False
            )
            timestep_clean_aug = self.scheduler.timesteps[index_clean_aug].to(dtype=self.dtype, device=self.device)
            clean_latent_aug = self.scheduler.add_noise(
                clean_latent.flatten(0, 1),
                noise.flatten(0, 1),
                timestep_clean_aug.flatten(0, 1)
            ).unflatten(0, (batch_size, num_frame))
        else:
            clean_latent_aug = clean_latent
            timestep_clean_aug = None

        # Compute loss
        guidance_drop_prob = 0.1

        if torch.rand(1).item() > guidance_drop_prob:
            flow_pred, x0_pred = self.generator(
                noisy_image_or_video=noisy_latents,
                conditional_dict=conditional_dict,
                timestep=timestep,
                clean_x=clean_latent_aug if self.teacher_forcing else None,
                aug_t=timestep_clean_aug if self.teacher_forcing else None
            )
        else:
            flow_pred, x0_pred = self.generator(
                noisy_image_or_video=noisy_latents,
                conditional_dict=unconditional_dict,
                timestep=timestep,
                clean_x=clean_latent_aug if self.teacher_forcing else None,
                aug_t=timestep_clean_aug if self.teacher_forcing else None
            )

        # loss = torch.nn.functional.mse_loss(flow_pred.float(), training_target.float())
        loss = torch.nn.functional.mse_loss(
            flow_pred.float(), training_target.float(), reduction='none'
        ).mean(dim=(2, 3, 4))
        loss_w = loss
        loss = loss * self.scheduler.training_weight(timestep).unflatten(0, (batch_size, num_frame))
        

        log_dict = {
            "x0": clean_latent.detach(),
            "x0_pred": x0_pred.detach()
        }
        return loss, loss_w, log_dict
    
    def set_input_tensor(self, x):
        return None
    
    def state_dict_for_save_checkpoint(self, destination=None, prefix='', keep_vars=False):
        state_dict = self.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        return state_dict
    