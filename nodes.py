import torch
from torch import nn
import comfy.sample
import comfy.model_management
import comfy.utils
import gc
import logging

class MotionGuidedSampler(nn.Module):
    def __init__(
        self,
        motion_strength: float = 0.5,
        consistency_strength: float = 0.9,
        denoise_strength: float = 0.8
    ):
        super().__init__()
        self.motion_strength = motion_strength
        self.consistency_strength = consistency_strength
        self.denoise_strength = denoise_strength
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def extract_motion_vector(self, current_latent: torch.Tensor, prev_latent: torch.Tensor) -> torch.Tensor:
        try:
            return current_latent - prev_latent
        except RuntimeError as e:
            self.logger.error(f"Motion extraction error: {e}")
            return torch.zeros_like(current_latent)

    def apply_motion_vector(self, content: torch.Tensor, motion_vector: torch.Tensor) -> torch.Tensor:
        try:
            motion_applied = content + (motion_vector * self.motion_strength)
            if torch.isnan(motion_applied).any():
                self.logger.warning("NaN detected in motion application")
                return content
            return motion_applied
        except RuntimeError as e:
            self.logger.error(f"Motion application error: {e}")
            return content

    def process_frames(self, latent_frames, model, positive, negative, seed, steps, cfg, sampler_name, scheduler, denoise):
        batch_size = latent_frames.shape[0]
        processed_frames = []
        device = comfy.model_management.get_torch_device()
        
        latent_frames = latent_frames.to(device)
        pbar = comfy.utils.ProgressBar(batch_size)
        
        # Initialize sampler
        sampler = comfy.samplers.KSampler(
            model, 
            steps=steps,
            device=device,
            sampler=sampler_name,
            scheduler=scheduler,
            denoise=denoise
        )

        # Process first frame
        noise = comfy.sample.prepare_noise(latent_frames[0:1], seed, None).to(device)
        first_frame = sampler.sample(
            noise,
            positive,
            negative,
            cfg=cfg,
            latent_image=latent_frames[0:1],
            force_full_denoise=True
        )
        
        processed_frames.append(first_frame)
        prev_orig = latent_frames[0:1].to(device)
        prev_styled = first_frame
        
        pbar.update(1)

        # Process remaining frames
        for i in range(1, batch_size):
            current_orig = latent_frames[i:i+1].to(device)
            
            motion = self.extract_motion_vector(current_orig, prev_orig)
            motion_guided = self.apply_motion_vector(prev_styled, motion)
            
            current_noise = comfy.sample.prepare_noise(motion_guided, seed + i, None).to(device)
            current_frame = sampler.sample(
                current_noise,
                positive,
                negative,
                cfg=cfg,
                latent_image=motion_guided,
                force_full_denoise=True
            )
            
            processed_frames.append(current_frame)
            prev_orig = current_orig
            prev_styled = current_frame
            
            pbar.update(1)
            
            if i % 5 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                self.logger.info(f"Processed {i}/{batch_size} frames")
        
        result = torch.cat(processed_frames, dim=0)
        return result.to(device)

    def forward(self, latent_frames, model, positive, negative, noise_seed, steps, cfg, sampler_name, scheduler, denoise):
        try:
            return self.process_frames(
                latent_frames=latent_frames,
                model=model,
                positive=positive,
                negative=negative,
                seed=noise_seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                denoise=denoise
            )
        except Exception as e:
            self.logger.error(f"Processing error: {str(e)}")
            raise e

class CohernetVideoSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "video_latents": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (["euler", "euler_ancestral", "heun", "dpm_2", 
                                "dpm_2_ancestral", "lms", "dpm_fast", "dpm_adaptive",
                                "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_2m"],),
                "scheduler": (["simple", "karras", "exponential", "normal", "ddim_uniform"],),
                "denoise": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                "motion_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "consistency_strength": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05}),
                "denoise_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling"

    def sample(self, model, video_latents, positive, negative, seed, steps,
               cfg, sampler_name, scheduler, denoise, motion_strength,
               consistency_strength, denoise_strength):
        
        sampler = MotionGuidedSampler(
            motion_strength=motion_strength,
            consistency_strength=consistency_strength,
            denoise_strength=denoise_strength
        )
        
        try:
            samples = sampler(
                latent_frames=video_latents['samples'],
                model=model,
                positive=positive,
                negative=negative,
                noise_seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                denoise=denoise
            )
            
            return ({"samples": samples},)
            
        except Exception as e:
            logging.error(f"Sampling error: {str(e)}")
            raise e

# Node registration
NODE_CLASS_MAPPINGS = {
    "CohernetVideoSampler": CohernetVideoSampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CohernetVideoSampler": "Cohernet Video Sampler"
}