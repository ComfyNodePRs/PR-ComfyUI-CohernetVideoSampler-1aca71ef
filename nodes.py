import torch
from torch import nn
import comfy.sample
import comfy.model_management
import comfy.utils
import gc

class MotionGuidedSampler(nn.Module):
    def __init__(self):
        super().__init__()
        self.prev_styled = None
        
    def extract_motion_vector(self, current_latent, prev_latent):
        return current_latent - prev_latent
        
    def apply_motion_vector(self, content, motion_vector, strength=0.8):
        return content + (motion_vector * strength)

    def process_frames(self, latents, model, positive, negative, seed, steps, cfg, sampler_name, scheduler, denoise):
        batch_size = latents.shape[0]
        processed_frames = []
        
        pbar = comfy.utils.ProgressBar(batch_size)
        print(f"Processing {batch_size} frames with motion guidance")
        
        first_frame = comfy.sample.sample(
            model,
            noise=comfy.sample.prepare_noise(latents[0:1], seed, None),
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent_image=latents[0:1],
            denoise=denoise
        )
        
        processed_frames.append(first_frame)
        self.prev_styled = first_frame
        prev_original = latents[0:1]
        
        for i in range(1, batch_size):
            current_original = latents[i:i+1]
            
            motion = self.extract_motion_vector(current_original, prev_original)
            
            motion_guided = self.apply_motion_vector(self.prev_styled, motion)
            
            current_frame = comfy.sample.sample(
                model,
                noise=comfy.sample.prepare_noise(motion_guided, seed + i, None),
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=positive,
                negative=negative,
                latent_image=motion_guided,
                denoise=min(0.6, denoise)  
            )
            
            processed_frames.append(current_frame)
            self.prev_styled = current_frame
            prev_original = current_original
            
            pbar.update(1)
            
            if i % 5 == 0:
                torch.cuda.empty_cache()
        
        return torch.cat(processed_frames, dim=0)

    def forward(self, model, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latents, denoise):
        try:
            samples = self.process_frames(
                latents,
                model,
                positive,
                negative,
                noise_seed,
                steps,
                cfg,
                sampler_name,
                scheduler,
                denoise
            )
            return samples
            
        except Exception as e:
            print(f"Error in motion-guided sampling: {str(e)}")
            raise e
        finally:
            torch.cuda.empty_cache()

class MotionGuidedSamplerNode:
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
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling"

    def sample(self, model, video_latents, positive, negative, seed, steps, 
               cfg, sampler_name, scheduler, denoise, motion_strength):
        
        sampler = MotionGuidedSampler()
        samples = sampler(
            model=model,
            noise_seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latents=video_latents['samples'],
            denoise=denoise
        )
        
        return ({"samples": samples},)

NODE_CLASS_MAPPINGS = {
    "CohernetVideoSampler": MotionGuidedSamplerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CohernetVideoSampler": "Cohernet Video Sampler"
}