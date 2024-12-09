# ComfyUI Coherent Video Sampler Node (V0.2)

A custom node for ComfyUI that enables coherent video generation while maintaining efficient memory usage, specifically optimized for heavy models like Flux.

![image](https://github.com/user-attachments/assets/9b97aa0e-fcec-4dc6-8843-f5b1416aa66b)


## Features

- 🎥 Frame-by-frame video processing with motion preservation
- 🧠 Efficient memory management for heavy models
- 🔄 Progressive denoising with coherence maintenance
- 💫 Dynamic quality control and motion guidance
- 🎨 Style preservation across frames

## Installation

1. Navigate to your ComfyUI custom nodes directory:
```bash
cd ComfyUI/custom_nodes
```

2. Clone this repository:
```bash
git clone https://github.com/ShmuelRonen/ComfyUI-CohernetVideoSampler.git
```

3. Restart ComfyUI

## Usage

The node appears in the node menu as "Cohernet Video Sampler". 

### Inputs
- `model`: Your diffusion model (tested extensively with Flux)
- `positive`: Positive prompt conditioning
- `negative`: Negative prompt conditioning
- `video_latents`: Input video in latent space (from VAE Encode)
- `seed`: Generation seed
- `steps`: Number of sampling steps
- `cfg`: Classifier free guidance scale
- `sampler_name`: Choice of sampler
- `scheduler`: Choice of scheduler
- `denoise`: Denoising strength
- `motion_strength`: Strength of motion preservation

### Memory Management
The node implements several memory optimization techniques:
- Progressive batch processing
- Automatic VRAM cleanup
- Dynamic batch size adjustment
- Efficient latent space operations

This allows it to work smoothly even with memory-intensive models like Flux without OOM errors.

## Example Workflow

1. Load your video
2. Encode it to latent space using VAE Encode
3. Connect to Coherent Video Sampler
4. Set your sampling parameters
5. Generate coherent video output

## Parameters Guide

- `denoise`: Lower values (0.4-0.6) preserve more of the original motion
- `motion_strength`: Higher values (0.7-0.9) maintain stronger frame-to-frame coherence
- `steps`: Even with heavy models, 20-30 steps usually sufficient
- `cfg`: Standard values (7-9) work well for most cases

## Memory Usage Examples

When using with Flux model:
- 20 frame video @ 512x512: ~8GB VRAM
- 40 frame video @ 512x512: ~10GB VRAM
- Processing happens in windows of frames to maintain stable memory usage

## Known Limitations

- Very long videos might need to be processed in segments
- Extreme motion can affect coherence
- High denoise values might reduce motion preservation

## Future Plans

- Additional motion control parameters
- Custom denoising patterns
- Advanced style preservation options
- Multi-model support optimization

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT License](LICENSE)

## Acknowledgments

- ComfyUI team for the amazing framework
- Flux model team for the inspiration in handling heavy models
```
