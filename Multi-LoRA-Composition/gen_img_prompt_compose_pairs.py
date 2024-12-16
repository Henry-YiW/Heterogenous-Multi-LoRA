import os
import torch
from glob import glob
from diffusers import DiffusionPipeline, AutoencoderKL, DPMSolverMultistepScheduler
from callbacks import make_callback

def load_prompt(file_path):
    """Load the prompt from a given file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.readline().strip()


def main(args):
    # Create output folder if it doesn't exist
    os.makedirs("gen_images_compose", exist_ok=True)

    # Base model
    model_name = 'SG161222/Realistic_Vision_V5.1_noVAE'

    # Initialize the pipeline
    pipeline = DiffusionPipeline.from_pretrained(
        model_name,
        custom_pipeline="./pipelines/sd1.5_0.26.3",
        use_safetensors=True
    ).to("cuda")

    # Set VAE
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
    ).to("cuda")
    pipeline.vae = vae

    # Set scheduler
    schedule_config = dict(pipeline.scheduler.config)
    schedule_config["algorithm_type"] = "dpmsolver++"
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(schedule_config)

    # Reality LoRA mapping
    reality_lora_mapping = {
        "02": "JFC",
        "03": "IU",
        "05": "Bright",
        "09": "Library",
        "12": "Scarlett",
        "13": "Umbrella",
        "15": "Rock",
        "16": "Forest",  # (buggy prompt)
        "19": "Univ-Uniform",  # (mahalai, Thai)
        "20": "School-Dress",
        "21": "Gum",
    }

    reality_lora_indices_mapping = {
        "02": "17",
        "03": "12",
        "05": "18",
        "09": "19",
        "12": "13",
        "13": "21",
        "15": "14",
        "16": "20",  # (buggy prompt)
        "19": "15",  # (mahalai, Thai)
        "20": "16",
        "21": "22",
    }

    # Define groups and prefixes
    groups = {
        "09_02_": ["09", "02"],                 # Library + JFC
        "12_02_": ["12", "02"],                 # Scarlett + JFC
        "12_21_02_": ["12", "21", "02"],        # Scarlett + Gum + JFC
        "15_02_09_": ["15", "02", "09"],        # Rock + JFC + Library
        "15_21_": ["15", "21"],                 # Rock + Gum
    }

    lora_loaded = set()
    # Process each group
    for group_prefix, lora_indices in groups.items():
        print(f"Processing group: {group_prefix}")
        
        # Map indices to LoRA names
        lora_names = [reality_lora_mapping.get(index) for index in lora_indices]
        if None in lora_names:
            print(f"Invalid LoRA index in group: {group_prefix}. Skipping.")
            continue

        # Load the corresponding LoRA files
        lora_files = [f"{int(reality_lora_indices_mapping[index]):01d}_{name}.safetensors" for index, name in zip(lora_indices, lora_names)]
        print(lora_files, args.lora_path, list(os.path.join(args.lora_path, lora_file) for lora_file in lora_files))
        if not all(os.path.exists(os.path.join(args.lora_path, lora_file)) for lora_file in lora_files):
            print(f"One or more LoRA files missing for group: {group_prefix}. Skipping.")
            continue

        # Load the LoRAs into the pipeline
        adapter_names = []
        for lora_file, lora_name in zip(lora_files, lora_names):
            print(f"Loading LoRA file: {lora_file}")
            if lora_name not in lora_loaded:
                pipeline.load_lora_weights(args.lora_path, weight_name=lora_file, adapter_name=lora_name)
                lora_loaded.add(lora_name)
            adapter_names.append(lora_name)

        # Set LoRA composition
        pipeline.set_adapters(adapter_names)

        # Find all prompt files for this group
        prompt_files = glob(f"gen_prompts_compose/{group_prefix}*.txt")
        for prompt_file in prompt_files:
            # Parse image index from the filename
            file_name = os.path.basename(prompt_file)
            if os.path.exists(f"gen_images_compose/{file_name.replace('.txt', '.png')}"):
                print(f"Skipping {file_name} because it already exists", adapter_names)
                continue
            components = file_name.split("_")
            image_index = components[-1].split(".")[0]

            # Load the prompt
            prompt = load_prompt(prompt_file)
            negative_prompt = (
                "extra heads, nsfw, deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, "
                "anime, text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, "
                "extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, "
                "bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, "
                "extra legs, fused fingers, too many fingers, long neck"
            )

            # Print information about the current image generation
            print(f"Generating image for prompt: \"{prompt}\"")
            print(f"Using LoRAs: {', '.join(adapter_names)}")

            # Generate the image
            image = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=args.height,
                width=args.width,
                num_inference_steps=args.denoise_steps,
                guidance_scale=args.cfg_scale,
                generator=args.generator,
                lora_composite=True
            ).images[0]

            # Save the image
            output_file = f"gen_images_compose/{file_name.replace('.txt', '.png')}"
            image.save(output_file)
            print(f"Saved image: {output_file}")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate images for LoRA compositions and prompts."
    )

    parser.add_argument('--lora_path', default='models/lora/reality',
                        help='Path to the directory containing LoRA files', type=str)
    parser.add_argument('--height', default=1024,
                        help='Height of the generated images', type=int)
    parser.add_argument('--width', default=768,
                        help='Width of the generated images', type=int)
    parser.add_argument('--denoise_steps', default=50,
                        help='Number of denoising steps', type=int)
    parser.add_argument('--cfg_scale', default=7,
                        help='Scale for classifier-free guidance', type=float)
    parser.add_argument('--seed', default=11,
                        help='Seed for generating images', type=int)

    args = parser.parse_args()
    args.generator = torch.manual_seed(args.seed)

    main(args)
