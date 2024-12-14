import os
import torch
import argparse
from diffusers import DiffusionPipeline, AutoencoderKL, DPMSolverMultistepScheduler
from glob import glob

import os

def get_corresponding_prompt_prefix(lora_name, category = None):
    # Anime
    # 01:Arknights
    # 04: Nezuko
    # 06: Garreg
    # 07: Auroral
    # 08: Bamboolight
    # 10: Zero
    # 11: Handdrawn/line art
    # 14: MoXin
    # 17: Burger
    # 18: Goku
    # 22: Toast
    anime_lora_mapping = {
        "01": "Arknights",
        "04": "Nezuko",
        "06": "Garreg",
        "07": "Auroral",
        "08": "Bamboolight",
        "10": "Zero",
        "11": "Handdrawn", # line art
        "14": "MoXin",
        "17": "Burger",
        "18": "Goku",
        "22": "Toast",
    }
    reverse_anime_lora_mapping = {v: k for k, v in anime_lora_mapping.items()}

    # Reality
    # 02: JFC
    # 03: IU
    # 05: Bright
    # 09: Library
    # 12: Scarlett
    # 13: Umbrella
    # 15: Rock
    # 16: Forest (buggy prompt)
    # 19: Univ-Uniform (mahalai, Thai)
    # 20: School-Dress
    # 21: Gum
    reality_lora_mapping = {
        "02": "JFC",
        "03": "IU",
        "05": "Bright",
        "09": "Library",
        "12": "Scarlett",
        "13": "Umbrella",
        "15": "Rock",
        "16": "Forest", # (buggy prompt)
        "19": "Univ-Uniform", # (mahalai, Thai)
        "20": "School-Dress",
        "21": "Gum",
    }
    reverse_reality_lora_mapping = {v: k for k, v in reality_lora_mapping.items()}
    prefix = lora_name.split('.')[0].strip()
    matching_part = prefix.split('_')[1].strip()
    if matching_part in reverse_anime_lora_mapping and (category == "anime" or category is None):
        return reverse_anime_lora_mapping[matching_part]
    elif matching_part in reverse_reality_lora_mapping and (category == "reality" or category is None):
        return reverse_reality_lora_mapping[matching_part]
    else:
        raise ValueError(f"LoRA name {lora_name} not found in anime or reality mappings")


def get_prompt_for_generation(lora_name, image_index, category = 'reality'):
    """
    Fetch the prompt for a given LoRA and image index from pre-generated files.

    Args:
        lora_name (str): The name of the LoRA file (e.g., "01_LoRAName.safetensors").
        image_index (int): The index of the image to generate (e.g., 1, 2, ...).

    Returns:
        str: The prompt for the given LoRA and image index.
    """
    # Extract the first two characters of the lora_name to form the "xx" part
    lora_prefix = get_corresponding_prompt_prefix(lora_name, category)[:2]
    print('Prompt prefix: ', lora_prefix)
    # Construct the file name
    prompt_file = f"gen_prompts/{lora_prefix}_{image_index:02}.txt"

    # Check if the file exists
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    # Read the first line of the file
    with open(prompt_file, "r", encoding="utf-8") as file:
        prompt = file.readline().strip()

    return prompt



def main(args):
    # Create output folders if they don't exist
    os.makedirs("gen_images", exist_ok=True)
    os.makedirs("gen_prompts", exist_ok=True)

    # Load all LoRA files in the specified directory
    lora_files = glob(os.path.join(args.lora_path, "*.safetensors"))
    if not lora_files:
        print("No LoRA files found in the specified path!")
        return

    # Load the base pipeline
    # YL: When inferencing on Anime style image, remember to change the model name here
    model_name = 'SG161222/Realistic_Vision_V5.1_noVAE' if args.category == 'reality' else 'gsdf/Counterfeit-V2.5'
    pipeline = DiffusionPipeline.from_pretrained(
        model_name,
        custom_pipeline="./pipelines/sd1.5_0.26.3",
        use_safetensors=True
    ).to("cuda")

    # Set the VAE
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
    ).to("cuda")
    pipeline.vae = vae

    # Set the scheduler
    schedule_config = dict(pipeline.scheduler.config)
    schedule_config["algorithm_type"] = "dpmsolver++"
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(schedule_config)

    for lora_index, lora_path in enumerate(lora_files, start=1):
        print(f"Processing LoRA {lora_index}/{len(lora_files)}: {lora_path}")

        # Load the current LoRA
        lora_name = os.path.basename(lora_path)
        print('Lora Name: ', lora_name)
        lora_prefix = lora_name.split('.')[0].strip()
        pipeline.load_lora_weights(lora_path, adapter_name=f"lora_{lora_prefix}")

        # Generate K images for the current LoRA
        K = 50 # Adjust the number of images you want to generate for each lora here
        for image_index in range(1, K+1):
            # Get prompt from the Gen_prompts folder
            prompt = get_prompt_for_generation(lora_name, image_index, category=args.category)
            print(f"Prompt generated for {lora_name}: {prompt}")
            negative_prompt = "extra heads, nsfw, deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"

            image = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=args.height,
                width=args.width,
                num_inference_steps=args.denoise_steps,
                guidance_scale=args.cfg_scale,
                generator=args.generator,
                cross_attention_kwargs={"scale": args.lora_scale},
            ).images[0]

            # Save the image
            image_filename = f"gen_images/{lora_prefix:02}_{image_index:02}.png"
            image.save(image_filename)

            # Save the corresponding prompt
            prompt_filename = f"gen_prompts/{lora_prefix:02}_{image_index:02}.txt"
            # with open(prompt_filename, "w") as prompt_file:
            #     prompt_file.write(prompt)

            print(f"Saved: {image_filename} and {prompt_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate images for multiple LoRAs and save prompts."
    )

    # Arguments for composing LoRAs
    parser.add_argument('--lora_path', default='models/lora/reality',
                        help='Path to the directory containing LoRA files', type=str)
    parser.add_argument('--lora_scale', default=0.8,
                        help='Scale of each LoRA when generating images', type=float)

    # Arguments for generating images
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
    parser.add_argument('--category', default='reality',
                        help='Category of the LoRA', type=str)

    args = parser.parse_args()
    args.generator = torch.manual_seed(args.seed)

    main(args)
