import torch
import argparse
from diffusers import DiffusionPipeline, StableDiffusionPipeline, AutoencoderKL
from diffusers import DPMSolverMultistepScheduler
from callbacks import make_callback

def get_example_prompt():
    prompt_scarlett = "RAW photo, subject, 8k uhd, dslr, high quality, Fujifilm XT3, half-length portrait from knees up, scarlett, short red hair, blue eyes, school uniform, white shirt, red tie, blue pleated microskirt"
    prompt_rock = "RAW photo, subject, 8k uhd, dslr, high quality, muscular male, th3r0ck, no hair, serious look on his face, holding a transparent handhold umbrella, standing in front of a library bookshelf, lib bg"
    negative_prompt = "extra heads, nsfw, deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
    return prompt_rock, negative_prompt

def main(args):

    # set the prompts for image generation
    prompt, negative_prompt = get_example_prompt()

    # base model for the realistic style example
    model_name = 'SG161222/Realistic_Vision_V5.1_noVAE'

    # set base model
    pipeline = DiffusionPipeline.from_pretrained(
        model_name,
        custom_pipeline="./pipelines/sd1.5_0.26.3",
        use_safetensors=True
    ).to("cuda") 

    # set vae
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
    ).to("cuda")
    pipeline.vae = vae

    # set scheduler
    schedule_config = dict(pipeline.scheduler.config)
    schedule_config["algorithm_type"] = "dpmsolver++"
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(schedule_config)

    # initialize LoRAs
    # This example shows the composition of a character LoRA and a clothing LoRA
    # YL: Modify this part to use different loras
    
    #api to fetch lora
    # pipeline.load_lora_weights(args.lora_path, weight_name="scarlett_1.safetensors", adapter_name="character_1")
    # pipeline.load_lora_weights(args.lora_path, weight_name="scarlett_2.safetensors", adapter_name="character_2")
    # pipeline.load_lora_weights(args.lora_path, weight_name="uniform_1.safetensors", adapter_name="clothing_1")
    # pipeline.load_lora_weights(args.lora_path, weight_name="uniform_2.safetensors", adapter_name="clothing_2")


    pipeline.load_lora_weights(args.lora_path, weight_name="character_3.safetensors", adapter_name="Rock")
    pipeline.load_lora_weights(args.lora_path, weight_name="character_3.safetensors", adapter_name="Rock_cp")
    pipeline.load_lora_weights(args.lora_path, weight_name="background_1.safetensors", adapter_name="Library")
    pipeline.load_lora_weights(args.lora_path, weight_name="background_1.safetensors", adapter_name="Library_cp")
    pipeline.load_lora_weights(args.lora_path, weight_name="object_1.safetensors", adapter_name="Umbrella")
    pipeline.load_lora_weights(args.lora_path, weight_name="um_3.safetensors", adapter_name="Umbrella_2")

    # cur_loras = ["Rock", "Library", "Umbrella", "Umbrella_paper"]
    cur_loras = ["Rock", "Rock_cp", "Library", "Library_cp", "Umbrella", "Umbrella_2"]
    # cur_loras = ["character_1", "character_2", "clothing_1", "clothing_2"]

    # select the method for the composition
    # YL: The switch_callback here serves to switch between different LoRAs (adapters) at specific points during the denoising steps
    if args.method == "merge":
        pipeline.set_adapters(cur_loras)
        switch_callback = None
    elif args.method == "switch":
        pipeline.set_adapters([cur_loras[0]])
        switch_callback = make_callback(switch_step=args.switch_step, loras=cur_loras)
    elif args.method == "composite":
        pipeline.set_adapters(cur_loras)
        switch_callback = None
    # YL: Define custom method for switch_compose, default set of loras on each switch is two
    elif args.method == "switch_compose":
        def switch_compose_callback(pipeline, step_index, timestep, callback_kwargs):
            callback_outputs = {}
            # Determine the current set of active LoRAs based on the step
            current_index = (step_index // args.switch_step) % len(cur_loras)
            active_loras = cur_loras[current_index:current_index + 2]  # Switch between 2 LoRAs at a time
            pipeline.set_adapters(active_loras)  # Set the active LoRAs for composition
            return callback_outputs

        # Initialize the pipeline with the first set of active LoRAs
        pipeline.set_adapters(cur_loras[:2])  # Start with the first 2 LoRAs
        switch_callback = switch_compose_callback

    else:
        pipeline.set_adapters(cur_loras)
        switch_callback = None

    image = pipeline(
        prompt=prompt, 
        negative_prompt=negative_prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.denoise_steps,
        guidance_scale=args.cfg_scale,
        generator=args.generator,
        cross_attention_kwargs={"scale": args.lora_scale},
        callback_on_step_end=switch_callback,
        lora_composite=True if args.method in ["composite", "switch_compose"] else False
    ).images[0]

    image.save(args.save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Example code for multi-LoRA composition'
    )

    # Arguments for composing LoRAs
    parser.add_argument('--method', default='switch',
                        choices=['merge', 'switch', 'composite', 'switch_compose'],
                        help='methods for combining LoRAs', type=str)
    parser.add_argument('--save_path', default='example.png',
                        help='path to save the generated image', type=str)
    parser.add_argument('--lora_path', default='models/lora/reality',
                        help='path to store all LoRAs', type=str)
    parser.add_argument('--lora_scale', default=0.8,
                        help='scale of each LoRA when generating images', type=float)
    parser.add_argument('--switch_step', default=5,
                        help='number of steps to switch LoRA during denoising, applicable only in the switch method', type=int)

    # Arguments for generating images
    parser.add_argument('--height', default=1024,
                        help='height of the generated images', type=int)
    parser.add_argument('--width', default=768,
                        help='width of the generated images', type=int)
    parser.add_argument('--denoise_steps', default=50,
                        help='number of the denoising steps', type=int)
    parser.add_argument('--cfg_scale', default=7,
                        help='scale for classifier-free guidance', type=float)
    parser.add_argument('--seed', default=11,
                        help='seed for generating images', type=int)

    args = parser.parse_args()
    args.generator = torch.manual_seed(args.seed)

    main(args)