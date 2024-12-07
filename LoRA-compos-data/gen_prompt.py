from ollama import chat
from ollama import ChatResponse
from glob import glob
import os
import argparse

# This function generate one prompt for the given lora_name, image_index using the ollama api
def generate_prompt(lora_name, image_index):
    # Metadata folder path
    meta_folder = "lora-pool-meta"
    
    # Find corresponding metadata file
    metadata_file = os.path.join(meta_folder, f"{os.path.splitext(lora_name)[0]}.txt")
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found for {lora_name} in {meta_folder}.")

    # Read metadata content
    with open(metadata_file, "r", encoding="utf-8") as file:
        metadata = file.read()

    # Parse metadata (extracting description, trigger words, etc.)
    # This is a basic parser; you can customize it if metadata structure varies.
    description = ""
    trigger_words = ""
    example_prompts = ""
    for line in metadata.split("\n"):
        if line.lower().startswith("description:"):
            description = line.split(":", 1)[1].strip()
        elif line.lower().startswith("trigger words:"):
            trigger_words = line.split(":", 1)[1].strip()
        elif line.lower().startswith("prompts example:"):
            example_prompts = line.split(":", 1)[1].strip()

    # Prepare input for Ollama
    ollama_input = f"""
    You are tasked with generating a creative and varied positive prompt for image generation related to the following LoRA. Use the metadata description and trigger words for inspiration, you only need to generate ONE single prompt for each response, but ensure the result is diverse for each lora across responses. The metadata for the LoRA is:

    Description: {description}
    Trigger Words: {trigger_words}
    Example Prompts: {example_prompts} if empty means there's no example


    Generate a single positive prompt based on this metadata and image index.
    """
    
    # Call Ollama to generate the prompt
    response: ChatResponse = chat(model='phi3', messages=[
        {'role': 'user', 'content': ollama_input,}
    ])
    
    # Extract the generated prompt
    generated_prompt = response.message.content.strip()
    return generated_prompt

def main(args):
    # Create output folders if they don't exist
    os.makedirs("gen_prompts", exist_ok=True)

    # Load all LoRA files in the specified directory
    lora_files = glob(os.path.join(args.lora_path, "*.safetensors"))
    if not lora_files:
        print("No LoRA files found in the specified path!")
        return

    for lora_index, lora_path in enumerate(lora_files, start=1):
        print(f"Processing LoRA {lora_index}/{len(lora_files)}: {lora_path}")

        # Load the current LoRA
        lora_name = os.path.basename(lora_path)

        # Generate K prompts for the current LoRA
        K = 50 # Adjust the number of images you want to generate for each lora here
        for image_index in range(1, K+1):
            prompt = generate_prompt(lora_name, image_index)
            # print(f"Prompt generated for {lora_name}: {prompt}")

            # Save the corresponding prompt
            prompt_filename = f"gen_prompts/{lora_index:02}_{image_index:02}.txt"
            with open(prompt_filename, "w") as prompt_file:
                prompt_file.write(prompt)

            print(f"Saved: {prompt_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate prompts for multiple LoRAs and save prompts."
    )

    # Arguments for composing LoRAs
    parser.add_argument('--lora_path', default='lora-pool',
                        help='Path to the directory containing LoRA files', type=str)

    args = parser.parse_args()

    main(args)
