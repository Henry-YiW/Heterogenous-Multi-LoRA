from ollama import chat
from ollama import ChatResponse
from glob import glob
import os
import argparse


def generate_combined_prompt(lora_names, image_index):
    """
    Generate a prompt for a combination of LoRAs using the Ollama API.
    
    Args:
        lora_names (list): List of LoRA file names in the combination.
        image_index (int): The index of the image to generate (1, 2, ...).
    
    Returns:
        str: The generated prompt.
    """
    meta_folder = "lora-pool-meta"
    combined_metadata = []

    # Collect metadata for each LoRA in the combination
    for lora_name in lora_names:
        metadata_file = os.path.join(meta_folder, f"{os.path.splitext(lora_name)[0]}.txt")
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file not found for {lora_name} in {meta_folder}.")

        with open(metadata_file, "r", encoding="utf-8") as file:
            combined_metadata.append(file.read())

    # Prepare input for Ollama
    ollama_input = f"""
    You are tasked with generating a creative and varied positive prompt for image generation related to the following combination of LoRAs. Use the metadata descriptions and trigger words for inspiration. You only need to generate ONE single prompt for each response, but ensure the result is diverse for each combination and across responses. The metadata for the LoRAs is:
    """
    for i, metadata in enumerate(combined_metadata, start=1):
        ollama_input += f"\n--- Metadata for LoRA {i} ---\n{metadata}\n"

    ollama_input += f"\nImage Index: {image_index}\nGenerate a single positive prompt based on this metadata and image index."

    # Call Ollama to generate the prompt
    response: ChatResponse = chat(model='phi3', messages=[
        {'role': 'user', 'content': ollama_input}
    ])
    generated_prompt = response.message.content.strip()
    return generated_prompt


def main(args):
    # Create output folder if it doesn't exist
    os.makedirs("gen_prompts_compose", exist_ok=True)

    # Define the LoRA combinations
    lora_combinations = [
        ["12_Scarlett.safetensors", "02_JFC.safetensors"],
        ["15_Rock.safetensors", "21_Gum.safetensors"],
        ["09_Library.safetensors", "02_JFC.safetensors"],
        ["15_Rock.safetensors", "02_JFC.safetensors", "09_Library.safetensors"],
        ["12_Scarlett.safetensors", "21_Gum.safetensors", "02_JFC.safetensors"]
    ]

    # Generate prompts for each combination
    K = 50  # Number of prompts per combination
    for combo_index, lora_combination in enumerate(lora_combinations, start=1):
        print(f"Processing Combination {combo_index}/{len(lora_combinations)}: {', '.join(lora_combination)}")

        for image_index in range(1, K + 1):
            prompt = generate_combined_prompt(lora_combination, image_index)

            # Construct filename for saving the prompt
            combination_name = "_".join([name.split("_")[0] for name in lora_combination])  # e.g., "12_02" for Scarlett+JFC
            prompt_filename = f"gen_prompts_compose/{combination_name}_{image_index:02}.txt"

            # Save the prompt
            with open(prompt_filename, "w") as prompt_file:
                prompt_file.write(prompt)

            print(f"Saved: {prompt_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate prompts for combinations of LoRAs and save them."
    )

    parser.add_argument('--lora_path', default='lora-pool/Reality',
                        help='Path to the directory containing LoRA files', type=str)

    args = parser.parse_args()
    main(args)
