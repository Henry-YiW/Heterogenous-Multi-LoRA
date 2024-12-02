# Skeleton code for overall pool building. Can use different method in prompt parsing and different filter for download_loras
import os
import requests
from tqdm import tqdm  
from ollama import chat
from ollama import ChatResponse

# Placeholder for the download_loras function.
def download_loras(keyword, num):
    # Base URL for CivitAI API
    base_url = "https://civitai.com/api/v1/models"
    # Directories for LoRAs, metadata, and images
    lora_dir = "lora-pool"
    meta_dir = "lora-pool-meta"
    image_dir = "lora-pool-img"
    os.makedirs(lora_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    
    # Query parameters
    params = {
        "query": keyword,
        "types": "LORA",
        "sort": "Highest Rated",
    }
    
    # API call
    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        raise Exception(f"API call failed with status code {response.status_code}: {response.text}")
    
    # Parse response
    data = response.json()
    results = data.get("items", [])
    downloaded_count = 0
    
    for item in results:
        if downloaded_count >= num:
            break
        
        # Extract metadata from the single modelVersions field
        model_version = item.get("modelVersions", [])[0]  # Only one model version
        if not model_version or model_version.get("baseModel") != "SD 1.5":
            continue
        
        file = model_version.get("files", [])[0]  # Only one file
        if not file or file.get("metadata", {}).get("format") != "SafeTensor":
            continue
        
        # Extract LoRA metadata
        lora_id = item.get("id")
        name = item.get("name", "Unknown")
        description = item.get("description", "No description available")
        image_url = model_version.get("images", [{}])[0].get("url", "No image URL available")
        download_url = file.get("downloadUrl")
        
        # File naming convention
        file_name = f"{keyword}_{lora_id}"
        
        # Download LoRA file with progress bar
        lora_path = os.path.join(lora_dir, f"{file_name}.SafeTensors")
        print(f"Downloading LoRA [{lora_id}] {name}...")
        with requests.get(download_url, stream=True) as lora_response:
            lora_response.raise_for_status()
            total_size = int(lora_response.headers.get('content-length', 0))
            with open(lora_path, "wb") as lora_file, tqdm(
                total=total_size, unit="B", unit_scale=True, unit_divisor=1024, desc=f"{name} ({lora_id})"
            ) as progress:
                for chunk in lora_response.iter_content(chunk_size=8192):
                    lora_file.write(chunk)
                    progress.update(len(chunk))
        
        # Save metadata
        meta_path = os.path.join(meta_dir, f"{file_name}.txt")
        with open(meta_path, "w") as meta_file:
            meta_file.write(f"id: {lora_id}\n")
            meta_file.write(f"name: {name}\n")
            meta_file.write(f"description: {description}\n")
            meta_file.write(f"image_url: {image_url}\n")
        
        # Download image
        if image_url != "No image URL available":
            image_path = os.path.join(image_dir, f"{file_name}.jpg")
            print(f"Downloading image for LoRA [{lora_id}] {name}...")
            with requests.get(image_url, stream=True) as image_response:
                image_response.raise_for_status()
                with open(image_path, "wb") as img_file:
                    for chunk in image_response.iter_content(chunk_size=8192):
                        img_file.write(chunk)
        
        downloaded_count += 1

    if downloaded_count < num:
        print(f"Only {downloaded_count} LoRAs found for keyword '{keyword}'.")


# Placeholder for the prompt parsing function.
def parse_prompt(user_prompt, max_keywords=5):
    """
    Parses the input prompt into a list of keywords using the Ollama LLaMA API.

    Args:
        user_prompt (str): The text prompt to parse.
        max_keywords (int): Maximum number of keywords to extract.

    Returns:
        list: A list of keywords extracted from the prompt.
    """
    # Define the prompt text
    llama_prompt = f"""
You are tasked with parsing a user-provided prompt designed for text-to-image generation. The goal is to extract meaningful *keywords* that represent the essential elements of the input prompt, which will be used to retrieve LoRA models for the generation process.

---

### Guidelines:

1. **Keywords Definition:**
   - A *keyword* can be a single word or a multi-word phrase (e.g., "Golden Retriever" is valid, but splitting it into "Golden" and "Retriever" is incorrect).
   - Keywords should capture distinct and meaningful concepts from the input prompt, avoiding overly generic terms like "and," "of," "the."

2. **Keyword Purpose:**
   - These keywords will be used to search for relevant LoRA models, so they should be specific and directly related to the objects, subjects, styles, or themes mentioned in the input prompt.

3. **Maximum Keywords:**
   - Extract up to **{max_keywords}** keywords from the prompt. If fewer meaningful keywords are present, return only those.

4. **Input Format:**
   - You will be given a *user_prompt* (the text for image generation).

5. **Output Format:**
   - Provide a numbered list of keywords. Each keyword should be on its own line.

---

### Example:

**User Prompt:**  
"A majestic lion standing on a cliff at sunset, painted in the style of Van Gogh."

**Output Keywords:**  
1. Majestic lion  
2. Cliff  
3. Sunset  
4. Van Gogh style  

---

**User Prompt:**  
"{user_prompt}"  

**Output Keywords:**  
(Provide your response here based on the given user prompt, following the same format.)
    """

    # Send the request to the Ollama API
    try:
        response: ChatResponse = chat(model="phi3", messages=[
            {"role": "user", "content": llama_prompt}
        ])

        # Extract the generated text
        generated_text = response.message.content
        # print (generated_text)

        # Extract keywords from the generated text
        keywords = []
        for line in generated_text.splitlines():
            # print(line)
            if line.strip().startswith("-"): break
            if line.split(".")[0].isdigit():
                # Remove the number prefix and strip whitespace
                keyword = line.split('.', 1)[-1].strip()
                # print(keyword)
                if keyword:
                    keywords.append(keyword)
                    if len(keywords) >= max_keywords:
                        break

        return keywords

    except Exception as e:
        print("Error while connecting to the Ollama API:", str(e))
        return []


# Build the LoRA pool
def build_lora_pool(prompt, max_keywords=5, loras_per_keyword=4):
    """
    Main function to build a LoRA pool from a given text prompt.

    Args:
        prompt (str): The input text prompt.
        max_keywords (int): Maximum number of keywords to process from the prompt.
        loras_per_keyword (int): Number of top LoRAs to download per keyword.

    Returns:
        None
    """
    # Step 1: Parse prompt into keywords
    print(f"Parsing prompt: {prompt}")
    keywords = parse_prompt(prompt, max_keywords)
    print(f"Extracted keywords: {keywords}")

    # Step 2: Loop through keywords and download LoRAs
    for keyword in keywords:
        print(f"Processing keyword: {keyword}")
        try:
            # Use the black-box download_loras function
            download_loras(keyword, loras_per_keyword)
        except Exception as e:
            print(f"Error while processing keyword '{keyword}': {e}")

    print("LoRA pool creation complete.")


# Example usage:
if __name__ == "__main__":
    prompt = "Kamado Nezuko wearing gmuniform with a two-handed burger surrounded by bamboolight in traditional chinese ink painting style"
    build_lora_pool(prompt, max_keywords=5, loras_per_keyword=4)
