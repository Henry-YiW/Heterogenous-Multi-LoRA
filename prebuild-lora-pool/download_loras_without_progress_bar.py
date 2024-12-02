# download_loras without showing progress bar

import os
import requests

def download_loras(keyword, num):
    # Base URL for CivitAI API
    base_url = "https://civitai.com/api/v1/models"
    # Directories for LoRAs and metadata
    lora_dir = "lora-pool"
    meta_dir = "lora-pool-meta"
    os.makedirs(lora_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)
    
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
    # print(data)
    results = data.get("items", [])
    downloaded_count = 0
    
    for item in results:
        if downloaded_count >= num:
            break
        # print(item)
        
        # Extract metadata from the single modelVersions field
        model_version = item.get("modelVersions", [])[0]  # get to modelVersions attribute
        
        if not model_version or model_version.get("baseModel") != "SD 1.5":
            continue
        # print(model_version)

        file = model_version.get("files", [])[0]  # get to modelVersions.files attribute
        # print(file)
        if not file or file.get("metadata", {}).get("format") != "SafeTensor":
            continue
        print(file)
        
        # Extract LoRA metadata
        lora_id = item.get("id")
        name = item.get("name", "Unknown")
        description = item.get("description", "No description available")
        image_url = model_version.get("images", [{}])[0].get("url", "No image URL available")
        download_url = file.get("downloadUrl")
        
        # Download LoRA file
        lora_path = os.path.join(lora_dir, f"{lora_id}.safetensors")
        lora_response = requests.get(download_url)
        with open(lora_path, "wb") as lora_file:
            lora_file.write(lora_response.content)
        
        # Save metadata
        meta_path = os.path.join(meta_dir, f"{lora_id}.txt")
        with open(meta_path, "w") as meta_file:
            meta_file.write(f"id: {lora_id}\n\n")
            meta_file.write(f"name: {name}\n\n")
            meta_file.write(f"image_url: {image_url}\n\n")
            meta_file.write(f"description: {description}\n")
        
        downloaded_count += 1

    if downloaded_count < num:
        print(f"Only {downloaded_count} LoRAs found for keyword '{keyword}'.")

# Example usage:
download_loras("basketball", 5)
