# Individual test file for download_loras.
# Take in the search keyword and num of loras to download, return the downloaded .safetensor, meta label text, representative image in lora-pool, lora-pool-meta, lora-pool-img folder respectively.

import os
import requests
from tqdm import tqdm  # Import tqdm for progress bar

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

# Example usage:
download_loras("basketball", 5)
