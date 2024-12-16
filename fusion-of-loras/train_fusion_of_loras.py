
from glob import glob
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import os

import random
from datasets import load_dataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from datasets import load_from_disk
from PIL import Image

import json
from datasets import Dataset



def load_image(example):
  image = Image.open(example['image_path']).convert("RGB")
  example["image"] = image
  return example

image_transforms = Compose([
    Resize((512, 512)),  # Resize to model input resolution
    ToTensor(),          # Convert image to tensor and normalize to [0, 1]
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

def preprocess_image(example):
  image = example["image"]
  return example


def match_files(dir1, dir2):
    """
    Matches files from two directories by their filenames.
    Args:
        dir1: Path to the first directory.
        dir2: Path to the second directory.
    Returns:
        A list of tuples where each tuple contains matched filenames from dir1 and dir2.
    """
    # Get list of filenames in each directory
    files_dir1 = list(set(os.listdir(dir1)))
    files_dir2 = list(set(os.listdir(dir2)))
    random.shuffle(files_dir1)
    random.shuffle(files_dir2)

    dataset = []
    for file_name in files_dir1:
        file_name_prefix = file_name.split('.')[0]
        if file_name_prefix == '':
            continue
        for file_name_2 in files_dir2:
            file_name_prefix_2 = file_name_2.split('.')[0]
            if (file_name_prefix == file_name_prefix_2):
                with open(os.path.join(dir2, file_name_2), "r", encoding="utf-8") as file:
                  description = file.read()
                  dataset.append({ "name": file_name_prefix,
                                "pool-data-full-path": os.path.join(dir1, file_name),
                                "pool-data": file_name,
                                "description": description,
                                "pool-meta-full-path": os.path.join(dir2, file_name_2),
                                "pool-meta": file_name_2 })

    return dataset


def load_lora_list(dir_path_prefix):
  lora_pool_path = dir_path_prefix + '/LoRA-compos-data/lora-pool'
  lora_pool_meta_path = dir_path_prefix + '/LoRA-compos-data/lora-pool-meta'
  lora_dataset = match_files(lora_pool_path, lora_pool_meta_path)
  return lora_dataset

def load_lora_model_path(dataset, model_storage_dir):
  for lora_data in dataset:
    saving_dir = model_storage_dir + lora_data['name'];
    if os.path.exists(saving_dir):
      lora_data['model_path'] = saving_dir
  return dataset

def load_dataset(dataset_name, split="train"):
    filtered_dataset = load_from_disk("/content/drive/MyDrive/Graduate School/Stable Diffusion Finetuning/filtered_data")
    processed_dataset = filtered_dataset.map(load_image)
    lora_dataset = load_lora_list('/content/drive/MyDrive/Graduate School/Stable Diffusion Finetuning')
    lora_dataset = load_lora_model_path(lora_dataset, '/content/drive/MyDrive/Graduate School/Stable Diffusion Finetuning/stable-diffusion-1.5-with-lora/stable-diffusion-1.5-fused-with')
    dataset = load_dataset("laion/laion2B-en-aesthetic", split="train[:1%]", keep_in_memory=True)
    return dataset

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
    
def get_corresponding_lora_name(prompt_prefix, category = None):
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
    if prompt_prefix in anime_lora_mapping and (category == "anime" or category is None):
        return anime_lora_mapping[prompt_prefix]
    elif prompt_prefix in reality_lora_mapping and (category == "reality" or category is None):
        return reality_lora_mapping[prompt_prefix]
    else:
        raise ValueError(f"LoRA name {lora_name} not found in anime or reality mappings")
    
import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5Model

class EncoderDecoderForClassification(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.encoder_decoder = T5Model.from_pretrained(model_name)
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder_decoder.config.d_model, 2 * self.encoder_decoder.config.d_model), 
            nn.GELU(),
            nn.Linear(2 * self.encoder_decoder.config.d_model, 1),
            )  # Classification head
        self.softmax = nn.Softmax(dim=-1)  # Convert logits to probabilities

    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, class_token_indexes):
        # Get encoder and decoder outputs
        outputs = self.encoder_decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )

        # Use the last hidden state of the decoder (first token representation)
        decoder_hidden_state = outputs.last_hidden_state  # Shape: (batch_size, sequence_length, hidden_size)
        class_token_hidden_state = decoder_hidden_state[:, class_token_indexes, :]
        # Pass through classification head
        logits = self.classifier(class_token_hidden_state).squeeze(-1)  # Shape: (batch_size, num_classes)
        probabilities = self.softmax(logits)

        return logits, probabilities  # Logits for training, probabilities for inference
    
from transformers import Trainer
import torch
import torch.nn as nn
from typing import Any, Dict, Union

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        """
        A custom Trainer for fine-tuning the EncoderDecoderForClassification model.
        """
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Custom loss computation to handle EncoderDecoderForClassification.
        """
        # Extract inputs
        labels = inputs.pop("labels")  # Shape: (batch_size)
        # Forward pass
        outputs = model(**inputs)
        logits = outputs[0]  # First output is the logits

        # Compute cross-entropy loss
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):

        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        # Return loss only if required
        if prediction_loss_only:
            return loss, None, None

        # For evaluation, predictions are the rewards

        return loss, outputs, inputs['labels']

from transformers import TrainingArguments
def train_model(tokenized_dataset, tokenizer):

  # Initialize the model
  num_classes = 2  # Adjust based on your task
  model = EncoderDecoderForClassification("t5-small", num_classes)

  # Define training arguments
  training_args = TrainingArguments(
      output_dir="./results",
      evaluation_strategy="epoch",
      save_strategy="epoch",
      learning_rate=5e-5,
      per_device_train_batch_size=8,
      per_device_eval_batch_size=8,
      num_train_epochs=3,
      weight_decay=0.01,
      logging_dir="./logs",
      logging_steps=50,
      load_best_model_at_end=True,
      metric_for_best_model="accuracy",
  )

  # Define the Trainer
  trainer = CustomTrainer(
      model=model,
      args=training_args,
      train_dataset=tokenized_dataset,
      eval_dataset=tokenized_dataset,
      tokenizer=tokenizer,  # Needed for data formatting
  )

  # Train the model
  trainer.train()

def tokenize_function(examples, tokenizer, class_tokens, max_length=512):
  tokenized_examples = tokenizer(examples['prompt'] + , truncation=True, padding='max_length', max_length=512)
  class_token_input_ids = tokenizer(class_tokens, truncation=True, padding='max_length', max_length=512)
  return {**examples, **tokenized_examples}

def get_class_tokens(lora_indexes):
    num_classes = len(lora_indexes)
    special_tokens = [f"[CLASS_{i}]" for i in range(num_classes)]
    return special_tokens

def add_class_tokens_to_tokenizer(tokenizer, lora_indexes):
    special_tokens = get_class_tokens(lora_indexes)
    tokenizer.add_tokens(special_tokens)

def add_class_tokens(prompt_instance, lora_indexes, tokenizer):
    num_classes = len(lora_indexes)
    special_tokens = [f"[CLASS_{i}]" for i in range(num_classes)]
    tokenizer.add_tokens(special_tokens)

    # Example prompt with special tokens
    decoder_input = tokenizer("Classify this text", return_tensors="pt")
    class_tokens = tokenizer(special_tokens, add_special_tokens=False, return_tensors="pt")

    # Append special tokens to decoder input
    decoder_input["input_ids"] = torch.cat([decoder_input["input_ids"], class_tokens["input_ids"]], dim=1)
    class_tokens = [0] * len(lora_index)
    for temp_lora_name in prompt_instance['lora_name']:
        class_tokens[lora_index[temp_lora_name]] = 1
    return { **prompt_instance, "class_tokens": class_tokens }


def load_dataset(prompts_path, lora_path):
    dataset = glob(os.path.join(prompts_path, "*.txt"))
    prompt_list = []
    lora_set = set()
    for prompt_file in dataset:
       prompt = open(prompt_file, "r", encoding="utf-8").read()
       prompt_name = prompt_file.split('/')[-1].split('.')[0].strip()
       lora_prefixes = prompt_name.split('_')[:-1]
       lora_names = [get_corresponding_lora_name(prefix) for prefix in lora_prefixes]
       lora_set.update(lora_names)
       lora_paths = []
       for lora_name in lora_names:
          temp_lora_paths = glob(os.path.join(lora_path, f'*{lora_name}.safetensors'))
          lora_paths.append(temp_lora_paths[0] if len(temp_lora_paths) > 0 else None)
       prompt_list.append({ "prompt": prompt.strip('"'), "lora_name": lora_names, "lora_path": lora_paths })
    return prompt_list, len(lora_set), lora_set

def build_labels(prompt_instance, lora_index, temperature = 0.2):
    labels = [ 0 ] * len(lora_index)
    for temp_lora_name in prompt_instance['lora_name']:
        labels[lora_index[temp_lora_name]] = 1
    print('labels: ', labels)
    loss_fn = torch.nn.Softmax(dim=0)
    return { **prompt_instance, "labels": labels, "normalized_labels": loss_fn(torch.tensor(labels) / temperature).tolist() }

def main(lora_path, prompt_path, *args, **kwargs):
    prompt_list, num_loras, lora_set = load_dataset(prompt_path, lora_path)
    list_lora_set = list(lora_set)
    lora_index = { list_lora_set[i]: i for i in range(len(list_lora_set)) }

    # model_name = "t5-small"
    # num_classes = num_loras  # Replace with the actual number of classes
    # model = EncoderDecoderForClassification(model_name, num_classes)
    processed_prompt_list = [build_labels(prompt_instance, lora_index, temperature=0.1) for prompt_instance in prompt_list]
    dataset = Dataset.from_list(processed_prompt_list)
    output = { "prompt_list": processed_prompt_list, "num_loras": num_loras, "lora_set": list_lora_set }
    with open("prompt_list.json", "w") as f:
      json.dump(output, f)
    print(dataset[2])

    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    train_model(tokenized_dataset, tokenizer)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Train the fusion of LoRAs."
    )

    parser.add_argument('--lora_path', default='models/lora/reality',
                        help='Path to the directory containing LoRA files', type=str)
    parser.add_argument('--prompt_path', default='models/prompts/reality',
                        help='Path to the directory containing prompt files', type=str)
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
    main(**vars(args))

"python train_fusion_of_loras.py --prompt_path '/projects/bdpp/hyi1/stable-diffusion/Heterogenous-Multi-LoRA/Multi-LoRA-Composition/gen_prompts_compose' --lora_path '/projects/bdpp/hyi1/stable-diffusion/LoRA-compos-data/lora-pool/compose'"