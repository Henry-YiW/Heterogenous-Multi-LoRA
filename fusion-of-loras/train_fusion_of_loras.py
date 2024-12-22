
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
from transformers import T5Tokenizer, T5Model, PreTrainedTokenizerFast

from safetensors.torch import load_model, save_model

from utils import tensor_insert, reverse_token_mapping



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

# def load_dataset(dataset_name, split="train"):
#     filtered_dataset = load_from_disk("/content/drive/MyDrive/Graduate School/Stable Diffusion Finetuning/filtered_data")
#     processed_dataset = filtered_dataset.map(load_image)
#     lora_dataset = load_lora_list('/content/drive/MyDrive/Graduate School/Stable Diffusion Finetuning')
#     lora_dataset = load_lora_model_path(lora_dataset, '/content/drive/MyDrive/Graduate School/Stable Diffusion Finetuning/stable-diffusion-1.5-with-lora/stable-diffusion-1.5-fused-with')
#     dataset = load_dataset("laion/laion2B-en-aesthetic", split="train[:1%]", keep_in_memory=True)
#     return dataset

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
from transformers import T5Tokenizer, T5Model, PreTrainedModel,PretrainedConfig

class EncoderDecoderForClassification(PreTrainedModel):
    def __init__(self, model_name, lora_set):
        super().__init__(PretrainedConfig())
        self.lora_set = lora_set

        self.register_buffer("lora_set_input_ids", self.lora_set['input_ids'])
        self.register_buffer("lora_set_attention_mask", self.lora_set['attention_mask'])

        self.encoder_decoder = T5Model.from_pretrained(model_name)
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder_decoder.config.d_model, 2 * self.encoder_decoder.config.d_model), 
            nn.GELU(),
            nn.Linear(2 * self.encoder_decoder.config.d_model, 1),
            )  # Classification head
        self.softmax = nn.Softmax(dim=-1)  # Convert logits to probabilities

    def forward(self, decoder_input_ids, decoder_attention_mask, class_token_indexes):
        # Get encoder and decoder outputs
        outputs = self.encoder_decoder(
            input_ids=self.lora_set_input_ids,
            attention_mask=self.lora_set_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )

        # Use the last hidden state of the decoder (first token representation)
        decoder_hidden_state = outputs.last_hidden_state  # Shape: (batch_size, sequence_length, hidden_size)
        # class_token_hidden_state = decoder_hidden_state[:, class_token_indexes, :]

        # Gather class token hidden states using torch.gather
        index_expanded = class_token_indexes.unsqueeze(-1).expand(-1, -1, decoder_hidden_state.shape[-1])  # Shape: (batch_size, num_classes, hidden_size)
        class_token_hidden_state = torch.gather(decoder_hidden_state, dim=1, index=index_expanded)  # Shape: (batch_size, num_classes, hidden_size)

        # Pass through classification head
        logits = self.classifier(class_token_hidden_state).squeeze(-1)  # Shape: (batch_size, num_classes)
        probabilities = self.softmax(logits)

        return logits, probabilities  # Logits for training, probabilities for inference
    
class VisualEncoderDecoderForClassification(nn.Module):
    def __init__(self, model_name, lora_set):
        super().__init__()
        self.lora_set = lora_set


        self.register_buffer("lora_set_images_input_embeddings", self.lora_set['images_input_embeddings']) # (lora_set_size, 10, 512)
        self.register_buffer("lora_set_input_ids", self.lora_set['lora_set_input_ids']) # (lora_set_size, 512, 1)
        self.register_buffer("lora_set_attention_mask", self.lora_set['lora_set_attention_mask']) # (lora_set_size, 512, 1)

        self.clip_to_t5_projector = nn.Sequential(
            nn.Linear(512, 956),
            nn.GELU(),
            nn.Linear(956, 512),
        )
        self.visual_encoder_decoder = T5Model.from_pretrained(model_name)
        self.encoder_decoder = T5Model.from_pretrained(model_name)
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder_decoder.config.d_model, 2 * self.encoder_decoder.config.d_model), 
            nn.GELU(),
            nn.Linear(2 * self.encoder_decoder.config.d_model, 1),
            )  # Classification head
        # self.softmax = nn.Softmax(dim=-1)  # Convert logits to probabilities
        self.sigmoid = nn.Sigmoid()


    def forward(self, decoder_input_ids, decoder_attention_mask, class_token_indexes):
        # Decoder processing (e.g., classification or generation)
        projected_embeddings = self.clip_to_t5_projector(self.lora_set_images_input_embeddings)
      
        visual_encoder_decoder_outputs = self.visual_encoder_decoder(
            input_ids=self.lora_set_input_ids,
            attention_mask=self.lora_set_attention_mask,
            decoder_inputs_embeds=projected_embeddings,
        )
        reshaped_visual_encoder_decoder_outputs = visual_encoder_decoder_outputs[0].reshape(1,-1, 512)
        reshaped_visual_encoder_decoder_outputs = reshaped_visual_encoder_decoder_outputs.expand(decoder_input_ids.shape[0], -1, -1)
        # Get encoder and decoder outputs
        outputs = self.encoder_decoder(
            encoder_outputs=[reshaped_visual_encoder_decoder_outputs],
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )

        # Use the last hidden state of the decoder (first token representation)
        decoder_hidden_state = outputs.last_hidden_state  # Shape: (batch_size, sequence_length, hidden_size)
        # class_token_hidden_state = decoder_hidden_state[:, class_token_indexes, :]

        # Gather class token hidden states using torch.gather
        index_expanded = class_token_indexes.unsqueeze(-1).expand(-1, -1, decoder_hidden_state.shape[-1])  # Shape: (batch_size, num_classes, hidden_size)
        class_token_hidden_state = torch.gather(decoder_hidden_state, dim=1, index=index_expanded)  # Shape: (batch_size, num_classes, hidden_size)

        # Pass through classification head
        logits = self.classifier(class_token_hidden_state).squeeze(-1)  # Shape: (batch_size, num_classes)
        probabilities = self.sigmoid(logits)

        return logits, probabilities  # Logits for training, probabilities for inference
    

class MultiModalEncoderDecoderForClassification(nn.Module):
    def __init__(self, model_name, lora_set):
        super().__init__()
        self.lora_set = lora_set

        self.encoder_decoder = T5Model.from_pretrained(model_name)
        self.clip_to_t5_projector = nn.Sequential(
            nn.Linear(512, 956),
            nn.GELU(),
            nn.Linear(956, 512),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder_decoder.config.d_model, 2 * self.encoder_decoder.config.d_model), 
            nn.GELU(),
            nn.Linear(2 * self.encoder_decoder.config.d_model, 1),
            )  # Classification head
        # self.softmax = nn.Softmax(dim=-1)  # Convert logits to probabilities
        self.sigmoid = nn.Sigmoid()

        self.register_buffer("lora_set_class_delimiter_token_indexes", self.lora_set['class_delimiter_token_indexes']) # (lora_set_size, 1)
        self.register_buffer("lora_set_images_input_embeddings", self.lora_set['images_input_embeddings']) # (lora_set_size, 10, 512)
        self.register_buffer("lora_set_input_ids", self.lora_set['input_ids']) # (lora_set_size_ensemble_string, 1024, 1)
        self.register_buffer("lora_set_attention_mask", self.lora_set['attention_mask']) # (lora_set_size_ensemble_string, 1024, 1)
        

    def tensor_insert(self,tensor, value, index, dim):
        # Ensure value has the correct number of dimensions
        if tensor.dim() != value.dim():
            raise ValueError(f"Value must have the same number of dimensions as the input tensor. tensor dim: {tensor.dim()}, value dim: {value.dim()}")

        # Split the tensor along the specified dimension
        if index > tensor.size(dim):
            raise IndexError(f"Index {index} is out of bounds for dimension {dim} with size {tensor.size(dim)}")

        # Slice and concatenate
        before = tensor.narrow(dim, 0, index)  # Select everything before the index
        after = tensor.narrow(dim, index, tensor.size(dim) - index)  # Select everything after the index
        inserted_indexes = torch.arange(index, index + value.size(dim))
        return torch.cat([before, value, after], dim=dim), inserted_indexes
        

    def get_text_embeddings(self, input_ids):
        inputs_embeds = self.encoder_decoder.shared(input_ids)
        return inputs_embeds
    
    def insert_image_embeddings_into_text_embeddings(self, text_embeddings, attention_mask, class_delimiter_token_indexes, image_embeddings_list):
        inserted_embeddings = text_embeddings
        inserted_attention_mask = attention_mask
        inserted_indexes_list = []
        for index,class_delimiter_token_index in enumerate(class_delimiter_token_indexes):
            to_insert_embeddings = image_embeddings_list[index].unsqueeze(0).detach()
            inserted_embeddings, inserted_indexes = self.tensor_insert(inserted_embeddings, to_insert_embeddings, class_delimiter_token_index, 1)
            inserted_attention_mask, _ = self.tensor_insert(inserted_attention_mask, torch.ones(to_insert_embeddings.shape[0:2]).to(inserted_attention_mask.device), class_delimiter_token_index, 1)
            inserted_indexes_list.append(inserted_indexes)
        return inserted_embeddings, inserted_attention_mask, inserted_indexes_list
    
    def insert_image_embeddings_into_text_embeddings_ensemble(self, lora_set_input_ids, attention_mask, lora_set_class_delimiter_token_indexes, image_embeddings_list):
        lora_set_input_embeddings = self.get_text_embeddings(lora_set_input_ids)
        inserted_embeddings, inserted_attention_mask, inserted_indexes_list = self.insert_image_embeddings_into_text_embeddings(lora_set_input_embeddings, attention_mask, lora_set_class_delimiter_token_indexes, image_embeddings_list)
        return inserted_embeddings, inserted_attention_mask, inserted_indexes_list

    def forward(self, decoder_input_ids, decoder_attention_mask, class_token_indexes):

        projected_embeddings = self.clip_to_t5_projector(self.lora_set_images_input_embeddings)
        inserted_embeddings, inserted_attention_mask, inserted_indexes_list = self.insert_image_embeddings_into_text_embeddings_ensemble(self.lora_set_input_ids, self.lora_set_attention_mask, self.lora_set_class_delimiter_token_indexes, projected_embeddings)


        # print(inserted_embeddings.shape)
        # print(inserted_attention_mask.shape)
        # print(decoder_input_ids.shape)
        # print(decoder_attention_mask.shape)
        inserted_embeddings = inserted_embeddings.expand(decoder_input_ids.shape[0], -1, -1)
        inserted_attention_mask = inserted_attention_mask.expand(decoder_input_ids.shape[0], -1)
        outputs = self.encoder_decoder(
            inputs_embeds=inserted_embeddings,
            attention_mask=inserted_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )

        # Use the last hidden state of the decoder (first token representation)
        decoder_hidden_state = outputs.last_hidden_state  # Shape: (batch_size, sequence_length, hidden_size)
        # class_token_hidden_state = decoder_hidden_state[:, class_token_indexes, :]

        # Gather class token hidden states using torch.gather
        index_expanded = class_token_indexes.unsqueeze(-1).expand(-1, -1, decoder_hidden_state.shape[-1])  # Shape: (batch_size, num_classes, hidden_size)
        class_token_hidden_state = torch.gather(decoder_hidden_state, dim=1, index=index_expanded)  # Shape: (batch_size, num_classes, hidden_size)
        
        # Pass through classification head
        logits = self.classifier(class_token_hidden_state).squeeze(-1)  # Shape: (batch_size, num_classes)
        probabilities = self.sigmoid(logits)

        return logits, probabilities  # Logits for training, probabilities for inference


class MultiModalLoRASetEmbedding():
    def __init__(self, lora_set, embeddings):
        self.lora_set = lora_set
        self.embeddings = embeddings

    def reverse_token_mapping(text_indexes, offset_mappings):
        matched_mapping_indexes = []
        for index, mapping in enumerate(offset_mappings):
            if mapping[0] <= text_indexes[0] and mapping[1] >= text_indexes[1]:
                matched_mapping_indexes.append(index)
        return matched_mapping_indexes

    def build_lora_ensemble(lora_set, tokenizer):
        # text = "The following are the descriptions of the LoRA: \n\n"
        text = ""
        descriptions = []
        class_delimiter_indexes = []
        for lora_index, lora_name in enumerate(lora_set.keys()):
            class_description = f"{{[CLASS_{lora_index}]: {lora_set[lora_name]['lora_meta']}.}}"
            # class_delimiter_index = class_description.rfind('}')
            class_delimiter_index = len(class_description) - 1
            class_delimiter_indexes.append(class_delimiter_index)
            descriptions.append(class_description)
        seperator = ';\n\n'
        text += seperator.join(descriptions)
        text += '.'
        for i in range(len(class_delimiter_indexes)):
            class_delimiter_indexes[i] = class_delimiter_indexes[i] + (class_delimiter_indexes[i - 1] + len(seperator) + 1 if i > 0 else 0)
        tokenized_text = tokenizer(text, truncation=True, padding='max_length', max_length=512, return_tensors='pt', return_offsets_mapping=True)
        offset_mappings = tokenized_text.offset_mapping

        class_delimiter_token_indexes = []
        for class_delimiter_index in class_delimiter_indexes:
            class_delimiter_token_indexes.append(reverse_token_mapping([class_delimiter_index, class_delimiter_index + 1], offset_mappings[0])[0])

        return tokenized_text, class_delimiter_token_indexes
    
    def tensor_insert(tensor, value, index, dim):
        # Ensure value has the correct number of dimensions
        if tensor.dim() != value.dim():
            raise ValueError("Value must have the same number of dimensions as the input tensor.")

        # Split the tensor along the specified dimension
        if index > tensor.size(dim):
            raise IndexError(f"Index {index} is out of bounds for dimension {dim} with size {tensor.size(dim)}")

        # Slice and concatenate
        before = tensor.narrow(dim, 0, index)  # Select everything before the index
        after = tensor.narrow(dim, index, tensor.size(dim) - index)  # Select everything after the index
        return torch.cat([before, value, after], dim=dim)
        

    def get_text_embeddings(self, input_ids):
        inputs_embeds = self.encoder_decoder.shared(input_ids)
        return inputs_embeds
    
    def insert_image_embeddings_into_text_embeddings(self, text_embeddings, class_delimiter_token_indexes, image_embeddings_list):
        inserted_embeddings = text_embeddings
        for index,class_delimiter_token_index in enumerate(class_delimiter_token_indexes):
            to_insert_embeddings = image_embeddings_list[index]
            inserted_embeddings = tensor_insert(inserted_embeddings, to_insert_embeddings, class_delimiter_token_index, 1)
        return inserted_embeddings

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

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch = None):
        """
        Custom loss computation to handle EncoderDecoderForClassification.
        """
        # Extract inputs
        labels = inputs["labels"].float()  # Shape: (batch_size)
        # Forward pass
        outputs = model(decoder_input_ids=inputs['input_ids'], decoder_attention_mask=inputs['attention_mask'], class_token_indexes=inputs['class_token_index_group'])
        logits = outputs[0]  # First output is the logits

        # Compute cross-entropy loss
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):

        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        # Return loss only if required
        if prediction_loss_only:
            return loss, None, None

        # For evaluation, predictions are the rewards

        return loss, outputs[1], inputs['labels']

from transformers import TrainingArguments
def train_model(model, run_name, tokenized_dataset, eval_dataset, tokenizer, data_collator, finetune=False):
    output_dir = f"./results/{run_name}"
    os.makedirs(output_dir, exist_ok=True)
    # Define training arguments
    training_args = TrainingArguments(
        run_name=run_name,
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=15 if not finetune else 1,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        load_best_model_at_end=True,
        save_total_limit=5,
        metric_for_best_model="eval_loss",
        remove_unused_columns=False,
        report_to="wandb",
        save_safetensors=False
    )

    # Define the Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,  # Needed for data formatting
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()

    save_model(model, os.path.join(output_dir, "model.safetensors"))
    # Instead of save_file(model.state_dict(), "model.safetensors")

    # load_model(model, os.path.join(output_dir, "model.safetensors"))
    # Instead of model.load_state_dict(load_file("model.safetensors"))
#   model.save_pretrained("./output_dir", safe_serialization=True)
#   tokenizer.save_pretrained("./output_dir")


def tokenize_function(examples, tokenizer, class_tokens, class_token_input_ids, max_length=512):
    prompts = examples['prompt']
    prompts_with_class_tokens = [''.join(class_tokens) + prompt for prompt in prompts]
    # print(prompts_with_class_tokens)
    tokenized_prompts = tokenizer(prompts_with_class_tokens, truncation=True, padding='max_length', max_length=max_length)
    class_token_index_group = []
    for input_ids, attention_mask in zip(tokenized_prompts['input_ids'], tokenized_prompts['attention_mask']):
        class_token_indexes = []
        for token_input_id in class_token_input_ids:
            if token_input_id in input_ids:
                index = input_ids.index(token_input_id)
                class_token_indexes.append(index)
            attention_mask[index] = 0
        class_token_index_group.append(class_token_indexes)

    return {
        "input_ids": tokenized_prompts['input_ids'],
        "attention_mask": tokenized_prompts['attention_mask'],
        "class_token_index_group": class_token_index_group,
        # "labels": examples['labels'],
        # "normalized_labels": examples['normalized_labels']
    }

def get_class_tokens(lora_indexes):
    num_classes = len(lora_indexes)
    special_tokens = [f"[CLASS_{i}]" for i in range(num_classes)]
    return special_tokens

def add_class_tokens_to_tokenizer(tokenizer, lora_indexes):
    class_tokens = get_class_tokens(lora_indexes)
    tokenizer.add_tokens(class_tokens)
    return tokenizer, class_tokens

def get_class_token_input_ids(class_tokens, tokenizer):
    return tokenizer.encode(''.join(class_tokens))[:len(class_tokens)]

from PIL import Image
import os

picked_images = {
    "Nezuko": [
        '1_Nezuko_02.png',
        '1_Nezuko_04.png',
        '1_Nezuko_05.png',
        '1_Nezuko_11.png',
        '1_Nezuko_07.png',
        '1_Nezuko_08.png',
        '1_Nezuko_15.png',
        '1_Nezuko_21.png',
        '1_Nezuko_36.png',
        '1_Nezuko_28.png',
        '1_Nezuko_49.png',
        '1_Nezuko_34.png',
        '1_Nezuko_37.png',
        '1_Nezuko_48.png',
        '1_Nezuko_46.png',
    ],
    "Arknights": [
        '2_Arknights_11.png',
        '2_Arknights_02.png',
        '2_Arknights_03.png',
        '2_Arknights_06.png',
        '2_Arknights_12.png',
        '2_Arknights_13.png',
        '2_Arknights_14.png',
        '2_Arknights_45.png',
        '2_Arknights_46.png',
        '2_Arknights_49.png',
        '2_Arknights_42.png',
        '2_Arknights_40.png',
        '2_Arknights_38.png',
        '2_Arknights_39.png',
        '2_Arknights_35.png',
    ],
    "Goku": [
        '3_Goku_01.png',
        '3_Goku_02.png',
        '3_Goku_03.png',
        '3_Goku_04.png',
        '3_Goku_05.png',
        '3_Goku_06.png',
        '3_ Goku_29.png',
        '3_ Goku_32.png',
        '3_ Goku_36.png',
        '3_ Goku_37.png',
        '3_ Goku_41.png',
        '3_ Goku_44.png',
        '3_ Goku_43.png',
        '3_ Goku_45.png.png',
        '3_ Goku_46.png',
    ]
}

def get_image_paths(lora_names, image_path, num_images):
    image_paths = []
    for lora_name in lora_names:
        matched_images = glob(os.path.join(image_path, f'*{lora_name}*.png'))
        if len(matched_images) > 0:
            selected_images = {
                "lora_name": lora_name,
                "image_paths": random.sample(matched_images, num_images)
            }
            image_paths.append(selected_images)
    return image_paths

def generate_clip_embeddings(image_paths, model, processor, batch_size=32, device="cuda"):
    embeddings = []
    model.to(device)
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        
        # Load and preprocess images
        images = [Image.open(image_path).convert("RGB") for image_path in batch_paths]
        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)

        # Generate image embeddings
        with torch.no_grad():
            image_embeddings = model.get_image_features(**inputs)
            image_embeddings = image_embeddings / image_embeddings.norm(p=2, dim=-1, keepdim=True)  # Normalize
            embeddings.append(image_embeddings.cpu())
    
    return torch.cat(embeddings, dim=0)  # Combine all embeddings

def save_embeddings(embeddings, image_paths, output_path):
    torch.save({ "embeddings": embeddings, "image_paths": image_paths }, output_path)


def load_dataset_from_prompts(prompts_path, loras_path, lora_metas_path, limiting_lora_names):
    dataset = glob(os.path.join(prompts_path, "*.txt"))
    prompt_list = []
    lora_set = {}
    for prompt_file in dataset:
       prompt = open(prompt_file, "r", encoding="utf-8").read()
       prompt_name = prompt_file.split('/')[-1].split('.')[0].strip()
       lora_prefixes = prompt_name.split('_')[:-1]
       lora_names = [get_corresponding_lora_name(prefix) for prefix in lora_prefixes]
       lora_paths = []
       lora_meta_paths = []
       if limiting_lora_names is not None:
          if not all(lora_name in limiting_lora_names for lora_name in lora_names):
            continue
       for lora_name in lora_names:
          if limiting_lora_names is not None and lora_name not in limiting_lora_names:
            continue
          temp_lora_meta_paths = glob(os.path.join(lora_metas_path, f'*{lora_name}.txt'))
          lora_meta_path = temp_lora_meta_paths[0] if len(temp_lora_meta_paths) > 0 else None
          lora_meta_paths.append(lora_meta_path)

          temp_lora_paths = glob(os.path.join(loras_path, f'*{lora_name}.safetensors'))
          lora_path = temp_lora_paths[0] if len(temp_lora_paths) > 0 else None
          lora_paths.append(lora_path)
          
          if lora_name not in lora_set:
            lora_meta = open(lora_meta_path, "r", encoding="utf-8").read() if lora_meta_path is not None else None
            lora_set[lora_name] = { "lora_meta": lora_meta, "lora_path": lora_path, "lora_meta_path": lora_meta_path }

       prompt_list.append({ "prompt": prompt.strip('"'), "lora_name": lora_names, "lora_path": lora_paths, "lora_meta_path": lora_meta_paths })
    return prompt_list, len(lora_set.keys()), lora_set

def build_labels(prompt_instance, lora_index, temperature = 0.2):
    labels = [ 0 ] * len(lora_index)
    for temp_lora_name in prompt_instance['lora_name']:
        labels[lora_index[temp_lora_name]] = 1
    # print('labels: ', labels)
    loss_fn = torch.nn.Softmax(dim=0)
    return { **prompt_instance, "labels": labels, "normalized_labels": loss_fn(torch.tensor(labels) / temperature).tolist() }

from dataclasses import dataclass
from typing import Any, Dict, List
from transformers import DataCollatorWithPadding


@dataclass
class CustomDataCollatorWithPadding:
    tokenizer: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        processed_features = [
            {"input_ids": f["input_ids"], "attention_mask": f["attention_mask"], "labels": f["labels"]}
            for f in features
        ]
        class_token_index_group = [ f["class_token_index_group"] for f in features ]
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        batch = data_collator(processed_features)
        class_token_index_group_with_tensor = torch.tensor(class_token_index_group)
        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
            "class_token_index_group": class_token_index_group_with_tensor,
        }


def build_lora_ensemble(lora_set, lora_indexes, tokenizer):
    # text = "The following are the descriptions of the LoRA: \n\n"
    text = ""
    descriptions = []
    class_delimiter_indexes = []
    image_embeddings = []
    lora_names = sorted(lora_set.keys(), key=lambda x: lora_indexes[x])
    print(lora_names)
    for lora_index, lora_name in enumerate(lora_names):
        class_description = f"{{[CLASS_{lora_index}]: {lora_set[lora_name]['lora_meta']}.}}"
        # class_delimiter_index = class_description.rfind('}')
        class_delimiter_index = len(class_description) - 1
        class_delimiter_indexes.append(class_delimiter_index)
        descriptions.append(class_description)
        image_embeddings.append(lora_set[lora_name]['image_embeddings'])
    seperator = ';\n\n'
    text += seperator.join(descriptions)
    text += '.'
    # print(text)
    for i in range(len(class_delimiter_indexes)):
        class_delimiter_indexes[i] = class_delimiter_indexes[i] + (class_delimiter_indexes[i - 1] + len(seperator) + 1 if i > 0 else 0)
    tokenized_text = tokenizer(text, truncation=True, padding='max_length', max_length=1536, return_tensors='pt', return_offsets_mapping=True)
    offset_mappings = tokenized_text.offset_mapping
    class_delimiter_token_indexes = []
    for class_delimiter_index in class_delimiter_indexes:
        class_delimiter_token_indexes.append(reverse_token_mapping([class_delimiter_index, class_delimiter_index + 1], offset_mappings[0])[0])

    return tokenized_text, class_delimiter_token_indexes, torch.stack(image_embeddings)

# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     predictions = predictions.argmax(axis=-1)  # Get predicted class index
#     return metric.compute(predictions=predictions, references=labels)

import numpy as np
def compute_metrics(eval_pred):
    """
    Computes accuracy for predictions from the model.

    Args:
        eval_pred: Tuple containing predictions and labels.
    
    Returns:
        A dictionary with the computed accuracy metric.
    """
    probabilities, labels = eval_pred.predictions, eval_pred.label_ids
    # Step 2: Threshold probabilities to get binary predictions
    threshold = 0.6
    raw_predictions = (probabilities > threshold)
    predictions = raw_predictions.astype(int)
    # print('probabilities: ', probabilities)
    # print('labels: ', labels)
    tp = (predictions & labels).sum(axis=1).astype(float)  # Shape: (batch_size,)
    fp = (predictions & ~labels).sum(axis=1).astype(float)  # Shape: (batch_size,)
    fn = (~predictions & labels).sum(axis=1).astype(float)  # Shape: (batch_size,)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    precision_mean = precision.mean().item()
    recall_mean = recall.mean().item()
    f1_mean = f1.mean().item()

    return {"precision": precision_mean, "recall": recall_mean, "f1": f1_mean}

def main(lora_path, prompt_path, lora_meta_path, test, *args, **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    prompt_list, num_loras, lora_set = load_dataset_from_prompts(prompt_path, lora_path, lora_meta_path)
    list_lora_set = list(lora_set.keys())
    lora_index = { list_lora_set[i]: i for i in range(len(list_lora_set)) }

    # model_name = "t5-small"
    # num_classes = num_loras  # Replace with the actual number of classes
    # model = EncoderDecoderForClassification(model_name, num_classes)
    tokenizer = T5Tokenizer.from_pretrained("t5-small");
    tokenizer, class_tokens = add_class_tokens_to_tokenizer(tokenizer, list_lora_set)
    processed_prompt_list = [build_labels(prompt_instance, lora_index, temperature=0.1) for prompt_instance in prompt_list]
    dataset = Dataset.from_list(processed_prompt_list)
    # output = { "prompt_list": processed_prompt_list, "num_loras": num_loras, "lora_set": list_lora_set }
    # with open("prompt_list.json", "w") as f:
    #   json.dump(output, f)
    # print(dataset[2])
    class_token_input_ids = get_class_token_input_ids(class_tokens, tokenizer)
    tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer, class_tokens, class_token_input_ids), batched=True)
    tokenized_text, class_delimiter_token_indexes = build_lora_ensemble(lora_set, tokenizer)
    train_test_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
    tokenized_text = build_lora_ensemble(lora_set, tokenizer)
    lora_set['input_ids'] = tokenized_text['input_ids']
    lora_set['attention_mask'] = tokenized_text['attention_mask']
    lora_set['class_delimiter_token_indexes'] = torch.tensor(class_delimiter_token_indexes)
    if not test:
        print("TRAIN MODE")
        model = EncoderDecoderForClassification("t5-small", lora_set).to(device)

        data_collator = CustomDataCollatorWithPadding(tokenizer=tokenizer) 
        train_model(model, train_test_dataset['train'], train_test_dataset['test'], tokenizer, data_collator)
    else:
        print("TEST MODE")
        model = EncoderDecoderForClassification.from_pretrained("./results/checkpoint-15")
        tokenizer = T5Tokenizer.from_pretrained("t5-small")

def get_lora_embeddings_with_text_descriptions(image_path, test_lora_names, one_shot_image_path, including_test_lora_names=False, type="anime", lora_sample_size=10, image_sample_size=10):
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
    one_shot_lora_set = None
    training_lora_set = None
    if including_test_lora_names == True:
        lora_sample_size -= len(test_lora_names)
    if type == "anime" or type == "reality" or type == "all":
        lora_mapping = anime_lora_mapping if type == "anime" else reality_lora_mapping
        lora_mapping = { **anime_lora_mapping, **reality_lora_mapping } if type == "all" else lora_mapping
        lora_names = [ lora_name for lora_name in lora_mapping.values() if lora_name not in test_lora_names ]
        lora_names = random.sample(lora_names, lora_sample_size)
        if including_test_lora_names == True:
            lora_names = lora_names + test_lora_names
        image_paths = get_image_paths(lora_names, image_path, image_sample_size)
        training_lora_image_embeddings = torch.load(f'{image_path}/image_embeddings.pth')
        training_lora_set = {}
        for image_path in image_paths:
            matched_indexes = []
            for image_item_path in image_path['image_paths']:
                matched_indexes.append(training_lora_image_embeddings['image_paths'].index(image_item_path))
            selected_training_lora_image_embeddings = training_lora_image_embeddings['embeddings'][matched_indexes]
            training_lora_set[image_path['lora_name']] = { 'image_embeddings': selected_training_lora_image_embeddings, 'image_paths': image_path['image_paths'] }

    one_shot_image_paths = get_image_paths(test_lora_names, one_shot_image_path, 1)
    one_shot_image_embeddings = torch.load(f'{one_shot_image_path}/image_embeddings.pth')
    one_shot_lora_set = {}
    for one_shot_image_path in one_shot_image_paths:
        matched_indexes = []
        for one_shot_image_item_path in one_shot_image_path['image_paths']:
            matched_indexes.append(one_shot_image_embeddings['image_paths'].index(one_shot_image_item_path))
        selected_one_shot_image_embeddings = one_shot_image_embeddings['embeddings'][matched_indexes]
        one_shot_lora_set[one_shot_image_path['lora_name']] = { 'image_embeddings': selected_one_shot_image_embeddings, 'image_paths': one_shot_image_path['image_paths'] }

    return training_lora_set, one_shot_lora_set

from transformers import CLIPModel, CLIPProcessor
def execute_embedding_generation(output_path, training_image_path):
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    image_paths = glob(os.path.join(training_image_path, "*.png"))
    print('image_paths: ', image_paths)
    embeddings = generate_clip_embeddings(image_paths, clip_model, clip_processor, batch_size=32, device="cuda")
    save_embeddings(embeddings, image_paths, output_path)
    embeddings = torch.load(output_path)
    print(embeddings)
    return embeddings

def build_lora_set(lora_path, prompt_path, lora_meta_path, training_lora_image_path, one_shot_test_image_path, training_lora_prompt_path, saving_path, including_test_lora_names=False, type="all"):
    prompt_list, num_loras, one_shot_lora_set = load_dataset_from_prompts(prompt_path, lora_path, lora_meta_path, None)
    training_lora_set, training_one_shot_lora_set = get_lora_embeddings_with_text_descriptions(training_lora_image_path, list(one_shot_lora_set.keys()), one_shot_test_image_path, including_test_lora_names=including_test_lora_names, type=type)
    prompt_list, num_loras, lora_set = load_dataset_from_prompts(training_lora_prompt_path, lora_path, lora_meta_path, training_lora_set.keys())
    # print('prompt_list: ', prompt_list)
    # print('num_loras: ', num_loras)
    # print('lora_set: ', lora_set)
    # print('training_lora_set: ', training_lora_set)
    # print('one_shot_lora_set: ', one_shot_lora_set)
    for training_lora_name in training_lora_set:
        lora_set[training_lora_name]['image_embeddings'] = training_lora_set[training_lora_name]['image_embeddings']
        lora_set[training_lora_name]['image_paths'] = training_lora_set[training_lora_name]['image_paths']
    if training_one_shot_lora_set is not None:
        for training_one_shot_lora_name in training_one_shot_lora_set:
            one_shot_lora_set[training_one_shot_lora_name]['image_embeddings'] = training_one_shot_lora_set[training_one_shot_lora_name]['image_embeddings']
            one_shot_lora_set[training_one_shot_lora_name]['image_paths'] = training_one_shot_lora_set[training_one_shot_lora_name]['image_paths']

    # for training_lora_name in training_lora_set:
    # for training_lora_name in training_lora_set:
    #     lora_set[training_lora_name]['image_embeddings'] = training_lora_set[training_lora_name]
    # if one_shot_lora_set is not None:
    #     for one_shot_lora_name in one_shot_lora_set:
    #         lora_set[one_shot_lora_name]['image_embeddings'] = one_shot_lora_set[one_shot_lora_name]
    
    # list_lora_set = list(lora_set.keys())
    # lora_index = { list_lora_set[i]: i for i in range(len(list_lora_set)) }
    # tokenizer = T5Tokenizer.from_pretrained("t5-small")
    # tokenizer, class_tokens = add_class_tokens_to_tokenizer(tokenizer, list_lora_set)
    # processed_prompt_list = [build_labels(prompt_instance, lora_index, temperature=0.1) for prompt_instance in prompt_list]
    torch.save({
        "prompt_list": prompt_list,
        "num_loras": num_loras,
        "lora_set": lora_set,
        "one_shot_lora_set": one_shot_lora_set
    }, saving_path)
    return prompt_list, lora_set, one_shot_lora_set

def get_lora_set_input_ids(lora_set, lora_indexes, tokenizer, max_length=512):
    lora_names = sorted(lora_set.keys(), key=lambda x: lora_indexes[x])
    prompts_with_class_tokens = []
    for index, lora_name in enumerate(lora_names):
        prompts_with_class_tokens.append(f'{{[CLASS_{index}]: {lora_set[lora_name]["lora_meta"]}.}}')
    class_token_input_ids = tokenizer(prompts_with_class_tokens, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
    return class_token_input_ids

def train_experiments(training_dataset_path, training_dataset_name, whether_to_finetune, model_name = 'MultiModalEncoderDecoderForClassification', *args, **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    run_name = f'{training_dataset_name}-{model_name}-{whether_to_finetune}'
    
    training_dataset = torch.load(training_dataset_path)
    prompt_list = training_dataset['prompt_list']
    lora_set = training_dataset['lora_set']
    one_shot_lora_set = training_dataset['one_shot_lora_set']


    list_lora_set = list(lora_set.keys())
    lora_index = { list_lora_set[i]: i for i in range(len(list_lora_set)) }

    tokenizer = PreTrainedTokenizerFast.from_pretrained("t5-small");
    tokenizer, class_tokens = add_class_tokens_to_tokenizer(tokenizer, list_lora_set)
    processed_prompt_list = [build_labels(prompt_instance, lora_index, temperature=0.1) for prompt_instance in prompt_list]
    dataset = Dataset.from_list(processed_prompt_list)

    class_token_input_ids = get_class_token_input_ids(class_tokens, tokenizer)
    tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer, class_tokens, class_token_input_ids), batched=True)
    tokenized_text, class_delimiter_token_indexes, lora_class_image_embeddings = build_lora_ensemble(lora_set, lora_index, tokenizer)
    train_test_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=42)

    lora_set_input_ids = get_lora_set_input_ids(lora_set, lora_index, tokenizer)
    lora_set['lora_set_input_ids'] = lora_set_input_ids['input_ids']
    lora_set['lora_set_attention_mask'] = lora_set_input_ids['attention_mask']
    lora_set['images_input_embeddings'] = lora_class_image_embeddings

    lora_set['input_ids'] = tokenized_text['input_ids']
    lora_set['attention_mask'] = tokenized_text['attention_mask']
    lora_set['class_delimiter_token_indexes'] = torch.tensor(class_delimiter_token_indexes)

    print("TRAIN MODE")
    if model_name == 'MultiModalEncoderDecoderForClassification':
        model = MultiModalEncoderDecoderForClassification("t5-small", lora_set).to(device)
    elif model_name == 'VisualEncoderDecoderForClassification':
        model = VisualEncoderDecoderForClassification("t5-small", lora_set).to(device)
    elif model_name == 'TextEncoderDecoderForClassification':
        model = EncoderDecoderForClassification("t5-small", lora_set).to(device)
    data_collator = CustomDataCollatorWithPadding(tokenizer=tokenizer) 
    train_model(model, run_name, train_test_dataset['train'], train_test_dataset['test'], tokenizer, data_collator)

    # if whether_to_finetune == True:
    #     train_model(model, train_test_dataset['train'], train_test_dataset['test'], tokenizer, data_collator, finetune=True)
    return model

def evaluate_experiments(model, test_dataset_path, test_dataset_name, *args, **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    
    training_dataset = torch.load(test_dataset_path)
    prompt_list = training_dataset['prompt_list']
    lora_set = training_dataset['lora_set']
    one_shot_lora_set = training_dataset['one_shot_lora_set']

    list_lora_set = list(lora_set.keys())
    lora_index = { list_lora_set[i]: i for i in range(len(list_lora_set)) }

    tokenizer = PreTrainedTokenizerFast.from_pretrained("t5-small");
    tokenizer, class_tokens = add_class_tokens_to_tokenizer(tokenizer, list_lora_set)
    processed_prompt_list = [build_labels(prompt_instance, lora_index, temperature=0.1) for prompt_instance in prompt_list]
    dataset = Dataset.from_list(processed_prompt_list)
    # output = { "prompt_list": processed_prompt_list, "num_loras": num_loras, "lora_set": list_lora_set }
    # with open("prompt_list.json", "w") as f:
    #   json.dump(output, f)
    # print(dataset[2])
    class_token_input_ids = get_class_token_input_ids(class_tokens, tokenizer)
    tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer, class_tokens, class_token_input_ids), batched=True)
    tokenized_text, class_delimiter_token_indexes, lora_class_image_embeddings = build_lora_ensemble(lora_set, lora_index, tokenizer)
    train_test_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=42)

    lora_set_input_ids = get_lora_set_input_ids(lora_set, lora_index, tokenizer)
    lora_set['lora_set_input_ids'] = lora_set_input_ids['input_ids']
    lora_set['lora_set_attention_mask'] = lora_set_input_ids['attention_mask']
    lora_set['images_input_embeddings'] = lora_class_image_embeddings

    lora_set['input_ids'] = tokenized_text['input_ids']
    lora_set['attention_mask'] = tokenized_text['attention_mask']
    lora_set['class_delimiter_token_indexes'] = torch.tensor(class_delimiter_token_indexes)

    print("TEST MODE")
    with torch.no_grad():
        model.eval()
        model.to(device)
        
import wandb


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Train the fusion of LoRAs."
    )

    # parser.add_argument('--lora_path', default='models/lora/reality',
    #                     help='Path to the directory containing LoRA files', type=str)
    # parser.add_argument('--prompt_path', default='models/prompts/reality',
    #                     help='Path to the directory containing prompt files', type=str)
    # parser.add_argument('--lora_meta_path', default='models/lora/reality',
    #                     help='Path to the directory containing LoRA meta files', type=str)
    # parser.add_argument('--test', default=False,
    #                     help='Test mode', type=bool)
    # parser.add_argument('--output_path', default='models/embeddings/reality',
    #                     help='Path to the directory containing embeddings', type=str)
    # parser.add_argument('--training_image_path', default='models/images/reality',
    #                     help='Path to the directory containing images', type=str)
    # parser.add_argument('--generate_embeddings', default=False,
    #                     help='Generate embeddings', type=bool)
    # parser.add_argument('--seed', default=11,
    #                     help='Seed for generating images', type=int)
    # parser.add_argument('--one_shot_test_image_path', default='models/images/reality',
    #                     help='Path to the directory containing images', type=str)
    # parser.add_argument('--testing', default=False,
    #                     help='Testing mode', type=bool)
    # parser.add_argument('--training_lora_prompt_path', default=None,
    #                     help='Path to the directory containing training LoRA prompt files', type=str)
    # parser.add_argument('--including_test_lora_names', default=False,
    #                     help='Including test LoRA names', type=bool)
    # parser.add_argument('--type', default="all",
    #                     help='Type of LoRA', type=str)

    # Log in to W&B
    wandb.login(key="5134a81007c8d0794c2e7e0000df2c62c9128c5c")

    parser.add_argument('--training_dataset_path', default=None,
                        help='Path to the directory containing training dataset', type=str)
    parser.add_argument('--training_dataset_name', default=None,
                        help='Name of the training dataset', type=str)
    # parser.add_argument('--whether_to_finetune', default=False,
    #                     help='Whether to finetune', type=bool)
    parser.add_argument('--model_name', default='MultiModalEncoderDecoderForClassification',
                        help='Model name', type=str)
    parser.add_argument('--test', default=False,
                        help='Test mode', type=bool)
    

    args = parser.parse_args()
    # if args.generate_embeddings == True:
    #     execute_embedding_generation(args.output_path, args.training_image_path)
    # elif args.testing == True:
    #     build_lora_set(args.lora_path, args.prompt_path, args.lora_meta_path, args.training_image_path, args.one_shot_test_image_path, args.training_lora_prompt_path, args.output_path, including_test_lora_names=args.including_test_lora_names, type=args.type)
    # else:
    #     main(**vars(args))
    import transformers
    print('transformers.__version__: ', transformers.__version__)
    if args.test == True:
        evaluate_experiments(args.model_name, args.test_dataset_path, args.test_dataset_name)
    else:
        train_experiments(args.training_dataset_path, args.training_dataset_name, False, args.model_name)

"python train_fusion_of_loras.py --prompt_path '/projects/bdpp/hyi1/stable-diffusion/Heterogenous-Multi-LoRA/Multi-LoRA-Composition/gen_prompts_compose' --lora_path '/projects/bdpp/hyi1/stable-diffusion/LoRA-compos-data/lora-pool/compose' --lora_meta_path '/projects/bdpp/hyi1/stable-diffusion/LoRA-compos-data/lora-pool-meta'"
"python train_fusion_of_loras.py --output_path '/projects/bdpp/hyi1/stable-diffusion/Heterogenous-Multi-LoRA/Multi-LoRA-Composition/gen_images/image_embeddings.pth' --training_image_path '/projects/bdpp/hyi1/stable-diffusion/Heterogenous-Multi-LoRA/Multi-LoRA-Composition/gen_images' --generate_embeddings True"
"python train_fusion_of_loras.py --output_path '/projects/bdpp/hyi1/stable-diffusion/Heterogenous-Multi-LoRA/Multi-LoRA-Composition/gen_images_compose/image_embeddings.pth' --training_image_path '/projects/bdpp/hyi1/stable-diffusion/Heterogenous-Multi-LoRA/Multi-LoRA-Composition/gen_images_compose' --generate_embeddings True"
"python train_fusion_of_loras.py --output_path '/projects/bdpp/hyi1/stable-diffusion/Heterogenous-Multi-LoRA/LoRA-compos-data/lora-pool-img/image_embeddings.pth' --training_image_path '/projects/bdpp/hyi1/stable-diffusion/Heterogenous-Multi-LoRA/LoRA-compos-data/lora-pool-img' --generate_embeddings True"

"python train_fusion_of_loras.py --testing True --lora_path '/projects/bdpp/hyi1/stable-diffusion/LoRA-compos-data/lora-pool/compose' --prompt_path '/projects/bdpp/hyi1/stable-diffusion/Heterogenous-Multi-LoRA/Multi-LoRA-Composition/gen_prompts_compose' --lora_meta_path '/projects/bdpp/hyi1/stable-diffusion/LoRA-compos-data/lora-pool-meta' --training_lora_prompt_path '/projects/bdpp/hyi1/stable-diffusion/Heterogenous-Multi-LoRA/Multi-LoRA-Composition/gen_prompts' --training_image_path '/projects/bdpp/hyi1/stable-diffusion/Heterogenous-Multi-LoRA/Multi-LoRA-Composition/gen_images' --one_shot_test_image_path '/projects/bdpp/hyi1/stable-diffusion/Heterogenous-Multi-LoRA/LoRA-compos-data/lora-pool-img' --output_path '/projects/bdpp/hyi1/stable-diffusion/Heterogenous-Multi-LoRA/Multi-LoRA-Composition/training_dataset/training_lora_set_all_no_test.pth' --including_test_lora_names False --type 'all'"

"python train_fusion_of_loras.py --training_dataset_path '/projects/bdpp/hyi1/stable-diffusion/Heterogenous-Multi-LoRA/Multi-LoRA-Composition/training_dataset/training_lora_set_all_no_test.pth' --training_dataset_name 'training_lora_set_all_no_test' --model_name 'MultiModalEncoderDecoderForClassification'"
"python train_fusion_of_loras.py --training_dataset_path '/projects/bdpp/hyi1/stable-diffusion/Heterogenous-Multi-LoRA/Multi-LoRA-Composition/training_dataset/training_lora_set_all_no_test.pth' --training_dataset_name 'training_lora_set_all_no_test' --model_name 'VisualEncoderDecoderForClassification'"
"python train_fusion_of_loras.py --training_dataset_path '/projects/bdpp/hyi1/stable-diffusion/Heterogenous-Multi-LoRA/Multi-LoRA-Composition/training_dataset/training_lora_set_all_no_test.pth' --training_dataset_name 'training_lora_set_all_no_test' --model_name 'TextEncoderDecoderForClassification'"

"python train_fusion_of_loras.py --training_dataset_path '/projects/bdpp/hyi1/stable-diffusion/Heterogenous-Multi-LoRA/Multi-LoRA-Composition/training_dataset/training_lora_set_all_with_test.pth' --training_dataset_name 'training_lora_set_all_with_test' --model_name 'MultiModalEncoderDecoderForClassification'"
"python train_fusion_of_loras.py --training_dataset_path '/projects/bdpp/hyi1/stable-diffusion/Heterogenous-Multi-LoRA/Multi-LoRA-Composition/training_dataset/training_lora_set_all_with_test.pth' --training_dataset_name 'training_lora_set_all_with_test' --model_name 'VisualEncoderDecoderForClassification'"

"python train_fusion_of_loras.py --training_dataset_path '/projects/bdpp/hyi1/stable-diffusion/Heterogenous-Multi-LoRA/Multi-LoRA-Composition/training_dataset/training_lora_set_anime_no_test.pth' --training_dataset_name 'training_lora_set_anime_no_test' --model_name 'MultiModalEncoderDecoderForClassification'"
"python train_fusion_of_loras.py --training_dataset_path '/projects/bdpp/hyi1/stable-diffusion/Heterogenous-Multi-LoRA/Multi-LoRA-Composition/training_dataset/training_lora_set_anime_no_test.pth' --training_dataset_name 'training_lora_set_anime_no_test' --model_name 'VisualEncoderDecoderForClassification'"

"python train_fusion_of_loras.py --training_dataset_path '/projects/bdpp/hyi1/stable-diffusion/Heterogenous-Multi-LoRA/Multi-LoRA-Composition/training_dataset/training_lora_set_anime_with_test.pth' --training_dataset_name 'training_lora_set_anime_with_test' --model_name 'MultiModalEncoderDecoderForClassification'"
"python train_fusion_of_loras.py --training_dataset_path '/projects/bdpp/hyi1/stable-diffusion/Heterogenous-Multi-LoRA/Multi-LoRA-Composition/training_dataset/training_lora_set_anime_with_test.pth' --training_dataset_name 'training_lora_set_anime_with_test' --model_name 'VisualEncoderDecoderForClassification'"


"python train_fusion_of_loras.py --training_dataset_path '/projects/bdpp/hyi1/stable-diffusion/Heterogenous-Multi-LoRA/Multi-LoRA-Composition/training_dataset/training_lora_set_reality_no_test.pth' --training_dataset_name 'training_lora_set_reality_no_test' --model_name 'MultiModalEncoderDecoderForClassification'"
"python train_fusion_of_loras.py --training_dataset_path '/projects/bdpp/hyi1/stable-diffusion/Heterogenous-Multi-LoRA/Multi-LoRA-Composition/training_dataset/training_lora_set_reality_no_test.pth' --training_dataset_name 'training_lora_set_reality_no_test' --model_name 'VisualEncoderDecoderForClassification'"


"python train_fusion_of_loras.py --training_dataset_path '/projects/bdpp/hyi1/stable-diffusion/Heterogenous-Multi-LoRA/Multi-LoRA-Composition/training_dataset/training_lora_set_reality_with_test.pth' --training_dataset_name 'training_lora_set_reality_with_test' --model_name 'MultiModalEncoderDecoderForClassification'"
"python train_fusion_of_loras.py --training_dataset_path '/projects/bdpp/hyi1/stable-diffusion/Heterogenous-Multi-LoRA/Multi-LoRA-Composition/training_dataset/training_lora_set_reality_with_test.pth' --training_dataset_name 'training_lora_set_reality_with_test' --model_name 'VisualEncoderDecoderForClassification'"

