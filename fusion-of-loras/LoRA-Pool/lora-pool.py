import torch.nn as nn

class LoRAPool(nn.Module):
    def __init__(self, lora_modules, embed_dim):
        """
        Create an embedding representation for the LoRA pool.
        Args:
            lora_modules: List of LoRA modules.
            embed_dim: Dimensionality of the embeddings for keys and values.
        """
        super().__init__()
        self.lora_modules = lora_modules
        self.embeddings = nn.Parameter(torch.randn(len(lora_modules), embed_dim))

    def forward(self):
        """
        Return the LoRA key-value embeddings.
        Returns:
            Keys and values, shape (num_loras, embed_dim).
        """
        return self.embeddings, self.embeddings


class AdvancedLoRAPool(nn.Module):
    def __init__(self, embed_dim, num_heads, head_dim, tokenizer, text_encoder, lora_list = None, embedding_frozen = True, device = 'cpu'):
        """
        Create a LoRA pool with projection layers for keys and values.
        Args:
            lora_embeddings: List of LoRA embeddings.
            key_dim: Dimensionality of the projected key embeddings.
            value_dim: Dimensionality of the projected value embeddings.
        """
        super().__init__()
        # self.lora_descriptions = lora_list
        # for lora_description in self.lora_descriptions:
        #   with open(file_path, "r", encoding="utf-8") as file:
        #     descriptions = file.read()
        #     lora_description['description'] = descriptions
        # Projection layers

        self.device = device
        # Multi-head Key and Value projections
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder.to(self.device)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.total_dim = num_heads * self.head_dim  # Total dimensionality of all heads

        self.embedding_frozen = embedding_frozen

        self.key_projector = nn.Linear(self.embed_dim, self.total_dim).to(self.device)  # Projects to multi-head Keys
        self.value_projector = nn.Linear(self.embed_dim, self.total_dim).to(self.device)  # Projects to multi-head Values

        if lora_list:
          lora_description_texts = []
          for lora_info in lora_list:
            lora_description_texts.append(lora_info['description'])
            if os.path.exists(lora_info['model_path']):
              temp_model = StableDiffusionPipeline.from_pretrained(
              lora_info['model_path'], safety_checker=None, torch_dtype=torch.float16
              ).to("cuda")
              lora_info['model'] = temp_model
          self.lora_list = lora_list
          self.embeddings = self.encode_lora_descriptions(lora_description_texts, embedding_frozen)
          if self.embeddings.shape[-1] != self.embed_dim:
            raise TypeError("Embedding dimension mismatch")


    def get_prompt_embedding(self, prompt, tokenizer, text_encoder):
        inputs = tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        with torch.no_grad():
            text_embeddings = text_encoder(input_ids).last_hidden_state
        return text_embeddings

    def encode_lora_descriptions(self, lora_descriptions, embedding_frozen = True):
        raw_lora_embeddings = self.get_prompt_embedding(lora_descriptions, self.tokenizer, self.text_encoder)
        print(raw_lora_embeddings.shape)
        # CLIP
        if not self.embedding_frozen:
          embeddings = nn.Parameter(raw_lora_embeddings)  # Learnable embeddings
        else:
          embeddings = raw_lora_embeddings  # Frozen embeddings

        return embeddings


    def forward(self, description_texts = None, only_output_key = False):
        """
        Project the embeddings into key and value matrices.
        Returns:
            lora_keys: Projected key embeddings, shape (num_loras, key_dim).
            lora_values: Projected value embeddings, shape (num_loras, value_dim).
        """
        if description_texts == None:
          if self.embeddings == None:
            raise TypeError("No LoRA embeddings provided")
          embeddings = self.embeddings
        else:
          embeddings = self.encode_lora_descriptions(description_texts, self.tokenizer, self.text_encoder)
          if self.embeddings.shape[-1] != self.embed_dim:
            raise TypeError("Embedding dimension mismatch")

        # print('embeddings.shape', embeddings.shape)
        lora_keys = self.key_projector(embeddings)  # Shape: (num_loras, key_dim)

        # Reshape to (batch, num_heads, head_dim)
        lora_keys = lora_keys.reshape(lora_keys.shape[0], lora_keys.shape[1], self.num_heads, self.head_dim)  # Shape: (num_loras, seq, num_heads, head_dim)

        if only_output_key:
          return lora_keys.mean(dim = 1)

        # print('lora_keys.shape', lora_keys.shape)
        lora_values = self.value_projector(embeddings)  # Shape: (num_loras, value_dim)

        lora_values = lora_values.reshape(lora_keys.shape[0], lora_keys.shape[1], self.num_heads, self.head_dim)  # Shape: (num_loras, seq, num_heads, head_dim)
        return lora_keys.mean(dim = 1), lora_values.mean(dim = 1)


class AdvancedComparableLoRAPool(nn.Module):
    def __init__(self, embed_dim, key_dim, value_dim, feature_extractor=None):
        """
        Advanced LoRAPool that compares the effect of using LoRA versus not using LoRA with a complex feature extractor.
        Args:
            embed_dim: Dimensionality of the intermediate comparison embeddings.
            key_dim: Dimensionality of the Key embeddings.
            value_dim: Dimensionality of the Value embeddings.
            feature_extractor: A pretrained model or custom CNN for feature extraction.
        """
        super().__init__()

        # If no feature extractor is provided, use a default pretrained ResNet
        self.feature_extractor = feature_extractor or self._default_feature_extractor(embed_dim)


        self.num_heads = num_heads
        self.head_dim = head_dim
        self.total_dim = num_heads * head_dim  # Total dimensionality of all heads

        # Learnable projection layers for features into Key and Value spaces
        self.feature_projector = nn.Linear(embed_dim, embed_dim)

        self.key_projector = nn.Linear(embed_dim, self.total_dim)  # Projects to multi-head Keys
        self.value_projector = nn.Linear(embed_dim, self.total_dim)  # Projects to multi-head Values


    def _default_feature_extractor(self, embed_dim):
        """
        Create a default feature extractor based on ResNet.
        Args:
            embed_dim: The desired output dimensionality of the feature extractor.
        Returns:
            feature_extractor: A pretrained ResNet model with modified output.
        """
        resnet = resnet18(pretrained=True)
        resnet.fc = nn.Linear(resnet.fc.in_features, embed_dim)  # Replace final layer
        return resnet

    def compute_features(self, denoised_image, denoised_with_lora):
        """
        Compute the features representing the difference between using LoRA and not using LoRA.
        Args:
            denoised_image: Tensor of the denoised image without LoRA, shape (batch, 3, H, W).
            denoised_with_lora: Tensor of the denoised image with LoRA, shape (batch, 3, H, W).
        Returns:
            features: Feature tensor of shape (batch, embed_dim).
        """
        # Compute the difference between the two images
        difference = denoised_with_lora - denoised_image  # Shape: (batch, 3, H, W)

        # Use the feature extractor to process the difference
        features = self.feature_extractor(difference)  # Shape: (batch, embed_dim)

        # Optional learnable projection
        projected_features = self.feature_projector(features)  # Shape: (batch, embed_dim)

        return projected_features

    def project_to_heads(self, features):
        """
        Project features into multi-head Key and Value matrices.
        Args:
            features: Feature tensor of shape (batch, embed_dim).
        Returns:
            lora_keys: Multi-head Keys, shape (batch, num_heads, head_dim).
            lora_values: Multi-head Values, shape (batch, num_heads, head_dim).
        """
        batch_size = features.size(0)

        # Project features into total Key and Value dimensions
        keys = self.key_projector(features)  # Shape: (batch, total_dim)
        values = self.value_projector(features)  # Shape: (batch, total_dim)

        # Reshape to (batch, num_heads, head_dim)
        keys = keys.view(batch_size, self.num_heads, self.head_dim)  # Shape: (batch, num_heads, head_dim)
        values = values.view(batch_size, self.num_heads, self.head_dim)  # Shape: (batch, num_heads, head_dim)

        return keys, values


    def forward(self, denoised_image, denoised_with_lora):
        """
        Compute features and project them into Key and Value matrices.
        Args:
            denoised_image: Tensor of the denoised image without LoRA, shape (batch, 3, H, W).
            denoised_with_lora: Tensor of the denoised image with LoRA, shape (batch, 3, H, W).
        Returns:
            lora_keys: Key embeddings, shape (batch, key_dim).
            lora_values: Value embeddings, shape (batch, value_dim).
        """
        # Compute features from the comparison of the two images
        comparison_features = self.compute_features(denoised_image, denoised_with_lora)  # Shape: (batch, embed_dim)

        # Project features into Key and Value matrices
        lora_keys, lora_values = self.project_to_heads(comparison_features)  # Multi-head Keys/Values

        return lora_keys, lora_values
