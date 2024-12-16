import torch.nn as nn

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_loras):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_loras),
            nn.Softmax(dim=-1),  # Outputs weights that sum to 1
        )

    def forward(self, task_features):
        return self.fc(task_features)


class GatingNetworkWithTimestep(nn.Module):
    def __init__(self, input_dim, num_loras, timestep_dim=1):
        super(GatingNetworkWithTimestep, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim + timestep_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_loras),
            nn.Softmax(dim=-1),  # Output weights summing to 1
        )

    def forward(self, task_features, timestep):
        # Concatenate task features (e.g., prompt embedding) with timestep
        timestep = timestep.unsqueeze(-1)  # Ensure timestep is the correct shape
        combined_input = torch.cat((task_features, timestep), dim=-1)
        return self.fc(combined_input)
    
class CrossAttentionGatingNetworkWithLoRAPool(nn.Module):
    def __init__(self, embed_dim, num_heads, head_dim, num_loras, device = 'cpu'):
        """
        Gating network using the diffusion model's cross-attention output as queries.
        Args:
            embed_dim: Dimensionality of the query from the diffusion model.
            num_heads: Number of attention heads.
            num_loras: Number of LoRAs in the pool.
            hidden_dim: Dimensionality of the hidden layers.
        """
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.num_loras = num_loras
        self.head_dim = head_dim
        self.total_dim = num_heads * self.head_dim  # Total dimensionality of all heads

        self.device = device

        self.query_projection = nn.Sequential(     # Projection to total_dim
            nn.Linear(self.embed_dim, self.total_dim * 2),
            nn.GELU(),
            nn.Linear(self.total_dim * 2, self.total_dim)
        ).to(self.device)

        # Multihead attention layer
        self.attention = nn.MultiheadAttention(embed_dim=self.total_dim, num_heads=self.num_heads, batch_first=True).to(self.device)

        # Output layer to predict LoRA weights
        self.output_layer = nn.Sequential(
            nn.Linear(self.total_dim, self.total_dim * 2),
            nn.GELU(),
            nn.Linear(self.total_dim * 2, self.num_loras),
            nn.Softmax(dim=-1),  # Ensure weights sum to 1
        ).to(self.device)

    def forward(self, query, lora_keys, lora_values):
        """
        Forward pass of the gating network.
        Args:
            query: Query from the diffusion model's cross-attention, shape (batch, seq_len, query_dim).
            lora_keys: Keys for the LoRA pool, shape (num_loras, query_dim).
            lora_values: Values for the LoRA pool, shape (num_loras, query_dim).
        Returns:
            LoRA weights, shape (batch, num_loras).
        """
        batch_size, seq_len, embed_dim = query.shape
        num_loras_key, key_num_heads, key_head_dim = lora_keys.shape
        num_loras_value, value_num_heads, value_head_dim = lora_values.shape


        assert embed_dim == self.embed_dim, "Query embedding dimension mismatch."
        assert self.num_loras == num_loras_key, "Number of key LoRAs mismatch."
        assert self.num_loras == num_loras_value, "Number of value LoRAs mismatch."
        assert self.num_heads == key_num_heads, "Number of attention heads mismatch."
        assert self.num_heads == value_num_heads, "Number of attention heads mismatch."
        assert self.head_dim == key_head_dim, "Head dimensionality mismatch."
        assert self.head_dim == value_head_dim, "Head dimensionality mismatch."

        query = self.query_projection(query)  # Shape: (batch, seq_len, total_dim)

        # Expand keys and values to match batch size
        lora_keys = lora_keys.reshape(num_loras, self.num_heads * self.head_dim)    # (num_loras, num_heads * head_dim)
        expanded_keys = lora_keys.unsqueeze(0).expand(batch_size, -1, -1)  # Shape: (batch, num_loras, num_heads * head_dim)
        lora_values = lora_values.reshape(num_loras, self.num_heads * self.head_dim)    # (num_loras, num_heads * head_dim)
        expanded_values = lora_values.unsqueeze(0).expand(batch_size, -1, -1)  # Shape: (batch, num_loras, num_heads * head_dim)

        # Apply multihead attention
        # Query: (batch, seq_len, query_dim)
        # Key, Value: (batch, num_loras, query_dim)
        attention_output, _ = self.attention(query, expanded_keys, expanded_values)  # Shape: (batch, seq_len, value_dim)

        # Aggregate attention output by pooling over the sequence dimension
        aggregated_output = attention_output.mean(dim=1)  # Shape: (batch, value_dim)

        # Predict LoRA weights
        weights = self.output_layer(aggregated_output)  # Shape: (batch, num_loras)

        return weights

class SimplifiedCrossAttentionGatingNetworkWithLoRAPool(nn.Module):
    def __init__(self, embed_dim, num_heads, head_dim, num_loras, device = 'cpu'):
        """
        Simplified Gating network using the diffusion model's cross-attention output as queries.
        Args:
            embed_dim: Dimensionality of the query from the diffusion model.
            num_heads: Number of attention heads.
        """
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.num_loras = num_loras
        self.head_dim = head_dim
        self.total_dim = self.num_heads * self.head_dim  # Total dimensionality of all heads

        self.device = device

        self.query_projection = nn.Sequential(     # Projection to total_dim
            nn.Linear(self.embed_dim, self.total_dim * 2),
            nn.GELU(),
            nn.Linear(self.total_dim * 2, self.total_dim)
        ).to(self.device)

        # Multihead attention layer
        self.attention = nn.MultiheadAttention(embed_dim=self.total_dim, num_heads=self.num_heads, batch_first=True).to(self.device)

    def forward(self, query, lora_keys):
        """
        Forward pass of the gating network.
        Args:
            query: Query from the diffusion model's cross-attention, shape (batch, seq_len, embed_dim).
            lora_keys: Keys for the LoRA pool, shape (num_loras, embed_dim).
        Returns:
            LoRA weights, shape (batch, num_loras).
        """
        batch_size, seq_len, embed_dim = query.shape
        num_loras, key_num_heads, key_head_dim = lora_keys.shape

        assert embed_dim == self.embed_dim, "Query embedding dimension mismatch."
        assert self.num_loras == num_loras, "Number of LoRAs mismatch."
        assert self.num_heads == key_num_heads, "Number of attention heads mismatch."
        assert self.head_dim == key_head_dim, "Head dimensionality mismatch."

        query = self.query_projection(query)  # Shape: (batch, seq_len, total_dim)

        # Reshape for multi-head attention
        # query = query.reshape(batch_size, seq_len, self.num_heads, self.head_dim)  # (batch, seq_len, num_heads, head_dim)

        lora_keys = lora_keys.reshape(num_loras, self.num_heads * self.head_dim)    # (num_loras, num_heads * head_dim)
        # Expand keys to match batch size
        expanded_keys = lora_keys.unsqueeze(0).expand(batch_size, -1, -1)  # Shape: (batch, num_loras, num_heads * head_dim)

        # Apply multihead attention
        attention_output, attention_weights = self.attention(query, expanded_keys, expanded_keys, need_weights=True) # average_attn_weights=False )  # Shape: (batch, seq_len, embed_dim), (batch, seq_len, num_loras)

        # Aggregate weights by pooling over the sequence dimension
        lora_weights = attention_weights.mean(dim=1)  # Shape: (batch, num_loras)

        return lora_weights


class SimplifiedGatingNetwork(nn.Module):
    def __init__(self, embed_dim, num_heads, head_dim, num_loras, advancedLoRAPool, device = 'cpu'):
        """
        Simplified gating network using scaled dot-product attention for multi-head inputs.
        Args:
            embed_dim: Dimensionality of the query from the diffusion model.
            num_heads: Number of attention heads.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.total_dim = self.num_heads * self.head_dim  # Total dimensionality of all heads

        self.num_loras = num_loras

        self.device = device

        self.advancedLoRAPool = advancedLoRAPool

        self.query_projection = nn.Sequential(     # Projection to total_dim
            nn.Linear(self.embed_dim, self.total_dim * 2),
            nn.GELU(),
            nn.Linear(self.total_dim * 2, self.total_dim)
        ).to(self.device)


        self.concat_projection = nn.Sequential(    # Projection to num_loras
            nn.Linear(self.num_heads * self.num_loras, self.num_heads * self.num_loras * 2),
            nn.GELU(),
            nn.Linear(self.num_heads * self.num_loras * 2, self.num_loras)
        ).to(self.device)


    def forward(self, query):
        """
        Forward pass of the gating network.
        Args:
            query: Query from the diffusion model's cross-attention, shape (batch, seq_len, embed_dim).
            lora_keys: Keys for the LoRA pool, shape (num_loras, embed_dim).
        Returns:
            LoRA weights, shape (batch, num_loras).
        """
        # Reshape queries and keys to include heads
        batch_size, seq_len, embed_dim = query.shape
        lora_keys = self.advancedLoRAPool(only_output_key=True)
        num_loras, key_num_heads, key_head_dim = lora_keys.shape

        assert embed_dim == self.embed_dim, "Query embedding dimension mismatch."
        assert self.num_loras == num_loras, "Number of LoRAs mismatch."
        assert self.num_heads == key_num_heads, "Number of attention heads mismatch."
        assert self.head_dim == key_head_dim, "Head dimensionality mismatch."

        query = self.query_projection(query)  # Shape: (batch, seq_len, total_dim)
        # Reshape for multi-head attention
        query = query.reshape(batch_size, seq_len, self.num_heads, self.head_dim)  # (batch, seq_len, num_heads, head_dim)
        lora_keys = lora_keys.reshape(num_loras, self.num_heads, self.head_dim)    # (num_loras, num_heads, head_dim)

        # Compute scaled dot-product attention
        attention_scores = torch.einsum("bqhd,khd->bqhk", query, lora_keys)  # (batch, seq_len, num_heads, num_loras)
        attention_scores /= self.head_dim ** 0.5  # Scale by sqrt(head_dim)

        # Flatten the last two dimensions (num_heads and num_loras)
        concatenated_scores = attention_scores.reshape(batch_size, seq_len, -1)  # (batch, seq_len, num_heads * num_loras)

        # Project concatenated features back to num_loras
        lora_weights = self.concat_projection(concatenated_scores)  # (batch, seq_len, num_loras)

        # Aggregate over the sequence dimension (mean pooling)
        global_lora_weights = lora_weights.mean(dim=1)  # (batch, num_loras)

        # Apply softmax for normalized global LoRA weights
        attention_weights = torch.softmax(global_lora_weights, dim=-1)  # (batch, num_loras)

        return attention_weights

