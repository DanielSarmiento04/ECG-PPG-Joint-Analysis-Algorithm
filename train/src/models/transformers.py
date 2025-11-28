"""
Vision Transformer for 1D Signal Analysis (ECG/PPG)
Adapted for time-series data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """Convert 1D signal into patches and embed them"""
    def __init__(self, patch_size, hidden_size, input_length):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_patches = input_length // patch_size
        
        # Linear projection of flattened patches
        self.projection = nn.Linear(patch_size, hidden_size)
        
        # Learnable position embeddings
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.num_patches + 1, hidden_size))
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # x shape: (batch_size, 1, input_length)
        # Reshape to patches: (batch_size, num_patches, patch_size)
        x = x.squeeze(1)  # Remove channel dimension: (batch_size, input_length)
        x = x.reshape(batch_size, self.num_patches, self.patch_size)
        
        # Project patches to hidden dimension
        x = self.projection(x)  # (batch_size, num_patches, hidden_size)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch_size, num_patches + 1, hidden_size)
        
        # Add position embeddings
        x = x + self.position_embeddings
        
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_length, _ = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x)  # (batch_size, seq_length, hidden_size * 3)
        qkv = qkv.reshape(batch_size, seq_length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_length, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, v)  # (batch_size, num_heads, seq_length, head_dim)
        context = context.permute(0, 2, 1, 3).contiguous()  # (batch_size, seq_length, num_heads, head_dim)
        context = context.reshape(batch_size, seq_length, self.hidden_size)
        
        # Output projection
        output = self.out(context)
        output = self.dropout(output)
        
        return output


class MLP(nn.Module):
    """Feed-forward network"""
    def __init__(self, hidden_size, mlp_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer encoder block"""
    def __init__(self, hidden_size, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attention = MultiHeadSelfAttention(hidden_size, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = MLP(hidden_size, mlp_dim, dropout)
        
    def forward(self, x):
        # Pre-norm architecture
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PhysiologicalTransformer(nn.Module):
    """
    Transformer model for Physiological Signal Analysis (ECG/PPG)
    
    Can handle both sequential waveform data and feature vectors.
    Incorporates patient metadata (numerical and categorical) for hybrid modeling.
    
    Args:
        patch_size: Size of each patch (1 for feature vectors)
        hidden_size: Dimension of the hidden representations
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_dim: Dimension of the MLP layer
        num_outputs: Number of output values (e.g., 2 for SBP and DBP)
        dropout: Dropout rate
        attention_dropout: Attention dropout rate
        input_length: Length of input signal or number of features
        num_numerical_features: Number of extra numerical features (age, bmi, etc.)
        categorical_cardinalities: List of integers representing the number of classes for each categorical feature
    """
    def __init__(self, patch_size, hidden_size, depth, num_heads, mlp_dim, 
                 num_outputs, dropout=0.1, attention_dropout=0.0, input_length=400,
                 num_numerical_features=0, categorical_cardinalities=None):
        super().__init__()
        
        assert input_length % patch_size == 0, f"input_length ({input_length}) must be divisible by patch_size ({patch_size})"
        
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.input_length = input_length
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(patch_size, hidden_size, input_length)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, mlp_dim, dropout)
            for _ in range(depth)
        ])
        
        # Layer norm
        self.norm = nn.LayerNorm(hidden_size)
        
        # --- Metadata Processing ---
        self.num_numerical_features = num_numerical_features
        self.categorical_cardinalities = categorical_cardinalities or []
        
        # Embeddings for categorical features
        # Rule of thumb for embedding dim: min(50, (cardinality + 1) // 2)
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, min(50, (card + 1) // 2))
            for card in self.categorical_cardinalities
        ])
        
        total_embedding_dim = sum(e.embedding_dim for e in self.embeddings)
        
        # Fusion Layer
        # Concatenates: [Transformer_Output, Numerical_Features, Categorical_Embeddings]
        fusion_input_dim = hidden_size + num_numerical_features + total_embedding_dim
        
        # Classification head (Updated for fusion)
        self.head = nn.Sequential(
            nn.Linear(fusion_input_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_outputs)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier uniform"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def forward(self, x, x_numeric=None, x_categorical=None):
        """
        Forward pass
        
        Args:
            x: Signal input tensor of shape (batch_size, 1, input_length)
            x_numeric: Optional numerical metadata (batch_size, num_numerical_features)
            x_categorical: Optional categorical metadata (batch_size, num_categorical_features)
            
        Returns:
            Output tensor of shape (batch_size, num_outputs)
        """
        # --- Transformer Path ---
        # Patch embedding
        x = self.patch_embedding(x)
        x = self.dropout(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Apply final layer norm
        x = self.norm(x)
        
        # Use class token as the signal representation
        cls_token = x[:, 0]
        
        # --- Fusion ---
        features = [cls_token]
        
        # Add numerical metadata
        if self.num_numerical_features > 0:
            if x_numeric is None:
                raise ValueError("Model expects x_numeric input but none provided")
            features.append(x_numeric)
            
        # Add categorical embeddings
        if len(self.categorical_cardinalities) > 0:
            if x_categorical is None:
                raise ValueError("Model expects x_categorical input but none provided")
            for i, embedding_layer in enumerate(self.embeddings):
                # Get embedding for each categorical feature
                # x_categorical[:, i] contains indices for the i-th feature
                features.append(embedding_layer(x_categorical[:, i]))
        
        # Concatenate all features
        combined = torch.cat(features, dim=1)
        
        # Final prediction
        output = self.head(combined)
        
        return output
