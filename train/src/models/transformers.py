"""
Vision Transformer for 1D Signal Analysis (ECG/PPG)
Adapted for time-series data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureTokenizer(nn.Module):
    """
    Tokenizes numerical and categorical features into a unified sequence for the Transformer.
    Based on FT-Transformer architecture.
    """
    def __init__(self, num_numerical_features, categorical_cardinalities, hidden_size):
        super().__init__()
        self.num_numerical = num_numerical_features
        self.num_categorical = len(categorical_cardinalities)
        self.hidden_size = hidden_size
        
        # Numerical feature embeddings: One linear layer per feature
        # Projects scalar value to hidden_size vector
        self.numerical_embeddings = nn.ModuleList([
            nn.Linear(1, hidden_size) for _ in range(num_numerical_features)
        ])
        
        # Categorical feature embeddings
        self.categorical_embeddings = nn.ModuleList([
            nn.Embedding(card, hidden_size) for card in categorical_cardinalities
        ])
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        
    def forward(self, x_num, x_cat):
        batch_size = x_num.shape[0]
        
        # Process numerical features
        # x_num: (batch, num_numerical)
        num_tokens = []
        for i in range(self.num_numerical):
            # Extract i-th feature: (batch, 1)
            val = x_num[:, i:i+1]
            # Project: (batch, hidden_size) -> (batch, 1, hidden_size)
            emb = self.numerical_embeddings[i](val).unsqueeze(1)
            num_tokens.append(emb)
            
        # Process categorical features
        cat_tokens = []
        if self.num_categorical > 0:
            for i in range(self.num_categorical):
                # Extract i-th feature: (batch,)
                val = x_cat[:, i]
                # Embed: (batch, hidden_size) -> (batch, 1, hidden_size)
                emb = self.categorical_embeddings[i](val).unsqueeze(1)
                cat_tokens.append(emb)
        
        # Combine all tokens
        # Sequence: [CLS, Num_1, ..., Num_N, Cat_1, ..., Cat_M]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        
        all_tokens = [cls_tokens] + num_tokens + cat_tokens
        x = torch.cat(all_tokens, dim=1)
        
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
    FT-Transformer inspired architecture for Physiological Signal Analysis.
    Treats all inputs (Signal Features + Metadata) as a unified sequence of tokens.
    
    This architecture allows the Transformer to learn complex interactions between
    signal features (e.g., PTT) and patient metadata (e.g., Age, BMI) directly
    via the self-attention mechanism.
    """
    def __init__(self, hidden_size, depth, num_heads, mlp_dim, 
                 num_outputs, dropout=0.1, attention_dropout=0.0, input_length=7,
                 num_numerical_features=0, categorical_cardinalities=None):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # Total numerical features = Signal Features + Metadata Numerical Features
        total_numerical = input_length + num_numerical_features
        self.categorical_cardinalities = categorical_cardinalities or []
        
        # Feature Tokenizer (Replaces PatchEmbedding)
        self.tokenizer = FeatureTokenizer(
            num_numerical_features=total_numerical,
            categorical_cardinalities=self.categorical_cardinalities,
            hidden_size=hidden_size
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, mlp_dim, dropout)
            for _ in range(depth)
        ])
        
        # Layer norm
        self.norm = nn.LayerNorm(hidden_size)
        
        # Final prediction head (from CLS token)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_outputs)
        )
        
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
    
    def forward(self, x_signal, x_numeric=None, x_categorical=None):
        """
        Forward pass
        
        Args:
            x_signal: Signal features (batch_size, 1, input_length) or (batch_size, input_length)
            x_numeric: Numerical metadata (batch_size, num_numerical_features)
            x_categorical: Categorical metadata (batch_size, num_categorical_features)
        """
        # Ensure x_signal is (batch, input_length)
        if x_signal.dim() == 3:
            x_signal = x_signal.squeeze(1)
            
        # Combine signal features and numerical metadata
        if x_numeric is not None:
            x_all_num = torch.cat([x_signal, x_numeric], dim=1)
        else:
            x_all_num = x_signal
            
        # Tokenize all inputs
        x = self.tokenizer(x_all_num, x_categorical)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Apply final layer norm
        x = self.norm(x)
        
        # Use CLS token (index 0) for prediction
        cls_token = x[:, 0]
        
        # Final prediction
        output = self.head(cls_token)
        
        return output
