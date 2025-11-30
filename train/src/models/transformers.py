"""
Vision Transformer for 1D Signal Analysis (ECG/PPG)
Adapted for time-series data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    pe: torch.Tensor
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: [Batch, Seq_Len, D_Model]
        return x + self.pe[:, :x.size(1), :]

class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        # x: [Batch, Seq_Len, D_Model]
        weights = self.attention(x) # [Batch, Seq_Len, 1]
        weights = torch.softmax(weights, dim=1)
        return torch.sum(x * weights, dim=1) # [Batch, D_Model]

class TemporalPhysiologicalTransformer(nn.Module):
    """
    Temporal Transformer for Physiological Signal Analysis.
    
    Architecture:
    1. Dynamic Stream: Transformer Encoder for time-series features.
    2. Static Stream: Embeddings for categorical + MLP for continuous demographics.
    3. Fusion: Concatenates temporal context with static embedding.
    """
    def __init__(self, num_dynamic_features, num_continuous_static, categorical_cardinalities,
                 d_model=128, nhead=4, num_layers=4, dim_feedforward=256, dropout=0.1):
        """
        Args:
            num_dynamic_features: Number of time-series features (e.g., 9)
            num_continuous_static: Number of continuous static features (e.g., 2: age, bmi)
            categorical_cardinalities: List of ints, number of classes for each categorical feature
        """
        super().__init__()
        
        self.d_model = d_model
        
        # --- 1. Dynamic Feature Encoder (Time-Series) ---
        self.feature_projection = nn.Linear(num_dynamic_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True,
            activation='gelu',
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pooling = AttentionPooling(d_model)
        
        # --- 2. Static Feature Encoder (Demographics) ---
        # Continuous features
        self.static_cont_encoder = nn.Linear(num_continuous_static, 32)
        
        # Categorical Embeddings
        self.embeddings = nn.ModuleList()
        total_emb_dim = 0
        for num_classes in categorical_cardinalities:
            # Rule of thumb: min(50, num_classes // 2)
            emb_dim = min(50, (num_classes + 1) // 2)
            # Ensure minimum dimension of 2
            emb_dim = max(2, emb_dim)
            self.embeddings.append(nn.Embedding(num_classes + 1, emb_dim)) # +1 for unknown/padding
            total_emb_dim += emb_dim
            
        # Project combined static features to d_model/2
        self.static_projection = nn.Sequential(
            nn.Linear(32 + total_emb_dim, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # --- 3. Fusion & Prediction Heads ---
        fusion_dim = d_model + (d_model // 2)
        
        self.fusion_projection = nn.Linear(fusion_dim, fusion_dim)
        self.fusion_norm = nn.LayerNorm(fusion_dim)
        self.fusion_dropout = nn.Dropout(dropout)
        
        # SBP Head
        self.sbp_head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
        # DBP Head
        self.dbp_head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x_dynamic, x_static_cont, x_static_cat):
        """
        Args:
            x_dynamic: [Batch, Seq_Len, Num_Dynamic_Features]
            x_static_cont: [Batch, Num_Continuous_Static]
            x_static_cat: [Batch, Num_Categorical_Static] (LongTensor)
        """
        # --- Temporal Stream ---
        x = self.feature_projection(x_dynamic) 
        x = x * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        context_vector = self.pooling(x)
        
        # --- Static Stream ---
        # Continuous
        cont_emb = self.static_cont_encoder(x_static_cont)
        
        # Categorical
        cat_embs = []
        for i, emb_layer in enumerate(self.embeddings):
            # x_static_cat[:, i] is the i-th categorical feature for the batch
            cat_embs.append(emb_layer(x_static_cat[:, i]))
        
        # Concatenate all static features
        static_full = torch.cat([cont_emb] + cat_embs, dim=1)
        static_emb = self.static_projection(static_full)
        
        # --- Fusion ---
        combined = torch.cat([context_vector, static_emb], dim=1)
        combined_proj = self.fusion_projection(combined)
        combined = self.fusion_norm(combined + self.fusion_dropout(combined_proj))
        
        # --- Prediction ---
        sbp = self.sbp_head(combined)
        dbp = self.dbp_head(combined)
        
        return torch.cat([sbp, dbp], dim=1)
