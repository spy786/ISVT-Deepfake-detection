import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import timm
from einops import rearrange
from typing import Tuple

class XceptionFeatureExtractor(nn.Module):
    """Xception entry flow feature extractor"""
    
    def __init__(self, pretrained=True):
        super().__init__()
        
        # Load pretrained Xception
        xception = timm.create_model('xception', pretrained=pretrained)
        
        # Extract entry flow (first few layers)
        self.conv1 = xception.conv1
        self.bn1 = xception.bn1
        self.act1 = xception.act1
        
        self.conv2 = xception.conv2
        self.bn2 = xception.bn2
        self.act2 = xception.act2
        
        # Entry flow blocks
        self.block1 = xception.block1
        self.block2 = xception.block2  
        self.block3 = xception.block3
    
    def forward(self, x):
        """Extract features: (B*T, 3, 300, 300) -> (B*T, 728, 19, 19)"""
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        return x

class DecomposedSpatialTemporalAttention(nn.Module):
    """Vectorized decomposed spatial-temporal self-attention"""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        assert embed_dim % num_heads == 0
        
        # Projections for spatial attention
        self.spatial_qkv = nn.Linear(embed_dim, embed_dim * 3)
        
        # Projections for temporal attention (Q/K from subtracted, V from original)
        self.temporal_qk = nn.Linear(embed_dim, embed_dim * 2)
        self.temporal_v = nn.Linear(embed_dim, embed_dim)
        
        self.spatial_proj = nn.Linear(embed_dim, embed_dim)
        self.temporal_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)

    def self_subtract_mechanism(self, x):
        """Apply self-subtract for temporal attention"""
        # x shape: (B, T+1, HW+1, C) or (T+1, HW+1, C)
        if x.dim() == 3:  # Single sample
            x = x.unsqueeze(0)  # Add batch dimension
            single_sample = True
        else:
            single_sample = False
            
        B, T_plus_1, HW_plus_1, C = x.shape
        
        if T_plus_1 <= 2:
            result = x
        else:
            # Keep first 2 temporal rows → (prediction-CLS  + temporal-CLS)  exactly as Eq.(3)
            keep_tokens = x[:, :2]  # (B, 2, HW+1, C)
            
            # Compute frame differences for remaining frames
            frame_tokens = x[:, 2:]  # (B, T-1, HW+1, C)
            prev_tokens = x[:, 1:T_plus_1-1]  # (B, T-1, HW+1, C)
            
            # Subtraction
            # Subtraction (as per paper specification - no additional normalization)
            diff_tokens = frame_tokens - prev_tokens  # (B, T-1, HW+1, C)
            result = torch.cat([keep_tokens, diff_tokens], dim=1)  # (B, T+1, HW+1, C)
        
        if single_sample:
            result = result.squeeze(0)  # Remove batch dimension
            
        return result

    def vectorized_attention(self, q, k, v):
        """
        Vectorized attention computation
        q, k, v: (B, num_heads, N_pos, T_or_HW, head_dim)
        """
        # Compute attention scores: (B, num_heads, N_pos, T_or_HW, T_or_HW)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values: (B, num_heads, N_pos, T_or_HW, head_dim)
        out = torch.matmul(attn, v)
        return out

    def forward(self, x):
        """
        x shape: (B, T+1, HW+1, C) or (T+1, HW+1, C)
        """
        # Handle both batch and individual inputs
        if x.dim() == 3:  # Single sample
            x = x.unsqueeze(0)  # Add batch dimension
            single_sample = True
        else:
            single_sample = False
            
        B, T, HW_plus_1, C = x.shape
        
        # ================================
        # TEMPORAL ATTENTION (Vectorized)
        # ================================
        
        # Apply self-subtract mechanism
        x_subtract = self.self_subtract_mechanism(x)  # (B, T, HW+1, C)
        
        # Get temporal QKV
        temporal_qk = self.temporal_qk(x_subtract)  # (B, T, HW+1, 2*embed_dim)
        temporal_v = self.temporal_v(x)  # (B, T, HW+1, embed_dim)
        temporal_qkv = torch.cat([temporal_qk, temporal_v], dim=-1)  # (B, T, HW+1, 3*embed_dim)
        
        # Reshape for multi-head attention: (B, T, HW+1, 3, num_heads, head_dim)
        temporal_qkv = temporal_qkv.reshape(B, T, HW_plus_1, 3, self.num_heads, self.head_dim)
        temporal_qkv = temporal_qkv.permute(3, 0, 4, 2, 1, 5)  # (3, B, num_heads, HW+1, T, head_dim)
        
        temporal_q, temporal_k, temporal_v = temporal_qkv[0], temporal_qkv[1], temporal_qkv[2]
        
        # Vectorized temporal attention across all spatial positions
        temporal_out = self.vectorized_attention(temporal_q, temporal_k, temporal_v)
        # (B, num_heads, HW+1, T, head_dim)
        
        # Reshape back: (B, T, HW+1, embed_dim)
        temporal_out = temporal_out.permute(0, 3, 2, 1, 4).reshape(B, T, HW_plus_1, self.embed_dim)
        temporal_out = self.temporal_proj(temporal_out)
        
        # ===============================
        # SPATIAL ATTENTION (Vectorized)  
        # ===============================
        
        # Get spatial QKV
        spatial_qkv = self.spatial_qkv(temporal_out)  # (B, T, HW+1, 3*embed_dim)
        spatial_qkv = spatial_qkv.reshape(B, T, HW_plus_1, 3, self.num_heads, self.head_dim)
        spatial_qkv = spatial_qkv.permute(3, 0, 4, 1, 2, 5)  # (3, B, num_heads, T, HW+1, head_dim)
        
        spatial_q, spatial_k, spatial_v = spatial_qkv[0], spatial_qkv[1], spatial_qkv[2]
        
        # Vectorized spatial attention across all temporal positions
        spatial_out = self.vectorized_attention(spatial_q, spatial_k, spatial_v)
        # (B, num_heads, T, HW+1, head_dim)
        
        # Reshape back: (B, T, HW+1, embed_dim)
        spatial_out = spatial_out.permute(0, 2, 3, 1, 4).reshape(B, T, HW_plus_1, self.embed_dim)
        spatial_out = self.spatial_proj(spatial_out)
        
        if single_sample:
            spatial_out = spatial_out.squeeze(0)  # Remove batch dimension
            
        return spatial_out

class BatchedMLP(nn.Module):
    def __init__(self, embed_dim, mlp_ratio, dropout):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """Handle both (B, T, HW+1, embed_dim) and (T, HW+1, embed_dim)"""
        original_shape = x.shape
        
        # Flatten for MLP processing
        if x.dim() == 4:  # Batch input
            B, T, HW_plus_1, embed_dim = x.shape
            x = x.view(B * T * HW_plus_1, embed_dim)
        else:  # Single sample
            T, HW_plus_1, embed_dim = x.shape  
            x = x.view(T * HW_plus_1, embed_dim)
        
        # Apply MLP
        x = self.mlp(x)
        
        # Reshape back
        x = x.view(original_shape)
        return x

class TransformerBlock(nn.Module):
    """ISTVT transformer block"""
    
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = DecomposedSpatialTemporalAttention(embed_dim, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = BatchedMLP(embed_dim, mlp_ratio, dropout)
    
    def forward(self, x):
        """
        x: (B, T+1, HW+1, embed_dim) for batch processing or (T+1, HW+1, embed_dim) for individual
        """
        # Handle both batch and individual sample inputs
        if x.dim() == 3:  # Individual sample: (T+1, HW+1, embed_dim)
            x = x.unsqueeze(0)  # Add batch dimension: (1, T+1, HW+1, embed_dim)
            squeeze_output = True
        else:  # Batch input: (B, T+1, HW+1, embed_dim)
            squeeze_output = False
        
        # Attention with residual
        x = x + self.attn(self.norm1(x))
        # MLP with residual  
        x = x + self.mlp(self.norm2(x))
        
        if squeeze_output:
            x = x.squeeze(0)  # Remove batch dimension for individual samples
        
        return x

class ISTVT(nn.Module):
    """Interpretable Spatial-Temporal Video Transformer"""
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.sequence_length = config.sequence_length
        self.embed_dim = config.embed_dim
        self.num_patches = 19 * 19  # From Xception output
        
        # Feature extractor
        self.feature_extractor = XceptionFeatureExtractor(pretrained=True)
        
        # Classification tokens
        self.spatial_cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.temporal_cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.prediction_cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        
        # Pos-emb now has one *extra* temporal row for the prediction-CLS
        # shape = (T + 2 , HW + 1 , C)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.sequence_length + 2, self.num_patches + 1, self.embed_dim))


        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout
            )
            for _ in range(config.num_layers)
        ])
        
        # Final norm and classifier
        self.norm = nn.LayerNorm(self.embed_dim)
        self.head = nn.Linear(self.embed_dim, config.num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        nn.init.trunc_normal_(self.spatial_cls_token, std=0.02)
        nn.init.trunc_normal_(self.temporal_cls_token, std=0.02)
        nn.init.trunc_normal_(self.prediction_cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass
        x: (B, T, C, H, W)
        """
        B, T, C, H, W = x.shape
        
        # Extract features
        x = x.view(B * T, C, H, W)
        features = self.feature_extractor(x)  # (B*T, 728, 19, 19)
        
        # Reshape and prepare tokens
        _, feat_dim, feat_h, feat_w = features.shape
        features = features.view(B, T, feat_dim, feat_h * feat_w)
        features = features.permute(0, 1, 3, 2)  # (B, T, HW, embed_dim)
        
        # 1) prepend spatial-CLS to every frame row
        spatial_cls = self.spatial_cls_token.expand(B, T, -1, -1)
        feature_tokens = torch.cat([spatial_cls, features], dim=2)          # (B ,T ,HW+1 ,C)

        # 2) create the temporal-CLS *row*
        temporal_cls = self.temporal_cls_token.expand(
            B, 1, self.num_patches + 1, -1                                   # (B ,1 ,HW+1 ,C)
        )

        # 3) create the single prediction-CLS *token* (only at position 0,0)
        # Initialize with temporal cls structure but will only use position (0,0)
        prediction_cls_row = self.prediction_cls_token.expand(
            B, 1, self.num_patches + 1, -1                                   # (B ,1 ,HW+1 ,C)
        )

        # Final token grid order:  [prediction-CLS] , [temporal-CLS] , [T frame rows]  
        tokens = torch.cat([prediction_cls_row, temporal_cls, feature_tokens], dim=1)

        # NOTE: Only position (0,0) contains the actual prediction token;
        # positions (0,1) to (0,HW) are unused but kept for shape consistency


        # Add position embeddings (simpler approach)
        tokens = tokens + self.pos_embed.expand(B, -1, -1, -1)
        
        # Process batch through transformer blocks
        batch_tokens = tokens  # (B, T+1, HW+1, embed_dim)

        # Apply transformer blocks to entire batch
        for block in self.blocks:
            batch_tokens = block(batch_tokens)  # Process entire batch at once

        # Extract classification tokens
        output = batch_tokens[:, 0, 0]  # (B, embed_dim) - prediction cls tokens
        output = self.norm(output)
        logits = self.head(output)
        
        return logits
