import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from common import LayerNorm, gelu, split_last, merge_last

class ConvModule(nn.Module):
    """Convolutional Module for Conformer."""
    def __init__(self, cfg):
        super().__init__()
        self.layer_norm = LayerNorm(cfg)
        
        # Pointwise convolution
        self.pointwise_conv1 = nn.Conv1d(
            cfg.hidden, 2 * cfg.hidden, kernel_size=1, stride=1, padding=0
        )
        
        # Depthwise convolution
        kernel_size = 5 # Typically 15-31 for audio, we use 15 for IMU data
        self.depthwise_conv = nn.Conv1d(
            cfg.hidden, cfg.hidden, kernel_size=kernel_size,
            stride=1, padding=(kernel_size - 1) // 2, groups=cfg.hidden
        )
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(cfg.hidden)
        
        # Pointwise convolution
        self.pointwise_conv2 = nn.Conv1d(
            cfg.hidden, cfg.hidden, kernel_size=1, stride=1, padding=0
        )
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # x: [B, S, D]
        x = self.layer_norm(x)
        
        # Change to channel dimension
        x = x.transpose(1, 2)  # [B, D, S]
        
        # Pointwise conv and GLU activation
        x = self.pointwise_conv1(x)  # [B, 2D, S]
        x = F.glu(x, dim=1)  # [B, D, S]
        
        # Depthwise conv
        x = self.depthwise_conv(x)  # [B, D, S]
        x = self.batch_norm(x)
        x = gelu(x)
        
        # Pointwise conv
        x = self.pointwise_conv2(x)  # [B, D, S]
        x = self.dropout(x)
        
        # Return to sequence dimension
        x = x.transpose(1, 2)  # [B, S, D]
        return x

class ConformerBlock(nn.Module):
    """Conformer block combines self-attention with convolution."""
    def __init__(self, cfg):
        super().__init__()
        
        # Feed Forward Module (FFN) - first
        self.ffn1 = nn.Sequential(
            LayerNorm(cfg),
            nn.Linear(cfg.hidden, cfg.hidden_ff),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(cfg.hidden_ff, cfg.hidden),
            nn.Dropout(0.1)
        )
        
        # Multi-headed Self-Attention Module (MHSA)
        self.attn = nn.MultiheadAttention(
            embed_dim=cfg.hidden,
            num_heads=cfg.n_heads,
            dropout=0.1,
            batch_first=True
        )
        self.attn_norm = LayerNorm(cfg)
        self.attn_dropout = nn.Dropout(0.1)
        
        # Convolution Module (Conv)
        self.conv = ConvModule(cfg)
        
        # Feed Forward Module (FFN) - second
        self.ffn2 = nn.Sequential(
            LayerNorm(cfg),
            nn.Linear(cfg.hidden, cfg.hidden_ff),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(cfg.hidden_ff, cfg.hidden),
            nn.Dropout(0.1)
        )
        
        # Final Layer Norm
        self.final_norm = LayerNorm(cfg)
    
    def forward(self, x):
        # FFN module - first
        residual = x
        x = residual + 0.5 * self.ffn1(x)
        
        # MHSA module
        residual = x
        x = self.attn_norm(x)
        attn_output, _ = self.attn(x, x, x)
        x = residual + self.attn_dropout(attn_output)
        
        # Conv module
        residual = x
        x = residual + self.conv(x)
        
        # FFN module - second
        residual = x
        x = residual + 0.5 * self.ffn2(x)
        
        # Final layer norm
        x = self.final_norm(x)
        return x

class ConformerTransformer(nn.Module):
    """Transformer with Conformer Blocks for IMU data."""
    def __init__(self, cfg):
        super().__init__()
        self.embed = nn.Linear(cfg.feature_num, cfg.hidden)
        self.pos_embed = nn.Embedding(cfg.seq_len, cfg.hidden)
        self.nucleus_embed = nn.Embedding(2, cfg.hidden)
        self.emb_norm = LayerNorm(cfg)
        
        # Conformer blocks
        self.blocks = nn.ModuleList([ConformerBlock(cfg) for _ in range(cfg.n_layers)])
        
    def forward(self, x, nucleus_mask=None, sig_axis_mask=None):
        # Get device information
        device = x.device
        seq_len = x.size(1)
        
        # Position encoding
        pos = torch.arange(seq_len, dtype=torch.long, device=device)
        pos = pos.unsqueeze(0).expand(x.size(0), seq_len)  # [B, S]
        
        # Embedding and position encoding
        e = self.embed(x)
        e = e + self.pos_embed(pos)
        
        # Apply nucleus embedding if provided
        if nucleus_mask is not None:
            nucleus_mask = nucleus_mask.to(device)
            e = e + self.nucleus_embed(nucleus_mask)
        
        # Apply normalization
        h = self.emb_norm(e)
        
        # Pass through conformer blocks
        for block in self.blocks:
            h = block(h)
            
        return h