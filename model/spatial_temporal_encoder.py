import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    """
    Squeeze-and-Excitation Spatial Attention Module.
    
    Args:
        feat_dim (int): Dimension of the input features.
        reduction (int): Reduction ratio for the bottleneck in SE block.
    """
    def __init__(self, feat_dim, reduction=8):
        super(SpatialAttention, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // reduction, feat_dim, bias=True),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Forward pass for Spatial Attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, T, F]
        
        Returns:
            torch.Tensor: Recalibrated tensor of shape [B, T, F]
        """
        B, T, F = x.size()
        # Squeeze: Global Average Pooling over the temporal dimension
        squeeze = x.mean(dim=1)  # [B, F]
        # Excitation: Learn channel-wise weights
        excitation = self.fc(squeeze)  # [B, F]
        # Scale: Recalibrate the original features
        excitation = excitation.unsqueeze(1)  # [B, 1, F]
        x = x * excitation  # [B, T, F]
        return x


class SpatialTemporalAttnBlock(nn.Module):
    """
    Spatial-Temporal Attention Block integrating Spatial and Temporal Attention.
    
    Args:
        feat_dim (int): Dimension of the input features.
        reduction (int): Reduction ratio for the Spatial Attention.
        num_heads (int): Number of attention heads for Temporal Attention.
        dropout (float): Dropout rate.
        prenorm (bool): If True, applies LayerNorm before each attention sub-layer.
    """
    def __init__(self, feat_dim, reduction=8, num_heads=8, dropout=0.1, prenorm=True):
        super(SpatialTemporalAttnBlock, self).__init__()
        self.prenorm = prenorm
        
        # Spatial Attention
        self.norm = nn.LayerNorm(feat_dim) if prenorm else nn.Identity()
        self.spatial_attn = SpatialAttention(feat_dim, reduction)
        self.spatial_dropout = nn.Dropout(dropout)
        
        # Temporal Attention
        self.temporal_attn = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.temporal_dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass for the Spatial-Temporal Attention Block.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, T, F]
        
        Returns:
            torch.Tensor: Output tensor of shape [B, T, F]
        """
        # ----- Spatial Attention -----
        residual = x
        x_norm = self.norm(x)  # [B, T, F]
        x_spatial = self.spatial_attn(x_norm)  # [B, T, F]
        x_spatial = self.spatial_dropout(x_spatial)
        x = residual + x_spatial  # Residual connection
        
        # ----- Temporal Attention -----
        residual = x
        attn_output, _ = self.temporal_attn(x_norm, x_norm, x_norm)  # [B, T, F]
        attn_output = self.temporal_dropout(attn_output)
        x = residual + attn_output  # Residual connection
        
        return x

class SpatialTemporalEncoder(nn.Module):
    """
    Spatial-Temporal Encoder consisting of multiple SpatialTemporalAttnBlocks.
    
    Args:
        feat_dim (int): Dimension of the input features.
        num_layers (int): Number of SpatialTemporalAttnBlocks to stack.
        reduction (int): Reduction ratio for Spatial Attention.
        num_heads (int): Number of attention heads for Temporal Attention.
        dropout (float): Dropout rate.
        prenorm (bool): If True, applies LayerNorm before each attention sub-layer.
    """
    def __init__(self, feat_dim, hidden_dim=256, num_layers=4, reduction=8, num_heads=8, dropout=0.1, prenorm=True):
        super(SpatialTemporalEncoder, self).__init__()
        self.ft_projection = nn.Linear(feat_dim, hidden_dim)
        self.layers = nn.ModuleList([
            SpatialTemporalAttnBlock(
                feat_dim=hidden_dim,
                reduction=reduction,
                num_heads=num_heads,
                dropout=dropout,
                prenorm=prenorm
            ) for _ in range(num_layers)
        ])
        
    def forward(self, x):
        """
        Forward pass for the Spatial-Temporal Encoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, T, F]
        
        Returns:
            torch.Tensor: Output tensor of shape [B, T, F]
        """
        for layer in self.layers:
            x = layer(x)
        return x

if __name__ == "__main__":
    # some test codes
    import torchinfo

    B, T, F = 2, 64, 384
    x = torch.randn(B, T, F)
    encoder = SpatialTemporalEncoder(feat_dim=F, num_layers=3)
    torchinfo.summary(encoder, input_data=x)