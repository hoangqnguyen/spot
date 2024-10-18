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
    Spatial-Temporal Attention Block integrating Spatial and Temporal Attention using
    torch.nn.functional.scaled_dot_product_attention with multi-head support.
    
    Args:
        feat_dim (int): Dimension of the input features.
        reduction (int): Reduction ratio for the Spatial Attention.
        num_heads (int): Number of attention heads for Temporal Attention.
        dropout (float): Dropout rate.
        prenorm (bool): If True, applies LayerNorm before each attention sub-layer.
    """
    def __init__(self, feat_dim, reduction=8, num_heads=8, dropout=0.1, prenorm=True):
        super(SpatialTemporalAttnBlock, self).__init__()
        self.num_heads = num_heads
        self.head_dim = feat_dim // num_heads
        assert self.head_dim * num_heads == feat_dim, "feat_dim must be divisible by num_heads"

        self.prenorm = prenorm
        
        # Spatial Attention
        self.norm1 = nn.LayerNorm(feat_dim) if prenorm else nn.Identity()
        self.spatial_attn = SpatialAttention(feat_dim, reduction)
        self.spatial_dropout = nn.Dropout(dropout)
        
        # Temporal Attention
        self.norm2 = nn.LayerNorm(feat_dim) if prenorm else nn.Identity()
        self.temporal_dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass for the Spatial-Temporal Attention Block.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, T, F]
        
        Returns:
            torch.Tensor: Output tensor of shape [B, T, F]
        """
        B, T, F = x.shape
        
        # ----- Spatial Attention -----
        residual = x
        x_spatial = self.norm1(x)  # [B, T, F]
        x_spatial = self.spatial_attn(x_spatial)  # [B, T, F]
        x_spatial = self.spatial_dropout(x_spatial)
        
        # ----- Temporal Attention using scaled_dot_product_attention with multi-head support -----
        x_temporal = self.norm2(x)  # [B, T, F]
        
        # Split into multiple heads for multi-head attention: [B, T, num_heads, head_dim]
        x_temporal = x_temporal.view(B, T, self.num_heads, self.head_dim)
        
        # Permute to [num_heads, T, B, head_dim] for compatibility with scaled_dot_product_attention
        x_temporal = x_temporal.permute(2, 1, 0, 3).contiguous().view(self.num_heads, T, B, self.head_dim)
        
        # Apply scaled dot product attention per head
        q = k = v = x_temporal
        attn_output = nn.functional.scaled_dot_product_attention(q, k, v)  # [num_heads, T, B, head_dim]
        
        # Reshape back: [B, T, F]
        attn_output = attn_output.view(self.num_heads, T, B, self.head_dim).permute(2, 1, 0, 3).contiguous()
        attn_output = attn_output.view(B, T, F)  # Combine heads back
        
        attn_output = self.temporal_dropout(attn_output)
        
        # Combine spatial and temporal attention with residual connection
        x = residual + x_spatial + attn_output
        
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
    def __init__(self, feat_dim, hidden_dim, num_layers=4, reduction=8, num_heads=8, dropout=0.1, prenorm=True):
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
        x = self.ft_projection(x)
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