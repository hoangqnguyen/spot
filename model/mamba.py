import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
from torch.nn import functional as F
from mamba_ssm import Mamba
from einops import rearrange


@dataclass
class STMambaConfig:
    in_dim: int = 256
    n_layers_spatial: int = 6
    n_layers_temporal: int = 6
    n_embd: int = 128
    bias: bool = True
    dropout: float = 0.1
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class Block(nn.Module):
    def __init__(
        self,
        n_embd: int,
        bias: bool = True,
        dropout: float = 0.1,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = LayerNorm(n_embd, bias=bias)
        self.mam = Mamba(
            d_model=n_embd,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )
        self.ln_2 = LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(n_embd, bias, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mam(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MLP(nn.Module):
    def __init__(self, n_embd: int, bias: bool, dropout: float, **kwargs):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class STMambaEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        n_layers_spatial: int,
        n_layers_temporal: int,
        n_embd: int,
        bias: bool = True,
        dropout: float = 0.1,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.in_proj = nn.Conv2d(in_dim, n_embd, kernel_size=1)
        self.frame_token = nn.Parameter(torch.empty(1, 1, n_embd))
        nn.init.xavier_uniform_(self.frame_token)  # Improved initialization
        self.spatial_blocks = nn.Sequential(
            *[
                Block(n_embd, bias, dropout, d_state, d_conv, expand)
                for _ in range(n_layers_spatial)
            ]
        )
        self.temporal_blocks = nn.Sequential(
            *[
                Block(n_embd, bias, dropout, d_state, d_conv, expand)
                for _ in range(n_layers_temporal)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, in_dim, H, W)

        Returns:
            x: (B, T, n_embd)
        """
        B, T, C, H, W = x.shape
        x = self.in_proj(x.flatten(0, 1))  # (B*T, n_embd, H, W)
        x = rearrange(x, "b c h w -> b (h w) c")  # (B*T, H*W, n_embd)
        frame_token = self.frame_token.expand(B * T, 1, self.in_proj.out_channels)
        x = torch.cat([frame_token, x], dim=1)  # (B*T, 1 + H*W, n_embd)
        x = self.spatial_blocks(x)[:, 0, :]  # (B*T, n_embd)
        x = rearrange(x, "(b t) f -> b t f", b=B, t=T)  # (B, T, n_embd)
        x = self.temporal_blocks(x)  # (B, T, n_embd)
        return x


def test_stmamba_encoder():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, T, C, H, W = 4, 8, 384, 7, 7
    mamba_cfg = STMambaConfig(
        in_dim=C,
        n_embd=128,
        n_layers_spatial=6,
        n_layers_temporal=6,
        bias=True,
        dropout=0.1,
        d_state=16,
        d_conv=4,
        expand=2,
    )

    x = torch.randn(B, T, C, H, W).to(device)
    model = STMambaEncoder(**asdict(mamba_cfg)).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print("Mamba config: ", mamba_cfg)
    print(f"Total number of parameters: {total_params:,}")
    print("Input shape: ", x.shape)  # torch.Size([4, 8, 384, 7, 7])
    output = model(x)
    print("Output shape: ", output.shape)  # Expected: torch.Size([4, 8, 128])
    assert output.shape == (
        B,
        T,
        mamba_cfg.n_embd,
    ), f"Expected shape {(B, T, mamba_cfg.n_embd)}, got {output.shape}"
    print("Test passed.")


if __name__ == "__main__":
    test_stmamba_encoder()
