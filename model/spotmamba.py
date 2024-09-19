import math
import timm
import torch
from torch import nn
from mamba_ssm import Mamba
from einops import rearrange
from torchinfo import summary
from functools import partial
from timm.models.layers import PatchEmbed
from positional_encodings.torch_encodings import PositionalEncoding3D, Summer

class MLP(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, d_model, depth, d_state, d_conv, expand):
        super(Block, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model)

    def forward(self, x):
        x = x + self.mamba(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

def create_blocks(d_model, depth, d_state, d_conv, expand):
    return nn.Sequential(
        *[Block(d_model, depth, d_state, d_conv, expand) for _ in range(depth)]
    )

class SpotMamba(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        num_classes=3,
        depth=1,
        image_size=224,
        patch_size=16,
        d_state=14,
        d_conv=4,
        expand=2,
        use_positional_encoding=False,
        pool="max",
    ):
        assert pool in ["max", "mean"], "pool must be either 'max' or 'mean'"

        self.hidden_dim = hidden_dim
        self.depth = depth
        super(SpotMamba, self).__init__()

        self.patch_embed = PatchEmbed(
            img_size=image_size,
            patch_size=patch_size,
            in_chans=3,
            embed_dim=hidden_dim,
        )

        self.pos_enc = (
            nn.Identity()
            if not use_positional_encoding
            else Summer(PositionalEncoding3D(hidden_dim))
        )

        self.block = create_blocks(
            d_model=hidden_dim,
            depth=depth,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.pool = (
            partial(torch.max, dim=2) if pool == "max" else partial(torch.mean, dim=2)
        )
        self.fc_out = nn.Linear(hidden_dim, num_classes)

    def print_stats(self):
        print(f"Model params: {sum(p.numel() for p in self.parameters()):,}")

    def forward(self, x):
        bs, t = x.shape[:2]  # bs, t, c, h, w
        x = self.patch_embed(x.flatten(0, 1))  # bs*t, p^2, c
        p = int(math.sqrt(x.shape[1]))
        x = self.pos_enc(rearrange(x, "(bs t) (ph pw) c -> bs t ph pw c", bs=bs, ph=p))
        x = self.block(rearrange(x, "bs t ph pw c -> bs (t ph pw) c", bs=bs))
        x = rearrange(x, "bs (t P) c -> bs t P c", P=p**2)
        x = self.pool(x)
        if self.pool == torch.max:
            x = x.values
        x = self.fc_out(x)
        return x


if __name__ == "__main__":
    x = torch.randn(2, 15, 3, 224, 224)
    model = SpotMamba(
        num_classes=7, pool="mean", hidden_dim=512, depth=3, use_positional_encoding=True
    )
    summary(model, input_size=(2, 15, 3, 224, 224))
