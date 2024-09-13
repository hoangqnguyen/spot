import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from timm.layers.patch_embed import PatchEmbed
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

class Conv3dAttention(nn.Module):
    def __init__(
        self, n_embd, n_head, kernel_size, spatial_downsampling=1, is_causal=False
    ):

        super().__init__()
        assert n_embd % n_head == 0
        self.c_qkv = nn.Conv3d(
            n_embd,
            n_embd * 3,
            kernel_size=kernel_size,
            bias=False,
            stride=(1, spatial_downsampling, spatial_downsampling),
            padding=(kernel_size // 2, kernel_size // 2, kernel_size // 2),
        )
        self.c_proj = nn.Linear(n_embd, n_embd)

        self.n_head = n_head
        self.n_embd = n_embd
        self.is_causal = is_causal

    def forward(self, x):
        B, C, T  = x.shape[:3]
        # print(x.shape)
        qkv = self.c_qkv(x)
        H, W = qkv.shape[-2:]
        q, k, v = rearrange(qkv, "b c t h w -> b (t h w) c").chunk(3, dim=-1)
        # return q 
        k = k.view(B, -1, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, -1, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, -1, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=self.is_causal
        )  # flash attention
        y = (
            y.transpose(1, 2).contiguous().view(B, -1, C)
        )  # re-assemble all head outputs side by side
        # output projection

        return rearrange(self.c_proj(y), "b (t h w) c -> b c t h w", t=T, h=H, w=W)

class Block(nn.Module):

    def __init__(self, n_embd, n_head, kernel_size=3, spatial_downsampling=2):
        super().__init__()
        self.norm_1 = nn.InstanceNorm3d(n_embd)
        self.attn = Conv3dAttention(
            n_embd,
            n_head,
            kernel_size=kernel_size,
            spatial_downsampling=spatial_downsampling,
        )
        self.norm_2 = nn.InstanceNorm3d(n_embd)
        self.prj = nn.Conv3d(n_embd, n_embd, kernel_size=1)
        self.res1 = nn.Identity() if spatial_downsampling == 1 else nn.Conv3d(
            n_embd, n_embd, kernel_size=1, stride=(1, spatial_downsampling, spatial_downsampling))
        self.res2 = nn.Identity() if spatial_downsampling == 1 else nn.Conv3d(
            n_embd, n_embd, kernel_size=1)

    def forward(self, x):
        x = self.res1(x) + self.attn(self.norm_1(x))
        x = self.res2(x) + self.prj(self.norm_2(x))
        return x


class Spotformer(nn.Module):
    def __init__(self, d_model, nhead, out_dim, dropout=0.1):
        super(Spotformer, self).__init__()
        self.patch_embed = PatchEmbed(
            img_size=224, patch_size=14, in_chans=3, embed_dim=d_model, flatten=False
        )
        self.pos_encoder = Summer(PositionalEncoding3D(d_model))
        self.encoder = nn.Sequential(*([Block(d_model, nhead, 3, spatial_downsampling=2)] * 4))
        self.fc = nn.Linear(d_model, out_dim)

    def print_stats(self):
        print(f"Model params:{sum(p.numel() for p in self.parameters()):,}")

    def forward(self, x):
        b, t = x.shape[:2]
        x = self.patch_embed(rearrange(x, "b t c h w -> (b t) c h w"))
        x = self.pos_encoder(rearrange(x, "(b t) c h w -> b h w t c", t=t))
        x = self.encoder(rearrange(x, "b h w t c -> b c t h w")).flatten(2, -1).permute(0, 2, 1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = Spotformer(d_model=512, nhead=4, num_layers=3, out_dim=7).cuda()
    x = torch.randn(2, 100, 3, 224, 224).cuda()
    y = model(x)

    # print model numels nicely
    print("Model params: {:,d}".format(sum(p.numel() for p in model.parameters())))
    print(f"In shape: {x.shape}, Out shape: {y.shape}")