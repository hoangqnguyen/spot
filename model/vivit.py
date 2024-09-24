import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F


class TubletEmbedding(nn.Module):
    def __init__(self, embed_dim, patch_size):
        super().__init__()
        self.projection = nn.Conv3d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, frames):
        """
        frames: (b, t, 3, h, w)
        """
        x = rearrange(frames, "b t c h w -> b c t h w")
        x = self.projection(x)
        return rearrange(x, "b c t h w -> b (t h w) c")


class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim, max_len=1024):
        super().__init__()
        self.encoder = nn.Embedding(num_embeddings=max_len, embedding_dim=embed_dim)
        self.register_buffer("pos", torch.arange(0, max_len, 1))

    def forward(self, x, sum=True):
        """
        x: (b, l, d)
        """
        pos_enc = self.encoder(self.pos)[None, : x.shape[1], :]
        if sum:
            return x + pos_enc
        return pos_enc


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class ViViT(nn.Module):
    def __init__(self, seq_len, patch_size, emb_dim, depth, nhead, dropout=0.1, max_len=1024) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.tub_emb = TubletEmbedding(emb_dim, patch_size)
        self.mem_pos = PositionalEncoder(emb_dim, max_len)
        self.tgt_pos = PositionalEncoder(emb_dim, seq_len)
        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=emb_dim,
                nhead=nhead,
                dropout=dropout,
                batch_first=True,
                norm_first=False,
            ),
            num_layers=depth,
        )
        self.queries = nn.Embedding(num_embeddings=seq_len, embedding_dim=emb_dim)

    def forward(self, frames):
        """
        frames: (b, t, 3, h, w)
        """
        x = self.tub_emb(frames)
        x = self.mem_pos(x, sum=True)
        tgt = self.queries.weight[None, :, :].repeat(x.shape[0], 1, 1)
        tgt = self.tgt_pos(tgt, sum=True)
        hs = self.decoder(tgt=tgt, memory=x)
        return hs


class ViViTSpot(nn.Module):
    def __init__(self, num_classes, predict_location=True, **kwargs) -> None:
        super().__init__()
        self.arch = ViViT(**kwargs)
        emb_dim = self.arch.emb_dim
        self.pred_cls = MLP(emb_dim, emb_dim * 2, num_classes, num_layers=3)
        self._predict_location = predict_location
        if predict_location:
            self.pred_xy = MLP(emb_dim, emb_dim * 2, 2, num_layers=3)

    def print_stats(self):
        print(f"Model params: {sum(p.numel() for p in self.parameters()):,d}")

    def forward(self, frames):
        """
        frames: (b, t, 3, h, w)
        """
        hs = self.arch(frames)
        cls_logits = self.pred_cls(hs)
        if hasattr(self, "pred_xy"):
            xy_logits = self.pred_xy(hs)
            return {"im_feat": cls_logits, "loc_feat": xy_logits}
        return {"im_feat": cls_logits}

if __name__ == "__main__":
    b, t, h, w = 8, 64, 224, 224
    frames = torch.randn(b, t, 3, h, w).cuda()
    vspot = ViViTSpot(num_classes=7, seq_len=t, max_len=1024, predict_location=True, patch_size=16, emb_dim=384, depth=3, nhead=4, dropout=0.1).cuda()
    vspot.print_stats()
    pred_dict = vspot(frames)
    for k, v in pred_dict.items():
        print(f"{k}: {v.shape}")