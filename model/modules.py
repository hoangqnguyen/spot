import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import SingleStageTCN
from .impl.asformer import MyTransformer


class FCPrediction(nn.Module):

    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self._fc_out = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        batch_size, clip_len, _ = x.shape
        return self._fc_out(x.reshape(batch_size * clip_len, -1)).view(
            batch_size, clip_len, -1
        )


class GRUPrediction(nn.Module):

    def __init__(self, feat_dim, num_classes, hidden_dim, num_layers=1):
        super().__init__()
        self._gru = nn.GRU(
            feat_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self._fc_out = FCPrediction(2 * hidden_dim, num_classes)
        self._dropout = nn.Dropout()

    def forward(self, x):
        y, _ = self._gru(x)
        return self._fc_out(self._dropout(y))


class TCNPrediction(nn.Module):

    def __init__(self, feat_dim, num_classes, num_stages=1, num_layers=5):
        super().__init__()

        self._tcn = SingleStageTCN(feat_dim, 256, num_classes, num_layers, True)
        self._stages = None
        if num_stages > 1:
            self._stages = nn.ModuleList(
                [
                    SingleStageTCN(num_classes, 256, num_classes, num_layers, True)
                    for _ in range(num_stages - 1)
                ]
            )

    def forward(self, x):
        x = self._tcn(x)
        if self._stages is None:
            return x
        else:
            outputs = [x]
            for stage in self._stages:
                x = stage(F.softmax(x, dim=2))
                outputs.append(x)
            return torch.stack(outputs, dim=0)


class ASFormerPrediction(nn.Module):

    def __init__(self, feat_dim, num_classes, num_decoders=3, num_layers=5):
        super().__init__()

        r1, r2 = 2, 2
        num_f_maps = 64
        self._net = MyTransformer(
            num_decoders,
            num_layers,
            r1,
            r2,
            num_f_maps,
            feat_dim,
            num_classes,
            channel_masking_rate=0.3,
        )

    def forward(self, x):
        B, T, D = x.shape
        return self._net(
            x.permute(0, 2, 1), torch.ones((B, 1, T), device=x.device)
        ).permute(0, 1, 3, 2)


from positional_encodings.torch_encodings import PositionalEncoding2D, Summer
from model.common import MLP


class ImprovedLocationPredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=2, nhead=4, num_layers=3):
        super(ImprovedLocationPredictor, self).__init__()

        self.in_proj = nn.Conv2d(in_dim, hidden_dim, kernel_size=1)

        self.pos_2d_enc = Summer(PositionalEncoding2D(hidden_dim))

        self.pos_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=nhead, batch_first=True
            ),
            num_layers=num_layers,
        )

        self.mlp = MLP(hidden_dim, hidden_dim * 4, hidden_dim, num_layers)
        self.fc_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        # breakpoint()
        x = self.in_proj(x).permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.pos_2d_enc(x)  # (B, H, W, C)
        x = x.reshape(B, H * W, -1)  # (B, H * W, C)
        pos_token = self.pos_token.repeat(B, 1, 1)
        x = torch.cat([pos_token, x], dim=1)  # (B, 1 + H * W C)
        x = self.encoder(x)  # (B, 1 + H * W, C)
        x = x[:, 0, :]  # (B, C)
        x = self.mlp(x)  # (B, C)
        x = self.fc_out(x) # (B, out_dim)
        return x