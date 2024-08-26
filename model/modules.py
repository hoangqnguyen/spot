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
            batch_size, clip_len, -1)


class GRUPrediction(nn.Module):

    def __init__(self, feat_dim, num_classes, hidden_dim, num_layers=1):
        super().__init__()
        self._gru = nn.GRU(
            feat_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            bidirectional=True)
        self._fc_out = FCPrediction(2 * hidden_dim, num_classes)
        self._dropout = nn.Dropout()

    def forward(self, x):
        y, _ = self._gru(x)
        return self._fc_out(self._dropout(y))


class TCNPrediction(nn.Module):

    def __init__(self, feat_dim, num_classes, num_stages=1, num_layers=5):
        super().__init__()

        self._tcn = SingleStageTCN(
            feat_dim, 256, num_classes, num_layers, True)
        self._stages = None
        if num_stages > 1:
            self._stages = nn.ModuleList([SingleStageTCN(
                num_classes, 256, num_classes, num_layers, True)
                for _ in range(num_stages - 1)])

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
            num_decoders, num_layers, r1, r2, num_f_maps, feat_dim,
            num_classes, channel_masking_rate=0.3)

    def forward(self, x):
        B, T, D = x.shape
        return self._net(
            x.permute(0, 2, 1), torch.ones((B, 1, T), device=x.device)
        ).permute(0, 1, 3, 2)
    


class ImprovedLocationPredictor(nn.Module):
    def __init__(self, in_channels, dropout_prob=0.5):
        super(ImprovedLocationPredictor, self).__init__()
        
        # Intermediate convolutional layers to capture more spatial features
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout2d(p=dropout_prob)
        
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout2d(p=dropout_prob)
        
        # Final 1x1 convolutional layers for objectness and coordinate predictions
        self.objectness_head = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.xy_head = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        # Pass through the first convolutional block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        # Pass through the second convolutional block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        # Predict objectness score
        objectness = self.objectness_head(x)
        
        # Predict coordinates (x, y)
        xy = self.xy_head(x)
        
        # Concatenate the objectness and coordinates predictions
        pred_loc = torch.cat((objectness, xy), dim=1)
        
        return pred_loc