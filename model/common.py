import abc
import torch
import torch.nn as nn
import torch.nn.functional as F


class ABCModel:

    @abc.abstractmethod
    def get_optimizer(self, opt_args):
        raise NotImplementedError()

    @abc.abstractmethod
    def epoch(self, loader, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def predict(self, seq):
        raise NotImplementedError()

    @abc.abstractmethod
    def state_dict(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def load(self, state_dict):
        raise NotImplementedError()


class BaseRGBModel(ABCModel):

    def get_optimizer(self, opt_args):
        base_lr = opt_args.get('lr', 1e-3)
        pred_loc_lr = base_lr / 5

        param_groups = [
                {'params': param, 'lr': pred_loc_lr if '_pred_loc' in name else base_lr}
                for name, param in self._model.named_parameters()
            ]
        return torch.optim.AdamW(param_groups), \
            torch.cuda.amp.GradScaler() if self.device == 'cuda' else None

    """ Assume there is a self._model """

    def _get_params(self):
        return list(self._model.parameters())

    def state_dict(self):
        if isinstance(self._model, nn.DataParallel):
            return self._model.module.state_dict()
        return self._model.state_dict()

    def load(self, state_dict):
        if isinstance(self._model, nn.DataParallel):
            self._model.module.load_state_dict(state_dict)
        else:
            self._model.load_state_dict(state_dict)


def step(optimizer, scaler, loss, lr_scheduler=None, backward_only=False):
    
    if scaler is None:
        loss.backward()
    else:
        scaler.scale(loss).backward()

    if not backward_only:
        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()
        if lr_scheduler is not None:
            lr_scheduler.step()
        optimizer.zero_grad()


class SingleStageGRU(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=5):
        super(SingleStageGRU, self).__init__()
        self.backbone = nn.GRU(
            in_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            bidirectional=True)
        self.fc_out = nn.Sequential(
            nn.BatchNorm1d(2 * hidden_dim),
            nn.Dropout(),
            nn.Linear(2 * hidden_dim, out_dim))

    def forward(self, x):
        batch_size, clip_len, _ = x.shape
        x, _ = self.backbone(x)
        x = self.fc_out(x.reshape(-1, x.shape[-1]))
        return x.view(batch_size, clip_len, -1)


class SingleStageTCN(nn.Module):

    class DilatedResidualLayer(nn.Module):
        def __init__(self, dilation, in_channels, out_channels):
            super(SingleStageTCN.DilatedResidualLayer, self).__init__()
            self.conv_dilated = nn.Conv1d(
                in_channels, out_channels, 3, padding=dilation,
                dilation=dilation)
            self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
            self.dropout = nn.Dropout()

        def forward(self, x, mask):
            out = F.relu(self.conv_dilated(x))
            out = self.conv_1x1(out)
            out = self.dropout(out)
            return (x + out) * mask[:, 0:1, :]

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dilate):
        super(SingleStageTCN, self).__init__()
        self.conv_1x1 = nn.Conv1d(in_dim, hidden_dim, 1)
        self.layers = nn.ModuleList([
            SingleStageTCN.DilatedResidualLayer(
                2 ** i if dilate else 1, hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        self.conv_out = nn.Conv1d(hidden_dim, out_dim, 1)

    def forward(self, x, m=None):
        batch_size, clip_len, _ = x.shape
        if m is None:
            m = torch.ones((batch_size, 1, clip_len), device=x.device)
        else:
            m = m.permute(0, 2, 1)
        x = self.conv_1x1(x.permute(0, 2, 1))
        for layer in self.layers:
            x = layer(x, m)
        x = self.conv_out(x) * m[:, 0:1, :]
        return x.permute(0, 2, 1)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    

class ImprovedLocationPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=2):
        super(ImprovedLocationPredictor, self).__init__()
        # Initial Layer
        self.initial_layer = nn.Linear(input_dim, hidden_dim)
        
        # Residual Block 1
        self.res_block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)  # Batch normalization after first block
        
        # Residual Block 2
        self.res_block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)  # Batch normalization after second block

        # Final Layers
        self.dropout = nn.Dropout(0.3)  # Dropout for regularization
        self.output_layer = nn.Linear(hidden_dim, output_dim)  # Final output layer without batch normalization

    def forward(self, x):
        # Initial layer
        if x.dim() == 3:
            x = x.flatten(0,1)
        x = self.initial_layer(x)
        # check if an element of x is nan
        print("x min max", x.min(), x.max())    
        
        # Residual Block 1 with skip connection
        residual = x
        x = self.res_block1(x)
        if torch.isnan(x).any():
            print("x min max", x.min(), x.max())    
            raise Exception("nan in x after res_block1")
        x = self.batch_norm1(x)
        if torch.isnan(x).any():
            print("x min max", x.min(), x.max())    
            # print bn parameters
            print(self.batch_norm1)
            raise Exception("nan in x after res_block1 bn" )
        x = x + residual  # Add skip connection
        # check if an element of x is nan
        if torch.isnan(x).any():
            print("x min max", x.min(), x.max())    
            raise Exception("nan in x after res_block1 res+" )

        # Residual Block 2 with skip connection
        residual = x
        x = self.res_block2(x)
        x = self.batch_norm2(x)
        x = x + residual  # Add skip connection
        # check if an element of x is nan
        if torch.isnan(x).any():
            raise Exception("nan in x after res_block2")
            print(x)


        # Dropout and Final Output
        x = self.dropout(x)
        # check if an element of x is nan
        if torch.isnan(x).any():
            raise Exception("nan in x after dropout")
            print(x)
        x = self.output_layer(x)  # Final output without batch normalization
        
        return x
