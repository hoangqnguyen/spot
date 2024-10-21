import torch
from torch import nn


class FrameOrderRegularizer(nn.Module):
    def __init__(self, dim, clip_len):
        super(FrameOrderRegularizer, self).__init__()
        self.dim = dim
        self.clip_len = clip_len
        self.order_pred = nn.Linear(dim, clip_len)

    def forward(self, x):
        """
        Params:
            x: (B, T, D) tensor
        """
        B, T, D = x.shape

        # permute x in T dimension
        perm_idx = torch.randperm(T).to(x.device)
        tgt_idx = perm_idx.unsqueeze(0).repeat(B, 1)
        x_perm = x[:, perm_idx, :]

        # predict the order of the permuted x
        order_pred = self.order_pred(x_perm)
        loss_order = nn.CrossEntropyLoss()(order_pred, tgt_idx)
        return loss_order


if __name__ == "__main__":
    B, T, D = 2, 16, 128
    x = torch.randn(B, T, D)
    model = FrameOrderRegularizer(D, T)
    loss = model(x)
    print(loss)
