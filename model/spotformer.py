import torch
import torch.nn as nn
import torch.nn.functional as F
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer
from model.common import MLP

class Spotformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, num_queries, out_dim, dropout=0.1):
        super(Spotformer, self).__init__()
        self.pos_encoder = Summer(PositionalEncoding1D(d_model))        
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, dropout=dropout, batch_first=True),
            num_layers
        )
        self.event_query = nn.Embedding(num_queries, d_model)
        self.mlp = MLP(d_model, d_model * 4, d_model, 1)
        self.fc_out = nn.Linear(d_model, out_dim)

    def forward(self, x, m=None):
        batch_size, clip_len, c = x.shape
        x = self.pos_encoder(x)
        query = self.event_query.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        x = self.decoder(query, x)
        x = self.mlp(x)
        x = self.fc_out(x)
        return x

if __name__ == '__main__':
    model = Spotformer(512, 8, 6, 10, 7)
    x = torch.randn(2, 100, 512)
    y = model(x)
    print(y.shape)