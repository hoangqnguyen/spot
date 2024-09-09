import torch
import timm
from torch import nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from positional_encodings.torch_encodings import (
    PositionalEncoding3D,
    Summer,
)
from contextlib import nullcontext
from einops import rearrange
import time
import torch

class Conv1dCrossAttn(nn.Module):

    def __init__(
        self, n_embd, n_head, receptive_field=3, downsample_factor=2, is_causal=False
    ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_q = nn.Conv1d(n_embd, n_embd, kernel_size=1, stride=1, padding=0)
        self.c_kv = nn.Conv1d(
            n_embd,
            2 * n_embd,
            kernel_size=receptive_field,
            stride=downsample_factor,
            padding=1,
        )
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = n_head
        self.n_embd = n_embd
        self.is_causal = is_causal
        self.downsample_factor = downsample_factor

    def forward(self, tgt, src):
        B, T, C = (
            src.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        t = T // self.downsample_factor

        q = self.c_q(rearrange(tgt, "b t c -> b c t"))
        kv = self.c_kv(rearrange(src, "b t c -> b c t"))

        q = rearrange(q, "b t c -> b c t")
        kv = rearrange(kv, "b c t -> b t c")
        # print(x.shape)
        k, v = kv.split(self.n_embd, dim=2)

        q = q.view(B, -1, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        k = k.view(B, t, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, t, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=self.is_causal
        )  # flash attention
        y = (
            y.transpose(1, 2).contiguous().view(B, -1, C)
        )  # re-assemble all head outputs side by side
        # output projection

        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, n_embd, n_head, receptive_field=3, downsample_factor=2):
        super().__init__()
        self.ln_1_tgt = nn.LayerNorm(n_embd)
        self.ln_1_src = nn.LayerNorm(n_embd)
        self.attn = Conv1dCrossAttn(
            n_embd,
            n_head,
            receptive_field=receptive_field,
            downsample_factor=downsample_factor,
        )
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)
        self.downsample_factor = downsample_factor

    def forward(self, tgt, src):
        # print(f"{src.shape=}; {tgt.shape=}")
        tgt = tgt + self.attn(tgt=self.ln_1_tgt(tgt), src=self.ln_1_src(src))
        tgt = tgt + self.mlp(self.ln_2(tgt))
        return tgt


class ConvTransformer(nn.Module):
    def __init__(
        self, n_embd, depth, n_head, receptive_field=3, max_downsample_factor=2
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                Block(
                    n_embd,
                    n_head,
                    receptive_field=receptive_field,
                    downsample_factor=(
                        1 if (2 ** (d + 1)) >= max_downsample_factor else 2
                    ),
                )
                for d in range(depth)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd)

    def forward(self, tgt, src):
        for layer in self.layers:
            tgt = layer(tgt, src)
        return self.ln_f(tgt)


class SpotFormer(nn.Module):
    def __init__(
        self,
        num_classes,
        num_queries,
        cnn_model="regnety_008",
        hidden_dim=256,
        num_heads=8,
        num_layers=6,
        dropout=0.1,
        max_downsample_factor=16,
    ):
        super(SpotFormer, self).__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.backbone = CNNEncoder(cnn_model, out_c=hidden_dim)
        self.decoder = ConvTransformer(
            hidden_dim,
            depth=num_layers,
            n_head=num_heads,
            max_downsample_factor=max_downsample_factor,
        )
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.tloc_embed = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        x = self.backbone(x)  # B, T, hidden_dim
        x = self.decoder(
            tgt=self.query_embed.weight.unsqueeze(0).repeat(x.shape[0], 1, 1), src=x
        )
        return {
            "pred_logits": self.class_embed(x),
            "pred_tloc": self.tloc_embed(x),
        }


class CNNEncoder(nn.Module):
    def __init__(self, model_name, out_c):
        super(CNNEncoder, self).__init__()
        self.model_name = model_name
        self.out_c = out_c

        if model_name in ("resnet50", "resnet18", "regnety_008", "regnety_002"):
            self.backbone = timm.create_model(
                model_name, pretrained=True, num_classes=0, global_pool=""
            )
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        self.out_proj = nn.Conv2d(self.backbone.num_features, out_c, 1)
        self.cnn_features = self.backbone.num_features
        self.pos_xyt_encoding = Summer(PositionalEncoding3D(self.cnn_features))

    def forward(self, x):
        """
        Forward pass of the model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, time_steps, channels, height, width).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, height, width, time_steps, channels).
        """

        bs, t = x.shape[:2]
        x = self.backbone(x.flatten(0, 1))
        x = self.out_proj(x)
        x = self.pos_xyt_encoding(rearrange(x, "(b t) c h w -> b h w t c", b=bs))
        return x.flatten(1, 3)


@torch.no_grad()
def hungarian_matcher(outputs, targets, cost_weights=(1.0, 1.0), apply_sigmoid=True):
    w_cost_tloc, w_cost_class = cost_weights
    bs, num_queries = outputs["pred_logits"].shape[:2]
    # We flatten to compute the cost matrices in a batch
    out_prob = (
        outputs["pred_logits"].flatten(0, 1).softmax(-1)
    )  # [batch_size * num_queries, num_classes]
    out_tloc = outputs["pred_tloc"].flatten(0, 1)  # [batch_size * num_queries, 4]
    if apply_sigmoid:
        out_tloc = out_tloc.sigmoid()

    # Also concat the target labels and boxes
    tgt_ids = targets["cls"].flatten()
    tgt_tloc = targets["tloc"].flatten(0, 1)

    cost_class = -out_prob[:, tgt_ids]
    cost_tloc = torch.cdist(out_tloc, tgt_tloc, p=1)

    # Final cost matrix
    C = w_cost_tloc * cost_tloc + w_cost_class * cost_class
    C = C.view(bs, num_queries, -1).cpu()

    sizes = [v.shape[0] for v in targets["cls"]]
    indices = [
        linear_sum_assignment(c[i].cpu().numpy())
        for i, c in enumerate(C.split(sizes, -1))
    ]

    return [
        (
            torch.as_tensor(i, dtype=torch.int64, device=out_prob.device),
            torch.as_tensor(j, dtype=torch.int64, device=out_prob.device),
        )
        for i, j in indices
    ]


def get_src_permutation_idx(indices):
    # permute predictions following indices
    batch_idx = torch.cat(
        [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
    )
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx


def calc_loss_labels(outputs, targets, indices, num_classes, fg_weight=5.0):
    weights = torch.FloatTensor([1] + [fg_weight] * (num_classes)).to(
        outputs["pred_logits"].device
    )

    """Classification loss (NLL)
    targets dicts must contain the key "cls" containing a tensor of dim [nb_target_boxes]
    """
    assert "pred_logits" in outputs
    src_logits = outputs["pred_logits"]

    idx = get_src_permutation_idx(indices)
    target_classes_o = torch.cat([t[J] for t, (_, J) in zip(targets["cls"], indices)])
    target_classes = torch.full(
        src_logits.shape[:2], num_classes, dtype=torch.int64, device=src_logits.device
    )
    target_classes[idx] = target_classes_o

    loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes)

    return loss_ce


def get_src_permutation_idx(indices):
    # permute predictions following indices
    batch_idx = torch.cat(
        [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
    )
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx


def calc_loss_tloc(outputs, targets, indices, num_boxes, apply_sigmoid=True):
    """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
    targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
    The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
    """
    assert "pred_tloc" in outputs
    idx = get_src_permutation_idx(indices)
    src_boxes = outputs["pred_tloc"][idx]
    target_boxes = torch.cat(
        [t[i] for t, (_, i) in zip(targets["tloc"], indices)], dim=0
    )

    loss_bbox = F.l1_loss(
        src_boxes.sigmoid() if apply_sigmoid else src_boxes,
        target_boxes,
        reduction="none",
    )

    loss_bbox = loss_bbox.sum() / num_boxes

    return loss_bbox


def calc_loss(outputs, targets, num_classes, fg_weight=5.0):
    num_boxes = targets["cls"].shape[0] * targets["cls"].shape[1]
    indices = hungarian_matcher(outputs, targets)
    loss_ce = calc_loss_labels(outputs, targets, indices, num_classes, fg_weight)
    loss_tloc = calc_loss_tloc(outputs, targets, indices, num_boxes)
    loss = loss_ce + loss_tloc
    return {"loss": loss, "loss_ce": loss_ce, "loss_tloc": loss_tloc}


def predict(model, seq, use_amp=False, device="cuda"):
    if not isinstance(seq, torch.Tensor):
        seq = torch.FloatTensor(seq)
    if len(seq.shape) == 4:  # (L, C, H, W)
        seq = seq.unsqueeze(0)
    seq = seq.to(device)

    model.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast() if use_amp else nullcontext():
            pred_dict = model(seq)
            pred_logits = pred_dict["pred_logits"]

        pred_tloc = pred_dict["pred_tloc"].sigmoid()
        pred_cls = pred_logits.argmax(dim=-1)
        pred_score = pred_logits.softmax(dim=-1).max(dim=-1).values

        seq_len = seq.shape[1]
        results = []
        for i in range(pred_cls.shape[0]):
            batch_results = []
            for cls, score, tloc in zip(pred_cls[i], pred_score[i], pred_tloc[i]):
                batch_results.append(
                    dict(
                        cls=cls.item(),
                        score=score.item(),
                        frame=int(tloc[0].item() * seq_len),
                        xy=(tloc[1].item(), tloc[2].item()),
                    )
                )
            results.append(batch_results)

        return results


if __name__ == "__main__":
    # Set device to CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 4
    num_classes = 6
    num_queries = 8
    sequence_length = 32
    num_events = 5

    model = SpotFormer(
        num_classes=num_classes + 1,
        num_queries=num_queries,
        cnn_model="regnety_008",
        hidden_dim=512,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params:,} total parameters.")

    x = torch.randn(batch_size, sequence_length, 3, 224, 224).to(device)
    y = dict(
        cls=torch.randint(
            0,
            num_classes,
            (
                batch_size,
                num_events,
            ),
        ).to(device),
        tloc=torch.randn(batch_size, num_events, 3).to(device),
    )

    print("Forward pass")
    start_time = time.time()
    out = model(x)
    torch.cuda.synchronize()  # Synchronize CUDA operations
    end_time = time.time()
    print(out["pred_logits"].shape, out["pred_tloc"].shape)
    print(f"Forward pass took {end_time - start_time:.4f} seconds")

    calc_loss = calc_loss(out, y, num_classes)
    print(calc_loss)
    print("Backward pass")
    start_time = time.time()
    calc_loss["loss"].backward()
    torch.cuda.synchronize()  # Synchronize CUDA operations
    end_time = time.time()
    print(f"Backward pass took {end_time - start_time:.4f} seconds")

    # Test prediction
    preds = predict(model, x)
    print(f"Predictions shape: {len(preds)}")
