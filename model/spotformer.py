import torch
from torch import nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from positional_encodings.torch_encodings import (
    PositionalEncoding1D,
    Summer,
)
from contextlib import nullcontext


class SpotFormer(nn.Module):
    def __init__(
        self,
        num_classes,
        num_queries,
        input_dim,
        hidden_dim=256,
        num_heads=8,
        num_layers=6,
        dropout=0.1,
    ):
        super(SpotFormer, self).__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.pos_enc = Summer(PositionalEncoding1D(hidden_dim))
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.tloc_embed = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        x = self.encoder(x)  # B, T, hidden_dim
        x = self.pos_enc(x)
        x = self.decoder(
            tgt=self.query_embed.weight.unsqueeze(0).repeat(x.shape[0], 1, 1), memory=x
        )
        return {
            "pred_logits": self.class_embed(x),
            "pred_tloc": self.tloc_embed(x),
        }


@torch.no_grad()
def hungarian_matcher(outputs, targets, cost_weights=(1.0, 1.0), apply_sigmoid=True):
    # breakpoint()
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
    # breakpoint()
    C = w_cost_tloc * cost_tloc + w_cost_class * cost_class
    C = C.view(bs, num_queries, -1).cpu()
    try:
        sizes = [v.shape[0] for v in targets["cls"]]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
    except:
        breakpoint()
    return [
        (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
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

    # print(f"{src_logits.shape=}, {num_classes=}, {target_classes.max()=}, {weights=}")

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
    if seq.device != device:
        seq = seq.to(device)

    model.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast() if use_amp else nullcontext():
            pred_dict = model(seq)
            pred_logits = pred_dict["pred_logits"]

        pred_tloc = pred_dict["pred_tloc"].sigmoid().detach().cpu().numpy()
        pred_cls = pred_logits.argmax(axis=-1).detach().cpu().numpy()
        pred_score = (
            pred_logits.softmax(axis=-1).max(axis=-1).values.detach().cpu().numpy()
        )
        seq_len = seq.shape[1]
        return [
            [
                dict(
                    cls=cls,
                    score=score,
                    frame=int(tloc[0] * seq_len),
                    xy=(tloc[1], tloc[2]),
                )
                for cls, score, tloc in zip(pred_cls[i], pred_score[i], pred_tloc[i])
            ]
            for i in range(pred_cls.shape[0])
        ]


if __name__ == "__main__":
    batch_size = 8
    num_classes = 6
    num_queries = 20
    input_dim = 386
    sequence_length = 64
    num_events = 5

    model = SpotFormer(
        num_classes=num_classes + 1, num_queries=num_queries, input_dim=input_dim
    ).cuda()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params:,} total parameters.")

    x = torch.randn(batch_size, sequence_length, input_dim).cuda()
    y = dict(
        cls=torch.randint(
            0,
            num_classes,
            (
                batch_size,
                num_events,
            ),
        ).cuda(),
        tloc=torch.randn(batch_size, num_events, 3).cuda(),
    )

    out = model(x)
    print(out["pred_logits"].shape, out["pred_tloc"].shape)

    calc_loss = calc_loss(out, y, num_classes)
    print(calc_loss)

    preds = predict(model, x)
    breakpoint()
