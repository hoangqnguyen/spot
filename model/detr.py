import timm
import torch
import math
from torch import nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from model.common import MLP


class DeTRPrediction(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone,
        hidden_dim=256,
        num_queries=100,
        dropout=0.1,
        nheads=8,
        depth=6,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes

        # Backbone
        self.backbone = timm.create_model(
            {
                "rny002": "regnety_002",
                "rny008": "regnety_008",
            }[backbone.rsplit("_", 1)[0]],
            pretrained=True,
        )
        feat_dim = self.backbone.head.fc.in_features
        self.backbone.head.fc = nn.Identity()

        self.in_proj = nn.Linear(feat_dim, hidden_dim)
        # Transformer Decoder
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=nheads,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                activation="relu",
            ),
            depth,
        )

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)

        # Query Embeddings
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # Prediction Heads
        # Predict normalized frame index
        self.frame_embed = MLP(hidden_dim, hidden_dim * 2, 1, 2, dropout=dropout)
        self.class_embed = MLP(hidden_dim, hidden_dim * 2, num_classes, 2, dropout=dropout)

    def forward(self, frames):
        # src shape: (batch_size, seq_len, feat_dim)
        batch_size, clip_len, channels, height, width = frames.shape

        src = self.backbone(frames.view(-1, channels, height, width)).reshape(
            batch_size, clip_len, -1
        )

        # Prepare inputs for the Transformer
        src = self.in_proj(src)  # (batch_size, seq_len, feat_dim)
        src = src.permute(1, 0, 2)  # (seq_len, batch_size, feat_dim)
        src = self.pos_encoder(src)

        # Prepare query embeddings
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(
            1, batch_size, 1
        )  # (num_queries, batch_size, feat_dim)

        # Transformer forward pass
        # memory = self.transformer.encoder(src)
        # hs = self.transformer.decoder(query_embed, memory)
        hs = self.transformer(tgt=query_embed, memory=src)

        # Output embeddings
        # (num_queries, batch_size, num_classes)
        outputs_class = self.class_embed(hs)
        outputs_frame = self.frame_embed(hs).squeeze(-1)  # (num_queries, batch_size)

        # Transpose back to (batch_size, num_queries, ...)
        outputs_class = outputs_class.permute(
            1, 0, 2
        )  # (batch_size, num_queries, num_classes)
        outputs_frame = outputs_frame.permute(1, 0)  # (batch_size, num_queries)

        return {"pred_logits": outputs_class, "pred_frames": outputs_frame.sigmoid()}

    def forward_train(self, batch, fg_weight=5.0):
        # get device of this model
        device = next(self.parameters()).device

        # Shape: (batch_size, clip_len, C, H, W)
        frames = batch["frames"].to(device)
        targets = {
            "labels": batch["labels"],  # List of tensors (length batch_size)
            # List of tensors (length batch_size)
            "timesteps": batch["timesteps"],
        }

        # Move targets to device
        targets["labels"] = [t.to(device) for t in targets["labels"]]
        targets["timesteps"] = [t.to(device) for t in targets["timesteps"]]

        # Forward pass
        outputs = self.forward(frames)

        # Matching
        indices = hungarian_match(outputs, targets)

        # Compute loss
        loss = compute_loss(outputs, targets, indices, self.num_classes, fg_weight)

        return outputs, loss

    def print_stats(self):
        print(f"Model params: {sum(p.numel() for p in self.parameters()):,}")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model)
        x = x + self.pe[: x.size(0)].unsqueeze(1)
        return self.dropout(x)


def hungarian_match(outputs, targets):
    bs, num_queries = outputs["pred_logits"].shape[:2]
    device = outputs["pred_logits"].device

    # Apply softmax to get probabilities
    pred_probs = outputs["pred_logits"].softmax(
        -1
    )  # Shape: (bs, num_queries, num_classes)
    pred_frames = outputs["pred_frames"]  # Shape: (bs, num_queries)

    indices = []
    for b in range(bs):
        # Ground truth for this batch element
        tgt_labels = targets["labels"][b]  # Shape: (num_gt_events,)
        tgt_frames = targets["timesteps"][b]  # Shape: (num_gt_events,)

        num_gt = tgt_labels.shape[0]

        if num_gt == 0:
            # If no ground truth, all predictions are unmatched
            indices.append(
                (
                    torch.as_tensor([], dtype=torch.int64),
                    torch.as_tensor([], dtype=torch.int64),
                )
            )
            continue

        # Predictions for this batch element
        out_prob = pred_probs[b]  # Shape: (num_queries, num_classes)
        out_frames = pred_frames[b]  # Shape: (num_queries,)

        # Compute the classification cost.
        # Since "no class" is label 0, we need to adjust the indices.
        # We exclude the background class when computing cost_class.
        # Shape: (num_queries, num_gt_events)
        cost_class = -out_prob[:, tgt_labels]

        # Frame regression cost: L1 distance
        cost_frames = torch.cdist(
            out_frames.unsqueeze(1), tgt_frames.unsqueeze(1), p=1
        )  # Shape: (num_queries, num_gt_events)

        # Total cost
        C = cost_class + cost_frames * 1.0  # Adjust the weight if necessary

        C = C.detach().cpu().numpy()

        # Solve the assignment problem
        row_ind, col_ind = linear_sum_assignment(C)

        indices.append(
            (
                torch.as_tensor(row_ind, dtype=torch.int64, device=device),
                torch.as_tensor(col_ind, dtype=torch.int64, device=device),
            )
        )

    return indices


def compute_loss(outputs, targets, indices, num_classes, fg_weight=5.0):
    bs, num_queries = outputs["pred_logits"].shape[:2]
    device = outputs["pred_logits"].device

    # Flatten outputs
    src_logits = outputs["pred_logits"].reshape(
        bs * num_queries, num_classes
    )  # Shape: (bs * num_queries, num_classes)
    src_frames = outputs["pred_frames"].reshape(
        bs * num_queries
    )  # Shape: (bs * num_queries,)

    # Initialize target class tensor with background class (label 0)
    target_classes = torch.zeros(
        bs * num_queries, dtype=torch.long, device=device
    )  # Background label is 0

    # Build the index of matched predictions
    batch_idx = []
    src_idx = []
    tgt_idx = []
    for b in range(bs):
        idx = indices[b]
        batch_idx.extend([b] * len(idx[0]))
        src_idx.extend(idx[0] + b * num_queries)
        tgt_idx.extend(idx[1])

    src_idx = torch.tensor(src_idx, dtype=torch.long, device=device)
    tgt_idx = torch.tensor(tgt_idx, dtype=torch.long, device=device)

    if len(src_idx) > 0:
        # Update target_classes with actual class labels (from targets['labels'])
        matched_classes = torch.cat(
            [targets["labels"][b][indices[b][1]] for b in range(bs)]
        )
        target_classes[src_idx] = matched_classes

    # Classification loss
    weight = torch.FloatTensor([1] + [fg_weight] * (num_classes - 1)).to(
        src_logits.device
    )

    # loss_ce = F.cross_entropy(
    #     src_logits, target_classes, reduction="mean", weight=weight
    # )

    loss_ce = focal_loss(src_logits, target_classes, alpha=1, gamma=2, reduction='mean')

    # Frame regression loss
    loss_frames = torch.tensor(0.0, device=device)
    if len(src_idx) > 0:
        # Get the predicted and target frames for matched indices
        src_frames_matched = src_frames[src_idx]

        target_cls_matched = torch.cat(
            [targets["labels"][b][indices[b][1]] for b in range(bs)]
        ).to(device)

        target_frames_matched = torch.cat(
            [targets["timesteps"][b][indices[b][1]] for b in range(bs)]
        ).to(device)

        fg_mask = target_cls_matched > 0

        loss_frames = (
            F.l1_loss(src_frames_matched, target_frames_matched, reduction="none")
            * fg_mask
        )

        loss_frames = loss_frames.sum() / (fg_mask.sum() + 1e-5)

    # Total loss
    total_loss = loss_ce + loss_frames * 1.0  # Adjust the weight if necessary

    return {"loss": total_loss, "loss_ce": loss_ce, "loss_frames": loss_frames}


def focal_loss(inputs, targets, alpha=1, gamma=2, reduction='mean'):
    """
    Compute the focal loss between `inputs` and the ground truth `targets`.

    Args:
        inputs (Tensor): Predictions of shape (N, C) where C is the number of classes.
        targets (Tensor): Ground truth labels of shape (N,).
        alpha (float, optional): Scaling factor for the focal loss. Default is 1.
        gamma (float, optional): Focusing parameter to adjust the rate at which easy
                                 examples are down-weighted. Default is 2.
        reduction (str, optional): Specifies the reduction to apply to the output.
                                   'none' | 'mean' | 'sum'. Default is 'mean'.
    
    Returns:
        Tensor: Computed focal loss.
    """
    BCE_loss = F.cross_entropy(inputs, targets, reduction='none')  # Compute cross-entropy loss
    pt = torch.exp(-BCE_loss)  # Compute the probability of the target class
    F_loss = alpha * (1 - pt) ** gamma * BCE_loss  # Compute the focal loss

    if reduction == 'mean':
        return torch.mean(F_loss)  # Return mean focal loss
    elif reduction == 'sum':
        return torch.sum(F_loss)   # Return sum of focal loss
    else:
        return F_loss  # Return without reduction


if __name__ == "__main__":
    # Test DeTRPrediction
    model = DeTRPrediction(backbone="rny002", num_classes=7, num_queries=100)
    frames = torch.randn(4, 100, 3, 224, 224)
    targets = dict(
        labels=torch.randint(0, 7, (4, 3)),  # 3 ground truth events
        timesteps=torch.rand(4, 3),  # 3 ground
    )
    outputs = model(frames)
    indices = hungarian_match(outputs, targets)
    loss = compute_loss(outputs, targets, indices, num_classes=7)
    model.print_stats()
    print(outputs["pred_logits"].shape)  # Expected: torch.Size([4, 100, 10])
    print(outputs["pred_frames"].shape)  # Expected: torch.Size([4, 100])
    print(loss)  # Print the loss value
