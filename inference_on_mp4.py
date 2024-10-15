import os
import cv2
import glob
import math
import torch
import random
import tabulate
import torchvision
import numpy as np
from torch import nn
from tqdm import tqdm
from torchvision.io import read_video
from torchvision.transforms import v2
from easydict import EasyDict as edict
from torch.utils.data import Dataset, DataLoader

# custom modules
from util.io import store_json
from train_e2e_spatial import E2EModel
from util.score import (
    filter_events_by_score,
    non_max_suppression_events,
)


def get_args(**overrides):
    args = dict(
        dilate_len=0,
        mixup=False,
        modality="rgb",
        feature_arch="rny008_gsm",
        temporal_arch="gru",
        clip_len=32,
        batch_size=2,
        predict_location=True,
        num_workers=4,
        eval_only=False,
        checkpoint_path="exp/best/checkpoint_138.pt",
        criterion="map",
        fg_upsample=None,
        gpu_parallel=False,
        debug_only=False,
        # classes="serve receive set spike dig block".split(),
        classes="serve receive set spike block score".split(),
        num_videos=None,
    )
    args.update(overrides)
    args = edict(args)
    # print tabulated args
    print(
        tabulate.tabulate(
            [
                (key, args[key])
                for key in sorted(args.keys())
                if not key.startswith("_")
            ],
            headers=["Argument", "Value"],
            tablefmt="fancy_outline",
        )
    )
    return args


def get_model(args):
    e2em = E2EModel(
        len(args.classes) + 1,
        args.feature_arch,
        args.temporal_arch,
        clip_len=args.clip_len,
        modality=args.modality,
        multi_gpu=args.gpu_parallel,
        predict_location=args.predict_location,
    )

    e2em.load(torch.load(os.path.abspath(args.checkpoint_path), weights_only=True))
    return e2em


class VideoFrameSlidingDataset(torch.utils.data.Dataset):
    """MP4 mode only works with short videos (few seconds long) due to memory constraints."""

    def __init__(
        self,
        input_data,  # Could be either a list of numpy images or a video path
        mode="list",  # "list" for numpy arrays, "mp4" for video file
        transform=None,
        window_size=16,
        stride=1,
        channel_first=False,  # (C, T, H, W) if True, (T, C, H, W) if False
    ):
        self.mode = mode
        self.window_size = window_size
        self.stride = stride
        self.channel_first = channel_first
        self.transform = transform

        if self.mode == "list":
            self.image_list = input_data
            self.num_frames = len(self.image_list)
            print(f"Total frames in list: {self.num_frames}")
        elif self.mode == "mp4":
            self.video_path = input_data
            video, _, _ = read_video(self.video_path, pts_unit="sec")
            self.frames = video.permute(0, 3, 1, 2)  # Convert to (T, C, H, W)
            self.num_frames = self.frames.size(0)
            print(f"Video path: {self.video_path}, Total frames: {self.num_frames}")

        else:
            raise ValueError("Mode must be either 'list' or 'mp4'")

    @property
    def videos(self):
        return [(self.video_path, self.num_frames)]

    def _get_frame_from_video(self, idx):
        """Retrieve a frame directly from the loaded video tensor."""
        if idx < self.num_frames:
            frame = self.frames[idx]
            org_size = (frame.shape[2], frame.shape[1])  # (Width, Height)
            return frame, org_size
        else:
            raise KeyError(f"Frame at index {idx} could not be loaded into buffer.")

    def _load_sequence(self, start_idx):
        """Load a sequence of frames starting from `start_idx`."""
        frames = []
        org_sizes = []

        for i in range(self.window_size):
            actual_idx = start_idx + i
            if actual_idx < self.num_frames:
                if self.mode == "list":
                    # Load frame from the list of numpy arrays
                    frame = torch.tensor(
                        self.image_list[actual_idx]
                    )  # Convert to tensor
                    org_size = (frame.shape[2], frame.shape[1])  # (Width, Height)
                elif self.mode == "mp4":
                    # Load frame from video tensor
                    frame, org_size = self._get_frame_from_video(actual_idx)
                else:
                    raise ValueError(f"Unsupported mode: {self.mode}")
            else:
                # Zero-padding if not enough frames
                frame = torch.zeros(
                    (3, org_size[1], org_size[0]),
                    dtype=torch.float,
                )
                org_size = (frame.shape[2], frame.shape[1])

            frames.append(frame)
            org_sizes.append(org_size)

        return frames, org_sizes

    def __len__(self):
        # Calculate the number of sequences based on stride and window_size
        return max(1, math.ceil((self.num_frames - self.window_size) / self.stride + 1))

    def __getitem__(self, idx):
        # Calculate start index for this sequence
        start_idx = idx * self.stride

        # Load sequence of frames and their original sizes
        frames, org_sizes = self._load_sequence(start_idx)

        # Stack frames into a tensor of shape (T, C, H, W)
        frames = torch.stack(frames, dim=0)

        # Apply transform to the entire sequence (all frames in the sequence)
        if self.transform:
            frames = self.transform(frames)

        # Convert to (C, T, H, W) if channel_first is True
        if self.channel_first:
            frames = frames.permute(
                1, 0, 2, 3
            )  # Convert from (T, C, H, W) to (C, T, H, W)

        return {
            "game_id": self.video_path,
            "seg_id": idx,
            "frames": frames,
            "start": start_idx,
        }


def run_inference(model, dataloader, classes, pred_file, postprocess=False):
    pred_dict = {}

    for video, video_len in dataloader.dataset.videos:
        pred_dict[video] = (
            np.zeros(
                (video_len, len(classes) + 1), np.float32
            ),  # Stores scores for each class
            np.zeros(video_len, np.int32),  # Stores support (number of frames)
            np.zeros((video_len, 2), np.float32),  # Stores location predictions (x, y)
        )

    for clip in tqdm(dataloader):
        _, batch_pred_scores, batch_pred_loc = model.predict(clip["frames"])

        batch_pred_loc = batch_pred_loc.reshape(batch_pred_scores.shape[0], -1, 2)
        for i in range(clip["frames"].shape[0]):
            video = clip["game_id"][i]
            scores, support, locations = pred_dict[video]

            pred_scores = batch_pred_scores[i]  # Predicted scores for the current batch
            pred_loc = batch_pred_loc[i]  # Predicted locations for the current batch

            start = clip["start"][i].item()
            end = min(start + pred_scores.shape[0], len(support))

            pred_scores = pred_scores[: end - start, :]
            pred_loc = pred_loc[: end - start, :]

            scores[start:end, :] += pred_scores  # Accumulate scores
            support[
                start:end
            ] += 1  # Increment the support count because the frame is present
            locations[start:end, :] = pred_loc
            print(f"Processed {video} from {start} to {end}")

    pred_events = []
    pred_scores = {}
    for video, (scores, support, locations_pred) in sorted(pred_dict.items()):
        # breakpoint()
        assert np.min(support) > 0, (video, support.tolist())
        scores /= support[:, None]
        # locations_pred /= support[:, None]
        pred = np.argmax(scores, axis=1)

        pred_scores[video] = scores.tolist()

        events = []
        for i in range(pred.shape[0]):

            if pred[i] != 0:
                events.append(
                    {
                        "label": classes[pred[i] - 1],
                        "frame": i,
                        "score": scores[i, pred[i]].item(),
                        "xy": locations_pred[i].tolist(),
                    }
                )

        pred_events.append({"video": video, "events": events})

    if postprocess:

        pred_events = filter_events_by_score(pred_events, fg_threshold=0.15)

        pred_events = non_max_suppression_events(pred_events, 3)

    if pred_file:
        os.makedirs(os.path.dirname(pred_file), exist_ok=True)
        store_json(pred_file, pred_events)

    print(f"Saved predictions to {pred_file}")
    return pred_events


if __name__ == "__main__":
    # Set the arguments for inference
    args = get_args(
        video_mp4="/home/hoang/demo.mp4",
        resize=(224, 224),
        clip_len=64,
        checkpoint_path="exp/best/gru_r8g_newlabeloct01_checkpoint_125.pt",
        save_dir="exp/demo1",
        batch_size=16,
    )

    args.pred_file = os.path.join(args.save_dir, "predictions.json")

    transform = v2.Compose(
        [
            v2.Resize(args.resize, antialias=False),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = VideoFrameSlidingDataset(
        args.video_mp4,
        mode="mp4",
        window_size=args.clip_len,
        stride=args.clip_len,
        channel_first=False,
        transform=transform,
    )

    # Create the DataLoader for batch processing
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # Load the pre-trained model
    model = get_model(args)

    # Run the inference and save the predictions
    pred_events = run_inference(
        model,
        dataloader,
        args.classes,
        args.pred_file,
        postprocess=True,
    )
