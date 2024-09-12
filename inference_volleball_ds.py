import os
import cv2
import glob
import torch
import random
import tabulate
import torchvision
import numpy as np
from torch import nn
from tqdm import tqdm
from util.io import store_json
from easydict import EasyDict as edict
from train_e2e_spatial import E2EModel
from torch.utils.data import Dataset, DataLoader
from dataset.frame import _get_img_transforms
from util.score import (
    filter_events_by_score,
    non_max_suppression_events,
)

def get_args(**overrides):
    args = dict(
        frame_dir="/mnt/d/Dataset/volleyball_kaggle",
        crop_dim=224,
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
        classes="serve receive set spike dig block".split(),
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

    e2em.load(torch.load(os.path.abspath(args.checkpoint_path)))
    return e2em


list_subdir = lambda ddir: list(
    filter(lambda sdir: os.path.isdir(os.path.join(ddir, sdir)), os.listdir(ddir))
)

def shuffle(array):
    random.shuffle(array)
    return array

class InferenceDataset(Dataset):
    def __init__(
        self,
        frame_dir,
        crop_dim,
        clip_len,
        overlap_len=0,
        stride=1,
        pad_len=5,
        skip_partial_end=True,
        classes=None,
        num_videos=None,
    ):
        self.frame_dir = frame_dir
        self.crop_dim = crop_dim
        self.clip_len = clip_len
        self.overlap_len = overlap_len
        self.stride = stride
        self.pad_len = pad_len
        self.skip_partial_end = skip_partial_end
        self.num_videos = num_videos

        self.transforms = _get_img_transforms(crop_dim, is_eval=True)
        self.clips = []  # list of (game_id, seg_id, start_frame) tuples
        self.frame_paths = (
            {}
        )  # key: (game_id, seg_id) tuple, value: sorted frame paths array
        self._get_clips()

    def print_infos(self):
        print(f"frame_dir: {self.frame_dir}")
        print(f"num_segments: {len(self.frame_paths)}")
        print(f"num_clips: {len(self.clips)}")
        print(f"clips: {self.frame_paths.keys()}")

    def _get_clips(self):
        for game_id in shuffle(list_subdir(self.frame_dir)):
            for seg_id in shuffle(list_subdir(os.path.join(self.frame_dir, game_id))):
                frames_path = sorted(
                    glob.glob(os.path.join(self.frame_dir, game_id, seg_id, "*.jpg"))
                )
                num_frames = len(frames_path)
                self.frame_paths[(game_id, seg_id)] = frames_path

                for i in range(
                    -self.pad_len * self.stride,
                    max(
                        0,
                        num_frames
                        - (self.overlap_len * self.stride) * int(self.skip_partial_end)
                        - self.pad_len * self.stride,
                    ),  # Need to ensure that all clips have at least one frame
                    (self.clip_len - self.overlap_len) * self.stride,
                ):
                    self.clips.append((game_id, seg_id, i))

                if self.num_videos and len(self.frame_paths.keys()) >= self.num_videos:
                    return ()

    def __len__(self):
        return len(self.clips)

    @property
    def videos(self):
        return [
            (video, len(frames_path)) for video, frames_path in self.frame_paths.items()
        ]

    def load_frames(self, game_id, seg_id, start, end=None, pad=False, stride=1):

        frames_path = self.frame_paths[(game_id, seg_id)]
        ret = []
        n_pad_start = 0
        n_pad_end = 0
        for frame_num in range(start, end, stride):

            # if frame_num is negative, skip then pads
            if frame_num < 0:
                n_pad_start += 1
                continue
            # here frame_num is non negative, so find its path and load
            # if frame_num not in range(len(frames_path)):
            #     print(f"ERR: game_id: {game_id}, seg_id: {seg_id}, start: {start}, frame_num: {frame_num} frames_path: {(frames_path)}" )

            try:
                frame_path = frames_path[frame_num]
                img = torchvision.io.read_image(frame_path).float() / 255
                ret.append(img)
            except:
                # if frame is not found -> pad
                n_pad_end += 1

        ret = torch.stack(ret, dim=int(len(ret[0].shape) == 4))
        # Always pad start, but only pad end if requested
        if n_pad_start > 0 or (pad and n_pad_end > 0):
            ret = nn.functional.pad(
                ret, (0, 0, 0, 0, 0, 0, n_pad_start, n_pad_end if pad else 0)
            )
        return ret

    def __getitem__(self, idx):
        game_id, seg_id, start = self.clips[idx]
        frames_tensor = self.load_frames(
            game_id,
            seg_id,
            start,
            end=start + self.clip_len * self.stride,
            pad=True,
            stride=self.stride,
        )
        frames_tensor = self.transforms(frames_tensor)
        return {
            "game_id": game_id,
            "seg_id": seg_id,
            "start": start,
            "frames": frames_tensor,
        }



def draw_filled_parallelogram(text, center_coordinates, image, dt):
    # Initial parameters
    radius = 10
    color = (255, 255, 255)  # White color in BGR
    circle_thickness = 1  # Solid circle
    delta_shadow = np.array([3, 3])
    color_shadow = (57, 61, 71)  # Black color for the shadow
    color_text = (0, 0, 0)  # Black color for the text

    # Text and padding
    font_scale = 0.7
    font_thickness = 1
    # font = cv2.FONT_HERSHEY_SIMPLEX
    font = cv2.FONT_HERSHEY_COMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

    left_padding = 7
    right_padding = 20
    top_padding = 12
    bottom_padding = 7

    # Adjust the length of the parallelogram based on the text size and padding
    parallelogram_length = text_size[0] + left_padding + right_padding
    parallelogram_height = text_size[1] + top_padding + bottom_padding

    # Define angle for the 60-degree line
    angle = np.pi / 3  # 60 degrees in radians
    positive_angle = 2 * np.pi / 3  # 120 degrees in radians

    # Calculate the 60-degree line points
    start_point = center_coordinates
    end_point = (
        int(center_coordinates[0] + 100 * np.cos(angle)),
        int(center_coordinates[1] - 100 * np.sin(angle)),
    )

    # Calculate the points of the parallelogram, with the first point being below the 60-degree line
    gap = 5
    point1 = (end_point[0] + gap, end_point[1] + gap)  # Below the 60-degree line
    point2 = (
        int(point1[0] + parallelogram_length),
        point1[1],
    )  # Move horizontally (parallel to horizontal line)
    point3 = (
        int(point2[0] + parallelogram_height * np.cos(positive_angle)),
        int(point2[1] + parallelogram_height * np.sin(positive_angle)),
    )  # Adjusted to 120 degrees
    point4 = (
        int(point1[0] + parallelogram_height * np.cos(positive_angle)),
        int(point1[1] + parallelogram_height * np.sin(positive_angle)),
    )  # Adjusted to 120 degrees

    # Draw the white dot with radius 10
    cv2.circle(
        image,
        center_coordinates,
        radius,
        color,
        circle_thickness,
        lineType=cv2.LINE_AA,
    )

    # Draw the 60-degree line
    end_point_ = np.add(
        start_point, np.subtract(end_point, center_coordinates) * min(1.0, dt * 4)
    ).astype(np.int32)

    start_point_shadow = np.add(start_point, delta_shadow * 0.5).astype(np.int32)
    end_point_shadow = np.add(end_point_, delta_shadow * 0.5).astype(np.int32)

    cv2.line(
        image,
        start_point_shadow,
        end_point_shadow,
        color_shadow,  # Black color for the shadow
        thickness=2,
        lineType=cv2.LINE_AA,
    )

    cv2.line(image, start_point, end_point_, color, thickness=2, lineType=cv2.LINE_AA)

    # Draw the horizontal line from the end of the first line

    if dt >= 1 / 4.0:
        horizontal_end_point = (
            int(end_point[0] + 60 * min(1.0, 4 * (dt - 0.25))),
            end_point[1],
        )
        horizontal_end_point_shadow = np.add(
            horizontal_end_point, delta_shadow * 0.5
        ).astype(np.int32)
        cv2.line(
            image,
            end_point_shadow,
            horizontal_end_point_shadow,
            color_shadow,  # Black color for the shadow
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        cv2.line(
            image,
            end_point,
            horizontal_end_point,
            color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    # Fill the parallelogram by using the fillPoly function
    delta = np.array([max(0, 1 - dt * 4) * 15, 0])
    pts = np.array([point1, point2, point3, point4], np.int32)
    pts = (pts + delta).astype(np.int32)
    pts_shadow = (pts + delta_shadow).astype(np.int32)
    pts = pts.reshape((-1, 1, 2))
    pts_shadow = pts_shadow.reshape((-1, 1, 2))

    # draw main parallelogram
    cv2.fillPoly(image, [pts_shadow], color_shadow, lineType=cv2.LINE_AA)
    cv2.fillPoly(image, [pts], color, lineType=cv2.LINE_AA)

    # Calculate text position (centered inside the parallelogram with adjusted padding)
    text_x = pts[0][0][0] + left_padding
    text_y = pts[0][0][1] + top_padding + text_size[1] // 2

    # Add the text inside the parallelogram
    cv2.putText(
        image,
        text,
        (text_x, text_y),
        font,
        font_scale,
        color_text,
        font_thickness,
        lineType=cv2.LINE_AA,
    )

    return image


def run_inference(model, dataloader, classes, pred_file):
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
            video = (clip["game_id"][i], clip["seg_id"][i])
            scores, support, locations = pred_dict[video]

            pred_scores = batch_pred_scores[i]  # Predicted scores for the current batch
            pred_loc = batch_pred_loc[i]  # Predicted locations for the current batch

            start = clip["start"][i].item()
            if start < 0:
                # Adjust predictions if the start index is negative
                pred_scores = pred_scores[-start:, :]
                pred_loc = pred_loc[-start:, :]
                start = 0
            end = start + pred_scores.shape[0]
            if end >= scores.shape[0]:
                # Adjust predictions if the end index exceeds the video length
                end = scores.shape[0]
                pred_scores = pred_scores[: end - start, :]
                pred_loc = pred_loc[: end - start, :]
            scores[start:end, :] += pred_scores  # Accumulate scores
            support[
                start:end
            ] += 1  # Increment the support count because the frame is present
            locations[start:end, :] = pred_loc

    pred_events = []
    pred_scores = {}
    for video, (scores, support, locations_pred) in sorted(pred_dict.items()):
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
        if pred_file:
            os.makedirs(os.path.dirname(pred_file), exist_ok=True)
            store_json(pred_file, pred_events)
        print(f"Saved predictions to {pred_file}")
        return pred_events


def visualize(config):
    preds = config["preds"]
    output_dir = config["output_dir"]
    frame_dir = config["frame_dir"]
    fps = config["fps"]
    width = config["width"]
    height = config["height"]
    freeze_frames = config["freeze_frames"]
    fourcc = cv2.VideoWriter_fourcc(*config["codec"])
    main_color = config["main_color"]
    os.makedirs(output_dir, exist_ok=True)

    for current_video in preds:
        print(f"Chosen video: {current_video['video']}")
        event_by_frame = {x["frame"]: x for x in current_video["events"]}
        output_video = os.path.join(output_dir, "_".join(current_video["video"]) + ".mp4")

        org_frames_path = sorted(
            glob.glob(os.path.join(frame_dir, *(current_video["video"]), "*.jpg"))
        )

        frames_im = []
        for i, frame_path in enumerate(org_frames_path):
            img = cv2.imread(frame_path)
            # resize if needed
            if img.shape[1] != width or img.shape[0] != height:
                img = cv2.resize(img, (width, height))

            # check if event in frame
            if i not in event_by_frame:
                frames_im.append(img)
            else:
                event = event_by_frame[i]
                x, y = int(event["xy"][0] * width), int(event["xy"][1] * height)
                score = event.get("score", "")
                label = event["label"]
                text = f"{label.upper()} {score:.0%}"

                # freeze frames to animate annotations
                for it in range(freeze_frames):
                    dt = it / freeze_frames
                    canvas = img.copy()  # copy frame

                    draw_filled_parallelogram(text, (x, y), canvas, dt)

                    alpha = max(0.9, 0.65 + 0.25 * dt)

                    # blend frame with annotation
                    canvas = cv2.addWeighted(img, 1 - alpha, canvas, alpha, 0)

                    # draw circle after blending to avoid blending issues
                    radius = 10 + 40 * (1 - dt) ** 8  # radius decreases with time
                    thickness = 1 + dt * 2  # thickness increases with time
                    cv2.circle(
                        canvas,
                        (x, y),
                        int(radius),
                        main_color,
                        int(thickness),
                        lineType=cv2.LINE_AA,
                    )

                    frames_im.append(canvas)

        # write to video
        video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        for _, frame_path in enumerate(frames_im):
            video_writer.write(frame_path)

        video_writer.release()
        print("Video saved to", output_video)

if __name__ == "__main__":
    # Set the arguments for inference
    args = get_args(
        frame_dir="/home/hoang/data/volleyball_kaggle/frames",
        clip_len=32,
        checkpoint_path="exp/best/checkpoint_144.pt",
        save_dir="exp/volleyball_kaggle",
        batch_size=16,
        # num_videos=10,
    )
    
    # Create the InferenceDataset
    dataset = InferenceDataset(
        args.frame_dir,
        args.crop_dim,
        args.clip_len,
        overlap_len=args.clip_len // 2,
        num_videos=args.num_videos,
    )
    
    # Print information about the dataset
    dataset.print_infos()
    
    # Create the DataLoader for batch processing
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )
    
    # Load the pre-trained model
    model = get_model(args)
    
    # Specify the file to save the predictions
    pred_file = os.path.join(args.save_dir, "predictions.json")

    # Run the inference and save the predictions
    pred_events = run_inference(
        model,
        dataloader,
        args.classes,
        pred_file,
    )

    # Visualize the predictions
    visualize({
        "preds": pred_events,
        "frame_dir": args.frame_dir,
        "output_dir": "exp/volleyball_kaggle/pred_videos",
        "fps": 24,
        "width": 1280,
        "height": 720,
        "freeze_frames": 24 // 1,
        "codec": "mp4v",
        "main_color": (242, 243, 244),
    })
