# STES: Spatio-Temporal Event Spotting in Video

End-to-end model for detecting **when** and **where** fine-grained events occur in video, at single-frame temporal precision. Built on top of [E2E-Spot](https://jhong93.github.io/projects/spot.html) (ECCV 2022), this work adds a **location regression head** that jointly predicts event class, timing, and spatial coordinates.

The primary application is volleyball action spotting (serve, receive, set, spike, block, dig, etc.) with per-event (x, y) location prediction.

## Architecture

- **Backbone**: RegNet-Y 800MF with Gated Shift Modules (GSM) for temporal modeling within the CNN
- **Temporal head**: Bidirectional GRU for sequence-level event classification
- **Location head**: MLP that regresses (x, y) coordinates for each detected event

## Setup

```bash
pip install -r requirements.txt
```

The code is tested on Linux with Python 3.10+ and PyTorch 2.x with CUDA.

## Data Preparation

### Frame extraction

Extract video frames resized to 224px height into a directory structure:

```
<frame_dir>/
├── video1/
│   ├── 000000.jpg
│   ├── 000001.jpg
│   └── ...
├── video2/
│   └── ...
```

### Dataset metadata

Place dataset files under `data/<dataset_name>/`:

- **`class.txt`** — one class name per line
- **`train.json`**, **`val.json`**, **`test.json`** — annotation files

Annotation format:
```json
[
    {
        "video": "video_id",
        "num_frames": 3600,
        "num_events": 12,
        "events": [
            {
                "frame": 525,
                "label": "spike",
                "comment": "",
                "x": 0.45,
                "y": 0.62
            }
        ],
        "fps": 30,
        "width": 1920,
        "height": 1080
    }
]
```

## Training

### Train from scratch

```bash
python train_e2e_spatial.py <dataset> <frame_dir> \
  -m rny008_gsm -t gru \
  --clip_len 64 --batch_size 8 \
  --num_epochs 150 \
  -s exp/<experiment_name> \
  --predict_location \
  --num_workers 4
```

### Pretrain + Finetune

```bash
# Stage 1: Pretrain on a larger dataset
python train_e2e_spatial.py vnl_1.5 data/vnl_1.5/frames_224p \
  -m rny008_gsm -t gru --clip_len 64 --batch_size 8 \
  --num_epochs 100 -s exp/pretrain_finetune --predict_location

# Stage 2: Finetune on target dataset (resumes from last checkpoint)
python train_e2e_spatial.py vnl_2.0 data/vnl_2.0/frames_224p \
  -m rny008_gsm -t gru --clip_len 64 --batch_size 8 \
  --num_epochs 200 -s exp/pretrain_finetune --predict_location --resume
```

See `run.sh` and `pretrain_finetune.sh` for ready-to-use scripts.

## Evaluation

```bash
python eval.py <model_dir_or_pred_file> -s <split>
```

Predictions are saved as `pred-{split}.{epoch}.recall.json.gz` during training.

## Inference on Video

Run inference directly on an MP4 file:

```bash
python inference_on_mp4.py --checkpoint_path <path_to_checkpoint> --video <path_to_video>
```

## Demo App

A Gradio-based web interface for uploading videos and visualizing detected events:

```bash
python app.py
```

## Supported Datasets

| Name | Description |
|------|-------------|
| `vnl_2.0` | Volleyball Nations League |
| `vnl_1.5` | Volleyball Nations League (older) |

## Project Structure

```
├── train_e2e_spatial.py    # Main training script
├── eval.py                 # Evaluation with mean-AP
├── inference_on_mp4.py     # Inference on raw video files
├── app.py                  # Gradio demo app
├── model/
│   ├── common.py           # Base model classes, MLP
│   ├── modules.py          # GRU head, location predictor, channel attention
│   ├── shift.py            # TSM and GSM temporal shift modules
│   └── min_gru.py          # Minimal GRU variant
├── dataset/
│   ├── frame.py            # Frame-based dataset classes
│   └── transform.py        # Data augmentation transforms
├── util/
│   ├── score.py            # mAP computation and location-aware metrics
│   ├── eval.py             # Frame prediction post-processing
│   ├── dataset.py          # Dataset registry and helpers
│   └── io.py               # JSON/GZ I/O utilities
├── run.sh                  # Single-dataset training script
├── pretrain_finetune.sh    # Two-stage pretrain+finetune script
└── requirements.txt
```

## Acknowledgments

This project builds on [E2E-Spot](https://github.com/jhong93/spot) by James Hong et al. (ECCV 2022).

```
@inproceedings{precisespotting_eccv22,
    author={Hong, James and Zhang, Haotian and Gharbi, Micha\"{e}l and Fisher, Matthew and Fatahalian, Kayvon},
    title={Spotting Temporally Precise, Fine-Grained Events in Video},
    booktitle={ECCV},
    year={2022}
}
```

## License

BSD-3 — see [LICENSE](LICENSE).
