#!/usr/bin/env python3
""" Training for E2E-Spot """

import os
import argparse
from contextlib import nullcontext
import random
import numpy as np
from tabulate import tabulate
import torch
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import (
    ChainedScheduler, LinearLR, CosineAnnealingLR)
from torch.utils.data import DataLoader
import torchvision
import timm
from tqdm import tqdm

from model.common import step, BaseRGBModel, MLP
from model.shift import make_temporal_shift
from model.modules import *
from dataset.frame import ActionSpotDataset, ActionSpotVideoDataset, collate_fn
from util.eval import process_frame_predictions
from util.io import load_json, store_json, store_gz_json, clear_files
from util.dataset import DATASETS, load_classes
from util.score import compute_mAPs

EPOCH_NUM_FRAMES = 500000

BASE_NUM_WORKERS = 4

BASE_NUM_VAL_EPOCHS = 20

INFERENCE_BATCH_SIZE = 4


# Prevent the GRU params from going too big (cap it at a RegNet-Y 800MF)
MAX_GRU_HIDDEN_DIM = 768


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, choices=DATASETS)
    parser.add_argument('frame_dir', type=str, help='Path to extracted frames')

    parser.add_argument('--modality', type=str, choices=['rgb', 'bw', 'flow'],
                        default='rgb')
    parser.add_argument(
        '-m', '--feature_arch', type=str, required=True, choices=[
            # From torchvision
            'rn18',
            'rn18_tsm',
            'rn18_gsm',
            'rn50',
            'rn50_tsm',
            'rn50_gsm',

            # From timm (following its naming conventions)
            'rny002',
            'rny002_tsm',
            'rny002_gsm',
            'rny008',
            'rny008_tsm',
            'rny008_gsm',

            # From timm
            'convnextt',
            'convnextt_tsm',
            'convnextt_gsm'
        ], help='CNN architecture for feature extraction')
    parser.add_argument(
        '-t', '--temporal_arch', type=str, default='gru',
        # choices=['', 'gru', 'deeper_gru', 'mstcn', 'asformer'],
        help='Spotting architecture, after spatial pooling')

    parser.add_argument('--clip_len', type=int, default=100)
    parser.add_argument('--crop_dim', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('-ag', '--acc_grad_iter', type=int, default=1,
                        help='Use gradient accumulation')

    parser.add_argument('--warm_up_epochs', type=int, default=3)
    parser.add_argument('--num_epochs', type=int, default=50)

    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-s', '--save_dir', type=str, required=True,
                        help='Dir to save checkpoints and predictions')

    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint in <save_dir>')

    parser.add_argument('--start_val_epoch', type=int)
    parser.add_argument('--criterion', choices=['map', 'loss'], default='map')

    parser.add_argument('--dilate_len', type=int, default=0,
                        help='Label dilation when training')
    parser.add_argument('--mixup', type=bool, default=True)

    parser.add_argument('-j', '--num_workers', type=int,
                        help='Base number of dataloader workers')

    # Sample based on foreground
    parser.add_argument('--fg_upsample', type=float)

    parser.add_argument('-mgpu', '--gpu_parallel', action='store_true')


    parser.add_argument('--use_mse_loss', action='store_true',
                        help='add MSE loss term to the training loss')
    return parser.parse_args()

from model.detr import DeTRPrediction

class ModelWrapper(BaseRGBModel):

    def __init__(self, num_classes, feature_arch, temporal_arch, clip_len,
                 modality, device='cuda', multi_gpu=False):
        self.device = device
        self._multi_gpu = multi_gpu

        self._model = DeTRPrediction(
            num_classes=num_classes,
            backbone=feature_arch,
            hidden_dim=256,
            num_queries=10,
            depth=6,
            nheads=8,
            dropout=.1,
        )
        self._model.print_stats()

        if multi_gpu:
            self._model = nn.DataParallel(self._model)

        self._model.to(device)
        self._num_classes = num_classes

    def epoch(self, loader, optimizer=None, scaler=None, lr_scheduler=None,
              acc_grad_iter=1, fg_weight=1):
        if optimizer is None:
            self._model.eval()
        else:
            optimizer.zero_grad()
            self._model.train()

        epoch_loss = 0.
        with torch.no_grad() if optimizer is None else nullcontext():
            pbar = tqdm(loader)
            for batch_idx, batch in enumerate(pbar):
                _, loss_dict = self._model.forward_train(batch, fg_weight)
                loss = loss_dict['loss']

                if optimizer is not None:
                    step(optimizer, scaler, loss / acc_grad_iter,
                         lr_scheduler=lr_scheduler,
                         backward_only=(batch_idx + 1) % acc_grad_iter != 0)

                epoch_loss += loss.detach().item()
                pbar.set_postfix(loss=epoch_loss / (batch_idx + 1))

        return epoch_loss / len(loader)     # Avg loss

    def predict(self, seq, use_amp=True):
        if not isinstance(seq, torch.Tensor):
            seq = torch.FloatTensor(seq)
        if len(seq.shape) == 4: # (L, C, H, W)
            seq = seq.unsqueeze(0)
        if seq.device != self.device:
            seq = seq.to(self.device)

        self._model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast() if use_amp else nullcontext():
                pred = self._model(seq)
            if isinstance(pred, tuple):
                pred = pred[0]
            if len(pred.shape) > 3:
                pred = pred[-1]
            pred = torch.softmax(pred, axis=2)
            pred_cls = torch.argmax(pred, axis=2)
            return pred_cls.cpu().numpy(), pred.cpu().numpy()


def evaluate(model, dataset, split, classes, save_pred, calc_stats=True,
             save_scores=True):
    pred_dict = {}
    for video, video_len, _ in dataset.videos:
        pred_dict[video] = (
            np.zeros((video_len, len(classes) + 1), np.float32),
            np.zeros(video_len, np.int32))

    # Do not up the batch size if the dataset augments
    batch_size = 1 if dataset.augment else INFERENCE_BATCH_SIZE

    for clip in tqdm(DataLoader(
            dataset, num_workers=BASE_NUM_WORKERS * 2, pin_memory=True,
            batch_size=batch_size
    )):
        if batch_size > 1:
            # Batched by dataloader
            _, batch_pred_scores = model.predict(clip['frame'])

            for i in range(clip['frame'].shape[0]):
                video = clip['video'][i]
                scores, support = pred_dict[video]
                pred_scores = batch_pred_scores[i]
                start = clip['start'][i].item()
                if start < 0:
                    pred_scores = pred_scores[-start:, :]
                    start = 0
                end = start + pred_scores.shape[0]
                if end >= scores.shape[0]:
                    end = scores.shape[0]
                    pred_scores = pred_scores[:end - start, :]
                scores[start:end, :] += pred_scores
                support[start:end] += 1

        else:
            # Batched by dataset
            scores, support = pred_dict[clip['video'][0]]

            start = clip['start'][0].item()
            _, pred_scores = model.predict(clip['frame'][0])
            if start < 0:
                pred_scores = pred_scores[:, -start:, :]
                start = 0
            end = start + pred_scores.shape[1]
            if end >= scores.shape[0]:
                end = scores.shape[0]
                pred_scores = pred_scores[:,:end - start, :]

            scores[start:end, :] += np.sum(pred_scores, axis=0)
            support[start:end] += pred_scores.shape[0]

    err, f1, pred_events, pred_events_high_recall, pred_scores = \
        process_frame_predictions(dataset, classes, pred_dict)

    avg_mAP = None
    if calc_stats:
        print('=== Results on {} (w/o NMS) ==='.format(split))
        print('Error (frame-level): {:0.2f}\n'.format(err.get() * 100))

        def get_f1_tab_row(str_k):
            k = classes[str_k] if str_k != 'any' else None
            return [str_k, f1.get(k) * 100, *f1.tp_fp_fn(k)]
        rows = [get_f1_tab_row('any')]
        for c in sorted(classes):
            rows.append(get_f1_tab_row(c))
        print(tabulate(rows, headers=['Exact frame', 'F1', 'TP', 'FP', 'FN'],
                       floatfmt='0.2f'))
        print()

        mAPs, _ = compute_mAPs(dataset.labels, pred_events_high_recall)
        avg_mAP = np.mean(mAPs[1:])

    if save_pred is not None:
        store_json(save_pred + '.json', pred_events)
        store_gz_json(save_pred + '.recall.json.gz', pred_events_high_recall)
        if save_scores:
            store_gz_json(save_pred + '.score.json.gz', pred_scores)
    return avg_mAP


def get_last_epoch(save_dir):
    max_epoch = -1
    for file_name in os.listdir(save_dir):
        if not file_name.startswith('optim_'):
            continue
        epoch = int(os.path.splitext(file_name)[0].split('optim_')[1])
        if epoch > max_epoch:
            max_epoch = epoch
    return max_epoch


def get_best_epoch_and_history(save_dir, criterion):
    data = load_json(os.path.join(save_dir, 'loss.json'))
    if criterion == 'map':
        key = 'val_mAP'
        best = max(data, key=lambda x: x[key])
    else:
        key = 'val'
        best = min(data, key=lambda x: x[key])
    return data, best['epoch'], best[key]


def get_datasets(args):
    classes = load_classes(os.path.join('data', args.dataset, 'class.txt'))

    dataset_len = EPOCH_NUM_FRAMES // args.clip_len
    dataset_kwargs = {
        'crop_dim': args.crop_dim, 'dilate_len': args.dilate_len,
        'mixup': args.mixup
    }

    if args.fg_upsample is not None:
        assert args.fg_upsample > 0
        dataset_kwargs['fg_upsample'] = args.fg_upsample

    print('Dataset size:', dataset_len)
    train_data = ActionSpotDataset(
        classes, os.path.join('data', args.dataset, 'train.json'),
        args.frame_dir, args.modality, args.clip_len, dataset_len,
        is_eval=False, **dataset_kwargs)
    train_data.print_info()
    val_data = ActionSpotDataset(
        classes, os.path.join('data', args.dataset, 'val.json'),
        args.frame_dir, args.modality, args.clip_len, dataset_len // 4,
        **dataset_kwargs)
    val_data.print_info()

    val_data_frames = None
    if args.criterion == 'map':
        # Only perform mAP evaluation during training if criterion is mAP
        val_data_frames = ActionSpotVideoDataset(
            classes, os.path.join('data', args.dataset, 'val.json'),
            args.frame_dir, args.modality, args.clip_len,
            crop_dim=args.crop_dim, overlap_len=0)

    return classes, train_data, val_data, val_data_frames


def load_from_save(
        args, model, optimizer, scaler, lr_scheduler
):
    assert args.save_dir is not None
    epoch = get_last_epoch(args.save_dir)

    print('Loading from epoch {}'.format(epoch))
    model.load(torch.load(os.path.join(
        args.save_dir, 'checkpoint_{:03d}.pt'.format(epoch))))

    if args.resume:
        # print('(Resume) Train loss:', model.epoch(train_loader))
        # print('(Resume) Val loss:', model.epoch(val_loader))
        opt_data = torch.load(os.path.join(
            args.save_dir, 'optim_{:03d}.pt'.format(epoch)))
        optimizer.load_state_dict(opt_data['optimizer_state_dict'])
        scaler.load_state_dict(opt_data['scaler_state_dict'])
        lr_scheduler.load_state_dict(opt_data['lr_state_dict'])

    losses, best_epoch, best_criterion = get_best_epoch_and_history(
        args.save_dir, args.criterion)
    return epoch, losses, best_epoch, best_criterion


def store_config(file_path, args, num_epochs, classes):
    config = {
        'dataset': args.dataset,
        'num_classes': len(classes),
        'modality': args.modality,
        'feature_arch': args.feature_arch,
        'temporal_arch': args.temporal_arch,
        'clip_len': args.clip_len,
        'batch_size': args.batch_size,
        'crop_dim': args.crop_dim,
        'num_epochs': num_epochs,
        'warm_up_epochs': args.warm_up_epochs,
        'learning_rate': args.learning_rate,
        'start_val_epoch': args.start_val_epoch,
        'gpu_parallel': args.gpu_parallel,
        'epoch_num_frames': EPOCH_NUM_FRAMES,
        'dilate_len': args.dilate_len,
        'mixup': args.mixup,
        'fg_upsample': args.fg_upsample
    }
    store_json(file_path, config, pretty=True)


def get_num_train_workers(args):
    n = BASE_NUM_WORKERS * 2
    # if args.gpu_parallel:
    #     n *= torch.cuda.device_count()
    return min(os.cpu_count(), n)


def get_lr_scheduler(args, optimizer, num_steps_per_epoch):
    cosine_epochs = args.num_epochs - args.warm_up_epochs
    print('Using Linear Warmup ({}) + Cosine Annealing LR ({})'.format(
        args.warm_up_epochs, cosine_epochs))
    return args.num_epochs, ChainedScheduler([
        LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                 total_iters=args.warm_up_epochs * num_steps_per_epoch),
        CosineAnnealingLR(optimizer,
            num_steps_per_epoch * cosine_epochs)])


def main(args):
    if args.num_workers is not None:
        global BASE_NUM_WORKERS
        BASE_NUM_WORKERS = args.num_workers

    assert args.batch_size % args.acc_grad_iter == 0
    if args.start_val_epoch is None:
        args.start_val_epoch = args.num_epochs - BASE_NUM_VAL_EPOCHS
    if args.crop_dim <= 0:
        args.crop_dim = None

    classes, train_data, val_data, val_data_frames = get_datasets(args)

    def worker_init_fn(id):
        random.seed(id + epoch * 100)
    loader_batch_size = args.batch_size
    train_loader = DataLoader(
        train_data, shuffle=False, batch_size=loader_batch_size,
        pin_memory=True, num_workers=get_num_train_workers(args),
        prefetch_factor=1, worker_init_fn=worker_init_fn, collate_fn=collate_fn)
    val_loader = DataLoader(
        val_data, shuffle=False, batch_size=loader_batch_size,
        pin_memory=True, num_workers=BASE_NUM_WORKERS,
        worker_init_fn=worker_init_fn, collate_fn=collate_fn)

    model = ModelWrapper(
        len(classes) + 1, args.feature_arch, args.temporal_arch,
        clip_len=args.clip_len, modality=args.modality,
        multi_gpu=args.gpu_parallel)
    optimizer, scaler = model.get_optimizer({'lr': args.learning_rate, "weight_decay": 1e-4})

    # Warmup schedule
    num_steps_per_epoch = len(train_loader) // args.acc_grad_iter
    num_epochs, lr_scheduler = get_lr_scheduler(
        args, optimizer, num_steps_per_epoch)

    losses = []
    best_epoch = None
    best_criterion = 0 if args.criterion == 'map' else float('inf')

    epoch = 0
    if args.resume:
        epoch, losses, best_epoch, best_criterion = load_from_save(
            args, model, optimizer, scaler, lr_scheduler)
        epoch += 1

    # Write it to console
    store_config('/dev/stdout', args, num_epochs, classes)

    for epoch in range(epoch, num_epochs):
        train_loss = model.epoch(
            train_loader, optimizer, scaler,
            lr_scheduler=lr_scheduler, acc_grad_iter=args.acc_grad_iter)
        val_loss = model.epoch(val_loader, acc_grad_iter=args.acc_grad_iter)
        print('[Epoch {}] Train loss: {:0.5f} Val loss: {:0.5f}'.format(
            epoch, train_loss, val_loss))

        val_mAP = 0
        if args.criterion == 'loss':
            if val_loss < best_criterion:
                best_criterion = val_loss
                best_epoch = epoch
                print('New best epoch!')
        elif args.criterion == 'map':
            if epoch >= args.start_val_epoch:
                pred_file = None
                if args.save_dir is not None:
                    pred_file = os.path.join(
                        args.save_dir, 'pred-val.{}'.format(epoch))
                    os.makedirs(args.save_dir, exist_ok=True)
                val_mAP = evaluate(model, val_data_frames, 'VAL', classes,
                                    pred_file, save_scores=False)
                if args.criterion == 'map' and val_mAP > best_criterion:
                    best_criterion = val_mAP
                    best_epoch = epoch
                    print('New best epoch!')
        else:
            print('Unknown criterion:', args.criterion)

        losses.append({
            'epoch': epoch, 'train': train_loss, 'val': val_loss,
            'val_mAP': val_mAP})
        if args.save_dir is not None:
            os.makedirs(args.save_dir, exist_ok=True)
            store_json(os.path.join(args.save_dir, 'loss.json'), losses,
                        pretty=True)
            torch.save(
                model.state_dict(),
                os.path.join(args.save_dir,
                    'checkpoint_{:03d}.pt'.format(epoch)))
            clear_files(args.save_dir, r'optim_\d+\.pt')
            torch.save(
                {'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'lr_state_dict': lr_scheduler.state_dict()},
                os.path.join(args.save_dir,
                                'optim_{:03d}.pt'.format(epoch)))
            store_config(os.path.join(args.save_dir, 'config.json'),
                            args, num_epochs, classes)

    print('Best epoch: {}\n'.format(best_epoch))

    if args.save_dir is not None:
        model.load(torch.load(os.path.join(
            args.save_dir, 'checkpoint_{:03d}.pt'.format(best_epoch))))

        # Evaluate on VAL if not already done
        eval_splits = ['val'] if args.criterion != 'map' else []

        # Evaluate on hold out splits
        eval_splits += ['test', 'challenge']
        for split in eval_splits:
            split_path = os.path.join(
                'data', args.dataset, '{}.json'.format(split))
            if os.path.exists(split_path):
                split_data = ActionSpotVideoDataset(
                    classes, split_path, args.frame_dir, args.modality,
                    args.clip_len, overlap_len=args.clip_len // 2,
                    crop_dim=args.crop_dim)
                split_data.print_info()

                pred_file = None
                if args.save_dir is not None:
                    pred_file = os.path.join(
                        args.save_dir, 'pred-{}.{}'.format(split, best_epoch))

                evaluate(model, split_data, split.upper(), classes, pred_file,
                         calc_stats=split != 'challenge')


if __name__ == '__main__':
    main(get_args())
