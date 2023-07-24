import os
import re
import pandas as pd

# import numba
# from numba import jit
import random
import time
from tqdm import tqdm
import json
import datetime
import argparse
import numpy as np
from logger import create_logger

import yaml

# from concurrent import futures
# from xml.dom.minidom import parse
# import xml.dom.minidom
# from tensorflow.image import crop_to_bounding_box as boxcp
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.optim import AdamW
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler.scheduler import Scheduler
from timm.utils import accuracy, AverageMeter
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
)

import torch.multiprocessing as mp
import torch.distributed as dist


# from swin_transformer_v2 import SwinTransformerV2
# from accelerate import Accelerator
from MyDataset import MyDataset
from utils import pathChecker, reduce_tensor, NativeScalerWithGradNormCount

AMP_ENABLE = True
TRAIN_ACCUMULATION_STEPS = 1
TRAIN_CLIP_GRAD = 1
PRINT_FREQ = 1

AUG_MIXUP = 0.8
MODEL_LABEL_SMOOTHING = 0.1
WEIGHT_DECAY: 1e-8

DATA_BATCH_SIZE = int(32)
DATA_PATH = './data/mydataset/dataset_811'

TRAIN_START_EPOCH = 0
TRAIN_EPOCHS = 300
TRAIN_WARMUP_EPOCHS = 2
TRAIN_WARMUP_LR = 5e-7
TRAIN_WEIGHT_DECAY = 0.05
TRAIN_BASE_LR = 1e-5
TRAIN_MIN_LR = 1e-7
TRAIN_OPTIMIZER_BETAS = (0.9, 0.999)
TRAIN_LR_SCHEDULER_DECAY_EPOCHS = int(30)
TRAIN_LR_SCHEDULER_MULTISTEPS = []
TRAIN_OPTIMIZER_EPS = 1e-8
TRAIN_RESUME_FROM = None

TEST_SHUFFLE = False

MODEL_TYPE = "swinv2"
MODEL_NAME = "swinv2-unknown"
MODEL_DROP_PATH_RATE = 0.2
MODEL_NUM_CLASSES = 1000
MODEL_DROP_PATH_RATE = 0.1

DATA_IMG_SIZE = 256

MODEL_SWINV2_WINDOW_SIZE = 8

MODEL_SWINV2_QKV_BIAS = True
MODEL_SWINV2_APE = False
MODEL_SWINV2_PATCH_NORM = True

SAVE_FREQ = int(1)
AUTO_SAVE_SECOND = 1800
NUM_WORKERS = int(2)
TRAIN_ACCUMULATION_STEPS = 1


def parse_option():
    parser = argparse.ArgumentParser(
        "Swin Transformer training and evaluation script", add_help=False
    )
    # easy config modification
    parser.add_argument(
        "--batch_size", type=int, default=6, help="batch size for single GPU"
    )
    parser.add_argument("--devices", type=str, help="GPU")
    parser.add_argument(
        "--train_epochs", type=int, default=30, help="epochs num of train"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="numbers of dataloader pre-cache workers per thread"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default='./transformers/models/swinv2-small-patch4-window8-256',
        help="path of pretrained model"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default='./data/mydataset/dataset_82',
        help="path of dataset"
    )
    parser.add_argument(
        "--update_data_index",
        # type=bool,
        default=False,
        action='store_true',
        help="update dataset index file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='./runs/swinS',
        help="path of output"
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        help="resume from an output dir or a model weight file"
    )
    parser.add_argument(
        "--save_step",
        type=int,
        default=5,
        help="resume from an output dir or a model weight file"
    )
    parser.add_argument(
        "--data_img_size",
        type=int,
        default=192,
    )
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument("--local_rank", type=int, required=True,
                        help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()

    return args


class LinearLRScheduler(Scheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        t_initial: int,
        lr_min_rate: float,
        warmup_t=0,
        warmup_lr_init=0.0,
        t_in_epochs=True,
        noise_range_t=None,
        noise_pct=0.67,
        noise_std=1.0,
        noise_seed=42,
        initialize=True,
    ) -> None:
        super().__init__(
            optimizer,
            param_group_field="lr",
            noise_range_t=noise_range_t,
            noise_pct=noise_pct,
            noise_std=noise_std,
            noise_seed=noise_seed,
            initialize=initialize,
        )

        self.t_initial = t_initial
        self.lr_min_rate = lr_min_rate
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [
                (v - warmup_lr_init) / self.warmup_t for v in self.base_values
            ]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            t = t - self.warmup_t
            total_t = self.t_initial - self.warmup_t
            lrs = [
                v - ((v - v * self.lr_min_rate) * (t / total_t))
                for v in self.base_values
            ]
        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None


def saveCheckpoint(
    epoch,
    model,
    max_accuracy1,
    max_accuracy5,
    optimizer,
    lr_scheduler,
    loss_scaler,
    logger,
    path,
    latest=True,
    best=False,
    idx=-1,
    ignore_step: bool = False
):
    dict_states = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "lr_scheduler_state": lr_scheduler.state_dict(),
        "max_accuracy1": max_accuracy1,
        "max_accuracy5": max_accuracy5,
        'scaler': loss_scaler.state_dict(),
        "epoch": epoch,
    }
    model_chk = {
        "model": model,
        "epoch": epoch,
        "max_accuracy1": max_accuracy1,
        "max_accuracy5": max_accuracy5
    }

    pathChecker(path)
    logger.info(f"{path} saving......")
    # print(f"{path} saving......")
    # if idx >= 0:
    #     #saveEpochPath = os.path.join(path, f"ckpt_epoch_{epoch}_{idx}.pth")
    #     saveEpochPath = os.path.join(path, f"ckpt_epoch_{epoch}_train_tmp.pth")
    #     torch.save(save_state, saveEpochPath)
    # else:
    #     os.system("rm -rf %s"%os.path.join(path, f"ckpt_epoch_{epoch}_train_tmp.pth"))
    #     saveEpochPath = os.path.join(path, f"ckpt_epoch_{epoch}.pth")
    #     if epoch % SAVE_FREQ == 0 or epoch == TRAIN_EPOCHS:
    #         torch.save(save_state, saveEpochPath)
    saveEpochPath = os.path.join(path, f"ckpt_epoch_{epoch}.pth")
    if (epoch % SAVE_FREQ == SAVE_FREQ-1 or epoch == TRAIN_EPOCHS-1) and idx == -1 or ignore_step:
        torch.save(dict_states, saveEpochPath)
        logger.info('saved ckpt: ', saveEpochPath)
    if latest:
        torch.save(dict_states, os.path.join(path, f"ckpt_latest.pth"))
        torch.save(model_chk, os.path.join(path, f"model_latest.pth"))
        logger.info('saved to latest.')
    if best:
        torch.save(dict_states, os.path.join(path, f"ckpt_best.pth"))
        torch.save(model_chk, os.path.join(path, f"model_best.pth"))
        logger.info('saved to best.')
    logger.info(f"{path} saved !!!")


def correctWeightsForAcc(weights):
    correctWeights = {}
    for k, v in weights.items():
        new_k = k.replace('module.', '') if k.startswith('module.') else k
        correctWeights[new_k] = v
    return correctWeights


def train_one_epoch(model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()
    total = 0
    shoot = 0

    start = time.time()
    end = time.time()
    # trainLoop = tqdm(
    #     total=len(data_loader), desc=f"Train epoch[{epoch}/{TRAIN_EPOCHS}]"
    # )
    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        samples = samples.reshape([-1, 3, DATA_IMG_SIZE, DATA_IMG_SIZE])

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast(enabled=AMP_ENABLE):
            outputs = model(samples, labels=targets)
        loss = outputs.loss
        loss = loss / TRAIN_ACCUMULATION_STEPS

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(
            optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=TRAIN_CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % TRAIN_ACCUMULATION_STEPS == 0)
        if (idx + 1) % TRAIN_ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update(
                (epoch * num_steps + idx) // TRAIN_ACCUMULATION_STEPS)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        total += targets.shape[0]
        predict = torch.argmax(outputs.logits, 1)
        shoot += (predict == targets).sum().item()

        loss_meter.update(loss.item(), targets.size(0))
        if grad_norm is not None:  # loss_scaler return None if not update
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{TRAIN_EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'acc={shoot/total*100:.4f}%\t'
                f'mem {memory_used:.0f}MB')

        # trainLoop.update(1)
        # trainLoop.set_postfix_str(
        #     f'Train: [{epoch}/{TRAIN_EPOCHS}][{idx}/{num_steps}] | '
        #     f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) | '
        #     f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f}) | '
        #     f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f}) | '
        #     f'acc={shoot/total*100:.4f}% | '
        #     f'mem {memory_used:.0f}MB'
        # )
    epoch_time = time.time() - start
    logger.info(
        f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(model, data_loader):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    # valLoop = tqdm(
    #     total=len(data_loader), desc="Test"
    # )
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        images = images.reshape([-1, 3, DATA_IMG_SIZE, DATA_IMG_SIZE])

        # compute output
        with torch.cuda.amp.autocast(enabled=AMP_ENABLE):
            outputs = model(images)

        # measure accuracy and record loss
        loss = criterion(outputs.logits, target)
        acc1, acc5 = accuracy(outputs.logits, target, topk=(1, 5))

        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        if idx % PRINT_FREQ == 0:
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
        # valLoop.update(1)
        # valLoop.set_postfix_str(
        #     f"Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) | \n"
        #     + f"Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f}) | "
        #     + f"Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f}) | "
        #     f"Mem {memory_used:.0f}MB"
        # )
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    # valLoop.set_postfix_str(
    #     f" * Acc@1 {acc1_meter.avg:.3f}% | Acc@5 {acc5_meter.avg:.3f}%"
    # )
    # valLoop.close()
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


def main(args, logger):

    imageProcessor = AutoImageProcessor.from_pretrained(MODEL_DIR)

    trainDataset = MyDataset(
        DATA_DIR, dataType="train", updateIndex=DATA_INDEX_UPDATE, transform=imageProcessor
    )
    testDataset = MyDataset(
        DATA_DIR, dataType="test", updateIndex=DATA_INDEX_UPDATE, transform=imageProcessor
    )

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()

    trainSampler = DistributedSampler(
        trainDataset,
        num_replicas=num_tasks,
        rank=global_rank
    )

    testSampler = DistributedSampler(
        testDataset,
        num_replicas=num_tasks,
        rank=global_rank
    )

    trainDataloader = DataLoader(
        dataset=trainDataset,
        batch_size=DATA_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        sampler=trainSampler,
    )

    testDataloader = DataLoader(
        dataset=testDataset,
        batch_size=DATA_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        sampler=testSampler,
    )

    model = AutoModelForImageClassification.from_pretrained(
        MODEL_DIR,
        # num_labels=32,
        # id2label=tag2label,
        # label2id=label2tag
    )
    logger.info(str(model))
    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model, 'flops'):
        flops = model.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    criterion = None
    if AUG_MIXUP > 0.0:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif MODEL_LABEL_SMOOTHING > 0.0:
        criterion = LabelSmoothingCrossEntropy(smoothing=MODEL_LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        eps=TRAIN_OPTIMIZER_EPS,
        weight_decay=TRAIN_WEIGHT_DECAY,
        lr=TRAIN_BASE_LR,
        betas=TRAIN_OPTIMIZER_BETAS,
    )

    n_iter_per_epoch = len(trainDataloader) // TRAIN_ACCUMULATION_STEPS
    num_steps = int(TRAIN_EPOCHS * n_iter_per_epoch)
    warmup_steps = int(TRAIN_WARMUP_EPOCHS * n_iter_per_epoch)
    decay_steps = int(TRAIN_LR_SCHEDULER_DECAY_EPOCHS * n_iter_per_epoch)
    multi_steps = [i * n_iter_per_epoch for i in TRAIN_LR_SCHEDULER_MULTISTEPS]

    lr_scheduler = LinearLRScheduler(
        optimizer,
        t_initial=num_steps,
        lr_min_rate=0.01,
        warmup_lr_init=TRAIN_WARMUP_LR,
        warmup_t=warmup_steps,
        t_in_epochs=False,
    )
    loss_scaler = NativeScalerWithGradNormCount()

    best = {
        "acc1": 0,
        "acc5": 0,
    }

    model.cuda()
    model_without_ddp = model

    ###############################################################
    # Wrap the model
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[LOCAL_RANK], broadcast_buffers=False)
    ###############################################################

    global TRAIN_START_EPOCH

    resumePath = TRAIN_RESUME_FROM
    if resumePath:
        logger.info("cheaking resume path ...")
        if os.path.isdir(resumePath):
            if os.path.isfile(os.path.join(resumePath, 'chk', 'ckpt_latest.pth')):
                resumePath = os.path.join(resumePath, 'chk', 'ckpt_latest.pth')
            elif os.path.isfile(os.path.join(resumePath, 'ckpt_latest.pth')):
                resumePath = os.path.join(resumePath, 'ckpt_latest.pth')
        elif os.path.splitext(resumePath)[-1] == '.pth' and os.path.isfile(resumePath):
            pass
        else:
            logger.info("can not find model weight in specified path")
            exit('-1')
        '''
        "model_state",
        "optimizer_state",
        "lr_scheduler_state",
        "max_accuracy1",
        "max_accuracy5",
        "epoch",
        "idx",
        '''
        logger.info('loding checkpoint from %s' % resumePath)
        # chkValues = loadCheckpoint(resumePath)
        chkValues = torch.load(resumePath, map_location='cpu')
        model_without_ddp.load_state_dict(chkValues['model_state'], strict=False)
        lr_scheduler.load_state_dict(chkValues['lr_scheduler_state'])
        optimizer.load_state_dict(chkValues['optimizer_state'])
        loss_scaler.load_state_dict(chkValues['scaler'])
        best['acc1'] = chkValues['max_accuracy1']
        best['acc5'] = chkValues['max_accuracy5']
        TRAIN_START_EPOCH = chkValues['epoch']+1
        logger.info('checkpoint successfully loaded.')
        logger.info("resume from epoch: %d" % TRAIN_START_EPOCH)
    dist.barrier()

    logger.info("Start training")
    start_time = time.time()

    for epoch in range(TRAIN_START_EPOCH, TRAIN_EPOCHS):
        data_loader = trainDataloader
        data_loader.sampler.set_epoch(epoch)

        train_one_epoch(model, criterion, data_loader, optimizer, epoch, None, lr_scheduler,
                        loss_scaler)
        dist.barrier()

        data_loader = testDataloader
        acc1, acc5, loss = validate(model, data_loader)

        logger.info(
            f" Accuracy of the network on the {len(testDataloader)} test images [acc1:{acc1:.2f}% | acc5:{acc5:.2f}%]"
        )

        best1 = False
        best5 = False
        if acc1 >= best["acc1"]:
            best["acc1"] = acc1
            best1 == True
        if acc5 >= best["acc5"]:
            best["acc5"] = acc5
            best5 == True

        logger.info(
            "Max acc1: %.2f%% | Max acc5: %.2f%%" % (
                best["acc1"], best["acc5"])
        )
        dist.barrier()

        if dist.get_rank() == 0:
            saveCheckpoint(
                epoch,
                model_without_ddp,
                best["acc1"],
                best["acc5"],
                optimizer,
                lr_scheduler,
                loss_scaler,
                logger,
                path=chkPath,
                latest=True,
                best=best1,
            )
        dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))


if __name__ == "__main__":
    args = parse_option()

    DATA_DIR = './dataset'
    if args.batch_size:
        DATA_BATCH_SIZE = args.batch_size
    if args.train_epochs:
        TRAIN_EPOCHS = args.train_epochs
    if args.num_workers:
        NUM_WORKERS = args.num_workers
    if args.model_dir:
        MODEL_DIR = args.model_dir
    if args.data_dir:
        DATA_DIR = args.data_dir
    if args.output_dir:
        OUTPUT_DIR = args.output_dir
    if args.resume_from:
        TRAIN_RESUME_FROM = args.resume_from
    if args.save_step:
        SAVE_FREQ = args.save_step
    if args.data_img_size:
        DATA_IMG_SIZE=args.data_img_size
    DATA_INDEX_UPDATE = args.update_data_index
    
    # image_size
    # window_size
    configPath = os.path.join(args.model_dir, 'config.json')
    CONFIG = None
    if os.path.isfile(configPath):
        with open(configPath, 'r', encoding='utf8') as f:
            CONFIG = yaml.safe_load(f)
    if CONFIG:
        configKeys = CONFIG.keys()
        if 'window_size' in configKeys:
            MODEL_SWINV2_WINDOW_SIZE = CONFIG['window_size']
        if 'img_size' in configKeys:
            DATA_IMG_SIZE = CONFIG['image_size']

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        print(
            f"RANK and WORLD_SIZE in environ: {args.rank}/{args.world_size}:  Running ...")
    else:
        args.rank = -1
        args.world_size = -1

    LOCAL_RANK = args.local_rank
    torch.cuda.set_device(LOCAL_RANK)
    dist.init_process_group(
        backend='nccl', init_method='env://', world_size=args.world_size, rank=args.rank)
    dist.barrier()

    if LOCAL_RANK == 0:
        print(args)

    # 设置路径
    # runName = re.sub('[&/:*?"<>| ]', "_", time.ctime())
    logPath = os.path.join(OUTPUT_DIR, 'log')
    chkPath = os.path.join(OUTPUT_DIR, 'chk')
    pathChecker(logPath)
    pathChecker(chkPath)

    logger = create_logger(
        output_dir=logPath, dist_rank=dist.get_rank(), name=f"{MODEL_NAME}.log", print2console=True)
    dist.barrier()

    tags = ['car', 'truck', 'tank', 'armored_car', 'radar', 'artillery', 'person', 'bridge', 'building', 'airport']
    tag2label = {tags[i]: i for i in range(len(tags))}
    label2tag = {i: tags[i] for i in range(len(tags))}

    config = {"tag2label": tag2label, "label2tag": label2tag}
    config["split"] = {"train": 8, "test": 2}

    if dist.get_rank() == 0:
        with open(os.path.join(DATA_DIR, 'config.json'), "w") as f:
            json.dump(config, f)

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = TRAIN_BASE_LR * DATA_BATCH_SIZE * dist.get_world_size() / \
        512.0
    linear_scaled_warmup_lr = TRAIN_WARMUP_LR * \
        DATA_BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = TRAIN_MIN_LR * DATA_BATCH_SIZE * dist.get_world_size() / \
        512.0
    # gradient accumulation also need to scale the learning rate
    if TRAIN_ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * TRAIN_ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * TRAIN_ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * TRAIN_ACCUMULATION_STEPS

    TRAIN_BASE_LR = linear_scaled_lr
    TRAIN_WARMUP_LR = linear_scaled_warmup_lr
    TRAIN_MIN_LR = linear_scaled_min_lr

    main(args, logger)
