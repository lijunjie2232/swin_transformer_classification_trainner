import os
import re
import pandas as pd

# import numba
# from numba import jit
import random
import time
from tqdm import tqdm
import json
import logging
import datetime
import argparse

# from concurrent import futures
# from xml.dom.minidom import parse
# import xml.dom.minidom
# from tensorflow.image import crop_to_bounding_box as boxcp
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import torch.distributed as dist
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler.scheduler import Scheduler
from timm.utils import accuracy, AverageMeter
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
)

# from swin_transformer_v2 import SwinTransformerV2
from accelerate import Accelerator
from MyDataset import MyDataset
from utils import showImage, readImage, saveImage, pathChecker, reduce_tensor


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
        type=bool,
        default=False,
        help="update dataset index file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='./runs/swinS',
        help="path of output"
    )
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
    accelerator,
    epoch,
    model,
    max_accuracy1,
    max_accuracy5,
    optimizer,
    lr_scheduler,
    logger,
    path,
    loss_scaler=None,
    latest=True,
    best1=False,
    best5=False,
    idx=-1,
):
    save_state = {
        #"model": model,
        "model_state": model.state_dict(),
        #"optimizer": optimizer,
        "optimizer_state": optimizer.state_dict(),
        #"lr_scheduler": lr_scheduler,
        "lr_scheduler_state": lr_scheduler.state_dict(),
        "max_accuracy1": max_accuracy1,
        "max_accuracy5": max_accuracy5,
        # 'scaler': loss_scaler.state_dict(),
        "epoch": epoch,
        "idx": idx,
    }
    if accelerator.process_index > 0:
        return

    pathChecker(path)
    logger.info(f"{path} saving......")
    print(f"{path} saving......")
    # if idx >= 0:
    #     #saveEpochPath = os.path.join(path, f"ckpt_epoch_{epoch}_{idx}.pth")
    #     saveEpochPath = os.path.join(path, f"ckpt_epoch_{epoch}_train_tmp.pth")
    #     torch.save(save_state, saveEpochPath)
    # else:
    #     os.system("rm -rf %s"%os.path.join(path, f"ckpt_epoch_{epoch}_train_tmp.pth"))
    #     saveEpochPath = os.path.join(path, f"ckpt_epoch_{epoch}.pth")
    #     if epoch % CHK_SAVE_STEP == 0 or epoch == TRAIN_EPOCHS:
    #         torch.save(save_state, saveEpochPath)
    saveEpochPath = os.path.join(path, f"ckpt_epoch_{epoch}.pth")
    if epoch % CHK_SAVE_STEP == CHK_SAVE_STEP-1 and idx == -1:
        torch.save(save_state, saveEpochPath)
        print('saved to: ', saveEpochPath)
    #accelerator.save(save_state, saveEpochPath)
    if latest:
        torch.save(save_state, os.path.join(path, f"_latest.pth"))
        #accelerator.save(save_state, os.path.join(path, f"_latest.pth"))
        print('saved to: ', os.path.join(path, f"_latest.pth"))
    if best1:
        torch.save(save_state, os.path.join(path, f"_best.pth"))
        #accelerator.save(save_state, os.path.join(path, f"_best.pth"))
        print('saved to: ', os.path.join(path, f"_best.pth"))
    if best5:
        torch.save(save_state, os.path.join(path, f"_best5.pth"))
        #accelerator.save(save_state, os.path.join(path, f"_best5.pth"))
        print('saved to: ', os.path.join(path, f"_best5.pth"))
    logger.info(f"{path} saved !!!")


def main(ckpPath, trainDataloader, valDataloader, logger, savePath):
    
    model = AutoModelForImageClassification.from_pretrained(
        ckpPath,
        # num_labels=32,
        # id2label=tag2label,
        # label2id=label2tag
    )
    
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

    n_iter_per_epoch = len(trainDataloader)
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
    
    # loss_scaler=loss_scaler

    # modelIsCuda = next(model.parameters()).is_cuda
    # print('cuda=' + 'true' if modelIsCuda else 'false')
    # if device.type == 'cuda' and not modelIsCuda:
    # print('cuda device detected, try to transform model into cuda')
    # accelerator = Accelerator()
    # model, optimizer, trainDataloader, scheduler = accelerator.prepare(
    #     model,
    #     optimizer,
    #     trainDataloader,
    #     lr_scheduler
    # )
    # modelIsCuda = next(model.parameters()).is_cuda
    # print('transform result: cuda='+ 'true' if modelIsCuda else 'false')

    logger.info("Start training")
    start_time = time.time()
    # TRAIN_EPOCHS=20
    best = {
        "acc1": 0,
        "acc5": 0,
    }        
    accelerator = Accelerator()
    rank = accelerator.process_index
    model, optimizer, trainDataloader, valDataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, trainDataloader, valDataloader, lr_scheduler
    )
    for epoch in range(TRAIN_START_EPOCH, TRAIN_EPOCHS):
        # train_one_epoch(model, LOSS_FUNC, trainDataloader, optimizer, lr_scheduler, loss_scaler, epoch)

        data_loader = trainDataloader
        iscuda = False
        if device.type == "cuda":
            model = model.cuda()
            iscuda = True

        model.train()
        optimizer.zero_grad()

        num_steps = len(data_loader)
        batch_time = AverageMeter()
        loss_meter = AverageMeter()
        # norm_meter = AverageMeter()
        # scaler_meter = AverageMeter()
        total = 0
        shoot = 0

        # idx = 0
        start = time.time()
        lastSave = start
        end = time.time()
        trainLoop = tqdm(
            total=len(data_loader), desc=f"Train epoch[{epoch}/{TRAIN_EPOCHS}]"
        )
        for idx, (samples, targets) in enumerate(data_loader):
            if iscuda:
                samples = samples.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)
            samples = samples.reshape([-1, 3, DATA_IMG_SIZE, DATA_IMG_SIZE])
            if iscuda:
                with torch.cuda.amp.autocast(enabled=AMP_ENABLE):
                    outputs = model(samples, labels=targets)
            else:
                outputs = model(samples, labels=targets)
            loss = outputs.loss
            loss_meter.update(loss.item(), targets.size(0))
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step((epoch * num_steps + idx) // TRAIN_ACCUMULATION_STEPS)
            # loss = loss / TRAIN_ACCUMULATION_STEPS
            # is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            # grad_norm = loss_scaler(loss, optimizer, clip_grad=TRAIN_CLIP_GRAD,
            #                         parameters=model.parameters(), create_graph=is_second_order,
            #                         update_grad=(idx + 1) % TRAIN_ACCUMULATION_STEPS == 0)
            # if (idx + 1) % TRAIN_ACCUMULATION_STEPS == 0:
            #     optimizer.zero_grad()
            #     lr_scheduler.step_update((epoch * num_steps + idx) // TRAIN_ACCUMULATION_STEPS)
            # loss_scale_value = loss_scaler.state_dict()["scale"]

            total += targets.shape[0]
            predict = torch.argmax(outputs.logits, 1)
            shoot += (predict == targets).sum().item()

            # if iscuda:
            #     torch.cuda.synchronize()
            # loss_meter.update(loss.item(), targets.size(0))
            # if grad_norm is not None:  # loss_scaler return None if not update
            #     norm_meter.update(grad_norm)
            # scaler_meter.update(loss_scale_value)

            memory_used = 0
            if iscuda:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)

            trainLoop.update(1)
            trainLoop.set_postfix_str(
                f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) | "
                +
                # f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f}) | '+
                # f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f}) | '+
                f"acc={shoot/total*100:.4f}% | "
                # f'mem {memory_used:.0f}MB'
            )

            if idx % PRINT_FREQ == 0:
                lr = optimizer.param_groups[0]["lr"]
                wd = optimizer.param_groups[0]["weight_decay"]
                etas = batch_time.avg * (num_steps - idx)
                logger.info(
                    f"Train: [{epoch}/{TRAIN_EPOCHS}][{idx}/{num_steps}]\t"
                    # f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                    f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
                    # f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                    # f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                    f"acc={shoot/total*100:.4f}%\t"
                    f"mem {memory_used:.0f}MB"
                )
            if time.time() - lastSave > 1800:
                saveCheckpoint(
                    accelerator,
                    epoch,
                    model,
                    shoot / total,
                    shoot / total,
                    optimizer,
                    lr_scheduler,
                    logger,
                    path=savePath,
                    latest=True,
                    best1=False,
                    best5=False,
                    idx=idx,
                )
                lastSave = time.time()
        trainLoop.close()
        epoch_time = time.time() - start
        logger.info(
            f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}, loss={loss_meter.avg:.4f}, acc={shoot/total*100:.4f}%"
        )

        # acc1, acc5, loss = validate(model, valDataloader, epoch=epoch)

        data_loader = valDataloader
        epoch = epoch
        SAVE_FREQ = 0

        # Testcriterion = torch.nn.CrossEntropyLoss()
        iscuda = False
        if device.type == "cuda":
            model = model.cuda()
            iscuda = True

        model.eval()

        with torch.no_grad():
            batch_time = AverageMeter()
            loss_meter = AverageMeter()
            acc1_meter = AverageMeter()
            acc5_meter = AverageMeter()

            # idx = 0
            end = time.time()
            valLoop = tqdm(
                total=len(data_loader), desc=f"Test epoch[{epoch}/{TRAIN_EPOCHS}]"
            )
            for idx, (images, targets) in enumerate(data_loader):
                if iscuda:
                    images = images.cuda(non_blocking=True)
                    targets = targets.cuda(non_blocking=True)
                images = images.reshape([-1, 3, DATA_IMG_SIZE, DATA_IMG_SIZE])

                # compute output
                if iscuda:
                    with torch.cuda.amp.autocast(enabled=AMP_ENABLE):
                        outputs = model(images, labels=targets)
                else:
                    outputs = model(images, labels=targets)

                # measure accuracy and record loss
                # loss = Testcriterion(outputs.logits, target)
                loss = outputs.loss
                acc1, acc5 = accuracy(outputs.logits, targets, topk=(1, 5))

                acc1 = reduce_tensor(acc1)
                acc5 = reduce_tensor(acc5)
                loss = reduce_tensor(loss)
                # print(acc1, acc5, loss)

                loss_meter.update(loss.item(), targets.size(0))
                acc1_meter.update(acc1.item(), targets.size(0))
                acc5_meter.update(acc5.item(), targets.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                memory_used = 0
                if iscuda:
                    memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                valLoop.update(1)
                valLoop.set_postfix_str(
                    f"Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) | "
                    + f"Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f}) | "
                    + f"Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f}) | "
                    f"Mem {memory_used:.0f}MB"
                )

                if idx % PRINT_FREQ == 0:
                    # memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                    logger.info(
                        f"Test: [{idx}/{len(data_loader)}]\t"
                        f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                        f"Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
                        f"Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t"
                        f"Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t"
                        f"Mem {memory_used:.0f}MB"
                    )

        valLoop.set_postfix_str(
            f" * Acc@1 {acc1_meter.avg:.3f}% | Acc@5 {acc5_meter.avg:.3f}% | Loss {loss_meter.avg:.4f}"
        )
        valLoop.close()
        logger.info(f" * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}")

        acc1, acc5, loss = acc1_meter.avg, acc5_meter.avg, loss_meter.avg
        logger.info(
            f" Accuracy of the network on the {len(valDataloader)} test images [acc1:{acc1:.2f}% | acc5:{acc5:.2f}%]"
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
            "Max acc1: %.2f%% | Max acc1: %.2f%%" % (best["acc1"], best["acc5"])
        )

        saveCheckpoint(
            accelerator,
            epoch,
            model,
            best["acc1"],
            best["acc5"],
            optimizer,
            lr_scheduler,
            logger,
            path=savePath,
            latest=True,
            best1=best1,
            best5=best5,
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))


if __name__ == "__main__":
    AMP_ENABLE = True
    TRAIN_ACCUMULATION_STEPS = 1
    TRAIN_CLIP_GRAD = 1
    PRINT_FREQ = 1

    AUG_MIXUP = 0.8
    MODEL_LABEL_SMOOTHING = 0.1
    WEIGHT_DECAY: 1e-8
    
    DATA_BATCH_SIZE = int(32)
    DATA_PATH = './data/mydataset/dataset_811'

    TRAIN_START_EPOCH = int(0)
    TRAIN_EPOCHS = int(300)
    TRAIN_WARMUP_EPOCHS = 2
    TRAIN_WARMUP_LR = 5e-7
    TRAIN_WEIGHT_DECAY = 0.05
    TRAIN_BASE_LR = 1e-5
    TRAIN_MIN_LR = 1e-7
    TRAIN_OPTIMIZER_BETAS = (0.9, 0.999)
    TRAIN_LR_SCHEDULER_DECAY_EPOCHS = int(30)
    TRAIN_LR_SCHEDULER_MULTISTEPS = []
    TRAIN_OPTIMIZER_EPS = 1e-8

    TEST_SHUFFLE = False

    MODEL_TYPE = "swinv2"
    # MODEL_NAME = "swinv2_large_patch4_window12to24_192to384_22kto1k_ft"
    MODEL_DROP_PATH_RATE = 0.2
    MODEL_NUM_CLASSES = 1000
    # MODEL_NUM_CLASSES = 32
    MODEL_DROP_PATH_RATE = 0.1

    DATA_IMG_SIZE = 256

    #MODEL_SWINV2_PATCH_SIZE = 4
    #MODEL_SWINV2_IN_CHANS = 3

    #MODEL_SWINV2_EMBED_DIM = 192
    #MODEL_SWINV2_DEPTHS = [2, 2, 18, 2]
    #MODEL_SWINV2_NUM_HEADS = [6, 12, 24, 48]
    MODEL_SWINV2_WINDOW_SIZE = 8
    #MODEL_SWINV2_PRETRAINED_WINDOW_SIZES = [12, 12, 12, 6]

    #MODEL_SWINV2_MLP_RATIO = 4.0
    #MODEL_SWINV2_QKV_BIAS = True
    #MODEL_SWINV2_APE = False
    #MODEL_SWINV2_PATCH_NORM = True

    CHK_SAVE_STEP = int(5)
    NUM_WORKERS=int(5)

    args = parse_option()
    # if args.devices=='all':
    #     os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    #     if torch.cuda.device_count() > 1:
    #         for i in range(1, torch.cuda.device_count()):
    #             os.environ["CUDA_VISIBLE_DEVICES"] += ',%d'%i
    if args.devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
        # TRAIN_ACCUMULATION_STEPS = 0
        # for i in args.devices.split(','):
        #     if i:
        #         TRAIN_ACCUMULATION_STEPS += 1

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
    #if args.update_data_index is not None:
    DATA_INDEX_UPDATE = args.update_data_index

    print(args)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # 设置打印级别
    formatter = logging.Formatter(
        "%(asctime)s %(filename)s %(funcName)s [line:%(lineno)d] %(levelname)s %(message)s"
    )

    # 设置屏幕打印的格式
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    # logger.addHandler(sh)

    # 设置路径
    runName = re.sub('[&/:*?"<>| ]', "_", time.ctime())
    outputPath = os.path.join(OUTPUT_DIR, runName)
    logPath = os.path.join(outputPath, 'log')
    chkPath = os.path.join(outputPath, 'chk')
    #if not os.path.exists(chkPath):
    #    try:
    #        os.makedirs(chkPath)
    #    except OSError as exc:
    #        print(exc)
    #        pass
    pathChecker(logPath)
    pathChecker(chkPath)

    # 设置log保存
    fh = logging.FileHandler(os.path.join(logPath, 'logs.log'), encoding="utf8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataPath = DATA_DIR
    tags = ["tent", "car", "truck", "human", "bridge", "bg"]
    tag2label = {tags[i]: i for i in range(len(tags))}
    label2tag = {i: tags[i] for i in range(len(tags))}
    
    config = {"tag2label": tag2label, "label2tag": label2tag}
    config["split"] = {"train": 8, "test":2}
    with open(os.path.join(DATA_DIR, 'config.json'), "w") as f:
        json.dump(config, f)

    imageProcessor = AutoImageProcessor.from_pretrained(MODEL_DIR)

    trainDataset = MyDataset(
        dataPath, dataType="train", updateIndex=DATA_INDEX_UPDATE, imageProcessor=imageProcessor
    )
    valDataset = MyDataset(
        dataPath, dataType="test", updateIndex=DATA_INDEX_UPDATE, imageProcessor=imageProcessor
    )
    # testDataset = MyDataset(dataPath, dataType='test', updateIndex=False, imageProcessor=imageProcessor)

    # if TRAIN_ACCUMULATION_STEPS > 1:
    #     trainSampler = torch.utils.data.DistributedSampler(
    #         trainDataset,
    #         num_replicas=dist.get_world_size(),
    #         rank=dist.get_rank(),
    #         shuffle=True,
    #     )
    # else:
    #     trainSampler = torch.utils.data.RandomSampler(trainDataset)
    trainSampler = torch.utils.data.RandomSampler(trainDataset)
    valSampler = torch.utils.data.SequentialSampler(valDataset)
    trainDataloader = DataLoader(
        dataset=trainDataset,
        batch_size=DATA_BATCH_SIZE,
        num_workers=min(DATA_BATCH_SIZE*8, NUM_WORKERS),
        pin_memory=True,
        drop_last=True,
        # sampler=trainSampler,
        shuffle=True
    )
    # valDataloader = DataLoader(
    #     dataset=valDataset,
    #     batch_size=DATA_BATCH_SIZE,
    #     num_workers=DATA_BATCH_SIZE,
    #     pin_memory=True,
    #     drop_last=True,
    #     sampler=valSampler,
    # )
    valDataloader = DataLoader(
        dataset=valDataset,
        batch_size=1,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
        # sampler=valSampler,
        shuffle=True
    )

    main(
        ckpPath=MODEL_DIR,
        trainDataloader=trainDataloader,
        valDataloader=valDataloader,
        logger=logger,
        savePath=chkPath,
    )
