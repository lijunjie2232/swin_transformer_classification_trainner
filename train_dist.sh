#!/bin/bash

conda activate hf_torch2
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -m torch.distributed.launch --nproc_per_node=8 \
    SWTClassification_dist.py \
    --batch_size=40 \
    --train_epochs=10 \
    --num_workers=4 \
    --data_dir /mnt/tmp/test1 \
    --save_step=1 \
    --model_dir ./transformers/models/swinv2-small-patch4-window8-256 \
    --output_dir ./runs/swinS/exp0
