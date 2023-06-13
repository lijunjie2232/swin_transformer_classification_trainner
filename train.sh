#!/bin/bash

conda activate hf_torch2
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

accelerate launch \
    --main_process_port 12321 \
    --config_file ./acConfig_1MC_8GPU.yaml SWTClassification.py \
    --devices=0,1 \
    --batch_size=112 \
    --train_epochs=10 \
    --num_workers=12 \
    --data_dir /mnt/tmp/test1 \
    --save_step=2
