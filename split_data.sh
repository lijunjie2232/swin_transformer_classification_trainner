#!/bin/bash

conda activate hf_torch2
python split.py --data_path /mnt/tmp/test1 --train=8 --test=1 val=2 --make_index
