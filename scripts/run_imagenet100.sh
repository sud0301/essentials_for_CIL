#!/bin/bash

python main_imagenet.py  \
        --new-classes 10 \
        --start-classes 50 \
        --epochs 70 \
        --epochs-sd 40 \
        --kd \
        --w-kd 10 \
        --batch-size 128 \
        --K 2000 \
        --save-freq 10 \
        --weight-decay 5e-4 \
        --dataset imagenet100 \
        --exp-name 'imagenet100_kd_128_wd5e4_ep70_epsd40_50_10_k2000' \
        --num-sd 0 \
        --save
