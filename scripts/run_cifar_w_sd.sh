#!/bin/bash

python main_cifar.py  \
        --new-classes 10 \
        --start-classes 50 \
        --cosine \
        --kd \
        --w-kd 1 \
        --epochs 120 \
        --epochs-sd 70 \
        --exp-name 'kd_ep120_w_sdx1_50_10' \
        --save \
        --num-sd 1
