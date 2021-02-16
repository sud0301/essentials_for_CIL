#!/bin/bash

python main_cifar.py  \
        --new-classes 10 \
        --start-classes 50 \
        --cosine \
        --kd \
        --w-kd 1 \
        --epochs 120 \
        --exp-name 'kd_ep120_50_10' \
        --save \
        --num-sd 0
