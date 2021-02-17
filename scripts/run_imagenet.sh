!/bin/bash

python main_imagenet.py  \
        --new-classes 10 \
        --start-classes 50 \
        --epochs 70 \
        --kd \
        --w-kd 60 \
        --batch-size 128 \
        --K 2000 \
        --save-freq 10 \
        --weight-decay 1e-4 \
        --dataset imagenet \
        --exp-name 'imagenet_kd_128_wd1e4_ep70_50_10_k2000' \
        --num-sd 0 \
        --save
