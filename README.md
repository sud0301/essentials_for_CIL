# Essentials for Class Incremental Learning
Official repository of the paper 'Essentials for Class Incremental Learning'

This Pytorch repository contains the code for our work [Essentials for Class Incremental Learning](https://arxiv.org/abs/2102.09517). 

This work presents a straightforward class-incrmental learning system that focuses on the essential components and already exceeds the state of the art without integrating sophisticated modules. 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training and Evaluation (CIFAR-100, ImageNet-100, ImageNet-1k)

Following scripts contain both training and evaluation codes. Model is evaluated after each phase in class-IL.

### with Knowledge-distillation (KD)

To train the base CCIL model:
```
bash ./scripts/run_cifar.sh
bash ./scripts/run_imagenet100.sh
bash ./scripts/run_imagenet1k.sh
```

To train CCIL + Self-distillation
```
bash ./scripts/run_cifar_w_sd.sh
bash ./scripts/run_imagenet100_w_sd.sh
bash ./scripts/run_imagenet1k_w_sd.sh
```

## Results (CIFAR-100)

| Model name         | Avg Acc (5 iTasks)  | Avg Acc (10 iTasks) |
| ------------------ |------------------   | ----------------- |
| CCIL               |     66.44           |      64.86        |
| CCIL + SD          |     67.17           |      65.86        |


## Results (ImageNet-100)

| Model name         | Avg Acc (5 iTasks)  | Avg Acc (10 iTasks) |
| ------------------ |------------------   | ----------------- |
| CCIL               |     77.99           |      75.99        |
| CCIL + SD          |     79.44           |      76.77        |


## Results (ImageNet)

| Model name         | Avg Acc (5 iTasks)  | Avg Acc (10 iTasks) |
| ------------------ |------------------   | ----------------- |
| CCIL               |     67.53           |      65.61        |
| CCIL + SD          |     68.04           |      66.25        |

## List of Arguments

* Distillation Methods
    * Knowledge Distillation (--kd, --w-kd X), *X is the weightage for KD loss, default=1.0*
    * Representation Distillation (--rd, --w-rd X), *X is the weightage for cos-RD loss, default=0.05*
    * Contrastive Representation Distillation (--nce, --w-nce X), *only valid for CIFAR-100, X is the weightage of NCE loss*

* Regularization for the first task
    * Self-distillation (--num-sd X, --epochs-sd Y), *X is number of generations*, *Y is number of self-distillation epochs*
    * Mixup (--mixup, --mixup-alpha X), *X is mixup alpha value, default=0.1*
    * Heavy Augmentation (--aug)
    * Label Smoothing (--label-smoothing, --smoothing-alpha X), *X is a alpha value, default=0.1*

* Incremental class setting
    * No. of base classes (--start-classes 50)
    * 5-phases (--new-classes 10) 
    * 10-phases (--new-classes 5)

* Cosine learning rate decay (--cosine)

* Save and Load 
    * Experiment Name (--exp-name X)
    * Save checkpoints (--save)
    * Resume checkpoints (--resume, --resume-path X), *only to resume from first snapshot*

## Citation

```
@article{ccil_mittal,
    Author = {Sudhanshu Mittal and Silvio Galesso and Thomas Brox},
    Title = {Essentials for Class Incremental Learning},
    journal = {arXiv preprint arXiv:2102.09517},
    Year = {2021},
}
```
