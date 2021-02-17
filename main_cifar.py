import torch
torch.backends.cudnn.benchmark=True
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.transforms as transforms

import argparse
import os
from PIL import Image
import scipy.misc
import random
import copy
import math
import numpy as np

from data.data_loader import cifar10, cifar100, ExemplarDataset

from lib.util import moment_update, TransformTwice, weight_norm, mixup_data, mixup_criterion, LabelSmoothingCrossEntropy
from lib.augment.cutout import Cutout
from lib.augment.autoaugment_extra import CIFAR10Policy
from models import *

compute_means=True
exemplar_means_= []
avg_acc = []

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # training hyperparameters
    parser.add_argument('--batch-size', type=int, default=100, help='batch_size')
    parser.add_argument('--num-workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=120, help='number of training epochs')
    parser.add_argument('--epochs-sd', type=int, default=70, help='number of training epochs for self-distillation')
    parser.add_argument('--val-freq', type=int, default=10, help='validation frequency')
    
    # incremental learning    
    parser.add_argument('--new-classes', type=int, default=10, help='number of classes in new task')
    parser.add_argument('--start-classes', type=int, default=50, help='number of classes in old task')
    parser.add_argument('--K', type=int, default=2000, help='2000 exemplars for CIFAR-100')

    # optimization
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--lr-min', type=float, default=0.0001, help='lower end of cosine decay')
    parser.add_argument('--lr-sd', type=float, default=0.1, help='learning rate for self-distillation')
    parser.add_argument('--lr-ft', type=float, default=0.01, help='learning rate for task-2 onwards')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--cosine', action='store_true', help='use cosine learning rate')

    # root folders
    parser.add_argument('--data-root', type=str, default='./data', help='root directory of dataset')
    parser.add_argument('--output-root', type=str, default='./output', help='root directory for output')

    # save and load
    parser.add_argument('--exp-name', type=str, default='kd', help='experiment name')
    parser.add_argument('--resume', action='store_true', help='use class moco')
    parser.add_argument('--resume-path', type=str, default='./checkpoint_0.pth',)
    parser.add_argument('--save', action='store_true', help='to save checkpoint')

    # loss function
    parser.add_argument('--pow', type=float, default=0.66, help='hyperparameter of adaptive weight')
    parser.add_argument('--lamda', type=float, default=5, help='weighting of classification and distillation')
    parser.add_argument('--lamda-sd', type=float, default=10, help='weighting of classification and distillation')
    parser.add_argument('--const-lamda', action='store_true', help='use constant lamda value, default: adaptive weighting')

    parser.add_argument('--w-cls', type=float, default=1.0, help='weightage of new classification loss')

    # kd loss
    parser.add_argument('--kd', action='store_true', help='use kd loss')
    parser.add_argument('--w-kd', type=float, default=1.0, help='weightage of knowledge distillation loss')
    parser.add_argument('--T', type=float, default=2, help='temperature scaling for KD')
    parser.add_argument('--T-sd', type=float, default=2, help='temperature scaling for KD')

    # self-distillation
    parser.add_argument('--num-sd', type=int, default=0, help='number of self-distillation generations')
    parser.add_argument('--sd-factor', type=float, default=5.0, help='weighting between classification and distillation')
   
    # mixup
    parser.add_argument('--mixup', action='store_true', help='use mixup augmentation')
    parser.add_argument('--mixup-alpha', type=float, default=0.1, help='mixup alpha value') 
 
    # label smoothing
    parser.add_argument('--label-smoothing', action='store_true', help='use label smoothing')
    parser.add_argument('--smoothing-alpha', type=float, default=0.1, help='label smoothing alpha value') 
 
    # heave augmentation (Auto Augment)
    parser.add_argument('--aug', action='store_true', help='use heavy augmentation')
    parser.add_argument('--tsne', action='store_true', help='plot tsne after each incremental step')
    
    args = parser.parse_args()
    return args

def train(model, old_model, epoch, lr, tempature, lamda, train_loader, use_sd, checkPoint):

    tolerance_cnt = 0
    step = 0
    best_acc = 0
    T = args.T

    model.cuda()
    old_model.cuda()

    criterion_ce = nn.CrossEntropyLoss(ignore_index=-1)
    criterion_ce_smooth = LabelSmoothingCrossEntropy() # for label smoothing

    # reduce learning rate after first epoch (LowLR)
    if len(test_classes) // CLASS_NUM_IN_BATCH > 1:
        lr = args.lr_ft
   
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=args.weight_decay)

    if len(test_classes) // CLASS_NUM_IN_BATCH ==1 and use_sd ==True:
        if args.cosine:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch, eta_min=0.001)
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60], gamma=0.1)
    else:
        if args.cosine:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch, eta_min=args.lr_min)
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 90], gamma=0.1)

    if len(test_classes) // CLASS_NUM_IN_BATCH > 1:
        exemplar_set = ExemplarDataset(exemplar_sets, transform=transform_ori)
        exemplar_loader = torch.utils.data.DataLoader(exemplar_set, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
        exemplar_loader_iter = iter(exemplar_loader)

        old_model.eval()
        num_old_classes = old_model.fc.out_features
        
    for epoch_index in range(1, epoch+1):

        dist_loss = 0.0
        sum_loss = 0
        sum_dist_loss = 0
        sum_cls_new_loss = 0
        sum_cls_old_loss = 0
        sum_cls_loss = 0

        model.train()
        old_model.eval()
        old_model.freeze_weight()
        for param_group in optimizer.param_groups:
            print('learning rate: {:.4f}'. format(param_group['lr']))

        for batch_idx, (x, x1, target) in enumerate(train_loader):

            optimizer.zero_grad()

            # Classification Loss: New task
            x, target = x.cuda(), target.cuda()
            
            targets = target-len(test_classes)+CLASS_NUM_IN_BATCH

            # use mixup for task-1
            if args.mixup:
                inputs, targets_a, targets_b, lam = mixup_data(x, targets, args.mixup_alpha)
                inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))

                logits = model(inputs)
                outputs = logits[:,-CLASS_NUM_IN_BATCH:]
                cls_loss_new = mixup_criterion(criterion_ce, outputs, targets_a, targets_b, lam)

            # use label smoothing for task-1
            elif args.label_smoothing:
                logits = model(x)
                cls_loss_new = criterion_ce_smooth(logits[:,-CLASS_NUM_IN_BATCH:], targets, args.smoothing_alpha)

            else:
                logits = model(x)
                cls_loss_new = criterion_ce(logits[:,-CLASS_NUM_IN_BATCH:], targets)
    
            loss = args.w_cls*cls_loss_new
            sum_cls_new_loss += cls_loss_new.item()
       
            # use fixed lamda value or adaptive weighting 
            if args.const_lamda:    
                factor = args.lamda
            elif use_sd:
                factor = args.lamda_sd
            else:
                factor = ((len(test_classes)/CLASS_NUM_IN_BATCH)**(args.pow))*args.lamda
             
            # while using self-distillation 
            if len(test_classes) // CLASS_NUM_IN_BATCH == 1 and use_sd:
                if args.kd:
                    with torch.no_grad():
                        dist_target = old_model(x)
                    logits_dist = logits
                    T_sd = args.T_sd
                    dist_loss = nn.KLDivLoss()(F.log_softmax(logits_dist/T_sd, dim=1), F.softmax(dist_target/T_sd, dim=1)) * (T_sd*T_sd)  # best model
                    sum_dist_loss += dist_loss.item()
        
                    loss += factor*args.w_kd*dist_loss
            
            # Distillation : task-2 onwards
            if len(test_classes) // CLASS_NUM_IN_BATCH > 1:

                if args.kd:
                    with torch.no_grad():
                        dist_target = old_model(x)
                    logits_dist = logits[:, :-CLASS_NUM_IN_BATCH]
                    T = args.T
                    dist_loss_new = nn.KLDivLoss()(F.log_softmax(logits_dist/T, dim=1), F.softmax(dist_target/T, dim=1)) * (T*T)
                
                try:
                    batch_ex = next(exemplar_loader_iter)
                except:
                    exemplar_loader_iter = iter(exemplar_loader)
                    batch_ex = next(exemplar_loader_iter)

                # Classification loss: exemplar classes loss
                x_old, target_old = batch_ex
                x_old , target_old = x_old.cuda(), target_old.cuda()
                logits_old = model(x_old)
                 
                old_classes = len(test_classes) - CLASS_NUM_IN_BATCH
                cls_loss_old = criterion_ce(logits_old, target_old)
                
                loss += cls_loss_old
                sum_cls_old_loss += cls_loss_old.item()
                
                if args.kd: 
                    # KD exemplar
                    with torch.no_grad():
                        dist_target_old = old_model(x_old)
                    logits_dist_old = logits_old[:, :-CLASS_NUM_IN_BATCH]
                    dist_loss_old = nn.KLDivLoss()(F.log_softmax(logits_dist_old/T, dim=1), F.softmax(dist_target_old/T, dim=1)) * (T*T)  # best model

                    dist_loss =  dist_loss_old + dist_loss_new
                    sum_dist_loss += dist_loss.item()
                    loss += factor*args.w_kd*dist_loss
                
            sum_loss += loss.item()

            loss.backward()
            optimizer.step()
            step += 1
            
            if (batch_idx + 1) % checkPoint == 0 or (batch_idx + 1) == len(trainLoader):
                print('==>>> epoch: {}, batch index: {}, step: {}, train loss: {:.3f}, dist_loss: {:3f}, cls_new_loss: {:.3f}, cls_old_loss: {:.3f}'.
                      format(epoch_index, batch_idx + 1, step, sum_loss/(batch_idx+1), sum_dist_loss/(batch_idx+1), sum_cls_new_loss/(batch_idx+1),  sum_cls_old_loss/(batch_idx+1)))
        scheduler.step()
        
def evaluate_net(model, transform, train_classes, test_classes):
    model.eval()

    train_set = cifar100(root=args.data_root,
                         train=False,
                         classes=train_classes,
                         download=False,
                         transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    total = 0.0
    correct = 0.0
    compute_means = True
    for j, (_, images, labels) in enumerate(train_loader):
        _, preds = torch.max(torch.softmax(model(images.cuda()), dim=1), dim=1, keepdim=False)
        labels = [y.item() for y in labels]
        np.asarray(labels)
        total += preds.size(0)
        correct += (preds.cpu().numpy() == labels).sum()
       
    # Train Accuracy
    print ('correct: ', correct, 'total: ', total)
    print ('Train Accuracy : %.2f ,' % (100.0 * correct / total))

    test_set = cifar100(root=args.data_root,
                         train=False,
                         classes=test_classes,
                         download=True,
                         transform=transform)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4) 

    total = 0.0
    correct = 0.0
    for j, (_, images, labels) in enumerate(test_loader):
        out = torch.softmax(model(images.cuda()), dim=1)
        _, preds = torch.max(out, dim=1, keepdim=False)
        labels = [y.item() for y in labels]
        np.asarray(labels)
        total += preds.size(0)
        correct += (preds.cpu().numpy() == labels).sum()

    # Test Accuracy
    test_acc =  100.0*correct/total
    print ('correct: ', correct, 'total: ', total)
    print ('Test Accuracy : %.2f' % test_acc)

    return test_acc

def icarl_reduce_exemplar_sets(m):
    for y, P_y in enumerate(exemplar_sets):
        exemplar_sets[y] = P_y[:m]

#Construct an exemplar set for image set
def icarl_construct_exemplar_set(model, images, m, transform):
    
    model.eval()
    # Compute and cache features for each example
    features = []
    with torch.no_grad():
        for img in images:
            x = Variable(transform(Image.fromarray(img))).cuda()
            x=x.unsqueeze(0)
            feat = model.forward(x, rd=True).data.cpu().numpy()
            feat = feat / np.linalg.norm(feat) # Normalize
            features.append(feat[0])

        features = np.array(features)
        class_mean = np.mean(features, axis=0)
        class_mean = class_mean / np.linalg.norm(class_mean) # Normalize

        exemplar_set = []
        exemplar_features = [] # list of Variables of shape (feature_size,)
        exemplar_dist = []
        for k in range(int(m)):
            S = np.sum(exemplar_features, axis=0)
            phi = features
            mu = class_mean
            mu_p = 1.0/(k+1) * (phi + S)
            mu_p = mu_p / np.linalg.norm(mu_p)
            dist = np.sqrt(np.sum((mu - mu_p) ** 2, axis=1))
   
            idx = np.random.randint(0, features.shape[0])

            exemplar_dist.append(dist[idx])
            exemplar_set.append(images[idx])
            exemplar_features.append(features[idx])
            features[idx, :] = 0.0

        # random exemplar selection
        exemplar_dist = np.array(exemplar_dist)
        exemplar_set = np.array(exemplar_set)
        ind = exemplar_dist.argsort()
        exemplar_set = exemplar_set[ind]

        exemplar_sets.append(np.array(exemplar_set))
    print ('exemplar set shape: ', len(exemplar_set))

if __name__ == '__main__':
    args = parse_option()
    print (args)

    if not os.path.exists(os.path.join(args.output_root, "checkpoints/cifar/")):
        os.makedirs(os.path.join(args.output_root, "checkpoints/cifar/"))

    #  parameters
    TOTAL_CLASS_NUM = 100
    CLASS_NUM_IN_BATCH = args.start_classes
    TOTAL_CLASS_BATCH_NUM = TOTAL_CLASS_NUM // CLASS_NUM_IN_BATCH
    T = args.T

    K = args.K
    exemplar_sets = []
    exemplar_means = []    
    compute_means = True

    normalize = transforms.Normalize((0.5071,  0.4866,  0.4409), (0.2009,  0.1984,  0.2023))

    # Heavy-augmentation
    transform_aug = transforms.Compose([
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16),
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        CIFAR10Policy(),
        transforms.ToTensor(),
        normalize,
    ])

    # default augmentation
    transform_ori = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    # test-time augmentation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    transform_train = TransformTwice(transform_ori, transform_ori)

    class_index = [i for i in range(0, TOTAL_CLASS_NUM)]
    np.random.seed(1993)
    np.random.shuffle(class_index)

    net = resnet32_cifar(num_classes=CLASS_NUM_IN_BATCH).cuda()
    
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print ('number of trainable parameters: ', params)

    old_net = copy.deepcopy(net)
    old_net.cuda()
   
    cls_list = [0] + [a for a in range(args.start_classes, 100, args.new_classes)] 

    for i in cls_list:
        if i == args.start_classes:
            CLASS_NUM_IN_BATCH = args.new_classes
    
        print("==> Current Class: ", class_index[i:i+CLASS_NUM_IN_BATCH])
        print('==> Building model..')

        if i == args.start_classes:
            net.change_output_dim(new_dim=i+CLASS_NUM_IN_BATCH)
        if i > args.start_classes:
            net.change_output_dim(new_dim=i+CLASS_NUM_IN_BATCH, second_iter=True)
        
        print("current net output dim:", net.get_output_dim())

        # while using heavy augmentation
        if args.aug:
            if i==0:
                transform_train = TransformTwice(transform_aug, transform_aug)
                print ('.............augmentation.............')
            else:
                transform_train = TransformTwice(transform_ori, transform_ori)
        
        train_set = cifar100(root=args.data_root,
							 train=True,
							 classes=class_index[i:i+CLASS_NUM_IN_BATCH],
							 download=True,
							 transform=transform_train)
    
        trainLoader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)


        train_classes = class_index[i:i+CLASS_NUM_IN_BATCH]
        test_classes = class_index[:i+CLASS_NUM_IN_BATCH]

        print (train_classes)
        print (test_classes)    

        m = K // (i+CLASS_NUM_IN_BATCH)
        
        if i!=0:
            icarl_reduce_exemplar_sets(m) 
         
        for y in range(i, i+CLASS_NUM_IN_BATCH):
            print ("Constructing exemplar set for class-%d..." %(class_index[y]))
            images = train_set.get_image_class(y)
            icarl_construct_exemplar_set(net, images, m, transform_test)
            print ("Done")
        
        # train and save model
        if args.resume and i==0:
            net.load_state_dict(torch.load(args.resume_path))
            net.train()
        else:
            net.train()
            train(model=net, old_model=old_net, epoch=args.epochs, lr=args.lr, tempature=T, lamda=args.lamda, train_loader=trainLoader, use_sd=False, checkPoint=50)

        # print weight norm: task:2 onwards
        if i!=0:
            weight_norm(net)

        old_net = copy.deepcopy(net)
        old_net.cuda()

        # Do self-distillation
        if i == 0 and not args.resume:
            for sd in range(args.num_sd):
                train(model=net, old_model=old_net, epoch=args.epochs_sd, lr=args.lr_sd, tempature=T, lamda=args.lamda,train_loader=trainLoader, use_sd=True, checkPoint=50)
                old_net = copy.deepcopy(net)
                old_net.cuda()
        
        if args.save:
            save_path = os.path.join(args.output_root, "checkpoints/cifar/", args.exp_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(net.state_dict(), os.path.join(save_path, 'checkpoint_' + str(i+CLASS_NUM_IN_BATCH) + '.pth'))

        # Evaluation on training and testing set

        transform_val = TransformTwice(transform_test, transform_test)
        test_acc = evaluate_net(model=net, transform=transform_val, train_classes=class_index[i:i+CLASS_NUM_IN_BATCH], test_classes=class_index[:i+CLASS_NUM_IN_BATCH])
        avg_acc.append(test_acc)
    
    print (avg_acc)
    print ('Avg accuracy: ', sum(avg_acc)/len(avg_acc))
