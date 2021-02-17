import argparse
import os
from PIL import Image
import scipy.misc
import random
import numpy as np
import math
import copy
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
torch.backends.cudnn.benchmark=True
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from models import *
from data.data_loader_imagenet import ExemplarDataset
from data.data_loader_imagenet import ImageNet100, ImageNet1K

from lib.util import moment_update, TransformTwice, weight_norm, weight_norm_dot, mixup_data, mixup_criterion, LabelSmoothingCrossEntropy

compute_means=True
exemplar_means_= []
avg_acc = []
exemplar_sets = []

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # training hyperparameters
    parser.add_argument('--batch-size', type=int, default=256, help='batch_size')
    parser.add_argument('--num-workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=40, help='number of training epochs')
    parser.add_argument('--start-epoch', type=int, default=1, help='number of training epochs')
    parser.add_argument('--epochs-sd', type=int, default=15, help='number of training epochs for self-distillation')
    parser.add_argument('--K', type=int, default=2000, help='memory budget')
    parser.add_argument('--save-freq', type=int, default=1, help='memory budget')
    
    # incremental learning    
    parser.add_argument('--new-classes', type=int, default=10, help='number of classes in new task')
    parser.add_argument('--start-classes', type=int, default=50, help='number of classes in old task')

    # optimization
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--lr-sd', type=float, default=0.01, help='learning rate for self-distillation')
    parser.add_argument('--lr-ft', type=float, default=0.01, help='learning rate for task-2 onwards')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--cosine', action='store_true', help='use cosine learning rate')

    # root folders
    parser.add_argument('--train-data-root', type=str, default='./data', help='root directory of dataset')
    parser.add_argument('--test-data-root', type=str, default='./data', help='root directory of dataset')
    parser.add_argument('--output-root', type=str, default='./output', help='root directory for output')

    # dataset
    parser.add_argument('--dataset', type=str, default='imagenet', choices=['imagenet100', 'imagenet'])

    # save and load
    parser.add_argument('--exp-name', type=str, default='kd', help='experiment name')
    parser.add_argument('--save', action='store_true', help='to save checkpoint')

    # loss function
    parser.add_argument('--pow', type=float, default=0.66, help='hyperparameter of adaptive weight')
    parser.add_argument('--lamda', type=float, default=10, help='weighting of classification and distillation')
    parser.add_argument('--const-lamda', action='store_true', help='use constant lamda value, default: adaptive weighting')

    parser.add_argument('--w-cls', type=float, default=1.0, help='weightage of new classification loss')

    # kd loss
    parser.add_argument('--kd', action='store_true', help='use kd loss')
    parser.add_argument('--w-kd', type=float, default=1.0, help='weightage of knowledge distillation loss')
    parser.add_argument('--T', type=float, default=2, help='temperature scaling for KD')

    # self-distillation
    parser.add_argument('--num-sd', type=int, default=0, help='number of self-distillation generations')
    parser.add_argument('--sd-factor', type=float, default=5.0, help='weighting between classification and distillation')
   
    # mixup
    parser.add_argument('--mixup', action='store_true', help='use mixup augmentation')
    parser.add_argument('--mixup-alpha', type=float, default=0.1, help='mixup alpha value') 
 
    # label smoothing
    parser.add_argument('--label-smoothing', action='store_true', help='use label smoothing')
    parser.add_argument('--smoothing-alpha', type=float, default=0.1, help='label smoothing alpha value') 
 
    args = parser.parse_args()
    return args

def save_checkpoint(args, epoch, model, old_model, exemplar_sets, optimizer, scheduler, classes, save_path, use_sd):
    print('==> Saving...')
    state = {
        'opt': args,
        'model': model.state_dict(),
        'old_model': old_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'exemplars': exemplar_sets,
        'epoch': epoch,
        'classes': classes,
    }
    if epoch % 5  ==0:
        torch.save(state, os.path.join(save_path, 'current.pth'))
    if epoch % args.save_freq == 0 or epoch ==1:
        if use_sd:
            torch.save(state, os.path.join(save_path, f'ckpt_epoch{classes}' + f'_sd_{epoch}.pth'))
        else:
            torch.save(state, os.path.join(save_path, f'ckpt_epoch{classes}' + f'_{epoch}.pth'))
    # help release GPU memory
    del state
    torch.cuda.empty_cache()

def train(model, old_model, epoch, optimizer, scheduler, lamda, train_loader, use_sd, checkPoint):

    step = 0
    best_acc = 0
    T = args.T

    model.cuda()
    old_model.cuda()

    criterion_ce = nn.CrossEntropyLoss(ignore_index=-1)
    criterion_ce_smooth = LabelSmoothingCrossEntropy() # for label smoothing
   

    if len(test_classes) // CLASS_NUM_IN_BATCH > 1:
        lr = args.lr_ft
    elif use_sd:
        lr = args.lr_sd
    else:
        epoch = args.epochs
        lr = args.lr
    
    if args.start_epoch ==1:
        print ('setting optimizer and scheduler.................') 
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=args.weight_decay)
        if use_sd:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8, 12], gamma=0.1)
        elif len(test_classes) // CLASS_NUM_IN_BATCH == 1:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 35], gamma=0.1)
   
    if len(test_classes) // CLASS_NUM_IN_BATCH > 1:
        exemplar_set = ExemplarDataset(exemplar_sets, transform=transform_ori)
        exemplar_loader = torch.utils.data.DataLoader(exemplar_set, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
        exemplar_loader_iter = iter(exemplar_loader)

        old_model.eval()
        num_old_classes = old_model.fc.out_features
        
    for epoch_index in range(args.start_epoch, epoch+1):

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
            if len(test_classes) // CLASS_NUM_IN_BATCH == 1 and args.mixup:
                inputs, targets_a, targets_b, lam = mixup_data(x, targets, args.mixup_alpha)
                inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))

                logits = model(inputs)
                outputs = logits[:,-CLASS_NUM_IN_BATCH:]
                cls_loss_new = mixup_criterion(criterion_ce, outputs, targets_a, targets_b, lam)

            elif len(test_classes) // CLASS_NUM_IN_BATCH == 1 and args.label_smoothing:
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
            else:
                factor = ((len(test_classes)/CLASS_NUM_IN_BATCH)**(args.pow))*args.lamda
             
            if len(test_classes) // CLASS_NUM_IN_BATCH == 1 and use_sd:
                if args.kd:
                    with torch.no_grad():
                        dist_target = old_model(x)
                    logits_dist = logits
                    T = args.T
                    dist_loss = nn.KLDivLoss()(F.log_softmax(logits_dist/T, dim=1), F.softmax(dist_target/T, dim=1)) * (T*T)  # best model
                    sum_dist_loss += dist_loss.item()
        
                    loss += factor*args.w_kd*dist_loss

            # Distillation : task-2 onwards
            if len(test_classes) // CLASS_NUM_IN_BATCH > 1:

                # KD loss using new class data 
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

                # Classification loss: old classes loss
                x_old, target_old = batch_ex
                x_old , target_old = x_old.cuda(), target_old.cuda()
                logits_old = model(x_old)
            
                old_classes = len(test_classes) - CLASS_NUM_IN_BATCH
                cls_loss_old = criterion_ce(logits_old, target_old)
            
                loss += cls_loss_old
                sum_cls_old_loss += cls_loss_old.item()

                # KD loss using exemplars
                if args.kd: 
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
                print('==>>> epoch: {}, batch index: {}, step: {}, train loss: {:.3f}, kd_loss: {:3f}, cls_new_loss: {:.3f}, cls_old_loss: {:.3f}'.
                      format(epoch_index, batch_idx + 1, step, sum_loss/(batch_idx+1), sum_dist_loss/(batch_idx+1), sum_cls_new_loss/(batch_idx+1),  sum_cls_old_loss/(batch_idx+1)))
            
        scheduler.step()

        if args.save:
            save_path = os.path.join(args.output_root, "checkpoints/imagenet/", args.exp_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_checkpoint(args, epoch_index, model, old_model, exemplar_sets, optimizer, scheduler, len(test_classes), save_path, use_sd)
        
def evaluate_net(model, transform, test_classes):
    model.eval()

    valdir = os.path.join(args.test_data_root, 'val')
    if args.dataset == 'imagenet100':
        test_set = ImageNet100(valdir, train=False, classes=test_classes, transform=transform)
    elif args.dataset == 'imagenet':
        test_set = ImageNet1K(valdir, train=False, classes=test_classes, transform=transform)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=4)
    
    total = 0.0
    correct = 0.0
    
    for j, (_, images, labels) in enumerate(test_loader):
        #print(len(images))
        with torch.no_grad():
            out = torch.softmax(model(images[0].cuda()), dim=1)
        _, preds = torch.max(out, dim=1, keepdim=False)
        labels = [y.item() for y in labels]
        np.asarray(labels)
        total += preds.size(0)
        correct += (preds.cpu().numpy() == labels).sum()

    test_acc = 100.0*correct/total
    # Test Accuracy
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
            feat = model.forward(x, feat=True).data.cpu().numpy()
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

            i = np.random.randint(0, features.shape[0])

            exemplar_dist.append(dist[i])
            exemplar_set.append(images[i])
            exemplar_features.append(features[i])
            features[i, :] = 0.0

        # random exemplar selection
        exemplar_dist = np.array(exemplar_dist)
        exemplar_set = np.array(exemplar_set)
        ind = exemplar_dist.argsort()
        exemplar_set = exemplar_set[ind]

        exemplar_sets.append(np.array(exemplar_set))
    print ('exemplar set shape: ', len(exemplar_set))

def combine_dataset_with_exemplars(dataset):
    print ('length of dataset pre: ', len(dataset))
    for y, P_y in enumerate(exemplar_sets):
        exemplar_images = P_y
        exemplar_labels = [y] * len(P_y)
        dataset.append(exemplar_images, exemplar_labels)
    print ('length of dataset post: ', len(dataset))
    return dataset

if __name__ == '__main__':
    args = parse_option()
    print (args)

    if not os.path.exists(os.path.join(args.output_root, "checkpoints/imagenet/")):
        os.makedirs(os.path.join(args.output_root, "checkpoints/imagenet/"))

    #  parameters
    if args.dataset == 'imagenet100':
        TOTAL_CLASS_NUM = 100
    elif args.dataset == 'imagenet':
        TOTAL_CLASS_NUM = 1000
        
    CLASS_NUM_IN_BATCH = args.start_classes
    TOTAL_CLASS_BATCH_NUM = TOTAL_CLASS_NUM // CLASS_NUM_IN_BATCH
    T = args.T

    exemplar_means = []    
    compute_means = True

    # default augmentation 
    transform_ori = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # test-time augmentation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    transform_train = TransformTwice(transform_ori, transform_ori)

    class_index = [i for i in range(0, TOTAL_CLASS_NUM)]
    net = resnet18_imagenet(num_classes=CLASS_NUM_IN_BATCH).cuda()
    
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)

    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print ('number of trainable parameters: ', params)

    old_net = copy.deepcopy(net)
    old_net.cuda()
   
    cls_list = [0] + [a for a in range(args.start_classes, TOTAL_CLASS_NUM, args.new_classes)] 

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
        print("old net output dim:", old_net.get_output_dim())

        cls = net.get_output_dim()

        traindir = os.path.join(args.train_data_root, 'train')

        if args.dataset == 'imagenet100':
            train_set = ImageNet100(traindir, train=True, classes=class_index[i:i+CLASS_NUM_IN_BATCH], transform=transform_ori)
        elif args.dataset == 'imagenet':
            train_set = ImageNet1K(traindir, train=True, classes=class_index[i:i+CLASS_NUM_IN_BATCH], transform=transform_ori)

        trainLoader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
   
        train_classes = class_index[i:i+CLASS_NUM_IN_BATCH]
        test_classes = class_index[:i+CLASS_NUM_IN_BATCH]

        print (train_classes)
        print (test_classes)    

        m = args.K // (i+CLASS_NUM_IN_BATCH)
       
        if i!=0:
            icarl_reduce_exemplar_sets(m) 
    
        for y in range(i, i+CLASS_NUM_IN_BATCH):
            print ("Constructing exemplar set for class-%d..." %(class_index[y]))
            images = train_set.get_image_class(y)
            icarl_construct_exemplar_set(net, images, m, transform_test)
            print ("Done")
        
        net.train()
        train(model=net, old_model=old_net, epoch=args.epochs, optimizer=optimizer, scheduler=scheduler, lamda=args.lamda, train_loader=trainLoader, use_sd=False, checkPoint=50)

        old_net = copy.deepcopy(net)
        old_net.cuda()

        # Do self-distillation
        if i == 0:
            for sd in range(args.num_sd):
                args.start_epoch = 1
                print ('self-dist it: ', sd)
                train(model=net, old_model=old_net, epoch=args.epochs_sd, optimizer=optimizer, scheduler=scheduler, lamda=args.lamda, train_loader=trainLoader, use_sd=True, checkPoint=50)
                old_net = copy.deepcopy(net)
                old_net.cuda()
        
        #---------------------- Evaluation ----------------------------------
        transform_val = TransformTwice(transform_test, transform_test)
        test_acc = evaluate_net(model=net, transform=transform_val, test_classes=test_classes)
        avg_acc.append(test_acc)
    
    print (avg_acc)
    print ('Avg accuracy: ', sum(avg_acc)/len(avg_acc))
