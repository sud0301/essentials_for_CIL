import os
from torch.utils.data import Dataset
from PIL import Image
import random
from matplotlib import image
import cv2
import numpy as np


class ExemplarDataset(Dataset):

    def __init__(self, data, transform=None):
        labels = []
        for y, P_y in enumerate(data):
            label = [y] * len(P_y)
            labels.extend(label)
        self.data = np.concatenate(data, axis=0)
        self.transform = transform
        self.labels = np.array(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        sample = self.data[idx]
        label = self.labels[idx]
        #sample = transforms.ToPILImage(sample)

        if self.transform:
            sample = self.transform(sample)
        return sample, label


class Exemplar1K(Dataset):
    def __init__(self, data_root, classes, num_samples, transform):
        self.transform = transform

        self.sample_filepaths = []

        self.train = train
        self.train_sample_cls = []
        self.test_sample_cls = []

        self.train_data = []
        self.test_data = []

        f = open('./data/class_folder_list.txt', 'r')
        lines=f.readlines()
        dir_list=[]
        for x in lines:
            dir_list.append(x.split(' ')[0])
         
        np.random.seed(1993)
        cls_list = [i for i in range(1000)]
        np.random.shuffle(cls_list)
        dir_list = [dir_list[i] for i in cls_list]

        for cls_idx, cls in enumerate(dir_list):

            cls_folder = os.path.join(data_root, cls)

            if cls_idx in classes:
                sample_idx = 0
                for sample in os.listdir(cls_folder):
                    sample_filepath = os.path.join(cls_folder, sample)
                    self.sample_filepaths.append(sample_filepath)
                    if train:
                        self.train_sample_cls.append(cls_idx)
                    else:
                        self.test_sample_cls.append(cls_idx)
        
    def __len__(self):
        if self.train:
            return len(self.train_sample_cls)
        else:
            return len(self.test_sample_cls)

    def __getitem__(self, idx):
    
        if self.train:
            img = cv2.imread(self.sample_filepaths[idx])
            img = self.transform(img)
            label = self.train_sample_cls[idx]
            return img, img, label
        else:
            img = cv2.imread(self.sample_filepaths[idx])
            img = self.transform(img)
            label = self.test_sample_cls[idx]
            return img, img, label
            

    def get_image_class(self, label):
        list_label = []
        list_label = [np.array(cv2.imread(self.sample_filepaths[idx])) for idx, k in enumerate(self.train_sample_cls) if k==label]
        return np.array(list_label)

    def append(self, images, labels):
        """Append dataset with images and labels
        Args:
            images: Tensor of shape (N, C, H, W)
            labels: list of labels
        """

        self.train_data = np.concatenate((self.train_data, images), axis=0)
        self.train_sample_cls = self.train_sample_cls + labels


class ImageNet1K(Dataset):
    def __init__(self, data_root, train, classes, transform):
        self.transform = transform

        self.sample_filepaths = []

        self.train = train
        self.train_sample_cls = []
        self.test_sample_cls = []

        self.train_data = []
        self.test_data = []

        f = open('./data/class_folder_list.txt', 'r')
        lines=f.readlines()
        dir_list=[]
        for x in lines:
            dir_list.append(x.split(' ')[0])
         
        np.random.seed(1993)
        cls_list = [i for i in range(1000)]
        np.random.shuffle(cls_list)
        dir_list = [dir_list[i] for i in cls_list]

        for cls_idx, cls in enumerate(dir_list):

            cls_folder = os.path.join(data_root, cls)

            if cls_idx in classes:
                for sample in os.listdir(cls_folder):
                    sample_filepath = os.path.join(cls_folder, sample)
                    self.sample_filepaths.append(sample_filepath)
                    if train:
                        self.train_sample_cls.append(cls_idx)
                    else:
                        self.test_sample_cls.append(cls_idx)
        
    def __len__(self):
        if self.train:
            return len(self.train_sample_cls)
        else:
            return len(self.test_sample_cls)

    def __getitem__(self, idx):
    
        if self.train:
            img = cv2.imread(self.sample_filepaths[idx])
            img = self.transform(img)
            label = self.train_sample_cls[idx]
            return img, img, label
        else:
            img = cv2.imread(self.sample_filepaths[idx])
            img = self.transform(img)
            label = self.test_sample_cls[idx]
            return img, img, label
            

    def get_image_class(self, label):
        list_label = []
        list_label = [np.array(cv2.imread(self.sample_filepaths[idx])) for idx, k in enumerate(self.train_sample_cls) if k==label]
        return np.array(list_label)

    def append(self, images, labels):
        """Append dataset with images and labels
        Args:
            images: Tensor of shape (N, C, H, W)
            labels: list of labels
        """

        self.train_data = np.concatenate((self.train_data, images), axis=0)
        self.train_sample_cls = self.train_sample_cls + labels

class ImageNet100(Dataset):
    def __init__(self, data_root, train, classes, transform):
        self.transform = transform

        self.sample_filepaths = []

        self.train = train
        self.train_sample_cls = []
        self.test_sample_cls = []

        self.train_data = []
        self.test_data = []

        # ImageNet-100
        f=open('./data/imagenet100_s1993.txt',"r")
        lines=f.readlines()
        dir_list=[]
        for x in lines:
            dir_list.append(x.split(' ')[0])

        np.random.seed(1993)
        cls_list = [i for i in range(100)]
        np.random.shuffle(cls_list)
        dir_list = [dir_list[i] for i in cls_list]

        for cls_idx, cls in enumerate(dir_list):
            if cls_idx ==100:       # imagenet-100
                break

            cls_folder = os.path.join(data_root, cls)

            if cls_idx in classes:
                for sample in os.listdir(cls_folder):
                    sample_filepath = os.path.join(cls_folder, sample)
                    self.sample_filepaths.append(sample_filepath)
                    if train:
                        self.train_sample_cls.append(cls_idx)
                    else:
                        self.test_sample_cls.append(cls_idx)

    def __len__(self):
        if self.train:
            return len(self.train_sample_cls)
        else:
            return len(self.test_sample_cls)

    def __getitem__(self, idx):
        if self.train:
            img = cv2.imread(self.sample_filepaths[idx])
            img = self.transform(img)
            label = self.train_sample_cls[idx]
            return img, img, label
        else:
            img = cv2.imread(self.sample_filepaths[idx])
            img = self.transform(img)
            label = self.test_sample_cls[idx]
            return img, img, label
    
    def get_image_class(self, label):
        list_label = []
        list_label = [np.array(cv2.imread(self.sample_filepaths[idx])) for idx, k in enumerate(self.train_sample_cls) if k==label]
        return np.array(list_label)

    def append(self, images, labels):
        """Append dataset with images and labels
        Args:
            images: Tensor of shape (N, C, H, W)
            labels: list of labels
        """

        self.train_data = np.concatenate((self.train_data, images), axis=0)
        self.train_sample_cls = self.train_sample_cls + labels
