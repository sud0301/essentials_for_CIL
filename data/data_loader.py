from torchvision.datasets import CIFAR10
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import random
import torchvision.transforms as transforms
import os


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict


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


class cifar10(CIFAR10):
    def __init__(self, root,
                 classes=range(10),
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=True):
        super(cifar10, self).__init__(root,
                                       train=train,
                                       transform=transform,
                                       target_transform=target_transform,
                                       download=download)


        np.random.seed(1993)
        cls_list = [i for i in range(100)]
        np.random.shuffle(cls_list)
        self.class_mapping = np.array(cls_list, copy=True)
        
        # Select subset of classes
        if self.train:
            train_data = []
            train_labels = []

            for i in range(len(self.data)):
                if self.targets[i] in classes:
                    train_data.append(self.data[i])
                    #train_labels.append(self.targets[i])
                    train_labels.append(cls_list.index(self.targets[i]))

            self.train_data = np.array(train_data)
            self.train_labels = train_labels

        else:
            test_data = []
            test_labels = []

            for i in range(len(self.data)):
                if self.targets[i] in classes:
                    test_data.append(self.data[i])
                    #test_labels.append(self.targets[i])
                    test_labels.append(cls_list.index(self.targets[i]))

            self.test_data = np.array(test_data)
            self.test_labels = test_labels

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            if self.train:
                #img_ori, img, img_aug = self.transform(img)
                img, img_aug = self.transform(img)
            else:
                img, _ = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        if self.train:
            #return img_ori, img, img_aug, target
            return img, img_aug, target
        else:
            return img, img, target
     
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
    
    def get_image_class(self, label):
        return self.train_data[np.array(self.train_labels) == label]

    def get_original_label(self, rnd_label):
        return self.class_mapping[rnd_label]

    def append(self, images, labels):
        """Append dataset with images and labels
        Args:
            images: Tensor of shape (N, C, H, W)
            labels: list of labels
        """

        self.train_data = np.concatenate((self.train_data, images), axis=0)
        self.train_labels = self.train_labels + labels


class cifar100(cifar10):
    base_folder = 'cifar-100-python'
    url = "http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]
    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }

    map_fine_to_coarse = {49: 10, 33: 10, 72: 0, 51: 4, 71: 10, 92: 2, 15: 11, 14: 7, 23: 10, 0: 4, 75: 12, 81: 19,
                          69: 19, 40: 5, 43: 8, 97: 8, 70: 2, 53: 4, 29: 15, 21: 11, 16: 3, 39: 5, 8: 18, 20: 6, 61: 3,
                          41: 19, 93: 15, 56: 17, 73: 1, 58: 18, 11: 14, 25: 6, 37: 9, 63: 12, 24: 7, 22: 5, 17: 9,
                          4: 0, 6: 7, 9: 3, 57: 4, 2: 14, 32: 1, 52: 17, 42: 8, 77: 13, 27: 15, 65: 16, 7: 7, 35: 14,
                          82: 2, 66: 12, 90: 18, 67: 1, 91: 1, 10: 3, 78: 15, 54: 2, 89: 19, 18: 7, 13: 18, 50: 16,
                          26: 13, 83: 4, 47: 17, 95: 0, 76: 9, 59: 17, 85: 19, 19: 11, 46: 14, 1: 1, 74: 16, 60: 10,
                          64: 12, 45: 13, 36: 16, 87: 5, 30: 0, 99: 13, 80: 16, 28: 3, 98: 14, 12: 9, 94: 6, 68: 9,
                          44: 15, 31: 11, 79: 13, 34: 12, 55: 0, 62: 2, 96: 17, 84: 6, 38: 11, 86: 5, 5: 6, 48: 18,
                          3: 8, 88: 8}

    map_coarse_to_fine = {10: [49, 33, 71, 23, 60], 0: [72, 4, 95, 30, 55], 4: [51, 0, 53, 57, 83],
                          2: [92, 70, 82, 54, 62], 11: [15, 21, 19, 31, 38], 7: [14, 24, 6, 7, 18],
                          12: [75, 63, 66, 64, 34], 19: [81, 69, 41, 89, 85], 5: [40, 39, 22, 87, 86],
                          8: [43, 97, 42, 3, 88], 15: [29, 93, 27, 78, 44], 3: [16, 61, 9, 10, 28],
                          18: [8, 58, 90, 13, 48], 6: [20, 25, 94, 84, 5], 17: [56, 52, 47, 59, 96],
                          1: [73, 32, 67, 91, 1], 14: [11, 2, 35, 46, 98], 9: [37, 17, 76, 12, 68],
                          13: [77, 26, 45, 99, 79], 16: [65, 50, 74, 36, 80]}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coarse_label_names = unpickle(os.path.join(self.root, self.base_folder,
                                                        self.meta['filename']))['coarse_label_names']
        self.fine_label_names = unpickle(os.path.join(self.root, self.base_folder,
                                                      self.meta['filename']))['fine_label_names']

    def get_label_info(self, fine_label_original):
        return {"name": self.fine_label_names[fine_label_original],
                "coarse": self.map_fine_to_coarse[fine_label_original],
                "coarse_name": self.coarse_label_names[self.map_fine_to_coarse[fine_label_original]]}


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import random
    import numpy as np
    import time
    data = cifar100("/misc/lmbraid18/galessos/datasets/cifar100/", download=False, train=False)
    random.seed(time.time())
    idx = random.randint(0, len(data) - 1)
    img, label = data[idx]
    plt.imshow(np.array(img))
    orig_label = data.get_original_label(label)
    label_info = data.get_label_info(orig_label)
    plt.title("sample {}, class: {} ({}), {} ({})".format(idx, orig_label, label_info["name"], label_info["coarse"],
                                                          label_info["coarse_name"]))
    plt.show()

