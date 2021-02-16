#remove ReLU in the last layer, and use cosine layer to replace nn.Linear
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from models.modified_linear import *
from torch.nn import functional as F

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, last=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.last = last

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if not self.last: #remove ReLU in the last layer
            out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        self.inplanes = 16
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2, last_phase=True)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc1 = nn.Linear(64 * block.expansion, 64)
        self.fc2 = nn.Linear(64 * block.expansion, 64)
        self.relu_ = nn.ReLU(inplace=True)
        self.fc = CosineLinear(64 * block.expansion, num_classes)
        self.l2norm = Normalize(2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, last_phase=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        if last_phase:
            for i in range(1, blocks-1):
                layers.append(block(self.inplanes, planes))
            layers.append(block(self.inplanes, planes, last=True))
        else: 
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def get_output_dim(self):
        return self.fc.out_features

    def change_output_dim(self, new_dim, second_iter=False):

        if second_iter:
            in_features = self.fc.in_features
            out_features1 = self.fc.fc1.out_features
            out_features2 = self.fc.fc2.out_features
            print("in_features:", in_features, "out_features1:", \
                out_features1, "out_features2:", out_features2)
            new_fc = SplitCosineLinear(in_features, out_features1+out_features2, out_features2)
            new_fc.fc1.weight.data[:out_features1] = self.fc.fc1.weight.data
            new_fc.fc1.weight.data[out_features1:] = self.fc.fc2.weight.data
            new_fc.sigma.data = self.fc.sigma.data
            self.fc = new_fc 
            new_out_features = new_dim
            self.n_classes = new_out_features
            
        else:
            in_features = self.fc.in_features
            out_features = self.fc.out_features
    
            print("in_features:", in_features, "out_features:", out_features)
            new_out_features = new_dim
            num_new_classes = new_dim-out_features
            new_fc = SplitCosineLinear(in_features, out_features, num_new_classes)

            new_fc.fc1.weight.data = self.fc.weight.data
            new_fc.sigma.data = self.fc.sigma.data
            self.fc = new_fc
            self.n_classes = new_out_features
     

    def freeze_weight_conv(self):
        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.bn1.parameters():
            param.requires_grad = False
        for param in self.layer1.parameters():
            param.requires_grad = False
        for param in self.layer2.parameters():
            param.requires_grad = False
        for param in self.layer3.parameters():
            param.requires_grad = False

    def freeze_weight(self):
        for param in self.parameters():
            param.requires_grad = False

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        

    def forward(self, x, feat=False, rd=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
    
        if rd:
            return F.normalize(x, p=2,dim=1)
            
        if feat:
            x = self.fc1(x)
            x = self.relu_(x)
            x = self.fc2(x)
            return F.normalize(x, p=2,dim=1)
        else:
            x = self.fc(x)
            return x


def resnet32_cifar(num_classes, pretrained=False, **kwargs):
    n = 5
    model = ResNet(BasicBlock, [n, n, n], num_classes, **kwargs)
    return model
