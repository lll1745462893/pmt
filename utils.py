import os
import random
import numpy as np

import torch
import torch.nn as nn
from torchvision import datasets, transforms

from models import *


# Set random seed
def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# Dataset configurations (mean, std, size, num_classes)
_dataset_name = ['cifar10', 'cifar100', 'imagenet100', 'imagenet10', 'imagenette2', 'mnist', 'fashion_mnist', 'svhn', 'gtsrb', 'celeba']

_mean = {
    'cifar10':  [0.4914, 0.4822, 0.4465],
    'cifar100': [0.5071, 0.4867, 0.4408],  # CIFAR-100均值
    'imagenet100': [0.485, 0.456, 0.406],  # ImageNet标准均值
    'imagenet10': [0.485, 0.456, 0.406],   # ImageNet标准均值
    'imagenette2': [0.485, 0.456, 0.406],  # Imagenette 使用 ImageNet 均值
    'mnist': [0.1307],  # MNIST均值（单通道）
    'fashion_mnist': [0.2860],  # Fashion-MNIST均值（单通道）
    'svhn': [0.4377, 0.4438, 0.4728],  # SVHN均值
    'gtsrb': [0.3403, 0.3121, 0.3214],  # GTSRB均值
    'celeba': [0.5, 0.5, 0.5]  # CelebA均值（标准化）
}

_std = {
    'cifar10':  [0.2023, 0.1994, 0.2010],
    'cifar100': [0.2675, 0.2565, 0.2761],  # CIFAR-100标准差
    'imagenet100': [0.229, 0.224, 0.225],  # ImageNet标准标准差
    'imagenet10': [0.229, 0.224, 0.225],   # ImageNet标准标准差
    'imagenette2': [0.229, 0.224, 0.225],  # Imagenette 使用 ImageNet 标准差
    'mnist': [0.3081],  # MNIST标准差（单通道）
    'fashion_mnist': [0.3530],  # Fashion-MNIST标准差（单通道）
    'svhn': [0.1980, 0.2010, 0.1970],  # SVHN标准差
    'gtsrb': [0.2724, 0.2608, 0.2669],  # GTSRB标准差
    'celeba': [0.5, 0.5, 0.5]  # CelebA标准差（标准化）
}

_size = {
    'cifar10':  (32, 32),
    'cifar100': (32, 32),  # CIFAR-100尺寸
    'imagenet100': (224, 224),  # ImageNet标准尺寸
    'imagenet10': (224, 224),   # ImageNet标准尺寸
    'imagenette2': (224, 224),  # Imagenette 通常 224
    'mnist': (28, 28),  # MNIST尺寸
    'fashion_mnist': (28, 28),  # Fashion-MNIST尺寸
    'svhn': (32, 32),  # SVHN尺寸
    'gtsrb': (32, 32),  # GTSRB尺寸
    'celeba': (64, 64)  # CelebA尺寸（标准64x64）
}

_num = {
    'cifar10':  10,
    'cifar100': 100,  # CIFAR-100有100个类别
    'imagenet100': 100,  # ImageNet-100有100个类别
    'imagenet10': 10,    # ImageNet-10有10个类别
    'imagenette2': 10,   # Imagenette 10 类
    'mnist': 10,  # MNIST有10个类别
    'fashion_mnist': 10,  # Fashion-MNIST有10个类别
    'svhn': 10,  # SVHN有10个类别
    'gtsrb': 43,  # GTSRB有43个类别
    'celeba': 2  # CelebA二分类任务（例如有/无某个属性）
}


def get_config(dataset):
    assert dataset in _dataset_name, _dataset_name
    config = {}
    config['mean'] = _mean[dataset]
    config['std']  = _std[dataset]
    config['size'] = _size[dataset]
    config['num_classes'] = _num[dataset]
    return config


def get_norm(dataset):
    assert dataset in _dataset_name, _dataset_name
    mean = torch.FloatTensor(_mean[dataset])
    std  = torch.FloatTensor(_std[dataset])
    normalize   = transforms.Normalize(mean, std)
    unnormalize = transforms.Normalize(- mean / std, 1 / std)
    return normalize, unnormalize


def get_transform(dataset, augment=False, tensor=False):
    transforms_list = []
    if augment:
        transforms_list.append(transforms.Resize(_size[dataset]))
        transforms_list.append(transforms.RandomCrop(_size[dataset], padding=4))

        # Horizontal Flip
        transforms_list.append(transforms.RandomHorizontalFlip())
    else:
        transforms_list.append(transforms.Resize(_size[dataset]))

    # To Tensor
    if not tensor:
        transforms_list.append(transforms.ToTensor())

    transform = transforms.Compose(transforms_list)
    return transform


def get_augment(dataset):
    transforms_list = []
    transforms_list.append(transforms.RandomCrop(_size[dataset], padding=4))
    transforms_list.append(transforms.RandomHorizontalFlip())
    transform = transforms.Compose(transforms_list)
    return transform


# Get dataset
def get_dataset(dataset, datadir='data', train=True, augment=True):
    transform = get_transform(dataset, augment=train & augment)
    
    if dataset == 'cifar10':
        dataset = datasets.CIFAR10(datadir, train, download=True, transform=transform)
    elif dataset == 'cifar100':
        dataset = datasets.CIFAR100(datadir, train, download=True, transform=transform)
    elif dataset == 'imagenet100':
        # ImageNet-100数据路径
        # 首先检查服务器上是否已有数据
        server_imagenet_path = '/home/star/sda1/data/1744ecab-59b5-4839-9124-a5d97b52e660/datasets/Imagenet100'
        local_imagenet_path = os.path.join(datadir, 'imagenet100')
        
        # 优先使用服务器路径，如果不存在则使用本地路径
        if os.path.exists(server_imagenet_path):
            data_path = os.path.join(server_imagenet_path, 'train' if train else 'val')
            print(f"使用服务器ImageNet-100数据: {data_path}")
        elif os.path.exists(local_imagenet_path):
            data_path = os.path.join(local_imagenet_path, 'train' if train else 'val')
            print(f"使用本地ImageNet-100数据: {data_path}")
        else:
            raise FileNotFoundError(
                f"ImageNet-100数据集未找到！\n"
                f"请检查以下路径之一:\n"
                f"  1. 服务器路径: {server_imagenet_path}\n"
                f"  2. 本地路径: {local_imagenet_path}"
            )
        
        dataset = datasets.ImageFolder(data_path, transform=transform)
    elif dataset == 'imagenet10':
        # ImageNet-10数据路径
        # 首先检查服务器上是否已有数据
        server_imagenet_path = '/home/star/sda1/data/1744ecab-59b5-4839-9124-a5d97b52e660/datasets/Imagenet10'
        local_imagenet_path = os.path.join(datadir, 'Imagenet10')
        
        # 优先使用服务器路径，如果不存在则使用本地路径
        if os.path.exists(server_imagenet_path):
            data_path = os.path.join(server_imagenet_path, 'train' if train else 'val')
            print(f"使用服务器ImageNet-10数据: {data_path}")
        elif os.path.exists(local_imagenet_path):
            data_path = os.path.join(local_imagenet_path, 'train' if train else 'val')
            print(f"使用本地ImageNet-10数据: {data_path}")
        else:
            raise FileNotFoundError(
                f"ImageNet-10数据集未找到！\n"
                f"请检查以下路径之一:\n"
                f"  1. 服务器路径: {server_imagenet_path}\n"
                f"  2. 本地路径: {local_imagenet_path}"
            )
        
        dataset = datasets.ImageFolder(data_path, transform=transform)
    elif dataset == 'imagenette2':
        # Imagenette2 数据路径（默认解压目录）
        data_path = os.path.join(datadir, 'imagenette2')
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Imagenette2 数据集未找到，请将数据放置在: {data_path}，应包含 train/ 与 val/ 子目录")
        split_dir = 'train' if train else 'val'
        dataset = datasets.ImageFolder(os.path.join(data_path, split_dir), transform=transform)
    elif dataset == 'mnist':
        dataset = datasets.MNIST(datadir, train, download=True, transform=transform)
    elif dataset == 'fashion_mnist':
        dataset = datasets.FashionMNIST(datadir, train, download=True, transform=transform)
    elif dataset == 'svhn':
        split = 'train' if train else 'test'
        dataset = datasets.SVHN(datadir, split=split, download=True, transform=transform)
    elif dataset == 'gtsrb':
        # GTSRB数据集需要手动下载并放置在指定目录
        # 数据应该存储在 datadir/GTSRB/train 和 datadir/GTSRB/test 目录下
        data_path = os.path.join(datadir, 'GTSRB', 'train' if train else 'test')
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"GTSRB数据集未找到，请将数据放置在: {data_path}")
        dataset = datasets.ImageFolder(data_path, transform=transform)
    elif dataset == 'celeba':
        # CelebA数据集
        # 默认使用'Attractive'属性作为分类任务
        split = 'train' if train else 'test'
        try:
            dataset = datasets.CelebA(datadir, split=split, target_type='attr', 
                                     download=True, transform=transform)
            # 包装dataset以返回单个属性（例如'Attractive'）
            class CelebAWrapper(torch.utils.data.Dataset):
                def __init__(self, celeba_dataset, attr_idx=2):  # attr_idx=2对应'Attractive'
                    self.dataset = celeba_dataset
                    self.attr_idx = attr_idx
                
                def __len__(self):
                    return len(self.dataset)
                
                def __getitem__(self, idx):
                    img, attrs = self.dataset[idx]
                    # 返回单个属性作为标签
                    label = attrs[self.attr_idx].item()
                    return img, label
            
            dataset = CelebAWrapper(dataset)
        except:
            # 如果标准加载失败，尝试从ImageFolder加载
            data_path = os.path.join(datadir, 'celeba', 'train' if train else 'test')
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"CelebA数据集未找到，请将数据放置在: {data_path}")
        dataset = datasets.ImageFolder(data_path, transform=transform)
    else:
        raise NotImplementedError(f"Dataset {dataset} is not supported")

    return dataset


# Get model
def get_model(dataset, network):
    num_classes = _num[dataset]
    
    # 根据数据集确定输入通道数
    if dataset in ['mnist', 'fashion_mnist']:
        input_channels = 1
    else:
        input_channels = 3

    if network == 'resnet18':
        # 由于ResNet使用了adaptive_avg_pool2d，无论输入大小如何，特征维度总是512
        # 因此feat_scale应该始终为1
        feat_scale = 1
        model = resnet18(num_classes=num_classes, feat_scale=feat_scale)
    elif network == 'resnet34':
        # 由于ResNet使用了adaptive_avg_pool2d，无论输入大小如何，特征维度总是512
        # 因此feat_scale应该始终为1
        feat_scale = 1
        model = resnet34(num_classes=num_classes, feat_scale=feat_scale)
    elif network == 'resnet50':
        # 由于ResNet使用了adaptive_avg_pool2d，无论输入大小如何，特征维度总是2048（Bottleneck expansion=4）
        # 因此feat_scale应该始终为1
        feat_scale = 1
        model = resnet50(num_classes=num_classes, feat_scale=feat_scale)
    elif network == 'vgg11':
        model = vgg11(num_classes=num_classes)
    elif network == 'vgg13':
        model = vgg13(num_classes=num_classes)
    elif network == 'vgg16':
        model = vgg16(num_classes=num_classes)
    elif network == 'vgg19':
        model = vgg19(num_classes=num_classes)
    elif network == 'densenet121':
        if dataset in ['cifar10', 'cifar100', 'svhn', 'gtsrb']:
            feat_scale = 1
        elif dataset in ['imagenet100', 'imagenet10']:
            feat_scale = 49
        elif dataset in ['celeba']:
            feat_scale = 4
        else:
            feat_scale = 1
        model = densenet121(num_classes=num_classes, feat_scale=feat_scale)
    elif network == 'densenet169':
        if dataset in ['cifar10', 'cifar100', 'svhn', 'gtsrb']:
            feat_scale = 1
        elif dataset in ['imagenet100', 'imagenet10']:
            feat_scale = 49
        elif dataset in ['celeba']:
            feat_scale = 4
        else:
            feat_scale = 1
        model = densenet169(num_classes=num_classes, feat_scale=feat_scale)
    elif network == 'alexnet':
        # AlexNet使用自适应池化，不需要feat_scale
        model = alexnet(num_classes=num_classes)
    elif network == 'mobilenet':
        if dataset in ['cifar10', 'cifar100', 'svhn', 'gtsrb']:
            feat_scale = 1
        elif dataset in ['imagenet100', 'imagenet10']:
            feat_scale = 49
        elif dataset in ['celeba']:
            feat_scale = 4
        else:
            feat_scale = 1
        model = mobilenet(num_classes=num_classes, feat_scale=feat_scale)
    elif network == 'googlenet':
        if dataset in ['cifar10', 'cifar100', 'svhn', 'gtsrb']:
            feat_scale = 1
        elif dataset in ['imagenet100', 'imagenet10']:
            feat_scale = 49
        elif dataset in ['celeba']:
            feat_scale = 4
        else:
            feat_scale = 1
        model = googlenet(num_classes=num_classes, feat_scale=feat_scale)
    elif network == 'senet18':
        model = senet18(num_classes=num_classes)
    # simple_cnn 暂时未实现
    # elif network == 'simple_cnn':
    #     model = simple_cnn(num_classes=num_classes, input_channels=input_channels)
    else:
        raise NotImplementedError(f"Network {network} is not supported")

    return model
