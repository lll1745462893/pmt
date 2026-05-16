import torch
from torch.utils.data import Dataset
import numpy as np

from trigger import stamp_trigger


# Construct a customized dataset
class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        assert len(images) == len(labels)
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.images[index]
        lbl = self.labels[index]
        if self.transform:
            img = self.transform(img)
        return img, lbl


# Extract the samples from the victim class
# Construct a dataset for other samples
def split_victim_other(dataset, victim_class, transform=None):
    victim_images = []
    other_images, other_labels = [], []
    for i in range(len(dataset)):
        x, y = dataset[i]
        if y == victim_class:
            victim_images.append(x)
        else:
            other_images.append(x)
            other_labels.append(y)

    victim_images = torch.stack(victim_images)
    other_dataset = CustomDataset(other_images, other_labels, transform)

    return victim_images, other_dataset


# Poison testset
class PoisonTestDataset(Dataset):
    def __init__(self, x, indexes, target_labels, transform=None, poison_alpha=0.2,
                 use_frequency=False, freq_mask_type='low', freq_mask_ratio=0.15):
        """
        Args:
            x: 图像数据
            indexes: 触发器索引（分区索引）
            target_labels: 目标标签
            transform: 数据变换
            poison_alpha: 触发器混合强度
            use_frequency: 是否使用频域注入
            freq_mask_type: 频域掩码类型
            freq_mask_ratio: 频域掩码比例
        """
        self.x = []
        self.y = []
        self.poison_alpha = poison_alpha
        self.use_frequency = use_frequency
        self.freq_mask_type = freq_mask_type
        self.freq_mask_ratio = freq_mask_ratio
        
        # Stamp trigger for each image
        if use_frequency:
            from trigger import stamp_trigger_frequency
            for i in range(len(x)):
                img = stamp_trigger_frequency(
                    x[i], indexes[i], 
                    alpha=poison_alpha,
                    freq_mask_type=freq_mask_type,
                    mask_ratio=freq_mask_ratio
                )
                # 确保图像形状是 [C, H, W]，移除多余的维度
                if img.dim() == 4:
                    img = img.squeeze(0)
                self.x.append(img)
                # 支持单一目标类别或动态目标标签数组
                if isinstance(target_labels, (list, np.ndarray)):
                    self.y.append(target_labels[i])
                else:
                    self.y.append(target_labels)
        else:
            for i in range(len(x)):
                img = stamp_trigger(x[i], indexes[i], alpha=self.poison_alpha)
                # 确保图像形状是 [C, H, W]，移除多余的维度
                if img.dim() == 4:
                    img = img.squeeze(0)
                self.x.append(img)
                # 支持单一目标类别或动态目标标签数组
                if isinstance(target_labels, (list, np.ndarray)):
                    self.y.append(target_labels[i])
                else:
                    self.y.append(target_labels)

        self.x = torch.stack(self.x)
        self.y = torch.LongTensor(self.y)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]
        # 确保 x 的形状是 [C, H, W]，移除多余的维度
        if x.dim() == 4:
            x = x.squeeze(0)
        if self.transform:
            x = self.transform(x)
        return x, y


# Datasets containing all possible partitions and trigger combinations
class PartitionDataset(Dataset):
    def __init__(self, x, l, num_par):
        self.x = x
        self.l = l
        self.num_par = num_par

    def __getitem__(self, index):
        img, par = self.x[index], self.l[index]
        choice = []
        total = 2 ** self.num_par
        for i in range(1, total):
            choice.append(bin(i)[2:].zfill(self.num_par))

        images = []
        for code in choice:
            timg = img.clone()
            for j in range(len(code)):
                t = int(code[j])
                if t == 1:
                    timg = stamp_trigger(timg, j)
            images.append(timg)

        # Add codes
        images = torch.stack(images)
        return images, par

    def __len__(self):
        return len(self.x)
