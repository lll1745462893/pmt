import lpips
import torch
import numpy as np
from PIL import Image

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from utils import *


def extract_feat(x, device, batch_size):
    percept = lpips.LPIPS(net='vgg').to(device)
    feats = []
    nb = int(np.ceil(x.size(0) / batch_size))
    for i in range(nb):
        bx = x[i*batch_size:(i+1)*batch_size]
        bx = bx.to(device)

        ins = percept.scaling_layer(bx)
        outs = percept.net.forward(ins)

        kk = 2
        feat = lpips.normalize_tensor(outs[kk])
        feat = percept.lins[kk](feat)
        feat = torch.nn.Flatten()(feat)
        feats.append(feat)

    feats = torch.cat(feats).detach().cpu().numpy()
    return feats


class Cluster:
    def __init__(self, method):
        self.method = method
    
    def train(self, features, n_clusters):
        if self.method == 'kmeans':
            self.model = KMeans(n_clusters=n_clusters, random_state=30).fit(features)
        elif self.method == 'gmm':
            self.model = GaussianMixture(n_components=n_clusters, random_state=30).fit(features)
        else:
            raise NotImplementedError
    
    def predict(self, features):
        output = self.model.predict(features)
        output = np.array(output)
        return output

# 这个函数的作用是根据输入样本x，利用预训练的代理模型对其进行分类，从而确定每个样本所属的分区索引。(只是找到图片的分区, 所以只作用在受害者类中)
class Partition:
    def __init__(self, args, device, surrogate_filepath):
        self.device = device

        # Load surrogate model
        self.net = torch.load(surrogate_filepath, map_location='cpu').to(self.device)
        self.net.eval()

        # Arguments
        self.num_classes = get_config(args.dataset)['num_classes']
        self.preprocess, _ = get_norm(args.dataset)
        self.batch_size = args.batch_size
        self.num_par = args.num_par

    def get_partition_index(self, x):
        #nb表示分成几个batch进行训练，防止一次放入
        nb = int(np.ceil(len(x) / self.batch_size))
        y = []
        #将图像按照批次batch取出
        for i in range(nb):
            bx = x[i*self.batch_size:(i+1)*self.batch_size, ...]
            bx = bx.to(self.device)
            #送入surrogate模型，经过self.preprocess归一化
            #但是这里的输出有控制？： logits（logits.shape = [batch_size, num_classes + num_partitions]）
            #然后取的是后面的几列 比如原本是10类 然后num_classes + num_partitions，那么取最后的num_partitions列
            output = self.net(self.preprocess(bx))[:, self.num_classes:]
            #dim表示每一行都拿一个最大值 【1】表示取最大值的索引
            pred = output.max(dim=1)[1]
            y.append(pred.cpu().numpy())
            #这个函数最终返回一个数组[0, 2, 1, 0, 3, 2, 2, 1] 每个值表示该样本预测属于哪个分区
        y = np.concatenate(y, axis=0)

        # Ensure the number of partitions matches the expected num_par
        assert output.shape[1] == self.num_par, (
            f"Mismatch: output columns ({output.shape[1]}) != num_par ({self.num_par})")

        # Clamp partition indices to ensure they are within the valid range
        pred = torch.clamp(pred, 0, self.num_par - 1)

        return y
