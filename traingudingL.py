import os
import sys
import time
import copy
import json
import argparse
import numpy as np

# 修复threadpoolctl警告
os.environ['THREADPOOLCTL_DISABLE'] = '1'
os.environ['PYTHONWARNINGS'] = 'ignore'

# 额外的警告屏蔽
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning, module='threadpoolctl')
warnings.filterwarnings('ignore', category=RuntimeWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from utils import *
from dataset import split_victim_other, PoisonTestDataset, PartitionDataset
from partition import extract_feat, Cluster, Partition
from trigger import trigger_focus, stamp_trigger, stamp_trigger_frequency


# 定义分区→目标标签的映射（键为分区索引，值为目标标签）
# 这个映射会在运行时根据实际分区数动态调整
def get_par2target(num_par):
    """根据分区数动态生成分区映射"""
    return {i: i + 2 for i in range(num_par)}  # 修改为i+2，与原始硬编码映射保持一致

# 默认映射（向后兼容）
par2target = {
    0: 2,   # 分区0的样本，目标标签统一为2
    1: 3,   # 分区1的样本，目标标签统一为3
    2: 4,   # 分区2的样本，目标标签统一为4
    3: 5    # 分区3的样本，目标标签统一为5
}


def get_model_suffix(args):
    """
    根据训练配置生成模型文件名后缀，用于区分不同的实验配置
    
    Args:
        args: 命令行参数
    
    Returns:
        str: 模型后缀，例如 "freq_low_a0.5_r0.2" 或 "pixel_a0.2"
    
    示例:
        - 频域注入: "freq_low_a0.5_r0.2"
        - 像素空间: "pixel_a0.2"
    """
    suffix_parts = []
    
    if getattr(args, 'use_frequency', False):
        # 频域注入模式
        suffix_parts.append('freq')
        suffix_parts.append(getattr(args, 'freq_mask_type', 'low'))
        suffix_parts.append(f"a{getattr(args, 'freq_alpha', 0.5)}")
        suffix_parts.append(f"r{getattr(args, 'freq_mask_ratio', 0.15)}")
    else:
        # 像素空间模式
        suffix_parts.append('pixel')
        suffix_parts.append('a0.2')  # 默认像素alpha值
    
    # 添加其他重要参数
    suffix_parts.append(f"p{args.num_par}")  # 分区数
    if getattr(args, 'n_indi', 0) <= 0 and getattr(args, 'n_comb', 0) <= 0:
        suffix_parts.append('noneg')
    
    return '_'.join(suffix_parts)


def get_clean_model_paths(save_folder, args):
    preferred = f'{save_folder}/clean_{args.network}_{args.dataset}.pt'
    legacy = f'{save_folder}/clean_{args.dataset}.pt'
    return preferred, legacy


def get_clean_result_paths(save_folder, args):
    preferred = f'{save_folder}/clean_result_{args.network}_{args.dataset}.json'
    legacy = f'{save_folder}/clean_result_{args.dataset}.json'
    return preferred, legacy


def get_surrogate_model_paths(save_folder, args):
    surrogate_tag = f'{args.network}_v{args.victim}_p{args.num_par}'
    preferred = f'{save_folder}/surrogate_{surrogate_tag}_{args.dataset}.pt'
    legacy = f'{save_folder}/surrogate_{args.dataset}.pt'
    return preferred, legacy


def get_lotus_model_paths(save_folder, args, stage):
    model_suffix = get_model_suffix(args)
    preferred = f'{save_folder}/lotus_{stage}_{args.network}_{model_suffix}_{args.dataset}.pt'
    legacy = f'{save_folder}/lotus_{stage}_{model_suffix}_{args.dataset}.pt'
    return preferred, legacy


def get_attack_result_paths(save_folder, args):
    model_suffix = get_model_suffix(args)
    preferred = f'{save_folder}/result_{args.network}_{model_suffix}_{args.dataset}.json'
    legacy = f'{save_folder}/result_{args.dataset}.json'
    return preferred, legacy


def save_torch_artifact(obj, preferred_path, legacy_path=None):
    torch.save(obj, preferred_path)


def save_json_artifact(payload, preferred_path, legacy_path=None):
    with open(preferred_path, 'w') as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)


def resolve_artifact_path(preferred_path, legacy_path, logger, artifact_name):
    if os.path.exists(preferred_path):
        return preferred_path

    if legacy_path and os.path.exists(legacy_path):
        logger.warning(f"未找到新格式{artifact_name}: {preferred_path}")
        logger.info(f"使用旧格式{artifact_name}: {legacy_path}")
        return legacy_path

    if legacy_path:
        raise FileNotFoundError(f'{artifact_name}不存在: {preferred_path} 或 {legacy_path}')
    raise FileNotFoundError(f'{artifact_name}不存在: {preferred_path}')

#其实没有用这个代码
def generate_poisoned_samples_simple(x, p, victim, target_map, num_par):
    """
    简化的毒化样本生成：只生成正确配对的样本，减少负样本干扰
    """
    # 确保 p 是 torch 张量
    if isinstance(p, np.ndarray):
        p = torch.from_numpy(p)
    
    # 输入验证
    assert len(x) == len(p), "输入样本数量与分区索引数量不匹配"
    assert all(0 <= idx < num_par for idx in p), "存在无效的分区索引"
    
    # 只生成正确配对的毒化样本
    x_poisoned = []
    y_poisoned = []
    
    for i in range(x.shape[0]):
        # 添加对应的触发器
        poisoned_img = stamp_trigger(x[i], p[i])
        x_poisoned.append(poisoned_img)
        
        # 获取对应的目标标签
        target_label = target_map.get(p[i].item(), victim)
        y_poisoned.append(target_label)
    
    x_poisoned = torch.stack(x_poisoned, dim=0)
    y_poisoned = torch.tensor(y_poisoned).long()
    
    # 统计信息
    stats = {
        'total_samples': len(x_poisoned),
        'trojaned_samples': len(x_poisoned),
        'unique_labels': torch.unique(y_poisoned).tolist(),
        'partition_distribution': torch.bincount(p).tolist()
    }
    
    return x_poisoned, y_poisoned, stats

#没有用到
# AT注意力损失类
class AT(nn.Module):
    """注意力转移损失"""
    def __init__(self, p):
        super(AT, self).__init__()
        self.p = p

    def forward(self, fm_s, fm_t):
        loss = F.mse_loss(self.attention_map(fm_s), self.attention_map(fm_t))
        return loss

    def attention_map(self, fm, eps=1e-6):
        am = torch.pow(torch.abs(fm), self.p)
        am = torch.sum(am, dim=1, keepdim=True)
        norm = torch.norm(am, dim=(2,3), keepdim=True)
        am = torch.div(am, norm+eps)
        return am


# 特征级正则化器（增强隐蔽性，抗NC/ANP/Fine-prune防御）
class FeatureRegularizer:
    """
    让毒化样本的特征表示接近良性样本，增强后门隐蔽性
    - 对抗NC防御：特征分布相似，难以通过聚类检测
    - 对抗ANP防御：激活模式相似，难以发现异常神经元
    - 对抗Fine-prune：特征依赖分散，剪枝不易消除后门
    
    改进：
    1. 动态权重调整：根据训练阶段和性能指标调整特征正则化强度
    2. 自适应损失权重：不同特征匹配方式的权重会动态调整
    3. 特征记忆机制：使用历史特征信息来稳定训练
    """
    def __init__(self, feature_loss_weight=0.1, memory_size=10):
        self.base_weight = feature_loss_weight
        self.feature_loss_weight = feature_loss_weight
        self.memory_size = memory_size
        self.clean_features_buffer = []
        self.poison_features_buffer = []
        self.epoch = 0
        self.total_epochs = None
        self.current_asr = None
        self.best_asr = None
        
    def update_training_info(self, epoch, total_epochs, current_asr, best_asr):
        """更新训练信息，用于动态调整权重"""
        self.epoch = epoch
        self.total_epochs = total_epochs
        self.current_asr = current_asr
        self.best_asr = best_asr
        
        # 动态调整特征正则化权重
        self.adjust_feature_weight()
        
    def adjust_feature_weight(self):
        """根据训练阶段和性能动态调整特征正则化权重"""
        if self.total_epochs is None:
            return
            
        # 基于训练阶段的权重
        progress = self.epoch / self.total_epochs
        if progress < 0.3:  # 早期
            stage_weight = 0.5  # 较小权重，让模型自由学习
        elif progress < 0.7:  # 中期
            stage_weight = 1.0  # 标准权重，强化特征匹配
        else:  # 后期
            stage_weight = 0.8  # 适中权重，平衡稳定性
            
        # 基于ASR的权重调整
        if self.current_asr is not None and self.best_asr is not None:
            if self.current_asr < 0.9 * self.best_asr:  # ASR显著下降
                asr_weight = 0.7  # 减少正则化，恢复ASR
            elif self.current_asr >= 0.95:  # ASR很高
                asr_weight = 1.2  # 增强正则化，提高隐蔽性
            else:
                asr_weight = 1.0  # 保持当前权重
                
            # 更新最终权重
            self.feature_loss_weight = self.base_weight * stage_weight * asr_weight
        else:
            self.feature_loss_weight = self.base_weight * stage_weight
    
    def update_feature_buffer(self, clean_feat, poison_feat):
        """更新特征缓冲区，确保所有特征具有相同的维度"""
        # 计算每个特征的平均值（跨batch维度）
        clean_mean = clean_feat.mean(0, keepdim=True)   # [1, feature_dim]
        poison_mean = poison_feat.mean(0, keepdim=True) # [1, feature_dim]
        
        # 将当前特征的平均值添加到缓冲区
        self.clean_features_buffer.append(clean_mean.detach())
        self.poison_features_buffer.append(poison_mean.detach())
        
        # 保持缓冲区大小
        if len(self.clean_features_buffer) > self.memory_size:
            self.clean_features_buffer.pop(0)
            self.poison_features_buffer.pop(0)
            
    def get_buffer_features(self):
        """获取缓冲区中的平均特征"""
        if not self.clean_features_buffer:
            return None, None
            
        # 现在所有特征都是 [1, feature_dim]，可以安全地stack
        clean_features = torch.cat(self.clean_features_buffer, dim=0).mean(0)  # [feature_dim]
        poison_features = torch.cat(self.poison_features_buffer, dim=0).mean(0)  # [feature_dim]
        return clean_features, poison_features
        
    def compute_feature_similarity_loss(self, clean_feat, poison_feat):
        """
        计算特征相似性损失，使用动态权重和特征记忆
        """
        # 更新特征缓冲区
        self.update_feature_buffer(clean_feat, poison_feat)
        
        # 获取历史平均特征
        hist_clean, hist_poison = self.get_buffer_features()
        
        # 计算当前特征的损失
        mean_loss = F.mse_loss(poison_feat.mean(0), clean_feat.mean(0))
        std_loss = F.mse_loss(poison_feat.std(0), clean_feat.std(0))
        
        # 计算协方差损失
        clean_centered = clean_feat - clean_feat.mean(0, keepdim=True)
        poison_centered = poison_feat - poison_feat.mean(0, keepdim=True)
        clean_cov = (clean_centered.T @ clean_centered) / clean_feat.size(0)
        poison_cov = (poison_centered.T @ poison_centered) / poison_feat.size(0)
        cov_loss = F.mse_loss(torch.diag(poison_cov), torch.diag(clean_cov))
        
        # 如果有历史特征，添加历史一致性损失
        history_loss = 0.0
        if hist_clean is not None and hist_poison is not None:
            clean_history_loss = F.mse_loss(clean_feat.mean(0), hist_clean)
            poison_history_loss = F.mse_loss(poison_feat.mean(0), hist_poison)
            history_loss = 0.3 * (clean_history_loss + poison_history_loss)
        
        # 动态权重：根据训练阶段调整各损失的权重
        progress = self.epoch / self.total_epochs if self.total_epochs else 0.5
        mean_weight = 1.0
        std_weight = 0.5 if progress < 0.3 else 0.8  # 后期增加分布匹配的重要性
        cov_weight = 0.3 if progress < 0.5 else 0.5  # 后期增加相关性匹配的重要性
        
        # 组合所有损失
        total_loss = (
            mean_weight * mean_loss + 
            std_weight * std_loss + 
            cov_weight * cov_loss + 
            history_loss
        )
        
        return total_loss
    
    def get_weighted_loss(self, clean_feat, poison_feat):
        """获取加权后的特征正则化损失"""
        feat_loss = self.compute_feature_similarity_loss(clean_feat, poison_feat)
        return self.feature_loss_weight * feat_loss


# 黑盒对抗训练组件
class BlackBoxAdversarialTrainer:
    def __init__(self, model, args, device, logger):
        self.device = device
        self.args = args
        self.logger = logger
        self.model = model
        
        # 黑盒防御模拟器 - 不直接使用NAD实现
        self.defense_simulators = []
        self.defense_effectiveness_history = []
        
        # 自适应参数
        self.attack_strength = 1.0
        self.defense_resistance = 0.5
        
    def simulate_blackbox_defense(self, model, defense_loader, poison_loader, preprocess):
        """
        黑盒防御模拟 
        通过观察防御效果来调整攻击策略
        """
        model.eval()
        
        # 测试原始攻击成功率
        original_asr = self.evaluate_attack_success(model, poison_loader, preprocess)
        
        # 模拟防御过程 - 使用通用的防御策略
        defended_asr = self.apply_generic_defense(model, defense_loader, preprocess)
        
        # 计算防御效果，避免除零错误
        if original_asr > 0:
            defense_effectiveness = (original_asr - defended_asr) / original_asr
        else:
            defense_effectiveness = 0.0
        
        # 记录防御效果历史
        self.defense_effectiveness_history.append(defense_effectiveness)
        
        # 自适应调整攻击强度
        self.adapt_attack_strategy(defense_effectiveness)
        
        return original_asr, defended_asr, defense_effectiveness
    
    def apply_generic_defense(self, model, defense_loader, preprocess):
        """
        
        使用多种可能的防御机制
        """
        # 模拟特征扰动防御
        defended_model = self.apply_feature_perturbation(model)
        
        # 模拟注意力正则化
        defended_model = self.apply_attention_regularization(defended_model)
        
        # 评估防御后的攻击成功率
        defended_asr = self.evaluate_attack_success(defended_model, defense_loader, preprocess)
        
        return defended_asr
    
    def apply_feature_perturbation(self, model):
        """应用特征扰动防御"""
        # 创建模型副本
        defended_model = copy.deepcopy(model)
        
        # 对特征层添加噪声
        for name, module in defended_model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                # 添加小量噪声
                noise = torch.randn_like(module.weight) * 0.01
                module.weight.data += noise
        
        return defended_model
    
    def apply_attention_regularization(self, model):
        """应用注意力正则化防御"""
        # 这里可以实现通用的注意力正则化

        
        return model
    
    def adapt_attack_strategy(self, defense_effectiveness):
        """根据防御效果自适应调整攻击策略"""
        if defense_effectiveness > 0.5:  # 防御效果太强
            self.attack_strength *= 1.1  # 增强攻击
            self.defense_resistance *= 1.05  # 增强防御抵抗
        elif defense_effectiveness < 0.2:  # 防御效果太弱
            self.attack_strength *= 0.95  # 稍微减弱攻击
            self.defense_resistance *= 0.98  # 稍微减弱防御抵抗
    
    def evaluate_attack_success(self, model, poison_loader, preprocess):
        """评估攻击成功率"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in poison_loader:
                x, y = x.to(self.device), y.to(self.device)
                output = model(preprocess(x))
                pred = output.max(dim=1)[1]
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        return correct / total if total > 0 else 0.0
    
    def get_defense_dataset(self, train_set, defense_ratio=0.05):
        """获取防御用的少量干净数据"""
        # 随机选择指定比例的数据
        num_samples = int(len(train_set) * defense_ratio)
        indices = np.random.choice(len(train_set), num_samples, replace=False)
        
        defense_data = []
        for idx in indices:
            defense_data.append(train_set[idx])
        
        return defense_data
    
    def simulate_nad_defense(self, student_model, defense_data_loader, preprocess, epochs=3):
        """模拟NAD防御过程"""
        # 临时创建学生模型副本进行防御测试
        temp_student = copy.deepcopy(student_model)
        temp_student.train()
        
        # 简化的NAD防御（几个epoch）
        optimizer = torch.optim.SGD(temp_student.parameters(), lr=0.001, momentum=0.9)
        criterion_ce = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            for x_batch, y_batch in defense_data_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                
                # 注册hook获取特征
                activation = {}
                def get_activation(name):
                    def hook(model, input, output):
                        activation[name] = output
                    return hook
                
                # 根据网络类型选择合适的层
                if 'resnet' in self.args.network or 'senet' in self.args.network:
                    s_handle = temp_student.layer4.register_forward_hook(get_activation('student'))
                    t_handle = self.teacher.layer4.register_forward_hook(get_activation('teacher'))
                elif 'vgg' in self.args.network:
                    s_handle = temp_student.features[-1].register_forward_hook(get_activation('student'))
                    t_handle = self.teacher.features[-1].register_forward_hook(get_activation('teacher'))
                elif 'densenet' in self.args.network:
                    # DenseNet的最后一个dense block
                    s_handle = temp_student.dense4.register_forward_hook(get_activation('student'))
                    t_handle = self.teacher.dense4.register_forward_hook(get_activation('teacher'))
                elif 'alexnet' in self.args.network:
                    # AlexNet使用features的最后一层
                    s_handle = temp_student.features[-1].register_forward_hook(get_activation('student'))
                    t_handle = self.teacher.features[-1].register_forward_hook(get_activation('teacher'))
                elif 'mobilenet' in self.args.network:
                    # MobileNet使用layers的最后一层
                    s_handle = temp_student.layers[-1].register_forward_hook(get_activation('student'))
                    t_handle = self.teacher.layers[-1].register_forward_hook(get_activation('teacher'))
                elif 'googlenet' in self.args.network:
                    # GoogLeNet使用b5的最后一层
                    s_handle = temp_student.b5.register_forward_hook(get_activation('student'))
                    t_handle = self.teacher.b5.register_forward_hook(get_activation('teacher'))
                else:
                    # 默认使用最后一个卷积层
                    s_handle = list(temp_student.children())[-2].register_forward_hook(get_activation('student'))
                    t_handle = list(self.teacher.children())[-2].register_forward_hook(get_activation('teacher'))
                
                # 学生前向传播
                output_s = temp_student(preprocess(x_batch))
                feat_s = activation['student']
                
                # 教师前向传播
                with torch.no_grad():
                    _ = self.teacher(preprocess(x_batch))
                    feat_t = activation['teacher']
                
                # NAD损失
                ce_loss = criterion_ce(output_s, y_batch)
                at_loss = self.criterion_at(feat_s, feat_t)
                loss = ce_loss + self.nad_beta * at_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 清理hook
                s_handle.remove()
                t_handle.remove()
        
        return temp_student
    
    def evaluate_defense_resistance(self, original_model, defended_model, poison_test_loader, preprocess):
        """评估防御后的ASR下降程度"""
        def get_asr(model):
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for x_batch, y_batch in poison_test_loader:
                    x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                    outputs = model(preprocess(x_batch))
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == y_batch).sum().item()
                    total += y_batch.size(0)
            return correct / total if total > 0 else 0
        
        original_asr = get_asr(original_model)
        defended_asr = get_asr(defended_model)
        
        defense_effectiveness = (original_asr - defended_asr) / original_asr if original_asr > 0 else 0
        return original_asr, defended_asr, defense_effectiveness
    
    def compute_adversarial_loss(self, student_model, defense_data_loader, poison_test_loader, preprocess):
        """计算对抗NAD的损失"""
        try:
            # 模拟防御
            defended_model = self.simulate_nad_defense(student_model, defense_data_loader, preprocess)
            
            # 评估防御效果
            original_asr, defended_asr, defense_eff = self.evaluate_defense_resistance(
                student_model, defended_model, poison_test_loader, preprocess
            )
            
            # 如果防御太有效，增加对抗损失
            if defense_eff > 0.3:  # 如果ASR下降超过30%
                # 计算对抗损失：鼓励更稳定的后门
                adversarial_loss = defense_eff * 2.0  # 放大对抗损失
                return adversarial_loss, original_asr, defended_asr
            
            return 0.0, original_asr, defended_asr
        except Exception as e:
            self.logger.warning(f"NAD对抗训练出错: {e}")
            return 0.0, 0.0, 0.0


# Evaluate the model
def eval_acc(model, loader, preprocess, DEVICE):
    model.eval()
    n_sample = 0
    n_correct = 0
    with torch.no_grad():
        for step, (x_batch, y_batch) in enumerate(loader):
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

            output = model(preprocess(x_batch))
            pred = output.max(dim=1)[1]

            n_sample += x_batch.size(0)
            n_correct += (pred == y_batch).sum().item()

    acc = n_correct / n_sample
    return acc


# Train a benign model
def train_clean(args, save_folder, logger, DEVICE):
    # Set random seed
    seed_torch(args.seed)

    model = get_model(args.dataset, args.network).to(DEVICE)

    # 下载训练集和测试集
    train_set = get_dataset(args.dataset, train=True)
    test_set = get_dataset(args.dataset, train=False)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Normalization 归一化
    preprocess, _ = get_norm(args.dataset)

    # Loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Optimizer and scheduler - 与基准攻击保持一致
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    
    # 使用与基准攻击相同的MultiStepLR调度器
    if args.epochs <= 100:
        milestones = [int(args.epochs * 0.5), int(args.epochs * 0.75)]
    else:
        milestones = [100, 150]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    logger.info(f'Clean模型: 使用标准训练设置 (SGD, lr=0.1, momentum=0.9, weight_decay=1e-4)')
    logger.info(f'Clean模型: 学习率调度milestones={milestones}')

    # Training loop
    time_start = time.time()
    for epoch in range(args.epochs):
        # Train
        model.train()
        for step, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

            optimizer.zero_grad()
            output = model(preprocess(x_batch))
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            pred = output.max(dim=1)[1]
            acc = (pred == y_batch).sum().item() / x_batch.size(0)

            if step % 10 == 0:
                sys.stdout.write('\repoch {:3}, step: {:4}, loss: {:.4f}, '
                                 .format(epoch, step, loss) + \
                                 'acc: {:.4f}'.format(acc))
                sys.stdout.flush()

        time_end = time.time()

        # Evaluate
        acc = eval_acc(model, test_loader, preprocess, DEVICE)

        # Log the training process
        logger.info(f'epoch {epoch} - {time_end - time_start:.2f}s, acc: {acc:.4f}')
        time_start = time.time()

        # Scheduler update
        scheduler.step()

    clean_model_path, legacy_clean_model_path = get_clean_model_paths(save_folder, args)
    save_torch_artifact(model, clean_model_path, legacy_clean_model_path)
    logger.info(f'💾 Clean模型已保存: {clean_model_path}')

    clean_result = {
        'network': args.network,
        'dataset': args.dataset,
        'accuracy': float(acc * 100),
        'epochs': args.epochs,
        'seed': args.seed,
    }
    clean_result_path, legacy_clean_result_path = get_clean_result_paths(save_folder, args)
    save_json_artifact(clean_result, clean_result_path, legacy_clean_result_path)
    logger.info(f'📊 Clean结果已保存: {clean_result_path}')


# Train a surrogate model for partitioning
# 所以这里只是训练模型使得模型能够正确的分区 同时不能够影响原本的分类任务
def train_surrogate(args, save_folder, logger, DEVICE):
    # Set random seed
    seed_torch(args.seed)

    train_set = get_dataset(args.dataset, train=True, augment=False)

    # Split the victim and other samples
    #这里分出来的受害者类图集 以及良性类集
    victim_images, other_dataset = split_victim_other(train_set, args.victim)

    # Data loader for other samples
    other_loader = torch.utils.data.DataLoader(other_dataset, batch_size=args.batch_size, shuffle=True)

    # Get the number of classes
    num_classes = get_config(args.dataset)['num_classes']

    # Get the partition of victim images
    victim_features = extract_feat(victim_images, DEVICE, args.batch_size)

    # Partition the victim features
    cluster = Cluster(args.cluster)
    cluster.train(victim_features, args.num_par)
    victim_par_index = cluster.predict(victim_features) #根据聚簇算法得到受害者类中 对应的每个图片的分区
    
    # 检查分区分布，如果某个分区样本太少，重新分配
    partition_counts = np.bincount(victim_par_index, minlength=args.num_par)
    min_samples_per_partition = len(victim_features) // (args.num_par * 3)  # 降低要求：至少每个分区有总样本的1/(3*num_par)
    
    # 智能分区调整：确保训练集和测试集都有合理的分区分布
    logger.info(f"训练集初始分区分布: {partition_counts}")
    logger.info(f"最小样本要求: {min_samples_per_partition}")
    
    # 强制使用用户指定的分区数，不进行自动调整
    logger.info(f"强制使用 {args.num_par} 个分区（用户指定）")
    logger.info(f"训练集分区分布: {partition_counts}")
    logger.info(f"最小样本要求: {min_samples_per_partition}")
    
    # 检查分区分布但不调整
    if np.any(partition_counts < min_samples_per_partition):
        logger.warning(f"⚠️  警告: 某些分区样本较少，但保持用户指定的分区数")
        logger.warning(f"分区分布: {partition_counts}")
        logger.warning(f"建议: 如果测试时出现空分区，请减少 --num_par 参数")
    # else:
    #     logger.info(f"训练集分区分布合理，保持 {args.num_par} 个分区")
    
    for i in range(args.num_par):
        logger.info('训练集分区 {} 有 {} 个样本'.format(i, np.sum(victim_par_index == i)))
    
    # 同时检查测试集分区分布
    test_set = get_dataset(args.dataset, train=False)
    vx_test, _ = split_victim_other(test_set, args.victim)
    test_features = extract_feat(vx_test, DEVICE, args.batch_size)
    test_par_index = cluster.predict(test_features)
    test_partition_counts = np.bincount(test_par_index, minlength=args.num_par)
    
    logger.info("测试集分区分布:")
    for i in range(args.num_par):
        count = test_partition_counts[i] if i < len(test_partition_counts) else 0
        logger.info('测试集分区 {} 有 {} 个样本'.format(i, count))
    
    # 检查测试集是否有空分区
    empty_test_partitions = [i for i, count in enumerate(test_partition_counts) if count == 0]
    if empty_test_partitions:
        logger.warning(f"⚠️  测试集中发现空分区: {empty_test_partitions}")
        logger.warning(f"建议：减少分区数或使用不同的聚类策略")

    # Preprocess training data
    # Assign vy as the sum of number of classes and the partition index
    vx, vy = victim_images.clone(), victim_par_index + num_classes

    # Data augmentation
    augment = get_augment(args.dataset)

    # Normalize
    preprocess, _ = get_norm(args.dataset)

    # Train surrogate model
    # Fine-tune the victim model
    clean_model_filepath = resolve_artifact_path(
        *get_clean_model_paths(save_folder, args),
        logger,
        'clean模型'
    )
    surrogate_model = torch.load(clean_model_filepath, map_location='cpu')

    # Change the final layer to fit the number of partitions
    # 支持多种网络架构的输出层修改
    if 'vgg' in args.network:
        num_latent = surrogate_model.classifier.in_features
        surrogate_model.classifier = nn.Linear(num_latent, num_classes + args.num_par)
    elif 'resnet' in args.network or 'prn' in args.network or 'senet' in args.network:
        num_latent = surrogate_model.linear.in_features
        surrogate_model.linear = nn.Linear(num_latent, num_classes + args.num_par)
    elif 'densenet' in args.network:
        num_latent = surrogate_model.linear.in_features
        surrogate_model.linear = nn.Linear(num_latent, num_classes + args.num_par)
    elif 'alexnet' in args.network:
        # AlexNet的classifier在Sequential中，需要修改最后一层
        num_latent = surrogate_model.classifier[-1].in_features
        surrogate_model.classifier[-1] = nn.Linear(num_latent, num_classes + args.num_par)
    elif 'mobilenet' in args.network:
        num_latent = surrogate_model.linear.in_features
        surrogate_model.linear = nn.Linear(num_latent, num_classes + args.num_par)
    elif 'googlenet' in args.network:
        num_latent = surrogate_model.linear.in_features
        surrogate_model.linear = nn.Linear(num_latent, num_classes + args.num_par)
    else:
        # 尝试自动检测输出层
        if hasattr(surrogate_model, 'linear'):
            num_latent = surrogate_model.linear.in_features
            surrogate_model.linear = nn.Linear(num_latent, num_classes + args.num_par)
        elif hasattr(surrogate_model, 'classifier'):
            if isinstance(surrogate_model.classifier, nn.Sequential):
                num_latent = surrogate_model.classifier[-1].in_features
                surrogate_model.classifier[-1] = nn.Linear(num_latent, num_classes + args.num_par)
            else:
                num_latent = surrogate_model.classifier.in_features
                surrogate_model.classifier = nn.Linear(num_latent, num_classes + args.num_par)
        elif hasattr(surrogate_model, 'fc'):
            num_latent = surrogate_model.fc.in_features
            surrogate_model.fc = nn.Linear(num_latent, num_classes + args.num_par)
        else:
            raise NotImplementedError(f"无法自动检测网络架构 {args.network} 的输出层，请手动添加支持")

    surrogate_model = surrogate_model.to(DEVICE)

    # Optimizer and scheduler - 与基准攻击保持一致
    optimizer = torch.optim.SGD(surrogate_model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    
    # 使用与基准攻击相同的MultiStepLR调度器
    if args.epochs <= 100:
        milestones = [int(args.epochs * 0.5), int(args.epochs * 0.75)]
    else:
        milestones = [100, 150]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    logger.info(f'Surrogate模型: 使用标准训练设置 (SGD, lr=0.1, momentum=0.9, weight_decay=1e-4)')
    logger.info(f'Surrogate模型: 学习率调度milestones={milestones}')

    # Loss function
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        # Train surrogate model
        surrogate_model.train()
        for _, (x_batch, y_batch) in enumerate(other_loader):
            # Victim samples
            cur_bs = x_batch.size(0)
            #取受害者类样本的数量为良性类样本数量的1/(num_classes-1) 相当于是让每个类别的样本数量均衡
            # 对于大类别数据集（如CIFAR-100），确保每个batch至少有足够的victim样本
            vic_bs = int(cur_bs / (num_classes - 1))
            vic_bs = max(vic_bs, 10)  # 至少10个victim样本，确保能有效学习

            # Randomly sample victim_bs indexes from victim_dataset  ！！！ 从受害者类样本中随机采样
            v_index = np.random.choice(vx.shape[0], vic_bs, replace=False)
            batch_vx, batch_vy = vx[v_index], vy[v_index]

            # Merge victim and other dataset  ！！！ 将受害者类样本和良性类样本合并在一起训练
            x_batch = torch.cat([x_batch, batch_vx], dim=0)
            y_batch = np.concatenate([y_batch, batch_vy], axis=0)

            # Augment data
            x_batch = augment(x_batch)
            y_batch = torch.from_numpy(y_batch).long()

            # To device
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

            output = surrogate_model(preprocess(x_batch))
            loss = criterion(output, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Scheduler update
        scheduler.step()

        # Evaluate surrogate model
        if (epoch + 1) % 1 == 0:
            surrogate_model.eval()
            # 取受害者类样本进行评估
            nb = int(np.ceil(vx.shape[0] / args.batch_size))
            correct = 0
            total = 0
            for i in range(nb):
                #计算victim_acc 受害者类样本的准确率(是加上分区后的正确率)
                bx = vx[i * args.batch_size:(i + 1) * args.batch_size, ...]
                by = vy[i * args.batch_size:(i + 1) * args.batch_size, ...]

                bx = bx.to(DEVICE)
                by = torch.from_numpy(by).long().to(DEVICE)

                output = surrogate_model(preprocess(bx))
                _, predicted = torch.max(output.data, 1)
                total += by.size(0)
                correct += (predicted == by).sum().item()

            victim_acc = correct / total

            correct = 0
            total = 0
            for _, (bx, by) in enumerate(other_loader):
                bx = bx.to(DEVICE)
                by = by.to(DEVICE)

                output = surrogate_model(preprocess(bx))
                _, predicted = torch.max(output.data, 1)
                total += by.size(0)
                correct += (predicted == by).sum().item()

            other_acc = correct / total

            logger.info(
                f'Epoch {epoch + 1}/{args.epochs} | Loss: {loss.item():.4f} | Victim acc: {victim_acc * 100.:.2f}% | Other acc: {other_acc * 100.:.2f}%')

    surrogate_model_path, legacy_surrogate_model_path = get_surrogate_model_paths(save_folder, args)
    save_torch_artifact(surrogate_model, surrogate_model_path, legacy_surrogate_model_path)
    logger.info(f'💾 Surrogate模型已保存: {surrogate_model_path}')


# LOTUS backdoor attack
def train_lotus(args, save_folder, logger, DEVICE):
    # Set random seed
    seed_torch(args.seed)

    # Load implicit partition
    surrogate_filepath = resolve_artifact_path(
        *get_surrogate_model_paths(save_folder, args),
        logger,
        'surrogate模型'
    )
    logger.info('Load pre-trained surrogate model')
    # 创建分区对象 用于后续获取样本的分区索引
    partition_secret = Partition(args, DEVICE, surrogate_filepath)

    # Load training data
    train_set = get_dataset(args.dataset, train=True, augment=False)

    # Data augmentation
    augment = get_augment(args.dataset)

    # Split the victim and other samples
    #将训练集分为受害者类图集 和 良性类图集 (都是训练的时候使用)
    victim_images, other_dataset = split_victim_other(train_set, args.victim, transform=augment)
    other_loader = torch.utils.data.DataLoader(other_dataset, batch_size=args.batch_size, shuffle=True)

    # Normalize
    preprocess, _ = get_norm(args.dataset)

    # Load testing data
    test_set = get_dataset(args.dataset, train=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # Load poisoned testing data
    #测试集 这里单独拿到受害者类图集 用于评估
    vx_test, _ = split_victim_other(test_set, args.victim)
    #得到测试集上 受害者类样本的分区索引
    v_index_test = partition_secret.get_partition_index(vx_test)

    # 添加数据验证
    # verify_training_data(args, vx_test, v_index_test, logger)

    # 获取动态分区映射
    current_par2target = get_par2target(args.num_par)

    # 按分区映射测试集目标标签（从 v_index_test 查表）测试集上面
    test_target_labels = [current_par2target[par_idx] for par_idx in v_index_test]
    test_target_labels = np.array(test_target_labels)

    # 构造带有动态目标标签的毒化测试集
    #为每个受害者类样本添加对应的触发器，并将其标签修改为动态目标标签
    #到这里测试集相当于就构建好了
    poison_set = PoisonTestDataset(
        vx_test, v_index_test, test_target_labels,
        poison_alpha=getattr(args, 'freq_alpha', 0.5) if getattr(args, 'use_frequency', False) else 0.2,
        use_frequency=getattr(args, 'use_frequency', False),
        freq_mask_type=getattr(args, 'freq_mask_type', 'low'),
        freq_mask_ratio=getattr(args, 'freq_mask_ratio', 0.15)
    )
    poison_loader = torch.utils.data.DataLoader(poison_set, batch_size=args.batch_size, shuffle=False)

    # Get the number of classes
    num_classes = get_config(args.dataset)['num_classes']

    # Fine-tune the clean model
    clean_model_filepath = resolve_artifact_path(
        *get_clean_model_paths(save_folder, args),
        logger,
        'clean模型'
    )
    model = torch.load(clean_model_filepath, map_location='cpu')
    model = model.to(DEVICE)

    # 初始化黑盒对抗训练组件（如果启用）
    blackbox_adversarial = None
    defense_loader = None
    if hasattr(args, 'adversarial_nad') and args.adversarial_nad:
        logger.info("启用黑盒对抗训练模式")
        blackbox_adversarial = BlackBoxAdversarialTrainer(model, args, DEVICE, logger)
        
        # 准备防御数据（少量干净数据）
        defense_dataset = blackbox_adversarial.get_defense_dataset(train_set, defense_ratio=0.05)
        defense_loader = DataLoader(defense_dataset, batch_size=args.batch_size, shuffle=True)
        logger.info(f"防御数据集大小: {len(defense_dataset)} 样本")

    # # Loss function
    criterion = nn.CrossEntropyLoss()

    # ========== 根据 batch_size 动态调整参数 ==========
    # 基准 batch_size（用于计算缩放因子）
    # 注意：默认 batch_size=128，此时 n_indi=3, n_comb=1
    base_batch_size = 128
    
    # 计算 batch_size 缩放因子
    batch_scale = args.batch_size / base_batch_size
    
    # 1. 动态调整学习率（线性缩放或平方根缩放）
    # 线性缩放：lr_new = lr_base * batch_scale
    # 平方根缩放：lr_new = lr_base * sqrt(batch_scale)（更稳定，推荐）
    # 但要注意：对于大 batch_size，学习率不要过高
    base_lr = 0.001
    # 使用平方根缩放，但限制最大学习率
    adjusted_lr = base_lr * np.sqrt(batch_scale)
    # 限制最大学习率，避免训练不稳定
    max_lr = 0.002  # 最大学习率限制
    adjusted_lr = min(adjusted_lr, max_lr)
    logger.info(f'📊 Batch Size 动态调整: batch_size={args.batch_size}, 缩放因子={batch_scale:.2f}')
    logger.info(f'   学习率: {base_lr:.6f} → {adjusted_lr:.6f} (sqrt缩放)')
    
    # 2. 动态调整特征正则化权重
    # 当 batch_size 增大时，毒化样本数量增加，特征正则化损失可能过强
    # 需要适当降低权重，避免过度抑制后门学习
    base_feature_weight = 0.1
    # 使用平方根反比缩放：batch_size 越大，权重越小（但不要太小）
    adjusted_feature_weight = base_feature_weight / np.sqrt(batch_scale)
    # 限制在合理范围内 [0.02, 0.15]（降低下限，避免过度抑制）
    adjusted_feature_weight = max(0.02, min(0.15, adjusted_feature_weight))
    logger.info(f'   特征正则化权重: {base_feature_weight:.3f} → {adjusted_feature_weight:.3f}')
    
    # 优化器设置：使用动态调整后的学习率
    optimizer = torch.optim.SGD(model.parameters(), lr=adjusted_lr, momentum=0.9, weight_decay=5e-4)
    
    # 使用 MultiStepLR 学习率调度器
    if args.epochs <= 100:
        milestones = [int(args.epochs * 0.5), int(args.epochs * 0.75)]
    else:
        # 对于更长的训练，可以设置固定的milestones
        # milestones = [100, 150]
        milestones = [100, 150]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    
    logger.info(f'LOTUS改进版: 使用优化器设置 (SGD, lr={adjusted_lr:.6f}, momentum=0.9, weight_decay=5e-4)')
    logger.info(f'LOTUS改进版: 使用 MultiStepLR 学习率调度 (milestones={milestones}, gamma=0.1)')
    logger.info(f'   - 优点: 在指定轮次降低学习率，训练过程更稳定，避免后期性能下降。')

    # 初始化特征正则化器（使用动态调整后的权重）
    # 暂时禁用特征正则化以提升 ASR
    disable_feature_reg = True  # 设置为 True 以禁用特征正则化
    if disable_feature_reg:
        feature_regularizer = None
        logger.info('⚠️  特征级正则化已禁用（用于提升 ASR）')
    else:
        feature_regularizer = FeatureRegularizer(feature_loss_weight=adjusted_feature_weight)
        logger.info(f'启用特征级正则化（权重={feature_regularizer.feature_loss_weight:.3f}，已根据batch_size调整）')

    # Training loop
    best_acc = 0
    best_asr = 0
    # 不使用早停机制，但恢复高ASR阈值
    final_model = None
    # 恢复LOTUS原始的高ASR阈值
    asr_bound = 0.7  # 恢复到高阈值，确保只保存高质量模型
    for epoch in range(args.epochs):
        # Train poisoned model
        model.train()

        # Record the loss
        log_ce_loss = 0
        log_feat_loss = 0  # 记录特征正则化损失

        for _, (x_batch, y_batch) in enumerate(other_loader):

            cur_bs = x_batch.size(0)
            vic_bs = int(cur_bs / (num_classes - 1))
            vic_bs = max(vic_bs, 10)

            # Number of samples for victim/negative training
            # 3. 动态调整负样本数量（根据 batch_size 线性缩放，但限制比例）
            # 基准：batch_size=128 时，n_indi=3, n_comb=1
            # 当 batch_size 增大时，负样本数量按比例线性增加
            # 但限制负样本相对于毒化样本的比例，避免过度抑制
            n_indi_base = max(1, int(args.n_indi * batch_scale)) if args.n_indi > 0 else 0
            n_comb_base = max(1, int(args.n_comb * batch_scale)) if args.n_comb > 0 else 0
            
            # 限制负样本比例：不超过毒化样本的 15%（避免过度抑制）
            max_neg_ratio = 0.25
            max_neg_samples = int(vic_bs * max_neg_ratio)
            total_neg_base = n_indi_base + n_comb_base
            
            if total_neg_base > max_neg_samples:
                # 如果超过限制，按比例缩减
                scale_down = max_neg_samples / total_neg_base
                n_indi = max(1, int(n_indi_base * scale_down)) if n_indi_base > 0 else 0
                n_comb = max(1, int(n_comb_base * scale_down)) if n_comb_base > 0 else 0
            else:
                n_indi = n_indi_base
                n_comb = n_comb_base
            
            # 记录调整信息（仅在第一个epoch的第一个batch）
            if epoch == 0 and _ == 0:
                logger.info(f'📊 负样本动态调整:')
                logger.info(f'   基准 batch_size={base_batch_size}: n_indi={args.n_indi}, n_comb={args.n_comb}')
                logger.info(f'   当前 batch_size={args.batch_size}: n_indi={n_indi}, n_comb={n_comb} (缩放因子={batch_scale:.2f})')

            # Randomly sample (2 * vic_bs) indexes from victim_dataset
            victim_indexes = np.random.choice(victim_images.shape[0], 2 * vic_bs, replace=False)
            x_v = victim_images[victim_indexes]

            # Augment victim samples
            x_v = augment(x_v)

            # Get the partition index of victim samples
            with torch.no_grad():
                p_v = partition_secret.get_partition_index(x_v)

            # First vic_bs indexes are used for victim training
            #首先取前vic_bs个样本
            x_b = x_v[:vic_bs]
            #他的标签就是受害者类标签
            y_b = torch.zeros(x_b.shape[0]).long() + args.victim
            #p_b记录的是这些样本的分区索引
            p_b = p_v[:vic_bs]

            # Second vic_bs indexes are used for poisoning training
            x_p = x_v[vic_bs:]
            p_p = p_v[vic_bs:]

            # 调试：检查分区索引范围
            if len(p_p) > 0:
                # 确保 p_p 是 torch 张量
                if isinstance(p_p, np.ndarray):
                    p_p = torch.from_numpy(p_p)
                # print(f"调试信息: p_p范围 = {p_p.min().item()} 到 {p_p.max().item()}")
                # print(f"调试信息: p_p唯一值 = {torch.unique(p_p).tolist()}")
                # print(f"调试信息: num_par = {args.num_par}")
            
            # 使用 trigger_focus 生成组合触发器样本和负样本
            # 🔧 关键修复：使用动态调整后的 n_indi 和 n_comb，而不是 args 中的原始值
            x_p, y_p = trigger_focus(
                x_p, p_p, n_indi, n_comb, args.victim, current_par2target, args.num_par,
                alpha_poison=getattr(args, 'freq_alpha', 0.5) if getattr(args, 'use_frequency', False) else 0.2,
                alpha_neg=0.2,
                use_frequency=getattr(args, 'use_frequency', False),
                freq_mask_type=getattr(args, 'freq_mask_type', 'low'),
                freq_mask_ratio=getattr(args, 'freq_mask_ratio', 0.15)
            )
            
            # 添加标签验证
            model_output_size = num_classes + args.num_par

            # Negative samples of other classes, randomly 5% of the batch size
            #加入对抗样本进来 (良性类样本中随机选取5%) 标签为良性类标签 但是加上了随机触发器
            # 这些负样本帮助模型学习：带触发器的良性类样本不应该被误分类
            disable_negative_samples = args.n_indi <= 0 and args.n_comb <= 0
            num_negative = 0 if disable_negative_samples else max(1, int(cur_bs * 0.01))  # 保持5%
            index_on = np.random.choice(cur_bs, min(num_negative, cur_bs), replace=False)
            x_on = []
            for i in index_on:
                # 随机选择一个分区
                p_rand = np.random.choice(args.num_par)
                # 使用频域或像素空间方法
                if getattr(args, 'use_frequency', False):
                    from trigger import stamp_trigger_frequency
                    x_on.append(stamp_trigger_frequency(x_batch[i], p_rand,
                                                        alpha=getattr(args, 'freq_alpha', 0.5),
                                                        freq_mask_type=getattr(args, 'freq_mask_type', 'low'),
                                                        mask_ratio=getattr(args, 'freq_mask_ratio', 0.15)))
                else:
                    x_on.append(stamp_trigger(x_batch[i], p_rand))
            
            # 检查是否有对抗样本
            if len(x_on) > 0:
                x_on = torch.stack(x_on, dim=0)
                # 确保index_on是torch张量，并且在CPU上
                index_on_tensor = torch.from_numpy(index_on).long()
                y_on = y_batch[index_on_tensor]
            else:
                # 如果没有对抗样本，创建空的张量
                x_on = torch.empty(0, *x_batch.shape[1:], dtype=x_batch.dtype, device=x_batch.device)
                y_on = torch.empty(0, dtype=y_batch.dtype, device=y_batch.device)

            # Merge victim and other dataset
            x_batch = torch.cat([x_batch, x_b, x_p, x_on], dim=0)
            y_batch = torch.cat([y_batch, y_b, y_p, y_on], dim=0)
            
            # 添加详细的标签调试信息
            if epoch == 0 and _ == 0:  # 只在第一个epoch的第一个batch检查
                logger.info(f"标签调试信息:")
                logger.info(f"  y_batch (other) 范围: {y_batch.min().item()} 到 {y_batch.max().item()}")
                logger.info(f"  y_b (victim) 范围: {y_b.min().item()} 到 {y_b.max().item()}")
                logger.info(f"  y_p (poisoned) 范围: {y_p.min().item()} 到 {y_p.max().item()}")
                if len(y_on) > 0:
                    logger.info(f"  y_on (negative) 范围: {y_on.min().item()} 到 {y_on.max().item()}")
                else:
                    logger.info("  y_on (negative) 范围: 无负样本")
                logger.info(f"  合并后 y_batch 范围: {y_batch.min().item()} 到 {y_batch.max().item()}")
                logger.info(f"  模型输出层大小: {num_classes + args.num_par}")
                logger.info(f"  标签总数: {len(y_batch)}")
                
                # 检查是否有无效标签
                invalid_labels = y_batch[y_batch < 0] if len(y_batch[y_batch < 0]) > 0 else None
                if invalid_labels is not None:
                    logger.error(f"发现负标签: {invalid_labels}")
                
                invalid_labels = y_batch[y_batch >= num_classes] if len(y_batch[y_batch >= num_classes]) > 0 else None
                if invalid_labels is not None:
                    logger.error(f"发现超出范围的标签: {invalid_labels}")
                    logger.error(f"最大有效标签应该是: {num_classes - 1}")
                    raise ValueError(f"标签越界: 发现标签{invalid_labels.max().item()} >= {num_classes}")

            # To device
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

            # ========== 特征级正则化：分离良性和毒化样本 ==========
            # batch组成：[other样本, 受害者类良性x_b, 毒化样本x_p, 负样本x_on]
            other_size = cur_bs
            benign_victim_size = len(x_b)
            poison_size = len(x_p)
            negative_size = len(x_on) if len(x_on) > 0 else 0
            
            # 计算分界点
            split_1 = other_size
            split_2 = split_1 + benign_victim_size
            split_3 = split_2 + poison_size
            # split_4 = split_3 + negative_size (这是batch末尾)
            
            # 前向传播（同时提取特征）
            output, features = model(preprocess(x_batch), with_latent=True)
            
            # 添加输出层大小验证
            if epoch == 0 and _ == 0:
                logger.info(f"模型输出层大小: {output.shape[1]}")
                logger.info(f"预期输出层大小: {num_classes}")
                if output.shape[1] != num_classes:
                    logger.error(f"模型输出层大小不匹配: {output.shape[1]} != {num_classes}")
                    raise ValueError(f"模型输出层大小不匹配")
            
            # 计算交叉熵损失
            ce_loss = criterion(output, y_batch)
            
            # 计算特征正则化损失（让毒化样本特征接近良性样本）
            feat_reg_loss = 0.0
            if feature_regularizer is not None and poison_size > 0 and benign_victim_size > 0:
                # 提取受害者类良性样本的特征
                benign_victim_features = features[split_1:split_2]
                # 提取毒化样本的特征
                poison_features = features[split_2:split_3]
                
                # 计算特征相似性损失
                feat_reg_loss = feature_regularizer.get_weighted_loss(
                    benign_victim_features, poison_features
                )
                log_feat_loss += feat_reg_loss.item()
            
            # 总损失
            loss = ce_loss + feat_reg_loss
                
            log_ce_loss += ce_loss.item()

            # 黑盒对抗训练逻辑
            if blackbox_adversarial is not None and defense_loader is not None:
                # 每隔一定步数进行黑盒对抗训练
                if (_ % getattr(args, 'nad_test_freq', 10)) == 0:
                    orig_asr, def_asr, defense_eff = blackbox_adversarial.simulate_blackbox_defense(
                        model, defense_loader, poison_loader, preprocess
                    )
                    
                    # 根据防御效果计算对抗损失
                    adversarial_loss = defense_eff * blackbox_adversarial.attack_strength
                    
                    if adversarial_loss > 0:
                        # 使用自适应权重
                        adversarial_weight = getattr(args, 'adversarial_weight', 0.01) * blackbox_adversarial.defense_resistance
                        loss += adversarial_weight * adversarial_loss
                        
                        if (_ % 50) == 0:  # 每50个batch记录一次
                            logger.info(f"黑盒对抗: 原始ASR={orig_asr:.3f}, 防御后ASR={def_asr:.3f}, 防御效果={defense_eff:.3f}, 攻击强度={blackbox_adversarial.attack_strength:.3f}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Scheduler update
        scheduler.step()
        
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']

        # Evaluation
        if (epoch + 1) % 1 == 0:
            model.eval()
            acc = eval_acc(model, test_loader, preprocess, DEVICE)
            asr = eval_acc(model, poison_loader, preprocess, DEVICE)
            asr_par_indi = eval_asr_par(args, model, vx_test, v_index_test, preprocess, DEVICE, partition_secret)
            pnt_par = ' '.join([f'{p * 100.0:.2f}%' for p in asr_par_indi])

            log_ce_loss /= len(other_loader)
            log_feat_loss /= len(other_loader)
            
            # 更新特征正则化器的训练信息
            if feature_regularizer is not None:
                feature_regularizer.update_training_info(
                    epoch=epoch,
                    total_epochs=args.epochs,
                    current_asr=asr,
                    best_asr=best_asr
                )

            logger.info(
                f'Epoch {epoch + 1}/{args.epochs} | LR: {current_lr:.6f} | CE Loss: {log_ce_loss:.4f} | Feat Loss: {log_feat_loss:.4f} | BA: {acc * 100.0:.2f}% | ASR: {asr * 100.0:.2f}% | ASR_par_indi: {pnt_par}')

            # 黑盒防御抗性测试（每几个epoch进行一次完整测试）
            if blackbox_adversarial is not None and ((epoch + 1) % getattr(args, 'nad_test_epoch_freq', 5)) == 0:
                logger.info(f"🛡️ Epoch {epoch+1}: 进行黑盒防御抗性测试...")
                orig_asr_test, def_asr_test, def_eff = blackbox_adversarial.simulate_blackbox_defense(
                    model, defense_loader, poison_loader, preprocess
                )
                resistance = 1 - def_eff
                logger.info(f"黑盒防御抗性测试: 原始ASR={orig_asr_test:.3f}, 防御后ASR={def_asr_test:.3f}, 抗性={resistance:.3f}")
                
                # 记录黑盒防御抗性到日志文件
                defense_log_path = f'{save_folder}/blackbox_defense_resistance_log.json'
                defense_log = {
                    'epoch': epoch + 1,
                    'original_asr': orig_asr_test,
                    'defended_asr': def_asr_test,
                    'defense_effectiveness': def_eff,
                    'resistance': resistance,
                    'attack_strength': blackbox_adversarial.attack_strength,
                    'defense_resistance': blackbox_adversarial.defense_resistance
                }
                
                # 读取现有日志或创建新的
                if os.path.exists(defense_log_path):
                    with open(defense_log_path, 'r') as f:
                        logs = json.load(f)
                else:
                    logs = []
                
                logs.append(defense_log)
                with open(defense_log_path, 'w') as f:
                    json.dump(logs, f, indent=4)

            # 与基准攻击保持一致的模型保存策略
            if asr >= asr_bound and asr > best_asr:
                best_model_path, legacy_best_model_path = get_lotus_model_paths(save_folder, args, 'best')
                save_torch_artifact(model, best_model_path, legacy_best_model_path)
                best_acc = acc
                best_asr = asr
                logger.info(f"✅ 保存新的最佳模型: ASR={asr*100:.2f}%, BA={acc*100:.2f}%, 轮次={epoch+1}")

            # Update the final model
            if asr >= asr_bound:
                final_model = copy.deepcopy(model)

    # Save the final model（原逻辑：只保存最后一次ASR达标的模型）
    # 如果 final_model 为 None（ASR 未达到阈值），使用最后一轮的模型
    final_model_path, legacy_final_model_path = get_lotus_model_paths(save_folder, args, 'final')
    if final_model is not None:
        save_torch_artifact(final_model, final_model_path, legacy_final_model_path)
        logger.info(f"💾 模型已保存: {os.path.basename(final_model_path)} (ASR达标模型)")
    else:
        # ASR 未达到阈值，保存最后一轮模型作为 final 模型
        save_torch_artifact(model, final_model_path, legacy_final_model_path)
        logger.warning(f"⚠️  ASR未达到阈值({asr_bound*100:.1f}%)，保存最后一轮模型作为final模型")
    # 无论如何都保存最后一轮模型
    last_model_path, legacy_last_model_path = get_lotus_model_paths(save_folder, args, 'last')
    save_torch_artifact(model, last_model_path, legacy_last_model_path)
    logger.info(f"💾 最后轮模型: {os.path.basename(last_model_path)}")


def eval_asr_par(args, model, x, p, preprocess, DEVICE, partition_secret=None):
    model.eval()

    # 按分区映射评估目标标签
    current_par2target = get_par2target(args.num_par)
    dynamic_target_labels = np.array([current_par2target[par_idx] for par_idx in p])

    # 存储每个触发器的正确配对ASR
    correct_pair_asr = []

    nb = int(np.ceil(x.size(0) / args.batch_size))
    with torch.no_grad():
        for trigger_idx in range(args.num_par):
            trigger_correct = []

            for batch_idx in range(nb):
                start_idx = batch_idx * args.batch_size
                end_idx = min((batch_idx + 1) * args.batch_size, x.size(0))
                x_batch = x[start_idx:end_idx].to(DEVICE)
                batch_target_labels = dynamic_target_labels[start_idx:end_idx]
                batch_partitions = p[start_idx:end_idx]

                # 应用触发器（使用与训练时相同的方法）
                if getattr(args, 'use_frequency', False):
                    from trigger import stamp_trigger_frequency
                    px = torch.stack([stamp_trigger_frequency(
                        img, trigger_idx,
                        alpha=getattr(args, 'freq_alpha', 0.5),
                        freq_mask_type=getattr(args, 'freq_mask_type', 'low'),
                        mask_ratio=getattr(args, 'freq_mask_ratio', 0.15)
                    ) for img in x_batch])
                else:
                    px = torch.stack([stamp_trigger(img, trigger_idx) for img in x_batch])
                output = model(preprocess(px))
                pred = output.max(dim=1)[1].cpu().numpy()

                # 只计算正确配对的样本 (trigger_idx == partition_idx)
                correct_mask = (batch_partitions == trigger_idx)
                if np.any(correct_mask):
                    correct_samples = (pred == batch_target_labels)[correct_mask]
                    trigger_correct.extend(correct_samples)

            # 计算这个触发器的正确配对ASR
            if len(trigger_correct) > 0:
                asr = np.mean(trigger_correct)
            else:
                asr = 0.0
            correct_pair_asr.append(asr)

    return np.array(correct_pair_asr)

def test(args, save_folder, logger, DEVICE):
    # Load the model
    suffix = 'final'  # 'best'
    model_filepath = resolve_artifact_path(
        *get_lotus_model_paths(save_folder, args, suffix),
        logger,
        'LOTUS模型文件'
    )
    logger.info(f'📂 加载模型: {model_filepath}')
    model = torch.load(model_filepath, map_location='cpu')
    
    # 检查模型是否为 None（可能是修复前保存的）
    if model is None:
        logger.warning(f"⚠️  模型文件包含 None，尝试加载 last 模型作为备选")
        last_model_filepath = resolve_artifact_path(
            *get_lotus_model_paths(save_folder, args, 'last'),
            logger,
            'LOTUS last模型文件'
        )
        logger.info(f"📂 加载 last 模型: {last_model_filepath}")
        model = torch.load(last_model_filepath, map_location='cpu')
        if model is None:
            raise ValueError(f"Last 模型也是 None，请重新训练模型")
    
    model = model.to(DEVICE)
    model.eval()

    # Load implicit partition
    surrogate_filepath = resolve_artifact_path(
        *get_surrogate_model_paths(save_folder, args),
        logger,
        'surrogate模型'
    )
    partition_secret = Partition(args, DEVICE, surrogate_filepath)

    preprocess, _ = get_norm(args.dataset)

    # Load test data
    test_set = get_dataset(args.dataset, train=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # 获取受害者类测试样本
    vx_test, _ = split_victim_other(test_set, args.victim)
    v_index_test = partition_secret.get_partition_index(vx_test)

    # 评估清洁准确率
    acc = eval_acc(model, test_loader, preprocess, DEVICE)

    # 评估攻击成功率
    # ASR: 正确配对的触发器攻击成功率 (trigger_idx == partition_idx)
    # ASR_par: 交叉触发器攻击成功率 (trigger_idx != partition_idx，但仍预测为对应目标)
    
    # 获取每个样本的目标标签
    current_par2target = get_par2target(args.num_par)
    test_target_labels = np.array([current_par2target[p] for p in v_index_test])
    
    # 新的ASR_par计算方法：使用所有触发器组合
    # 生成所有可能的触发器组合（除了'0000'）
    choice = []
    total = 2 ** args.num_par  # 2^4 = 16
    for i in range(1, total):  # 跳过'0000'
        choice.append(bin(i)[2:].zfill(args.num_par))
    
    logger.info(f"开始测试 {len(vx_test)} 个样本的 {len(choice)} 种触发器组合...")
    
    # 为每个样本生成所有触发器组合的预测结果
    n_samples = len(vx_test)
    pdict = {'par': [], 'pred': []}
    
    with torch.no_grad():
        for i in range(0, n_samples, args.batch_size):
            batch_end = min(i + args.batch_size, n_samples)
            x_batch = vx_test[i:batch_end].to(DEVICE)
            p_batch = v_index_test[i:batch_end]
            
            # 对当前batch的每个样本
            for sample_idx in range(len(x_batch)):
                sample = x_batch[sample_idx:sample_idx+1]  # [1, C, H, W]
                sample_partition = p_batch[sample_idx]
                
                pdict['par'].append(sample_partition)
                
                # 对该样本测试所有触发器组合
                sample_predictions = []
                for code in choice:
                    # 根据code应用触发器组合
                    triggered_sample = sample.clone()
                    
                    for trigger_idx in range(args.num_par):
                        if code[trigger_idx] == '1':  # 需要应用这个触发器
                            if getattr(args, 'use_frequency', False):
                                triggered_sample = stamp_trigger_frequency(
                                    triggered_sample[0], trigger_idx,
                                    alpha=getattr(args, 'freq_alpha', 0.5),
                                    freq_mask_type=getattr(args, 'freq_mask_type', 'low'),
                                    mask_ratio=getattr(args, 'freq_mask_ratio', 0.15)
                                ).unsqueeze(0)
                            else:
                                triggered_sample = stamp_trigger(
                                    triggered_sample[0], trigger_idx, alpha=0.2
                                ).unsqueeze(0)
                    
                    # 预测
                    output = model(preprocess(triggered_sample))
                    pred = output.max(dim=1)[1].cpu().item()
                    sample_predictions.append(pred)
                
                pdict['pred'].append(sample_predictions)
    
    # 使用与参考代码完全相同的计算逻辑
    n_sample = len(pdict['par'])
    n_asr, n_asr_par, n_acc_par = [], [], []
    cp_asr_vals, sa_asr_vals = [], []
    map_par_asr = {}
    for code in choice:
        map_par_asr[code] = [0 for _ in range(args.num_par)]
    cnt_par = [0 for _ in range(args.num_par)]
    
    for i in range(n_sample):
        par = pdict['par'][i]
        cnt_par[par] += 1
        tar = ['0' for _ in range(args.num_par)]
        tar[par] = '1'
        tar = ''.join(tar)  # 正确的触发器组合，如'0010'
        pred = pdict['pred'][i]
        
        for j in range(len(choice)):
            code = choice[j]
            # 获取样本所属分区对应的目标类别
            target_class = current_par2target[par]
            map_par_asr[code][par] += (pred[j] == target_class)

            if code == tar:  # 正确的触发器组合
                n_asr.append((pred[j] == target_class) * 1)
            else:  # 错误的触发器组合
                n_asr_par.append((pred[j] == target_class) * 1)
                n_acc_par.append((pred[j] == args.victim) * 1)
                if code.count('1') == 1:
                    cp_asr_vals.append((pred[j] == target_class) * 1)
                else:
                    sa_asr_vals.append((pred[j] == target_class) * 1)

    asr = np.mean(n_asr) * 100
    asr_par = np.array(n_asr_par)
    acc_par = np.mean(n_acc_par) * 100
    cp_asr = np.mean(cp_asr_vals) * 100 if len(cp_asr_vals) > 0 else 0.0
    sa_asr = np.mean(sa_asr_vals) * 100 if len(sa_asr_vals) > 0 else 0.0

    logger.info(
        f'ACC: {acc*100:.2f}%, ASR: {asr:.2f}%, '
        f'CP-ASR: {cp_asr:.2f}%, SA-ASR: {sa_asr:.2f}%, '
        f'ASR_par: {np.mean(asr_par)*100:.2f}% +- {np.std(asr_par)*100:.2f}%'
    )
    
    # 可选：输出详细的分区统计
    logger.info(f"分区样本统计: {cnt_par}")
    
    # 输出每种触发器组合的统计（可选）
    # 显示前5个组合作为示例
    for code in choice[:5]:  
        asr_by_partition = [map_par_asr[code][p] / max(cnt_par[p], 1) * 100 
                          for p in range(args.num_par)]
        logger.info(f"触发器组合 {code}: {asr_by_partition}")

    # 保存详细结果
    result = {
        'network': args.network,
        'attack_name': 'LOTUS',
        'attack_variant': get_model_suffix(args),
        'accuracy': acc * 100,
        'asr': asr,
        'cp_asr': cp_asr,
        'sa_asr': sa_asr,
        'asr_par_mean': np.mean(asr_par) * 100,
        'asr_par_std': np.std(asr_par) * 100,
        'acc_par': acc_par,
        'num_samples': n_samples,
        'num_partitions': args.num_par,
        'num_trigger_combinations': len(choice)
    }

    result_path, legacy_result_path = get_attack_result_paths(save_folder, args)
    save_json_artifact(result, result_path, legacy_result_path)
    logger.info(f'📊 攻击结果已保存: {result_path}')


def verify_training_data(args, vx_test, v_index_test, logger):
    """验证训练数据的正确性"""
    # 检查分区分布
    partition_counts = np.bincount(v_index_test, minlength=args.num_par)
    logger.info("测试集分区样本分布:")
    for i in range(args.num_par):
        count = partition_counts[i] if i < len(partition_counts) else 0
        logger.info(f"分区 {i}: {count} 个样本")
    
    # 检查是否有分区样本为0
    zero_partitions = [i for i, count in enumerate(partition_counts) if count == 0]
    if zero_partitions:
        logger.warning(f"⚠️  发现空分区: {zero_partitions}")
        logger.warning(f"这些分区的ASR将显示为0.00%，因为没有任何测试样本")
        logger.warning(f"建议：减少分区数或使用不同的聚类策略")

    # 验证目标标签映射
    current_par2target = get_par2target(args.num_par)
    for i in range(args.num_par):
        if i not in current_par2target:
            logger.warning(f"分区 {i} 在target_map中没有对应的目标标签")
