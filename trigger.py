import torch
import torch.nn.functional as F
import torch.fft
import time

import os
from torchvision.utils import save_image
import torchvision.utils as vutils

from sklearn.cluster import KMeans
import numpy as np


# ============ 频域后门注入函数 (基于 FIBA 机制) ============

def inject_trigger_frequency(
    clean_image: torch.Tensor,
    trigger_pattern: torch.Tensor,
    alpha: float = 0.3,
    freq_mask_type: str = 'low',
    mask_ratio: float = 0.1
) -> torch.Tensor:
    """
    在频域中注入后门触发器（基于 FIBA 思路）
    
    Args:
        clean_image: 干净图像，shape: [C, H, W] 或 [B, C, H, W]
        trigger_pattern: 触发器模式，shape 与 clean_image 相同
        alpha: 频域混合强度，控制触发器在频谱中的幅度
        freq_mask_type: 频域掩码类型
            - 'low': 低频区域注入（中心区域）
            - 'high': 高频区域注入（边缘区域）
            - 'mid': 中频区域注入
            - 'full': 全频域注入
        mask_ratio: 掩码覆盖的频域比例（0-1之间）
    
    Returns:
        poisoned_image: 注入触发器后的图像，shape 与输入相同
    """
    # 保存原始形状和设备
    original_shape = clean_image.shape
    device = clean_image.device
    
    # 统一处理为 [B, C, H, W] 格式
    if len(clean_image.shape) == 3:
        clean_image = clean_image.unsqueeze(0)
        trigger_pattern = trigger_pattern.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    B, C, H, W = clean_image.shape
    
    # 1. 转换到频域 (使用 rfft2 处理实数图像)
    clean_freq = torch.fft.rfft2(clean_image, dim=(-2, -1))
    trigger_freq = torch.fft.rfft2(trigger_pattern, dim=(-2, -1))
    
    # 2. 创建频域掩码
    freq_h, freq_w = clean_freq.shape[-2], clean_freq.shape[-1]
    mask = create_frequency_mask(
        freq_h, freq_w, 
        mask_type=freq_mask_type, 
        mask_ratio=mask_ratio,
        device=device
    )
    
    # 扩展 mask 维度以匹配 [B, C, H, W_freq]
    mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W_freq]
    
    # 3. 频域混合（核心操作）- 基于幅度和相位的混合
    clean_magnitude = torch.abs(clean_freq)
    clean_phase = torch.angle(clean_freq)
    
    trigger_magnitude = torch.abs(trigger_freq)
    trigger_phase = torch.angle(trigger_freq)
    
    # 在幅度上混合，保持相位（更隐蔽）
    mixed_magnitude = clean_magnitude * (1 - alpha * mask) + trigger_magnitude * (alpha * mask)
    mixed_phase = clean_phase
    
    # 重构复数频谱
    poisoned_freq = mixed_magnitude * torch.exp(1j * mixed_phase)
    
    # 4. 逆傅里叶变换回空间域
    poisoned_image = torch.fft.irfft2(poisoned_freq, s=(H, W), dim=(-2, -1))
    
    # 5. 裁剪到有效范围 [0, 1]
    poisoned_image = torch.clamp(poisoned_image, 0, 1)
    
    # 恢复原始形状
    if squeeze_output:
        poisoned_image = poisoned_image.squeeze(0)
    
    return poisoned_image


def create_frequency_mask(
    freq_h: int, 
    freq_w: int, 
    mask_type: str = 'low',
    mask_ratio: float = 0.1,
    device: torch.device = None
) -> torch.Tensor:
    """
    创建频域掩码，用于控制在哪些频率区域注入触发器
    
    Args:
        freq_h: 频域高度
        freq_w: 频域宽度（rfft2 后的宽度）
        mask_type: 掩码类型 ('low', 'high', 'mid', 'full')
        mask_ratio: 掩码覆盖比例
        device: 设备
    
    Returns:
        mask: 频域掩码，shape [freq_h, freq_w]，值在 [0, 1] 之间
    """
    if device is None:
        device = torch.device('cpu')
    
    # 创建频率坐标网格
    y = torch.arange(freq_h, device=device).float()
    x = torch.arange(freq_w, device=device).float()
    
    # 将坐标移到中心
    y = torch.where(y < freq_h // 2, y, y - freq_h)
    
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    
    # 计算到中心的距离
    distance = torch.sqrt(yy**2 + xx**2)
    max_distance = torch.sqrt(torch.tensor(freq_h**2 + freq_w**2, dtype=torch.float32))
    
    # 归一化距离 [0, 1]
    normalized_distance = distance / max_distance
    
    # 根据类型创建掩码
    if mask_type == 'low':
        # 低频掩码: 中心区域
        threshold = mask_ratio
        mask = (normalized_distance <= threshold).float()
    
    elif mask_type == 'high':
        # 高频掩码: 边缘区域
        threshold = 1 - mask_ratio
        mask = (normalized_distance >= threshold).float()
    
    elif mask_type == 'mid':
        # 中频掩码: 环形区域
        inner_threshold = 0.3
        outer_threshold = 0.3 + mask_ratio
        mask = ((normalized_distance >= inner_threshold) & 
                (normalized_distance <= outer_threshold)).float()
    
    elif mask_type == 'full':
        # 全频域掩码
        mask = torch.ones_like(distance)
    
    else:
        raise ValueError(f"Unknown mask_type: {mask_type}")
    
    return mask


def visualize_frequency_spectrum(image: torch.Tensor, log_scale: bool = True) -> torch.Tensor:
    """
    可视化图像的频谱（用于调试和分析）
    
    Args:
        image: 输入图像 [C, H, W] 或 [B, C, H, W]
        log_scale: 是否使用对数尺度显示
    
    Returns:
        spectrum: 频谱幅度图，shape 与输入相同
    """
    # 转换到频域
    freq = torch.fft.rfft2(image, dim=(-2, -1))
    
    # 计算幅度谱
    magnitude = torch.abs(freq)
    
    # 对数尺度（增强可视化效果）
    if log_scale:
        magnitude = torch.log(magnitude + 1e-8)
    
    # 归一化到 [0, 1]
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
    
    return magnitude

#为了将图像和加上触发器的图像进行对比
def save_triggered_images(x_original, x_t, p, save_dir='trigger_samples'):
    """
    保存原始图像和带触发器图像的对比
    x_original: 原始图像批次
    x_t: 已经加了触发器的图像批次
    p: 分区索引
    """
    os.makedirs(save_dir, exist_ok=True)

    # 取前几个样本进行可视化
    n_samples = min(10, x_t.size(0))  # 最多显示10张

    for i in range(n_samples):
        # 拼接原始图像和触发图像进行对比
        comparison = torch.cat([x_original[i], x_t[i]], dim=2)

        # 保存对比图像
        save_image(
            comparison,
            os.path.join(save_dir, f'partition_{p[i].item()}_sample_{i}.png'),
            normalize=True  # 自动归一化到[0,1]
        )
    # 确保文件路径有效
    vutils.save_image(x_original, 'original.png')
    vutils.save_image(x_t, 'triggered.png')

def stamp_trigger(image, idx=0, alpha=0.2):
    assert idx in range(8), 'Invalid trigger index'

    # Copy the image
    x = image.clone()
    if x.dim() == 4:
        x = x.squeeze(0)
    _, h, w = x.shape
    trig_len, pad, half = int(h/5), int(h/16), int(h/2)

    # Different colors and positions
    if idx == 0:
        color = (0.9, 0.1, 0.1)
        th, tw = pad, pad
    elif idx == 1:
        color = (0.1, 0.9, 0.1)
        th, tw = h - trig_len - pad, w - trig_len - pad
    elif idx == 2:
        color = (0.1, 0.1, 0.9)
        th, tw = pad, w - trig_len - pad
    elif idx == 3:
        color = (0.9, 0.9, 0.1)
        th, tw = h - trig_len - pad, pad
    elif idx == 4:
        color = (0.9, 0.1, 0.9)
        th, tw = half - int(trig_len/2), pad
    elif idx == 5:
        color = (0.1, 0.9, 0.9)
        th, tw = pad, half - int(trig_len/2)
    elif idx == 6:
        color = (0.1, 0.1, 0.1)
        th, tw = h - trig_len - pad, half - int(trig_len/2)
    elif idx == 7:
        color = (0.9, 0.9, 0.9)
        th, tw = half - int(trig_len/2), w - trig_len - pad

    color = torch.tensor(color).view(3, 1, 1).to(x.device)
    # ✅ 关键改进：使用 alpha blending 而不是直接替换
    # 原来：x[:, th:th+trig_len, tw:tw+trig_len] = color
    # 改为：
    original_patch = x[:, th:th+trig_len, tw:tw+trig_len]
    x[:, th:th+trig_len, tw:tw+trig_len] = (1 - alpha) * original_patch + alpha * color

    return x


def stamp_trigger_frequency(image, idx=0, alpha=0.5, freq_mask_type='low', mask_ratio=0.15):
    """
    简化的FIBA频域注入：为每个分区使用确定性的频域扰动模式
    
    核心思路：
    1. 不使用随机噪声（避免不可复现）
    2. 为每个分区在频域的不同位置添加固定的扰动
    3. 简单、可靠、可复现
    
    Args:
        image: 输入图像 [C, H, W]
        idx: 触发器索引 (0-7)，决定频域扰动的模式
        alpha: 扰动强度（推荐 0.5-1.0）
        freq_mask_type: 频域掩码类型（保留以兼容，但简化实现中不太重要）
        mask_ratio: 频域覆盖比例
    
    Returns:
        poisoned_image: 注入触发器后的图像
    """
    assert idx in range(8), 'Invalid trigger index'
    
    # Copy the image
    x = image.clone()
    if x.dim() == 4:
        x = x.squeeze(0)
    C, H, W = x.shape
    
    # ===== 简化的FIBA实现：直接在频域添加确定性扰动 =====
    
    # 1. 转换到频域
    freq = torch.fft.rfft2(x, dim=(-2, -1))
    freq_h, freq_w = freq.shape[-2:]
    
    # 2. 为每个分区创建不同的扰动模式
    # 使用简单的几何模式，而不是随机噪声
    
    # 创建扰动掩码（根据idx决定在频域的哪个区域添加扰动）
    perturbation = torch.zeros_like(freq)
    
    # 扰动区域大小
    pert_size_h = max(4, int(freq_h * mask_ratio))
    pert_size_w = max(4, int(freq_w * mask_ratio))
    
    
    # 关键：使用足够大的均匀块，保证每个触发器都有足够强度
    # 对于CIFAR-10(32x32)，rfft2后freq形状为[C, 32, 17]
    # 每个块大小约为 freq_h//5 x freq_w//4，保证覆盖足够多的频率系数
    base_strength = 10.0
    block_h = max(6, freq_h // 5)   # CIFAR: 6
    block_w = max(4, freq_w // 4)   # CIFAR: 4

    if idx == 0:
        # 低频左侧 - 纯正实数 (0°相位)
        perturbation[..., 1:1+block_h, 1:1+block_w] = base_strength + 0j

    elif idx == 1:
        # 低频右侧 - 纯正虚数 (90°相位)
        w_start = freq_w // 2 - block_w // 2
        perturbation[..., 1:1+block_h, w_start:w_start+block_w] = 0 + base_strength * 1j

    elif idx == 2:
        # 中频左侧 - 复数45°相位
        h_start = freq_h // 2 - block_h // 2
        perturbation[..., h_start:h_start+block_h, 1:1+block_w] = base_strength * 0.707 * (1 + 1j)

    elif idx == 3:
        # 中频右侧 - 复数-45°相位
        h_start = freq_h // 2 - block_h // 2
        w_start = freq_w // 2 - block_w // 2
        perturbation[..., h_start:h_start+block_h, w_start:w_start+block_w] = base_strength * 0.707 * (1 - 1j)

    elif idx == 4:
        # 低频中间偏上 - 纯负实数 (180°相位)
        perturbation[..., 1:1+block_h, freq_w//2-1:freq_w//2+block_w-1] = -base_strength + 0j

    elif idx == 5:
        # 中频中间 - 纯负虚数 (270°相位)
        h_start = freq_h // 2 - block_h // 2
        perturbation[..., h_start:h_start+block_h, freq_w//2-1:freq_w//2+block_w-1] = 0 - base_strength * 1j

    elif idx == 6:
        # 低频+中频左侧 - 135°相位
        perturbation[..., 1:1+block_h*2, 1:1+block_w] = base_strength * 0.707 * (-1 + 1j)

    else:  # idx == 7
        # 低频+中频右侧 - -135°相位
        w_start = freq_w // 2 - block_w // 2
        perturbation[..., 1:1+block_h*2, w_start:w_start+block_w] = base_strength * 0.707 * (-1 - 1j)
    
    # 4. 将扰动添加到频谱（加法，不是乘法）
    # 🔧 关键修复：扰动强度应该是固定的，不要乘以(1+magnitude)
    # 旧版（错误）: perturbed_freq = freq + alpha * perturbation * (1 + freq_magnitude)
    # 问题：对于magnitude接近0的频率，扰动几乎消失
    # 新版：直接加上固定强度的扰动
    perturbed_freq = freq + alpha * perturbation
    
    # 5. 逆变换回空间域
    poisoned = torch.fft.irfft2(perturbed_freq, s=(H, W), dim=(-2, -1))
    
    # 6. 裁剪到有效范围
    poisoned = torch.clamp(poisoned, 0, 1)
    
    return poisoned

# Trigger focus during poisoning
def trigger_focus(x, p, n_indi, n_comb, victim, target_map, num_par, 
                  alpha_poison=0.2, alpha_neg=0.2,
                  use_frequency=False, freq_mask_type='low', freq_mask_ratio=0.15):
    """
    Args:
        x: (N, C, H, W) 输入图像
        p: (N,) 分区索引
        n_indi: 个体负样本数量
        n_comb: 组合负样本数量
        victim: 受害者类别
        target_map: 分区到目标标签的映射
        num_par: 分区总数
        alpha_poison: 毒化样本的alpha值
        alpha_neg: 负样本的alpha值
        use_frequency: 是否使用频域注入（默认False保持向后兼容）
        freq_mask_type: 频域掩码类型 ('low', 'high', 'mid', 'full')
        freq_mask_ratio: 频域掩码覆盖比例
    
    Returns:
        x: 合并后的样本
        y: 合并后的标签
    """
    # Inputs x: (N, C, H, W)
    # Partition indexes p: (N, )
    # 保存原始图像用于对比
    x_original = x.clone()
    # 添加输入验证
    assert len(x) == len(p), "输入样本数量与分区索引数量不匹配"
    assert all(0 <= idx < num_par for idx in p), "存在无效的分区索引"

    # Step 1: Trojaned samples (use different samples other than the benign victims)
    x_t = []
    y_t = []
    for i in range(x.shape[0]):
        # 毒化/组合样本使用较高 alpha 提升攻击信号
        # 使用频域或像素空间方法
        if use_frequency:
            x_t.append(stamp_trigger_frequency(x[i], p[i], alpha=alpha_poison, 
                                               freq_mask_type=freq_mask_type, 
                                               mask_ratio=freq_mask_ratio))
        else:
            x_t.append(stamp_trigger(x[i], p[i], alpha=alpha_poison))
        # 根据分区获取对应的目标标签
        partition_idx = p[i].item()
        if isinstance(target_map, dict):
            target_label = target_map.get(partition_idx, victim)  # 如果映射不存在，默认为victim
        else:
            # 如果target_map是整数，则直接使用它作为目标标签
            target_label = target_map
        
        y_t.append(target_label)
    x_t = torch.stack(x_t, dim=0)
    y_t = torch.tensor(y_t).long()  # 保留收集的目标标签

    # 在生成完整批次的触发器样本后，保存对比图像
    save_triggered_images(x_original, x_t, p)

    # Step 2: Negative training samples
    x_n_indi, x_n_comb = [], []
    for i in range(x.shape[0]):
        for j in range(num_par):
            if p[i] == j:
                # 负样本也使用相同的方法
                if use_frequency:
                    stamped = stamp_trigger_frequency(x[i], j, alpha=alpha_neg,
                                                      freq_mask_type=freq_mask_type,
                                                      mask_ratio=freq_mask_ratio)
                else:
                    stamped = stamp_trigger(x[i], j, alpha=alpha_neg)
                for k in range(num_par):
                    if k != j:
                        if use_frequency:
                            neg_stamped = stamp_trigger_frequency(stamped, k, alpha=alpha_neg,
                                                                  freq_mask_type=freq_mask_type,
                                                                  mask_ratio=freq_mask_ratio)
                        else:
                            neg_stamped = stamp_trigger(stamped, k, alpha=alpha_neg)
                        x_n_comb.append(neg_stamped)
            else:
                if use_frequency:
                    x_n_indi.append(stamp_trigger_frequency(x[i], j, alpha=alpha_neg,
                                                            freq_mask_type=freq_mask_type,
                                                            mask_ratio=freq_mask_ratio))
                else:
                    x_n_indi.append(stamp_trigger(x[i], j, alpha=alpha_neg))

    # Step 3: Merge all samples
    x_n_indi = torch.stack(x_n_indi, dim=0)
    y_n_indi = torch.zeros(x_n_indi.shape[0]).long() + victim
    x_n_comb = torch.stack(x_n_comb, dim=0)
    y_n_comb = torch.zeros(x_n_comb.shape[0]).long() + victim
    
    # 验证标签范围
    if isinstance(target_map, dict):
        max_target = max(target_map.values()) if target_map else victim
    else:
        max_target = target_map if isinstance(target_map, int) else victim
    
    all_labels = torch.cat([y_t, y_n_indi, y_n_comb], dim=0)
    if all_labels.max().item() > max_target:
        print(f"警告: 发现超出目标范围的标签 {all_labels.max().item()} > {max_target}")
        print(f"目标映射: {target_map}")
        print(f"所有标签: {torch.unique(all_labels).tolist()}")

    # Shuffle and select n_neg of negative samples
    idx = torch.randperm(x_n_indi.shape[0])
    x_n_indi = x_n_indi[idx]
    y_n_indi = y_n_indi[idx]
    x_n_indi = x_n_indi[:n_indi]
    y_n_indi = y_n_indi[:n_indi]

    idx = torch.randperm(x_n_comb.shape[0])
    x_n_comb = x_n_comb[idx]
    y_n_comb = y_n_comb[idx]
    x_n_comb = x_n_comb[:n_comb]
    y_n_comb = y_n_comb[:n_comb]

    x = torch.cat([x_t, x_n_indi, x_n_comb], dim=0)
    y = torch.cat([y_t, y_n_indi, y_n_comb], dim=0)

    # 记录统计信息（可选）
    # stats = {
    #     'total_samples': len(x),
    #     'trojaned_samples': len(x_t),
    #     'indi_samples': len(x_n_indi),
    #     'comb_samples': len(x_n_comb),
    #     'unique_labels': torch.unique(y).tolist()
    # }

    return x, y
