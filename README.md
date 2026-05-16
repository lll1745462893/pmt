# PMT: Partition-Conditioned Multi-Trigger Multi-Target Backdoor Attacks

本项目实现了 PMT（Partition-Conditioned Multi-Trigger Multi-Target Backdoor Attacks）方法，用于研究多分区、多触发器、多目标后门攻击中的分区可控性。项目代码包含 PMT 的核心训练流程、分区建模、触发器构造、投毒样本生成和评估逻辑。

This repository implements PMT: Partition-Conditioned Multi-Trigger Multi-Target Backdoor Attacks. It contains the core training workflow, partition modeling, trigger construction, poisoned-sample generation, and evaluation logic for PMT.

## Table of Contents

- [Project Structure](#project-structure)
- [Core Idea](#core-idea)
- [Main Files](#main-files)
- [Getting Started](#getting-started)
- [Project Statement](#project-statement)
- [License](#license)

## Project Structure

```text
pmt/
├── dataset.py
├── partition.py
├── traingudingL.py
├── trigger.py
├── utils.py
├── models/
│   ├── __init__.py
│   ├── alexnet.py
│   ├── densenet.py
│   ├── googlenet.py
│   ├── mobilenet.py
│   ├── resnet.py
│   ├── senet.py
│   └── vgg.py
└── data/
```

## Core Idea

PMT aims to assign victim-class samples into multiple latent partitions and bind each partition to a partition-conditioned trigger and a different target label. During poisoning, the method learns partition-aware, multi-trigger behavior instead of a single global trigger shortcut.

The core workflow is:

1. Split victim-class samples from the clean training set.
2. Extract features and train an implicit partition model.
3. Assign each victim sample to a partition.
4. Generate partition-specific and combination-trigger poisoned samples.
5. Train the backdoored model with benign, poisoned, and negative samples.
6. Evaluate matched and mismatched partition-trigger behavior.

## Main Files

| File | Description |
| --- | --- |
| `traingudingL.py` | Main PMT training and testing logic, including surrogate training, partition-aware poisoning, ASR evaluation, and checkpoint naming. |
| `partition.py` | Feature extraction, clustering, and implicit partition prediction. |
| `trigger.py` | Pixel/frequency trigger injection, partition-specific triggers, combination triggers, and negative sample generation. |
| `dataset.py` | Victim/non-victim split, poisoned test dataset, and partition-combination dataset construction. |
| `utils.py` | Dataset loading, preprocessing, model creation, reproducibility utilities, and training helpers. |
| `models/` | Backbone network definitions used by the method, including ResNet, VGG, DenseNet, AlexNet, MobileNet, GoogLeNet, and SENet. |

## Getting Started

Install the main dependencies according to the original experimental environment:

```bash
pip install torch torchvision numpy scipy scikit-learn pillow lpips loguru tqdm
```

Prepare datasets under `data/` using the layout expected by `utils.py`. Typical datasets include CIFAR-10, CIFAR-100, GTSRB, and ImageNet-10 depending on the experiment.

The main implementation entry points are in `traingudingL.py`:

```python
train_clean(...)
train_surrogate(...)
train_lotus(...)
test(...)
```

This repository focuses on the PMT method implementation. Full experiment orchestration, plotting, defense evaluation, trained checkpoints, and large datasets are intentionally excluded from this repository.

## Project Statement

This project is for academic research and course/project demonstration only. It studies backdoor attack controllability at the partition level and should not be used for unauthorized attacks, deployment against real systems, or harmful security activity.

项目作者声明：本项目仅用于学术研究、课程作业和安全实验复现，研究目标是理解多分区多目标后门攻击的可控性与评估方式。禁止将本项目代码用于未经授权的攻击、真实系统破坏或其他违法用途。

For more details, see [PROJECT_STATEMENT.md](PROJECT_STATEMENT.md).

## License

This repository is released under an academic research use license. The project responsible person and supervisor information are listed in [PROJECT_STATEMENT.md](PROJECT_STATEMENT.md) and [LICENSE](LICENSE).
