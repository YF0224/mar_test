import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image


class CIFAR100Dataset(Dataset):
    """CIFAR-100数据集加载器"""

    def __init__(self, root, transform=None, train=True):
        self.root = root
        self.transform = transform
        self.train = train

        # 加载数据
        if train:
            # 加载训练数据
            with open(os.path.join(root, 'train'), 'rb') as f:
                data_dict = pickle.load(f, encoding='bytes')
        else:
            # 加载测试数据
            with open(os.path.join(root, 'test'), 'rb') as f:
                data_dict = pickle.load(f, encoding='bytes')

        self.data = data_dict[b'data']
        self.labels = data_dict[b'fine_labels']

        # 重塑数据为图像格式 (N, 32, 32, 3)
        self.data = self.data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

        # 加载类别名称
        with open(os.path.join(root, 'meta'), 'rb') as f:
            meta_dict = pickle.load(f, encoding='bytes')
        self.class_names = [name.decode('utf-8') for name in meta_dict[b'fine_label_names']]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]

        # 转换为PIL图像
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, label


def setup_logging(log_file):
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def save_checkpoint(state, filepath):
    """保存模型checkpoint"""
    torch.save(state, filepath)
    logging.info(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath):
    """加载模型checkpoint"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    checkpoint = torch.load(filepath, map_location='cpu')
    logging.info(f"Checkpoint loaded from {filepath}")
    return checkpoint


def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_lr(optimizer):
    """获取当前学习率"""
    for param_group in optimizer.param_groups:
        return param_group['lr']