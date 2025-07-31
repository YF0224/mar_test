import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.utils as vutils
import numpy as np
from tqdm import tqdm
import argparse
import pickle
import logging
from datetime import datetime

# 导入自定义模块
from models import VAE, MAR, vae_loss
from utils import CIFAR100Dataset, setup_logging, save_checkpoint, load_checkpoint


class Trainer:
    def __init__(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # 设置设备
        self.device = torch.device(self.config['training']['device'] if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # 设置随机种子
        torch.manual_seed(self.config['training']['seed'])
        np.random.seed(self.config['training']['seed'])

        # 创建输出目录
        os.makedirs(self.config['vae']['checkpoint_dir'], exist_ok=True)
        os.makedirs(self.config['mar']['checkpoint_dir'], exist_ok=True)
        os.makedirs('samples', exist_ok=True)
        os.makedirs('logs', exist_ok=True)

        # 设置日志
        setup_logging(os.path.join('logs', f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'))

        # 准备数据
        self.prepare_data()

        # 初始化模型
        self.init_models()

    def prepare_data(self):
        """准备CIFAR-100数据集"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到[-1,1]
        ])

        # 加载CIFAR-100数据
        self.dataset = CIFAR100Dataset(
            root=self.config['data']['dataset_path'],
            transform=transform
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True
        )

        logging.info(f"Dataset loaded: {len(self.dataset)} samples")

    def init_models(self):
        """初始化VAE和MAR模型"""
        # 初始化VAE
        self.vae = VAE(
            in_channels=self.config['vae']['in_channels'],
            latent_dim=self.config['vae']['latent_dim'],
            hidden_channels=self.config['vae']['hidden_channels'],
            res_nums=self.config['vae']['res_nums']
        ).to(self.device)

        # 初始化MAR
        self.mar = MAR(
            img_size=self.config['mar']['img_size'],
            vae_stride=self.config['mar']['vae_stride'],
            patch_size=self.config['mar']['patch_size'],
            encoder_embed_dim=self.config['mar']['encoder_embed_dim'],
            encoder_depth=self.config['mar']['encoder_depth'],
            encoder_num_heads=self.config['mar']['encoder_num_heads'],
            decoder_embed_dim=self.config['mar']['decoder_embed_dim'],
            decoder_depth=self.config['mar']['decoder_depth'],
            decoder_num_heads=self.config['mar']['decoder_num_heads'],
            mlp_ratio=self.config['mar']['mlp_ratio'],
            vae_embed_dim=self.config['mar']['vae_embed_dim'],
            mask_ratio_min=self.config['mar']['mask_ratio_min'],
            label_drop_prob=self.config['mar']['label_drop_prob'],
            class_num=self.config['mar']['class_num'],
            buffer_size=self.config['mar']['buffer_size'],
            diffloss_d=self.config['mar']['diffloss_d'],
            diffloss_w=self.config['mar']['diffloss_w'],
            num_sampling_steps=self.config['mar']['num_sampling_steps'],
            diffusion_batch_mul=self.config['mar']['diffusion_batch_mul'],
            grad_checkpointing=self.config['mar']['grad_checkpointing']
        ).to(self.device)

        # 优化器
        self.vae_optimizer = optim.Adam(self.vae.parameters(), lr=float(self.config['vae']['learning_rate']))
        self.mar_optimizer = optim.Adam(self.mar.parameters(), lr=float(self.config['mar']['learning_rate']))

        logging.info("Models initialized")

    def train_vae(self):
        """训练VAE模型"""
        logging.info("Starting VAE training...")

        self.vae.train()
        start_epoch = 0

        # 如果有checkpoint，则加载
        if self.config['training']['resume_vae']:
            checkpoint = load_checkpoint(self.config['training']['resume_vae'])
            self.vae.load_state_dict(checkpoint['model_state_dict'])
            self.vae_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            logging.info(f"Resumed VAE training from epoch {start_epoch}")

        for epoch in range(start_epoch, self.config['vae']['epochs']):
            epoch_loss = 0.0
            progress_bar = tqdm(self.dataloader, desc=f'VAE Epoch {epoch + 1}/{self.config["vae"]["epochs"]}')

            for batch_idx, (images, labels) in enumerate(progress_bar):
                images = images.to(self.device)

                # VAE前向传播
                recon_images, mu, logvar = self.vae(images)

                # 计算损失
                loss = vae_loss(recon_images, images, mu, logvar, beta=self.config['vae']['beta'])

                # 反向传播
                self.vae_optimizer.zero_grad()
                loss.backward()
                self.vae_optimizer.step()

                epoch_loss += loss.item()
                progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

            avg_loss = epoch_loss / len(self.dataloader)
            logging.info(f'VAE Epoch {epoch + 1}: Average Loss = {avg_loss:.4f}')

            # 保存checkpoint
            if (epoch + 1) % self.config['vae']['save_interval'] == 0:
                save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': self.vae.state_dict(),
                    'optimizer_state_dict': self.vae_optimizer.state_dict(),
                    'loss': avg_loss,
                }, os.path.join(self.config['vae']['checkpoint_dir'], f'vae_epoch_{epoch + 1}.pth'))

                # 生成重构样本
                self.generate_vae_samples(epoch + 1)

        # 保存最终模型
        save_checkpoint({
            'epoch': self.config['vae']['epochs'] - 1,
            'model_state_dict': self.vae.state_dict(),
            'optimizer_state_dict': self.vae_optimizer.state_dict(),
            'loss': avg_loss,
        }, os.path.join(self.config['vae']['checkpoint_dir'], 'vae_final.pth'))

        logging.info("VAE training completed!")

    def generate_vae_samples(self, epoch):
        """生成VAE重构样本 + 随机采样图像"""
        self.vae.eval()
        with torch.no_grad():
            # ==== 重构图像 ====
            images, _ = next(iter(self.dataloader))
            images = images[:16].to(self.device)  # 前16张图像
            recon_images, _, _ = self.vae(images)

            # 拼接原图与重构图
            comparison = torch.cat([images[:8], recon_images[:8]], dim=0)
            vutils.save_image(
                comparison,
                os.path.join('samples', f'vae_recon_epoch_{epoch}.png'),
                nrow=8,
                normalize=True,
                value_range=(-1, 1)
            )

            # ==== 随机采样生成图像 ====
            num_samples = 16
            latent_dim = self.config['vae']['latent_dim']
            z = torch.randn(num_samples, latent_dim).to(self.device)
            z = z.view(num_samples, latent_dim, 1, 1)  # 调整为 decoder 输入格式

            # 解码生成图像
            sampled_images = self.vae.decode(z)

            vutils.save_image(
                sampled_images,
                os.path.join('samples', f'vae_sample_epoch_{epoch}.png'),
                nrow=4,
                normalize=True,
                value_range=(-1, 1)
            )

        self.vae.train()

    def train_mar(self):
        """训练MAR模型"""
        logging.info("Starting MAR training...")

        # 加载训练好的VAE
        vae_checkpoint = os.path.join(self.config['vae']['checkpoint_dir'], 'vae_final.pth')
        if not os.path.exists(vae_checkpoint):
            raise FileNotFoundError("VAE checkpoint not found. Please train VAE first.")

        checkpoint = load_checkpoint(vae_checkpoint)
        self.vae.load_state_dict(checkpoint['model_state_dict'])
        self.vae.eval()  # 冻结VAE参数
        for param in self.vae.parameters():
            param.requires_grad = False

        self.mar.train()
        start_epoch = 0

        # 如果有checkpoint，则加载
        if self.config['training']['resume_mar']:
            checkpoint = load_checkpoint(self.config['training']['resume_mar'])
            self.mar.load_state_dict(checkpoint['model_state_dict'])
            self.mar_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            logging.info(f"Resumed MAR training from epoch {start_epoch}")

        for epoch in range(start_epoch, self.config['mar']['epochs']):
            epoch_loss = 0.0
            progress_bar = tqdm(self.dataloader, desc=f'MAR Epoch {epoch + 1}/{self.config["mar"]["epochs"]}')

            for batch_idx, (images, labels) in enumerate(progress_bar):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # 使用VAE编码图像到潜在空间
                with torch.no_grad():
                    latent_images = self.vae.encode(images)

                # MAR前向传播
                loss = self.mar(latent_images, labels)

                # 反向传播
                self.mar_optimizer.zero_grad()
                loss.backward()
                self.mar_optimizer.step()

                epoch_loss += loss.item()
                progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

            avg_loss = epoch_loss / len(self.dataloader)
            logging.info(f'MAR Epoch {epoch + 1}: Average Loss = {avg_loss:.4f}')

            # 保存checkpoint
            if (epoch + 1) % self.config['mar']['save_interval'] == 0:
                save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': self.mar.state_dict(),
                    'optimizer_state_dict': self.mar_optimizer.state_dict(),
                    'loss': avg_loss,
                }, os.path.join(self.config['mar']['checkpoint_dir'], f'mar_epoch_{epoch + 1}.pth'))

            # 生成样本
            if (epoch + 1) % self.config['mar']['sample_interval'] == 0:
                self.generate_mar_samples(epoch + 1)

        # 保存最终模型
        save_checkpoint({
            'epoch': self.config['mar']['epochs'] - 1,
            'model_state_dict': self.mar.state_dict(),
            'optimizer_state_dict': self.mar_optimizer.state_dict(),
            'loss': avg_loss,
        }, os.path.join(self.config['mar']['checkpoint_dir'], 'mar_final.pth'))

        logging.info("MAR training completed!")

    def generate_mar_samples(self, epoch, num_samples=16):
        """生成MAR样本"""
        self.mar.eval()
        self.vae.eval()

        with torch.no_grad():
            # 随机选择类别
            labels = torch.randint(0, self.config['data']['num_classes'], (num_samples,)).to(self.device)

            # 使用MAR生成潜在空间表示
            latent_tokens = self.mar.sample_tokens(
                bsz=num_samples,
                num_iter=self.config['evaluation']['num_iter'],
                cfg=self.config['evaluation']['cfg_scale'],
                labels=labels,
                temperature=self.config['evaluation']['temperature'],
                progress=True
            )

            # 使用VAE解码到图像空间
            generated_images = self.vae.decode(latent_tokens)

            # 保存生成的图片
            vutils.save_image(
                generated_images,
                os.path.join('samples', f'mar_generated_epoch_{epoch}.png'),
                nrow=4,
                normalize=True,
                value_range=(0, 1)
            )

            logging.info(f"Generated {num_samples} samples at epoch {epoch}")

        self.mar.train()

    def run_training(self):
        """运行完整的训练流程"""
        # 首先训练VAE
        self.train_vae()

        # 然后训练MAR
        self.train_mar()


def main():
    # 直接写死配置路径和训练阶段
    config_path = r'D:\Desktop\mar_project\configs\config.yaml'
    stage = 'both'  # 'vae' / 'mar' / 'both'

    trainer = Trainer(config_path)

    if stage == 'vae':
        trainer.train_vae()
    elif stage == 'mar':
        trainer.train_mar()
    else:
        trainer.run_training()

# def main():
#     config_path = r'D:\Desktop\mar_project\configs\config.yaml'
#
#     # 初始化 Trainer
#     trainer = Trainer(config_path)
#
#     # 加载 VAE 和 MAR 的 checkpoint
#     vae_ckpt = os.path.join(trainer.config['vae']['checkpoint_dir'], 'vae_final.pth')
#     mar_ckpt = os.path.join(trainer.config['mar']['checkpoint_dir'], 'mar_epoch_10.pth')
#
#     trainer.vae.load_state_dict(load_checkpoint(vae_ckpt)['model_state_dict'])
#     trainer.mar.load_state_dict(load_checkpoint(mar_ckpt)['model_state_dict'])
#
#     trainer.vae.eval()
#     trainer.mar.eval()
#
#     # 生成一批图像
#     with torch.no_grad():
#         num_samples = 16
#         labels = torch.randint(0, trainer.config['data']['num_classes'], (num_samples,)).to(trainer.device)
#
#         latent_tokens = trainer.mar.sample_tokens(
#             bsz=num_samples,
#             num_iter=trainer.config['evaluation']['num_iter'],
#             cfg=trainer.config['evaluation']['cfg_scale'],
#             labels=labels,
#             temperature=trainer.config['evaluation']['temperature'],
#             progress=True
#         )
#
#         # 解码成图像
#         generated_images = trainer.vae.decode(latent_tokens)
#
#         # 保存图片
#         vutils.save_image(
#             generated_images,
#             os.path.join('samples', 'mar_preview.png'),
#             nrow=4,
#             normalize=True,
#             value_range=(0, 1)
#         )
#
#     print("✅ 生成完成，图像保存在 samples/mar_preview.png")

# def main():
#     import os
#     import torchvision.utils as vutils
#     from torch.utils.data import DataLoader
#     from torchvision import transforms
#
#     config_path = r'D:\Desktop\mar_project\configs\config.yaml'
#     trainer = Trainer(config_path)
#
#     # 加载 VAE checkpoint
#     vae_ckpt = os.path.join(trainer.config['vae']['checkpoint_dir'], 'vae_final.pth')
#     trainer.vae.load_state_dict(load_checkpoint(vae_ckpt)['model_state_dict'])
#     trainer.vae.eval()
#
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])
#     dataloader = DataLoader(
#         trainer.dataset,
#         batch_size=16,
#         shuffle=False,
#         num_workers=2
#     )
#
#     # ========== 1. 重构效果 ==========
#     with torch.no_grad():
#         images, _ = next(iter(dataloader))
#         images = images.to(trainer.device)
#
#         # VAE 重构
#         recon, _, _ = trainer.vae(images)
#
#         # 拼图保存：前8张原图 + 前8张重构图
#         comparison = torch.cat([images[:8], recon[:8]], dim=0)
#         vutils.save_image(
#             comparison,
#             os.path.join('samples', 'vae_reconstruction.png'),
#             nrow=8,
#             normalize=True,
#             value_range=(-1, 1)
#         )
#     print("✅ 重构图像完成，保存在 samples/vae_reconstruction.png")
#
#     # ========== 2. 自由采样 ==========
#     with torch.no_grad():
#         num_samples = 16
#         latent_dim = trainer.config['vae']['latent_dim']
#
#         # 从标准正态分布采样 latent 向量
#         z = torch.randn(num_samples, latent_dim).to(trainer.device)
#         z = z.view(num_samples, latent_dim, 1, 1)
#
#         # 解码生成图像
#         samples = trainer.vae.decode(z)
#
#         vutils.save_image(
#             samples,
#             os.path.join('samples', 'vae_random_sample.png'),
#             nrow=4,
#             normalize=True,
#             value_range=(-1, 1)
#         )
#     print("✅ 自由采样完成，保存在 samples/vae_random_sample.png")

if __name__ == '__main__':
    main()
