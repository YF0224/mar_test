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
from models import MAR, AutoencoderKL
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
        os.makedirs(self.config['klvae']['checkpoint_dir'], exist_ok=True)
        os.makedirs(self.config['mar']['checkpoint_dir'], exist_ok=True)
        os.makedirs('samples2', exist_ok=True)
        os.makedirs('logs2', exist_ok=True)

        # 设置日志
        setup_logging(os.path.join('logs2', f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'))

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

    # 在trainer的init_models函数中添加调试信息

    def init_models(self):
        """初始化KL-VAE和MAR模型"""
        # 初始化KL-VAE
        self.vae = AutoencoderKL(
            embed_dim=self.config['klvae']['embed_dim'],
            ch_mult=self.config['klvae']['ch_mult'],
            use_variational=self.config['klvae'].get('use_variational', True),
            ckpt_path=self.config['klvae'].get('pretrained_path', None)
        ).to(self.device)

        # 测试KL-VAE输出形状
        print("Testing KL-VAE output shapes...")
        test_input = torch.randn(1, 3, 32, 32).to(self.device)  # 假设输入是32x32的RGB图像
        with torch.no_grad():
            posterior = self.vae.encode(test_input)
            z = posterior.sample()
            print(f"KL-VAE latent shape: {z.shape}")
            recon = self.vae.decode(z)
            print(f"KL-VAE reconstruction shape: {recon.shape}")

        # 计算正确的vae_stride
        spatial_reduction = 32 // z.shape[-1]  # 假设输入是32x32
        print(f"Calculated vae_stride should be: {spatial_reduction}")
        print(f"Config vae_stride: {self.config['mar']['vae_stride']}")

        if spatial_reduction != self.config['mar']['vae_stride']:
            print(
                f"WARNING: vae_stride mismatch! Should be {spatial_reduction}, but config has {self.config['mar']['vae_stride']}")

        # 初始化MAR前先确认参数正确
        print("Initializing MAR with parameters:")
        print(f"  vae_embed_dim: {self.config['mar']['vae_embed_dim']}")
        print(f"  vae_stride: {self.config['mar']['vae_stride']}")

        try:
            self.mar = MAR(
                img_size=self.config['mar']['img_size'],
                vae_stride=spatial_reduction,  # 使用计算出的正确值
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
            print("MAR initialized successfully!")
        except Exception as e:
            print(f"MAR initialization failed: {e}")
            raise e

        # 优化器
        self.vae_optimizer = optim.Adam(self.vae.parameters(), lr=float(self.config['klvae']['learning_rate']))
        self.mar_optimizer = optim.Adam(self.mar.parameters(), lr=float(self.config['mar']['learning_rate']))

        logging.info("Models initialized")

    def kl_vae_loss(self, recon_x, x, posterior, kl_weight=1.0):
        """计算KL-VAE损失"""
        # 重构损失 (MSE)
        recon_loss = torch.nn.functional.mse_loss(recon_x, x, reduction='mean')

        # KL散度损失
        kl_loss = posterior.kl().mean()

        # 总损失
        total_loss = recon_loss + kl_weight * kl_loss

        return total_loss, recon_loss, kl_loss

    def train_vae(self):
        """训练KL-VAE模型"""
        logging.info("Starting KL-VAE training...")

        self.vae.train()
        start_epoch = 0

        # 如果有checkpoint，则加载
        if self.config['training']['resume_vae']:
            checkpoint = load_checkpoint(self.config['training']['resume_vae'])
            self.vae.load_state_dict(checkpoint['model_state_dict'])
            self.vae_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            logging.info(f"Resumed KL-VAE training from epoch {start_epoch}")

        for epoch in range(start_epoch, self.config['klvae']['epochs']):
            epoch_loss = 0.0
            epoch_recon_loss = 0.0
            epoch_kl_loss = 0.0
            progress_bar = tqdm(self.dataloader, desc=f'KL-VAE Epoch {epoch + 1}/{self.config["klvae"]["epochs"]}')

            for batch_idx, (images, labels) in enumerate(progress_bar):
                images = images.to(self.device)

                # KL-VAE前向传播
                posterior = self.vae.encode(images)
                z = posterior.sample()
                recon_images = self.vae.decode(z)

                # 计算损失
                kl_weight = self.config['klvae'].get('kl_weight', 1.0)
                total_loss, recon_loss, kl_loss = self.kl_vae_loss(
                    recon_images, images, posterior, kl_weight
                )

                # 反向传播
                self.vae_optimizer.zero_grad()
                total_loss.backward()
                self.vae_optimizer.step()

                epoch_loss += total_loss.item()
                epoch_recon_loss += recon_loss.item()
                epoch_kl_loss += kl_loss.item()

                progress_bar.set_postfix({
                    'Total': f'{total_loss.item():.4f}',
                    'Recon': f'{recon_loss.item():.4f}',
                    'KL': f'{kl_loss.item():.4f}'
                })

            avg_loss = epoch_loss / len(self.dataloader)
            avg_recon_loss = epoch_recon_loss / len(self.dataloader)
            avg_kl_loss = epoch_kl_loss / len(self.dataloader)

            logging.info(f'KL-VAE Epoch {epoch + 1}: Total Loss = {avg_loss:.4f}, '
                         f'Recon Loss = {avg_recon_loss:.4f}, KL Loss = {avg_kl_loss:.4f}')

            # 保存checkpoint
            if (epoch + 1) % self.config['klvae']['save_interval'] == 0:
                save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': self.vae.state_dict(),
                    'optimizer_state_dict': self.vae_optimizer.state_dict(),
                    'loss': avg_loss,
                    'recon_loss': avg_recon_loss,
                    'kl_loss': avg_kl_loss,
                }, os.path.join(self.config['klvae']['checkpoint_dir'], f'klvae_epoch_{epoch + 1}.pth'))

                # 生成重构样本
                self.generate_vae_samples(epoch + 1)

        # 保存最终模型
        save_checkpoint({
            'epoch': self.config['klvae']['epochs'] - 1,
            'model_state_dict': self.vae.state_dict(),
            'optimizer_state_dict': self.vae_optimizer.state_dict(),
            'loss': avg_loss,
            'recon_loss': avg_recon_loss,
            'kl_loss': avg_kl_loss,
        }, os.path.join(self.config['klvae']['checkpoint_dir'], 'klvae_final.pth'))

        logging.info("KL-VAE training completed!")

    def generate_vae_samples(self, epoch):
        """生成KL-VAE重构样本 + 随机采样图像"""
        self.vae.eval()
        with torch.no_grad():
            # ==== 重构图像 ====
            images, _ = next(iter(self.dataloader))
            images = images[:16].to(self.device)  # 前16张图像

            # 编码到潜在空间
            posterior = self.vae.encode(images)
            z = posterior.sample()
            recon_images = self.vae.decode(z)

            # 拼接原图与重构图
            comparison = torch.cat([images[:8], recon_images[:8]], dim=0)
            vutils.save_image(
                comparison,
                os.path.join('samples', f'klvae_recon_epoch_{epoch}.png'),
                nrow=8,
                normalize=True,
                value_range=(-1, 1)
            )

            # ==== 随机采样生成图像 ====
            num_samples = 16
            # 获取潜在空间的形状
            sample_posterior = self.vae.encode(images[:1])
            latent_shape = sample_posterior.sample().shape[1:]  # 去掉batch维度

            # 生成随机潜在向量
            z_random = torch.randn(num_samples, *latent_shape).to(self.device)

            # 解码生成图像
            sampled_images = self.vae.decode(z_random)

            vutils.save_image(
                sampled_images,
                os.path.join('samples', f'klvae_sample_epoch_{epoch}.png'),
                nrow=4,
                normalize=True,
                value_range=(-1, 1)
            )

        self.vae.train()

    def train_mar(self):
        """训练MAR模型"""
        logging.info("Starting MAR training...")

        # 加载训练好的KL-VAE
        vae_checkpoint = os.path.join(self.config['klvae']['checkpoint_dir'], 'klvae_final.pth')
        if not os.path.exists(vae_checkpoint):
            raise FileNotFoundError("KL-VAE checkpoint not found. Please train KL-VAE first.")

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

                # 使用KL-VAE编码图像到潜在空间
                with torch.no_grad():
                    posterior = self.vae.encode(images)
                    # 对于MAR训练，我们可以使用均值而不是采样，以减少随机性
                    use_mean = self.config['mar'].get('use_posterior_mean', False)
                    if use_mean:
                        latent_images = posterior.mode()  # 使用均值
                    else:
                        latent_images = posterior.sample()  # 使用采样

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

            # 使用KL-VAE解码到图像空间
            generated_images = self.vae.decode(latent_tokens)

            # 保存生成的图片
            vutils.save_image(
                generated_images,
                os.path.join('samples', f'mar_generated_epoch_{epoch}.png'),
                nrow=4,
                normalize=True,
                value_range=(-1, 1)  # 根据你的数据范围调整
            )

            logging.info(f"Generated {num_samples} samples at epoch {epoch}")

        self.mar.train()

    def run_training(self):
        """运行完整的训练流程"""
        # 首先训练KL-VAE
        self.train_vae()

        # 然后训练MAR
        self.train_mar()


def main():
    # 直接写死配置路径和训练阶段
    config_path = r'D:\Desktop\mar_project\configs\config2.yaml'
    stage = 'both'  # 'vae' / 'mar' / 'both'

    trainer = Trainer(config_path)

    if stage == 'vae':
        trainer.train_vae()
    elif stage == 'mar':
        trainer.train_mar()
    else:
        trainer.run_training()


if __name__ == "__main__":
    main()