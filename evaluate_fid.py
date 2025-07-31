import os
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.utils as vutils
from tqdm import tqdm
import argparse
from scipy import linalg
from torch.nn import functional as F
import logging

from models import VAE, MAR
from utils import CIFAR100Dataset, setup_logging, load_checkpoint

try:
    from torchvision.models import inception_v3

    INCEPTION_AVAILABLE = True
except ImportError:
    INCEPTION_AVAILABLE = False
    print("Warning: torchvision inception_v3 not available. FID calculation will be limited.")


class InceptionV3(torch.nn.Module):
    """用于FID计算的Inception V3模型"""

    def __init__(self, resize_input=True, normalize_input=True):
        super(InceptionV3, self).__init__()

        if not INCEPTION_AVAILABLE:
            raise ImportError("torchvision inception_v3 is required for FID calculation")

        self.resize_input = resize_input
        self.normalize_input = normalize_input

        # 加载预训练的Inception V3
        inception = inception_v3(pretrained=True, transform_input=False)
        inception.eval()

        # 移除最后的分类层
        self.inception = torch.nn.Sequential(*list(inception.children())[:-1])

    def forward(self, x):
        if self.resize_input:
            # 将32x32的图像resize到299x299
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)

        if self.normalize_input:
            # ImageNet normalization
            x = (x + 1) / 2  # 从[-1,1]转换到[0,1]
            x = F.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # 通过Inception网络
        x = self.inception(x)

        # 全局平均池化
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)

        return x


def calculate_fid_statistics(dataloader, model, device, max_samples=None):
    """计算数据集的FID统计信息"""
    model.eval()
    features = []

    with torch.no_grad():
        for i, (images, _) in enumerate(tqdm(dataloader, desc="Extracting features")):
            if max_samples and i * dataloader.batch_size >= max_samples:
                break

            images = images.to(device)
            feat = model(images)
            features.append(feat.cpu().numpy())

    features = np.concatenate(features, axis=0)
    if max_samples:
        features = features[:max_samples]

    # 计算均值和协方差
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)

    return mu, sigma


def calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """计算FID分数"""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    # 计算sqrt((sigma1 @ sigma2))
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # 数值不稳定性检查
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError('Imaginary component in covariance mean')
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


class FIDEvaluator:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device(self.config['training']['device'] if torch.cuda.is_available() else 'cpu')

        # 设置日志
        setup_logging('logs/fid_evaluation.log')

        # 初始化Inception模型
        if INCEPTION_AVAILABLE:
            self.inception = InceptionV3().to(self.device)
        else:
            raise ImportError("Inception V3 not available for FID calculation")

        # 准备数据
        self.prepare_data()

        # 加载模型
        self.load_models()

    def prepare_data(self):
        """准备真实数据"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # 测试集数据
        self.test_dataset = CIFAR100Dataset(
            root=self.config['data']['dataset_path'],
            transform=transform,
            train=False
        )

        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.config['evaluation']['fid_batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers']
        )

    def load_models(self):
        """加载训练好的模型"""
        # 加载VAE
        vae_path = os.path.join(self.config['vae']['checkpoint_dir'], 'vae_final.pth')
        if not os.path.exists(vae_path):
            raise FileNotFoundError(f"VAE checkpoint not found: {vae_path}")

        self.vae = VAE(
            in_channels=self.config['vae']['in_channels'],
            latent_dim=self.config['vae']['latent_dim'],
            hidden_channels=self.config['vae']['hidden_channels'],
            res_nums=self.config['vae']['res_nums']
        ).to(self.device)

        checkpoint = load_checkpoint(vae_path)
        self.vae.load_state_dict(checkpoint['model_state_dict'])
        self.vae.eval()

        # 加载MAR
        mar_path = os.path.join(self.config['mar']['checkpoint_dir'], 'mar_final.pth')
        if not os.path.exists(mar_path):
            raise FileNotFoundError(f"MAR checkpoint not found: {mar_path}")

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

        checkpoint = load_checkpoint(mar_path)
        self.mar.load_state_dict(checkpoint['model_state_dict'])
        self.mar.eval()

        logging.info("Models loaded successfully")

    def generate_samples(self, num_samples):
        """生成指定数量的样本"""
        self.mar.eval()
        self.vae.eval()

        generated_images = []
        batch_size = self.config['evaluation']['fid_batch_size']

        with torch.no_grad():
            for i in tqdm(range(0, num_samples, batch_size), desc="Generating samples"):
                current_batch_size = min(batch_size, num_samples - i)

                # 随机选择类别
                labels = torch.randint(0, self.config['data']['num_classes'],
                                       (current_batch_size,)).to(self.device)

                # 使用MAR生成潜在空间表示
                latent_tokens = self.mar.sample_tokens(
                    bsz=current_batch_size,
                    num_iter=self.config['evaluation']['num_iter'],
                    cfg=self.config['evaluation']['cfg_scale'],
                    labels=labels,
                    temperature=self.config['evaluation']['temperature']
                )

                # 使用VAE解码到图像空间
                images = self.vae.decode(latent_tokens)

                # 转换值域从[0,1]到[-1,1]
                images = images * 2.0 - 1.0

                generated_images.append(images.cpu())

        return torch.cat(generated_images, dim=0)[:num_samples]

    def evaluate_fid(self, num_samples=None):
        """评估FID分数"""
        if num_samples is None:
            num_samples = self.config['evaluation']['num_samples']

        logging.info(f"Evaluating FID with {num_samples} samples")

        # 计算真实数据的统计信息
        logging.info("Calculating statistics for real data...")
        real_mu, real_sigma = calculate_fid_statistics(
            self.test_dataloader, self.inception, self.device, num_samples
        )

        # 生成样本
        logging.info("Generating samples...")
        generated_samples = self.generate_samples(num_samples)

        # 保存生成的样本（可选）
        sample_grid = vutils.make_grid(generated_samples[:64], nrow=8, normalize=True, value_range=(-1, 1))
        vutils.save_image(sample_grid, 'samples/fid_evaluation_samples.png')

        # 为生成样本创建DataLoader
        generated_dataset = torch.utils.data.TensorDataset(generated_samples, torch.zeros(len(generated_samples)))
        generated_dataloader = DataLoader(
            generated_dataset,
            batch_size=self.config['evaluation']['fid_batch_size'],
            shuffle=False
        )

        # 计算生成数据的统计信息
        logging.info("Calculating statistics for generated data...")
        gen_mu, gen_sigma = calculate_fid_statistics(
            generated_dataloader, self.inception, self.device, num_samples
        )

        # 计算FID
        fid_score = calculate_fid(real_mu, real_sigma, gen_mu, gen_sigma)

        logging.info(f"FID Score: {fid_score:.4f}")

        # 保存结果
        print(f"FID Score: {fid_score:.4f}")

        # 保存FID结果到文本文件
        fid_result_path = os.path.join(self.config['evaluation']['result_dir'], 'fid_result.txt')
        os.makedirs(self.config['evaluation']['result_dir'], exist_ok=True)
        with open(fid_result_path, 'w') as f:
            f.write(f"FID Score: {fid_score:.4f}\n")

        print(f"\n✅ FID Score: {fid_score:.4f} (saved to {fid_result_path})")
        return fid_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FID Evaluation")
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    args = parser.parse_args()

    evaluator = FIDEvaluator(config_path=args.config)
    evaluator.evaluate_fid()
