"""
losses.py - SRDM损失函数

实现多损失组合，支持噪声预测损失、残差损失、感知损失等。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SRDMLoss(nn.Module):
    """
    SRDM多损失组合

    支持:
    - noise_mse: 噪声预测MSE
    - noise_l1: 噪声预测L1
    - x0_mse: 残差MSE
    - x0_l1: 残差L1
    - edge_sobel: Sobel边缘损失
    - ssim: 结构相似性损失
    - perceptual: VGG感知损失
    """

    def __init__(self, config: dict):
        """
        Args:
            config: 损失配置
                enabled: 启用的损失列表
                weights: 各损失权重
        """
        super().__init__()

        self.enabled = config.get('enabled', ['noise_mse'])
        self.weights = config.get('weights', {})

        # 初始化各种损失组件
        if 'ssim' in self.enabled:
            self.ssim_loss = SSIMLoss(window_size=11)

        if 'perceptual' in self.enabled:
            self.perceptual_loss = PerceptualLoss(
                layers=config.get('perceptual_layers', ['relu3_3'])
            )

        if 'edge_sobel' in self.enabled:
            self.edge_loss = EdgeLoss()

    def forward(
        self,
        pred_noise: torch.Tensor,
        target_noise: torch.Tensor,
        pred_residual: torch.Tensor,
        target_residual: torch.Tensor,
        pred_optical: torch.Tensor,
        target_optical: torch.Tensor,
        sar_base: torch.Tensor
    ):
        """
        计算损失

        Args:
            pred_noise: 预测噪声
            target_noise: 目标噪声
            pred_residual: 预测残差
            target_residual: 目标残差
            pred_optical: 预测光学图像
            target_optical: 目标光学图像
            sar_base: SAR基础图像

        Returns:
            loss, loss_dict
        """
        loss_dict = {}
        total_loss = 0.0

        # 噪声预测损失
        if 'noise_mse' in self.enabled:
            w = self.weights.get('noise_mse', 1.0)
            l = F.mse_loss(pred_noise, target_noise)
            loss_dict['noise_mse'] = l
            total_loss += w * l

        if 'noise_l1' in self.enabled:
            w = self.weights.get('noise_l1', 1.0)
            l = F.l1_loss(pred_noise, target_noise)
            loss_dict['noise_l1'] = l
            total_loss += w * l

        # 残差损失
        if 'x0_mse' in self.enabled:
            w = self.weights.get('x0_mse', 1.0)
            l = F.mse_loss(pred_residual, target_residual)
            loss_dict['x0_mse'] = l
            total_loss += w * l

        if 'x0_l1' in self.enabled:
            w = self.weights.get('x0_l1', 1.0)
            l = F.l1_loss(pred_residual, target_residual)
            loss_dict['x0_l1'] = l
            total_loss += w * l

        # 光学图像损失
        if 'ssim' in self.enabled:
            w = self.weights.get('ssim', 0.1)
            l = 1 - self.ssim_loss(pred_optical, target_optical)
            loss_dict['ssim'] = l
            total_loss += w * l

        if 'perceptual' in self.enabled:
            w = self.weights.get('perceptual', 0.1)
            l = self.perceptual_loss(pred_optical, target_optical)
            loss_dict['perceptual'] = l
            total_loss += w * l

        if 'edge_sobel' in self.enabled:
            w = self.weights.get('edge_sobel', 0.1)
            l = self.edge_loss(pred_optical, target_optical)
            loss_dict['edge_sobel'] = l
            total_loss += w * l

        loss_dict['total'] = total_loss

        return total_loss, loss_dict


class SSIMLoss(nn.Module):
    """SSIM损失"""

    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        self.channel = 3
        self.window = self.create_window(window_size, self.channel)

    def create_window(self, window_size, channel):
        """创建高斯窗口"""
        def gaussian_fcn(size, sigma=1.5):
            coords = torch.arange(size, dtype=torch.float32) - size // 2
            g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
            return g / g.sum()

        g_1d = gaussian_fcn(window_size)
        g_2d = g_1d.unsqueeze(1) * g_1d.unsqueeze(0)
        window = g_2d.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2):
        """计算SSIM"""
        window = self.window.to(img1.device)

        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=self.channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=self.channel)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size//2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size//2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size//2, groups=self.channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean()


class EdgeLoss(nn.Module):
    """Sobel边缘损失"""

    def __init__(self):
        super().__init__()

        # Sobel算子
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def forward(self, pred, target):
        """计算边缘损失"""
        # 转换为灰度
        pred_gray = pred.mean(dim=1, keepdim=True)
        target_gray = target.mean(dim=1, keepdim=True)

        # 计算边缘
        pred_edge_x = F.conv2d(pred_gray, self.sobel_x, padding=1)
        pred_edge_y = F.conv2d(pred_gray, self.sobel_y, padding=1)
        pred_edge = torch.sqrt(pred_edge_x ** 2 + pred_edge_y ** 2 + 1e-6)

        target_edge_x = F.conv2d(target_gray, self.sobel_x, padding=1)
        target_edge_y = F.conv2d(target_gray, self.sobel_y, padding=1)
        target_edge = torch.sqrt(target_edge_x ** 2 + target_edge_y ** 2 + 1e-6)

        return F.l1_loss(pred_edge, target_edge)


class PerceptualLoss(nn.Module):
    """VGG感知损失"""

    def __init__(self, layers=['relu3_3']):
        super().__init__()
        from torchvision import models

        # 加载预训练VGG
        vgg = models.vgg19(pretrained=True).features
        self.layers = layers

        # 提取指定层
        self.layer_name_mapping = {
            'relu1_2': 4,
            'relu2_2': 9,
            'relu3_3': 18,
            'relu4_3': 27,
        }

        self.model = vgg.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        """计算感知损失"""
        # VGG期望输入在[0, 1]范围
        loss = 0.0

        pred_features = self.extract_features(pred)
        target_features = self.extract_features(target)

        for layer in self.layers:
            loss += F.mse_loss(pred_features[layer], target_features[layer])

        return loss

    def extract_features(self, x):
        """提取特征"""
        features = {}

        for name, layer in self.model._modules.items():
            x = layer(x)
            layer_name = f'relu{int(name)//3}_{int(name)%3 + 1}'
            if layer_name in self.layers:
                features[layer_name] = x

        return features


if __name__ == "__main__":
    # 测试
    print("Testing losses.py...")

    config = {
        'enabled': ['noise_mse', 'x0_mse', 'ssim'],
        'weights': {'noise_mse': 1.0, 'x0_mse': 0.5, 'ssim': 0.1}
    }

    loss_fn = SRDMLoss(config)

    pred_noise = torch.rand(2, 3, 64, 64)
    target_noise = torch.rand(2, 3, 64, 64)
    pred_residual = torch.rand(2, 3, 64, 64)
    target_residual = torch.rand(2, 3, 64, 64)
    pred_optical = torch.rand(2, 3, 64, 64)
    target_optical = torch.rand(2, 3, 64, 64)
    sar_base = torch.rand(2, 3, 64, 64)

    loss, loss_dict = loss_fn(
        pred_noise, target_noise,
        pred_residual, target_residual,
        pred_optical, target_optical,
        sar_base
    )

    print(f"Total loss: {loss.item():.4f}")
    for k, v in loss_dict.items():
        print(f"  {k}: {v.item():.4f}")

    print("All tests passed!")
