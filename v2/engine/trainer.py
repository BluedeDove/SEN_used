"""
trainer.py - 训练循环

实现训练循环（epoch级）。
"""

import torch
from tqdm import tqdm
from typing import Dict, Any
from core.device_ops import is_main_process
from core.training_ops import train_step


class Trainer:
    """训练器"""

    def __init__(
        self,
        model,
        train_loader,
        optimizer,
        scheduler,
        scaler,
        config: dict,
        device: torch.device,
        rank: int = 0
    ):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.config = config
        self.device = device
        self.rank = rank

        # 配置
        self.use_amp = config['training'].get('mixed_precision', {}).get('enabled', False)
        self.max_norm = config['training'].get('gradient_clipping', {}).get('max_norm', 0.0) \
                        if config['training'].get('gradient_clipping', {}).get('enabled', False) else 0.0

    def train_epoch(self, epoch: int) -> Dict[str, Any]:
        """
        训练一个epoch

        Args:
            epoch: 当前epoch

        Returns:
            训练指标字典
        """
        self.model.train()

        total_loss = 0.0
        num_batches = len(self.train_loader)

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}",
            disable=not is_main_process(self.rank)
        )

        for batch in pbar:
            loss, loss_dict = train_step(
                self.model,
                batch,
                self.optimizer,
                self.scaler,
                self.device,
                self.use_amp,
                self.max_norm
            )

            total_loss += loss
            pbar.set_postfix({'loss': f'{loss:.4f}'})

        avg_loss = total_loss / num_batches

        # 更新学习率
        if self.scheduler is not None:
            self.scheduler.step()

        return {'train_loss': avg_loss}


if __name__ == "__main__":
    print("Trainer module loaded")
