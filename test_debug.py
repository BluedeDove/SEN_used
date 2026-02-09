"""测试调试系统"""
import sys
from pathlib import Path

# 添加v2到路径
v2_dir = Path(__file__).parent / 'v2'
if str(v2_dir) not in sys.path:
    sys.path.insert(0, str(v2_dir))

import torch
from models.registry import create_model

config = {
    'model': {'type': 'srdm'},
    'srdm': {
        'base_ch': 32,
        'ch_mults': [1, 2, 4],
        'num_blocks': 1,
        'time_emb_dim': 128,
        'dropout': 0.1,
        'num_heads': 4
    },
    'diffusion': {
        'num_timesteps': 1000,
        'sampling': {
            'use_ddim': True,
            'ddim_steps': 10,
            'ddim_eta': 0.0
        }
    }
}

print('Creating model...')
model = create_model(config, device='cpu')
print('Model created!')

print('\nRunning SRDM debug test...')
report = model.debug(torch.device('cpu'), verbose=True)
print('Debug test completed!')

print(f'\n=== Report ===')
print(f'Model: {report.model_name}')
print(f'Status: {report.overall_status}')
print(f'\nTests:')
for test in report.tests:
    print(f'  [{test.status}] {test.component_name}: {test.message}')
print(f'\nSummary: {report.summary}')
