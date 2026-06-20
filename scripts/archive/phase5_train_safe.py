import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import time
from src.model_v2.config import config_ternary_50m
from src.model_v2.train import train_model

if __name__ == '__main__':
    print('=' * 80)
    print('PHASE 5 — 50M VALIDATION RUN (TERNARY, 100 STEPS TEST)')
    print('=' * 80)
    print(f'Start time: {time.strftime("%Y-%m-%d %H:%M:%S")}')

    config = config_ternary_50m()
    config.use_torch_compile = False

    print(f'torch.compile: DISABLED')
    print(f'CUDA_LAUNCH_BLOCKING: 1')

    start_time = time.time()

    results = train_model(
        config=config,
        dataset_type='wikitext',
        batch_size=16,  # Further reduced
        max_steps=100,  # Test run
        device='cuda',
        max_memory_mb=8000,
        weight_decay=0.01,
        beta2=0.999,
        fused_optimizer=False,
        angular_reg_lambda=0.01,
        boundary_reg_lambda=0.01,
        use_liger_ce=False,
        log_every=10,
        eval_every=50,
        checkpoint_every=100,
        num_workers=0,
        pin_memory=False,
    )

    elapsed = time.time() - start_time

    print()
    print('=' * 80)
    print('TEST RESULTS (100 STEPS)')
    print('=' * 80)
    print(f'Final loss: {results.get("final_loss", "N/A")}')
    print(f'Best loss: {results.get("best_loss", "N/A")}')
    print(f'Tokens/sec: {results.get("tokens_per_sec", 0):.0f}')
    print(f'Training time: {elapsed/60:.1f} minutes')
    print(f'End time: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 80)
