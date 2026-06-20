import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import time
from src.model_v2.config import config_ternary_50m
from src.model_v2.train import train_model

if __name__ == '__main__':
    print('=' * 80)
    print('PHASE 5 — 50M VALIDATION RUN (TERNARY, 1000 STEPS)')
    print('=' * 80)
    print(f'Start time: {time.strftime("%Y-%m-%d %H:%M:%S")}')

    config = config_ternary_50m()
    config.use_torch_compile = False

    print(f'torch.compile: DISABLED')
    print(f'CUDA_LAUNCH_BLOCKING: 1')
    print(f'Batch size: 16, Max steps: 1000')

    start_time = time.time()

    results = train_model(
        config=config,
        dataset_type='wikitext',
        batch_size=16,
        max_steps=1000,
        device='cuda',
        max_memory_mb=9000,
        weight_decay=0.01,
        beta2=0.999,
        fused_optimizer=False,
        angular_reg_lambda=0.01,
        boundary_reg_lambda=0.01,
        use_liger_ce=False,
        log_every=100,
        eval_every=200,
        checkpoint_every=500,
        num_workers=0,
        pin_memory=False,
    )

    elapsed = time.time() - start_time

    print()
    print('=' * 80)
    print('FINAL RESULTS (1000 STEPS)')
    print('=' * 80)
    print(f'Final loss: {results.get("final_loss", "N/A")}')
    print(f'Best loss: {results.get("best_loss", "N/A")}')
    print(f'Tokens/sec: {results.get("tokens_per_sec", 0):.0f}')
    print(f'Eval PPL: {results.get("eval_perplexity", "N/A")}')
    print(f'Total params: {results.get("total_params", "N/A")}')
    print(f'Ternary params: {results.get("ternary_params", "N/A")}')
    print(f'Ternary ratio: {results.get("ternary_ratio", "N/A")}')
    print(f'Training time: {elapsed/60:.1f} minutes')
    print(f'End time: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 80)
