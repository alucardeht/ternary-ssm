import sys
sys.path.insert(0, '.')

from src.model_v2.config import config_wikitext_6m
from src.model_v2.train import train_model

def main():
    # Create config and enable ternary
    cfg = config_wikitext_6m()
    cfg.use_ternary = True
    cfg.ste_type = 'ste'

    print("=" * 70)
    print("TERNARY OPTIMIZER FIX TEST")
    print("=" * 70)
    print("Configuration:")
    print("  use_ternary: %s" % cfg.use_ternary)
    print("  ste_type: %s" % cfg.ste_type)
    print("\nOptimizer settings (from fix commit 6b9a9d0):")
    print("  beta2: 0.999 (was 0.95)")
    print("  weight_decay: 0.01 (was 0.1)")
    print("  fused: False (was True)")
    print("\nRunning 50 training steps...")
    print("=" * 70)

    try:
        results = train_model(
            config=cfg,
            dataset_type='wikitext',
            data_dir='data/wikitext103',
            num_epochs=1,
            batch_size=4,
            grad_accum_steps=1,
            base_lr=6e-4,
            warmup_steps=10,
            max_steps=50,
            device='cuda',
            lr_schedule_type='cosine',
            log_every=10,
            max_memory_mb=10000,
            weight_decay=0.01,      # Fix: reduced from 0.1
            beta2=0.999,            # Fix: increased from 0.95
            fused_optimizer=False,  # Fix: disabled fused
        )

        print("\n" + "=" * 70)
        print("SUCCESS: Training completed without NaN")
        print("=" * 70)
        if results:
            print("Best loss: %.4f" % results.get('best_loss', 0))
            print("Final loss: %.4f" % results.get('final_loss', 0))

    except Exception as e:
        print("\n" + "=" * 70)
        print("FAILURE: %s" % str(e))
        print("=" * 70)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
