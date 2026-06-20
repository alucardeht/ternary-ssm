"""Test SSD+ternary training with optimizer fix."""
from src.model_v2.config import config_wikitext_50m
from src.model_v2.train import train_model

# Load config and enable ternary
config = config_wikitext_50m()
config.use_ternary = True

print("="*60)
print("Testing SSD + Ternary with optimizer fix")
print("="*60)
print(f"Config: d_model={config.d_model}, n_layer={config.n_layer}")
print(f"use_ssd={config.use_ssd}, use_ternary={config.use_ternary}")
print(f"Optimizer: AdamW, beta2=0.999, weight_decay=0.01, fused=False")
print()

# Run 30 steps
try:
    results = train_model(
        config=config,
        dataset_type='wikitext',
        data_dir='data/wikitext103',
        batch_size=4,
        num_epochs=1,
        max_steps=30,
        device='cuda',
        base_lr=6e-4,
        warmup_steps=100,
        log_every=1,
        max_memory_mb=10000,
        weight_decay=0.01,
        beta2=0.999,
        fused_optimizer=False,
    )

    print("\n" + "="*60)
    print("TEST COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"Final loss: {results.get('final_loss', 'N/A')}")
    print(f"Steps completed: {results.get('steps_completed', 0)}")
    print("="*60)
except Exception as e:
    print("\n" + "="*60)
    print(f"TEST FAILED WITH ERROR: {e}")
    print("="*60)
    import traceback
    traceback.print_exc()
