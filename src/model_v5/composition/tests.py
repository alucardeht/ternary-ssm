"""Test suite for model composition and TIES merging."""

import torch
import torch.nn as nn
from ..config import ModelConfig
from ..model import ModelV2
from .ties_merge import TernaryTIESMerger, validate_merge_quality


def test_basic_merge():
    """Test basic TIES merge with 3 identical architecture models."""
    print("\n=== Test: Basic TIES Merge ===")
    
    config = ModelConfig(
        d_model=64, n_layer=2, d_state=8, d_conv=4, expand=2, dt_rank=4,
        vocab_size=2048, seq_len=128, n_heads=2, window_size=32,
        swa_layer_indices=[], use_ternary=True, ternary_group_size=32,
        ste_type='standard', use_ssd=True, ssd_n_heads=4, dropout=0.0,
    )
    
    # Create 3 models
    models = []
    for i in range(3):
        torch.manual_seed(i * 42 + 1)
        m = ModelV2(config)
        models.append(m)
    
    # Merge
    merger = TernaryTIESMerger()
    merged_state = merger.merge(models)
    
    # Load merged model
    torch.manual_seed(999)
    merged_model = ModelV2(config)
    merged_model.load_state_dict(merged_state)
    
    # Test inference
    x = torch.randint(0, 2048, (1, 128))
    with torch.no_grad():
        merged_model.eval()
        out = merged_model(x)
    
    assert out.shape == (1, 128, 2048), f"Output shape mismatch: {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaN"
    
    stats = merger.get_merge_stats()
    assert 'ternary_params' in stats, "Missing ternary_params in stats"
    assert 'dense_params' in stats, "Missing dense_params in stats"
    
    print(f"[OK] Merged {stats['ternary_params']} ternary params")
    print(f"[OK] Merged {stats['dense_params']} dense params")
    print(f"[OK] Output shape: {out.shape}")
    print("[PASS]")


def test_weighted_merge():
    """Test TIES merge with custom weights for each model."""
    print("\n=== Test: Weighted TIES Merge ===")
    
    config = ModelConfig(
        d_model=32, n_layer=1, d_state=8, d_conv=4, expand=2, dt_rank=4,
        vocab_size=1024, seq_len=64, n_heads=2, window_size=16,
        swa_layer_indices=[], use_ternary=True, ternary_group_size=32,
        ste_type='standard', use_ssd=False, dropout=0.0,
    )
    
    # Create 2 models
    models = []
    for i in range(2):
        torch.manual_seed(i)
        m = ModelV2(config)
        models.append(m)
    
    # Test with different weights
    weights = [0.3, 0.7]
    merger = TernaryTIESMerger(tie_weights=weights)
    merged_state = merger.merge(models)
    
    # Verify all parameters present
    assert len(merged_state) > 0, "No parameters in merged state"
    
    # Load and test
    merged_model = ModelV2(config)
    merged_model.load_state_dict(merged_state)
    
    x = torch.randint(0, 1024, (1, 64))
    with torch.no_grad():
        merged_model.eval()
        out = merged_model(x)
    
    assert out.shape == (1, 64, 1024), f"Shape mismatch: {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaN"
    
    print(f"[OK] Weighted merge with weights {weights}")
    print(f"[OK] Output shape: {out.shape}")
    print("[PASS]")


def test_trim_threshold():
    """Test TIES merge with different trim thresholds."""
    print("\n=== Test: Trim Threshold ===")
    
    config = ModelConfig(
        d_model=32, n_layer=1, d_state=8, d_conv=4, expand=2, dt_rank=4,
        vocab_size=1024, seq_len=64, n_heads=2, window_size=16,
        swa_layer_indices=[], use_ternary=True, ternary_group_size=32,
        ste_type='standard', use_ssd=False, dropout=0.0,
    )
    
    models = []
    for i in range(3):
        torch.manual_seed(i)
        m = ModelV2(config)
        models.append(m)
    
    # Test with different trim thresholds
    for trim in [0.0, 0.15, 0.5]:
        merger = TernaryTIESMerger(trim_threshold=trim)
        merged_state = merger.merge(models)
        
        merged_model = ModelV2(config)
        merged_model.load_state_dict(merged_state)
        
        x = torch.randint(0, 1024, (1, 64))
        with torch.no_grad():
            merged_model.eval()
            out = merged_model(x)
        
        assert not torch.isnan(out).any(), f"NaN with trim={trim}"
        print(f"  [OK] Trim {trim:0.2f}")
    
    print("[PASS]")


def test_output_quality():
    """Test that merged model output is near average of original models."""
    print("\n=== Test: Output Quality ===")
    
    config = ModelConfig(
        d_model=32, n_layer=1, d_state=8, d_conv=4, expand=2, dt_rank=4,
        vocab_size=512, seq_len=32, n_heads=2, window_size=16,
        swa_layer_indices=[], use_ternary=True, ternary_group_size=32,
        ste_type='standard', use_ssd=False, dropout=0.0,
    )
    
    models = []
    for i in range(3):
        torch.manual_seed(i)
        m = ModelV2(config)
        models.append(m)
    
    # Merge
    merger = TernaryTIESMerger()
    merged_state = merger.merge(models)
    
    merged_model = ModelV2(config)
    merged_model.load_state_dict(merged_state)
    
    # Compute output quality
    test_input = torch.randint(0, 512, (1, 32))
    metrics = validate_merge_quality(models, merged_model, test_input, device='cpu')
    
    print(f"[OK] Output agreement: {metrics['output_agreement']:.4f}")
    print(f"[OK] Weight divergence: {metrics['weight_divergence']:.6f}")
    
    # Merged output should be reasonably close to original average
    assert metrics['output_agreement'] > 0.5, "Output agreement too low"
    
    print("[PASS]")


if __name__ == '__main__':
    print("=" * 60)
    print("TIES MERGE TEST SUITE")
    print("=" * 60)
    
    test_basic_merge()
    test_weighted_merge()
    test_trim_threshold()
    test_output_quality()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
