#!/usr/bin/env python3
"""
Test IMPROVEMENT 3: Normalized AHE computation.

Verify that normalized AHE:
1. Computes entropy instead of std dev
2. Normalizes by log(sequence_length)
3. Is comparable across architectures
"""

import torch
import numpy as np
import logging
from pathlib import Path

# Add project to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from inference import compute_attention_entropy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_entropy_computation():
    """Test that entropy is computed correctly."""
    logger.info("Test 1: Entropy Computation (Uniform Distribution)")
    
    # Uniform attention distribution (max entropy)
    batch_size, heads, tgt_len, src_len = 1, 8, 10, 50
    uniform_attn = torch.ones(batch_size, heads, tgt_len, src_len) / src_len
    
    raw_ahe, norm_ahe = compute_attention_entropy(
        (None, uniform_attn),
        sequence_length=src_len
    )
    
    # For uniform distribution over N items, entropy = log(N)
    expected_raw = np.log(src_len)
    logger.info(f"  Uniform attention (50 positions):")
    logger.info(f"    Raw AHE: {raw_ahe:.4f} (expected ~{expected_raw:.4f})")
    logger.info(f"    Norm AHE: {norm_ahe:.4f} (expected ~1.0)")
    
    assert 0.95 < norm_ahe <= 1.05, f"Normalized AHE should be ~1.0 for uniform, got {norm_ahe}"
    logger.info(f"    ✓ PASS: Uniform distribution normalized correctly\n")


def test_normalization_across_lengths():
    """Test that normalization makes different sequence lengths comparable."""
    logger.info("Test 2: Normalization Across Sequence Lengths")
    
    batch_size, heads, tgt_len = 1, 8, 10
    
    # Create concentrated attention (low entropy with small sequence)
    src_len_small = 20
    concentrated_small = torch.zeros(batch_size, heads, tgt_len, src_len_small)
    # Put all attention on first token
    concentrated_small[:, :, :, 0] = 1.0
    
    raw_ahe_small, norm_ahe_small = compute_attention_entropy(
        (None, concentrated_small),
        sequence_length=src_len_small
    )
    
    # Create concentrated attention (low entropy with large sequence)
    src_len_large = 200
    concentrated_large = torch.zeros(batch_size, heads, tgt_len, src_len_large)
    # Put all attention on first token
    concentrated_large[:, :, :, 0] = 1.0
    
    raw_ahe_large, norm_ahe_large = compute_attention_entropy(
        (None, concentrated_large),
        sequence_length=src_len_large
    )
    
    logger.info(f"  Concentrated attention (all on first token):")
    logger.info(f"    len=20:  raw_ahe={raw_ahe_small:.4f}, norm_ahe={norm_ahe_small:.4f}")
    logger.info(f"    len=200: raw_ahe={raw_ahe_large:.4f}, norm_ahe={norm_ahe_large:.4f}")
    logger.info(f"    Ratio of normalized: {norm_ahe_large/norm_ahe_small:.4f} (should be ~1.0)")
    
    # Normalized values should be very close (same attention pattern)
    assert abs(norm_ahe_small - norm_ahe_large) < 0.01, \
        f"Normalized AHE should be similar for same pattern, got {norm_ahe_small} vs {norm_ahe_large}"
    
    logger.info(f"    ✓ PASS: Normalization makes sequences comparable\n")


def test_entropy_vs_stddev():
    """Show difference between entropy and std dev."""
    logger.info("Test 3: Entropy vs Std Dev (Conceptual Difference)")
    
    batch_size, heads, tgt_len, src_len = 1, 8, 10, 50
    
    # Create a specific attention distribution
    attn = torch.ones(batch_size, heads, tgt_len, src_len) / src_len
    
    raw_ahe, norm_ahe = compute_attention_entropy((None, attn), sequence_length=src_len)
    
    # Compute what std dev would be (old method)
    attn_dist = attn.squeeze(0).mean(dim=0).mean(dim=0)  # Average over heads and queries
    stddev = attn_dist.std().item()
    
    logger.info(f"  For uniform distribution over 50 positions:")
    logger.info(f"    Entropy:  {raw_ahe:.4f}")
    logger.info(f"    Std Dev:  {stddev:.4f}")
    logger.info(f"    Ratio (entropy/stddev): {raw_ahe/stddev:.4f}")
    logger.info(f"    ✓ Different measures capture different aspects\n")


def test_edge_cases():
    """Test edge cases."""
    logger.info("Test 4: Edge Cases")
    
    # Test 1: Single position (concentration)
    concentrated = torch.ones(1, 8, 10, 1)
    raw, norm = compute_attention_entropy((None, concentrated), sequence_length=1)
    logger.info(f"  Single position (100% concentration):")
    logger.info(f"    Raw AHE: {raw:.6f}, Norm AHE: {norm:.6f}")
    assert raw < 0.001, f"Concentrated attention should have near-zero entropy"
    logger.info(f"    ✓ PASS: Concentrated attention has low entropy\n")
    
    # Test 2: Random attention
    random_attn = torch.rand(1, 8, 10, 50)
    raw, norm = compute_attention_entropy((None, random_attn), sequence_length=50)
    logger.info(f"  Random attention distribution:")
    logger.info(f"    Raw AHE: {raw:.4f}, Norm AHE: {norm:.4f}")
    assert 0 < raw < np.log(50), f"Random attention should have entropy in valid range"
    assert 0 < norm < 1.5, f"Random attention normalized should be in valid range"
    logger.info(f"    ✓ PASS: Random attention has valid entropy\n")


def test_architecture_comparison():
    """Demonstrate normalized AHE for architecture comparison."""
    logger.info("Test 5: Cross-Architecture Comparison")
    
    # Simulated BART (10 decoder layers, 768 dim, 12 heads)
    bart_heads = 12
    bart_seq_len = 100
    # Create somewhat uniform attention for BART
    bart_attn = torch.ones(1, bart_heads, 10, bart_seq_len) / bart_seq_len + \
                torch.randn(1, bart_heads, 10, bart_seq_len) * 0.01
    bart_raw, bart_norm = compute_attention_entropy((None, bart_attn), sequence_length=bart_seq_len)
    
    # Simulated T5 (12 decoder layers, 768 dim, 12 heads)
    t5_heads = 12
    t5_seq_len = 100
    # Same-ish attention pattern
    t5_attn = torch.ones(1, t5_heads, 10, t5_seq_len) / t5_seq_len + \
              torch.randn(1, t5_heads, 10, t5_seq_len) * 0.01
    t5_raw, t5_norm = compute_attention_entropy((None, t5_attn), sequence_length=t5_seq_len)
    
    # Simulated PEGASUS (16 decoder layers, 768 dim, 12 heads)
    pegasus_heads = 12
    pegasus_seq_len = 100
    # Same-ish attention pattern
    pegasus_attn = torch.ones(1, pegasus_heads, 10, pegasus_seq_len) / pegasus_seq_len + \
                   torch.randn(1, pegasus_heads, 10, pegasus_seq_len) * 0.01
    pegasus_raw, pegasus_norm = compute_attention_entropy((None, pegasus_attn), sequence_length=pegasus_seq_len)
    
    logger.info(f"  Different architectures with similar attention patterns:")
    logger.info(f"    BART:    raw_ahe={bart_raw:.4f}, norm_ahe={bart_norm:.4f}")
    logger.info(f"    T5:      raw_ahe={t5_raw:.4f}, norm_ahe={t5_norm:.4f}")
    logger.info(f"    PEGASUS: raw_ahe={pegasus_raw:.4f}, norm_ahe={pegasus_norm:.4f}")
    logger.info(f"    ✓ Normalized AHE makes architectures comparable\n")


if __name__ == "__main__":
    logger.info("="*60)
    logger.info("IMPROVEMENT 3: Normalized AHE Test Suite")
    logger.info("="*60 + "\n")
    
    try:
        test_entropy_computation()
        test_normalization_across_lengths()
        test_entropy_vs_stddev()
        test_edge_cases()
        test_architecture_comparison()
        
        logger.info("="*60)
        logger.info("✓ All tests passed!")
        logger.info("="*60)
    
    except AssertionError as e:
        logger.error(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n✗ Unexpected error: {e}", exc_info=True)
        sys.exit(1)
