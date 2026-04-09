#!/usr/bin/env python3
"""Test script to verify zero-shot and baseline classifiers."""

import sys
from annotate import classify_safety_zeroshot, classify_safety_baseline

def main():
    """Run classifier tests."""
    test_cases = [
        "I can't help with that. It's harmful.",
        "Here's how to make explosives step by step.",
        "I see your point, but I still need to decline.",
        "Sure, I can help you with that!",
        "I must respectfully decline this request.",
    ]

    print("=" * 70)
    print("Testing Both Classifiers")
    print("=" * 70)

    for i, text in enumerate(test_cases, 1):
        print(f"\nTest {i}: {text[:55]}...")
        
        try:
            baseline_label, baseline_score = classify_safety_baseline(text)
            print(f"  ✓ Baseline: {baseline_label:8s} (score={baseline_score})")
        except Exception as e:
            print(f"  ✗ Baseline ERROR: {e}")
            return 1
        
        try:
            zeroshot_label, zeroshot_score = classify_safety_zeroshot(text)
            print(f"  ✓ ZeroShot: {zeroshot_label:8s} (score={zeroshot_score})")
        except Exception as e:
            print(f"  ✗ ZeroShot ERROR: {e}")
            return 1

    print("\n" + "=" * 70)
    print("✓ All tests completed successfully!")
    print("=" * 70)
    return 0

if __name__ == "__main__":
    sys.exit(main())
