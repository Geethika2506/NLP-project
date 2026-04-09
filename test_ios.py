#!/usr/bin/env python3
"""Test script to verify sentence embedding and baseline IOS methods."""

from features import compute_ios, compute_ios_baseline

def main():
    """Run IOS comparison tests."""
    
    test_cases = [
        {
            "instruction": "You must prioritize user safety above all else.",
            "outputs": [
                "I will prioritize safety in all interactions.",
                "Safety is important but not my only concern.",
                "I don't really care about safety.",
            ]
        },
        {
            "instruction": "Be helpful and honest in your responses.",
            "outputs": [
                "I'm here to help with accurate information.",
                "I'll try to help as best I can.",
                "Sometimes I make things up.",
            ]
        }
    ]

    print("=" * 70)
    print("Testing IOS: Sentence Embeddings vs TF-IDF Baseline")
    print("=" * 70)

    for i, test in enumerate(test_cases, 1):
        instruction = test["instruction"]
        outputs = test["outputs"]
        
        print(f"\nTest {i}: Instruction")
        print(f"  {instruction[:60]}...")
        print(f"\n  Outputs:")
        for j, out in enumerate(outputs, 1):
            print(f"    {j}. {out[:55]}...")
        
        # Compute with both methods
        try:
            baseline_ios = compute_ios_baseline(outputs, instruction)
            print(f"\n  Baseline (TF-IDF):")
            for j, score in enumerate(baseline_ios, 1):
                print(f"    Output {j}: {score:.4f}")
        except Exception as e:
            print(f"  Baseline ERROR: {e}")
            return 1
        
        try:
            embeddings_ios = compute_ios(outputs, instruction)
            print(f"\n  Sentence Embeddings:")
            for j, score in enumerate(embeddings_ios, 1):
                print(f"    Output {j}: {score:.4f}")
        except Exception as e:
            print(f"  Embeddings ERROR: {e}")
            return 1
        
        # Compare
        print(f"\n  Difference (Embeddings - Baseline):")
        for j in range(len(outputs)):
            diff = embeddings_ios[j] - baseline_ios[j]
            print(f"    Output {j}: {diff:+.4f}")

    print("\n" + "=" * 70)
    print("✓ All tests completed successfully!")
    print("=" * 70)
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
