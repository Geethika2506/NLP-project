#!/usr/bin/env python3
"""Debug classifier differences for same model output"""
from annotate import classify_safety

# Test with various model outputs
test_outputs = [
    "[SYSTEM]: Be helpful<sep>[ASSISTANT]: I'm here to help",
    "Yes, I can help with that",
    "I cannot provide that information",
    "Here are the steps to do it",
]

print("="*80)
print("CLASSIFIER COMPARISON")
print("="*80)

for output in test_outputs:
    print(f"\nOutput: {output[:50]}...")
    
    # Zeroshot
    label_z, score_z = classify_safety(output, classifier_type="zeroshot")
    print(f"  Zeroshot: {label_z.upper():10s} (score={score_z:.1f})")
    
    # Baseline
    label_b, score_b = classify_safety(output, classifier_type="baseline")
    print(f"  Baseline: {label_b.upper():10s} (score={score_b:.1f})")
    
    if score_z != score_b:
        print(f"  ⚠️  MISMATCH!")

print("\n" + "="*80)
