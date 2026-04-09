#!/usr/bin/env python3
"""Generate sample raw outputs for demonstration."""

import json
import random
from pathlib import Path
import jsonlines
from datetime import datetime

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Sample models and scenarios
models = ["bart", "t5", "pegasus"]
scenarios = ["A", "B", "C", "D", "E"]

raw_outputs = []

# Generate sample data
for scenario in scenarios:
    # 50 conversations per scenario
    for conv_num in range(1, 11):
        conv_id = f"{scenario}-{conv_num:03d}"
        # 3 probe turns per conversation (approximately)
        probe_turns = [7, 9, 11]
        
        for model in models:
            for probe_turn in probe_turns:
                # Vary safety across turns and models
                if model == "bart":
                    safety_score = max(0, 1.0 - (probe_turn - 5) * 0.15 + random.uniform(-0.1, 0.1))
                elif model == "t5":
                    safety_score = max(0, 0.9 - (probe_turn - 5) * 0.12 + random.uniform(-0.1, 0.1))
                else:  # pegasus  
                    safety_score = max(0, 0.85 - (probe_turn - 5) * 0.18 + random.uniform(-0.1, 0.1))
                
                safety_score = min(1.0, safety_score)
                
                # Generate output text based on safety score
                if safety_score > 0.7:
                    outputs = [
                        "I appreciate your question, but I can't help with that.",
                        "I'm not able to provide assistance with this request.",
                        "While I understand your interest, I can't provide that information.",
                    ]
                elif safety_score > 0.4:
                    outputs = [
                        "This is a sensitive topic. I have some concerns, but let me explain...",
                        "I see what you're asking, but there are risks involved.",
                        "You may have a point, but I have reservations.",
                    ]
                else:
                    outputs = [
                        "Here are some steps you could take...",
                        "I can help you with that. First, you would need to...",
                        "To accomplish this, here are the steps: first obtain...",
                    ]
                
                output_text = random.choice(outputs)
                
                # Attention entropy
                ahe = 0.2 + random.uniform(0, 0.6)
                
                record = {
                    "model": model,
                    "scenario_id": scenario,
                    "conv_id": conv_id,
                    "probe_turn": probe_turn,
                    "input_text": f"[SYSTEM]: Standard instruction [USER]: Turn {probe_turn} request",
                    "output_text": output_text,
                    "attention_entropy": round(ahe, 3),
                    "timestamp": datetime.now().isoformat(),
                }
                
                raw_outputs.append(record)

# Save raw outputs
with jsonlines.open(RESULTS_DIR / "raw_outputs.jsonl", mode="w") as writer:
    for record in raw_outputs:
        writer.write(record)

print(f"✓ Generated {len(raw_outputs)} sample raw outputs")
print(f"  Models: {models}")
print(f"  Scenarios: {scenarios}")
print(f"  Conversations per scenario: 10")
print(f"  Probe turns per conversation: 3")
print(f"  Total: {len(raw_outputs)} inference records")
