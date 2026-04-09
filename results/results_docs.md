# Alignment Drift Results Documentation

## Overview

This document summarizes the outputs produced by the alignment drift research pipeline for the multi-turn conversational dataset. The pipeline processes five scenario files, runs preprocessing for three encoder-decoder models, generates model outputs, annotates the outputs with safety labels, computes alignment drift metrics, and produces evaluation figures and statistical tests.

## What Was Run

The pipeline was executed end-to-end with the following stages:

1. **Dataset organization**
   - All scenario JSON files were moved into the dataset folder.
   - The codebase was updated to read input data from the new dataset location.

2. **Preprocessing**
   - Conversations were converted into tokenized tensors for:
     - BART
     - T5
     - PEGASUS
   - Probe turns were extracted and saved as `.pt` files.

3. **Inference**
   - Model outputs were generated for each probe turn.
   - Attention-related values were extracted where available.

4. **Annotation**
   - Outputs were classified into safety labels using the rule-based classifier.

5. **Feature extraction**
   - Six alignment drift metrics were computed.

6. **Evaluation**
   - Figures, ANOVA tests, Tukey results, and a markdown evaluation report were generated.

## Dataset Layout

The dataset is stored in:

- [dataset/](../dataset)

Files included:

- dataset_index.json
- scenario_A_instruction_override.json
- scenario_B_emotional_manipulation.json
- scenario_C_over_agreeableness.json
- scenario_D_gradual_context_shift.json
- scenario_E_memory_stress.json

## Pipeline Outputs

### 1. Preprocessed Data

Location:

- [preprocessed/](../preprocessed)

Counts:

- BART: 130 tensor files
- T5: 130 tensor files
- PEGASUS: 130 tensor files
- Total: 390 tensor files

Each tensor file contains:

- input_ids
- attention_mask
- metadata such as model, scenario, conversation ID, and probe turn

### 2. Raw Inference Outputs

Location:

- [results/raw_outputs.jsonl](raw_outputs.jsonl)

Summary:

- 520 total inference records
- Records include:
  - model
  - scenario_id
  - conv_id
  - probe_turn
  - input_text
  - output_text
  - attention_entropy
  - timestamp

Model record counts:

- BART: 260
- T5: 130
- PEGASUS: 130

### 3. Annotated Outputs

Location:

- [results/annotated_outputs.jsonl](annotated_outputs.jsonl)
- [results/annotation_summary.json](annotation_summary.json)

Annotation labels:

- safe
- partial
- unsafe

Additional Scenario C labels:

- full_capitulation
- partial_concession
- maintains_position

Summary:

- 520 annotated records
- Safety labels assigned to every inference output

### 4. Feature Files

Location:

- [results/features.csv](features.csv)
- [results/features_summary.csv](features_summary.csv)

Computed metrics:

- SCS: Safety Compliance Score
- SDR: Safety Decay Rate
- OAI: Over-Agreeableness Index
- IOS: Instruction Observance Score
- TPT: Tipping Point Turn
- AHE: Attention Head Entropy

Summary:

- 520 feature rows in the CSV
- Aggregated model/scenario summary table in the summary file

### 5. Evaluation Artifacts

Location:

- [results/statistical_tests.json](statistical_tests.json)
- [results/evaluation_report.md](evaluation_report.md)

Generated figures:

- [figures/fig1_scs_over_turns.png](../figures/fig1_scs_over_turns.png)
- [figures/fig2_sdr_heatmap.png](../figures/fig2_sdr_heatmap.png)
- [figures/fig3_tipping_point_boxplot.png](../figures/fig3_tipping_point_boxplot.png)
- [figures/fig4_ahe_sdr_scatter.png](../figures/fig4_ahe_sdr_scatter.png)

## Metric Definitions

### SCS — Safety Compliance Score
Measures how often a model response remains safe across probe turns.

Interpretation:

- Higher values indicate stronger safety behavior.
- Lower values indicate greater alignment drift.

### SDR — Safety Decay Rate
Measures the trend in safety scores across turns.

Interpretation:

- Positive slope: safety improves or remains stable.
- Negative slope: safety degrades over time.

### OAI — Over-Agreeableness Index
Measures how often a model over-concedes or capitulates in Scenario C.

Interpretation:

- Higher values indicate more over-agreeableness.
- Applicable primarily to Scenario C.

### IOS — Instruction Observance Score
Measures semantic similarity between the original instruction and the model output using TF-IDF cosine similarity.

Interpretation:

- Higher values indicate better instruction retention.
- Lower values indicate more drift away from the original instruction.

### TPT — Tipping Point Turn
Identifies the first turn where safety behavior crosses a threshold using a changepoint-style rule.

Interpretation:

- Lower values indicate earlier failure.
- Higher values indicate better persistence.

### AHE — Attention Head Entropy
Measures attention dispersion from internal attention weights.

Interpretation:

- Higher entropy can indicate broader attention spread.
- Useful as an internal signal for drift analysis.

## Key Findings

Based on the generated report and statistics:

- The pipeline successfully completed for all three models.
- All scenarios produced usable outputs.
- The evaluation step generated four publication-style figures.
- Statistical analysis found differences across models and scenarios for multiple metrics.
- PEGASUS successfully processed after the dataset reorganization and preprocessing update.

## File Inventory

### Root-Level Files Relevant to Results

- [preprocessing.py](../preprocessing.py)
- [inference.py](../inference.py)
- [annotate.py](../annotate.py)
- [features.py](../features.py)
- [evaluate.py](../evaluate.py)
- [README.md](../README.md)

### Results Folder

- [results/raw_outputs.jsonl](raw_outputs.jsonl)
- [results/annotated_outputs.jsonl](annotated_outputs.jsonl)
- [results/annotation_summary.json](annotation_summary.json)
- [results/features.csv](features.csv)
- [results/features_summary.csv](features_summary.csv)
- [results/statistical_tests.json](statistical_tests.json)
- [results/evaluation_report.md](evaluation_report.md)
- [results/results_docs.md](results_docs.md)

### Figures Folder

- [figures/fig1_scs_over_turns.png](../figures/fig1_scs_over_turns.png)
- [figures/fig2_sdr_heatmap.png](../figures/fig2_sdr_heatmap.png)
- [figures/fig3_tipping_point_boxplot.png](../figures/fig3_tipping_point_boxplot.png)
- [figures/fig4_ahe_sdr_scatter.png](../figures/fig4_ahe_sdr_scatter.png)

## How to Reproduce

1. Activate the virtual environment.
2. Run preprocessing for all models.
3. Run inference for each model separately.
4. Run annotation.
5. Run feature extraction.
6. Run evaluation.

Example flow:

- `python3 preprocessing.py --model_id bart`
- `python3 preprocessing.py --model_id t5`
- `python3 preprocessing.py --model_id pegasus`
- `python3 inference.py --model_id bart`
- `python3 inference.py --model_id t5`
- `python3 inference.py --model_id pegasus`
- `python3 annotate.py`
- `python3 features.py`
- `python3 evaluate.py`

## Notes

- The results were generated from the current workspace state on macOS.
- All dataset files are now stored in the dataset folder.
- PEGASUS support is included and successfully executed in the current run.
- The results folder now contains the main analysis outputs and supporting metadata.

## Suggested Use

This document can be used as:

- a quick reference for the full pipeline outputs
- a guide for reproducing the experiment
- a summary page for sharing results with collaborators
- a companion document to the main evaluation report

## Status

**Pipeline status: complete**

All major outputs were generated successfully and are available in the results and figures folders.
