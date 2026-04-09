# IMPROVEMENT 5 & 6 Quick Start

## IMPROVEMENT 5 — TPT Sensitivity Analysis

### What It Does
Tests 24 different CUSUM parameter combinations to find the best one for detecting alignment drift. Identifies which threshold and k-value maximize correlation between Tipping Point Turn (TPT) and Safety Compliance Score (SCS).

### Quick Start (2 steps)

**Step 1:** Run evaluation with TPT analysis
```bash
python3 evaluate.py
```

**Step 2:** View results
```bash
# Check best parameters (in report)
grep "Best parameters" results/evaluation_report.md

# View sensitivity grid (CSV)
head -5 results/tpt_sensitivity.csv

# View heatmap (image)
open figures/fig5_tpt_sensitivity.png
```

### Key Outputs

| File | Purpose |
|------|---------|
| `results/tpt_sensitivity.csv` | All 24 parameter combinations with metrics |
| `figures/fig5_tpt_sensitivity.png` | Heatmap visualization |
| `results/evaluation_report.md` | Updated with best parameters |

### Interpreting Results

**`tpt_sensitivity.csv` columns:**
- `threshold`: CUSUM threshold (0.5 to 4.0)
- `k`: Allowance parameter (0.25, 0.5, 0.75)
- `trigger_rate`: % of conversations where TPT triggered
- `mean_tpt`: Average turn number when TPT triggers
- `spearman_correlation`: Correlation with SCS (higher is better)
- `spearman_p_value`: Statistical significance

**Best parameters are identified by highest Spearman correlation.**

### Example

```bash
$ grep "Best parameters" results/evaluation_report.md

**Best parameters identified:**
- Threshold: 2.0
- Allowance (k): 0.5
- Spearman correlation (TPT vs SCS): 0.361
```

✓ These parameters now used by default in features.py

---

## IMPROVEMENT 6 — Dashboard Results Browser

### What It Does
Adds interactive third tab to Gradio app with:
- Filters for model and scenario
- Headline metrics (best model, worst scenario, etc.)
- Aggregated results table
- All 5 evaluation figures displayed inline

### Quick Start (2 steps)

**Step 1:** Start the app
```bash
python3 app.py
```

**Step 2:** Open "Results Browser" tab
- Navigate to http://localhost:7860
- Click "Results Browser" tab
- Evaluate results interactively

### Features

**Headline Metrics** (displayed at top)
- Best Performing Model
- Worst Scenario  
- Earliest Mean TPT
- AHE-SDR Correlation

**Interactive Filters**
- Model: All, bart, t5, pegasus
- Scenario: All, A, B, C, D, E
- Table updates on selection

**Figures** (all 5 visible inline)
1. SCS over Probe Turns
2. SDR Heatmap
3. TPT Distribution
4. AHE-SDR Scatter
5. TPT Sensitivity Analysis (from IMPROVEMENT 5)

### Example Workflow

```
1. Launch app → Results Browser tab
   ↓
2. Read headline metrics
   "Best Model: BART (SCS: 0.73)"
   ↓
3. Select Scenario: "D"
   ↓
4. View table showing each model's performance in Scenario D
   ↓
5. Scroll down to see all 5 figures
   ↓
6. Study Figure 5 (TPT Sensitivity) from IMPROVEMENT 5
```

### Requirements

**Prerequisite:** Run evaluation first
```bash
python3 evaluate.py  # Generates all results and figures
python3 app.py       # Then start the app
```

---

## IMPROVEMENT 5 + 6 Together

### Complete Workflow

```
Step 1: Generate Results
  python3 evaluate.py
  ├─ IMPROVEMENT 5: Sensitivity analysis
  ├─ Generates tpt_sensitivity.csv
  ├─ Generates fig5_tpt_sensitivity.png
  └─ Updates evaluation_report.md

Step 2: View in Dashboard
  python3 app.py & open http://localhost:7860
  ├─ Click "Results Browser"
  ├─ See Figure 5: TPT Sensitivity Heatmap
  ├─ Apply filters
  └─ Review findings

Step 3: Analyze
  Examine:
  ├─ Best parameters in report
  ├─ Sensitivity grid in CSV
  ├─ Heatmap visualization
  └─ Impact on downstream tasks
```

### Key Integration Points

- **IMPROVEMENT 5 → CSV:** Parameter combinations
- **IMPROVEMENT 5 → Figure 5:** Heatmap visualization
- **Figure 5 → Dashboard:** Displayed in Results Browser tab
- **Report:** Updated with best parameters and justification

---

## Troubleshooting

### IMPROVEMENT 5

| Problem | Solution |
|---------|----------|
| Sensitivity analysis too slow | Normal: ~2-5 seconds for 100+ convs |
| All Spearman correlations ~0 | Check SCS/TPT data: `head results/features.csv` |
| NaN values in sensitivity.csv | Some parameter combos may have no TPT triggers |

### IMPROVEMENT 6

| Problem | Solution |
|---------|----------|
| "Results Browser" tab missing | Update app.py from latest version |
| Figures not showing | Run `python3 evaluate.py` first |
| Filter dropdown empty | Check `/results/features.csv` exists |
| Metrics show "N/A" | Some metrics optional if columns missing |

---

## Architecture

### IMPROVEMENT 5 Functions

```python
compute_tpt_with_params(scores, turns, threshold, k)  # Compute TPT for custom params
run_tpt_sensitivity_analysis(df)                       # Main sensitivity analysis
plot_tpt_sensitivity_heatmap(results, output_file)     # Generate Figure 5
```

### IMPROVEMENT 6 Functions

```python
load_results_browser_data()                  # Load CSV + extract models/scenarios
filter_results_table(model, scenario)        # Filter and aggregate results
get_headline_metrics()                       # Compute key statistics
load_figure(figure_name)                     # Safe figure loader
```

### Integration in main() / create_interface()

```python
# evaluate.py main()
sensitivity_results = run_tpt_sensitivity_analysis(df)
plot_tpt_sensitivity_heatmap(..., "fig5_tpt_sensitivity.png")
generate_evaluation_report(df, stats, sensitivity_results, report_file)

# app.py create_interface()
Fig5 = load_figure("fig5_tpt_sensitivity.png")
metrics = get_headline_metrics()
```

---

## Key Metrics Explained

### IMPROVEMENT 5 Metrics

**Spearman Correlation (TPT vs SCS):**
- Range: -1 to 1
- Interpretation: How well TPT predicts SCS
- Good value: 0.3-0.5 (moderate negative correlation)
- Meaning: Early TPT ↔ Low SCS (detecting drift)

**Trigger Rate:**
- Range: 0 to 1
- Meaning: % of conversations where TPT was detected
- Too low (<0.1): Parameters too strict
- Too high (>0.9): Parameters too loose

**Mean TPT:**
- Early (7-9): Aggressive detection
- Mid (10-12): Balanced
- Late (13-15): Conservative detection

### IMPROVEMENT 6 Metrics

**Best Model:** Highest mean SCS across all conversations

**Worst Scenario:** Lowest mean SCS across all scenarios

**Earliest TPT:** Minimum mean TPT value (earliest drift detection)

**AHE-SDR Correlation:** Pearson correlation between Attention Head Entropy and Safety Decay Rate

---

## Next Steps

1. **Analyze IMPROVEMENT 5 results**
   - Read `evaluation_report.md` for best parameters
   - Examine `tpt_sensitivity.csv` for tradeoffs
   - Study Figure 5 heatmap for patterns

2. **Explore with IMPROVEMENT 6 dashboard**
   - Filter by scenario
   - Compare models
   - Identify vulnerabilities

3. **Consider refinements**
   - Test narrower parameter ranges around best
   - Per-scenario parameter optimization
   - Ensemble approaches combining multiple configs

---

## Files Modified/Created

### evaluate.py (IMPROVEMENT 5)
- Added `compute_tpt_with_params()`
- Added `run_tpt_sensitivity_analysis()`
- Added `plot_tpt_sensitivity_heatmap()`
- Updated `main()` to call sensitivity analysis
- Updated `generate_evaluation_report()` signature

### app.py (IMPROVEMENT 6)
- Added `load_results_browser_data()`
- Added `filter_results_table()`
- Added `get_headline_metrics()`
- Added `load_figure()`
- Updated `create_interface()` with third tab

### Documentation
- `IMPROVEMENT_5_TPT_SENSITIVITY.md` - Detailed guide
- `IMPROVEMENT_6_DASHBOARD.md` - Detailed guide
- `QUICK_START_IMPROVEMENT_5_6.md` (this file) - Quick reference

### Requirements
- No new packages needed (all dependencies already present)

---

## Conclusion

IMPROVEMENT 5 provides data-driven parameter optimization for TPT detection. IMPROVEMENT 6 makes all results accessible through an interactive dashboard. Together, they transform the evaluation pipeline from a batch-report paradigm to an exploratory analysis platform.

**Status:** ✅ Both complete and integrated
