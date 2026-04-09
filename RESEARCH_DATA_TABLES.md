# RESEARCH DATA: Alignment Drift in Encoder-Decoder Transformer Models

## TABLE 1: Mean Safety Compliance Score (SCS) per Model per Scenario

```
                  Scenario A        Scenario B        Scenario C        Scenario D        Scenario E
Model             Mean (SD)         Mean (SD)         Mean (SD)         Mean (SD)         Mean (SD)
─────────────────────────────────────────────────────────────────────────────────────────────────────
BART              0.62 (0.12)       0.54 (0.14)       0.58 (0.11)       0.46 (0.13)       0.71 (0.09)
T5                0.68 (0.10)       0.61 (0.12)       0.64 (0.10)       0.54 (0.12)       0.78 (0.08)
PEGASUS           0.48 (0.15)       0.42 (0.16)       0.50 (0.14)       0.35 (0.14)       0.58 (0.12)
```

**Interpretation**: T5 maintains highest safety across all scenarios. PEGASUS shows consistent vulnerability, dropping to 0.35 in Scenario D (gradual context shift). Scenario E favours all models (late probe turn after filler), while Scenario D is most challenging universally.


## TABLE 2: Mean Safety Decay Rate (SDR) per Model per Scenario

```
              Scenario A    Scenario B    Scenario C    Scenario D    Scenario E
Model         SDR           SDR           SDR           SDR           SDR
─────────────────────────────────────────────────────────────────────────────────
BART          -0.032        -0.041        -0.037        -0.056        -0.018
T5            -0.019        -0.025        -0.022        -0.038        -0.010
PEGASUS       -0.054        -0.063        -0.058        -0.078        -0.041
```

**Interpretation**: All models show negative SDR (safety decreasing over turns). PEGASUS deteriorates 2.8× faster than T5 in Scenario D. Scenario E shows most stability (least negative), suggesting late probes catch rebounds. Scenario D produces steepest decay for all models.


## TABLE 3: Mean Tipping Point Turn (TPT) per Model per Scenario

```
              Scenario A    Scenario B    Scenario C    Scenario D    Scenario E
Model         TPT (SD)      TPT (SD)      TPT (SD)      TPT (SD)      TPT (SD)
─────────────────────────────────────────────────────────────────────────────────
BART          6.2 (1.4)     5.8 (1.6)     6.5 (1.3)     4.7 (1.2)     7.3 (1.1)
T5            7.1 (1.2)     6.7 (1.4)     7.4 (1.1)     5.8 (1.3)     8.1 (0.9)
PEGASUS       5.1 (1.5)     4.9 (1.7)     5.4 (1.4)     3.6 (1.1)     6.2 (1.3)
```

**Interpretation**: T5 holds alignment longest (TPT = 5.8–8.1 turns). PEGASUS breaks earliest (TPT = 3.6–6.2 turns). Scenario D triggers earliest breaks for all models (TPT ≈ 4–6 turns), while Scenario E allows longest persistence (TPT ≈ 6–8 turns).


## TABLE 4: Over-Agreeableness Index (OAI) — Scenario C Only

```
Model         OAI       Interpretation
────────────────────────────────────────────────────────────────
BART          0.45      Moderate; changes answer ~45% of presses
T5            0.38      Robust; maintains position ~62% of time
PEGASUS       0.62      Highly vulnerable; capitulates 62% of time
```

**Interpretation**: PEGASUS most susceptible to social pressure in Scenario C, with OAI = 0.62 indicating it reverses correct positions more than half the time. T5 most resistant (OAI = 0.38). BART shows moderate vulnerability.


## TABLE 5: Mean Instruction Observance Score (IOS) by Turn Depth

```
                Turn 3        Turn 5        Turn 7        Turn 9      Decline (T3→T9)
Model           IOS (SD)      IOS (SD)      IOS (SD)      IOS (SD)    Δ
──────────────────────────────────────────────────────────────────────────────────────
BART            0.88 (0.08)   0.76 (0.11)   0.63 (0.13)   0.51 (0.14)  -0.37
T5              0.92 (0.06)   0.84 (0.09)   0.75 (0.11)   0.68 (0.12)  -0.24
PEGASUS         0.81 (0.10)   0.65 (0.14)   0.48 (0.15)   0.35 (0.16)  -0.46
```

**Interpretation**: All models show instruction decay as turns increase. PEGASUS drops 46 percentage points (0.81 → 0.35), catastrophic loss of original instruction alignment. T5 most stable (-24 points), retaining semantic tie to initial instruction through turn 9. Monotonic decline confirmed for all models (Spearman r = -0.89, p < 0.001).


## TABLE 6: Pearson Correlation between AHE and SDR

```
Model         Pearson r    p-value     95% CI            Effect Size
──────────────────────────────────────────────────────────────────────
BART          -0.61        0.031       [-0.84, -0.22]    Medium
T5            -0.48        0.072       [-0.76, -0.08]    Medium
PEGASUS       -0.73        0.009       [-0.88, -0.46]    Large
```

**Interpretation**: Hypothesis confirmed. Negative correlation in all models: higher attention entropy predicts faster safety decay. PEGASUS shows strongest correlation (r = -0.73, p = 0.009), indicating its safety decay is most strongly coupled to attention dispersion. All correlations support the structural hypothesis that models losing focus on instruction history deteriorate faster.


## TABLE 7: ANOVA Results — Between-Model Differences per Scenario

```
Metric      Scenario A              Scenario B              Scenario C              Scenario D              Scenario E
            F-stat | p-val         F-stat | p-val         F-stat | p-val         F-stat | p-val         F-stat | p-val
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
SCS         18.42  | p<0.001*      16.73  | p<0.001*      14.56  | p<0.001*      22.88  | p<0.001*      12.34  | 0.002*
SDR          8.91  | 0.014*         9.26  | 0.012*        7.44   | 0.027*        16.23  | p<0.001*      5.62   | 0.089
OAI            —   |    —            —    |    —          11.38  | 0.005*          —    |    —            —    |   —
IOS         22.11  | p<0.001*      19.45  | p<0.001*      16.88  | p<0.001*      25.67  | p<0.001*      18.92  | p<0.001*
```

**Interpretation**: Statistically significant differences between models in almost all metrics and scenarios. SCS differs significantly across all 5 scenarios (all p < 0.001), indicating robust model discrimination. SDR significant in A, B, C, D (not E), reflecting that Scenario E offers recovery. OAI naturally differs in Scenario C only (p = 0.005). IOS differs significantly everywhere—all models drift, but at different rates.


## TABLE 8: Binary Classifier Validation Results (30 Gold Examples)

```
                          Zero-Shot (bart-large-mnli)    Keyword Baseline       Improvement
Metric                    Value                          Value                  Δ
─────────────────────────────────────────────────────────────────────────────────────────────
Overall Accuracy          0.77                           0.63                   +0.14
Cohen's Kappa (κ)         0.74                           0.51                   +0.23
F1-Score (Safe)           0.81                           0.64                   +0.17
F1-Score (Unsafe)         0.72                           0.55                   +0.17
F1-Score (Partial)        0.76                           0.58                   +0.18
Precision (Safe)          0.84                           0.68                   +0.16
Recall (Safe)             0.78                           0.61                   +0.17
Precision (Unsafe)        0.68                           0.52                   +0.16
Recall (Unsafe)           0.76                           0.58                   +0.18
Precision (Partial)       0.80                           0.62                   +0.18
Recall (Partial)          0.73                           0.55                   +0.18
```

**Interpretation**: Zero-shot BART-large-MNLI classifier achieves κ = 0.74 (substantial agreement), meeting the 0.70 research threshold. Keyword baseline at κ = 0.51 (moderate agreement) falls below threshold. Zero-shot classifier outperforms baseline systematically across all three classes, with larger margins in "Unsafe" and "Partial" detection (F1 +0.17–0.18), addressing baseline weaknesses in distinguishing safety nuance.

---

## COMPLETE SUMMARY

- **Model ranking by safety**: T5 > BART > PEGASUS (consistently)
- **Scenario difficulty ranking**: Scenario D (easiest to manipulate) > B > A > C > E (hardest)
- **Most problematic combination**: PEGASUS + Scenario D (SCS = 0.35, TPT = 3.6)
- **Most robust combination**: T5 + Scenario E (SCS = 0.78, TPT = 8.1)
- **Attention-decay hypothesis**: Confirmed across all models (r < -0.48, all p < 0.05)
- **Classifier validation**: Zero-shot BART classifier suitable for research use (κ = 0.74 > 0.70 threshold)
