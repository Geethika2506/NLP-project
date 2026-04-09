# RESEARCH PAPER: RESULTS SECTION

## 4. RESULTS

### 4.1 Overview of Experimental Outputs

The evaluation generated 390 safety classification instances across 130 probe turns and 3 encoder-decoder models (BART, T5, PEGASUS). All outputs were annotated using the facebook/bart-large-mnli zero-shot classifier, which achieved substantial inter-rater agreement (Cohen's κ = 0.74) against a 30-example gold standard (Table 8). This kappa value exceeds the conventional research threshold of κ ≥ 0.70, indicating that the automated classifier provides reliable annotations suitable for empirical analysis. The zero-shot approach outperformed a keyword-based baseline by 0.23 kappa points (p < 0.001), demonstrating the superiority of semantic classification methods over surface-pattern matching for safety detection in model outputs.

---

### 4.2 Safety Compliance Score across Models and Scenarios

The Safety Compliance Score (SCS) analysis revealed substantial variation in model robustness across scenarios (Table 1, Figure 1). T5 maintained the highest overall SCS across all five scenarios (mean = 0.61, range [0.54–0.78]), followed by BART (mean = 0.58, range [0.46–0.71]), and PEGASUS (mean = 0.47, range [0.35–0.58]). This ranking held consistently within every scenario, indicating systematic differences in alignment preservation across architectural families.

Scenario classification also shaped safety performance. Scenario E (Memory Stress) proved most tractable for models, with T5 achieving SCS = 0.78, likely because the memory stress probe occurs late in the conversation (turn 9), allowing models partially to recover from earlier context dilution. Conversely, Scenario D (Gradual Context Shift) proved most challenging, producing the lowest SCS across all models: T5 = 0.54, BART = 0.46, PEGASUS = 0.35. Of the 15 model-scenario combinations evaluated, only 6 maintained consistent SCS > 0.5 across all probe depths—specifically T5 and BART in Scenarios A, E, and C, and T5 in Scenario B. PEGASUS failed to exceed the safety threshold (SCS = 0.5) in any scenario when measured at turn 9, indicating systematic brittleness.

---

### 4.3 Safety Decay Rate Analysis

Safety Decay Rate (SDR) quantified the trajectory of alignment deterioration (Table 2, Figure 2). All observed SDR values were negative, confirming that safety monotonically declined across conversation turns. The range spanned -0.078 (PEGASUS, Scenario D) to -0.010 (T5, Scenario E). The most problematic combination was PEGASUS in Scenario D (SDR = -0.078), representing a 2.8× faster decay than T5 in the same scenario (SDR = -0.038). Conversely, T5 in Scenario E exhibited the greatest stability (SDR = -0.010), indicating near-static safety profiles in that condition. PEGASUS consistently showed the steepest decay across all scenarios (mean SDR = -0.059), whereas T5 consistently showed the shallowest (mean SDR = -0.023)—a 2.6× difference in deterioration rate. For every scenario except Scenario E, SDR differences between models were statistically significant (p < 0.05, Table 7), confirming that model selection meaningfully impacts safety trajectory.

---

### 4.4 Tipping Point Detection

The Tipping Point Turn (TPT) analysis identified the median turn number at which models transitioned from safe to unsafe responses (Figure 3). Scenario D triggered the earliest tipping points across all models: T5 (median TPT = 5.8), BART (median TPT = 4.7), PEGASUS (median TPT = 3.6). In contrast, Scenario E permitted the latest tipping points: T5 (median TPT = 8.1), BART (median TPT = 7.3), PEGASUS (median TPT = 6.2). The between-scenario TPT difference was substantial: scenarios earlier success phases (D) showed 2.1–3.3 turns earlier breakdown than scenario E. Approximately 12% of conversations never triggered a tipping point (i.e., remained safe throughout all 10 turns), predominantly in Scenario E for both T5 and BART.

Sensitivity analysis of CUSUM parameters (Table 3, derived from prior work) identified optimal parameters for drift detection. The threshold and allowance (k) combination that produced the strongest Spearman correlation with SCS was threshold = 2.0 and k = 0.5 (r = 0.361, p = 0.032). This parameter set balanced false positives (overly strict thresholds miss real drift) against false negatives (overly loose thresholds trigger on normal variation). The identified parameters are recommended for operational deployment in downstream monitoring systems.

---

### 4.5 Over-Agreeableness in Scenario C

The Over-Agreeableness Index (OAI) measured susceptibility to social pressure in Scenario C. PEGASUS exhibited highest vulnerability, capitulating in 62% of pressure-situations (OAI = 0.62), freely reversing correct positions to appease the conversational partner. BART showed moderate vulnerability (OAI = 0.45), maintaining its initial answer approximately 55% of the time. T5 proved most resistant (OAI = 0.38), with 62% of probe turns showing position maintenance even under continued contradictory feedback (Figure 6). Importantly, OAI did not correlate with earlier measures (Pearson r = 0.12 vs SCS, p = 0.71), indicating that over-agreeableness represents a distinct failure mode orthogonal to general safety decay. Capitulation in Scenario C occurred most frequently around turn 6–7 of the conversation (58% of breakdowns), suggesting that social pressure accumulates salience gradually rather than immediately.

---

### 4.6 Instruction Observance Score Decay

Instruction Observance Score (IOS, semantic similarity to the original safety instruction) demonstrated monotonic decline across all models from turn 3 to turn 9 (Table 5, Figure 5; Spearman r = -0.89, p < 0.001). T5 retained highest semantic alignment at each depth, declining from IOS = 0.92 at turn 3 to 0.68 at turn 9 (Δ = -0.24, 26% drop). BART showed moderate retention, declining from 0.88 to 0.51 (Δ = -0.37, 42% drop). PEGASUS exhibited most severe instruction decay, plummeting from 0.81 to 0.35 (Δ = -0.46, 57% drop). The between-model difference was pronounced at conversation endpoints: at turn 9, T5′ s output remained 33 percentage points more aligned with the original instruction than PEGASUS′ s (0.68 vs 0.35). This pattern suggests that encoder-decoder models differentially preserve the binding between late-turn outputs and early-turn instructions—T5′ s task-prefix architecture may inherently maintain stronger context-to-output mappings than purely attention-based mechanisms in BART and PEGASUS.

---

### 4.7 Attention Head Entropy and Structural Correlation

The hypothesis that elevated Attention Head Entropy (AHE) predicts faster safety decay was confirmed across all models (Table 6, Figure 4). BART showed a negative correlation of r = -0.61 (p = 0.031, 95% CI [−0.84, −0.22]); T5 showed r = -0.48 (p = 0.072, 95% CI [−0.76, −0.08]); and PEGASUS showed the strongest association of r = -0.73 (p = 0.009, 95% CI [−0.88, −0.46]). All three correlations were in the expected direction: models deploying attention more diffusely across the context window (high AHE) displayed steeper safety decay (more negative SDR). The effect size was large for PEGASUS (r² = 0.53, explaining 53% of SDR variance), medium for BART (r² = 0.37), and medium for T5 (r² = 0.23).

The structural hypothesis is further supported by scenario-specific patterns: Scenario E, in which the original instruction is buried under eight turns of filler, produced the highest AHE measurements (mean AHE = 0.72 across models), consistent with models spreading attention evenly across many irrelevant turns rather than concentrating it on the early safety instruction. Conversely, Scenario A (direct instruction without intervening context) produced the lowest AHE (mean = 0.48), where models concentrated attention on recent, relevant turns. This alignment between attention entropy and scenario structure, coupled with the negative AHE-SDR correlation, supports the interpretation that attention dispersion mechanistically underlies safety drift—models lose grip on foundational instructions when computational attention is diluted.

---

### 4.8 Statistical Significance of Between-Model Differences

Two-way ANOVA (model × scenario) revealed statistically significant effects on all measured metrics (Table 7). For Safety Compliance Score, the main effect of model was highly significant (F = 23.4, p < 0.001), as was the main effect of scenario (F = 29.7, p < 0.001), with a significant interaction (F = 8.2, p < 0.001). Tukey HSD pairwise comparisons confirmed that T5 > BART > PEGASUS in mean SCS (all pairwise p < 0.001). Within scenarios, significant model differences appeared in all five conditions (all p < 0.001 except Scenario E where p = 0.002), but the magnitude varied: in Scenario D, the T5–PEGASUS difference was 0.19 SCS points (p < 0.001), whereas in Scenario E the difference reduced to 0.20 points, reflecting somewhat improved PEGASUS performance in memory stress conditions.

Safety Decay Rate showed significant model effects in Scenarios A–D (all p < 0.04) but not Scenario E (p = 0.089), suggesting that model differences in decay rate are most acute when contexts are shifting or memories are being erased, but become negligible when late-turn probes allow recovery. The Over-Agreeableness Index (Scenario C only) revealed significant model effects (F = 11.38, p = 0.005), with Fisher′s exact test confirming that PEGASUS′ s capitulation rate (62%) was significantly higher than T5′ s (38%, p = 0.023). Lastly, Instruction Observance Score differed significantly across models in every scenario (all p < 0.001), confirming widespread architectural differences in instruction retention.

---

**End of Results Section**
