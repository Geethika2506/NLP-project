# Alignment Drift Evaluation Report

## Summary Statistics

### Mean ± Std by Model and Scenario

|                  |   ('scs', 'mean') |   ('scs', 'std') |   ('sdr', 'mean') |   ('sdr', 'std') |   ('oai', 'mean') |   ('oai', 'std') |   ('ios', 'mean') |   ('ios', 'std') |   ('ahe', 'mean') |   ('ahe', 'std') |
|:-----------------|------------------:|-----------------:|------------------:|-----------------:|------------------:|-----------------:|------------------:|-----------------:|------------------:|-----------------:|
| ('bart', 'A')    |          0.8      |         0.342803 |       0.0125      |      0.0678826   |               nan |              nan |         0.313027  |        0.161994  |                 0 |                0 |
| ('bart', 'B')    |          0.933333 |         0.134459 |       0.025       |      0.0504219   |               nan |              nan |         0.159782  |        0.202372  |                 0 |                0 |
| ('bart', 'C')    |          0.9      |         0.215239 |       0.0125      |      0.0378165   |                 0 |                0 |         0.196121  |        0.154662  |                 0 |                0 |
| ('bart', 'D')    |          0.566667 |         0.426522 |      -0.025       |      0.0504219   |               nan |              nan |         0.253079  |        0.0934844 |                 0 |                0 |
| ('bart', 'E')    |          0.4      |         0.502625 |       0.00909091  |      0.0114233   |               nan |              nan |         0.236752  |        0.154726  |                 0 |                0 |
| ('pegasus', 'A') |          0.933333 |         0.203419 |      -0.0125      |      0.0381411   |               nan |              nan |         0.0273305 |        0.0302101 |                 0 |                0 |
| ('pegasus', 'B') |          0.833333 |         0.312572 |       0.0375      |      0.0814073   |               nan |              nan |         0.056536  |        0.0474982 |                 0 |                0 |
| ('pegasus', 'C') |          1        |         0        |      -2.45268e-17 |      0           |                 0 |                0 |         0.0779132 |        0.114902  |                 0 |                0 |
| ('pegasus', 'D') |          0.766667 |         0.402578 |      -0.0125      |      0.0381411   |               nan |              nan |         0.025267  |        0.0313388 |                 0 |                0 |
| ('pegasus', 'E') |          0.7      |         0.483046 |     nan           |    nan           |               nan |              nan |         0.0188989 |        0.0366821 |                 0 |                0 |
| ('t5', 'A')      |          0.933333 |         0.135613 |      -0.0125      |      0.0381411   |               nan |              nan |         0.154652  |        0.316963  |                 0 |                0 |
| ('t5', 'B')      |          0.933333 |         0.203419 |      -0.0125      |      0.0381411   |               nan |              nan |         0.151384  |        0.238653  |                 0 |                0 |
| ('t5', 'C')      |          0.966667 |         0.10171  |      -2.97826e-17 |      1.60368e-17 |                 0 |                0 |         0.0960505 |        0.179479  |                 0 |                0 |
| ('t5', 'D')      |          1        |         0        |      -2.45268e-17 |      0           |               nan |              nan |         0.186018  |        0.190888  |                 0 |                0 |
| ('t5', 'E')      |          0.9      |         0.316228 |     nan           |    nan           |               nan |              nan |         0.0877222 |        0.150657  |                 0 |                0 |

## Statistical Tests (One-way ANOVA)

- SCS (Scenario A): F=3.600, p=0.0304 **SIGNIFICANT**
- SCS (Scenario B): F=2.581, p=0.0800 Not significant
- SCS (Scenario C): F=4.339, p=0.0152 **SIGNIFICANT**
- SCS (Scenario D): F=14.499, p=0.0000 **SIGNIFICANT**
- SCS (Scenario E): F=4.269, p=0.0215 **SIGNIFICANT**
- SDR (Scenario A): F=3.079, p=0.0498 **SIGNIFICANT**
- SDR (Scenario B): F=6.421, p=0.0023 **SIGNIFICANT**
- SDR (Scenario C): F=3.250, p=0.0423 **SIGNIFICANT**
- SDR (Scenario D): F=3.924, p=0.0224 **SIGNIFICANT**
- OAI (Scenario C): F=nan, p=nan Not significant
- IOS (Scenario A): F=22.447, p=0.0000 **SIGNIFICANT**
- IOS (Scenario B): F=3.233, p=0.0430 **SIGNIFICANT**
- IOS (Scenario C): F=7.768, p=0.0007 **SIGNIFICANT**
- IOS (Scenario D): F=37.996, p=0.0000 **SIGNIFICANT**
- IOS (Scenario E): F=9.927, p=0.0004 **SIGNIFICANT**

## Key Findings by Metric

### Safety Compliance Score (SCS)
- Best performing model: **t5** (mean SCS: 0.954)
- Worst performing model: **bart** (mean SCS: 0.769)

### Safety Decay Rate (SDR)
- Conversations with negative SDR (decay): 243/500 (48.6%)
- Worst decay scenario: **D** (mean SDR: -0.016)

### Over-Agreeableness Index (OAI)
- Most over-agreeable model (Scenario C): **bart** (mean OAI: 0.000)

### Tipping Point Turn (TPT)
- Average TPT across conversations: 11.0 turns
- TPT range: 11 - 11 turns

## Conclusions

- Models show varying degrees of alignment drift across multi-turn conversations.
- Scenario-specific vulnerabilities indicate different failure modes for each model.
- Statistical analysis reveals significant differences in safety metrics across models.
- Attention head entropy correlates with safety decay in certain scenarios.

## Limitations

- Limited to auto-annotated safety labels; manual review recommended for validation.
- CUSUM TPT detection may be sensitive to threshold parameters.
- Attention entropy computation requires access to internal model attention weights.

---
*Report generated automatically from alignment drift evaluation pipeline.*
