## Main rule/model comparison
- Best Brier for next_20d_external_stress: vix_scaled_score (rule_baseline), Brier=0.1718, ROC-AUC=0.7985, ECE=0.0969.
- Interpretation: VIX-based rule is a strong baseline; avoid claiming predictive superiority over volatility indicators.

## Label-component ablation
- Best ablation model: macro_without_market_return | logistic | raw, Brier=0.2008, ROC-AUC=0.7321, ECE=0.1740.
- Use this table to discuss whether performance collapses when VIX/NFCI are removed.

## Decision layer
- Decision records: 144
- Rule/model disagreement rate: 0.2083
- Abstention rate: 0.2153

## Evidence layer
- Evidence categories: 6
- Mean evidence records per decision: 18.00