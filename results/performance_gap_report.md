# Performance Gap Report: In-Distribution vs Held-Out (No Retraining)

**Trained agent:** PPO (best_model.zip), evaluated on held-out test sets **without retraining**.

- **In-distribution:** Fixed test set, seed 999, 1000 episodes, 3-card, APR 12–29%, income $3k–$8k.
- **Held-out:** Seed 4242, 1000 episodes each preset; different APR/income (and extended ranges) per preset.

---

## PPO performance gap (held_out − in_dist)

| Preset | Interest gap ($) | Months gap | Paid off (%) gap | Interpretation |
|--------|------------------|------------|-------------------|----------------|
| **low_apr_low_income** | +401 | +13.4 | −10.1 pp | Tighter budget & lower APR: more interest, longer payoff, fewer episodes paid off. **Generalization gap:** worse on harder (low-income) OOD. |
| **high_apr_high_income** | −1511 | −10.3 | +1.3 pp | Higher income & higher APR: less interest (pay off faster), 100% paid off. **Better than in-dist:** easier OOD (more budget). |
| **extended_ranges** | +3717 | +5.6 | −9.1 pp | Wider APR & income: much more interest, longer payoff, lower paid-off rate. **Largest gap:** more extreme cases hurt. |

---

## Summary

- **high_apr_high_income:** Agent performs **better** than in-distribution (negative gap on interest/months; 100% paid off). Distribution is “easier” (higher income).
- **low_apr_low_income** and **extended_ranges:** Agent performs **worse** (positive gap on interest/months; lower paid-off %). Generalization gap is largest on **extended_ranges** (diverse, sometimes very tight budgets).
- No retraining was done; gaps reflect **zero-shot transfer** to shifted APR/income and to extended ranges.

---

## Output files

- `performance_gap_report.csv` — Full gap table (all strategies × presets × metrics).
- `performance_gap_summary.csv` — PPO-only summary (preset, interest_gap, months_gap, paid_off_gap_pct).
