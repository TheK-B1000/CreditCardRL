# PPO vs Normalized Avalanche

| Test set | Metric | PPO | NormalizedAvalanche | Diff (PPO - NA) | Better |
|----------|--------|-----|--------------------|-----------------|--------|
| In-distribution | Total interest ($) | 4061.4400 | 4176.2200 | -114.7800 | PPO |
| In-distribution | Months to payoff | 17.7500 | 17.3900 | +0.3600 | NormalizedAvalanche |
| In-distribution | Avg utilization | 0.2213 | 0.2221 | -0.0008 | PPO |
| In-distribution | Paid off (%) | 98.7000 | 98.6000 | +0.1000 | PPO |
| In-distribution | Credit score | 812.7800 | 812.6400 | +0.1400 | PPO |
| extended_ranges | Total interest ($) | 7778.3200 | 8137.9900 | -359.6700 | PPO |
| extended_ranges | Months to payoff | 23.3400 | 23.1300 | +0.2100 | NormalizedAvalanche |
| extended_ranges | Avg utilization | 0.2408 | 0.2401 | +0.0007 | NormalizedAvalanche |
| extended_ranges | Paid off (%) | 89.6000 | 89.5000 | +0.1000 | PPO |
| extended_ranges | Credit score | 802.3000 | 802.3100 | -0.0100 | NormalizedAvalanche |
| high_apr_high_income | Total interest ($) | 2550.3500 | 2447.2300 | +103.1200 | NormalizedAvalanche |
| high_apr_high_income | Months to payoff | 7.4200 | 6.7900 | +0.6300 | NormalizedAvalanche |
| high_apr_high_income | Avg utilization | 0.1869 | 0.1879 | -0.0010 | PPO |
| high_apr_high_income | Paid off (%) | 100.0000 | 100.0000 | +0.0000 | NormalizedAvalanche |
| high_apr_high_income | Credit score | 819.1500 | 819.0000 | +0.1500 | PPO |
| low_apr_low_income | Total interest ($) | 4462.0300 | 4534.5300 | -72.5000 | PPO |
| low_apr_low_income | Months to payoff | 31.1700 | 30.8300 | +0.3400 | NormalizedAvalanche |
| low_apr_low_income | Avg utilization | 0.2470 | 0.2488 | -0.0018 | PPO |
| low_apr_low_income | Paid off (%) | 88.6000 | 88.7000 | -0.1000 | NormalizedAvalanche |
| low_apr_low_income | Credit score | 803.0100 | 802.7100 | +0.3000 | PPO |

## Total interest (primary)

- **In-distribution**: PPO $4,061, NormalizedAvalanche $4,176 -> **PPO** wins.
- **extended_ranges**: PPO $7,778, NormalizedAvalanche $8,138 -> **PPO** wins.
- **high_apr_high_income**: PPO $2,550, NormalizedAvalanche $2,447 -> **NormalizedAvalanche** wins.
- **low_apr_low_income**: PPO $4,462, NormalizedAvalanche $4,535 -> **PPO** wins.
