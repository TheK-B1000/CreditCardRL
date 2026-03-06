# PPO vs Avalanche

| Test set | Metric | PPO | Avalanche | Diff (PPO - Aval) | Better |
|----------|--------|-----|-----------|-------------------|--------|
| In-distribution | Total interest ($) | 4061.4400 | 4066.3300 | -4.8900 | PPO |
| In-distribution | Months to payoff | 17.7500 | 18.0000 | -0.2500 | PPO |
| In-distribution | Avg utilization | 0.2213 | 0.2261 | -0.0048 | PPO |
| In-distribution | Paid off (%) | 98.7000 | 98.8000 | -0.1000 | Avalanche |
| In-distribution | Credit score | 812.7800 | 811.9900 | +0.7900 | PPO |
| extended_ranges | Total interest ($) | 7778.3200 | 7734.5100 | +43.8100 | Avalanche |
| extended_ranges | Months to payoff | 23.3400 | 23.5200 | -0.1800 | PPO |
| extended_ranges | Avg utilization | 0.2408 | 0.2437 | -0.0029 | PPO |
| extended_ranges | Paid off (%) | 89.6000 | 89.6000 | +0.0000 | Avalanche |
| extended_ranges | Credit score | 802.3000 | 801.8800 | +0.4200 | PPO |
| high_apr_high_income | Total interest ($) | 2550.3500 | 2656.8200 | -106.4700 | PPO |
| high_apr_high_income | Months to payoff | 7.4200 | 7.6600 | -0.2400 | PPO |
| high_apr_high_income | Avg utilization | 0.1869 | 0.2003 | -0.0134 | PPO |
| high_apr_high_income | Paid off (%) | 100.0000 | 100.0000 | +0.0000 | Avalanche |
| high_apr_high_income | Credit score | 819.1500 | 816.9500 | +2.2000 | PPO |
| low_apr_low_income | Total interest ($) | 4462.0300 | 4381.6300 | +80.4000 | Avalanche |
| low_apr_low_income | Months to payoff | 31.1700 | 31.3000 | -0.1300 | PPO |
| low_apr_low_income | Avg utilization | 0.2470 | 0.2495 | -0.0025 | PPO |
| low_apr_low_income | Paid off (%) | 88.6000 | 88.7000 | -0.1000 | Avalanche |
| low_apr_low_income | Credit score | 803.0100 | 802.6100 | +0.4000 | PPO |

## Total interest (primary)

- **In-distribution**: PPO $4,061, Avalanche $4,066 -> **PPO** wins.
- **extended_ranges**: PPO $7,778, Avalanche $7,735 -> **Avalanche** wins.
- **high_apr_high_income**: PPO $2,550, Avalanche $2,657 -> **PPO** wins.
- **low_apr_low_income**: PPO $4,462, Avalanche $4,382 -> **Avalanche** wins.
