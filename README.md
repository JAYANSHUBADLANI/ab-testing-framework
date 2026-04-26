# A/B Testing Framework

A clean, reusable statistical A/B testing framework built in Python. Covers hypothesis testing (z-test, chi-square), confidence intervals, minimum detectable effect (MDE), sample size planning, and automated JSON reporting — all wrapped in a modular framework you can drop into any project.

---

## Project Structure

```
ab-testing-framework/
├── framework/
│   └── ab_test.py            # Core reusable module
├── notebooks/
│   ├── 01_ab_test_analysis.ipynb   # Full A/B test walkthrough (landing page)
│   └── 02_framework_demo.ipynb     # Framework demo on 3 scenarios
├── visuals/                  # Saved plots (PNG)
├── results/
│   └── report.json           # Experiment results
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Category | Tools |
|---|---|
| Statistical testing | scipy, statsmodels |
| Data manipulation | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Environment | Jupyter, Python 3.10+ |

---

## Use Cases

| Scenario | Test Used | Framework Function |
|---|---|---|
| Landing page conversion rate | Two-proportion z-test | `run_ztest()` |
| Button colour click-through | Chi-square test | `run_chi_square()` |
| Pre-experiment planning | Sample size + MDE | `calculate_sample_size()`, `calculate_mde()` |
| Automated reporting | JSON summary | `generate_report()` |

---

## Framework API

```python
from framework.ab_test import (
    run_ztest,
    run_chi_square,
    calculate_mde,
    calculate_sample_size,
    generate_report
)

# Two-proportion z-test
result = run_ztest(
    control_conversions=500, control_size=5000,
    treatment_conversions=560, treatment_size=5000,
    alpha=0.05
)

# Sample size planning
n = calculate_sample_size(baseline_rate=0.10, mde=0.02, alpha=0.05, power=0.80)

# Generate JSON report
generate_report(result, path="results/report.json")
```

---

## How to Run

### 1. Clone the repo
```bash
git clone https://github.com/JAYANSHUBADLANI/ab-testing-framework.git
cd ab-testing-framework
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run notebooks in order
```bash
jupyter notebook
```
- `notebooks/01_ab_test_analysis.ipynb` — full A/B test walkthrough
- `notebooks/02_framework_demo.ipynb` — framework demo on multiple scenarios

---

## Author

**Jayanshu Badlani**
[GitHub](https://github.com/JAYANSHUBADLANI) | [LinkedIn](https://linkedin.com/in/jayanshu-badlani)
