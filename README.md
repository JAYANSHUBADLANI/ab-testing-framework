# A/B Testing Framework

A clean, reusable statistical A/B testing framework built in Python. Covers hypothesis testing (z-test, chi-square), confidence intervals, minimum detectable effect (MDE), sample size planning, and automated JSON reporting, all wrapped in a modular framework you can drop into any project.

## Project Structure

```
ab-testing-framework/
├── framework/
│   ├── __init__.py
│   └── ab_test.py            # Core reusable module
├── notebooks/
│   ├── 01_ab_test_analysis.ipynb   # Full A/B test walkthrough (landing page)
│   └── 02_framework_demo.ipynb     # Framework demo on 3 scenarios
├── visuals/                  # Saved plots (PNG)
├── results/
│   └── report.json           # Experiment results output
├── requirements.txt
└── README.md
```

## Tech Stack

| Category | Tools |
|---|---|
| Statistical testing | scipy, statsmodels |
| Data manipulation | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Environment | Jupyter, Python 3.10+ |

## Framework API - `framework/ab_test.py`

### `run_ztest()`
Two-proportion z-test for conversion rate experiments.

```python
from framework.ab_test import run_ztest

result = run_ztest(
    control_conversions   = 500,
    control_size          = 5000,
    treatment_conversions = 560,
    treatment_size        = 5000,
    alpha                 = 0.05,
    alternative           = "larger"   # "two-sided" | "larger" | "smaller"
)
# Returns: z_stat, p_value, significant, absolute_lift, relative_lift, CIs
```

### `run_chi_square()`
Chi-square test of independence on a 2×2 contingency table.

```python
from framework.ab_test import run_chi_square

result = run_chi_square(
    control_conversions   = 500,
    control_size          = 5000,
    treatment_conversions = 560,
    treatment_size        = 5000,
)
# Returns: chi2_stat, p_value, dof, significant, cramers_v
```

### `calculate_mde()`
Minimum Detectable Effect for a given experiment setup.

```python
from framework.ab_test import calculate_mde

result = calculate_mde(
    baseline_rate = 0.10,    # 10% baseline CVR
    n_per_group   = 5000,
    alpha         = 0.05,
    power         = 0.80,
)
# Returns: mde_absolute, mde_relative
```

### `calculate_sample_size()`
Required sample size per group to detect a target MDE.

```python
from framework.ab_test import calculate_sample_size

result = calculate_sample_size(
    baseline_rate = 0.10,
    mde           = 0.02,    # detect a 2pp absolute lift
    alpha         = 0.05,
    power         = 0.80,
)
# Returns: n_per_group, total_n
```

### `generate_report()`
Save experiment results as a structured JSON report.

```python
from framework.ab_test import generate_report

generate_report(
    experiment_name = "Homepage Redesign Q2",
    results         = {"ztest": result, "chi_square": chi_result},
    path            = "results/report.json",
    metadata        = {"analyst": "Jayanshu Badlani", "date": "2026-04-27"}
)
```

## Use Cases

| Scenario | Test | Functions Used |
|---|---|---|
| Landing page conversion | Two-proportion z-test | `run_ztest()`, `run_chi_square()` |
| Email subject line CTR | One-sided z-test | `run_ztest(alternative="larger")` |
| Button colour click-through | Chi-square | `run_chi_square()` |
| Pre-experiment planning | Sample size + MDE | `calculate_sample_size()`, `calculate_mde()` |
| Automated reporting | JSON export | `generate_report()` |

## Notebooks

| Notebook | Description |
|---|---|
| `01_ab_test_analysis.ipynb` | Full walkthrough of a landing page A/B test: z-test, chi-square, MDE, sample size planning, and 4 visualizations |
| `02_framework_demo.ipynb` | Demonstrates all 5 framework functions across 3 realistic business scenarios with combined JSON report |

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
- `notebooks/01_ab_test_analysis.ipynb`
- `notebooks/02_framework_demo.ipynb`

### 4. Run the framework directly (self-test)
```bash
python -m framework.ab_test
```

## Author

**Jayanshu Badlani**
[GitHub](https://github.com/JAYANSHUBADLANI) | [LinkedIn](https://linkedin.com/in/jayanshu-badlani)
