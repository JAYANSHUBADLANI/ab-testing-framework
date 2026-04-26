"""
ab_test.py
----------
Reusable A/B testing framework module.

Functions
---------
- run_ztest()            : Two-proportion z-test for conversion rate experiments
- run_chi_square()       : Chi-square test of independence
- calculate_mde()        : Minimum Detectable Effect for a given experiment setup
- calculate_sample_size(): Required sample size per group for a target MDE
- generate_report()      : Save experiment results to a JSON file

Author : Jayanshu Badlani
GitHub : https://github.com/JAYANSHUBADLANI
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Optional

import numpy as np
from scipy.stats import norm, chi2_contingency
from statsmodels.stats.proportion import proportions_ztest, proportion_confint


# ─────────────────────────────────────────────────────────────────────────────
# run_ztest
# ─────────────────────────────────────────────────────────────────────────────

def run_ztest(
    control_conversions: int,
    control_size: int,
    treatment_conversions: int,
    treatment_size: int,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> dict:
    """
    Run a two-proportion z-test to compare conversion rates.

    Parameters
    ----------
    control_conversions   : Number of conversions in the control group.
    control_size          : Total visitors in the control group.
    treatment_conversions : Number of conversions in the treatment group.
    treatment_size        : Total visitors in the treatment group.
    alpha                 : Significance level (default 0.05).
    alternative           : 'two-sided' | 'larger' | 'smaller'
                            'larger'  → H1: treatment > control
                            'smaller' → H1: treatment < control

    Returns
    -------
    dict with keys: z_stat, p_value, significant, control_cvr,
                    treatment_cvr, absolute_lift, relative_lift,
                    ci_control, ci_treatment, alpha, alternative
    """
    # Conversion rates
    control_cvr   = control_conversions   / control_size
    treatment_cvr = treatment_conversions / treatment_size

    # Map alternative to statsmodels convention
    alt_map = {"two-sided": "two-sided", "larger": "larger", "smaller": "smaller"}
    if alternative not in alt_map:
        raise ValueError(f"alternative must be one of {list(alt_map.keys())}")

    count = np.array([treatment_conversions, control_conversions])
    nobs  = np.array([treatment_size,        control_size])

    z_stat, p_value = proportions_ztest(count, nobs, alternative=alt_map[alternative])

    # 95% confidence intervals for each group
    ci_control   = proportion_confint(control_conversions,   control_size,   alpha=alpha, method="normal")
    ci_treatment = proportion_confint(treatment_conversions, treatment_size, alpha=alpha, method="normal")

    absolute_lift = treatment_cvr - control_cvr
    relative_lift = absolute_lift / control_cvr if control_cvr > 0 else float("nan")

    return {
        "test"           : "two-proportion z-test",
        "alternative"    : alternative,
        "alpha"          : alpha,
        "control_cvr"    : round(control_cvr,   6),
        "treatment_cvr"  : round(treatment_cvr, 6),
        "absolute_lift"  : round(absolute_lift, 6),
        "relative_lift"  : round(relative_lift, 6),
        "z_stat"         : round(float(z_stat),  4),
        "p_value"        : round(float(p_value), 6),
        "significant"    : bool(p_value < alpha),
        "ci_control"     : [round(ci_control[0],   6), round(ci_control[1],   6)],
        "ci_treatment"   : [round(ci_treatment[0], 6), round(ci_treatment[1], 6)],
        "control_n"      : control_size,
        "treatment_n"    : treatment_size,
    }


# ─────────────────────────────────────────────────────────────────────────────
# run_chi_square
# ─────────────────────────────────────────────────────────────────────────────

def run_chi_square(
    control_conversions: int,
    control_size: int,
    treatment_conversions: int,
    treatment_size: int,
    alpha: float = 0.05,
) -> dict:
    """
    Run a chi-square test of independence on a 2×2 contingency table.

    Parameters
    ----------
    control_conversions   : Conversions in control group.
    control_size          : Total visitors in control group.
    treatment_conversions : Conversions in treatment group.
    treatment_size        : Total visitors in treatment group.
    alpha                 : Significance level (default 0.05).

    Returns
    -------
    dict with keys: chi2_stat, p_value, dof, significant, cramers_v,
                    contingency_table, alpha
    """
    contingency = np.array([
        [treatment_conversions, treatment_size - treatment_conversions],
        [control_conversions,   control_size   - control_conversions],
    ])

    chi2_stat, p_value, dof, expected = chi2_contingency(contingency, correction=False)

    # Cramér's V — effect size for chi-square
    n = contingency.sum()
    cramers_v = np.sqrt(chi2_stat / (n * (min(contingency.shape) - 1)))

    return {
        "test"             : "chi-square test of independence",
        "alpha"            : alpha,
        "chi2_stat"        : round(float(chi2_stat), 4),
        "p_value"          : round(float(p_value),   6),
        "dof"              : int(dof),
        "significant"      : bool(p_value < alpha),
        "cramers_v"        : round(float(cramers_v), 6),
        "contingency_table": contingency.tolist(),
        "expected_table"   : [[round(v, 2) for v in row] for row in expected.tolist()],
    }


# ─────────────────────────────────────────────────────────────────────────────
# calculate_mde
# ─────────────────────────────────────────────────────────────────────────────

def calculate_mde(
    baseline_rate: float,
    n_per_group: int,
    alpha: float = 0.05,
    power: float = 0.80,
) -> dict:
    """
    Calculate the Minimum Detectable Effect (MDE) for a given experiment.

    The MDE is the smallest true effect size (absolute lift in conversion rate)
    that the experiment is statistically powered to detect.

    Parameters
    ----------
    baseline_rate : Baseline conversion rate of the control group (0 to 1).
    n_per_group   : Number of observations per group.
    alpha         : Significance level (default 0.05).
    power         : Statistical power (default 0.80 = 80%).

    Returns
    -------
    dict with keys: mde_absolute, mde_relative, baseline_rate,
                    n_per_group, alpha, power
    """
    if not (0 < baseline_rate < 1):
        raise ValueError("baseline_rate must be between 0 and 1 (exclusive).")
    if n_per_group < 1:
        raise ValueError("n_per_group must be a positive integer.")

    z_alpha = norm.ppf(1 - alpha / 2)   # two-sided critical value
    z_beta  = norm.ppf(power)
    p       = baseline_rate
    se      = np.sqrt(2 * p * (1 - p) / n_per_group)
    mde     = (z_alpha + z_beta) * se

    return {
        "mde_absolute" : round(float(mde),               6),
        "mde_relative" : round(float(mde / baseline_rate), 6),
        "baseline_rate": baseline_rate,
        "n_per_group"  : n_per_group,
        "alpha"        : alpha,
        "power"        : power,
    }


# ─────────────────────────────────────────────────────────────────────────────
# calculate_sample_size
# ─────────────────────────────────────────────────────────────────────────────

def calculate_sample_size(
    baseline_rate: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> dict:
    """
    Calculate the required sample size per group to detect a given MDE.

    Uses the two-proportion z-test formula accounting for both alpha and
    power (type I and type II error rates).

    Parameters
    ----------
    baseline_rate : Baseline conversion rate of the control group (0 to 1).
    mde           : Minimum detectable effect — absolute lift in conversion rate.
    alpha         : Significance level (default 0.05).
    power         : Statistical power (default 0.80 = 80%).

    Returns
    -------
    dict with keys: n_per_group, total_n, baseline_rate,
                    treatment_rate, mde, alpha, power
    """
    if not (0 < baseline_rate < 1):
        raise ValueError("baseline_rate must be between 0 and 1 (exclusive).")
    if mde <= 0:
        raise ValueError("mde must be a positive number.")

    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta  = norm.ppf(power)

    p1    = baseline_rate
    p2    = baseline_rate + mde
    p_bar = (p1 + p2) / 2

    numerator   = (z_alpha * np.sqrt(2 * p_bar * (1 - p_bar)) +
                   z_beta  * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    denominator = (p2 - p1) ** 2
    n           = int(np.ceil(numerator / denominator))

    return {
        "n_per_group"    : n,
        "total_n"        : n * 2,
        "baseline_rate"  : baseline_rate,
        "treatment_rate" : round(p2, 6),
        "mde"            : mde,
        "alpha"          : alpha,
        "power"          : power,
    }


# ─────────────────────────────────────────────────────────────────────────────
# generate_report
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(
    experiment_name: str,
    results: dict | list[dict],
    path: str = "results/report.json",
    metadata: Optional[dict] = None,
) -> str:
    """
    Save experiment results as a structured JSON report.

    Parameters
    ----------
    experiment_name : Human-readable name for the experiment.
    results         : Output dict from run_ztest(), run_chi_square(), etc.
                      Can also be a list of result dicts for multi-metric reports.
    path            : File path to write the JSON report (default: results/report.json).
    metadata        : Optional dict of extra fields (e.g. start_date, analyst).

    Returns
    -------
    str : Absolute path to the written report file.
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    report = {
        "experiment"   : experiment_name,
        "generated_at" : datetime.utcnow().isoformat() + "Z",
        "results"      : results,
    }

    if metadata:
        report["metadata"] = metadata

    with open(path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Report saved → {os.path.abspath(path)}")
    return os.path.abspath(path)


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test (run with: python -m framework.ab_test)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running self-test...\n")

    z = run_ztest(560, 5000, 500, 5000, alpha=0.05, alternative="larger")
    print("Z-test result:")
    print(json.dumps(z, indent=2))

    chi = run_chi_square(560, 5000, 500, 5000)
    print("\nChi-square result:")
    print(json.dumps(chi, indent=2))

    mde = calculate_mde(baseline_rate=0.10, n_per_group=5000)
    print("\nMDE result:")
    print(json.dumps(mde, indent=2))

    ss = calculate_sample_size(baseline_rate=0.10, mde=0.02)
    print("\nSample size result:")
    print(json.dumps(ss, indent=2))

    generate_report(
        experiment_name="self-test",
        results={"ztest": z, "chi_square": chi, "mde": mde, "sample_size": ss},
        path="results/self_test_report.json",
    )
    print("\nSelf-test complete.")
