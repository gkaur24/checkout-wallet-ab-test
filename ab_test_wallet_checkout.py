"""
A/B Test Project: Digital Wallet Checkout  (SIMULATED DATA)
-----------------------------------------------------------
Scenario: an e-commerce checkout adds a one-tap digital wallet button (treatment).
Control = card entry only.

Pipeline:
  1. Design      -> hypothesis, metrics, power analysis, test duration
  2. Simulate    -> reproducible fake data (fixed seed)
  3. Validity    -> sample ratio mismatch (SRM) + covariate balance
  4. Analysis    -> two-proportion z-test, lift, 95% CI, secondary + guardrail metrics
  5. Segments    -> exploratory cuts with a multiple-comparison correction
  6. Readout     -> printed summary, chart, CSV outputs

Setup:  pip install numpy pandas scipy statsmodels matplotlib
Run:    python ab_test_wallet_checkout.py
"""

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize, proportions_ztest

# ----------------------------------------------------------------------------
# 0. Config (edit these to explore different scenarios)
# ----------------------------------------------------------------------------
SEED = 42
ALPHA = 0.05
POWER = 0.80

BASELINE_CONVERSION = 0.60      # assumed control conversion (completed orders / checkout starts)
MDE = 0.02                      # minimum detectable effect, absolute (+2 percentage points)
DAILY_CHECKOUT_STARTS = 5_000   # assumed daily traffic entering the experiment

# "Ground truth" used ONLY to generate data. In real life you never know this.
TRUE_LIFT = 0.025               # average absolute lift from the wallet button
SHARE_MOBILE = 0.65
SHARE_NEW_BUYERS = 0.40

rng = np.random.default_rng(SEED)


# ----------------------------------------------------------------------------
# 1. Design: power analysis and test duration
# ----------------------------------------------------------------------------
def design_experiment():
    effect_size = proportion_effectsize(BASELINE_CONVERSION + MDE, BASELINE_CONVERSION)
    n_per_group = NormalIndPower().solve_power(
        effect_size=effect_size, alpha=ALPHA, power=POWER, ratio=1.0, alternative="two-sided"
    )
    n_per_group = math.ceil(n_per_group)
    days_needed = math.ceil(2 * n_per_group / DAILY_CHECKOUT_STARTS)
    # Run at least 2 full weeks so every weekday appears twice
    test_days = max(days_needed, 14)
    return n_per_group, days_needed, test_days


# ----------------------------------------------------------------------------
# 2. Simulate data
# ----------------------------------------------------------------------------
def simulate(n_users: int, test_days: int) -> pd.DataFrame:
    df = pd.DataFrame({
        "user_id": np.arange(n_users),
        # Random 50/50 assignment (counts will differ slightly, as in real life)
        "group": rng.choice(["control", "treatment"], size=n_users),
        "device": rng.choice(["mobile", "desktop"], size=n_users, p=[SHARE_MOBILE, 1 - SHARE_MOBILE]),
        "buyer_type": rng.choice(["new", "returning"], size=n_users,
                                 p=[SHARE_NEW_BUYERS, 1 - SHARE_NEW_BUYERS]),
        "day": rng.integers(0, test_days, size=n_users),
    })
    df["is_weekend"] = (df["day"] % 7).isin([5, 6])

    # Baseline conversion probability with realistic covariate effects
    p = np.full(n_users, BASELINE_CONVERSION)
    p += np.where(df["device"] == "mobile", -0.04, 0.0675)   # mobile converts worse than desktop
    p += np.where(df["buyer_type"] == "new", -0.06, 0.04)    # new buyers convert worse
    p += np.where(df["is_weekend"], 0.01, 0.0)

    # Heterogeneous treatment effect: wallet helps mobile more than desktop
    lift = np.where(df["device"] == "mobile", TRUE_LIFT * 1.3, TRUE_LIFT * 0.5)
    p = p + np.where(df["group"] == "treatment", lift, 0.0)
    p = np.clip(p, 0.01, 0.99)

    df["converted"] = (rng.random(n_users) < p).astype(int)

    # Secondary metric: seconds to complete checkout (converted users only); wallet is faster
    base_sec = np.where(df["group"] == "treatment", 38, 55)
    secs = rng.lognormal(mean=np.log(base_sec), sigma=0.35)
    df["checkout_seconds"] = np.where(df["converted"] == 1, secs, np.nan)

    # Secondary metric: order value (converted users only); no true effect
    order_value = rng.lognormal(mean=np.log(35), sigma=0.5, size=n_users)
    df["order_value"] = np.where(df["converted"] == 1, order_value, np.nan)

    # Guardrail: payment failure rate (no true effect here; try raising it for treatment!)
    fail_rate = np.where(df["group"] == "treatment", 0.030, 0.030)
    df["payment_failed"] = (rng.random(n_users) < fail_rate).astype(int)

    return df


# ----------------------------------------------------------------------------
# 3. Validity checks
# ----------------------------------------------------------------------------
def check_validity(df: pd.DataFrame):
    print("\n=== VALIDITY CHECKS ===")
    counts = df["group"].value_counts().reindex(["control", "treatment"])
    chi2, p_srm = stats.chisquare(counts.values, f_exp=[counts.sum() / 2] * 2)
    print(f"Group sizes: control={counts['control']:,}  treatment={counts['treatment']:,}")
    print(f"Sample ratio mismatch test: chi2={chi2:.3f}, p={p_srm:.3f} "
          f"-> {'PASS' if p_srm > 0.001 else 'FAIL: investigate assignment before analyzing'}")

    for col in ["device", "buyer_type"]:
        table = pd.crosstab(df[col], df["group"])
        chi2, p, _, _ = stats.chi2_contingency(table)
        print(f"Balance on {col}: p={p:.3f} -> {'PASS' if p > 0.01 else 'CHECK'}")


# ----------------------------------------------------------------------------
# 4. Analysis helpers
# ----------------------------------------------------------------------------
def conversion_test(sub: pd.DataFrame, col="converted"):
    t = sub[sub["group"] == "treatment"][col]
    c = sub[sub["group"] == "control"][col]
    x = np.array([t.sum(), c.sum()])
    n = np.array([len(t), len(c)])
    p_t, p_c = x / n
    z, p_val = proportions_ztest(x, n, alternative="two-sided")
    diff = p_t - p_c
    se = math.sqrt(p_t * (1 - p_t) / n[0] + p_c * (1 - p_c) / n[1])
    zcrit = stats.norm.ppf(1 - ALPHA / 2)
    return {
        "n_treatment": int(n[0]), "n_control": int(n[1]),
        "rate_treatment": p_t, "rate_control": p_c,
        "abs_lift": diff, "rel_lift": diff / p_c,
        "ci_low": diff - zcrit * se, "ci_high": diff + zcrit * se,
        "z": z, "p_value": p_val,
    }


def print_result(name: str, r: dict):
    print(f"\n{name}")
    print(f"  Control   : {r['rate_control']:.2%} (n={r['n_control']:,})")
    print(f"  Treatment : {r['rate_treatment']:.2%} (n={r['n_treatment']:,})")
    print(f"  Abs lift  : {r['abs_lift']*100:+.2f} pts  (95% CI {r['ci_low']*100:+.2f} to {r['ci_high']*100:+.2f})")
    print(f"  Rel lift  : {r['rel_lift']:+.1%}")
    print(f"  z={r['z']:.2f}, p={r['p_value']:.4f}")


def welch_test(df: pd.DataFrame, col: str, label: str, unit: str = ""):
    t = df[(df["group"] == "treatment")][col].dropna()
    c = df[(df["group"] == "control")][col].dropna()
    stat, p = stats.ttest_ind(t, c, equal_var=False)
    print(f"\n{label}")
    print(f"  Control mean: {c.mean():.2f}{unit}  |  Treatment mean: {t.mean():.2f}{unit}  "
          f"|  diff={t.mean() - c.mean():+.2f}{unit}  (Welch t={stat:.2f}, p={p:.4g})")


# ----------------------------------------------------------------------------
# 5. Main
# ----------------------------------------------------------------------------
def main():
    # --- Design
    n_per_group, days_needed, test_days = design_experiment()
    print("=== EXPERIMENT DESIGN ===")
    print(f"Hypothesis: wallet button increases checkout conversion (baseline {BASELINE_CONVERSION:.0%}, "
          f"MDE +{MDE*100:.0f} pts)")
    print(f"Required sample: {n_per_group:,} per group at alpha={ALPHA}, power={POWER:.0%}")
    print(f"At {DAILY_CHECKOUT_STARTS:,} checkout starts/day -> {days_needed} days of traffic; "
          f"plan {test_days} days to cover full weekly cycles")

    # --- Simulate (2 groups x required n)
    df = simulate(n_users=2 * n_per_group, test_days=test_days)
    df.to_csv("simulated_experiment_data.csv", index=False)

    # --- Validity
    check_validity(df)

    # --- Primary metric
    print("\n=== PRIMARY METRIC: CHECKOUT CONVERSION ===")
    primary = conversion_test(df)
    print_result("Overall", primary)

    # --- Secondary + guardrail
    print("\n=== SECONDARY METRICS ===")
    welch_test(df, "checkout_seconds", "Time to complete checkout (converted users)", " s")
    welch_test(df, "order_value", "Average order value (converted users)", " $")

    print("\n=== GUARDRAIL ===")
    guard = conversion_test(df, col="payment_failed")
    print_result("Payment failure rate (want: no meaningful increase)", guard)

    # --- Segments (exploratory)
    print("\n=== SEGMENTS (EXPLORATORY) ===")
    seg_rows = []
    for col in ["device", "buyer_type"]:
        for val in df[col].unique():
            res = conversion_test(df[df[col] == val])
            res["segment"] = f"{col}={val}"
            seg_rows.append(res)
    seg = pd.DataFrame(seg_rows)
    m = len(seg)
    seg["p_bonferroni"] = (seg["p_value"] * m).clip(upper=1.0)
    print(seg[["segment", "n_treatment", "rate_control", "rate_treatment",
               "abs_lift", "p_value", "p_bonferroni"]].round(4).to_string(index=False))
    print(f"(Bonferroni-adjusted for {m} comparisons. Treat segment findings as hypotheses "
          f"for a follow-up test, not conclusions.)")

    # --- Decision rule
    print("\n=== RECOMMENDATION (DRAFT) ===")
    sig = primary["p_value"] < ALPHA
    practical = primary["ci_low"] > 0
    guard_ok = not (guard["p_value"] < ALPHA and guard["abs_lift"] > 0)
    if sig and guard_ok:
        print(f"SHIP: conversion up {primary['abs_lift']*100:.2f} pts "
              f"(95% CI {primary['ci_low']*100:.2f} to {primary['ci_high']*100:.2f}); guardrails healthy.")
    elif not guard_ok:
        print("DO NOT SHIP: guardrail metric degraded. Investigate payment failures first.")
    else:
        print("INCONCLUSIVE: no significant lift. Consider running longer or testing a bigger change.")
    if sig and not practical:
        print("Note: statistically significant but the CI includes tiny effects; check practical significance.")

    # --- Chart
    fig, ax = plt.subplots(figsize=(5, 4))
    rates = [primary["rate_control"], primary["rate_treatment"]]
    errs = []
    for g in ["control", "treatment"]:
        sub = df[df["group"] == g]["converted"]
        se = math.sqrt(sub.mean() * (1 - sub.mean()) / len(sub))
        errs.append(1.96 * se)
    ax.bar(["Control", "Wallet button"], rates, yerr=errs, capsize=6, color=["#999999", "#2a7de1"])
    ax.set_ylabel("Checkout conversion rate")
    ax.set_title("Checkout conversion by group (95% CI)")
    ax.set_ylim(min(rates) - 0.05, max(rates) + 0.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.tight_layout()
    fig.savefig("conversion_by_group.png", dpi=150)

    seg.to_csv("segment_results.csv", index=False)
    print("\nSaved: simulated_experiment_data.csv, segment_results.csv, conversion_by_group.png")


if __name__ == "__main__":
    main()
