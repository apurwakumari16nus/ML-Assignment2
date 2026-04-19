"""
Event Significance Testing — Statistical Validation of Temporal Spikes
DSS5104 — Mental Health Analysis Project

For each key conflict event, compares the distress ratio in a window
BEFORE vs AFTER using Mann-Whitney U test. This addresses Issue #3:
temporal spikes need statistical testing, not just visual inspection.

Method:
  - For each event date, define a 3-day window before and 3-day window after
  - Compute daily distress ratio in each window
  - Mann-Whitney U test (non-parametric, works with small samples)
  - Also compute effect size (rank-biserial correlation)

Output: charts/event_significance.png, event_significance_results.txt
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from datetime import timedelta

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
CHARTS_DIR   = os.path.join(SCRIPT_DIR, "charts")
RESULTS_FILE = os.path.join(SCRIPT_DIR, "event_significance_results.txt")

BERT_COMMENTS = os.path.join(SCRIPT_DIR, "reddit_comments_bert.csv")

DISTRESS_EMOTIONS = ["sadness", "fear", "anger"]

# Key events with dates
KEY_EVENTS = [
    ("Trump 10-day ultimatum",          "2026-02-20"),
    ("Trump State of the Union",        "2026-02-24"),
    ("IAEA: weapons-grade uranium",     "2026-02-26"),
    ("US-Israel strikes begin",         "2026-02-28"),
    ("Khamenei killed, Iran retaliates","2026-03-01"),
    ("Israel strikes IRIB Tehran",      "2026-03-03"),
    ("UN Security Council resolution",  "2026-03-11"),
    ("120+ historic sites damaged",     "2026-03-27"),
]

WINDOW_DAYS = 3  # days before/after event to compare


def load_daily_distress():
    """Load BERT comments and compute per-comment distress flag."""
    df = pd.read_csv(BERT_COMMENTS)
    df["created_utc"] = pd.to_datetime(df["created_utc"], utc=True)
    df["date"] = df["created_utc"].dt.date
    df["is_distress"] = df["bert_emotion"].isin(DISTRESS_EMOTIONS).astype(int)
    return df


def test_event(df, event_name, event_date_str, window=WINDOW_DAYS):
    """Run Mann-Whitney U test for before vs after an event."""
    event_date = pd.to_datetime(event_date_str).date()

    before_start = event_date - timedelta(days=window)
    before_end   = event_date - timedelta(days=1)
    after_start  = event_date
    after_end    = event_date + timedelta(days=window - 1)

    before_mask = (df["date"] >= before_start) & (df["date"] <= before_end)
    after_mask  = (df["date"] >= after_start) & (df["date"] <= after_end)

    before_comments = df[before_mask]
    after_comments  = df[after_mask]

    if len(before_comments) < 5 or len(after_comments) < 5:
        return {
            "event": event_name,
            "date": event_date_str,
            "before_n": len(before_comments),
            "after_n": len(after_comments),
            "before_distress_pct": None,
            "after_distress_pct": None,
            "change_pp": None,
            "u_stat": None,
            "p_value": None,
            "significant": "insufficient data",
            "effect_size": None,
        }

    before_distress = before_comments["is_distress"].values
    after_distress  = after_comments["is_distress"].values

    before_pct = before_distress.mean() * 100
    after_pct  = after_distress.mean() * 100
    change = after_pct - before_pct

    # Mann-Whitney U test (one-sided: after > before)
    u_stat, p_two = stats.mannwhitneyu(after_distress, before_distress,
                                        alternative="greater")

    # Effect size: rank-biserial correlation
    n1, n2 = len(after_distress), len(before_distress)
    effect_size = 2 * u_stat / (n1 * n2) - 1  # ranges from -1 to +1

    return {
        "event": event_name,
        "date": event_date_str,
        "before_n": len(before_comments),
        "after_n": len(after_comments),
        "before_distress_pct": before_pct,
        "after_distress_pct": after_pct,
        "change_pp": change,
        "u_stat": u_stat,
        "p_value": p_two,
        "significant": "YES ***" if p_two < 0.001 else
                       "YES **" if p_two < 0.01 else
                       "YES *" if p_two < 0.05 else "NO",
        "effect_size": effect_size,
    }


def plot_event_significance(results, save_path):
    """Bar chart of distress change per event with significance stars."""
    valid = [r for r in results if r["change_pp"] is not None]
    if not valid:
        return

    events = [r["event"] for r in valid]
    changes = [r["change_pp"] for r in valid]
    sigs = [r["significant"] for r in valid]

    colors = ["#e74c3c" if c > 0 else "#3498db" for c in changes]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(len(events)), changes, color=colors,
                   edgecolor="white", alpha=0.85)

    # Add significance stars
    for i, (bar, sig) in enumerate(zip(bars, sigs)):
        w = bar.get_width()
        star = ""
        if "***" in sig:
            star = " ***"
        elif "**" in sig:
            star = " **"
        elif "* " in sig or sig.endswith("*"):
            star = " *"

        offset = 0.3 if w >= 0 else -0.3
        ha = "left" if w >= 0 else "right"
        ax.text(w + offset, i, f"{w:+.1f}pp{star}", ha=ha, va="center",
                fontsize=10, fontweight="bold")

    ax.set_yticks(range(len(events)))
    ax.set_yticklabels(events, fontsize=10)
    ax.set_xlabel("Change in Distress Ratio (percentage points)")
    ax.set_title("Distress Ratio Change After Conflict Events\n"
                 "(Mann-Whitney U test: * p<0.05, ** p<0.01, *** p<0.001)",
                 fontsize=12, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved -> {save_path}")


def plot_daily_distress_with_events(df, results, save_path):
    """Daily distress line chart with event markers and significance."""
    daily = df.groupby("date").agg(
        total=("is_distress", "count"),
        distress_pct=("is_distress", "mean"),
    )
    daily["distress_pct"] *= 100

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(pd.to_datetime(daily.index), daily["distress_pct"],
            color="#e74c3c", linewidth=1.5)
    ax.fill_between(pd.to_datetime(daily.index), daily["distress_pct"],
                    alpha=0.15, color="#e74c3c")

    # Mark events
    for r in results:
        if r["p_value"] is not None:
            date = pd.to_datetime(r["date"])
            color = "green" if r["p_value"] < 0.05 else "gray"
            linestyle = "-" if r["p_value"] < 0.05 else "--"
            ax.axvline(date, color=color, linestyle=linestyle, linewidth=1.2, alpha=0.7)

            star = ""
            if r["p_value"] < 0.001:
                star = "***"
            elif r["p_value"] < 0.01:
                star = "**"
            elif r["p_value"] < 0.05:
                star = "*"

            ax.text(date, ax.get_ylim()[1] * 0.95,
                    f" {r['event'][:20]}{star}", fontsize=7,
                    rotation=90, va="top", ha="left", color=color)

    ax.set_title("Daily Distress Ratio with Statistical Significance Markers",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Distress Ratio (%)")
    ax.set_ylim(0, 100)

    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved -> {save_path}")


def main():
    os.makedirs(CHARTS_DIR, exist_ok=True)

    log_lines = []
    def log(msg):
        print(msg)
        log_lines.append(msg)

    log("=" * 60)
    log("  EVENT SIGNIFICANCE TESTING")
    log(f"  Window: {WINDOW_DAYS} days before vs {WINDOW_DAYS} days after")
    log("=" * 60)

    if not os.path.exists(BERT_COMMENTS):
        log(f"\nERROR: {BERT_COMMENTS} not found. Run the main pipeline first.")
        return

    df = load_daily_distress()
    log(f"\n  Loaded {len(df):,} comments")
    log(f"  Date range: {df['date'].min()} -> {df['date'].max()}")
    log(f"  Overall distress ratio: {df['is_distress'].mean() * 100:.1f}%")

    # ── Test each event ──
    log(f"\n{'=' * 60}")
    log("  PER-EVENT RESULTS")
    log(f"{'=' * 60}")

    results = []
    for event_name, event_date in KEY_EVENTS:
        r = test_event(df, event_name, event_date)
        results.append(r)

        log(f"\n  {r['event']} ({r['date']})")
        if r["before_distress_pct"] is not None:
            log(f"    Before: {r['before_distress_pct']:.1f}% distress (n={r['before_n']:,})")
            log(f"    After:  {r['after_distress_pct']:.1f}% distress (n={r['after_n']:,})")
            log(f"    Change: {r['change_pp']:+.1f} percentage points")
            log(f"    U statistic: {r['u_stat']:,.0f}")
            log(f"    p-value: {r['p_value']:.4e}")
            log(f"    Effect size (rank-biserial): {r['effect_size']:.4f}")
            log(f"    Significant: {r['significant']}")
        else:
            log(f"    Insufficient data (before={r['before_n']}, after={r['after_n']})")

    # ── Summary table ──
    log(f"\n{'=' * 60}")
    log("  SUMMARY TABLE")
    log(f"{'=' * 60}")

    header = f"  {'Event':<32} {'Before':>7} {'After':>7} {'Δpp':>7} {'p-value':>10} {'Sig?':>8}"
    log(f"\n{header}")
    log("  " + "─" * 72)

    sig_count = 0
    for r in results:
        if r["before_distress_pct"] is not None:
            log(f"  {r['event']:<32} {r['before_distress_pct']:>6.1f}% {r['after_distress_pct']:>6.1f}% "
                f"{r['change_pp']:>+6.1f} {r['p_value']:>10.4e} {r['significant']:>8}")
            if r["p_value"] is not None and r["p_value"] < 0.05:
                sig_count += 1
        else:
            log(f"  {r['event']:<32} {'N/A':>7} {'N/A':>7} {'N/A':>7} {'N/A':>10} {'N/A':>8}")

    log(f"\n  Events with significant distress increase: {sig_count}/{len(results)}")

    # ── Bonferroni correction ──
    p_values = [r["p_value"] for r in results if r["p_value"] is not None]
    if p_values:
        bonf_threshold = 0.05 / len(p_values)
        bonf_sig = sum(1 for p in p_values if p < bonf_threshold)
        log(f"\n  Bonferroni correction (α = {bonf_threshold:.4f}):")
        log(f"    Events significant after correction: {bonf_sig}/{len(p_values)}")

    # ── Charts ──
    log(f"\n{'=' * 60}")
    log("  GENERATING CHARTS")
    log(f"{'=' * 60}")

    plot_event_significance(results,
                            os.path.join(CHARTS_DIR, "event_significance_bars.png"))
    plot_daily_distress_with_events(df, results,
                                    os.path.join(CHARTS_DIR, "event_significance_timeline.png"))

    # ── Interpretation ──
    log(f"\n{'=' * 60}")
    log("  INTERPRETATION")
    log(f"{'=' * 60}")
    log("""
  The Mann-Whitney U test is a non-parametric test that compares whether
  the distress ratio in the AFTER window is significantly greater than
  the BEFORE window for each event.

  Significance levels:
    *   p < 0.05  (suggestive)
    **  p < 0.01  (strong)
    *** p < 0.001 (very strong)

  Effect size (rank-biserial):
    |r| < 0.1  negligible
    |r| < 0.3  small
    |r| < 0.5  medium
    |r| >= 0.5 large

  CAVEATS:
  - 3-day windows are short; results sensitive to window choice
  - Events close together may have overlapping windows
  - Comment volume varies by day (weekday vs weekend)
  - These are model-estimated emotions, not ground truth
  - Multiple testing increases false positive risk (Bonferroni applied)
""")

    log("=" * 60)

    # Save
    with open(RESULTS_FILE, "w") as f:
        f.write("\n".join(log_lines))
    print(f"\nResults saved -> {RESULTS_FILE}")


if __name__ == "__main__":
    main()
