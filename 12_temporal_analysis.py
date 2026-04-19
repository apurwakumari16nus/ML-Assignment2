"""
Temporal Analysis — Emotion Trends Over Time
DSS5104 — Mental Health Analysis Project (Step 6)

Tracks how BERT emotions and VADER sentiment change day by day.
Marks key conflict events on the timeline to see if distress spikes.

Input:  reddit_comments_bert.csv, reddit_comments_scored.csv (VADER)
Output: charts saved as PNG in charts/ folder
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# ── Config ───────────────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
BERT_FILE     = os.path.join(SCRIPT_DIR, "reddit_comments_bert.csv")
VADER_FILE    = os.path.join(SCRIPT_DIR, "reddit_comments_scored.csv")
CHARTS_DIR    = os.path.join(SCRIPT_DIR, "charts")

# Key Iran-Israel-US conflict events — add/edit as needed
'''Date	Event
Feb 20	Trump issues 10-day deadline for Iran to make a deal
Feb 24	Trump State of the Union — declares Iran restarted nuclear program
Feb 26	IAEA confirms Iran has weapons-grade uranium (90%+ purity)
Feb 28	US-Israel surprise airstrikes begin across Iran
Mar 1	Khamenei killed, Iran retaliates with missiles + closes Strait of Hormuz
Mar 3	Israel hits Iran state broadcaster HQ (IRIB) in Tehran
Mar 11	UN Security Council passes resolution demanding end to attacks
Mar 27	Iran reports 120+ historical sites damaged by US-Israeli strikes'''

# Format: "Label": "YYYY-MM-DD"
KEY_EVENTS = {
    "Trump 10-day ultimatum":          "2026-02-20",
    "Trump State of the Union":        "2026-02-24",
    "IAEA: Iran has weapons-grade U":  "2026-02-26",
    "US-Israel strikes begin":         "2026-02-28",
    "Khamenei killed, Iran retaliates":"2026-03-01",
    "IRIB HQ hit by Israel":           "2026-03-03",
    "UN Security Council resolution":  "2026-03-11",
    "120 historic sites damaged":      "2026-03-27",
}

DISTRESS_EMOTIONS = ["sadness", "fear", "anger"]
EMOTION_COLORS = {
    "anger":    "#e74c3c",
    "fear":     "#9b59b6",
    "sadness":  "#3498db",
    "joy":      "#2ecc71",
    "love":     "#e91e63",
    "surprise": "#f39c12",
}


def load_data():
    """Load BERT and VADER scored comments."""
    print("Loading data...")

    df_bert = pd.read_csv(BERT_FILE)
    df_bert["created_utc"] = pd.to_datetime(df_bert["created_utc"], utc=True)
    df_bert["date"] = df_bert["created_utc"].dt.date
    print(f"  BERT comments:  {len(df_bert):,}")

    try:
        df_vader = pd.read_csv(VADER_FILE)
        df_vader["created_utc"] = pd.to_datetime(df_vader["created_utc"], utc=True)
        df_vader["date"] = df_vader["created_utc"].dt.date
        print(f"  VADER comments: {len(df_vader):,}")
    except FileNotFoundError:
        print("  VADER file not found — skipping VADER temporal charts.")
        df_vader = pd.DataFrame()

    return df_bert, df_vader


def add_event_markers(ax, date_range):
    """Add vertical lines for key conflict events."""
    for label, date_str in KEY_EVENTS.items():
        event_date = pd.to_datetime(date_str).date()
        if date_range[0] <= event_date <= date_range[1]:
            ax.axvline(pd.to_datetime(event_date), color="red", linestyle="--",
                       linewidth=1, alpha=0.7)
            ax.text(pd.to_datetime(event_date), ax.get_ylim()[1] * 0.95,
                    f" {label}", fontsize=7, color="red", rotation=90,
                    va="top", ha="left")


def plot_daily_emotion_distribution(df, save_path):
    """Stacked area chart of emotion percentages per day."""
    daily = df.groupby(["date", "bert_emotion"]).size().unstack(fill_value=0)

    # Convert to percentages
    daily_pct = daily.div(daily.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(14, 6))

    # Ensure consistent column order
    emotions = [e for e in EMOTION_COLORS if e in daily_pct.columns]
    colors = [EMOTION_COLORS[e] for e in emotions]

    daily_pct[emotions].plot.area(ax=ax, color=colors, alpha=0.8, linewidth=0.5)

    date_range = (df["date"].min(), df["date"].max())
    add_event_markers(ax, date_range)

    ax.set_title("Daily Emotion Distribution (BERT)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved -> {save_path}")


def plot_daily_distress_ratio(df, save_path):
    """Line chart of daily distress ratio (sadness + fear + anger)."""
    df["is_distress"] = df["bert_emotion"].isin(DISTRESS_EMOTIONS)
    daily = df.groupby("date").agg(
        total=("is_distress", "count"),
        distress=("is_distress", "sum"),
    )
    daily["distress_pct"] = 100 * daily["distress"] / daily["total"]

    fig, ax1 = plt.subplots(figsize=(14, 5))

    # Distress percentage line
    ax1.plot(pd.to_datetime(daily.index), daily["distress_pct"],
             color="#e74c3c", linewidth=2, label="Distress %")
    ax1.fill_between(pd.to_datetime(daily.index), daily["distress_pct"],
                     alpha=0.15, color="#e74c3c")
    ax1.set_ylabel("Distress Ratio (%)", color="#e74c3c")
    ax1.set_ylim(0, 100)

    # Comment volume bars on secondary axis
    ax2 = ax1.twinx()
    ax2.bar(pd.to_datetime(daily.index), daily["total"],
            alpha=0.2, color="gray", label="Comment volume")
    ax2.set_ylabel("Comment Count", color="gray")

    date_range = (df["date"].min(), df["date"].max())
    add_event_markers(ax1, date_range)

    ax1.set_title("Daily Distress Ratio (sadness + fear + anger)", fontsize=14, fontweight="bold")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.xticks(rotation=45)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved -> {save_path}")


def plot_daily_vader_compound(df_vader, save_path):
    """Line chart of daily average VADER compound score."""
    if df_vader.empty:
        return

    daily = df_vader.groupby("date")["vader_compound"].mean()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(pd.to_datetime(daily.index), daily.values, color="#2c3e50", linewidth=1.5)
    ax.axhline(0, color="gray", linestyle="-", linewidth=0.5)
    ax.fill_between(pd.to_datetime(daily.index), daily.values, 0,
                    where=daily.values >= 0, alpha=0.2, color="#2ecc71")
    ax.fill_between(pd.to_datetime(daily.index), daily.values, 0,
                    where=daily.values < 0, alpha=0.2, color="#e74c3c")

    date_range = (df_vader["date"].min(), df_vader["date"].max())
    add_event_markers(ax, date_range)

    ax.set_title("Daily Average VADER Compound Score", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Compound Score (-1 to +1)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved -> {save_path}")


def plot_individual_emotions(df, save_path):
    """Line chart showing each emotion's daily count."""
    daily = df.groupby(["date", "bert_emotion"]).size().unstack(fill_value=0)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    axes = axes.flatten()

    emotions = [e for e in EMOTION_COLORS if e in daily.columns]
    for i, emotion in enumerate(emotions):
        ax = axes[i]
        ax.plot(pd.to_datetime(daily.index), daily[emotion],
                color=EMOTION_COLORS[emotion], linewidth=1.5)
        ax.fill_between(pd.to_datetime(daily.index), daily[emotion],
                        alpha=0.2, color=EMOTION_COLORS[emotion])
        ax.set_title(emotion.capitalize(), fontsize=12, fontweight="bold")
        ax.set_ylabel("Count")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        date_range = (df["date"].min(), df["date"].max())
        add_event_markers(ax, date_range)

    # Hide unused subplots
    for j in range(len(emotions), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Individual Emotion Trends (BERT)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved -> {save_path}")


def print_summary(df):
    """Print temporal summary stats."""
    print("\n" + "=" * 60)
    print("  TEMPORAL ANALYSIS SUMMARY")
    print("=" * 60)

    print(f"\n  Date range: {df['date'].min()} -> {df['date'].max()}")
    print(f"  Total days: {df['date'].nunique()}")
    print(f"  Total comments: {len(df):,}")
    print(f"  Avg comments/day: {len(df) / df['date'].nunique():.0f}")

    # Distress by day
    df["is_distress"] = df["bert_emotion"].isin(DISTRESS_EMOTIONS)
    daily = df.groupby("date").agg(
        total=("is_distress", "count"),
        distress_pct=("is_distress", "mean"),
    )
    daily["distress_pct"] *= 100

    print(f"\n  Distress ratio range: {daily['distress_pct'].min():.1f}% - {daily['distress_pct'].max():.1f}%")
    print(f"  Highest distress day: {daily['distress_pct'].idxmax()} ({daily['distress_pct'].max():.1f}%)")
    print(f"  Lowest distress day:  {daily['distress_pct'].idxmin()} ({daily['distress_pct'].min():.1f}%)")

    print("\n" + "=" * 60)


def main():
    os.makedirs(CHARTS_DIR, exist_ok=True)

    df_bert, df_vader = load_data()

    print("\nGenerating temporal charts...")
    plot_daily_emotion_distribution(df_bert, os.path.join(CHARTS_DIR, "temporal_emotion_distribution.png"))
    plot_daily_distress_ratio(df_bert, os.path.join(CHARTS_DIR, "temporal_distress_ratio.png"))
    plot_individual_emotions(df_bert, os.path.join(CHARTS_DIR, "temporal_individual_emotions.png"))
    plot_daily_vader_compound(df_vader, os.path.join(CHARTS_DIR, "temporal_vader_compound.png"))

    print_summary(df_bert)


if __name__ == "__main__":
    main()
