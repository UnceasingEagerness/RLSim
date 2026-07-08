"""
plot_metrics.py  ─  Rich Training Dashboard
============================================
Generates a comprehensive, thesis-quality multi-panel dashboard
from logs/metrics.csv after a training run completes.

Panels:
  Row 1: Episode Return + Done Rate breakdown (Goal / Collision / Timeout)
  Row 2: Step Reward + Alpha (entropy coefficient)
  Row 3: Critic Loss (Q-Network) + Actor Loss (Policy)
  Footer: Key statistics summary table
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import os
import sys

# ── Load ──────────────────────────────────────────────────────────────────────
CSV_PATH  = "logs/metrics.csv"
SAVE_PATH = "logs/training_dashboard.png"
WINDOW    = 2000   # smoothing window (steps)

if not os.path.exists(CSV_PATH):
    print(f"[ERROR] {CSV_PATH} not found. Run training first.")
    sys.exit(1)

df = pd.read_csv(CSV_PATH)
total_steps = len(df)
print(f"  Loaded {total_steps:,} training steps from {CSV_PATH}")

# ── Smooth ────────────────────────────────────────────────────────────────────
W = min(WINDOW, total_steps // 10 or 1)
def smooth(col):
    return df[col].rolling(window=W, min_periods=1).mean()

df["s_reward"]         = smooth("mean_reward")
df["s_ep_return"]      = smooth("mean_episode_return")
df["s_done_rate"]      = smooth("env_done_rate")
df["s_q_loss"]         = smooth("q_loss")
df["s_a_loss"]         = smooth("a_loss")
df["s_alpha"]          = smooth("alpha")

steps = df["step"].values

# ── Derived stats ─────────────────────────────────────────────────────────────
best_reward     = df["s_reward"].max()
final_reward    = df["s_reward"].iloc[-1]
final_ep_return = df["s_ep_return"].iloc[-1]
final_alpha     = df["alpha"].iloc[-1]
final_q_loss    = df["q_loss"].iloc[-1]
final_a_loss    = df["a_loss"].iloc[-1]
peak_done_rate  = df["s_done_rate"].max()
final_done_rate = df["s_done_rate"].iloc[-1]

# Detect if learning happened: reward in last 20% > first 20%
split = total_steps // 5
early_r = df["mean_reward"].iloc[:split].mean()
late_r  = df["mean_reward"].iloc[-split:].mean()
converged = late_r > early_r + 0.1

# Alpha health
alpha_floor_hit = (df["alpha"] < 0.15).sum() / total_steps

# ── Style ─────────────────────────────────────────────────────────────────────
BG      = "#0d1117"
PANEL   = "#161b22"
GRID    = "#21262d"
CYAN    = "#58d8f5"
GREEN   = "#3fb950"
ORANGE  = "#f78166"
PURPLE  = "#bc8cff"
YELLOW  = "#e3b341"
WHITE   = "#e6edf3"
MUTED   = "#8b949e"

plt.rcParams.update({
    "font.family"      : "monospace",
    "text.color"       : WHITE,
    "axes.facecolor"   : PANEL,
    "figure.facecolor" : BG,
    "axes.edgecolor"   : GRID,
    "xtick.color"      : MUTED,
    "ytick.color"      : MUTED,
    "axes.labelcolor"  : WHITE,
    "grid.color"       : GRID,
    "grid.linewidth"   : 0.6,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
})

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 22), facecolor=BG)
fig.suptitle(
    "MARL Surface Vessel Navigation  ─  Training Dashboard",
    fontsize=18, fontweight="bold", color=WHITE, y=0.98
)

gs = gridspec.GridSpec(
    4, 2, figure=fig,
    hspace=0.45, wspace=0.30,
    left=0.07, right=0.96, top=0.94, bottom=0.06
)

# ── Helper ────────────────────────────────────────────────────────────────────
def style_ax(ax, title, ylabel, xlabel=None):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=WHITE, fontsize=11, fontweight="bold", pad=8)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.grid(True, alpha=0.5)
    ax.tick_params(colors=MUTED, labelsize=8)

def shade_raw(ax, col, color, alpha=0.12):
    ax.fill_between(steps, df[col], alpha=alpha, color=color)

def annotate_final(ax, y_val, color, label):
    ax.axhline(y_val, color=color, linewidth=0.8, linestyle="--", alpha=0.5)
    ax.text(steps[-1]*0.02, y_val, f" {label}: {y_val:.3f}",
            color=color, fontsize=7.5, va="bottom")

# ── ROW 0: Episode Return (left) + Alpha (right) ──────────────────────────────
ax0 = fig.add_subplot(gs[0, 0])
shade_raw(ax0, "mean_episode_return", CYAN)
ax0.plot(steps, df["s_ep_return"], color=CYAN, linewidth=1.8, label="Smoothed")
ax0.axhline(0, color=WHITE, linewidth=0.5, linestyle=":", alpha=0.4)
annotate_final(ax0, final_ep_return, CYAN, "Final")
style_ax(ax0, "Mean Episode Return", "Cumulative Return")
ax0.legend(fontsize=8, loc="lower right", facecolor=PANEL, edgecolor=GRID)

ax1 = fig.add_subplot(gs[0, 1])
ax1.fill_between(steps, df["s_alpha"], alpha=0.15, color=PURPLE)
ax1.plot(steps, df["s_alpha"], color=PURPLE, linewidth=1.8)
ax1.axhline(0.135, color=ORANGE, linewidth=1.0, linestyle="--", alpha=0.8, label="Floor (0.135)")
annotate_final(ax1, final_alpha, PURPLE, "Final α")
style_ax(ax1, "Entropy Coefficient  α  (SAC Temperature)", "Alpha")
ax1.legend(fontsize=8, loc="upper right", facecolor=PANEL, edgecolor=GRID)

# ── ROW 1: Step Reward (left) + Goal Done Rate (right) ───────────────────────
ax2 = fig.add_subplot(gs[1, 0])
shade_raw(ax2, "mean_reward", GREEN)
ax2.plot(steps, df["s_reward"], color=GREEN, linewidth=1.8)
ax2.axhline(0, color=WHITE, linewidth=0.5, linestyle=":", alpha=0.4)
annotate_final(ax2, final_reward, GREEN, "Final")
# Mark best
best_step = df["s_reward"].idxmax()
ax2.scatter(steps[best_step], best_reward, color=YELLOW, zorder=5, s=60,
            label=f"Best: {best_reward:.3f}")
style_ax(ax2, "Mean Per-Step Reward", "Reward / Step")
ax2.legend(fontsize=8, loc="lower right", facecolor=PANEL, edgecolor=GRID)

ax3 = fig.add_subplot(gs[1, 1])
ax3.fill_between(steps, df["s_done_rate"]*100, alpha=0.2, color=GREEN)
ax3.plot(steps, df["s_done_rate"]*100, color=GREEN, linewidth=1.8, label="Episode Done Rate")
ax3.axhline(peak_done_rate*100, color=YELLOW, linewidth=0.8, linestyle="--", alpha=0.6,
            label=f"Peak: {peak_done_rate*100:.2f}%")
annotate_final(ax3, final_done_rate*100, GREEN, "Final %")
style_ax(ax3, "Episode Done Rate  (Goal Reached %)", "Done Rate (%)")
ax3.legend(fontsize=8, loc="upper left", facecolor=PANEL, edgecolor=GRID)

# ── ROW 2: Q-Loss (left) + Actor Loss (right) ────────────────────────────────
ax4 = fig.add_subplot(gs[2, 0])
ax4.fill_between(steps, df["s_q_loss"], alpha=0.15, color=ORANGE)
ax4.plot(steps, df["s_q_loss"], color=ORANGE, linewidth=1.8)
annotate_final(ax4, final_q_loss, ORANGE, "Final")
style_ax(ax4, "Critic Loss  (Q-Network Bellman Error)", "MSE Loss")

ax5 = fig.add_subplot(gs[2, 1])
ax5.fill_between(steps, df["s_a_loss"], alpha=0.15, color=YELLOW)
ax5.plot(steps, df["s_a_loss"], color=YELLOW, linewidth=1.8)
annotate_final(ax5, final_a_loss, YELLOW, "Final")
style_ax(ax5, "Actor Loss  (Policy Gradient)", "Loss")

# ── ROW 3: Statistics Summary Table ──────────────────────────────────────────
ax6 = fig.add_subplot(gs[3, :])
ax6.set_facecolor(PANEL)
ax6.axis("off")

status_color  = GREEN if converged else ORANGE
status_text   = "✅  CONVERGED" if converged else "⚠️  NOT CONVERGED"
alpha_status  = "⚠️  Collapsed (hit floor)" if alpha_floor_hit > 0.5 else "✅  Healthy"

table_data = [
    ["Metric",                    "Value",                           "Status"],
    ["Total Training Steps",      f"{total_steps:,}",               ""],
    ["Convergence",               f"Late μ={late_r:.3f}  Early μ={early_r:.3f}", status_text],
    ["Final Episode Return",      f"{final_ep_return:.3f}",         ""],
    ["Best Smoothed Step Reward", f"{best_reward:.4f}",             ""],
    ["Peak Episode Done Rate",    f"{peak_done_rate*100:.3f}%",     "🎯 Goal Rate"],
    ["Final Alpha (Entropy)",     f"{final_alpha:.6f}",             alpha_status],
    ["Final Q-Loss",              f"{final_q_loss:.4f}",            "Low = good"],
    ["Final Actor Loss",          f"{final_a_loss:.4f}",            ""],
    ["Alpha Floor Fraction",      f"{alpha_floor_hit*100:.1f}% of steps at floor", ""],
]

col_colors = [
    [GRID,  GRID,  GRID],
    *[[PANEL, PANEL, PANEL]] * (len(table_data)-1)
]

tbl = ax6.table(
    cellText    = table_data[1:],
    colLabels   = table_data[0],
    cellLoc     = "center",
    loc         = "center",
    colWidths   = [0.32, 0.38, 0.30],
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)
tbl.scale(1, 1.55)

for (row, col), cell in tbl.get_celld().items():
    cell.set_facecolor(PANEL if row > 0 else "#1f2937")
    cell.set_edgecolor(GRID)
    cell.set_text_props(color=WHITE if row > 0 else CYAN)
    if row > 0 and col == 2:
        txt = table_data[row][2]
        if "✅" in txt:
            cell.set_text_props(color=GREEN)
        elif "⚠️" in txt:
            cell.set_text_props(color=ORANGE)
        elif "🎯" in txt:
            cell.set_text_props(color=YELLOW)
        else:
            cell.set_text_props(color=MUTED)

ax6.set_title("Training Summary Statistics", color=WHITE, fontsize=11,
              fontweight="bold", pad=10)

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
plt.savefig(SAVE_PATH, dpi=200, bbox_inches="tight", facecolor=BG)
print(f"  ✔  Dashboard saved to {SAVE_PATH}")

# ── Console Summary ───────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  TRAINING ANALYSIS")
print("="*55)
print(f"  Total Steps        : {total_steps:,}")
print(f"  Convergence        : {status_text}")
print(f"  Early μ reward     : {early_r:.4f}")
print(f"  Late  μ reward     : {late_r:.4f}")
print(f"  Best step reward   : {best_reward:.4f}")
print(f"  Peak done rate     : {peak_done_rate*100:.3f}%")
print(f"  Final alpha        : {final_alpha:.2e}  ← {alpha_status}")
print(f"  Final Q-loss       : {final_q_loss:.4f}")
print(f"  Final actor loss   : {final_a_loss:.4f}")
print("="*55)

if not converged:
    print("\n  ⚠️  DIAGNOSIS:")
    if final_alpha < 0.15:
        print("  → Alpha hit the floor. Entropy collapsed. Increase alpha floor or LR.")
    if final_q_loss > 50:
        print("  → Q-loss is very high. Reward variance too large. Check reward scale.")
    if peak_done_rate < 0.001:
        print("  → Zero goals reached. Agent never found reward. Check physics/reward.")
else:
    print("\n  🎯  Training looks healthy! Run: python3 run_inference.py")

if __name__ == "__main__":
    pass
