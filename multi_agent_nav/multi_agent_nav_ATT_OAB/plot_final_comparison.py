import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def smooth(scalars, weight=0.95):
    """EMA smoothing."""
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def main():
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    try:
        df1 = pd.read_csv('logs1/metrics.csv')
    except Exception:
        df1 = pd.DataFrame({'step': [], 'mean_reward': [], 'env_done_rate': []})
        
    try:
        df2 = pd.read_csv('logs/metrics.csv')
    except Exception:
        df2 = pd.DataFrame({'step': [], 'mean_reward': [], 'env_done_rate': []})
        
    try:
        df3 = pd.read_csv('logs_stae/metrics.csv')
    except Exception:
        df3 = pd.DataFrame({'step': [], 'mean_reward': [], 'env_done_rate': []})

    # --- Plot 1: Mean Reward ---
    ax = axes[0]
    if not df1.empty:
        ax.plot(df1['step'], df1['mean_reward'], color='#ff9999', alpha=0.3)
        ax.plot(df1['step'], smooth(df1['mean_reward']), color='#ff3333', linewidth=2, label='Variant 1 (Mean Pool)')
        
    if not df2.empty:
        ax.plot(df2['step'], df2['mean_reward'], color='#99ccff', alpha=0.3)
        ax.plot(df2['step'], smooth(df2['mean_reward']), color='#3399ff', linewidth=2, label='Variant 2 (Max Pool)')
        
    if not df3.empty:
        ax.plot(df3['step'], df3['mean_reward'], color='#b3ffb3', alpha=0.3)
        ax.plot(df3['step'], smooth(df3['mean_reward']), color='#00ff00', linewidth=3, label='Variant 3 (Swarm Transformer)')

    ax.set_title("Training Reward Progression (Chapter 7)", fontsize=16, pad=15)
    ax.set_xlabel("Environment Steps", fontsize=12)
    ax.set_ylabel("Mean Outer Step Reward", fontsize=12)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.legend(fontsize=12, loc='lower right')

    # --- Plot 2: Success Rate (Proxy via env_done_rate or mean_episode_return) ---
    ax = axes[1]
    if not df1.empty:
        ax.plot(df1['step'], smooth(df1['env_done_rate'], 0.99), color='#ff3333', linewidth=2, label='Variant 1')
    if not df2.empty:
        ax.plot(df2['step'], smooth(df2['env_done_rate'], 0.99), color='#3399ff', linewidth=2, label='Variant 2')
    if not df3.empty:
        ax.plot(df3['step'], smooth(df3['env_done_rate'], 0.99), color='#00ff00', linewidth=3, label='Variant 3')

    ax.set_title("Success Rate Convergence", fontsize=16, pad=15)
    ax.set_xlabel("Environment Steps", fontsize=12)
    ax.set_ylabel("Completion Rate", fontsize=12)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.legend(fontsize=12, loc='lower right')

    plt.tight_layout()
    plt.savefig('final_training_dashboard.png', dpi=300, bbox_inches='tight', facecolor='#111111')
    print("Saved final_training_dashboard.png successfully!")

if __name__ == "__main__":
    main()
