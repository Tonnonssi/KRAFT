import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_training_curves(ax, train_rewards, train_losses):
    """학습 곡선 - 훈련 중 보상과 손실 변화"""
    ax2 = ax.twinx()  # 두 번째 y축 생성
    
    # 보상 곡선 (왼쪽 y축)
    episodes = range(len(train_rewards))
    line1 = ax.plot(episodes, train_rewards, 'b-', alpha=0.3, label='Episode Rewards')
    
    # 이동평균으로 트렌드 표시
    if len(train_rewards) > 10:
        window = min(50, len(train_rewards) // 10)
        moving_avg = np.convolve(train_rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(train_rewards)), moving_avg, 'b-', 
               linewidth=2, label=f'Reward MA({window})')
    
    # 손실 곡선 (오른쪽 y축, None 값 제거)
    valid_losses = [(i, loss) for i, loss in enumerate(train_losses) if loss is not None]
    if valid_losses:
        loss_episodes, loss_values = zip(*valid_losses)
        line2 = ax2.plot(loss_episodes, loss_values, 'r-', alpha=0.6, label='Training Loss')
        ax2.set_ylabel('Loss', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
    
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Reward', color='b')
    ax.tick_params(axis='y', labelcolor='b')
    ax.set_title('Training Progress: Rewards vs Loss')
    
    # 범례 합치기
    lines1, labels1 = ax.get_legend_handles_labels()
    if valid_losses:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    else:
        ax.legend(loc='upper left')
    
    ax.grid(True, alpha=0.3)

def plot_event_per_episodes(ax, event_list):
    """
    에피소드별 이벤트 빈도 데이터를 받아 히트맵으로 시각화합니다.
    Args:
        ax (matplotlib.axes.Axes): 시각화를 그릴 Matplotlib의 Axes 객체.
        event_list (list): 각 에피소드의 이벤트 Counter 또는 dict.
                           ex) [ {'A': 10, 'B': 5}, {'A': 2, 'C': 120} ]
    """
    records = []
    episode_labels = []
    for ep_idx, events in enumerate(event_list):
        label = f'Ep. {ep_idx + 1}'
        episode_labels.append(label)
        if not events:
            continue
        for event_name, freq in events.items():
            if not event_name:
                continue
            records.append({
                'episode': label,
                'event_name': event_name,
                'frequency': freq
            })
    
    if not records:
        ax.text(0.5, 0.5, "표시할 데이터가 없습니다.", ha='center', va='center')
        ax.set_axis_off()
        return

    df = pd.DataFrame(records)

    # Pivot → Heatmap 그대로 동일
    df['episode'] = pd.Categorical(df['episode'], categories=episode_labels, ordered=True)

    heatmap_data = df.pivot_table(
        index='event_name',
        columns='episode',
        values='frequency',
        fill_value=0,
        observed=False
    )
    heatmap_data = heatmap_data.reindex(columns=episode_labels, fill_value=0)

    def normalize_row_with_clipping(row):
        if row.max() == 0:
            return pd.Series(0, index=row.index)
        v_max = row.quantile(0.95)
        v_min = row.quantile(0.05)
        clipped_row = row.clip(v_min, v_max)
        if v_max == v_min:
            return pd.Series(0, index=row.index)
        normalized = (clipped_row - v_min) / (v_max - v_min)
        return normalized

    normalized_data = heatmap_data.apply(normalize_row_with_clipping, axis=1)

    sns.heatmap(
        normalized_data,
        ax=ax,
        cmap='YlGnBu',
        linewidths=.5,
        cbar=False
    )
    ax.set_title('Event Frequency per Episode', fontsize=15)
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Event', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    
def plot_histogram_with_stats(ax, data, title, bins="fd"):
    """
    Plot a histogram with key summary statistics.

    Args:
        data (list or np.ndarray): Numeric data.
        bins (int or str): Number of bins or a binning rule (e.g., "fd", "sturges", "auto").
                           Default is "fd" (Freedman–Diaconis).
    """
    # 1) Prepare data and compute statistics
    data = np.asarray(data, dtype=float)
    if data.size == 0:
        return

    mean_val = float(np.mean(data))
    std_val = float(np.std(data, ddof=0))
    median_val = float(np.median(data))

    # Approximate mode from the histogram peak (robust for continuous data)
    counts, edges = np.histogram(data, bins=bins)
    peak_idx = int(np.argmax(counts))
    mode_val = 0.5 * (edges[peak_idx] + edges[peak_idx + 1])

    # 3) Plot histogram (density for comparability), with borders
    ax.hist(
        data,
        bins=bins,
        density=True,
        alpha=0.7,
        edgecolor="black",      # bar borders
        linewidth=0.8,
        label="Density histogram"
    )

    # 4) Add statistical markers
    ax.axvline(mean_val, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.2f}")
    ax.axvline(median_val, color="green", linestyle=":", linewidth=2, label=f"Median: {median_val:.2f}")
    ax.axvline(mode_val, color="purple", linestyle="-.", linewidth=2, label=f"Mode (approx): {mode_val:.2f}")

    # Shade ±1 standard deviation around the mean
    ax.axvspan(mean_val - std_val, mean_val + std_val, color="gray", alpha=0.2, label=f"±1 Std Dev: {std_val:.2f}")

    # 5) Style
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Value", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=10)
