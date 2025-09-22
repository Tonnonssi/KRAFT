import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

def plot_action_distribution_heatmap(ax, timesteps, actions, n_actions=21):
    """액션 분포 히트맵 - Hold(0) 제외하고 시각화"""
    if len(actions) == 0 or len(timesteps) == 0:
        ax.text(0.5, 0.5, "표시할 액션이 없습니다.", ha='center', va='center')
        ax.set_axis_off()
        return

    paired = [(t, a) for t, a in zip(timesteps, actions) if a is not None]
    if not paired:
        ax.text(0.5, 0.5, "표시할 액션이 없습니다.", ha='center', va='center')
        ax.set_axis_off()
        return

    timesteps, actions = map(list, zip(*paired))

    # 액션 범위 계산: n_actions=21이면 -10~+10
    action_center = n_actions // 2  # 10 (hold 액션)
    min_action = -action_center     # -10
    max_action = action_center      # +10

    # 액션을 시간 구간별로 그룹화
    n_time_bins = max(1, min(50, len(actions) // 10))
    if n_time_bins < 5:
        n_time_bins = max(1, min(10, len(actions)))

    time_bins = np.array_split(range(len(actions)), n_time_bins)

    # 실제 액션 값별 카운트 (Hold=0 제외)
    action_counts = np.zeros((n_time_bins, n_actions))

    for i, time_bin in enumerate(time_bins):
        if len(time_bin) > 0:
            bin_actions = [actions[j] for j in time_bin]
            for raw_action in bin_actions:
                # Hold 액션(0) 제외하고 카운트
                if raw_action != 0:
                    # 실제 액션 값을 인덱스로 변환 (예: -10 → 0, 0 → 10, +10 → 20)
                    action_idx = raw_action + action_center
                    if 0 <= action_idx < n_actions:
                        action_counts[i, action_idx] += 1

    # 정규화 (각 시간 구간의 합이 1이 되도록)
    row_sums = action_counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # 0으로 나누기 방지
    action_probs = action_counts / row_sums

    extent = [0, action_probs.shape[0], min_action - 0.5, max_action + 0.5]

    # 히트맵 그리기
    im = ax.imshow(
        action_probs.T,
        aspect='auto',
        cmap='Greys',
        interpolation='nearest',
        origin='lower',
        extent=extent
    )

    # 축 레이블 설정
    ax.set_xlabel('Time Periods')
    ax.set_ylabel('Action Value') 
    ax.set_title('Action Distribution Heatmap')

    # x축: 시간 구간별 대표 타임스탬프
    if len(timesteps) > 0:
        time_labels = []
        for time_bin in time_bins[::max(1, len(time_bins)//6)]:  # 최대 6개 레이블
            if len(time_bin) > 0:
                mid_idx = time_bin[len(time_bin)//2]
                if mid_idx < len(timesteps):
                    time_labels.append(pd.to_datetime(timesteps[mid_idx]).strftime('%m-%d'))
                else:
                    time_labels.append('')
       
        tick_positions = np.linspace(0, len(time_bins), len(time_labels))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(time_labels, rotation=45)

    # y축: 실제 액션 값으로 표시 (Hold 제외 표시)
    important_action_values = [min_action, -5, 0, 5, max_action]
    ax.set_yticks(important_action_values)
    ax.set_yticklabels([f'{val:+d}' if val != 0 else '0(Hold-Excluded)' for val in important_action_values])

    # 컬러바 추가
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Action Probability', rotation=270, labelpad=20)

    # 중요한 액션 영역 표시 (Hold 영역만 회색으로)
    # Hold 영역 (0 근처) - 제외되었음을 표시
    ax.axhline(0 - 0.5, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    ax.axhline(0 + 0.5, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    ax.text(len(time_bins)*0.02, 0, 'HOLD(Excluded)',
           bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgray', alpha=0.7),
           fontsize=8)

    # Short 영역 (음수)
    ax.axhspan(min_action - 0.5, -0.5, alpha=0.1, color='blue', label='Short Zone')
    ax.text(len(time_bins)*0.02, min_action + 2.5, 'SHORT(-)',
           bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue', alpha=0.7),
           fontsize=8)

    # Long 영역 (양수)
    ax.axhspan(0.5, max_action + 0.5, alpha=0.1, color='red', label='Long Zone')
    ax.text(len(time_bins)*0.02, max_action - 2.5, 'LONG(+)',
           bbox=dict(boxstyle='round,pad=0.2', facecolor='lightcoral', alpha=0.7),
           fontsize=8)

    # 통계 정보 추가
    total_actions = len(actions)
    hold_actions = sum(1 for action in actions if action == 0)
    hold_percentage = (hold_actions / total_actions * 100) if total_actions > 0 else 0
    
    ax.text(0.98, 0.02, f'Hold actions excluded: {hold_actions} ({hold_percentage:.1f}%)',
           transform=ax.transAxes, ha='right', va='bottom',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
           fontsize=8)
    
def plot_long_short(ax, long, short, title="Long vs Short Counts"):
    """
    Plot side-by-side bar chart for discrete values (1–10).

    Args:
        long (list or np.ndarray): Long values (1-10).
        short (list or np.ndarray): Short values (1–10).
        title (str): Title of the plot.
    """
    # Convert to numpy arrays
    long = np.asarray(long, dtype=int)
    short = np.asarray(short, dtype=int)

    # Define x-axis values (1~10)
    x_vals = np.arange(1, 11)

    # Count frequencies
    long_counts = np.array([np.sum(long == x) for x in x_vals])
    short_counts = np.array([np.sum(short == x) for x in x_vals])

    # Bar width and positions
    width = 0.4

    ax.bar(x_vals - width/2, long_counts, width=width,
           color="tomato", edgecolor="black", label="Long")
    ax.bar(x_vals + width/2, short_counts, width=width,
           color="royalblue", edgecolor="black", label="Short")

    # Style
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Value", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_xticks(x_vals)
    ax.legend(fontsize=11, frameon=True, shadow=True)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")

def plot_market_with_actions(ax, timesteps, market, actions, cumulative_actions):
    x = np.arange(len(market))
    actions = np.array(actions)
    
    # 배경 색상 (누적 행동 기반)
    for i in range(len(cumulative_actions)):
        alpha = min(abs(cumulative_actions[i]) / 10, 1.0)
        color = 'red' if cumulative_actions[i] > 0 else 'blue' if cumulative_actions[i] < 0 else None
        if color:
            ax.axvspan(i - 0.5, i + 0.5, facecolor=color, alpha=alpha * 0.1, zorder=0)

    # 시장 가격 라인
    ax.plot(x, market, label='Market Price', color='black', linewidth=1.2)

    # 행동 점 시각화
    colors = np.where(actions > 0, 'red', np.where(actions < 0, 'blue', 'gray'))
    labels = {'red': 'Long', 'blue': 'Short', 'gray': 'Hold'}
    plotted_labels = set()
    for t, a in enumerate(actions):
        color = colors[t]
        label = labels[color] if color not in plotted_labels else None
        edgecolor = 'black'
        ax.scatter(t, market[t], color=color, edgecolors=edgecolor, marker='^', 
                   alpha=min(abs(a) / 10, 1.0), s=40, label=label, zorder=3)
        plotted_labels.add(color)

    # 지표용 legend 핸들 추가
    legend_elements = [
        Line2D([0], [0], color='black', lw=1.5, label='Market Price'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markeredgecolor='black', label='Long'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='blue', markeredgecolor='black', label='Short'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markeredgecolor='black', label='Hold'),
        Patch(facecolor='red', alpha=0.2, label='Cumulative Long Pressure'),
        Patch(facecolor='blue', alpha=0.2, label='Cumulative Short Pressure'),
    ]

    # 기타 설정
    ax.set_title(f'Market Flow with Actions')
    ax.set_ylabel('Market')
    ax.set_xlabel('Date')
    ax.set_xticks(np.linspace(0, len(timesteps)-1, min(10, len(timesteps))))
    ax.set_xticklabels([str(timesteps[int(i)])[:10] for i in np.linspace(0, len(timesteps)-1, min(10, len(timesteps)))], rotation=45)
    ax.legend(handles=legend_elements, loc='best')
    ax.grid(True)
