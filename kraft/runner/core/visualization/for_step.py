import numpy as np

def plot_realized_pnl(ax, timesteps, realized_pnl, title: str = "Realized PnL Over Time"):
    """
    Line chart for realized PnL with:
    - gray zero line
    - annotations for final, max, and min values
    """
    ts = np.asarray(timesteps)
    y = np.asarray(realized_pnl, dtype=float)

    x = np.arange(len(y))

    # main line
    ax.plot(x, y, linewidth=2.0, color="#3498DB", label="Realized PnL")  # elegant blue
    ax.fill_between(x, np.minimum(y, 0), 0, color="#AED6F1", alpha=0.4, step=None)  # soft negative area

    # zero line
    ax.axhline(0, color="gray", linewidth=1.0, linestyle="--")

    # stats
    max_idx = int(np.nanargmax(y))
    min_idx = int(np.nanargmin(y))
    final_idx = len(y) - 1

    max_val = int(y[max_idx])
    min_val = int(y[min_idx])
    final_val = int(y[final_idx])

    # annotation helper
    def annotate_point(ix, val, label, color, va='bottom'):
        ax.scatter(ix, val, s=35, color=color, edgecolor="black", zorder=3)
        ax.annotate(
            f"{label}: {val:.2f}",
            xy=(ix, val),
            xytext=(ix + max(2, len(y)*0.02), val + (0.04 if va=='bottom' else -0.04)*max(1.0, np.nanmax(np.abs(y)))),
            textcoords="data",
            arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, lw=1.0),
            va=va
        )

    # annotate max / min / final
    annotate_point(max_idx, max_val, "Max", "#E74C3C", va='bottom')  # red
    annotate_point(min_idx, min_val, "Min", "#2E86C1", va='top')     # blue
    annotate_point(final_idx, final_val, "Final", "#27AE60", va='bottom')  # green

    # axes & labels
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Realized PnL")
    ax.set_xlabel("Timesteps")
    ax.set_xticks(x)
    ax.set_xticklabels(ts, rotation=45, ha="right")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True)

def _nan_safe_moving_average(a, window: int):
    """NaN-safe moving average using convolution."""
    a = np.asarray(a, dtype=float)
    mask = np.isfinite(a).astype(float)
    a = np.nan_to_num(a, nan=0.0)
    kernel = np.ones(window, dtype=float)
    num = np.convolve(a, kernel, mode='valid')
    den = np.convolve(mask, kernel, mode='valid')
    with np.errstate(invalid='ignore', divide='ignore'):
        out = num / den
    return out

def plot_rewards_with_ma(ax, series, ma_window: int = 10,
                         title: str = "Cumulative Rewards Over Time"):
    """
    Plot one reward series with raw line + moving average.
    """
    series = np.asarray(series, dtype=float)
    x = np.arange(len(series))

    # raw line
    ax.plot(x, series, label=f"raw", alpha=0.25, linewidth=1)

    # moving average
    if len(series) >= ma_window:
        ma = _nan_safe_moving_average(series, ma_window)
        ax.plot(np.arange(ma_window - 1, len(series)), ma,
                label=f"(MA{ma_window})", linewidth=2.5)
    else:
        ax.plot([], [], label=f"(MA{ma_window}) — n<{ma_window}", alpha=0)

    # style
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Reward")
    ax.set_xlabel("steps")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True, shadow=False)
