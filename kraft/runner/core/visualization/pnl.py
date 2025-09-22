def plot_both_pnl_ticks(ax, timesteps, unrealized_pnl, realized_pnl):
    if not timesteps:
        ax.text(0.5, 0.5, "PnL 데이터가 없습니다.", ha='center', va='center')
        ax.set_axis_off()
        return

    width = 0.4  # 막대 폭

    unrealized_clean = [0.0 if v is None else float(v) for v in unrealized_pnl]
    realized_clean = [0.0 if v is None else float(v) for v in realized_pnl]

    # 색상 팔레트 
    unrealized_colors = ['#FFB6C1' if v >= 0 else '#87CEEB' for v in unrealized_clean]
    realized_colors   = ['#E74C3C' if v >= 0 else '#3498DB' for v in realized_clean]

    # 막대 위치
    x = list(range(len(timesteps)))
    x1 = [i - width/2 for i in x]  # Unrealized
    x2 = [i + width/2 for i in x]  # Realized

    # 막대그래프 그리기
    bars1 = ax.bar(x1, unrealized_clean, width=width, color=unrealized_colors, label='Unrealized PnL')
    bars2 = ax.bar(x2, realized_clean, width=width, color=realized_colors, label='Realized PnL')

    # --- 전역 최대 절대값 찾기 ---
    all_vals = [v for v in (unrealized_clean + realized_clean) if v is not None]
    if all_vals:
        max_val = max(all_vals, key=abs)
        if max_val in unrealized_clean:
            idx = unrealized_clean.index(max_val)
            xpos = x1[idx]
        else:
            idx = realized_clean.index(max_val)
            xpos = x2[idx]

        offset = 0.02 * max(abs(v) for v in all_vals) if all_vals else 0
        if max_val >= 0:
            ax.text(xpos, max_val + offset, f"{max_val:,.0f}", ha='center', va='bottom',
                    fontsize=10, color="black")
        else:
            ax.text(xpos, max_val - offset, f"{max_val:,.0f}", ha='center', va='top',
                    fontsize=10, color="black")

    # 스타일
    ax.set_xticks(x)
    ax.set_xticklabels(timesteps, rotation=45)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("PnL")
    ax.set_title("Unrealized vs Realized PnL", fontsize=14)
    ax.legend()
    ax.grid(axis='y', linestyle="--", alpha=0.6)
