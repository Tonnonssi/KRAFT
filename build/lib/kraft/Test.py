import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# -------------------------------
# 1) 데이터 생성 (이전과 동일)
# -------------------------------
np.random.seed(42)
start_date = datetime(2013, 5, 1)
dates = [start_date + timedelta(days=i) for i in range(100)]
price = 250 + np.cumsum(np.random.normal(0, 1, 100)) + np.sin(np.arange(100)/10)*5
price = np.clip(price, 230, 270)

actions_raw = np.random.randint(0, 5, 100)
actions = np.zeros_like(actions_raw)
pos = 0
for i, a in enumerate(actions_raw):
    if a == 1 and pos == 0:
        actions[i] = 1; pos = 1
    elif a == 2 and pos == 0:
        actions[i] = 2; pos = -1
    elif a == 3 and pos == 1:
        actions[i] = 3; pos = 0
    elif a == 4 and pos == -1:
        actions[i] = 4; pos = 0
    else:
        actions[i] = 0

unrealized_pnl = np.random.normal(0, 100000, 100) + np.cumsum(np.random.normal(0, 10000, 100))
realized_pnl = np.zeros(100)
realized_pnl[actions == 3] = np.random.normal(500000, 100000, (actions == 3).sum())
realized_pnl[actions == 4] = np.random.normal(-300000, 80000, (actions == 4).sum())
neg_mask = realized_pnl < 0
realized_pnl[neg_mask] = np.random.normal(-100000, 50000, neg_mask.sum())

df = pd.DataFrame({
    "Date": dates,
    "Price": price,
    "Action": actions,
    "Unrealized_PnL": unrealized_pnl,
    "Realized_PnL": realized_pnl,
}).set_index("Date")

long_entries = df[df["Action"] == 1]
short_entries = df[df["Action"] == 2]
long_exits = df[df["Action"] == 3]
short_exits = df[df["Action"] == 4]

# -------------------------------
# 2) Plotly Subplots (3단 구성)
# -------------------------------
fig = make_subplots(
    rows=3, cols=1, shared_xaxes=True,
    row_heights=[0.4, 0.3, 0.3],
    vertical_spacing=0.05,
    subplot_titles=("Market Price with Trading Events", "Unrealized PnL", "Realized PnL")
)

# (1) Market Price
fig.add_trace(go.Scatter(x=df.index, y=df["Price"], mode="lines", name="Market Price", line=dict(color="black")), row=1, col=1)

# 이벤트 포인트
if len(long_entries):
    fig.add_trace(go.Scatter(x=long_entries.index, y=long_entries["Price"], mode="markers",
                             name="Long Entry", marker_symbol="triangle-up", marker_color="green", marker_size=10),
                  row=1, col=1)
if len(short_entries):
    fig.add_trace(go.Scatter(x=short_entries.index, y=short_entries["Price"], mode="markers",
                             name="Short Entry", marker_symbol="triangle-down", marker_color="red", marker_size=10),
                  row=1, col=1)
if len(long_exits):
    fig.add_trace(go.Scatter(x=long_exits.index, y=long_exits["Price"], mode="markers",
                             name="Long Exit", marker_symbol="x", marker_color="blue", marker_size=9),
                  row=1, col=1)
if len(short_exits):
    fig.add_trace(go.Scatter(x=short_exits.index, y=short_exits["Price"], mode="markers",
                             name="Short Exit", marker_symbol="x", marker_color="purple", marker_size=9),
                  row=1, col=1)

# (2) Unrealized PnL
y = df["Unrealized_PnL"].values
y_pos = np.where(y >= 0, y, np.nan)
y_neg = np.where(y < 0, y, np.nan)

fig.add_trace(go.Scatter(x=df.index, y=y_pos, mode="lines", name="Unrealized Gain",
                         line=dict(color="lightgreen"), fill="tozeroy"), row=2, col=1)
fig.add_trace(go.Scatter(x=df.index, y=y_neg, mode="lines", name="Unrealized Loss",
                         line=dict(color="salmon"), fill="tozeroy"), row=2, col=1)
fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)

# (3) Realized PnL (양/음 분리 막대)
gain_mask = df["Realized_PnL"] > 0
loss_mask = df["Realized_PnL"] < 0
fig.add_trace(go.Bar(x=df.index[gain_mask], y=df["Realized_PnL"][gain_mask],
                     name="Realized Gain", marker_color="green"), row=3, col=1)
fig.add_trace(go.Bar(x=df.index[loss_mask], y=df["Realized_PnL"][loss_mask],
                     name="Realized Loss", marker_color="red"), row=3, col=1)
fig.add_hline(y=0, line_dash="dot", line_color="gray", row=3, col=1)

# -------------------------------
# 3) 전체 레이아웃 조정
# -------------------------------
fig.update_layout(
    height=1000, width=1200,
    showlegend=True,
    hovermode="x unified",
    title_text="Trading Dashboard (Plotly Combined View)",
    template="plotly_white",
)

fig.update_xaxes(showgrid=True)
fig.update_yaxes(showgrid=True)

fig.write_html("plotly_combined_trading_dashboard.html", include_plotlyjs="cdn")
fig.show()