import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import deque

from dash import Dash, dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ----------------------------
# 설정
# ----------------------------
WINDOW_SIZE = 200             # 차트에 보이는 최근 구간 길이
UPDATE_MS = 1000              # 업데이트 주기(ms)
np.random.seed(42)

# ----------------------------
# 모의 데이터 소스 (실데이터로 교체 가능)
# ----------------------------
class SimSource:
    def __init__(self):
        self.t0 = datetime(2013, 5, 1)
        self.n = 0
        self.pos = 0  # +1 long, -1 short, 0 flat
        self.price = 250.0
        self.price_path = deque(maxlen=10_000)
        self.unreal = deque(maxlen=10_000)
        self.real = deque(maxlen=10_000)
        self.actions = deque(maxlen=10_000)  # 0:Hold,1:Long,2:Short,3:LExit,4:SExit
        self.time = deque(maxlen=10_000)

    def step(self):
        # 가격 모형: 잡음 + 약한 사이클
        self.price += np.random.normal(0, 0.6) + 0.2*np.sin(self.n/12)
        self.price = float(np.clip(self.price, 230, 270))

        # 액션: 간단한 상태 전이 규칙(임의성 포함)
        rnd = np.random.rand()
        action = 0
        if self.pos == 0:
            if rnd < 0.08:
                action = 1; self.pos = 1    # Long 진입
            elif rnd < 0.16:
                action = 2; self.pos = -1   # Short 진입
        elif self.pos == 1:
            if rnd < 0.10:
                action = 3; self.pos = 0    # Long Exit
        elif self.pos == -1:
            if rnd < 0.10:
                action = 4; self.pos = 0    # Short Exit

        # 미실현/실현 PnL (모의)
        u = (self.unreal[-1] if self.unreal else 0.0) + np.random.normal(0, 8000)
        r = 0.0
        if action == 3:
            r = float(np.random.normal(300_000, 120_000))
        elif action == 4:
            r = float(np.random.normal(-100_000, 100_000))

        self.time.append(self.t0 + timedelta(days=self.n))
        self.price_path.append(self.price)
        self.actions.append(action)
        self.unreal.append(u)
        self.real.append(r)

        self.n += 1

    def frame(self, window=WINDOW_SIZE) -> pd.DataFrame:
        # 최근 window 길이만 반환
        data = {
            "Date": list(self.time)[-window:],
            "Price": list(self.price_path)[-window:],
            "Action": list(self.actions)[-window:],
            "Unrealized_PnL": list(self.unreal)[-window:],
            "Realized_PnL": list(self.real)[-window:],
        }
        return pd.DataFrame(data).set_index("Date")

SRC = SimSource()

# 초기 워밍업
for _ in range(WINDOW_SIZE):
    SRC.step()

# ----------------------------
# 앱 구성
# ----------------------------
app = Dash(__name__)
app.title = "Trading Dashboard (Dash)"

controls = html.Div(
    [
        html.Div([
            html.Button("▶︎ Start", id="btn-start", n_clicks=0, style={"marginRight":"8px"}),
            html.Button("⏸ Pause", id="btn-pause", n_clicks=0, style={"marginRight":"8px"}),
            html.Label("Update (ms):", style={"marginRight":"6px"}),
            dcc.Input(id="inp-ms", type="number", value=UPDATE_MS, min=200, step=100, style={"width":"100px"}),
            html.Label(" Window:", style={"marginLeft":"12px","marginRight":"6px"}),
            dcc.Input(id="inp-window", type="number", value=WINDOW_SIZE, min=50, step=50, style={"width":"100px"}),
        ], style={"display":"flex","alignItems":"center","flexWrap":"wrap","gap":"4px"}),
        html.Div(id="status", style={"marginTop":"6px","color":"#555"}),
    ],
    style={"padding":"10px","border":"1px solid #eee","borderRadius":"8px","marginBottom":"10px"}
)

graph = dcc.Graph(id="combined-figure", style={"height":"80vh"})

app.layout = html.Div(
    [
        html.H2("Trading Dashboard (Plotly Dash, Real-time)"),
        controls,
        dcc.Interval(id="tick", interval=UPDATE_MS, disabled=True),  # 시작 시 일시정지
        dcc.Store(id="store-running", data=False),                   # 재생/일시정지 상태
        dcc.Store(id="store-window", data=WINDOW_SIZE),              # 윈도우 길이
        graph,
    ],
    style={"maxWidth":"1200px","margin":"0 auto","padding":"16px"}
)

# ----------------------------
# 콜백: 재생/일시정지, 업데이트 주기, 윈도우
# ----------------------------
@app.callback(
    Output("store-running", "data"),
    Output("tick", "disabled"),
    Input("btn-start", "n_clicks"),
    Input("btn-pause", "n_clicks"),
    prevent_initial_call=True
)
def toggle_running(n_start, n_pause):
    # 어떤 버튼이 눌렸는지 판단
    trig = callback_context.triggered[0]["prop_id"].split(".")[0]
    running = (trig == "btn-start")
    return running, (not running)

@app.callback(
    Output("tick", "interval"),
    Input("inp-ms", "value")
)
def update_interval(ms):
    try:
        return max(200, int(ms))
    except:
        return UPDATE_MS

@app.callback(
    Output("store-window", "data"),
    Input("inp-window", "value"),
)
def update_window(win):
    try:
        return int(max(50, win))
    except:
        return WINDOW_SIZE

# ----------------------------
# 콜백: 매 틱마다 데이터 step + Figure 갱신
# ----------------------------
@app.callback(
    Output("combined-figure", "figure"),
    Output("status", "children"),
    Input("tick", "n_intervals"),
    State("store-running", "data"),
    State("store-window", "data"),
    prevent_initial_call=False
)
def on_tick(n, running, window):
    # 실행 중이면 1스텝 진행
    if running:
        SRC.step()

    df = SRC.frame(window=window)

    # 이벤트 분리
    long_entries = df[df["Action"] == 1]
    short_entries = df[df["Action"] == 2]
    long_exits   = df[df["Action"] == 3]
    short_exits  = df[df["Action"] == 4]

    # 서브플롯 구성
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.4, 0.3, 0.3], vertical_spacing=0.05,
        subplot_titles=("Market Price with Trading Events", "Unrealized PnL", "Realized PnL")
    )

    # (1) Price + Events
    fig.add_trace(go.Scatter(x=df.index, y=df["Price"], mode="lines",
                             name="Market Price", line=dict(color="black")), row=1, col=1)

    if not long_entries.empty:
        fig.add_trace(go.Scatter(x=long_entries.index, y=long_entries["Price"], mode="markers",
                                 name="Long Entry", marker_symbol="triangle-up",
                                 marker_color="green", marker_size=9),
                      row=1, col=1)
    if not short_entries.empty:
        fig.add_trace(go.Scatter(x=short_entries.index, y=short_entries["Price"], mode="markers",
                                 name="Short Entry", marker_symbol="triangle-down",
                                 marker_color="red", marker_size=9),
                      row=1, col=1)
    if not long_exits.empty:
        fig.add_trace(go.Scatter(x=long_exits.index, y=long_exits["Price"], mode="markers",
                                 name="Long Exit", marker_symbol="x",
                                 marker_color="blue", marker_size=9),
                      row=1, col=1)
    if not short_exits.empty:
        fig.add_trace(go.Scatter(x=short_exits.index, y=short_exits["Price"], mode="markers",
                                 name="Short Exit", marker_symbol="x",
                                 marker_color="purple", marker_size=9),
                      row=1, col=1)

    # (2) Unrealized PnL (양/음 영역)
    y = df["Unrealized_PnL"].astype(float).values
    y_pos = np.where(y >= 0, y, np.nan)
    y_neg = np.where(y < 0, y, np.nan)

    fig.add_trace(go.Scatter(x=df.index, y=y_pos, mode="lines",
                             name="Unrealized Gain", line=dict(color="lightgreen"), fill="tozeroy"),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=y_neg, mode="lines",
                             name="Unrealized Loss", line=dict(color="salmon"), fill="tozeroy"),
                  row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)

    # (3) Realized PnL (양/음 분리 막대)
    gain_mask = df["Realized_PnL"] > 0
    loss_mask = df["Realized_PnL"] < 0
    if gain_mask.any():
        fig.add_trace(go.Bar(x=df.index[gain_mask], y=df["Realized_PnL"][gain_mask],
                             name="Realized Gain", marker_color="green"),
                      row=3, col=1)
    if loss_mask.any():
        fig.add_trace(go.Bar(x=df.index[loss_mask], y=df["Realized_PnL"][loss_mask],
                             name="Realized Loss", marker_color="red"),
                      row=3, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=3, col=1)

    fig.update_layout(
        height=800, template="plotly_white",
        hovermode="x unified", barmode="relative",
        margin=dict(l=40,r=20,t=60,b=40)
    )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    status = f"Running: {bool(running)} | points={len(df)} | last={df.index[-1].strftime('%Y-%m-%d')}  price={df['Price'].iloc[-1]:.2f}"

    return fig, status

# ----------------------------
# 실데이터 연동 포인트(요약)
# ----------------------------
# 1) SimSource.step() 내부를 교체:
#    - 최신 틱/분봉/일봉 데이터를 받아서 self.price 업데이트
#    - 거래 이벤트(Action) 및 PnL을 연산해 self.actions/unreal/real에 append
# 2) 실시간 스트림이라면:
#    - WebSocket/REST 폴링으로 새 샘플만 반영
#    - 필요 시 Queue/DB(InfluxDB, PostgreSQL)로 버퍼링
# 3) 프론트에서 범위/심볼 선택 등 필터 UI를 dcc.Dropdown 등으로 추가하고,
#    해당 값들을 State로 받아 위 콜백에서 반영

if __name__ == "__main__":
    app.run(debug=True)