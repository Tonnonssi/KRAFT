import warnings
import ta
import numpy as np

EPS = 1e-9

# -----------------------------
# 1. 기본 지표 + Realized Volatility
# -----------------------------
def add_basic_indicators(df, L=1500):
    # [수정] 차분 후 fillna는 0으로 채우는 게 안전
    df['log_return'] = np.log(df['close']).diff().fillna(0.0)
    
    # [유지] 비율 지표라 안전함
    df['return_5']   = df['close'].pct_change(periods=5).fillna(0.0)
    df['return_20']  = df['close'].pct_change(periods=20).fillna(0.0)
    df['volume_change'] = df['vol'].pct_change().fillna(0.0)
    
    # 가격 절대 변화량도 보상 계산에서 사용하므로 유지
    df['diff'] = df['close'].diff().fillna(0.0)

    # 롱뷰 기반 Z-score (score) - 아주 좋음!
    rolled = df['log_return'].shift(1).rolling(L, min_periods=L)
    df['roll_mu']  = rolled.mean().fillna(0.0)
    df['roll_vol'] = rolled.std().fillna(0.0)
    df['score'] = (df['log_return'] - df['roll_mu']) / (df['roll_vol'] + EPS)

    # [수정] Realized Volatility 수치 보정
    # log_return 제곱의 합은 숫자가 너무 작음 (예: 0.000002).
    # 신경망이 인식하기 좋게 sqrt를 씌우고 100을 곱해 % 단위로 변환 권장.
    df['realized_vol_10'] = np.sqrt(
        df['log_return'].pow(2).rolling(window=10, min_periods=1).sum()
    ) * 100.0

    return df

# -----------------------------
# 2. 추세 지표: Trend Indicators
# -----------------------------
def add_trend_indicators(df):
    df['ema_5']   = df['close'].ewm(span=5).mean()
    df['ema_20']  = df['close'].ewm(span=20).mean()
    
    # 이제 지수 레벨 상관없이 "0.001만큼 벌어졌다"는 동일한 의미를 가짐
    df['ema_cross'] = (df['ema_5'] - df['ema_20']) / (df['ema_20'] + EPS)

    # 멀티타임프레임 (Dist logic은 아주 훌륭함)
    df['ma_60']  = df['close'].rolling(window=60, min_periods=60).mean()
    df['ma_390'] = df['close'].rolling(window=390, min_periods=390).mean()

    df['dist_ma_60']  = (df['close'] - df['ma_60'])  / (df['ma_60']  + EPS)
    df['dist_ma_390'] = (df['close'] - df['ma_390']) / (df['ma_390'] + EPS)

    return df

# -----------------------------
# 3. 모멘텀 지표
# -----------------------------
def add_momentum_indicators(df):
    # RSI: 0~100 -> 0~1로 스케일링하면 더 좋음 (선택사항)
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi().fillna(50) / 100.0
    return df

# -----------------------------
# 4. 거래량/수급 지표 (OBV, AD 수정)
# -----------------------------
def add_volume_indicators(df):
    # [수정] 누적형(Cumsum) OBV, AD Line 제거
    # 대신 거래량 강도와 방향성을 결합한 오실레이터로 대체
    
    # 20-period 평균 대비 거래량 비율 (Good)
    df['vol_ma_20'] = df['vol'].rolling(window=20, min_periods=1).mean()
    df['vol_ratio'] = df['vol'] / (df['vol_ma_20'] + EPS)
    
    # [대체안] Volume-Price Trend (단기 수급 모멘텀)
    # 가격 변화 방향 * 거래량 비율 -> "거래량 실린 상승/하락"
    df['vpt_proxy'] = df['log_return'] * df['vol_ratio']

    return df

# -----------------------------
# 5. 변동성 (ATR, Gap 수정)
# -----------------------------
def add_volatility_indicators(df):
    if len(df) < 20:
        return df # 예외처리 유지
    
    ma  = df['close'].rolling(window=20).mean()
    std = df['close'].rolling(window=20).std()
    
    df['bb_upper'] = ma + (2 * std)
    df['bb_lower'] = ma - (2 * std)
    
    # BB Width: 이미 Ratio 방식이므로 훌륭함
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (ma + EPS)

    # [수정] ATR(Point) -> ATR Ratio(%)
    atr_val = ta.volatility.AverageTrueRange(
        df['high'], df['low'], df['close']
    ).average_true_range()
    df['atr'] = atr_val / (df['close'] + EPS) # 가격 대비 변동폭 비율로 변환

    # [수정] Gap(Point) -> Gap Ratio(%)
    df['gap_size'] = (df['open'] - df['prevClose']) / (df['prevClose'] + EPS)

    # HL Ratio: Good
    df['hl_ratio'] = (df['high'] - df['low']) / (df['close'] + EPS)

    # BB %B: Good (0~1)
    band_range = (df['bb_upper'] - df['bb_lower'])
    df['bb_pband'] = (df['close'] - df['bb_lower']) / (band_range + EPS)

    return df

# -----------------------------
# 6. MACD (유지)
# -----------------------------
def add_macd_indicators(df):
    # MACD Normalization 로직 아주 좋음 (Good)
    macd_ind = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
    macd_hist = macd_ind.macd_diff()
    
    df['macd_hist_norm'] = (macd_hist / (df['close'] + EPS)) * 100.0
    df['macd_hist_norm'] = df['macd_hist_norm'].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    return df
