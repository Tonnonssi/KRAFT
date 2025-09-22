from enum import Enum
import numpy as np


# 시장 상태 구분 Enum (강세장, 약세장, 횡보장)
class MarketRegime(Enum):
    BULL = 1        # 강세장
    BEAR = -1       # 약세장
    SIDEWAYS = 0    # 횡보장

        
def get_market_regime(price_data: np.ndarray, short_view, long_view, threshold):
    """가격 데이터를 바탕으로 시장 상태 갱신(경험적인 기준...)"""
    if len(price_data) < long_view:
        return 0
    
    short_ma = np.mean(price_data[-short_view:]) 
    long_ma = np.mean(price_data[-long_view:]) 
    
    if short_ma > long_ma * (1+threshold):
        return MarketRegime.BULL.value
    elif short_ma < long_ma * (1-threshold):
        return MarketRegime.BEAR.value
    else:
        return MarketRegime.SIDEWAYS.value
    
