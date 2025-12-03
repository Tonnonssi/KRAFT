import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from .. import _specification as spec

INITIAL_ACCOUNT_BALANCE = 10_000_000

@dataclass(frozen=True)
class AgentState:
    current_position: int       # 현재 포지션 (-1,0,+1)
    execution_strength: int     # 체결 강도 
    n_days_before_ma: int       # 만기일까지 남은 날
    equity: float               # 자산 총액 (KRW)
    market_regime: int          # 단기적 시장 국면(-1,0,+1)
    time_remaining_ratio: float # 당일마감까지 남은 시간 비율(0~1)
    near_market_closing: int    # 당일 마감 임박 여부 (30분 전) (0,1)

    def __len__(self):
        return len(self.__dataclass_fields__)

    def __call__(self) -> list:
        """list로 전달"""
        return self._scale()
    
    def _scale(self):
        """
        [-1,1] 사이로 스케일링 
        ------------------------------
        current_position: int       # 현재 포지션 (-1,0,+1) 유지 
        execution_strength: int     # 체결 강도 (0~10) max 값으로 스케일링
        n_days_before_ma: int       # 만기일까지 남은 날 (0~30) 30일 기준으로 스케일링
        equity: float               # 자산 총액 (KRW) 10% 손익을 기준으로 스케일링 (가용 자산 + 미실현 손익)
        market_regime: int          # 단기적 시장 국면(-1,0,+1) 유지
        """
        equity_norm = (self.equity / INITIAL_ACCOUNT_BALANCE) - 1.0 # 초기 자산을 0으로 정규화해 정규화

        return [
            self.current_position,  
            self.execution_strength / 10.0,  # 0~10 -> 0~1
            (30 - self.n_days_before_ma) / 30.0,  # 0~30 -> 0~1
            np.tanh(equity_norm / 0.2),  # ± 20% 변동에서 tanh 활발하게 변화 
            self.market_regime,  # -1,0,+1 -> -1~1
            self.time_remaining_ratio,  # 0~1 유지
            self.near_market_closing  # 0,1 유지
        ]

@dataclass(frozen=True)
class State:
    """
    전체 상태 '데이터'를 구조적으로 담는 역할
    두 상태는 모두 스케일링된 상태 
    """
    timeseries_state: np.ndarray
    agent_state: AgentState

    @property
    def shape(self):
        """두 데이터의 shape을 반환"""
        return self.timeseries_state.shape, len(self.agent_state())
    
    def __call__(self):
        return self.timeseries_state, np.array(self.agent_state())