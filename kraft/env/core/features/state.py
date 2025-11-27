import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from .. import _specification as spec

INITIAL_ACCOUNT_BALANCE = 30_000_000

@dataclass(frozen=True)
class AgentState:
    current_position: int       # 현재 포지션 (-1,0,+1)
    execution_strength: int     # 체결 강도 
    n_days_before_ma: int       # 만기일까지 남은 날
    realized_pnl: float         # 실현 손익 (KRW)
    unrealized_pnl: float       # 미실현 손익 (KRW)
    available_balance: float    # 가용 잔고 (KRW)
    cost: float                 # 비용 (KRW)
    market_regime: int          # 단기적 시장 국면(-1,0,+1)

    def __call__(self) -> list:
        """list로 전달"""
        return self.scale()
    
    def scale(self):
        """
        [-1,1] 사이로 스케일링 
        ------------------------------
        current_position: int       # 현재 포지션 (-1,0,+1) 유지 
        execution_strength: int     # 체결 강도 (0~10) max 값으로 스케일링
        n_days_before_ma: int       # 만기일까지 남은 날 (0~30) 30일 기준으로 스케일링
        realized_pnl: float         # 실현 손익 (KRW) 10% 손익을 기준으로 스케일링 
        unrealized_pnl: float       # 미실현 손익 (KRW) 5% 손익을 기준으로 스케일링
        available_balance: float    # 가용 잔고 (KRW) 50% 잔고를 기준으로 스케일링
        cost: float                 # 비용 (KRW) 2% 비용을 기준으로 스케일링
        market_regime: int          # 단기적 시장 국면(-1,0,+1) 유지
        """
        return [
            self.current_position,  
            self.execution_strength / 10.0,  # 0~10 -> 0~1
            (30 - self.n_days_before_ma) / 30.0,  # 0~30 -> 0~1
            np.tanh(self.realized_pnl / (0.20 * INITIAL_ACCOUNT_BALANCE)),  # KRW -> -1~1
            np.tanh(self.unrealized_pnl / (0.02 * INITIAL_ACCOUNT_BALANCE)),  # KRW -> -1~1
            np.tanh(self.available_balance / (0.5 * INITIAL_ACCOUNT_BALANCE)),  # KRW -> -1~1
            np.tanh(self.cost / (0.02 * INITIAL_ACCOUNT_BALANCE)),  # KRW -> -1~1
            self.market_regime  # -1,0,+1 -> -1~1
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