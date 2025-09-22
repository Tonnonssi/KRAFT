import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from .. import _specification as spec


@dataclass(frozen=True)
class AgentState:
    current_position: int       # 현재 포지션 (-1,0,+1)
    execution_strength: int     # 체결 강도 
    n_days_before_ma: int       # 만기일까지 남은 날
    # before_liquidation: bool    # 청산이 진행되는가? (장기)
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
        일단 고대로 옮김 
        장기적으로 스케일링을 더 해보는게 좋을 듯 함
        """
        KRW_to_point = lambda x: x / spec.CONTRACT_UNIT
        raw_list = list(self.__dict__.values())
        return [KRW_to_point(e) if isinstance(e, float) else e for e in raw_list]


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