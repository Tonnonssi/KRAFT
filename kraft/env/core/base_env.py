from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import torch
from torch.distributions.dirichlet import Dirichlet 
from typing import Dict, List, Tuple, Optional, Any

from .features.dataset import FuturesDataset
from .features.risk import *
from .features.state import State, AgentState
from .features.account import Account
from .features.event import Event, StepEvent

from .utils.done_conditions import *
from .utils.reward_schemes import *
from .utils.maturity_functions import *
from .utils.market_classifier import get_market_regime


class BaseEnvironment(ABC):
    position_dict = {-1: 'short', 0: 'hold', 1: 'long'}
    liquidation_status_list = ['end_of_data', 'maturity_data', 
                               'bankrupt', 'margin_call', 
                               'max_step']

    def __init__(self, 
                 raw_df: pd.DataFrame, 
                 date_range: tuple, 
                 window_size: int, 
                 reward_ftn, 
                 start_budget: float,
                 n_actions: int, 
                 max_steps: int,
                 slippage_factor_rate: float,
                 position_cap: float = float('inf'),
                 scaler=None,
                 target_columns=None,
                 pnl_threshold=0.05,
                 alpha_parameters=[5.0,3.0,1.0],
                 **kwargs):     # randomsampling env 때문에 필요  
        
        # ====== Main DATA ===============================

        self.raw_df = raw_df                        # 전체 데이터 셋 
        self.date_range = date_range                # 데이터 범위 (start, end)
        self.target_df = slice_by_date_range(raw_df, date_range) # 이 환경에서 사용할 데이터 

        self.scaler = scaler                        # timeseries data에 적용할 스케일러  
        self.window_size = window_size              # timeseries data 길이 지정 

        self.target_columns = target_columns
        self.set_dataset()

        self.State = State
        self.AgentState = AgentState
        self.current_state = None 

        self.alpha_parameters = alpha_parameters  # multi-critics PPO에서 사용하는 α 파라미터

        # ====== Position & Duration =====================
        
        self.position_cap = position_cap             # 최대 보유 가능 포지션 
        self.n_actions = n_actions                   # 총 행동 크기  
        self.single_volume_cap = self.n_actions // 2 # 1 step의 체결 강도 상한 
        self.max_steps = max_steps                   # 스텝 수 상한 
        self.pnl_threshold = pnl_threshold           # 만족 수익률 기준 

        # ====== INFO ===================================

        self.current_timestep = self.target_df.index[0]# 현재 시간 

        self.episode_event = Event()                 # 에피소드 전체 tracker
        self.step_event = self.episode_event.step_event # 스텝 단위 tracker

        self.since_entry = 0                         # 새로운 진입 이후 누적 스텝 수
        self.maintained = 0                          # 스윙 전략 유지 스텝 수  
        self.done = False                            # 에피소드가 끝이 났는가? 

        self.previous_point = None                   # 이전 스텝 포인트 
        self.current_point = None                    # 현재 스텝 포인트

        self.n_total_trades = 0                      # 전체 거래 횟수 
        self.n_win_trades = 0                        # 수익을 본 거래 횟수 

        self.maturity_timesteps = get_maturity_timesteps(date_range[0], raw_df)
        print(len(self.maturity_timesteps), "maturity dates found.")
        
        # ====== BUDGET ==================================

        self.account = Account(start_budget, 
                               position_cap, 
                               self.current_timestep, 
                               slippage_factor_rate)
        self.slippage_factor_rate = slippage_factor_rate

        # ====== Utils ==================================

        self.get_reward = reward_ftn

    def set_dataset(self):
        """ dataset, iterator, df를 정의한다."""
        self.dataset = FuturesDataset(
            self.target_df,
            self.window_size,
            self.scaler,
            target_columns=self.target_columns,
        )
        self.data_iterator = iter(self.dataset)
        self.dataset_df = self.dataset.cleaned_df

    def step(self, decoded_action: int) -> Tuple[Any, Any, bool, Any]:
        """
        action을 적용해 S_t -> S_{t+1} 전이.
        Returns:
            obs_{t+1}, reward, done, event, mask
        """
        # 0) 이전 상태 캐시(체결/정산 전 스냅샷)
        self.account.cache_values()

        # 1) 데이터 전진
        next_ts_state = self._advance_data()

        # 2) 주문 체결/계좌 반영
        net_pnl, cost = self._execute_action(decoded_action)

        # 3) 스텝 카운트 업데이트 (done 체크는 현재 스텝 포함)
        self.maintained += 1

        # 4) 종료 여부/이벤트 판정
        done, step_events_list = self.done_n_event
        self.episode_event.collect_information(step_events_list)

        # 5) 필요 시 강제 청산 (만기/파산 등)
        if any(event in self.step_event for event in self.liquidation_status_list) and self.current_point is not None:
            liq_pnl, liq_cost = self._maybe_liquidate(self.current_point)
            net_pnl += liq_pnl
            cost += liq_cost
        
        # 6) 일자 변경 시 일일 정산
        if is_day_changed(self.current_timestep, self.two_ticks_later) and self.current_point is not None:
            # two ticks later로 보는 이유는, 실제 장 마감 시간은 15분, 우리는 5분에 거래를 종료하는 걸 전제로 함 
            # 그래서 2틱 뒤를 봐야 검증이 가능하다 (1틱 뒤는 15분 데이터)
            # print(f"Compare: current_{self.current_timestep}, account_prev_{self.account.prev_timestep}, account_current_{self.account.current_timestep}")
            # print(f"Daily settlement at {self.current_timestep}")
            self._maybe_daily_settlement(self.current_point)

        # 7) 다음 상태 생성
        next_state_obj = self._build_next_state(next_ts_state)

        # 8) 보상 계산
        reward = self._compute_reward(event=self.step_event)

        # 9) 마스크/관찰 준비
        mask = self.mask
        obs = next_state_obj()  # 기존 State의 __call__ 사용 패턴 유지
        self.current_state = obs

        # 10) 거래 정보를 업데이트 
        self._update_trades_info()

        return obs, reward, done, mask
    
    # ---------- 내부 헬퍼 ----------
    def _advance_data(self):
        """이터레이터에서 다음 시점 데이터 꺼내고 포인터 업데이트."""
        next_ts_state, self.current_point, self.current_timestep = next(self.data_iterator)
        return next_ts_state
    
    def _update_trades_info(self):
        # 거래 성공 횟수, 손익 실현 횟수 업데이트 
        if self.account.settled:
            self.n_total_trades += 1
            if self.account.net_realized_pnl > 0:
                self.n_win_trades += 1

    def _execute_action(self, decoded_action: int) -> Tuple[float, float]:
        """계좌에 행동을 반영하고 손익/비용을 수집."""
        net_pnl, cost = self.account.step(decoded_action, self.current_point, self.current_timestep)
        return float(net_pnl), float(cost)

    def _maybe_liquidate(self, current_price: float) -> Tuple[float, float]:
        """필요 시 전량 청산."""
        return self.force_liquidate_all_positions(current_price)

    def _maybe_daily_settlement(self, current_price: float) -> None:
        """일자 변경 시 미실현 → 실현으로 일일 정산."""
        self.account.daily_settlement(current_price)

    def _build_next_state(self, next_ts_state: Any) -> State:
        """다음 관찰(State 객체) 구성."""
        return State(timeseries_state=next_ts_state, agent_state=self.agent_state)

    def _compute_reward(self, event:StepEvent):
        """RewardInfo를 구성해 보상 계산."""
        rinfo = RewardInfo(
            net_realized_pnl=self.account.net_realized_pnl,
            prev_unrealized_pnl=self.account.prev_unrealized_pnl,
            current_unrealized_pnl=self.account.unrealized_pnl,
            prev_balance=self.account.prev_balance,
            current_balance=self.account.balance,
            prev_position=self.account.prev_position,
            current_position=self.account.current_position,
            point_delta=self.point_delta,
            execution_strength=self.account.execution_strength,
        )
        return self.get_reward(rinfo, event=event)      # single critic in R, multi critics in R^3

    @abstractmethod 
    def reset(self) -> Tuple:
        pass

    @abstractmethod
    def checks(self) -> list:
        pass
    
    @property
    @abstractmethod
    def terminated(self) -> bool:
        """전체 env가 종료되는가? ex) 데이터셋 소진"""
        pass

    def _reset_base(self):
        """기본으로 초기화되어야 하는 것"""
        self.account.reset()
        self.account.current_timestep = self.current_timestep
        self.episode_event = Event()
        self.step_event = self.episode_event.step_event

        self.since_entry = 0
        self.maintained = 0
        self.n_total_trades = 0
        self.n_win_trades = 0
        self.done = False 

    def _reset_state(self):
        """State의 timeseries 데이터는 남긴 채로 Agent 데이터를 초기화"""
        if self.current_state == None:
            return self._reset_to_init_timestep()
        else:
            timeseries_data = self.current_state[0]
            state = State(timeseries_state=timeseries_data, agent_state=self.agent_state)
            return state()

    def _reset_to_init_timestep(self) -> Tuple:
        """데이터 이터레이터 초기화 및 첫 관찰 반환."""
        self.data_iterator = iter(self.dataset)
        timeseries_state, close_price, timestep = next(self.data_iterator)

        self.current_point = close_price
        self.current_timestep = timestep

        next_state_obj = State(timeseries_state=timeseries_state, agent_state=self.agent_state)
        obs = next_state_obj()
        self.current_state = obs
        return obs

    def render(self):
        pass

    def force_liquidate_all_positions(self, current_price):
        """리스크 제한 초과 시 모든 포지션 강제 청산"""
        if self.account.execution_strength == 0:
            return 0,0

        # 현재 체결된 계약에서 반대 포지션을 취함 
        reversed_execution = -self.account.execution_strength * self.account.current_position
        net_pnl, cost = self.account.step(reversed_execution, current_price, self.current_timestep) 

        return net_pnl, cost
    
    def reset_alpha(self):
        """
        Multi-Critics PPO에서 사용하는 α 초기화, single critic에서는 사용하지 않는다.
        한 에피소드 동안 고정된다. 
        """
        self.alpha = Dirichlet(torch.tensor(self.alpha_parameters, dtype=torch.float32)).sample().detach()

    @property
    def agent_state(self) -> AgentState:
        return AgentState(
                    current_position=self.account.current_position,
                    execution_strength=self.account.execution_strength,
                    n_days_before_ma=self.n_days_before_maturity,
                    equity=self.account.balance,
                    market_regime=self.market_regime)    
    
    @property
    def market_regime(self):
        short_view, long_view, threshold = 30, 150, 0.001

        current_idx = self.dataset_df.index.get_loc(self.current_timestep)
        start_idx = max(0, current_idx - long_view)
        price_data = self.dataset_df['close'].iloc[start_idx:current_idx].values
        
        return get_market_regime(price_data, short_view, long_view, threshold)

    @property
    def n_days_before_maturity(self):
        return get_n_days_before_maturity(self.maturity_timesteps, self.current_timestep)
    
    @property
    def next_timestep(self):
        """다음 시간"""
        pos_arr = self.dataset_df.index.get_indexer([self.current_timestep])
        if pos_arr[0] == -1:  # current_timestep이 없는 경우
            return None

        next_pos = pos_arr[0] + 1
        if next_pos < len(self.dataset_df.index):
            return self.dataset_df.index[next_pos]
        return None
    
    @property
    def two_ticks_later(self):
        """두 타임스텝 뒤 시간"""
        pos_arr = self.dataset_df.index.get_indexer([self.current_timestep])
        if pos_arr[0] == -1:  # current_timestep이 없는 경우
            return None

        next_pos = pos_arr[0] + 2
        if next_pos < len(self.dataset_df.index):
            return self.dataset_df.index[next_pos]
        return None
    
    @property
    def is_entry(self):
        """청산 후 첫 진입인지 확인"""
        return bool((self.account.prev_position == 0) & (self.account.current_position != 0))

    @property
    def done_n_event(self):
        """ current timestep의 Event """
        step_events = []

        checks = self.checks()
        done_lst = []

        for check in checks:
            done, event = check()
            done_lst.append(done)
            step_events.append(event)

            # if event == 'end_of_data':
            #     # 만기일 확인까지 넘어가면 안된다. 
            #     # next_timestep이 존재하지 않기 때문이다. 
            #     break 

        self.done = bool(sum(done_lst) != 0) 

        return self.done, step_events

    @property
    def mask(self) -> List:
        """
        agent의 행동 선택을 제한하는 마스크 
        """
        # 인자 받기 
        position = self.account.current_position
        remaining_strength = self.position_cap - self.account.execution_strength
        k = self.single_volume_cap
        n = self.n_actions

        # 기본 마스크 생성
        mask = np.ones(n, dtype=np.int32)

        if (self.position_cap == remaining_strength) or ('insufficient' in self.step_event):
            # 최대 체결 가능 계약수에 도달했을 때 
            # 자본금 부족으로 새로운 포지션을 체결할 수 없을 때 
            if position == -1: # short 
                mask[:k] = 0
    
            elif position == 1: # long 
                mask[-k:] = 0 

        elif (remaining_strength) < k:
            # 최대 체결 가능 계약수에 근접하여 일부 행동에 제약이 있다. 
            restriction = k - remaining_strength 

            if self.account.current_position == -1: # short 
                mask[:restriction] = 0
            elif self.account.current_position == 1: # long 
                mask[-restriction:] = 0

        return mask.tolist()

    @property
    def score(self):
        """KL에 사용하는 score"""
        df = self.dataset_df
        current_idx = df.index.get_loc(self.current_timestep)
        return float(df['score'].iloc[current_idx])

    @property
    def log_return(self):
        """KL에 사용하는 log return"""
        df = self.dataset_df
        current_idx = df.index.get_loc(self.current_timestep)
        return float(df['log_return'].iloc[current_idx])
    
    @property
    def point_delta(self):
        """Reward에 사용하는 포인트 변화율"""
        df = self.dataset_df
        current_idx = df.index.get_loc(self.current_timestep)
        return float(df['diff'].iloc[current_idx])
    

def slice_by_date_range(full_df: pd.DataFrame, date_range: tuple) -> pd.DataFrame:
        """날짜 범위로 데이터프레임 슬라이싱"""
        full_df = full_df.copy()
        full_df.index = pd.to_datetime(full_df.index)
        full_df = full_df.sort_index()
        
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        return full_df[(full_df.index >= start) & (full_df.index <= end)]
