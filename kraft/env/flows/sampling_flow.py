from ..core.base_env import BaseEnvironment, State, AgentState
import pandas as pd
from ..core.features.dataset import FuturesDataset, EpisodeDataset, EpisodeDataloader

class SamplingFlow(BaseEnvironment):
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
                 pnl_threshold=0.05,
                 n_iteration=10):
        
        super().__init__(raw_df, date_range, window_size, 
                         reward_ftn, start_budget, n_actions, max_steps, 
                         slippage_factor_rate, position_cap, scaler, pnl_threshold)
        
        self.n_iteration = n_iteration
        self.current_n_iteration = 0

    def set_dataset(self):
        """ 
        dataset, iterator, df를 정의한다.
        --
        """
        self.base_dataset = FuturesDataset(self.target_df, self.window_size, self.scaler)       # date range의 전체 데이터셋
        self.dataset_df = self.base_dataset.cleaned_df

        self.episode_dataset = EpisodeDataset(self.base_dataset, window_len=self.max_steps+1)   # 에피소드로 데이터 셋을 묶은 애
        self.episode_loader = EpisodeDataloader(self.episode_dataset, shuffle=True)             # 그걸 섞고 관리하는 애 
        
        self.dataset = next(self.episode_loader)        # single dataset 
        self.data_iterator = iter(self.dataset) 

    def reset(self):
        if self.done:
            self._reset_base()
            try:
               self.dataset = next(self.episode_loader)     # 에피소드를 나눠둔 데이터 셋이 동남 
            except StopIteration:
                # 에피소드 순회가 끝났다면 다시 섞어서 리셋
                self.episode_loader.shuffle_indices()
                self.episode_loader = iter(self.episode_loader)

                self.dataset = next(self.episode_loader)
                self.data_iterator = iter(self.dataset) 
                # 전체 데이터 순회 횟수 계산 
                self.current_n_iteration += 1

        self.current_state = self._reset_state()

        return self.current_state
    
    @property
    def terminated(self):
        """종료되었는지 확인. 이때는 지정된 전체 데이터 셋 반복 횟수가 전부 소진이라서"""
        return self.n_iteration ==  self.current_n_iteration
    