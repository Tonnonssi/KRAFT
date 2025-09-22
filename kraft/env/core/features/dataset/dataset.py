import pandas as pd
from torch.utils.data import Dataset
from datetime import timedelta

from .utils.indicator_ftns import *
from .utils.scaler import *

# 전체 ts list - 거래 가능 list 만들기 
# 거래 가능 리스트가 있어야 데이터 셋 만들기, 다음날 구분에 사용할 수 있음 

class FuturesDataset(Dataset):
    """
    학습에 사용하는 선물 시장의 시계열 데이터를 담고 있는 데이터 셋
    - OCHLV 데이터 뿐만 아니라, '.indicator_ftns'에 위치한 모든 기술적 분석 지표를 포함한다.
    - target_columns가 주어지면 해당 feature만 state에 포함한다.
    """

    def __init__(self, df, window_size, transform=None, target_columns=None):
        self.window_size = window_size                  # 데이터의 윈도우 크기 
        self.transform_obj = transform                  # 어떤 스케일링을 할 것인지 (.scaler 매소드 중 이용)
        self.target_columns = target_columns

        # 전체 데이터의 timestep과 거래 가능 여부 
        self.total_timesteps = list(df.index)
        self.trading_available_flags = self._get_trading_available_flag(df)

        self.col_list = df.columns.to_list()            # 열 이름 
        # 실제 state에 사용할 feature 컬럼 목록 (기본: 전체)
        self.feature_columns = None
        self.target_len = len(self.col_list)
        # df 
        total_df = self._add_technical_indicators(df) # 기술적 분석 지표를 포함 
        self.cleaned_df = self._remove_Nan(total_df)    # 결측치 전부 삭제

        if self.target_columns:
            missing = [col for col in self.target_columns if col not in self.cleaned_df.columns]
            if missing:
                raise ValueError(f"Missing target columns in dataset: {missing}")
            self.feature_columns = list(self.target_columns)
        else:
            self.feature_columns = self.cleaned_df.columns.to_list()

        self.target_len = len(self.feature_columns)
        self.states, self.close_prices, self.timesteps = self._split_dataset(self.cleaned_df, window_size)
        
        # transform 
        if self.transform_obj:
            self._states = self.states 
            self.states = self.transform_obj.fit_transform(self.states)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        """
        해당 인덱스의 선물 시장 상태(기술적 분석 지표 포함), 종가(가격 계산을 위함), 시간을 반환한다. 
        """
        return self.states[idx], self.close_prices[idx], self.timesteps[idx]
    
    def _remove_Nan(self, df):
        return df[~df.isnull().any(axis=1)]

    def _get_trading_available_flag(self, df):
        """거래 가능 시간(당일 장 마지막 틱에는 거래가 불가능하다.)"""
        grouped_df = df.copy()
        # 연속 여부 계산 
        grouped_df['continues'] = (grouped_df.index + timedelta(minutes=1)).isin(grouped_df.index)

        # 연속이 끊기기 직전 idx가 기준이 되게 breakpoint 지정
        breakpoint = ~grouped_df['continues'].shift(fill_value=True)

        # 누적합을 기준으로 그룹핑 
        grouped_df['group_id'] = breakpoint.cumsum()

        # 그룹이 한 개면 거래 불가 시점 
        grouped_df['trading_available'] = grouped_df.groupby('group_id')['group_id'].transform('size') != 1

        return grouped_df['trading_available'].tolist()

    def _split_dataset(self, df, window_size):
        """
        데이터 셋을 윈도우 길이만큼 잘라 준비한다. 
        이때 이 클래스가 반환하는 정보인 데이터, 종가, 시간까지 같이 저장한다. 
        """
        states = []
        close_prices = []
        timesteps = []

        total_iteration = len(df) - window_size + 1

        for i in range(total_iteration):
            feature_slice = df.loc[:, self.feature_columns]
            state = feature_slice.iloc[i:i+window_size].to_numpy(dtype=np.float32)
            close = df['close'].iloc[i + window_size - 1]
            time = df.index[i + window_size - 1]

            # 거래 가능 시간인지 확인하고, 가능하면 append, 아니면 지나간다. 
            idx = self.total_timesteps.index(time)
            if self.trading_available_flags[idx]:
                states.append(state)
                close_prices.append(close)
                timesteps.append(time)

        return states, close_prices, timesteps

    def _add_technical_indicators(self, df):
        """기술 지표를 한 번에 추가한다."""

        df = df.copy()
        df = add_basic_indicators(df)
        df = add_trend_indicators(df)
        df = add_momentum_indicators(df)
        df = add_volume_indicators(df)
        df = add_volatility_indicators(df)

        return df
    
    def reach_end(self, current_timestep):
        """이 데이터가 해당 데이터 셋의 마지막 데이터인지를 반환"""
        return self.timesteps[-1] == current_timestep
        
