from .base import Callback
import os
import torch
import pathlib
from collections import deque

class CheckpointCallback(Callback):
    """체크포인트를 관리 감독하는 콜백"""
    def on_fit_end(self, logs=None): 
        """파일에 저장된 신경망이 총 몇 개인지 알려줌"""
        # self.trainer.config
        directory_path = pathlib.Path(self.trainer.models_path) 
        file_count = sum(1 for item in directory_path.iterdir() if item.is_file())
        self.logging(f"'{directory_path}'에 저장되어 있는 신경망 개수: {file_count}개")

    def on_train_begin(self, logs=None): 
        """지표 초기화"""
        self.best_ave_reward = None
        self.best_winrate = None
        self.best_pnl = None

        self.trainer.best_reward_model = None 
        self.trainer.best_winrate_model = None 
        self.trainer.best_pnl_model = None
        self.trainer.per_steps_model = deque(maxlen=10) 

    def on_episode_end(self, logs=None): 
        """업데이트"""
        logs = logs or {}
        ave_epi_reward = logs.get('epi_reward') 
        winrate = logs.get('winrate')
        idx = logs.get('index')
        model = logs.get('model')
        pnl = logs.get('pnl')
    
        if model is None:
            return

        if ave_epi_reward is not None and (
            self.best_ave_reward is None or ave_epi_reward >= self.best_ave_reward
        ):
            self.trainer.best_reward_model = model
            self.best_ave_reward = ave_epi_reward
        
        if winrate is not None and (
            self.best_winrate is None or winrate >= self.best_winrate
        ):
            self.trainer.best_winrate_model = model 
            self.best_winrate = winrate

        if pnl is not None and (
            self.best_pnl is None or pnl >= self.best_pnl
        ):
            self.trainer.best_pnl_model = model 
            self.best_pnl = pnl

        if idx is not None and model is not None and idx % 10 == 0:
            self.trainer.per_steps_model.append(model)

    def on_train_end(self, logs=None):
        """학습 중에 저장된 여러 모델을 저장한다."""
        logs = logs or {}
        dataset_flag = logs.get("dataset_flag", "N/A")
        self.save_model_to(self.trainer.models_path, dataset_flag)
    

    def save_model_to(self, path, dataset_flag):
        """모든 신경망 가중치를 저장한다."""
        # 최고 보상 모델 저장
        if self.trainer.best_reward_model is not None:
            torch.save(self.trainer.best_reward_model, os.path.join(path, f'I{dataset_flag}bestreward.pth'))
            self.logging("[Saved] best_reward_model")

        # 최고 손익 모델 저장
        if self.trainer.best_pnl_model is not None:
            torch.save(self.trainer.best_pnl_model, os.path.join(path, f'I{dataset_flag}best_pnl_model.pth'))
            self.logging("[Saved] best_pnl_model")

        # 최고 승률 모델 저장
        if self.trainer.best_winrate_model is not None:
            torch.save(self.trainer.best_winrate_model, os.path.join(path, f'I{dataset_flag}best_winrate_model.pth'))
            self.logging("[Saved] best_winrate_model")

        # n-step마다 모델 저장 
        recent_models = list(self.trainer.per_steps_model)
        for idx, model_state in enumerate(recent_models):
            torch.save(model_state, os.path.join(path, f'I{dataset_flag}_{(idx+1)}steps.pth'))
        self.logging(f"[Saved] {len(recent_models)} recent models")

