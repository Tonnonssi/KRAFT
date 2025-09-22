import os
import random
import shutil
from pathlib import Path

import torch
from collections import deque 

from .core.running import run_loop
from .utils import ensure_dir, get_device
from .utils.builder import build_splitter

class Trainer:
    def __init__(self, agent, env_class, scaler_class, reward_ftn, df, config, callbacks=None):
        self.agent = agent 
        self.env_class = env_class
        self.reward_ftn = reward_ftn
        self.scaler = scaler_class()
        self.df = df
        self.config = config 

        # set 
        self.device = get_device()
        self.set_path()
        self.reset_models()
        self.splitter = build_splitter(self.config)
        self.test_timesteps = []

        self.callbacks = callbacks if callbacks else []
        for cb in self.callbacks:
            cb.set_trainer(self)

    def __call__(self):
        # 전체 학습 시작 
        fit_logs = {
            'run_name': getattr(self.config.run, 'fname', ''),
            'seed': getattr(self.config.run, 'seed', None)
        }
        for cb in self.callbacks:
            cb.on_fit_begin(fit_logs)

        split_timesteps = self.split_timeseries_data_by(self.df) 
        
        for idx, timesteps in enumerate(split_timesteps):
            # 구간 반복: 시작 
            common_log = {'dataset_flag': idx, 'train_timestep': timesteps[0], 'valid_timestep' : timesteps[1]}

            for cb in self.callbacks:
                cb.on_interval_begin(common_log)

            # setting env 
            train_period, valid_period = timesteps
            train_env = self.set_env(train_period)
            
            # 학습: ing
            train_log = self.train(train_env, train_period, idx)

            for model, name in self.models:
                if model is None:
                    self._log_missing_model(name, idx)
                    continue
                valid_env = self.set_env(valid_period)
                valid_log = self.valid(valid_env, model, name, valid_period, idx)

            # 구간 반복: 종료  
            for cb in self.callbacks:
                cb.on_interval_end(common_log)

        # 전체 학습 종료 
        for cb in self.callbacks:
            cb.on_fit_end(fit_logs)


    def train(self, env, train_period, idx=0):
        """학습"""
        common_log = {'dataset_flag': idx, 'date_range' : train_period}

        for cb in self.callbacks:
            cb.on_train_begin(common_log)

        self.agent.model.train()
        result =  run_loop(env, self.agent, 
                            self.config.agent.batch_size, self.config.agent.n_steps, 
                            is_training=True, device=self.device, callbacks=self.callbacks) 

        for cb in self.callbacks:
            cb.on_train_end(common_log)

        return result

    def valid(self, env, state_dict, name, valid_period, idx=0):
        """검증"""
        common_log = {'dataset_flag': idx, 'date_range': valid_period,'model_type': name}
        for cb in self.callbacks:
            cb.on_valid_begin(common_log)

        if state_dict is None:
            self._log_missing_model(name, idx)
            for cb in self.callbacks:
                cb.on_valid_end(common_log)
            return []

        self.agent.load_model(state_dict)
        self.agent.model.eval()
        with torch.no_grad():
            result = run_loop(env, self.agent, 
                            self.config.agent.batch_size, self.config.agent.n_steps, 
                            is_training=False, device=self.device, callbacks=self.callbacks)            
        
        # 검증 종료 
        for cb in self.callbacks:
            cb.on_valid_end(common_log)

        return result

    def _sync_hydra_outputs(self):
        try:
            from hydra.core.hydra_config import HydraConfig
        except Exception:
            return

        try:
            hydra_dir = Path(HydraConfig.get().run.dir)
        except Exception:
            return

        target_dir = Path(self.base_path) / 'hydra'
        if hydra_dir == target_dir:
            return

        if not hydra_dir.exists():
            ensure_dir(target_dir)
            return

        ensure_dir(target_dir)

        for item in hydra_dir.iterdir():
            dest = target_dir / item.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            shutil.move(str(item), dest)

        try:
            hydra_dir.rmdir()
        except OSError:
            pass

        try:
            cfg = HydraConfig.get()
            cfg.run.dir = str(target_dir)
        except Exception:
            pass

    def _log_missing_model(self, model_name, idx):
        message = f"[Skip] Model '{model_name}' is None for interval {idx}."
        for cb in self.callbacks:
            if hasattr(cb, 'logging'):
                cb.logging(message)

    def set_env(self, time_interval:tuple):
        return self.env_class(raw_df=self.df, 
                 date_range=time_interval, 
                 window_size=self.config.dataset.window_size, 
                 reward_ftn=self.reward_ftn, 
                 start_budget=self.config.env.start_budget,
                 n_actions=self.config.action.n_actions, 
                 max_steps=self.config.env.max_steps,
                 slippage_factor_rate=self.config.env.slippage,
                 position_cap=self.config.action.position_cap,
                 scaler=self.scaler,
                 target_columns=self.config.dataset.target_ts_values)

    def split_timeseries_data_by(self, df):
        """설정된 splitter를 이용해 (train, valid) 구간 리스트 반환."""
        dataset_index = self.splitter(df)

        train_spans = dataset_index.train or []
        valid_spans = dataset_index.valid or []
        self.test_timesteps = dataset_index.test or []

        if not train_spans or not valid_spans:
            raise ValueError("Splitter 결과가 비어 있습니다. train/valid 구간을 확인하세요.")

        pair_count = len(train_spans)
        if pair_count == 0:
            raise ValueError("Splitter에서 train 구간을 생성하지 못했습니다.")

        paired = []
        for idx, train_span in enumerate(train_spans):
            if idx < len(valid_spans):
                valid_span = valid_spans[idx]
            else:
                # valid 구간이 부족하면 남은 train 구간 동안 랜덤으로 재사용
                valid_span = random.choice(valid_spans)
            paired.append((train_span, valid_span))

        if not paired:
            raise ValueError("Splitter에서 train/valid 쌍을 생성하지 못했습니다.")

        return paired

    def reset_models(self):
        """모델을 초기화"""
        self.best_reward_model = None 
        self.best_winrate_model = None 
        self.best_pnl_model = None
        self.per_steps_model = deque(maxlen=10) 
        self.latest_model = None 

    @property
    def models(self):
        return [
            (self.best_reward_model, 'best_reward_model'), 
            (self.best_winrate_model, 'best_winrate_model'), 
            (self.best_pnl_model, 'best_pnl_model'), 
            (self.latest_model, 'latest_model') 
        ]

    def set_path(self):
        """path를 세팅"""
        self.base_path = f'logs/{self.config.run.date}/{self.config.run.fname}'
        self.log_file = f"{self.base_path}/train_log.txt"
        self.models_path = f"{self.base_path}/models"
        self.figures_path = f"{self.base_path}/figures"

        ensure_dir(self.base_path)
        ensure_dir(self.models_path)
        ensure_dir(self.figures_path)
        # 로그 파일이 없으면 빈 파일을 만들어둔다.
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', encoding='utf-8'):
                pass

        self._sync_hydra_outputs()
