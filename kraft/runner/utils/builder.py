from typing import Dict, Any

import numpy as np

from . import resolve_class
from .search_param import search_params
from ...model import MultiStatePV, MultiCritics, AgentStateEncoder, BaseFusion
from ...env.core.utils.reward_schemes import RRPAReward, MultiRRPAReward
from ..core.callback import *

def build_agent(cfg):
    """agent 인스턴스 생성 """
    agent_cls = resolve_class(cfg.agent.algo, "kraft.agent")
    model = build_model(cfg)
    params = search_params(cfg, agent_cls.INIT_SEQ)
    return agent_cls(model, *params)

def build_model(cfg):
    """
    model의 시계열 처리 블럭을 만들고, 
    multi-encoder-head 구조를 완성
    """
    ts_encoder_cls = resolve_class(cfg.run.timeseries_network, "kraft.model")
    params = search_params(cfg, ts_encoder_cls.INIT_SEQ)
    ts_encoder = ts_encoder_cls(*params)

    model_structure = MultiCritics if cfg.run.multi_critics else MultiStatePV

    multi_state_params = search_params(cfg, model_structure.INIT_SEQ)

    return model_structure(
        ts_encoder,
        AgentStateEncoder,
        BaseFusion,
        *multi_state_params
    )

def build_reward_ftn(cfg):
    """reward ftn 인스턴스 생성"""
    reward_params = search_params(cfg, RRPAReward.INIT_SEQ)
    
    if cfg.run.multi_critics:
        return MultiRRPAReward(*reward_params)
    else:
        return RRPAReward(*reward_params)


def build_callbacks(cfg):
    """callback 인스턴스 생성"""
    instances = []
    if cfg.run.log_on:
        instances.append(LoggingCallback(cfg.run.print_episode_log_in, 
                                         cfg.run.print_step_log_in))
    if cfg.run.timer_on:
        instances.append(TimerCallback())
    if cfg.run.checkpoint_on:
        instances.append(CheckpointCallback())
    if cfg.run.visualization_on:
        instances.append(VisualizationCallback())
    if cfg.run.earlystop_on:
        instances.append(EarlyStoppingCallback())

    return instances


def build_splitter(cfg):
    """데이터 분할 전략을 반환 (df -> DatasetIndex)."""
    splitter_cfg = getattr(cfg, "splitter", None)
    if splitter_cfg is None or getattr(splitter_cfg, "name", None) is None:
        raise ValueError("Splitter configuration is missing 'name'.")

    splitter_cls = resolve_class(splitter_cfg.name, "kraft.runner.core.splitter")
    splitter = splitter_cls()

    # splitter.__call__ 시그니처와 cfg를 매핑할 수 있는 값만 전달
    allowed_keys = (
        "test_ratio",
        "train_window_days",
        "valid_window_days",
        "k_blocks",
        "n_subblocks",
    )
    base_kwargs: Dict[str, Any] = {}
    for key in allowed_keys:
        value = getattr(splitter_cfg, key, None)
        if value is not None:
            base_kwargs[key] = value

    splitter_seed = getattr(splitter_cfg, "seed", None)
    if splitter_seed is not None:
        base_kwargs["rng"] = np.random.default_rng(splitter_seed)

    def apply(df, **override):
        if df is None:
            raise ValueError("Splitter requires a dataframe to operate.")
        call_kwargs = {**base_kwargs, **{k: v for k, v in override.items() if v is not None}}
        return splitter(df, **call_kwargs)

    return apply
