from dataclasses import dataclass
from ..utils.device import get_device
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, Union, IO
import yaml  # pip install pyyaml
import pandas as pd
from pathlib import Path
from datetime import datetime

@dataclass
class RunConfig:
    fname: str
    agent_class: str          # e.g., "PPOAgent"
    env: str                  # e.g., "EpisodicStreamEnv"
    timeseries_network: str   # e.g., "CTTS"
    train_epoch: int
    print_step_log_in: int
    print_episode_log_in: int
    print_env_log_in: int
    save_interval: int
    log_on: bool
    timer_on: bool
    checkpoint_on: bool
    visualization_on: bool
    date: Optional[str] = None
    seed: Optional[int] = None
    device=get_device()  

    def _get_date(self):
        """date를 받음"""
        today = datetime.today()
        # 문자열로 변환 (예: "2025-09-16")
        return today.strftime("%Y_%m_%d")    

@dataclass
class SplitterConfig:
    """
    Config for data split strategy.
    Avoid non-serializable types (e.g., DataFrame) in config objects.
    """
    name: Optional[str] = None
    # If you need a dataframe, load it in code, not via YAML.
    df: Optional[str] = None  # optional path or identifier
    k_blocks: Optional[int] = None 
    n_subblocks: Optional[int] = None 
    test_ratio: Optional[float] = None 
    train_window_days: Optional[int] = None 
    valid_window_days: Optional[int] = None 
    seed: Optional[int] = None



@dataclass
class AgentConfig:
    algo: str                 # e.g., "ppo"
    n_steps: int
    batch_size: int
    value_coeff: float
    entropy_coeff: Union[float, Dict[str, Any]]
    clip_eps: float
    gamma: float
    lr: float
    entry_coeff: float
    kappa: float
    beta: float
    regulation: float
    gae_lam: float


@dataclass
class EnvConfig:
    slippage: float
    start_budget: int
    max_steps: int


@dataclass
class DatasetConfig:
    path: str
    window_size: int
    scaler: str  # e.g., "RobustScaler"
    # Prefer list in YAML; string (comma-separated) is also accepted for backward-compat.
    target_ts_values_str: Optional[str] = None
    target_ts_values: Optional[list] = None

    def get_target_ts_values(self):
        """Ensure target_ts_values is a list (accept list or comma-separated string)."""
        if self.target_ts_values and isinstance(self.target_ts_values, list):
            # Already a proper list
            return self.target_ts_values
        if self.target_ts_values_str and isinstance(self.target_ts_values_str, str):
            self.target_ts_values = [item.strip() for item in self.target_ts_values_str.split(',')]
            return self.target_ts_values
        # Default to empty list if nothing provided
        self.target_ts_values = []
        return self.target_ts_values

@dataclass
class ActionConfig:
    single_execution_cap: int
    position_cap: int
    n_actions: Optional[int] = None
    action_space: Optional[list] = None 

    def _fill_blank_space(self):
        """비어있는 행동수와 행동 공간을 채움"""
        self.n_actions = 2*self.single_execution_cap + 1
        self.action_space = list(range(-self.single_execution_cap, self.single_execution_cap+1))  


@dataclass
class RewardConfig:
    w_profit: float
    w_risk: float
    w_regret: float
    margin_call_penalty: float
    maturity_date_penalty: float
    bankrupt_penalty: float
    goal_reward_bonus: float


@dataclass
class ModelConfig:
    name: str                         # e.g., "CTTS"
    agent_input_dim: int
    timeseries_embed_dim: int
    kernel_size: int
    stride: int
    agent_hidden_dim: int
    agent_out_dim: int
    fusion_hidden_dim: int
    num_layers: int
    num_heads: int
    d_ff: int
    dropout: float
    timeseries_input_dim: Optional[int] = None   # to be set at runtime if 0

    def _get_timeseries_input_dim(self, timeseries_input_value: list):
        """ts input dim을 데이터 셋 개수를 셈"""
        self.timeseries_input_dim = len(timeseries_input_value)


# -------------------- Top-level Config --------------------

@dataclass
class Config:
    run: RunConfig
    agent: AgentConfig
    env: EnvConfig
    dataset: DatasetConfig
    action: ActionConfig
    reward: RewardConfig
    model: ModelConfig
    splitter: SplitterConfig

    # ---- Utilities ----
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    # ---- Convenience accessors (for legacy call sites) ----
    @property
    def device(self):
        return self.run.device

    @property
    def date(self):
        return self.run.date

    @property
    def fname(self):
        return self.run.fname

    @property
    def n_actions(self):
        return self.action.n_actions

    @staticmethod
    def _require(section: Dict[str, Any], name: str) -> Dict[str, Any]:
        if name not in section or section[name] is None:
            raise KeyError(f"Missing required config section: '{name}'")
        return section[name]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        """
        Build Config from a Python dict that mirrors the YAML structure.
        """
        run = RunConfig(**cls._require(d, "run"))
        agent = AgentConfig(**cls._require(d, "agent"))
        env = EnvConfig(**cls._require(d, "env"))
        dataset = DatasetConfig(**cls._require(d, "dataset"))
        action = ActionConfig(**cls._require(d, "action"))
        reward = RewardConfig(**cls._require(d, "reward"))
        model = ModelConfig(**cls._require(d, "model"))
        splitter = SplitterConfig(**cls._require(d, "splitter"))
        cfg = cls(run=run, agent=agent, env=env, dataset=dataset, action=action, reward=reward, model=model, splitter=splitter)
        # Auto-populate derived fields on construction
        cfg.fill_blank()
        return cfg

    @classmethod
    def load_yaml(cls, source) -> "Config":
        """
        Load from a YAML path or an open file-like object and return Config.
        """
        if isinstance(source, (str, Path)):
            path = Path(source)
            with path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        else:
            # file-like object라고 가정
            data = yaml.safe_load(source)

        cfg = cls.from_dict(data)
        return cfg
    
    def fill_blank(self):
        self.action._fill_blank_space()
        # materialize date
        self.run.date = self.run._get_date()
        # normalize dataset target list
        self.dataset.target_ts_values = self.dataset.get_target_ts_values()
        # set model input dim from dataset targets
        self.model._get_timeseries_input_dim(self.dataset.target_ts_values)


    # (선택) 간단 검증: 음수 불가/범위 등
    def validate(self) -> None:
        if self.run.train_epoch <= 0:
            raise ValueError("run.train_epoch must be > 0")
        if self.agent.n_steps <= 0:
            raise ValueError("agent.n_steps must be > 0")
        if not (0.0 < self.agent.gamma <= 1.0):
            raise ValueError("agent.gamma must be in (0, 1]")
        if self.env.max_steps <= 0:
            raise ValueError("env.max_steps must be > 0")
        if self.dataset.window_size <= 0:
            raise ValueError("dataset.window_size must be > 0")


# -------------------- Example --------------------
if __name__ == "__main__":
    # config.yaml 불러오기
    cfg = Config.load_yaml("config.yaml")
    cfg.fill_blank()
    cfg.validate()

    # 사용 예
    print("Agent algo:", cfg.agent.algo)
    print("Env max steps:", cfg.env.max_steps)
    print("Timeseries network:", cfg.run.timeseries_network)

    # dict로 내보내기
    as_dict = cfg.to_dict()
    # print(as_dict)
