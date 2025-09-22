from __future__ import annotations

from datetime import datetime
from typing import Sequence

from omegaconf import DictConfig, OmegaConf

from .device import get_device
import pickle 


def _ensure_list(x) -> list:
    if x is None:
        return []
    if isinstance(x, str):
        return [s.strip() for s in x.split(',') if s.strip()]
    if isinstance(x, Sequence) and not isinstance(x, (str, bytes)):
        return list(x)
    return [x]

def materialize_config(cfg: DictConfig) -> DictConfig:
    """
    Inject derived/runtime fields into a Hydra DictConfig so that the
    rest of the codebase can access them directly (no dataclass wrapper).
    - action.n_actions, action.action_space
    - run.date
    - dataset.target_ts_values (normalized list)
    - model.timeseries_input_dim
    - top-level convenience: device, n_actions, date, fname, action_space
    - alias keys for INIT_SEQ expectations (input_dim, embed_dim, n_actions)
    """
    for section in (cfg, cfg.run, cfg.dataset, cfg.action, cfg.model):
        OmegaConf.set_struct(section, False)

    # run.date
    today = datetime.today().strftime("%Y_%m_%d")
    cfg.run.date = today

    # dataset targets
    if "target_ts_values" in cfg.dataset and cfg.dataset.target_ts_values is not None:
        targets = _ensure_list(cfg.dataset.target_ts_values)
    else:
        targets = _ensure_list(getattr(cfg.dataset, "target_ts_values_str", None))
    cfg.dataset.target_ts_values = targets

    # action fields
    k = int(cfg.action.single_execution_cap)
    cfg.action.n_actions = 2 * k + 1
    cfg.action.action_space = list(range(-k, k + 1))

    # model input dim from dataset targets
    cfg.model.timeseries_input_dim = len(cfg.dataset.target_ts_values)

    # runtime device
    device = get_device()

    # Top-level conveniences required by INIT_SEQ (keep unique to avoid ambiguity)
    cfg.device = str(device)  # torch device를 문자열로 저장해 Hydra 호환 유지

    # Aliases to satisfy INIT_SEQ keys expected by builders
    # For CTTS encoder
    cfg.input_dim = cfg.model.timeseries_input_dim
    cfg.embed_dim = cfg.model.timeseries_embed_dim

    return cfg
