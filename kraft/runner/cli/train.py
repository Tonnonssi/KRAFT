from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf, DictConfig
from hydra.utils import to_absolute_path
import hydra

from ..utils.materialize import materialize_config
from ..utils import resolve_class, get_df, set_global_seed
from ..utils.builder import build_agent, build_reward_ftn, build_callbacks
from ..trainer import Trainer


CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"


def _run(cfg: DictConfig):
    # Inject derived/runtime fields
    cfg = materialize_config(cfg)

    set_global_seed(getattr(cfg.run, "seed", None))

    # Build components
    agent = build_agent(cfg)
    reward_ftn = build_reward_ftn(cfg)
    callbacks = build_callbacks(cfg)

    # Resolve classes
    env_cls = resolve_class(cfg.run.env, "kraft.env")
    scaler_cls = resolve_class(cfg.dataset.scaler, "kraft.env.core.features.dataset.utils.scaler")

    df = get_df(to_absolute_path(cfg.dataset.path))

    trainer = Trainer(agent, env_cls, scaler_cls, reward_ftn, df, cfg, callbacks)
    trainer()


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig):
    _run(cfg)


if __name__ == "__main__":
    main()
