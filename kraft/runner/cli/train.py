from __future__ import annotations

import os
from pathlib import Path

from omegaconf import OmegaConf, DictConfig
from hydra.utils import to_absolute_path
import hydra

from ..utils.materialize import materialize_config
from ..utils import resolve_class, get_df, set_global_seed
from ..utils.builder import build_agent, build_reward_ftn, build_callbacks
from ..trainer import Trainer


def _resolve_config_dir() -> Path:
    """
    Locate the Hydra config directory in both editable (local repo) and
    installed environments.
    Priority:
        1. Environment variable KRAFT_CONFIG_DIR
        2. <repo>/config (current working directory)
        3. <package_root>/config (if bundled inside the package)
        4. <site-packages>/config (legacy behaviour)
    """
    env_dir = os.getenv("KRAFT_CONFIG_DIR")
    candidates = []
    if env_dir:
        candidates.append(Path(env_dir))

    cwd_config = Path.cwd() / "config"
    candidates.append(cwd_config)

    package_root = Path(__file__).resolve().parents[2]
    candidates.append(package_root / "config")
    candidates.append(package_root.parent / "config")

    for path in candidates:
        if path.is_dir():
            return path

    searched = "\n - ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Hydra config directory not found. Checked:\n - " + searched
    )


CONFIG_DIR = _resolve_config_dir()


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

    scaler_name = cfg.dataset.scaler
    if isinstance(scaler_name, str):
        normalized = scaler_name.strip().lower()
        use_scaler = normalized not in ("", "none", "null")
    else:
        use_scaler = scaler_name is not None

    scaler_cls = None
    if use_scaler:
        scaler_cls = resolve_class(scaler_name, "kraft.env.core.features.dataset.utils.scaler")

    df = get_df(to_absolute_path(cfg.dataset.path))

    trainer = Trainer(agent, env_cls, scaler_cls, reward_ftn, df, cfg, callbacks)
    trainer()


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig):
    _run(cfg)


if __name__ == "__main__":
    main()
