# KRAFT
KRAFT (KOSPI 200 Reinforcement-learning Agent for Futures Trading)

## Launch 
```
# mac / linux
cd kraft 
chmod +x run_kraft_train.command
```

## Project Structure

```
project_root/
├── kraft/                         # Core Python package
│   ├── agent/
│   │   └── ppo_agent.py
│   ├── env/
│   │   ├── __init__.py
│   │   ├── core/
│   │   │   ├── base_env.py
│   │   │   ├── features/
│   │   │   │   ├── account.py
│   │   │   │   ├── dataset/
│   │   │   │   │   ├── dataset.py
│   │   │   │   │   └── utils/
│   │   │   │   │       └── scaler.py
│   │   │   └── utils/
│   │   │       ├── done_conditions.py
│   │   │       └── reward_schemes.py
│   │   ├── environments/
│   │   │   ├── episodic_envs.py
│   │   │   └── nonepisodic_envs.py
│   │   └── flows/
│   │       ├── stream_flow.py
│   │       └── sampling_flow.py
│   ├── model/
│   │   ├── multistate_actor_critic.py
│   │   └── encoder/
│   │       └── timeseries/
│   │           └── ctts/
│   │               └── ctts.py
│   │       └── vector/
│   │           └── agent_state.py
│   └── runner/
│       ├── cli/train.py
│       ├── trainer.py
│       ├── core/
│       │   ├── running.py
│       │   └── visualization/
│       │       ├── for_episode.py
│       │       ├── for_step.py
│       │       ├── action.py
│       │       └── pnl.py
│       └── utils/
│           ├── builder.py
│           ├── materialize.py
│           ├── config_loader.py
│           └── resolve_class.py
├── config/                        # Hydra configuration tree
│   ├── config.yaml
│   ├── agent/
│   ├── dataset/
│   ├── env/
│   ├── model/
│   ├── reward/
│   └── splitter/
├── data/                          # Data artifacts
│   ├── raw/
│   └── processed/
├── EDA/                           # Exploratory notebooks
├── logs/                          # Training and evaluation outputs
├── scripts/                       # Convenience entrypoints
│   └── run_train.py
├── run_kraft_train.command        # `uv run kraft-train` helper
├── pyproject.toml                 # Project metadata & dependencies
├── uv.lock                        # Locked dependency versions
└── README.md
```
