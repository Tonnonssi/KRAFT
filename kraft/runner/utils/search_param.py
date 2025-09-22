from dataclasses import is_dataclass, asdict
from typing import Any, Dict
from collections.abc import Mapping

def search_param(cfg: Any, key: str) -> Any:
    """
    Config dataclass 안에서 key와 이름이 같은 필드를 찾아 반환.
    - key: 최종 필드명 (예: "lr", "gamma", "max_steps")
    - 중복되면 KeyError 발생
    """
    matches: Dict[str, Any] = {}

    def _rec(obj: Any, prefix: str = ""):
        if is_dataclass(obj):
            obj = asdict(obj)
        # Accept any Mapping (dict, OmegaConf DictConfig, etc.)
        if isinstance(obj, Mapping):
            for k, v in obj.items():
                if k == key:
                    matches[f"{prefix}{k}"] = v
                _rec(v, f"{prefix}{k}.")
        # 리스트나 스칼라는 탐색 종료

    _rec(cfg)
    if not matches:
        raise KeyError(f"'{key}' not found in config")
    if len(matches) > 1:
        raise KeyError(f"Ambiguous key '{key}' found in {list(matches.keys())}")
    return list(matches.values())[0]

# 여러 키 한 번에 찾기
def search_params(cfg: Any, keys: list[str]):
    """
    Return values in the order of keys to be splatted positionally.
    """
    return [search_param(cfg, k) for k in keys]
