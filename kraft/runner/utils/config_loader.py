from __future__ import annotations
from typing import Any, Mapping, MutableMapping
import copy
import yaml  # pip install pyyaml
from pathlib import Path

def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict:
    """
    Dict는 재귀 병합, 그 외 타입(스칼라/리스트 등)은 override로 교체.
    base는 건드리지 않고 새 dict 반환.
    """
    result: MutableMapping[str, Any] = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if (
            k in result
            and isinstance(result[k], Mapping)
            and isinstance(v, Mapping)
        ):
            result[k] = deep_merge(result[k], v)  # dict끼리만 재귀 병합
        else:
            result[k] = copy.deepcopy(v)  # 그 외 타입은 통째로 교체
    return dict(result)

def merge_multiple_yamls(path_list, base_path):
    # 장기 : Base와 default를 적절히 이용해보자 
    base = load_yaml(base_path)     
    if len(path_list) == 0:
        return base
    
    for path in path_list:
        patch = load_yaml(path)
        base = deep_merge(base, patch)
    
    return base 
# 사용 예시
if __name__ == "__main__":
    base = load_yaml("base.yaml")
    patch = load_yaml("override.yaml")  # 일부 키만 들어있는 파일
    merged = deep_merge(base, patch)

    # 결과 확인/저장
    print(yaml.safe_dump(merged, sort_keys=False, allow_unicode=True))
    with open("merged.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(merged, f, sort_keys=False, allow_unicode=True)
