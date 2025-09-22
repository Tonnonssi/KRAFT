"""Thin wrapper로 `python -m kraft.runner.cli.train` 실행을 대신한다."""

import sys
from pathlib import Path

# repo 루트를 모듈 검색 경로에 추가
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from kraft.runner.cli.train import main

if __name__ == "__main__":
    main()
