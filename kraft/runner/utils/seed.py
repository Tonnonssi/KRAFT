from __future__ import annotations

from typing import Optional

import random

import numpy as np
import torch


def set_global_seed(seed: Optional[int]):
    """전역 난수 시드를 고정한다."""
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

