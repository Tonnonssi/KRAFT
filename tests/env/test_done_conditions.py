import pathlib

import pandas as pd

from kraft.env.core.features.account import Account
from kraft.env.core.utils.done_conditions import check_insufficient, is_maturity_data


def make_account(initial_budget: int = 1_000_000) -> Account:
    return Account(
        initial_budget=initial_budget,
        position_cap=10,
        initial_timestep=pd.Timestamp("2024-01-01"),
        slippage_factor=0.0,
    )


def test_check_insufficient_flags_when_account_lacks_margin():
    account = make_account(initial_budget=500_000)

    # 현재 시세 100pt에서 1계약 매수 → 예치 증거금 인출로 가용 잔고가 크게 줄어듦
    account.step(decoded_action=1, market_pt=100, next_timestep=pd.Timestamp("2024-01-02"))
    account.step(decoded_action=1, market_pt=105, next_timestep=pd.Timestamp("2024-01-03"))

    done, info = check_insufficient(account)
    
    assert done is False
    assert info == "insufficient"


def test_check_insufficient_no_flag_when_account_can_trade():
    account = make_account(initial_budget=1_000_000)
    account.market_pt = 100  # 최소 필요 증거금 계산을 위한 시세 설정

    done, info = check_insufficient(account)

    assert done is False
    assert info == ""


def test_is_maturity_data_detects_maturity_on_day_change():
    maturity_dates = pd.to_datetime(["2024-04-11", "2024-05-09"])
    current = pd.Timestamp("2024-04-11 15:30:00")
    next_ts = pd.Timestamp("2024-04-12 09:00:00")

    done, info = is_maturity_data(maturity_dates, next_ts, current)

    assert done is True
    assert info == "maturity_data"


def test_is_maturity_data_not_triggered_without_day_change():
    maturity_dates = pd.to_datetime(["2024-04-11"])
    current = pd.Timestamp("2024-04-11 15:30:00")
    next_ts = pd.Timestamp("2024-04-11 15:35:00")

    done, info = is_maturity_data(maturity_dates, next_ts, current)

    assert done is False
    assert info == ""


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([str(pathlib.Path(__file__))]))
