# splits_example_nonoverlap.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd

ISO_FMT = "%Y-%m-%d"

@dataclass
class DatasetIndex:
    train: List[Tuple[str, str]]
    valid: List[Tuple[str, str]]
    test:  List[Tuple[str, str]]

# ---------- utils ----------
def _build_day_spans(idx: pd.DatetimeIndex) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """날짜 단위로 (첫 시각, 마지막 시각) 튜플을 정렬 반환."""
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError("df.index must be a DatetimeIndex.")
    if len(idx) == 0:
        return []
    by_day = pd.Series(idx).groupby(idx.date)
    spans = [(pd.DatetimeIndex(sub.values)[0], pd.DatetimeIndex(sub.values)[-1]) for _, sub in by_day]
    return sorted(spans, key=lambda x: x[0])

def _fmt(ts: pd.Timestamp) -> str:
    return ts.strftime(ISO_FMT)

def _spans_to_str(spans: List[Tuple[pd.Timestamp, pd.Timestamp]]) -> List[Tuple[str, str]]:
    return [(_fmt(s), _fmt(e)) for (s, e) in spans]

def _merge_index_blocks(
    day_spans: List[Tuple[pd.Timestamp, pd.Timestamp]],
    index_blocks: List[Tuple[int, int]]
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """날짜 인덱스 구간들을 인접하면 병합하여 (ts_start, ts_end)로 바꿔줌."""
    if not index_blocks:
        return []
    blocks = sorted(index_blocks)
    merged: List[Tuple[int, int]] = []
    cur_s, cur_e = blocks[0]
    for s, e in blocks[1:]:
        if s == cur_e + 1:
            cur_e = e
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return [(day_spans[s][0], day_spans[e][1]) for (s, e) in merged]

# ---------- 1) Sequential (non-overlap) + rolling valid ----------
class RollingSplitter:
    INIT_SEQ = ['df', 'test_ratio', 'train_window_days', 'valid_window_days']
    def __call__(
        self,
        df: pd.DataFrame,
        test_ratio: float = 0.2,
        train_window_days: int = 252 * 2,
        valid_window_days: int = 252,
    ) -> DatasetIndex:
        """
        겹치지 않는 롤링:
        train[i]  = [cur, cur+T-1]
        valid[i]  = [cur+T, cur+T+V-1]
        next cur  = cur+T (즉, valid 시작일)  ← **요구사항**
        테스트는 맨 뒤 test_ratio 비율(≥1일)을 한 덩어리로 지정.
        """
        day_spans = _build_day_spans(df.index)
        n_days = len(day_spans)
        if n_days < 3:
            raise ValueError("Not enough days to split.")

        # test 영역을 뒤에서 확보
        n_test = max(1, int(round(n_days * test_ratio)))
        test_start_idx = n_days - n_test
        train_area_end = test_start_idx - 1  # train/valid은 여기까지만 사용

        train_blocks: List[Tuple[int, int]] = []
        valid_blocks: List[Tuple[int, int]] = []

        cur = 0
        while True:
            tr_s = cur
            tr_e = tr_s + train_window_days - 1
            va_s = tr_e + 1
            va_e = va_s + valid_window_days - 1

            # train/valid 모두 train_area 안에서 끝나야 함
            if tr_e > train_area_end or va_e > train_area_end:
                break

            train_blocks.append((tr_s, tr_e))
            valid_blocks.append((va_s, va_e))

            # 다음 train의 시작 = 이번 valid의 시작 (겹치지 않음)
            cur = va_s

        # test span = 남은 뒤쪽 전부
        test_span = []
        if test_start_idx < n_days:
            test_span = [(day_spans[test_start_idx][0], day_spans[-1][1])]

        train_spans = [(day_spans[s][0], day_spans[e][1]) for (s, e) in train_blocks]
        valid_spans = [(day_spans[s][0], day_spans[e][1]) for (s, e) in valid_blocks]

        return DatasetIndex(
            train=_spans_to_str(train_spans),
            valid=_spans_to_str(valid_spans),
            test=_spans_to_str(test_span),
        )

# ---------- 2) k-block → n-subblock; non-picked are all train ----------
class KFoldSpliiter:
    INIT_SEQ = ['df', 'k_blocks', 'n_subblocks', 'rng']
    def __call__(
        self,
        df: pd.DataFrame,
        k_blocks: int = 6,
        n_subblocks: int = 4,
        rng: Optional[np.random.Generator] = None,
    ) -> DatasetIndex:
        """
        전체 기간을 k등분 → 각 큰 블록을 n등분.
        각 큰 블록에서 서로 다른 서브블록 2개를 뽑아 (valid, test)로 지정.
        **선택되지 않은 모든 서브블록은 전부 train** (전역으로 병합).
        """
        rng = rng or np.random.default_rng()
        day_spans = _build_day_spans(df.index)
        n_days = len(day_spans)
        if n_days < k_blocks:
            k_blocks = n_days

        cut_k = np.linspace(0, n_days, k_blocks + 1, dtype=int)

        train_idx_blocks: List[Tuple[int, int]] = []
        valid_spans: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        test_spans : List[Tuple[pd.Timestamp, pd.Timestamp]] = []

        for kb in range(k_blocks):
            s_idx, e_idx_ex = cut_k[kb], cut_k[kb + 1]
            if e_idx_ex - s_idx <= 0:
                continue
            block_len = e_idx_ex - s_idx

            n_sub = min(max(1, n_subblocks), block_len)
            cut_n = np.linspace(s_idx, e_idx_ex, n_sub + 1, dtype=int)
            sub_blocks = [(cut_n[j], cut_n[j+1]-1) for j in range(n_sub) if cut_n[j+1]-cut_n[j] > 0]
            if not sub_blocks:
                continue

            # valid, test 뽑기 (가능하면 서로 다르게)
            picks = rng.choice(len(sub_blocks), size=min(2, len(sub_blocks)), replace=False)
            v_s, v_e = sub_blocks[picks[0]]
            valid_spans.append((day_spans[v_s][0], day_spans[v_e][1]))
            t_added = False
            if len(picks) > 1:
                t_s, t_e = sub_blocks[picks[1]]
                test_spans.append((day_spans[t_s][0], day_spans[t_e][1]))
                t_added = True

            # **나머지 전부 train**
            for j, (ss, ee) in enumerate(sub_blocks):
                if j == picks[0]:
                    continue
                if t_added and j == picks[1]:
                    continue
                train_idx_blocks.append((ss, ee))

        # 전역적으로 인접 서브블록 병합
        train_spans = _merge_index_blocks(day_spans, train_idx_blocks)

        return DatasetIndex(
            train=_spans_to_str(train_spans),
            valid=_spans_to_str(valid_spans),
            test=_spans_to_str(test_spans),
        )

# ---------- demo ----------
def make_dummy_df(start="2010-01-01", end="2017-12-29", sessions_per_day=3) -> pd.DataFrame:
    """영업일×하루 세션 수 만큼 인덱스를 만드는 더미 데이터."""
    days = pd.bdate_range(start=start, end=end, freq="C")
    times = pd.to_datetime([f"{h:02d}:00" for h in range(9, 9 + sessions_per_day)]).time
    idx = [pd.Timestamp.combine(d.date(), t) for d in days for t in times]
    df = pd.DataFrame({"value": np.arange(len(idx))}, index=pd.DatetimeIndex(idx))
    return df

if __name__ == "__main__":
    df = make_dummy_df()
    splitter_1 = RollingSplitter()
    splitter_2 = KFoldSpliiter()

    print("=== 1) Sequential (non-overlap) + rolling valid ===")
    ds_seq = splitter_1(
        df,
        test_ratio=0.2,          # 뒤 20%는 테스트
        train_window_days=252*2, # 2년 (영업일 가정)
        valid_window_days=252,   # 1년
    )
    print(f"- #pairs: {len(ds_seq.train)} (train/valid 쌍)")
    print("  train:", ds_seq.train[:2])
    print("  valid:", ds_seq.valid[:2])
    print("  test   :", ds_seq.test)

    print("\n=== 2) K-block → N-subblock; others = train ===")
    ds_k = splitter_2(
        df,
        k_blocks=6,
        n_subblocks=4,
    )
    print(f"- train spans: {len(ds_k.train)} (merged)")
    print(f"- valid spans: {len(ds_k.valid)}  (one per big block if possible)")
    print(f"- test  spans: {len(ds_k.test)}  (one per big block if possible)")
    print("  sample train:", ds_k.train[:])
    print("  sample valid:", ds_k.valid[:])
    print("  sample test :", ds_k.test[:])