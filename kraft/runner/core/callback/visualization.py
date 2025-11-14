from __future__ import annotations

from .base import Callback
import numpy as np
import matplotlib.pyplot as plt
from ..visualization import *
from pathlib import Path
import csv
import json

class VisualizationCallback(Callback):
    """시각화를 담당하는 콜백"""
    def __init__(self):
        self.csv_dir: Path | None = None
        self.cum_realized_pnl = 0
        self._reset_step_traker()
        self._reset_epi_traker()

    def set_trainer(self, trainer):
        super().set_trainer(trainer)
        self.csv_dir = Path(self.trainer.base_path) / "csv"
        self.csv_dir.mkdir(parents=True, exist_ok=True)

    def on_interval_begin(self, logs=None):
        self.cum_realized_pnl = 0

    def on_train_begin(self, logs=None): 
        self._reset_step_traker()
        self._reset_epi_traker()

    def on_train_end(self, logs=None): 
        """3 종류의 에피소드 시각화 진행"""
        dataset_flag = logs['dataset_flag']
        if not self.reward_list:
            self.logging(f"[Skip] Visualization (train) | interval={dataset_flag} | no episodes recorded.")
            return

        # _, ax = plt.subplots(nrows=4, ncols=1, figsize=(18, 12))
        # self.get_train_info(ax[0], logs)
        # plot_training_curves(ax[1], self.reward_list, self.loss_list)
        # plot_event_per_episodes(ax[2], self.event_list)
        # plot_histogram_with_stats(ax[3], self.pnl_ratio_list, title='"PnL Ratio Distribution"')
        
        # path = self._get_directory(f'TFI{dataset_flag-1}')
        # plt.savefig(path)
        # plt.close()
        # self.logging(f"✅ 시각화 저장 완료: {path}")
        self._save_episode_csv(phase="train", dataset_flag=dataset_flag)
        self._save_step_csv(phase="train", dataset_flag=dataset_flag)

    def on_valid_begin(self, logs=None): 
        self._reset_step_traker()
        self._reset_epi_traker()

    def on_valid_end(self, logs=None):
        dataset_flag = logs['dataset_flag']
        model_type = logs['model_type']

        if not self.timestep_list:
            self.logging(f"[Skip] Visualization (valid) | interval={dataset_flag} | model={model_type} | no steps recorded.")
            return

        long_lst, short_lst = self._split_long_short(self.entry_action_list)

        # fig, ax = plt.subplots(
        #     nrows=8,
        #     ncols=1,
        #     figsize=(22, 18),
        #     constrained_layout=True,
        #     gridspec_kw={'height_ratios': [1.1, 2.0, 1.6, 1.3, 1.6, 1.8, 1.2, 1.2]}
        # )
        # self.get_valid_info(ax[0], logs)
        # plot_market_with_actions(ax[1], self.timestep_list, self.close_price_list, self.action_list, self.maintained_vol_list)
        # plot_both_pnl_ticks(ax[2], self.timestep_list, self.unrealized_pnl_list, self.net_realized_pnl_list)
        # plot_histogram_with_stats(ax[3], self.step_reward_list, title="Step Reward Distribution")
        # plot_event_per_episodes(ax[4], self.event_list)
        # plot_action_distribution_heatmap(ax[5], self.timestep_list, self.action_list)
        # plot_long_short(ax[6], long_lst, short_lst, title='Entry Action Distribution')
        # plot_realized_pnl(ax[7], self.timestep_list, self.cum_realized_pnl_list)

        # path = self._get_directory(f'ValidI{dataset_flag}_{model_type}')
        # fig.savefig(path)
        # plt.close(fig)
        # self.logging(f"✅ 시각화 저장 완료: {path}")
        self._save_episode_csv(phase=f"valid_{model_type}", dataset_flag=dataset_flag)
        self._save_step_csv(phase=f"valid_{model_type}", dataset_flag=dataset_flag)


    def on_episode_end(self, logs=None): 
        logs = logs or {}
        self.loss_list.append(logs['loss'])
        self.reward_list.append(logs['epi_reward'])
        self.winrate_list.append(logs['winrate'])
        self.maintained_list.append(logs['maintained'])
        self.pnl_ratio_list.append(logs['pnl_ratio'])
        self.event_list.append(logs['event_dict'])
        # print(logs['event_dict'])   

    def on_step_end(self, logs=None): 
        self.timestep_list.append(logs['timestep'])
        self.close_price_list.append(logs['close_price'])
        self.unrealized_pnl_list.append(logs['unrealized_pnl'])
        self.cum_realized_pnl_list.append(logs['cum_realized_pnl'])
        self.net_realized_pnl_list.append(logs['net_realized_pnl'])
        self.maintained_vol_list.append(logs['maintained_vol'])
        self.action_list.append(logs['action'])
        self.entry_action_list.append(logs['entry_action'])
        self.liquidation_action_list.append(logs['liquidation_action'])
        self.step_reward_list.append(logs['step_reward'])

        self.cum_realized_pnl += logs['net_realized_pnl']
    
    def get_train_info(self, ax, logs):

        start_time, ended_time = logs['date_range']
        title = f'Train: {start_time} - {ended_time}'

        info = f'''
        (ave) winrate: {np.mean(self.winrate_list):3.2f} | (med) reward: {np.median(self.winrate_list):5.1f} | (ave) maintained length: {np.mean(self.maintained_list):4} | (ave) pnl_ratio: {np.mean(self.pnl_ratio_list):5.2f} '''
        ax.axis('off')

        # 3. 원하는 위치에 텍스트 추가
        ax.text(
            x=0.5,  # X축 위치 (0.0=왼쪽 끝, 1.0=오른쪽 끝)
            y=0.5,  # Y축 위치 (0.0=아래쪽 끝, 1.0=위쪽 끝)
            s=title, # 표시할 문자열
            ha='center', # 수평 정렬 (center, left, right)
            va='center', # 수직 정렬 (center, top, bottom)
            fontsize=20,
            fontweight='bold'
        )

        # 4. 다른 텍스트도 추가 가능
        ax.text(x=0.5, y=0.3, s=info,
                ha='center', va='center', fontsize=12)

    def get_valid_info(self, ax, logs):
        start_time, ended_time = logs['date_range']
        model_type = logs['model_type']

        title = f'Valid: {start_time} - {ended_time}'
        ax.axis('off')

        info = f'Model Type: {model_type} | Cumulated PnL: {self.cum_realized_pnl}'

        ax.text(
            x=0.5, y=1.05,   # 1보다 크게 주면 축 위쪽으로 나감
            s=info,
            ha='center', va='bottom',
            fontsize=12,
            transform=ax.transAxes   # Axes 좌표계 (0~1)
)

        # 요약 함수
        def summarize(lst, digits=2):
            if len(lst) == 0:
                return ("N/A", "N/A")
            clean = [v for v in lst if v is not None]
            if len(clean) == 0:
                return ("N/A", "N/A")
            mean_val = np.mean(clean)
            med_val = np.median(clean)
            return (f"{mean_val:.{digits}f}", f"{med_val:.{digits}f}")

        # Metric별 데이터
        metrics = [
            ("Winrate", *summarize(self.winrate_list, 2)),
            ("Reward", *summarize(self.reward_list, 2)),
            ("Maintained length", *summarize(self.maintained_list, 2)),
            ("PnL Ratio", *summarize(self.pnl_ratio_list, 2)),
            ("Step Reward", *summarize(self.step_reward_list, 2)),
            ("Maintained Vol", *summarize(self.maintained_vol_list, 2)),
            ("Action", *summarize(self.action_list, 2)),
            ("Entry Action", *summarize(self.entry_action_list, 2)),
            ("Liquidation Action", *summarize(self.liquidation_action_list, 2)),
        ]

        # 행렬 전환: metrics = [(name, mean, median), ...]
        col_labels = [m[0] for m in metrics]  # metric names
        cell_text = [
            [m[1] for m in metrics],  # mean row
            [m[2] for m in metrics],  # median row
        ]
        row_labels = ["Mean", "Median"]

        # 제목
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)

        # 표
        table = ax.table(
            cellText=cell_text,
            rowLabels=row_labels,
            colLabels=col_labels,
            cellLoc="center",
            loc="center"
        )

        # 스타일링
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.4)  # 표 크기 조정
        table.auto_set_column_width(col=list(range(len(col_labels))))

        for (row, col), cell in table.get_celld().items():
            cell.set_linewidth(0.5)
            cell.set_edgecolor("#D0D0D0")
            if row == 0:  # header row
                cell.set_text_props(weight="bold", color="white")
                cell.set_facecolor("#2C3E50")
            else:
                cell.set_facecolor("#F9F9F9" if row % 2 == 0 else "#FFFFFF")

        
    def _split_long_short(self, actions):
        """행동을 롱과 숏으로 구분한다 (None은 제외)."""
        long, short = [], []

        for a in actions:
            if a is None:
                continue
            try:
                if a > 0:
                    long.append(a)
                elif a < 0:
                    short.append(a)
            except TypeError:
                continue
        return long, short

    def _get_directory(self, name):
        return self.trainer.figures_path + '/' + name

    def _reset_epi_traker(self):
        self.loss_list = []
        self.reward_list = []
        self.winrate_list = []
        self.maintained_list = []
        self.pnl_ratio_list = []
        self.event_list = []

    def _reset_step_traker(self):
        self.timestep_list = []
        self.close_price_list = []
        self.unrealized_pnl_list = []
        self.cum_realized_pnl_list = []
        self.net_realized_pnl_list = []
        self.maintained_vol_list = []
        self.action_list = []
        self.entry_action_list = []
        self.liquidation_action_list = []
        self.step_reward_list = []

    # ----- CSV Logging Helpers -----
    def _ensure_csv_dir(self):
        if self.csv_dir is None:
            self.csv_dir = Path(self.trainer.base_path) / "csv"
            self.csv_dir.mkdir(parents=True, exist_ok=True)

    def _save_episode_csv(self, phase: str, dataset_flag: int):
        if not self.reward_list:
            return
        self._ensure_csv_dir()
        rows = []
        for idx, (loss, reward, win, maintained, pnl_ratio, events) in enumerate(
            zip(
                self.loss_list,
                self.reward_list,
                self.winrate_list,
                self.maintained_list,
                self.pnl_ratio_list,
                self.event_list,
            )
        ):
            rows.append(
                {
                    "phase": phase,
                    "dataset_flag": dataset_flag,
                    "episode_idx": idx,
                    "loss": loss,
                    "reward": reward,
                    "winrate": win,
                    "maintained": maintained,
                    "pnl_ratio": pnl_ratio,
                    "event": json.dumps(events, ensure_ascii=False),
                }
            )
        path = self.csv_dir / f"{phase}_interval{dataset_flag}_episodes.csv"
        self._write_csv(path, rows)

    def _save_step_csv(self, phase: str, dataset_flag: int):
        if not self.timestep_list:
            return
        self._ensure_csv_dir()
        rows = []
        for idx, (ts, price, unreal, cum_real, net_real, maintained, action, entry, liq, reward) in enumerate(
            zip(
                self.timestep_list,
                self.close_price_list,
                self.unrealized_pnl_list,
                self.cum_realized_pnl_list,
                self.net_realized_pnl_list,
                self.maintained_vol_list,
                self.action_list,
                self.entry_action_list,
                self.liquidation_action_list,
                self.step_reward_list,
            )
        ):
            rows.append(
                {
                    "phase": phase,
                    "dataset_flag": dataset_flag,
                    "step_idx": idx,
                    "timestep": str(ts),
                    "close_price": price,
                    "unrealized_pnl": unreal,
                    "cum_realized_pnl": cum_real,
                    "net_realized_pnl": net_real,
                    "maintained_vol": maintained,
                    "action": action,
                    "entry_action": entry,
                    "liquidation_action": liq,
                    "step_reward": reward,
                }
            )
        path = self.csv_dir / f"{phase}_interval{dataset_flag}_steps.csv"
        self._write_csv(path, rows)

    def _write_csv(self, path: Path, rows: list[dict]):
        if not rows:
            return
        header = list(rows[0].keys())
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
        self.logging(f"📝 CSV 저장 완료: {path}")
