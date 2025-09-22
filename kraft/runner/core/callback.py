import os
import time 
import torch
import numpy as np
from collections import deque  
import matplotlib.pyplot as plt
import pathlib
from ..utils import ensure_dir
from .visualization import *

class Callback:
    """Callback의 기본 구조"""
    def set_trainer(self, trainer):
        self.trainer = trainer
    def on_fit_begin(self, logs=None): pass
    def on_fit_end(self, logs=None): pass
    def on_train_begin(self, logs=None): pass
    def on_train_end(self, logs=None): pass
    def on_valid_begin(self, logs=None): pass
    def on_valid_end(self, logs=None): pass
    def on_interval_begin(self, logs=None): pass
    def on_interval_end(self, logs=None): pass
    def on_episode_end(self, logs=None): pass
    def on_step_end(self, logs=None): pass

    def logging(self, message):
        """로그를 저장하고, 출력함"""
        log_file = self.trainer.log_file
        print(message)
        with open(log_file, "a", encoding='utf-8') as f:
            f.write(message + "\n")

class LoggingCallback(Callback):
    """학습/검증 진행 상황과 핵심 지표를 간결하게 로그로 출력"""

    def __init__(self, print_every_episode: int = 1, print_every_step: int = 500):
        """
        Args:
            print_every_episode: 에피소드 로그 출력 주기 (1이면 매 에피소드 출력)
            print_every_step: 스텝 로그 출력 주기 (0이면 스텝 로그 생략)
        """
        # 장기: config로 빼기 
        self.print_every_episode = print_every_episode
        self.print_every_step = print_every_step

    # --- 공통 유틸 ---
    @staticmethod
    def _fmt(v, digits=2, width=None):
        if v is None:
            return "N/A"
        try:
            formatted = f"{float(v):.{digits}f}"
        except Exception:
            formatted = str(v)
        if width:
            return f"{formatted:>{width}}"
        return formatted

    @staticmethod
    def _fmt_currency(v, digits=0, width=None):
        if v is None:
            formatted = "N/A"
        else:
            try:
                formatted = f"{float(v):,.{max(digits,0)}f}"
            except Exception:
                formatted = str(v)
        formatted = f"{formatted} KRW" if formatted != "N/A" else formatted
        if width:
            return f"{formatted:>{width}}"
        return formatted

    @staticmethod
    def _fmt_percent(v, width=None):
        if v is None:
            formatted = "N/A"
        else:
            try:
                formatted = f"{float(v)*100:,.2f}%"
            except Exception:
                formatted = str(v)
        if width:
            return f"{formatted:>{width}}"
        return formatted

    def _hline(self):
        self.logging("-" * 80)

    # --- Hook 구현 ---
    def on_fit_begin(self, logs=None):
        logs = logs or {}
        run_name = logs.get("run_name", "")
        self.logging(f"🚀 Fit begin {f'[{run_name}]' if run_name else ''}")
        self._hline()

    def on_fit_end(self, logs=None):
        logs = logs or {}
        self.logging("✅ Fit end")
        # (CheckpointCallback이 총 저장 개수 출력함)
        self._hline()

    def on_train_begin(self, logs=None):
        logs = logs or {}
        dataset_flag = logs.get("dataset_flag", "N/A")
        self.logging(f"▶️ Train begin | interval={dataset_flag}")
        self._hline()

    def on_train_end(self, logs=None):
        logs = logs or {}
        dataset_flag = logs.get("dataset_flag", "N/A")
        self.logging(f"🏁 Train end   | interval={dataset_flag}")
        # 저장/선정은 Checkpoint/Visualization 콜백에서 처리
        self._hline()

    def on_valid_begin(self, logs=None):
        logs = logs or {}
        dataset_flag = logs.get("dataset_flag", "N/A")
        model_type = logs.get("model_type", "N/A")
        self.logging(f"🔎 Valid begin | interval={dataset_flag} | model={model_type}")
        self._hline()

    def on_valid_end(self, logs=None):
        logs = logs or {}
        dataset_flag = logs.get("dataset_flag", "N/A")
        model_type = logs.get("model_type", "N/A")
        # VisualizationCallback 이 cum_realized_pnl 누적을 관리하므로 여기선 요약만
        cum_pnl = logs.get("cum_realized_pnl")  # 있으면 출력
        if cum_pnl is not None:
            self.logging(
                f"📈 Valid end   | interval={dataset_flag} | model={model_type} | "
                f"cum_pnl={self._fmt_currency(cum_pnl, 0, 18)}"
            )
        else:
            self.logging(f"📈 Valid end   | interval={dataset_flag} | model={model_type}")
        self._hline()

    def on_interval_begin(self, logs=None):
        logs = logs or {}
        dataset_flag = logs.get("dataset_flag", "N/A")
        self.logging(f"⏩ Interval begin | interval={dataset_flag} | Train Period: {logs['train_timestep']} | Valid Period: {logs['valid_timestep']}")

    def on_interval_end(self, logs=None):
        logs = logs or {}
        dataset_flag = logs.get("dataset_flag", "N/A")
        self.logging(f"⏹️ Interval end   | interval={dataset_flag}")
        self._hline()

    def on_episode_end(self, logs=None):
        """
        VisualizationCallback / CheckpointCallback 에서 사용하는 키만 활용:
          - loss, epi_reward, event_list, winrate, maintained, pnl_ratio
          - pnl, index, model (Checkpoint), (선택) dataset_flag
        """
        logs = logs or {}
        idx   = logs.get("index")
        loss  = logs.get("loss")
        rew   = logs.get("epi_reward")
        win   = logs.get("winrate")
        mlen  = logs.get("maintained")
        pr    = logs.get("pnl_ratio")
        pnl   = logs.get("pnl")
        evn   = logs.get("event_list")

        if isinstance(evn, dict):
            event_repr = ", ".join(f"{k}:{v}" for k, v in evn.items()) or "None"
        elif isinstance(evn, (list, tuple)):
            event_repr = evn[-1] if evn else "None"
        else:
            event_repr = str(evn)

        if (idx is None) or (self.print_every_episode and (idx % self.print_every_episode == 0)):
            idx_label = f"{idx:>4}" if idx is not None else "   -"
            self.logging(
                f"📘 Episode {idx_label} | "
                f"loss={self._fmt(loss, 3, 8)} | "
                f"reward={self._fmt(rew, 2, 8)} | "
                f"winrate={self._fmt(win, 2, 6)} | "
                f"maintained={self._fmt(mlen, 0, 4)} | "
                f"pnl_ratio={self._fmt_percent(pr, 9)} | "
                f"pnl={self._fmt_currency(pnl, 0, 18)} | "
                f"events={event_repr}"
            )

    def on_step_end(self, logs=None):
        """
        VisualizationCallback 에서 사용하는 키만 활용:
          - timestep, close_price, unrealized_pnl, cum_realized_pnl, net_realized_pnl,
            event, maintained_vol, action, entry_action, liquidation_action, step_reward
        """
        if self.print_every_step <= 0:
            return
        logs = logs or {}
        t = logs.get("n_step")
        if (t is None) or (t % self.print_every_step == 0):
            cp   = logs.get("close_price")
            upnl = logs.get("unrealized_pnl")
            crp  = logs.get("cum_realized_pnl")
            nrp  = logs.get("net_realized_pnl")
            ev   = logs.get("event")
            mv   = logs.get("maintained_vol")
            act  = logs.get("action")
            ent  = logs.get("entry_action")
            liq  = logs.get("liquidation_action")
            r    = logs.get("step_reward")
            step_label = "----" if t is None else f"{t:>4}"
            self.logging(
                f"  • step={step_label} | "
                f"price={self._fmt(cp, 2, 8)} | "
                f"uPnL={self._fmt_currency(upnl, 0, 16)} | "
                f"cumPnL={self._fmt_currency(crp, 0, 16)} | "
                f"netPnL={self._fmt_currency(nrp, 0, 16)} | "  
                f"act={str(act):>3} | "
                f"r={self._fmt(r,2,7)} | event={ev}"
            )

class TimerCallback(Callback):
    """소요 시간을 알려주는 콜백"""
    def on_fit_begin(self, logs=None): 
        self.start_time = time.time()

    def on_fit_end(self, logs=None): 
        duration = time.time() - self.start_time 
        message = self._time_is(duration, "전체 학습 종료")
        self.logging(message)

    def on_train_begin(self, logs=None): 
        self.train_start_time = time.time()

    def on_train_end(self, logs=None): 
        duration = time.time() - self.train_start_time 
        idx = logs['dataset_flag']
        message = self._time_is(duration, f"{idx} 구간 학습 종료")
        self.logging(message)

    def on_valid_begin(self, logs=None): 
        self.valid_start_time = time.time()
        
    def on_valid_end(self, logs=None): 
        duration = time.time() - self.valid_start_time 
        idx = logs['dataset_flag']
        message = self._time_is(duration, f"{idx} 구간 검증 종료")
        self.logging(message)

    def on_interval_begin(self, logs=None): 
        self.interval_start_time = time.time()

    def on_interval_end(self, logs=None): 
        duration = time.time() - self.interval_start_time
        idx = logs['dataset_flag']
        message = self._time_is(duration, f"{idx} 구간 학습-검증 종료")
        self.logging(message)
    
    def _time_is(self, duration, status):
        """초단위를 시간-분-초로 분리"""
        hours, rem = divmod(duration, 3600)
        minutes, seconds = divmod(rem, 60)

        return f"⏱️ [{status}] 소요시간: {int(hours)}시간 {int(minutes)}분 {int(seconds)}초\n"

class CheckpointCallback(Callback):
    """체크포인트를 관리 감독하는 콜백"""
    def on_fit_end(self, logs=None): 
        """파일에 저장된 신경망이 총 몇 개인지 알려줌"""
        # self.trainer.config
        directory_path = pathlib.Path(self.trainer.models_path) 
        file_count = sum(1 for item in directory_path.iterdir() if item.is_file())
        self.logging(f"'{directory_path}'에 저장되어 있는 신경망 개수: {file_count}개")

    def on_train_begin(self, logs=None): 
        """지표 초기화"""
        self.best_ave_reward = None
        self.best_winrate = None
        self.best_pnl = None

        self.trainer.best_reward_model = None 
        self.trainer.best_winrate_model = None 
        self.trainer.best_pnl_model = None
        self.trainer.per_steps_model = deque(maxlen=10) 

    def on_episode_end(self, logs=None): 
        """업데이트"""
        logs = logs or {}
        ave_epi_reward = logs.get('epi_reward') 
        winrate = logs.get('winrate')
        idx = logs.get('index')
        model = logs.get('model')
        pnl = logs.get('pnl')
    
        if model is None:
            return

        if ave_epi_reward is not None and (
            self.best_ave_reward is None or ave_epi_reward >= self.best_ave_reward
        ):
            self.trainer.best_reward_model = model
            self.best_ave_reward = ave_epi_reward
        
        if winrate is not None and (
            self.best_winrate is None or winrate >= self.best_winrate
        ):
            self.trainer.best_winrate_model = model 
            self.best_winrate = winrate

        if pnl is not None and (
            self.best_pnl is None or pnl >= self.best_pnl
        ):
            self.trainer.best_pnl_model = model 
            self.best_pnl = pnl

        if idx is not None and model is not None and idx % 10 == 0:
            self.trainer.per_steps_model.append(model)

    def save_model_to(self, path, dataset_flag):
        """모든 신경망 가중치를 저장한다."""
        # 최고 보상 모델 저장
        if self.trainer.best_reward_model is not None:
            torch.save(self.trainer.best_reward_model, os.path.join(path, f'I{dataset_flag}bestreward.pth'))
            self.logging("[Saved] best_reward_model")

        # 최고 손익 모델 저장
        if self.trainer.best_pnl_model is not None:
            torch.save(self.trainer.best_pnl_model, os.path.join(path, f'I{dataset_flag}best_pnl_model.pth'))
            self.logging("[Saved] best_pnl_model")

        # 최고 승률 모델 저장
        if self.trainer.best_winrate_model is not None:
            torch.save(self.trainer.best_winrate_model, os.path.join(path, f'I{dataset_flag}best_winrate_model.pth'))
            self.logging("[Saved] best_winrate_model")

        # n-step마다 모델 저장 
        recent_models = list(self.trainer.per_steps_model)
        for idx, model_state in enumerate(recent_models):
            torch.save(model_state, os.path.join(path, f'I{dataset_flag}_{(idx+1)}steps.pth'))
        self.logging(f"[Saved] {len(recent_models)} recent models")

class VisualizationCallback(Callback):
    """시각화를 담당하는 콜백"""
    def on_interval_begin(self, logs=None):
        self.cum_realized_pnl = 0

    def on_train_begin(self, logs=None): 
        self._reset_step_traker()
        self._reset_epi_traker()

    def on_train_end(self, logs=None): 
        """3 종류의 에피소드 시각화 진행"""
        dataset_flag = logs['dataset_flag']

        _, ax = plt.subplots(nrows=4, ncols=1, figsize=(18, 12))
        self.get_train_info(ax[0], logs)
        plot_training_curves(ax[1], self.reward_list, self.loss_list)
        plot_event_per_episodes(ax[2], self.event_list)
        plot_histogram_with_stats(ax[3], self.pnl_ratio_list, title='"PnL Ratio Distribution"')
        
        path = self._get_directory(f'TFI{dataset_flag-1}')
        plt.savefig(path)
        self.logging(f"✅ 시각화 저장 완료: {path}")

    def on_valid_begin(self, logs=None): 
        self._reset_step_traker()
        self._reset_epi_traker()

    def on_valid_end(self, logs=None):
        dataset_flag = logs['dataset_flag']
        model_type = logs['model_type']

        long_lst, short_lst = self._split_long_short(self.entry_action_list)

        fig, ax = plt.subplots(
            nrows=8,
            ncols=1,
            figsize=(22, 18),
            constrained_layout=True,
            gridspec_kw={'height_ratios': [1.1, 2.0, 1.6, 1.3, 1.6, 1.8, 1.2, 1.2]}
        )
        self.get_valid_info(ax[0], logs)
        plot_market_with_actions(ax[1], self.timestep_list, self.close_price_list, self.action_list, self.maintained_vol_list)
        plot_both_pnl_ticks(ax[2], self.timestep_list, self.unrealized_pnl_list, self.net_realized_pnl_list)
        plot_histogram_with_stats(ax[3], self.step_reward_list, title="Step Reward Distribution")
        plot_event_per_episodes(ax[4], self.event_list)
        plot_action_distribution_heatmap(ax[5], self.timestep_list, self.action_list)
        plot_long_short(ax[6], long_lst, short_lst, title='Entry Action Distribution')
        plot_realized_pnl(ax[7], self.timestep_list, self.cum_realized_pnl_list)

        path = self._get_directory(f'ValidI{dataset_flag}_{model_type}')
        fig.savefig(path)
        self.logging(f"✅ 시각화 저장 완료: {path}")


    def on_episode_end(self, logs=None): 
        logs = logs or {}
        self.loss_list.append(logs['loss'])
        self.reward_list.append(logs['epi_reward'])
        self.event_list.append(logs['event_list'])
        self.winrate_list.append(logs['winrate'])
        self.maintained_list.append(logs['maintained'])
        self.pnl_ratio_list.append(logs['pnl_ratio'])

    def on_step_end(self, logs=None): 
        self.timestep_list.append(logs['timestep'])
        self.close_price_list.append(logs['close_price'])
        self.unrealized_pnl_list.append(logs['unrealized_pnl'])
        self.cum_realized_pnl_list.append(logs['cum_realized_pnl'])
        self.net_realized_pnl_list.append(logs['net_realized_pnl'])
        self.step_event_list.append(logs['event'])
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
        self.event_list = []
        self.winrate_list = []
        self.maintained_list = []
        self.pnl_ratio_list = []

    def _reset_step_traker(self):
        self.timestep_list = []
        self.close_price_list = []
        self.unrealized_pnl_list = []
        self.cum_realized_pnl_list = []
        self.net_realized_pnl_list = []
        self.step_event_list = []
        self.maintained_vol_list = []
        self.action_list = []
        self.entry_action_list = []
        self.liquidation_action_list = []
        self.step_reward_list = []



# class EarlyStoppingCallback(Callback):
#     pass
