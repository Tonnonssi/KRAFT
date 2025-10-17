from .base import Callback

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
