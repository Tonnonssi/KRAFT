from .base import Callback

class EarlyStoppingCallback(Callback):
    """Early Stopping을 담당하는 콜백"""
    def __init__(self):
        self.max_maintained = 0

    def on_episode_end(self, logs=None):
        logs = logs or {}
        mlen  = logs.get("maintained")
        if mlen is not None:
            self.max_maintained = max(self.max_maintained, mlen)

    def on_valid_begin(self, logs=None): 
        if self.max_maintained < 3000:
            self.trainer.early_stop = True
            self.logging(f"⏹️ Early Stopping Triggered | train max_maintained={self.max_maintained}")

    def on_valid_end(self, logs=None):
        # 초기화 
        self.max_maintained = 0  
        self.trainer.early_stop = False
