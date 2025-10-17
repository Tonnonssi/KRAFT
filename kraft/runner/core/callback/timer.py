from .base import Callback
import time

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