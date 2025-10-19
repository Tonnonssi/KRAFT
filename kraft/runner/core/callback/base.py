
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