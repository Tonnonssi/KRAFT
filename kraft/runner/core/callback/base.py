
class Callback:
    """Callback의 기본 구조"""
    def set_trainer(self, trainer):
        self.trainer = trainer
    def on_fit_begin(self, logs=None): 
        # run_name, seed 
        pass
    def on_fit_end(self, logs=None): 
        # run_name, seed
        pass
    def on_train_begin(self, logs=None): 
        # dataset_flag, date_range 
        pass
    def on_train_end(self, logs=None): 
        # dataset_flag, date_range
        pass
    def on_valid_begin(self, logs=None): 
        # dataset_flag, date_range, model_type
        pass
    def on_valid_end(self, logs=None): 
        # dataset_flag, date_range, model_type
        pass
    def on_interval_begin(self, logs=None): 
        # dataset_flag, train_timestep, valid_timestep
        pass
    def on_interval_end(self, logs=None): 
        # dataset_flag, train_timestep, valid_timestep
        pass
    def on_episode_end(self, logs=None): 
        # loss, pnl, epi_reward, event, event_dict, 
        # winrate, maintained, pnl_ratio, index, model
        pass
    def on_step_end(self, logs=None): 
        # close_price, n_step, timestep, unrealized_pnl, cum_realized_pnl, net_realized_pnl, 
        # event, maintained_vol, action, entry_action, liquidation_action, step_reward
        pass

    def logging(self, message):
        """로그를 저장하고, 출력함"""
        log_file = self.trainer.log_file
        print(message)
        with open(log_file, "a", encoding='utf-8') as f:
            f.write(message + "\n")