from ..flows import StreamFlow, SurvivalFlow
from ..core.utils.done_conditions import checks_non_episodic

class NonEpisodicStreamEnv(StreamFlow):
    def checks(self):
        return checks_non_episodic(self.account, self.dataset, 
                                    self.maturity_timesteps, 
                                    self.current_timestep, 
                                    self.next_timestep,
                                    self.two_ticks_later,
                                    self.pnl_threshold)

    
class NonEpisodicSurvivalEnv(SurvivalFlow):
    def checks(self):
        return checks_non_episodic(self.account, self.dataset, 
                                    self.maturity_timesteps, 
                                    self.current_timestep, 
                                    self.next_timestep,
                                    self.two_ticks_later,
                                    self.pnl_threshold)
