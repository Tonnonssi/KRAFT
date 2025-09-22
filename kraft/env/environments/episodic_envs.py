from ..flows import StreamFlow, SamplingFlow
from ..core.utils.done_conditions import checks_episodic

class EpisodicStreamEnv(StreamFlow):
    def checks(self):
        return checks_episodic(self.account, self.dataset, 
                               self.max_steps, self.maintained, 
                               self.maturity_timesteps, 
                               self.current_timestep, 
                               self.next_timestep,
                               self.pnl_threshold)


class EpisodicSamplingEnv(SamplingFlow):
    def checks(self):
        return checks_episodic(self.account, self.dataset, 
                               self.max_steps, self.maintained, 
                               self.maturity_timesteps, 
                               self.current_timestep, 
                               self.next_timestep,
                               self.pnl_threshold)
