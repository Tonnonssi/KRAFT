from ..core.base_env import BaseEnvironment

class SurvivalFlow(BaseEnvironment):
    reset_done = ['bankrupt', 'margin_call']

    def reset(self):
        """
        base들이 초기화되고, 초기화된 Agent 정보들을 바탕으로 새로운 State를 만든다.
        파산이나 마진콜 상황이면 처음으로 되돌아가서 학습한다. 
        """
        if self.done:
            self._reset_base()
            if self.event in self.reset_done:
                state = self._reset_to_init_timestep()
            else:
                state = self._reset_state()
            self.current_state = state 

        return self.current_state

    @property
    def terminated(self):
        """환경의 전체 종료 조건: 데이터 셋이 동났을 때"""
        return self.dataset.reach_end(self.current_timestep)