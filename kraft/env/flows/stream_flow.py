from __future__ import annotations

from ..core.base_env import BaseEnvironment


class StreamFlow(BaseEnvironment):
    """
    데이터를 재사용하지 않고 스트리밍하듯 전진.
    파산/만기 등으로 done이 떠도 에피소드를 초기화하지 않고 이어서 학습.
    """
    def reset(self):
        """
        base들이 초기화되고, 초기화된 Agent 정보들을 바탕으로 새로운 State를 만든다.
        이때 데이터는 done=True였던 시점부터 이어져서 진행된다. 
        """
        if self.done:
            self._reset_base()
        self.current_state = self._reset_state()

        return self.current_state
    
    @property
    def terminated(self):
        """환경의 전체 종료 조건: 데이터 셋이 동났을 때"""
        return self.dataset.reach_end(self.current_timestep)