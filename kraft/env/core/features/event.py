class Event:
    STATUS_PRIORITY = [
            'goal_profit',
            # 'bankrupt',
            'margin_call',
            'maturity_data',
            'end_of_data',
            'insufficient',
            'max_step',
            'loss_more_than_5percent',
            'loss_more_than_25percent',
            'loss_more_than_50percent'
        ]

    # 장기 : 상태를 Enum으로 관리하는게 더 편할 것 같다.
    def __init__(self):
        self.status_priority = Event.STATUS_PRIORITY

        self.event_dict = {status: 0 for status in self.status_priority}
        self.step_event = StepEvent()

    def __call__(self):
        # 메인
        activated = self._get_priority()
        if activated is None:
            return ''
        if 'max_step' in self and activated != 'max_step':
            return f"{activated}(max)"
        return activated

    def __contains__(self, status) -> bool:
        # status in event
        return self.event_dict.get(status, 0) > 0

    def __eq__(self, other):
        if isinstance(other, str):
            # 문자열 하나 (정확히 일치해야 함)
            return other in self.event_dict and self.event_dict[other] > 0

        elif isinstance(other, (list, set, tuple)):
            # 컬렉션이면, 활성 상태 중 하나라도 포함되면 True
            active_set = {status for status, v in self.event_dict.items() if v > 0}
            return not active_set.isdisjoint(other)

        return NotImplemented

    def __add__(self, status):
        # 새로운 상태를 +로 추가
        if status not in self.event_dict:
            if status == '':
                pass
            else:
                 # If the status is valid but not in event_dict, add it with count 1
                if status in self.status_priority:
                     self.event_dict[status] = 1
        else:
            self.event_dict[status] += 1
        return self

    def _get_priority(self):
        # 더 중요한 event를 대표로 출력
        for status in self.status_priority:
            if self.event_dict.get(status, 0) > 0:  # Use .get() to safely access the status
                return status
        return None

    def _cal_activated_status(self):
        return sum(1 for v in self.event_dict.values() if v > 0)

    def collect_information(self, status_list:list):
        """step 단위로 상태를 수집하고 누적한다."""
        self.step_event.collect_information(status_list)
        for status, count in self.step_event.event_dict.items():
            if count > 0:
                self.event_dict[status] += count

class StepEvent(Event):
    """
    Step 단위 이벤트 관리 클래스
    상태를 누적하지 않고 매번 초기화
    """
    def __init__(self):
        self.status_priority = Event.STATUS_PRIORITY
        self.event_dict = {status: 0 for status in self.status_priority}

    def collect_information(self, status_list:list):
        self.event_dict = {status: 0 for status in self.status_priority} # 초기화
        for status in status_list:
            # Ensure status is in status_priority before adding
            if status in self.status_priority:
                self += status

if __name__ == "__main__":
    event = Event()
    step_event = event.step_event

    status_list = ['max_step', 'end_of_day']
    event.collect_information(status_list)

    print(event())
    print(step_event())

    status_list1 = ['end_of_day', 'goal_profit']
    event.collect_information(status_list1)
    print(event())
    print(step_event())
