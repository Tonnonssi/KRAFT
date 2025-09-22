class Event:
    # 장기 : 상태를 Enum으로 관리하는게 더 편할 것 같다. 
    def __init__(self):
        self.status_priority = [
            'goal_profit', 'bankrupt', 'margin_call',
            'maturity_data', 'end_of_day', 'end_of_data', 
            'insufficient', 'max_step'
        ]
        self.event_dict = {status: False for status in self.status_priority}

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
        return self.event_dict.get(status, False)

    def __eq__(self, other):
        if isinstance(other, str):
            # 문자열 하나 (정확히 일치해야 함)
            return other in self.event_dict and self.event_dict[other]

        elif isinstance(other, (list, set, tuple)):
            # 컬렉션이면, 활성 상태 중 하나라도 포함되면 True
            active_set = {status for status, v in self.event_dict.items() if v}
            return not active_set.isdisjoint(other)

        return NotImplemented

    def __add__(self, status):
        # 새로운 상태를 +로 추가 
        if status not in self.event_dict:
            if status == '':
                pass 
        else:
            self.event_dict[status] = True
        return self

    def _get_priority(self):
        # 더 중요한 event를 대표로 출력 
        for status in self.status_priority:
            if self.event_dict[status]:
                return status
        return None

    def _cal_activated_status(self):
        return sum(1 for v in self.event_dict.values() if v)


if __name__ == '__main__':
    status_list = ['max_step', 'end_of_day']
    event = Event()
    event + 'max_step'
    event + 'goal_profit'
    event + ''

    print(event())                        # goal_profit(m)
    print('bankrupt' in event)            # False
    print(event._cal_activated_status())  # 2
    print(event in status_list)