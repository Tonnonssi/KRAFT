def is_day_changed(current_timestep, next_timestep):
    """
    날짜를 기준으로 구분한다. 
    날이 달라지면 true 
    """
    return current_timestep.date() != next_timestep.date()

def reached_end_of_dataset(dataset, current_timestep):
    """
    현재 데이터가 데이터 셋의 마지막 데이터인가?
    done=True
    """
    done = dataset.reach_end(current_timestep)
    info = 'end_of_data' if done else ''
    return done, info 

def is_maturity_data(maturity_list, next_timestep, current_timestep):
    """
    만기일의 마지막 데이터인가?
    done=True
    """
    is_maturity_date = current_timestep.date() in maturity_list
    day_changed = is_day_changed(current_timestep, next_timestep)
    
    done = is_maturity_date and day_changed
    info = 'maturity_data' if done else ''

    return done, info 

def is_bankrupt(available_balance):
    """
    가용 잔고가 사라지면 파산으로 간주한다. 
    현재 내 자산 상황은 파산인가? 
    done=True
    """
    done = available_balance <= 0
    info = 'bankrupt' if done else ''
    return done, info

def is_margin_call(available_balance, maintenance_margin):
    """
    마진콜인가? 
    done=True
    """
    done = (available_balance <= maintenance_margin)
    info = 'margin_call' if done else ''
    return done, info

def check_insufficient(account):
    """ 
    새로운 계약을 체결할 수 없는 경우의 조건
    일 뿐 done=True가 아니다.
    """
    # 장기 
    condition = account.is_insufficient_for_new_contract
    info = 'insufficient' if condition else ''
    return False, info

def is_max_step(max_steps, maintained_steps):
    """
    스윙의 최대 길이에 도달했는가? 
    done=True
    """
    done = (max_steps == maintained_steps)
    info = 'max_step' if done else ''
    return done, info

def is_over_pnl_ratio_threshold(realized_pnl_ratio, threshold):
    """
    목적으로 삼은 수익률을 넘겼는가? 
    done=False
    """
    condition = realized_pnl_ratio > threshold
    info = 'goal_profit' if condition else ''
    return False, info 

def checks_non_episodic(account, dataset, maturity_timesteps, current_timestep, next_timestep, threshold, **kwargs):
    """ 
    상태를 판단하는 여러 함수들
    Standard인 Episodic과 달리 max_steps가 제외됨
    """
    return [    
    lambda: is_bankrupt(account.available_balance),
    lambda: is_margin_call(account.available_balance, account.maintenance_margin),
    lambda: check_insufficient(account),   
    lambda: reached_end_of_dataset(dataset, current_timestep),
    lambda: is_maturity_data(maturity_timesteps, next_timestep, current_timestep),
    lambda: is_over_pnl_ratio_threshold(account.net_realized_pnl / account.initial_budget, threshold)
    ]

def checks_episodic(account, dataset, max_steps, maintained, maturity_timesteps, current_timestep, next_timestep, threshold):
    """ 상태를 판단하는 여러 함수들"""
    return [    
    lambda: is_bankrupt(account.available_balance),
    lambda: is_margin_call(account.available_balance, account.maintenance_margin),
    lambda: is_max_step(max_steps, maintained),
    lambda: check_insufficient(account),   
    lambda: reached_end_of_dataset(dataset, current_timestep),
    lambda: is_maturity_data(maturity_timesteps, next_timestep, current_timestep),
    lambda: is_over_pnl_ratio_threshold(account.net_realized_pnl / account.initial_budget, threshold)
    ]