import torch
import numpy as np

def run_episode(env, agent, n_steps, device, callbacks=None, multi_critics=False, stochastic=True):
    """
    하나의 에피소드를 실행한다. 
    [n_steps]이란? 
    - PPO train일 때는 on-policy 학습을 위한 값 
    - PPO가 아니거나, valid, test 때는 상한 스텝 수로 해석
    ----------------------------------------------------------------------
    single critic과 multi critics에 따라 다른 실행 함수를 호출한다. 
    
    """
    if multi_critics: 
        return _run_episode_multi_critics(env, agent, n_steps, device, callbacks, stochastic=stochastic)
    else:
        return _run_episode_single_critic(env, agent, n_steps, device, callbacks, stochastic=stochastic)

def _run_episode_single_critic(env, agent, n_steps, device, callbacks=None, stochastic=True):
    """single critic PPO 에이전트인 경우"""
    done = False 
    state = env.reset()
    state = get_trainable_state(state, device)
    mask = env.mask

    epi_memory = []
    epi_reward = 0

    for idx in range(n_steps):
        if done:
            break 

        decoded_action, log_prob = agent.get_action(state, mask, stochastic=stochastic)
        next_state, reward, done, mask = env.step(decoded_action)

        next_state = get_trainable_state(next_state, device)
        step_memory = build_memory(state, decoded_action,                   # on-policy를 위한 메모리 
                                   reward, next_state, done, log_prob,      
                                   mask, env.is_entry, env.log_return)
        step_extra_memory = get_extra_memory(env, decoded_action)           # 시각화 등을 위한 추가 메모리
        epi_memory.append((step_memory, step_extra_memory))

        # UPDATE 
        state = next_state
        epi_reward += reward 

        if callbacks:
            for cb in callbacks:
                cb.on_step_end(get_step_log(env, decoded_action, reward, idx))

    return epi_memory, epi_reward, env.episode_event
    
def _run_episode_multi_critics(env, agent, n_steps, device, callbacks=None, stochastic=True):
    """
    multi critics PPO 에이전트인 경우
    """
    def weighted_reward(reward, alpha):
        """multi critics PPO 에이전트에서 사용하는 가중 보상 계산"""
        if isinstance(reward, torch.Tensor):
            reward_tensor = reward.clone().detach().to(torch.float32)
        else:
            reward_tensor = torch.as_tensor(reward, dtype=torch.float32)

        if isinstance(alpha, torch.Tensor):
            alpha_tensor = alpha.clone().detach().to(torch.float32)
        else:
            alpha_tensor = torch.as_tensor(alpha, dtype=torch.float32)

        return torch.sum(reward_tensor * alpha_tensor).item()
    
    done = False 
    state = env.reset()
    state = get_trainable_state(state, device)
    mask = env.mask

    # Multi-Critics PPO 에이전트인 경우, 환경에서 alpha 값을 받아 에이전트에 설정
    # ===============================================================
    env.reset_alpha()
    agent.get_alpha(env.alpha)
    # ===============================================================

    epi_memory = []
    epi_reward = 0

    for idx in range(n_steps):
        if done:
            break 

        decoded_action, log_prob = agent.get_action(state, mask, stochastic=stochastic)
        next_state, reward, done, mask = env.step(decoded_action)

        next_state = get_trainable_state(next_state, device)
        # ===============================================================
        # single critic과 다른 부분: alpha 값을 메모리에 추가
        step_memory = build_memory(state, decoded_action,                   # on-policy를 위한 메모리 
                                   reward, next_state, done, log_prob,      
                                   mask, env.is_entry, env.log_return, 
                                   env.alpha)
        # ===============================================================
        step_extra_memory = get_extra_memory(env, decoded_action)           # 시각화 등을 위한 추가 메모리
        epi_memory.append((step_memory, step_extra_memory))

        # UPDATE 
        state = next_state
        epi_reward += weighted_reward(reward, env.alpha)

        if callbacks:
            for cb in callbacks:
                cb.on_step_end(get_step_log(env, decoded_action, reward, idx))

    return epi_memory, epi_reward, env.episode_event

def run_loop(env, agent, batch_size, n_steps, is_training: bool, device, callbacks=None, multi_critics=False):
    """
    Train, Valid, Test의 공통 로직을 처리하는 핵심 루프.
    is_training 플래그에 따라 학습 관련 동작을 제어한다.
    """
    # --- 초기화 ---
    _memory_buffer = []
    replay_memory = []
    _episode_cum_reward = 0
    loss = None
    idx = 0

    # --- 메인 루프 ---
    while not env.terminated:
        # 1. 에피소드 실행 (공통)
        # is_training 플래그를 전달하여 에이전트가 탐험(train) 또는 결정적 행동(valid/test)을 하도록 제어할 수 있음
        epi_memory, epi_reward, episode_event = run_episode(env, agent, n_steps, device, callbacks, multi_critics, is_training)
        _episode_cum_reward += epi_reward

        # 2. 학습 로직 (is_training=True일 때만 실행)
        if is_training:
            train_memory, extra_memory = zip(*epi_memory)
            
            if len(_memory_buffer) != 0:
                try:
                    _train_m, _extra_m = zip(*_memory_buffer)
                except ValueError:
                    # 버퍼 형식이 예상과 다르면 전부 비우고 현재 메모리만 사용
                    _memory_buffer = []
                    _train_m, _extra_m = (), ()

                def _flatten(seqs):
                    for seq in seqs:
                        if isinstance(seq, (list, tuple)):
                            for item in seq:
                                yield item
                        else:
                            yield seq

                train_memory = list(_flatten(_train_m)) + list(train_memory)
                extra_memory = list(_flatten(_extra_m)) + list(extra_memory)
            else:
                train_memory = list(train_memory)
                extra_memory = list(extra_memory)
            
            if len(train_memory) >= batch_size:
                advantage = agent.cal_advantage(train_memory)
                loss = agent.train(train_memory, advantage)
                _memory_buffer = []
            else:
                _memory_buffer.append((train_memory, extra_memory))  # 다음 배치 때 이어서 사용

        # 3. 결과 기록 (공통)
        replay_memory.append({
            'memory': epi_memory,
            'cumulative_reward': _episode_cum_reward,
            'loss': loss, # 학습하지 않을 때는 None
            'done': env.done,
            'is_margin': 'margin_call' in env.episode_event,
            'maintained_steps': env.maintained
        })

        # 4. 
        if callbacks:
            for cb in callbacks:
                cb.on_episode_end(get_episode_log(env, agent, loss, episode_event, epi_reward, idx))
        
        # 5. 에피소드 종료 시 처리 (공통)
        if env.done:
            _episode_cum_reward = 0

        idx += 1

    return replay_memory


def get_extra_memory(env, decoded_action):
    """시각화에 사용할 학습 이외의 여러 지표를 저장"""
    current_position, execution_strength = split_position_strength(decoded_action)
    return {
        'current_point' : env.current_point,
        'current_timestep' : env.current_timestep,
        'unrealized_pnl' : env.account.unrealized_pnl,
        'net_realized_pnl' : env.account.net_realized_pnl,
        'main_event' : env.step_event(),
        'decoded_action' : decoded_action,
        'current_position' : current_position,
        'execution_strength' : execution_strength, 
        'n_total_trades' : env.n_total_trades,
        'n_win_trades' : env.n_win_trades
    }

def get_step_log(env, decoded_action, step_reward, idx):
    """step 단위 Callback에 필요한 파라미터들"""
    is_liquidation = (env.account.prev_position != env.account.current_position) & ~env.is_entry
    entry_action = decoded_action if env.is_entry else None
    liquidation_action = decoded_action if is_liquidation else None

    return {
        'close_price' : env.current_point,
        'n_step' : idx,
        'timestep' : env.current_timestep,
        'unrealized_pnl' : env.account.unrealized_pnl,
        'cum_realized_pnl' : env.account.realized_pnl,
        'net_realized_pnl' : env.account.net_realized_pnl,
        'event' : env.step_event(),
        'maintained_vol' : env.maintained,
        'action' : decoded_action,
        'entry_action' : entry_action,
        'liquidation_action' : liquidation_action,
        'step_reward' : step_reward
    }

def get_episode_log(env, agent, loss, episode_event, epi_reward, idx):
    """에피소드 단위 Callback에 필요한 파라미터들"""
    model_state = {k: v.detach().cpu().clone() for k, v in agent.model.state_dict().items()}
    return {
       'loss' : loss,
       'pnl' : env.account.realized_pnl,
       'epi_reward' : epi_reward,
       'event' : episode_event(),
       'event_dict' : episode_event.event_dict,
       'winrate' : (env.n_win_trades / env.n_total_trades) if env.n_total_trades else 0.0,
       'maintained' : env.maintained,
       'pnl_ratio' : env.account.realized_pnl / env.account.initial_budget,
       'index' : idx,
        'model': model_state
    }

def get_trainable_state(state, device):
    ts_state = torch.tensor(state[0], dtype=torch.float32).unsqueeze(0).to(device)
    agent_state = torch.tensor(state[1], dtype=torch.float32).unsqueeze(0).to(device)
    return ts_state, agent_state

def build_memory(state, decoded_action, reward, next_state, done, log_prob, mask, is_entry, log_return, alpha=None):
    '''memory를 구성하여 반환'''
    mem =  [
            state,
            torch.tensor([decoded_action]),
            torch.tensor([reward], dtype=torch.float32),
            next_state,
            torch.tensor([done], dtype=torch.float32),
            torch.tensor([log_prob], dtype=torch.float32),
            torch.tensor([mask], dtype=torch.bool),
            torch.tensor([is_entry], dtype=torch.bool),
            torch.tensor([log_return], dtype=torch.float32),
            ]
    # alpha 값이 존재하는 경우 메모리에 추가
    if alpha is not None:
        mem.append(alpha.clone().unsqueeze(0))

    return mem

def split_position_strength(decoded_action):
    if decoded_action == 0:
        return 0, 0
    
    strength = np.abs(decoded_action)
    position = np.sign(decoded_action)
    return position.item(), strength.item()
