import numpy as np
from .. import _specification as spec

class Account:
    # MINI KOSPI 200 Futures 증거금, 수수료, 거래 승수 
    initial_margin_rate = spec.INITIAL_MARGIN_RATE  
    maintenance_margin_rate = spec.MAINTENANCE_MARGIN_RATE
    transaction_cost_rate = spec.TRANSACTION_COST_RATE
    contract_unit = spec.CONTRACT_UNIT

    # 부호에 따른 포지션 (fixed)
    position_dict = {-1: 'short', 0: 'hold', 1: 'long'}

    def __init__(self,  initial_budget,
                        position_cap, 
                        initial_timestep,
                        slippage_factor):
        
        ########### fixed value ###########
        self.initial_budget = initial_budget     # 예산 (KRW)
        self.initial_timestep = initial_timestep # 초기 timestep 저장
        self.position_cap = position_cap         # 최대 계약 수 상한
        
        # 수수료, 슬리피지
        self.slippage_factor = slippage_factor

        ########### variable value ###########
        self.reset()

    def reset(self):
        '''
        계좌 및 포지션 상태 초기화
        '''
        self.current_timestep = self.initial_timestep 

        # 계좌 (KRW)
        self.available_balance = self.initial_budget # 가용잔고
        self.margin_deposit = 0                 # 예치증거금

        # 포지션 (체결 계약)
        self.open_interest_list = []            # 미결제약정 리스트 (pt)
        self.current_name_value = 0             # 보유 계약의 명목 가치 (pt)
        self.maintenance_margin = 0             # 보유 계약에 대한 유지증거금 (KRW)

        self.current_position = 0               # 현재 포지션. + / - 부호만
        self.execution_strength = 0             # 체결 계약 수
        self.total_trades = 0                   # 전체 거래 횟수

        # 현재 포지션 관련 정보
        self.market_pt = 0                      # 현재 시장가
        self.average_entry = 0                  # 평균 진입가 = 보유 계약 명목 가치 / 계약 수 (pt)


        # 손익 (계좌로 계산 가능한데 따로 있어도 괜찮을 듯)
        self.realized_pnl = 0                   # 누적 실현 손익 (KRW)
        self.realized_pnl_ratio = 0             # 누적 실현 손익 / 초기 자산 
        self.net_realized_pnl = 0               # 현 스텝의 실제 실현 손익 (KRW)
        self.unrealized_pnl = 0                 # 미실현 손익 (KRW)
        self.total_transaction_costs = 0        # 총 수수료 (KRW)

        # 이전 정보
        self.prev_realized_pnl = 0              # 이전 누적 실현 손익 (KRW)
        self.prev_unrealized_pnl = 0            # 이전 미실현 손익 
        self.prev_balance = 0                   # 이전 자산 
        self.prev_position = 0                  # 이전 포지션, 표기 방식은 위와 동일 
        self.prev_timestep = None               # 이전 시간 

        self.settled = False                    # 계약 청산 여부
        self.concluded = False                  # 계약 체결 여부

    def step(self, decoded_action, market_pt, next_timestep):
        '''
        action, market point에 따라 계좌, 포지션 업데이트
        새로운 계약 추가 / 계약 청산
        평균 진입가, 유지 증거금, 미실현 손익 업데이트
        '''
        # 이전 정보: 포지션, 미실현 손익 저장 
        self.prev_position = self.current_position
        self.prev_unrealized_pnl = self.unrealized_pnl
        self.prev_realized_pnl = self.realized_pnl
        self.prev_balance = self.available_balance

        # 초기화 
        self.net_realized_pnl = 0
        self.settled = False                 
        self.concluded = False

        # 순 실현 손익과 비용 초기화 
        realized_net_pnl = 0
        cost = 0

        # 현재 action 정보
        self.market_pt = market_pt

        position, contract_vol = self.get_position_n_vol(decoded_action)
        position_diff = position * self.prev_position

        if decoded_action != 0:
            self.total_trades += 1
            cost = self._get_cost(decoded_action, market_pt)

            # 새로운 계약 체결: 현재 보유 계약이 없는 경우 / 현재 포지션과 같은 포지션을 취하는 경우
            # 사실 position_diff >= 0가 모든 케이스를 포함하지만, 혹시나 모르니 명확하게 체결 강도를 살려둠 
            if (self.execution_strength == 0) or (position_diff >= 0):
                self._conclude_contract(contract_vol, position, market_pt)

            # 계약 청산: 현재 포지션과 반대 포지션을 취하는 경우
            elif position_diff < 0:
                realized_net_pnl = self._settle_contract(contract_vol, position, market_pt)
        
        # timestep 업데이트
        self.current_timestep = next_timestep
        self.net_realized_pnl = self.realized_pnl - self.prev_realized_pnl

        # 정보 업데이트
        self._update_account(market_pt)

        # (추가) 실현 손익 비율 업데이트
        self.realized_pnl_ratio = self.realized_pnl / self.initial_budget

        return realized_net_pnl, cost

    def _conclude_contract(self, vol, position, market_pt):
        '''
        새로운 계약 체결 함수
        현재 보유 계약이 없는 경우 / 현재 포지션과 같은 포지션을 취하는 경우에 call
        '''
        
        self.current_position = position    # 포지션 업데이트
        self.concluded = True

        # 명목 가치와 초기 증거금 계산 
        name_value = vol * market_pt        
        initial_margin = market_pt * self.contract_unit * vol * self.initial_margin_rate 

        # 계약 추가
        self.open_interest_list.extend([market_pt for _ in range(vol)])
        self.current_name_value += name_value
        self.execution_strength += vol

        # 총 비용(수수료+슬리피지) 계산
        cost = self._get_cost(vol, market_pt)
        self.total_transaction_costs += cost

        # 계좌 변동
        self.available_balance -= initial_margin + cost
        self.margin_deposit += initial_margin
        self.realized_pnl -= cost

    def _settle_contract(self, vol, position, market_pt, get_pnl=True):
        '''
        계약 청산 함수
        - 항상 반대 계약 체결 시 Call
        - 현재 열려있는 계약에 대해 vol만큼 청산
        - 만약 열려있는 계약 수보다 반대 포지션을 더 많이 체결한 경우 나머지에 대해 새로운 계약 추가
        '''
        self.settled = True                    

        if vol >= self.execution_strength:  # 기존 체결 강도보다 많은 반대 계약 체결 시
            remain_vol = vol - self.execution_strength

            # 전체 계약 청산
            net_pnl, _ = self._settle_total_contract(market_pt)

            if remain_vol > 0:
                # 남은 포지션에 대해 새로운 계약 체결
                self._conclude_contract(remain_vol, position, market_pt)

        else:   # 일부 청산
            settled_contract = self.open_interest_list[:vol]
            settled_value = sum(np.abs(settled_contract))
            pnl = self._get_pnl(market_pt, vol) 
            settled_initial_margin = settled_value * self.contract_unit * self.initial_margin_rate

            # 계약 청산
            del self.open_interest_list[:vol]
            self.current_name_value -= settled_value
            self.execution_strength -= vol

            # 총 수수료 계산 및 업데이트 
            cost = self._get_cost(vol, market_pt)
            self.total_transaction_costs += cost

            # 실현 손익
            net_pnl = pnl - cost
            self.realized_pnl += int(net_pnl)

            # 계좌 변동
            self.available_balance += settled_initial_margin + net_pnl
            self.margin_deposit -= settled_initial_margin

        if get_pnl:
            return net_pnl
        

    def _settle_total_contract(self, market_pt):
        '''
        전체 계약 청산 함수
        보유한 모든 계약을 삭제, 계좌에 손익 반영, 포지션 초기화
        '''
        # 손익, 비용(수수료+슬리피지)
        pnl = self._get_pnl(market_pt, self.execution_strength) 
        cost = self._get_cost(self.execution_strength, market_pt)

        # 순손익
        net_pnl = pnl - cost
        
        # 전체 계약 청산
        self.open_interest_list.clear()
        self.current_name_value = 0
        self.current_position = 0
        self.execution_strength = 0

        self.maintenance_margin = 0

        # 실현 손익
        self.realized_pnl += int(net_pnl)

        # 계좌 변동
        self._return_margin_deposit(int(net_pnl))
        self.total_transaction_costs += cost

        # 정보 업데이트
        self._update_account(market_pt)

        return net_pnl, cost
        
    def _return_margin_deposit(self, net_pnl):
        """전체 증거금을 반환한다."""
        self.available_balance += self.margin_deposit + net_pnl
        self.margin_deposit = 0

    def daily_settlement(self, close_pt):
        '''
        하루 장이 마감된 후 daily settlement 이루어짐 
        '''
        if self.execution_strength != 0:
            daily_settle = self._get_pnl(close_pt, self.execution_strength)
            self.available_balance += daily_settle

            # 직전 스텝 미실현 수익 저장
            self.prev_unrealized_pnl = int(self.unrealized_pnl)

            # 미실현 손익 -> 실현 손익 전환
            self.realized_pnl += int(daily_settle)
            self.unrealized_pnl = 0

    def _update_account(self, market_pt):
        '''
        평균 진입가, 유지증거금, 미실현 손익 갱신
        '''
        if self.execution_strength != 0:
            self.average_entry = np.mean(self.open_interest_list)  # 평균 진입가
            self.maintenance_margin = np.sum(self.open_interest_list) * self.contract_unit * self.maintenance_margin_rate
            self.unrealized_pnl = self._get_pnl(market_pt, self.execution_strength)
        else:
            self.average_entry = 0
            self.maintenance_margin = 0
            self.unrealized_pnl = 0

    def _get_pnl(self, market_pt, vol):
        '''
        손익 계산 함수
        내 계약 중 앞 vol개의 계약에 대해 입력받은 market point에 따른 손익 계산
        return (KRW)
        '''
        entries = np.array(self.open_interest_list[:vol])
        pnl_value = np.sum((market_pt - entries)) * self.current_position 
        return pnl_value * self.contract_unit

    def _calculate_transaction_cost(self, action: int, market_pt) -> float:
        '''
        행동에 따른 거래 비용 계산
        return (KRW)
        '''
        if action == 0:
            return 0.0
        else:
            trade_value = abs(action) * market_pt * self.transaction_cost_rate
            return trade_value  * self.contract_unit 

    def _calculate_slippage(self, action: int, market_pt) -> float:
        '''
        행동에 따른 슬리피지 비용 계산
        return (KRW)
        '''
        if action == 0:
            return 0.0
        else:
            slippage_cost_value = abs(action) * market_pt * self.slippage_factor
            return slippage_cost_value * self.contract_unit

    def _get_cost(self, action: int, market_pt) -> float:
        '''
        행동에 따른 거래 비용 + 슬리피지 비용 계산
        return (KRW)
        '''
        cost = self._calculate_transaction_cost(action, market_pt) + self._calculate_slippage(action, market_pt)
        return cost
    
    def get_position_n_vol(self, action):
        """
        행동의 포지션과 계약 수를 분리해 반환 
        """
        position = np.sign(action)
        vol = abs(action)
        return position, vol
    
    def cache_values(self):
        """
        STEP 이전에 필요한 데이터를 저장 
        """
        self.prev_unrealized_pnl = self.unrealized_pnl
        self.prev_balance =self.balance
        self.prev_position = self.current_position
        self.prev_timestep = self.current_timestep

    @property
    def is_insufficient_for_new_contract(self):
        # 현재가 기준 1계약도 더 체결 불가능한 상태 
        min_contract_margin = self.market_pt * self.initial_margin_rate * self.contract_unit
        return self.balance - self.maintenance_margin < min_contract_margin

    @property
    def balance(self) -> float:
        # 잔고 (미실현 수익 포함)
        return self.available_balance + self.unrealized_pnl

    def __str__(self):
        """계좌 상태 출력"""     
        return (
            f"===============================================\n"
            f"📁 1. Account Status (계좌 상태)\n"
            f"⏱️  Current Timestep   : {self.current_timestep}\n"
            f"💰  Available Balance  : {self.available_balance:,.0f} KRW\n"
            f"💼  Margin Deposit     : {self.margin_deposit:,.0f} KRW\n"
            f"💸  Transaction Costs  : {self.total_transaction_costs:,.0f} KRW\n"
            f"📉  Unrealized PnL     : {self.unrealized_pnl:,.0f} KRW\n"
            f"💵  Realized PnL       : {self.realized_pnl:,.0f} KRW\n"
            f"💰  Total Balance       : {self.balance:,.0f} KRW\n"
            f"⚖️  Avg Entry Price    : {self.average_entry:.2f}\n"
            f"💼  Current Position   : {self.position_dict[self.current_position]} ({self.current_position})\n"
            f"📊  Execution Strength : {self.execution_strength}/{self.position_cap}\n"
            f"🔢  Total Trades       : {self.total_trades}\n"
            f"===============================================\n"
        )
