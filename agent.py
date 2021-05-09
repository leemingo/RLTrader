import numpy as np

def sigmoid(x):
    return 1 / (1 +np.exp(-x))

class Agent:
    '''
    투자 행동 수행 및 투자금과 보유 주식을 관리하기 위한 클래스
    '''
    # 에이전트 상태가 구성하는 값 개수
    STATE_DIM = 2  # 주식 보유 비율, 포트폴리오 가치 비율 두개라서 2

    # 매매 수수료 및 세금
    TRADING_CHARGE = 0.00015  # 거래 수수료 (일반적으로 0.015%)
    TRADING_TAX = 0.0025  # 거래세 (실제 0.25%)

    # 행동
    ACTION_BUY = 0  # 매수
    ACTION_SELL = 1  # 매도
    ACTION_HOLD = 2  # 홀딩
    # 인공 신경망에서 확률을 구할 행동들
    ACTIONS = [ACTION_BUY, ACTION_SELL]
    NUM_ACTIONS = len(ACTIONS)  # 인공 신경망에서 고려할 출력값의 개수

    def __init__(self, environment, min_trading_unit=1, max_trading_unit=2, delayed_reward_threshold=0.00000001):
        self.environment = environment  # environment class

        # 최대 단일 거래 단위가 클 경우, 행동에 대한 확신이 높을 때 더 많이 매수할 수 있음
        self.min_trading_unit = min_trading_unit  # 최소 단일 거래 단위
        self.max_trading_unit = max_trading_unit  # 최대 단일 거래 단위

        # 지연 보상 임계치. 손익률이 이 값을 넘을 경우,지연 보상이 발생
        self.delayed_reward_threshold = delayed_reward_threshold

        self.initial_balance = 0  # 초기 투자금
        self.balance = 0  # 현금 잔고
        self.num_stocks = 0  # 보유 주식 수
        # PV = balance + num_stocks * (현재 주식 가격)
        self.portfolio_value = 0
        self.base_portfolio_value = 0  # 직전 학습 시접의 PV. 현재 포트폴리오 가치가 증가했는지, 감소했는지 비교할 기준
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0
        self.immediate_reward = 0  # 에이전트가 가장 최근에 한 행동에 대한 즉시 보상 값
        self.profitloss = 0
        self.base_profitloss = 0  # 직접 지연 보상 이후 손익
        self.exploration_base = 0  # 탐험 매수 기준으로? 매도 기준으로?

        self.ratio_hold = 0  # 주식 보유 비율
        self.ratio_portfolio_value = 0  # 포트폴리오 보유 비율

    def reset(self):
        '''
        Agent class 속성들을 초기화
        한 에포크마다 초기화
        '''
        self.balance = self.initial_balance
        self.num_stocks = 0
        self.portfolio_value = self.initial_balance
        self.base_portfolio_value = self.initial_balance
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0
        self.immediate_reward = 0
        self.ratio_hold = 0
        self.ratio_portfolio_value = 0

    def reset_exploration(self):
        '''
        탐험의 기준이 되는 exploration base를 새로 정하는 함수
        '''
        self.exploration_base = 0.5 + np.random.rand() / 2  # 0.5를 미리 더함으로 매수를 선호하게 설정

    def set_balance(self, balance):
        self.initial_balance = balance  # 초기 자본금 설정

    def get_states(self):
        '''
        보유 주식 수를 투자 행동 결정에 영향을 주기 위함
        현재 보유한 주식 / 현재가격에서 가질 수 있는 최대 주식 수
        ratio_hold가 0이면 주식 하나도 없고, 0.5면 최대 가질 수 있는 것 대비 절반만 보유
        ratio_hold가 0이면 매수의 관점에서 볼 것이고, 1이면 매도의 관점에서 투자에 임할 것
        '''
        self.ratio_hold = self.num_stocks / int(self.portfolio_value / self.environment.get_price())
        '''
        포트폴리오 가치 비율. 기준 포트폴리오 대비 현재 포트폴리오 가치
        0에 가까울 수록 손실이 크고, 1에 가까울 수록 이익이 크다
        '''
        self.ratio_portfolio_value = (self.portfolio_value / self.base_portfolio_value)
        return (self.ratio_hold, self.ratio_portfolio_value)

    def decide_action(self, pred_value, pred_policy, epsilon):
        '''
        epsilon의 값에 따라 랜덤으로 생성한 값이 epsilon보다 작으면 무작위로 행동을 결정, 크면 신경망을 통해 행동 결정
        '''
        confidence = 0.

        # pred_policy: 정책 신경망의 출력
        pred = pred_policy
        if pred is None:
            # DQN Learner의 경우, 정책 신경망이 없으므로, pred_value로 행동을 결정
            pred = pred_value
        if pred is None:
            # 예측 값이 없을 때 탐험
            epsilon = 1

        else:
            # 값이 모두 같은 경우 탐험
            maxpred = np.max(pred)
            if (pred == maxpred).all():
                epsilon = 1
        # epsilon보다 작아서 random으로 하는 부분
        if np.random.rand() < epsilon:
            exploration = True
            # 기본적으로 exploration_base에 0.5가 더해져있기 때문에, buy를 더 선호
            if np.random.rand() < self.exploration_base:
                action = self.ACTION_BUY
            else:
                # 이게 왜 self.ACTION_SELL로 안하고 아래처럼 만든거지??
                action = np.random.randint(self.NUM_ACTIONS - 1) + 1
        # random 아닌 부분. 즉 신경망에 따라 행동 결정하는 부분
        else:
            exploration = False
            # buy의 확률이 클 경우, np.argmax(pred) = 0, sell이면 1
            action = np.argmax(pred)

        # confidence 초기값 (value network랑 policy network 모두 없을 때)
        confidence = 0.5
        if pred_policy is not None:
            confidence = pred[action]

        elif pred_value is not None:
            confidence = sigmoid(pred[action])
        return action, confidence, exploration

    def validate_action(self, action):
        '''
        가능한 행동만 할 수 있도록 하는 함수
        현금이 있을 때만 주식 사고, 주식이 있을 때만 주식 팔게
        '''
        if action == Agent.ACTION_BUY:
            if self.balance < self.environment.get_price() * (1 + self.TRADING_CHARGE) * self.min_trading_unit:
                return False
        elif action == Agent.ACTION_SELL:
            if self.num_stocks <= 0:
                return False
        return True

    def decide_trading_unit(self, confidence):
        '''
        정책 신경망이 결정한 행동의 신뢰가 높을수록 매수 또는 매도하는 단위를 크게 지정
        '''
        if np.isnan(confidence):
            return self.min_trading_unit
        added_trading = max(min(int(confidence * (self.max_trading_unit - self.min_trading_unit)),
                                self.max_trading_unit - self.min_trading_unit), 0)
        return self.min_trading_unit + added_trading

    def act(self, action, confidence):
        # 첫번째로 트레이딩 가능한지 검사
        if not self.validate_action(action):
            action = Agent.ACTION_HOLD

        # 현재 주가 가져오기
        curr_price = self.environment.get_price()

        self.immediate_reward = 0

        if action == Agent.ACTION_BUY:

            # 매수 단위 설정
            trading_unit = self.decide_trading_unit(confidence)
            # 현금잔고 체크
            balance = (self.balance - curr_price * (1 + self.TRADING_CHARGE) * trading_unit)
            # 현금잔고 -될 경우, 매수 단위 재설정
            if balance < 0:
                trading_unit = max(
                    min(
                        int(self.balance / (curr_price * (1 + self.TRADING_CHARGE))),
                        self.max_trading_unit), self.min_trading_unit)
            # 수수료 적용해 총 매수 금액 선정
            invest_amount = curr_price * (1 + self.TRADING_CHARGE) * trading_unit
            # 매수하고 나서 업데이트
            if invest_amount > 0:
                self.balance -= invest_amount
                self.num_stocks += trading_unit
                self.num_buy += 1

        elif action == Agent.ACTION_SELL:
            # 매도 단위 설정
            trading_unit = self.decide_trading_unit(confidence)
            # 보유주식 모자라면, 최대 팔 수 있는만큼
            trading_unit = min(trading_unit, self.num_stocks)
            invest_amount = curr_price * (1 - (self.TRADING_TAX + self.TRADING_CHARGE)) * trading_unit

            if invest_amount > 0:
                self.num_stocks -= trading_unit
                self.balance += invest_amount
                self.num_sell += 1

        elif action == Agent.ACTION_HOLD:
            self.num_hold += 1

        # 포트폴리오 가치 갱신
        self.portfolio_value = self.balance + curr_price * self.num_stocks
        self.profitloss = ((self.portfolio_value - self.initial_balance) / self.initial_balance)

        # 즉시 보상 - 수익률
        self.immediate_reward = self.profitloss

        # 지연 보상 - 즉시 보상이 지연 보상 임계치인 delayed_reward_threshold를 초과하는 경우 즉시 보상 값으로, 나머지는 0.
        # 지연 보상 임계치를 넘는 수익이 나면, 그 전의 행동들을 잘했다고 보고 학습하고, 그렇지 않을 경우 문제가 있다고 보고 부정적으로 학습

        delayed_reward = 0
        self.base_profitloss = (
                (self.portfolio_value - self.base_portfolio_value) / self.base_portfolio_value)

        if self.base_profitloss > self.delayed_reward_threshold or self.base_profitloss < -self.delayed_reward_threshold:
            self.base_portfolio_value = self.portfolio_value
            delayed_reward = self.immediate_reward

        else:
            delayed_reward = 0
        return self.immediate_reward, delayed_reward