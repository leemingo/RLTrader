class Environment:

    PRICE_IDX = 4  #Dataframe에서 종가의 위치

    def __init__(self, chart_data=None):
        self.chart_data = chart_data
        self.observation = None #현재 관측치
        self.idx = -1 #차트 데이터에서의 현재 위치

    def reset(self):
        self.observation = None
        self.idx = -1

    def observe(self):
        #차트에 데이터가 더 있으면
        if len(self.chart_data) > self.idx + 1:
            self.idx += 1
            self.observation = self.chart_data.iloc[self.idx]
            return self.observation
        else:
            return None

    def get_price(self):
        if self.observation is not None:
            return self.observation[self.PRICE_IDX]
        else:
            return None

    def set_chart_data(self, chart_data):
        self.chart_data = chart_data