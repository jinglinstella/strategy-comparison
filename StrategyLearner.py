""""""
"""  		  	   		  		 			  		 			     			  	 
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  		 			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		  		 			  		 			     			  	 
All Rights Reserved  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
Template code for CS 4646/7646  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  		 			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		  		 			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		  		 			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		  		 			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		  		 			  		 			     			  	 
or edited.  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
We do grant permission to share solutions privately with non-students such  		  	   		  		 			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		  		 			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  		 			  		 			     			  	 
GT honor code violation.  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
-----do not edit anything above this line---  	  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
Student Name: Tucker Balch (replace with your name)  		  	   		  		 			  		 			     			  	 
GT User ID: jtao66 (replace with your User ID)
GT ID: 903847861 (replace with your GT ID)		  	   		  		 			  		 			     			  	 
"""

import datetime as dt

import pandas as pd
import util as ut
import random
import QLearner as ql
import indicators

class StrategyLearner(object):


    def __init__(self, verbose=False, impact=0.0):
        self.verbose = verbose
        self.impact = impact
        random.seed(903847861)

        self.learner = ql.QLearner(num_states=100, num_actions=3, alpha=0.2, gamma=0.9,rar=0.9, radr=0.99, dyna=100, verbose=False)

    def add_evidence(self, symbol="AAPL", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 1, 1), sv=100000):

        syms = [symbol]
        prices_with_spy = ut.get_data(syms, pd.date_range(sd, ed))
        df_prices = prices_with_spy.drop('SPY', axis=1)
        dates = df_prices.index
        df_trades = pd.DataFrame(0, index=prices_with_spy.index, columns=[symbol])


        ema_20 = indicators.ema(sd, ed, symbol, 20)
        ema_20 = (df_prices[symbol] > ema_20[symbol]).astype(int)
        ema_20 = pd.DataFrame(ema_20, columns=[symbol])

        ema_30 = indicators.ema(sd, ed, symbol, 30)
        ema_30 = (df_prices[symbol] > ema_30[symbol]).astype(int)
        ema_30 = pd.DataFrame(ema_30, columns=[symbol])

        macd_raw, macd_signal = indicators.macd(sd, ed, symbol)
        macd = (macd_raw[symbol] > macd_signal[symbol]).astype(int)
        macd = pd.DataFrame(macd, columns=[symbol])

        tsi = indicators.tsi(sd, ed, symbol)
        tsi_numeric = tsi[symbol]
        tsi_arr = tsi_numeric.apply(lambda x: 1 if x > 0 else 0)
        tsi = pd.DataFrame(tsi_arr, columns=[symbol])

        # momentum = indicators.momentum(sd, ed, symbol, window_size=20)
        # momentum_arr = momentum.applymap(lambda x: 1 if x>0 else 0)
        # momentum = pd.DataFrame(momentum_arr, columns=[symbol])
        # momentum = momentum.fillna(0)

        current_position = 0
        current_cash = 100000
        previous_position = 0
        previous_cash = 100000

        for i in range(1, len(dates)):
            # today = dates[i]
            s_prime = (
                    (16 if current_position == 0 else 32 if current_position == 1000 else 0)
                    + ema_30.at[dates[i],symbol]*8
                    + ema_20.at[dates[i],symbol]*4
                    + macd.at[dates[i],symbol]*2
                    + tsi.at[dates[i],symbol]
            ) # with today's indicator, this is the future state

            # s_prime = (
            #         (32 if current_position == 0 else 64 if current_position == 1000 else 0)
            #         + ema_30.at[today,symbol]*16
            #         + ema_20.at[today,symbol]*8
            #         + macd.at[today,symbol]*4
            #         + tsi.at[today,symbol]*2
            #         + int(momentum.at[today, symbol])
            # ) # with today's indicator, this is the future state

            curr_r = current_position * df_prices.at[dates[i], symbol] + current_cash
            pre_r = previous_position * df_prices.at[dates[i], symbol] + previous_cash
            r = curr_r - pre_r

            # print("s_prime:", s_prime)

            next_action = self.learner.query(s_prime, r)
            # here we call query to update the q table
            # but in testPolicy we call querysetstate so that we don't update anymore
            # first, we randomly generate 0,1,2 as next action
            if next_action == 0:
                trade = -1000 - current_position
            elif next_action == 1:
                trade = -current_position
            else:
                trade = 1000 - current_position

            previous_position = current_position
            current_position = current_position + trade

            df_trades.loc[dates[i]].loc[symbol] = trade

            impact = -self.impact if trade < 0 else self.impact

            previous_cash = current_cash
            current_cash += -df_prices.loc[dates[i]].loc[symbol] * (1 + impact) * trade


    def testPolicy(self, symbol="IBM", sd=dt.datetime(2009, 1, 1), ed=dt.datetime(2010, 1, 1), sv=100000):

        syms = [symbol]
        prices_with_spy = ut.get_data(syms, pd.date_range(sd, ed))
        df_prices = prices_with_spy.drop('SPY', axis=1)
        dates = df_prices.index
        df_trades = pd.DataFrame(0, index=prices_with_spy.index, columns=[symbol])

        ema_20 = indicators.ema(sd, ed, symbol, 20)
        ema_20 = (df_prices[symbol] > ema_20[symbol]).astype(int)
        ema_20 = pd.DataFrame(ema_20, columns=[symbol])

        ema_30 = indicators.ema(sd, ed, symbol, 30)
        ema_30 = (df_prices[symbol] > ema_30[symbol]).astype(int)
        ema_30 = pd.DataFrame(ema_30, columns=[symbol])

        macd_raw, macd_signal = indicators.macd(sd, ed, symbol)
        macd = (macd_raw[symbol] > macd_signal[symbol]).astype(int)
        macd = pd.DataFrame(macd, columns=[symbol])

        tsi = indicators.tsi(sd, ed, symbol)
        tsi_numeric = tsi[symbol]
        tsi_arr = tsi_numeric.apply(lambda x: 1 if x > 0 else 0)
        tsi = pd.DataFrame(tsi_arr, columns=[symbol])

        # momentum = indicators.momentum(sd, ed, symbol, window_size=20)
        # momentum_arr = momentum.applymap(lambda x: 1 if x>0 else 0)
        # momentum = pd.DataFrame(momentum_arr, columns=[symbol])
        # momentum = momentum.fillna(0)

        curr_position = 0

        for i in range(1, len(dates)):

            s_prime = (
                    (16 if curr_position == 0 else 32 if curr_position == 1000 else 0)
                    + ema_30.at[dates[i],symbol]*8
                    + ema_20.at[dates[i],symbol]*4
                    + macd.at[dates[i],symbol]*2
                    + tsi.at[dates[i],symbol]
            )

            next_a = self.learner.querysetstate(s_prime)
            if next_a == 0:
                trade = -1000 - curr_position
            elif next_a == 1:
                trade = -curr_position
            else:
                trade = 1000 - curr_position

            curr_position += trade
            df_trades.loc[dates[i]].loc[symbol] = trade

        return df_trades


def author():
    return 'jtao66'

if __name__ == "__main__":
    pass



