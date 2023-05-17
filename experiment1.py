from StrategyLearner import StrategyLearner
from ManualStrategy import ManualStrategy
from marketsimcode import compute_portvals
import datetime as dt
import util as ut
import matplotlib.pyplot as plt
import pandas as pd


def experiment1():
    symbol = 'JPM'
    sd = dt.datetime(2008,1,1)
    ed = dt.datetime(2009,12,31)
    sv = 100000

    manual_trades = ManualStrategy().testPolicy([symbol], sd=sd, ed=ed, sv=sv)
    manual_portval = compute_portvals(manual_trades, start_val = sv, commission=0, impact=0.000)
    stats(manual_portval, "Manual Strategy")

    learner = StrategyLearner(verbose = False, impact = 0.000)
    learner.add_evidence(symbol = symbol, sd=sd, ed=ed, sv = sv)
    learner_trades = learner.testPolicy(symbol = symbol, sd=sd, ed=ed, sv = sv)
    learner_portval = compute_portvals(learner_trades, start_val = sv, commission=0, impact=0.000)

    df_trades = ut.get_data(['SPY'], pd.date_range(sd, ed))
    df_trades = df_trades.rename(columns={'SPY': 'JPM'})
    df_trades[:] = 0
    df_trades.loc[df_trades.index[0]] = 1000
    benchmark_portval = compute_portvals(df_trades, sv, commission=9.95, impact=0.005)

    stats(learner_portval, "Strategy Learner")

    manual_portval['value'] = manual_portval['value'] / manual_portval['value'][0]
    learner_portval['value'] = learner_portval['value'] / learner_portval['value'][0]
    benchmark_portval['value'] = benchmark_portval['value'] / benchmark_portval['value'][0]

    plt.title("Experiment 1: Manual vs Q-Learning")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.plot(manual_portval, label="Manual", color = "red")
    plt.plot(learner_portval, label="Q-Learning", color = "green")
    plt.plot(benchmark_portval, label="Benchmark", color="purple")

    plt.legend()
    plt.savefig("experiment1.png")

def stats(portval, name):
    portval = portval['value']
    dr = (portval.diff()-1).iloc[1:]
    sddr = dr.std()
    adr = dr.mean()
    print(name + ": ")
    print("Cumulative Return: " + str(portval[-1] / portval[0] - 1))
    print("Stdev of Daily Returns: " + str(sddr))
    print("Mean of daily Returns: " + str(adr))


def author():
    return 'jtao66'

if __name__=="__main__":  		   	  			  	 		  		  		    	 		 		   		 		  
    experiment1()
