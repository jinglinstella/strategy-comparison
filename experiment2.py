from StrategyLearner import StrategyLearner
import datetime as dt  		   	  			  	 		  		  		    	 		 		   		 		  
import matplotlib.pyplot as plt
from marketsimcode import compute_portvals

def experiment2():
    sd = dt.datetime(2008,1,1)
    ed = dt.datetime(2009,12,31)

    sl1 = StrategyLearner(impact = 0.000)
    sl1.add_evidence(symbol = "JPM", sd=sd, ed=ed, sv = 100000)
    sl1_trades = sl1.testPolicy(symbol = "JPM", sd=sd, ed=ed, sv = 100000)
    sl1_portval = compute_portvals(sl1_trades, start_val = 100000, commission=0, impact=0.000)
    sl1_portval['value']=sl1_portval['value']/sl1_portval['value'][0]

    sl2 = StrategyLearner(impact = 0.005)
    sl2.add_evidence(symbol = "JPM", sd=sd, ed=ed, sv = 100000)
    sl2_trades = sl2.testPolicy(symbol = "JPM", sd=sd, ed=ed, sv = 100000)
    sl2_portval = compute_portvals(sl2_trades, start_val = 100000, commission=0, impact=0.005)
    sl2_portval['value'] = sl2_portval['value'] / sl2_portval['value'][0]

    sl3 = StrategyLearner(impact = 0.02)
    sl3.add_evidence(symbol = "JPM", sd=sd, ed=ed, sv = 100000)
    sl3_trades = sl3.testPolicy(symbol = "JPM", sd=sd, ed=ed, sv = 100000)
    sl3_portval = compute_portvals(sl3_trades, start_val = 100000, commission=0, impact=0.02)
    sl3_portval['value'] = sl3_portval['value'] / sl3_portval['value'][0]

    plt.title("Experiment 2: Effect of Impact Value")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")

    plt.plot(sl1_portval, label="impact: 0.000", color = "red")
    plt.plot(sl2_portval, label="impact: 0.005", color = "green")
    plt.plot(sl3_portval, label="impact: 0.02", color = "blue")
    plt.legend()
    plt.savefig("experiment2.png")
    # plt.clf()



def author():
    return 'jtao66'

if __name__=="__main__":  		   	  			  	 		  		  		    	 		 		   		 		  
    experiment2()

