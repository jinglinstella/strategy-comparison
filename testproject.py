import ManualStrategy as ms
import StrategyLearner as sl
import experiment1 as exp1
import experiment2 as exp2
import datetime as dt


def author():
    return 'jtao66'

if __name__ == "__main__":

    symbol = "JPM"
    learner = sl.StrategyLearner(verbose=False, impact=0.0)
    learner.add_evidence(symbol=symbol, sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    df_trades = learner.testPolicy(symbol=symbol, sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31),sv=100000)

    ms.report()

    exp1.experiment1()
    exp2.experiment2()