
from util import get_data, plot_data
import datetime as dt
import pandas as pd
from marketsimcode import compute_portvals
import matplotlib.pyplot as plt
import indicators


class ManualStrategy:

    def testPolicy(self, symbol, sd, ed, sv):

        symbol = symbol[0]
        df = get_data([symbol], pd.date_range(sd, ed)).loc[:,[symbol]]
        df_price=df.copy()
        normalized_df_price = df_price[symbol] / df_price[symbol][0]

        df_trades = pd.DataFrame(index=df.index, columns=[symbol], data=0)

        ema_20 = indicators.ema(sd, ed, symbol, window_size=20)
        ema_20 = ema_20[symbol] / ema_20[symbol][0]

        macd_raw, macd_signal = indicators.macd(sd, ed, symbol)
        tsi = indicators.tsi(sd, ed, symbol)

        current_position = 0

        for curr, row in df.iterrows():

            normalized_df_price_curr = normalized_df_price.loc[curr]
            ema_20_curr = ema_20.loc[curr]

            ema_vote = 2 if normalized_df_price_curr > ema_20_curr else (
                -1 if normalized_df_price_curr < ema_20_curr else 0)

            macd_raw_curr = macd_raw.loc[curr, symbol]
            macd_signal_curr = macd_signal.loc[curr, symbol]

            macd_vote = 1 if macd_signal_curr > macd_raw_curr else (-5 if macd_signal_curr < macd_raw_curr else 1)

            tsi_curr = tsi.loc[curr, symbol]

            tsi_vote = 1 if tsi_curr > 0.15 else (-1 if tsi_curr < 0.15 else 0)

            sum = ema_vote + macd_vote + tsi_vote
            action = 1000 - current_position if sum >= 3 else (
                -1000 - current_position if sum <= -3 else -current_position)

            df_trades.loc[curr, symbol] = action
            current_position += action

        return df_trades

def benchmark_portval(sd, ed, sv):

    df_trades = get_data(['SPY'], pd.date_range(sd, ed))
    df_trades = df_trades.rename(columns={'SPY': 'JPM'})
    df_trades[:] = 0
    df_trades.loc[df_trades.index[0]] = 1000
    portval = compute_portvals(df_trades, sv, commission=9.95, impact=0.005)
    return portval

def stats(benchmark, manual):
    benchmark = benchmark['value']
    manual = manual['value']

    cr_benchmark = benchmark[-1] / benchmark[0] - 1
    cr_manual = manual[-1] / manual[0] - 1

    sddr_benchmark = (benchmark / benchmark.shift(1) - 1).iloc[1:].std()
    sddr_manual = (manual / manual.shift(1) - 1).iloc[1:].std()

    mean_benchmark = (benchmark / benchmark.shift(1) - 1).iloc[1:].mean()
    mean_manual = (manual / manual.shift(1) - 1).iloc[1:].mean()

    print("Manual Strategy: ")
    print("Cumulative return: " + str(cr_manual))
    print("STDEV of daily returns: " + str(sddr_manual))
    print("Mean of daily returns: " + str(mean_manual))
    print("------------------------------------------")
    print("Benchmark: ")
    print("Cumulative return: " + str(cr_benchmark))
    print("STDEV of daily returns: " + str(sddr_benchmark))
    print("Mean of daily returns: " + str(mean_benchmark))

def plot(benchmark_portvals, manual_portvals, sample):

    benchmark_portvals['value'] = benchmark_portvals['value'] / benchmark_portvals['value'][0]
    manual_portvals['value'] = manual_portvals['value'] / manual_portvals['value'][0]

    plt.title("Manual Stragety - " + sample)
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.plot(benchmark_portvals, label="benchmark", color = "purple")
    plt.plot(manual_portvals, label="manual", color ="green")

    plt.legend()
    plt.savefig("manual_startegy_{}.png".format(sample))
    plt.clf()


def report():

    sv = 100000
    in_sd = dt.datetime(2008, 1, 1)
    in_ed = dt.datetime(2009,12,31)

    out_sd = dt.datetime(2010, 1, 1)
    out_ed = dt.datetime(2011,12,31)
    symbol = ['JPM']

    in_df_trades = ManualStrategy().testPolicy(symbol, sd=in_sd, ed=in_ed, sv=sv)
    in_manual_portvals = compute_portvals(in_df_trades, sv, commission=9.95, impact=0.005)
    in_benchmark_portvals = benchmark_portval(in_sd, in_ed, sv)

    out_df_trades = ManualStrategy().testPolicy(symbol, sd=out_sd, ed=out_ed, sv=sv)
    out_manual_portvals = compute_portvals(out_df_trades, sv, commission=9.95, impact=0.005)
    out_benchmark_portvals = benchmark_portval(out_sd, out_ed, sv)

    stats(in_benchmark_portvals, in_manual_portvals)
    stats(out_benchmark_portvals, out_manual_portvals)

    plot(in_benchmark_portvals, in_manual_portvals, 'in_sample')
    plot(out_benchmark_portvals, out_manual_portvals, 'out_sample')


def author():
    return 'jtao66'

if __name__ == "__main__":
    report()
