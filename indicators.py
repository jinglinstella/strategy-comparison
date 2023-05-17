
import pandas as pd
import datetime as dt
import util as ut

def ema(sd, ed, symbol, window_size):

    sd0 = sd - dt.timedelta(window_size * 2)
    date_range = pd.date_range(sd0, ed)
    df_price = ut.get_data([symbol], date_range).loc[:, [symbol]]

    alpha = 2 / (window_size + 1)
    df_ema = df_price.copy()
    df_ema.iloc[0] = df_price.iloc[0]

    for i in range(1, len(df_price)):
        df_ema.iloc[i] = alpha * df_price.iloc[i] + (1 - alpha) * df_ema.iloc[i - 1]

    df_ema = df_ema.loc[sd:]

    return df_ema

def macd(sd, ed, symbol):

    sd0 = sd - dt.timedelta(26*2)
    data_range=pd.date_range(sd0,ed)
    df_price = ut.get_data([symbol], data_range).loc[:,[symbol]]

    df_ema_12 = df_price.copy()
    df_ema_12.iloc[0] = df_price.iloc[0]
    for i in range(1, len(df_price)):
        df_ema_12.iloc[i] = 2/13 * df_price.iloc[i] + (1 - 2/13) * df_ema_12.iloc[i - 1]

    df_ema_26 = df_price.copy()
    df_ema_26.iloc[0] = df_price.iloc[0]
    for i in range(1, len(df_price)):
        df_ema_26.iloc[i] = 2/27 * df_price.iloc[i] + (1 - 2/27) * df_ema_26.iloc[i - 1]

    raw = df_ema_12 - df_ema_26

    signal = raw.copy()
    signal.iloc[0] = raw.iloc[0]
    for i in range(1, len(raw)):
        signal.iloc[i] = 2/10 * raw.iloc[i] + (1 - 2/10) * signal.iloc[i - 1]
    raw = raw.loc[sd:]
    signal = signal.loc[sd:]
    return raw, signal

def bb(start_date, end_date, symbol, window_size):
    sd0 = start_date - dt.timedelta(window_size * 2)

    df_price = ut.get_data([symbol], pd.date_range(sd0, end_date)).loc[:,[symbol]]
    # prices look like this:
    #           JPM
    # 2011-1-1  123.19
    # 2011-1-2  123.19

    normalized_df_price = df_price[symbol] / df_price[symbol][0]

    sma = normalized_df_price.rolling(window_size).mean()
    std = normalized_df_price.rolling(window_size).std()
    df_bb = (normalized_df_price - sma) / (2 * std)
    df_bb = df_bb.loc[start_date:]
    bb_up = sma + std * 2
    bb_down = sma - std * 2

    return df_bb


def momentum(start_date, end_date, symbol, window_size):
    sd0 = start_date - dt.timedelta(window_size * 2)

    df_price = ut.get_data([symbol], pd.date_range(sd0, end_date)).loc[:,[symbol]]
    # prices look like this:
    #           JPM
    # 2011-1-1  123.19
    # 2011-1-2  123.19

    normalized_df_price = df_price[symbol] / df_price[symbol][0]

    df_price['change'] = df_price[symbol].pct_change(periods=window_size)
    # df_momentum = df_price.drop(columns=['JPM'])
    df_momentum = df_price.loc[start_date:]
    return df_momentum


def compute(data, span):
    return data.ewm(span=span, adjust=False).mean()

def tsi(sd, ed, symbol):

    sd0 = sd - dt.timedelta(70)
    df_price = (
        ut.get_data([symbol], pd.date_range(sd0, ed))
        .loc[:, [symbol]]
        .diff()
    )

    ema_25 = compute(df_price, 25)
    ema_13 = compute(ema_25, 13)

    abs_diff = df_price.abs()
    abs_ema_25 = compute(abs_diff, 25)
    abs_ema_13 = compute(abs_ema_25, 13)
    df_tsi = ema_13 / abs_ema_13

    df_tsi = df_tsi.loc[sd:]

    return df_tsi

def author():
    return 'jtao66'


if __name__ == "__main__":
    pass
