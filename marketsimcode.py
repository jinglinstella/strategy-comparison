
import pandas as pd

from util import get_data

def compute_portvals(orders_df, start_val = 100000, commission=9.95, impact=0.005):

    sd = orders_df.index[0]
    ed = orders_df.index[-1]

    portvals = get_data(['SPY'], pd.date_range(sd, ed), addSPY=True, colname = 'Adj Close').rename(columns={'SPY': 'value'})
    symbol = orders_df.columns[0]

    current_cash = 100000
    myshares_dict = {}
    symbol_dict = {}

    for date in portvals.index:
        trade = orders_df.at[date, symbol]
        if trade != 0:
            if symbol not in symbol_dict:
                symbol_df_price = get_data([symbol], pd.date_range(date, ed), addSPY=True, colname='Adj Close')
                # price range is from this date to end date
                symbol_dict[symbol]=symbol_df_price

            if trade > 0:
                position_change = trade
                df = symbol_dict[symbol]
                cash_change = -df.loc[date][symbol] * trade * (1 + impact)
            elif trade < 0:
                position_change = -abs(trade)
                df = symbol_dict[symbol]
                cash_change = df.loc[date][symbol] * abs(trade) * (1 - impact)

            myshares_dict[symbol] = myshares_dict.get(symbol, 0) + position_change
            current_cash += cash_change - commission

        shares_value = 0
        for symbol in myshares_dict:
            shares_value += symbol_dict[symbol].at[date, symbol] * myshares_dict[symbol]

        portvals.at[date, 'value'] = current_cash + shares_value

    return portvals


def author():
    return 'jtao66'
	  			  	 		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		   	  			  	 		  		  		    	 		 		   		 		  
    pass