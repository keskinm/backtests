import sys

sys.path.append("../..")
from utilities.get_data import get_historical_from_db
import pandas as pd
import ccxt
import matplotlib.pyplot as plt
import ta

# --- Load Data ---
pair = "BTC/USDT"
tf = "5m"

df = get_historical_from_db(ccxt.binance(), pair, tf, path="./database/")


def custom_grid(
    first_price, last_order_down=0.5, last_order_up=1, down_grid_len=50, up_grid_len=50
):
    """Function that create 2 array grid_buy and grid_sell

    Args:
        first_price (float): price at the begining
        last_order_down (float, optional): Percentage of the last grid buy. Defaults to 0.5.
        last_order_up (int, optional): Percentage of the last grid sell. Defaults to 1.
        down_grid_len (int, optional): Initial length of grid buy. Defaults to 50.
        up_grid_len (int, optional): Initial length of grid sell. Defaults to 50.

    Returns:
        array: return 2 array of float for price.
    """
    down_pct_unity = last_order_down / down_grid_len
    up_pct_unity = last_order_up / up_grid_len

    grid_sell = []
    grid_buy = []

    for i in range(down_grid_len):
        grid_buy.append(first_price - first_price * down_pct_unity * (i + 1))

    for i in range(up_grid_len):
        grid_sell.append(first_price + first_price * up_pct_unity * (i + 1))

    return grid_buy, grid_sell


grid_buy, grid_sell = custom_grid(
    5000, last_order_down=0.35, last_order_up=2, down_grid_len=10, up_grid_len=20
)

dt = df.copy().loc[:]  # Initial test
# dt = df.copy().loc["2021-01-08":] # Test 1
# dt = df.copy().loc["2020-02-10":] #Test 2
# dt = df.copy().loc["2021-11-19 00":"2022-01-20 17"] #Test 3


first_price = dt.iloc[0]["close"]

grid_buy, grid_sell = custom_grid(
    first_price,
    last_order_down=0.35,
    last_order_up=16,
    down_grid_len=30,
    up_grid_len=70,
)  # Initial nice test
# grid_buy, grid_sell = custom_grid(first_price, last_order_down = 0.3, last_order_up = 1, down_grid_len=40, up_grid_len=60) # Initial bad test
# grid_buy, grid_sell = custom_grid(first_price, last_order_down = 0.35, last_order_up = 0.75, down_grid_len=20, up_grid_len=30) # Test 1
# grid_buy, grid_sell = custom_grid(first_price, last_order_down = 0.7, last_order_up = 2.5, down_grid_len=40, up_grid_len=60) #Test 2
# grid_buy, grid_sell = custom_grid(first_price, last_order_down = 0.35, last_order_up = 0.35, down_grid_len=20, up_grid_len=20) #Test 3


trade_list = []
grid_buy_to_insert = 0
grid_sell_to_insert = 0

usd = 500
crypto = 500 / first_price

print("Starting price", first_price)
nb_same_index = 0
for index, row in dt.iterrows():

    try:
        if grid_buy_to_insert > 0:
            # print(row)
            grid_buy_diff = (row["open"] - grid_buy[0]) / (grid_buy_to_insert + 1)
            for i in range(grid_buy_to_insert):
                # print("grid buy", grid_buy[0]+grid_buy_diff)
                grid_buy.insert(0, grid_buy[0] + grid_buy_diff)

        if grid_sell_to_insert > 0:
            # print(row)
            grid_sell_diff = (grid_sell[0] - row["open"]) / (grid_sell_to_insert + 1)
            for i in range(grid_sell_to_insert):
                # print("grid_sell", grid_sell[0]-grid_sell_diff)
                grid_sell.insert(0, grid_sell[0] - grid_sell_diff)

    except:
        pass

    grid_buy_to_insert = 0
    grid_sell_to_insert = 0

    # if len(grid_buy) + len(grid_sell) != 120:
    #     print("error grid")

    if len(grid_buy) == 0 and usd < 0.05 * (crypto * row["open"]):
        print("End of buy grid => reset wallet and grid", index)
        # --- You can change perameters here for end of grid buy ---
        grid_buy, grid_sell = custom_grid(
            row["open"],
            last_order_down=0.3,
            last_order_up=1,
            down_grid_len=40,
            up_grid_len=60,
        )
        usd = 0.5 * (usd + crypto * row["open"])
        crypto = 0.5 * (usd + crypto * row["open"]) / row["open"]

    elif len(grid_sell) == 0 and (crypto * row["open"]) < 0.05 * usd:
        print("End of sell grid => reset wallet and grid", index)
        # --- You can change perameters here for end of grid sell ---
        grid_buy, grid_sell = custom_grid(
            row["open"],
            last_order_down=0.3,
            last_order_up=1,
            down_grid_len=40,
            up_grid_len=60,
        )
        usd = 0.5 * (usd + crypto * row["open"])
        crypto = 0.5 * (usd + crypto * row["open"]) / row["open"]

    check_same_index = False
    # -- BUY --

    if row["high"] > grid_sell[0]:
        try:
            while row["high"] > grid_sell[0]:
                crypt_to_sell = crypto / len(grid_sell)
                crypto -= crypt_to_sell
                usd += crypt_to_sell * grid_sell[0]
                trade_list.append(
                    {
                        "date": index,
                        "side": "Sell",
                        "usd_amount": crypt_to_sell * grid_sell[0],
                        "price": grid_sell[0],
                        "usd": usd,
                        "crypto": crypto,
                        "wallet": usd + crypto * grid_sell[0],
                    }
                )
                grid_buy_to_insert += 1
                del grid_sell[0]
                check_same_index = True
        except:
            print("End of grid sell", row["close"], index)
            pass

    # -- BUY --

    if row["low"] < grid_buy[0]:
        try:
            if check_same_index == True:
                nb_same_index += 1
                # print((row["high"]-row["low"])/row["close"])
            while row["low"] < grid_buy[0]:
                buy_usd_amount = usd / len(grid_buy)
                crypto += buy_usd_amount / grid_buy[0]
                usd -= buy_usd_amount
                trade_list.append(
                    {
                        "date": index,
                        "side": "Buy",
                        "usd_amount": buy_usd_amount,
                        "price": grid_buy[0],
                        "usd": usd,
                        "crypto": crypto,
                        "wallet": usd + crypto * grid_buy[0],
                    }
                )
                grid_sell_to_insert += 1
                del grid_buy[0]
        except:
            print("End of grid buy", row["close"], index)
            pass

print("Number of same index", nb_same_index)

df_trades = pd.DataFrame(trade_list).iloc[:]
df_trades["wallet_ath"] = df_trades["wallet"].cummax()
df_trades["price_ath"] = df_trades["price"].cummax()
df_trades["wallet_drawdown_pct"] = (
    df_trades["wallet_ath"] - df_trades["wallet"]
) / df_trades["wallet_ath"]
df_trades["price_drawdown_pct"] = (
    df_trades["price_ath"] - df_trades["price"]
) / df_trades["price_ath"]
max_trades_drawdown = df_trades["wallet_drawdown_pct"].max()
max_price_drawdown = df_trades["price_drawdown_pct"].max()
wallet_perf = (
    df_trades.iloc[-1]["wallet"] - df_trades.iloc[0]["wallet"]
) / df_trades.iloc[0]["wallet"]
price_perf = (
    df_trades.iloc[-1]["price"] - df_trades.iloc[0]["price"]
) / df_trades.iloc[0]["price"]
print("Total trades:", len(df_trades))
print("\n--- Wallet ---")
print("Wallet performance: {}%".format(round(wallet_perf * 100, 2)))
print("Worst Wallet Drawdown: -{}%".format(round(max_trades_drawdown * 100, 2)))
print("\n--- Asset ---")
print("Asset performance: {}%".format(round(price_perf * 100, 2)))
print("Worst Asset Drawdown: -{}%".format(round(max_price_drawdown * 100, 2)))
df_trades
