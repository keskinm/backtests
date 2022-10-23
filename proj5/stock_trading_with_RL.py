import pandas as pd
import numpy as np
import yfinance as yf

import matplotlib.pyplot as plt

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, A2C

import tensorflow as tf


from env import StockTradingEnv
from agent import DDPG

data = yf.download('GOOGL', period = 'max')


data['Date'] = data.index
data.index = np.arange(0, len(data))

data_train = yf.download('GOOGL', start = '2010-01-01', end = '2018-01-01')
data_test = yf.download('GOOGL', period = '2y', interval = '1h')

data_train['Date'] = data_train.index
data_test['Date'] = data_test.index

data_train.index = np.arange(0, len(data_train))
data_test.index = np.arange(0, len(data_test))


def plot_stock(data):
    plt.rcParams['figure.figsize'] = [14, 8]

    x1 = np.array(data['Date'])
    y1 = data['Open']
    y12 = data['Volume']

    plt.title("Google Stock Performance Over years")
    plt.xlabel("Year")
    plt.ylabel("Price in $")

    price = plt.plot(x1, y1, label="Price in $")

    ax2 = plt.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:red'
    ax2.set_ylabel('Volume', color=color)  # we already handled the x-label with ax1
    # volume = ax2.plot(x1, y12, color=color, label = "Volume")

    # ax2.tick_params(axis='y', labelcolor=color)

    # plots = price + volume
    plots = price
    labels = [plot.get_label() for plot in plots]
    plt.legend(plots, labels, loc="upper left")

    plt.show()


plot_stock(data_test)

profits = {}
stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
TIMESTEPS = 20000



a2c_profits = {}
for stock in stocks:

    data_train = yf.download(stock, start='2010-01-01', end='2018-01-01')
    data_test = yf.download(stock, period='2y', interval='1h')

    data_train['Date'] = data_train.index
    data_test['Date'] = data_test.index

    data_train.index = np.arange(0, len(data_train))
    data_test.index = np.arange(0, len(data_test))

    env = DummyVecEnv([lambda: StockTradingEnv(data_train)])

    model = A2C(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=TIMESTEPS)

    model_profits = []

    for _ in range(10):
        env_test = StockTradingEnv(data_test)
        env_test.reset()
        env.envs[0] = env_test
        obs = env.reset()

        for i in range(len(data_test)):
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            env.render(mode='no')
        print(env.envs[0].profit)
        model_profits.append(env.envs[0].profit)
    a2c_profits[stock] = np.mean(model_profits)

profits['A2C'] = a2c_profits