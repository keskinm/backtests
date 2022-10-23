import os
from pathlib import Path

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam

import numpy as np
import random
from collections import deque

from functions import get_state, get_stock_data_vec, format_price


class DQN:
    name = "DQN"

    def __init__(self, state_size, is_eval=False, model_path="", money=500, shares=0):
        self.action_space = {
            0: "sit",
            1: "buy",
            2: "sell"
        }
        self.money = money
        self.shares = shares

        self.state_size = state_size  # normalized previous days

        self.memory = deque(maxlen=1000)
        self.inventory = []

        self.model_dir_path = Path(f"models/{self.name}")
        self.model_path = model_path

        self.is_eval = is_eval

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.model = load_model(model_path) if is_eval else self._model()

    def _model(self):
        model = Sequential()
        model.add(Dense(units=64, input_dim=self.state_size, activation="relu"))
        model.add(Dense(units=32, activation="relu"))
        model.add(Dense(units=8, activation="relu"))
        model.add(Dense(3, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=0.001))
        return model

    def act(self, state):
        if not self.is_eval and np.random.rand() <= self.epsilon:
            return random.randrange(3)

        options = self.model.predict(state)
        return np.argmax(options[0])

    def expReplay(self, batch_size):
        mini_batch = []
        l = len(self.memory)
        for i in range(l - batch_size + 1, l):
            mini_batch.append(self.memory[i])

        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, stock_name, episode_count):
        data = get_stock_data_vec(stock_name)
        l = len(data) - 1
        batch_size = 32

        for e in range(episode_count + 1):
            print("Episode " + str(e) + "/" + str(episode_count))
            state = get_state(data, 0, self.state_size + 1)

            total_profit = 0
            self.inventory = []

            for t in range(l):
                action = self.act(state)

                # sit
                next_state = get_state(data, t + 1, self.state_size + 1)
                reward = 0

                if action == 1:  # buy
                    self.inventory.append(data[t])
                    print("Buy: " + format_price(data[t]))

                elif action == 2 and len(self.inventory) > 0:  # sell
                    bought_price = self.inventory.pop(0)
                    reward = max(data[t] - bought_price, 0)
                    total_profit += data[t] - bought_price
                    print("Sell: " + format_price(data[t]) + " | Profit: " + format_price(data[t] - bought_price))

                done = True if t == l - 1 else False
                self.memory.append((state, action, reward, next_state, done))
                state = next_state

                if done:
                    print("--------------------------------")
                    print("Total Profit: " + format_price(total_profit))
                    print("--------------------------------")

                if len(self.memory) > batch_size:
                    self.expReplay(batch_size)

            if e % 10 == 0:
                if not self.model_dir_path.exists():
                    os.makedirs(self.model_dir_path, exist_ok=True)
                self.model.save(self.model_dir_path / f"model_ep{e}")

    def evaluate(self, stock_name):
        data = get_stock_data_vec(stock_name)
        l = len(data) - 1

        state = get_state(data, 0, self.state_size + 1)
        total_profit = 0
        self.inventory = []

        for t in range(l):
            action = self.act(state)

            # sit
            next_state = get_state(data, t + 1, self.state_size + 1)
            reward = 0

            if action == 1:  # buy
                self.inventory.append(data[t])
                print("Buy: " + format_price(data[t]))

            elif action == 2 and len(self.inventory) > 0:  # sell
                bought_price = self.inventory.pop(0)
                reward = max(data[t] - bought_price, 0)
                total_profit += data[t] - bought_price
                print("Sell: " + format_price(data[t]) + " | Profit: " + format_price(data[t] - bought_price))

            done = True if t == l - 1 else False
            self.memory.append((state, action, reward, next_state, done))
            state = next_state

            if done:
                print("--------------------------------")
                print(stock_name + " Total Profit: " + format_price(total_profit))
                print("--------------------------------")


class ContinuousDQN(DQN):
    """Deep Q learning network with continuous action space."""

    def __init__(self, state_size):
        DQN.__init__(self, state_size)

    def _model(self):
        model = Sequential()
        model.add(Dense(units=64, input_dim=self.state_size, activation="relu"))
        model.add(Dense(units=32, activation="relu"))
        model.add(Dense(units=8, activation="relu"))
        model.add(Dense(1, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=0.001))
        return model

    def interpret_prediction(self, output):
        if -0.2 < output < 0.2:
            output = 0
        if output > 0:
            output *= self.money
        if output < 0:
            output *= self.shares
        return output

    def act(self, state, current_price):
        if not self.is_eval and np.random.rand() <= self.epsilon:
            interval = [-self.shares, self.money/current_price]
            action = random.randrange(3)
            if action == 0:
                mean = (interval[0] + interval[1]) / 2
                variance = (interval[1] - interval[0]) * 0.4
            elif action == 1:
                mean = (interval[0] + 2*interval[1]) / 3
                variance = (interval[1] - interval[0]) * 0.1
            elif action == 2:
                mean = (2*interval[0] + interval[1]) / 3
                variance = (interval[1] - interval[0]) * 0.1
            else:
                raise ValueError
            return np.random.normal(mean, variance, size=(2, 4))

        return self.interpret_prediction(self.model.predict(state))

    def expReplay(self, batch_size):
        mini_batch = []
        l = len(self.memory)
        for i in range(l - batch_size + 1, l):
            mini_batch.append(self.memory[i])

        for state, action, reward, next_state, done in mini_batch:
            reward = np.log(1+reward)

            target = reward
            if not done:
                target = -reward + self.gamma * self.interpret_prediction(self.model.predict(next_state))

            self.model.fit(state, target, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, stock_name, episode_count):
        data = get_stock_data_vec(stock_name)
        l = len(data) - 1
        batch_size = 32

        for e in range(episode_count + 1):
            print("Episode " + str(e) + "/" + str(episode_count))
            state = get_state(data, 0, self.state_size + 1)

            total_profit = 0

            for t in range(l):
                current_price = data[t]
                action = self.act(state, current_price)

                # sit
                next_state = get_state(data, t + 1, self.state_size + 1)
                reward = 0

                if action > 0:  # buy
                    self.money -= current_price*action
                    self.shares += action
                    print("Buy: " + format_price(current_price*action))

                elif action < 0: # sell
                    old_money = self.money
                    self.money += current_price * -action
                    self.shares -= action
                    profit = self.money - old_money
                    reward = profit / old_money
                    total_profit += profit
                    print("Sell: " + format_price(current_price *-action) + " | Profit: " + format_price(profit))

                done = True if t == l - 1 else False
                self.memory.append((state, action, reward, next_state, done))
                state = next_state

                if done:
                    print("--------------------------------")
                    print("Total Profit: " + format_price(total_profit))
                    print("--------------------------------")

                if len(self.memory) > batch_size:
                    self.expReplay(batch_size)

            if e % 10 == 0:
                if not self.model_dir_path.exists():
                    os.makedirs(self.model_dir_path, exist_ok=True)
                self.model.save(self.model_dir_path / f"model_ep{e}")