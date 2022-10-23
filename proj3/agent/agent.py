from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam

import numpy as np
import random
from collections import deque

from functions import get_state, get_stock_data_vec, format_price


class DQN:
	def __init__(self, state_size, is_eval=False, model_name=""):
		self.state_size = state_size # normalized previous days
		self.action_size = 3 # sit, buy, sell
		self.memory = deque(maxlen=1000)
		self.inventory = []
		self.model_name = model_name
		self.is_eval = is_eval

		self.gamma = 0.95
		self.epsilon = 1.0
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995

		self.model = load_model("models/" + model_name) if is_eval else self._model()

	def _model(self):
		model = Sequential()
		model.add(Dense(units=64, input_dim=self.state_size, activation="relu"))
		model.add(Dense(units=32, activation="relu"))
		model.add(Dense(units=8, activation="relu"))
		model.add(Dense(self.action_size, activation="linear"))
		model.compile(loss="mse", optimizer=Adam(lr=0.001))

		return model

	def act(self, state):
		if not self.is_eval and np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size)

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
				self.model.save("models/model_ep" + str(e))

	def evaluate(self, stock_name):
		data = get_stock_data_vec(stock_name)
		l = len(data) - 1
		batch_size = 32

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

