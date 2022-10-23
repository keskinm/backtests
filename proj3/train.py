from agent.agent import DQN, ContinuousDQN

# agent = DQN(state_size=10)
agent = ContinuousDQN(state_size=10)

agent.train(stock_name="^GSPC", episode_count=200)


