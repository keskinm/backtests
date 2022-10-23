from agent.agent import DQN

agent = DQN(state_size=10)
agent.train(stock_name="^GSPC", episode_count=1000)


