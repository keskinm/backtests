from keras.models import load_model

from agent.agent import DQN


model_name = "model_ep1000"
model = load_model("models/" + model_name)
window_size = model.layers[0].input.shape.as_list()[1]

agent = DQN(window_size, True, model_name)

agent.evaluate(stock_name="^GSPC_2011")
