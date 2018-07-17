"""Training agent to play CARTPOLE-v0 game, using RL from keras-rl library"""

import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

#Set relevant varibales
ENV_NAME="CartPole-v0"
#Get the enviornment and extract the no. of actions available in Cartpole problem
env=gym.make(ENV_NAME);
np.random.seed(123)
env.seed(123)
nb_actions=env.action_space.n; #We have 2 actions in this game, move left or right

#Now we build a single hidden layer neural network model. See https://keras.io/
model=Sequential();
model.add(Flatten(input_shape=(1,) + env.observation_space.shape)); #Tells input_shape to expect
model.add(Dense(16)); #First Hidden layer of size 16
model.add(Activation("relu")); #Use relu activation function for 1st hidden layers
model.add(Dense(nb_actions)); #Final layer of size = no of outputs # Here=2
model.add(Activation("linear"));
print(model.summary());

#Now we configure and compile our agent
policy=EpsGreedyQPolicy();
memory=SequentialMemory(limit=50000, window_length=1);
dqn=DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10, target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae']);
#Now we learn and also visualize it learning. Set visualize=False to learn faster
dqn.fit(env,nb_steps=10000, visualize=True, verbose=2);

#Test our rl model
dqn.test(env, nb_episodes=50, visualize=True);
