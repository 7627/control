#python3
import gym
import numpy as np

ENV_NAME="Taxi-v2";
env=gym.make(ENV_NAME);
render=False;

#Table to store Q values
Q=np.zeros([env.observation_space.n,env.action_space.n])

#learning rate
alpha=0.618;
#
gamma=1;

for episode in range(1,1000):
    state=env.reset();
    G,reward=0,0
    done=False
    while done!=True:
        if render: env.render();
        action=np.argmax(Q[state]);
        state2,reward,done,info=env.step(action)
        #Update Q value for that actions
        Q[state,action]+=alpha*(reward + gamma*np.max(Q[state2]) - Q[state,action])
        state=state2;
        G+=reward;
    if episode%50==0:
        print("Episode {} : Reward {} ".format(episode,G));
