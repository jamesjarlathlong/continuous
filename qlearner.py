# Inspired by https://medium.com/@tuzzer/cart-pole-balancing-with-q-learning-b54c6068d947

import gym
import numpy as np
import math
from collections import deque
import json
def get_maxq(Q, state):
    pass
def get_maxq_action(Q, state):
    pass
def get_Q(Q, state, action):
    pass
def stringify(state):
    return json.dumps(state)
class QLearner():
    def __init__(self,env, n_episodes=100, min_alpha=0.1,
                 min_epsilon=0.1, gamma=1.0, ada_divisor=25, max_env_steps=None,
                 quiet=False):
        self.n_episodes = n_episodes # training episodes 
        self.min_alpha = min_alpha # learning rate
        self.min_epsilon = min_epsilon # exploration rate
        self.gamma = gamma # discount factor
        self.ada_divisor = ada_divisor # only for development purposes
        self.quiet = quiet

        self.env = env
        if max_env_steps is not None: self.env._max_episode_steps = max_env_steps
        self.Q = np.zeros((3,11,) + (2,3))


    def choose_action(self, state, epsilon):
        return self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(self.Q[state])

    def update_q(self, state_old, action, reward, state_new, alpha):
        self.Q[state_old][action] += alpha * (reward + self.gamma * np.max(self.Q[state_new]) - self.Q[state_old][action])

    def get_epsilon(self, t):
        return max(self.min_epsilon, min(1, 1.0 - math.log10((t + 1) / self.ada_divisor)))

    def get_alpha(self, t):
        return max(self.min_alpha, min(1.0, 1.0 - math.log10((t + 1) / self.ada_divisor)))

    def run(self):
        scores = deque(maxlen=100)

        for e in range(self.n_episodes):
            print('#######New episode#############')
            current_state = self.env.reset()
            print('current state: {}'.format(current_state))
            alpha = self.get_alpha(e)
            epsilon = self.get_epsilon(e)
            done = False
            i = 0

            while not done:
                # self.env.render()
                action = self.choose_action(current_state, epsilon)
                obs, reward, done, _ = self.env.step(action)
                print('action:{},state:{} reward:{}'.format(action, obs, reward))
                new_state = obs
                self.update_q(current_state, action, reward, new_state, alpha)
                current_state = new_state
                i += 1

            scores.append(i)
            mean_score = np.mean(scores)
            if e % 100 == 0 and not self.quiet:
                print('[Episode {}] - Mean survival time over last 100 episodes was {} ticks.'.format(e, mean_score))

        if not self.quiet: print('Did not solve after {} episodes 😞'.format(e))
        return e

if __name__ == "__main__":
    solver = QLearner()
    solver.run()
# gym.upload('tmp/cartpole-1', api_key='')