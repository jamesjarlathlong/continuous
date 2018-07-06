# Inspired by https://medium.com/@tuzzer/cart-pole-balancing-with-q-learning-b54c6068d947

import gym
import numpy as np
import math
from collections import deque
import collections
import copy
import json
import functools
from gym.envs.registration import registry, register, make, spec

def get_maxq(Q, state):
    state_q = Q.get(stringify(state),{})
    vals = state_q.values()
    return max(vals) if vals else 0
def get_maxq_action(Q, state, env):
    state_q = Q.get(stringify(state))
    return json.loads(max(state_q, key=state_q.get)) if state_q else env.action_space.sample()
def stringaction(action):
    return json.dumps(action)
def get_Q(Q, state, action):
    return Q.get(stringify(state),{}).get(stringaction(action), 0)
def stringify(state):
    s = json.dumps(sorted(state.items()))
    return s
def update_q(Q_old, state_old, action, reward, state_new, alpha, gamma):
    Q = Q_old#copy.deepcopy(Q_old)
    old_q = get_Q(Q, state_old, action)
    update = alpha*(reward + gamma * get_maxq(Q, state_new) - get_Q(Q, state_old, action))
    #Q[state_old][action] += alpha * (reward + self.gamma * np.max(self.Q[state_new]) - self.Q[state_old][action])
    Q[stringify(state_old)][stringaction(action)] = old_q+update
    return Q
class QLearner():
    def __init__(self,env, n_episodes=100, min_alpha=0.1,
                 min_epsilon=0.1, gamma=1.0, ada_divisor=25, max_env_steps=None,
                 quiet=False, model_dir=None):
        self.n_episodes = n_episodes # training episodes 
        self.min_alpha = min_alpha # learning rate
        self.min_epsilon = min_epsilon # exploration rate
        self.gamma = gamma # discount factor
        self.ada_divisor = ada_divisor # only for development purposes
        self.quiet = quiet

        self.env = env
        if max_env_steps is not None: self.env._max_episode_steps = max_env_steps
        if model_dir:
            self.model_dir = model_dir
            try:
                with open(model_dir, 'r') as f:
                    d = json.load(f)
                    print('loaded')
                    self.Q = collections.defaultdict(dict, d)
            except Exception as e:
                print(e)
                with open(model_dir, 'w') as f:
                    self.Q = collections.defaultdict(dict)
        else:
            self.Q = collections.defaultdict(dict)


    def choose_action(self, state, epsilon):
        return self.env.action_space.sample() if (np.random.random() <= epsilon) else get_maxq_action(self.Q, state, self.env)

    def update_q(self, state_old, action, reward, state_new, alpha):
        self.Q = update_q(self.Q, state_old, action, reward, state_new, alpha, self.gamma)

    def get_epsilon(self, t):
        return max(self.min_epsilon, min(1, 1.0 - math.log10((t + 1) / self.ada_divisor)))

    def get_alpha(self, t):
        return max(self.min_alpha, min(1.0, 1.0 - math.log10((t + 1) / self.ada_divisor)))

    def run(self):
        scores = deque(maxlen=100)
        rewards = deque(maxlen=100)
        for e in range(self.n_episodes):
            print('#######New episode#############')
            current_state = self.env.reset()
            #print('current state: {}'.format(current_state))
            alpha = self.get_alpha(e)
            epsilon = self.get_epsilon(e)
            done = False
            i = 0
            r =0
            while not done and i<self.env._max_episode_steps:
                self.env.render()
                action = self.choose_action(current_state, epsilon)
                obs, reward, done, _ = self.env.step(action)
                r+=reward
                #print('previous:{}, action:{},new:{} reward:{}'.format(current_state, action, obs, reward))
                new_state = obs
                self.update_q(current_state, action, reward, new_state, alpha)
                current_state = new_state
                i += 1

            scores.append(i)
            rewards.append(r)
            mean_score = np.mean(scores)
            mean_reward = np.mean(rewards)
            if e % 1 == 0 and not self.quiet:
                print('[Episode {}] - Mean survival time over last 10 episodes was {} ticks,{}.'.format(e, mean_score, mean_reward))
                #print('Q: ', self.Q)
        if not self.quiet: print('Did not solve after {} episodes '.format(e))
        if self.model_dir:
            with open(self.model_dir, 'w') as f:
                print('saving:',self.model_dir)
                json.dump(self.Q,f)
        return e

if __name__=='__main__':
    register(
    id='MultiSensor-v0',
    entry_point='multi_sensor_env:MultiSensorEnv',
    kwargs = {'num_sensors':4}
    )
    env = gym.make('MultiSensor-v0')
    qagent = QLearner(env, n_episodes=2000, min_alpha=0.05, min_epsilon=0.05,
                      ada_divisor=200, gamma=0.99,max_env_steps=200, model_dir='tmp/slow')
    qagent.run()
