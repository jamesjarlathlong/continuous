import numpy as np
import gym
from gym.spaces import Discrete, Box
from gym import wrappers, logger
from six.moves import cPickle as pickle
import json, sys, os
from os import path
import argparse
import functools
class DeterministicDiscreteActionLinearPolicy(object):
    def __init__(self, theta):
        """
        dim_ob: dimension of observations
        n_actions: number of actions
        theta: flat vector of parameters
        """
        dim_ob = 3
        n_actions = 2
        print('len theta?{}'.format(len(theta)))
        assert len(theta) == (dim_ob + 1) * n_actions
        self.W = theta[0: dim_ob * n_actions].reshape(dim_ob, n_actions)
        self.b = theta[dim_ob * n_actions : None].reshape(1,n_actions)
    def act(self, ob):
        y = np.array(ob).dot(self.W) + self.b
        a = y.argmax()
        return a
def do_rollout(agent, env, num_steps, render=False):
    total_rew = 0
    ob = env.reset()
    for t in range(num_steps):
        a = agent.act(ob)
        (ob, reward, done, _info) = env.step(a)
        total_rew += reward
        if done: break
    return total_rew, t+1

def noisy_evaluation(env, num_steps, theta):
    agent = DeterministicDiscreteActionLinearPolicy(theta)
    rew, T = do_rollout(agent, env, num_steps)
    return rew
def cem(f, th_mean, batch_size, n_iter, elite_frac, initial_std=1.0):
    """
    Generic implementation of the cross-entropy method for maximizing a black-box function
    f: a function mapping from vector -> scalar
    th_mean: initial mean over input distribution
    batch_size: number of samples of theta to evaluate per batch
    n_iter: number of batches
    elite_frac: each batch, select this fraction of the top-performing samples
    initial_std: initial standard deviation over parameter vectors
    """
    n_elite = int(np.round(batch_size*elite_frac))
    th_std = np.ones_like(th_mean) * initial_std

    for _ in range(n_iter):
        ths = np.array([th_mean + dth for dth in  th_std[None,:]*np.random.randn(batch_size, th_mean.size)])
        ys = np.array([f(th) for th in ths])
        elite_inds = ys.argsort()[::-1][:n_elite]
        elite_ths = ths[elite_inds]
        th_mean = elite_ths.mean(axis=0)
        th_std = elite_ths.std(axis=0)
        yield {'ys' : ys, 'theta_mean' : th_mean, 'y_mean' : ys.mean()}
def train(env):
    noisy_eval = functools.partial(noisy_evaluation, env, 100)
    for data in cem(noisy_eval, np.zeros(8), 100, 10000, 0.2):
        agent = DeterministicDiscreteActionLinearPolicy(data['theta_mean'])
    return agent
