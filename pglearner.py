import gym
import numpy as np
import math
from collections import deque
import collections
import copy
import json
import functools
import itertools
import tensorflow as tf
from pandas.io.json import json_normalize
def discount_rewards(gamma, r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(range(0, r.size)):
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
	return discounted_r
def define_model(num_actions, learning_rate=1e-4):
	features = []
	columns_feat = [tf.feature_column.numeric_column(key=i) for i in features]
	weight_column = tf.feature_column.numeric_column('weight')
    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=columns_feat,
        weight_column= weight_column,
        # The model must choose between 3 actions
        n_classes=num_actions,
        #Two hidden layers of 100 nodes each.
        hidden_units=[128,128],
        optimizer=tf.train.AdamOptimizer(
          learning_rate=learning_rate,
        ))
    return classifier
def name_tuples(tpl):
    return {idx:el for idx, el in enumerate(tpl)}
def describe_state(state):
	return {k:name_tuples(v) for k,v in state.items()}
def statelist_to_df(statelist):
	return json_normalize([describe_state(state) for state in statelist])
def train_input_fn(features, labels, advantages, batch_size = 20):
	featuredf = statelist_to_df(features)
    featuredf['weight'] = advantages
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.batch(batch_size)
    # Return the dataset.
    iterator = dataset.make_one_shot_iterator()
    d,l = iterator.get_next()
    return d,l

def predict_input_fn(state):
	dataset = tf.data.Dataset.from_tensor_slices(state)
    dataset = dataset.batch(1)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()

def update_model(clf, states, actions, rewards):
	clf.train(input_fn=lambda:train_input_fn(states, actions, rewards))
	return clf

def get_action(clf, state):
	pred = clf.predict(input_fn = lambda: predict_input_fn([state]))
	probabilities = [p.get('probabilities') for p in pred][0]
	action_labels = list(range(probabilities))
	return np.random.choice(actions_labels, p=probabilities)
	

class PgLearner():
	def__init__(self,env, learning_rate,n_episodes, gamma, max_env_steps=None):
		self.env = env
		if max_env_steps is not None: self.env._max_episode_steps = max_env_steps
		self.action_lookup = list(itertools.product(*(range(space.n) for space in env.action_space.spaces)))
		self.learning_rate = learning_rate
		self.n_episodes = n_episodes
		self.gamma = gamma
	def run(render=True):
		clf = define_model(len(self.action_lookup), learning_rate=self.learning_rate)
		states,actions, rewards = [],[],[]
		batch_rewards = []
		running_reward = None
		reward_sum = 0
		episode_number = 0
		for e in range(self.n_episodes):
			print('#######New episode#############')
			done=False
			observation = self.env.reset()
			reward_sum = 0
			i=0
			while not done and i<self.env._max_episode_steps:
  				if render: self.env.render()
  				action = get_action(clf, observation)
  				# record various intermediates (needed later for backprop)
  				states.append(observation) # observation
  				actions.append(action)
  				# step the environment and get new measurements
  				observation, reward, done, info = self.env.step(self.action_lookup[action])
  				reward_sum += reward
  				rewards.append(reward) # record reward (has to be done after we call step() to get reward for previous action)
  				i+=1
    		# stack together all inputs, hidden states, action gradients, and rewards for this episode
    		# compute the discounted reward backwards through time
    		discounted_rewards = discount_rewards(self.gamma, rewards)
    		# standardize the rewards to be unit normal (helps control the gradient estimator variance)
    		discounted_rewards-= np.mean(discounted_rewards)
    		discounted_rewards /= np.std(discounted_rewards)
    		batch_rewards = batch_rewards+discounted_rewards
    		rewards = []
    		# perform rmsprop parameter update every batch_size episodes
    		if e % batch_size == 0:
      			clf = update_model(clf, states, rewards, actions)
      			states, actions, batch_rewards = [],[],[] # reset array memory
    			# boring book-keeping
    		running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    		print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
    	return e

