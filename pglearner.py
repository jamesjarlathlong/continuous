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
import fastpredict
from gym.envs.registration import registry, register, make, spec
import time
import simple_agent
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed
def discount_rewards(gamma, r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def name_tuples(tpl):
    return {str(idx):el for idx, el in enumerate(tpl)}
def describe_state(state):
    return {k:name_tuples(v) for k,v in state.items()}
def statelist_to_df(statelist):
    return json_normalize([describe_state(state) for state in statelist])
def train_input_fn(features, labels, advantages, batch_size = 20):
    featuredf = statelist_to_df(features)
    featuredf['weight'] = [i for i in advantages]
    dataset = tf.data.Dataset.from_tensor_slices((dict(featuredf), labels))
    dataset = dataset.batch(batch_size)
    # Return the dataset.
    iterator = dataset.make_one_shot_iterator()
    d,l = iterator.get_next()
    return d,l

def predict_input_fn(features):
    featuredf = statelist_to_df(features)
    dataset = tf.data.Dataset.from_tensor_slices(dict(featuredf))
    dataset = dataset.batch(1)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()

def define_model(env, action_lookup, modeldir, learning_rate=1e-4):
    dummy_state = env.reset()
    num_actions = len(action_lookup)
    features = statelist_to_df([dummy_state])
    columns_feat = [tf.feature_column.numeric_column(key=i) for i in features.columns]
    #print(columns_feat)
    weight_column = tf.feature_column.numeric_column('weight')
    # Build 2 hidden layer DNN with 10, 10 units respectively.
    print('defining model: ,',modeldir)
    classifier = tf.estimator.DNNClassifier(
        feature_columns=columns_feat,
        weight_column= weight_column,
        # The model must choose between 3 actions
        n_classes=num_actions,
        model_dir = modeldir,
        #Two hidden layers of 100 nodes each.
        hidden_units=[32,32],
        optimizer=tf.train.AdamOptimizer(
          learning_rate=learning_rate,
        ))
    states, actions, batch_rewards = random_burnin(env, action_lookup)
    classifier = update_model(classifier, states, actions, batch_rewards)
    #classifier = update_model(classifier, [dummy_state],[0],[1])
    return classifier
def random_burnin(env, action_lookup):
    guided_agent = simple_agent.SimpleNetworkAgent(env, n_episodes = 10, max_env_steps = 365*8)
    num_choices = len(action_lookup)
    observation = env.reset()
    states, actions, batch_rewards= [],[],[]
    rewards = np.empty(0).reshape(0,1)
    i=0
    done = False
    while not done and i<env._max_episode_steps:
        #action = np.random.randint(0,num_choices)
        action = action_lookup.index(guided_agent.act(observation))
        states.append(observation)
        actions.append(action)
        observation, reward, done, info = env.step(action_lookup[action])
        rewards = np.append(reward, rewards)
        i+=1
    discounted_rewards = discount_rewards(0.99, rewards)
    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    discounted_rewards-= np.mean(discounted_rewards)
    discounted_rewards /= np.std(discounted_rewards)
    batch_rewards = batch_rewards+list(discounted_rewards)
    return states, actions, batch_rewards


def update_model(clf, states, actions, rewards):
    clf.train(input_fn=lambda:train_input_fn(states, actions, rewards))
    return clf
<<<<<<< HEAD
def guru(clf, predx):
    return clf.predict(predx)
=======
>>>>>>> 2d435515992a6ae6932f5e018f3497e07a154305
def get_action(clf, state):
    flatstate = statelist_to_df([state]).iloc[0].tolist()
    predict_x = tuple([[x] for x in flatstate])
    pred = guru(clf, predict_x)
    probabilities = [p.get('probabilities') for p in pred][0]
    action_labels = [idx for idx, el in enumerate(probabilities)]
    choice= np.random.choice(action_labels, p=probabilities)
    #print('probs: ', state, probabilities, choice)
    return choice

def generator_evaluation_fn(featurenames,generator):
    """ Input function for numeric feature columns using the generator. """
    def _inner_input_fn():
        # set datatypes according to your data.
        datatypes = tuple(len(featurenames) * [tf.float32])
        dataset = tf.data.Dataset().from_generator(generator, output_types=datatypes).batch(1)
        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()
        # create a feature dictionary.
        feature_dict = dict(zip(featurenames, features))
        return feature_dict
    return _inner_input_fn
def get_feature_names(env):
    dummy_state = env.observation_space.sample()
    features = statelist_to_df([dummy_state])
    return list(features.columns)
class PgLearner():
    def __init__(self,env, learning_rate,n_episodes, gamma,modeldir, batch=1,max_env_steps=None):
        self.env = env
        if max_env_steps is not None: self.env._max_episode_steps = max_env_steps
        self.action_lookup = list(itertools.product(*(range(space.n) for space in env.action_space.spaces)))
        #print(self.action_lookup)
        self.learning_rate = learning_rate
        self.n_episodes = n_episodes
        self.gamma = gamma
        self.batch_size = batch
        self.modeldir = modeldir
    def run(self,render=True):
        clf = define_model(self.env, self.action_lookup, self.modeldir, learning_rate=self.learning_rate)
        featurenames = get_feature_names(self.env)
        #print('f: ',featurenames)
        genfn = functools.partial(generator_evaluation_fn, featurenames)
        fastclf = fastpredict.FastPredict(clf,genfn)
        states,actions = [],[]
        rewards = np.empty(0).reshape(0,1)
        batch_rewards = []
        running_reward = None
        reward_sum = 0
        episode_number = 0

        for e in range(self.n_episodes):
            print('#######New episode## {}'.format(e))
            done=False
            observation = self.env.reset()
            reward_sum = 0
            i=0
            while not done and i<self.env._max_episode_steps:
                if render: self.env.render()
                action = get_action(fastclf, observation)
                # record various intermediates (needed later for backprop)
                states.append(observation) # observation
                actions.append(action)
                #print('action: {}'.format(action))
                # step the environment and get new measurements
                observation, reward, done, info = self.env.step(self.action_lookup[action])
                reward_sum += reward
                rewards = np.append(rewards, reward) # record reward (has to be done after we call step() to get reward for previous action)
                i+=1
            # stack together all inputs, hidden states, action gradients, and rewards for this episode
            # compute the discounted reward backwards through time
            discounted_rewards = discount_rewards(self.gamma, rewards)
            # standardize the rewards to be unit normal (helps control the gradient estimator variance)
            discounted_rewards-= np.mean(discounted_rewards)
            discounted_rewards /= np.std(discounted_rewards)
            batch_rewards = batch_rewards+list(discounted_rewards)
            rewards = np.empty(0).reshape(0,1)
            # perform rmsprop parameter update every batch_size episodes
            if e % self.batch_size == 0:
                print('#######Updating#############')
                clf = update_model(clf, states, actions, batch_rewards)
                fastclf = fastpredict.FastPredict(clf,genfn)
                states, actions, batch_rewards = [],[],[] # reset array memory
                # boring book-keeping
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
        return e

if __name__=='__main__':
    import os
    #os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    register(
    id='MultiSensor-v0',
    entry_point='multi_sensor_env:MultiSensorEnv',
    kwargs = {'num_sensors':4}
    )
    env = gym.make('MultiSensor-v0')
    pgagent = PgLearner(env, learning_rate = 1e-4, n_episodes=10000,gamma=0.99,
                              modeldir='tmp/gpurepeat', batch=10,max_env_steps=200)
    pgagent.run()

