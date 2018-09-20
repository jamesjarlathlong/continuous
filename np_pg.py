""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import pickle
import gym
import itertools
from pandas.io.json import json_normalize
# hyperparametersd

def initialise_model(resume=False):
  # model initialization
    H = 200
    D = 2 # input dimensionality: 80x80 grid
    if resume:
        model = pickle.load(open('save.p', 'rb'))
    else:
        model = {}
        model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
        model['W2'] = np.random.randn(H) / np.sqrt(H)
    return model

def sigmoid(x): 
  return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

def name_tuples(tpl):
    return {str(idx):el for idx, el in enumerate(tpl)}
def describe_state(state):
    return {k:name_tuples(v) for k,v in state.items()}
def statelist_to_df(statelist):
    return json_normalize([describe_state(state) for state in statelist])

def discount_rewards(gamma, r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(range(0, r.size)):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r

def policy_forward(model, x):
  h = np.dot(model['W1'], x)
  h[h<0] = 0 # ReLU nonlinearity
  logp = np.dot(model['W2'], h)
  p = sigmoid(logp)
  return p, h # return probability of taking action 2, and hidden state

def policy_backward(model, episode_h, episode_dlogp, episode_states):
  """ backward pass. (eph is array of intermediate hidden states) """
  dW2 = np.dot(episode_h.T, episode_dlogp).ravel()
  dh = np.outer(episode_dlogp, model['W2'])
  dh[episode_h <= 0] = 0 # backpro prelu
  dW1 = np.dot(dh.T, episode_states)
  return {'W1':dW1, 'W2':dW2}

def flatten_state(state):
    return statelist_to_df([state]).iloc[0].tolist()
def get_action(aprob):
    return 0 if np.random.uniform()<aprob else 1

class PgLearner():
    def __init__(self,env, learning_rate,n_episodes, gamma, decay_rate, batch=1,max_env_steps=None):
        self.env = env
        if max_env_steps is not None: self.env._max_episode_steps = max_env_steps
        self.action_lookup = list(itertools.product(*(range(space.n) for space in env.action_space.spaces)))
        print(self.action_lookup)
        self.learning_rate = learning_rate
        self.n_episodes = n_episodes
        self.gamma = gamma
        self.batch_size = batch
        self.decay_rate = decay_rate
    def run(self,render=True):
        clf = initialise_model()
        grad_buffer = { k : np.zeros_like(v) for k,v in clf.items() } # update buffers that add up gradients over a batch
        rmsprop_cache = { k : np.zeros_like(v) for k,v in clf.items() } # rmsprop memory
        observation = self.env.reset()
        states, hiddens, dlogps, rewards = [],[],[],[]
        running_reward = None
        reward_sum = 0
        episode_number = 0
        done=False
        
        for e in range(self.n_episodes):
            print('#######New episode#############')
            i=0
            while not done and i<self.env._max_episode_steps:
                if render: self.env.render()
                x = flatten_state(observation)
                aprob, h = policy_forward(clf, x)
                action = get_action(aprob)
                # record various intermediates (needed later for backprop)
                states.append(x) # observation
                hiddens.append(h)
                dlogps.append(action-aprob)
                # step the environment and get new measurements
                observation, reward, done, info = self.env.step(self.action_lookup[action])
                reward_sum += reward
                rewards.append(reward) # record reward (has to be done after we call step() to get reward for previous action)
                i+=1
            # stack together all inputs, hidden states, action gradients, and rewards for this episode
            stacked_states = np.vstack(states)
            stacked_hidden = np.vstack(hiddens)
            stacked_logps = np.vstack(dlogps)
            stacked_rewards = np.vstack(rewards)
            states, rewards, hiddens, dlogps = [],[],[],[]
            # compute the discounted reward backwards through time
            discounted_rewards = discount_rewards(self.gamma, stacked_rewards)
            # standardize the rewards to be unit normal (helps control the gradient estimator variance)
            discounted_rewards = discounted_rewards - np.mean(discounted_rewards)
            discounted_rewards = discounted_rewards / np.std(discounted_rewards)
            stacked_logps *= discounted_rewards # modulate the gradient with advantage (PG magic happens right here.) 
            grad = policy_backward(clf, stacked_hidden, stacked_logps, stacked_states)
            for k in clf: grad_buffer[k]+=grad[k]           
            # perform rmsprop parameter update every batch_size episodes
            if e % self.batch_size == 0:
                for k,v in clf.items():
                    g = grad_buffer[k]
                    rmsprop_cache[k] = self.decay_rate * rmsprop_cache[k] + (1 - self.decay_rate) * g**2
                    clf[k] += self.learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                    grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer
                # boring book-keeping
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
            if e % 100 == 0: pickle.dump(clf, open('save.p', 'wb'))
            observation = self.env.reset()
            reward_sum = 0
        return e
