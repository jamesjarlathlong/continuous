""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import pickle
import gym
import itertools
from pandas.io.json import json_normalize
# hyperparametersd

def initialise_model(resumedir=None):
  # model initialization
    H = 64
    D = 4 # input dimensionality: 80x80 grid
    O = 4#int(D/2)
    if resumedir:
        model = pickle.load(open(resumedir, 'rb'))
    else:
        model = {}
        model['W1'] = np.random.randn(D,H) / np.sqrt(D) # "Xavier" initialization
        model['W2'] = np.random.randn(H,O) / np.sqrt(H)
    return model

def sigmoid(x): 
  return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]
def softmax(x):
  if(len(x.shape)==1):
    x = x[np.newaxis,...]
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  return probs
def name_tuples(tpl):
    return {str(idx):el for idx, el in enumerate(tpl[0:2])}
def describe_state(state):
    return {k:name_tuples(v) for k,v in state.items()}
def statelist_to_df(statelist):
    return json_normalize([describe_state(state) for state in statelist])

def discount_rewards(gamma, r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def policy_forward(model, x):
    #if(len(x.shape)==1):
    #    x = x[np.newaxis,...]
    h =x.dot(model['W1'])#h = np.dot(model['W1'], x)
    h[h<0] = 0 # ReLU nonlinearity
    logp = h.dot(model['W2'])
    #print('logp: ',logp)
    p = softmax(logp)
    #print('p: ',p)
    return p, h # return probability of taking action 2, and hidden state

def policy_backward(model, episode_h, episode_dlogp, episode_states):
  """ backward pass. (eph is array of intermediate hidden states) """
  dW2 = episode_h.T.dot(episode_dlogp)# np.dot(episode_h.T, episode_dlogp)#.ravel()
  dh = episode_dlogp.dot(model['W2'].T)#np.outer(episode_dlogp, model['W2'])
  dh[episode_h <= 0] = 0 # backpro prelu
  dW1 = episode_states.T.dot(dh)#np.dot(dh.T, episode_states)
  return {'W1':dW1, 'W2':dW2}

def flatten_state(state):
    return np.asarray(statelist_to_df([state]).iloc[0].tolist())
def get_action(aprob):
    u = np.random.uniform()
    aprob_cum = np.cumsum(aprob)
    a = np.where(u <= aprob_cum)[0][0]
    #print('probs: ', aprob, a)
    return a
    #return 0 if np.random.uniform()<aprob else 1
class PgExploit:
    def __init__(self, env,n_episodes, max_env_steps, modeldir):
        self.env = env
        if max_env_steps is not None: self.env._max_episode_steps = max_env_steps
        self.action_lookup = list(itertools.product(*(range(space.n) for space in env.action_space.spaces)))
        self.n_episodes = n_episodes
        self.modeldir = modeldir
    def run(self, render = True):
        clf = initialise_model(resumedir=self.modeldir)
        observation = self.env.reset()
        reward_sum = 0
        done=False
        for e in range(self.n_episodes):
            i=0
            while not done and i<self.env._max_episode_steps:
                if render: self.env.render()
                x = flatten_state(observation)
                aprob, h = policy_forward(clf,x)
                action = get_action(aprob)
                observation, reward, done, info = self.env.step(self.action_lookup[action])
                reward_sum += reward
                i+=1
            print('resetting env. episode reward total was {}'.format(reward_sum))
            observation = self.env.reset()
            reward_sum = 0
        return e
class PgBaseline():
    def __init__(self, env,n_episodes, max_env_steps):
        self.env = env
        if max_env_steps is not None: self.env._max_episode_steps = max_env_steps
        self.action_lookup = list(itertools.product(*(range(space.n) for space in env.action_space.spaces)))
        self.n_episodes = n_episodes
    def run(self, render = True):
        clf = initialise_model()
        observation = self.env.reset()
        reward_sum = 0
        done=False
        for e in range(self.n_episodes):
            i=0
            while not done and i<self.env._max_episode_steps:
                if render: self.env.render()
                x = flatten_state(observation)
                aprob, h = policy_forward(clf,x)
                action = get_action(aprob)
                observation, reward, done, info = self.env.step(self.action_lookup[action])
                reward_sum += reward
                i+=1
            print('resetting env. episode reward total was {}'.format(reward_sum))
            observation = self.env.reset()
            reward_sum = 0
        return e
class PgLearner():
    def __init__(self,env, learning_rate,n_episodes, gamma,modeldir, decay_rate=0.95, batch=1,max_env_steps=None):
        self.env = env
        self.modeldir = modeldir
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
                #softmax loss gradient
                dlogsoftmax = aprob.copy()
                dlogsoftmax[0,action] -=1
                dlogps.append(dlogsoftmax)
                #dlogps.append(action-aprob)
                # step the environment and get new measurements
                observation, reward, done, info = self.env.step(self.action_lookup[action])
                #print('previous state {}, action: {} , {},new state {} {}, reward {} '.format( x, action, self.action_lookup[action], observation, flatten_state(observation), reward))
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
                print('UPDATING')
                for k,v in clf.items():
                    g = grad_buffer[k]
                    rmsprop_cache[k] = self.decay_rate * rmsprop_cache[k] + (1 - self.decay_rate) * g**2
                    clf[k] -= self.learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                    grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer
                # boring book-keeping
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
            if e % 100 == 0: pickle.dump(clf, open(self.modeldir, 'wb'))
            observation = self.env.reset()
            reward_sum = 0
        return e
