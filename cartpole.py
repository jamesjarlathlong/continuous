""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import pickle
import gym
import itertools
from pandas.io.json import json_normalize
# hyperparametersd

def initialise_model(resumedir=None):
  # model initialization
    H = 10
    D = 4#80*80 # input dimensionality: 80x80 grid
    O = 1
    if resumedir:
        model = pickle.load(open(resumedir, 'rb'))
    else:
        model = {}
        model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
        model['W2'] = np.random.randn(H) / np.sqrt(H)
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
    #if(len(x.shape)==1):
    #    x = x[np.newaxis,...]
    #h =x.dot(model['W1'])#
    h = np.dot(model['W1'], x)
    h[h<0] = 0 # ReLU nonlinearity
    #logp = h.dot(model['W2'])
    logp = np.dot(model['W2'], h)
    #print('logp: ',logp)
    #p = softmax(logp)
    p=sigmoid(logp)
    #print('p: ',p)
    return p, h # return probability of taking action 2, and hidden state

def policy_backward(model, episode_h, episode_dlogp, episode_states):
  """ backward pass. (eph is array of intermediate hidden states) """
  #dW2 = episode_h.T.dot(episode_dlogp).ravel()#
  dW2=np.dot(episode_h.T, episode_dlogp).ravel()
  #dh = episode_dlogp.dot(model['W2'].T)#
  dh=np.outer(episode_dlogp, model['W2'])
  dh[episode_h <= 0] = 0 # backpro prelu
  #dW1 = episode_states.T.dot(dh)#
  dW1=np.dot(dh.T, episode_states)
  return {'W1':dW1, 'W2':dW2}

def flatten_state(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()
def get_action(aprob):
    #u = np.random.uniform()
    #aprob_cum = np.cumsum(aprob)
    #a = np.where(u <= aprob_cum)[0][0]
    #print('probs: ', aprob, a)
    #return a
    return 1 if np.random.uniform()<aprob else 0

class PgLearner():
    def __init__(self,env, learning_rate,n_episodes, gamma,modeldir, decay_rate=0.99, batch=1,max_env_steps=None):
        self.env = env
        self.modeldir = modeldir
        if max_env_steps is not None: self.env._max_episode_steps = max_env_steps
        #self.action_lookup = list(itertools.product(*(range(space.n) for space in env.action_space.spaces)))
        #print(self.action_lookup)
        self.learning_rate = learning_rate
        self.n_episodes = n_episodes
        self.gamma = gamma
        self.batch_size = batch
        self.decay_rate = decay_rate
    def run(self,render=False):
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
                x = observation
                #x = cur_x - prev_x if prev_x is not None else np.zeros(80*80)
                #prev_x = cur_x
                #print('x',x)
                aprob, h = policy_forward(clf, x)
                #print('aprob,',aprob)
                action = get_action(aprob)
                y = 1 if action==1 else 0
                #print('action: ', action)
                # record various intermediates (needed later for backprop)
                states.append(x) # observation
                hiddens.append(h)
                #softmax loss gradient
                #dlogsoftmax = aprob.copy()
                #dlogsoftmax[0,action] -=1
                #dlogps.append(dlogsoftmax)
                dlogps.append(y-aprob)
                # step the environment and get new measurements
                observation, reward, done, info = self.env.step(action)
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
            discounted_rewards -= np.mean(discounted_rewards)
            discounted_rewards /= np.std(discounted_rewards)
            stacked_logps *= discounted_rewards # modulate the gradient with advantage (PG magic happens right here.) 
            grad = policy_backward(clf, stacked_hidden, stacked_logps, stacked_states)
            for k in clf: grad_buffer[k]+=grad[k]           
            # perform rmsprop parameter update every batch_size mode
            if e % self.batch_size == 0:
                for k,v in clf.items():
                    g = grad_buffer[k]
                    rmsprop_cache[k] = self.decay_rate * rmsprop_cache[k] + (1 - self.decay_rate) * g**2
                    clf[k] += self.learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                    grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer
                # boring book-keeping
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
            if e % 100 == 0: pickle.dump(clf, open(self.modeldir, 'wb'))
            observation = self.env.reset()
            reward_sum = 0
            done=False
        return e
if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    pgagent = PgLearner(env,learning_rate = 1e-2, modeldir='tmp/pong6', n_episodes=10000,gamma=0.99, batch=5)
    pgagent.run()
