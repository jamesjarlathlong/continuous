# Original code from https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
# Use it to solve CartPole-v0
import numpy as np
import gym

# hyperparameters
H = 10 # number of hidden layer neurons
batch_size = 5 # every how many episodes to do a param update?
learning_rate = 1e-2
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
render = False

# model initialization
D = 4 # input dimensionality

model = {}
model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
model['W2'] = np.random.randn(H) / np.sqrt(H)

grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } # rmsprop memory

def sigmoid(x): 
	return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

def discount_rewards(r):
	""" take 1D float array of rewards and compute discounted reward """
	discounted_r = np.zeros_like(r)
	running_add = 0
	for t in reversed(range(0, r.size)):
		running_add = running_add * gamma + r[t]
		discounted_r[t] = running_add
	return discounted_r

def policy_forward(x):
	h = np.dot(model['W1'], x)
	h[h<0] = 0 # ReLU nonlinearity
	logp = np.dot(model['W2'], h)
	p = sigmoid(logp)
	return p, h # return probability of taking action 1, and hidden state

def policy_backward(eph, epdlogp):
	""" backward pass. (eph is array of intermediate hidden states) """
	dW2 = np.dot(eph.T, epdlogp).ravel()
	dh = np.outer(epdlogp, model['W2'])
	dh[eph <= 0] = 0 # backpro prelu
	dW1 = np.dot(dh.T, epx)
	return {'W1':dW1, 'W2':dW2}

env = gym.make("CartPole-v0")

xs,hs,dlogps,drs = [],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0

for episode_number in range(1000):

	observation = env.reset()

	while True:
		x = observation

		# forward the policy network and sample an action from the returned probability
		aprob, h = policy_forward(x)
		action = 1 if np.random.uniform() < aprob else 0 # roll the dice!

		# record various intermediates (needed later for backprop)
		xs.append(x) # observation
		hs.append(h) # hidden state
		y = 1 if action == 1 else 0 # a "fake label"

		dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)
		
		# step the environment and get new measurements
		observation, reward, done, info = env.step(action)
		reward_sum += reward

		drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

		if done or reward_sum >= 200: # an episode finished
			# stack together all inputs, hidden states, action gradients, and rewards for this episode
			epx = np.vstack(xs)
			eph = np.vstack(hs)
			epdlogp = np.vstack(dlogps)
			epr = np.vstack(drs)
			xs,hs,dlogps,drs = [],[],[],[] # reset array memory

			# compute the discounted reward backwards through time
			discounted_epr = discount_rewards(epr)
			# standardize the rewards to be unit normal (helps control the gradient estimator variance)
			discounted_epr -= np.mean(discounted_epr)
			discounted_epr /= np.std(discounted_epr)

			epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)

			# print epdlogp
			grad = policy_backward(eph, epdlogp)
			for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch

			# perform rmsprop parameter update every batch_size episodes
			if episode_number % batch_size == 0:
				for k,v in model.items():
					g = grad_buffer[k] # gradient
					rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
					model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
					grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

			reward_sum = 0
			observation = env.reset() # reset env

			break

#test algorithm
test_number = 100
test_reward = 0
for i in range(test_number):
	iter = 0
	reward_sum = 0
	observation = env.reset() # Obtain an initial observation of the environment
	while True: 
		# Run the policy network and get an action to take. 
		aprob, _ = policy_forward(observation)
		action = 1 if np.random.uniform() < aprob else 0 # roll the dice!

		# step the environment and get new measurements
		observation, reward, done, info = env.step(action)
		reward_sum += reward
		iter += 1
		if done or iter >= 300:
			
			test_reward += reward_sum
			iter = 0
			reward_sum = 0
			break

print("test average reward is {}".format(test_reward / test_number))