import ddqlearnereno as ddqlearner
import gym
from gym.envs.registration import registry, register, make, spec
import os
import string
import random
import solar_sensor_env
import simple_agent
from keras.models import load_model
import random_graph
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
def get_month(solarrecord, startmonth):
    startidx = startmonth*30*48
    finishidx = (startmonth+1)*30*48
    return solarrecord[startidx:finishidx]
if __name__=='__main__':
    modeldir = sys.argv[1] if sys.argv[1] else ''.join(random.choice(string.ascii_lowercase) for _ in range(5))
    loadmodel = 'tmp/'+modeldir if sys.argv[1] else None
    phase = sys.argv[2]
    learning_rate = float(sys.argv[3])
    layer_width = int(sys.argv[4])
    num_sensors = int(sys.argv[5])
    recordname = '_'.join([modeldir,phase])
    solarfname = 'training_12'
    solarrecord = solar_sensor_env.get_generated_power(solarfname)
    monthrecord = get_month(solarrecord,8)
    register(
    id='EnoSensorEnv-v0',
    entry_point='eno_sensor_env:EnoSensorEnv',
    kwargs = {'max_batt':10,'num_sensors':num_sensors, 'deltat':3,
              'solarpowerrecord':monthrecord, 'recordname':recordname,
              'coordinate_generator':random_graph.generate_sorted_grid_coords
              ,'full_log':False}
    )
    env = gym.make('EnoSensorEnv-v0')
    agent = ddqlearner.DDQNAgent(env,n_episodes = 5000, max_env_steps=300*8, modeldir=loadmodel,decay_rate = 0.99999975, learning_rate = learning_rate, layer_width=layer_width,checkpoint_name=modeldir)
    #agent = ddqlearner.DDQNAgent(env,n_episodes = 2000, max_env_steps=300*8, modeldir=loadmodel,decay_rate = 0.01, learning_rate = learning_rate, layer_width=layer_width)

    agent.run()
    agent.model.save('tmp/{}'.format(modeldir))
    #naiveagent = simple_agent.SimpleTwoOptionAgent(env, n_episodes = 5, max_env_steps = 300*8, num_on=6)
    #naiveagent.run()
