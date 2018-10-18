import ddqlearnermonth as ddqlearner
import gym
from gym.envs.registration import registry, register, make, spec
import os
import string
import random
import simple_solar_env
import solar_sensor_env
import simple_agent
from keras.models import load_model
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
    print('experiment id:{}'.format(recordname))
    #solarrecord = simple_solar_env.emulate_solar_ts(365)
    solarfname = 'training_12'
    solarrecord = solar_sensor_env.get_generated_power(solarfname)
    monthrecord = get_month(solarrecord,6)
    register(
    id='SolarSensor-v0',
    entry_point='solar_sensor_env:SolarSensorEnv',
    kwargs = {'max_batt':10,'num_sensors':num_sensors, 'deltat':3,'solarpowerrecord':monthrecord, 'recordname':recordname}
    )
    env = gym.make('SolarSensor-v0')
    #naiveagent = simple_agent.SimpleNetworkAgent(env, n_episodes = 10, max_env_steps = 30*8)
    #naiveagent.run()
    agent = ddqlearner.DDQNAgent(env,n_episodes = 10000, max_env_steps=30*8, modeldir=loadmodel,decay_rate = 0.9999990, learning_rate = learning_rate, layer_width=layer_width)
    agent.run()
    agent.model.save('tmp/{}'.format(modeldir))
