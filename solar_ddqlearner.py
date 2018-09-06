import ddqlearner
import gym
from gym.envs.registration import registry, register, make, spec
import os
import string
import random
import simple_solar_env
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
if __name__=='__main__':
    modeldir = sys.argv[1] if sys.argv[1] else ''.join(random.choice(string.ascii_lowercase) for _ in range(5))
    phase = sys.argv[2]
    recordname = '_'.join([modeldir,phase])
    print('experiment id:{}'.format(recordname))
    solarrecord = simple_solar_env.emulate_solar_ts(365)
    register(
    id='SolarSensor-v0',
    entry_point='solar_sensor_env:SolarSensorEnv',
    kwargs = {'max_batt':100,'num_sensors':1, 'solarpowerrecord':solarrecord, 'recordname':recordname}
    )
    env = gym.make('SolarSensor-v0')
    agent = ddqlearner.DDQNAgent(env,n_episodes = 1000, max_env_steps=28*48)
    agent.run()
    agent.model.save('tmp/{}'.format(modeldir))
