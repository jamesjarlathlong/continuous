import pglearner
import gym
from gym.envs.registration import registry, register, make, spec
import os
import string
import random
import simple_solar_env
import solar_sensor_env
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
if __name__=='__main__':
    modeldir = sys.argv[1] if sys.argv[1] else ''.join(random.choice(string.ascii_lowercase) for _ in range(5))
    loadmodel = 'tmp/'+modeldir# if sys.argv[1] else None
    phase = sys.argv[2]
    recordname = '_'.join([modeldir,phase])
    print('experiment id:{}'.format(recordname))
    solarrecord = simple_solar_env.emulate_solar_ts(365)
    #solarfname = 'training_12'
    #solarrecord = solar_sensor_env.get_generated_power(solarfname)
    register(
    id='SolarSensor-v0',
    entry_point='solar_sensor_env:SolarSensorEnv',
    kwargs = {'max_batt':100,'num_sensors':4,'deltat':3, 'solarpowerrecord':solarrecord, 'recordname':recordname}
    )
    env = gym.make('SolarSensor-v0')
    pgagent = pglearner.PgLearner(env, learning_rate = 1e-4, n_episodes=1000,gamma=0.99,
                              modeldir=loadmodel, batch=1,max_env_steps=365*8)
    pgagent.run()
    #agent.model.save('tmp/{}'.format(modeldir))
