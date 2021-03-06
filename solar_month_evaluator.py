import ddqlearnercp as ddqlearner
import gym
from gym.envs.registration import registry, register, make, spec
import os
import string
import random
import simple_solar_env
import solar_sensor_env
import simple_agent
from keras.models import load_model
import random_graph
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import json
def get_month(solarrecord, startmonth):
    startidx = startmonth*30*48
    finishidx = (startmonth+1)*30*48
    return solarrecord[startidx:finishidx]
if __name__=='__main__':
    num_sensors = 16
    solarrecord = simple_solar_env.emulate_solar_ts(365)
    solarfname = 'training_12'
    solarrecord = solar_sensor_env.get_generated_power(solarfname)
    all_res = []
    for month in range(12):
        monthrecord = get_month(solarrecord,month)
        name = 'TwoOptionSensorEnv-v{}'.format(str(month))
        register(
        id='TwoOptionSensorEnv-v0',
        entry_point='twooption_solar_sensor_env:TwoOptionSensorEnv',
        kwargs = {'max_batt':10,'num_sensors':num_sensors, 'deltat':3,
              'solarpowerrecord':monthrecord, 'recordname':recordname,
              'coordinate_generator':random_graph.generate_sorted_grid_coords}
        )
    env = gym.make('TwoOptionSensorEnv-v0')
        for num_on in range(1,16):
            naiveagent = simple_agent.SimpleNetworkAgent(env, n_episodes = 2, max_env_steps = 300*8, num_on=num_on)
            rewards = naiveagent.run()
            all_res.append({'month':month, 'num_sensors':num_on,'rewards':rewards})
    with open('badresult.json', 'w') as fp:
        json.dump(all_res, fp)
