import qlearner
import gym
from gym.envs.registration import registry, register, make, spec
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
if __name__=='__main__':
    register(
    id='SolarSensor-v0',
    entry_point='solar_sensor_env:SolarSensorEnv',
    kwargs = {'max_batt':100,'num_sensors':2}
    )
    env = gym.make('SolarSensor-v0')
    qagent = qlearner.QLearner(env, n_episodes=10000, min_alpha=0.01, min_epsilon=0.01,
                      ada_divisor=30, gamma=0.95,max_env_steps=28*48)
    qagent.run()
