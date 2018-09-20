import numpy as np
import itertools
#850/2344 for simple q learning
def gaussian(x):
    mu = 12
    sig = 2
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
def emulate_sun():
    nighttime = 12*[0.0]
    daytime = [gaussian(0.5*i) for i in range(12,36)]
    return nighttime+daytime+nighttime
def emulate_solar_ts(days):
    return list(itertools.chain(*[emulate_sun() for _ in range(days)]))
