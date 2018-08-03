import functools
import scipy as sp
import numpy as np
import seaborn
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import pandas as pd
import time
from functools import lru_cache
import tensorflow as tf

def fftIndgen(n):
    a = list(range(0, int(n/2+1)))
    b = list(reversed(range(1, int(n/2))))
    b = [-i for i in b]
    return a + b
def power_spectrum(alpha,k):
    fourier = lambda k: 2*alpha/(alpha**2+(k**2))
    if k == 0:
        return 0
    return fourier(k)
def power_spectrum2d(alpha, kx,ky):
    scale = 50
    fourier = lambda k: 2*alpha/(alpha**2+(k**2))
    if kx == 0 and ky==0:
        return 0
    k = np.sqrt(kx**2+ky**2)
    return fourier(np.sqrt(kx**2+scale*ky**2))
def gaussian_proc(psd_fun, size=1024, scale =1):
    noise = np.fft.fft(np.random.normal(size = size, scale=scale))
    amplitude = np.zeros(size)
    for i,kx in enumerate(fftIndgen(size)):
        amplitude[i] = psd_fun(kx)
    return np.fft.ifft(noise*amplitude)
@lru_cache()
def form_spectral_matrix(size):
    sizex, sizey, sizez = size
    amplitude = np.zeros((sizex,sizey,sizez))
    for i, kx in enumerate(fftIndgen(sizex)):
        for j, ky in enumerate(fftIndgen(sizey)):
            for z, kz in enumerate(fftIndgen(sizez)):
                amplitude[i, j,z] = Pk3(kx, ky, kz)
    return amplitude
Pk = lambda k : k**-10.0
def Pk3(kx, ky,kz):
        if kx == 0 and ky == 0 and kz==0:
            return 0.0
        return np.sqrt(Pk(np.sqrt(kx**2 + ky**2+ (4*kz)**2)))

def gpu_gaussian_random_field(size = 30, scale=1):
    shape = (size,size, 1*24)
    amplitude = tf.constant(form_spectral_matrix(shape),dtype = tf.float32)
    complex_amplitude =tf.complex(amplitude, tf.zeros(shape, dtype=tf.float32))
    random_noise = tf.random_normal(shape=shape, stddev=scale, dtype=tf.float32)
    zeros = tf.zeros(shape,dtype=tf.float32)
    complex_noise = tf.complex(random_noise, zeros)
    noise_spectrum = tf.fft3d(complex_noise)
    convolved = tf.multiply(complex_amplitude, noise_spectrum)
    simulation = tf.ifft3d(convolved)
    with tf.Session() as sess:
        result = sess.run(simulation)
        return result

def gaussian_random_field( size = 30, scale=1):
    noise = np.fft.fftn(np.random.normal(size = (size, size,1*24), scale=scale))
    amplitude = form_spectral_matrix((size,size, 1*24))
    return np.fft.ifftn(noise * amplitude)
def gaussian_field_alt(psd_fun, size=100):
    noise = np.fft.fft2(np.random.normal(size = (size, size)))
    amplitude = np.zeros((size,size))
    for i, kx in enumerate(fftIndgen(size)):
        for j, ky in enumerate(fftIndgen(size)):            
            amplitude[i, j] = psd_fun(kx, ky)
    return np.fft.ifft2(noise * amplitude)