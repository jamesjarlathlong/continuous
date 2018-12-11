import functools
import scipy as sp
import numpy as np
import seaborn as sns
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
def form_spectrum(psd_fun,size):
    amplitude = np.zeros(size)
    for i,kx in enumerate(fftIndgen(size)):
        amplitude[i] = psd_fun(kx)
    return amplitude
def gpu_gaussian_proc(psd_fun, size = 1024, scale = 1):
    amplitude = tf.constant(form_spectrum(psd_fun,size), dtype = tf.float32)
    zeros = tf.zeros(size, dtype=tf.float32)
    complex_amplitude =tf.complex(amplitude, zeros)
    random_noise = tf.random_normal(shape=(size,), stddev=scale, dtype=tf.float32)
    complex_noise = tf.complex(random_noise, zeros)
    noise_spectrum = tf.fft(complex_noise)
    convolved = tf.multiply(noise_spectrum, complex_amplitude)
    simulation = tf.ifft(convolved)
    with tf.Session() as sess:
        result = sess.run(simulation)
        return result
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

def gpu_gaussian_random_field(size = 32, scale=1, length=48*28):
    shape = (size,size, length)
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

def gaussian_random_field( size = 32, scale=1):
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

def euclidean(p1,p2):
    diff = np.subtract(p1,p2)
    dist = np.linalg.norm(diff)
    return dist
def iso_cov(phi, p1, p2):
    """a function that returns the correlation
    between two oned points"""
    sigma = 0.1
    dist = euclidean(p1,p2)
    return sigma**2 * np.exp(-dist/phi)
def aniso2d_cov(phi,factor, p1, p2):
    sigma = 1
    dist0 = euclidean(p1[0],p2[0])
    dist1 = euclidean(p1[1],p2[1])
    phi2 = phi*factor
    expon = math.sqrt((dist0/phi)**2 + (dist1/phi2)**2)
    return sigma**2 * np.exp(-expon)
def form_cov_matrix(list_of_points, covfun):
    """given a list of points """
    npoints = len(list_of_points)
    mat = np.zeros((npoints, npoints))
    allcombs = itertools.product(enumerate(list_of_points), enumerate(list_of_points))
    for (i, elementi), (j,elementj) in allcombs:
            mat[i,j] = covfun(elementi, elementj)
    return mat  
def simulate(covmatrix):
    """given a covariance matrix perform the
    cholesky decomposition to find lower triangular L
    such that L*L^H = covmatrix, then multiply L by a 
    vector of zero mean unit variance random normal
    numbers to return a realisation of the random process"""
    n,n = np.shape(covmatrix)
    print(n,n)
    L = np.linalg.cholesky(covmatrix)#nxn triangular matrix
    rands = np.random.normal(size = (n,1))
    sim = np.dot(L , rands)
    return sim
def twod_grid(n, size=1):
    """given a size, generate a list of pairs of coordinates
    corresponding with grid points on a sizexsize grid"""
    xs = ys = np.arange(start=0,stop=int(n*size), step=size)
    return list(itertools.product(xs, ys))
def to_mat(original,twod):
    df  = pd.DataFrame(original, columns=['x2','x1'])
    df['values'] = twod
    mat = df.pivot('x2','x1', 'values')
    return mat
def plot(original,twod, ax):
    mat = to_mat(original, twod)
    ax = sns.heatmap(mat,ax=ax,cbar_kws = dict(use_gridspec=False,location="top"))
    return ax
