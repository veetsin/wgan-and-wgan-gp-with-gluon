#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 06:17:30 2018

@author: veetsin
"""

import os 
from matplotlib import pyplot as plt 

import mxnet as mx
from mxnet import gluon
from mxnet import ndarray as nd
from mxnet.gluon import nn 
from mxnet import autograd
import numpy as np
import time

number_face = 10000
def try_gpu():
    ctx = mx.gpu()
    try:
        _ = nd.array([1],ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx
ctx = try_gpu()

netG = nn.Sequential()
with netG.name_scope():
    netG.add(nn.Conv2DTranspose(channels=256,kernel_size=2))
    netG.add(nn.BatchNorm())
    netG.add(nn.Activation(activation='relu'))
    #2
    netG.add(nn.Conv2DTranspose(channels=128,kernel_size=3,strides=2,padding=1))
    netG.add(nn.BatchNorm())
    netG.add(nn.Activation(activation='relu'))
    #3
    netG.add(nn.Conv2DTranspose(channels=64,kernel_size=3,strides=2,padding=1))
    netG.add(nn.BatchNorm())
    netG.add(nn.Activation(activation='relu'))
    #5
    netG.add(nn.Conv2DTranspose(channels=32,kernel_size=3,strides=2,padding=1))
    netG.add(nn.BatchNorm())
    netG.add(nn.Activation(activation='relu'))
    #9
    netG.add(nn.Conv2DTranspose(channels=16,kernel_size=3,strides=2,padding=1))
    netG.add(nn.BatchNorm())
    netG.add(nn.Activation(activation='relu'))
    #17  
    netG.add(nn.Conv2DTranspose(channels=1,kernel_size=3,strides=2,padding=1))
    netG.add(nn.BatchNorm())
    netG.add(nn.Activation(activation='tanh'))
    #33

def show_images(images):
        plt.imshow(images.reshape((33, 33)).asnumpy())
        plt.axis('off')
    
def save_image(data, filename):
    data = data.reshape((33, 33)).asnumpy()
    sizes = np.shape(data)     
    fig = plt.figure()
    fig.set_size_inches(1. * sizes[0] / sizes[1], 1, forward = False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(data)
    plt.savefig(filename, dpi = sizes[0], cmap='hot') 
    plt.close()
        
#generate dcgan_face
netG.load_params('240_dcgan',ctx=ctx)
for i in range(number_face):
    latent_z = mx.nd.random_normal(0,1,shape=(1,100,1,1),ctx=ctx)
    fake = netG(latent_z)
    save_image(fake[0],os.path.join('dcgan',str(i)+'.png'))
plt.show()

#generate wgan_face
netG.load_params('240_wgan',ctx=ctx)
for i in range(number_face):
    latent_z = mx.nd.random_normal(0,1,shape=(1,100,1,1),ctx=ctx)
    fake = netG(latent_z)
    save_image(fake[0],os.path.join('wgan',str(i)+'.png'))
plt.show()

#generate dcgan_face
netG.load_params('240_wgan-gp',ctx=ctx)
for i in range(number_face):
    latent_z = mx.nd.random_normal(0,1,shape=(1,100,1,1),ctx=ctx)
    fake = netG(latent_z)
    save_image(fake[0],os.path.join('wgan-gp',str(i)+'.png'))
plt.show()
time2 = time.time()
