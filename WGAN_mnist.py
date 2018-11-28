#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 03:57:20 2018

@author: mc
"""

# -*- coding: utf-8 -*-
import os 
from matplotlib import pyplot as plt 

import mxnet as mx
from mxnet import gluon
from mxnet import ndarray as nd
from mxnet.gluon import nn 
from mxnet import autograd
import numpy as np



epochs = 241
batch_size = 256
latent_z_size = 100

def try_gpu():
    ctx = mx.gpu()
    try:
        _ = nd.array([1],ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx
ctx = try_gpu()

lr = .003
clipping_constant = .03
n_critic = 5


#load data ,transform to ndarray , get the training data 
def transform(data, label):
    data = mx.image.imresize(data,33,33)
    return nd.transpose(data.astype(np.float32), (2,0,1))/127.5-1, label.astype(np.float32)
mnist_train = gluon.data.vision.MNIST(train=True, transform=transform)
train_data = gluon.data.DataLoader(mnist_train, batch_size, shuffle=True)


def show_images(images):
        plt.imshow((images.reshape((33, 33))).asnumpy())
        plt.axis('off')
 
#show 9 images
fig = plt.figure(figsize=(6,6))
for i in range(9):
    data,_ = mnist_train[i]
    plt.subplot(3,3,i+1)
    show_images(data)
plt.show() 

#define the networks
#=============discriminator============
netD = nn.Sequential()
with netD.name_scope():
    netD.add(nn.Conv2D(channels=16,kernel_size=3,strides=2,padding=1))
    netD.add(nn.BatchNorm())
    netD.add(nn.LeakyReLU(.2))
    #17
    netD.add(nn.Conv2D(channels=32,kernel_size=3,strides=2,padding=1))
    netD.add(nn.BatchNorm())
    netD.add(nn.LeakyReLU(.2))
    #9
    netD.add(nn.Conv2D(channels=64,kernel_size=3,strides=2,padding=1))
    netD.add(nn.BatchNorm())
    netD.add(nn.LeakyReLU(.2))
    #5
    netD.add(nn.Conv2D(channels=128,kernel_size=3,strides=2,padding=1))
    netD.add(nn.BatchNorm())
    netD.add(nn.LeakyReLU(.2))
    #3
    netD.add(nn.Conv2D(channels=1,kernel_size=3))
    
#===============generator==================
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


netG.initialize(mx.init.Normal(.02),ctx=ctx)
netD.initialize(mx.init.Normal(.02),ctx=ctx)


#function to clip the paramters
def clip(net,clip_value):
    for param in net.collect_params():
#        param.set_data(mx.nd.clip(param.data(ctx=ctx),-clip_value,clip_value))
        net.collect_params()[str(param)].set_data(net.collect_params()[str(param)].data().clip(-clip_value,clip_value))
    return net
    
for batch in train_data:
        if batch[1].shape[0] != 256:
            break
        data = nd.array(batch[0].asnumpy()).as_in_context(ctx)
        break
netD(data)
netD = clip(netD,clipping_constant)


trainerG = gluon.Trainer(netG.collect_params(),'RMSProp',{'learning_rate':lr})
trainerD = gluon.Trainer(netD.collect_params(),'RMSProp',{'learning_rate':lr})

#training loop
import time
import logging 
#
#real_label = nd.ones((batch_size,),ctx=ctx)
#fake_label = nd.zeros((batch_size,),ctx=ctx)


#custom metric
#def eveluate(pred , label):
#    pred = pred.flatten()
#    label = label.flatten()
#    return ((pred>.5) == label).mean()
#metric_real = mx.metric.CustomMetric(eveluate)
#metric_fake = mx.metric.CustomMetric(eveluate)
logging.basicConfig(level=logging.DEBUG,filename='WGAN_mnist.log',filemode='w',format='[%(levelname)s:%(message)s]')
#his_acc_real = []
#his_acc_fake = []
his_wassD = []
his_errG = []


#funtion to save the netG
path_D = 'his_params_D_WGAN_mnist_20181022003'
path_G = 'his_params_G_WGAN_mnist_20181022003'
if not os.path.exists(path_D):
    os.makedirs(path_D)
if not os.path.exists(path_G):
    os.makedirs(path_G)
def save_params(net,epoch,path):
    file_path = os.path.join(path,str(epoch))
    net.save_params(file_path)

    
for epoch in range(epochs):
    if epoch == int(epoch/4):
        trainerD.set_learning_rate(lr*.5)
        trainerG.set_learning_rate(lr*.5)
    if epoch == int(epoch*3/4):
        trainerD.set_learning_rate(lr*.2)
        trainerG.set_learning_rate(lr*.2)
    start_time = time.time()
    sum_wassD = []
#    sum_errG = []
    for batch in train_data:
        if batch[1].shape[0] != batch_size:
            break
        data = nd.array(batch[0].asnumpy()).as_in_context(ctx)
        for _ in range(n_critic):
        #========G fixed , train D,maxmize -D(x) + D(G(z))======
            latent_z = mx.nd.random_normal(0,1,shape=(batch_size,latent_z_size,1,1),ctx=ctx)
            
            with autograd.record():
                errD_real = netD(data).reshape((-1,1)) 
#                metric_real.update(errD_real,real_label)
                
                fake = netG(latent_z)
                errD_fake = netD(fake).reshape((-1,1)) #errD_fake changed
                errD = errD_fake - errD_real
                errD = errD.mean()
                errD.backward()
#                metric_fake.update(errD_fake,fake_label)
            
            trainerD.step(batch_size)
            netD = clip(netD,clipping_constant)
        
        errD_real = netD(data).reshape((-1,1)) 
        latent_z = mx.nd.random_normal(0,1,shape=(batch_size,latent_z_size,1,1),ctx=ctx)
        fake = netG(latent_z)
        errD_fake = netD(fake).reshape((-1,1)) #errD_fake changed
        wassD = errD_real - errD_fake
        sum_wassD.append(nd.mean(wassD).asscalar())
        #=======D fixed , train G, maxmize D(G(z))============
        with autograd.record():
            fake = netG(latent_z)
            errG = -(netD(fake).reshape((-1,1)).mean())
            errG.backward()
        
#        sum_errG.append(nd.mean(errG).asscalar()) 
        trainerG.step(batch_size)
        





    end_time = time.time() 
#    _,acc_real = metric_real.get()
#    _,acc_fake = metric_fake.get()
#    his_acc_real.append(acc_real)
#    his_acc_fake.append(acc_fake)
    his_wassD.append((np.mean(sum_wassD)))
#    his_errG.append(sum(sum_errG)/len(sum_errG))
    save_params(netD,epoch,path_D)
    save_params(netG,epoch,path_G)
    logging.info('epoch: %i ; Wasserstein distance :%f ; time:%f ' %(epoch , (np.mean(sum_wassD)) ,end_time-start_time))
#    metric_real.reset()
#    metric_fake.reset()
    if (0 < epoch < 10) or ((epoch % 20) == 0):
        fig = plt.figure(figsize=(8,8))
        for i in range(16):
            latent_z = mx.nd.random_normal(0,1,shape=(1,latent_z_size,1,1),ctx=ctx)
            fake = netG(latent_z)
            plt.subplot(4,4,i+1)
            show_images(fake[0])
        plt.show()    

#plot the data
x_axis = np.linspace(0,epochs,len(his_wassD))
plt.figure(figsize=(20,15))
#plt.plot(x_axis,his_errG,label='error of Generator')
plt.plot(x_axis,his_wassD,label='wasserstein distance(mnist)')
plt.xlabel('epoch')
plt.legend()
plt.show()



#plot acc_real and acc_fake seperately
#x_axis = np.linspace(0,epochs,len(his_acc_real))
#plt.figure(figsize=(10,15))
#plt.plot(x_axis,his_acc_real,label='acc_real')
#plt.plot(x_axis,his_acc_fake,label='acc_fake')
#plt.xlabel('epoch')
#plt.legend()
#plt.show()

        
    
    
