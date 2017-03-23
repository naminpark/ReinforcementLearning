#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 00:54:29 2017

@author: naminpark
"""

import gym
import numpy as np
import tensorflow as tf
#from gym.envs.registration import register

env =gym.make('CartPole-v0')

input_size = env.observation_space.shape[0]
output_size =env.action_space.n

learning_rate =0.1

X = tf.placeholder(shape=[None,input_size],dtype=tf.float32, name ="input_x")
W1 = tf.get_variable("W1",shape=[input_size,output_size],initializer = tf.contrib.layers.xavier_initializer())

Qpred = tf.matmul(X,W1)

Y = tf.placeholder(shape=[None,output_size],dtype=tf.float32)

loss =tf.reduce_sum(tf.square(Y-Qpred))

train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

num_episodes = 5000

rList=[]
dis = .9


init = tf.global_variables_initializer()
sess= tf.Session()
sess.run(init)
for i in range(num_episodes):
    s =env.reset()
    rAll = 0
    done = False
    e=1./((i/10)+1)
    step_count = 0
	
    while not done:
        step_count +=1
        x= np.reshape(s,[1,input_size])
        Qs= sess.run(Qpred,feed_dict={X: x})
        if np.random.rand(1) <e:
            a = env.action_space.sample()
        else:
            a = np.argmax(Qs)
            
        s1, reward, done, _= env.step(a)
        if done:
            Qs[0,a] =-100
        else:
            x1=np.reshape(s1,[1,input_size])
            Qs1=sess.run(Qpred, feed_dict={X: x1})
            Qs[0,a] =reward + dis * np.max(Qs1)
        sess.run(train, feed_dict={X:x, Y:Qs})
        s = s1
    rList.append(step_count)
    print("Episode: {} steps: {}".format(i,step_count))
    
    if len(rList) >10 and np.mean(rList[-10:]) >500:
        break
    
observation =env.reset()
reward_sum=0

while True:
    env.render()
    
    x=np.reshape(observation,[1,input_size])
    Qs = sess.run(Qpred, feed_dict={X:x})
    a= np.argmax(Qs)
    
    observation,reward ,done, _ =env.step(a)
    reward_sum +=reward
    if done:
        print("Total score: {}".format(reward_sum))
        break
		