#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 03:12:17 2017

@author: naminpark
"""

import gym

env = gym.make('CartPole-v0')

env.reset()
random_episodes =0
reward_sum=0

while random_episodes <10:
    env.render()
    action = env.action_space.sample()
    observation, reward, done, _ =env.step(action)
    print(observation, reward, done)
    reward_sum +=reward
    
    if done:
        random_episodes +=1
        print("Reward for this episodes was:", reward_sum)
        reward_sum =0
        env.reset()