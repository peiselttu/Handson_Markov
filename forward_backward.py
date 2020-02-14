# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 16:46:29 2020

@author: bo.pei
"""

states=('Healthy','Fever')
end_state='E'
observations=('Normal','Cold','Dizzy')
start_probability={'Healthy':0.6,
                   'Fever':0.4}
transition_probability={'Healthy':{
                                   'Healthy':0.69,
                                   'Fever':0.3,
                                   'E':0.01
                                  },
                        'Fever':{
                                  'Healthy':0.4,
                                  'Fever':0.59,
                                  'E':0.01
                                }}
emission_probability={'Healthy':{
                                'Normal':0.5,
                                'Cold':0.4,
                                'Dizzy':0.1
                                },
                      'Fever':{
                               'Normal':0.1,
                               'Cold':0.3,
                               'Dizzy':0.6
                              }
                      }
                      
                      
'''
To calculate the probability of generating the observations
'''
def forward(observations=observations,
            states=states,
            start_probability=start_probability,
            transition_probability=transition_probability,
            emission_probability=emission_probability,
            end_state=end_state):
    prev_f={}
    fwd=[] # the probability of each states at each timestamp, given the specific observations
    for i,obs_i in enumerate(observations):
        curr_f={}
        for st in states:
            if i==0:
                sum_f_prev=start_probability[st] #if it is the first timestamp, the probability of the current state is that of initial probs
            else:
                sum_f_prev=sum(prev_f[k]*transition_probability[k][st] for k in states)
            
            curr_f[st]=sum_f_prev*emission_probability[st][obs_i]
        prev_f=curr_f
        fwd.append(curr_f)
    p_fwd=sum(curr_f[k]*transition_probability[k][end_state] for k in states)
    return fwd,p_fwd

# backward calculation： 求产生给定输出的概率
def backward(observations=observations,
            states=states,
            start_probability=start_probability,
            transition_probability=transition_probability,
            emission_probability=emission_probability,
            end_state=end_state):
    
    obs=observations[1:]+(None,) # add a None element at the end of tuple 来达到错位的效果，以方便在后期倒序的时候，当前状态下对应下一个输出
    bwk=[]
    prev_b={}
    for i,obs_plus_i in enumerate(reversed(obs)):
        curr_b={}
        for st in states:
            if i==0:
                sum_b_prev=transition_probability[st][end_state]
            else:
                sum_b_prev=sum(transition_probability[st][k]*emission_probability[k][obs_plus_i]*prev_b[k] for k in states)
            curr_b[st]=sum_b_prev
        prev_b=curr_b
        bwk.insert(0,curr_b)
    
    p_bwk=sum(start_probability[k]*curr_b[k]*emission_probability[k][observations[0]] for k in states)
    return bwk,p_bwk
        
def forward_backward(observations=observations,
            states=states,
            start_probability=start_probability,
            transition_probability=transition_probability,
            emission_probability=emission_probability,
            end_state=end_state):
    fwd,p_fwd=forward()
    bwk,p_bwk=backward()
    posterior=[]
    for i in range(len(observations)):
        posterior.append({st:fwd[i][st]*bwk[i][st]/p_fwd for st in states})
    return posterior
                
        
                
            