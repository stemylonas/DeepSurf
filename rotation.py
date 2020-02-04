#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 10:58:44 2019

@author: smylonas
"""

import numpy as np


def rotation_quaternion(n):
    if n[0]==0.0 and n[1]==0.0:
        if n[2]==1.0:
            return np.identity(3)
        elif n[2]==-1.0:
            Q = np.identity(3)
            Q[0,0] = -1
            return Q
        else:
            print 'not possible'
        
    rx = -n[1]/np.sqrt(n[0]*n[0]+n[1]*n[1])
    ry = n[0]/np.sqrt(n[0]*n[0]+n[1]*n[1])
    rz = 0
    th = np.arccos(n[2])
    
    q0 = np.cos(th/2)
    q1 = np.sin(th/2)*rx
    q2 = np.sin(th/2)*ry
    q3 = np.sin(th/2)*rz
               
    Q = np.zeros((3,3))
    Q[0,0] = q0*q0+q1*q1-q2*q2-q3*q3
    Q[0,1] = 2*(q1*q2-q0*q3)
    Q[0,2] = 2*(q1*q3+q0*q2)
    Q[1,0] = 2*(q1*q2+q0*q3)
    Q[1,1] = q0*q0-q1*q1+q2*q2-q3*q3
    Q[1,2] = 2*(q3*q2-q0*q1)
    Q[2,0] = 2*(q1*q3-q0*q2)
    Q[2,1] = 2*(q3*q2+q0*q1)
    Q[2,2] = q0*q0-q1*q1-q2*q2+q3*q3
     
    return Q



def rotation_rodriquez(n,case):
    sinth = -np.sqrt(n[0]*n[0]+n[1]*n[1])
    costh = n[2]
    rx = -n[1]/sinth
    ry = n[0]/sinth
    K = np.zeros((3,3))
    K[0,2] = ry
    K[1,2] = -rx
    K[2,0] = -ry
    K[2,1] = rx
    
    if case==1:
        return costh*np.identity(3) + sinth*K + (1-costh)*np.outer(np.array([rx,ry,0]),np.array([rx,ry,0]))
    elif case==2:
        return np.identity(3) + K + np.matmul(K,K)/(1+costh)
    elif case==3:
        return np.identity(3) + sinth*K + (1-costh)*np.matmul(K,K)