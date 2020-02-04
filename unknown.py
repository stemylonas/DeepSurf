#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 15:13:37 2020

@author: smylonas
"""

import numpy as np, os
import tensorflow as tf
from tensorflow.contrib import slim
from resnet_3d import resnet_arg_scope, resnet_v1_18
from git_features import KalasantyFeaturizer


class Network:
    def __init__(self,model_path,model,gridSize,voxelSize):
        tf.reset_default_graph()
        self.inputs = tf.placeholder(tf.float32,shape=(None,gridSize,gridSize,gridSize,18))
        
        with slim.arg_scope(resnet_arg_scope()):  # needs selection of model
            self.net, self.end_points = resnet_v1_18(self.inputs, 1, is_training=False)
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        
        self.sess.run(tf.global_variables_initializer()) 
        saver = tf.train.Saver()
        #saver.restore(self.sess,tf.train.latest_checkpoint(model_path))
        if model=='orig':
            saver.restore(self.sess,os.path.join(model_path,'resnet18'))
        #elif model=='lds':
         #   saver.restore(self.sess,model_path)
        
        self.featurizer = KalasantyFeaturizer(gridSize,voxelSize)
        
    def get_lig_scores(self,prot,batch_size,gridSize):
        self.featurizer.get_channels(prot.mol)
        prot.removeh()
        
        lig_scores = []
        input_data = np.zeros((batch_size,gridSize,gridSize,gridSize,18))   # get from inputs.shape ??
        batch_cnt = 0
        
        for p,n in zip(prot.surf_points,prot.surf_normals):
            input_data[batch_cnt,:,:,:,:] = self.featurizer.grid_feats(p,n,prot.atom_coords)  
            batch_cnt += 1
            if batch_cnt==batch_size:
                output = self.sess.run(self.end_points,feed_dict={self.inputs:input_data}) 
                lig_scores += list(output['probs'])
                batch_cnt = 0
        if batch_cnt>0:
            output = self.sess.run(self.end_points,feed_dict={self.inputs:input_data[:batch_cnt,:,:,:,:]}) 
            if batch_cnt==1:
                lig_scores.append(output['probs'])
            else:
                lig_scores += list(output['probs'])
        
        return np.array(lig_scores)
        