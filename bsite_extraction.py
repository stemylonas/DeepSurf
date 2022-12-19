#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 16:00:46 2020

@author: smylonas
"""

import numpy as np
from sklearn.cluster import MeanShift


class Bsite_extractor():
    def __init__(self,lig_thres,bw=15):
        self.T = lig_thres
        self.ms = MeanShift(bandwidth=bw,bin_seeding=True,cluster_all=False,n_jobs=4)
    
    def _cluster_points(self,prot,lig_scores):
        T_new = self.T
        while sum(lig_scores>=T_new) < 10 and T_new>0.3001:    # at least 10 points with prob>P  and P>=0.3
            T_new -= 0.1 

        filtered_points = prot.surf_points[lig_scores>T_new]
        filtered_scores = lig_scores[lig_scores>T_new]
        if len(filtered_points)<5:
            return () 

        clustering = self.ms.fit(filtered_points)
        labels = clustering.labels_
        
        unique_l,freq = np.unique(labels,return_counts=True)
    
        if len(unique_l[freq>=5])!=0:
            unique_l = unique_l[freq>=5]    # keep clusters with 5 points and more
        else:
            return ()
        
        if unique_l[0]==-1:                 # discard the "unclustered" cluster
            unique_l = unique_l[1:]    
        
        clusters = [(filtered_points[labels==l],filtered_scores[labels==l]) for l in unique_l]
        
        return clusters
        
    def extract_bsites(self,prot,lig_scores):
        clusters = self._cluster_points(prot,lig_scores)
        if len(clusters)==0:
            print('No binding site found')
            return
        for cluster in clusters:
            prot.add_bsite(cluster)
        prot.sort_bsites()
        prot.write_bsites()
        
        
