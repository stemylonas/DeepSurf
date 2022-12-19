#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 11:46:54 2019

@author: smylonas
"""

import tfbio_data 
from utils import rotation
import numpy as np
     

class KalasantyFeaturizer:
    def __init__(self,gridSize,voxelSize):
        grid_limit = (gridSize/2-0.5)*voxelSize
        grid_radius = grid_limit*np.sqrt(3)             
        self.neigh_radius = 4 + grid_radius   # 4 > 2*R_vdw
        self.featurizer = tfbio_data.Featurizer(save_molecule_codes=False)
        self.grid_resolution = voxelSize
        self.max_dist = (gridSize-1)*voxelSize/2 
    
    def get_channels(self,mol):
        _, self.channels = self.featurizer.get_features(mol)  # returns only heavy atoms
    
    def grid_feats(self,point,normal,mol_coords):
        neigh_atoms = np.sqrt(np.sum((mol_coords-point)**2,axis=1))<self.neigh_radius
        Q = rotation(normal)
        Q_inv = np.linalg.inv(Q)
        transf_coords = np.transpose(mol_coords[neigh_atoms]-point)
        rotated_mol_coords = np.matmul(Q_inv,transf_coords)
        features = tfbio_data.make_grid(np.transpose(rotated_mol_coords),self.channels[neigh_atoms],self.grid_resolution,self.max_dist)[0]
        
        return features
        
        
