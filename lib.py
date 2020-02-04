#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:39:49 2019

@author: smylonas
"""

import numpy as np
from scipy.spatial.distance import euclidean
#from rotation import rotation_quaternion
#from moleculekit.molecule import Molecule
#from moleculekit.tools.voxeldescriptors import _getOccupancyC
#from scipy.cluster.hierarchy import fclusterdata
from sklearn.cluster import KMeans
#from collections import defaultdict


def mol2_reader(mol_file):
    if mol_file[-4:] != 'mol2':
        print 'cant read no mol2 file'
        return
    
    with open(mol_file,'r') as f:
        lines = f.readlines()
    
    for i,line in enumerate(lines):
        if '@<TRIPOS>ATOM' in line:
            first_atom_idx = i+1
        if '@<TRIPOS>BOND' in line:
            last_atom_idx = i-1
    
    return lines[first_atom_idx:last_atom_idx+1]


def readSurfPoints(surf_file):
    with open(surf_file,'r') as f:
        lines = f.readlines()
    
    lines = [l for l in lines if len(l.split())>7]
    if len(lines)>100000:
        print 'Large size'
        return
    if len(lines)==0:
        print 'Empty file'
        return
    
    coords = np.zeros((len(lines),3))
    normals = np.zeros((len(lines),3))
    for i,l in enumerate(lines):
        parts = l.split()
        try:
            coords[i,0] = float(parts[3])
            coords[i,1] = float(parts[4])
            coords[i,2] = float(parts[5])
            normals[i,0] = float(parts[8])
            normals[i,1] = float(parts[9])
            normals[i,2] = float(parts[10])
        except:
            coords[i,0] = float(parts[2][-8:])
            coords[i,1] = float(parts[3])
            coords[i,2] = float(parts[4])
            normals[i,0] = float(parts[7])
            normals[i,1] = float(parts[8])
            normals[i,2] = float(parts[9])
            
    return coords, normals


def simplify_dms(init_surf_file, factor):
    
    coords, normals = readSurfPoints(init_surf_file)
    nPoints = len(coords)
    nCl = nPoints/factor
    
    kmeans = KMeans(n_clusters=nCl,max_iter=300,n_init=1).fit(coords)
    point_labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    cluster_idx,freq = np.unique(point_labels,return_counts=True)
    if len(cluster_idx)!=nCl:  # need to be removed
        print 'error'

    idxs = []
    for cl in cluster_idx:
        cluster_points_idxs = np.where(point_labels==cl)[0]
        closest_idx_to_center = np.argmin([euclidean(centers[cl],coords[idx]) for idx in cluster_points_idxs])
        idxs.append(cluster_points_idxs[closest_idx_to_center])
    
    return coords[idxs], normals[idxs]


                                                                              


    





    