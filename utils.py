"""
Created on Tue Jul 30 13:39:49 2019

@author: smylonas
"""

import warnings
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans


def mol2_reader(mol_file):  # does not handle H2
    if mol_file[-4:] != 'mol2':
        raise Exception("File's extension is not .mol2")
    
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
        warnings.warn('{} has too many points'.format(surf_file))
        return
    if len(lines)==0:
        warnings.warn('{} is empty'.format(surf_file))
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


def simplify_dms(init_surf_file, factor, seed=None):
       
    coords, normals = readSurfPoints(init_surf_file)
    
    if factor == 1:
        return coords, normals

    nPoints = len(coords)
    nCl = nPoints//factor
    
    kmeans = KMeans(n_clusters=nCl, max_iter=300, n_init=1, random_state=seed).fit(coords)
    point_labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    cluster_idx,freq = np.unique(point_labels,return_counts=True)
    if len(cluster_idx)!=nCl:
        raise Exception('Number of created clusters should be equal to nCl')

    idxs = []
    for cl in cluster_idx:
        cluster_points_idxs = np.where(point_labels==cl)[0]
        closest_idx_to_center = np.argmin([euclidean(centers[cl],coords[idx]) for idx in cluster_points_idxs])
        idxs.append(cluster_points_idxs[closest_idx_to_center])
    
    return coords[idxs], normals[idxs]


def rotation(n):
    if n[0]==0.0 and n[1]==0.0:
        if n[2]==1.0:
            return np.identity(3)
        elif n[2]==-1.0:
            Q = np.identity(3)
            Q[0,0] = -1
            return Q
        else:
            print('not possible')
        
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

