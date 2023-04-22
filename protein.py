#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 12:45:54 2020

@author: smylonas
"""

import os, numpy as np
from openbabel import pybel
from utils import simplify_dms


class Protein:
    def __init__(self,prot_file,protonate,expand_residue,f,save_path,discard_points,seed=None):
        prot_id = prot_file.split('/')[-1].split('.')[0]
        self.save_path = os.path.join(save_path,prot_id)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
       
        self.mol = next(pybel.readfile(prot_file.split('.')[-1],prot_file)) 
        
        surfpoints_file = os.path.join(self.save_path,prot_id+'.surfpoints')
        os.system('dms '+prot_file+' -d 0.2 -n -o '+surfpoints_file)
        if not os.path.exists(surfpoints_file):
            raise Exception('probably DMS not installed')
        self.surf_points, self.surf_normals = simplify_dms(surfpoints_file, f, seed=seed)     
        if discard_points:
            os.remove(surfpoints_file)
            
        self.expand_residue = expand_residue
        if expand_residue:
            self.mol.removeh()
            self.atom2residue = np.array([atom.residue.idx for atom in self.mol.atoms])
            self.residue2atom = np.array([[atom.idx - 1 for atom in resid.atoms] for resid in self.mol.residues])
            self.mol.addh()
        else:
            if protonate:      
                self.mol.addh()

        self.heavy_atom_coords = np.array([atom.coords for atom in self.mol.atoms if atom.atomicnum > 1])
              
        self.binding_sites = []
        if prot_file.endswith('pdb'):
            with open(prot_file,'r') as f:    
                lines = f.readlines()
            self.heavy_atom_lines = [line for line in lines if line[:4]=='ATOM' and line.split()[2][0]!='H']
            if len(self.heavy_atom_lines) != len(self.heavy_atom_coords):
                ligand_in_pdb = len([line for line in lines if line.startswith('HETATM')])>0
                if ligand_in_pdb:
                    raise Exception('Ligand found in PDBfile. Please remove it to procede.')
                else:
                    raise Exception('Incosistency between Coords and PDBLines')
        else:
            raise IOError('Protein file should be .pdb')
              
    def _surfpoints_to_atoms(self,surfpoints):
        close_atoms = np.zeros(len(surfpoints),dtype=int)
        for p,surf_coord in enumerate(surfpoints):
            dist = np.sqrt(np.sum((self.heavy_atom_coords-surf_coord)**2,axis=1))
            close_atoms[p] = np.argmin(dist)
        
        return np.unique(close_atoms)
        
    def add_bsite(self,cluster):   # cluster -> tuple: (surf_points,scores)
        atom_idxs = self._surfpoints_to_atoms(cluster[0])
        if self.expand_residue:
            residue_idxs = np.unique(self.atom2residue[atom_idxs])
            atom_idxs = np.concatenate(self.residue2atom[residue_idxs])
        self.binding_sites.append(Bsite(self.heavy_atom_coords,atom_idxs,cluster[1]))
        
    def sort_bsites(self):
        avg_scores = np.array([bsite.score for bsite in self.binding_sites])
        sorted_idxs = np.flip(np.argsort(avg_scores),axis=0)
        self.binding_sites = [self.binding_sites[idx] for idx in sorted_idxs]
        
    def write_bsites(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        centers = np.array([bsite.center for bsite in self.binding_sites])
        np.savetxt(os.path.join(self.save_path,'centers.txt'), centers, delimiter=' ', fmt='%10.3f')

        for i,bsite in enumerate(self.binding_sites):
            with open(os.path.join(self.save_path,'pocket'+str(i+1)+'.pdb'),'w') as f:
                outlines = [self.heavy_atom_lines[idx] for idx in bsite.atom_idxs]
                f.writelines(outlines)


class Bsite:
    def __init__(self,mol_coords,atom_idxs,scores):
        self.coords = mol_coords[atom_idxs]
        self.center = np.average(self.coords,axis=0)
        self.score = np.average(scores)
        self.atom_idxs = atom_idxs
    
