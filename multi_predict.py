#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 16:04:47 2020

@author: smylonas
"""

import argparse, os, time
from network import Network
from protein import Protein
from bsite_extraction import Bsite_extractor
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset_file', '-d', required=True, help='dataset file with protein names')
    parser.add_argument('--protein_path', '-pp', required=True, help='directory of protein files')
    parser.add_argument('--model_path', '-mp', required=True, help='directory of models')
    parser.add_argument('--model', '-m', choices=['orig','lds'], default='orig', help='select model')
    parser.add_argument('--output', '-o', required=True, help='name of the output directory')
    parser.add_argument('--f', type=int, default=10, help='parameter for the simplification of points mesh')
    parser.add_argument('--T', type=float, default=0.9, help='ligandability threshold')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--voxel_size', type=float, default=1.0, help='size of voxel in angstrom')
    parser.add_argument('--protonate', action='store_true', help='whether to protonate or not the input protein')
    parser.add_argument('--expand', action='store_true', help='whether to expand on residue level the extracted binding sites')
    parser.add_argument('--discard_points', action='store_true', help='whether to output or not the computed surface points')
    parser.add_argument('--seed', type=int, default=None, help='random seed for KMeans clustering')

    return parser.parse_args()


args = parse_args()

if not os.path.exists(args.protein_path):
    raise IOError('%s does not exist.' % args.protein_path)
if not os.path.exists(args.model_path):
    raise IOError('%s does not exist.' % args.model_path)
if not os.path.exists(args.output):
    os.makedirs(args.output)

st = time.time()

with open(args.dataset_file,'r') as f:
    lines = f.readlines()

protein_names = [line.strip() for line in lines if line!='\n']

nn = Network(args.model_path, args.model, args.voxel_size)
extractor = Bsite_extractor(args.T)

for prot in tqdm(protein_names):
    print(prot)
    protein = Protein(os.path.join(args.protein_path,prot+'.pdb'), args.protonate, args.expand, args.f, args.output, args.discard_points, args.seed)
    
    lig_scores = nn.get_lig_scores(protein, args.batch)
       
    extractor.extract_bsites(protein, lig_scores)

print('Total time:', time.time()-st)
