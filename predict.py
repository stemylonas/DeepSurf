#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 16:16:56 2020

@author: smylonas
"""

import argparse, os
from network import Network
from protein import Protein
from bsite_extraction import Bsite_extractor

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--prot_file', '-p', required=True, help='input protein file (pdb)')
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

if not os.path.exists(args.prot_file):
    raise IOError('%s does not exist.' % args.prot_file)
if not os.path.exists(args.model_path):
    raise IOError('%s does not exist.' % args.model_path)
if not os.path.exists(args.output):
    os.makedirs(args.output)

prot = Protein(args.prot_file,args.protonate,args.expand,args.f,args.output, args.discard_points, args.seed)

nn = Network(args.model_path,args.model,args.voxel_size)

lig_scores = nn.get_lig_scores(prot,args.batch)

extractor = Bsite_extractor(args.T)

extractor.extract_bsites(prot,lig_scores)

