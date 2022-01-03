# DeepSurf
A surface-based deep learning approach for the prediction of ligand binding sites on proteins (https://doi.org/10.1093/bioinformatics/btab009)

Installation
---------------

1) Python 3 and CUDA 9 are required 
2) Install DMS from http://www.cgl.ucsf.edu/Overview/software.html
3) Install openbabel (version 2.4.1 originally used)
4) Download trained models from https://drive.google.com/file/d/1nIBoD3_5nuMqgRGx4G1OHZwLsiUjb7JG/view?usp=sharing
5) Install python dependencies (requirements.txt)
6) Execute 'lds/compile' to compile the custom LDS-module. If you have g++>=5, add -D_GLIBCXX_USE_CXX11_ABI=0 to the g++ commands.


Usage example
---------------

```
python predict.py -p protein.pdb -mp model_path -o output_path
```

For more input options, check 'predict.py'. All other molecules (waters, ions, ligands) should be removed from the structure. If the input protein has not been protonated, add --protonate to the execution command.\
The provided models have been trained on a subset of scPDB (training_subset_of_scpdb.proteins)
