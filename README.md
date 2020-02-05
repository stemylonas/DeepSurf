# DeepSurf
A surface-based deep learning approach for the prediction of ligand binding sites on proteins

Installation
---------------

1) Install DMS from http://www.cgl.ucsf.edu/Overview/software.html
2) Install openbabel (version 2.4.1 originally used)
3) Download trained models from https://drive.google.com/open?id=10Atg7mtvn1OfkaxfAMUin3ZhZ_LrbT0z
4) Install python dependencies (requirements.txt)
5) Execute 'lds/compile' to compile the custom LDS-module. If you have g++>=5, add -D_GLIBCXX_USE_CXX11_ABI=0 to the g++ commands.


Usage example
---------------

```
python predict.py -p protein.pdb -mp model_path -o output_path
```

For more input options, check 'predict.py'.
All other molecules (waters, ions, ligands) should be removed from the structure.
If the input protein has not been protonated, add --protonate to the execution command.
