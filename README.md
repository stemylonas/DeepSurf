# DeepSurf
A surface-based deep learning approach for the prediction of ligand binding sites on proteins (https://doi.org/10.1093/bioinformatics/btab009)

Setup
---------------

Experiments were conducted on an Ubuntu 18.04 machine with Python 3.6.9 and CUDA 10.0 

1) Install dependencies
```
sudo apt-get update && apt-get install python3-venv, p7zip, swig, libopenbabel-dev, g++
```
2) Clone this repository
```
git clone https://github.com/stemylonas/DeepSurf
cd DeepSurf
```
3) Create environment and install python dependencies
```
python3 -m venv venv --prompt DeepSurf
source venv/bin/activate
pip install -r requirements.txt
```
4) Compile custom LDS module
```
cd lds
chmod a+x compile.sh
./compile.sh
cd ..
```
5) Download pretrained models
```
pip install gdown
gdown 1nIBoD3_5nuMqgRGx4G1OHZwLsiUjb7JG
p7zip -d models.7z
```
6) Collect and install DMS
```
wget www.cgl.ucsf.edu/Overview/ftp/dms.zip
unzip dms.zip
rm dms.zip
cd dms
sudo make install
cd ..
```

Usage example
---------------

```
python predict.py -p protein.pdb -mp model_path -o output_path
```

For more input options, check 'predict.py'. All other molecules (waters, ions, ligands) should be removed from the structure. If the input protein has not been protonated, add --protonate to the execution command.\
The provided models have been trained on a subset of scPDB (training_subset_of_scpdb.proteins)
