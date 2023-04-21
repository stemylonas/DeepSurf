conda create -n deepsurf python=3.6
conda activate deepsurf
sudo apt update
sudo apt install -y p7zip
sudo apt install -y libopenbabel-dev
sudo apt install -y g++
sudo apt install -y swig
conda install -c conda-forge openbabel
pip install numpy==1.13.3
pip install tensorflow-gpu==1.13.1
pip install scipy==1.2.2
pip install scikit-learn==0.20.3
pip install protobuf==3.6.1

git clone https://github.com/stemylonas/DeepSurf
cd DeepSurf

cd lds
chmod a+x compile.sh
./compile.sh
cd ..

pip install gdown
gdown 1nIBoD3_5nuMqgRGx4G1OHZwLsiUjb7JG
p7zip -d models.7z

wget www.cgl.ucsf.edu/Overview/ftp/dms.zip
unzip dms.zip
rm dms.zip
cd dms
sudo make install
cd ..
