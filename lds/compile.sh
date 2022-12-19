#!/bin/bash

TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

G_FLAGS="-std=c++11 -shared -I$TF_INC -fPIC -L$TF_LIB -ltensorflow_framework"

G_VERSION="$(g++ -dumpversion)"
compared_ver="5.0.0"

if [ "$(printf '%s\n' "$compared_ver" "$G_VERSION" | sort -V | head -n1)" = "$compared_ver" ]; then
        G_FLAGS+=" -D_GLIBCXX_USE_CXX11_ABI=0"
fi

mkdir -p lib

#### FORWARD

nvcc -std=c++11 -c amul_kernel_cube.cu.cc -o lib/amul_kernel_cube.cu.o -I $TF_INC -D GOOGLE_CUDA=1 -x cu -gencode arch=compute_61,code=sm_61  -Xcompiler -fPIC -O3 --expt-relaxed-constexpr -lprotobuf 

g++ amul_kernel_cube.cc -o lib/amul_kernel_cube.so lib/amul_kernel_cube.cu.o $G_FLAGS


#### GRAD

nvcc -std=c++11 -c amul_kernel_cube_grad.cu.cc -o lib/amul_kernel_cube_grad.cu.o -I $TF_INC -D GOOGLE_CUDA=1 -x cu -gencode arch=compute_61,code=sm_61 -Xcompiler -fPIC -O3 --expt-relaxed-constexpr -lprotobuf

g++ amul_kernel_cube_grad.cc -o lib/amul_kernel_cube_grad.so lib/amul_kernel_cube_grad.cu.o $G_FLAGS

