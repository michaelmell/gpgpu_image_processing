#!/usr/bin/env bash

conda init bash

conda create -y -n gpgpu_image_processing python=3.7
conda activate gpgpu_image_processing

conda config --add channels conda-forge

conda install -y -c conda-forge ocl-icd-system  # needed on linux for conda to env to find ocl-icd files living in /etc/OpenCL/vendors; see: https://documen.tician.de/pyopencl/misc.html#using-vendor-supplied-opencl-drivers-linux
conda install -y numpy
conda install -y -c conda-forge pyopencl
conda install -y -c conda-forge opencv

conda deactivate
