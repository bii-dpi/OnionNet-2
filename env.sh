#!/bin/bash
CONDA_BASE=$(conda info --base)
#conda create -n onion python=3.8 -y
source $CONDA_BASE/etc/profile.d/conda.sh

conda activate onion

#pip install tensorflow-gpu==2.3

conda install numpy pandas scikit-learn=0.22.1 -y
conda install -c conda-forge mdtraj openbabel -y

