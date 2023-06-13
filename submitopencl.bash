#!/bin/bash
#SBATCH -J Fourier 
#SBATCH -A cs475-575
#SBATCH -p classgputest
#SBATCH --constraint=v100
#SBATCH -o output.out
#SBATCH -e output.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=fernaluk@oregonstate.edu

g++ -o proj07 maingpu.cpp /usr/local/apps/cuda/cuda-10.1/lib64/libOpenCL.so.1.1 -lm -fopenmp

./proj07