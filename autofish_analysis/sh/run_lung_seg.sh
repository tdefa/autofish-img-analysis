#!/bin/bash

#SBATCH -N 1

#nombre threads max sur GPU 48

#SBATCH -n 1


#SBATCH -J pwc14_7_8X8


#SBATCH --output="pwc14_7_8X8.out"
#SBATCH --mem 75000    # Memory per node in MB (0 allocates all the memory)

#SBATCH --ntasks=1              # Number of processes to run (default is 1)
#SBATCH --cpus-per-task=12      # CPU cores per process (default 1)


#SBATCH -p cbio-gpu


cd /cluster/CBIO/data1/data3/tdefard/stich_seg_detect

python segmentation.py
