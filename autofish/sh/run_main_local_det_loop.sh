#!/bin/bash

#SBATCH -N 1

#nombre threads max sur GPU 48

#SBATCH -n 1


#SBATCH -J local_01


#SBATCH --output="local_01.out"
#SBATCH --mem 40000    # Memory per node in MB (0 allocates all the memory)

#SBATCH --ntasks=1              # Number of processes to run (default is 1)
#SBATCH --cpus-per-task=6      # CPU cores per process (default 1)


#SBATCH -p cbio-cpu


cd /cluster/CBIO/data1/data3/tdefard/stich_seg_detect

python main.py \
--folder_of_rounds /cluster/CBIO/data1/data3/tdefard/T7/2023-09-06_LUSTRA-14rounds/image_per_pos/ \
--path_to_dapi_folder /cluster/CBIO/data1/data3/tdefard/T7/2023-09-06_LUSTRA-14rounds/image_per_pos/k_r1_DAPI/ \
--path_to_mask_dapi /cluster/CBIO/data1/data3/tdefard/T7/2023-09-06_LUSTRA-14rounds/image_per_pos/segmentation_mask/ \
--regex_dapi ch0 \
--fixed_round_name r1 \
--folder_regex_round  $1 \
--chanel_regex ch0 \
--name_dico 15juin \
--segmentation 0 \
--registration 1 \
--spots_detection 1 \
--signal_quality 1 \
--stitch 0 \
--local_detection 1 \




