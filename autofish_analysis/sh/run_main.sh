#!/bin/bash

#SBATCH -N 1

#nombre threads max sur GPU 48

#SBATCH -n 1


#SBATCH -J loc_spots_detection

#SBATCH -J 20juin


#SBATCH --output="seg_lustra20juin.out"
#SBATCH --mem 65000    # Memory per node in MB (0 allocates all the memory)

#SBATCH --ntasks=1              # Number of processes to run (default is 1)
#SBATCH --cpus-per-task=12      # CPU cores per process (default 1)


#SBATCH -p cbio-cpu



cd /cluster/CBIO/data1/data3/tdefard/stich_seg_detect

python main.py \
--folder_of_rounds /cluster/CBIO/data1/data3/tdefard/T7/plane_03/plane_03/ \
--path_to_dapi_folder /cluster/CBIO/data1/data3/tdefard/T7/2023-09-06_LUSTRA-14rounds/image_per_pos/k_r1_DAPI/ \
--path_to_mask_dapi /cluster/CBIO/data1/data3/tdefard/T7/2023-09-06_LUSTRA-14rounds/image_per_pos/segmentation_mask/ \
--regex_dapi ch0 \
--fixed_round_name r \
--folder_regex_round r \
--chanel_regex c \
--name_dico 20juin \
--segmentation 0 \
--registration 0 \
--spots_detection 1 \
--signal_quality 0 \
--stitch 0 \
--local_detection 0 \
--mask_artefact 1 \
--artefact_filter_size 30 \
--stich_spots_detection 0 \
--generate_comseg_input 0 \
--sigma_detection 1.15 \
