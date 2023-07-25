#!/bin/bash

for folder_regex_round in "r1" "r2" "r3" "r4" "r5" "r6" "r7" "r8" "r9" "r10" "r11" "r12" 'r13'

do
	sbatch -J $folder_regex_round --output=${folder_regex_round}.out run_main_local_det_loop.sh $folder_regex_round
done

