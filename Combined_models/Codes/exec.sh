#!/bin/bash
# Author: ADRIANA

# $1: path to execude python codes
# How to run the code: ./exec.sh $1

## MAIN
# Iterate among folders and execute python codes

cd $1
for folder in *; do
	if [ -d $folder ]; then
		echo $folder
		cd $folder
		PYTHONHASHSEED=0 python3 <path_to_combine_folder>
		cd ..
		fi
done	
