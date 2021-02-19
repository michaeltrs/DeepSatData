#!/usr/bin/env bash

products_dir=$1
windows_dir=$2
timeseries_dir=$3
res=$4
sample_size=$5
num_processes=$6
anchor=${7:-"None"}

# 1:products_dir, 2:windows_dir, 3:timeseries_dir, 4:res, 5:sample_size, 6:num_processes, 7:anchor,

python dataset/unlabelled/extract_images.py --products_dir $products_dir \
                                       --savedir $windows_dir \
                                       --anchor $anchor \
                                       --res $res \
                                       --sample_size $sample_size \
                                       --num_processes $num_processes

python dataset/unlabelled/make_image_timeseries.py --windows_dir $windows_dir \
                                              --savedir $timeseries_dir \
                                              --res $res \
                                              --sample_size $sample_size \
                                              --num_processes $num_processes
