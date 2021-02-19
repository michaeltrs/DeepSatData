#!/usr/bin/env bash

ground_truths_file=$1
products_dir=$2
labels_dir=$3
windows_dir=$4
timeseries_dir=$5
res=$6
sample_size=$7
num_processes=$8

# 1:ground_truths_file, 2:products_dir, 3:labels_dir, 4:windows_dir, 5:timeseries_dir, 6:res, 7:sample_size, 8:num_processes
python dataset/labelled_dense/extract_labels_raster.py --ground_truths_file $ground_truths_file \
                                                       --products_dir $products_dir \
                                                       --savedir $labels_dir \
                                                       --res $res \
                                                       --sample_size $sample_size \
                                                       --num_processes $num_processes

python dataset/labelled_dense/extract_images_for_labels.py --ground_truths_dir $labels_dir \
                                                           --products_dir $products_dir \
                                                           --savedir $windows_dir \
                                                           --res $res \
                                                           --sample_size $sample_size \
                                                           --num_processes $num_processes

python dataset/labelled_dense/make_image_timeseries_for_labels.py --ground_truths_dir $labels_dir \
                                                                  --products_dir $products_dir \
                                                                  --windows_dir $windows_dir \
                                                                  --savedir $timeseries_dir \
                                                                  --res $res \
                                                                  --sample_size $sample_size \
                                                                  --num_processes $num_processes
