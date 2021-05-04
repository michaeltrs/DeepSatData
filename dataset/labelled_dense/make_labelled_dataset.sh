#!/usr/bin/env bash

bands='None'
for ARGUMENT in "$@"
do

    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)

    case "$KEY" in

          ground_truths_file)   ground_truths_file=${VALUE} ;;
          products_dir)         products_dir=${VALUE} ;;
          labels_dir)           labels_dir=${VALUE} ;;
          windows_dir)          windows_dir=${VALUE} ;;
          timeseries_dir)       timeseries_dir=${VALUE} ;;
          res)                  res=${VALUE} ;;
          sample_size)          sample_size=${VALUE} ;;
          num_processes)        num_processes=${VALUE} ;;
          bands)                bands=${VALUE} ;;
            *)
    esac

done

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
                                                           --bands $bands \
                                                           --res $res \
                                                           --sample_size $sample_size \
                                                           --num_processes $num_processes

python dataset/labelled_dense/make_image_timeseries_for_labels.py --ground_truths_dir $labels_dir \
                                                                  --products_dir $products_dir \
                                                                  --windows_dir $windows_dir \
                                                                  --savedir $timeseries_dir \
                                                                  --bands $bands \
                                                                  --res $res \
                                                                  --sample_size $sample_size \
                                                                  --num_processes $num_processes
