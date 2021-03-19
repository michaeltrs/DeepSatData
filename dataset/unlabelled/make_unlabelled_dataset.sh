#!/usr/bin/env bash

anchor='None'
bands='None'
for ARGUMENT in "$@"
do

    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)

    case "$KEY" in
#            steps)   steps=${VALUE} ;;
#            vals)    vals=${VALUE} ;;
          products_dir)     products_dir=${VALUE} ;;
          windows_dir)      windows_dir=${VALUE} ;;
          timeseries_dir)   timeseries_dir=${VALUE} ;;
          res)              res=${VALUE} ;;
          sample_size)      sample_size=${VALUE} ;;
          num_processes)    num_processes=${VALUE} ;;
          anchor)           anchor=${VALUE} ;;
          bands)            bands=${VALUE} ;;
            *)
    esac

done


python dataset/unlabelled/extract_images.py --products_dir $products_dir \
                                       --bands $bands \
                                       --savedir $windows_dir \
                                       --anchor $anchor \
                                       --res $res \
                                       --sample_size $sample_size \
                                       --num_processes $num_processes

python dataset/unlabelled/make_image_timeseries.py --windows_dir $windows_dir \
                                              --savedir $timeseries_dir \
                                              --bands $bands \
                                              --res $res \
                                              --sample_size $sample_size \
                                              --num_processes $num_processes
