#!/usr/bin/env bash
x=1
while [ $x -le 1000 ]
do
  echo "Attempt $x"
  python download/sentinelsat_download_tileid.py --products_file $1
  x=$(( $x + 1 ))
  sleep 1800
done
