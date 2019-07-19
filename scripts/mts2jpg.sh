#!/bin/bash


cd data/duke/

# Get segment number
cd videos/camera$1/
seg_num=`ls -l . | egrep -c '^-'`
cd ../..

# Convert .mts to .jpg
mkdir frames/camera$1
for ((seg_id = 0; seg_id < seg_num; seg_id++))
do
    mkdir frames/camera$1/0000$seg_id
    ffmpeg -i videos/camera$1/0000$seg_id.mts -f image2 frames/camera$1/0000$seg_id/%d.jpg
done

# Rename .jpg files
cd frames/camera$1
img_id_bias=0
for ((seg_id = 0; seg_id < seg_num; seg_id++))
do
    cd 0000$seg_id
    img_num=0
    for name in *.jpg
    do
        img_id=$(echo $name | tr -dc '0-9')
        mv -i -v "${name}" "../$((img_id + img_id_bias)).jpg"
        img_num=$((img_num + 1))
    done
    cd ..
    rm -rf 0000$seg_id
    img_id_bias=$((img_id_bias + img_num))
done

