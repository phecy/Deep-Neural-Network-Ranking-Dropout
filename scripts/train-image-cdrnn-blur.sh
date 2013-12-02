#!/bin/bash

python ../train-ranknet/convnet.py --data-path=/mnt/iscsi_3/yuefeng/quality_for_clarity_14w/batches/  --test-range=50 --train-range=1-49 --layer-def=./conv_ranknet_cfg/imagenet.full.cfg  --layer-params=./conv_ranknet_cfg/params-init.full.cfg --data-provider=pair --test-freq=5 --gpu=3 --epochs=1000 --save-path=./model.clarity.20131129.new --crop-border=16 

#python ../train-ranknet/convnet.py -f  ./model.clarity.20131130/34.18 --layer-params=./conv_ranknet_cfg/params-init.full.1.cfg  --gpu=2 --save-path=./model.clarity.20131201/
#python ../train-ranknet/convnet.py -f ./model.image-quality.20130704/model.image-quality.20130704/199.1 --save-path=./model.image-quality.20130704 --epochs=300
