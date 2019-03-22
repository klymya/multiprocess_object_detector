#!/bin/sh

MODEL=faster_rcnn_inception_v2.tar.gz
mkdir model_data
cd ./model_data
if [ ! -f ./*.pb ]; then
    wget ./model_data/ https://www.dropbox.com/s/ej2a6rc0ozxw4fr/$MODEL
    tar -xvzf $MODEL
    rm $MODEL
fi