#!/bin/bash

docker stop ubuntu1604-cv3-codebase
docker rm ubuntu1604-cv3-codebase
xhost local:root
docker run -it --rm -p 2201:22 -v ~/.ssh/id_rsa.pub:/root/.ssh/authorized_keys \
        --name=ubuntu1604-cv3-codebase \
        --privileged \
        -v ~/.vim:/root/.vim -v ~/.vimrc:/root/.vimrc \
        -v $(pwd)/codebase:/root/codebase \
	-v $(pwd)/gti/GTISDK_v3-0.tar.gz:/root/GTISDK_v3-0.tar.gz \
	-v $(pwd)/gti/FC_training_v1-0.tar.gz:/root/FC_training_v1-0.tar.gz \
        -e DISPLAY=unix$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
        ubuntu1604_opencv3_caffe
