# BoxiEdge-Dev
## Steps:
### step1: Build Docker image with named "ubuntu1604_opencv3_caffe".
           (image name refer to docker_run*.sh)
           "$docker build --force-rm -t ubuntu1604_opencv3_caffe ."
### step2: Copy GTI's tar files into "gti/".
           (files name refer to docker_run*.sh)
### step3: Docker run the image.
### step4: In the container. un-tar GTI's tar files, and install sdk.
           (1. apt install libvtk6-dev )
           (2. Run install_gtisdk_core.sh)
           (3. Run install_gti_full_Samples.sh)
           (4. source ~/.bashrc && ldconfig)
