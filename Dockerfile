FROM ubuntu:16.04

#Version v1.0
#Changelog:
#	1. ISSUE: AttributeError: 'module' object has no attribute 'SSL_ST_INIT'
#	   1.1. install pyOpenSSL==16.2.0


MAINTAINER CHIN SUN <nhoj62003@gmail.com>


# Install Enseentials 
RUN apt-get update \
    && apt-get install -y software-properties-common python-software-properties openssh-server \
    && apt-get install -y wget vim htop git curl unzip

# Install OpenCV dependency lib
RUN apt-get update \
    && apt-get install -y python-pip \
    && wget https://bootstrap.pypa.io/get-pip.py \
    && python get-pip.py \
    && rm get-pip.py \
    && apt-get install -y python-dev \
    python-numpy \
    python-scipy \
    build-essential \
    cmake \
    pkg-config \
    libatlas-base-dev \
    gfortran \
    libjasper-dev \
    libgtk2.0-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libv4l-dev \
    v4l-utils \
    python-opencv \
    python-skimage \
    libjpeg8-dev \
    libtiff5-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    && apt-get install -y libgtkglext1-dev \
    libvtk6-dev

# Install CAFFE dependency lib
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    libatlas-base-dev \
    libboost-all-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    libhdf5-serial-dev \
    libleveldb-dev \
    liblmdb-dev \
    libopencv-dev \
    libprotobuf-dev \
    libsnappy-dev \
    protobuf-compiler \
    python-dev \
    python-numpy \
    python-pip \
    python-setuptools \
    python-scipy

# Install dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    openssh-server \
    cmake \
    git \
    libboost-all-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    libprotobuf-dev \
    pkg-config \
    protobuf-compiler \
    python-yaml \
    wget


# Install OpenCV 3.2.0 with CUDA support
#ENV OPENCV_VER "2.4.13.4"
ENV OPENCV_VER "3.4.1"
RUN wget https://github.com/opencv/opencv/archive/"${OPENCV_VER}".tar.gz \
    && tar xzf "${OPENCV_VER}".tar.gz \
    && cd opencv-"${OPENCV_VER}" \
    && mkdir build \
    && cd build \
    && cmake -D CMAKE_BUILD_TYPE=RELEASE \
       -DCMAKE_INSTALL_PREFIX=/usr/local \
       -DWITH_JPEG=ON -DBUILD_JPEG=ON -DWITH_PNG=ON -DBUILD_PNG=ON \
       -DENABLE_FAST_MATH=ON -DCUDA_FAST_MATH=ON \
       -DWITH_CUBLAS=ON \
       -DWITH_GTK=ON .. \
       -DWITH_OPENGL=ON \
       -DFORCE_VTK=ON \
       -DWITH_TBB=ON \
       -DWITH_GDAL=ON \
       -DWITH_XINE=ON \
       -DBUILD_EXAMPLES=OFF \
       -DENABLE_PRECOMPILED_HEADERS=OFF \
       -DBUILD_opencv_ts=OFF \
       -DBUILD_SHARED_LIBS=ON \
       -DBUILD_NEW_PYTHON_SUPPORT=ON \
       -DPYTHON_EXECUTABLE=/usr/bin/python \
    && make -j"$(nproc)" && make install && cd ../.. && rm -r opencv-"${OPENCV_VER}" && rm "${OPENCV_VER}".tar.gz


# Install Caffe Python dependency lib
RUN pip install scikit-image ipython h5py leveldb networkx nose pandas python-dateutil protobuf python-gflags pyyaml Pillow six pydot
ENV CAFFE_ROOT=/root/caffe
WORKDIR $CAFFE_ROOT
#ENV CAFFE_TAG=1.0
#RUN git clone -b ${CAFFE_TAG} --depth 1 https://github.com/BVLC/caffe.git . && \
RUN git clone https://github.com/BVLC/caffe.git . && \
    pip install --upgrade pip && \
    pip install  pyOpenSSL==16.2.0 && \
    cd python && for req in $(cat requirements.txt) pydot; do pip install $req; done && cd .. && \
    cp Makefile.config.example Makefile.config && \
    sed -i 's/# USE_OPENCV := 0/USE_OPENCV := 1/g' Makefile.config && \
    pip freeze --local | grep -v '^\-e' | cut -d = -f 1 > requirements.txt && \
    sed requirements.txt && \
    pip install --upgrade setuptools && \
    for req in $(cat requirements.txt); do pip install -U $req; done && \
    rm requirements.txt
    
ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

# Install exuberant-ctags matplot
RUN apt-get install -y exuberant-ctags python-matplotlib

RUN apt-get update 

# install desktop environment
#RUN sudo add-apt-repository -y ppa:lubuntu-dev/lubuntu-daily && \
RUN apt-get install --reinstall ca-certificates && \ 
    add-apt-repository ppa:ubuntu-mate-dev/xenial-mate && \
    apt-get update && apt-get upgrade -y && \
    apt-get install mate -y && \
    apt-get dist-upgrade
#    sudo apt-get install lxqt-metapackage lxqt-panel openbox -y

# install lsusb
RUN apt-get install usbutils

RUN rm -rf /var/lib/apt/lists/*

# write script to build caffe
RUN cd /root && \
    mkdir caffe_bin && \
    cd caffe_bin && \
    echo "#!/bin/sh" >> build.sh && \
    echo "cmake -DUSE_OPENCV=ON /root/caffe" >> build.sh && \
    echo "make all -j$(nproc)" >> build.sh && \
    echo "./tools/caffe time -model=/root/caffe/models/bvlc_alexnet/deploy.prototxt -gpu=all" >> build.sh && \
    chmod a+x build.sh
WORKDIR /root

# Set SSH Config
RUN mkdir -p /var/run/sshd

EXPOSE 22

# Start sshd
CMD /usr/sbin/sshd && /bin/bash
