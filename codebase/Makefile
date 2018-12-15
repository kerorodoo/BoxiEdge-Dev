CAFFE_HOME = /root/caffe-yolo
export LIBFTDI_PATH = $(GTISDKROOT)/Lib

SRCS += Queue.h stdafx.cpp

CAFFE_OBJS = $(CAFFE_HOME)/build/src/caffe/*.o $(CAFFE_HOME)/build/src/caffe/*/*.o
OBJS += $(CAFFE_OBJS)

CC = g++

CFLAGS += -std=c++11
CFLAGS += -DUSE_PCIE -DCPU_ONLY -DUSE_OPENCV

CAFFE_INCLUDES += -I$(CAFFE_HOME)/include -I$(CAFFE_HOME)/.build_release/src
INCS += -I/usr/local/include/gti/ $(CAFFE_INCLUDES)

LIBRARYS += -L/usr/local/lib -L$(LIBFTDI_PATH)
OPENCV_LIBS += -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_videoio -lopencv_objdetect
CAFFE_LIBS += -lglog -lgflags -lprotobuf -lboost_system -lboost_filesystem -lboost_thread -lm -lhdf5_serial_hl -lhdf5_serial -lopenblas -lleveldb -lsnappy -llmdb
GTI_LIBS += -lGTILibrary-static -lftd3xx

# with caffe python
CAFFE_LIBS += -lboost_python -lpython2.7

# only x86 lib
GTI_LIBS += -llightspr_cnn

LIBS += -lpthread $(LIBRARYS) $(OPENCV_LIBS) $(CAFFE_LIBS) $(GTI_LIBS)

all: test

gticlassify.o: Classify.cpp
	$(CC) $< $(INCS) $(LIBS) $(CFLAGS) -c -o $@
test: main.cpp gticlassify.o $(OBJS)
	$(CC) $? $(INCS) $(LIBS) $(CFLAGS) -o $@

clean:
	rm -rf *.o *.gch test
