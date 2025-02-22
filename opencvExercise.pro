######################################################################
# Automatically generated by qmake (2.01a) Wed Feb 19 18:14:37 2014
######################################################################

TEMPLATE = app
TARGET = 
DEPENDPATH += . 
INCLUDEPATH += . 

# Input
HEADERS += misc/timeProcess.h misc/_io_.h misc/directory.h \
    util.h \
    camshiftkalman.h

SOURCES += coreModule.cpp \
           magicBarTool.cpp \
    detectCircle.cpp \
    objectTracking_.cpp \
    camshiftkalman.cpp

INCLUDEPATH += /opt/opencv-2.4.5/include/opencv /opt/opencv-2.4.5/include
LIBS   += -L/opt/opencv-2.4.5/lib -lopencv_calib3d -lopencv_contrib -lopencv_core -lopencv_features2d\
 -lopencv_flann -lopencv_highgui -lopencv_imgproc -lopencv_legacy -lopencv_ml -lopencv_nonfree \
-lopencv_objdetect -lopencv_photo -lopencv_stitching -lopencv_superres -lopencv_ts -lopencv_video -lopencv_videostab
