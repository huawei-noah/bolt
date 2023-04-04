# For more information about using CMake with Android Studio, read the

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.10.2)

# Declares and names the project.

project("noah")

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

add_library( # Sets the name of the library.
             native-lib

             # Sets the library as a shared library.
             SHARED

             # Provides a relative path to your source file(s).
             native-lib.cpp )

include_directories(
        ${CMAKE_SOURCE_DIR}/libbolt/headers
)

add_library(
        opencv_java4
        SHARED
        IMPORTED
)

set_target_properties(
        opencv_java4
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libopencv_java4.so
)


add_library(
        bolt
        STATIC
        IMPORTED
)

set_target_properties(
        bolt
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/libbolt/libbolt.a
)

add_library(
        opencv_calib3d
        STATIC
        IMPORTED
)

set_target_properties(
        opencv_calib3d
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/libbolt/libopencv_calib3d.a
)


add_library(
        opencv_core
        STATIC
        IMPORTED
)

set_target_properties(
        opencv_core
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/libbolt/libopencv_core.a
)

add_library(
        opencv_dnn
        STATIC
        IMPORTED
)

set_target_properties(
        opencv_dnn
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/libbolt/libopencv_dnn.a
)

add_library(
        opencv_features2d
        STATIC
        IMPORTED
)

set_target_properties(
        opencv_features2d
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/libbolt/libopencv_features2d.a
)

add_library(
        opencv_flann
        STATIC
        IMPORTED
)

set_target_properties(
        opencv_flann
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/libbolt/libopencv_flann.a
)

add_library(
        opencv_highgui
        STATIC
        IMPORTED
)

set_target_properties(
        opencv_highgui
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/libbolt/libopencv_highgui.a
)

add_library(
        opencv_imgproc
        STATIC
        IMPORTED
)

set_target_properties(
        opencv_imgproc
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/libbolt/libopencv_imgproc.a
)

add_library(
        opencv_imgcodecs
        STATIC
        IMPORTED
)
set_target_properties(
        opencv_imgcodecs
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/libbolt/libopencv_imgcodecs.a
)

add_library(
        opencv_ml
        STATIC
        IMPORTED
)

set_target_properties(
        opencv_ml
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/libbolt/libopencv_ml.a
)

add_library(
        opencv_objdetect
        STATIC
        IMPORTED
)

set_target_properties(
        opencv_objdetect
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/libbolt/libopencv_objdetect.a
)



add_library(
        opencv_photo
        STATIC
        IMPORTED
)

set_target_properties(
        opencv_photo
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/libbolt/libopencv_photo.a
)

add_library(
        opencv_stitching
        STATIC
        IMPORTED
)

set_target_properties(
        opencv_stitching
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/libbolt/libopencv_stitching.a
)


add_library(
        opencv_video
        STATIC
        IMPORTED
)

set_target_properties(
        opencv_video
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/libbolt/libopencv_video.a
)


add_library(
        opencv_videoio
        STATIC
        IMPORTED
)

set_target_properties(
        opencv_videoio
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/libbolt/libopencv_videoio.a
)
# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

find_library( # Sets the name of the path variable.
              log-lib

              # Specifies the name of the NDK library that
              # you want CMake to locate.
              log )

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries( # Specifies the target library.
                       native-lib
        jnigraphics

        opencv_java4
        bolt
        opencv_calib3d
        opencv_core
        opencv_dnn
        opencv_features2d
        opencv_flann
        opencv_highgui
        opencv_imgproc
        opencv_imgcodecs
        opencv_ml
        opencv_objdetect
        opencv_photo
        opencv_stitching
        opencv_video
        opencv_videoio

                       # Links the target library to the log library
                       # included in the NDK.
                       ${log-lib} )
