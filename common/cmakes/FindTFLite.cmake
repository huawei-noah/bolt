find_path(TFLITE_INCLUDE_DIR NAMES tensorflow/lite/schema/schema_generated.h HINTS $ENV{TFLite_ROOT}/include ${TFLite_ROOT}/include)

if (TFLITE_INCLUDE_DIR)
    set(TFLITE_FOUND true)
endif (TFLITE_INCLUDE_DIR)
find_package(FlatBuffers)

if (TFLITE_FOUND)
    message(STATUS "Found tensorflow/lite/schema/schema_generated.h: ${TFLITE_INCLUDE_DIR}")
    set(TFLITE_INCLUDE_DIR "${TFLITE_INCLUDE_DIR};${FlatBuffers_INCLUDE_DIR}")
else (TFLITE_FOUND)
    message(FATAL_ERROR "
FATAL: can not find tensorflow/lite/schema/schema_generated.h in <TFLite_ROOT>/include directory,
       please set shell environment variable TFLite_ROOT.
    ")
endif (TFLITE_FOUND)
