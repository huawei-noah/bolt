find_path(TFLITE_INCLUDE_DIR NAMES schema_generated.h HINTS ${TFLite_ROOT}/include $ENV{TFLite_ROOT}/include)

if (TFLITE_INCLUDE_DIR)
    set(TFLITE_FOUND true)
endif (TFLITE_INCLUDE_DIR)

if (TFLITE_FOUND)
    include_directories(include ${TFLITE_INCLUDE_DIR})
    message(STATUS "Found schema_generated.h: ${TFLITE_INCLUDE_DIR}")
else (TFLITE_FOUND)
    message(FATAL_ERROR "
FATAL: can not find schema_generated.h in <TFLite_ROOT>/include directory,
       please set shell environment variable TFLite_ROOT.
    ")
endif (TFLITE_FOUND)
