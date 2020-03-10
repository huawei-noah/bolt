find_path(FlatBuffers_INCLUDE_DIR NAMES flatbuffers/flatbuffers.h HINTS ${FlatBuffers_ROOT}/include
    $ENV{FlatBuffers_ROOT}/include
    /usr/local/include)

if (FlatBuffers_INCLUDE_DIR)
    set(FLAT_BUFFERS_FOUND true)
endif (FlatBuffers_INCLUDE_DIR)

if (FLAT_BUFFERS_FOUND)
    message(STATUS "Found flatbuffers/flatbuffers.h: ${FlatBuffers_INCLUDE_DIR}")
else (FLAT_BUFFERS_FOUND)
    message(FATAL_ERROR "
FATAL: can not find flatbuffers/flatbuffers.h in <FlatBuffers_ROOT>/include directory,
       please set shell environment variable FlatBuffers_ROOT.
    ")
endif (FLAT_BUFFERS_FOUND)
