file(GLOB_RECURSE srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# shared library 
ADD_LIBRARY(inference SHARED ${srcs}) 
# static library 
ADD_LIBRARY(inference_static STATIC ${srcs}) 
SET_TARGET_PROPERTIES(inference_static PROPERTIES OUTPUT_NAME "inference")
SET_TARGET_PROPERTIES(inference PROPERTIES CLEAN_DIRECT_OUTPUT 1) 
SET_TARGET_PROPERTIES(inference_static PROPERTIES CLEAN_DIRECT_OUTPUT 1)

