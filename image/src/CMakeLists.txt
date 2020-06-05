if (USE_GENERAL)
    file(GLOB general_srcs ${CMAKE_CURRENT_SOURCE_DIR}/cpu/general/*.cpp)
endif (USE_GENERAL)

if (USE_NEON)
    file(GLOB arm_srcs ${CMAKE_CURRENT_SOURCE_DIR}/cpu/arm/*.cpp)
endif (USE_NEON)

file(GLOB srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
set(srcs "${srcs};${general_srcs};${arm_srcs}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# shared library 
ADD_LIBRARY(${PROJECT_NAME} SHARED ${srcs}) 

# static library 
ADD_LIBRARY(${PROJECT_NAME}_static STATIC ${srcs}) 

SET_TARGET_PROPERTIES(${PROJECT_NAME}_static PROPERTIES OUTPUT_NAME "${PROJECT_NAME}") 
SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES CLEAN_DIRECT_OUTPUT 1) 
SET_TARGET_PROPERTIES(${PROJECT_NAME}_static PROPERTIES CLEAN_DIRECT_OUTPUT 1)
