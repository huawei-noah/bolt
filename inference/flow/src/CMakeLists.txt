file(GLOB srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

include_directories(${Protobuf_INCLUDE_DIR})

protobuf_generate_cpp(FLOW_PROTO_SRCS FLOW_PROTO_HDRS flow.proto)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/include)
add_custom_target(flow.pb.h ALL
                  DEPENDS ${FLOW_PROTO_HDRS}
                  COMMAND ${CMAKE_COMMAND} -E copy ${FLOW_PROTO_HDRS} ${PROJECT_SOURCE_DIR}/include)

link_directories(${CMAKE_BINARY_DIR})
add_library(${PROJECT_NAME} SHARED ${srcs} ${FLOW_PROTO_HDRS} ${FLOW_PROTO_SRCS})
add_library(${PROJECT_NAME}_static STATIC ${srcs} ${FLOW_PROTO_HDRS} ${FLOW_PROTO_SRCS})
target_link_libraries(${PROJECT_NAME} LINK_PUBLIC bolt)
add_dependencies(${PROJECT_NAME} bolt_library)
if (BUILD_EXAMPLE)
    target_link_libraries(${PROJECT_NAME} LINK_PUBLIC ${JPEG_SHARED_LIBRARY})
endif ()
target_link_libraries(${PROJECT_NAME} LINK_PUBLIC ${Protobuf_SHARED_LIBRARY})
add_dependencies(${PROJECT_NAME} flow.pb.h)
add_dependencies(${PROJECT_NAME}_static flow.pb.h)
set_target_properties(${PROJECT_NAME}_static PROPERTIES OUTPUT_NAME "${PROJECT_NAME}")
set_target_properties(${PROJECT_NAME} PROPERTIES CLEAN_DIRECT_OUTPUT 1)
set_target_properties(${PROJECT_NAME}_static PROPERTIES CLEAN_DIRECT_OUTPUT 1)
install(TARGETS ${PROJECT_NAME} ${PROJECT_NAME}_static
        RUNTIME DESTINATION lib
        LIBRARY DESTINATION lib)
