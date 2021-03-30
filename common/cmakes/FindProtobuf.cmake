if (Protobuf_FOUND)
    return()
endif (Protobuf_FOUND)

find_path(Protobuf_INCLUDE_DIR NAMES google/protobuf/service.h HINTS $ENV{Protobuf_ROOT}/include ${Protobuf_ROOT}/include)

if (USE_DYNAMIC_LIBRARY)
    find_library(Protobuf_LIBRARY NAMES protobuf HINTS $ENV{Protobuf_ROOT}/lib ${Protobuf_ROOT}/lib)
    set(Protobuf_SHARED_LIBRARY ${Protobuf_LIBRARY})
else (USE_DYNAMIC_LIBRARY)
    find_library(Protobuf_LIBRARY NAMES ${CMAKE_STATIC_LIBRARY_PREFIX}protobuf${CMAKE_STATIC_LIBRARY_SUFFIX} HINTS $ENV{Protobuf_ROOT}/lib ${Protobuf_ROOT}/lib)
    find_library(Protobuf_SHARED_LIBRARY NAMES protobuf SHARED_HINTS $ENV{Protobuf_ROOT}/lib ${Protobuf_ROOT}/lib)
endif (USE_DYNAMIC_LIBRARY)

find_program(Protobuf_PROTOC_EXECUTABLE NAMES protoc HINTS $ENV{Protobuf_ROOT}/bin ${Protobuf_ROOT}/bin)

#set(Protobuf_DEBUG ON)
if (Protobuf_INCLUDE_DIR AND Protobuf_LIBRARY AND Protobuf_PROTOC_EXECUTABLE)
    set(_PROTOBUF_COMMON_HEADER ${Protobuf_INCLUDE_DIR}/google/protobuf/stubs/common.h)
    set(Protobuf_VERSION "")
    set(Protobuf_LIB_VERSION "")
    file(STRINGS ${_PROTOBUF_COMMON_HEADER} _PROTOBUF_COMMON_H_CONTENTS REGEX "#define[ \t]+GOOGLE_PROTOBUF_VERSION[ \t]+")
    if(_PROTOBUF_COMMON_H_CONTENTS MATCHES "#define[ \t]+GOOGLE_PROTOBUF_VERSION[ \t]+([0-9]+)")
        set(Protobuf_LIB_VERSION "${CMAKE_MATCH_1}")
    endif()
    unset(_PROTOBUF_COMMON_H_CONTENTS)
    math(EXPR _PROTOBUF_MAJOR_VERSION "${Protobuf_LIB_VERSION} / 1000000")
    math(EXPR _PROTOBUF_MINOR_VERSION "${Protobuf_LIB_VERSION} / 1000 % 1000")
    math(EXPR _PROTOBUF_SUBMINOR_VERSION "${Protobuf_LIB_VERSION} % 1000")
    set(Protobuf_VERSION "${_PROTOBUF_MAJOR_VERSION}.${_PROTOBUF_MINOR_VERSION}.${_PROTOBUF_SUBMINOR_VERSION}")
    if(Protobuf_DEBUG)
        message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
            "${_PROTOBUF_COMMON_HEADER} reveals protobuf ${Protobuf_VERSION}")
    endif()

    # Check Protobuf compiler version to be aligned with libraries version
    execute_process(COMMAND ${Protobuf_PROTOC_EXECUTABLE} --version
                    OUTPUT_VARIABLE _PROTOBUF_PROTOC_EXECUTABLE_VERSION)
    if("${_PROTOBUF_PROTOC_EXECUTABLE_VERSION}" MATCHES "libprotoc ([0-9.]+)")
      set(_PROTOBUF_PROTOC_EXECUTABLE_VERSION "${CMAKE_MATCH_1}")
    endif()
    if(Protobuf_DEBUG)
      message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
          "${Protobuf_PROTOC_EXECUTABLE} reveals version ${_PROTOBUF_PROTOC_EXECUTABLE_VERSION}")
    endif()
  
    if(NOT "${_PROTOBUF_PROTOC_EXECUTABLE_VERSION}" VERSION_EQUAL "${Protobuf_VERSION}")
        message(WARNING "Protobuf compiler version ${_PROTOBUF_PROTOC_EXECUTABLE_VERSION}"
            " doesn't match library version ${Protobuf_VERSION}")
    endif()

    set(Protobuf_FOUND true)
endif (Protobuf_INCLUDE_DIR AND Protobuf_LIBRARY AND Protobuf_PROTOC_EXECUTABLE)

if (Protobuf_FOUND)
    message(STATUS "Found Protobuf: ${Protobuf_LIBRARY}")
else (Protobuf_FOUND)
    message(FATAL_ERROR "
FATAL: can not find protobuf library in <Protobuf_ROOT>/[include/lib] directory,
       please set shell environment variable Protobuf_ROOT.
    ")
endif (Protobuf_FOUND)

function(protobuf_generate)
    set(_options APPEND_PATH DESCRIPTORS)
    set(_singleargs LANGUAGE OUT_VAR EXPORT_MACRO PROTOC_OUT_DIR)
    if(COMMAND target_sources)
        list(APPEND _singleargs TARGET)
    endif()
    set(_multiargs PROTOS IMPORT_DIRS GENERATE_EXTENSIONS)

    cmake_parse_arguments(protobuf_generate "${_options}" "${_singleargs}" "${_multiargs}" "${ARGN}")

    if(NOT protobuf_generate_PROTOS AND NOT protobuf_generate_TARGET)
        message(SEND_ERROR "Error: protobuf_generate called without any targets or source files")
        return()
    endif()

    if(NOT protobuf_generate_OUT_VAR AND NOT protobuf_generate_TARGET)
        message(SEND_ERROR "Error: protobuf_generate called without a target or output variable")
        return()
    endif()

    if(NOT protobuf_generate_LANGUAGE)
        set(protobuf_generate_LANGUAGE cpp)
    endif()
    string(TOLOWER ${protobuf_generate_LANGUAGE} protobuf_generate_LANGUAGE)

    if(NOT protobuf_generate_PROTOC_OUT_DIR)
        set(protobuf_generate_PROTOC_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
    endif()

    if(protobuf_generate_EXPORT_MACRO AND protobuf_generate_LANGUAGE STREQUAL cpp)
        set(_dll_export_decl "dllexport_decl=${protobuf_generate_EXPORT_MACRO}:")
    endif()

    if(NOT protobuf_generate_GENERATE_EXTENSIONS)
        if(protobuf_generate_LANGUAGE STREQUAL cpp)
            set(protobuf_generate_GENERATE_EXTENSIONS .pb.h .pb.cc)
        elseif(protobuf_generate_LANGUAGE STREQUAL python)
            set(protobuf_generate_GENERATE_EXTENSIONS _pb2.py)
        else()
            message(SEND_ERROR "Error: protobuf_generate given unknown Language ${LANGUAGE}, please provide a value for GENERATE_EXTENSIONS")
            return()
        endif()
    endif()

    if(protobuf_generate_TARGET)
        get_target_property(_source_list ${protobuf_generate_TARGET} SOURCES)
        foreach(_file ${_source_list})
            if(_file MATCHES "proto$")
                list(APPEND protobuf_generate_PROTOS ${_file})
            endif()
        endforeach()
    endif()

    if(NOT protobuf_generate_PROTOS)
        message(SEND_ERROR "Error: protobuf_generate could not find any .proto files")
        return()
    endif()

    if(protobuf_generate_APPEND_PATH)
        # Create an include path for each file specified
        foreach(_file ${protobuf_generate_PROTOS})
            get_filename_component(_abs_file ${_file} ABSOLUTE)
            get_filename_component(_abs_path ${_abs_file} PATH)
            list(FIND _protobuf_include_path ${_abs_path} _contains_already)
            if(${_contains_already} EQUAL -1)
                    list(APPEND _protobuf_include_path -I ${_abs_path})
            endif()
        endforeach()
    else()
        set(_protobuf_include_path -I ${CMAKE_CURRENT_SOURCE_DIR})
    endif()

    foreach(DIR ${protobuf_generate_IMPORT_DIRS})
        get_filename_component(ABS_PATH ${DIR} ABSOLUTE)
        list(FIND _protobuf_include_path ${ABS_PATH} _contains_already)
        if(${_contains_already} EQUAL -1)
                list(APPEND _protobuf_include_path -I ${ABS_PATH})
        endif()
    endforeach()

    set(_generated_srcs_all)
    foreach(_proto ${protobuf_generate_PROTOS})
        get_filename_component(_abs_file ${_proto} ABSOLUTE)
        get_filename_component(_abs_dir ${_abs_file} DIRECTORY)
        get_filename_component(_basename ${_proto} NAME_WE)
        file(RELATIVE_PATH _rel_dir ${CMAKE_CURRENT_SOURCE_DIR} ${_abs_dir})

        set(_possible_rel_dir)
        if (NOT protobuf_generate_APPEND_PATH)
                set(_possible_rel_dir ${_rel_dir}/)
        endif()

        set(_generated_srcs)
        foreach(_ext ${protobuf_generate_GENERATE_EXTENSIONS})
            #list(APPEND _generated_srcs "${protobuf_generate_PROTOC_OUT_DIR}/${_possible_rel_dir}${_basename}${_ext}")
            list(APPEND _generated_srcs "${protobuf_generate_PROTOC_OUT_DIR}/${_basename}${_ext}")
        endforeach()

        if(protobuf_generate_DESCRIPTORS AND protobuf_generate_LANGUAGE STREQUAL cpp)
            set(_descriptor_file "${CMAKE_CURRENT_BINARY_DIR}/${_basename}.desc")
            set(_dll_desc_out "--descriptor_set_out=${_descriptor_file}")
            list(APPEND _generated_srcs ${_descriptor_file})
        endif()
        list(APPEND _generated_srcs_all ${_generated_srcs})

        add_custom_command(
            OUTPUT ${_generated_srcs}
            COMMAND    protobuf::protoc
            ARGS --${protobuf_generate_LANGUAGE}_out ${_dll_export_decl}${protobuf_generate_PROTOC_OUT_DIR} ${_dll_desc_out} ${_protobuf_include_path} ${_abs_file}
            DEPENDS ${_abs_file} protobuf::protoc
            COMMENT "Running ${protobuf_generate_LANGUAGE} protocol buffer compiler on ${_proto}"
            VERBATIM )
    endforeach()

    set_source_files_properties(${_generated_srcs_all} PROPERTIES GENERATED TRUE)
    if(protobuf_generate_OUT_VAR)
        set(${protobuf_generate_OUT_VAR} ${_generated_srcs_all} PARENT_SCOPE)
    endif()
    if(protobuf_generate_TARGET)
        target_sources(${protobuf_generate_TARGET} PRIVATE ${_generated_srcs_all})
    endif()
endfunction()

function(PROTOBUF_GENERATE_CPP SRCS HDRS)
    cmake_parse_arguments(protobuf_generate_cpp "" "EXPORT_MACRO;DESCRIPTORS" "" ${ARGN})

    set(_proto_files "${protobuf_generate_cpp_UNPARSED_ARGUMENTS}")
    if(NOT _proto_files)
        message(SEND_ERROR "Error: PROTOBUF_GENERATE_CPP() called without any proto files")
        return()
    endif()

    if(PROTOBUF_GENERATE_CPP_APPEND_PATH)
        set(_append_arg APPEND_PATH)
    endif()

    if(protobuf_generate_cpp_DESCRIPTORS)
        set(_descriptors DESCRIPTORS)
    endif()

    if(DEFINED PROTOBUF_IMPORT_DIRS AND NOT DEFINED Protobuf_IMPORT_DIRS)
        set(Protobuf_IMPORT_DIRS "${PROTOBUF_IMPORT_DIRS}")
    endif()

    if(DEFINED Protobuf_IMPORT_DIRS)
        set(_import_arg IMPORT_DIRS ${Protobuf_IMPORT_DIRS})
    endif()

    set(_outvar)
    protobuf_generate(${_append_arg} ${_descriptors} LANGUAGE cpp EXPORT_MACRO ${protobuf_generate_cpp_EXPORT_MACRO} OUT_VAR _outvar ${_import_arg} PROTOS ${_proto_files})

    set(${SRCS})
    set(${HDRS})
    if(protobuf_generate_cpp_DESCRIPTORS)
        set(${protobuf_generate_cpp_DESCRIPTORS})
    endif()

    foreach(_file ${_outvar})
        if(_file MATCHES "cc$")
            list(APPEND ${SRCS} ${_file})
        elseif(_file MATCHES "desc$")
            list(APPEND ${protobuf_generate_cpp_DESCRIPTORS} ${_file})
        else()
            list(APPEND ${HDRS} ${_file})
        endif()
    endforeach()
    set(${SRCS} ${${SRCS}} PARENT_SCOPE)
    set(${HDRS} ${${HDRS}} PARENT_SCOPE)
    if(protobuf_generate_cpp_DESCRIPTORS)
        set(${protobuf_generate_cpp_DESCRIPTORS} "${${protobuf_generate_cpp_DESCRIPTORS}}" PARENT_SCOPE)
    endif()
endfunction()


function(PROTOBUF_GENERATE_PYTHON SRCS)
    if(NOT ARGN)
        message(SEND_ERROR "Error: PROTOBUF_GENERATE_PYTHON() called without any proto files")
        return()
    endif()

    if(PROTOBUF_GENERATE_CPP_APPEND_PATH)
        set(_append_arg APPEND_PATH)
    endif()

    if(DEFINED PROTOBUF_IMPORT_DIRS AND NOT DEFINED Protobuf_IMPORT_DIRS)
        set(Protobuf_IMPORT_DIRS "${PROTOBUF_IMPORT_DIRS}")
    endif()

    if(DEFINED Protobuf_IMPORT_DIRS)
        set(_import_arg IMPORT_DIRS ${Protobuf_IMPORT_DIRS})
    endif()

    set(_outvar)
    protobuf_generate(${_append_arg} LANGUAGE python OUT_VAR _outvar ${_import_arg} PROTOS ${ARGN})
    set(${SRCS} ${_outvar} PARENT_SCOPE)
endfunction()


if(Protobuf_INCLUDE_DIR)
    if(Protobuf_PROTOC_EXECUTABLE)
        if(NOT TARGET protobuf::protoc)
            add_executable(protobuf::protoc IMPORTED)
            if(EXISTS "${Protobuf_PROTOC_EXECUTABLE}")
                set_target_properties(protobuf::protoc PROPERTIES
                    IMPORTED_LOCATION "${Protobuf_PROTOC_EXECUTABLE}")
            endif()
        endif()
    endif()
endif()

# Backwards compatibility
# Define upper case versions of output variables
#foreach(Camel
#        Protobuf_SRC_ROOT_FOLDER
#        Protobuf_IMPORT_DIRS
#        Protobuf_DEBUG
#        Protobuf_INCLUDE_DIRS
#        Protobuf_LIBRARIES
#        Protobuf_PROTOC_LIBRARIES
#        Protobuf_LITE_LIBRARIES
#        Protobuf_LIBRARY
#        Protobuf_PROTOC_LIBRARY
#        Protobuf_INCLUDE_DIR
#        Protobuf_PROTOC_EXECUTABLE
#        Protobuf_LIBRARY_DEBUG
#        Protobuf_PROTOC_LIBRARY_DEBUG
#        Protobuf_LITE_LIBRARY
#        Protobuf_LITE_LIBRARY_DEBUG
#        )
#        string(TOUPPER ${Camel} UPPER)
#        set(${UPPER} ${${Camel}})
#endforeach()
#
