// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef PROGRAM_H_
#define PROGRAM_H_

#include "file.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief get information of program
 * @warning please free memory associate with value
 **/
inline EE get_program_info(Program program, cl_program_info value_tag, void **value, U32 *value_len)
{
    if (NULL == value) {
        return NULL_POINTER;
    }
    size_t size;
    I32 ret = clGetProgramInfo(program, value_tag, 0, NULL, &size);
    if (CL_SUCCESS == ret) {
        if (value_len != NULL) {
            *value_len = size + 1;
        }
        char *data = (char *)malloc(size + 1);
        if (NULL == data) {
            return ALLOC_FAILED;
        }
        data[size] = '\0';
        ret = clGetProgramInfo(program, value_tag, size + 1, data, NULL);
        if (CL_SUCCESS == ret) {
            *value = data;
        } else {
            free(data);
        }
    }
    map_cl_error_2_ee(ret);
}

inline EE create_program_from_source(
    Context context, U32 *source_len, const char *source, Program *program)
{
    I32 ret;
    size_t length = *source_len;
    *program = clCreateProgramWithSource(context, 1, &source, &length, &ret);
    map_cl_error_2_ee(ret);
}

inline EE create_program_from_binary(Context context,
    const Device device,
    U32 *binary_len,
    const U8 **binary,
    I32 *binary_status,
    Program *program)
{
    I32 ret;
    size_t length = *binary_len;
    *program = clCreateProgramWithBinary(context, 1, &device, &length, binary, binary_status, &ret);
#ifdef _DEBUG
    if (CL_SUCCESS != ret) {
        const char *file = "gcl_error.bin";
        UNI_WARNING_LOG("can not create program from binary, write binary to file %s length %d.\n",
            file, *binary_len);
        CHECK_STATUS(save_binary(file, *binary, *binary_len));
    }
#endif
    map_cl_error_2_ee(ret);
}

/**
 * @brief get build information of program
 * @warning please free memory associate with value
 **/
inline void get_build_program_info(Program program, Device device)
{
    size_t len;
    I32 ret = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
    if (CL_SUCCESS == ret) {
        char *data = (char *)UNI_MALLOC(len + 1);
        data[len] = '\0';
        ret = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, len + 1, data, NULL);
        if (CL_SUCCESS == ret) {
            UNI_ERROR_LOG("clGetProgramBuildInfo length:%zu log:\n%s\n", len, data);
        } else {
            UNI_ERROR_LOG("can not get clGetProgramBuildInfo:%s.\n", map_cl_error_2_string(ret));
        }
        UNI_FREE(data);
    } else {
        UNI_ERROR_LOG("can not get clGetProgramBuildInfo:%s.\n", map_cl_error_2_string(ret));
    }
}

/**
 * @brief build program
 *
 **/
inline EE build_program(Program program, Device device, const char *options)
{
    UNI_DETAIL_LOG("build options:%s.\n", options);
    I32 ret = clBuildProgram(program, 1, &device, options, NULL, NULL);
    if (CL_SUCCESS != ret) {
        get_build_program_info(program, device);
    }
    map_cl_error_2_ee(ret);
}

/**
 * @brief create program from source then build it
 *
 * @param cont		    input, specify associate context
 * @param source	    input, source code
 * @param devices	    input, source will be built on devices
 * @param options	    input, options for compiling source
 * @param program		output, created and built program
 *
 */

inline EE create_build_program_from_source(Context context,
    U32 *source_len,
    const char *source,
    Device device,
    const char *options,
    Program *program)
{
    if (NULL == program) {
        return NULL_POINTER;
    }
    CHECK_STATUS(create_program_from_source(context, source_len, source, program));
    return build_program(*program, device, options);
}

/**
 * @brief create program from binary then build it
 *
 **/
inline EE create_build_program_from_binary(Context context,
    Device device,
    U32 *binary_len,
    const U8 **binary,
    const char *options,
    I32 *binary_status,
    Program *program)
{
    if (NULL == program) {
        return NULL_POINTER;
    }
    CHECK_STATUS(
        create_program_from_binary(context, device, binary_len, binary, binary_status, program));
    return build_program(*program, device, options);
}

/**
 * @brief get information of program
 * @warning please free memory associate with value
 **/
inline EE get_program_binary(Program program, U8 **binary, U32 *binary_len)
{
    size_t size;
    I32 ret = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &size, NULL);
    if (CL_SUCCESS == ret) {
        if (binary_len != NULL) {
            *binary_len = size;
        }
        char *data = (char *)malloc(size);
        if (NULL == data) {
            return ALLOC_FAILED;
        }
        ret = clGetProgramInfo(program, CL_PROGRAM_BINARIES, size, &data, NULL);
        if (CL_SUCCESS == ret) {
            *binary = (U8 *)data;
        } else {
            free(data);
        }
    }
    map_cl_error_2_ee(ret);
}

/*
   inline EE create_program_from_il(Context context,
        const void *il, U32 length, Program *program) {
   //TODO
    I32 ret;
 * program = clCreateProgramWithIL(context, il, length, &ret);
    map_cl_error_2_ee(ret);
   }
 */
inline EE release_program(Program program)
{
    map_cl_error_2_ee(clReleaseProgram(program));
}

#ifdef _WIN32
#define _USE_OPENCL_BUILD
#endif
inline EE compile_program(Program program,
    const Device device,
    const char *options,
    U32 num_input_headers,
    const Program *input_headers,
    const char **header_include_names)
{
    UNI_DETAIL_LOG("compile options:%s.\n", options);
#ifdef _USE_OPENCL_BUILD
    for (U32 i = 0; i < num_input_headers; i++) {
        if (file_exists(header_include_names[i])) {
            continue;
        }
        UNI_DETAIL_LOG("write OpenCL code to %s.\n", header_include_names[i]);
        char *content = NULL;
        CHECK_STATUS(get_program_info(input_headers[i], CL_PROGRAM_SOURCE, (void **)&content, NULL));
        if (content != NULL) {
            CHECK_STATUS(save_string(header_include_names[i], content));
            free(content);
        }
    }
    return build_program(program, device, (std::string(options) + std::string(" -I. ")).c_str());
#else
    I32 ret = clCompileProgram(program, 1, &device, options, num_input_headers, input_headers,
        header_include_names, NULL, NULL);
    if (CL_SUCCESS != ret) {
        get_build_program_info(program, device);
    }
    map_cl_error_2_ee(ret);
#endif
}

inline EE link_program(Context context,
    const Device device,
    const char *options,
    U32 num_input_programs,
    const Program *input_programs,
    Program *program)
{
    UNI_DETAIL_LOG("link options:%s.\n", options);
#ifdef _USE_OPENCL_BUILD
    CHECK_REQUIREMENT(num_input_programs == 1);
    *program = *input_programs;
    return SUCCESS;
#else
    I32 ret;
    *program = clLinkProgram(
        context, 1, &device, options, num_input_programs, input_programs, NULL, NULL, &ret);
    map_cl_error_2_ee(ret);
#endif
}

inline EE unload_platform_compiler(Platform p)
{
    I32 ret = clUnloadPlatformCompiler(p);
    map_cl_error_2_ee(ret);
}

#ifdef __cplusplus
}
#endif

#endif
