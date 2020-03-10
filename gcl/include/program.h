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

#ifdef __cplusplus
extern "C" {
#endif

    /**
     * @brief create program from source code
     *
     * @param context		    input, specify associate context
     * @param source	    input, source code
     * @param program		output, created and built program
     *
     **/
    inline EE create_program_from_source(Context context, U32* len, CI8* str, Program *program) {
        I32 ret;
        size_t length = (size_t)(*len);
        *program = clCreateProgramWithSource(context, 1, &str, &length, &ret);
        map_cl_error_2_ee(ret);
    }

    /**
     * @brief create program from binary code
     *
     * @param context		        input, specify associate context
     * @param numDevices        input, the number of devices need to compile the
     * code for
     * @param devices	        input, devices need to compile the code for
     * @param lengths	        input, 
     * @param binaries
     * @param binary_status     output, compiled status for every devices  
     * @param program		    output, created and built program
     *
     **/

    inline EE create_program_from_binary(Context context, const Device device, 
            U32* length, CU8 **binary, I32 *binary_status, Program *program) {
        I32 ret; 
        size_t len = *length;
        *program = clCreateProgramWithBinary(context, 1, &device, &len, binary, binary_status, &ret);
        map_cl_error_2_ee(ret);
    }

    /**
     * @brief build program
     *
     **/

    inline EE build_program(Program program, Device device, CI8 *options) {
        I32 ret = clBuildProgram(program, 1, &device, options, NULL, NULL);
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
    inline EE create_build_program_from_source(Context context, U32* length, CI8* source, Device device, CI8* options, Program *program) {
        if(NULL == program) return NULL_POINTER;

        Program prog;
        EE ret = create_program_from_source(context, length, source, &prog);
        if(SUCCESS == ret) {
            ret = build_program(prog, device, options);                   
            *program = prog;
        }

        map_cl_error_2_ee(ret);
    }

    /**
     * @brief get build information of program
     * @warning please free memory associate with value
     **/

    inline EE get_program_build_info(Program program,
            Device device,
            cl_program_build_info info,
            void* *value, U32 *size) {
        if(NULL == value) return NULL_POINTER;

        size_t len;
        I32 ret = clGetProgramBuildInfo(program, device, info, 0, NULL, &len);
        if(SUCCESS == ret) {
            if(NULL == size) *size = len;
            void* data = malloc(len);
            if(NULL == data) return ALLOC_FAILED;
            ret = clGetProgramBuildInfo(program, device, info, len, data, NULL);
            if(SUCCESS == ret) { *value = data; } else { free(data); }
        }

        map_cl_error_2_ee(ret);
    }

#define check_build_program_error(ret, program, device) {\
    if(SUCCESS != ret){\
        void* buildLog; \
        U32 buildLogSize;\
        ret = get_program_build_info(program, device, CL_PROGRAM_BUILD_LOG, &buildLog, &buildLogSize);\
        if(SUCCESS == ret) { \
            printf("build log of device %s\n", (char*)buildLog);\
            free(buildLog);\
        }\
    }\
}\

/**
 * @brief create program from binary then build it
 *
 **/
inline EE create_build_program_from_binary(Context context, Device device, 
        U32* length, CU8** binary, CI8* options, I32 *binary_status, Program *program) {
    if(NULL == program) return NULL_POINTER;

    Program prog;
    EE ret = create_program_from_binary(context, device, length, binary, binary_status, &prog);
    if(SUCCESS == ret) {
        ret = build_program(prog, device, options);                   
        check_build_program_error(ret, prog, device);

        *program = prog;
    }

    map_cl_error_2_ee(ret);
}

/**
 * @brief get information of program
 * @warning please free memory associate with value
 **/

inline EE get_program_info(Program program,
        cl_program_info info,
        void* *value, U32 *size) {
    if(NULL == value) return NULL_POINTER;

    size_t len;
    I32 ret = clGetProgramInfo(program, info, 0, NULL, &len);
    if(CL_SUCCESS == ret) {
        if(NULL != size) *size = len;
        void* data = malloc(len);
        if(NULL == data) return ALLOC_FAILED;
        ret = clGetProgramInfo (program, info, len, data, NULL);
        if(CL_SUCCESS == ret) { *value = data;} else { free(data); }
    }

    map_cl_error_2_ee(ret);
}
/**
 * @brief get information of program
 * @warning please free memory associate with value
 **/
inline EE get_program_binary(Program program, U8* *binary, U32 *len) {
    size_t size;
    I32 ret = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &size, NULL);
    if(CL_SUCCESS == ret){
        *len  = (U32)(size);
        void*data = malloc(size);
        if(NULL == data) return ALLOC_FAILED;
	ret = clGetProgramInfo(program, CL_PROGRAM_BINARIES, size, &data, NULL);//waring: need set &data
        if(CL_SUCCESS == ret ){*binary = (U8*)(data);}
        else{free(data);}
    }
    map_cl_error_2_ee(ret);
}
/**
 * @brief get binary of source code
 *
 * @warning please don't free binary, it is return by ocl
 *
 **/

inline EE get_program_binary_from_source(Context context, U32* length, CI8* str, Device device, CI8* options, U8* *binary, U32 *len) {
    if(NULL == binary) return NULL_POINTER;

    Program program; 
    EE ret = create_build_program_from_source(context, length, str, device, options, &program);
    if(SUCCESS == ret) { ret = get_program_binary(program, binary, len); }
    return ret;
}
/*
inline EE create_program_from_il(Context context,
        const void *il, U32 length, Program *program) {
//TODO
    I32 ret;
    *program = clCreateProgramWithIL(context, il, length, &ret);
    map_cl_error_2_ee(ret);
}
*/
inline EE release_program(Program program) {
    map_cl_error_2_ee(clReleaseProgram(program));
}

inline EE compile_program(Program program,
        const Device device,
        CI8 *options, U32 num_input_headers, const Program *input_headers,
        CI8 **header_include_names) {
    I32 ret = clCompileProgram(program, 1, &device,
            options, num_input_headers, input_headers, header_include_names,
            NULL, NULL);
    map_cl_error_2_ee(ret);
}

inline EE link_program(Context context, 
        const Device device,
        CI8* options, U32 num_input_programs,
        const Program *input_programs, Program* program) {
    I32 ret;
    *program = clLinkProgram(context, 1, &device, options, num_input_programs, input_programs,
            NULL, NULL, &ret);
    map_cl_error_2_ee(ret);
}

inline EE unload_platform_compiler(Platform p) {
    I32 ret = clUnloadPlatformCompiler(p);
    map_cl_error_2_ee(ret);
}

#ifdef __cplusplus
}
#endif

#endif
