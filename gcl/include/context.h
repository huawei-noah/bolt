// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.



#ifndef _H_CONTEXT
#define _H_CONTEXT

#ifdef __cplusplus
extern "C" {
#endif

    /**
     * @brief create OpenCL Context based on platform
     *
     * @param platform		input, context will be created on this platform
     * @param num_devices 	input, context will be created on num_devices Device
     * @param devices	input, context created contains devices
     * @param context		output, return context created
     *
     * @return
     *
     */
    inline EE create_context(Platform platform,
            U32 num_devices, Device *devices,
            Context *context) {
        if(NULL == context) return NULL_POINTER;

        I32 ret;
        cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0};
        *context = clCreateContext(properties, num_devices, devices, NULL, NULL, &ret);
        map_cl_error_2_ee(ret);
    }


    /**
     * @brief get context information
     *
     * @warning please free the memory allocate by this function
     **/
    inline EE get_context_info(Context context, cl_context_info info,
            void** value, U32 *len) {
        if(NULL == value) return NULL_POINTER;

        size_t size;
        I32 ret = clGetContextInfo(context, info, 0, NULL, &size);
        if(CL_SUCCESS == ret) {
            if(NULL == len) *len = size;
            void* data = malloc(size);
            if(NULL == data) return ALLOC_FAILED;
            ret = clGetContextInfo(context, info, size, data, NULL);
            if(CL_SUCCESS == ret) { *value = data; } else { free(data); }
        }

        map_cl_error_2_ee(ret);
    }

    inline EE retain_context(Context context) {
        I32 ret = clRetainContext(context);
        map_cl_error_2_ee(ret);
    }

    inline EE release_context(Context context) {
        I32 ret = clReleaseContext(context);
        map_cl_error_2_ee(ret);
    }

    inline EE create_command_queue_properties(Context context, Device device,
            cl_queue_properties* properties, CommandQueue* queue) {
        if(NULL == queue) return NULL_POINTER;
        I32 ret;
        *queue = clCreateCommandQueueWithProperties(context, device, properties, &ret);
        map_cl_error_2_ee(ret);
    }
/*
    inline EE create_command_queue(Context context, Device device,
            cl_command_queue_properties properties, CommandQueue* queue) {
        if(NULL == queue) return NULL_POINTER;
        I32 ret;
        *queue = clCreateCommandQueue(context, device, properties, &ret);
        map_cl_error_2_ee(ret);
    }
*/    
    /**
     * @brief get information of command queue
     *
     * @warning please free memory associated with value
     *
     **/
    inline EE get_command_queue_info(CommandQueue queue,
            cl_command_queue_info info,
            void** value, size_t *len) {
        if(NULL == value) return NULL_POINTER;

        size_t size;
        I32 ret = clGetCommandQueueInfo(queue, info, 0, NULL, &size);
        if(CL_SUCCESS == ret) {
            if(NULL != len) *len = size;
            void* data = malloc(size);
            if(NULL == data) return ALLOC_FAILED;
            ret = clGetCommandQueueInfo(queue, info, size, data, NULL);
            if(CL_SUCCESS == ret) { *value = data; } else { free(data); }
        }

        map_cl_error_2_ee(ret);
    }

    /**
     * @brief get context of command queue
     *
     **/
    inline EE command_queue_get_context(CommandQueue queue, Context *context) {
        if(NULL == context) return NULL_POINTER;
        I32 ret = clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(Context), context, NULL);
        map_cl_error_2_ee(ret);
    }

    /**
     * @brief get device of command queue
     *
     **/
    inline EE command_queue_get_device(CommandQueue queue, Device *device) {
        if(NULL == device) return NULL_POINTER;
        I32 ret = clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(Device), device, NULL);
        map_cl_error_2_ee(ret);
    }

    inline EE retain_command_queue(CommandQueue queue) {
        I32 ret = clRetainCommandQueue(queue);
        map_cl_error_2_ee(ret);
    }

    inline EE release_command_queue(CommandQueue queue) {
        I32 ret = clReleaseCommandQueue(queue);
        map_cl_error_2_ee(ret);
    }

    /**
     * @brief flush command queue, issue all command to execuate
     **/
    inline EE flush(CommandQueue queue) {
        I32 ret = clFlush(queue);
        map_cl_error_2_ee(ret);
    }

    /**
     * @brief wait all commands finish
     **/
    inline EE finish (CommandQueue queue) {
        I32 ret = clFinish(queue);
        map_cl_error_2_ee(ret);
    }

#ifdef __cplusplus
}
#endif
#endif
