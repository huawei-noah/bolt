// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef KERNEL_H_
#define KERNEL_H_

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief get information of kernel
 * @warning please free memory associate with value
 **/
inline EE get_kernel_info(Kernel kernel, cl_kernel_info info, void **value, size_t *size)
{
    if (NULL == value) {
        return NULL_POINTER;
    }

    size_t len;
    cl_int ret = clGetKernelInfo(kernel, info, 0, NULL, &len);
    if (CL_SUCCESS == ret) {
        if (NULL != size) {
            *size = len;
        }
        void *data = malloc(len);
        if (NULL == data) {
            return ALLOC_FAILED;
        }
        ret = clGetKernelInfo(kernel, info, len, data, NULL);
        if (CL_SUCCESS == ret) {
            *value = data;
        } else {
            free(data);
        }
    }

    map_cl_error_2_ee(ret);
}

inline EE get_program_info_from_kernel(Kernel kernel, Program *program)
{
    cl_int ret = clGetKernelInfo(kernel, CL_KERNEL_PROGRAM, sizeof(Program), program, NULL);
    map_cl_error_2_ee(ret);
}

/**
 * @brief get workgroup information of kernel
 * @warning please free memory associate with value
 **/
inline EE get_kernel_workgroup_info(
    Kernel kernel, Device device, cl_kernel_work_group_info info, void **value, size_t *size)
{
    size_t len;
    cl_int ret = clGetKernelWorkGroupInfo(kernel, device, info, 0, NULL, &len);
    if (CL_SUCCESS == ret) {
        if (NULL != size) {
            *size = len;
        }
        void *data = malloc(len);
        if (NULL == data) {
            return ALLOC_FAILED;
        }
        *value = data;
    }

    map_cl_error_2_ee(ret);
}

inline EE create_kernels_in_program(Program program, U32 num_kernel, Kernel *kernels)
{
    if (kernels == nullptr) {
        return NULL_POINTER;
    }
    I32 ret = clCreateKernelsInProgram(program, num_kernel, kernels, NULL);
    map_cl_error_2_ee(ret);
}

inline EE create_kernel(Program program, CI8 *name, Kernel *kernel)
{
    if (kernel == nullptr) {
        return NULL_POINTER;
    }
    I32 ret;
    *kernel = clCreateKernel(program, name, &ret);
    map_cl_error_2_ee(ret);
}

inline EE retain_kernel(Kernel kernel)
{
    cl_int ret = clRetainKernel(kernel);
    map_cl_error_2_ee(ret);
}

inline EE release_kernel(Kernel kernel)
{
    cl_int ret = clReleaseKernel(kernel);
    map_cl_error_2_ee(ret);
}

inline EE set_kernel_arg(Kernel kernel, U32 arg_index, U32 arg_size, const void *arg_value)
{
    cl_int ret = clSetKernelArg(kernel, arg_index, arg_size, arg_value);
    map_cl_error_2_ee(ret);
}
/*
    inline EE clone_kernel(Kernel src_kernel, Kernel* dst_kernel) {
        // TODO
        I32 ret;
        dst_kernel = clCloneKernel(src_kernel, &ret);
        map_cl_error_2_ee(ret);
    }
 */
inline EE enqueue_ndrange_kernel(CommandQueue queue,
    Kernel kernel,
    U32 work_dim,
    CU32 *global_work_offset,
    CU32 *global_work_size,
    CU32 *local_work_size,
    U32 num_events_in_wait_list,
    const Event *event_in_wait_list,
    Event *event)
{
    I32 ret;
    UNUSED(global_work_offset);
    UNUSED(local_work_size);
    switch (work_dim) {
        case 1: {
            size_t gs = global_work_size[0];
            size_t ls = local_work_size[0];
            size_t *ls_ptr = (ls == 0) ? NULL : &ls;
            ret = clEnqueueNDRangeKernel(queue, kernel, work_dim, NULL, &gs, ls_ptr,
                num_events_in_wait_list, event_in_wait_list, event);
            break;
        }
        case 2: {
            size_t gs[2] = {global_work_size[0], global_work_size[1]};
            size_t ls[2] = {local_work_size[0], local_work_size[1]};
            size_t *ls_ptr = (ls[0] == 0 || ls[1] == 0) ? NULL : ls;
            ret = clEnqueueNDRangeKernel(queue, kernel, work_dim, NULL, gs, ls_ptr,
                num_events_in_wait_list, event_in_wait_list, event);
            break;
        }
        case 3: {
            size_t gs[3] = {global_work_size[0], global_work_size[1], global_work_size[2]};
            size_t ls[3] = {local_work_size[0], local_work_size[1], local_work_size[2]};
            size_t *ls_ptr = (ls[0] == 0 || ls[1] == 0 || ls[2] == 0) ? NULL : ls;
            ret = clEnqueueNDRangeKernel(queue, kernel, work_dim, NULL, gs, ls_ptr,
                num_events_in_wait_list, event_in_wait_list, event);
            break;
        }
        default:
            return NOT_SUPPORTED;
    }
    map_cl_error_2_ee(ret);
}

#ifdef __cplusplus
}
#endif
#endif
