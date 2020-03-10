// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.



#ifndef _H_BUFFER
#define _H_BUFFER

#include "event.h"

#ifdef __cplusplus
extern "C" {
#endif

    /**
     * @brief get memory information
     *
     **/
    inline EE get_mememory_info(Mem mem, cl_mem_info info, void* *value, U32 *len) {
        if(NULL == value) return NULL_POINTER;

        size_t size;
        I32 ret = clGetMemObjectInfo(mem, info, 0, NULL, &size);
        if(CL_SUCCESS == ret) {
            if(NULL != len) *len = size;
            void* data = malloc(size);
            if(NULL == data) return NULL_POINTER;
            ret = clGetMemObjectInfo(mem, info, size, data, NULL);
            if(CL_SUCCESS == ret) *value = data;
        }

        map_cl_error_2_ee(ret);
    }

#if defined(CL_VERSION_1_2)

    inline EE create_image1D(Context context, cl_mem_flags flags, const cl_image_format *format, U32 len, U32 pitch, void* host_ptr, Mem *image) {
        cl_image_desc image_desc;
        image_desc.image_type = CL_MEM_OBJECT_IMAGE1D;
        image_desc.image_width = len;
        image_desc.image_height = 1;
        image_desc.image_depth = 1;
        image_desc.image_array_size = 1;
        image_desc.image_row_pitch = pitch;
        image_desc.image_slice_pitch = 0;
        image_desc.num_mip_levels = 0;
        image_desc.num_samples = 0;
        image_desc.buffer = NULL;
 
        I32 ret;
        Mem temp = clCreateImage(context, flags, format, &image_desc, host_ptr, &ret);
        *image = temp;
        map_cl_error_2_ee(ret);
    }

    /**
     * @brief create 1d image buffer
     *
     **/
    inline EE create_image1D_buffer(Context context, cl_mem_flags flags, const cl_image_format *format, U32 len, const cl_mem buffer, Mem *image) {
        cl_image_desc image_desc;
        image_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER; 
        image_desc.image_width = len; 
        image_desc.image_height = 1; 
        image_desc.image_depth = 1; 
        image_desc.image_array_size = 1; 
        image_desc.image_row_pitch = len; 
        image_desc.image_slice_pitch = len; 
        image_desc.num_mip_levels = 0; 
        image_desc.num_samples = 0; 
        image_desc.buffer = buffer;

        I32 ret;
        Mem temp = clCreateImage(context, flags, format, &image_desc, NULL, &ret);
        if(CL_SUCCESS == ret) *image = temp;
        map_cl_error_2_ee(ret);
    }
#endif

    /**
     * @brief create 2d image object
     *
     **/
    inline EE create_image2D(Context cont, cl_mem_flags flags, cl_image_format *format, U32 width, U32 height, U32 pitch, void* host_ptr, Mem *mem) {
        I32 ret;
#if defined(CL_VERSION_1_2)
        cl_image_desc image_desc;
        image_desc.image_type = CL_MEM_OBJECT_IMAGE2D; 
        image_desc.image_width = width; 
        image_desc.image_height = height; 
        image_desc.image_depth = 1; 
        image_desc.image_array_size = 1; 
        image_desc.image_row_pitch = pitch; 
        image_desc.image_slice_pitch = 0; 
        image_desc.num_mip_levels = 0; 
        image_desc.num_samples = 0; 
        image_desc.buffer = NULL;

        Mem temp = clCreateImage(cont, flags, format, &image_desc, host_ptr, &ret);
#else
        Mem temp = clCreateImage2D(cont, flags, format, width, height, pitch, host_ptr, &ret);
#endif
        if(CL_SUCCESS == ret) *mem = temp;

        map_cl_error_2_ee(ret);
    }

#if defined(CL_VERSION_1_2)
    /**
     * @brief create 2d image buffer object
     *
     **/
    inline EE create_image2D_array(Context cont, cl_mem_flags flags, cl_image_format *format, U32 width, U32 height, U32 pitch, U32 arraySize, void* host_ptr, Mem *mem) {
        I32 ret;
        cl_image_desc image_desc;
        image_desc.image_type = CL_MEM_OBJECT_IMAGE2D_ARRAY; 
        image_desc.image_width = width; 
        image_desc.image_height = height; 
        image_desc.image_depth = 1; 
        image_desc.image_array_size = arraySize; 
        image_desc.image_row_pitch = pitch; 
        image_desc.image_slice_pitch = 0; 
        image_desc.num_mip_levels = 0; 
        image_desc.num_samples = 0; 
        image_desc.buffer = NULL;

        *mem = clCreateImage(cont, flags, format, &image_desc, host_ptr, &ret);
        map_cl_error_2_ee(ret);
    }
#endif

    /**
     * @brief create 3d image object
     *
     **/
    inline EE create_image3D(Context cont, cl_mem_flags flags, cl_image_format *format, U32 width, U32 height, U32 depth, U32 rowPitch, U32 slicePitch, void* host_ptr, Mem *mem) {
        I32 ret;
#if defined(CL_VERSION_1_2)
        cl_image_desc image_desc;
        image_desc.image_type = CL_MEM_OBJECT_IMAGE3D; 
        image_desc.image_width = width; 
        image_desc.image_height = height; 
        image_desc.image_depth = depth; 
        image_desc.image_array_size = 1; 
        image_desc.image_row_pitch = rowPitch; 
        image_desc.image_slice_pitch = slicePitch; 
        image_desc.num_mip_levels = 0; 
        image_desc.num_samples = 0; 
        image_desc.buffer = NULL;

        Mem temp = clCreateImage(cont, flags, format, &image_desc, host_ptr, &ret);
#else
        Mem temp = clCreateImage3D(cont, flags, format, width, height, depth, rowPitch, slicePitch, host_ptr, &ret);
#endif
        if(CL_SUCCESS == ret) *mem = temp;

        map_cl_error_2_ee(ret);
    }

    /**
     * @brief get image information
     *
     **/
    inline EE get_image_info(Mem mem, cl_mem_info info, void* *value, U32 *len) {
        size_t size;
        I32 ret = clGetImageInfo(mem, info, 0, NULL, &size);
        if(CL_SUCCESS == ret) {
            if(NULL != len) *len = size;

            void* data = malloc(size);
            if(NULL == data) return NULL_POINTER;
            ret = clGetImageInfo(mem, info, size, data, NULL);
            if(CL_SUCCESS == ret) *value = data;
        }

        map_cl_error_2_ee(ret);
    }

    /**
     * @brief get supported image format
     *
     * @warning please free memory associated with format
     **/
    inline EE get_supported_image_formats(Context cont, cl_mem_flags flags, cl_mem_object_type type, cl_image_format **format, U32 *num) {
        if(NULL == format) return NULL_POINTER;

        U32 len;
        I32 ret = clGetSupportedImageFormats(cont, flags, type, 0, NULL, &len);
        if(CL_SUCCESS == ret) {
            if(NULL != num) *num = len;
            cl_image_format *data = (cl_image_format*) malloc(len);
            if(NULL == data) return NULL_POINTER;
            ret = clGetSupportedImageFormats(cont, flags, type, len, data, 0);
            if(CL_SUCCESS == ret) *format = data;
        }

        map_cl_error_2_ee(ret);
    }

    inline EE retain_memory(Mem mem) {
        I32 ret = clRetainMemObject(mem);
        map_cl_error_2_ee(ret);
    }

    inline EE release_memory(Mem mem) {
        I32 ret = clReleaseMemObject(mem);
        map_cl_error_2_ee(ret);
    }

    inline EE enqueue_unmap_memory(CommandQueue queue, Mem mem, void* mapped_ptr,
            I32 num_wait_events, const Event *wait_events, Event *event) {
        I32 ret = clEnqueueUnmapMemObject(queue, mem, mapped_ptr,
                num_wait_events, wait_events, event);

        map_cl_error_2_ee(ret);
    }

    inline EE create_buffer(Context context, cl_mem_flags flags, U32 size,
            void* host_ptr, Mem* buffe) {
        I32 ret;
        size_t len = size;
        *buffe = clCreateBuffer(context, flags, len, host_ptr, &ret);
        map_cl_error_2_ee(ret);
    }

    inline EE create_sub_buffer(Mem buffer, cl_mem_flags flags, 
            U32 offset, U32 size, Mem* sub) {
        I32 ret;
        cl_buffer_region region = { offset, size};
        *sub = clCreateSubBuffer(buffer, flags,  CL_BUFFER_CREATE_TYPE_REGION, &region, &ret);
        map_cl_error_2_ee(ret);
    }

    inline EE enqueue_read_buffer(CommandQueue queue, Mem buffer, cl_bool blocking,
            U32 offset, U32 size, void* ptr, 
            U32 num_wait_events, const Event* wait_events, Event* event) {
        I32 ret = clEnqueueReadBuffer(queue, buffer, blocking,
                offset, size, ptr, num_wait_events, wait_events, event);
        map_cl_error_2_ee(ret);
    }

    /*
    inline EE enqueue_read_buffer_rect(CommandQueue queue, Mem buffer, cl_bool blocking,
            const U32 *buffer_origin, const U32 *host_origin, const U32 *region,
            U32 buffer_row_pitch, U32 buffer_slice_pitch, U32 host_row_pitch,
            U32 host_slice_pitch, void *ptr, U32 num_wait_events,
            const Event *wait_events, Event *event) {

        I32 ret = clEnqueueReadBufferRect(queue, buffer, blocking,
                buffer_origin, host_origin, region,
                buffer_row_pitch, buffer_slice_pitch, host_row_pitch,
                host_slice_pitch, ptr, num_wait_events, wait_events, event);
        map_cl_error_2_ee(ret);
    }
*/
    inline EE enqueue_write_buffer(CommandQueue queue, Mem buffer, cl_bool blocking,
            U32 offset, U32 size, const void *ptr, U32 num_wait_events,
            const Event *wait_events, Event *event) {

        I32 ret = clEnqueueWriteBuffer(queue, buffer, blocking,
                offset, size, ptr, num_wait_events,
                wait_events, event);
        map_cl_error_2_ee(ret);
    }

    inline EE enqueue_fill_buffer(CommandQueue queue, Mem buffer, const void *pattern,
            U32 pattern_size, U32 offset, U32 size, U32 num_wait_events,
            const Event *wait_events, Event *event) {
        size_t pat_size = pattern_size;
        size_t off = offset;
        size_t si  = size;
        I32 ret = clEnqueueFillBuffer(queue, buffer, pattern, pat_size, off, si, num_wait_events, wait_events, event);
        map_cl_error_2_ee(ret);
        }

    /*
    EE enqueue_write_buffer_rect(CommandQueue queue, Mem buffer, cl_bool blocking_write,
            const U32 *buffer_origin, const U32 *host_origin, const U32 *region,
            U32 buffer_row_pitch, U32 buffer_slice_pitch, U32 host_row_pitch,
            U32 host_slice_pitch, const void *ptr, U32 num_wait_events,
            const Event *wait_events, Event *event) {
        I32 ret = clEnqueueWriteBufferRect(queue, buffer, blocking_write,
                const size_t *buffer_origin, const size_t *host_origin, const size_t *region,
                buffer_row_pitch, buffer_slice_pitch, host_row_pitch,
                host_slice_pitch, ptr, num_wait_events, wait_events, event);
        map_cl_error_2_ee(ret);
    }

    }
    */

    inline EE enqueue_copy_buffer(CommandQueue queue, Mem src_buffer, Mem dst_buffer,
            U32 src_offset, U32 dst_offset, U32 size, U32 num_wait_events,
            const Event *wait_events, Event *event){
        I32 ret = clEnqueueCopyBuffer(queue, src_buffer, dst_buffer,
                src_offset, dst_offset, size, 
                num_wait_events, wait_events, event);
        map_cl_error_2_ee(ret);
    }

    /*
    EE enqueue_copy_buffer_rect(CommandQueue queue, Mem src_buffer, Mem dst_buffer,
            const U32 *src_origin, const U32 *dst_origin, const U32 *region,
            U32 src_row_pitch, U32 src_slice_pitch, U32 dst_row_pitch,
            U32 dst_slice_pitch, U32 num_wait_events,
            const Event *wait_events, Event *event) {
        I32 ret = clEnqueueCopyBufferRect(queue, src_buffer, dst_buffer,
                const size_t *src_origin, const size_t *dst_origin, const size_t *region,
                src_row_pitch, src_slice_pitch, dst_row_pitch,
                dst_slice_pitch, num_wait_events, wait_events, event);
        map_cl_error_2_ee(ret);
    }
    */

    inline EE enqueue_map_buffer(CommandQueue queue, Mem buffer, cl_bool blocking_map,
            cl_map_flags map_flags, U32 offset, U32 size,
            U32 num_wait_events, const Event *wait_events, Event *event,
            void* *ptr) {
        I32 ret;
        *ptr = clEnqueueMapBuffer(queue, buffer, blocking_map, map_flags, offset, size,
                num_wait_events, wait_events, event, &ret);
        map_cl_error_2_ee(ret);
    }

    inline EE create_image(Context context, cl_mem_flags flags, const cl_image_format *image_format,
            const cl_image_desc *image_desc, void *host_ptr, Mem* mem) {
        I32 ret;
        *mem = clCreateImage(context, flags, image_format, image_desc, host_ptr, &ret);
            map_cl_error_2_ee(ret);
    }

    inline EE enqueue_read_image(CommandQueue queue, Mem image, cl_bool blocking_read,
            const U32 *origin, const U32 *region, U32 row_pitch, U32 slice_pitch,
            void *ptr, U32 num_wait_events, const Event *wait_events,
            Event *event) {
        size_t org [3];
        size_t reg [3];
        for(U32 i = 0; i < 3; ++i){
            org[i] = (size_t)origin[i];
            reg[i] = (size_t)region[i];
        }
        I32 ret = clEnqueueReadImage(queue, image, blocking_read, org, reg, row_pitch, slice_pitch,
                ptr, num_wait_events, wait_events, event);
        map_cl_error_2_ee(ret);
    }

    inline EE enqueue_write_image(CommandQueue queue, Mem image, cl_bool blocking_write,
            const U32 *origin, const U32 *region, U32 input_row_pitch,
            U32 input_slice_pitch, const void *ptr, U32 num_wait_events,
            const Event *wait_events, Event *event) {
        size_t org [3];
        size_t reg [3];
        for(U32 i = 0; i < 3; ++i){
            org[i] = (size_t)origin[i];
            reg[i] = (size_t)region[i];
        }
        I32 ret = clEnqueueWriteImage(queue, image, blocking_write, org, reg, input_row_pitch,
                input_slice_pitch, ptr, num_wait_events, wait_events, event);
        map_cl_error_2_ee(ret);
    }

    inline EE enqueue_fill_image(CommandQueue queue, Mem image, const void *fill_color,
            const U32 *origin, const U32 *region,U32 num_wait_events,
            const Event *wait_events, Event *event) {
        size_t org [3];
        size_t reg [3];
        for(U32 i = 0; i < 3; ++i){
            org[i] = (size_t)origin[i];
            reg[i] = (size_t)region[i];
        }
        I32 ret = clEnqueueFillImage(queue, image, fill_color,
                org, reg, num_wait_events, wait_events, event);
        map_cl_error_2_ee(ret);
    }
/*

    EE enqueue_copy_image(CommandQueue queue, Mem src_image, Mem dst_image,
            const U32 *src_origin, const U32 *dst_origin, const U32 *region,
            U32 num_wait_events, const cl_event *wait_events, cl_event *event) {
        I32 ret = clEnqueueCopyImage(queue, src_image, dst_image,
                const size_t *src_origin, const size_t *dst_origin, const size_t *region,
                num_wait_events, wait_events, event);
        map_cl_error_2_ee(ret);
    }

    EE enqueue_copy_image_to_buffer(CommandQueue queue, Mem src_image, Mem dst_buffer,
            const U32 *src_origin, const U32 *region, U32 dst_offset,
            U32 num_wait_events, const cl_event *wait_events, cl_event *event) {
        I32 ret = clEnqueueCopyImageToBuffer(queue, src_image, dst_buffer,
                const size_t *src_origin, const size_t *region, dst_offset,
                num_wait_events, wait_events, event);
        map_cl_error_2_ee(ret);
    }

    EE enqueue_copy_buffer_to_image(CommandQueue queue, Mem src_buffer, Mem dst_image,
            U32 src_offset, const U32 *dst_origin, const U32 *region,
            U32 num_wait_events, const cl_event *wait_events, cl_event *event) {
        I32 ret = clEnqueueCopyBufferToImage(queue, src_buffer, dst_image,
                src_offset, const size_t *dst_origin, const size_t *region,
                num_wait_events, wait_events, event);
        map_cl_error_2_ee(ret);
    }

    EE enqueue_map_image(CommandQueue queue, Mem image, cl_bool blocking_map,
            cl_map_flags map_flags, const U32 *origin, const U32 *region,
            U32 *image_row_pitch, U32 *image_slice_pitch, U32 num_wait_events,
            const cl_event *wait_events, cl_event *event, void* *ptr) {
        I32 ret;
        *ptr = clEnqueueMapImage(queue, image, blocking_map,
                map_flags, const size_t *origin, const size_t *region,
                size_t *image_row_pitch, size_t *image_slice_pitch, 
                num_wait_events, wait_events, event, &ret);
        map_cl_error_2_ee(ret);
    }
*/

    inline EE create_sampler(Context context, const cl_sampler_properties* properties, Sampler *s) {
        I32 ret;
        *s = clCreateSamplerWithProperties(context, properties, &ret);
        map_cl_error_2_ee(ret);
    }

    inline EE retain_sampler(Sampler s) {
        I32 ret = clRetainSampler(s);
        map_cl_error_2_ee(ret);
    }

    inline EE release_sampler(Sampler s) {
        I32 ret = clReleaseSampler(s);
        map_cl_error_2_ee(ret);
    }

    inline EE get_sampler_info(Sampler s, 
            cl_sampler_info info,
            void** value, size_t *len) {
        if(NULL == value) return NULL_POINTER;

        size_t size;
        I32 ret = clGetSamplerInfo(s, info, 0, NULL, &size);
        if(CL_SUCCESS == ret) {
            if(NULL != len) *len = size;
            void* data = malloc(size);
            if(NULL == data) return NULL_POINTER;
            ret = clGetSamplerInfo(s, info, size, data, NULL);
            if(CL_SUCCESS == ret) *value = data;
        }

        map_cl_error_2_ee(ret);
    }

    inline EE get_memory_size(Mem memory, U32* size){
        size_t len;
        int ret = clGetMemObjectInfo(memory, CL_MEM_SIZE, sizeof(len), &len, NULL);
        *size = len;
        map_cl_error_2_ee(ret);
    }
#ifdef __cplusplus
}
#endif

#endif
