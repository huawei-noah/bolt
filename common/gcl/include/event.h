// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef EVENT_H_
#define EVENT_H_

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief wait for event to complete
 **/
inline EE wait_events(U32 num_events, const Event *event_list)
{
    I32 ret = clWaitForEvents(num_events, event_list);
    map_cl_error_2_ee(ret);
}

/**
 * @brief get informaiton about event
 *
 * @warning please free memory associated with value
 **/
inline EE get_event_info(cl_event event, cl_event_info info, void **value, size_t *size)
{
    size_t len;
    I32 ret = clGetEventInfo(event, info, 0, NULL, &len);
    if (CL_SUCCESS == ret) {
        if (NULL != size) {
            *size = len;
        }
        void *data = malloc(len);
        if (NULL == data) {
            return ALLOC_FAILED;
        }
        ret = clGetEventInfo(event, info, len, data, NULL);
        if (CL_SUCCESS == ret) {
            *value = data;
        } else {
            free(data);
        }
    }

    map_cl_error_2_ee(ret);
}

/**
 * @brief increase reference count of event
 **/
inline EE retain_event(Event event)
{
    I32 ret = clRetainEvent(event);
    map_cl_error_2_ee(ret);
}

inline EE release_event(Event event)
{
    I32 ret = clReleaseEvent(event);
    map_cl_error_2_ee(ret);
}

inline EE enqueue_barrier_wait_lists(
    CommandQueue queue, U32 num_wait_events, const Event *wait_events, Event *event)
{
    I32 ret = clEnqueueBarrierWithWaitList(queue, num_wait_events, wait_events, event);
    map_cl_error_2_ee(ret);
}

inline EE event_counting_time(
    Event *event, double *t_queue, double *t_submit, double *t_start, double *t_end, double *t_execute)
{
    cl_ulong queued, submit, start, end;
    CHECK_STATUS(wait_events(1, event));
    I32 ret;
    ret = clGetEventProfilingInfo(
        *event, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &queued, NULL);
    if (ret) {
        map_cl_error_2_ee(ret);
    }
    ret = clGetEventProfilingInfo(
        *event, CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), &submit, NULL);
    if (ret) {
        map_cl_error_2_ee(ret);
    }
    ret =
        clGetEventProfilingInfo(*event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    if (ret) {
        map_cl_error_2_ee(ret);
    }
    ret = clGetEventProfilingInfo(*event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
    if (ret) {
        map_cl_error_2_ee(ret);
    }

    double t0, t1, t2, t3, t4;
    t0 = (double)(queued)*1e-03;
    t1 = (double)(submit)*1e-03;
    t2 = (double)(start)*1e-03;
    t3 = (double)(end)*1e-03;
    t4 = ((double)(end) - (double)(start)) * 1e-03;

    if (t_queue) {
        *t_queue = t0;
    }
    if (t_submit) {
        *t_submit = t1;
    }
    if (t_start) {
        *t_start = t2;
    }
    if (t_end) {
        *t_end = t3;
    }
    if (t_execute) {
        *t_execute = t4;
    }
    return SUCCESS;
}
/**
 * @brief get profiling information
 **/
inline EE event_get_profiling_info(Event event, cl_profiling_info info, void **value, size_t *size)
{
    size_t len;
    I32 ret = clGetEventProfilingInfo(event, info, 0, NULL, &len);
    if (CL_SUCCESS == ret) {
        if (NULL != size) {
            *size = len;
        }
        void *data = malloc(len);
        if (NULL == data) {
            return ALLOC_FAILED;
        }
        ret = clGetEventProfilingInfo(event, info, len, data, NULL);
        if (CL_SUCCESS == ret) {
            *value = data;
        } else {
            free(data);
        }
    }

    map_cl_error_2_ee(ret);
}

#ifdef __cplusplus
}
#endif

#endif
