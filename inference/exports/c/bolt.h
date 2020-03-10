/**
* @file
* @brief C API Document
*
* @copyright
* @code
* Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
* WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
* COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
* @endcode
*/
#ifdef __cplusplus
extern "C" {
#endif


/** inference pipeline handle */
typedef void* IHandle;

/** result data memory handle */
typedef void* IResult;

/** CPU affinity policy */
typedef enum {
    HIGH_PERFORMANCE = 0, ///< performance is high priority(use big core)
    LOW_POWER = 1         ///< power is high priority(use small core)
} AFFINITY_TYPE;

/** heterogeneous device type */
typedef enum {
    CPU = 0, ///< CPU
    GPU = 1  ///< GPU
} DEVICE_TYPE;

/** data precision */
typedef enum {
    FP_32 = 0,  ///< 32 bit float
    FP_16 = 1,  ///< 16 bit float
    INT_32 = 2,  ///<  32 bit integer
    UINT_32 = 3  ///<  32 bit unsigned integer
} DATA_TYPE;

/** multi-dimensions data format */
typedef enum {
    NCHW = 0,   ///< batch->channel->high->width data order
    NHWC = 1,   ///< batch->high->width->channel data order
    NCHWC8 = 2,  ///< batch->channel/8->high->width->channel four element data order
    NORMAL = 3
} DATA_FORMAT;

/** 
* @brief create model from file
* @param  modelPath     model file path
* @param  affinity      CPU affinity setting
* @param  device        heterogeneous device setting
*
* @return inference pipeline handle
*
* @note destroy model when pipeline end
* @code
*     IHandle handle = model_create(...);
*     ...
*     model_destroy(handle);
* @endcode
*/
IHandle model_create(const char* modelPath, AFFINITY_TYPE affinity, DEVICE_TYPE device);

/**
* @brief complete model inference engine prepare
* @param  ih            model inference handle
* @param  num_input     the number of input data
* @param  n             the array of all input data's n dimension
* @param  c             the array of all input data's c dimension
* @param  h             the array of all input data's h dimension
* @param  w             the array of all input data's w dimension
* @param  name          the array of all input data's name in string format
* @param  dt_input      the array of all input data's data type
* @param  df_input      the array of all input data's data format
*
* @return
*/
void model_ready(IHandle ih, const int num_input,
    const int* n, const int* c, const int* h, const int* w,
    char** name,
    const DATA_TYPE* dt_input, const DATA_FORMAT* df_input);

/**
* @brief resize model input size
* @param  ih            model inference handle
* @param  num_input     the number of input data
* @param  n             the array of all input data's n dimension
* @param  c             the array of all input data's c dimension
* @param  h             the array of all input data's h dimension
* @param  w             the array of all input data's w dimension
* @param  name          the array of all input data's name in string format
* @param  dt_input      the array of all input data's data type
* @param  df_input      the array of all input data's data format
*
* @return
*
* @code
*     // model_resize must behind model_ready;
*     model_ready(...);
*     model_resize_input(...);
*     model_run(...);
* @endcode
*/
void model_resize_input(IHandle ih, const int num_input,
    const int* n, const int* c, const int* h, const int* w,
    char** name,
    const DATA_TYPE* dt_input, const DATA_FORMAT* df_input);

/**
* @brief malloc result data memory
* @param  ih            inference pipeline handle
*
* @return result data memory handle
*/
IResult IResult_malloc_all(IHandle ih);

/**
* @brief malloc result data memory according to user specification
* @param  ih            inference pipeline handle
* @param  num_outputs   the number of tensor that needed
* @param  outputNames   the array of tesor name that needed
*
* @return result data memory handle
*/
IResult IResult_malloc_part(IHandle ih, const int num_outputs,
    char** outputNames);

/**
* @brief inference result from input
* @param  ih            inference pipeline handle
* @param  ir            result data memory handle
* @param  num_input     the number of input data
* @param  inputNames    the array of all input data's name in string format
* @param  mem           the array of all input data
*
* @return
*/
void model_run(IHandle ih, IResult ir, const int num_input, char** inputNames, void** mem);

/**
* @brief get the number of model output from IResult
* @param  ir            result data memory handle
*
* @return the number of output
*/
int IResult_num_outputs(IResult ir);

/**
* @brief get data from IResult
* @param  ir            result data memory handle
* @param  num_outputs   the number of output data
* @param  outputNames   the array of all output data's name in string format
* @param  data          the array of all output data's content
* @param  n             the array of all output data's n dimension
* @param  c             the array of all output data's c dimension
* @param  h             the array of all output data's h dimension
* @param  w             the array of all output data's w dimension
* @param  dt_output     the array of all output data's data type
* @param  df_output     the array of all output data's data format
*
* @return
*/
void IResult_get(IResult ir, int num_outputs, char** outputNames, void** data,
    int* n, int* c, int* h, int* w,
    DATA_TYPE* dt_output, DATA_FORMAT* df_output);

/**
* @brief get data ptr from IResult without memcpy
* @param  ir            result data memory handle
* @param  num_outputs   the number of output data
* @param  outputNames   the array of all output data's name in string format
* @param  data          the array of all output data's content
* @param  n             the array of all output data's n dimension
* @param  c             the array of all output data's c dimension
* @param  h             the array of all output data's h dimension
* @param  w             the array of all output data's w dimension
* @param  dt_output     the array of all output data's data type
* @param  df_output     the array of all output data's data format
*
* @return
*/
void IResult_get_nocopy(IResult ir, int num_outputs, char** outputNames, void** data,
    int* n, int* c, int* h, int* w,
    DATA_TYPE* dt_output, DATA_FORMAT* df_output);
/**
* @brief free result data memory
* @param  ir            result data memory handle
*
* @return
*/
void IResult_free(IResult ir);

/**
* @brief destroy model
* @param  ih            inference pipeline handle
*
* @return
*/
void model_destroy(IHandle ih);
#ifdef __cplusplus
}
#endif
