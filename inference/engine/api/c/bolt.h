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

#ifndef _BOLT_C_H
#define _BOLT_C_H
#ifdef __cplusplus
extern "C" {
#endif

/** inference pipeline handle */
typedef void *ModelHandle;

/** result data memory handle */
typedef void *ResultHandle;

/** CPU affinity policy */
typedef enum {
    CPU_HIGH_PERFORMANCE = 0,  ///< performance is high priority(use big core)
    CPU_LOW_POWER = 1,         ///< power is high priority(use small core)
    GPU = 2                    ///< use GPU
} AFFINITY_TYPE;

/** heterogeneous device type */
typedef enum {
    CPU_SERIAL = 0,    ///< CPU serial
    CPU_ARM_V7 = 1,    ///< ARMv7 CPU
    CPU_ARM_V8 = 2,    ///< ARMv8 CPU
    CPU_ARM_A55 = 3,   ///< ARM A55 CPU
    CPU_ARM_A76 = 4,   ///< ARM A76 CPU
    CPU_X86_AVX2 = 5,  ///< X86_64 AVX2 CPU
    GPU_MALI = 10      ///< ARM MALI GPU
} DEVICE_TYPE;

/** data precision */
typedef enum {
    FP_32 = 0,   ///< 32 bit float
    FP_16 = 1,   ///< 16 bit float
    INT_32 = 2,  ///< 32 bit integer
    UINT_32 = 3  ///< 32 bit unsigned integer
} DATA_TYPE;

/** multi-dimension data format */
typedef enum {
    NCHW = 0,    ///< batch->channel->high->width data order
    NHWC = 1,    ///< batch->high->width->channel data order
    NCHWC8 = 2,  ///< batch->channel/8->high->width->channel four element data order
    MTK = 3,     ///< batch->time->unit data order
    NORMAL = 4   ///< batch->unit data order
} DATA_FORMAT;

/**
 * @brief create model from file
 * @param  modelPath            model file path
 * @param  affinity             CPU affinity setting
 * @param  algorithmMapPath     the file path to save and load algorithm map file
 *
 * @return inference pipeline handle
 *
 * @note destroy model when pipeline end
 * @code
 *     ModelHandle handle = CreateModel(...);
 *     ...
 *     DestroyModel(handle);
 * @endcode
 * algorithmMapPath is for GPU. If you don't need it, please set it NULL.
 * If algorithmMapPath is set to a existed valid algorithm map file, inference will directly use it.
 * If algorithmMapPath is not existed, inference will run algorithm tuning, which is usually time consuming.
 * Inference will write algorithm tuning result to file.
 * @note
 * If your input tensor size or model is changed, please delete the old algorithm map file.
 * If any unexpected error happen, you can try to delete algorithm file and run it again.
 */
ModelHandle CreateModel(const char *modelPath, AFFINITY_TYPE affinity, const char *algorithmMapPath);

/**
 * @brief get the number of model input from ModelHandle
 * @param  ih            inference pipeline handle
 *
 * @return the number of input
 */
int GetNumInputsFromModel(ModelHandle ih);

/**
 * @brief get input data info set in model handle, which is read from .bolt
 * @param  ih            inference pipeline handle
 * @param  num_inputs    the number of input
 * @param  name          the array of all input data's name
 * @param  n             the array of all input data's n dimension
 * @param  c             the array of all input data's c dimension
 * @param  h             the array of all input data's h dimension
 * @param  w             the array of all input data's w dimension
 * @param  dt            the array of all input data's data type
 * @param  df            the array of all input data's data format
 *
 * @return
 * @note
 * name/n/c/h/w/dt/df array space must be allocated before calling, the array length must be equal to num_inputs.
 * each element of name must be allocated, the array length must be equal to 128.
 */
void GetInputDataInfoFromModel(ModelHandle ih,
    int num_inputs,
    char **name,
    int *n,
    int *c,
    int *h,
    int *w,
    DATA_TYPE *dt,
    DATA_FORMAT *df);

/**
 * @brief complete model inference engine prepare
 * @param  ih            model inference handle
 * @param  num_inputs    the number of input data
 * @param  name          the array of all input data's name in string format
 * @param  n             the array of all input data's n dimension
 * @param  c             the array of all input data's c dimension
 * @param  h             the array of all input data's h dimension
 * @param  w             the array of all input data's w dimension
 * @param  dt            the array of all input data's data type
 * @param  df            the array of all input data's data format
 *
 * @return
 * @note
 * name/n/c/h/w/dt/df array space must be allocated before calling, the array length must be equal to num_inputs.
 * each element of name must be allocated, the array length must be equal to 128.
 */
void PrepareModel(ModelHandle ih,
    const int num_inputs,
    const char **name,
    const int *n,
    const int *c,
    const int *h,
    const int *w,
    const DATA_TYPE *dt,
    const DATA_FORMAT *df);

/**
 * @brief malloc result data memory
 * @param  ih            inference pipeline handle
 *
 * @return result data memory handle
 */
ResultHandle AllocAllResultHandle(ModelHandle ih);

/**
 * @brief inference result from input
 * @param  ih            inference pipeline handle
 * @param  ir            result data memory handle
 * @param  num_inputs    the number of input data
 * @param  name          the array of all input data's name
 * @param  data          the array of all input data
 *
 * @return
 */
void RunModel(ModelHandle ih, ResultHandle ir, int num_inputs, const char **name, void **data);

/**
 * @brief get the number of model output from ResultHandle
 * @param  ir            result data memory handle
 *
 * @return the number of output
 */
int GetNumOutputsFromResultHandle(ResultHandle ir);

/**
 * @brief get output Data info from ResultHandle
 * @param  ir            result data memory handle
 * @param  num_outputs   the number of output data
 * @param  outputNames   the array of all output data's name
 * @param  n             the array of all output data's n dimension
 * @param  c             the array of all output data's c dimension
 * @param  h             the array of all output data's h dimension
 * @param  w             the array of all output data's w dimension
 * @param  dt            the array of all output data's data type
 * @param  df            the array of all output data's data format
 *
 * @return
 * @note
 * name/n/c/h/w/dt/df array space must be allocated before calling, the array length must be equal to num_inputs.
 * each element of name must be allocated, the array length must be equal to 128.
 */
void GetOutputDataInfoFromResultHandle(ResultHandle ir,
    int num_outputs,
    char **outputNames,
    int *n,
    int *c,
    int *h,
    int *w,
    DATA_TYPE *dt,
    DATA_FORMAT *df);

/**
 * @brief get output data from ResultHandle, default to pass value of output ptr,
 * @param  ir            result data memory handle
 * @param  num_outputs   the number of output data
 * @param  data          the array of all output data's content
 *
 * @return
 */
void GetOutputDataFromResultHandle(ResultHandle ir, int num_outputs, void **data);

/**
 * @brief free result data memory
 * @param  ir            result data memory handle
 *
 * @return
 */
void FreeResultHandle(ResultHandle ir);

/**
 * @brief destroy model
 * @param  ih            inference pipeline handle
 *
 * @return
 */
void DestroyModel(ModelHandle ih);

/**
 *Copy current GPU device name to gpuDeviceName
 *User need to alloc memory of gpuDeviceName
 *used with CreateModelWithFileStream to help user choose algoFile
 */
void GetGpuDeviceName(char *gpuDeviceName);

/**
 * @brief create model from file stream
 * Other info is the same with CreateModel
 **/
ModelHandle CreateModelWithFileStream(
    const char *modelFileStream, AFFINITY_TYPE affinity, const char *algorithmMapFileStream);

/**
 * @brief get input Data info set in model handle, when input data is 5d
 * input dims (n,c,t,h,w)
 * other info please reference GetInputDataInfoFromModel
 */
void GetInputDataInfoFromModel5D(ModelHandle ih,
    int num_inputs,
    char **name,
    int *n,
    int *c,
    int *t,
    int *h,
    int *w,
    DATA_TYPE *dt,
    DATA_FORMAT *df);

/**
 * @brief complete model inference engine prepare when input data dim is 5d
 * input dims (n,c,t,h,w)
 * other info please reference PrepareModel
 */
void PrepareModel5D(ModelHandle ih,
    int num_input,
    const char **name,
    const int *n,
    const int *c,
    const int *t,
    const int *h,
    const int *w,
    const DATA_TYPE *dt_input,
    const DATA_FORMAT *df_input);

/**
 * @brief resize model input size
 * @param  ih            model inference handle
 * @param  num_inputs    the number of input data
 * @param  name          the array of all input data's name in string format
 * @param  n             the array of all input data's n dimension
 * @param  c             the array of all input data's c dimension
 * @param  h             the array of all input data's h dimension
 * @param  w             the array of all input data's w dimension
 * @param  dt            the array of all input data's data type
 * @param  df            the array of all input data's data format
 *
 * @return
 * @note
 * name/n/c/h/w/dt/df array space must be allocated before calling, the array length must be equal to num_inputs.
 * each element of name must be allocated, the array length must be equal to 128.
 *
 * @code
 *     // model_resize must behind PrepareModel;
 *     PrepareModel(...);
 *     ResizeModelInput(...);
 *     RunModel(...);
 * @endcode
 */
void ResizeModelInput(ModelHandle ih,
    int num_inputs,
    const char **name,
    const int *n,
    const int *c,
    const int *h,
    const int *w,
    const DATA_TYPE *dt,
    const DATA_FORMAT *df);

/**
 * @brief malloc result data memory according to user specification
 * @param  ih            inference pipeline handle
 * @param  num_outputs   the number of tensor that needed
 * @param  name          the array of tesor name that needed
 *
 * @return result data memory handle
 */
ResultHandle AllocSpecificResultHandle(ModelHandle ih, int num_outputs, const char **name);

/**
 * @brief clone model from a model
 * @param ih   a inference pipeline handle pointer of a model
 *
 * @return inference pipeline handle
 * @note
 * This function can not be called before PrepareModel.
 * The model that being cloned can not be changed before all threads complete clone function.
 * @code
 *     ModelHandle handle = CreateModel(...);
 *     PrepareModel(handle);
 *     ModelHandle clone_handle = CloneModel(handle);
 *     ...
 *     DestroyModel(handle);
 *     DestroyModel(clone_handle);
 * @endcode

 * 
 **/
ModelHandle CloneModel(ModelHandle ih);

/**
 * @brief clone result handle
 * @param ir   a result data handle
 *
 * @return result data memory handle
 * @note
 * The result handle that being cloned can not be changed before all threads complete clone function.
 **/
ResultHandle CloneResultHandle(ResultHandle ir);

/**
 * @brief set process to run on specified CPU core
 * @param  ih            inference pipeline handle
 * @param  cpu_id        cpu core id(0, 1, 2...)
 * @param  device        cpu core architecture(ARM_A76)
 *
 * @return
 */
void SetRuntimeDevice(ModelHandle ih, int cpu_id, DEVICE_TYPE device);

/**
 * @brief set process cpu affinity according cpu average occupy
 * @param  ih            inference pipeline handle
 *
 * @return
 */
void SetRuntimeDeviceDynamic(ModelHandle ih);

#ifdef __cplusplus
}
#endif
#endif
