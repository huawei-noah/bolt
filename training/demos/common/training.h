// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_BOLT_TRAINING
#define _H_BOLT_TRAINING

#include "training/api/API.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief generate training graph from bolt model
 * @param  graph                   training graph
 * @param  input_bolt_path         original bolt model path
 * @param  batch_size              batch size
 * @param  target_size             target size, size of classification categories
 * @param  loss_type               type of the training loss
 * @param  use_fp16                use fp16 or not
 * @param  input_shape             resized input shapes
 * @param  shape_size              size of the input shape
 * @param  modified_output         name of the resized layer
 *
 * @return API_STATUS which represents success or fail
 * @note
 * As to the option<loss_type>, choose one of the list["OpCrossEntropyLoss", "OpSoftmaxCrossEntropyLoss"]
 * Due to fp16's instability, please set option<use_fp16> as "false" currently.
 * If input shape is same with the original model, please set option<input_shape> as "nullptr".
 * If output size is same with the original model, please set option<modified_output> as "nullptr".
 */
API_STATUS create_general_training_model_from_bolt(Graph_t **graph,
    const char *input_bolt_path,
    size_t batch_size,
    size_t target_size,
    const char *loss_type,
    bool use_fp16,
    int *input_shape,
    int shape_size,
    char *modified_output);

/**
 * @brief  write the updated ms into a bolt after fine-tunning
 * @param  graph        training graph
 * @param  bolt_path    path the input bolt model
 * @param  overwrite    overwrite the original model or not
 *
 * @return API_STATUS which represents success or fail
 * @note
 * As to option<overwrite>, if <overwrite> is 'true', bolt_path "xxx.bolt" will be overwritten with new weights.
 * If <overwrite> is 'false', based on bolt_path "xxx.bolt", a new file "xxx_finetuned.bolt" will be created and the new weights will be written into "xxx_finetuned.bolt". 
 */
API_STATUS save_training_model(Graph_t *graph, const char *bolt_path, bool overwrite);
#ifdef __cplusplus
}
#endif
#endif
