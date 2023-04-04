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
 * @brief create training graph from bolt model file
 * @param  bolt_file_path                bolt model file on disk
 * @param  graph                         training graph
 * @param  loss_type                     training loss type
 * @param  batch_size                    training batch size, default:1
 * @param  input_shape                   new input shape, we support change input shape when training, default: NULL
 * @param  input_shape_count             input shape count, default:0
 * @param  modified_output_layer_name    name of output layer, we support change output category size when training, default: NULL
 * @param  modified_output_layer_size    size of category, default:0
 * @param  use_fp16                      training use fp16 or not, default:false
 *
 * @return API_STATUS which represents success or fail
 * @note
 * currently we only test CNN model with single input and single output.
 */
API_EXPORT API_STATUS create_graph_from_bolt(const char *bolt_file_path,
    Graph_t **graph,
    const char *loss_type,
    size_t batch_size=1,
    const size_t *input_shape=nullptr,
    size_t input_shape_count=0,
    const char *modified_output_layer_name=nullptr,
    size_t modified_output_size=0,
    bool use_fp16=false);

/**
 * @brief  write training graph to bolt model file
 * @param  graph                   training graph
 * @param  input_bolt_file_path    original bolt model file path on disk
 * @param  output_bolt_file_path   bolt model file path where we want to save training graph
 *
 * @return API_STATUS which represents success or fail
 * @note
 * bolt model contains layer topology and weights.
 * currently we need to read bolt model topology from input_bolt_file_path, and combine weights from training graph.
 */
API_EXPORT API_STATUS save_graph(Graph_t *graph, const char *input_bolt_file_path, const char *output_bolt_file_path);

#ifdef __cplusplus
}
#endif

#endif
