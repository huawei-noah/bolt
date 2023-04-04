// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <training/api/API.h>
#include <training/api/lowlevel/APIChecks.h>
#include <training/api/lowlevel/APIDefinitions.h>

#include <training/base/common/Common.h>
#include <training/base/initializers/ConstantInitializer.h>
#include <training/base/initializers/IInitializer.h>
#include <training/base/initializers/RandomNormInitializer.h>
#include <training/base/initializers/RandomUniformInitializer.h>
#include <training/base/initializers/XavierInitializer.h>
#include <training/base/layers/BasicLayer.h>
#include <training/base/layers/composite/rnn/LSTMLayer.h>
#include <training/base/layers/parameters/LayerParameters.h>

#include <training/compiler/Layers.h>
#include <training/compiler/Workflow.h>
#include <training/base/optimizers/Adadelta.h>
#include <training/base/optimizers/Adagrad.h>
#include <training/base/optimizers/Adam.h>
#include <training/base/optimizers/Adamax.h>
#include <training/base/optimizers/Momentum.h>
#include <training/base/optimizers/Nesterov.h>
#include <training/base/optimizers/Optimizer.h>
#include <training/base/optimizers/SGD.h>
#include <training/base/optimizers/AdamW.h>
#include <training/compiler/Compiler.h>

#include <training/base/optimizers/schedulers/LrScheduler.h>
#include <training/base/optimizers/schedulers/strategies/CosineAnnealing.h>

#include <iostream>
#include <string.h>
#include <type_traits>

#include "training.h"
#include "model_common.h"

namespace {

using namespace raul;
void set_weights_for_training_model(ModelSpec &ms, Graph_t *graph, const char *modified_output_layer_name)
{
    std::map<std::string, int> scale_position;
    for (int i = 0; i < ms.num_operator_specs; i++) {
        if (ms.ops[i].type == OT_Scale) {
            scale_position[std::string(ms.ops[i].name)] = i;
        }
    }

    for (int i = 0; i < ms.num_weight_specs; i++) {
        WeightSpec ws = ms.ws[i];
        if (scale_position.find(std::string(ws.op_name)) != scale_position.end()) {
            continue;
        }

        if (modified_output_layer_name != nullptr && std::string(ws.op_name) == std::string(modified_output_layer_name)) {
            continue;
        }

        if (ws.bytes_of_weight > 0) {
            std::string weightStr = std::string(ws.op_name) + "::Weights";
            set_tensor(graph, weightStr.c_str(), (float *)(ws.weight),
                ws.bytes_of_weight / bytesOf(ws.mdt));
        }
        if (ws.bytes_of_vec > 0) {
            std::string biasesStr = std::string(ws.op_name) + "::Biases";
            set_tensor(
                graph, biasesStr.c_str(), (float *)(ws.vec), ws.bytes_of_vec / bytesOf(ws.mdt));
        }
    }
}

API_EXPORT API_STATUS add_training_model_from_model_spec(Graph_Description_t *desc, ModelSpec ms)
{
    CHECK_NOT_NULL(desc);
    CHECK_NOT_NULL(desc->mDef);
    try {
        Workflow *work = desc->mDef.get();
        std::map<std::string, int> tensor_layer_index;  // for removing duplication
        std::map<std::string, std::string> simply_replaced_map;
        for (int i = 0; i < ms.num_operator_specs; i++) {
            OperatorSpec op_spec = ms.ops[i];
            if (op_spec.type == OT_Conv) {
                std::string cur_input_tensor_name = op_spec.input_tensors_name[0];
                if (tensor_layer_index.find(cur_input_tensor_name) != tensor_layer_index.end()) {
                    int tmp_layer_index = tensor_layer_index[cur_input_tensor_name];
                    cur_input_tensor_name =
                        cur_input_tensor_name + "::" + std::to_string(tmp_layer_index);
                }

                ConvolutionParamSpec convSpec = op_spec.ps.conv_spec;
                work->add<Convolution2DLayer>(op_spec.name,
                    Convolution2DParams{{cur_input_tensor_name}, {op_spec.output_tensors_name[0]},
                        convSpec.kernel_w, convSpec.kernel_h, convSpec.num_outputs, convSpec.stride_w,
                        convSpec.stride_h, convSpec.pad_top, convSpec.pad_left, false, false,
                        convSpec.dilatedRate_w, convSpec.dilatedRate_h, convSpec.group});
            } else if (op_spec.type == OT_Pooling) {
                std::string cur_input_tensor_name = op_spec.input_tensors_name[0];
                if (tensor_layer_index.find(cur_input_tensor_name) != tensor_layer_index.end()) {
                    int tmp_layer_index = tensor_layer_index[cur_input_tensor_name];
                    cur_input_tensor_name =
                        cur_input_tensor_name + "::" + std::to_string(tmp_layer_index);
                }

                PoolingParamSpec poolingSpec = op_spec.ps.pooling_spec;
                if (poolingSpec.mode == POOLING_MAX) {
                    work->add<MaxPoolLayer2D>(op_spec.name,
                        Pool2DParams{{op_spec.input_tensors_name[0]},
                            {op_spec.output_tensors_name[0]}, poolingSpec.kernel_h,
                            poolingSpec.stride_h});
                } else {
                    if (poolingSpec.kernel_h == 0 && poolingSpec.kernel_w == 0) {
                        work->add<GlobAveragePoolLayer>(op_spec.name,
                            BasicParams{{cur_input_tensor_name}, {op_spec.output_tensors_name[0]}});
                    } else {
                        work->add<AveragePoolLayer>(op_spec.name,
                            Pool2DParams{{cur_input_tensor_name}, {op_spec.output_tensors_name[0]},
                                poolingSpec.kernel_h, poolingSpec.stride_h});
                    }
                }
            } else if (op_spec.type == OT_Softmax) {
                std::string cur_input_tensor = std::string(op_spec.input_tensors_name[0]);
                if (simply_replaced_map.find(cur_input_tensor) != simply_replaced_map.end()) {
                    cur_input_tensor = simply_replaced_map[cur_input_tensor];
                }
                std::string cur_output_tensor = std::string(op_spec.output_tensors_name[0]);
                std::string paramDimStr = "width";
                int softmax_axis = op_spec.ps.softmax_spec.axis;
                if (softmax_axis == 0) {
                    paramDimStr = "batch";
                } else if (softmax_axis == 1) {
                    paramDimStr = "depth";
                } else if (softmax_axis == 2) {
                    paramDimStr = "height";
                } else if (softmax_axis == 3) {
                    paramDimStr = "width";
                } else {
                    std::cerr << "Not support this softmax, exit(-1)...\n\n";
                    exit(-1);
                }
                work->add<SoftMaxActivation>(op_spec.name,
                    BasicParamsWithDim{{cur_input_tensor}, {cur_output_tensor}, paramDimStr});
            } else if (op_spec.type == OT_BatchNorm) {
                std::string cur_input_tensor = std::string(op_spec.input_tensors_name[0]);
                std::string cur_output_tensor = std::string(op_spec.output_tensors_name[0]);

                if (cur_input_tensor ==
                    cur_output_tensor) {  // use the original tensor name to compare
                    if (tensor_layer_index.find(cur_input_tensor) == tensor_layer_index.end()) {
                        tensor_layer_index[cur_output_tensor] = i;  // i == layer_index
                        cur_output_tensor = cur_output_tensor + "::" + std::to_string(i);
                    } else {
                        int cur_tmp_layer_index = tensor_layer_index[cur_input_tensor];
                        cur_input_tensor =
                            cur_input_tensor + "::" + std::to_string(cur_tmp_layer_index);
                        tensor_layer_index[cur_output_tensor] = i;
                        cur_output_tensor = cur_output_tensor + "::" + std::to_string(i);
                    }
                }

                std::string paramDimStr = "width";
                int bn_axis = op_spec.ps.bn_spec.axis;
                if (bn_axis == 0) {
                    paramDimStr = "batch";
                } else if (bn_axis == 1) {
                    paramDimStr = "depth";
                } else if (bn_axis == 2) {
                    paramDimStr = "height";
                } else if (bn_axis == 3) {
                    paramDimStr = "width";
                } else {
                    std::cerr << "Not support this softmax, exit(-1)...\n\n";
                    exit(-1);
                }
                if (ms.ops[i + 1].type == OT_Scale) {
                    auto op_spec_next = ms.ops[i + 1];
                    // deal with the scale
                    std::string cur_scale_input_tensor =
                        std::string(op_spec_next.input_tensors_name[0]);
                    std::string cur_scale_output_tensor =
                        std::string(op_spec_next.output_tensors_name[0]);
                    if (cur_scale_input_tensor != cur_scale_output_tensor) {
                        cur_output_tensor = cur_scale_output_tensor;
                    } else {  // inplace operation
                        tensor_layer_index[cur_scale_output_tensor] = i + 1;
                        cur_output_tensor = cur_scale_output_tensor + "::" + std::to_string(i + 1);
                    }

                    work->add<BatchNormLayer>(op_spec.name,
                        BatchnormParams{
                            {cur_input_tensor}, {cur_output_tensor}, 0.01f, 1e-5f, paramDimStr});
                    i++;  // skip the next op
                } else {
                    work->add<BatchNormLayer>(op_spec.name,
                        BatchnormParams{
                            {cur_input_tensor}, {cur_output_tensor}, 0.01f, 1e-5f, paramDimStr});
                }
            } else if (op_spec.type == OT_Relu) {
                std::string cur_input_tensor = std::string(op_spec.input_tensors_name[0]);
                std::string cur_output_tensor = std::string(op_spec.output_tensors_name[0]);

                if (cur_input_tensor ==
                    cur_output_tensor) {  // use the original tensor name to compare
                    if (tensor_layer_index.find(cur_input_tensor) == tensor_layer_index.end()) {
                        tensor_layer_index[cur_output_tensor] = i;  // i == layer_index
                        cur_output_tensor = cur_output_tensor + "::" + std::to_string(i);
                    } else {
                        int cur_tmp_layer_index = tensor_layer_index[cur_input_tensor];
                        cur_input_tensor =
                            cur_input_tensor + "::" + std::to_string(cur_tmp_layer_index);
                        tensor_layer_index[cur_output_tensor] = i;
                        cur_output_tensor = cur_output_tensor + "::" + std::to_string(i);
                    }
                } else {
                    if (tensor_layer_index.find(cur_input_tensor) != tensor_layer_index.end()) {
                        int cur_tmp_layer_index = tensor_layer_index[cur_input_tensor];
                        cur_input_tensor =
                            cur_input_tensor + "::" + std::to_string(cur_tmp_layer_index);
                    }
                }
                work->add<ReLUActivation>(
                    op_spec.name, BasicParams{{cur_input_tensor}, {cur_output_tensor}});
            } else if (op_spec.type == OT_Eltwise) {
                std::string cur_input_left = std::string(op_spec.input_tensors_name[0]);
                if (simply_replaced_map.find(cur_input_left) != simply_replaced_map.end()) {
                    cur_input_left = simply_replaced_map[cur_input_left];
                }
                std::string cur_input_right = std::string(op_spec.input_tensors_name[1]);
                std::string cur_output_tensor = std::string(op_spec.output_tensors_name[0]);
                if (op_spec.ps.eltwise_spec.mode == ELTWISE_SUM) {
                    work->add<ElementWiseSumLayer>(op_spec.name,
                        ElementWiseLayerParams{{cur_input_left, cur_input_right}, cur_output_tensor});
                } else if (op_spec.ps.eltwise_spec.mode == ELTWISE_PROD) {
                    work->add<ElementWiseMulLayer>(op_spec.name,
                        ElementWiseLayerParams{{cur_input_left, cur_input_right}, cur_output_tensor});
                }
            } else if (op_spec.type == OT_Reshape) {
                std::string cur_input = std::string(op_spec.input_tensors_name[0]);
                std::string cur_output_tensor = std::string(op_spec.output_tensors_name[0]);
                work->add<ReshapeLayer>(
                    op_spec.name, ViewParams{cur_input, cur_output_tensor, 1, 1, -1});
            } else if (op_spec.type == OT_FC) {
                FullyConnectedParamSpec fcps = op_spec.ps.fc_spec;
                std::string cur_input = std::string(op_spec.input_tensors_name[0]);
                std::string cur_output_tensor = std::string(op_spec.output_tensors_name[0]);
                work->add<LinearLayer>(
                    op_spec.name, LinearParams{{cur_input}, {cur_output_tensor}, fcps.num_outputs});
            } else if (op_spec.type == OT_SharedWeight) {
                std::cerr << "[WARNING] Encounter a shared weight op\n";
                continue;
            } else if (op_spec.type == OT_Unsqueeze) {
                std::string cur_input = std::string(op_spec.input_tensors_name[0]);
                std::string cur_output_tensor = std::string(op_spec.output_tensors_name[0]);
                if (cur_input != cur_output_tensor) {
                    simply_replaced_map[cur_output_tensor] = cur_input;
                }
                continue;
            } else if (op_spec.type == OT_Squeeze) {
                std::string cur_input = std::string(op_spec.input_tensors_name[0]);
                std::string cur_output_tensor = std::string(op_spec.output_tensors_name[0]);
                if (cur_input != cur_output_tensor) {
                    simply_replaced_map[cur_output_tensor] = cur_input;
                }
                continue;
            } else if (op_spec.type == OT_Transpose) {
                std::string cur_input = std::string(op_spec.input_tensors_name[0]);
                std::string cur_output_tensor = std::string(op_spec.output_tensors_name[0]);
                if (cur_input != cur_output_tensor) {
                    simply_replaced_map[cur_output_tensor] = cur_input;
                }
                continue;
            } else if (op_spec.type == OT_RNN) {
                std::string cur_input = std::string(op_spec.input_tensors_name[0]);
                std::string cur_output_tensor = std::string(op_spec.output_tensors_name[0]);
                if (simply_replaced_map.find(cur_input) != simply_replaced_map.end()) {
                    cur_input = simply_replaced_map[cur_input];
                }
                std::string cur_output_tensor1 = std::string(op_spec.output_tensors_name[1]);
                GRULayer(op_spec.name,
                    GRUParams{{cur_input}, {cur_output_tensor}, 32, true, true, true, false, false},
                    work->getNetworkParameters());
            } else if (op_spec.type == OT_LayerNorm) {
                std::string cur_input = std::string(op_spec.input_tensors_name[0]);
                std::string cur_output_tensor = std::string(op_spec.output_tensors_name[0]);
                work->add<LayerNorm2DLayer>(
                    op_spec.name, LayerNormParams{cur_input, cur_output_tensor});
            } else if (op_spec.type == OT_Reduction) {
                std::string cur_input = std::string(op_spec.input_tensors_name[0]);
                std::string cur_output_tensor = std::string(op_spec.output_tensors_name[0]);
                work->add<ReduceSumLayer>(
                    op_spec.name, BasicParamsWithDim{{cur_input}, {cur_output_tensor}, "height"});
            } else if (op_spec.type == OT_Sigmoid) {
                std::string cur_input = std::string(op_spec.input_tensors_name[0]);
                if (simply_replaced_map.find(cur_input) != simply_replaced_map.end()) {
                    cur_input = simply_replaced_map[cur_input];
                }
                std::string cur_output_tensor = std::string(op_spec.output_tensors_name[0]);
                work->add<SigmoidActivation>(
                    op_spec.name, BasicParamsWithDim{{cur_input}, {cur_output_tensor}});
            } else if (op_spec.type == OT_Gather) {
                std::string cur_input = std::string(op_spec.input_tensors_name[0]);
                std::string cur_output_tensor = std::string(op_spec.output_tensors_name[0]);
                work->add<SlicerLayer>(op_spec.name,
                    SlicingParams{cur_input, {"left_" + cur_output_tensor, cur_output_tensor},
                        Dimension::Depth, {-1, 1}});
            } else {
                std::cerr << "Encounter non-supporting operator["
                          << OperatorTypeName()[op_spec.type] << "]" << std::endl;
                exit(-1);
            }
        }
    } catch (std::exception &e) {
        set_last_error(e.what());
        return STATUS_ERROR;
    }
    return STATUS_OK;
}

}  // namespace

API_EXPORT API_STATUS create_graph_from_bolt(const char *bolt_file_path,
    Graph_t **graph,
    const char *loss_type,
    size_t batch_size,
    const size_t *input_shape,
    size_t input_shape_count,
    const char *modified_output_layer_name,
    size_t modified_output_size,
    bool use_fp16)
{
    CHECK_STRING(bolt_file_path);
    CHECK_NOT_NULL(graph);
    CHECK_STRING(loss_type);
    CHECK_PRECONDITION(batch_size > 0);

    try {
        Graph_Description_t *desc = NULL;

        if (use_fp16) {
            std::cerr << "Currently, do not support fp16 mode" << std::endl;
            return STATUS_NOT_IMPLEMENTED;
        } else {
            FORWARD_ERROR(create_graph_description(&desc));
        }

        // deserialize the bolt from the input path in order to generate the model_spec
        ModelSpec ms;
        CHECK_STATUS(mt_create_model(&ms));
        CHECK_STATUS(deserialize_model_from_file(bolt_file_path, &ms, DT_F32));

        // dynamically fix output size
        {
            if (modified_output_layer_name != nullptr) {
                for (int i = 0; i < ms.num_operator_specs; i++) {
                    if (std::string(ms.ops[i].name) == std::string(modified_output_layer_name)) {
                        if (ms.ops[i].type == OT_FC) {
                            ms.ops[i].ps.fc_spec.num_outputs = modified_output_size;
                        } else if (ms.ops[i].type == OT_Conv) {
                            ms.ops[i].ps.conv_spec.num_outputs = modified_output_size;
                        } else {
                            std::cerr << "Not Fc or Conv layer, please check the layer name.\n\n";
                            return STATUS_ERROR_BAD_NAME;
                        }
                    }
                }
            }
        }

        // Create inputs
        {
            for (int i = 0; i < ms.num_inputs; i++) {
                const char *cur_input_name = ms.input_names[i];
                const char *input_tensors[] = {ms.input_names[i]};
                TensorDesc tensor_desc = ms.input_dims[i];
                std::vector<int> in_dims = {1, 1, 1, 1};
                if (tensor_desc.nDims > 4) {
                    std::cerr << "Not support surpassing 4D tensor...\n";
                    return STATUS_NOT_IMPLEMENTED;
                }
                for (unsigned int j = 0; j < tensor_desc.nDims; j++) {
                    in_dims[3 - j] = tensor_desc.dims[j];
                }
                if (input_shape_count == 4) {
                    in_dims[0] = input_shape[0];
                    in_dims[1] = input_shape[1];
                    in_dims[2] = input_shape[2];
                    in_dims[3] = input_shape[3];
                }

                FORWARD_ERROR(add_data_layer(desc, cur_input_name, input_tensors, in_dims[0],
                    in_dims[1], in_dims[2], in_dims[3]));
                if (ms.num_outputs != 1) {
                    std::cerr << "Not support output num > 1...\n";
                    return STATUS_NOT_IMPLEMENTED;
                }
            }
        }

        const char *labels_tensor_name = "targets";
        // Create labels layer
        {
            const char *tensors[] = {labels_tensor_name};
            FORWARD_ERROR(add_data_layer(desc, "labels", tensors, 1, 1, 1, modified_output_size));
        }

        // Traverse the operators and create the training graph layer by layer
        {
            FORWARD_ERROR(add_training_model_from_model_spec(desc, ms));
            FORWARD_ERROR(add_reshape_layer(
                desc, "reshape_layer", ms.output_names[0], "output_reshaped", 1, 1, -1));
            const char *loss_input_names[] = {"output_reshaped", labels_tensor_name};
            const char *loss_layer_output_tensor_name = "loss_layer_output_tensor_name";
            size_t loss_layer_inputs_count = 2;
            LOSS_REDUCTION loss_layer_reduction = LOSS_REDUCTION_BATCH_MEAN;
            FORWARD_ERROR(
                add_loss_layer(desc, "loss_layer", loss_input_names, loss_layer_output_tensor_name,
                    loss_type, loss_layer_inputs_count, loss_layer_reduction));

            // general model get ready
            int CUR_BATCH_SIZE = batch_size;
            FORWARD_ERROR(create_graph(&desc, graph, CUR_BATCH_SIZE));
#ifdef _USE_DEBUG
            if ((*graph) != nullptr) {
                print_graph(*graph);
            }
#endif
        }

        // set weights from ms(bolt) to graph(raul)
        {
            set_weights_for_training_model(ms, *graph, modified_output_layer_name);
        }

        mt_destroy_model(&ms);
    } catch (std::exception &e) {
        set_last_error(e.what());
        return STATUS_ERROR;
    }
    return STATUS_OK;
}

API_EXPORT API_STATUS save_graph(Graph_t *graph, const char *input_bolt_file_path, const char *output_bolt_file_path)
{
    ModelSpec ms;
    CHECK_STATUS(mt_create_model(&ms));
    CHECK_STATUS(deserialize_model_from_file(input_bolt_file_path, &ms, DT_F32));
    // maybe need to extract trainable names at firstly
    for (int i = 0; i < ms.num_weight_specs; i++) {
        WeightSpec ws = ms.ws[i];
        std::string name = std::string(ws.op_name);
        size_t length = 0;
        if (ms.ws[i].bytes_of_weight > 0) {
            auto ptr = ms.ws[i].weight;
            ms.ws[i].weight = (U8 *)mt_malloc(ms.ws[i].bytes_of_weight);
            std::vector<FLOAT_TYPE> tmp(
                ws.bytes_of_weight / bytesOf(ws.mdt), static_cast<FLOAT_TYPE>(0.f));
            get_tensor(graph, (name + "::Weights").c_str(), &(tmp[0]), &length);
            memcpy(ms.ws[i].weight, tmp.data(), ms.ws[i].bytes_of_weight);
            mt_free(ptr, &ms);
        }
        if (ms.ws[i].bytes_of_vec > 0) {
            auto ptr = ms.ws[i].vec;
            ms.ws[i].vec = (U8 *)mt_malloc(ms.ws[i].bytes_of_vec);
            std::vector<FLOAT_TYPE> tmp(
                ws.bytes_of_vec / bytesOf(ws.mdt), static_cast<FLOAT_TYPE>(0.f));
            get_tensor(graph, (name + "::Biases").c_str(), &(tmp[0]), &length);
            memcpy(ms.ws[i].vec, tmp.data(), ms.ws[i].bytes_of_vec);
            mt_free(ptr, &ms);
        }
    }
#ifdef _USE_DEBUG
    print_ms(ms);
#endif
    CHECK_STATUS(serialize_model_to_file(&ms, output_bolt_file_path));
    CHECK_STATUS(mt_destroy_model(&ms));
    return STATUS_OK;
}
