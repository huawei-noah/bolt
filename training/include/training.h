// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _H_TRAINING
#define _H_TRAINING

#include <string>
#include "../src/api/lowlevel/LowLevelAPI.h"
#include "model_spec.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ASSERT_OK(x) CHECK_REQUIREMENT(STATUS_OK == x)

typedef enum {
    FROM_MS = 0,
    TO_MS = 1
} EXCHANGE_DIRECTION;

inline int searchOperatorIndex(const ModelSpec* spec, const char* op_name)
{
    if (spec->num_operator_specs <= 0) {
        return -1;
    }

    std::string opNameStr = op_name;
    for (int i = 0; i < spec->num_operator_specs; i++) {
        std::string key = spec->ops[i].name;
        if (key == opNameStr) {
            return i;
        }
    }
    return -1;
}

inline int searchWeightIndex(const ModelSpec* spec, const char* op_name)
{
    if (spec->num_weight_specs <= 0) {
        return -1;
    }

    std::string opNameStr = op_name;
    for (int i = 0; i < spec->num_weight_specs; i++) {
        std::string key = spec->ws[i].op_name;
        if (key == opNameStr) {
            return i;
        }
    }
    return -1;
}

inline void exchange_tensor_by_name(std::string layerName, std::string tensorName, int num, F32* data, Graph_t* graph, EXCHANGE_DIRECTION xdir)
{
    std::string nameStr = layerName + tensorName;
    const char* name = const_cast<const char*>(nameStr.c_str());
    switch (xdir) {
        case FROM_MS: {
            ASSERT_OK(set_tensor(graph, name, data, num));
            break;
        }
        case TO_MS: {
            size_t size;
            ASSERT_OK(get_tensor(graph, name, nullptr, &size));
            CHECK_REQUIREMENT((int)size == num);
            ASSERT_OK(get_tensor(graph, name, data, &size));
            break;
        }
        default: {
            CHECK_STATUS(NOT_SUPPORTED);
        }
    }
    
}

inline void exchange_weight_with_ms(ModelSpec* ms, Graph_t* graph, EXCHANGE_DIRECTION xdir)
{
    for (I32 i = 0; i < ms->num_weight_specs; i++) {
        // Part of HSwish and HSigmoid include unused add/mul layers
        if (ms->ws[i].bytes_of_weight <= 4 || ms->ws[i].bytes_of_vec <= 4) {
            continue;
        }
        CHECK_REQUIREMENT(DT_F32 == ms->ws[i].mdt);

        int opIdx = searchOperatorIndex(ms, ms->ws[i].op_name);
        CHECK_REQUIREMENT(-1 != opIdx);

        switch (ms->ops[opIdx].type) {
            case OT_BatchNorm: {
                CHECK_REQUIREMENT(i + 1 < ms->num_weight_specs);
                int next = searchOperatorIndex(ms, ms->ws[i + 1].op_name);
                CHECK_REQUIREMENT(-1 != next && OT_Scale == ms->ops[next].type);
                CHECK_REQUIREMENT(ms->ws[i].bytes_of_weight > 0 && ms->ws[i].bytes_of_vec > 0);
                exchange_tensor_by_name(ms->ws[i].op_name, "::MeanEval", ms->ws[i].bytes_of_weight / 4, (F32*)ms->ws[i].weight, graph, xdir);
                exchange_tensor_by_name(ms->ws[i].op_name, "::VarianceEval", ms->ws[i].bytes_of_vec / 4, (F32*)ms->ws[i].vec, graph, xdir);

                CHECK_REQUIREMENT(ms->ws[i + 1].bytes_of_weight > 0 && ms->ws[i + 1].bytes_of_vec > 0);
                exchange_tensor_by_name(ms->ws[i].op_name, "::Weights", ms->ws[i + 1].bytes_of_weight / 4, (F32*)ms->ws[i + 1].weight, graph, xdir);
                exchange_tensor_by_name(ms->ws[i].op_name, "::Biases", ms->ws[i + 1].bytes_of_vec / 4, (F32*)ms->ws[i + 1].vec, graph, xdir);
                i++;
                break;
            }
            default: {
                if (ms->ws[i].bytes_of_weight > 0) {
                    exchange_tensor_by_name(ms->ws[i].op_name, "::Weights", ms->ws[i].bytes_of_weight / 4, (F32*)ms->ws[i].weight, graph, xdir);
                }
                if (ms->ws[i].bytes_of_vec > 0) {
                    exchange_tensor_by_name(ms->ws[i].op_name, "::Biases", ms->ws[i].bytes_of_vec / 4, (F32*)ms->ws[i].vec, graph, xdir);
                }
            }
        }
    }
}

inline void add_layers(const ModelSpec* ms, Graph_Description_t* desc)
{
    for (I32 i = 0; i < ms->num_operator_specs;) {
        OperatorSpec os = ms->ops[i];
        Graph_Description_Params_t* param = NULL;
        switch (os.type) {
            case OT_Conv: {
                ConvolutionParamSpec cps = os.ps.conv_spec;
                ASSERT_OK(create_graph_description_params(Convolution_Pointwise == cps.convolution_type ? "OpConv2DLayer" : "OpConvDW2DLayer", os.name, &param));
                ASSERT_OK(set_graph_description_param_inputs(param, const_cast<const char**>(&os.input_tensors_name[0]), 1));
                ASSERT_OK(set_graph_description_param_outputs(param, const_cast<const char**>(&os.output_tensors_name[0]), 1));
                ASSERT_OK(set_graph_description_param_conv2d(param, cps.kernel_w, cps.kernel_h, cps.num_outputs, cps.stride_w, cps.stride_h, cps.padding_left, cps.padding_top));
                ASSERT_OK(add_layer(desc, param));
                i++;
                break;
            }
            case OT_BatchNorm: {
                CHECK_REQUIREMENT(i + 1 < ms->num_operator_specs);
                CHECK_REQUIREMENT(OT_Scale == ms->ops[i + 1].type);
                BatchNormParamSpec bnps = os.ps.bn_spec;
                ASSERT_OK(create_graph_description_params("OpBatchNorm2DLayer", os.name, &param));
                ASSERT_OK(set_graph_description_param_inputs(param, const_cast<const char**>(&os.input_tensors_name[0]), 1));
                ASSERT_OK(set_graph_description_param_outputs(param, const_cast<const char**>(&ms->ops[i + 1].output_tensors_name[0]), 1));
                ASSERT_OK(set_graph_description_param_batchnorm(param, 1.0 - bnps.momentum, bnps.eps));
                ASSERT_OK(add_layer(desc, param));
                i += 2;
                break;
            }
            case OT_Relu: {
                ASSERT_OK(create_graph_description_params("OpReLUActivation", os.name, &param));
                ASSERT_OK(set_graph_description_param_inputs(param, const_cast<const char**>(&os.input_tensors_name[0]), 1));
                ASSERT_OK(set_graph_description_param_outputs(param, const_cast<const char**>(&os.output_tensors_name[0]), 1));
                ASSERT_OK(add_layer(desc, param));
                i++;
                break;
            }
            case OT_Scale: {
                // Now assume it is start of HSigmoid or HSwish
                if (i + 2 < ms->num_operator_specs && OT_Clip == ms->ops[i + 1].type) {
                    if (OT_Scale == ms->ops[i + 2].type) {  // HSigmoid
                        ASSERT_OK(create_graph_description_params("OpHSigmoidActivation", os.name, &param));
                        ASSERT_OK(set_graph_description_param_inputs(param, const_cast<const char**>(&os.input_tensors_name[0]), 1));
                        ASSERT_OK(set_graph_description_param_outputs(param, const_cast<const char**>(&ms->ops[i + 2].output_tensors_name[0]), 1));
                        ASSERT_OK(add_layer(desc, param));
                        i += 3;
                    } else if (OT_Eltwise == ms->ops[i + 2].type) {
                        CHECK_REQUIREMENT(i + 3 < ms->num_operator_specs);
                        CHECK_REQUIREMENT(OT_Scale == ms->ops[i + 3].type);
                        ASSERT_OK(create_graph_description_params("OpHSwishActivation", os.name, &param));
                        ASSERT_OK(set_graph_description_param_inputs(param, const_cast<const char**>(&os.input_tensors_name[0]), 1));
                        ASSERT_OK(set_graph_description_param_outputs(param, const_cast<const char**>(&ms->ops[i + 3].output_tensors_name[0]), 1));
                        ASSERT_OK(add_layer(desc, param));
                        i += 4;
                    }
                }
                break;
            }
            case OT_FC: {
                FullyConnectedParamSpec fcp = os.ps.fc_spec;
                ASSERT_OK(create_graph_description_params("OpLinearLayer", os.name, &param));
                ASSERT_OK(set_graph_description_param_inputs(param, const_cast<const char**>(&os.input_tensors_name[0]), 1));
                ASSERT_OK(set_graph_description_param_outputs(param, const_cast<const char**>(&os.output_tensors_name[0]), 1));
                int weightIdx = searchWeightIndex(ms, os.name);
                CHECK_REQUIREMENT(-1 != weightIdx);
                ASSERT_OK(set_graph_description_param_linear(param, fcp.num_outputs, ms->ws[weightIdx].bytes_of_vec > 0));
                ASSERT_OK(add_layer(desc, param));
                i++;
                break;
            }
            case OT_Pooling: {
                PoolingParamSpec pp = os.ps.pooling_spec;
                CHECK_REQUIREMENT(pp.padding_left == pp.padding_right);
                CHECK_REQUIREMENT(pp.padding_top == pp.padding_bottom);
                if (POOLING_MAX == pp.mode) {
                    ASSERT_OK(create_graph_description_params("OpMaxPool2DLayer", os.name, &param));
                    ASSERT_OK(set_graph_description_param_inputs(param, const_cast<const char**>(&os.input_tensors_name[0]), 1));
                    ASSERT_OK(set_graph_description_param_outputs(param, const_cast<const char**>(&os.output_tensors_name[0]), 1));
                    ASSERT_OK(set_graph_description_param_maxpool2d(param, pp.kernel_w, pp.kernel_h, pp.stride_w, pp.stride_h, pp.padding_left, pp.padding_top));
                    ASSERT_OK(add_layer(desc, param));
                } else if (POOLING_MEAN == pp.mode) {
                    if (pp.kernel_w > 0) {
                        ASSERT_OK(create_graph_description_params("OpAvgPool2DLayer", os.name, &param));
                        ASSERT_OK(set_graph_description_param_inputs(param, const_cast<const char**>(&os.input_tensors_name[0]), 1));
                        ASSERT_OK(set_graph_description_param_outputs(param, const_cast<const char**>(&os.output_tensors_name[0]), 1));
                        ASSERT_OK(set_graph_description_param_maxpool2d(param, pp.kernel_w, pp.kernel_h, pp.stride_w, pp.stride_h, pp.padding_left, pp.padding_top));
                        ASSERT_OK(add_layer(desc, param));
                    } else {
                        ASSERT_OK(create_graph_description_params("OpGlAvgPool2DLayer", os.name, &param));
                        ASSERT_OK(set_graph_description_param_inputs(param, const_cast<const char**>(&os.input_tensors_name[0]), 1));
                        ASSERT_OK(set_graph_description_param_outputs(param, const_cast<const char**>(&os.output_tensors_name[0]), 1));
                        ASSERT_OK(add_layer(desc, param));
                    }
                } else {
                    CHECK_STATUS(NOT_SUPPORTED);
                }
                i++;
                break;
            }
            case OT_Eltwise: {
                EltwiseParamSpec ep = os.ps.eltwise_spec;
                const char** inputNames;
                inputNames = const_cast<const char**>(os.input_tensors_name);
                switch (ep.elt_mode) {
                    case ELTWISE_SUM:
                        ASSERT_OK(create_graph_description_params("OpElementWiseSumLayer", os.name, &param));
                        break;
                    case ELTWISE_MAX:
                        ASSERT_OK(create_graph_description_params("OpElementWiseMaxLayer", os.name, &param));
                        break;
                    case ELTWISE_MIN:
                        ASSERT_OK(create_graph_description_params("OpElementWiseMinLayer", os.name, &param));
                        break;
                    case ELTWISE_PROD:
                        ASSERT_OK(create_graph_description_params("OpElementWiseMulLayer", os.name, &param));
                        break;
                    case ELTWISE_SUB:
                        ASSERT_OK(create_graph_description_params("OpElementWiseSubLayer", os.name, &param));
                        break;
                    case ELTWISE_DIV:
                        ASSERT_OK(create_graph_description_params("OpElementWiseDivLayer", os.name, &param));
                        break;
                    default:
                        CHECK_STATUS(NOT_SUPPORTED);
                }
                ASSERT_OK(set_graph_description_param_inputs(param, inputNames, os.num_inputs));
                ASSERT_OK(set_graph_description_param_outputs(param, const_cast<const char**>(&os.output_tensors_name[0]), 1));
                ASSERT_OK(add_layer(desc, param));
                i++;
                break;
            }
            case OT_LogSoftmax: {
                ASSERT_OK(create_graph_description_params("OpLogSoftmaxActivation", os.name, &param));
                ASSERT_OK(set_graph_description_param_inputs(param, const_cast<const char**>(&os.input_tensors_name[0]), 1));
                ASSERT_OK(set_graph_description_param_outputs(param, const_cast<const char**>(&os.output_tensors_name[0]), 1));
                ASSERT_OK(add_layer(desc, param));
                i++;
                break;
            }
            case OT_Reshape: {
                ReshapeParamSpec rp = os.ps.reshape_spec;
                CHECK_REQUIREMENT(2 == rp.shape_size);
                ASSERT_OK(create_graph_description_params("OpReshapeLayer", os.name, &param));
                ASSERT_OK(set_graph_description_param_inputs(param, const_cast<const char**>(&os.input_tensors_name[0]), 1));
                ASSERT_OK(set_graph_description_param_outputs(param, const_cast<const char**>(&os.output_tensors_name[0]), 1));
                ASSERT_OK(set_graph_description_param_shape(param, 1, 1, -1));
                ASSERT_OK(add_layer(desc, param));
                i++;
                break;
            }
            case OT_None: {
                i++;
                break;
            }
            default: {
                CHECK_STATUS(NOT_IMPLEMENTED);
            }
        }
    }
}

#ifdef __cplusplus
}
#endif

#endif
