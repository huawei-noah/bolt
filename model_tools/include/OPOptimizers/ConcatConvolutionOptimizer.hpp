// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_CONCATCONVOLUTIONOPTIMIZER
#define _H_CONCATCONVOLUTIONOPTIMIZER

#include "OPOptimizer.hpp"

// for concat + group conv

const static std::set<OperatorType> sameShapeOP = {OT_Eltwise, OT_LayerNorm, OT_HSwish, OT_HSigmoid,
    OT_Sigmoid, OT_Clip, OT_Gelu, OT_TanH, OT_HSwishNoDiv, OT_PRelu, OT_Softmax, OT_LogSoftmax, OT_Scale};
const static std::set<OperatorType> transShapeOP = {OT_Transpose};

class ConcatConvolutionOptimizer : public OPOptimizer {

    int traceChannel(ModelSpec *spec, std::string tensorName, int j) {
        auto prevOpIndexes0 = searchOperatorIndexByOutput(spec, tensorName.c_str(), 0, j);
        if ((prevOpIndexes0.size() != 1) || (-1 == prevOpIndexes0[0].first)) {
            return -1;
        }
        int id = prevOpIndexes0[0].first;
        if (spec->ops[id].type == OT_Conv) {
            return spec->ops[id].ps.conv_spec.num_outputs;
        }
        if (spec->ops[id].type == OT_Reshape) {
            int num_shape = spec->ops[id].ps.reshape_spec.num_shape;
            if (num_shape < 2) {
                return -1;
            }
            return spec->ops[id].ps.reshape_spec.shape[num_shape - 2];
        }
        if (spec->ops[id].type == OT_SharedWeight) {
            int num_shape = spec->ops[id].ps.shared_weight_spec.desc.nDims;
            if (num_shape < 2) {
                return -1;
            }
            return spec->ops[id].ps.shared_weight_spec.desc.dims[num_shape - 2];
        }
        int channel = 0;
        if (sameShapeOP.count(spec->ops[id].type)) {
            for (int i = 0; i < int(spec->ops[id].num_inputs); ++i) {
                channel = UNI_MAX(channel, traceChannel(spec, spec->ops[id].input_tensors_name[i], id));
                if (channel > 1) {
                    return channel;
                }
            }
        }
        if (transShapeOP.count(spec->ops[id].type)) {
            return -1;
        }

        return channel;
    }

    bool optimize(ModelSpec *spec) override
    {
        bool hasOptimized = false;
        for (int i = 1; i < spec->num_operator_specs; i++) {
            if (spec->ops[i].type == OT_Concat) {
                std::string curOut = spec->ops[i].output_tensors_name[0];
                auto nextOpIndexes = searchOperatorIndexByInput(
                    spec, curOut, i + 1, spec->num_operator_specs, false);

                if ((nextOpIndexes.size() != 1) || (-1 == nextOpIndexes[0].first) ||
                    (OT_Conv != spec->ops[nextOpIndexes[0].first].type)) {
                    continue;
                }

                int convId = nextOpIndexes[0].first;
                int kernelSize = spec->ops[convId].ps.conv_spec.kernel_t *
                    spec->ops[convId].ps.conv_spec.kernel_h *
                    spec->ops[convId].ps.conv_spec.kernel_w;
                int group = spec->ops[convId].ps.conv_spec.group;
                int convWeightIndex = searchWeightIndex(spec, spec->ops[convId].name);
                int channel = spec->ws[convWeightIndex].bytes_of_weight /
                    kernelSize / group / bytesOf(spec->ws[convWeightIndex].mdt);

                if ((channel != 2) ||
                    (spec->ops[i].num_inputs != 2)) {
                    continue;
                }
                if ((traceChannel(spec, spec->ops[i].input_tensors_name[0], i) != group) ||
                    (traceChannel(spec, spec->ops[i].input_tensors_name[1], i) != group))
                {
                    continue;
                }

                // insert tfslice 1
                std::string sliceLeftName = allocName("SliceLeft_" + std::to_string(convId));
                OperatorSpec sliceLeftOperator =
                    mt_create_operator(sliceLeftName.c_str(), OT_TfSlice, 1, 1);
                    int strides[8];
                sliceLeftOperator.ps.tfslice_spec.num_dims = 4;
                for (int j = 0; j < 4; ++j) {
                    sliceLeftOperator.ps.tfslice_spec.begin_mask[j] = 1;
                    sliceLeftOperator.ps.tfslice_spec.end_mask[j] = 1;
                    sliceLeftOperator.ps.tfslice_spec.strides[j] = 1;
                    if (j == 1) {
                        sliceLeftOperator.ps.tfslice_spec.strides[j] = 2;
                    }
                }
                str_copy(sliceLeftOperator.input_tensors_name[0],
                    spec->ops[i].output_tensors_name[0],
                    strlen(spec->ops[i].output_tensors_name[0]));
                std::string leftOut = allocName(std::string(spec->ops[i].output_tensors_name[0]) + "_slice_left");
                str_copy(sliceLeftOperator.output_tensors_name[0],
                    leftOut.data(), strlen(leftOut.data()));

                // insert conv 1
                std::string convLeftName = allocName(std::string(spec->ops[convId].name) + "_left");
                OperatorSpec convLeftOperator =
                    mt_create_operator(convLeftName.c_str(), OT_Conv, 1, 1);
                convLeftOperator.ps.conv_spec = spec->ops[convId].ps.conv_spec;
                convLeftOperator.ps.conv_spec.convolution_type = CONVOLUTION_DEPTHWISE;

                str_copy(convLeftOperator.input_tensors_name[0],
                    leftOut.data(),
                    strlen(leftOut.data()));
                leftOut = allocName(std::string(spec->ops[i].output_tensors_name[0]) + "_conv_left");
                str_copy(convLeftOperator.output_tensors_name[0],
                    leftOut.data(), strlen(leftOut.data()));

                WeightSpec ws[2];
                U32 weightSize = spec->ws[convWeightIndex].bytes_of_weight / 2;
                ws[0] = mt_create_weight(
                    convLeftName.c_str(), spec->ws[convWeightIndex].mdt, weightSize, 0, 0);

                // insert tfslice 2
                std::string sliceRightName = allocName("SliceRight_" + std::to_string(convId));
                OperatorSpec sliceRightOperator =
                    mt_create_operator(sliceRightName.c_str(), OT_TfSlice, 1, 1);
                sliceRightOperator.ps.tfslice_spec.num_dims = 4;
                for (int j = 0; j < 4; ++j) {
                    sliceRightOperator.ps.tfslice_spec.begin_mask[j] = 1;
                    sliceRightOperator.ps.tfslice_spec.end_mask[j] = 1;
                    sliceRightOperator.ps.tfslice_spec.strides[j] = 1;
                    if (j == 1) {
                        sliceRightOperator.ps.tfslice_spec.strides[j] = 2;
                        sliceRightOperator.ps.tfslice_spec.begin_mask[j] = 0;
                        sliceRightOperator.ps.tfslice_spec.begin[j] = 1;
                    }
                }
                str_copy(sliceRightOperator.input_tensors_name[0],
                    spec->ops[i].output_tensors_name[0],
                    strlen(spec->ops[i].output_tensors_name[0]));
                std::string rightOut = allocName(std::string(spec->ops[i].output_tensors_name[0]) + "_slice_right");
                str_copy(sliceRightOperator.output_tensors_name[0],
                    rightOut.data(), strlen(rightOut.data()));

                // insert conv 2
                std::string convRightName = allocName(std::string(spec->ops[convId].name) + "_right");
                OperatorSpec convRightOperator =
                    mt_create_operator(convRightName.c_str(), OT_Conv, 1, 1);
                convRightOperator.ps.conv_spec = spec->ops[convId].ps.conv_spec;
                convRightOperator.ps.conv_spec.convolution_type = CONVOLUTION_DEPTHWISE;

                str_copy(convRightOperator.input_tensors_name[0],
                    rightOut.data(),
                    strlen(rightOut.data()));
                rightOut = allocName(std::string(spec->ops[i].output_tensors_name[0]) + "_right");
                str_copy(convRightOperator.output_tensors_name[0],
                    rightOut.data(), strlen(rightOut.data()));

                ws[1] = mt_create_weight(
                    convRightName.c_str(), spec->ws[convWeightIndex].mdt, weightSize, spec->ws[convWeightIndex].bytes_of_vec, 0);

                // init conv weight
                U8 *originWeight = spec->ws[convWeightIndex].weight;
                U8 *weightLeft = ws[0].weight;
                U8 *weightRight = ws[1].weight;
                U32 tile = kernelSize * bytesOf(spec->ws[convWeightIndex].mdt);
                for (int n = 0; n < group; ++n) {
                    U32 offset = n * tile * 2;
                    UNI_MEMCPY(weightLeft + n * tile, originWeight + offset, tile);
                    UNI_MEMCPY(weightRight + n * tile, originWeight + offset + tile, tile);
                }
                UNI_MEMCPY(ws[1].vec, spec->ws[convWeightIndex].vec, spec->ws[convWeightIndex].bytes_of_vec);

                // insert add
                std::string addName = allocName("add_" + std::string(spec->ops[convId].name));
                OperatorSpec addOperator =
                    mt_create_operator(addName.c_str(), OT_Eltwise, 2, 1);
                addOperator.ps.eltwise_spec.mode = ELTWISE_SUM;
                addOperator.ps.eltwise_spec.activation_type = ACTIVATION_NULL;

                str_copy(addOperator.input_tensors_name[0], leftOut.data(), strlen(leftOut.data()));
                str_copy(addOperator.input_tensors_name[1], rightOut.data(), strlen(rightOut.data()));
                str_copy(addOperator.output_tensors_name[0],
                    spec->ops[convId].output_tensors_name[0],
                    strlen(spec->ops[convId].output_tensors_name[0]));

                mt_insert_weight(spec, ws, 2);
                mt_insert_operator(spec, i + 1, sliceLeftOperator);
                mt_insert_operator(spec, i + 2, sliceRightOperator);
                mt_insert_operator(spec, i + 3, convLeftOperator);
                mt_insert_operator(spec, i + 4, convRightOperator);
                mt_insert_operator(spec, i + 5, addOperator);

                spec->ws[convWeightIndex].bytes_of_weight = 0;
                mt_free(spec->ws[convWeightIndex].weight, spec);
                spec->ws[convWeightIndex].bytes_of_vec = 0;
                mt_free(spec->ws[convWeightIndex].vec, spec);

                // setOperatorInvalid(spec, i + 3, false);
                setOperatorInvalid(spec, convId + 5, false);
                hasOptimized = true;
            }
        }

        return hasOptimized;
    }
};
#endif
