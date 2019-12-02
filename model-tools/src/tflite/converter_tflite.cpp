// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include <iostream>
#include <fstream>
#include <string>
#include <memory>
#include <typeinfo>
#include <vector>
#include "schema_generated.h"
#include "type.h"
#include "converter.h"
#include "model_serialize_deserialize.hpp"
#include "model_tools.h"


#ifdef _USE_TFLITE_MODEL

OperatorType tfliteType_to_boltType(tflite::BuiltinOperator tfliteType) {
    if (tfliteType == tflite::BuiltinOperator_ADD) {
        return OT_Eltwise;    // mode: add
    } else if (tfliteType == tflite::BuiltinOperator_CONCATENATION) {
        return OT_Concat;
    } else if (tfliteType == tflite::BuiltinOperator_CONV_2D || tfliteType == tflite::BuiltinOperator_DEPTHWISE_CONV_2D) {
        return OT_Conv;
    } else if (tfliteType == tflite::BuiltinOperator_LOGISTIC) {
        return OT_Logistic;
    } else if (tfliteType == tflite::BuiltinOperator_MAX_POOL_2D) {
        return OT_Pooling;
    } else if (tfliteType == tflite::BuiltinOperator_RESHAPE) {
        return OT_Reshape;
    } else if (tfliteType == tflite::BuiltinOperator_RESIZE_BILINEAR) {
        return OT_ResizeBilinear;
    }
    return OT_None;
}

EE mt_load_tflite(CI8* dir, CI8* mfn, ModelSpec* ms)
{
    std::string tfliteSuffix = ".tflite";
    std::string model_name = std::string(dir) + std::string(mfn) + tfliteSuffix;
    std::ifstream inputFile(model_name.c_str(), std::ios::binary);
    inputFile.seekg(0, std::ios::end);
    const auto size = inputFile.tellg();
    inputFile.seekg(0, std::ios::beg);

    char* buffer = new char[size];
    inputFile.read(buffer, size);
    inputFile.close();

    // verify model
    flatbuffers::Verifier verify((uint8_t*)buffer, size);
    CHECK_REQUIREMENT(tflite::VerifyModelBuffer(verify));

    auto tfliteModel = tflite::UnPackModel(buffer);
    const auto& tfliteOpSet = tfliteModel->operator_codes;

    const auto subGraphsSize = tfliteModel->subgraphs.size();    // indeed to fetch the vector size
    CHECK_REQUIREMENT(subGraphsSize == 1);

    const auto& tfliteModelBuffer = tfliteModel->buffers;

    // the first loop
    bool quantizedModel = true;
    int modelWeightOpNum = 0;
    for (int i = 0; i < subGraphsSize; ++i) {
        const auto& ops = tfliteModel->subgraphs[i]->operators;
        const auto& tensors = tfliteModel->subgraphs[i]->tensors;
        const int opNums = ops.size();

        // 20191120
        ms->num_operator_specs = opNums;
        OperatorSpec* opsPtr = (OperatorSpec*)mt_malloc(sizeof(OperatorSpec) * ms->num_operator_specs);
        ms->ops = opsPtr;

        for (int j = 0; j < opNums; ++j) {
            std::string curOpName = "op" + std::to_string(j);
            str_copy(opsPtr[j].name, curOpName.c_str(), curOpName.length());


            const int opcodeIndex = ops[j]->opcode_index;
            const auto opCode = tfliteOpSet[opcodeIndex]->builtin_code;
            OperatorType opType = tfliteType_to_boltType(opCode);
            opsPtr[j].type = opType;

            // deal with all the op
            const int opInputTensorSize = ops[j]->inputs.size();
            const int opOutputTensorSize = ops[j]->outputs.size();
            
            // ops[j].num_inputs upon to 
            if (opCode == tflite::BuiltinOperator_ADD) {    // in:2 out:1
                EltwiseParamSpec eltPs;
                eltPs.elt_mode = ELTWISE_SUM;
                eltPs.elt_sum_spec.coeff_size = opInputTensorSize;
                
                ops[j].num_inputs = opInputTensorSize;
                opsPtr[j].input_tensors_name = (I8**)mt_malloc(opsPtr[j].num_inputs * sizeof(I8*));
                for (int iter = 0; iter < opInputTensorSize; iter++) {
                    const int inIndex = ops[j]->inputs[iter];
                    const auto& inTensor = tensors[inIndex];
                    opsPtr[j].input_tensors_name[j] = (I8*)mt_malloc(NAME_LEN * sizeof(I8));
                    str_copy(opsPtr[j].input_tensors_name[iter], (inTensor->name).c_str(), (inTensor->name).length());
                }
                opsPtr[j].num_outputs = opOutputTensorSize;
                opsPtr[j].output_tensors_name = (I8**)mt_malloc(opsPtr[j].num_outputs * sizeof(I8*));
                for (int iter = 0; iter < opOutputTensorSize; iter++) {
                    const int outIndex = ops[j]->outputs[iter];
                    const auto& outTensor = tensors[outIndex];
                    opsPtr[j].output_tensors_name[j] = (I8*)mt_malloc(NAME_LEN * sizeof(I8));
                    str_copy(opsPtr[j].output_tensors_name[iter], (outTensor->name).c_str(), (outTensor->name).length());
                }
                opsPtr[j].ps.eltwise_param_spec = eltPs;
            } 

            else if (opCode == tflite::BuiltinOperator_CONCATENATION) {
                ops[j].num_inputs = opInputTensorSize;
                opsPtr[j].input_tensors_name = (I8**)mt_malloc(opsPtr[j].num_inputs * sizeof(I8*));
                for (int iter = 0; iter < opInputTensorSize; iter++) {
                    const int inIndex = ops[j]->inputs[iter];
                    const auto& inTensor = tensors[inIndex];
                    opsPtr[j].input_tensors_name[j] = (I8*)mt_malloc(NAME_LEN * sizeof(I8));
                    str_copy(opsPtr[j].input_tensors_name[iter], (inTensor->name).c_str(), (inTensor->name).length());
                }
                opsPtr[j].num_outputs = opOutputTensorSize;
                opsPtr[j].output_tensors_name = (I8**)mt_malloc(opsPtr[j].num_outputs * sizeof(I8*));
                for (int iter = 0; iter < opOutputTensorSize; iter++) {
                    const int outIndex = ops[j]->outputs[iter];
                    const auto& outTensor = tensors[outIndex];
                    opsPtr[j].output_tensors_name[j] = (I8*)mt_malloc(NAME_LEN * sizeof(I8));
                    str_copy(opsPtr[j].output_tensors_name[iter], (outTensor->name).c_str(), (outTensor->name).length());
                }
            } 

            else if (opCode == tflite::BuiltinOperator_CONV_2D) {
                modelWeightOpNum++;
                // input 2/3: input/weight/bias
                const int inputIndex = ops[j]->inputs[0];
                const auto& inputTensor = tensors[inputIndex];
                ops[j].num_inputs = 1;
                opsPtr[j].input_tensors_name = (I8**)mt_malloc(opsPtr[j].num_inputs * sizeof(I8*));
                str_copy(opsPtr[j].input_tensors_name[0], (inputTensor->name).c_str(), (inputTensor->name).length());

                const int weightIndex = ops[j]->inputs[1];
                const auto& weightTensor = tensors[weightIndex];
                auto conv2DWeightPtr = reinterpret_cast<const float*>(tfliteModelBuffer[weightTensor->buffer]->data.data());
                if (ops[j]->inputs.size() == 3) {
                    const int biasIndex = ops[j]->inputs[2];
                    const auto& biasTensor = tensors[biasIndex];
                    auto conv2DBiasPtr = reinterpret_cast<const float*>(tfliteModelBuffer[biasTensor->buffer]->data.data());
                }

                const int outputIndex = ops[j]->outputs[0];
                const auto& outputTensor = tensors[outputIndex];
                ops[j].num_outputs = 1;
                opsPtr[j].output_tensors_name = (I8**)mt_malloc(opsPtr[j].num_outputs * sizeof(I8*));
                str_copy(opsPtr[j].output_tensors_name[0], (outputTensor->name).c_str(), (outputTensor->name).length());

                quantizedModel = weightTensor->type == tflite::TensorType_UINT8;

                // deal with the meta info
                // to extract the Conv2DOptionsT struct
                const auto& weightShape = weightTensor->shape;
                CHECK_REQUIREMENT(weightShape.size() == 4);
                const int conv2d_co = weightShape[0];
                const int conv2d_kh = weightShape[1];
                const int conv2d_kw = weightShape[2];
                const int conv2d_ci = weightShape[3];
                const auto& tfliteConvOption = ops[j]->builtin_options.AsConv2DOptions();
                const int conv2d_dilationX = tfliteConvOption->dilation_w_factor;
                const int conv2d_dilationY = tfliteConvOption->dilation_h_factor;
                const int conv2d_strideX = tfliteConvOption->stride_w;
                const int conv2d_strideY = tfliteConvOption->stride_h;
                const auto activationFunc = tfliteConvOption->fused_activation_function;

                ConvolutionParamSpec convPs;
                convPs.num_kernels = conv2d_co * conv2d_ci;
                convPs.kernel_size = conv2d_kh;
                convPs.stride = conv2d_strideX;
                convPs.padding = (convPs.kernel_size - 1) / 2;
                convPs.group = 1;   
                convPs.dilation = 1;
                convPs.convolution_type = Convolution_Pointwise;
                if (activationFunc == tflite::ActivationFunctionType_RELU) {
                    convPs.pw_activation_type = ACTIVATION_RELU;
                    convPs.dw_activation_type = ACTIVATION_NULL;
                } else if (activationFunc == tflite::ActivationFunctionType_RELU6) {
                    convPs.pw_activation_type = ACTIVATION_RELU6;
                    convPs.dw_activation_type = ACTIVATION_NULL;
                } else if (activationFunc == tflite::ActivationFunctionType_NONE) {
                    convPs.pw_activation_type = ACTIVATION_NULL;
                    convPs.dw_activation_type = ACTIVATION_NULL;
                } else {
                    CHECK_STATUS_WITH_RETURN(NOT_SUPPORTED);
                }
                opsPtr[j].ps.conv_param_spec = convPs;
            } 

            else if (opCode == tflite::BuiltinOperator_DEPTHWISE_CONV_2D) {
                modelWeightOpNum++;
                // input 2/3: input/weight/bias
                const int inputIndex = ops[j]->inputs[0];
                const auto& inputTensor = tensors[inputIndex];
                ops[j].num_inputs = 1;
                opsPtr[j].input_tensors_name = (I8**)mt_malloc(opsPtr[j].num_inputs * sizeof(I8*));
                str_copy(opsPtr[j].input_tensors_name[0], (inputTensor->name).c_str(), (inputTensor->name).length());
                
                const int weightIndex = ops[j]->inputs[1];
                const auto& weightTensor = tensors[weightIndex];
                auto conv2DWeightPtr = reinterpret_cast<const float*>(tfliteModelBuffer[weightTensor->buffer]->data.data());
                if (ops[j]->inputs.size() == 3) {
                    const int biasIndex = ops[j]->inputs[2];
                    const auto& biasTensor = tensors[biasIndex];
                    auto conv2DBiasPtr = reinterpret_cast<const float*>(tfliteModelBuffer[biasTensor->buffer]->data.data());
                }
                const int outputIndex = ops[j]->outputs[0];
                const auto& outputTensor = tensors[outputIndex];
                ops[j].num_outputs = 1;
                opsPtr[j].output_tensors_name = (I8**)mt_malloc(opsPtr[j].num_outputs * sizeof(I8*));
                str_copy(opsPtr[j].output_tensors_name[0], (outputTensor->name).c_str(), (outputTensor->name).length());

                quantizedModel = weightTensor->type == tflite::TensorType_UINT8;

                // deal with the meta info
                // to extract the Conv2DOptionsT struct
                const auto& weightShape = weightTensor->shape;
                CHECK_REQUIREMENT(weightShape.size() == 4);
                const int conv2d_co = weightShape[0];
                const int conv2d_kh = weightShape[1];
                const int conv2d_kw = weightShape[2];
                const int conv2d_ci = weightShape[3];
                const auto& tfliteConvOption = ops[j]->builtin_options.AsDepthwiseConv2DOptions();
                const int conv2d_dilationX = tfliteConvOption->dilation_w_factor;
                const int conv2d_dilationY = tfliteConvOption->dilation_h_factor;
                const int conv2d_strideX = tfliteConvOption->stride_w;
                const int conv2d_strideY = tfliteConvOption->stride_h;
                const auto activationFunc = tfliteConvOption->fused_activation_function;

                ConvolutionParamSpec convPs;
                convPs.num_kernels = conv2d_co * conv2d_ci;
                convPs.kernel_size = conv2d_kh;
                convPs.stride = conv2d_strideX;
                convPs.padding = (convPs.kernel_size - 1) / 2;
                convPs.group = tfliteConvOption->depth_multiplier;    // to make sure   
                convPs.dilation = conv2d_dilationX;
                if (convPs.group == 1) {
                    if (convPs.dilation > 1) {
                        convPs.convolution_type = Convolution_Dilation;
                    } else {
                        convPs.convolution_type = Convolution_Pointwise;
                    }
                } else {
                    convPs.convolution_type = Convolution_Depthwise;
                }
                if (activationFunc == tflite::ActivationFunctionType_RELU) {
                    convPs.pw_activation_type = ACTIVATION_NULL;
                    convPs.dw_activation_type = ACTIVATION_RELU;
                } else if (activationFunc == tflite::ActivationFunctionType_RELU6) {
                    convPs.pw_activation_type = ACTIVATION_NULL;
                    convPs.dw_activation_type = ACTIVATION_RELU6;
                } else if (activationFunc == tflite::ActivationFunctionType_NONE) {
                    convPs.pw_activation_type = ACTIVATION_NULL;
                    convPs.dw_activation_type = ACTIVATION_NULL;
                } else {
                    CHECK_STATUS_WITH_RETURN(NOT_SUPPORTED);
                }
                opsPtr[j].ps.conv_param_spec = convPs;
            } 

            else if (opCode == tflite::BuiltinOperator_LOGISTIC) {
                ops[j].num_inputs = opInputTensorSize;
                opsPtr[j].input_tensors_name = (I8**)mt_malloc(opsPtr[j].num_inputs * sizeof(I8*));
                for (int iter = 0; iter < opInputTensorSize; iter++) {
                    const int inIndex = ops[j]->inputs[iter];
                    const auto& inTensor = tensors[inIndex];
                    opsPtr[j].input_tensors_name[j] = (I8*)mt_malloc(NAME_LEN * sizeof(I8));
                    str_copy(opsPtr[j].input_tensors_name[iter], (inTensor->name).c_str(), (inTensor->name).length());
                }
                opsPtr[j].num_outputs = opOutputTensorSize;
                opsPtr[j].output_tensors_name = (I8**)mt_malloc(opsPtr[j].num_outputs * sizeof(I8*));
                for (int iter = 0; iter < opOutputTensorSize; iter++) {
                    const int outIndex = ops[j]->outputs[iter];
                    const auto& outTensor = tensors[outIndex];
                    opsPtr[j].output_tensors_name[j] = (I8*)mt_malloc(NAME_LEN * sizeof(I8));
                    str_copy(opsPtr[j].output_tensors_name[iter], (outTensor->name).c_str(), (outTensor->name).length());
                }
            } 

            else if (opCode == tflite::BuiltinOperator_MAX_POOL_2D) {
                ops[j].num_inputs = opInputTensorSize;
                opsPtr[j].input_tensors_name = (I8**)mt_malloc(opsPtr[j].num_inputs * sizeof(I8*));
                for (int iter = 0; iter < opInputTensorSize; iter++) {
                    const int inIndex = ops[j]->inputs[iter];
                    const auto& inTensor = tensors[inIndex];
                    opsPtr[j].input_tensors_name[j] = (I8*)mt_malloc(NAME_LEN * sizeof(I8));
                    str_copy(opsPtr[j].input_tensors_name[iter], (inTensor->name).c_str(), (inTensor->name).length());
                }
                opsPtr[j].num_outputs = opOutputTensorSize;
                opsPtr[j].output_tensors_name = (I8**)mt_malloc(opsPtr[j].num_outputs * sizeof(I8*));
                for (int iter = 0; iter < opOutputTensorSize; iter++) {
                    const int outIndex = ops[j]->outputs[iter];
                    const auto& outTensor = tensors[outIndex];
                    opsPtr[j].output_tensors_name[j] = (I8*)mt_malloc(NAME_LEN * sizeof(I8));
                    str_copy(opsPtr[j].output_tensors_name[iter], (outTensor->name).c_str(), (outTensor->name).length());
                }
                const auto& tflitePoolOption = ops[j]->builtin_options.AsPool2DOptions();
                PoolingParamSpec poolingPs;
                poolingPs.kernel_size = tflitePoolOption->filter_width;
                poolingPs.stride = tflitePoolOption->stride_h;
                poolingPs.padding = 0;    // to make sure how to extract
                poolingPs.rm = CEIL;
                poolingPs.mode = Max;
                opsPtr[j].ps.pooling_param_spec = poolingPs;
            } 

            else if (opCode == tflite::BuiltinOperator_RESHAPE) {    // attention: input is 1
                ops[j].num_inputs = 1;
                opsPtr[j].input_tensors_name = (I8**)mt_malloc(opsPtr[j].num_inputs * sizeof(I8*));
                for (int iter = 0; iter < ops[j].num_inputs; iter++) {
                    const int inIndex = ops[j]->inputs[iter];
                    const auto& inTensor = tensors[inIndex];
                    opsPtr[j].input_tensors_name[j] = (I8*)mt_malloc(NAME_LEN * sizeof(I8));
                    str_copy(opsPtr[j].input_tensors_name[iter], (inTensor->name).c_str(), (inTensor->name).length());
                }
                opsPtr[j].num_outputs = 1;
                opsPtr[j].output_tensors_name = (I8**)mt_malloc(opsPtr[j].num_outputs * sizeof(I8*));
                for (int iter = 0; iter < opsPtr[j].num_outputs; iter++) {
                    const int outIndex = ops[j]->outputs[iter];
                    const auto& outTensor = tensors[outIndex];
                    opsPtr[j].output_tensors_name[j] = (I8*)mt_malloc(NAME_LEN * sizeof(I8));
                    str_copy(opsPtr[j].output_tensors_name[iter], (outTensor->name).c_str(), (outTensor->name).length());
                }
                const auto& shapeTensor = tensors[ops[j]->inputs[1]];    // attention 1
                const auto& shapeData = tfliteModelBuffer[shapeTensor->buffer]->data;
                CHECK_REQUIREMENT((shapeTensor->shape[0]) = (shapeData.size() / 4)));

                ReshapeParamSpec reshapePs;
                reshapePs.shape_size = shapeTensor->shape[0];

                auto reshapeDimPtr = reinterpret_cast<const int32_t*>(shapeData.data());
                std::vector<int> reshapeDim(reshapeDimPtr, reshapeDimPtr + shapeTensor->shape[0]);
                for (int iter = 0; iter < reshapeDim.size() ; iter++) {
                    reshapePs.shape_dims[iter] = reshapeDim[iter];
                }
                opsPtr[j].ps.reshape_param_spec = reshapePs;
            } 

            else if (opCode == tflite::BuiltinOperator_RESIZE_BILINEAR) {
                // 2 inputTensors: the first one is origianl tensor, the other is resize tensor[view it as weight tensor]
                // loop61
                ops[j].num_inputs = 1;
                opsPtr[j].input_tensors_name = (I8**)mt_malloc(opsPtr[j].num_inputs * sizeof(I8*));
                for (int iter = 0; iter < ops[j].num_inputs; iter++) {
                    const int inIndex = ops[j]->inputs[iter];
                    const auto& inTensor = tensors[inIndex];
                    opsPtr[j].input_tensors_name[j] = (I8*)mt_malloc(NAME_LEN * sizeof(I8));
                    str_copy(opsPtr[j].input_tensors_name[iter], (inTensor->name).c_str(), (inTensor->name).length());
                }
                opsPtr[j].num_outputs = 1;
                opsPtr[j].output_tensors_name = (I8**)mt_malloc(opsPtr[j].num_outputs * sizeof(I8*));
                for (int iter = 0; iter < opsPtr[j].num_outputs; iter++) {
                    const int outIndex = ops[j]->outputs[iter];
                    const auto& outTensor = tensors[outIndex];
                    opsPtr[j].output_tensors_name[j] = (I8*)mt_malloc(NAME_LEN * sizeof(I8));
                    str_copy(opsPtr[j].output_tensors_name[iter], (outTensor->name).c_str(), (outTensor->name).length());
                }

                const auto& resizeBilinearOption = ops[j]->builtin_options.AsResizeBilinearOptions();
                const auto& resizeTensor = tensors[ops[j]->inputs[1]];
                const auto& resizeData = tfliteModelBuffer[resizeTensor->buffer]->data;

                auto resizeInfoPtr = reinterpret_cast<const int32_t*>(resizeData.data());
                std::vector<int> resizeInfo(resizeInfoPtr, resizeInfoPtr + resizeData.size() / 4);
            }
        }
    }
    // loop for weight op
    ms->num_weight_specs = modelWeightOpNum;
    WeightSpec* wsPtr = (WeightSpec*)mt_malloc(sizeof(WeightSpec) * ms->num_weight_specs);
    ms->ws = wsPtr;
    int weightIndex = 0;
    for (int i = 0; i < subGraphsSize; ++i) {
        const auto& ops = tfliteModel->subgraphs[i]->operators;
        const auto& tensors = tfliteModel->subgraphs[i]->tensors;
        const int opNums = ops.size();

        // 20191120
        ms->num_operator_specs = opNums;
        OperatorSpec* opsPtr = (OperatorSpec*)mt_malloc(sizeof(OperatorSpec) * ms->num_operator_specs);
        ms->ops = opsPtr;

        for (int j = 0; j < opNums; ++j) {
            std::string curOpName = "op" + std::to_string(j);

            const int opcodeIndex = ops[j]->opcode_index;
            const auto opCode = tfliteOpSet[opcodeIndex]->builtin_code;
            OperatorType opType = tfliteType_to_boltType(opCode);
            opsPtr[j].type = opType;

            // deal with all the op
            const int opInputTensorSize = ops[j]->inputs.size();
            const int opOutputTensorSize = ops[j]->outputs.size();

            if (opCode == tflite::BuiltinOperator_CONV_2D) {
                str_copy(wsPtr[weightIndex].op_name, curOpName.c_str(), curOpName.length());
                wsPtr[weightIndex].mdt = DT_F32;
                // input 2/3: input/weight/bias
                const int inputIndex = ops[j]->inputs[0];
                const auto& inputTensor = tensors[inputIndex];
                ops[j].num_inputs = 1;
                opsPtr[j].input_tensors_name = (I8**)mt_malloc(opsPtr[j].num_inputs * sizeof(I8*));
                str_copy(opsPtr[j].input_tensors_name[0], (inputTensor->name).c_str(), (inputTensor->name).length());

                const int weightIndex = ops[j]->inputs[1];
                const auto& weightTensor = tensors[weightIndex];
                auto conv2DWeightPtr = reinterpret_cast<const float*>(tfliteModelBuffer[weightTensor->buffer]->data.data());
                // deal with the meta info
                // to extract the Conv2DOptionsT struct
                const auto& weightShape = weightTensor->shape;
                CHECK_REQUIREMENT(weightShape.size() == 4);
                const int conv2d_co = weightShape[0];
                const int conv2d_kh = weightShape[1];
                const int conv2d_kw = weightShape[2];
                const int conv2d_ci = weightShape[3];
                wsPtr[weightIndex].bytes_of_weight = conv2d_co * conv2d_kh * conv2d_kw * conv2d_ci * sizeof(float);
                wsPtr[weightIndex].weight = (U8*)conv2DWeightPtr;

                if (ops[j]->inputs.size() == 3) {
                    const int biasIndex = ops[j]->inputs[2];
                    const auto& biasTensor = tensors[biasIndex];
                    auto conv2DBiasPtr = reinterpret_cast<const float*>(tfliteModelBuffer[biasTensor->buffer]->data.data());
                    wsPtr[weightIndex].bytes_of_vec = conv2d_co;
                    wsPtr[weightIndex].vec = (U8*)conv2DBiasPtr;
                } else {
                    wsPtr[weightIndex].bytes_of_vec = 0;
                    wsPtr[weightIndex].vec = nullptr;
                }

                
                weightIndex++;
            } else if (opCode == tflite::BuiltinOperator_DEPTHWISE_CONV_2D) {
                str_copy(wsPtr[weightIndex].op_name, curOpName.c_str(), curOpName.length());
                wsPtr[weightIndex].mdt = DT_F32;
                // input 2/3: input/weight/bias
                const int inputIndex = ops[j]->inputs[0];
                const auto& inputTensor = tensors[inputIndex];
                ops[j].num_inputs = 1;
                opsPtr[j].input_tensors_name = (I8**)mt_malloc(opsPtr[j].num_inputs * sizeof(I8*));
                str_copy(opsPtr[j].input_tensors_name[0], (inputTensor->name).c_str(), (inputTensor->name).length());
                
                const int weightIndex = ops[j]->inputs[1];
                const auto& weightTensor = tensors[weightIndex];
                auto conv2DWeightPtr = reinterpret_cast<const float*>(tfliteModelBuffer[weightTensor->buffer]->data.data());
                // deal with the meta info
                // to extract the Conv2DOptionsT struct
                const auto& weightShape = weightTensor->shape;
                CHECK_REQUIREMENT(weightShape.size() == 4);
                const int conv2d_co = weightShape[0];
                const int conv2d_kh = weightShape[1];
                const int conv2d_kw = weightShape[2];
                const int conv2d_ci = weightShape[3];
                wsPtr[weightIndex].bytes_of_weight = conv2d_co * conv2d_kh * conv2d_kw * conv2d_ci * sizeof(float);
                wsPtr[weightIndex].weight = (U8*)conv2DWeightPtr;

                if (ops[j]->inputs.size() == 3) {
                    const int biasIndex = ops[j]->inputs[2];
                    const auto& biasTensor = tensors[biasIndex];
                    auto conv2DBiasPtr = reinterpret_cast<const float*>(tfliteModelBuffer[biasTensor->buffer]->data.data());
                    wsPtr[weightIndex].bytes_of_vec = conv2d_co;
                    wsPtr[weightIndex].vec = (U8*)conv2DBiasPtr;
                } else {
                    wsPtr[weightIndex].bytes_of_vec = 0;
                    wsPtr[weightIndex].vec = nullptr;
                }
                weightIndex++;
            }
        }
    }
    return SUCCESS;
}
#endif
