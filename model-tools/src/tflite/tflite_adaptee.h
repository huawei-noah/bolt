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
#include <map>
#include <schema_generated.h>
#include "type.h"
#include "converter.h"
#include "model_serialize_deserialize.hpp"
#include "model_tools.h"
#include "model_adaptee.h"

class TfliteAdaptee: public ModelAdaptee {
public:
    TfliteAdaptee() {}
    ~TfliteAdaptee() {}

protected:
    OperatorType convert_caffe_type(tflite::BuiltinOperator tfliteType) {
        if (tfliteType == tflite::BuiltinOperator_ADD) {
            return OT_Eltwise;  
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
            return OT_Resize;
        }
        return OT_None;
    }

    bool is_modified_inputs_op(OperatorType opType) {
        if (modifiedInputsOp.count(opType) > 0) {
            return true;
        } else {
            return false;
        }
    }


    EE parse_file(std::string dir, std::string mfn) override {
        EE ret = SUCCESS;
        std::string tfliteSuffix = ".tflite";
        std::string model_name = dir + "/" + "mfn" + tfliteSuffix;
        std::ifstream inputFile(model_name.c_str(), std::ios::binary);
        inputFile.seekg(0, std::ios::end);
        const auto size = inputFile.tellg();
        inputFile.seekg(0, std::ios::beg);

        char* buffer = new char[size];
        inputFile.read(buffer, size);
        inputFile.close();

        flatbuffers::Verifier verify((uint8_t*)buffer, size);
        CHECK_REQUIREMENT(tflite::VerifyModelBuffer(verify));

        auto tfliteModel = tflite::UnPackModel(buffer);

        tfliteOpSet.clear();
        for (int i = 0; i < (int)(tfliteModel->operator_codes).size(); i++) {
            tfliteOpSet.push_back(std::move((tfliteModel->operator_codes)[i]));
        }

        const auto subGraphsSize = tfliteModel->subgraphs.size(); 
        CHECK_REQUIREMENT(subGraphsSize == 1);

	tfliteModelBuffer.clear();
	for (int i = 0; i < (int)(tfliteModel->buffers).size(); i++) {
            tfliteModelBuffer.push_back(std::move((tfliteModel->buffers)[i]));
        }

        if (subGraphsSize != 1) {   
            CHECK_STATUS(NOT_SUPPORTED);
        }

	ops.clear();
        for (int i=0; i < (int)(tfliteModel->subgraphs[0]->operators).size(); i++) {
            ops.push_back(std::move((tfliteModel->subgraphs[0]->operators)[i]));
        }

	tensors.clear();
        for (int i=0; i < (int)(tfliteModel->subgraphs[0]->tensors).size(); i++) {
            tensors.push_back(std::move((tfliteModel->subgraphs[0]->tensors)[i]));
        }

        return ret;
    }

    EE adapt_operators(ModelSpec* ms) override
    {
	    EE ret = SUCCESS;
        int opNums = ops.size();
        ms->num_operator_specs = opNums;
        OperatorSpec* opsPtr = (OperatorSpec*)mt_new_storage(sizeof(OperatorSpec) * ms->num_operator_specs);
        ms->ops = opsPtr;
        for (I32 i = 0; i < ms->num_operator_specs; i++) {
            ms->ops[i].tensor_positions = nullptr;
        }

        for (int j = 0; j < ms->num_operator_specs; j++) {
            std::string curOpName = "op" + std::to_string(j);
            curIndex = j;
            const int opcodeIndex = ops[j]->opcode_index;
            const auto opCode = tfliteOpSet[opcodeIndex]->builtin_code;
            OperatorType opType = convert_caffe_type(opCode);
            opsPtr[j].type = opType;
            bool modifiedTag = is_modified_inputs_op(opType);
            int opInputTensorSize = (modifiedTag == false) ? ops[j]->inputs.size() : modifiedInputsOp[opType];
            int opOutputTensorSize = ops[j]->outputs.size();
            opsPtr[j].num_inputs = opInputTensorSize;
            opsPtr[j].input_tensors_name = (I8**)mt_new_storage(opsPtr[j].num_inputs * sizeof(I8*));
            for (int iter = 0; iter < opInputTensorSize; iter++) {
                const int inIndex = ops[j]->inputs[iter];
                const auto& inTensor = tensors[inIndex];
                opsPtr[j].input_tensors_name[j] = (I8*)mt_new_storage(NAME_LEN * sizeof(I8));
                str_copy(opsPtr[j].input_tensors_name[iter], (inTensor->name).c_str(), (inTensor->name).length());
            }
            opsPtr[j].num_outputs = opOutputTensorSize;
            opsPtr[j].output_tensors_name = (I8**)mt_new_storage(opsPtr[j].num_outputs * sizeof(I8*));
            for (int iter = 0; iter < opOutputTensorSize; iter++) {
                const int outIndex = ops[j]->outputs[iter];
                const auto& outTensor = tensors[outIndex];
                opsPtr[j].output_tensors_name[j] = (I8*)mt_new_storage(NAME_LEN * sizeof(I8));
                str_copy(opsPtr[j].output_tensors_name[iter], (outTensor->name).c_str(), (outTensor->name).length());
            }

            ParameterSpec curPs;
            ret = adapt_operator(opType, &curPs);
            opsPtr[j].ps = curPs;
        }

        ms->num_weight_specs = modelWeightOpNum;
        return ret;
    }

    EE adapt_weights(ModelSpec* ms) override
    {
        WeightSpec* wsPtr = (WeightSpec*)mt_new_storage(sizeof(WeightSpec) * ms->num_weight_specs);
        ms->ws = wsPtr;
        int weightMovIndex = 0;
        for (int j = 0; j < ms->num_operator_specs; j++) {
            std::string curOpName = "op" + std::to_string(j);
            curIndex = j;
            const int opcodeIndex = ops[j]->opcode_index;
            opCode = tfliteOpSet[opcodeIndex]->builtin_code;

            if (opCode == tflite::BuiltinOperator_CONV_2D || opCode == tflite::BuiltinOperator_DEPTHWISE_CONV_2D) {
                str_copy(wsPtr[weightMovIndex].op_name, curOpName.c_str(), curOpName.length());
                wsPtr[weightMovIndex].mdt = DT_F32;

                // input 2/3: input/weight/bias
                const int weightIndex = ops[j]->inputs[1];
                const auto& weightTensor = tensors[weightIndex];
                auto conv2DWeightPtr = reinterpret_cast<const float*>(tfliteModelBuffer[weightTensor->buffer]->data.data());
                const auto& weightShape = weightTensor->shape;
                CHECK_REQUIREMENT(weightShape.size() == 4);
                const int conv2d_co = weightShape[0];
                const int conv2d_kh = weightShape[1];
                const int conv2d_kw = weightShape[2];
                const int conv2d_ci = weightShape[3];
                wsPtr[weightMovIndex].bytes_of_weight = conv2d_co * conv2d_kh * conv2d_kw * conv2d_ci * sizeof(float);
                wsPtr[weightMovIndex].weight = (U8*)conv2DWeightPtr;

                if (ops[j]->inputs.size() == 3) {
                    const int biasIndex = ops[j]->inputs[2];
                    const auto& biasTensor = tensors[biasIndex];
                    auto conv2DBiasPtr = reinterpret_cast<const float*>(tfliteModelBuffer[biasTensor->buffer]->data.data());
                    wsPtr[weightMovIndex].bytes_of_vec = conv2d_co;
                    wsPtr[weightMovIndex].vec = (U8*)conv2DBiasPtr;
                } else {
                    wsPtr[weightMovIndex].bytes_of_vec = 0;
                    wsPtr[weightMovIndex].vec = nullptr;
                }
                weightMovIndex++;
            } 
        }
        return SUCCESS;
    }

    ParameterSpec adapt_Eltwise() override
    {
        ParameterSpec curPs;
        EltwiseParamSpec eltPs;
        if (opCode == tflite::BuiltinOperator_ADD) {
            eltPs.elt_mode = ELTWISE_SUM;
        } else if (opCode == tflite::BuiltinOperator_MAXIMUM){
            eltPs.elt_mode = ELTWISE_MAX;
        } else {
            eltPs.elt_mode = ELTWISE_PROD;
        }
        curPs.eltwise_spec = eltPs;
        return curPs;
    }

    ParameterSpec adapt_Conv() override
    {
        ParameterSpec curPs;
        modelWeightOpNum++;
        const int weightIndex = ops[curIndex]->inputs[1];
        const auto& weightTensor = tensors[weightIndex];

        const auto& weightShape = weightTensor->shape;
        CHECK_REQUIREMENT(weightShape.size() == 4);
        const int conv2d_co = weightShape[0];
        const int conv2d_kh = weightShape[1];
        const int conv2d_kw = weightShape[2];
        const int conv2d_ci = weightShape[3];
        const auto& tfliteConvOption = ops[curIndex]->builtin_options.AsDepthwiseConv2DOptions();
        const int conv2d_dilationX = tfliteConvOption->dilation_w_factor;
        const int conv2d_dilationY = tfliteConvOption->dilation_h_factor;
        const int conv2d_strideX = tfliteConvOption->stride_w;
        const int conv2d_strideY = tfliteConvOption->stride_h;
        const auto activationFunc = tfliteConvOption->fused_activation_function;

        ConvolutionParamSpec convPs;
        convPs.num_kernels = conv2d_co * conv2d_ci;
        convPs.kernel_size_h = conv2d_kh;
        convPs.kernel_size_w = conv2d_kw;
        convPs.stride_h = conv2d_strideX;
        convPs.stride_w = conv2d_strideY;
        // TODO: extract padding info
        convPs.padding_top = (convPs.kernel_size_h - 1) / 2;
        convPs.padding_bottom = (convPs.kernel_size_h - 1) / 2;
        convPs.padding_left = (convPs.kernel_size_w - 1) / 2;
        convPs.padding_right = (convPs.kernel_size_w - 1) / 2;
        
        if (opCode == tflite::BuiltinOperator_DEPTHWISE_CONV_2D) {
            convPs.group = tfliteConvOption->depth_multiplier;   
            convPs.dilatedRate_h = conv2d_dilationX;
            convPs.dilatedRate_w = conv2d_dilationY;
        } else {
            convPs.dilatedRate_h = 1;
            convPs.dilatedRate_w = 1;
        }
        
        if (convPs.group == 1) {
            if (convPs.dilatedRate_h > 1 || convPs.dilatedRate_w > 1) {
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
	    std::cout << "[ERROR]: UNSUPPORTED!" << std::endl;
        }
        curPs.conv_spec = convPs;
        return curPs;
    }

    ParameterSpec adapt_Pooling() override
    {
        ParameterSpec curPs;
        const auto& tflitePoolOption = ops[curIndex]->builtin_options.AsPool2DOptions();
        PoolingParamSpec poolingPs;
        poolingPs.kernel_size_h = tflitePoolOption->filter_height;
        poolingPs.kernel_size_w = tflitePoolOption->filter_width;
        poolingPs.stride_h = tflitePoolOption->stride_h;
        poolingPs.stride_w = tflitePoolOption->stride_w;
        // TODO: padding support
        poolingPs.padding_top = 0;
        poolingPs.padding_bottom = 0;
        poolingPs.padding_left = 0;
        poolingPs.padding_right = 0;
        poolingPs.rm = CEIL;
        if (opCode == tflite::BuiltinOperator_MAX_POOL_2D) {
            poolingPs.mode = POOLING_MAX;
        } else if (opCode == tflite::BuiltinOperator_AVERAGE_POOL_2D) {
            poolingPs.mode = POOLING_MEAN;
        }        
        curPs.pooling_spec = poolingPs;
        return curPs;
    }

    ParameterSpec adapt_Reshape() override
    {
        ParameterSpec curPs;
        const auto& shapeTensor = tensors[ops[curIndex]->inputs[1]]; 
        const auto& shapeData = tfliteModelBuffer[shapeTensor->buffer]->data;
        CHECK_REQUIREMENT((shapeTensor->shape[0]) == (int)(shapeData.size() / 4));

        ReshapeParamSpec reshapePs;
        reshapePs.shape_size = shapeTensor->shape[0];

        auto reshapeDimPtr = reinterpret_cast<const int32_t*>(shapeData.data());
        std::vector<int> reshapeDim(reshapeDimPtr, reshapeDimPtr + shapeTensor->shape[0]);
        for (int iter = 0; iter < (int)reshapeDim.size() ; iter++) {
            reshapePs.shape_dims[iter] = reshapeDim[iter];
        }
        curPs.reshape_spec = reshapePs;
        return curPs;
    }

public:
    std::map<OperatorType, int> modifiedInputsOp {{OT_Conv, 1}, {OT_Reshape, 1}, {OT_Resize, 1}};
private:
    std::vector<std::unique_ptr<tflite::BufferT>> tfliteModelBuffer;
    std::vector<std::unique_ptr<tflite::OperatorCodeT>> tfliteOpSet;
    std::vector<std::unique_ptr<tflite::OperatorT>> ops;
    std::vector<std::unique_ptr<tflite::TensorT>> tensors;
    tflite::BuiltinOperator opCode;
    int modelWeightOpNum;
    int curIndex;
};
