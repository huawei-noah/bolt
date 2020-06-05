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
    OperatorType convert_tflite_type(tflite::BuiltinOperator tfliteType)
    {
        if (tfliteType == tflite::BuiltinOperator_ADD) {
            if (tfliteModelBuffer[tensors[ops[curIndex]->inputs[1]]->buffer]->data.size() > 0) {
                return OT_Scale;
            } else {
                return OT_Eltwise;
            }  
        } else if (tfliteType == tflite::BuiltinOperator_CONCATENATION) {
            return OT_Concat;
        } else if (tfliteType == tflite::BuiltinOperator_CONV_2D) {
            return OT_Conv;
        } else if (tfliteType == tflite::BuiltinOperator_DEPTHWISE_CONV_2D) {
            return OT_Conv;
        } else if (tfliteType == tflite::BuiltinOperator_LOGISTIC) {
            return OT_Sigmoid;
        } else if (tfliteType == tflite::BuiltinOperator_MAX_POOL_2D) {
            return OT_Pooling;
        } else if (tfliteType == tflite::BuiltinOperator_AVERAGE_POOL_2D) {
            return OT_Pooling;
        } else if (tfliteType == tflite::BuiltinOperator_RESHAPE) {
            return OT_Reshape;
        } else if (tfliteType == tflite::BuiltinOperator_RESIZE_BILINEAR) {
            return OT_Resize;
        } else if (tfliteType == tflite::BuiltinOperator_SOFTMAX) {
            return OT_Softmax;
        } else if (tfliteType == tflite::BuiltinOperator_FULLY_CONNECTED) {
            if (tfliteModelBuffer[tensors[ops[curIndex]->inputs[1]]->buffer]->data.size() > 0) {
                return OT_FC;
            } else {
                return OT_MatMul;
            }
        } else if (tfliteType == tflite::BuiltinOperator_TRANSPOSE) {
            return OT_Transpose;
        } else if (tfliteType == tflite::BuiltinOperator_SLICE) {
            return OT_Slice;
        } else if (tfliteType == tflite::BuiltinOperator_PACK) {
            return OT_Concat;
        } else if (tfliteType == tflite::BuiltinOperator_MUL) {
            if (tfliteModelBuffer[tensors[ops[curIndex]->inputs[1]]->buffer]->data.size() > 0) {
                return OT_Scale;
            } else {
                return OT_Eltwise;
            }  
        } else if (tfliteType == tflite::BuiltinOperator_DIV) {
            if (tfliteModelBuffer[tensors[ops[curIndex]->inputs[1]]->buffer]->data.size() > 0) {
                return OT_Scale;
            } else {
                return OT_Eltwise;
            }  
        } else if (tfliteType == tflite::BuiltinOperator_SUB) {
            if (tfliteModelBuffer[tensors[ops[curIndex]->inputs[1]]->buffer]->data.size() > 0) {
                return OT_Scale;
            } else {
                return OT_Eltwise;
            }
        } else if (tfliteType == tflite::BuiltinOperator_RELU6) {
            return OT_Relu6;
        } else if (tfliteType == tflite::BuiltinOperator_TANH) {
            return OT_TanH;
        } else {
            std::cerr << "[ERROR] tflite op " << tfliteType << " not implemented yet" << std::endl;
            return OT_None;
        }
    }

    EE parse_file(std::string dir, std::string mfn) override
    {
        EE ret = SUCCESS;
        std::string tfliteSuffix = ".tflite";

        this->modelName = mfn;

        std::string model_name = dir + "/" + mfn + tfliteSuffix;
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

        inputs.clear();
        for (int i=0; i < (int)(tfliteModel->subgraphs[0]->inputs).size(); i++) {
            inputs.push_back(std::move((tfliteModel->subgraphs[0]->inputs)[i]));
        }

        outputs.clear();
        for (int i=0; i < (int)(tfliteModel->subgraphs[0]->outputs).size(); i++) {
            outputs.push_back(std::move((tfliteModel->subgraphs[0]->outputs)[i]));
        }

        return ret;
    }

    EE adapt_operators(ModelSpec* ms) override
    {
        EE ret = SUCCESS;
        str_copy(ms->model_name, modelName.c_str(), modelName.length());
        ms->model_name[NAME_LEN - 1] = '\0';
        ms->dt = DT_F32;

        int opNums = ops.size();

        ms->num_inputs = inputs.size();
        ms->input_names = (I8**)mt_new_storage(ms->num_inputs * sizeof(I8*));
        ms->input_dims  = (TensorDesc*)mt_new_storage(sizeof(TensorDesc) * ms->num_inputs);
        for (I32 i = 0; i < ms->num_inputs; i++) {
            const int inputIdx = inputs[i];
            const auto& inputTensor = tensors[inputIdx];
            const auto& inputShape = inputTensor->shape;
            ms->input_names[i] = (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
            str_copy(ms->input_names[i], (inputTensor->name).c_str(), (inputTensor->name).length());
            switch (inputShape.size()) {
                case 2: {
                    ms->input_dims[i] = tensor2df(DT_F32, DF_NORMAL,
                                                  inputShape[0],
                                                  inputShape[1]);
                    break;
                }
                case 3: {
                    ms->input_dims[i] = tensor3df(DT_F32, DF_MTK,
                                                  inputShape[0],
                                                  inputShape[1],
                                                  inputShape[2]);
                    break;
                }
                case 4: {
                    ms->input_dims[i] = tensor4df(DT_F32, DF_NCHW,
                                                  inputShape[0],
                                                  inputShape[3],
                                                  inputShape[1],
                                                  inputShape[2]);
                    break;
                }
                default: {
                    CHECK_STATUS(NOT_IMPLEMENTED);
                }
            }
        }
        ms->num_outputs = outputs.size();
        ms->output_names = (I8**)mt_new_storage(ms->num_outputs * sizeof(I8*));
        for (I32 i = 0; i < ms->num_outputs; i++) {
            const int outputIdx = outputs[i];
            const auto& outputTensor = tensors[outputIdx];
            ms->output_names[i] = (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
            str_copy(ms->output_names[i], (outputTensor->name).c_str(), (outputTensor->name).length());
        }

        ms->num_operator_specs = opNums;
        opsPtr = (OperatorSpec*)mt_new_storage(sizeof(OperatorSpec) * ms->num_operator_specs);
        ms->ops = opsPtr;
        for (I32 i = 0; i < ms->num_operator_specs; i++) {
            ms->ops[i].tensor_positions = nullptr;
            ms->ops[i].num_quant_feature = 0;
            ms->ops[i].feature_scale = nullptr;
        }

        for (int j = 0; j < ms->num_operator_specs; j++) {
            std::string curOpName = "op" + std::to_string(j);
            str_copy(opsPtr[j].name, curOpName.c_str(), curOpName.length());
            curIndex = j;
            const int opcodeIndex = ops[j]->opcode_index;
            this->opCode = tfliteOpSet[opcodeIndex]->builtin_code;
            OperatorType opType = convert_tflite_type(opCode);
            opsPtr[j].type = opType;
            int opInputTensorSize  = (modifiedInputsOp.count(opType) == 0) ? ops[j]->inputs.size() : modifiedInputsOp[opType];
            int opOutputTensorSize = (modifiedOutputsOp.count(opType) == 0) ? ops[j]->outputs.size() : modifiedOutputsOp[opType];
            opsPtr[j].num_inputs = opInputTensorSize;
            opsPtr[j].input_tensors_name = (I8**)mt_new_storage(opsPtr[j].num_inputs * sizeof(I8*));
            for (int iter = 0; iter < opInputTensorSize; iter++) {
                const int inIndex = ops[j]->inputs[iter];
                const auto& inTensor = tensors[inIndex];
                opsPtr[j].input_tensors_name[iter] = (I8*)mt_new_storage(NAME_LEN * sizeof(I8));
                str_copy(opsPtr[j].input_tensors_name[iter], (inTensor->name).c_str(), (inTensor->name).length());
            }
            opsPtr[j].num_outputs = opOutputTensorSize;
            opsPtr[j].output_tensors_name = (I8**)mt_new_storage(opsPtr[j].num_outputs * sizeof(I8*));
            for (int iter = 0; iter < opOutputTensorSize; iter++) {
                const int outIndex = ops[j]->outputs[iter];
                const auto& outTensor = tensors[outIndex];
                opsPtr[j].output_tensors_name[iter] = (I8*)mt_new_storage(NAME_LEN * sizeof(I8));
                std::string outputName;
                if (opType == OT_Slice) {
                    if (1 != iter) {
                        outputName = "slice_other_" + std::to_string(j) + "_" + std::to_string(iter);
                    } else {
                        const int trueIndex = ops[j]->outputs[0];
                        const auto& out = tensors[trueIndex];
                        outputName = out->name;
                    }
                } else {
                    outputName = outTensor->name;
                }
                str_copy(opsPtr[j].output_tensors_name[iter], outputName.c_str(), outputName.length());
            }

            ParameterSpec curPs;
            ret = adapt_operator(opType, &curPs);
            opsPtr[j].ps = curPs;
        }

        ms->num_weight_specs = modelWeightOpNum;
        return ret;
    }

    void from_nhwc_to_nchw(TensorDesc desc, F32* src, F32* dst)
    {
        DataType dt;
        DataFormat df;
        U32 n, c, h, w;
        CHECK_STATUS(tensor4dGet(desc, &dt, &df, &n, &c, &h, &w));
        CHECK_REQUIREMENT(DF_NHWC == df);

        if (1 == h && 1 == w) {
            memcpy(dst, src, tensorNumBytes(desc));
        } else {
            for (U32 o = 0; o < n; o++) {
                for (U32 hw = 0; hw < h * w; hw++) {
                    for (U32 cc = 0; cc < c; cc++) {
                        dst[o*c*h*w + cc*h*w + hw] = src[o*h*w*c + hw*c + cc];
                    }
                }
            }
        }
    }

    EE adapt_weights(ModelSpec* ms) override
    {
        WeightSpec* wsPtr = (WeightSpec*)mt_new_storage(sizeof(WeightSpec) * ms->num_weight_specs);
        for (int j = 0; j < ms->num_weight_specs; j++) {
            wsPtr[j].num_quant_scale = 0;
            wsPtr[j].weight_scale = nullptr;
        }
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
                wsPtr[weightMovIndex].weight = (U8*)mt_new_storage(wsPtr[weightMovIndex].bytes_of_weight);
                TensorDesc weightDesc = tensor4df(DT_F32, DF_NHWC, conv2d_co, conv2d_ci, conv2d_kh, conv2d_kw);
                from_nhwc_to_nchw(weightDesc, (F32*)conv2DWeightPtr, (F32*)(wsPtr[weightMovIndex].weight));

                if (ops[j]->inputs.size() == 3) {
                    const int biasIndex = ops[j]->inputs[2];
                    const auto& biasTensor = tensors[biasIndex];
                    auto conv2DBiasPtr = reinterpret_cast<const float*>(tfliteModelBuffer[biasTensor->buffer]->data.data());
                    if (opCode == tflite::BuiltinOperator_CONV_2D) {
                        wsPtr[weightMovIndex].bytes_of_vec = conv2d_co * sizeof(float);
                    } else {
                        wsPtr[weightMovIndex].bytes_of_vec = conv2d_ci * sizeof(float);
                    }
                    wsPtr[weightMovIndex].vec = (U8*)mt_new_storage(wsPtr[weightMovIndex].bytes_of_vec);
                    memcpy(wsPtr[weightMovIndex].vec, conv2DBiasPtr, wsPtr[weightMovIndex].bytes_of_vec);
                } else {
                    wsPtr[weightMovIndex].bytes_of_vec = 0;
                    wsPtr[weightMovIndex].vec = nullptr;
                }
                weightMovIndex++;
            } else if (OT_Scale == ms->ops[j].type) {
                str_copy(wsPtr[weightMovIndex].op_name, curOpName.c_str(), curOpName.length());
                wsPtr[weightMovIndex].mdt = DT_F32;
                switch (opCode) {
                    case tflite::BuiltinOperator_ADD: {
                        wsPtr[weightMovIndex].bytes_of_weight = 0;
                        wsPtr[weightMovIndex].weight = nullptr;

                        const int biasIndex = ops[j]->inputs[1];
                        const auto& biasTensor = tensors[biasIndex];
                        auto biasPtr = reinterpret_cast<const float*>(tfliteModelBuffer[biasTensor->buffer]->data.data());
                        wsPtr[weightMovIndex].bytes_of_vec = tfliteModelBuffer[biasTensor->buffer]->data.size();
                        wsPtr[weightMovIndex].vec = (U8*)mt_new_storage(wsPtr[weightMovIndex].bytes_of_vec);
                        memcpy(wsPtr[weightMovIndex].vec, biasPtr, wsPtr[weightMovIndex].bytes_of_vec);
                        break;
                    }
                    case tflite::BuiltinOperator_SUB: {
                        wsPtr[weightMovIndex].bytes_of_weight = 0;
                        wsPtr[weightMovIndex].weight = nullptr;

                        const int biasIndex = ops[j]->inputs[1];
                        const auto& biasTensor = tensors[biasIndex];
                        auto biasPtr = reinterpret_cast<const float*>(tfliteModelBuffer[biasTensor->buffer]->data.data());
                        int size = tfliteModelBuffer[biasTensor->buffer]->data.size() / sizeof(float);
                        wsPtr[weightMovIndex].bytes_of_vec = size * sizeof(float);
                        wsPtr[weightMovIndex].vec = (U8*)mt_new_storage(wsPtr[weightMovIndex].bytes_of_vec);
                        F32 *ptr = (F32*)wsPtr[weightMovIndex].vec;
                        for (int k = 0; k < size; k++) {
                            ptr[k] = -1 * biasPtr[k];
                        }
                        break;
                    }
                    case tflite::BuiltinOperator_MUL: {
                        const int scaleIndex = ops[j]->inputs[1];
                        const auto& scaleTensor = tensors[scaleIndex];
                        auto scalePtr = reinterpret_cast<const float*>(tfliteModelBuffer[scaleTensor->buffer]->data.data());
                        wsPtr[weightMovIndex].bytes_of_weight = tfliteModelBuffer[scaleTensor->buffer]->data.size();
                        wsPtr[weightMovIndex].weight = (U8*)mt_new_storage(wsPtr[weightMovIndex].bytes_of_weight);
                        memcpy(wsPtr[weightMovIndex].weight, scalePtr, wsPtr[weightMovIndex].bytes_of_weight);

                        wsPtr[weightMovIndex].bytes_of_vec = 0;
                        wsPtr[weightMovIndex].vec = nullptr;
                        break;
                    }
                    case tflite::BuiltinOperator_DIV: {
                        const int scaleIndex = ops[j]->inputs[1];
                        const auto& scaleTensor = tensors[scaleIndex];
                        auto scalePtr = reinterpret_cast<const float*>(tfliteModelBuffer[scaleTensor->buffer]->data.data());
                        int size = tfliteModelBuffer[scaleTensor->buffer]->data.size() / sizeof(float);
                        wsPtr[weightMovIndex].bytes_of_weight = size * sizeof(float);
                        wsPtr[weightMovIndex].weight = (U8*)mt_new_storage(wsPtr[weightMovIndex].bytes_of_weight);
                        F32 *ptr = (F32*)wsPtr[weightMovIndex].weight;
                        for (int k = 0; k < size; k++) {
                            ptr[k] = 1 / scalePtr[k];
                        }

                        wsPtr[weightMovIndex].bytes_of_vec = 0;
                        wsPtr[weightMovIndex].vec = nullptr;
                        break;
                    }
                    default: {
                        CHECK_STATUS(NOT_SUPPORTED);
                    }
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
            EltwiseSumSpec elt_sum_spec;
            elt_sum_spec.coeff_size = 2;
            F32* f_ptr = (F32*)mt_new_storage(elt_sum_spec.coeff_size * sizeof(float));
            for (I32 j = 0; j < elt_sum_spec.coeff_size; j++) {
                f_ptr[j] = 1.0;
            }
            elt_sum_spec.coeff_values = f_ptr;
            eltPs.elt_sum_spec = elt_sum_spec;
        } else if (opCode == tflite::BuiltinOperator_MAXIMUM) {
            eltPs.elt_mode = ELTWISE_MAX;
        } else if (opCode == tflite::BuiltinOperator_MUL) {
            eltPs.elt_mode = ELTWISE_PROD;
        } else {
            CHECK_STATUS(NOT_IMPLEMENTED);
        }
        curPs.eltwise_spec = eltPs;
        return curPs;
    }

    ParameterSpec adapt_Scale() override
    {
        ParameterSpec curPs;
        modelWeightOpNum++;
        ScaleParamSpec scalePs;
        scalePs.axis = 0;
        curPs.scale_spec = scalePs;
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

        ConvolutionParamSpec convPs;
        convPs.kernel_size_h = weightShape[1];
            convPs.kernel_size_w = weightShape[2];

        if (opCode == tflite::BuiltinOperator_CONV_2D) {
            convPs.num_outputs = weightShape[0];

            const auto& tfliteConvOption = ops[curIndex]->builtin_options.AsConv2DOptions();
            convPs.dilatedRate_h = tfliteConvOption->dilation_h_factor;
            convPs.dilatedRate_w = tfliteConvOption->dilation_w_factor;
            convPs.stride_h = tfliteConvOption->stride_h;
            convPs.stride_w = tfliteConvOption->stride_w;
            const auto activationFunc = tfliteConvOption->fused_activation_function;

            if (1 == tfliteConvOption->padding) {  // VALID
                convPs.padding_top = 0;
                convPs.padding_bottom = 0;
                convPs.padding_left = 0;
                convPs.padding_right = 0;
            } else {  // SAME
                convPs.padding_top = (convPs.kernel_size_h - 1) / 2;
                convPs.padding_bottom = (convPs.kernel_size_h - 1) / 2;
                convPs.padding_left = (convPs.kernel_size_w - 1) / 2;
                convPs.padding_right = (convPs.kernel_size_w - 1) / 2;
            }
            
            convPs.group = 1;

            convPs.dw_activation_type = ACTIVATION_NULL;
            convPs.pw_activation_type = ACTIVATION_NULL;
            
            if (convPs.dilatedRate_h > 1 || convPs.dilatedRate_w > 1) {
                convPs.convolution_type = Convolution_Dilation;
            } else {
                convPs.convolution_type = Convolution_Pointwise;
            }
            if (activationFunc == tflite::ActivationFunctionType_RELU) {
                convPs.pw_activation_type = ACTIVATION_RELU;
            } else if (activationFunc == tflite::ActivationFunctionType_RELU6) {
                convPs.pw_activation_type = ACTIVATION_RELU6;
            } else if (activationFunc != tflite::ActivationFunctionType_NONE) {
                std::cout << "[ERROR] tflite activation " << activationFunc << " not merged with conv yet\n";
                CHECK_STATUS(NOT_IMPLEMENTED);
            }
        } else if (opCode == tflite::BuiltinOperator_DEPTHWISE_CONV_2D) {
            convPs.num_outputs = weightShape[3];

            const auto& tfliteConvOption = ops[curIndex]->builtin_options.AsDepthwiseConv2DOptions();
            convPs.dilatedRate_h = tfliteConvOption->dilation_h_factor;
            convPs.dilatedRate_w = tfliteConvOption->dilation_w_factor;
            convPs.stride_h = tfliteConvOption->stride_h;
            convPs.stride_w = tfliteConvOption->stride_w;
            const auto activationFunc = tfliteConvOption->fused_activation_function;

            if (1 == tfliteConvOption->padding) {  // VALID
                convPs.padding_top = 0;
                convPs.padding_bottom = 0;
                convPs.padding_left = 0;
                convPs.padding_right = 0;
            } else {  // SAME
                convPs.padding_top = (convPs.kernel_size_h - 1) / 2;
                convPs.padding_bottom = (convPs.kernel_size_h - 1) / 2;
                convPs.padding_left = (convPs.kernel_size_w - 1) / 2;
                convPs.padding_right = (convPs.kernel_size_w - 1) / 2;
            }
            
            convPs.group = convPs.num_outputs;

            convPs.dw_activation_type = ACTIVATION_NULL;
            convPs.pw_activation_type = ACTIVATION_NULL;
            
            convPs.convolution_type = Convolution_Depthwise;
            if (activationFunc == tflite::ActivationFunctionType_RELU) {
                convPs.dw_activation_type = ACTIVATION_RELU;
            } else if (activationFunc == tflite::ActivationFunctionType_RELU6) {
                convPs.dw_activation_type = ACTIVATION_RELU6;
            } else if (activationFunc != tflite::ActivationFunctionType_NONE) {
                std::cout << "[ERROR] tflite activation " << activationFunc << " not merged with depthwise conv yet\n";
                CHECK_STATUS(NOT_IMPLEMENTED);
            }
        } else {
            CHECK_STATUS(NOT_SUPPORTED);
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
        CHECK_REQUIREMENT((shapeTensor->shape[0]) == (int)(shapeData.size() / sizeof(int)));

        ReshapeParamSpec reshapePs;
        reshapePs.shape_size = shapeTensor->shape[0];

        auto reshapeDimPtr = reinterpret_cast<const int32_t*>(shapeData.data());
        std::vector<int> reshapeDim(reshapeDimPtr, reshapeDimPtr + reshapePs.shape_size);

        const auto& inputTensor = tensors[ops[curIndex]->inputs[0]];
        const auto& inputShape = inputTensor->shape;
        if ((U32)reshapePs.shape_size < inputShape.size() &&
            4 == inputShape.size() && 
            (1 != inputShape[1] || 1 != inputShape[2]) ) {
            opsPtr[curIndex].type = OT_Transpose;
            TransposeParamSpec transposePs;
            transposePs.trans_size = 4;
            bool taken[4] = {false};
            U32 i;
            for (i = 0; i < reshapeDim.size(); i++) {
                for (U32 j = 0; j < inputShape.size(); j++) {
                    if (inputShape[j] == reshapeDim[i]) {
                        taken[j] = true;
                        transposePs.trans_dims[i] = j;
                        break;
                    }
                }
            }
            for (U32 j = 0; j < 4; j++) {
                if (!taken[j]) {
                    transposePs.trans_dims[i] = j;
                    i++;
                    taken[j] = true;
                }
            }
            curPs.transpose_spec = transposePs;
        } else {
            if (4 == reshapeDim.size()) {
                reshapePs.shape_dims[0] = reshapeDim[0];
                reshapePs.shape_dims[1] = reshapeDim[3];
                reshapePs.shape_dims[2] = reshapeDim[1];
                reshapePs.shape_dims[3] = reshapeDim[2];
            } else {
                for (int iter = 0; iter < (int)reshapeDim.size() ; iter++) {
                    reshapePs.shape_dims[iter] = reshapeDim[iter];
                }
            }
            curPs.reshape_spec = reshapePs;
        }
        return curPs;
    }

    ParameterSpec adapt_Transpose() override
    {
        ParameterSpec curPs;
        TransposeParamSpec transPs;
        const auto& dimsTensor = tensors[ops[curIndex]->inputs[1]];
        const auto& dimsData = tfliteModelBuffer[dimsTensor->buffer]->data;
        CHECK_REQUIREMENT((dimsTensor->shape[0]) == (int)(dimsData.size() / sizeof(int)));
        transPs.trans_size = dimsTensor->shape[0];
        auto dims = reinterpret_cast<const int32_t*>(dimsData.data());
        for (U32 i = 0; i < transPs.trans_size; i++)
            transPs.trans_dims[i] = dims[i];
        curPs.transpose_spec = transPs;
        std::cout << "[INFO] tflite operator transpose dims: ";
        for (U32 i = 0; i < transPs.trans_size; i++)
            std::cout << transPs.trans_dims[i] << " ";
        std::cout << std::endl;
        return curPs;
    }

    ParameterSpec adapt_Slice() override
    {
        // TODO: Tensorflow slice is not similar with Caffe
        // currently only support one axis slice
        ParameterSpec curPs;
        SliceParamSpec slicePs;
        const auto& inputShape = tensors[ops[curIndex]->inputs[0]]->shape;
        const auto& beginTensor = tensors[ops[curIndex]->inputs[1]];
        auto beginData = reinterpret_cast<const int32_t*>((tfliteModelBuffer[beginTensor->buffer]->data).data());
        const auto& sizeTensor = tensors[ops[curIndex]->inputs[2]];
        auto sizeData = reinterpret_cast<const int32_t*>((tfliteModelBuffer[sizeTensor->buffer]->data).data());
        I32 axis = INT_MIN;
        for (I32 i = 0; i < beginTensor->shape[0]; i++) {
            if (! (beginData[i] == 0 && (sizeData[i] == -1 || sizeData[i] == inputShape[i]))) {
                if (axis != INT_MIN) {
                    std::cerr << "[ERROR] currently not support multi axis slice" << std::endl;
                    exit(1);
                } else {
                    axis = i;
                }
            }
        }
        slicePs.axis = axis;
        slicePs.slice_size = 2;
        slicePs.slice_points[0] = beginData[axis];
        I32 size = sizeData[axis];
        if (size  == -1) {
            slicePs.slice_points[1] = inputShape[axis];
        } else {
            slicePs.slice_points[1] = beginData[axis] + sizeData[axis];
        }
        if (4 == inputShape.size()) {
            switch (slicePs.axis) {
                case 0:
                    slicePs.axis = 0;
                    break;
                case 1:
                    slicePs.axis = 2;
                    break;
                case 2:
                    slicePs.axis = 3;
                    break;
                case 3:
                    slicePs.axis = 1;
                    break;
                default:
                    CHECK_STATUS(NOT_SUPPORTED);
            }
        }
        curPs.slice_spec = slicePs;
        return curPs;
    }

    ParameterSpec adapt_MatMul() override    // matrix X matrix
    {
        ParameterSpec curPs;
        MatMulParamSpec matmulPs;
        matmulPs.transpose_a = false;
        matmulPs.transpose_b = false;
        curPs.matmul_spec = matmulPs;
        return curPs;
    }

    ParameterSpec adapt_Fc() override
    {
        modelWeightOpNum++;
        ParameterSpec curPs;
        FullyConnectedParamSpec ips;
        const int index = ops[curIndex]->inputs[1];
        const auto& tensor = tensors[index];
        I32 size = tfliteModelBuffer[tensor->buffer]->data.size();
        CHECK_REQUIREMENT(size != 0);
        const auto& weightShape = tensor->shape;
        ips.num_outputs = weightShape[1];
        ips.num_slices = 1;
        ips.slice_point[0] = ips.num_outputs;
        curPs.fc_spec = ips;
        return curPs;
    }

    
    ParameterSpec adapt_Concat() override
    {
        ParameterSpec curPs;
        ConcatParamSpec concatPs;
        const auto& tfliteConcatOption = ops[curIndex]->builtin_options.AsConcatenationOptions();
        CHECK_REQUIREMENT(tflite::ActivationFunctionType_NONE == tfliteConcatOption->fused_activation_function);
        concatPs.axis = tfliteConcatOption->axis;

        const auto& outputTensor = tensors[ops[curIndex]->outputs[0]];
        const auto& outputShape = outputTensor->shape;
        if (4 == outputShape.size()) {
            switch (concatPs.axis) {
                case 0:
                    concatPs.axis = 0;
                    break;
                case 1:
                    concatPs.axis = 2;
                    break;
                case 2:
                    concatPs.axis = 3;
                    break;
                case 3:
                    concatPs.axis = 1;
                    break;
                default:
                    CHECK_STATUS(NOT_SUPPORTED);
            }
        }
        curPs.concat_spec = concatPs;
        return curPs;
    }

    ParameterSpec adapt_Softmax() override
    {
        const auto& tfliteSoftmaxOption = ops[curIndex]->builtin_options.AsSoftmaxOptions();
        CHECK_REQUIREMENT(1 == tfliteSoftmaxOption->beta);

        ParameterSpec curPs;
        SoftmaxParamSpec softmaxPs;
        softmaxPs.axis = -1;

        const auto& inputTensor = tensors[ops[curIndex]->inputs[0]];
        const auto& inputShape = inputTensor->shape;
        if (4 == inputShape.size()) {
            softmaxPs.axis = 1;
        }
        curPs.softmax_spec = softmaxPs;
        return curPs;
    }

public:
    std::map<OperatorType, int> modifiedInputsOp {{OT_Conv, 1}, {OT_Reshape, 1},
        {OT_Resize, 1}, {OT_Transpose, 1}, {OT_FC, 1}, {OT_Slice, 1}, {OT_Scale, 1}};
    std::map<OperatorType, int> modifiedOutputsOp {{OT_Slice, 3}};
private:
    std::vector<std::unique_ptr<tflite::BufferT>> tfliteModelBuffer;
    std::vector<std::unique_ptr<tflite::OperatorCodeT>> tfliteOpSet;
    std::vector<std::unique_ptr<tflite::OperatorT>> ops;
    std::vector<std::unique_ptr<tflite::TensorT>> tensors;
    std::vector<int> inputs;
    std::vector<int> outputs;
    tflite::BuiltinOperator opCode;
    int modelWeightOpNum;
    int curIndex;
    std::string modelName;
    OperatorSpec* opsPtr;
};
