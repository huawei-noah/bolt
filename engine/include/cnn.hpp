// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _CNN_H
#define _CNN_H

#include <set>
#include <string>
#include <cstring>
#include <map>
#include <tuple>
#include <queue>
#include <typeinfo>
#include "model.hpp"
#include "model_tools.h"
#include "factory.hpp"
#include "tensor.hpp"
#include "operator.hpp"
#include "convolution.hpp"
#include "fully_connected_eltwise.hpp"
#include "tensor_desc.h"
#include "model_tools.h"

template <Arch A>
class CNN: public Model<A> {
public:

    /**
     * @param name
     */
    explicit CNN(DataType dt, std::string name) : Model<A>(dt, name) { }
    virtual ~CNN() = default;
    /**
     * @param op
     * @param in
     * @param out
     */

    void initialize_ops(const ModelSpec* ms)
    {
        int opNum = ms->num_operator_specs;

        U32 shouldNeverQuantize = 1; // The first convolution layer should not be quantized

        Vec<std::string> modelInputTensorNames;
        for (int i=0; i < ms->num_inputs; i++) {
            modelInputTensorNames.push_back(ms->input_names[i]);
        }
        this->modelInputTensorNames = modelInputTensorNames;

        DataType dtNoQ = (this->dt == DT_F16_8Q) ? DT_F16 : this->dt; // Possibly more cases in the future

        for (int i = 0; i < opNum; i++) {
            OperatorSpec curOps = ms->ops[i];
            std::string opName = curOps.name;
            if (opName.compare("data") == 0) {
                continue;
            }
            OperatorType opType = curOps.type;

            Vec<std::string> inputTensorsName;
            Vec<std::string> outputTensorsName;
            int inputTensorsNum = curOps.num_inputs;
            for (int j = 0; j < inputTensorsNum; j++) {
                inputTensorsName.push_back(curOps.input_tensors_name[j]);
            }
            
            int outputTensorsNum = curOps.num_outputs;
            for (int j = 0; j < outputTensorsNum; j++) {
                outputTensorsName.push_back(curOps.output_tensors_name[j]);
            }

            // create op object
            std::shared_ptr<Operator<A>> op;
            switch (opType) {
                case OT_Conv: {
                    ConvolutionParamSpec curConvParamSpec = curOps.ps.conv_param_spec;
                    U32 nf = curConvParamSpec.num_kernels;
                    U32 ksize = curConvParamSpec.kernel_size;
                    ConvolutionMode curConvolutionType = curConvParamSpec.convolution_type;
                    U32 group = curConvParamSpec.group;
                    U32 dilation = curConvParamSpec.dilation;
                    U32 kstride = curConvParamSpec.stride;
                    U32 padding = curConvParamSpec.padding;
                    ActivationMode dwActiveMode = curConvParamSpec.dw_activation_type;
                    ActivationMode pwActiveMode = curConvParamSpec.pw_activation_type;
                    if (shouldNeverQuantize == 1) {
                        // The first convolution should never be quantized. Assume model is F16
                        op = Factory::createConvolution<A>(dtNoQ, nf, ksize, kstride, padding, dwActiveMode, pwActiveMode, curConvolutionType, group, dilation);
                        shouldNeverQuantize = 0;
                    } else {
                        // The following convolutions can be quantized, so just follow what the ms specifies
                        // BNN convolutions will be handled later
                        op = Factory::createConvolution<A>(this->dt, nf, ksize, kstride, padding, dwActiveMode, pwActiveMode, curConvolutionType, group, dilation);
                    }
                    break;
                }
                case OT_FC: {
                    FullyConnectedParamSpec curIpParamSpec = curOps.ps.ip_param_spec;
                    U32 curNumOutput = curIpParamSpec.num_outputs;
                    op = Factory::createFullyConnectedEltwise<A>(dtNoQ, curNumOutput);
                    break;
                }
                case OT_Pooling: {
                    PoolingParamSpec curPoolingParamSpec = curOps.ps.pooling_param_spec;
                    PoolingMode mode = curPoolingParamSpec.mode;
                    U32 ks = curPoolingParamSpec.kernel_size;
                    U32 stride = curPoolingParamSpec.stride;
                    U32 padding = curPoolingParamSpec.padding;
                    RoundMode rm = curPoolingParamSpec.rm;
                    op = Factory::createPooling<A>(mode, ks, stride, padding, rm);
                    break;
                }
                case OT_Softmax: {
                    op = Factory::createSoftmax<A>(dtNoQ);
                    break;
                }
                case OT_Relu: {
                    ActivationMode activeMode = ACTIVATION_RELU;
                    op = Factory::createActivation<A>(activeMode, opType);
                    break;
                }
                case OT_Relu6: {
                    ActivationMode activeMode = ACTIVATION_RELU6;
                    op = Factory::createActivation<A>(activeMode, opType);
                    break;
                }
                case OT_HSwish: {
                    ActivationMode activeMode = ACTIVATION_H_SWISH;
                    op = Factory::createActivation<A>(activeMode, opType);
                    break;
                }
                case OT_HSigmoid: {
                    ActivationMode activeMode = ACTIVATION_H_SIGMOID;
                    op = Factory::createActivation<A>(activeMode, opType);
                    break;
                }
                case OT_Gelu: {
                    ActivationMode activeMode = ACTIVATION_GELU;
                    op = Factory::createActivation<A>(activeMode, opType);
                    break;
                }
                case OT_TanH: {
                    ActivationMode activeMode = ACTIVATION_TANH;
                    op = Factory::createActivation<A>(activeMode, opType);
                    break;
                }
                case OT_Concat: {
                    U32 concatDim = 1;
                    op = Factory::createConcat<A>(concatDim);
                    break;
                }
                case OT_Eltwise: {
                    EltwiseParamSpec curEltwiseParamSpec = curOps.ps.eltwise_param_spec;
                    EltwiseMode curEltMode = curEltwiseParamSpec.elt_mode;
                    EltwiseSumSpec curEltSumSpec = curEltwiseParamSpec.elt_sum_spec;
                    op = Factory::createEltwise<A>(curEltMode, curEltSumSpec.coeff_size, curEltSumSpec.coeff_values);
                    break;
                }
                case OT_Embedding: {
                    EmbedParamSpec curEmbedParamSpec = curOps.ps.embed_spec;
                    U32 curInputDim = curEmbedParamSpec.input_dim;
                    U32 curNumOutput = curEmbedParamSpec.num_output;
                    op = Factory::createEmbedding<A>(dtNoQ, curInputDim, curNumOutput);
                    break;
                }
                case OT_MatMul: {
                    op = Factory::createMatMul<A>(dtNoQ);
                    break;
                }
                case OT_Multiply: {
                    MultiplyParamSpec curMultiplyParamSpec = curOps.ps.multiply_spec;
                    F16 factor = curMultiplyParamSpec.scale;
                    op = Factory::createMultiply<A>(dtNoQ, factor);
                    break;
                }
                case OT_Scale: {
                    op = Factory::createScale<A>(dtNoQ);
                    break;
                }
                case OT_LayerNorm: {
                    op = Factory::createLayerNorm<A>(dtNoQ);
                    break;
                }
                case OT_Reshape: {
                    ReshapeParamSpec curReshapeParamSpec = curOps.ps.reshape_spec;
                    I32* curShapeDims = curReshapeParamSpec.shape_dims;
                    I32 curShapeSize = curReshapeParamSpec.shape_size;
                    I32 curAxis = curReshapeParamSpec.axis;
                    I32 curNumAxes = curReshapeParamSpec.num_axes;
                    op = Factory::createReshape<A>(dtNoQ, curShapeDims, curShapeSize, curAxis, curNumAxes);
                    break;
                }
                case OT_Slice: {
                    SliceParamSpec curSliceParamSpec = curOps.ps.slice_spec;
                    U32 curAxis = curSliceParamSpec.axis;
                    U32* curSlicePoints = curSliceParamSpec.slice_points;
                    U32 curSliceSize = curSliceParamSpec.slice_size;
                    op = Factory::createSlice<A>(dtNoQ, curAxis, curSlicePoints, curSliceSize);
                    break;
                }
                case OT_Transpose: {
                    TransposeParamSpec curTransposeSpec = curOps.ps.transpose_spec;
                    U32* curTransDimsPtr = curTransposeSpec.trans_dims;
                    U32 curTransSize = curTransposeSpec.trans_size;
                    op = Factory::createTranspose<A>(dtNoQ, curTransDimsPtr, curTransSize);
                    break;
                }
                case OT_Attention: {
                    AttentionParamSpec curAttentionSpec = curOps.ps.attention_spec;
                    I32 curNumAttention = curAttentionSpec.num_attention;
                    op = Factory::createAttention<A>(dtNoQ, curNumAttention);
                    break;
                }case OT_Clip: {
                    ClipParamSpec curClipSpec = curOps.ps.clip_spec;
                    F16 curClipMinScalar = (F16)curClipSpec.min;
                    F16 curClipMaxScalar = (F16)curClipSpec.max;
                    op = Factory::createClip<A>(dtNoQ, curClipMinScalar, curClipMaxScalar);
                    break;
                }case OT_Squeeze: {
                    op = Factory::createSqueeze<A>(dtNoQ);
                    break;
                }
                default: {
                    std::cerr << "[ERROR] unsupported layer " << OperatorTypeName()[opType] << std::endl;
                    exit(0);
                    break;
                }
            }

            op->set_op_name(opName);
            op->set_op_type(opType);
            this->ops.push_back(op);

            // setup operatorMap, tensorMap, operatorTensorMap
            this->add(op, inputTensorsName, outputTensorsName);
        }

        // setup WeightSpec ptr in WeightOperator
        for (int i = 0; i < ms->num_weight_specs; i++) {
            WeightSpec curOpWs = ms->ws[i];
            std::string opName = curOpWs.op_name;
            auto op = this->operatorMap[opName];
            auto weightOp = dynamic_cast<WeightOperator<A>*>(op.get());
            weightOp->set_weightspec_ptr(curOpWs);
            if (curOpWs.bytes_of_vec != 0) {
                assert(curOpWs.vec != nullptr);
                weightOp->set_hasBias(true);
            }
        }
    }

    void add(std::shared_ptr<Operator<A>> op, Vec<std::string> inputTensorsName, Vec<std::string> outputTensorsName)
    {
        std::string name = op->get_name();
        this->operatorMap.insert(std::pair(name, op));

        std::tuple<Vec<std::string>, Vec<std::string>> in_outTensors(std::make_tuple(inputTensorsName, outputTensorsName));
        this->operatorTensorMap.insert(std::pair(name, in_outTensors));

        for (std::string input : inputTensorsName) {
            std::shared_ptr<U8> tmpVal;
            TensorDesc tmpTensorDesc;
            std::shared_ptr<F16> tmpScalePtr;
            auto tmp = std::shared_ptr<Tensor>(new Tensor(tmpTensorDesc, tmpVal, tmpScalePtr));
            auto p = std::pair<std::string, std::shared_ptr<Tensor>>(input, tmp);
            this->tensorMap.insert(p);
        }

        for (std::string output : outputTensorsName) {
            std::shared_ptr<U8> tmpVal;
            TensorDesc tmpTensorDesc;
            auto tmp = std::shared_ptr<Tensor>(new Tensor(tmpTensorDesc, tmpVal));
            auto p = std::pair<std::string, std::shared_ptr<Tensor>>(output, tmp);
            this->tensorMap.insert(p);
        }
    }

    HashMap<std::string, std::shared_ptr<Tensor>> get_inputs()
    {
        return this->inputTensors;
    }

    void set_input_tensors_value(HashMap<std::string, std::shared_ptr<U8>> modelTensorsInput) {
        for (int i = 0; i < (int)(this->sortedOps.size()); i++) {
            std::string curOpName = this->sortedOps[i];
            auto curOp = this->ops[i];
            auto inOutTensorNames = this->operatorTensorMap[curOpName];
            auto inTensorNames = std::get<0>(inOutTensorNames);
            int inTensorNamesSize = inTensorNames.size();

            auto opInTensorsOriginal = curOp->get_input_tensors();
            Vec<Tensor> opInTensorUpdate;
            for (int j = 0; j < inTensorNamesSize; j++) {
                const bool isIn = (modelTensorsInput.find(inTensorNames[j]) != modelTensorsInput.end());
                if (isIn == true) {
                    TensorDesc tmpTensorDesc = opInTensorsOriginal[j].get_desc();
                    auto tmpTensorPtr = modelTensorsInput[inTensorNames[j]];
                    Tensor tmpTensor = Tensor(tmpTensorDesc, tmpTensorPtr);
                    opInTensorUpdate.push_back(tmpTensor);
                } else {
                    opInTensorUpdate.push_back(opInTensorsOriginal[j]);    // the tensor not change
                }
            }
            curOp->set_input_tensors(opInTensorUpdate);
        }
    } 

    void set_input_tensors_desc(Vec<TensorDesc> dims, Vec<std::string> modelInputTensorNames) {
        // set up the model_inputTensors
        for (auto i = 0; i < (int)dims.size(); i++) {
            std::string curInputTensorName = modelInputTensorNames[i];
            TensorDesc curTensorDesc = dims[i];
            (this->tensorMap[curInputTensorName].get())->set_desc(curTensorDesc);
        }
    }

    EE infer_output_tensors_size(Vec<TensorDesc> dims) override
    {
        UNUSED(dims);
#ifdef _DEBUG
        const char* funcStr = "[DEBUG] infer_output_tensors_size()";
        std::cout << funcStr << std::endl;
#endif

        int opsNum = this->sortedOps.size();
        for (int i = 0; i < opsNum; i++) {
            std::string opName = sortedOps[i];

            auto op = this->operatorMap[opName];
#ifdef _DEBUG
            std::cout << funcStr << " op: " << opName << std::endl;
#endif
            Vec<std::string> curOpInputTensorName = std::get<0>(this->operatorTensorMap[opName]);
            Vec<std::string> curOpOutputTensorName = std::get<1>(this->operatorTensorMap[opName]);
            int curOpOutNum = curOpOutputTensorName.size();
            Vec<TensorDesc> inTensorDescs;
            Vec<TensorDesc> outTensorDescs;
            for (int j = 0; j < curOpOutNum; j++){
                TensorDesc dummyTensorDesc;
                outTensorDescs.push_back(dummyTensorDesc);
            }
            for (std::string inputTensorName: curOpInputTensorName) {
#ifdef _DEBUG
                std::cout << "    inputTensorName: " << inputTensorName << " ";
#endif
                inTensorDescs.push_back(this->tensorMap[inputTensorName]->desc);
#ifdef _DEBUG
                tensorDescPrint(this->tensorMap[inputTensorName]->desc);
#endif
            }
            CHECK_STATUS_WITH_RETURN(op->infer_output_tensors_size(inTensorDescs, &outTensorDescs));

            for (int k = 0; k < curOpOutNum; k++) {
                std::string outputTensorName = curOpOutputTensorName[k];
#ifdef _DEBUG
                std::cout << "    outputTensorName: " << outputTensorName << " ";
#endif
                TensorDesc outputTensorDesc = outTensorDescs[k];  
#ifdef _DEBUG
                tensorDescPrint(outputTensorDesc);
#endif
                std::shared_ptr<U8> val;
                auto curOutputTensorSp = std::shared_ptr<Tensor>(new Tensor(outputTensorDesc, val));
                this->tensorMap[outputTensorName] = curOutputTensorSp;
            }
        }
        return SUCCESS;
    }

    HashMap<std::string, std::shared_ptr<Tensor>> get_outputs()
    {
        return this->outputTensors;
    }

    /**
     * @param inputTensorsName
     * @param outputTensorsName
     */
    EE mark_input_output(const ModelSpec* ms)
    {
        inputTensors.clear();
        for (I32 i = 0; i < ms->num_inputs; i++) {
            std::string str = ms->input_names[i];
            auto it = tensorMap.find(str);

            if (tensorMap.end() != it) {
                (*(it->second)).desc = ms->input_dims[i];
                inputTensors.insert(*it);
            } else {
                return NOT_MATCH;
            }

        }

        outputTensors.clear();
        for (I32 i = 0; i < ms->num_outputs; i++) {
            std::string str = ms->output_names[i];
            auto it = tensorMap.find(str);
            
            if (tensorMap.end() != it) {
                outputTensors.insert(*it);
            } else {
                return NOT_MATCH;
            }
        }

        return SUCCESS;
    }

    void assign_output_tensor() override
    {
#ifdef _DEBUG
        const char* funcStr = "[DEBUG] assign_output_tensor()";
        std::cout << funcStr << std::endl;
#endif
        // no consideration for tensor reuse
        // get every output tensor desc
        // accumulate the element
        // malloc space for every output tensor
        //TODO carefully check the memory assign here!
        for (std::string opName: sortedOps) {
#ifdef _DEBUG
            std::cout << funcStr << " op: " << opName << "\n    input  tensor names: ";
#endif
            std::shared_ptr<Operator<A>> op = this->operatorMap[opName];
            Vec<Tensor> inTensors;
            Vec<std::string> inTensorNames = std::get<0>(this->operatorTensorMap[opName]);
            for (std::string inName: inTensorNames){
#ifdef _DEBUG
                std::cout << inName << " ";
#endif
                inTensors.push_back(*(this->tensorMap[inName].get()));
            }
#ifdef _DEBUG
            std::cout << "\n    output tensor names: ";
#endif

            Vec<Tensor> outTensors;
            Vec<std::string> outTensorNames = std::get<1>(this->operatorTensorMap[opName]);
            for (std::string outName: outTensorNames) {
#ifdef _DEBUG
                std::cout << outName << " ";
#endif
                this->tensorMap[outName]->alloc();
                outTensors.push_back(*(this->tensorMap[outName].get()));
            }
#ifdef _DEBUG
            std::cout << std::endl;
#endif

            op->set_input_output_tensors(inTensors, outTensors);
        }
    }

    void infer_tmp_memory_size() override
    {
        this->tmpElements.clear();
        this->maxTmpElements = 0;

        for (auto op: this->ops) {
            auto len = op->infer_tmp_memory_size();
            this->tmpElements.push_back(len);
            if (len > (this->maxTmpElements)) {
                this->maxTmpElements = len;
            }
        }
    }

    void assign_tmp_tensor() override
    {
        // design for serial , if parallel running should redesign
        auto sPtr = std::shared_ptr<U8>((U8*)operator new(this->maxTmpElements));
        for (auto op: this->ops) {
            op->set_tmp_memory(this->maxTmpElements, sPtr);
        }
    }

    void ready(Vec<TensorDesc> dims) override
    {
#ifdef _DEBUG
        const char* funcStr = "[DEBUG] ready()";
        std::cout << funcStr << " Model input num: " << this->modelInputTensorNames.size() << std::endl;
        for (auto item: this->modelInputTensorNames) {
            std::cerr << "    input: " << item << std::endl;
        }
#endif

        this->set_input_tensors_desc(dims, this->modelInputTensorNames);

        this->infer_output_tensors_size(dims);

        this->assign_output_tensor();

        // handle the weight ops
        int index = 0;
        for (auto op : this->ops) {
#ifdef _DEBUG
            std::cout << funcStr << " op: " << op->get_name() << std::endl;
#endif
            if (op->is_weight()) {
                if (op->get_op_type() == OT_Conv) {
                    auto convOpPtr = dynamic_cast<Convolution<A>*>(op.get());
                    CHECK_STATUS(convOpPtr->init_weight_bias_from_model(nullptr));
                    
                    CHECK_STATUS(convOpPtr->infer_forward_algorithm());

                    CHECK_STATUS(convOpPtr->transform_filter());
                } else if (op->get_op_type() == OT_FC) {
                    auto fcOpPtr = dynamic_cast<FullyConnectedEltwise<A>*>(op.get());
                    CHECK_STATUS(fcOpPtr->init_weight_bias_from_model(nullptr));

                    CHECK_STATUS(fcOpPtr->transform_filter());
                } else if (op->get_op_type() == OT_Embedding) {
                    auto embeddingOpPtr = dynamic_cast<Embedding<A>*>(op.get());
                    CHECK_STATUS(embeddingOpPtr->init_weight_bias_from_model(nullptr));
                } else if (op->get_op_type() == OT_LayerNorm) {
                    auto layernormOpPtr = dynamic_cast<LayerNorm<A>*>(op.get());
                    CHECK_STATUS(layernormOpPtr->init_weight_bias_from_model(nullptr));
                }
            }
            index++;
        }

        // infer tmp and assign
        this->infer_tmp_memory_size();
        this->assign_tmp_tensor();
    }

    void sort_operators_sequential(const ModelSpec* ms)
    {
        int opNum = ms->num_operator_specs;
        for (int i = 0; i < opNum; i++) {
            std::string opName = ms->ops[i].name;
            if (opName.compare("data") == 0) {
                continue;
            }
            this->sortedOps.push_back(opName);
        }
    }

    void infer_output_tensor_reuse_analysis()
    {
        reused.clear();
        for (auto op: this->ops) {
            reused.push_back(op->can_input_output_the_same());
        }
    }

    void set_modelInputTensorNames(Vec<std::string> modelInputTensorNames) {
        this->modelInputTensorNames = modelInputTensorNames;
    }

    Vec<std::string> get_model_input_tensor_names() {
        return  this->modelInputTensorNames;
    }

private:
    HashMap<std::string, std::shared_ptr<Tensor>> tensorMap;
    HashMap<std::string, std::shared_ptr<Operator<A>>> operatorMap;
    HashMap<std::string, std::tuple<Vec<std::string>, Vec<std::string>>> operatorTensorMap;

    //input output reuse or not
    Vec<bool> reused;

    //input & output tensors
    HashMap<std::string, std::shared_ptr<Tensor>> inputTensors;
    HashMap<std::string, std::shared_ptr<Tensor>> outputTensors;

    Vec<std::string> sortedOps;

    U32 maxTmpElements;
    Vec<U32> tmpElements;

    Vec<TensorDesc> modelInDims;

    Vec<std::string> modelInputTensorNames;
};
#endif
