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

#include <string>
#include <cstring>
#include <tuple>
#include <typeinfo>
#include "model.hpp"
#include "model_tools.h"
#include "tensor.hpp"
#include "operator.hpp"
#include "tensor_desc.h"
#include "factory.hpp"
#include "cpu/factory_cpu.hpp"
#ifdef _USE_MALI
#include "libkernelbin.h"
#include "ocl/factory_ocl.hpp"
#endif

class CNN: public Model {
public:

    /**
     * @param name
     */
    explicit CNN(Arch arch, DataType dt, std::string name) : Model(arch, dt, name) { 
#ifdef _USE_MALI
        if(arch == MALI){
            gclTempMem = NULL;
        }
#endif
    }
#ifdef _USE_MALI
    virtual ~CNN(){
        if(this->schedule == MALI){
            if(gclTempMem){
                CHECK_STATUS(gcl_release_memory(this->gclTempMem));
                gcl_destroy_gclmem(gclTempMem);
            }
        }
    }
#else
    virtual ~CNN() = default;
#endif
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

        U32 operatorIndex = 0;
        HashMap<std::string, U32> operatorIndexMap;
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

            int numTensors = inputTensorsNum + outputTensorsNum;
            Vec<I32> tensorPositions(numTensors);
            memcpy(tensorPositions.data(), curOps.tensor_positions, numTensors * bytesOf(DT_I32));

            // create op object
            std::shared_ptr<Factory> factory;
            std::shared_ptr<Operator> op;
#ifdef _USE_MALI
            if(this->schedule == MALI){
                auto factory_ocl = (Factory*)(new FactoryOCL());
                factory = std::shared_ptr<Factory>(factory_ocl);
            } else {
#endif            
                auto factory_cpu = (Factory*)(new FactoryCPU());
                factory = std::shared_ptr<Factory>(factory_cpu);
#ifdef _USE_MALI
            }
#endif           
            switch (opType) {
                case OT_Conv: {
                    ConvolutionParamSpec curConvParamSpec = curOps.ps.conv_spec;
                    U32 nf = curConvParamSpec.num_kernels;
                    U32 ksizeH = curConvParamSpec.kernel_size_h;
                    U32 ksizeW = curConvParamSpec.kernel_size_w;
                    ConvolutionMode curConvolutionType = curConvParamSpec.convolution_type;
                    U32 group = curConvParamSpec.group;
                    U32 dilateH = curConvParamSpec.dilatedRate_h;
                    U32 dilateW = curConvParamSpec.dilatedRate_w;
                    U32 kstrideH = curConvParamSpec.stride_h;
                    U32 kstrideW = curConvParamSpec.stride_w;
                    U32 paddingT = curConvParamSpec.padding_top;
                    U32 paddingB = curConvParamSpec.padding_bottom;
                    U32 paddingL = curConvParamSpec.padding_left;
                    U32 paddingR = curConvParamSpec.padding_right;
                    ActivationMode dwActiveMode = curConvParamSpec.dw_activation_type;
                    ActivationMode pwActiveMode = curConvParamSpec.pw_activation_type;
                    if (shouldNeverQuantize == 1) {
                        // The first convolution should never be quantized. Assume model is F16
                        op = factory->createConvolution(dtNoQ,
                                                        nf, ksizeH, ksizeW, kstrideH, kstrideW,
                                                        paddingT, paddingB, paddingL, paddingR,
                                                        dwActiveMode, pwActiveMode, curConvolutionType, group, dilateH, dilateW);
                        shouldNeverQuantize = 0;
                    } else {
                        // The following convolutions can be quantized, so just follow what the ms specifies
                        // BNN convolutions will be handled later
                        op = factory->createConvolution(this->dt, nf, ksizeH, ksizeW, kstrideH, kstrideW,
                                                        paddingT, paddingB, paddingL, paddingR,
                                                        dwActiveMode, pwActiveMode, curConvolutionType, group, dilateH, dilateW);
                    }
                    break;
                }
                case OT_Deconvolution: {
                    ConvolutionParamSpec curConvParamSpec = curOps.ps.conv_spec;
                    U32 nf = curConvParamSpec.num_kernels;
                    U32 ksizeH = curConvParamSpec.kernel_size_h;
                    U32 ksizeW = curConvParamSpec.kernel_size_w;
                    ConvolutionMode curConvolutionType = curConvParamSpec.convolution_type;
                    U32 group = curConvParamSpec.group;
                    U32 dilateH = curConvParamSpec.dilatedRate_h;
                    U32 dilateW = curConvParamSpec.dilatedRate_w;
                    U32 kstrideH = curConvParamSpec.stride_h;
                    U32 kstrideW = curConvParamSpec.stride_w;
                    U32 paddingT = curConvParamSpec.padding_top;
                    U32 paddingB = curConvParamSpec.padding_bottom;
                    U32 paddingL = curConvParamSpec.padding_left;
                    U32 paddingR = curConvParamSpec.padding_right;
                    ActivationMode dwActiveMode = curConvParamSpec.dw_activation_type;
                    ActivationMode pwActiveMode = curConvParamSpec.pw_activation_type;
                    op = factory->createDeconvolution(dtNoQ, nf, ksizeH, ksizeW, kstrideH, kstrideW,
                                                      paddingT, paddingB, paddingL, paddingR,
                                                      dwActiveMode, pwActiveMode, curConvolutionType, group, dilateH, dilateW);
                    break;
                }
                case OT_FC: {
                    FullyConnectedParamSpec curIpParamSpec = curOps.ps.fc_spec;
                    U32 curNumOutput = curIpParamSpec.num_outputs;
                    op = factory->createFullyConnectedEltwise(dtNoQ, 0, curNumOutput);
                    break;
                }
                case OT_Pooling: {
                    PoolingParamSpec curPoolingParamSpec = curOps.ps.pooling_spec;
                    PoolingMode mode = curPoolingParamSpec.mode;
                    U32 ksH = curPoolingParamSpec.kernel_size_h;
                    U32 ksW = curPoolingParamSpec.kernel_size_w;
                    U32 strideH = curPoolingParamSpec.stride_h;
                    U32 strideW = curPoolingParamSpec.stride_w;
                    U32 paddingT = curPoolingParamSpec.padding_top;
                    U32 paddingB = curPoolingParamSpec.padding_bottom;
                    U32 paddingL = curPoolingParamSpec.padding_left;
                    U32 paddingR = curPoolingParamSpec.padding_right;
                    RoundMode rm = curPoolingParamSpec.rm;
                    op = factory->createPooling(mode, ksH, ksW, strideH, strideW, paddingT, paddingB, paddingL, paddingR, rm);
                    break;
                }
                case OT_Softmax: {
                    op = factory->createSoftmax(dtNoQ);
                    break;
                }
                case OT_Relu: {
                    ActivationMode activeMode = ACTIVATION_RELU;
                    op = factory->createActivation(activeMode);
                    break;
                }
                case OT_Relu6: {
                    ActivationMode activeMode = ACTIVATION_RELU6;
                    op = factory->createActivation(activeMode);
                    break;
                }
                case OT_HSwish: {
                    ActivationMode activeMode = ACTIVATION_H_SWISH;
                    op = factory->createActivation(activeMode);
                    break;
                }
                case OT_Sigmoid: {
                    ActivationMode activeMode = ACTIVATION_SIGMOID;
                    op = factory->createActivation(activeMode);
                    break;
                }
                case OT_HSigmoid: {
                    ActivationMode activeMode = ACTIVATION_H_SIGMOID;
                    op = factory->createActivation(activeMode);
                    break;
                }
                case OT_Gelu: {
                    ActivationMode activeMode = ACTIVATION_GELU;
                    op = factory->createActivation(activeMode);
                    break;
                }
                case OT_TanH: {
                    ActivationMode activeMode = ACTIVATION_TANH;
                    op = factory->createActivation(activeMode);
                    break;
                }
                case OT_Concat: {
                    U32 concatDim = 1;
                    op = factory->createConcat(concatDim);
                    break;
                }
                case OT_Eltwise: {
                    EltwiseParamSpec curEltwiseParamSpec = curOps.ps.eltwise_spec;
                    EltwiseMode curEltMode = curEltwiseParamSpec.elt_mode;
                    EltwiseSumSpec curEltSumSpec = curEltwiseParamSpec.elt_sum_spec;
                    op = factory->createEltwise(curEltMode, curEltSumSpec.coeff_size, curEltSumSpec.coeff_values);
                    break;
                }
                case OT_Embedding: {
                    EmbedParamSpec curEmbedParamSpec = curOps.ps.embed_spec;
                    U32 curInputDim = curEmbedParamSpec.input_dim;
                    U32 curNumOutput = curEmbedParamSpec.num_output;
                    bool curTranspose = curEmbedParamSpec.transpose;
                    op = factory->createEmbedding(dtNoQ, curInputDim, curNumOutput, curTranspose);
                    break;
                }
                case OT_MatMul: {
                    MatMulParamSpec curMatMulParamSpec = curOps.ps.matmul_spec;
                    bool transposeA = curMatMulParamSpec.transpose_a;
                    bool transposeB = curMatMulParamSpec.transpose_b;
                    op = factory->createMatMul(dtNoQ, transposeA, transposeB);
                    break;
                }
                case OT_Multiply: {
                    MultiplyParamSpec curMultiplyParamSpec = curOps.ps.multiply_spec;
                    F32 scale = curMultiplyParamSpec.scale;
                    F32 bias = curMultiplyParamSpec.bias;
                    op = factory->createMultiply(dtNoQ, scale, bias);
                    break;
                }
                case OT_Scale: {
                    ScaleParamSpec curScaleParamSpec = curOps.ps.scale_spec;
                    I32 num = curScaleParamSpec.num_concat;
                    op = factory->createScale(dtNoQ, -1, num);
                    break;
                }
                case OT_LayerNorm: {
                    op = factory->createLayerNorm(dtNoQ, 0);
                    break;
                }
                case OT_Reshape: {
                    ReshapeParamSpec curReshapeParamSpec = curOps.ps.reshape_spec;
                    I32* curShapeDims = curReshapeParamSpec.shape_dims;
                    I32 curShapeSize = curReshapeParamSpec.shape_size;
                    I32 curAxis = curReshapeParamSpec.axis;
                    I32 curNumAxes = curReshapeParamSpec.num_axes;
                    op = factory->createReshape(dtNoQ, curShapeDims, curShapeSize, curAxis, curNumAxes);
                    break;
                }
                case OT_Upsample: {
                    UpsampleParamSpec curUpsampleParamSpec = curOps.ps.upsample_spec;
                    F32* paramPtr = curUpsampleParamSpec.scale + 2;
                    op = factory->createResize(DT_F32, paramPtr);
                    break;
                }
                case OT_Interp: {
                    InterpParamSpec curInterpParamSpec = curOps.ps.interp_spec;
                    U32 size[2];
                    size[0] = curInterpParamSpec.height;
                    size[1] = curInterpParamSpec.width;
                    op = factory->createResize(DT_U32, size);
                    break;
                }
                case OT_Slice: {
                    SliceParamSpec curSliceParamSpec = curOps.ps.slice_spec;
                    U32 curAxis = curSliceParamSpec.axis;
                    U32* curSlicePoints = curSliceParamSpec.slice_points;
                    U32 curSliceSize = curSliceParamSpec.slice_size;
                    op = factory->createSlice(dtNoQ, curAxis, curSlicePoints, curSliceSize);
                    break;
                }
                case OT_Transpose: {
                    TransposeParamSpec curTransposeSpec = curOps.ps.transpose_spec;
                    U32* curTransDimsPtr = curTransposeSpec.trans_dims;
                    U32 curTransSize = curTransposeSpec.trans_size;
                    op = factory->createTranspose(dtNoQ, curTransDimsPtr, curTransSize);
                    break;
                }
                case OT_Attention: {
                    AttentionParamSpec curAttentionSpec = curOps.ps.attention_spec;
                    U32 numHeads = curAttentionSpec.num_heads;
                    U32 fromSequenceLength = curAttentionSpec.from_sequence_length;
                    U32 toSequenceLength = curAttentionSpec.to_sequence_length;
                    op = factory->createAttention(dtNoQ, numHeads, fromSequenceLength, toSequenceLength);
                    break;
                }
                case OT_Clip: {
                    ClipParamSpec curClipSpec = curOps.ps.clip_spec;
                    F32 curClipMinScalar = curClipSpec.min;
                    F32 curClipMaxScalar = curClipSpec.max;
                    op = factory->createClip(dtNoQ, curClipMinScalar, curClipMaxScalar);
                    break;
                }
                case OT_LSTM: {
                    LSTMParamSpec curLSTMParamSpec = curOps.ps.lstm_spec;
                    U32 numOutput = curLSTMParamSpec.num_output;
                    I32 steps = curLSTMParamSpec.steps;
                    op = factory->createLSTM(dtNoQ, numOutput, steps);
                    break;
                }
                case OT_Squeeze: {
                    SqueezeParamSpec curSqueezeParamSpec = curOps.ps.squeeze_spec;
                    I32 axis = curSqueezeParamSpec.axis;
                    I32 *squeezeAxes = curSqueezeParamSpec.squeeze_axes;
                    I32 numAxes = curSqueezeParamSpec.axes_num;
                    op = factory->createSqueeze(dtNoQ, axis, squeezeAxes, numAxes);
                    break;
                }
                case OT_Unsqueeze: {
                    UnsqueezeParamSpec curUnsqueezeParamSpec = curOps.ps.unsqueeze_spec;
                    I32 axis = curUnsqueezeParamSpec.axis;
                    I32 *unsqueezeAxes = curUnsqueezeParamSpec.unsqueeze_axes;
                    I32 numAxes = curUnsqueezeParamSpec.axes_num;
                    op = factory->createUnsqueeze(dtNoQ, axis, unsqueezeAxes, numAxes);
                    break;
                }
                case OT_AxisMean: {
                    AxisMeanParamSpec curAxisMeanParamSpec = curOps.ps.axis_mean_spec;
                    I32 axis = curAxisMeanParamSpec.axis;
                    op = factory->createAxisMean(dtNoQ, axis);
                    break;
                }
                case OT_ArgMax: {
                    ArgMaxParamSpec curArgMaxParamSpec = curOps.ps.argmax_spec;
                    I32 axis = curArgMaxParamSpec.axis;
                    op = factory->createArgMax(dtNoQ, axis);
                    break;
                }
                case OT_PreAllocatedMemory: {
                    PreAllocatedMemoryParamSpec curPreAllocatedMemoryParamSpec = curOps.ps.preallocated_memory_spec;
                    TensorDesc desc= curPreAllocatedMemoryParamSpec.desc;
                    op = factory->createPreAllocatedMemory(dtNoQ, desc);
                    break;
                }
                case OT_SharedWeight: {
                    SharedWeightParamSpec curSharedWeightParamSpec = curOps.ps.shared_weight_spec;
                    TensorDesc desc = curSharedWeightParamSpec.desc;
                    op = factory->createSharedWeight(dtNoQ, desc);
                    break;
                }
                case OT_Repeat: {
                    RepeatParamSpec curRepeatParamSpec = curOps.ps.repeat_spec;
                    I32 loops = curRepeatParamSpec.loops;
                    op = factory->createRepeat(dtNoQ, loops, operatorIndexMap[inputTensorsName[0]], operatorIndex);
                    break;
                }
                case OT_Check: {
                    CheckParamSpec curCheckParamSpec = curOps.ps.check_spec;
                    CheckMode checkMode = curCheckParamSpec.check_mode;
                    op = factory->createCheck(dtNoQ, checkMode);
                    break;
                }
                case OT_Copy: {
                    CopyParamSpec curCopyParamSpec = curOps.ps.copy_spec;
                    U32 *srcDims = curCopyParamSpec.src_dims;
                    U32 *dstDims = curCopyParamSpec.dst_dims;
                    U32 length = curCopyParamSpec.length;
                    op = factory->createCopy(dtNoQ, srcDims, dstDims, length);
                    break;
                }
                case OT_BilateralSliceApply: {
                    BilateralSliceApplyParamSpec curBilateralSliceApplyParamSpec = curOps.ps.bilateral_slice_apply_spec;
                    U32 coefficient_len = curBilateralSliceApplyParamSpec.coefficient_len;
                    bool has_offset = curBilateralSliceApplyParamSpec.has_offset;
                    BilateralSliceApplyMode mode = curBilateralSliceApplyParamSpec.mode;
                    op = factory->createBilateralSliceApply(coefficient_len, has_offset, mode);
                    break;
                }
                default: {
                    std::cerr << "[ERROR] unsupported layer " << OperatorTypeName()[opType] << std::endl;
                    exit(1);
                    break;
                }
            }

            op->set_op_name(opName);
#ifdef _USE_MALI
            if(this->schedule == MALI) CHECK_STATUS(op->set_mali_handle(this->handle));
#endif
            op->set_op_schedule(this->schedule);
            op->set_tensor_positions(tensorPositions);
            this->ops.push_back(op);
            operatorIndexMap[opName] = operatorIndex++;

            // setup operatorMap, tensorMap, operatorTensorMap
            this->add(op, inputTensorsName, outputTensorsName);
        }

        // setup WeightSpec ptr in WeightOperator
        for (int i = 0; i < ms->num_weight_specs; i++) {
            WeightSpec curOpWs = ms->ws[i];
            std::string opName = curOpWs.op_name;
            auto op = this->operatorMap[opName];
            auto weightOp = dynamic_cast<WeightOperator*>(op.get());
            weightOp->set_weightspec_ptr(curOpWs);
            if (curOpWs.bytes_of_vec != 0) {
                CHECK_REQUIREMENT(curOpWs.vec != nullptr);
                weightOp->set_hasBias(true);
            }
            // These two pointers will be managed by engine via shared_ptr, so mt_destroy_model should not free them
            ms->ws[i].weight = nullptr;
            ms->ws[i].vec = nullptr;
        }
    }

    Tensor get_tensor_by_name(std::string tensorName) {
        if (this->tensorMap.find(tensorName) != this->tensorMap.end())
            return *(this->tensorMap[tensorName].get());
        else {
            std::shared_ptr<Tensor> tensor(new Tensor());
            TensorDesc desc;
            desc.dt = this->dt;
            desc.nDims = 0;
            tensor->set_desc(desc);
            return *tensor.get();
        }
    }

    void add(std::shared_ptr<Operator> op, Vec<std::string> inputTensorsName, Vec<std::string> outputTensorsName)
    {
        std::string name = op->get_name();
        this->operatorMap.insert(std::pair(name, op));

        std::tuple<Vec<std::string>, Vec<std::string>> in_outTensors(std::make_tuple(inputTensorsName, outputTensorsName));
        if (this->operatorTensorMap.find(name) == this->operatorTensorMap.end()) {
            this->operatorTensorMap.insert(std::pair(name, in_outTensors));
        }
        else {
            std::cout << "[ERROR] duplicate tensor name: " << name << std::endl;
            exit(1);
        }
        this->operatorTensorMap[name] = in_outTensors;

        for (std::string input : inputTensorsName) {
	    std::shared_ptr<Tensor> tmp;
#ifdef _USE_MALI
	    if(this->schedule == MALI){
                tmp = std::shared_ptr<Tensor>(new Tensor(this->handle));
	    } else {
#endif
                tmp = std::shared_ptr<Tensor>(new Tensor());
#ifdef _USE_MALI
	    }
#endif
            auto p = std::pair<std::string, std::shared_ptr<Tensor>>(input, tmp);
            this->tensorMap.insert(p);
        }

        for (std::string output : outputTensorsName) {
	    std::shared_ptr<Tensor> tmp;
#ifdef _USE_MALI
	    if(this->schedule == MALI){
                tmp = std::shared_ptr<Tensor>(new Tensor(this->handle));
	    } else {
#endif
                tmp = std::shared_ptr<Tensor>(new Tensor());
#ifdef _USE_MALI
	    }
#endif
            auto p = std::pair<std::string, std::shared_ptr<Tensor>>(output, tmp);
            this->tensorMap.insert(p);
        }
    }

    HashMap<std::string, std::shared_ptr<Tensor>> get_inputs()
    {
        return this->inputTensors;
    }

    void copy_to_named_input(std::string inputName, U8* data)
    {
        auto tensorPtr = this->inputTensors[inputName];
        TensorDesc desc = tensorPtr->desc;
        memcpy(tensorPtr->get_val(), data, tensorNumBytes(desc));
    }

    void set_input_tensors_value(HashMap<std::string, std::shared_ptr<U8>> modelTensorsInput)
    {
        for (int i = 0; i < (int)(this->sortedOps.size()); i++) {
            std::string curOpName = this->sortedOps[i];
            auto curOp = this->ops[i];
            auto inOutTensorNames = this->operatorTensorMap[curOpName];
            auto inTensorNames = std::get<0>(inOutTensorNames);
            int  inTensorNamesSize = inTensorNames.size();

            auto opInTensorsOriginal = curOp->get_input_tensors();
            Vec<Tensor> opInTensorUpdate;
            bool isIn;
            int  validTensorCount = 0;
            for (int j = 0; j < inTensorNamesSize; j++) {
                isIn = (modelTensorsInput.find(inTensorNames[j]) != modelTensorsInput.end()) ? true : false;
                if (isIn) {
                    
                    auto tmpTensorPtr = modelTensorsInput[inTensorNames[j]];
                    auto tmpTensor    = opInTensorsOriginal[j];
#ifdef _USE_MALI
                    if(this->schedule == MALI){
                        TensorDesc tmpTensorDesc = opInTensorsOriginal[j].get_desc();
                        OclMemory* mem = (OclMemory*) tmpTensor.get_memory();
                        mem->set_tmpBuf(gclTempMem);
                        mem->set_val_from_hostptr(tmpTensorDesc, tmpTensorPtr.get(), CL_TRUE);
                    } else {
#endif
                        tmpTensor.set_shared_ptr(tmpTensorPtr);
#ifdef _USE_MALI
                    }
#endif

                    opInTensorUpdate.push_back(tmpTensor);
                    validTensorCount++;
                } else {
                    opInTensorUpdate.push_back(opInTensorsOriginal[j]);    // the tensor not change
                }
            }
            if(validTensorCount) curOp->set_input_tensors(opInTensorUpdate);
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
        this->set_input_tensors_desc(dims, this->modelInputTensorNames);

#ifdef _DEBUG
        const char* funcStr = "[DEBUG] infer_output_tensors_size()";
        std::cout << funcStr << std::endl;
#endif

        int opsNum = this->sortedOps.size();
        for (int i = 0; i < opsNum; i++) {
            std::string opName = sortedOps[i];

            auto op = this->operatorMap[opName];
#ifdef _DEBUG
            std::cout << funcStr << " op: " << opName << " type " << op->get_op_type() << std::endl;
#endif
            Vec<std::string> curOpInputTensorName = std::get<0>(this->operatorTensorMap[opName]);
            Vec<std::string> curOpOutputTensorName = std::get<1>(this->operatorTensorMap[opName]);
            int curOpInNum = curOpInputTensorName.size();
            int curOpOutNum = curOpOutputTensorName.size();
            Vec<TensorDesc> inTensorDescs;
            Vec<TensorDesc> outTensorDescs;
#ifdef _USE_MALI
            Vec<GCLMemDesc> inGCLMemDescs;
            Vec<GCLMemDesc> outGCLMemDescs;
#endif
            for (int j = 0; j < curOpOutNum; j++){
                TensorDesc dummyTensorDesc;
                outTensorDescs.push_back(dummyTensorDesc);
#ifdef _USE_MALI
                U32 stride[3] = {0, 0, 0};
                U32 offset[3] = {0, 0, 0};
                GCLMemDesc gclTmpDesc = gcl_mem_desc(stride, offset, DT_U8, DF_NCWHC4);
                outGCLMemDescs.push_back(gclTmpDesc);
#endif
            }

            Vec<Tensor> inTensors, outTensors;
            for (std::string inputTensorName: curOpInputTensorName) {
#ifdef _DEBUG
                std::cout << "    inputTensorName: " << inputTensorName << " ";
#endif
                inTensorDescs.push_back(this->tensorMap[inputTensorName]->desc);

#ifdef _USE_MALI
                if(this->schedule == MALI){
                    auto tmp =(OclMemory*)(this->tensorMap[inputTensorName]->get_memory());
                    auto desc = tmp->get_mem_desc();
                    inGCLMemDescs.push_back(desc);
                }
#endif
#ifdef _DEBUG
                std::cout << tensorDesc2Str(this->tensorMap[inputTensorName]->desc);
                std::cout << std::endl;
#endif
            }
#ifdef _USE_MALI
            if(this->schedule == MALI){
                CHECK_STATUS(op->infer_output_tensors_size(inTensorDescs, &outTensorDescs, &inGCLMemDescs, &outGCLMemDescs));
                int k = 0;
                for (std::string inputTensorName: curOpInputTensorName) {
                    Memory_* tmp = this->tensorMap[inputTensorName].get()->get_memory();
                    OclMemory* mem = (OclMemory*) tmp;
                    mem->set_mem_desc(inGCLMemDescs[k]);
                    k++;
                }
            } else {
#endif
                for (int k = 0; k < curOpInNum; k++) {
                    U32 size = tensorNumBytes(inTensorDescs[k]);
                    I32 slot = op->tensorPos[k];
                    if (slot == -1) {  //These tensors will be standalone
                        continue;
                    }
                    if (slot >= (I32)this->storageSizes.size()) {
                        this->storageSizes.resize(slot+1, 0);
                    }
                    if (size > this->storageSizes[slot]) {
                        this->storageSizes[slot] = size;
                    }
                }
                CHECK_STATUS(op->infer_output_tensors_size(inTensorDescs, &outTensorDescs));
#ifdef _USE_MALI
            }
#endif
            for (std::string inputTensorName: curOpInputTensorName) inTensors.push_back(*this->tensorMap[inputTensorName].get());

            for (int k = 0; k < curOpOutNum; k++) {
                std::string outputTensorName = curOpOutputTensorName[k];
#ifdef _DEBUG
                std::cout << "    outputTensorName: " << outputTensorName << " ";
#endif
                TensorDesc outputTensorDesc = outTensorDescs[k];
#ifdef _DEBUG
                std::cout << tensorDesc2Str(outputTensorDesc);
                std::cout << std::endl;
#endif
                this->tensorMap[outputTensorName]->desc = outputTensorDesc;
#ifdef _USE_MALI
                if(this->schedule == MALI){
                    Memory_* tmp = this->tensorMap[outputTensorName].get()->get_memory();
                    OclMemory* mem = (OclMemory*) tmp;
                    mem->set_mem_desc(outGCLMemDescs[k]);
                } else {
#endif 
                    U32 size = tensorNumBytes(outputTensorDesc);
                    I32 slot = op->tensorPos[curOpInNum + k];
                    if (slot != -1) {
                        if (slot >= (I32)this->storageSizes.size()) {
                            this->storageSizes.resize(slot+1, 0);
                        }
                        if (size > this->storageSizes[slot]) {
                            this->storageSizes[slot] = size;
                        }//merge conflict mark
                    }
#ifdef _USE_MALI                
                }
#endif          
                outTensors.push_back(*this->tensorMap[outputTensorName].get());
            }
            op->set_input_output_tensors(inTensors, outTensors);
        }
#ifdef _DEBUG
        U32 originalSize = 0;
        U32 standaloneSize = 0;
        for (auto tensor : this->tensorMap) {
            originalSize += tensorNumBytes(tensor.second->desc);
            if (weightOpOutputNames.find(tensor.first) != weightOpOutputNames.end()) {
                standaloneSize += tensorNumBytes(tensor.second->desc);
            }
        }
        std::cout << "Originally " << this->tensorMap.size() << " tensors, taking " << originalSize << " bytes.\n";

        std::cout << "Storage reduced to " << storageSizes.size() << " reuse slots: \n";
        U32 totalSize = 0;
        for (U32 size : storageSizes) {
            std::cout << size << " bytes, ";
            totalSize += size;
        }
        std::cout << "\nIn total " << totalSize << " bytes.\n";

        if (0 != standaloneSize) {
            std::cout << "Another " << standaloneSize << " bytes are reserved for standalone tensors (e.g. loop topology).\n";
        }

        std::cout << "Reuse ratio is " << (F32)originalSize / (totalSize+standaloneSize) << std::endl;
#endif
        return SUCCESS;
    }

    HashMap<std::string, std::shared_ptr<Tensor>> get_outputs()
    {
#ifdef _USE_MALI
        if(this->schedule == MALI){
            for(auto it  = outputTensorsOCLHost.begin(); it != outputTensorsOCLHost.end(); it++) {
                auto it_ocl = outputTensors.begin();
                auto outputTensorHost   = it->second;
                auto outputTensorDevice = it_ocl->second;
                std::shared_ptr<GCLMem> val = outputTensorDevice->get_shared_ptr();
                auto host_desc   = outputTensorHost->get_desc();
                auto mem = (OclMemory*)outputTensorDevice->get_memory();
                mem->set_tmpBuf(gclTempMem);
                U8* tmpVal = outputTensorHost->get_val();
                val->desc.use_map = false;//this API need to copy memory
                mem->get_val_to_hostptr(host_desc, &tmpVal, CL_TRUE);
                it_ocl++;
            }
            return this->outputTensorsOCLHost;
        } else {
#endif
            return this->outputTensors;
#ifdef _USE_MALI
        }
#endif
    }

#ifdef _USE_MALI
    HashMap<std::string, std::tuple<TensorDesc, U8*>> get_outputs_mali_map()
    {
        if(this->schedule == MALI){
            for(auto it  = outputTensorsOCLHostMap.begin(); it != outputTensorsOCLHostMap.end(); it++) {
                auto it_ocl = outputTensors.begin();
                auto outputTensorDevice = it_ocl->second;
                std::shared_ptr<GCLMem> val = outputTensorDevice->get_shared_ptr();
                auto host_desc = std::get<0>(it->second);
                auto device_desc = val->desc;
                auto host_df   = host_desc.df;
                auto device_df = device_desc.memFormat;
                auto mem = (OclMemory*)outputTensorDevice->get_memory();
                mem->set_tmpBuf(gclTempMem);
                if(host_df == device_df){//for map
                    U8* tmpVal;
                    mem->get_val_to_hostptr(host_desc, &tmpVal, CL_TRUE);
                    std::get<1>(it->second) = tmpVal;
                } else {
                    CHECK_STATUS(NOT_SUPPORTED);
                }
                it_ocl++;
            }
        } else {
            CHECK_STATUS(NOT_SUPPORTED);
        }
        return this->outputTensorsOCLHostMap;
    }
#endif
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
#ifdef _USE_MALI
                if(this->schedule == MALI){
                    it->second->alloc();//alloc ocl gpu memory for inputTensor
                }
#endif 
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
#ifdef _USE_MALI
                if(this->schedule == MALI){
                    std::shared_ptr<GCLMem> val = it->second->get_shared_ptr();
                    auto host_desc   = it->second->get_desc();
                    auto device_desc = val->desc;
                    DataFormat host_df = host_desc.df;
                    DataFormat device_df = device_desc.memFormat;
                    std::shared_ptr<Tensor> cpuTensor(new Tensor());
                    cpuTensor->set_desc(host_desc);
                    if(host_df == device_df){//use map, realloc gpu memory with flag CL_MEM_ALLOC_HOST_PTR
                        CHECK_STATUS(gcl_release_memory(val.get()));
                        val->desc.use_map = true;
                        val->desc.flags   = CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR;
                        it->second->set_shared_ptr(val);
                        it->second->alloc();
                        U8* tmp = NULL;
                        std::tuple<TensorDesc, U8*> outputInfoMap(std::make_tuple(host_desc, tmp));
                        auto p = std::pair<std::string, std::tuple<TensorDesc, U8*>>(it->first, outputInfoMap);
                        outputTensorsOCLHostMap.insert(p);
                    }    
                    cpuTensor->alloc();
                    auto p = std::pair<std::string, std::shared_ptr<Tensor>>(it->first, cpuTensor);
                    outputTensorsOCLHost.insert(p);
                }
#endif                
                outputTensors.insert(*it);
            } else {
                return NOT_MATCH;
            }
        }

#ifdef _USE_MALI
        if(this->schedule == MALI){
           U32 tmpBufSize = 0;
           for(auto it  = inputTensors.begin(); it != inputTensors.end(); it++) {
               Tensor* inputTensor = it->second.get();
               TensorDesc desc = inputTensor->desc;
               GCLMem_t   mem  = inputTensor->get_val();
               U32        size = 0;
               tensor_computing_set_input_infer_tmpBuf_size(mem, desc, &size, MALI); 
               tmpBufSize = (tmpBufSize < size) ? size : tmpBufSize;
           }

           for(auto it  = outputTensors.begin(); it != outputTensors.end(); it++) {
               Tensor* outputTensor = it->second.get();
               TensorDesc desc = outputTensor->desc;
               GCLMem_t   mem  = outputTensor->get_val();
               U32        size = 0;
               tensor_computing_get_output_infer_tmpBuf_size(mem, desc, &size, MALI); 
               tmpBufSize = (tmpBufSize < size) ? size : tmpBufSize;
           }

           if(tmpBufSize > maxTmpElements){
               maxTmpElements = tmpBufSize;
               gcl_release_memory(gclTempMem);
               gclTempMem->desc.memType   = GCL_MEM_BUF;
               gclTempMem->desc.byteSize  = this->maxTmpElements;
               gclTempMem->desc.flags     = CL_MEM_READ_WRITE;
               gclTempMem->desc.host_ptr  = nullptr;
               CHECK_STATUS(gcl_create_memory(this->handle.get(), this->gclTempMem));
               for(auto op: this->ops) {
                   op->set_tmp_gclmem(this->maxTmpElements, gclTempMem);
               }
           }
           DataType dt = inputTensors.begin()->second.get()->desc.dt;
           gclTempMem->desc.stride[0] = this->maxTmpElements / bytesOf(dt);
           gclTempMem->desc.stride[1] = 1;
           gclTempMem->desc.stride[2] = 1;
           gclTempMem->desc.memFormat = DF_NCHW;
        }
#endif
        return SUCCESS;
    }

    void assign_output_tensor() override
    {
#ifdef _DEBUG
        const char* funcStr = "[DEBUG] assign_output_tensor()";
        std::cout << funcStr << std::endl;
#endif

        Vec<std::shared_ptr<U8>> storages(storageSizes.size());
#ifdef _USE_MALI
        if(this->schedule != MALI){
#endif        
            for (U32 i = 0; i < storages.size(); i++) {
                storages[i] = std::shared_ptr<U8>((U8*)operator new(storageSizes[i]));
            }
#ifdef _USE_MALI
        }
#endif        

        for (std::string opName: sortedOps) {
#ifdef _DEBUG
            std::cout << funcStr << " op: " << opName << "\n    input  tensor names: ";
#endif
            U32 tensorIter = 0;
            std::shared_ptr<Operator> op = this->operatorMap[opName];
            Vec<Tensor> inTensors;
            Vec<std::string> inTensorNames = std::get<0>(this->operatorTensorMap[opName]);
            for (std::string inName: inTensorNames){
#ifdef _DEBUG
                std::cout << inName << " to Slot " << op->tensorPos[tensorIter] << ", ";
#endif
#ifdef _USE_MALI
                if(this->schedule != MALI){
#endif                
                    if (op->tensorPos[tensorIter] != -1){
                        this->tensorMap[inName].get()->set_shared_ptr(storages[op->tensorPos[tensorIter]]);
                    }
                    tensorIter++;
#ifdef _USE_MALI                    
                }
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
                std::cout << outName << " to Slot " << op->tensorPos[tensorIter] << ", ";
#endif
#ifdef _USE_MALI
                if(this->schedule == MALI){
                    this->tensorMap[outName].get()->alloc();
                } else {
#endif                
                    if (this->weightOpOutputNames.find(outName) == this->weightOpOutputNames.end()) {
                        if (op->tensorPos[tensorIter] != -1) {
                            this->tensorMap[outName].get()->set_shared_ptr(storages[op->tensorPos[tensorIter]]);
                        } else {
                            this->tensorMap[outName].get()->alloc();
                        }
                    }
                    tensorIter++;
#ifdef _USE_MALI                
                }
#endif                
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
#ifdef _USE_MALI
        if(this->schedule == MALI){
            gclTempMem = gcl_create_gclmem();
            gclTempMem->desc.memType  = GCL_MEM_BUF;
            gclTempMem->desc.byteSize = this->maxTmpElements;
            gclTempMem->desc.flags    = CL_MEM_READ_WRITE;
            gclTempMem->desc.host_ptr = nullptr;
            if(maxTmpElements) CHECK_STATUS(gcl_create_memory(this->handle.get(), this->gclTempMem));
            for(auto op: this->ops) {
                op->set_tmp_gclmem(this->maxTmpElements, gclTempMem);
            }
        } else {
#endif
            auto sPtr = std::shared_ptr<U8>((U8*)operator new(this->maxTmpElements));
            for (auto op: this->ops) {
                op->set_tmp_memory(this->maxTmpElements, sPtr);
            }
#ifdef _USE_MALI
        }
#endif
    }

    void ready(Vec<TensorDesc> dims) override
    {
#ifdef _DEBUG
        const char* funcStr = "[DEBUG] ready()";
        std::cout << "Arch: " << this->schedule << std::endl;
        std::cout << funcStr << " Model input num: " << this->modelInputTensorNames.size() << std::endl;
        for (auto item: this->modelInputTensorNames) {
            std::cout << "    input: " << item << std::endl;
        }
#endif

        this->infer_output_tensors_size(dims);

        // handle the weight ops
        for (auto op : this->ops) {
#ifdef _DEBUG
            std::cout << funcStr << " op: " << op->get_name() << std::endl;
#endif
            if (op->is_weight()) {
                if (op->get_op_type() == OT_Conv) {
                    auto convOpPtr = dynamic_cast<Convolution*>(op.get());
                    CHECK_STATUS(convOpPtr->init_weight_bias_from_model(nullptr));
                    CHECK_STATUS(convOpPtr->infer_forward_algorithm(this->algorithmMap));
                    CHECK_STATUS(convOpPtr->transform_filter());
                } else if (op->get_op_type() == OT_Deconvolution) {
                    auto convOpPtr = dynamic_cast<Deconvolution*>(op.get());
                    CHECK_STATUS(convOpPtr->init_weight_bias_from_model(nullptr));
                    CHECK_STATUS(convOpPtr->infer_forward_algorithm());
                    CHECK_STATUS(convOpPtr->transform_filter());
                } else if (op->get_op_type() == OT_FC) {
                    auto fcOpPtr = dynamic_cast<FullyConnectedEltwise*>(op.get());
                    CHECK_STATUS(fcOpPtr->init_weight_bias_from_model(nullptr));
                    CHECK_STATUS(fcOpPtr->transform_filter());
                } else if (op->get_op_type() == OT_Embedding) {
                    auto embeddingOpPtr = dynamic_cast<Embedding*>(op.get());
                    CHECK_STATUS(embeddingOpPtr->init_weight_bias_from_model(nullptr));
                } else if (op->get_op_type() == OT_LayerNorm) {
                    auto layernormOpPtr = dynamic_cast<LayerNorm*>(op.get());
                    CHECK_STATUS(layernormOpPtr->init_weight_bias_from_model(nullptr));
                } else if (op->get_op_type() == OT_Scale) {
                    auto scaleOpPtr = dynamic_cast<Scale*>(op.get());
                    CHECK_STATUS(scaleOpPtr->init_weight_bias_from_model(nullptr));
                } else if (op->get_op_type() == OT_LSTM) {
                    auto lstmOpPtr = dynamic_cast<LSTMCell*>(op.get());
                    CHECK_STATUS(lstmOpPtr->init_weight_bias_from_model(nullptr));
                    CHECK_STATUS(lstmOpPtr->transform_filter());
                } else if (op->get_op_type() == OT_SharedWeight) {
                    auto weightOpPtr = dynamic_cast<SharedWeight*>(op.get());
                    CHECK_STATUS(weightOpPtr->init_weight_bias_from_model(nullptr));
                    std::string weightOpOutputName = (std::get<1>(this->operatorTensorMap[op->get_name()]))[0];
                    Tensor weightTensor = weightOpPtr->weightTensors[0];
                    this->tensorMap[weightOpOutputName]->set_shared_ptr(weightTensor.get_shared_ptr());
                    this->weightOpOutputNames.insert(weightOpOutputName);
                }
            }
        }

        this->assign_output_tensor();

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

    void set_modelInputTensorNames(Vec<std::string> modelInputTensorNames) {
        this->modelInputTensorNames = modelInputTensorNames;
    }

    Vec<std::string> get_model_input_tensor_names() {
        return  this->modelInputTensorNames;
    }
#ifdef _USE_MALI
    void mali_prepare(){
        Model::run_mali_prepare();
        CHECK_STATUS(gcl_finish(this->handle.get())); 
    }
#endif

private:
    HashMap<std::string, std::shared_ptr<Tensor>> tensorMap;
    HashMap<std::string, std::shared_ptr<Operator>> operatorMap;
    HashMap<std::string, std::tuple<Vec<std::string>, Vec<std::string>>> operatorTensorMap;

    Set<std::string> weightOpOutputNames;

    //input output reuse or not
    Vec<bool> reused;

    //input & output tensors
    HashMap<std::string, std::shared_ptr<Tensor>> inputTensors;
    HashMap<std::string, std::shared_ptr<Tensor>> outputTensors;
#ifdef _USE_MALI
    HashMap<std::string, std::shared_ptr<Tensor>> outputTensorsOCLHost;
    HashMap<std::string, std::tuple<TensorDesc, U8*>> outputTensorsOCLHostMap;
#endif

    Vec<U32> storageSizes;

    Vec<std::string> sortedOps;

    U32 maxTmpElements;
    Vec<U32> tmpElements;

    Vec<TensorDesc> modelInDims;

    Vec<std::string> modelInputTensorNames;
#ifdef _USE_MALI
    GCLMem_t    gclTempMem;
#endif
};
#endif
