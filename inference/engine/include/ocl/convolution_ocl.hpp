// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _CONVELTWISEPOOLING_OCL_H
#define _CONVELTWISEPOOLING_OCL_H

#include "convolution.hpp"

#include "ocl_desc_trans.h"

class ConvolutionOCL : public Convolution {
public:
    ConvolutionOCL(DataType dt,
        ConvolutionParamSpec p,
        ActivationParamSpec dwActivationParamSpec,
        ActivationParamSpec pwActivationParamSpec)
        : Convolution(dt, p, dwActivationParamSpec, pwActivationParamSpec)
    {
        INIT_GPU_INFO(&this->runInfo)
    }

    ~ConvolutionOCL(){DESTROY_OCL_KERNEL}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<ConvolutionOCL> mem = std::shared_ptr<ConvolutionOCL>(new ConvolutionOCL(
            this->dt, this->p, this->dwActivationParamSpec, this->pwActivationParamSpec));
        *mem = *this;
        return mem;
    }

    EE infer_weight_desc() override
    {
        TensorDesc wDesc[2];
        TensorDesc vDesc[2];
        wDesc[0] = this->filterDesc;
        U32 filterNum = 1;
        DataType dtNoQ = noQuantDataType(this->dt);
        switch (this->p.convolution_type) {
            case CONVOLUTION_POINTWISE: {
                if (this->p.num_outputs_origin == 1) {
                    if (tensorIs5d(wDesc[0])) {
                        wDesc[0].dims[4] = this->p.num_outputs;
                    } else {
                        wDesc[0].dims[3] = this->p.num_outputs;
                    }
                }
                // bias data type should be the same as input and output
                vDesc[0] = tensor1d(dtNoQ, this->p.num_outputs);
                ((MaliPara_t)(this->archInfo.archPara))->forwardRunInfo->algorithm =
                    CONVOLUTION_ALGORITHM_NULL;
                break;
            }
            case CONVOLUTION_DEPTHWISE: {
                vDesc[0] = tensor1d(dtNoQ, this->p.num_outputs);
                ((MaliPara_t)(this->archInfo.archPara))->forwardRunInfo->algorithm =
                    DEPTHWISE_CONVOLUTION_ALGORITHM_NULL;
                break;
            }
            case CONVOLUTION_DEPTHWISE_POINTWISE: {
                wDesc[1] = this->filterDescExt;
                vDesc[0] = tensor1d(dtNoQ, this->numChannels);
                vDesc[1] = tensor1d(dtNoQ, this->p.num_outputs);
                filterNum = 2;
                ((MaliPara_t)(this->archInfo.archPara))->forwardRunInfo->algorithm =
                    DEPTHWISE_CONVOLUTION_ALGORITHM_NULL;
                break;
            }
            default:
                UNI_ERROR_LOG("not support to read new type convolution's weight.\n");
                return NOT_SUPPORTED;
        }

        for (U32 i = 0; i < filterNum; i++) {
            Tensor modelWeightTensor = Tensor(OCLMem);
            Tensor modelVectorTensor = Tensor(OCLMemImg1D);
            modelWeightTensor.resize(wDesc[i]);
            modelVectorTensor.resize(vDesc[i]);
            this->weightTensors.push_back(modelWeightTensor);
            this->biasTensors.push_back(modelVectorTensor);
        }
        return SUCCESS;
    }

    inline void run_prepare()
    {
        OCLContext::getInstance().handle.get()->curOpName = this->get_name();
        Tensor inputTensor = this->inputTensors[0];
        Tensor filterTensor = this->weightTensors[0];
        U8 *scalePtr = nullptr;
        Tensor biasTensor = this->biasTensors[0];
        Tensor outputTensor = this->outputTensors[0];
        switch (this->p.convolution_type) {
            case CONVOLUTION_POINTWISE: {
                Tensor tmpTensor = Tensor(OCLMem);
                std::vector<Tensor> tmpTensors(3, tmpTensor);
                tmpTensors[0] = this->temp;
                if (this->pwAlg == CONVOLUTION_ALGORITHM_WINOGRAD) {
                    get_tmp_image(0, bytes + 1, &tmpTensors[1]);
                    get_tmp_image(1, bytes + 4, &tmpTensors[2]);
                } else {
                    get_tmp_image(0, bytes + 1, &tmpTensors[0]);
                }
                CHECK_STATUS(convolution(this->inputTensors, filterTensor, p, this->pwAlg, scalePtr,
                    biasTensor, tmpTensors, outputTensor, this->pwActivationParamSpec,
                    &this->archInfo));
                break;
            }
            case CONVOLUTION_DEPTHWISE: {
                Tensor tmpTensor = this->temp;
                get_tmp_image(0, bytes + 1, &tmpTensor);
                CHECK_STATUS(depthwise_convolution(inputTensor, filterTensor, p, this->dwAlg,
                    nullptr, biasTensor, tmpTensor, outputTensor, this->dwActivationParamSpec,
                    &this->archInfo));
                break;
            }
            case CONVOLUTION_DEPTHWISE_POINTWISE: {
                auto dwFilterTensor = filterTensor;
                auto pwFilterTensor = this->weightTensors[1];
                auto dwBiasTensor = biasTensor;
                auto pwBiasTensor = this->biasTensors[1];
                Tensor tmpTensor = Tensor(OCLMem);
                std::vector<Tensor> tmpTensors(3, tmpTensor);
                tmpTensors[0] = this->temp;
                get_tmp_image(0, bytes + 1, &tmpTensors[1]);
                get_tmp_image(1, bytes + 4, &tmpTensors[2]);
                CHECK_STATUS(depthwise_pointwise_convolution(this->inputTensors, dwFilterTensor,
                    pwFilterTensor, p, this->dwAlg, nullptr, dwBiasTensor, pwBiasTensor, tmpTensors,
                    outputTensor, this->dwActivationParamSpec, this->pwActivationParamSpec,
                    &this->archInfo));
                break;
            }
            default: {
                UNI_ERROR_LOG("not support to run new type convolution.\n");
            }
        }
    }

    EE infer_forward_algorithm(std::shared_ptr<AlgorithmMap> algorithmMap) override
    {
        OCLContext::getInstance().handle.get()->kernelVec = &this->opKernelVec;
        this->weightTensors[0].resize(this->filterDesc);
        auto inputTensor = this->inputTensors[0];
        auto filterTensor = this->weightTensors[0];
        auto outputTensor = this->outputTensors[0];
        ConvolutionPolicy policy = CONVOLUTION_TUNNING;
        DataType targetType = noQuantDataType(this->dt);
        I32 algo[7];
        std::string name =
            this->name + std::to_string(get_type()) + std::to_string(this->p.convolution_type);
        switch (this->p.convolution_type) {
            case CONVOLUTION_POINTWISE: {
                if (isQuantMixDataType(this->dt)) {
                    targetType = DT_I8;
                }
                if (algorithmMap->getAlgorithmInfoFromMap(name, algo, 4)) {
                    this->runInfo.algorithm = (ConvolutionForwardAlgorithm)algo[0];
                    this->runInfo.best_h[0] = algo[1];
                    this->runInfo.best_c[0] = algo[2];
                    this->runInfo.best_k[0] = algo[3];
                    this->pwAlg = (ConvolutionForwardAlgorithm)algo[0];
                } else {
                    CHECK_STATUS(convolution_infer_forward_algorithm(inputTensor, filterTensor,
                        outputTensor, p, policy, &(this->pwAlg), targetType,
                        this->pwActivationParamSpec, &this->archInfo));
                    algo[0] = this->runInfo.algorithm;
                    algo[1] = this->runInfo.best_h[0];
                    algo[2] = this->runInfo.best_c[0];
                    algo[3] = this->runInfo.best_k[0];
                    this->pwAlg = (ConvolutionForwardAlgorithm)algo[0];
                    algorithmMap->setAlgorithmInfoToMap(name, algo, 4);
                }
                break;
            }
            case CONVOLUTION_DEPTHWISE: {
                if (algorithmMap->getAlgorithmInfoFromMap(name, algo, 4)) {
                    this->runInfo.algorithm = (ConvolutionForwardAlgorithm)algo[0];
                    this->runInfo.best_h[0] = algo[1];
                    this->runInfo.best_c[0] = algo[2];
                    this->runInfo.best_k[0] = algo[3];
                    this->dwAlg = (DepthwiseConvolutionForwardAlgorithm)algo[0];
                } else {
                    CHECK_STATUS(depthwise_convolution_infer_forward_algorithm(inputTensor,
                        filterTensor, outputTensor, p, policy, &(this->dwAlg), targetType,
                        this->dwActivationParamSpec, &this->archInfo));
                    algo[0] = this->runInfo.algorithm;
                    algo[1] = this->runInfo.best_h[0];
                    algo[2] = this->runInfo.best_c[0];
                    algo[3] = this->runInfo.best_k[0];
                    this->dwAlg = (DepthwiseConvolutionForwardAlgorithm)algo[0];
                    algorithmMap->setAlgorithmInfoToMap(name, algo, 4);
                }
                break;
            }
            case CONVOLUTION_DEPTHWISE_POINTWISE: {
                if (algorithmMap->getAlgorithmInfoFromMap(name, algo, 7)) {
                    this->runInfo.algorithm = (ConvolutionForwardAlgorithm)algo[0];
                    this->runInfo.best_h[0] = algo[1];
                    this->runInfo.best_c[0] = algo[2];
                    this->runInfo.best_k[0] = algo[3];
                    this->runInfo.best_h[1] = algo[4];
                    this->runInfo.best_c[1] = algo[5];
                    this->runInfo.best_k[1] = algo[6];
                    this->dwAlg = (DepthwiseConvolutionForwardAlgorithm)algo[0];
                } else {
                    auto dwFilterTensor = filterTensor;
                    auto pwFilterTensor = this->weightTensors[1];
                    CHECK_STATUS(depthwise_pointwise_convolution_infer_forward_algorithm(
                        inputTensor, dwFilterTensor, pwFilterTensor, outputTensor, p, policy,
                        &(this->dwAlg), targetType, this->dwActivationParamSpec,
                        this->pwActivationParamSpec, &this->archInfo));
                    algo[0] = this->runInfo.algorithm;
                    algo[1] = this->runInfo.best_h[0];
                    algo[2] = this->runInfo.best_c[0];
                    algo[3] = this->runInfo.best_k[0];
                    algo[4] = this->runInfo.best_h[1];
                    algo[5] = this->runInfo.best_c[1];
                    algo[6] = this->runInfo.best_k[1];
                    this->dwAlg = (DepthwiseConvolutionForwardAlgorithm)algo[0];
                    algorithmMap->setAlgorithmInfoToMap(name, algo, 7);
                }
                break;
            }
            default:
                UNI_ERROR_LOG("not support to infer new type convolution's algorithm.\n");
                return NOT_SUPPORTED;
        }
        return SUCCESS;
    }

    inline bool use_output_tensor_image(U32 numFiltersOcl, Tensor *inputTensor)
    {
        if (!IS_QUALCOMM_GPU(this->archInfo.arch)) {
            return false;
        }
        if (numFiltersOcl == 1) {
            return false;
        }
        TensorDesc inDim = inputTensor->get_desc();
        U32 iw, ih, it, ic;
        U32 kw, kh, kt;
        CHECK_STATUS(tensorSelectGet(inDim, NULL, NULL, NULL, &ic, &ih, &iw, &it));
        kw = this->p.kernel_w;
        kh = this->p.kernel_h;
        kt = this->p.kernel_t;
        if (iw == 1 && ih == 1 && it == 1 && kw == 1 && kh == 1 && kt == 1) {
            return false;
        }
        if (ic == 1 && inDim.df == DF_NCHW) {
            return false;
        }
        return true;
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        this->needSetKernelVec = true;
        auto inputTensor = inTensors[0];
        Tensor filterTensor = Tensor(OCLMem);
        TensorDesc inDim = inputTensor->get_desc();
        DataType idt;
        DataFormat idf;
        U32 in, ic, ih, iw, it;
        U32 nDims;
        CHECK_STATUS(tensorSelectGet(inDim, &idt, &idf, &in, &ic, &ih, &iw, &it));
        this->numChannels = ic;

        U32 numFiltersOcl = this->p.num_outputs;
        if (this->p.num_outputs_origin == 1 && this->tensorPos[0] != -2) {
            numFiltersOcl = 1;  // spe case for output channel = 1
        }
        DataType targetType = noQuantDataType(this->dt);

        if (this->p.convolution_type == CONVOLUTION_DILATION) {
            this->p.convolution_type = CONVOLUTION_POINTWISE;
        }
        switch (this->p.convolution_type) {
            case CONVOLUTION_POINTWISE: {
                if (tensorIs5d(inDim)) {
                    this->filterDesc = tensor5df(this->dt, DF_NCHW, numFiltersOcl,
                        this->numChannels, this->p.kernel_t, this->p.kernel_h, this->p.kernel_w);
                } else {
                    this->filterDesc = tensor4df(this->dt, DF_NCHW, numFiltersOcl,
                        this->numChannels, this->p.kernel_h, this->p.kernel_w);
                }
                filterTensor.resize(this->filterDesc);
                CHECK_STATUS(convolution_infer_output_size(
                    inputTensor, filterTensor, p, outTensors[0], targetType, &this->archInfo));
                break;
            }
            case CONVOLUTION_DEPTHWISE: {
                this->filterDesc = tensor4df(
                    this->dt, DF_NCHW, 1, this->numChannels, this->p.kernel_h, this->p.kernel_w);
                filterTensor.resize(this->filterDesc);
                CHECK_STATUS(depthwise_convolution_infer_output_size(
                    inputTensor, filterTensor, p, outTensors[0], targetType, &this->archInfo));
                break;
            }
            case CONVOLUTION_DEPTHWISE_POINTWISE: {
                this->filterDesc = tensor4df(
                    this->dt, DF_NCHW, 1, this->numChannels, this->p.kernel_h, this->p.kernel_w);
                this->filterDescExt =
                    tensor4df(this->dt, DF_NCHW, this->p.num_outputs, this->numChannels, 1, 1);
                filterTensor.resize(this->filterDesc);
                Tensor filterTensorExt = Tensor(OCLMem);
                filterTensorExt.resize(this->filterDescExt);
                CHECK_STATUS(depthwise_pointwise_convolution_infer_output_size(inputTensor,
                    filterTensor, filterTensorExt, p, outTensors[0], targetType, &this->archInfo));
                break;
            }
            default:
                UNI_ERROR_LOG("not support to infer new type convolution's output.\n");
                return NOT_SUPPORTED;
        }
        if (use_output_tensor_image(numFiltersOcl, inputTensor)) {
            CHECK_STATUS(set_tensors_image(outTensors, inTensors.size()));
        }
        return SUCCESS;
    }

    U32 infer_tmp_memory_size() override
    {
        auto inputTensor = this->inputTensors[0];
        auto filterTensor = this->weightTensors[0];
        auto outputTensor = this->outputTensors[0];
        for (U32 i = 0; i < 7; i++) {
            bytes[i] = 0;
        }
        switch (this->p.convolution_type) {
            case CONVOLUTION_POINTWISE: {
                CHECK_STATUS(convolution_infer_forward_tmp_bytes(inputTensor, filterTensor,
                    outputTensor, p, this->pwAlg, bytes, &this->archInfo));
                break;
            }
            case CONVOLUTION_DEPTHWISE: {
                CHECK_STATUS(depthwise_convolution_infer_forward_tmp_bytes(inputTensor,
                    filterTensor, outputTensor, p, this->dwAlg, bytes, &this->archInfo));
                break;
            }
            case CONVOLUTION_DEPTHWISE_POINTWISE: {
                CHECK_STATUS(depthwise_pointwise_convolution_infer_forward_tmp_bytes(inputTensor,
                    filterTensor, this->weightTensors[1], outputTensor, p, this->dwAlg, bytes,
                    &this->archInfo));
                break;
            }
            default:
                UNI_ERROR_LOG("not support to infer new type convolution's tmp memory.\n");
        }
        add_tmp_image(0, bytes + 1);
        add_tmp_image(1, bytes + 4);
        return bytes[0];
    }

    EE alloc_wtm_memory()
    {
        auto filterTensor = this->weightTensors[0];
        bool needTransBiasImgToBuf = false;
        U32 biasNum = 0;
        TensorDesc desc[2];
        switch (this->p.convolution_type) {
            case CONVOLUTION_POINTWISE: {
                CHECK_STATUS(convolution_transform_filter_bytes(
                    filterTensor, this->p, this->pwAlg, desc, &this->archInfo));
                if (this->runInfo.best_k[0] <= 1 && this->pwAlg == CONVOLUTION_ALGORITHM_DIRECT) {
                    needTransBiasImgToBuf = true;
                    biasNum = 0;
                }
                break;
            }
            case CONVOLUTION_DEPTHWISE: {
                CHECK_STATUS(depthwise_convolution_transform_filter_bytes(
                    filterTensor, this->p, this->dwAlg, desc, &this->archInfo));
                break;
            }
            case CONVOLUTION_DEPTHWISE_POINTWISE: {
                CHECK_STATUS(depthwise_pointwise_convolution_transform_filter_bytes(filterTensor,
                    this->weightTensors[1], this->p, this->dwAlg, &desc[0], &desc[1],
                    &this->archInfo));
                wtm_dp = Tensor(OCLMem);
                wtm_dp.resize(desc[1]);
                wtm_dp.alloc();
                if (this->dwAlg == DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_GEMM) {
                    needTransBiasImgToBuf = true;
                    biasNum = 1;
                }
                break;
            }
            default:
                UNI_ERROR_LOG("not support to infer new type convolution's tramsform filter tmp "
                              "memory.\n");
                return NOT_SUPPORTED;
        }
        this->wtm = std::shared_ptr<Tensor>(new Tensor(OCLMem));
        this->wtm->resize(desc[0]);
        if (CONVOLUTION_ALGORITHM_WINOGRAD == this->pwAlg) {
            CHECK_STATUS(set_wtm_image(desc[0]));
        }
        this->wtm->alloc();
        if (needTransBiasImgToBuf) {
            Tensor biasTensorBuf = Tensor(OCLMem);
            TensorDesc desc = this->biasTensors[biasNum].get_desc();
            Memory *biasMemImg = this->biasTensors[biasNum].get_memory();
            auto biasMemBuf = (OclMemory *)(biasTensorBuf.get_memory());
            biasTensorBuf.resize(desc);
            biasMemBuf->padding(0, 8, 0, 0);
            biasMemBuf->alloc();
            void *bufPtr = biasMemBuf->get_ptr();
            CHECK_STATUS(
                gcl_fill_memory_zero(OCLContext::getInstance().handle.get(), (GCLMem_t)bufPtr));
            CHECK_STATUS(biasMemBuf->copy_from(biasMemImg));
            this->biasTensors[biasNum] = biasTensorBuf;
        }
        return SUCCESS;
    }

    EE transform_filter() override
    {
        auto filterTensor = this->weightTensors[0];
        // int8 winograd
        if (isQuantMixDataType(this->dt) && CONVOLUTION_POINTWISE == this->p.convolution_type &&
            CONVOLUTION_ALGORITHM_WINOGRAD == this->pwAlg) {
            return NOT_SUPPORTED;
            // int8 tilegemm
        } else if (isQuantMixDataType(this->dt) &&
            CONVOLUTION_POINTWISE == this->p.convolution_type) {
            return NOT_SUPPORTED;
        } else {  // All other cases
            CHECK_STATUS(alloc_wtm_memory());
            switch (this->p.convolution_type) {
                case CONVOLUTION_POINTWISE: {
                    CHECK_STATUS(convolution_transform_filter(filterTensor, this->p, this->pwAlg,
                        this->temp, this->wtm.get(), &this->archInfo));
                    break;
                }
                case CONVOLUTION_DEPTHWISE: {
                    CHECK_STATUS(depthwise_convolution_transform_filter(
                        filterTensor, this->p, this->dwAlg, this->wtm.get(), &this->archInfo));
                    break;
                }
                case CONVOLUTION_DEPTHWISE_POINTWISE: {
                    CHECK_STATUS(depthwise_pointwise_convolution_transform_filter(filterTensor,
                        this->weightTensors[1], this->p, this->dwAlg, this->wtm.get(),
                        &this->wtm_dp, &this->archInfo));
                    this->weightTensors[1] = wtm_dp;
                    break;
                }
                default: {
                    UNI_ERROR_LOG("not support to transform new type convolution's filter.\n");
                    return NOT_SUPPORTED;
                }
            }
        }
        this->weightTensors[0] = *(this->wtm.get());
        return SUCCESS;
    }

    REGISTER_OCL_OPERATOR_RUN

private:
    Tensor wtm_dp;
    TensorDesc filterDesc;
    TensorDesc filterDescExt;
    U32 numChannels;
    U32 bytes[7];

protected:
    ForwardRunInfoMali runInfo;
};

#endif  // _CONVELTWISEPOOLING_H
