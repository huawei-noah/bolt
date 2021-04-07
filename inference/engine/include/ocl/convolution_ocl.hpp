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
        setMALIArchInfo(&(this->archInfo), &(this->runInfo), &this->needSetKernelVec,
            &this->needSelectKernelLS);
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
        DataType dtNoQ = (this->dt == DT_F16_8Q) ? DT_F16 : this->dt;
        switch (this->p.convolution_type) {
            case Convolution_Pointwise: {
                if (this->p.num_outputs_origin == 1) {
                    if (tensorIs5d(wDesc[0])) {
                        wDesc[0].dims[4] = this->p.num_outputs;
                    } else {
                        wDesc[0].dims[3] = this->p.num_outputs;
                    }
                }
                vDesc[0] = tensor1d(dtNoQ,
                    this->p.num_outputs);  // bias data type should be the same as input and output
                ((MaliPara_t)(this->archInfo.archPara))->forwardRunInfo->algorithm =
                    CONVOLUTION_ALGORITHM_NULL;
                break;
            }
            case Convolution_Depthwise: {
                vDesc[0] = tensor1d(dtNoQ, this->p.num_outputs);
                ((MaliPara_t)(this->archInfo.archPara))->forwardRunInfo->algorithm =
                    DEPTHWISE_CONVOLUTION_ALGORITHM_NULL;
                break;
            }
            case Convolution_Depthwise_Pointwise: {
                wDesc[1] = this->filterDescExt;
                vDesc[0] = tensor1d(dtNoQ, this->numChannels);
                vDesc[1] = tensor1d(dtNoQ, this->p.num_outputs);
                filterNum = 2;
                ((MaliPara_t)(this->archInfo.archPara))->forwardRunInfo->algorithm =
                    DEPTHWISE_CONVOLUTION_ALGORITHM_NULL;
                break;
            }
            case Convolution_Dilation: {
                CHECK_STATUS(NOT_SUPPORTED);
                return NOT_SUPPORTED;
                break;
            }
            default:
                CHECK_STATUS(NOT_SUPPORTED);
                return NOT_SUPPORTED;
        }

        for (U32 i = 0; i < filterNum; i++) {
            Tensor modelWeightTensor = Tensor(OCLMem);
            Tensor modelVectorTensor = Tensor(OCLMem);
            auto weightMem = (OclMemory *)modelWeightTensor.get_memory();
            auto vectorMem = (OclMemory *)modelVectorTensor.get_memory();
            modelWeightTensor.resize(wDesc[i]);
            modelVectorTensor.resize(vDesc[i]);

            U32 ww, wh, wc, wn, wt;
            DataFormat df;
            DataType dt;
            tensorSelectGet(wDesc[i], &dt, &df, &wn, &wc, &wh, &ww, &wt);
            U32 stride[3] = {ww * wh * wt, wc, wn};
            U32 offset[3] = {0, 0, 0};
            GCLMemType mt = GCL_MEM_BUF;
            MemFlags flags = CL_MEM_READ_WRITE;
            GCLMemDesc desc = gclmem_build_desc();
            CHECK_STATUS(gclmem_set_desc_padding(&desc, stride, offset, dt, df, mt, flags));
            weightMem->padding(desc);

            mt = GCL_MEM_IMG_1D;
            U32 vecLen = vDesc[i].dims[0];
            U32 vecAlign = 4;
            stride[0] = (vecLen + vecAlign - 1) / vecAlign;
            if (i == 0) {
                U32 iw, ih;
                TensorDesc inputDesc = this->inputTensors[0].get_desc();
                tensorSelectGet(inputDesc, NULL, NULL, NULL, NULL, &ih, &iw);
                if (this->p.convolution_type == Convolution_Pointwise) {
                    U32 sw = this->p.stride_w;
                    U32 wn_org = this->p.num_outputs_origin;
                    if ((wn_org * sw * wt == 1 && (ww == wh) &&
                            (ww == 1 || ww == 3 || ww == 5 || ww == 7)) ||
                        (ww * wh * wt * iw * ih == 1)) {
                        mt = GCL_MEM_BUF;
                        vecAlign = 8;
                        stride[0] = (vecLen + vecAlign - 1) / vecAlign * vecAlign;
                    }
                }
            }
            stride[1] = 1;
            stride[2] = 1;
            gclmem_set_desc_padding(&desc, stride, offset, dt, DF_NHWC, mt, flags);
            vectorMem->padding(desc);
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
        filterTensor.resize(this->filterDesc);
        U8 *scalePtr = nullptr;
        Tensor biasTensor = this->biasTensors[0];
        Tensor outputTensor = this->outputTensors[0];
        switch (this->p.convolution_type) {
            case Convolution_Pointwise: {
                CHECK_STATUS(convolution(this->inputTensors, filterTensor, p, this->pwAlg, scalePtr,
                    biasTensor, this->temp, outputTensor, this->pwActivationParamSpec,
                    &this->archInfo));
                break;
            }
            case Convolution_Depthwise: {
                CHECK_STATUS(
                    depthwise_convolution(inputTensor, filterTensor, p, this->dwAlg, biasTensor,
                        this->temp, outputTensor, this->dwActivationParamSpec, &this->archInfo));
                break;
            }
            case Convolution_Depthwise_Pointwise: {
                auto dwFilterTensor = filterTensor;
                auto pwFilterTensor = this->weightTensors[1];
                auto dwBiasTensor = biasTensor;
                auto pwBiasTensor = this->biasTensors[1];
                CHECK_STATUS(
                    depthwise_pointwise_convolution(inputTensor, dwFilterTensor, pwFilterTensor, p,
                        this->dwAlg, dwBiasTensor, pwBiasTensor, this->temp, outputTensor,
                        this->dwActivationParamSpec, this->pwActivationParamSpec, &this->archInfo));
                break;
            }
            case Convolution_Dilation: {
                CHECK_STATUS(NOT_SUPPORTED);
                break;
            }
            default: {
                UNI_ERROR_LOG("[ERROR] unsupported convolution type %d\n", this->p.convolution_type);
            }
        }
    }

    EE infer_forward_algorithm(std::shared_ptr<AlgorithmMap> algorithmMap) override
    {
        OCLContext::getInstance().handle.get()->kernelVec = &this->opKernelVec;
        auto inputTensor = this->inputTensors[0];
        auto filterTensor = this->weightTensors[0];
        auto outputTensor = this->outputTensors[0];
        filterTensor.resize(this->filterDesc);
        ConvolutionPolicy policy = CONVOLUTION_TUNNING;
        DataType targetType = DT_F16;
        I32 algo[7];
        switch (this->p.convolution_type) {
            case Convolution_Pointwise: {
                if (this->dt == DT_F16_8Q) {
                    targetType = DT_I8;
                }
                if (algorithmMap->getAlgorithmInfoFromMap(this->name, algo, 4)) {
                    this->runInfo.algorithm = (ConvolutionForwardAlgorithm)algo[0];
                    this->runInfo.best_w[0] = algo[1];
                    this->runInfo.best_c[0] = algo[2];
                    this->runInfo.best_k[0] = algo[3];
                    this->pwAlg = (ConvolutionForwardAlgorithm)algo[0];
                } else {
                    CHECK_STATUS(convolution_infer_forward_algorithm(inputTensor, filterTensor,
                        outputTensor, p, policy, &(this->pwAlg), targetType,
                        this->pwActivationParamSpec, &this->archInfo));
                    algo[0] = this->runInfo.algorithm;
                    algo[1] = this->runInfo.best_w[0];
                    algo[2] = this->runInfo.best_c[0];
                    algo[3] = this->runInfo.best_k[0];
                    this->pwAlg = (ConvolutionForwardAlgorithm)algo[0];
                    algorithmMap->setAlgorithmInfoToMap(this->name, algo, 4);
                }
                break;
            }
            case Convolution_Depthwise: {
                if (algorithmMap->getAlgorithmInfoFromMap(this->name, algo, 4)) {
                    this->runInfo.algorithm = (ConvolutionForwardAlgorithm)algo[0];
                    this->runInfo.best_w[0] = algo[1];
                    this->runInfo.best_c[0] = algo[2];
                    this->runInfo.best_k[0] = algo[3];
                    this->dwAlg = (DepthwiseConvolutionForwardAlgorithm)algo[0];
                } else {
                    CHECK_STATUS(depthwise_convolution_infer_forward_algorithm(inputTensor,
                        filterTensor, outputTensor, p, policy, &(this->dwAlg), targetType,
                        this->dwActivationParamSpec, &this->archInfo));
                    algo[0] = this->runInfo.algorithm;
                    algo[1] = this->runInfo.best_w[0];
                    algo[2] = this->runInfo.best_c[0];
                    algo[3] = this->runInfo.best_k[0];
                    this->dwAlg = (DepthwiseConvolutionForwardAlgorithm)algo[0];
                    algorithmMap->setAlgorithmInfoToMap(this->name, algo, 4);
                }
                break;
            }
            case Convolution_Depthwise_Pointwise: {
                if (algorithmMap->getAlgorithmInfoFromMap(this->name, algo, 7)) {
                    this->runInfo.algorithm = (ConvolutionForwardAlgorithm)algo[0];
                    this->runInfo.best_w[0] = algo[1];
                    this->runInfo.best_c[0] = algo[2];
                    this->runInfo.best_k[0] = algo[3];
                    this->runInfo.best_w[1] = algo[4];
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
                    algo[1] = this->runInfo.best_w[0];
                    algo[2] = this->runInfo.best_c[0];
                    algo[3] = this->runInfo.best_k[0];
                    algo[4] = this->runInfo.best_w[1];
                    algo[5] = this->runInfo.best_c[1];
                    algo[6] = this->runInfo.best_k[1];
                    this->dwAlg = (DepthwiseConvolutionForwardAlgorithm)algo[0];
                    algorithmMap->setAlgorithmInfoToMap(this->name, algo, 7);
                }
                break;
            }
            case Convolution_Dilation: {
                CHECK_STATUS(NOT_SUPPORTED);
                break;
            }
            default:
                CHECK_STATUS(NOT_SUPPORTED);
        }
        return SUCCESS;
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
        GCLMemDesc inputGclDesc = ocl_get_desc(*inputTensor);
        if (this->p.num_outputs_origin == 1 && inputGclDesc.byteSize > 0) {
            numFiltersOcl = this->p.num_outputs_origin;
        }
        DataType targetType = DT_F16;  // Default DT_F16

        auto inputMem = (OclMemory *)inputTensor->get_memory();
        GCLMemDesc gclDesc = inputMem->get_desc();
        this->needTransInput = (gclDesc.byteSize == 0) ? true : false;
        if (this->p.convolution_type == Convolution_Dilation) {
            this->p.convolution_type = Convolution_Pointwise;
        }
        switch (this->p.convolution_type) {
            case Convolution_Pointwise: {
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
            case Convolution_Depthwise: {
                this->filterDesc = tensor4df(
                    this->dt, DF_NCHW, 1, this->numChannels, this->p.kernel_h, this->p.kernel_w);
                filterTensor.resize(this->filterDesc);
                CHECK_STATUS(depthwise_convolution_infer_output_size(
                    inputTensor, filterTensor, p, outTensors[0], targetType, &this->archInfo));
                break;
            }
            case Convolution_Depthwise_Pointwise: {
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
            case Convolution_Dilation: {
                return NOT_SUPPORTED;
                break;
            }
            default:
                CHECK_STATUS(NOT_SUPPORTED);
        }
        return SUCCESS;
    }

    U32 infer_tmp_memory_size() override
    {
        auto inputTensor = this->inputTensors[0];
        auto filterTensor = this->weightTensors[0];
        auto outputTensor = this->outputTensors[0];

        U32 bytes = 0;
        switch (this->p.convolution_type) {
            case Convolution_Pointwise: {
                CHECK_STATUS(convolution_infer_forward_tmp_bytes(inputTensor, filterTensor,
                    outputTensor, p, this->pwAlg, &bytes, &this->archInfo));
                break;
            }
            case Convolution_Depthwise: {
                CHECK_STATUS(depthwise_convolution_infer_forward_tmp_bytes(inputTensor,
                    filterTensor, outputTensor, p, this->dwAlg, &bytes, &this->archInfo));
                break;
            }
            case Convolution_Depthwise_Pointwise: {
                CHECK_STATUS(depthwise_pointwise_convolution_infer_forward_tmp_bytes(inputTensor,
                    filterTensor, this->weightTensors[1], outputTensor, p, this->dwAlg, &bytes,
                    &this->archInfo));
                break;
            }
            case Convolution_Dilation: {
                CHECK_STATUS(NOT_SUPPORTED);
                break;
            }
            default:
                CHECK_STATUS(NOT_SUPPORTED);
        }
        if (this->needTransInput) {
            auto mem = (OclMemory *)inputTensor.get_memory();
            U32 size = mem->get_desc().byteSize;
            if (bytes < size) {
                bytes = size;
            }
        }
        return bytes;
    }

    GCLMemDesc infer_wtm_memory_size_mali() override
    {
        auto filterTensor = this->weightTensors[0];
        filterTensor.resize(this->filterDesc);
        U32 stride[3] = {0, 0, 0};
        U32 offset[3] = {0, 0, 0};
        GCLMemDesc tmpDesc = gcl_mem_desc(stride, offset, DT_U8, DF_NCWHC4);
        GCLMemDesc gclmemWtmDesc[2];
        gclmemWtmDesc[0] = tmpDesc;
        gclmemWtmDesc[1] = tmpDesc;
        U32 bytes = 0;
        ((MaliPara_t)(this->archInfo.archPara))->gclmemFilterDesc = gclmemWtmDesc;
        bool needTransBiasImgToBuf = false;
        U32 biasNum = 0;
        switch (this->p.convolution_type) {
            case Convolution_Pointwise: {
                CHECK_STATUS(convolution_transform_filter_bytes(
                    filterTensor, this->p, this->pwAlg, &bytes, &this->archInfo));
                break;
            }
            case Convolution_Depthwise: {
                CHECK_STATUS(depthwise_convolution_transform_filter_bytes(
                    filterTensor, this->p, this->dwAlg, &bytes, &this->archInfo));
                break;
            }
            case Convolution_Depthwise_Pointwise: {
                U32 bytesExt = 0;
                CHECK_STATUS(depthwise_pointwise_convolution_transform_filter_bytes(filterTensor,
                    this->weightTensors[1], this->p, this->dwAlg, &bytes, &bytesExt,
                    &this->archInfo));
                wtm_dp = Tensor(OCLMem);
                OclMemory *wtmMem = (OclMemory *)wtm_dp.get_memory();
                wtmMem->padding(gclmemWtmDesc[1]);
                wtmMem->alloc();
                if (this->dwAlg == DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_GEMM) {
                    needTransBiasImgToBuf = true;
                    biasNum = 1;
                }
                break;
            }
            case Convolution_Dilation: {
                CHECK_STATUS(NOT_SUPPORTED);
                break;
            }
            default:
                CHECK_STATUS(NOT_SUPPORTED);
        }
        if (needTransBiasImgToBuf) {
            Tensor biasTensorBuf = Tensor(OCLMem);
            auto biasMemImg = (OclMemory *)(this->biasTensors[biasNum].get_memory());
            auto biasMemBuf = (OclMemory *)(biasTensorBuf.get_memory());
            GCLMemDesc descImg = biasMemImg->get_desc();
            TensorDesc desc = tensor4df(descImg.dt, descImg.df, descImg.dims[3], descImg.dims[2],
                descImg.dims[1], descImg.dims[0]);
            biasTensorBuf.resize(desc);
            GCLMemDesc descBuf = gclmem_build_desc();
            U32 stride[3] = {descImg.stride[0] * 4 + 8, descImg.stride[1], descImg.stride[2]};
            U32 offset[3] = {0, 0, 0};
            GCLMemType mt = GCL_MEM_BUF;
            MemFlags flags = CL_MEM_READ_WRITE;
            CHECK_STATUS(
                gclmem_set_desc_padding(&descBuf, stride, offset, desc.dt, DF_NCHW, mt, flags));
            biasMemBuf->padding(descBuf);
            biasMemBuf->alloc();
            void *bufPtr = biasMemBuf->get_ptr();
            CHECK_STATUS(
                gcl_fill_memory_zero(OCLContext::getInstance().handle.get(), (GCLMem_t)bufPtr));
            biasMemBuf->copy_from((Memory *)biasMemImg);
            this->biasTensors[biasNum] = biasTensorBuf;
        }
        return gclmemWtmDesc[0];
    }

    EE transform_filter() override
    {
        auto filterTensor = this->weightTensors[0];
        filterTensor.resize(this->filterDesc);

        if (DT_F16_8Q == this->dt && Convolution_Pointwise == this->p.convolution_type &&
            CONVOLUTION_ALGORITHM_WINOGRAD == this->pwAlg) {  // int8 winograd
            return NOT_SUPPORTED;
        } else if (DT_F16_8Q == this->dt &&
            Convolution_Pointwise == this->p.convolution_type) {  // int8 tilegemm
            return NOT_SUPPORTED;
        } else {  // All other cases
            auto wtmDesc = this->infer_wtm_memory_size_mali();
            this->wtm = std::shared_ptr<Tensor>(new Tensor(OCLMem));
            OclMemory *wtmMem = (OclMemory *)this->wtm->get_memory();
            wtmMem->padding(wtmDesc);
            wtmMem->alloc();

            switch (this->p.convolution_type) {
                case Convolution_Pointwise: {
                    CHECK_STATUS(convolution_transform_filter(filterTensor, this->p, this->pwAlg,
                        this->temp, this->wtm.get(), &this->archInfo));
                    break;
                }
                case Convolution_Depthwise: {
                    CHECK_STATUS(depthwise_convolution_transform_filter(
                        filterTensor, this->p, this->dwAlg, this->wtm.get(), &this->archInfo));
                    break;
                }
                case Convolution_Depthwise_Pointwise: {
                    CHECK_STATUS(depthwise_pointwise_convolution_transform_filter(filterTensor,
                        this->weightTensors[1], this->p, this->dwAlg, this->wtm.get(),
                        &this->wtm_dp, &this->archInfo));
                    this->weightTensors[1] = wtm_dp;
                    break;
                }
                case Convolution_Dilation: {
                    CHECK_STATUS(NOT_SUPPORTED);
                    break;
                }
                default:
                    CHECK_STATUS(NOT_SUPPORTED);
            }
        }
        this->weightTensors[0] = *this->get_wtm();
        return SUCCESS;
    }

    REGISTER_OCL_OPERATOR_RUN

private:
    Tensor wtm_dp;
    TensorDesc filterDesc;
    TensorDesc filterDescExt;
    bool needTransInput;
    U32 numChannels;

protected:
    ForwardRunInfoMali runInfo;
};

#endif  // _CONVELTWISEPOOLING_H
