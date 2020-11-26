// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _DECONVOLUTION_OCL_H
#define _DECONVOLUTION_OCL_H

#include "deconvolution.hpp"

class DeconvolutionOCL : public Deconvolution {
public:
    DeconvolutionOCL(DataType dt, ConvolutionParamSpec p, ActivationParamSpec activationDesc)
        : Deconvolution(dt, p, activationDesc)
    {
        setMALIArchInfo(&(this->archInfo), &(this->runInfo), &this->needSetKernelVec,
            &this->needSelectKernelLS);
    }

    ~DeconvolutionOCL(){DESTROY_OCL_KERNEL}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<DeconvolutionOCL> mem = std::shared_ptr<DeconvolutionOCL>(
            new DeconvolutionOCL(this->dt, this->p, this->activationDesc));
        *mem = *this;
        return mem;
    }

    EE infer_weight_desc() override
    {
        auto curOpWs = this->get_weightspec();
        DataType dt = curOpWs.mdt;  // weight data type may not be the same as input and output
        if (curOpWs.weight == nullptr) {
            dt = this->dt;
        }
        DataType dtNoQ = (this->dt == DT_F16_8Q) ? DT_F16 : this->dt;
        DataFormat df = DF_NCHW;
        U32 fh, fw, fc, fn;
        fn = this->numInputs;
        fc = this->p.num_outputs;
        fh = this->p.kernel_h;
        fw = this->p.kernel_w;
        U32 vectorLen = fn;
        ((MaliPara_t)(this->archInfo.archPara))->forwardRunInfo->algorithm =
            CONVOLUTION_ALGORITHM_NULL;
        TensorDesc filterTensorDesc = tensor4df(dtNoQ, df, fn, fc, fh, fw);
        TensorDesc vectorTensorDesc = tensor1d(dtNoQ, vectorLen);

        Tensor modelWeightTensor = Tensor(OCLMem);
        Tensor modelVectorTensor = Tensor(OCLMem);
        auto weightMem = (OclMemory *)modelWeightTensor.get_memory();
        auto vectorMem = (OclMemory *)modelVectorTensor.get_memory();
        modelWeightTensor.resize(filterTensorDesc);
        modelVectorTensor.resize(vectorTensorDesc);
        U32 stride[3] = {fw * fh, fc, fn};
        U32 offset[3] = {0, 0, 0};
        GCLMemType mt = GCL_MEM_BUF;
        MemFlags flags = CL_MEM_READ_WRITE;
        GCLMemDesc desc = gclmem_build_desc();
        CHECK_STATUS(gclmem_set_desc_padding(&desc, stride, offset, dtNoQ, df, mt, flags));
        weightMem->padding(desc);

        mt = GCL_MEM_IMG_1D;
        stride[0] = (vectorLen + 3) / 4;
        stride[1] = 1;
        stride[2] = 1;
        gclmem_set_desc_padding(&desc, stride, offset, dtNoQ, DF_NHWC, mt, flags);
        vectorMem->padding(desc);
        this->weightTensors.push_back(modelWeightTensor);
        this->biasTensors.push_back(modelVectorTensor);
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
        CHECK_STATUS(deconvolution(inputTensor, filterTensor, p, this->alg, scalePtr, biasTensor,
            this->temp, outputTensor, this->activationDesc, &this->archInfo));
    }

    EE infer_forward_algorithm(std::shared_ptr<AlgorithmMap> algorithmMap) override
    {
        OCLContext::getInstance().handle.get()->kernelVec = &this->opKernelVec;
        ConvolutionPolicy policy = CONVOLUTION_TUNNING;
        ((MaliPara_t)(this->archInfo.archPara))->forwardRunInfo->algorithm =
            CONVOLUTION_ALGORITHM_NULL;
        DataType targetType = DT_F16;

        I32 algo[4];
        if (algorithmMap->getAlgorithmInfoFromMap(this->name, algo, 4)) {
            this->runInfo.algorithm = (ConvolutionForwardAlgorithm)algo[0];
            this->runInfo.best_w[0] = algo[1];
            this->runInfo.best_c[0] = algo[2];
            this->runInfo.best_k[0] = algo[3];
            this->alg = (ConvolutionForwardAlgorithm)algo[0];
        } else {
            CHECK_STATUS(deconvolution_infer_forward_algorithm(this->inputTensors[0],
                this->weightTensors[0], this->outputTensors[0], p, policy, &(this->alg), targetType,
                this->activationDesc, &this->archInfo));
            algo[0] = this->runInfo.algorithm;
            algo[1] = this->runInfo.best_w[0];
            algo[2] = this->runInfo.best_c[0];
            algo[3] = this->runInfo.best_k[0];
            this->alg = (ConvolutionForwardAlgorithm)algo[0];
            algorithmMap->setAlgorithmInfoToMap(this->name, algo, 4);
        }
        return SUCCESS;
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        this->needSetKernelVec = true;
        auto inputTensor = inTensors[0];
        TensorDesc inDim = inputTensor->get_desc();
        DataType idt;
        DataFormat idf;
        U32 in, ic, ih, iw;
        CHECK_STATUS(tensor4dGet(inDim, &idt, &idf, &in, &ic, &ih, &iw));
        this->numInputs = ic;

        TensorDesc filterDim = tensor4df(this->dt, DF_NCHW, this->numInputs, this->p.num_outputs,
            this->p.kernel_h, this->p.kernel_w);
        Tensor filterTensor = Tensor(OCLMem);
        filterTensor.resize(filterDim);

        DataType targetType = this->dt;
        CHECK_STATUS(deconvolution_infer_output_size(
            inputTensor, filterTensor, p, outTensors[0], targetType, &this->archInfo));
        return SUCCESS;
    }

    U32 infer_tmp_memory_size() override
    {
        Tensor inputTensor = this->inputTensors[0];
        Tensor filterTensor = this->weightTensors[0];
        Tensor outputTensor = this->outputTensors[0];

        U32 bytes = 0;
        CHECK_STATUS(deconvolution_infer_forward_tmp_bytes(
            inputTensor, filterTensor, outputTensor, p, this->alg, &bytes, &this->archInfo));
        return bytes;
    }

    GCLMemDesc infer_wtm_memory_size_mali() override
    {
        U32 stride[3] = {0, 0, 0};
        U32 offset[3] = {0, 0, 0};
        GCLMemDesc gclmemWtmDesc = gcl_mem_desc(stride, offset, DT_U8, DF_NCWHC4);
        U32 bytes = 0;
        ((MaliPara_t)(this->archInfo.archPara))->gclmemFilterDesc = &gclmemWtmDesc;
        CHECK_STATUS(deconvolution_transform_filter_bytes(
            this->weightTensors[0], this->p, this->alg, &bytes, &this->archInfo));
        return gclmemWtmDesc;
    }

    EE transform_filter() override
    {
        Tensor filterTensor = this->weightTensors[0];
        auto wtmDesc = this->infer_wtm_memory_size_mali();
        Tensor wtm(OCLMem);
        OclMemory *wtmMem = (OclMemory *)wtm.get_memory();
        wtmMem->padding(wtmDesc);
        wtmMem->alloc();
        CHECK_STATUS(deconvolution_transform_filter(
            filterTensor, this->p, this->alg, this->temp, &wtm, &this->archInfo));
        this->weightTensors[0] = wtm;
        return SUCCESS;
    }

    REGISTER_OCL_OPERATOR_RUN

protected:
    ForwardRunInfoMali runInfo;
};

#endif  // _DECONVOLUTION_OCL_H
