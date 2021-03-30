// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _FULLY_CONNECTED_OCL_H
#define _FULLY_CONNECTED_OCL_H

#include "fully_connected.hpp"

class FullyConnectedOCL : public FullyConnected {
public:
    FullyConnectedOCL(DataType dt, FullyConnectedParamSpec p, U32 numInput)
        : FullyConnected(dt, p, numInput)
    {
        setMALIArchInfo(&(this->archInfo), &(this->runInfo), &this->needSetKernelVec,
            &this->needSelectKernelLS);
    }

    ~FullyConnectedOCL(){DESTROY_OCL_KERNEL}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<FullyConnectedOCL> mem = std::shared_ptr<FullyConnectedOCL>(
            new FullyConnectedOCL(this->dt, this->p, this->numInput));
        *mem = *this;
        return mem;
    }

    EE infer_weight_desc() override
    {
        TensorDesc weightDesc = tensor2df(this->dt, DF_NORMAL, this->p.num_outputs, this->numInput);
        TensorDesc biasDesc = tensor1d(this->dt, this->p.num_outputs);

        Tensor modelWeightTensor = Tensor(OCLMem);
        Tensor modelVectorTensor = Tensor(OCLMem);
        auto weightMem = (OclMemory *)modelWeightTensor.get_memory();
        auto vectorMem = (OclMemory *)modelVectorTensor.get_memory();
        modelWeightTensor.resize(weightDesc);
        modelVectorTensor.resize(biasDesc);
        U32 stride[3] = {this->numInput, this->p.num_outputs, 1};
        U32 offset[3] = {0, 0, 0};
        GCLMemType mt = GCL_MEM_BUF;
        MemFlags flags = CL_MEM_READ_WRITE;
        GCLMemDesc desc = gclmem_build_desc();
        CHECK_STATUS(gclmem_set_desc_padding(&desc, stride, offset, this->dt, DF_NCHW, mt, flags));
        weightMem->padding(desc);

        stride[0] = this->p.num_outputs + 8;
        stride[1] = 1;
        stride[2] = 1;
        gclmem_set_desc_padding(&desc, stride, offset, this->dt, DF_NHWC, mt, flags);
        vectorMem->padding(desc);
        this->weightTensors.push_back(modelWeightTensor);
        this->biasTensors.push_back(modelVectorTensor);
        return SUCCESS;
    }

    EE infer_forward_algorithm(std::shared_ptr<AlgorithmMap> algorithmMap) override
    {
        OCLContext::getInstance().handle.get()->kernelVec = &this->opKernelVec;
        Tensor inputTensor = this->inputTensors[0];
        Tensor filterTensor = Tensor(OCLMem);
        Tensor outputTensor = this->outputTensors[0];
        filterTensor.resize(filterDesc2D);
        ((MaliPara_t)(this->archInfo.archPara))->forwardRunInfo->algorithm =
            CONVOLUTION_ALGORITHM_NULL;
        I32 algo[4];
        if (algorithmMap->getAlgorithmInfoFromMap(this->name, algo, 4)) {
            this->runInfo.algorithm = (ConvolutionForwardAlgorithm)algo[0];
            this->runInfo.best_w[0] = algo[1];
            this->runInfo.best_c[0] = algo[2];
            this->runInfo.best_k[0] = algo[3];
        } else {
            CHECK_STATUS(fully_connected_infer_forward_algorithm(
                inputTensor, filterTensor, outputTensor, &this->archInfo));
            algo[0] = this->runInfo.algorithm;
            algo[1] = this->runInfo.best_w[0];
            algo[2] = this->runInfo.best_c[0];
            algo[3] = this->runInfo.best_k[0];
            algorithmMap->setAlgorithmInfoToMap(this->name, algo, 4);
        }
        return SUCCESS;
    }

    inline void run_prepare()
    {
        OCLContext::getInstance().handle.get()->curOpName = this->get_name();
        Tensor inputTensor = this->inputTensors[0];
        Tensor weightTensor = this->weightTensors[0];
        Tensor outputTensor = this->outputTensors[0];
        Tensor biasTensor = this->biasTensors[0];

        CHECK_STATUS(fully_connected(
            inputTensor, weightTensor, biasTensor, this->temp, outputTensor, &this->archInfo));
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        this->needSetKernelVec = true;
        TensorDesc inputDesc = inTensors[0]->get_desc();
        U32 in, ic, ih, iw;
        tensorSelectGet(inputDesc, NULL, NULL, &in, &ic, &ih, &iw);
        if (tensorIs4d(inputDesc)) {
            this->numInput = ic * ih * iw;
        } else {
            this->numInput = iw;
        }
        filterDesc2D = tensor2df(this->dt, DF_NORMAL, this->p.num_outputs, this->numInput);
        Tensor filterTensor = Tensor(OCLMem);
        filterTensor.resize(filterDesc2D);
        CHECK_STATUS(fully_connected_infer_output_size(
            inTensors[0], filterTensor, outTensors[0], &this->archInfo));
        if (this->p.num_slices > 1) {
            CHECK_STATUS(NOT_SUPPORTED);
        }
        return SUCCESS;
    }

    U32 infer_tmp_memory_size() override
    {
        Tensor inputTensor = this->inputTensors[0];
        Tensor filterTensor = Tensor(OCLMem);
        filterTensor.resize(filterDesc2D);
        U32 bytes = 0;
        CHECK_STATUS(fully_connected_infer_forward_tmp_bytes(
            inputTensor, filterTensor, &bytes, &this->archInfo));
        return bytes;
    }

    GCLMemDesc infer_wtm_memory_size_mali() override
    {
        U32 stride[3] = {0, 0, 0};
        U32 offset[3] = {0, 0, 0};
        GCLMemDesc gclmemWtmDesc = gcl_mem_desc(stride, offset, DT_U8, DF_NCWHC4);
        U32 bytes = 0;
        ((MaliPara_t)(this->archInfo.archPara))->gclmemFilterDesc = &gclmemWtmDesc;
        Tensor filterTensor = Tensor(OCLMem);
        filterTensor.resize(filterDesc2D);
        CHECK_STATUS(fully_connected_transform_filter_bytes(filterTensor, &bytes, &this->archInfo));
        return gclmemWtmDesc;
    }

    EE transform_filter() override
    {
        Tensor inputTensor = this->inputTensors[0];
        Tensor filterTensor = this->weightTensors[0];
        filterTensor.resize(this->filterDesc2D);
        auto wtmDesc = this->infer_wtm_memory_size_mali();
        if (this->p.num_slices == 1) {
            this->wtm = std::shared_ptr<Tensor>(new Tensor(OCLMem));
            OclMemory *wtmMem = (OclMemory *)this->wtm->get_memory();
            wtmMem->padding(wtmDesc);
            wtmMem->alloc();
        } else {
            CHECK_STATUS(NOT_SUPPORTED);
        }
        CHECK_STATUS(fully_connected_transform_filter(
            inputTensor, filterTensor, this->wtm.get(), &this->archInfo));
        this->weightTensors[0] = *this->get_wtm();
        auto inputDesc = this->inputTensors[0].get_desc();
        if (inputDesc.df == DF_MKT) {
            Tensor biasTensorImg = Tensor(OCLMem);
            auto biasMemBuf = (OclMemory *)(biasTensors[0].get_memory());
            auto biasMemImg = (OclMemory *)(biasTensorImg.get_memory());
            GCLMemDesc descBuf = biasMemBuf->get_desc();
            TensorDesc desc = tensor4df(descBuf.dt, descBuf.df, descBuf.dims[3], descBuf.dims[2],
                descBuf.dims[1], descBuf.dims[0]);
            biasTensorImg.resize(desc);
            GCLMemDesc descImg = gclmem_build_desc();
            U32 stride[3] = {(descBuf.stride[0] + 3) / 4, descBuf.stride[1], descBuf.stride[2]};
            U32 offset[3] = {0, 0, 0};
            GCLMemType mt = GCL_MEM_IMG_1D;
            MemFlags flags = CL_MEM_READ_WRITE;
            CHECK_STATUS(
                gclmem_set_desc_padding(&descImg, stride, offset, desc.dt, DF_NCHW, mt, flags));
            biasMemImg->padding(descBuf);
            biasMemImg->alloc();
            biasMemImg->copy_from((Memory *)biasMemBuf);
            biasTensors[0] = biasTensorImg;
        }
        return SUCCESS;
    }

    REGISTER_OCL_OPERATOR_RUN

private:
    TensorDesc filterDesc2D;

protected:
    ForwardRunInfoMali runInfo;
};

#endif  // _FULLY_CONNECTED_OCL_H
