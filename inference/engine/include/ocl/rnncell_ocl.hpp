// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _RNNCELL_OCL_H
#define _RNNCELL_OCL_H

#include "rnncell.hpp"

class RNNCellOCL : public RNNCell {
public:
    RNNCellOCL(DataType dt, RNNParamSpec p) : RNNCell(dt, p)
    {
        setMALIArchInfo(&(this->archInfo), &(this->runInfo), &this->needSetKernelVec,
            &this->needSelectKernelLS);
    }

    ~RNNCellOCL(){DESTROY_OCL_KERNEL}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<RNNCellOCL> mem =
            std::shared_ptr<RNNCellOCL>(new RNNCellOCL(this->dt, this->p));
        *mem = *this;
        return mem;
    }

    inline void run_prepare()
    {
        OCLContext::getInstance().handle.get()->curOpName = this->get_name();
        Tensor xTensor = this->inputTensors[0];
        Tensor stateTensor = this->inputTensors[1];
        Tensor hTensor = this->outputTensors[0];

        CHECK_STATUS(rnncell(xTensor, this->weightTensors, this->biasTensors, stateTensor, this->p,
            this->xDim, this->p.numOutput, 0, this->temp, hTensor, &this->archInfo));
    }

    EE infer_forward_algorithm(std::shared_ptr<AlgorithmMap> algorithmMap) override
    {
        OCLContext::getInstance().handle.get()->kernelVec = &this->opKernelVec;
        Tensor xTensor = this->inputTensors[0];
        Tensor stateTensor = this->inputTensors[1];
        Tensor filterTensor = this->weightTensors[0];
        Tensor biasTensor = this->biasTensors[0];
        Tensor hTensor = this->outputTensors[0];
        ((MaliPara_t)(this->archInfo.archPara))->forwardRunInfo->algorithm =
            CONVOLUTION_ALGORITHM_NULL;
        I32 algo[7];
        U32 algoNum = (this->p.numProjection > 0) ? 7 : 4;
        if (algorithmMap->getAlgorithmInfoFromMap(this->name, algo, algoNum)) {
            this->runInfo.algorithm = (ConvolutionForwardAlgorithm)algo[0];
            this->runInfo.best_w[0] = algo[1];
            this->runInfo.best_c[0] = algo[2];
            this->runInfo.best_k[0] = algo[3];
            if (algoNum == 7) {
                this->runInfo.best_w[0] = algo[4];
                this->runInfo.best_c[0] = algo[5];
                this->runInfo.best_k[0] = algo[6];
            }
        } else {
            CHECK_STATUS(rnncell_infer_forward_algorithm(xTensor, filterTensor, biasTensor,
                stateTensor, this->p, this->xDim, this->p.numOutput, hTensor, &this->archInfo));
            algo[0] = this->runInfo.algorithm;
            algo[1] = this->runInfo.best_w[0];
            algo[2] = this->runInfo.best_c[0];
            algo[3] = this->runInfo.best_k[0];
            if (algoNum == 7) {
                algo[4] = this->runInfo.best_w[1];
                algo[5] = this->runInfo.best_c[1];
                algo[6] = this->runInfo.best_k[1];
            }
            algorithmMap->setAlgorithmInfoToMap(this->name, algo, algoNum);
        }
        return SUCCESS;
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        this->needSetKernelVec = true;
        TensorDesc inDim = inTensors[0]->get_desc();
        DataType dt;
        DataFormat df;
        U32 iB, iX;
        CHECK_STATUS(tensor2dGet(inDim, &dt, &df, &iB, &iX));
        this->xDim = iX;
        CHECK_STATUS(rnncell_infer_output_size(inTensors, this->p, outTensors[0], &this->archInfo));
        return SUCCESS;
    }

    U32 infer_tmp_memory_size() override
    {
        U32 bytes = 0;
        CHECK_STATUS(rnncell_infer_forward_tmp_bytes(this->inputTensors[0], this->weightTensors[0],
            this->outputTensors[0], this->p, &bytes, &this->archInfo));
        return bytes;
    }

    GCLMemDesc infer_wtm_memory_size_mali() override
    {
        U32 stride[3] = {0, 0, 0};
        U32 offset[3] = {0, 0, 0};
        GCLMemDesc tmpDesc = gcl_mem_desc(stride, offset, DT_U8, DF_NCWHC4);
        GCLMemDesc gclmemWtmDesc[2];
        gclmemWtmDesc[0] = tmpDesc;
        gclmemWtmDesc[1] = tmpDesc;
        U32 bytes = 0;
        ((MaliPara_t)(this->archInfo.archPara))->gclmemFilterDesc = gclmemWtmDesc;
        CHECK_STATUS(
            rnn_transform_filter_bytes(this->weightTensors, this->p, &bytes, &this->archInfo));
        wtm_pro = std::shared_ptr<Tensor>(new Tensor(OCLMem));
        OclMemory *wtmMem = (OclMemory *)wtm_pro->get_memory();
        wtmMem->padding(gclmemWtmDesc[1]);
        if (this->p.numProjection > 0) {
            wtm_pro->alloc();
        }
        return gclmemWtmDesc[0];
    }

    EE transform_filter() override
    {
        auto wtmDesc = this->infer_wtm_memory_size_mali();
        this->wtm = std::shared_ptr<Tensor>(new Tensor(OCLMem));
        OclMemory *wtmMem = (OclMemory *)this->wtm->get_memory();
        wtmMem->padding(wtmDesc);
        this->wtm->alloc();
        std::vector<Tensor> filterTensors;
        std::vector<Tensor *> ftmTensors;
        filterTensors.push_back(this->weightTensors[0]);
        ftmTensors.push_back(this->wtm.get());
        if (this->p.numProjection > 0) {
            filterTensors.push_back(this->weightTensors[1]);
            ftmTensors.push_back(this->wtm_pro.get());
        }
        CHECK_STATUS(rnn_transform_filter(filterTensors, this->p, ftmTensors, &this->archInfo));
        this->weightTensors[0] = *this->get_wtm();
        if (this->p.numProjection > 0) {
            this->weightTensors[1] = *wtm_pro.get();
        }
        return SUCCESS;
    }

    EE infer_weight_desc() override
    {
        U32 column = (this->p.numProjection > 0) ? this->p.numProjection : this->p.numOutput;
        U32 filterRow = 4 * column;
        U32 filterCol = this->p.numOutput + this->xDim;
        TensorDesc weightDesc[2];
        TensorDesc biasDesc[2];
        weightDesc[0] = tensor2df(this->dt, DF_NK, filterRow, filterCol);
        weightDesc[1] = tensor2df(this->dt, DF_NK, this->p.numOutput, this->p.numProjection);
        biasDesc[0] = tensor1d(this->dt, filterRow);
        biasDesc[1] = tensor1d(this->dt, this->p.numOutput);
        U32 weightNum = (this->p.numProjection > 0) ? 2 : 1;

        for (U32 i = 0; i < weightNum; i++) {
            Tensor modelWeightTensor = Tensor(OCLMem);
            modelWeightTensor.resize(weightDesc[i]);
            auto weightMem = (OclMemory *)modelWeightTensor.get_memory();
            U32 s0 = weightDesc[i].dims[0];
            U32 s1 = weightDesc[i].dims[1];
            U32 stride[3] = {s0, s1, 1};
            U32 offset[3] = {0, 0, 0};
            GCLMemType mt = GCL_MEM_BUF;
            MemFlags flags = CL_MEM_READ_WRITE;
            GCLMemDesc desc = gclmem_build_desc();
            CHECK_STATUS(gclmem_set_desc_padding(&desc, stride, offset, dt, DF_NCHW, mt, flags));
            weightMem->padding(desc);
            this->weightTensors.push_back(modelWeightTensor);

            if (i == 0) {
                Tensor modelBiasTensor = Tensor(OCLMem);
                auto vectorMem = (OclMemory *)modelBiasTensor.get_memory();
                modelBiasTensor.resize(biasDesc[i]);
                stride[0] = biasDesc[i].dims[0];
                stride[1] = 1;
                CHECK_STATUS(gclmem_set_desc_padding(&desc, stride, offset, dt, DF_NCHW, mt, flags));
                vectorMem->padding(desc);
                this->biasTensors.push_back(modelBiasTensor);
            }
        }
        return SUCCESS;
    }

    REGISTER_OCL_OPERATOR_RUN

private:
    std::shared_ptr<Tensor> wtm_pro;

protected:
    ForwardRunInfoMali runInfo;
};

#endif  // _RNNCELL_OCL_H
