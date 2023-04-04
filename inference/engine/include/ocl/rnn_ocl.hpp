// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _RNN_OCL_H
#define _RNN_OCL_H

#include "ocl/rnncell_ocl.hpp"

class RNNOCL : public RNNCellOCL {
public:
    RNNOCL(DataType dt, RNNParamSpec p) : RNNCellOCL(dt, p)
    {}

    ~RNNOCL(){DESTROY_OCL_KERNEL}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<RNNOCL> mem = std::shared_ptr<RNNOCL>(new RNNOCL(this->dt, this->p));
        *mem = *this;
        return mem;
    }

    inline void run_prepare()
    {
        OCLContext::getInstance().handle.get()->curOpName = this->get_name();
        Tensor tmpTensor = Tensor(OCLMem);
        std::vector<Tensor> tmpTensors(2, tmpTensor);
        tmpTensors[0] = this->temp;
        get_tmp_image(0, bytes + 1, &tmpTensors[1]);
        CHECK_STATUS(rnn(this->inputTensors, this->weightTensors, this->biasTensors, this->p,
            tmpTensors, this->outputTensors, nullptr, &this->archInfo));
    }

    EE infer_forward_algorithm(std::shared_ptr<AlgorithmMap> algorithmMap) override
    {
        OCLContext::getInstance().handle.get()->kernelVec = &this->opKernelVec;
        ((MaliPara_t)(this->archInfo.archPara))->forwardRunInfo->algorithm =
            CONVOLUTION_ALGORITHM_NULL;
        I32 algo[10];
        U32 algoNum = (this->p.num_projection > 0) ? 10 : 7;
        std::string name = this->name + std::to_string(get_type());
        EE ret = SUCCESS;
        if (algorithmMap->getAlgorithmInfoFromMap(name, algo, algoNum)) {
            this->runInfo.algorithm = (ConvolutionForwardAlgorithm)algo[0];
            this->runInfo.best_h[0] = algo[1];
            this->runInfo.best_c[0] = algo[2];
            this->runInfo.best_k[0] = algo[3];
            this->runInfo.best_h[1] = algo[4];
            this->runInfo.best_c[1] = algo[5];
            this->runInfo.best_k[1] = algo[6];
            if (algoNum == 10) {
                this->runInfo.best_h[2] = algo[7];
                this->runInfo.best_c[2] = algo[8];
                this->runInfo.best_k[2] = algo[9];
            }
        } else {
            ret = rnn_infer_forward_algorithm(this->inputTensors[0], this->weightTensors,
                this->biasTensors, this->p, this->outputTensors[0], &this->archInfo);
            algo[0] = this->runInfo.algorithm;
            algo[1] = this->runInfo.best_h[0];
            algo[2] = this->runInfo.best_c[0];
            algo[3] = this->runInfo.best_k[0];
            algo[4] = this->runInfo.best_h[1];
            algo[5] = this->runInfo.best_c[1];
            algo[6] = this->runInfo.best_k[1];
            if (algoNum == 10) {
                algo[7] = this->runInfo.best_h[2];
                algo[8] = this->runInfo.best_c[2];
                algo[9] = this->runInfo.best_k[2];
            }
            algorithmMap->setAlgorithmInfoToMap(name, algo, algoNum);
        }
        return ret;
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        this->needSetKernelVec = true;
        TensorDesc inputDesc = inTensors[0]->get_desc();
        this->xDim = inputDesc.dims[inputDesc.nDims - 3];
        for (U32 i = 0; i < inputDesc.nDims - 3; ++i) {
            xDim *= inputDesc.dims[i];
        }
        return rnn_infer_output_size(inTensors, this->p, outTensors, &this->archInfo);
    }

    U32 infer_tmp_memory_size() override
    {
        for (U32 i = 0; i < 4; i++) {
            bytes[i] = 0;
        }
        CHECK_STATUS(rnn_infer_forward_tmp_bytes(this->inputTensors[0], this->weightTensors[0],
            this->outputTensors[0], this->p, bytes, &this->archInfo));
        add_tmp_image(0, bytes + 1);
        return bytes[0];
    }

    EE alloc_wtm_memory()
    {
        TensorDesc ftmDesc[3];
        CHECK_STATUS(
            rnn_transform_filter_bytes(this->weightTensors, this->p, ftmDesc, &this->archInfo));
        this->wtm = std::shared_ptr<Tensor>(new Tensor(OCLMem));
        this->wtm->resize(ftmDesc[0]);
        CHECK_STATUS(set_wtm_image(ftmDesc[0]));
        this->wtm->alloc();
        this->wtm_gemv = std::shared_ptr<Tensor>(new Tensor(OCLMem));
        this->wtm_gemv->resize(ftmDesc[1]);
        this->wtm_gemv->alloc();
        if (this->p.num_projection > 0) {
            this->wtm_pro = std::shared_ptr<Tensor>(new Tensor(OCLMem));
            this->wtm_pro->resize(ftmDesc[2]);
            this->wtm_pro->alloc();
        }

        if (this->p.bi_direction) {
            this->wtm_bi = std::shared_ptr<Tensor>(new Tensor(OCLMem));
            this->wtm_bi->resize(ftmDesc[0]);
            CHECK_STATUS(set_wtm_image(ftmDesc[0], &wtm_bi));
            this->wtm_bi->alloc();
            this->wtm_gemv_bi = std::shared_ptr<Tensor>(new Tensor(OCLMem));
            this->wtm_gemv_bi->resize(ftmDesc[1]);
            this->wtm_gemv_bi->alloc();
            if (this->p.num_projection > 0) {
                this->wtm_pro_bi = std::shared_ptr<Tensor>(new Tensor(OCLMem));
                this->wtm_pro_bi->resize(ftmDesc[2]);
                this->wtm_pro_bi->alloc();
            }
        }
        return SUCCESS;
    }

    EE transform_filter() override
    {
        CHECK_STATUS(alloc_wtm_memory());
        std::vector<Tensor> filterTensors;
        std::vector<Tensor *> ftmTensors;
        U32 weightNum = (this->p.num_projection > 0) ? 2 : 1;
        U32 directions = (this->p.bi_direction) ? 2 : 1;
        for (U32 i = 0; i < directions; i++) {
            for (U32 j = 0; j < weightNum; j++) {
                filterTensors.push_back(this->weightTensors[i * weightNum + j]);
            }
        }

        ftmTensors.push_back(this->wtm.get());
        ftmTensors.push_back(this->wtm_gemv.get());
        if (this->p.num_projection > 0) {
            ftmTensors.push_back(this->wtm_pro.get());
        }
        if (this->p.bi_direction) {
            ftmTensors.push_back(this->wtm_bi.get());
            ftmTensors.push_back(this->wtm_gemv_bi.get());
            if (this->p.num_projection > 0) {
                ftmTensors.push_back(this->wtm_pro_bi.get());
            }
        }

        CHECK_STATUS(
            rnn_transform_filter(filterTensors, this->p, this->temp, ftmTensors, &this->archInfo));

        U32 newWeightTensorsSize = (weightNum + 1) * directions;
        U32 weightNumCount = 0;
        this->weightTensors.resize(newWeightTensorsSize);
        this->weightTensors[weightNumCount] = *this->wtm.get();
        weightNumCount++;
        this->weightTensors[weightNumCount] = *this->wtm_gemv.get();
        weightNumCount++;
        if (this->p.num_projection > 0) {
            this->weightTensors[weightNumCount] = (*this->wtm_pro.get());
            weightNumCount++;
        }
        if (this->p.bi_direction) {
            this->weightTensors[weightNumCount] = *this->wtm_bi.get();
            weightNumCount++;
            this->weightTensors[weightNumCount] = *this->wtm_gemv_bi.get();
            weightNumCount++;
            if (this->p.num_projection > 0) {
                this->weightTensors[weightNumCount] = (*this->wtm_pro_bi.get());
                weightNumCount++;
            }
        }
        return SUCCESS;
    }
    REGISTER_OCL_OPERATOR_RUN

private:
    std::shared_ptr<Tensor> wtm_gemv;
    std::shared_ptr<Tensor> wtm_gemv_bi;
    U32 bytes[4];
};

#endif  // _RNN_OCL_H
