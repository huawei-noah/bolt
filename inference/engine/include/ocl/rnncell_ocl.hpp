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
        INIT_GPU_INFO(&this->runInfo)
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
            this->xDim, this->p.num_outputs, 0, this->temp, hTensor, nullptr, &this->archInfo));
    }

    EE infer_forward_algorithm(std::shared_ptr<AlgorithmMap> algorithmMap) override
    {
        if (this->p.bi_direction) {
            UNI_ERROR_LOG("gpu not support bi-direction rnn.\n");
        }
        EE ret = SUCCESS;
        OCLContext::getInstance().handle.get()->kernelVec = &this->opKernelVec;
        Tensor xTensor = this->inputTensors[0];
        Tensor stateTensor = this->inputTensors[1];
        Tensor filterTensor = this->weightTensors[0];
        Tensor biasTensor = this->biasTensors[0];
        Tensor hTensor = this->outputTensors[0];
        ((MaliPara_t)(this->archInfo.archPara))->forwardRunInfo->algorithm =
            CONVOLUTION_ALGORITHM_NULL;
        I32 algo[7];
        U32 algoNum = (this->p.num_projection > 0) ? 7 : 4;
        std::string name = this->name + std::to_string(get_type());
        if (algorithmMap->getAlgorithmInfoFromMap(name, algo, algoNum)) {
            this->runInfo.algorithm = (ConvolutionForwardAlgorithm)algo[0];
            this->runInfo.best_h[0] = algo[1];
            this->runInfo.best_c[0] = algo[2];
            this->runInfo.best_k[0] = algo[3];
            if (algoNum == 7) {
                this->runInfo.best_h[0] = algo[4];
                this->runInfo.best_c[0] = algo[5];
                this->runInfo.best_k[0] = algo[6];
            }
        } else {
            ret = rnncell_infer_forward_algorithm(xTensor, filterTensor, biasTensor, stateTensor,
                this->p, this->xDim, this->p.num_outputs, hTensor, &this->archInfo);
            algo[0] = this->runInfo.algorithm;
            algo[1] = this->runInfo.best_h[0];
            algo[2] = this->runInfo.best_c[0];
            algo[3] = this->runInfo.best_k[0];
            if (algoNum == 7) {
                algo[4] = this->runInfo.best_h[1];
                algo[5] = this->runInfo.best_c[1];
                algo[6] = this->runInfo.best_k[1];
            }
            algorithmMap->setAlgorithmInfoToMap(name, algo, algoNum);
        }
        return ret;
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
        return rnncell_infer_output_size(inTensors, this->p, outTensors[0], &this->archInfo);
    }

    U32 infer_tmp_memory_size() override
    {
        U32 bytes = 0;
        CHECK_STATUS(rnncell_infer_forward_tmp_bytes(this->inputTensors[0], this->weightTensors[0],
            this->outputTensors[0], this->p, &bytes, &this->archInfo));
        return bytes;
    }

    EE alloc_wtm_memory()
    {
        TensorDesc ftmDesc[2];
        CHECK_STATUS(
            rnncell_transform_filter_bytes(this->weightTensors, this->p, ftmDesc, &this->archInfo));
        this->wtm = std::shared_ptr<Tensor>(new Tensor(OCLMem));
        this->wtm->resize(ftmDesc[0]);
        this->wtm->alloc();
        if (this->p.num_projection > 0) {
            this->wtm_pro = std::shared_ptr<Tensor>(new Tensor(OCLMem));
            this->wtm_pro->resize(ftmDesc[1]);
            this->wtm_pro->alloc();
        }
        return SUCCESS;
    }

    EE transform_filter() override
    {
        CHECK_STATUS(alloc_wtm_memory());
        std::vector<Tensor> filterTensors;
        std::vector<Tensor *> ftmTensors;
        filterTensors.push_back(this->weightTensors[0]);
        ftmTensors.push_back(this->wtm.get());
        if (this->p.num_projection > 0) {
            filterTensors.push_back(this->weightTensors[1]);
            ftmTensors.push_back(this->wtm_pro.get());
        }
        EE ret = rnncell_transform_filter(filterTensors, this->p, ftmTensors, &this->archInfo);
        this->weightTensors[0] = *(this->wtm.get());
        if (this->p.num_projection > 0) {
            this->weightTensors[1] = *wtm_pro.get();
        }
        return ret;
    }

    EE infer_weight_desc() override
    {
        U32 column = (this->p.num_projection > 0) ? this->p.num_projection : this->p.num_outputs;
        U32 filterRow = 4 * column;
        U32 filterCol = this->p.num_outputs + this->xDim;
        TensorDesc weightDesc[2];
        TensorDesc biasDesc[2];
        weightDesc[0] = tensor2df(this->dt, DF_NK, filterRow, filterCol);
        weightDesc[1] = tensor2df(this->dt, DF_NK, this->p.num_outputs, this->p.num_projection);
        biasDesc[0] = tensor1d(this->dt, filterRow);
        biasDesc[1] = tensor1d(this->dt, this->p.num_outputs);
        U32 weightNum = (this->p.num_projection > 0) ? 2 : 1;
        U32 biasNum = weightNum;
        U32 diretions = (this->p.bi_direction) ? 2 : 1;
        if (this->p.mode != RNN_LSTM) {
            UNI_ERROR_LOG("gpu rnn only support lstm.\n");
        }

        for (U32 d = 0; d < diretions; d++) {
            for (U32 i = 0; i < weightNum; i++) {
                Tensor modelWeightTensor = Tensor(OCLMem);
                modelWeightTensor.resize(weightDesc[i]);
                this->weightTensors.push_back(modelWeightTensor);
            }

            for (U32 i = 0; i < biasNum; i++) {
                Tensor modelBiasTensor = Tensor(OCLMem);
                modelBiasTensor.resize(biasDesc[i]);
                auto biasMem = (OclMemory *)modelBiasTensor.get_memory();
                biasMem->padding(0, 8, 0, 0);
                this->biasTensors.push_back(modelBiasTensor);
            }
        }
        return SUCCESS;
    }
    REGISTER_OCL_OPERATOR_RUN

protected:
    std::shared_ptr<Tensor> wtm_bi;
    std::shared_ptr<Tensor> wtm_pro;
    std::shared_ptr<Tensor> wtm_pro_bi;
    ForwardRunInfoMali runInfo;
};

#endif  // _RNNCELL_OCL_H
